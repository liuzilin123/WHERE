import json
import os
import pickle
from typing import List, Set, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from sklearn.neighbors import BallTree

# =========================
# 工具函数
# =========================

def _dedup_sets(edges: List[Set[int]]) -> List[Set[int]]:
    """按集合完全相同去重，并移除 |set| < 2 的边"""
    seen, out = set(), []
    for s in edges:
        if len(s) < 2:
            continue
        fs = frozenset(s)
        if fs not in seen:
            seen.add(fs)
            out.append(set(s))
    return out

def _build_incidence(node_ids: Sequence[int], edges: List[Set[int]]) -> csr_matrix:
    """构建 |V| x |E| 的关联矩阵 H（节点顺序= node_ids）"""
    n = len(node_ids)
    if n == 0 or not edges:
        return csr_matrix((n, 0), dtype=np.int8)
    node_pos = {nid: i for i, nid in enumerate(node_ids)}
    rows, cols = [], []
    for j, e in enumerate(edges):
        for nid in e:
            i = node_pos.get(nid)
            if i is not None:
                rows.append(i)
                cols.append(j)
    data = np.ones(len(rows), dtype=np.int8)
    return csr_matrix((data, (rows, cols)), shape=(n, len(edges)), dtype=np.int8)

def _to_radians(latlon: np.ndarray) -> np.ndarray:
    return np.radians(latlon)

def _meters_to_radians(meters: float) -> float:
    return meters / 6371000.0  # 地球半径约 6371000m

def _mode_or_first(s: pd.Series):
    m = s.mode(dropna=True)
    if len(m) > 0:
        return m.iloc[0]
    s2 = s.dropna()
    return s2.iloc[0] if len(s2) > 0 else np.nan

# =========================
# 构建超图
# =========================
class WeeklyHypergraphBuilder:

    def __init__(self, poi2id: Optional[Dict[str, int]] = None):
        # 可选：当输入 CSV 没有 poi_index 时，用 POI_id -> 索引 的映射补齐
        self.poi2id = poi2id

    # ---------- 三类超边 ----------
    def user_behavior_edges(self, df_week: pd.DataFrame, min_size: int = 2) -> List[Set[int]]:
        """
        用户行为边：同一用户在该周访问过的 poi_index 集合作为一条超边
        仅当集合大小 >= min_size 时保留
        """
        if 'user_index' not in df_week.columns or 'poi_index' not in df_week.columns:
            return []
        edges: List[Set[int]] = []
        for _, g in df_week.groupby('user_index'):
            s = set(g['poi_index'].astype(int))
            if len(s) >= min_size:
                edges.append(s)
        return _dedup_sets(edges)

    def category_edges(self, df_week: pd.DataFrame, min_size: int) -> List[Set[int]]:
        """
        类别边：同一 cat_index（>=0）下的 poi_index 集合作为一条超边
        """
        if 'cat_index' not in df_week.columns or 'poi_index' not in df_week.columns:
            return []
        edges: List[Set[int]] = []
        valid = df_week[df_week['cat_index'].fillna(-1).astype(int) >= 0]
        for _, g in valid.groupby('cat_index'):
            s = set(g['poi_index'].astype(int))
            if len(s) >= min_size:
                edges.append(s)
        return _dedup_sets(edges)

    def geo_edges(
            self,
            df_week: pd.DataFrame,
            radius_m: float,
            min_size: int = 5,
            *,
            center_sampling: str = "grid_medoid",  # "grid_medoid" | "grid_centroid" | "grid" | "all"
            cell_m: Optional[float] = None,  # grid 网格边长（米），默认 0.8 * radius_m
            max_size: int = 128,  # 每条超边最多保留的 POI 数
            jaccard_merge_thr: float = 0.92,  # 合并近重复边；设为 0 或 None 可关闭
            medoid_cap: int = 400  # 单格内点数>此阈值时，为避免 O(n^2) 代价会下采样
    ) -> List[Set[int]]:

        needed = {'poi_index', 'latitude', 'longitude'}
        if not needed.issubset(df_week.columns) or df_week.empty:
            return []

        # 去重&过滤缺经纬度（不要把 NaN 填 0，否则会跑到(0,0)）
        g = (
            df_week.drop_duplicates(subset=['poi_index'])
            .dropna(subset=['latitude', 'longitude'])
            [['poi_index', 'latitude', 'longitude']]
        )
        if g.empty:
            return []

        poi_ids = g['poi_index'].to_numpy(int)
        lat = g['latitude'].to_numpy(float)
        lon = g['longitude'].to_numpy(float)
        latlon = np.stack([lat, lon], axis=1)

        def _to_radians(latlon_np: np.ndarray) -> np.ndarray:
            return np.radians(latlon_np)

        def _meters_to_radians(m: float) -> float:
            return float(m) / 6371000.0

        # —— grid 划分（把米换成度；经度步长要乘 cos(lat)）——
        lat_mean = float(np.mean(lat))
        cm = float(cell_m) if cell_m is not None else float(radius_m * 0.8)
        dlat = max(1e-12, cm / 111111.0)
        dlon = max(1e-12, cm / (111111.0 * max(1e-6, np.cos(np.radians(abs(lat_mean))))))

        lat_bin = np.floor(lat / dlat).astype(np.int64)
        lon_bin = np.floor(lon / dlon).astype(np.int64)

        cs = center_sampling.lower()
        if cs == "grid":
            cs = "grid_medoid"

        # —— 选择每个格子的代表中心：medoid 或 centroid ——
        def _haversine_pairwise_rad(P_rad: np.ndarray) -> np.ndarray:
            """
            P_rad: (k, 2) = (lat_rad, lon_rad). 返回 k×k 的球面大圆距离(弧度)矩阵。
            """
            lat1 = P_rad[:, 0][:, None]
            lon1 = P_rad[:, 1][:, None]
            lat2 = P_rad[:, 0][None, :]
            lon2 = P_rad[:, 1][None, :]
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
            return 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))

        centers_idx: list[int] = []
        if cs in {"grid_medoid", "grid_centroid"}:
            # 建桶：同一 (lat_bin, lon_bin) -> 索引列表
            from collections import defaultdict
            buckets = defaultdict(list)
            for i, key in enumerate(zip(lat_bin.tolist(), lon_bin.tolist())):
                buckets[key].append(i)

            P_deg = latlon  # (N,2) in degrees
            P_rad = _to_radians(P_deg)

            for _, idxs in buckets.items():
                if not idxs:
                    continue
                cell_idx = np.array(idxs, dtype=np.int64)
                if cs == "grid_centroid":
                    # 几何中心（度），再选离中心最近的 POI 当代表
                    c_lat = float(np.mean(lat[cell_idx]))
                    c_lon = float(np.mean(lon[cell_idx]))
                    c_rad = _to_radians(np.array([[c_lat, c_lon]], dtype=float))
                    # 到中心的球面距离
                    d = _haversine_pairwise_rad(
                        np.vstack([c_rad, P_rad[cell_idx]])  # 第一行是中心
                    )[0, 1:]  # 中心到每个点的距离
                    pick_local = int(np.argmin(d))
                    centers_idx.append(int(cell_idx[pick_local]))
                else:
                    # grid_medoid：格内 medoid（与其它点的总球面距离最小）
                    # 大桶下采样避免 O(m^2) 爆炸
                    if len(cell_idx) > medoid_cap:
                        rng = np.random.default_rng(0)
                        cell_idx = rng.choice(cell_idx, size=medoid_cap, replace=False)
                    P_cell = P_rad[cell_idx]  # (k,2)
                    D = _haversine_pairwise_rad(P_cell)  # k×k
                    s = D.sum(axis=1)
                    pick_local = int(np.argmin(s))
                    centers_idx.append(int(cell_idx[pick_local]))
            centers_idx = np.array(centers_idx, dtype=np.int64)

        elif cs == "all":
            centers_idx = np.arange(len(poi_ids), dtype=np.int64)
        else:
            raise ValueError(f"Unknown center_sampling='{center_sampling}', "
                             f"use 'grid_medoid' | 'grid_centroid' | 'all'.")

        # —— 基于所选中心，做半径邻域查询 ——
        tree = BallTree(_to_radians(latlon), metric='haversine')
        rad = _meters_to_radians(radius_m)
        centers_rad = _to_radians(latlon[centers_idx])
        inds_list, _ = tree.query_radius(centers_rad, r=rad, return_distance=True, sort_results=True)

        edges: List[Set[int]] = []
        for nbr_idx in inds_list:
            if len(nbr_idx) < min_size:
                continue
            # 对超密区域避免只保留下前 max_size 个最近点的偏置：可选重要度采样
            if max_size is not None and len(nbr_idx) > max_size:
                # 这里仍保留“按距离截断”，如果你想更公平，可替换为分层/随机子采样
                nbr_idx = nbr_idx[:max_size]
            e = set(poi_ids[nbr_idx].tolist())
            if len(e) >= min_size:
                edges.append(e)

        if jaccard_merge_thr and jaccard_merge_thr > 0:
            edges = self._merge_by_jaccard(edges, jaccard_merge_thr)

        return _dedup_sets(edges)

    # ====== 辅助：Jaccard 合并高度相似的边（贪心，保大边/代表边） ======
    @staticmethod
    def _merge_by_jaccard(edges: List[Set[int]], thr: float) -> List[Set[int]]:
        if not edges:
            return edges
        edges = sorted(edges, key=lambda s: -len(s))  # 大边先保留
        kept: List[Set[int]] = []
        for e in edges:
            drop = False
            for k in kept:
                inter = len(e & k)
                if inter == 0:
                    continue
                jac = inter / (len(e) + len(k) - inter)
                if jac >= thr:
                    drop = True
                    break
            if not drop:
                kept.append(e)
        return kept

    # ---------- 每周构建 ----------
    def build_week(self,
                   week_csv_path: str,
                   out_dir: str,
                   build_user: bool = True,
                   build_cat: bool = True,
                   build_geo: bool = True,
                   geo_radius_m: float = 300.0,
                   min_edge_size: int = 2,
                   save_sparse: bool = True) -> Dict:
        week_name = os.path.splitext(os.path.basename(week_csv_path))[0]
        base = os.path.join(out_dir, week_name)
        os.makedirs(base, exist_ok=True)

        df = pd.read_csv(week_csv_path)

        # 若没有 poi_index，尝试用映射从 POI_id 推断（可选）
        if 'poi_index' not in df.columns:
            if 'POI_id' in df.columns and self.poi2id is not None:
                df = df[df['POI_id'].isin(self.poi2id)].copy()
                df['poi_index'] = df['POI_id'].map(self.poi2id).astype(int)
            else:
                meta = {'week': week_name, 'nodes': 0, 'edges_user': 0, 'edges_cat': 0, 'edges_geo': 0}
                with open(os.path.join(base, "meta.json"), "w") as f:
                    json.dump(meta, f, indent=2)
                return meta

        # 节点集合
        node_ids = np.sort(df['poi_index'].dropna().astype(int).unique())

        if len(node_ids) == 0:
            meta = {'week': week_name, 'nodes': 0, 'edges_user': 0, 'edges_cat': 0, 'edges_geo': 0}
            with open(os.path.join(base, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
            return meta

        if 'local_time' in df.columns:
            df['local_time'] = pd.to_datetime(df['local_time'], errors='coerce')

        cols_exist = set(df.columns)
        agg_dict = {}
        if 'cat_index' in cols_exist:
            agg_dict['cat_index'] = _mode_or_first

        if {'latitude', 'longitude'}.issubset(cols_exist):
            agg_dict['latitude'] = _mode_or_first
            agg_dict['longitude'] = _mode_or_first

        node_meta = (df.sort_values('local_time') if 'local_time' in df.columns else df) \
            .groupby('poi_index', as_index=True) \
            .agg(agg_dict) \
            .reindex(node_ids)

        cat_idx = node_meta['cat_index'].fillna(-1).astype(np.float32).to_numpy() if 'cat_index' in node_meta.columns else np.full(len(node_ids), -1, dtype=np.float32)
        lat = node_meta['latitude'].astype(float).fillna(0.0).to_numpy() if 'latitude' in node_meta.columns else np.zeros(len(node_ids), dtype=np.float32)
        lon = node_meta['longitude'].astype(float).fillna(0.0).to_numpy() if 'longitude' in node_meta.columns else np.zeros(len(node_ids), dtype=np.float32)

        poi_index_feat = node_ids.astype(np.float32)
        Xt = np.stack([poi_index_feat, cat_idx, lat, lon], axis=1).astype(np.float32)

        # 三类超边（地理超边的 min_size 与上层保持一致）
        edges_user = self.user_behavior_edges(df, min_edge_size) if build_user else []
        edges_cat  = self.category_edges(df,  min_edge_size)     if build_cat  else []
        edges_geo  = self.geo_edges(df, geo_radius_m, min_size=min_edge_size) if build_geo else []

        # 关联矩阵
        H_user = _build_incidence(node_ids, edges_user) if build_user else csr_matrix((len(node_ids), 0))
        H_cat  = _build_incidence(node_ids, edges_cat)  if build_cat  else csr_matrix((len(node_ids), 0))
        H_geo  = _build_incidence(node_ids, edges_geo)  if build_geo  else csr_matrix((len(node_ids), 0))

        # 保存
        np.save(os.path.join(base, 'nodes.npy'), node_ids.astype(np.int32))
        np.save(os.path.join(base, 'Xt.npy'), Xt)
        if save_sparse:
            save_npz(os.path.join(base, 'Ht_user.npz'), H_user)
            save_npz(os.path.join(base, 'Ht_cat.npz'),  H_cat)
            save_npz(os.path.join(base, 'Ht_geo.npz'),  H_geo)

        meta = {
            'week': week_name,
            'nodes': int(len(node_ids)),
            'edges_user': int(H_user.shape[1]),
            'edges_cat': int(H_cat.shape[1]),
            'edges_geo': int(H_geo.shape[1]),
        }
        with open(os.path.join(base, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        return meta

# =========================
# 批量处理入口
# =========================

def build_all_weeks(snapshot_dir: str,
                    out_root: str,
                    prefix: Optional[str] = None,
                    **kwargs):
    builder = WeeklyHypergraphBuilder()
    os.makedirs(out_root, exist_ok=True)

    stats_all = []
    for fn in sorted(os.listdir(snapshot_dir)):
        if not fn.endswith('.csv'):
            continue
        if prefix and not fn.startswith(prefix):
            continue
        stats = builder.build_week(os.path.join(snapshot_dir, fn), out_dir=out_root, **kwargs)
        stats_all.append(stats)

    print(pd.DataFrame(stats_all).tail())
    return stats_all

# =========================
# 入口
# =========================
if __name__ == "__main__":

    # 训练集（train_weeks -> train_hypergraphs）
    build_all_weeks(
        snapshot_dir="data/nyc/snapshots/train_weeks",
        out_root="data/nyc/graph/train_hypergraphs",
        prefix="train_",
        build_user=True, build_cat=True, build_geo=True,
        geo_radius_m=800.0,
        min_edge_size=3
    )

    # 验证集（val_weeks -> val_hypergraphs）
    build_all_weeks(
        snapshot_dir="data/nyc/snapshots/val_weeks",
        out_root="data/nyc/graph/val_hypergraphs",
        prefix="val_",
        build_user=True, build_cat=True, build_geo=True,
        geo_radius_m=800.0,
        min_edge_size=3
    )

    # 测试集（test_weeks -> test_hypergraphs）
    build_all_weeks(
        snapshot_dir="data/nyc/snapshots/test_weeks",
        out_root="data/nyc/graph/test_hypergraphs",
        prefix="test_",
        build_user=True, build_cat=True, build_geo=True,
        geo_radius_m=800.0,
        min_edge_size=3
    )
