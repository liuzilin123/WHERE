import os
import math
import time
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from scipy.sparse import load_npz

from model import NextPOIModel, WeeklyGraph, scipy_csr_to_torch_coo


# ------------------ Utilities ------------------
def set_global_seed(seed: int = 42, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ------------------ Dataset ------------------
@dataclass
class Sample:
    seq_pois: List[int]     # 历史 POI
    seq_weeks: List[int]    # 对齐的周次 ID
    target: int             # 下一个 POI（id）
    week_id: int            # 目标所在周次


class SeqDataset(torch.utils.data.Dataset):
    """
    读取 sessions_*.csv（已切分 train/val/test），
    按 (user_index, week_start, session_id) 构建 next-POI 样本。
    """
    def __init__(
        self,
        sessions_csv: str,
        split_prefix: str,           # 'train' | 'val' | 'test'
        all_pois: int,
        max_seq_len: int = 20,
        week_id_map: Optional[Dict[str, int]] = None,
    ):
        self.max_seq_len = max_seq_len
        self.all_pois = all_pois
        self.split_prefix = split_prefix

        df = pd.read_csv(sessions_csv, parse_dates=["local_time", "week_start"])
        df = df.sort_values(["user_index", "week_start", "local_time"]).reset_index(drop=True)

        if week_id_map is None:
            week_id_map = {}
        self.week_id_map = week_id_map

        samples: List[Sample] = []
        for (uid, wk, sid), g in df.groupby(
            ["user_index", df["week_start"].dt.strftime("%Y-%m-%d"), "session_id"]
        ):
            pois = g["poi_index"].astype(int).tolist()
            wkname = f"{split_prefix}_{wk}"
            wkid = self.week_id_map.get(wkname)
            if wkid is None:
                wkid = len(self.week_id_map)
                self.week_id_map[wkname] = wkid

            for t in range(1, len(pois)):
                hist = pois[:t][-self.max_seq_len:]
                target = pois[t]
                seq_weeks = [wkid] * len(hist)
                samples.append(Sample(seq_pois=hist, seq_weeks=seq_weeks, target=target, week_id=wkid))

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "seq_pois": torch.tensor(s.seq_pois, dtype=torch.long),
            "seq_weeks": torch.tensor(s.seq_weeks, dtype=torch.long),
            "target": torch.tensor(s.target, dtype=torch.long),
            "week_id": torch.tensor(s.week_id, dtype=torch.long),
            "length": torch.tensor(len(s.seq_pois), dtype=torch.long),
        }


def collate_batch(batch, pad_val: int = -1):
    B = len(batch)
    maxL = max(int(b["length"]) for b in batch) if B > 0 else 0

    seq_pois = torch.full((B, maxL), pad_val, dtype=torch.long)
    seq_weeks = torch.zeros((B, maxL), dtype=torch.long)
    key_padding_mask = torch.ones((B, maxL), dtype=torch.bool)  # True=pad

    targets = torch.stack([b["target"] for b in batch], dim=0)
    week_ids = torch.stack([b["week_id"] for b in batch], dim=0)

    for i, b in enumerate(batch):
        L = int(b["length"])
        if L == 0:
            continue
        seq_pois[i, :L] = b["seq_pois"]
        seq_weeks[i, :L] = b["seq_weeks"]
        key_padding_mask[i, :L] = False

    return {
        "seq_pois": seq_pois,
        "seq_weeks": seq_weeks,
        "key_padding_mask": key_padding_mask,
        "targets": targets,
        "week_ids": week_ids,
    }


# ------------------ Weekly graph loader ------------------
def load_weekly_graphs(graph_root: str, split_prefix: str, device: torch.device):
    """
    读取每周超图（Xt.npy / nodes.npy / Ht_*.npz），组装成 {week_id: WeeklyGraph}。
    返回: (weeks_dict, week_names_sorted)
    其中 week_names_sorted 形如 ["train_2020-01-06", ...] / ["val_2020-03-02", ...] / ["test_2020-07-13", ...]
    """
    base = os.path.join(graph_root, f"{split_prefix}_hypergraphs")
    week_names = sorted([
        d for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d)) and d.startswith(split_prefix)
    ])
    weeks: Dict[int, WeeklyGraph] = {}
    for wkid, wkname in enumerate(week_names):
        wkdir = os.path.join(base, wkname)
        Xt = np.load(os.path.join(wkdir, "Xt.npy"))
        nodes = np.load(os.path.join(wkdir, "nodes.npy")).astype(np.int64)

        def load_h(name):
            path = os.path.join(wkdir, f"Ht_{name}.npz")
            if os.path.exists(path):
                return scipy_csr_to_torch_coo(load_npz(path), device=str(device))
            else:
                from scipy.sparse import csr_matrix
                return scipy_csr_to_torch_coo(csr_matrix((len(nodes), 0)), device=str(device))

        H_user = load_h("user")
        H_geo  = load_h("geo")
        H_cat  = load_h("cat")

        # Xt: [poi_index, cat_index, lat, lon]
        poi_idx = torch.from_numpy(nodes).to(device=device, dtype=torch.long)
        cat_idx_np = Xt[:, 1].astype(np.int64)
        cat_idx = torch.from_numpy(cat_idx_np).to(device=device, dtype=torch.long)
        geo     = torch.from_numpy(Xt[:, 2:4]).to(device=device, dtype=torch.float32)

        weeks[wkid] = WeeklyGraph(
            H_user=H_user, H_geo=H_geo, H_cat=H_cat,
            poi_idx=poi_idx, cat_idx=cat_idx, geo=geo
        )
    return weeks, week_names


# ------------------ Metrics ------------------
def recall_at_k(scores: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    topk = torch.topk(scores, k=min(k, scores.size(1)), dim=1).indices
    hit = (topk == targets[:, None]).any(dim=1).float()
    return float(hit.mean().item())


def ndcg_at_k(scores: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    B, M = scores.shape
    topk = torch.topk(scores, k=min(k, M), dim=1).indices
    gains = []
    for i in range(B):
        idxs = topk[i].tolist()
        try:
            pos = idxs.index(int(targets[i].item()))
            gains.append(1.0 / math.log2(pos + 2))
        except ValueError:
            gains.append(0.0)
    return float(np.mean(gains))


def _make_full_candidates(batch_size: int, num_pois: int, device: torch.device) -> torch.Tensor:
    # 全库采样：B 行，每行是 [0..num_pois-1]
    row = torch.arange(num_pois, dtype=torch.long, device=device)
    return row.unsqueeze(0).expand(batch_size, -1)


# ------------------ Training ------------------
def train_one_epoch(model, weeks_train, loader, optimizer, device, num_pois: int):
    model.train()
    ce_poi = nn.CrossEntropyLoss()

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        # 1) 仅构建到本 batch 需要的最大周，避免时间泄露
        seq_pois_cpu  = batch["seq_pois"]   # [B, T] (CPU)
        seq_weeks_cpu = batch["seq_weeks"]  # [B, T] (CPU)
        valid_mask = (seq_pois_cpu != -1)
        if valid_mask.any():
            w_max = int(seq_weeks_cpu[valid_mask].max().item())
        else:
            w_max = min(sorted(weeks_train.keys()))
        week_ids_to_build = [w for w in sorted(weeks_train.keys()) if w <= w_max]
        if not week_ids_to_build:
            week_ids_to_build = [min(sorted(weeks_train.keys()))]

        # 2) 为当前 batch 构建 Z_global（允许反向传播）
        model.build_global(weeks_train, device=device, week_ids=week_ids_to_build, detach=False)

        # 3) 前向/反向
        seq_pois = batch["seq_pois"].to(device)
        seq_weeks = batch["seq_weeks"].to(device)
        kpm = batch["key_padding_mask"].to(device)
        targets = batch["targets"].to(device)

        B = seq_pois.size(0)
        candidates = _make_full_candidates(B, num_pois, device)

        poi_scores = model.forward(
            weeks_train, seq_pois, seq_weeks, candidates, key_padding_mask=kpm
        )
        poi_loss = ce_poi(poi_scores, targets)

        optimizer.zero_grad(set_to_none=True)
        poi_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


# ------------------ Evaluation ------------------
@torch.no_grad()
def evaluate(model, weeks_eval, loader, device, k_list, num_pois: int):
    """
    评估仅使用 weeks_eval；只计算 POI 指标
    """
    model.eval()
    ce_poi = nn.CrossEntropyLoss()

    # 推理态构建全局表示（detach 以节省显存与图）
    model.build_global(weeks_eval, device=device, detach=True)

    total_loss = 0.0
    n_batches = 0
    mets = {f"recall@{k}": 0.0 for k in k_list}
    mets.update({f"ndcg@{k}": 0.0 for k in k_list})
    n_items = 0

    pbar = tqdm(loader, desc="Eval", leave=False)
    for batch in pbar:
        seq_pois = batch["seq_pois"].to(device)
        seq_weeks = batch["seq_weeks"].to(device)
        kpm = batch["key_padding_mask"].to(device)
        targets = batch["targets"].to(device)

        B = seq_pois.size(0)
        candidates = _make_full_candidates(B, num_pois, device)

        poi_scores = model.forward(
            weeks_eval, seq_pois, seq_weeks, candidates, key_padding_mask=kpm
        )

        poi_loss = ce_poi(poi_scores, targets)
        total_loss += float(poi_loss.item())
        n_batches += 1
        n_items   += B

        for k in k_list:
            mets[f"recall@{k}"] += recall_at_k(poi_scores, targets, k) * B
            mets[f"ndcg@{k}"]   += ndcg_at_k(poi_scores, targets, k) * B

    avg_loss = total_loss / max(n_batches, 1)
    for k in k_list:
        mets[f"recall@{k}"] /= max(n_items, 1)
        mets[f"ndcg@{k}"]   /= max(n_items, 1)

    return avg_loss, mets


# ------------------ Checkpoint I/O ------------------
def save_checkpoint(path, epoch, model, optimizer, rng_states, stats):
    ensure_dir(os.path.dirname(path))
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "rng_states": rng_states,
        "stats": stats,
    }
    torch.save(ckpt, path)


# ---- helper: 将任意 RNG 状态转成 ByteTensor（uint8）----
def _to_byte_tensor(obj):
    import torch, numpy as np
    if isinstance(obj, torch.Tensor):
        return obj.to(dtype=torch.uint8).contiguous()
    elif isinstance(obj, np.ndarray):
        return torch.as_tensor(obj, dtype=torch.uint8)
    else:
        # list/tuple 等
        return torch.as_tensor(obj, dtype=torch.uint8)

# ------------------ Checkpoint I/O ------------------
def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    # 显式声明 weights_only=False（你信任本地 ckpt）
    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        except Exception as e:
            print(f"[Warn] optimizer state load failed: {e}")

    # 兼容恢复 RNG
    rs = ckpt.get("rng_states", {})
    try:
        if "torch" in rs and rs["torch"] is not None:
            torch.set_rng_state(_to_byte_tensor(rs["torch"]))
    except Exception as e:
        print(f"[Warn] restore torch RNG failed: {e}")

    try:
        if torch.cuda.is_available():
            cuda_all = rs.get("cuda_all", None)
            if cuda_all is not None:
                cuda_all_bt = [_to_byte_tensor(s) for s in cuda_all]
                torch.cuda.set_rng_state_all(cuda_all_bt)
    except Exception as e:
        print(f"[Warn] restore CUDA RNG failed: {e}")

    # numpy/python 的状态若不是 tuple，就转成 tuple 再设
    import numpy as _np, random as _random
    try:
        if "numpy" in rs and rs["numpy"] is not None:
            st = rs["numpy"]
            if not isinstance(st, tuple):
                st = tuple(st)
            _np.random.set_state(st)
    except Exception as e:
        print(f"[Warn] restore NumPy RNG failed: {e}")

    try:
        if "python" in rs and rs["python"] is not None:
            st = rs["python"]
            if not isinstance(st, tuple):
                st = tuple(st)
            _random.setstate(st)
    except Exception as e:
        print(f"[Warn] restore Python RNG failed: {e}")

    return ckpt



# ------------------ Main ------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/nyc")
    parser.add_argument("--graph_root", type=str, default="data/nyc/graph")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max_seq_len", type=int, default=20)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_dir", type=str, default="runs/exp_nyc")
    parser.add_argument("--resume", type=str, default="")
    # 验证与早停
    parser.add_argument("--early_metric", type=str, default="ndcg@10",
                        choices=["recall@1","recall@5","recall@10","ndcg@1","ndcg@5","ndcg@10"])
    parser.add_argument("--early_stop_patience", type=int, default=5)
    args = parser.parse_args()

    ensure_dir(args.log_dir)
    set_global_seed(args.seed)

    device = torch.device(args.device)

    # mappings
    map_dir = os.path.join(args.data_root, "mappings")
    with open(os.path.join(map_dir, "poi2id.pkl"), "rb") as f:
        import pickle as pkl
        poi2id = pkl.load(f)
    with open(os.path.join(map_dir, "catid2id.pkl"), "rb") as f:
        import pickle as pkl
        cat2id = pkl.load(f)
    num_pois = len(poi2id)
    num_cats = len(cat2id)

    # -------- 三套周图：train / val / test --------
    weeks_tr, week_names_tr = load_weekly_graphs(args.graph_root, "train", device)
    weeks_va, week_names_va = load_weekly_graphs(args.graph_root, "val",   device)
    weeks_te, week_names_te = load_weekly_graphs(args.graph_root, "test",  device)

    # -------- 统一的 week_id_map（train 从 0 开始，val 接着，test 再接着） --------
    name2id_tr  = {n: i for i, n in enumerate(week_names_tr)}
    name2id_va  = {n: i + len(name2id_tr) for i, n in enumerate(week_names_va)}
    name2id_te  = {n: i + len(name2id_tr) + len(name2id_va) for i, n in enumerate(week_names_te)}
    week_id_map = {}
    week_id_map.update(name2id_tr)
    week_id_map.update(name2id_va)
    week_id_map.update(name2id_te)

    # -------- Datasets / Dataloaders --------
    sess_dir = os.path.join(args.data_root, "sessions")
    train_csv = os.path.join(sess_dir, "sessions_train.csv")
    val_csv   = os.path.join(sess_dir, "sessions_val.csv")
    test_csv  = os.path.join(sess_dir, "sessions_test.csv")

    ds_tr = SeqDataset(train_csv, "train", all_pois=num_pois, max_seq_len=args.max_seq_len, week_id_map=week_id_map)
    ds_va = SeqDataset(val_csv,   "val",   all_pois=num_pois, max_seq_len=args.max_seq_len, week_id_map=week_id_map)
    ds_te = SeqDataset(test_csv,  "test",  all_pois=num_pois, max_seq_len=args.max_seq_len, week_id_map=week_id_map)

    dl_tr = torch.utils.data.DataLoader(
        ds_tr, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=str(device).startswith("cuda"),
        collate_fn=collate_batch, drop_last=False
    )
    dl_va = torch.utils.data.DataLoader(
        ds_va, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=str(device).startswith("cuda"),
        collate_fn=collate_batch, drop_last=False
    )
    dl_te = torch.utils.data.DataLoader(
        ds_te, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=str(device).startswith("cuda"),
        collate_fn=collate_batch, drop_last=False
    )

    # -------- Model / Optim --------
    model = NextPOIModel(num_pois=num_pois, num_cats=num_cats, d_model=args.d_model).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    start_epoch = 1

    # -------- 断点续训 --------
    loaded = False
    if args.resume and os.path.exists(args.resume):
        ckpt = load_checkpoint(args.resume, model, optimizer, map_location=device)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"[Resume] Loaded from {args.resume}, resume at epoch {start_epoch}")
        loaded = True
    else:
        default_resume = os.path.join(args.log_dir, "last.ckpt")
        if os.path.exists(default_resume):
            ckpt = load_checkpoint(default_resume, model, optimizer, map_location=device)
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            print(f"[Auto Resume] Loaded from {default_resume}, resume at epoch {start_epoch}")
            loaded = True

    if not loaded:
        print("[Start Fresh] No checkpoint found. Train from scratch.")

    # -------- 日志文件（单一 CSV） --------
    log_csv = os.path.join(args.log_dir, "metrics.csv")
    if not os.path.exists(log_csv):
        with open(log_csv, "w", encoding="utf-8") as f:
            cols = [
                "epoch",
                "val_loss",
                "val_recall@1", "val_recall@5", "val_recall@10",
                "val_ndcg@1",  "val_ndcg@5",  "val_ndcg@10",
                "test_loss",
                "test_recall@1", "test_recall@5", "test_recall@10",
                "test_ndcg@1",  "test_ndcg@5",  "test_ndcg@10",
                "time_sec",
            ]
            f.write(",".join(cols) + "\n")

    # ========= Train & Evaluate with Early Stopping =========
    best_metric = -float("inf")
    bad_epochs = 0
    best_ckpt_path = os.path.join(args.log_dir, "best.ckpt")
    k_list = [1, 5, 10]

    # 按照统一 week_id_map，准备对齐后的 weeks 容器
    weeks_eval_val = {name2id_va[name]: weeks_va[i] for i, name in enumerate(week_names_va)}
    weeks_eval_te  = {name2id_te[name]: weeks_te[i] for i, name in enumerate(week_names_te)}
    # 训练时 weeks_train 的键必须与数据集的周 id 对齐（0..len(train)-1）
    weeks_train = {name2id_tr[name]: weeks_tr[i] for i, name in enumerate(week_names_tr)}

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # --- 训练（仅训练周，避免时间泄露） ---
        train_one_epoch(
            model, weeks_train, dl_tr, optimizer, device,
            num_pois=num_pois
        )

        # --- 验证 ---
        val_loss, val_mets = evaluate(
            model, weeks_eval_val, dl_va, device, k_list, num_pois=num_pois
        )

        early_key = args.early_metric  # e.g., 'ndcg@10'
        cur_metric = val_mets[early_key]

        improved = cur_metric > best_metric
        if improved:
            best_metric = cur_metric
            bad_epochs = 0

            # 在验证集提升时，再评测一次测试集
            test_loss, test_mets = evaluate(
                model, weeks_eval_te, dl_te, device, k_list, num_pois=num_pois
            )

            def fmt_mets(m):
                return " | ".join([
                    f"R@1:{m['recall@1']:.4f}", f"R@5:{m['recall@5']:.4f}", f"R@10:{m['recall@10']:.4f}",
                    f"N@1:{m['ndcg@1']:.4f}",   f"N@5:{m['ndcg@5']:.4f}",   f"N@10:{m['ndcg@10']:.4f}",
                ])
            elapsed = time.time() - t0
            print(f"* Epoch {epoch:02d} [BEST] | Val loss {val_loss:.4f} | {fmt_mets(val_mets)} | "
                  f"Test loss {test_loss:.4f} | {fmt_mets(test_mets)} | time {elapsed:.1f}s")

            with open(log_csv, "a", encoding="utf-8") as f:
                row = [
                    epoch,
                    f"{val_loss:.6f}",
                    f"{val_mets['recall@1']:.6f}", f"{val_mets['recall@5']:.6f}", f"{val_mets['recall@10']:.6f}",
                    f"{val_mets['ndcg@1']:.6f}",   f"{val_mets['ndcg@5']:.6f}",   f"{val_mets['ndcg@10']:.6f}",
                    f"{test_loss:.6f}",
                    f"{test_mets['recall@1']:.6f}", f"{test_mets['recall@5']:.6f}", f"{test_mets['recall@10']:.6f}",
                    f"{test_mets['ndcg@1']:.6f}",   f"{test_mets['ndcg@5']:.6f}",   f"{test_mets['ndcg@10']:.6f}",
                    f"{elapsed:.3f}",
                ]
                f.write(",".join(map(str, row)) + "\n")

            # 保存 best.ckpt
            rng_states = {
                "torch": torch.get_rng_state(),
                "cuda_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            }
            ensure_dir(args.log_dir)
            save_checkpoint(best_ckpt_path, epoch, model, optimizer, rng_states, stats={"best_metric": best_metric})

        else:
            bad_epochs += 1
            elapsed = time.time() - t0
            print(f"Epoch {epoch:02d} | Val loss {val_loss:.4f} | "
                  f"R@10:{val_mets['recall@10']:.4f} | N@10:{val_mets['ndcg@10']:.4f} "
                  f"| no improve ({bad_epochs}/{args.early_stop_patience}) | time {elapsed:.1f}s")

            # 记录仅验证结果（测试列留空）
            with open(log_csv, "a", encoding="utf-8") as f:
                row = [
                    epoch,
                    f"{val_loss:.6f}",
                    f"{val_mets['recall@1']:.6f}", f"{val_mets['recall@5']:.6f}", f"{val_mets['recall@10']:.6f}",
                    f"{val_mets['ndcg@1']:.6f}",   f"{val_mets['ndcg@5']:.6f}",   f"{val_mets['ndcg@10']:.6f}",
                    "", "", "", "", "", "", "",
                    f"{elapsed:.3f}",
                ]
                f.write(",".join(map(str, row)) + "\n")

        # 常规 last.ckpt
        rng_states = {
            "torch": torch.get_rng_state(),
            "cuda_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }
        last_path = os.path.join(args.log_dir, "last.ckpt")
        ensure_dir(args.log_dir)
        save_checkpoint(last_path, epoch, model, optimizer, rng_states, stats={"best_metric": best_metric})

        # 早停
        if bad_epochs >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch}. Best {args.early_metric}={best_metric:.6f}.")
            break

    print("Training finished.")
    print(f"Best checkpoint: {best_ckpt_path}")
    print(f"Last checkpoint: {os.path.join(args.log_dir, 'last.ckpt')}")
    print(f"Metrics CSV   : {log_csv}")


if __name__ == "__main__":
    main()
