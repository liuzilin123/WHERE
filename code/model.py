import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Utils
# =========================
def scipy_csr_to_torch_coo(H_csr, device: str = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Convert SciPy CSR to a coalesced PyTorch COO tensor.
    """
    coo = H_csr.tocoo()
    idx = torch.stack([
        torch.from_numpy(coo.row.astype("int64")),
        torch.from_numpy(coo.col.astype("int64")),
    ], dim=0)
    val = torch.from_numpy(coo.data).to(torch.float32)
    return torch.sparse_coo_tensor(idx, val, size=coo.shape, dtype=dtype, device=device).coalesce()


@torch.jit.script
def _segment_softmax(scores: torch.Tensor, group_idx: torch.Tensor) -> torch.Tensor:
    """
    Softmax over 1D 'scores' grouped by integer 'group_idx' (same length).
    """
    if scores.numel() == 0:
        return scores
    scores = scores.to(torch.float32)
    G = int(torch.max(group_idx).item()) + 1

    # max per group
    max_g = torch.full((G,), float("-inf"), device=scores.device)
    max_g.scatter_reduce_(0, group_idx, scores, reduce="amax", include_self=True)

    # exp normalized by group max
    exps = torch.exp(scores - max_g[group_idx])

    # sum per group
    sum_g = torch.zeros(G, device=scores.device)
    sum_g.scatter_add_(0, group_idx, exps)

    return (exps / (sum_g[group_idx] + 1e-12)).to(scores.dtype)

# =========================
# 1) POI feature encoder
# =========================
class POIFeatureEncoder(nn.Module):
    """
    Encode POI: id embedding + category embedding + geo Fourier features -> d_model.
    """
    def __init__(
        self,
        num_pois: int,
        num_cats: int,
        d_model: int,
        d_poi: int = 64,
        d_cat: int = 32,
        d_geo: int = 32,
        fourier_dim: int = 8,
    ):
        super().__init__()
        self.emb_poi = nn.Embedding(num_pois, d_poi)
        self.emb_cat = nn.Embedding(num_cats + 1, d_cat, padding_idx=num_cats)  # +1 for unknown
        self.lin_geo = nn.Linear(4 + 4 * fourier_dim, d_geo)
        self.proj = nn.Linear(d_poi + d_cat + d_geo, d_model)

        # frequencies for Fourier features on radians
        self.register_buffer(
            "freq",
            torch.exp(torch.linspace(math.log(0.5), math.log(50.0), fourier_dim))
        )

        for m in [self.emb_poi, self.emb_cat, self.lin_geo, self.proj]:
            if hasattr(m, "weight"):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, poi_idx: torch.Tensor, cat_idx: torch.Tensor, geo: torch.Tensor) -> torch.Tensor:
        e_poi = self.emb_poi(poi_idx)

        cats = cat_idx.clone()
        cats[cats < 0] = self.emb_cat.num_embeddings - 1  # unknown
        e_cat = self.emb_cat(cats)

        lat, lon = geo[:, 0] * math.pi / 180.0, geo[:, 1] * math.pi / 180.0
        s = torch.stack([torch.sin(lat), torch.cos(lat), torch.sin(lon), torch.cos(lon)], dim=-1)

        lat_f = lat[:, None] * self.freq[None, :]
        lon_f = lon[:, None] * self.freq[None, :]
        fourier = torch.cat([torch.sin(lat_f), torch.cos(lat_f), torch.sin(lon_f), torch.cos(lon_f)], dim=-1)

        e_geo = self.lin_geo(torch.cat([s, fourier], dim=-1))
        return self.proj(torch.cat([e_poi, e_cat, e_geo], dim=-1))


# =========================
# 2) Dual-attn hypergraph conv
# =========================
class HypergraphDualAttnConv(nn.Module):
    """
    Dual attention on a hypergraph incidence H (V x E, COO & coalesced):
      inner attn (node -> edge)  with D_v^{-1/2} H D_e^{-1/2}
      outer attn (edge -> node)  with D_v^{-1/2} H D_e^{-1/2}
    """
    def __init__(
        self,
        d_in: int,
        d_out: int,
        dropout: float = 0.1,
        bias: bool = True,
        attn_temperature: float = 1.0,
    ):
        super().__init__()
        self.xi = nn.Parameter(torch.randn(d_in))   # node->edge attention vector
        self.rho = nn.Parameter(torch.randn(d_in))  # edge->node attention vector
        self.proj = nn.Linear(d_in, d_out, bias=bias)
        self.ln = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(dropout)
        self.tau = float(attn_temperature)

        nn.init.normal_(self.xi, std=math.sqrt(2.0 / d_in))
        nn.init.normal_(self.rho, std=math.sqrt(2.0 / d_in))
        nn.init.xavier_uniform_(self.proj.weight)

    @torch.no_grad()
    def _degrees(self, H: torch.Tensor):
        # dv[i]: node degree; de[j]: hyperedge size
        dv = torch.sparse.sum(H, dim=1).to_dense()  # [V]
        de = torch.sparse.sum(H, dim=0).to_dense()  # [E]
        dv.clamp_(min=1.0)
        de.clamp_(min=1.0)
        return dv, de

    def forward(self, H: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        assert H.layout == torch.sparse_coo, "H must be a sparse COO tensor"
        if not H.is_coalesced():
            H = H.coalesce()

        v_idx, e_idx = H.indices()           # [nnz]
        V, E = H.size()
        d = Z.size(1)
        device, dtype = Z.device, Z.dtype
        # degrees
        dv, de = self._degrees(H)            # [V], [E]
        # ---------- inner: node -> edge ----------
        scores_ne = (Z[v_idx].float() @ self.xi.float())  # [nnz]
        if self.tau != 1.0:
            scores_ne = scores_ne / self.tau
        alpha = _segment_softmax(scores_ne, e_idx)        # [nnz]
        # D_v^{-1/2} D_e^{-1/2}
        scale_ne = dv[v_idx].rsqrt() * de[e_idx].rsqrt()  # [nnz]
        E_emb = torch.zeros(E, d, device=device, dtype=dtype)
        E_emb.index_add_(0, e_idx, (alpha * scale_ne)[:, None] * Z[v_idx])

        # ---------- outer: edge -> node ----------
        scores_en = (E_emb.float() @ self.rho.float())[e_idx]  # [nnz]
        if self.tau != 1.0:
            scores_en = scores_en / self.tau
        beta = _segment_softmax(scores_en, v_idx)              # [nnz]

        scale_en = dv[v_idx].rsqrt() * de[e_idx].rsqrt()       # [nnz]

        Z_new = torch.zeros(V, d, device=device, dtype=dtype)
        Z_new.index_add_(0, v_idx, (beta * scale_en)[:, None] * E_emb[e_idx])

        # projection / dropout / layernorm
        return self.ln(self.proj(self.drop(Z_new)))


# =========================
# 3) Multi-type HGNN with gating
# =========================
class TypeHGNN(nn.Module):
    """
    Stack hypergraph convs for multiple types and fuse by learned gate.
    """
    def __init__(
        self,
        d_model: int,
        num_layers: int = 2,
        types: Tuple[str, ...] = ("user", "geo", "cat"),
        attn_drop: float = 0.1,
        gate_temp: float = 1.0,
    ):
        super().__init__()
        self.types = types
        self.stacks = nn.ModuleDict({
            t: nn.ModuleList([HypergraphDualAttnConv(d_model, d_model, dropout=attn_drop) for _ in range(num_layers)])
            for t in types
        })
        self.fuse_proj = nn.Linear(d_model, d_model, bias=False)
        self.fuse_query = nn.ParameterDict({t: nn.Parameter(torch.randn(d_model)) for t in types})
        self.gate_temp = nn.Parameter(torch.tensor(gate_temp))
        self.out_ln = nn.LayerNorm(d_model)

        nn.init.xavier_uniform_(self.fuse_proj.weight)
        for t in types:
            nn.init.normal_(self.fuse_query[t], std=math.sqrt(2.0 / d_model))

    def forward(
        self,
        H_dict: Dict[str, torch.Tensor],
        X: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:

        Z_type: Dict[str, torch.Tensor] = {}
        for t in self.types:
            Ht = H_dict.get(t)
            if Ht is not None and Ht._nnz() > 0:
                Z = X
                for conv in self.stacks[t]:
                    Z = F.relu(Z + conv(Ht, Z))  # backward-compatible call
                Z_type[t] = Z
            else:
                Z_type[t] = X

        # gating
        logits = []
        for t in self.types:
            logits.append(torch.tanh(self.fuse_proj(Z_type[t])) @ self.fuse_query[t])  # [V]
        G = torch.stack(logits, dim=-1) / torch.clamp(self.gate_temp, min=1e-3)        # [V, T]
        G = torch.softmax(G, dim=-1)

        Z = sum(G[:, i:i+1] * Z_type[t] for i, t in enumerate(self.types))
        return self.out_ln(Z + X), Z_type, G


# =========================
# 4) LSTM sequence encoder with attention pooling
# =========================
class LSTMSeqEncoder(nn.Module):
    """
    LSTM over padded sequence (batch_first).
    Returns (all_hidden, context) where context can be:
      last hidden (context_mode='last')
      attention pooled (context_mode='attn', default)
      gated fusion of last & attn (context_mode='gate')
    """
    def __init__(self, d_model: int, num_layers: int = 2, dropout: float = 0.1,
                 context_mode: str = "attn"):
        super().__init__()
        assert context_mode in ("last", "attn", "gate")
        self.context_mode = context_mode

        self.lstm = nn.LSTM(d_model, d_model, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.pre_ln = nn.LayerNorm(d_model)

        # attention parameters
        if context_mode in ("attn", "gate"):
            self.attn_w = nn.Linear(d_model, d_model, bias=True) # 为每个隐藏状态计算注意力得分
            self.attn_v = nn.Linear(d_model, 1, bias=False)  # score = v^T tanh(W h_t)

        # gate when fusing last & attn
        if context_mode == "gate":
            self.gate = nn.Linear(3 * d_model, d_model, bias=True)  # sigmoid(gate) mixing

    def _attention_pool(self, H: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        H: [B,T,d]; key_padding_mask: [B,T] True=pad
        returns: h_attn [B,d]
        """
        # energy: [B,T,1]
        e = torch.tanh(self.attn_w(H))
        e = self.attn_v(e).squeeze(-1)  # [B,T]

        if key_padding_mask is not None:
            # mask padded positions to -inf
            e = e.masked_fill(key_padding_mask, float("-inf"))

        alpha = torch.softmax(e, dim=1)           # [B,T]
        h_attn = torch.bmm(alpha.unsqueeze(1), H).squeeze(1)  # [B,d]
        return h_attn

    def forward(self, X: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = X.shape
        X = self.pre_ln(X)

        if key_padding_mask is None:
            H, (h_n, _) = self.lstm(X)
        else:
            lengths = (T - key_padding_mask.sum(dim=1)).clamp_min(0).to(torch.int64).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(X, lengths, batch_first=True, enforce_sorted=False)
            packed_out, (h_n, _) = self.lstm(packed)
            H, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=T)

        H = self.drop(H)
        h_last = h_n[-1]  # [B,d]

        if self.context_mode == "last":
            h_ctx = h_last
        elif self.context_mode == "attn":
            h_attn = self._attention_pool(H, key_padding_mask)
            h_ctx = h_attn
        else:  # "gate"
            h_attn = self._attention_pool(H, key_padding_mask)
            # gate ∈ (0,1) per-dim
            g = torch.sigmoid(self.gate(torch.cat([h_last, h_attn, h_last - h_attn], dim=-1)))
            h_ctx = g * h_last + (1.0 - g) * h_attn

        return H, h_ctx


# =========================
# 5) Weekly graph container
# =========================
@dataclass
class WeeklyGraph:
    H_user: torch.Tensor
    H_geo: torch.Tensor
    H_cat: torch.Tensor
    poi_idx: torch.Tensor         # [n_week]
    cat_idx: torch.Tensor         # [n_week]
    geo: torch.Tensor             # [n_week, 2]


# =========================
# 6) model (仅 POI 打分)
# =========================
class NextPOIModel(nn.Module):
    def __init__(
        self,
        num_pois: int,
        num_cats: int,
        d_model: int = 128,
        num_hyper_layers: int = 2 ,
        attn_drop: float = 0.1,
        use_l2norm: bool = True,
        seq_context_mode: str = "attn",   # 'last' | 'attn' | 'gate'
    ):
        super().__init__()
        self.num_pois = num_pois
        self.num_cats = num_cats
        self.d_model = d_model
        self.use_l2norm = use_l2norm

        self.feat = POIFeatureEncoder(num_pois, num_cats, d_model)
        self.hg = TypeHGNN(d_model, num_layers=num_hyper_layers, types=("user", "geo", "cat"), attn_drop=attn_drop)
        self.seq_encoder = LSTMSeqEncoder(d_model, num_layers=2, dropout=attn_drop, context_mode=seq_context_mode)

        # POI head: query & candidate projection
        self.proj_q = nn.Linear(d_model, d_model, bias=False)
        self.proj_c = nn.Linear(d_model, d_model, bias=False)

        self.global_gru = nn.GRUCell(d_model, d_model)
        self.poi_bias = nn.Parameter(torch.zeros(num_pois))

        for m in [self.proj_q, self.proj_c]:
            nn.init.xavier_uniform_(m.weight)

        # runtime state/cache (not parameters)
        self.Z_global: Optional[torch.Tensor] = None                 # [N, d], built by build_global()
        self._Z_weeks_cache: Dict[int, torch.Tensor] = {}            # week_id -> [N, d]

        # ========== learnable temperature (logit scale) ==========
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1/0.03), dtype=torch.float32))

    @staticmethod
    def _l2norm_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        return x / torch.clamp(x.norm(dim=1, keepdim=True), min=eps)

    def _normalize_if_needed(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1) if self.use_l2norm else x

    def _current_scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(1e-2, 100.0)

    def scaled_dot(self, Q: torch.Tensor, Z_pos: torch.Tensor) -> torch.Tensor:
        Qn = self._normalize_if_needed(Q)
        Zn = self._normalize_if_needed(Z_pos)
        return (Qn * Zn).sum(dim=1) * self._current_scale()

    def scaled_matmul(self, Q: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """
        矩阵乘 + 温度：返回 [B, C]
        Q: [B, d], Z: [C, d] 或 [B, C, d]
        """
        Qn = self._normalize_if_needed(Q)
        if Z.dim() == 2:
            Zn = self._normalize_if_needed(Z)               # [C, d]
            logits = Qn @ Zn.T                               # [B, C]
        else:
            # Z: [B, C, d]
            Zn = self._normalize_if_needed(Z)               # [B, C, d]
            logits = torch.einsum("bd,bcd->bc", Qn, Zn)     # [B, C]
        return logits * self._current_scale()

    def clear_week_cache(self) -> None:
        self._Z_weeks_cache.clear()

    # ------- Weekly encoding & alignment (keeps graph) -------
    def encode_week(self, wg: WeeklyGraph) -> torch.Tensor:
        """
        Encode a week and align to full catalog size [N, d].
        Unseen POIs for that week are zeros.
        """
        X0 = self.feat(wg.poi_idx, wg.cat_idx, wg.geo)
        Z_local, _, _ = self.hg({"user": wg.H_user, "geo": wg.H_geo, "cat": wg.H_cat}, X0)
        N, d = self.num_pois, Z_local.size(1)
        Z_full = Z_local.new_zeros(N, d)
        Z_full.index_copy_(0, wg.poi_idx.long(), Z_local)
        return Z_full

    def _get_week_Z(self, weeks: Dict[int, WeeklyGraph], wk: int, device: torch.device) -> torch.Tensor:
        """
        Return week-aligned embeddings [N, d].
        If cached, reuse; if not, compute and cache.
        """
        if wk not in self._Z_weeks_cache:
            Zw = self.encode_week(weeks[wk]).to(device)
            self._Z_weeks_cache[wk] = Zw
        return self._Z_weeks_cache[wk].to(device)

    # ------- Build Z_global (E2E if detach=False) -------
    def build_global(
        self,
        weeks: Dict[int, WeeklyGraph],
        device: Optional[torch.device] = None,
        week_ids: Optional[Sequence[int]] = None,
        detach: bool = False,
    ) -> torch.Tensor:
        """
        Build Z_global by iterating weeks in time order.
        ▪ detach=False: End-to-end training (keeps graph, backprop through GRU & hypergraphs).
        ▪ detach=True : Inference/fast mode (detaches to save memory).
        Returns: Z_global [N, d]
        """
        if device is None:
            device = next(self.parameters()).device

        # reset global state
        Zg = torch.zeros(self.num_pois, self.d_model, device=device)
        self.Z_global = Zg
        self._Z_weeks_cache = {}  # rebuild cache to keep graph consistent

        if week_ids is None:
            week_ids = sorted(weeks.keys())

        for wk in week_ids:
            # 编码当前周的poi表示
            Zw = self.encode_week(weeks[int(wk)]).to(device)   # [N, d]
            self._Z_weeks_cache[int(wk)] = Zw

            # mask: which rows appear this week
            mask_row = (Zw.abs().sum(dim=1) > 0)               # [N]
            if not mask_row.any():
                continue

            idx = mask_row.nonzero(as_tuple=False).squeeze(1)  # [M]

            x_t = Zw.index_select(0, idx)                      # [M, d]
            h_prev_rows = self.Z_global.index_select(0, idx)   # [M, d]

            # GRU update on active rows
            h_new_rows = self.global_gru(x_t, h_prev_rows)     # [M, d]
            if self.use_l2norm:
                h_new_rows = self._l2norm_rows(h_new_rows)

            # write updates into a copy tensor to avoid in-place on graph-critical tensor
            updates_full = torch.zeros_like(self.Z_global)
            updates_full.index_copy_(0, idx, h_new_rows)
            # combine: active rows take updates; others keep previous
            self.Z_global = torch.where(mask_row[:, None], updates_full, self.Z_global)

        if detach:
            self.Z_global = self.Z_global.detach()

        return self.Z_global

    # ------- Pull seq embeddings from per-week Z_full -------
    def _gather_sequence_embeddings(
        self,
        weeks: Dict[int, WeeklyGraph],
        seq_pois: torch.Tensor,   # [B, T]
        seq_weeks: torch.Tensor,  # [B, T]
    ) -> torch.Tensor:
        """
        Pull embeddings from cached per-week Z_full into a [B, T, d] tensor.
        Requires that the weeks used here were either cached during build_global()
        or will be computed on-the-fly.
        """
        device = seq_pois.device
        B, T = seq_pois.shape
        out = torch.zeros(B, T, self.d_model, device=device)
        valid = (seq_pois >= 0)
        if not valid.any():
            return out

        uniq_weeks = torch.unique(seq_weeks[valid]).tolist()
        cache = {int(wk): self._get_week_Z(weeks, int(wk), device) for wk in uniq_weeks}

        idx = valid.nonzero(as_tuple=False)  # [M, 2] -> (b, t)
        wk = seq_weeks[idx[:, 0], idx[:, 1]].tolist()
        pois = seq_pois[idx[:, 0], idx[:, 1]].long()
        for k, (b, t) in enumerate(idx.tolist()):
            out[b, t] = cache[int(wk[k])][pois[k]]
        return out

    # ------- POI scoring -------
    def forward(
        self,
        weeks: Dict[int, WeeklyGraph],
        seq_pois: torch.Tensor,           # [B, T]
        seq_weeks: torch.Tensor,          # [B, T]
        candidates: torch.Tensor,         # [B, M]
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Score given candidates for each sequence (POI head only).
        NOTE: call build_global(..., detach=False) before forward during training.
        Returns: scores [B, M]
        """
        assert self.Z_global is not None, "Call build_global(...) to build Z_global first."

        # (1) sequence -> query
        X_seq = self._gather_sequence_embeddings(weeks, seq_pois, seq_weeks)  # [B, T, d]
        _, h_ctx = self.seq_encoder(X_seq, key_padding_mask=key_padding_mask) # attention context (or last)
        Q = self.proj_q(h_ctx)                                                # [B, d]

        # (2) candidates from global catalog (vectorized)
        Zc = self.proj_c(self.Z_global.to(Q.device)).contiguous()             # [N, d]
        B, M = candidates.shape
        cand_flat = candidates.reshape(-1).to(dtype=torch.long, device=Q.device)   # [B*M]
        Zcand_flat = Zc.index_select(0, cand_flat)                                # [B*M, d]
        Zcand = Zcand_flat.reshape(B, M, -1)
        bias = self.poi_bias.to(Q.device).index_select(0, cand_flat).reshape(B, M)
        # [B, M, d]

        # (3) 统一：归一化 + 温度
        scores = self.scaled_matmul(Q, Zcand) + bias                                   # [B, M]
        return scores
