from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rsna_knee.config.config import ContextConfig

__all__ = ["ContextBranch"]


class ContextBranch(nn.Module):
    """所見ごとの窓特徴 ``L`` に、実在窓への self-attention 1 層で作った窓間の文脈 ``m`` で条件付けた残差 ``Δ`` を返す枝。

    ``Δ_kc = W_out(GELU(W_L LN(L_kc) + W_h m_k))``。``W_out`` はゼロ初期化なので学習開始時の ``Δ`` は 0。
    """

    def __init__(self, cfg: ContextConfig, gap_dim: int, head_dim: int) -> None:
        super().__init__()
        self.inp = nn.Linear(gap_dim, cfg.dim)
        self.attn_norm = nn.LayerNorm(cfg.dim)
        self.attn = nn.MultiheadAttention(cfg.dim, cfg.heads, batch_first=True)
        self.finding_norm = nn.LayerNorm(head_dim)
        self.finding_in = nn.Linear(head_dim, cfg.hidden)
        self.context_in = nn.Linear(cfg.dim, cfg.hidden, bias=False)
        self.out = nn.Linear(cfg.hidden, head_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, h: torch.Tensor, per_finding: torch.Tensor, window_mask: torch.Tensor | None) -> torch.Tensor:
        """``h [B, K, gap_dim]`` と ``per_finding [B, K, 12, head_dim]`` -> ``Δ [B, K, 12, head_dim]``。fp32 で計算し、実在窓だけを混ぜ、padding 窓の ``Δ`` は 0。"""
        real = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device) if window_mask is None else window_mask
        with torch.autocast(device_type=h.device.type, enabled=False):
            p = self.inp(h.float())
            x = self.attn_norm(p)
            m = p + self.attn(x, x, x, key_padding_mask=~real, need_weights=False)[0]
            x = self.finding_in(self.finding_norm(per_finding.float())) + self.context_in(m)[:, :, None, :]
            delta = self.out(F.gelu(x))
            return torch.where(real[:, :, None, None], delta, 0.0)
