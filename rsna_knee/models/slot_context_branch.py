from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rsna_knee.config.config import SlotContextConfig
from rsna_knee.constants import NUM_LABELS
from rsna_knee.data.slots import NUM_SLOTS

__all__ = ["SlotContextBranch"]


class SlotContextBranch(nn.Module):
    """所見ごとの窓特徴 ``F`` から Slot 内 -> Slot 間の所見別 attention で slot 文脈と study 文脈を作り、窓へ戻した残差 ``Δ`` を返す枝。

    ``Δ_kc = W_out(GELU(W_F LN(F_kc) + W_s Hslot_{s(k)c} + W_t Hstudy_c))``。``W_out`` はゼロ初期化なので学習開始時の ``Δ`` は 0。
    """

    def __init__(self, cfg: SlotContextConfig, head_dim: int) -> None:
        super().__init__()
        self.finding_norm = nn.LayerNorm(head_dim)
        self.proj = nn.Linear(head_dim, cfg.dim)
        self.slot_att = nn.Linear(cfg.dim, cfg.att_hidden)
        self.slot_w = nn.Parameter(nn.init.trunc_normal_(torch.zeros(NUM_LABELS, cfg.att_hidden), std=0.02))
        self.slot_emb = nn.Parameter(nn.init.trunc_normal_(torch.zeros(NUM_SLOTS, cfg.dim), std=0.02))
        self.slot_norm = nn.LayerNorm(cfg.dim)
        self.study_att = nn.Linear(cfg.dim, cfg.att_hidden)
        # 読み出しと定数を 0 から始めるので、学習開始時の Slot 間の重みは窓のある Slot の一様平均。
        self.study_w = nn.Parameter(torch.zeros(NUM_LABELS, cfg.att_hidden))
        self.study_b = nn.Parameter(torch.zeros(NUM_SLOTS, NUM_LABELS))
        self.finding_in = nn.Linear(head_dim, cfg.hidden)
        self.slot_in = nn.Linear(cfg.dim, cfg.hidden, bias=False)
        self.study_in = nn.Linear(cfg.dim, cfg.hidden, bias=False)
        self.out = nn.Linear(cfg.hidden, head_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, per_finding: torch.Tensor, window_mask: torch.Tensor | None, window_slot: torch.Tensor) -> torch.Tensor:
        """``per_finding [B, K, 12, head_dim]`` と ``window_slot [B, K]`` -> ``Δ [B, K, 12, head_dim]``。fp32 で計算し、padding 窓の ``Δ`` は 0。"""
        real = torch.ones(per_finding.shape[:2], dtype=torch.bool, device=per_finding.device) if window_mask is None else window_mask
        with torch.autocast(device_type=per_finding.device.type, enabled=False):
            x = self.finding_norm(per_finding.float())
            p = self.proj(x)  # [B, K, 12, d]
            member = (window_slot[..., None] == torch.arange(NUM_SLOTS, device=window_slot.device)) & real[..., None]  # [B, K, S]
            present = member.any(dim=1)  # [B, S]
            s1 = (torch.tanh(self.slot_att(p)) * self.slot_w).sum(-1)  # [B, K, 12]
            # 窓の無い Slot は全窓で一様にして有限に保ち、後で 0 を掛ける。
            logits1 = s1[:, :, None, :].masked_fill(~member[..., None], -torch.inf).masked_fill(~present[:, None, :, None], 0.0)
            a = logits1.softmax(dim=1) * present[:, None, :, None]  # [B, K, S, 12]
            hslot = torch.einsum("bksc,bkcd->bscd", a, p)  # [B, S, 12, d]
            q = self.slot_norm(hslot + self.slot_emb[None, :, None, :])
            s2 = (torch.tanh(self.study_att(q)) * self.study_w).sum(-1) + self.study_b  # [B, S, 12]
            beta = s2.masked_fill(~present[..., None], -torch.inf).softmax(dim=1)
            hstudy = torch.einsum("bsc,bscd->bcd", beta, q)  # [B, 12, d]
            slot_index = window_slot.clamp(0, NUM_SLOTS - 1)[:, :, None, None].expand(-1, -1, NUM_LABELS, hslot.shape[-1])
            hslot_k = hslot.gather(1, slot_index)  # [B, K, 12, d]
            delta = self.out(F.gelu(self.finding_in(x) + self.slot_in(hslot_k) + self.study_in(hstudy)[:, None]))
            return torch.where(real[:, :, None, None], delta, 0.0)
