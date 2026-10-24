from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rsna_knee.config.config import FsxConfig
from rsna_knee.constants import NUM_LABELS
from rsna_knee.data.slots import NUM_SLOTS

__all__ = ["FSX_HEADS", "FSX_RANK", "FSX_WIDTH", "FindingSlotFilm", "slot_features"]

#: cross-attention の幅・head 数と、所見別 query adapter の rank。
FSX_WIDTH = 128
FSX_HEADS = 2
FSX_RANK = 8


def slot_features(h: torch.Tensor, window_mask: torch.Tensor | None, window_slot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """窓特徴 ``[B, K, D]`` を Slot ごとに実在窓だけで平均し、``(g [B, S, D], present [B, S])`` を返す。"""
    member = F.one_hot(window_slot, NUM_SLOTS).to(h.dtype)
    if window_mask is not None:
        member = member * window_mask[..., None].to(h.dtype)
    counts = member.sum(dim=1)
    g = torch.einsum("bks,bkd->bsd", member, h) / counts.clamp_min(1.0)[..., None]
    return g, counts > 0


class FindingSlotFilm(nn.Module):
    """所見 query が Slot 特徴に cross-attention し、その context で所見特徴を FiLM 変調した logits の差分を上限付きで足す。"""

    def __init__(self, cfg: FsxConfig, feat_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.finding_norm = nn.LayerNorm(feat_dim)
        self.slot_norm = nn.LayerNorm(feat_dim)
        self.slot_identity = nn.Parameter(torch.zeros(NUM_SLOTS, feat_dim))
        nn.init.trunc_normal_(self.slot_identity, std=0.02)
        self.q = nn.Linear(feat_dim, FSX_WIDTH)
        self.k = nn.Linear(feat_dim, FSX_WIDTH)
        self.v = nn.Linear(feat_dim, FSX_WIDTH)
        self.o = nn.Linear(FSX_WIDTH, FSX_WIDTH)
        for layer in (self.q, self.k, self.v, self.o):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        # 所見ごとの係数で混ぜる rank FSX_RANK の query adapter。係数が 0 の初期状態では q に何も足さない。
        self.q_down = nn.Linear(feat_dim, FSX_RANK, bias=False)
        self.q_up = nn.Linear(FSX_RANK, FSX_WIDTH, bias=False)
        nn.init.xavier_uniform_(self.q_down.weight)
        nn.init.xavier_uniform_(self.q_up.weight)
        self.q_coeff = nn.Parameter(torch.zeros(NUM_LABELS, FSX_RANK))
        self.finding_identity = nn.Parameter(torch.zeros(NUM_LABELS, FSX_WIDTH))
        self.finding_slot_bias = nn.Parameter(torch.zeros(NUM_LABELS, NUM_SLOTS))
        # 温度は正に保つため log で持つ。初期値 0 は scale 1。
        self.log_scale = nn.Parameter(torch.zeros(()))
        self.film_scale = nn.Linear(FSX_WIDTH, feat_dim)
        self.film_shift = nn.Linear(FSX_WIDTH, feat_dim)
        for layer in (self.film_scale, self.film_shift):
            nn.init.normal_(layer.weight, std=1e-3)
            nn.init.zeros_(layer.bias)

    def attention(self, finding: torch.Tensor, g: torch.Tensor, present: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """所見特徴 ``[B, 12, D]`` と Slot 特徴 ``[B, S, D]`` -> ``(context [B, 12, W], attention [B, H, 12, S])``。欠けた Slot は softmax から除く。"""
        b = finding.shape[0]
        f = self.finding_norm(finding)
        s = self.slot_norm(g + self.slot_identity)
        q = self.q(f) + self.finding_identity + self.q_up(self.q_down(f) * self.q_coeff)
        head_dim = FSX_WIDTH // FSX_HEADS
        q = F.normalize(q.view(b, NUM_LABELS, FSX_HEADS, head_dim).transpose(1, 2), dim=-1, eps=1e-6)
        k = F.normalize(self.k(s).view(b, NUM_SLOTS, FSX_HEADS, head_dim).transpose(1, 2), dim=-1, eps=1e-6)
        v = self.v(s).view(b, NUM_SLOTS, FSX_HEADS, head_dim).transpose(1, 2)
        logits = self.log_scale.exp() * q @ k.transpose(-1, -2) + self.finding_slot_bias
        attention = logits.masked_fill(~present[:, None, None, :], -torch.inf).softmax(dim=-1)
        context = (attention @ v).transpose(1, 2).reshape(b, NUM_LABELS, FSX_WIDTH)
        return self.o(context), attention

    def forward(self, pooled: torch.Tensor, h: torch.Tensor, window_mask: torch.Tensor | None, window_slot: torch.Tensor, cls_w: torch.Tensor, cls_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """所見特徴 ``pooled`` と窓特徴 ``h`` -> ``(logits [B, 12], attention [B, H, 12, S])``。fp32 で計算する。"""
        with torch.autocast(device_type=h.device.type, enabled=False):
            pooled, h = pooled.float(), h.float()
            g, present = slot_features(h, window_mask, window_slot)
            context, attention = self.attention(pooled, g, present)
            modulated = pooled * (1.0 + 0.25 * torch.tanh(self.film_scale(context))) + 0.25 * torch.tanh(self.film_shift(context))
            base = (pooled * cls_w).sum(-1) + cls_b
            film = (modulated * cls_w).sum(-1) + cls_b
            cap = self.cfg.cap_train if self.training else self.cfg.cap_eval
            return base + cap * torch.tanh((film - base) / cap), attention
