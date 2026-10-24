from __future__ import annotations

import math

import torch
import torch.nn as nn

from rsna_knee.constants import NUM_LABELS
from rsna_knee.data.slots import NUM_SLOTS

__all__ = ["SpatialFindingPool"]


class SpatialFindingPool(nn.Module):
    """特徴マップ ``[N, F, h, w]`` を所見 x Slot の query で位置ごとに重み付けして平均し、所見ごとの窓特徴 ``[N, 12, F]`` にする。

    key は特徴の射影と位置の MLP の和、query は窓の Slot のもの。query は 0 から始まるので、最初は空間平均（GAP）と同じ。
    """

    def __init__(self, feat_dim: int, dim: int) -> None:
        super().__init__()
        self.key = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, dim))
        self.pos = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.query = nn.Parameter(torch.zeros(NUM_SLOTS, NUM_LABELS, dim))

    def forward(self, maps: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
        n, f, gh, gw = maps.shape
        content = maps.flatten(2).transpose(1, 2)
        ys, xs = torch.meshgrid(torch.linspace(-1.0, 1.0, gh, device=maps.device), torch.linspace(-1.0, 1.0, gw, device=maps.device), indexing="ij")
        keys = self.key(content) + self.pos(torch.stack([xs, ys], dim=-1).view(1, gh * gw, 2).to(content.dtype))
        scores = torch.einsum("ntd,npd->ntp", self.query[slot].to(keys.dtype), keys).float() / math.sqrt(keys.shape[-1])
        return torch.einsum("ntp,npf->ntf", scores.softmax(dim=-1).to(content.dtype), content)
