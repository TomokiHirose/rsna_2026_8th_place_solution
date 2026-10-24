from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rsna_knee.config.config import SeqConfig

__all__ = ["NeighborConv", "SeqBranch", "neighbor_index"]


def neighbor_index(window_slot: torch.Tensor, window_pos: torch.Tensor, window_mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """各窓の左隣・右隣（同じ Bag・同じ Slot で ``pos`` が 1 小さい / 大きい実在窓）の番号 ``[B, K]`` と、隣があるかの ``[B, K]`` を返す。

    隣が無い窓の番号は 0。同じ隣が複数あれば先頭を採る。
    """
    real = torch.ones_like(window_slot, dtype=torch.bool) if window_mask is None else window_mask
    # [B, K(i), K(j)]: 窓 j が窓 i の隣の候補か
    candidate = (window_slot[:, :, None] == window_slot[:, None, :]) & real[:, None, :] & real[:, :, None]
    step = window_pos[:, None, :] - window_pos[:, :, None]
    left, right = candidate & (step == -1), candidate & (step == 1)
    return left.int().argmax(dim=-1), right.int().argmax(dim=-1), left.any(dim=-1), right.any(dim=-1)


class NeighborConv(nn.Module):
    """``W_c x_i + W_l x_left(i) + W_r x_right(i)``。隣が無い側は 0 を入れる。"""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.center = nn.Linear(width, width, bias=False)
        self.left = nn.Linear(width, width, bias=False)
        self.right = nn.Linear(width, width, bias=False)

    def forward(self, x: torch.Tensor, left: torch.Tensor, right: torch.Tensor, has_left: torch.Tensor, has_right: torch.Tensor) -> torch.Tensor:
        """``x [B, K, W]`` と :func:`neighbor_index` の返り -> ``[B, K, W]``。"""
        width = x.shape[-1]
        x_left = torch.gather(x, 1, left[..., None].expand(-1, -1, width)) * has_left[..., None].to(x.dtype)
        x_right = torch.gather(x, 1, right[..., None].expand(-1, -1, width)) * has_right[..., None].to(x.dtype)
        return self.center(x) + self.left(x_left) + self.right(x_right)


class SeqBranch(nn.Module):
    """LayerNorm 後の窓特徴の差分 ``Δh`` を返す。

    流れは h → inp → [近傍 Conv → LayerNorm → GELU] × layers → ゼロ初期化の out → Δh。
    """

    def __init__(self, cfg: SeqConfig, feat_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.inp = nn.Linear(feat_dim, cfg.width)
        self.convs = nn.ModuleList(NeighborConv(cfg.width) for _ in range(cfg.layers))
        self.norms = nn.ModuleList(nn.LayerNorm(cfg.width) for _ in range(cfg.layers))
        self.out = nn.Linear(cfg.width, feat_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, h: torch.Tensor, window_mask: torch.Tensor | None, window_slot: torch.Tensor, window_pos: torch.Tensor) -> torch.Tensor:
        """LayerNorm 後の窓特徴 ``h [B, K, D]`` -> ``Δh [B, K, D]``。fp32 で計算し、padding 窓の ``Δh`` は 0。"""
        neighbors = neighbor_index(window_slot, window_pos, window_mask)
        with torch.autocast(device_type=h.device.type, enabled=False):
            x = self.inp(h.float())
            for conv, norm in zip(self.convs, self.norms, strict=True):
                x = F.gelu(norm(conv(x, *neighbors)))
            delta = self.out(x)
            if window_mask is not None:
                delta = delta * window_mask[..., None].to(delta.dtype)
            return delta
