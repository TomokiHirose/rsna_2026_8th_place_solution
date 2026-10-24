from __future__ import annotations

import math
import random
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from rsna_knee.config.config import AugConfig
from rsna_knee.data.slots import make_slots

__all__ = ["mirror_windows", "window_augment"]


def mirror_windows(windows: torch.Tensor, slots: np.ndarray, planes: Sequence[str]) -> tuple[torch.Tensor, np.ndarray]:
    """膝を反対側の膝に見せる。Coronal/Axial は列（患者の左右）を反転し、Sagittal は窓内のスライス順を反転する。

    返り値は ``(窓, スライス順を反転した窓の bool [K])``。

    Sagittal を面内で左右反転すると前後が入れ替わるので触らない。窓は位置埋め込みの無い MIL で集約されるので、
    窓内のチャネル順の反転はスライス順の反転と同じ。
    """
    out = windows.clone()
    flipped = np.array([planes[s] == "Sagittal" for s in np.asarray(slots)], dtype=bool)
    for i, depth in enumerate(flipped):
        out[i] = windows[i].flip(0) if depth else windows[i].flip(-1)
    return out, flipped


def _affine(x: torch.Tensor, degrees: float, scale: float, shift: tuple[float, float]) -> torch.Tensor:
    """画像中心まわりの回転・拡縮・並進。はみ出しは 0（MRI の背景）で埋める。"""
    a = math.radians(degrees)
    c, s = math.cos(a) / scale, math.sin(a) / scale  # 出力座標 -> 入力座標の写像なので逆変換
    theta = torch.tensor([[[c, -s, shift[0]], [s, c, shift[1]]]], dtype=x.dtype).expand(x.shape[0], 2, 3)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """分離可能なガウスぼかし。端は複製で埋める。"""
    radius = max(1, math.ceil(3 * sigma))
    t = torch.arange(-radius, radius + 1, dtype=x.dtype)
    kernel = torch.exp(-(t**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    n, c, h, w = x.shape
    y = x.reshape(n * c, 1, h, w)
    y = F.conv2d(F.pad(y, (radius, radius, 0, 0), mode="replicate"), kernel.view(1, 1, 1, -1))
    y = F.conv2d(F.pad(y, (0, 0, radius, radius), mode="replicate"), kernel.view(1, 1, -1, 1))
    return y.reshape(n, c, h, w)


def _series_transform(x: torch.Tensor, aug: AugConfig) -> torch.Tensor:
    """1 Slot 分の窓 ``[n, C, R, R]`` に、Slot で共有するパラメータで affine -> 低解像度 -> ぼかし -> ノイズを掛ける。"""
    size = x.shape[-1]
    if aug.affine_prob > 0 and random.random() < aug.affine_prob:
        degrees = random.uniform(-aug.rotate_deg, aug.rotate_deg)
        scale = random.uniform(*aug.affine_scale)
        # grid 座標は [-1, 1] で幅が 2 なので、画像幅に対する割合を 2 倍する。
        shift = (random.uniform(-aug.affine_translate, aug.affine_translate) * 2.0, random.uniform(-aug.affine_translate, aug.affine_translate) * 2.0)
        x = _affine(x, degrees, scale, shift)
    if aug.lowres_prob > 0 and random.random() < aug.lowres_prob:
        low = max(8, round(size * random.uniform(*aug.lowres_range)))
        x = F.interpolate(F.interpolate(x, size=(low, low), mode="area"), size=(size, size), mode="bilinear", align_corners=False)
    if aug.blur_prob > 0 and random.random() < aug.blur_prob:
        x = _gaussian_blur(x, random.uniform(*aug.blur_sigma))
    if aug.noise_prob > 0 and random.random() < aug.noise_prob:
        std = random.uniform(0.0, aug.noise_std)
        gen = torch.Generator().manual_seed(random.getrandbits(63))
        x = x + torch.randn(x.shape, generator=gen, dtype=x.dtype) * std
    return x.clamp(0.0, 1.0)


def window_augment(windows: torch.Tensor, slots: np.ndarray, slot_table: Sequence[int], aug: AugConfig) -> tuple[torch.Tensor, np.ndarray]:
    """窓 ``[K, C, R, R]`` に鏡像（Study 単位）-> Slot ごとの変換を掛け、``(窓, スライス順を反転した窓の bool [K])`` を返す。

    ``slots`` は各窓の Slot 番号 ``[K]``。

    同じ Slot（= 1 Series）の窓には同じパラメータを掛け、スライスごとに別の変換で偽の動きを作らない。
    乱数はすべて ``random`` モジュールから引き、無効な変換は乱数を引かない。
    """
    slots = np.asarray(slots)
    flipped = np.zeros(len(slots), dtype=bool)
    if aug.mirror_prob > 0 and random.random() < aug.mirror_prob:
        windows, flipped = mirror_windows(windows, slots, [slot.plane for slot in make_slots(slot_table)])
    out = windows.clone()
    for s in np.unique(slots):
        idx = torch.as_tensor(np.flatnonzero(slots == s))
        out[idx] = _series_transform(windows[idx], aug)
    return out, flipped
