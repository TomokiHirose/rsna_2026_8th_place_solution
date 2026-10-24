from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np
import torch

__all__ = ["build_windows", "build_windows_on", "gather_windows", "pick_window_runs", "pick_windows", "segment_window_indices", "select_window_indices", "slot_of_rows", "stack_window_indices", "window_indices"]


def slot_of_rows(rows: np.ndarray, slot_table: Sequence[int]) -> np.ndarray:
    """Stack 上のスライス番号を Slot 番号に写す。"""
    return np.searchsorted(np.cumsum(slot_table), np.asarray(rows), side="right")


def window_indices(presence: np.ndarray, slot_table: Sequence[int], context: int = 3) -> np.ndarray:
    """Slot 内の連続実在区間ごとに窓の添字 ``[N, C]`` を作り、区間端の隣接スライスは端で複製する。"""
    mask = np.asarray(presence)
    if not slot_table or any(int(n) != n or n <= 0 for n in slot_table) or sum(slot_table) != len(mask):
        raise ValueError("slot_table must match presence length")
    stops = np.cumsum(slot_table)
    return segment_window_indices(mask, list(zip(stops - np.asarray(slot_table), stops)), context)[0]


def segment_window_indices(presence: np.ndarray, segments: Sequence[tuple[int, int]], context: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Stack 上の区間 ``[start, stop)`` ごとに、連続実在区間から窓の添字 ``[N, C]`` と各窓の区間番号 ``[N]`` を作る。"""
    mask = np.asarray(presence)
    if mask.ndim != 1 or not np.isfinite(mask).all():
        raise ValueError("presence must be a finite vector")
    if context < 1 or context % 2 == 0:
        raise ValueError(f"context must be positive and odd, got {context}")
    rows, owner = [], []
    delta = np.arange(-(context // 2), context // 2 + 1)
    for i, (start, stop) in enumerate(segments):
        valid = np.flatnonzero(mask[start:stop] != 0) + start
        if len(valid):
            for run in np.split(valid, np.flatnonzero(np.diff(valid) != 1) + 1):
                rows.append(np.clip(run[:, None] + delta, run[0], run[-1]))
                owner.append(np.full(len(run), i, dtype=np.int64))
    if not rows:
        raise ValueError("study has no usable slices")
    return np.concatenate(rows).astype(np.int64, copy=False), np.concatenate(owner)


def pick_windows(n: int, k: int | None, train: bool = False) -> np.ndarray:
    """``n`` 本の窓から使う窓の番号を返す。学習は固定 K 本を無作為に（足りなければ重複して）、評価は重複なしで最大 K 本を等間隔に。``None`` は全窓。"""
    if k is not None and k < 1:
        raise ValueError("k must be positive or None")
    if train:
        if k is None:
            raise ValueError("training requires a fixed k")
        if n >= k:
            chosen = random.sample(range(n), k)
        else:
            chosen = list(range(n)) + random.choices(range(n), k=k - n)
            random.shuffle(chosen)
        return np.asarray(chosen, dtype=np.int64)
    if k is None or n <= k:
        return np.arange(n)
    return np.linspace(0, n - 1, k).round().astype(int)


def pick_window_runs(index: np.ndarray, owner: np.ndarray, k: int) -> np.ndarray:
    """学習用に、窓の添字 ``[N, C]`` と Slot 番号 ``[N]`` から、窓を持つ Slot ごとに 1 本の連続区間で K 本を選び、窓の番号を返す。

    連続とは同じ Slot で中心の行番号が 1 ずつ増える並び。K を Slot 数で割り、余りは無作為な Slot に 1 ずつ足す。
    その長さが取れない Slot は最長の連続区間を採り、不足は長さに空きのある Slot のうち最も短いものへ 1 ずつ回し、それでも足りない分は重複で埋めて shuffle する。
    """
    if k < 1:
        raise ValueError("k must be positive")
    center = index[:, index.shape[1] // 2]
    link = (owner[1:] == owner[:-1]) & (np.diff(center) == 1)  # 窓 i と i + 1 が連続か
    pieces = np.split(np.arange(len(index)), np.flatnonzero(~link) + 1)
    slots = np.unique(owner)
    by_slot = [[piece for piece in pieces if owner[piece[0]] == s] for s in slots]
    cap = np.array([max(len(piece) for piece in group) for group in by_slot])
    want = np.full(len(slots), k // len(slots))
    want[random.sample(range(len(slots)), k % len(slots))] += 1
    short = int(np.maximum(want - cap, 0).sum())
    want = np.minimum(want, cap)
    while short > 0 and (want < cap).any():
        room = np.flatnonzero(want < cap)
        want[random.choice(room[want[room] == want[room].min()].tolist())] += 1
        short -= 1
    chosen: list[int] = []
    for group, n in zip(by_slot, want):
        if n > 0:
            piece, start = random.choice([(piece, start) for piece in group for start in range(len(piece) - n + 1)])
            chosen.extend(int(i) for i in piece[start : start + n])
    if len(chosen) < k:
        chosen += random.choices(chosen, k=k - len(chosen))
    random.shuffle(chosen)
    return np.asarray(chosen, dtype=np.int64)


def select_window_indices(presence: np.ndarray, slot_table: Sequence[int], k: int | None, context: int = 3, train: bool = False, train_runs: str = "window") -> np.ndarray:
    """学習は固定 K 窓を無作為に（``train_runs="slot"`` は Slot ごとに 1 本の連続区間で）選び、評価は重複なしで最大 K 窓を等間隔に選ぶ。``None`` は全窓。"""
    index = window_indices(presence, slot_table, context)
    if train and train_runs == "slot":
        return index[pick_window_runs(index, slot_of_rows(index[:, context // 2], slot_table), k)]
    return index[pick_windows(len(index), k, train)]


def build_windows(volume: np.ndarray, presence: np.ndarray, k: int | None, img_size: int, context: int = 3, train: bool = False, *, slot_table: Sequence[int]) -> torch.Tensor:
    """Stack から Slot 境界・欠損をまたがない ``[K, C, R, R]`` の Bag を CPU 上に作る。値域は ``[0, 1]``。"""
    return gather_windows(volume, stack_window_indices(volume, presence, k, img_size, context, train, slot_table))


def gather_windows(volume: np.ndarray, index: np.ndarray) -> torch.Tensor:
    """添字 ``[K, C]`` の窓を uint8 Stack から CPU 上に取り出して 255 で割る。"""
    return torch.from_numpy(np.ascontiguousarray(volume[index], dtype=np.float32) / 255.0)


def stack_window_indices(volume: np.ndarray, presence: np.ndarray, k: int | None, img_size: int, context: int, train: bool, slot_table: Sequence[int]) -> np.ndarray:
    """Stack の寸法を検め、窓が参照するスライスの添字を返す。"""
    if volume.ndim != 3 or tuple(volume.shape[1:]) != (img_size, img_size) or len(presence) != len(volume):
        raise ValueError(f"expected stack [S, {img_size}, {img_size}] and presence [S], got {volume.shape}")
    return select_window_indices(presence, slot_table, k, context, train)


def build_windows_on(
    volume: np.ndarray, presence: np.ndarray, k: int | None, img_size: int, context: int = 3, device: str | torch.device = "cpu", *, slot_table: Sequence[int], row_slots: Sequence[int] | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Slot 内の窓 Bag ``[K, C, R, R]``、各窓の Slot 番号 ``[K]``、中心スライスの Stack 行番号 ``[K]``、各チャネルの Stack 行番号 ``[K, C]`` を指定 device に作る。

    uint8 Stack は転送後に窓へ展開して 255 で割る。窓の Slot 番号は ``slot_table`` の区間から決めるが、``row_slots``（Stack の行ごとの Slot
    番号）があれば中心スライスの行の値を使う。そのとき Slot の決まらない行（負の値）に中心がある窓があれば例外にする。
    """
    index = stack_window_indices(volume, presence, k, img_size, context, False, slot_table)
    device = torch.device(device)
    stack = torch.from_numpy(np.array(volume, copy=True)).to(device)
    windows = stack[torch.as_tensor(index, device=device)].float() / torch.tensor(255.0, device=device)
    center = index[:, context // 2]
    rows = torch.as_tensor(index, dtype=torch.int64, device=device)
    if row_slots is None:
        slots = slot_of_rows(center, slot_table)
    else:
        slots = np.asarray(row_slots, dtype=np.int64)[center]
        if (slots < 0).any():
            raise ValueError("a window is centred on a stack row with no slot in row_slots")
    return windows, torch.as_tensor(slots, device=device), rows[:, context // 2], rows
