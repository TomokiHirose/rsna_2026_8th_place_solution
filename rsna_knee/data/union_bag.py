from __future__ import annotations

import random
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from rsna_knee.data.depth_prior import window_depths
from rsna_knee.data.slots import make_slots, slot_for_series
from rsna_knee.models.windows import segment_window_indices

if TYPE_CHECKING:
    from rsna_knee.modules.window_dataset import WindowStudyDataset

__all__ = ["cap_windows", "read_subseries", "sample_versions", "sample_windows", "study_windows", "version_depths", "version_pixels", "version_windows"]


def version_windows(masks: Sequence[np.ndarray], context: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """版の実在の印（bool、実在が 1 枚以上）の列から、窓ごとの ``(版の番号 [N], 版の中の中心の行 [N], 版の実在だけを版の順に詰めた画素の行 [N, C])``。窓は版をまたがない。"""
    owner, center, rows, packed = [], [], [], 0
    for v, mask in enumerate(masks):
        index, _ = segment_window_indices(mask, [(0, len(mask))], context)
        owner.append(np.full(len(index), v, dtype=np.int64))
        center.append(index[:, context // 2].astype(np.int64))
        rows.append(packed + (np.cumsum(mask) - 1)[index])
        packed += int(mask.sum())
    return np.concatenate(owner), np.concatenate(center), np.concatenate(rows).astype(np.int64)


def read_subseries(dataset: WindowStudyDataset) -> dict[int, dict[int, list[tuple[int, int]]]]:
    """``subseries.csv`` の全行を cache の study 行 -> unit -> ``(start, stop)`` の列（CSV の行順 = sub-series の番号順）にする。"""
    table = pd.read_csv(dataset.cache_dir / "subseries.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    out: dict[int, dict[int, list[tuple[int, int]]]] = {}
    for rec in table.itertuples():
        row = dataset.uid_index[rec.StudyInstanceUID][1]
        if dataset.units[row][int(rec.unit)]["SeriesInstanceUID"] != rec.SeriesInstanceUID:
            raise ValueError(f"{dataset.cache_dir / 'subseries.csv'}: unit {rec.unit} of {rec.StudyInstanceUID} is not {rec.SeriesInstanceUID}")
        out.setdefault(row, {}).setdefault(int(rec.unit), []).append((int(rec.start), int(rec.stop)))
    return out


def study_windows(dataset: WindowStudyDataset, row: int, subs: dict[int, list[tuple[int, int]]], sub_masks: np.ndarray | None, series: frozenset[str] | None = None, lat: int | None = None) -> dict[str, Any]:
    """study ``row`` の全 unit の全版の全窓。``subs`` は unit -> sub-series の ``(start, stop)``（空なら元の版だけ）。膝ごとの子 study は ``series`` で unit を絞り、``lat`` に子の左右を渡す。

    返り値は ``sources``（版の画素の在処 ``(kind, start, stop)`` の列。``"stack"`` は study の詰めた行、``"sub"`` は ``sub_vols`` の行で実在だけを使う）、``rows``（:func:`version_windows`）、
    版ごとの ``version_slot``（series の撮像面と液体強調）と ``version_series``（版の series の ``groups`` での番号）、series ごとの ``groups``（:func:`sample_versions` の入力）と、
    窓ごとの ``owner``（版の番号）・``center``・``u``・``u_sym``。実在の無い版は除く。
    """
    presence = np.asarray(dataset.masks[dataset.stack_offsets[row] : dataset.stack_offsets[row + 1]]) != 0
    counts = np.concatenate([[0], np.cumsum(presence)])
    slots = make_slots(dataset.slot_table)
    masks, sources, info, groups = [], [], [], []
    for j, unit in enumerate(dataset.units.get(row, [])):
        uid = str(unit["SeriesInstanceUID"])
        if series is not None and uid not in series:
            continue
        slot = slot_for_series(unit["Anatomical_Plane"], unit["Fluid_Sensitive"], slots).index
        first, cap = int(unit["stack_offset"]), int(unit["cap"])
        candidates = [(presence[first : first + cap], ("stack", int(counts[first]), int(counts[first + cap])), {})] + [(sub_masks[start:stop] != 0, ("sub", start, stop), {uid: (start, stop)}) for start, stop in subs.get(j, [])]
        versions = []
        for mask, where, swapped in candidates:
            if mask.any():
                versions.append(len(masks))
                masks.append(mask)
                sources.append(where)
                info.append((slot, uid, swapped))
        if versions:
            groups.append((slot, versions))
    owner, center, rows = version_windows(masks, dataset.window.context)
    u, u_sym = np.empty(len(owner), dtype=np.float32), np.zeros(len(owner), dtype=bool)
    for v, (slot, uid, swapped) in enumerate(info):
        members = owner == v
        u[members], u_sym[members] = dataset.prior_depths(row, np.zeros(int(members.sum()), dtype=np.int64), center[members], [slot], [uid], swapped, lat)
    version_slot = np.asarray([slot for slot, _, _ in info], dtype=np.int64)
    version_series = np.empty(len(masks), dtype=np.int64)
    for s, (_, versions) in enumerate(groups):
        version_series[versions] = s
    return {"owner": owner, "version_slot": version_slot, "version_series": version_series, "groups": groups, "center": center, "u": u, "u_sym": u_sym, "sources": sources, "rows": rows}


def version_depths(owner: np.ndarray, center: np.ndarray, units: Sequence[tuple[np.ndarray, int, int]]) -> np.ndarray:
    """窓ごとの深さ u ``[N]``。``units`` は版ごとの ``(Stack 行 j が span 内の何番目のスライスか, span 内の枚数 m, 向き)`` で、``center`` は版の中の中心の行。"""
    u = np.empty(len(owner), dtype=np.float32)
    for v, (rows, m, orientation) in enumerate(units):
        members = owner == v
        u[members] = window_depths(rows[center[members]], m, orientation)
    return u


def version_pixels(dataset: WindowStudyDataset, row: int, sources: list[tuple[str, int, int]], sub_vols: np.ndarray | None, sub_masks: np.ndarray | None) -> np.ndarray:
    """study の版の実在スライスを版の順に詰めた画素 ``[P, R, R]``（:func:`study_windows` の ``rows`` が指す）。"""
    volume = np.asarray(dataset._volume()[dataset.offsets[row] : dataset.offsets[row + 1]])
    return np.concatenate([volume[start:stop] if kind == "stack" else np.asarray(sub_vols[start:stop])[sub_masks[start:stop] != 0] for kind, start, stop in sources])


def cap_windows(owner: np.ndarray, version_slot: np.ndarray, cap: int) -> np.ndarray:
    """推論と検証の窓。``cap`` 以下なら全窓、超えたら Slot ごとの窓の比を保って（端数は余りの大きい Slot から）固定の乱数で ``cap`` 窓を採る。添字は昇順。"""
    n = len(owner)
    if n <= cap:
        return np.arange(n)
    slots = version_slot[owner]
    names, counts = np.unique(slots, return_counts=True)
    exact = cap * counts / n
    quota = np.floor(exact).astype(np.int64)
    for i in sorted(range(len(names)), key=lambda i: (-(exact[i] - quota[i]), names[i]))[: cap - int(quota.sum())]:
        quota[i] += 1
    rng = np.random.default_rng(0)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(slots == name), size=int(q), replace=False) for name, q in zip(names, quota)]))


def sample_versions(groups: Sequence[tuple[int, Sequence[int]]]) -> list[int]:
    """学習の版。``groups`` は series ごとの ``(Slot 番号, その series の版の番号)``。

    Slot ごとに series の本数を 1〜候補数から一様に引いて無作為に選び、選んだ series ごとに版の数を 1〜版数から一様に引いて無作為に選ぶ。版の番号は昇順で返す。
    """
    chosen: list[int] = []
    for slot in sorted({slot for slot, _ in groups}):
        candidates = [list(ids) for s, ids in groups if s == slot]
        for ids in random.sample(candidates, random.randint(1, len(candidates))):
            chosen.extend(random.sample(ids, random.randint(1, len(ids))))
    return sorted(chosen)


def sample_windows(n: int, k_min: int, k_max: int) -> np.ndarray:
    """学習の窓。``k_min``〜``k_max`` の一様な数（``n`` が少なければ全部）を重複なしで採る。添字は昇順。"""
    k = random.randint(k_min, k_max)
    return np.arange(n) if n <= k else np.sort(np.asarray(random.sample(range(n), k), dtype=np.int64))
