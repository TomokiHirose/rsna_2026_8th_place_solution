from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_ndtr, logsumexp

from rsna_knee.constants import LABELS
from rsna_knee.data.dicom import span_bounds
from rsna_knee.data.slots import SLOT_NAMES
from rsna_knee.models.windows import pick_windows

__all__ = ["finding_density_table", "interval_log_probs", "load_depth_prior", "normal_sign", "pick_prior_windows", "span_rows", "study_laterality", "weighted_sample", "window_depths", "window_intervals", "window_log_weights"]

_LATERALITY = {"R": 1, "RIGHT": 1, "L": -1, "LEFT": -1}


def study_laterality(meta: pd.DataFrame) -> dict[str, int]:
    """study ごとの左右。全 series の欠損を除いた Laterality が R だけなら +1、L だけなら -1、空・混在・B は 0。"""
    sides = {uid: {_LATERALITY.get(str(v).strip().upper(), 0) for v in values.dropna()} for uid, values in meta.groupby("StudyInstanceUID")["Laterality"]}
    return {uid: seen.pop() if len(seen) == 1 and 0 not in seen else 0 for uid, seen in sides.items()}


def normal_sign(iop: str) -> int:
    """IOP の法線 row x col の支配成分の符号。Stack はこの法線への射影の昇順に並ぶので、掛けると患者座標の昇順になる。"""
    v = np.array([float(x) for x in str(iop).split("|")])
    n = np.cross(v[:3], v[3:])
    return int(np.sign(n[np.argmax(np.abs(n))]))


def _read_prior(path: str | Path) -> dict:
    """prior の JSON を読み、版と打ち切り区間を検める。"""
    data = json.loads(Path(path).read_text())
    if data["version"] != 1 or [float(x) for x in data["truncate"]] != [-1.0, 1.0]:
        raise ValueError(f"{path}: expected version 1 truncated to [-1, 1], got version {data['version']} truncate {data['truncate']}")
    return data


def load_depth_prior(path: str | Path) -> dict[str, np.ndarray]:
    """prior の JSON を Slot 名 -> 成分 ``[J, 3]``（weight, mean, sigma）にする。``findings`` は読まない。"""
    data = _read_prior(path)
    return {name: np.array([[c["weight"], c["mean"], c["sigma"]] for c in slot["components"]], dtype=np.float64) for name, slot in data["slots"].items()}


def finding_density_table(path: str | Path, n_grid: int = 401) -> np.ndarray:
    """Slot x 所見の GMM 密度を u の格子 ``linspace(-1, 1, n_grid)`` で評価し、最大値で割って [0, 1] にした ``[2, len(SLOT_NAMES), 12, n_grid]``。

    先頭の軸の 0 は prior そのもの、1 は左右対称化した ``(g(u) + g(-u)) / 2`` を同じく最大値で割ったもの。
    その Slot に所見の成分が無いか、Slot が prior に無ければ 0。
    """
    data = _read_prior(path)
    grid = np.linspace(-1.0, 1.0, n_grid)
    table = np.zeros((2, len(SLOT_NAMES), len(LABELS), n_grid), dtype=np.float64)
    for name, slot in data["slots"].items():
        for c in slot["components"]:
            density = c["weight"] * np.exp(-0.5 * ((grid - c["mean"]) / c["sigma"]) ** 2) / c["sigma"]
            table[0, SLOT_NAMES.index(name), LABELS.index(c["finding"])] += density
    table[1] = (table[0] + table[0][..., ::-1]) / 2.0  # 格子は 0 について対称なので、反転が g(-u)
    peak = table.max(axis=-1, keepdims=True)
    return np.divide(table, peak, out=np.zeros_like(table), where=peak > 0).astype(np.float32)


def span_rows(selected: list[int], n_files: int, span: tuple[float, float]) -> tuple[np.ndarray, int]:
    """cache が採ったスライスの元の番号を span 内の番号 ``idx - lo`` と span 内の枚数 m にする。"""
    lo, hi = span_bounds(n_files, span)
    return np.asarray(selected, dtype=np.int64) - lo, hi - lo + 1


def window_depths(rows: np.ndarray, m: int, orientation: int) -> np.ndarray:
    """span 内の番号 ``rows`` のスライスの符号つき正規化深さ u ``-1 + (2j + 1) / m``（:func:`window_intervals` と同じ座標）。``orientation == -1`` は符号を反転する。"""
    return orientation * (-1.0 + (2.0 * np.asarray(rows, dtype=np.float64) + 1.0) / m)


def _log_gauss_mass(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """標準正規の区間 ``[lo, hi]`` の確率の対数。裾で引き算が桁落ちしないよう、上側の裾は対称な下側の裾に折り返す。"""
    upper = lo > 0
    lo, hi = np.where(upper, -hi, lo), np.where(upper, -lo, hi)
    log_hi = log_ndtr(hi)
    return log_hi + np.log1p(-np.exp(log_ndtr(lo) - log_hi))


def interval_log_probs(components: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """[-1, 1] で打ち切って再正規化した混合ガウスの、区間 ``[a, b]`` ごとの確率の対数。"""
    w, m, s = components.T
    a, b = np.asarray(a, dtype=np.float64)[:, None], np.asarray(b, dtype=np.float64)[:, None]
    mass = logsumexp(_log_gauss_mass((a - m) / s, (b - m) / s), b=w, axis=1)
    return mass - logsumexp(_log_gauss_mass((-1.0 - m) / s, (1.0 - m) / s), b=w)


def window_intervals(rows: np.ndarray, m: int, orientation: int) -> tuple[np.ndarray, np.ndarray]:
    """span 内の番号 ``rows``（昇順）のスライスを中心とする窓が受け持つ深さ u の区間 ``[a, b]``。``orientation == -1`` は ``[-b, -a]`` に反転する。

    span の [-1, 1] を m 等分し、番号 j の中心は ``-1 + (2j + 1) / m``。窓の境は隣の窓の中心との中点で、両端は ±1。
    """
    # 半スライス単位の整数で中点を取る。中心 2j + 1 と 2k + 1 の中点は j + k + 1。
    rows = np.asarray(rows, dtype=np.float64)
    edges = -1.0 + np.concatenate([[0.0], rows[:-1] + rows[1:] + 1.0, [2.0 * m]]) / m
    a, b = edges[:-1], edges[1:]
    return (a, b) if orientation == 1 else (-b, -a)


def window_log_weights(components: np.ndarray, rows: np.ndarray, m: int, orientation: int, symmetric: bool, alpha: float) -> np.ndarray:
    """unit の ``n = len(rows)`` 本の窓の採取重み ``q = (1 - alpha) / n + alpha * q_prior`` の対数 ``[n]``。``q_prior`` は unit 内で和 1。

    区間は :func:`window_intervals`。``symmetric`` は prior を左右対称化した ``(P([a, b]) + P([-b, -a])) / 2`` を使う。
    """
    n = len(rows)
    a, b = window_intervals(rows, m, orientation)
    log_p = interval_log_probs(components, a, b)
    if symmetric:
        log_p = np.logaddexp(log_p, interval_log_probs(components, -b, -a)) - math.log(2.0)
    # alpha が 0 や 1 の端では片方の項が log(0) = -inf になり、logaddexp がもう片方だけを残す。
    with np.errstate(divide="ignore"):
        return np.logaddexp(np.log1p(-alpha) - math.log(n), np.log(alpha) + log_p - logsumexp(log_p))


def weighted_sample(log_q: np.ndarray, c: int) -> np.ndarray:
    """重み ``exp(log_q)`` で ``c`` 本を非復元に引く（指数分布の競争、Efraimidis-Spirakis と同値）。乱数は ``random``。"""
    race = np.log([random.expovariate(1.0) for _ in range(len(log_q))]) - log_q
    return np.argsort(race, kind="stable")[:c]


def pick_prior_windows(owner: np.ndarray, k: int, log_q: np.ndarray) -> np.ndarray:
    """学習用に ``N`` 本の窓から K 本の番号を返す。``N <= K`` は :func:`pick_windows` のまま。

    ``N > K`` は :func:`pick_windows` が引いた K 本の unit ごとの本数だけを使い、各 unit の中で窓ごとの重み ``exp(log_q)`` で引き直して shuffle する。
    """
    chosen = pick_windows(len(owner), k, train=True)
    if len(owner) <= k:
        return chosen
    picked: list[int] = []
    for s, c in zip(*np.unique(owner[chosen], return_counts=True)):
        members = np.flatnonzero(owner == s)
        picked.extend(int(i) for i in members[weighted_sample(log_q[members], int(c))])
    random.shuffle(picked)
    return np.asarray(picked, dtype=np.int64)
