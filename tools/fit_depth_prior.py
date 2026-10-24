from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import click
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from rsna_knee.constants import LABELS
from rsna_knee.data.depth_prior import interval_log_probs
from rsna_knee.data.dicom import span_count
from rsna_knee.data.slots import SLOT_NAMES
from rsna_knee.modules.labels import gold_ids

# [-1, 1] 上ではこれより広い成分はほぼ一様なので、sigma の上限にする。
SIGMA_CAP = 4.0


def load_windows(path: Path, cache: Path, train_csv: Path) -> pd.DataFrame:
    """窓 attention を読み、長い series と左右不明の Sagittal を外して ``gap_*`` を study x Slot 内で和 1 に割り直す。"""
    frame = pd.read_csv(path, dtype={"study_id": str, "series_uid": str})
    cols = [f"gap_{label}" for label in LABELS]
    span = tuple(json.loads((cache / "manifest.json").read_text())["span"])
    units = pd.read_csv(cache / "series.csv", usecols=["SeriesInstanceUID", "cap"], dtype={"SeriesInstanceUID": str})
    units = units.merge(pd.read_csv(cache / "series_meta.csv", usecols=["SeriesInstanceUID", "n_files"], dtype={"SeriesInstanceUID": str}), on="SeriesInstanceUID")
    long_series = set(units.loc[[span_count(int(n), span) > cap for n, cap in zip(units["n_files"], units["cap"])], "SeriesInstanceUID"])
    gold = gold_ids(train_csv)
    if not gold:
        raise ValueError(f"{train_csv}: no gold studies found")
    leaked = gold & set(frame["study_id"])
    if leaked:
        raise ValueError(f"{len(leaked)} gold studies are in {path}")
    sagittal = frame["slot"].map(lambda s: SLOT_NAMES[s].startswith("SAG"))
    frame = frame[~frame["series_uid"].isin(long_series) & (~sagittal | (frame["lat"] != 0))].copy()
    frame[cols] = frame[cols] / frame.groupby(["study_id", "slot"])[cols].transform("sum")
    print(f"{len(frame)} windows after dropping {len(long_series)} long series and Sagittal of unknown side", flush=True)
    return frame


def window_intervals(part: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """窓の区間の中心 c と unit の実在枚数 n。窓は unit の [-1, 1] を n 等分した区間 ``[c - 1/n, c + 1/n]`` を受け持つ。"""
    n = part["n_real"].to_numpy(dtype=np.float64)
    return part["u"].to_numpy(dtype=np.float64) * (n - 1) / n, n


def log_components(c: np.ndarray, pi: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """``[N, k]`` の log(pi_j N(c_i; mu_j, sigma_j))。"""
    z = (c[:, None] - mu[None, :]) / sigma[None, :]
    return np.log(pi)[None, :] - 0.5 * z**2 - np.log(sigma)[None, :] - 0.5 * math.log(2.0 * math.pi)


def fit_gmm(c: np.ndarray, w: np.ndarray, k: int, sigma_floor: float, tol: float, max_iter: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
    """区間の中心を点とみなした、打ち切りなしの重み付き 1 次元 GMM の EM。sigma は下限で止める。:func:`fit_truncated` の初期値に使う。

    Returns:
        (pi, mu, sigma, 重み付き対数尤度 sum w log p(c), 収束したか)。
    """
    order = np.argsort(c)
    cdf = np.cumsum(w[order]) - 0.5 * w[order]
    mu = np.interp((np.arange(k) + 0.5) / k, cdf, c[order])
    spread = math.sqrt(float(np.sum(w * (c - np.sum(w * c)) ** 2)))
    sigma = np.full(k, max(spread / k, sigma_floor))
    pi = np.full(k, 1.0 / k)
    previous = -math.inf
    for _ in range(max_iter):
        logp = log_components(c, pi, mu, sigma)
        top = logp.max(axis=1, keepdims=True)
        total = top[:, 0] + np.log(np.exp(logp - top).sum(axis=1))
        loglik = float(np.sum(w * total))
        if abs(loglik - previous) < tol:
            return pi, mu, sigma, loglik, True
        previous = loglik
        r = w[:, None] * np.exp(logp - total[:, None])
        mass = r.sum(axis=0)
        pi = mass / mass.sum()
        mu = (r * c[:, None]).sum(axis=0) / mass
        sigma = np.maximum(np.sqrt((r * ((c[:, None] - mu[None, :]) ** 2)).sum(axis=0) / mass), sigma_floor)
    return pi, mu, sigma, previous, False


def fit_truncated(a: np.ndarray, b: np.ndarray, w: np.ndarray, init: np.ndarray, sigma_floor: float) -> tuple[np.ndarray, float, bool]:
    """[-1, 1] で打ち切った混合の区間確率の重み付き対数尤度 ``L = sum w log P([a, b])`` を L-BFGS-B で最大化する。

    Returns:
        (成分 ``[k, 3]``（weight, mean, sigma）, L, 最適化が収束したか)。
    """
    k = len(init)

    def unpack(theta: np.ndarray) -> np.ndarray:
        # 重みは先頭の logit を 0 に固定した softmax、sigma は log で持つ。
        logits = np.concatenate([[0.0], theta[: k - 1]])
        weights = np.exp(logits - logits.max())
        return np.column_stack([weights / weights.sum(), theta[k - 1 : 2 * k - 1], np.exp(theta[2 * k - 1 :])])

    def loss(theta: np.ndarray) -> float:
        return -float(np.sum(w * interval_log_probs(unpack(theta), a, b)))

    pi = np.maximum(init[:, 0], 1e-12)
    theta = np.concatenate([np.log(pi[1:] / pi[0]), np.clip(init[:, 1], -1.5, 1.5), np.log(np.clip(init[:, 2], sigma_floor, SIGMA_CAP))])
    bounds = [(None, None)] * (k - 1) + [(-1.5, 1.5)] * k + [(math.log(sigma_floor), math.log(SIGMA_CAP))] * k
    result = minimize(loss, theta, method="L-BFGS-B", bounds=bounds)
    return unpack(result.x), -float(result.fun), bool(result.success)


def add_component(comp: np.ndarray, a: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
    """k - 1 成分の解に、経験の質量 / GMM の区間確率が最大の区間の中心へ sigma 0.15・重み 0.1 の成分を足す（既存の重みは 0.9 倍）。"""
    worst = int(np.argmax(np.log(w) - interval_log_probs(comp, a, b)))
    return np.vstack([comp * [0.9, 1.0, 1.0], [0.1, (a[worst] + b[worst]) / 2, 0.15]])


def as_array(components: list[dict]) -> np.ndarray:
    """JSON の成分の列を :func:`interval_log_probs` が取る ``[J, 3]``（weight, mean, sigma）にする。"""
    return np.array([[c["weight"], c["mean"], c["sigma"]] for c in components], dtype=np.float64)


def bin_ratio(components: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """打ち切り混合の区間 ``[edges[i], edges[i + 1]]`` の確率の、一様に対する倍率。"""
    return np.exp(interval_log_probs(components, edges[:-1], edges[1:])) / (np.diff(edges) / 2.0)


def fit_slot(part: pd.DataFrame, max_k: int, sigma_floor: float, tol: float, max_iter: int) -> tuple[dict, int, int]:
    """1 Slot の 12 所見を k = 1..max_k で fit し、BIC 最小の k の成分を所見等重みで並べる。

    Returns:
        (Slot の JSON, max-iter で止まった EM の数, 収束しなかった L-BFGS-B の数)。
    """
    n_eff = part["study_id"].nunique()
    cols = [f"gap_{label}" for label in LABELS]
    # 同じ区間（u と n が等しい窓）は尤度で区別されないので、重みを足して 1 つにまとめる。
    points = part.groupby(["u", "n_real"], as_index=False)[cols].sum()
    c, n = window_intervals(points)
    a, b = c - 1.0 / n, c + 1.0 / n
    components, findings, stalled, failed = [], {}, 0, 0
    for label in LABELS:
        w = points[f"gap_{label}"].to_numpy(dtype=np.float64) / n_eff
        fits, logliks, bics = [], [], []
        for k in range(1, max_k + 1):
            pi, mu, sigma, _, converged = fit_gmm(c, w, k, sigma_floor, tol, max_iter)
            if not converged:
                stalled += 1
                print(f"  EM did not converge: {label} k={k}", flush=True)
            # 初期値は打ち切りなし EM の解と、k >= 2 では k - 1 の解に 1 成分を足したもの。L の良い方を採る。
            inits = [np.column_stack([pi, mu, sigma])] + ([add_component(fits[-1], a, b, w)] if fits else [])
            candidates = [fit_truncated(a, b, w, init, sigma_floor) for init in inits]
            for _, _, success in candidates:
                if not success:
                    failed += 1
                    print(f"  L-BFGS-B did not converge: {label} k={k}", flush=True)
            comp, loglik, _ = max(candidates, key=lambda fit: fit[1])
            if logliks and loglik < logliks[-1] - 1e-9:
                raise RuntimeError(f"{label}: L decreased from k={k - 1} ({logliks[-1]:.9f}) to k={k} ({loglik:.9f})")
            fits.append(comp)
            logliks.append(loglik)
            bics.append(-2.0 * n_eff * loglik + (3 * k - 1) * math.log(n_eff))
        best = int(np.argmin(bics))
        comp = fits[best]
        for weight, mean, sigma in comp[np.argsort(comp[:, 1])]:
            components.append({"finding": label, "weight": round(float(weight) / len(LABELS), 6), "mean": round(float(mean), 6), "sigma": round(float(sigma), 6)})
        findings[label] = {"k": best + 1, "bic": [round(b, 3) for b in bics], "n_study_slots": n_eff}
    return {"components": components, "findings": findings}, stalled, failed


@click.command(help="Fit the per-Slot depth prior (equal-weight mixture of per-finding GMMs) from the teacher's out-of-fold GAP window attention.")
@click.option("--attention", type=click.Path(path_type=Path), default=Path("output/attention_prior/exp242/window_attention.csv.gz"))
@click.option("--cache", type=click.Path(path_type=Path), default=Path("data/cache/cache_native_384_series_all_exact_uniform_sub"))
@click.option("--train-csv", type=click.Path(path_type=Path), default=Path("input/rsna-knee-abnormality-detection/train.csv"))
@click.option("--out", type=click.Path(path_type=Path), default=Path("data/depth_prior/v1.json"))
@click.option("--sigma-floor", type=float, default=0.08)
@click.option("--max-k", type=int, default=4, help="Max components per finding.")
@click.option("--tol", type=float, default=1e-9, help="Change in weighted log-likelihood that stops the initial EM.")
@click.option("--max-iter", type=int, default=5000, help="Max iterations of the initial EM.")
def main(**options: Any) -> None:
    """Fit the depth prior and write it as JSON."""
    args = SimpleNamespace(**options)

    frame = load_windows(args.attention, args.cache, args.train_csv)
    slots, stalled, failed = {}, 0, 0
    for index, name in enumerate(SLOT_NAMES):
        print(f"fitting {name}", flush=True)
        slots[name], em_count, opt_count = fit_slot(frame[frame["slot"] == index], args.max_k, args.sigma_floor, args.tol, args.max_iter)
        stalled, failed = stalled + em_count, failed + opt_count
    payload = {"version": 1, "truncate": [-1.0, 1.0], "sigma_floor": args.sigma_floor, "attention": "gap", "slots": slots}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out}", flush=True)

    print("k per Slot x finding")
    print(f"{'':18s}" + "".join(f"{name:>12s}" for name in SLOT_NAMES))
    for label in LABELS:
        print(f"{label:18s}" + "".join(f"{slots[name]['findings'][label]['k']:>12d}" for name in SLOT_NAMES))
    edges = np.linspace(-1, 1, 11)
    print("12-finding mixture (truncated) per u bin, interval probability as ratio to uniform")
    print(f"{'':12s}" + "".join(f"{f'{lo:+.1f}..{hi:+.1f}':>12s}" for lo, hi in zip(edges[:-1], edges[1:])))
    for name in SLOT_NAMES:
        print(f"{name:12s}" + "".join(f"{v:12.2f}" for v in bin_ratio(as_array(slots[name]["components"]), edges)))
    floored = [(name, c["finding"], c["mean"]) for name in SLOT_NAMES for c in slots[name]["components"] if c["sigma"] <= args.sigma_floor + 1e-9]
    print(f"components at sigma floor ({len(floored)}): {floored}")
    capped = [(name, c["finding"], c["mean"]) for name in SLOT_NAMES for c in slots[name]["components"] if c["sigma"] >= SIGMA_CAP - 1e-6]
    print(f"components at sigma cap ({len(capped)}): {capped}")
    print(f"initial EM stopped at max-iter: {stalled}, L-BFGS-B not converged: {failed}")


if __name__ == "__main__":
    main()
