import hashlib
import json
from collections import defaultdict
from pathlib import Path

import click
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from rsna_knee.constants import ID_COLUMN, LABELS

DEFAULT_TRAIN_CSV = Path("input/rsna-knee-abnormality-detection/train.csv")
DEFAULT_SERIES_META = Path("data/cache/cache_native_384_series_all_exact/series_meta.csv")
DEFAULT_LABELS_CSV = Path("input/rsna-knee-labels/labels_llm_soft.csv")
FINGERPRINT_KEYS = ["Manufacturer", "ManufacturerModelName", "SoftwareVersions", "ReceiveCoilName"]
SOFT_THRESHOLD = 0.5


def resolve_series_meta(path: Path) -> Path:
    """``--series-meta`` を解決する。既定のものが無ければ ``data/cache/*/series_meta.csv`` の一番新しいものを拾う。"""
    if path.exists():
        return path
    candidates = sorted(Path("data/cache").glob("*/series_meta.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise click.ClickException(f"{path} not found and no data/cache/*/series_meta.csv either; build a cache first (tools/build_cache.py --resample exact)")
    return candidates[0]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def study_fingerprints(meta: pd.DataFrame, study_ids: pd.Index) -> pd.Series:
    """study ごとの scanner fingerprint。series が複数あるときは最頻値、同数なら辞書順で小さい方を採る。"""
    missing = [key for key in FINGERPRINT_KEYS if key not in meta.columns]
    if missing:
        raise click.ClickException(f"series_meta is missing columns {missing}")
    per_series = meta[FINGERPRINT_KEYS].fillna("").astype(str).agg("|".join, axis=1)
    frame = pd.DataFrame({ID_COLUMN: meta[ID_COLUMN], "fingerprint": per_series})
    counts = frame.value_counts([ID_COLUMN, "fingerprint"]).reset_index(name="n")
    counts = counts.sort_values([ID_COLUMN, "n", "fingerprint"], ascending=[True, False, True])
    picked = counts.drop_duplicates(subset=ID_COLUMN).set_index(ID_COLUMN)["fingerprint"]
    absent = study_ids.difference(picked.index)
    if len(absent):
        raise click.ClickException(f"{len(absent)} studies have no row in series_meta (e.g. {list(absent[:3])}); rebuild the cache over the full train set")
    return picked.reindex(study_ids)


def connected_groups(study_ids: list[str], keys: list[list[str]]) -> dict[str, int]:
    """同じ鍵を 1 つでも共有する study を union-find で束ねて、study -> 群 ID を返す。"""
    parent: dict[object, object] = {}

    def find(node: object) -> object:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: object, right: object) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[left] = right

    for uid, study_keys in zip(study_ids, keys, strict=True):
        find(("study", uid))
        for key in study_keys:
            union(("study", uid), key)
    roots: dict[object, int] = {}
    groups: dict[str, int] = {}
    for uid in study_ids:  # 群 ID は study の順で振るので決定的
        root = find(("study", uid))
        groups[uid] = roots.setdefault(root, len(roots))
    return groups


def binary_labels(train: pd.DataFrame, labels_csv: Path) -> pd.DataFrame:
    """層化と統計に使う 12 所見の 0/1 表。gold の 58 件は train.csv の本物、それ以外は LLM soft label を 0.5 で 2 値化する。"""
    frame = train.set_index(ID_COLUMN)[LABELS]
    gold = frame.notna().all(axis=1)
    if not labels_csv.exists():
        raise click.ClickException(f"{labels_csv} not found; it supplies the stratification labels for the {int((~gold).sum())} non-gold studies")
    soft = pd.read_csv(labels_csv).drop_duplicates(subset=ID_COLUMN).set_index(ID_COLUMN)
    missing_labels = [label for label in LABELS if label not in soft.columns]
    if missing_labels:
        raise click.ClickException(f"{labels_csv} is missing columns {missing_labels}")
    uncovered = frame.index[~gold & ~frame.index.isin(soft.index)]
    if len(uncovered):
        raise click.ClickException(f"{len(uncovered)} studies are neither gold nor in {labels_csv.name} (e.g. {list(uncovered[:3])})")
    binary = (soft.reindex(frame.index)[LABELS] >= SOFT_THRESHOLD).astype(int)
    binary[gold] = frame[gold][LABELS].astype(int)
    return binary


def verify(folds: pd.Series, fingerprint: pd.Series, groups: dict[str, int]) -> None:
    """群化の鍵を持つ study が 2 つの fold に散っていたら例外。"""
    for name, key in (("scanner fingerprint", fingerprint), ("group", pd.Series(groups))):
        spread = folds.groupby(key.reindex(folds.index)).nunique()
        bad = spread[spread > 1]
        if len(bad):
            raise click.ClickException(f"{len(bad)} {name} values straddle more than one fold (e.g. {list(bad.index[:3])})")


@click.command()
@click.option("--train-csv", type=click.Path(path_type=Path), default=DEFAULT_TRAIN_CSV, show_default=True, help="Competition train.csv (study list, reports, gold labels).")
@click.option("--series-meta", type=click.Path(path_type=Path), default=DEFAULT_SERIES_META, show_default=True, help="series_meta.csv written by tools/build_cache.py; falls back to the newest data/cache/*/series_meta.csv.")
@click.option("--labels-csv", type=click.Path(path_type=Path), default=DEFAULT_LABELS_CSV, show_default=True, help="LLM soft labels used to stratify the non-gold studies.")
@click.option("--out", type=click.Path(path_type=Path), default=Path("data/folds_v1.csv"), show_default=True, help="Output CSV; the recipe goes next to it as .json.")
@click.option("--folds", type=int, default=5, show_default=True, help="Number of folds.")
@click.option("--seed", type=int, default=42, show_default=True, help="Shuffle seed for StratifiedGroupKFold.")
@click.option("--dedup-reports", is_flag=True, help="Drop studies whose stripped Report repeats an earlier one (reproduces the published fold's 4,273 studies).")
def main(train_csv: Path, series_meta: Path, labels_csv: Path, out: Path, folds: int, seed: int, dedup_reports: bool) -> None:
    """``StudyInstanceUID,fold`` を作る。scanner fingerprint で群化し、12 所見の陽性数 (0/1/2/3+) で層化する。"""
    series_meta = resolve_series_meta(series_meta)
    if not train_csv.exists():
        raise click.ClickException(f"{train_csv} not found")
    train = pd.read_csv(train_csv).drop_duplicates(subset=ID_COLUMN).sort_values(ID_COLUMN).reset_index(drop=True)
    study_ids = train[ID_COLUMN].tolist()
    index = pd.Index(study_ids, name=ID_COLUMN)

    meta = pd.read_csv(series_meta, dtype=str, low_memory=False)
    fingerprint = study_fingerprints(meta, index)
    if train["Report"].isna().any():
        raise click.ClickException(f"{int(train['Report'].isna().sum())} studies have no Report; cannot hash them")
    report_hash = pd.Series([hashlib.sha256(text.strip().encode("utf-8")).hexdigest() for text in train["Report"]], index=index)
    keep = ~report_hash.duplicated(keep="first").to_numpy() if dedup_reports else np.ones(len(study_ids), dtype=bool)
    dropped = int((~keep).sum())
    if dropped:
        train = train.loc[keep].reset_index(drop=True)
        study_ids = train[ID_COLUMN].tolist()
        index = pd.Index(study_ids, name=ID_COLUMN)
        fingerprint = fingerprint[index]
    groups = connected_groups(study_ids, [[("fingerprint", fingerprint[uid])] for uid in study_ids])
    sizes = pd.Series(groups).value_counts()
    largest = int(sizes.iloc[0])
    if largest > len(study_ids) / folds:
        raise click.ClickException(f"the largest group holds {largest} of {len(study_ids)} studies ({largest / len(study_ids):.1%}), more than one {folds}-fold split; grouping keys need rethinking")

    binary = binary_labels(train, labels_csv)
    strata = binary.sum(axis=1).clip(upper=3).to_numpy()  # 0 / 1 / 2 / 3+ 所見
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    assignment = pd.Series(-1, index=index, dtype=int)
    for fold, (_, holdout) in enumerate(splitter.split(study_ids, strata, groups=[groups[uid] for uid in study_ids])):
        assignment.iloc[holdout] = fold
    if (assignment < 0).any():
        raise click.ClickException(f"{int((assignment < 0).sum())} studies were left without a fold")
    verify(assignment, fingerprint, groups)

    out.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame({ID_COLUMN: index, "fold": assignment.to_numpy()}).sort_values(ID_COLUMN).reset_index(drop=True)
    table.to_csv(out, index=False)

    counts = assignment.value_counts().sort_index()
    rates = binary.groupby(assignment).mean().round(4)
    fingerprint_groups = int(fingerprint.nunique())
    recipe = {
        "note": "公開 fold の写しではなく、tools/make_folds.py がこの手順で生成したもの",
        "grouping": {
            "keys": ["scanner fingerprint: " + " + ".join(FINGERPRINT_KEYS) + " (mode over the study's series)"],
            "rule": "studies on the same scanner land in the same fold",
            "duplicate_reports_dropped": dropped,
            "reference": "scanner-fingerprint grouping as in flight0234/rsna-knee-dual-grouped-folds; --dedup-reports additionally drops the same 134 studies they drop",
            "n_groups": len(sizes),
            "largest_group": largest,
            "largest_group_fraction": round(largest / len(study_ids), 4),
            "fingerprint_values": fingerprint_groups,
            "report_values": int(report_hash.nunique()),
        },
        "split": {
            "n_folds": folds,
            "seed": seed,
            "splitter": "sklearn.model_selection.StratifiedGroupKFold(shuffle=True)",
            "stratum": "number of positive findings clipped to 0/1/2/3+ (gold labels for the 58 fully labelled studies, LLM soft labels >= 0.5 otherwise)",
            "n_studies": len(study_ids),
            "excluded": 0,
        },
        "fold_sizes": {str(fold): int(size) for fold, size in counts.items()},
        "positive_rate": {str(fold): {label: float(rates.loc[fold, label]) for label in LABELS} for fold in counts.index},
        "inputs": {str(path): sha256_of(path) for path in (train_csv, series_meta, labels_csv)},
    }
    recipe_path = out.with_suffix(".json")
    recipe_path.write_text(json.dumps(recipe, ensure_ascii=False, indent=2) + "\n")

    click.echo(f"dropped {dropped} studies whose stripped Report repeats an earlier one" if dropped else "kept every study (pass --dedup-reports to reproduce the published 4,273)")
    click.echo(f"fingerprints: {fingerprint_groups} values, groups after union-find: {len(sizes)} (largest {largest} = {largest / len(study_ids):.1%})")
    click.echo(f"wrote {out} with {len(table)} studies and {recipe_path}")
    for fold, size in counts.items():
        click.echo(f"  fold {fold}: {size} studies, positives/study {binary[assignment == fold].sum(axis=1).mean():.2f}")
    spread = defaultdict(float)
    for label in LABELS:
        spread[label] = float(rates[label].max() - rates[label].min())
    worst = sorted(spread.items(), key=lambda item: -item[1])[:3]
    click.echo("positive-rate spread across folds (worst 3): " + ", ".join(f"{label} {gap:.3f}" for label, gap in worst))


if __name__ == "__main__":
    main()
