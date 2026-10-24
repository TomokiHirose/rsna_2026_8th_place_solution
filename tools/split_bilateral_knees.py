"""両膝の series が 1 つの study に入っている検査を見つけ、series を右膝・左膝に振り分けた表を書く（膝分割の前処理）。

``tools/build_cache.py`` が書いた cache の ``series_meta.csv`` の幾何だけを読む（DICOM を開き直さない）。出力は
``data/label/ayfukushima/bilateral_knees/knee_series.csv`` と同じ形で、学習では ``data.subcases_csv`` に渡す。
規則は :mod:`rsna_knee.data.bilateral`。

    uv run python tools/split_bilateral_knees.py --cache data/cache/cache_native_384_series_all_exact_uniform_sub \\
        --train-series input/rsna-knee-abnormality-detection/train_series.csv --out /tmp/knee_series_from_cache.csv \\
        --compare data/label/ayfukushima/bilateral_knees/knee_series.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import pandas as pd

from rsna_knee.data.bilateral import MIN_SERIES, MIN_SPREAD_MM, split_bilateral_studies

KEY = ["StudyInstanceUID", "SeriesInstanceUID", "knee", "child_study_id"]


@click.command(help="Find studies holding series of both knees and write a table assigning each series to the right or left knee (knee-split preprocessing).")
@click.option("--cache", type=click.Path(path_type=Path), required=True, help="Cache directory holding series_meta.csv.")
@click.option("--out", type=click.Path(path_type=Path), required=True, help="Output table (CSV).")
@click.option("--train-series", type=click.Path(path_type=Path), default=None, help="train_series.csv; adds the plane, fat-suppression and fluid-sensitive columns.")
@click.option("--min-spread-mm", type=float, default=MIN_SPREAD_MM, show_default=True, help="Left-right spread of series centres above which a study counts as bilateral.")
@click.option("--min-series", type=int, default=MIN_SERIES, show_default=True, help="Series per knee needed to make a child study.")
@click.option("--compare", type=click.Path(path_type=Path), default=None, help="Reference table (knee_series.csv) to compare against; exits non-zero on any mismatch.")
def main(cache: Path, out: Path, train_series: Path | None, min_spread_mm: float, min_series: int, compare: Path | None) -> None:
    """Split bilateral studies into per-knee tables and optionally compare with a reference."""
    raise SystemExit(split(cache, out, train_series, min_spread_mm, min_series, compare))


def split(cache: Path, out: Path, train_series: Path | None, min_spread_mm: float, min_series: int, compare: Path | None) -> int:
    """Write the per-knee table; returns 1 when ``compare`` disagrees, else 0."""
    if compare is not None and out.resolve() == compare.resolve():
        raise click.UsageError("--out and --compare are the same file; write the new table elsewhere and compare it against the reference")

    meta = pd.read_csv(cache / "series_meta.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    table = split_bilateral_studies(meta, min_spread_mm, min_series)
    if train_series is not None:
        extra = pd.read_csv(train_series, dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
        table = table.merge(extra[["SeriesInstanceUID", "Anatomical_Plane", "Fat_Suppression", "Fluid_Sensitive"]], on="SeriesInstanceUID", how="left")
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    summary = {
        "studies_scanned": int(meta["StudyInstanceUID"].nunique()),
        "bilateral_studies": int(table["StudyInstanceUID"].nunique()),
        "children": int(table["child_study_id"].nunique()),
        "series": table["knee"].value_counts().to_dict(),
        "out": str(out),
    }
    status = 0
    if compare is not None:
        ref = pd.read_csv(compare, dtype=str)[KEY].fillna("")
        new = table[KEY].astype(object).where(table[KEY].notna(), "").astype(str)
        merged = ref.merge(new, on=["StudyInstanceUID", "SeriesInstanceUID"], how="outer", suffixes=("_ref", "_new"), indicator=True)
        diff = merged[(merged["_merge"] != "both") | (merged["knee_ref"] != merged["knee_new"]) | (merged["child_study_id_ref"] != merged["child_study_id_new"])]
        summary["compare"] = {"reference": str(compare), "series": len(ref), "mismatches": len(diff)}
        if len(diff):
            print(diff.to_string(index=False), file=sys.stderr)
            status = 1
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return status


if __name__ == "__main__":
    main()
