from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["CACHE_IDS", "CACHE_MANIFEST", "CACHE_MASKS", "CACHE_OFFSETS", "CACHE_SERIES", "CACHE_STACK_OFFSETS", "CACHE_VOLS", "build_uid_index", "read_cache_manifest", "read_series_units"]

#: ``tools/build_cache.py`` が書く merged cache ディレクトリのファイル名。
CACHE_VOLS = "vols.npy"
CACHE_IDS = "ids.npy"
CACHE_MASKS = "masks.npy"
CACHE_OFFSETS = "offsets.npy"
CACHE_MANIFEST = "manifest.json"
#: series レイアウト（manifest の ``series_layout == "all"``）だけが持つファイル。
CACHE_STACK_OFFSETS = "stack_offsets.npy"
CACHE_SERIES = "series.csv"


def read_cache_manifest(cache_dir: str | Path) -> dict[str, Any]:
    """study cache ディレクトリの ``manifest.json`` を読んで検証し、``n_slices`` を足して返す。

    正本は ``slot_table``（Slot ごとのスライス枚数）で、loader とモデルが形を取るのはここからだけ。
    ``slots`` はその表の *名前* であって枚数ではないので、数値でありながら表と食い違う manifest は
    はっきりエラーにする。画素の作られ方（``span`` / ``size`` / ``crop_mm`` / ``percentiles``）を言えない
    manifest も受け付けない。``vols.npy`` は実在スライスだけを study 順に詰めた ``[P, H, W]`` で、
    ``ids.npy[i]`` の study は ``vols[offsets[i]:offsets[i + 1]]``、Stack 上の位置は ``masks[i]`` が持つ。
    ``series_layout`` は ``"slots"``（キーが無いときも）か ``"all"`` に正規化する。``"all"`` の ``masks`` は 1 次元で、study i の Stack は
    ``masks[stack_offsets[i]:stack_offsets[i + 1]]``、unit の並びは ``series.csv``（:func:`read_series_units`）が持つ。
    """
    cache_dir = Path(cache_dir)
    path = cache_dir / CACHE_MANIFEST
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; a cache directory needs {CACHE_MANIFEST}")
    manifest = json.loads(path.read_text())

    layout = manifest.get("series_layout", "slots")
    if layout not in ("slots", "all"):
        raise ValueError(f"{path}: series_layout must be 'slots' or 'all', got {layout!r}")
    for name in (CACHE_VOLS, CACHE_IDS, CACHE_MASKS, CACHE_OFFSETS, *((CACHE_STACK_OFFSETS, CACHE_SERIES) if layout == "all" else ())):
        if not (cache_dir / name).exists():
            raise FileNotFoundError(f"{cache_dir / name} not found")

    table = [int(n) for n in manifest["slot_table"]]
    if not table or any(n <= 0 for n in table):
        raise ValueError(f"{path}: slot_table must be positive slice counts, got {table}")
    n_slices = sum(table)

    # レシピは重みと一緒に提出 notebook まで運ばれる。自分のレシピを言えない cache はここで弾く。
    missing = [key for key in ("size", "span", "crop_mm", "percentiles", "long_series_sampling") if key not in manifest]
    if missing:
        raise ValueError(f"{path}: missing {missing}; the manifest must describe how its pixels were produced")

    label = manifest.get("slots")
    if isinstance(label, int) or (isinstance(label, str) and label.isdigit()):
        if int(label) != n_slices:
            raise ValueError(f"{path}: slot_table {table} sums to {n_slices}, not slots={label}")
    manifest["slot_table"] = table
    manifest["n_slices"] = n_slices
    manifest["series_layout"] = layout
    return manifest


def read_series_units(cache_dir: str | Path) -> pd.DataFrame:
    """series レイアウトの ``series.csv`` を ``ids.npy`` の順・unit 順で読み、``study_row``（``ids.npy`` の行）を足して返す。

    unit j は study の Stack の ``[stack_offset, stack_offset + cap)`` を占め、直後の 1 枚は非実在。占める長さが ``stack_offsets.npy`` と食い違えばエラー。
    """
    cache_dir = Path(cache_dir)
    ids = [str(uid) for uid in np.load(cache_dir / CACHE_IDS, allow_pickle=True).tolist()]
    stack = np.load(cache_dir / CACHE_STACK_OFFSETS)
    units = pd.read_csv(cache_dir / CACHE_SERIES, dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    row_of = {uid: row for row, uid in enumerate(ids)}
    units["study_row"] = units["StudyInstanceUID"].map(row_of)
    if units["study_row"].isna().any():
        raise ValueError(f"{cache_dir / CACHE_SERIES}: {int(units['study_row'].isna().sum())} units belong to studies missing from {CACHE_IDS}")
    units = units.astype({"study_row": np.int64}).sort_values(["study_row", "unit"], kind="stable").reset_index(drop=True)
    lengths = (units["cap"] + 1).groupby(units["study_row"]).sum().reindex(range(len(ids)), fill_value=0).to_numpy()
    if len(stack) != len(ids) + 1 or not np.array_equal(lengths, np.diff(stack)):
        raise ValueError(f"{cache_dir}: {CACHE_SERIES} does not cover {CACHE_STACK_OFFSETS}")
    return units


def build_uid_index(parts: list[tuple[str | Path, str | Path]]) -> dict[str, tuple[int, int]]:
    """すべての ``StudyInstanceUID`` を ``(part_index, row_index)`` に対応づける。

    cache 1 つが 1 part で、同じ uid が 2 度出てきたら先の part が勝つ。行番号は ``ids.npy`` の並びそのもの。
    """
    index: dict[str, tuple[int, int]] = {}
    for part_idx, (_, ids_path) in enumerate(parts):
        ids = np.load(Path(ids_path), allow_pickle=True)
        for row, uid in enumerate(ids.tolist()):
            index.setdefault(str(uid), (part_idx, row))
    return index
