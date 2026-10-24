from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rsna_knee.data.cache_dataset import CACHE_SERIES, CACHE_STACK_OFFSETS, read_series_units
from rsna_knee.data.dicom import META_FIELDS, PreprocessParams, StudyTensor, all_series_units, build_study, count_dcm_files, probe_series, resolve_slots, span_count
from rsna_knee.data.slots import NUM_SLOTS, SLOT_NAMES, _as_flag, assign_slots

ID_COLUMN = "StudyInstanceUID"
#: merged レイアウトの組み立て中に study ごと ``[S, H, W]`` で書く中間ファイル。完成時に ``pack_volumes`` が詰める。
PADDED_VOLS = "vols_padded.npy"
SERIES_COLUMN = "SeriesInstanceUID"
#: series レイアウトで study i の Stack が ``masks`` / ``positions`` / ``vols_padded`` 上に占める範囲 ``[N+1]``。
STACK_OFFSETS = CACHE_STACK_OFFSETS
#: series レイアウトの unit（Stack に置いた series）の表。
SERIES_UNITS = CACHE_SERIES
#: ``--subseries`` の組み立て中に worker が study ごとに書く sub-series の置き場（詰め終えたら消す）と、詰めた後のファイル。
SUBSERIES_PARTS = "subseries_parts"
SUBSERIES_TABLE = "subseries.csv"
SUBSERIES_FILES = ("sub_vols.npy", "sub_masks.npy", "sub_positions.npy")

INDEX_FIELDS: list[str] = [ID_COLUMN, "n_series", "n_slots", "n_present", "mask"] + [f"slot{i}_{suffix}" for i in range(NUM_SLOTS) for suffix in ("series", "n_slices", "n_present")] + ["n_warnings"]
SERIES_INDEX_FIELDS: list[str] = [ID_COLUMN, "n_series", "n_units", "n_present", "mask", "n_warnings"]
SERIES_UNIT_FIELDS: list[str] = [ID_COLUMN, SERIES_COLUMN, "unit", "plane", "fluid", "stack_offset", "cap", "n_present"]
META_ROW_FIELDS: list[str] = [ID_COLUMN, SERIES_COLUMN, "slot_index", "slot_name", "assigned", *META_FIELDS]
CENSUS_FIELDS: list[str] = [ID_COLUMN, SERIES_COLUMN, "slot_index", "slot_name", "n_files", "in_span", "SliceThickness", "SpacingBetweenSlices", "PixelSpacing_0", "Rows", "Columns", "SeriesDescription"]


def read_split(data_dir: Path, split: str) -> tuple[list[str], pd.DataFrame]:
    """1 split の study id（CSV 順）と series 表。"""
    series_path = data_dir / f"{split}_series.csv"
    if not series_path.exists():
        raise SystemExit(f"missing {series_path}")
    series = pd.read_csv(series_path)
    study_csv = data_dir / f"{split}.csv"
    if study_csv.exists():
        studies = pd.read_csv(study_csv, usecols=[ID_COLUMN])[ID_COLUMN].astype(str).tolist()
    else:  # split の表が無いこともある（test を部分的にしか落としていない場合など）
        studies = series[ID_COLUMN].astype(str).drop_duplicates().tolist()
    seen: set[str] = set()
    return [s for s in studies if not (s in seen or seen.add(s))], series


def select_studies(studies: list[str], shard: tuple[int, int], limit: int | None) -> list[str]:
    index, total = shard
    selected = [s for i, s in enumerate(studies) if i % total == index]
    return selected[:limit] if limit else selected


def git_commit(explicit: str | None = None) -> str:
    """``--code-commit`` があればそれ、無ければこの checkout の HEAD、それも駄目なら ``"unknown"``。"""
    if explicit:
        return explicit
    try:
        done = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return done.stdout.strip() if done.returncode == 0 and done.stdout.strip() else "unknown"


def write_manifest(out_dir: Path, params: PreprocessParams, split: str, shard: str | None, n_studies: int, commit: str, caps_rule: str | None = None) -> dict[str, Any]:
    """``out/manifest.json`` を書く。正本は ``slot_table`` で、読み手は Slot の個数とブロックをそこから導く。"""
    slots = resolve_slots(params)
    manifest = {
        "slots": str(params.slots),
        "slot_table": [slot.n_slices for slot in slots],
        "span": [float(params.span[0]), float(params.span[1])],
        "size": int(params.size),
        "crop_mm": float(params.crop_mm),
        "percentiles": [float(params.percentiles[0]), float(params.percentiles[1])],
        "split": split,
        "shard": shard or "0/1",
        "n_studies": int(n_studies),
        "code_commit": commit,
        "native_spacing": bool(params.native_spacing),
        "resample": str(params.resample),
        "long_series_sampling": params.long_series_sampling,
        "crop_pad": bool(params.crop_pad),
        "subseries": bool(params.subseries),
        "histogram_standard": None if params.histogram_standard is None else [list(row) for row in params.histogram_standard],
        "caps_rule": caps_rule,
        "series_layout": params.series_layout,
        "anterior_shift_mm": float(params.anterior_shift_mm),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def check_recipe_matches(out_dir: Path, params: PreprocessParams, split: str) -> None:
    """別のレシピで焼かれた cache ディレクトリへの追記を拒む。

    resume は study id しか見ないうえ、``--span`` / ``--resample`` などを変えても配列の形は変わらない。
    ``resample`` を持たない manifest は ``"area"`` で焼かれたものとして比較する。
    """
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        return
    try:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not any(out_dir.glob("chunk_*")) and not (out_dir / PADDED_VOLS).exists():
        return  # まだ何も焼かれていない。manifest はただの残骸
    slots = resolve_slots(params)
    wanted = {
        "slots": str(params.slots),
        "slot_table": [slot.n_slices for slot in slots],
        "span": [float(params.span[0]), float(params.span[1])],
        "size": int(params.size),
        "crop_mm": float(params.crop_mm),
        "percentiles": [float(params.percentiles[0]), float(params.percentiles[1])],
        "split": split,
        "native_spacing": bool(params.native_spacing),
        "resample": str(params.resample),
        "long_series_sampling": params.long_series_sampling,
        "crop_pad": bool(params.crop_pad),
        "subseries": bool(params.subseries),
        "histogram_standard": None if params.histogram_standard is None else [list(row) for row in params.histogram_standard],
        "series_layout": params.series_layout,
        "anterior_shift_mm": float(params.anterior_shift_mm),
    }
    existing.setdefault("subseries", False)  # このキーを持たない cache は sub-series を持たない
    existing.setdefault("crop_pad", False)  # このキーを持たない cache はすべてはみ出しを縮めたボックスで焼かれている
    existing.setdefault("resample", "area")  # このキーを持たない cache はすべて "area" で焼かれている
    existing.setdefault("histogram_standard", None)  # このキーを持たない cache はすべて輝度窓で焼かれている
    existing.setdefault("series_layout", "slots")  # このキーを持たない cache はすべて 5 Slot 固定で焼かれている
    existing.setdefault("anterior_shift_mm", 0.0)  # このキーを持たない cache はすべて画像中心でクロップされている
    mismatch = {key: (existing.get(key), value) for key, value in wanted.items() if (key in existing or key == "long_series_sampling") and existing.get(key) != value}
    if mismatch:
        raise SystemExit(f"{out_dir} already holds a cache built with a different recipe (was, now): {mismatch}")


def _init_worker() -> None:
    """worker は 1 スレッドに抑える。並列度は --workers で出す。"""
    import cv2

    cv2.setNumThreads(1)


def _process_study(task: dict[str, Any]) -> dict[str, Any]:
    """worker 側の処理。1 Study をデコードして画素を書き出し、小さな行だけを返す。

    series レイアウトは ``task["stack"]`` の ``(start, stop)`` に Stack を書く。固定レイアウトは ``position`` 行に書く。
    """
    study_id = task["study_id"]
    position = task.get("position", 0)
    try:
        import cv2

        cv2.setNumThreads(1)  # 1 study につき 1 プロセス。並列化は --workers に任せる
        result = build_study(task["study_dir"], task["rows"], task["params"], study_id=study_id)
        # 互いに重ならない行への MAP_SHARED なので、親も同じページキャッシュを読む以上 flush は要らない。
        volume = np.load(task["chunk_path"], mmap_mode="r+")
        if task["params"].series_layout == "all":
            start, stop = task["stack"]
            if len(result.pixels) != stop - start:
                raise ValueError(f"Stack has {len(result.pixels)} slices, the series table reserved {stop - start}")
            volume[start:stop] = result.pixels
            if result.subseries:
                write_subseries_part(Path(task["chunk_path"]).parent, study_id, result)
        else:
            volume[position] = result.pixels
        del volume
    except Exception as exc:  # 1 件の失敗で run 全体を落としてはいけない
        return {"study_id": study_id, "position": position, "error": f"{type(exc).__name__}: {exc}", "warnings": [], "meta_rows": []}

    if task["params"].series_layout == "all":
        row, unit_rows = _series_rows(task, result)
    else:
        row, unit_rows = _slot_row(task, result), []
    return {
        "study_id": study_id,
        "position": position,
        "error": None,
        "index_row": row,
        "meta_rows": result.series_headers,
        "unit_rows": unit_rows,
        "warnings": result.warnings,
        "mask": result.mask.astype(np.uint8).tolist(),
        "positions": result.positions.astype(np.float32).tolist(),
    }


def _slot_row(task: dict[str, Any], result: StudyTensor) -> dict[str, Any]:
    """固定レイアウトの ``index.csv`` の行。Slot ごとの series と実在枚数を持つ。"""
    study_id = task["study_id"]
    groups = []
    offset = 0
    row: dict[str, Any] = {ID_COLUMN: study_id, "n_series": len(task["rows"]), "n_warnings": len(result.warnings)}
    for slot in resolve_slots(task["params"]):
        slice_mask = result.mask[offset : offset + slot.n_slices]
        groups.append("".join(str(int(v)) for v in slice_mask))
        offset += slot.n_slices
        row[f"slot{slot.index}_n_present"] = int(slice_mask.sum())
    for assignment in result.assignments:
        row[f"slot{assignment.slot_index}_series"] = assignment.series_uid or ""
        row[f"slot{assignment.slot_index}_n_slices"] = assignment.n_slices
    row["n_slots"] = sum(1 for a in result.assignments if a.present)
    row["n_present"] = int(result.mask.sum())
    row["mask"] = "|".join(groups)  # Slot ごとの塊。"|" があるので pandas が数値として解釈しない
    return row


def _series_rows(task: dict[str, Any], result: StudyTensor) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """series レイアウトの ``index.csv`` の行と、unit ごとの ``series.csv`` の行。"""
    study_id = task["study_id"]
    units, _ = all_series_units(study_id, task["rows"], resolve_slots(task["params"]))
    unit_rows: list[dict[str, Any]] = []
    groups = []
    for j, (slot, offset, row) in enumerate(units):
        slice_mask = result.mask[offset : offset + slot.n_slices]
        groups.append("".join(str(int(v)) for v in slice_mask))
        fluid = _as_flag(row.get("Fluid_Sensitive"))
        unit_rows.append({ID_COLUMN: study_id, SERIES_COLUMN: str(row[SERIES_COLUMN]), "unit": j, "plane": slot.plane, "fluid": -1 if fluid is None else fluid, "stack_offset": offset, "cap": slot.n_slices, "n_present": int(slice_mask.sum())})
    index_row = {ID_COLUMN: study_id, "n_series": len(task["rows"]), "n_units": len(units), "n_present": int(result.mask.sum()), "mask": "|".join(groups), "n_warnings": len(result.warnings)}
    return index_row, unit_rows


def write_subseries_part(out_dir: Path, study_id: str, result: StudyTensor) -> Path:
    """1 study の sub-series を ``subseries_parts/<study>.npz`` に書く（一時ファイルから置き換えるので、途中で殺されても半端なファイルは残らない）。"""
    parts = out_dir / SUBSERIES_PARTS
    parts.mkdir(exist_ok=True)
    meta = [{"SeriesInstanceUID": sub.series_uid, "sub": sub.index, "n_sub": sub.count, "cap": len(sub.mask), "selected_indices": json.dumps(sub.picks)} for sub in result.subseries]
    path = parts / f"{study_id}.npz"
    tmp = parts / f"{study_id}.tmp.npz"
    # 撮像面で cap が違うので、sub-series を面の方向に連結して書く（j 本目は meta の cap の累積で切り出す）
    pixels = np.concatenate([sub.pixels for sub in result.subseries])
    np.savez(tmp, pixels=pixels, mask=np.concatenate([sub.mask for sub in result.subseries]), positions=np.concatenate([sub.positions for sub in result.subseries]), meta=np.array(json.dumps(meta)))
    os.replace(tmp, path)
    return path


def pack_subseries(out_dir: Path) -> pd.DataFrame:
    """``subseries_parts/`` を ``ids.npy`` の順に 1 本の ``sub_vols.npy`` ``[R, size, size]``・``sub_masks.npy``・``sub_positions.npy`` に詰め、
    ``subseries.csv`` を書いて置き場を消す。``subseries.csv`` の 1 行が sub-series 1 本で、``unit`` は ``series.csv`` の元の series の unit、
    画素は ``sub_vols[start:stop]``（``stop - start`` は cap、デコードに失敗した面は mask 0）。
    """
    parts = out_dir / SUBSERIES_PARTS
    manifest = json.loads((out_dir / "manifest.json").read_text())
    size = int(manifest["size"])
    ids = [str(uid) for uid in np.load(out_dir / "ids.npy", allow_pickle=True).tolist()]
    units = pd.read_csv(out_dir / SERIES_UNITS, dtype={ID_COLUMN: str, SERIES_COLUMN: str})
    unit_of = {(r[ID_COLUMN], r[SERIES_COLUMN]): int(r["unit"]) for r in units.to_dict(orient="records")}
    files = [(uid, parts / f"{uid}.npz") for uid in ids if (parts / f"{uid}.npz").exists()]
    metas = []
    for uid, path in files:
        with np.load(path) as part:
            metas.append(json.loads(str(part["meta"])))
    total = sum(int(meta["cap"]) for study in metas for meta in study)
    tmp = [tmp_npy(out_dir / name) for name in SUBSERIES_FILES]
    vols = np.lib.format.open_memmap(tmp[0], mode="w+", dtype=np.uint8, shape=(total, size, size))
    masks = np.zeros(total, dtype=np.uint8)
    positions = np.full(total, np.nan, dtype=np.float32)
    rows: list[dict[str, Any]] = []
    start = 0
    for (uid, path), study in zip(files, metas):
        with np.load(path) as part:
            pixels, mask, pos = part["pixels"], part["mask"], part["positions"]
        at = 0
        for meta in study:
            cap = int(meta["cap"])
            vols[start : start + cap] = pixels[at : at + cap]
            masks[start : start + cap] = mask[at : at + cap].astype(np.uint8)
            positions[start : start + cap] = pos[at : at + cap]
            uid_series = (uid, meta["SeriesInstanceUID"])
            rows.append({ID_COLUMN: uid, SERIES_COLUMN: uid_series[1], "unit": unit_of[uid_series], "sub": meta["sub"], "n_sub": meta["n_sub"], "start": start, "stop": start + cap})
            rows[-1] |= {"n_present": int(mask[at : at + cap].sum()), "selected_indices": meta["selected_indices"]}
            start += cap
            at += cap
    vols.flush()
    del vols
    np.save(tmp[1], masks)
    np.save(tmp[2], positions)
    table = pd.DataFrame(rows, columns=[ID_COLUMN, SERIES_COLUMN, "unit", "sub", "n_sub", "start", "stop", "n_present", "selected_indices"])
    table.to_csv(out_dir / SUBSERIES_TABLE, index=False)
    for path, name in zip(tmp, SUBSERIES_FILES):
        os.replace(path, out_dir / name)
    for _, path in files:
        path.unlink()
    for stale in parts.glob("*.tmp.npz"):
        stale.unlink()
    if parts.exists():
        parts.rmdir()
    return table


def series_stack_offsets(studies: list[str], series: pd.DataFrame, params: PreprocessParams) -> np.ndarray:
    """series レイアウトの ``stack_offsets`` ``[N+1]``。DICOM を読まず series 表だけから決まる。"""
    by_study = {str(k): v for k, v in series.groupby(ID_COLUMN)}
    slots = resolve_slots(params)
    lengths = [0 if by_study.get(uid) is None else all_series_units(uid, by_study[uid], slots)[1] for uid in studies]
    return np.concatenate([[0], np.cumsum(lengths, dtype=np.int64)]).astype(np.int64)


class AppendWriter:
    """ヘッダを 1 回だけ書き、行ごとに flush する CSV の追記器。"""

    def __init__(self, path: Path, fields: list[str]) -> None:
        is_new = not path.exists() or path.stat().st_size == 0
        self.handle = path.open("a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=fields, extrasaction="ignore", restval="")
        if is_new:
            self.writer.writeheader()
            self.handle.flush()

    def write(self, row: dict[str, Any]) -> None:
        self.writer.writerow(row)
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def indexed_studies(out_dir: Path) -> set[str]:
    """``index.csv`` に行を残した study、すなわちエラー無しで焼けた study。"""
    index_path = out_dir / "index.csv"
    if not index_path.exists():
        return set()
    try:
        return set(pd.read_csv(index_path, usecols=[ID_COLUMN])[ID_COLUMN].astype(str))
    except (ValueError, pd.errors.EmptyDataError):
        return set()


# --------------------------------------------------------------------------------------
# chunk レイアウト
# --------------------------------------------------------------------------------------


def tmp_npy(path: Path) -> Path:
    """``chunk_000.npy`` -> ``chunk_000.tmp.npy``。拡張子を ``.npy`` のままにするのは、
    それ以外の名前だと ``np.save`` が勝手に ``.npy`` を足すから。"""
    return path.with_name(f"{path.stem}.tmp.npy")


def chunk_paths(out_dir: Path, index: int) -> tuple[Path, Path, Path, Path]:
    """非圧縮の chunk ``index`` の ``(volumes, ids, masks, positions)``。"""
    stem = f"chunk_{index:03d}"
    return out_dir / f"{stem}.npy", out_dir / f"{stem}_ids.npy", out_dir / f"{stem}_masks.npy", out_dir / f"{stem}_positions.npy"


def chunk_npz_path(out_dir: Path, index: int) -> Path:
    """``--compress`` が上の 4 ファイルの代わりに書く、deflate 済みの 1 ファイル。"""
    return out_dir / f"chunk_{index:03d}.npz"


def chunk_is_complete(out_dir: Path, index: int, expected_ids: list[str], indexed: set[str] | None = None) -> bool:
    """chunk が完成しているのは、ファイルが在り、この id 列を持ち、``indexed`` が全 id を覆うとき。

    デコードで例外を出した study も行（全部ゼロ）と ``_ids`` を占めるので、``index.csv`` に行を残した
    study の集合と突き合わせないと、未完了の chunk が永久に真っ黒のまま完了扱いになる。
    """
    npz_path = chunk_npz_path(out_dir, index)
    try:
        if npz_path.exists():
            with np.load(npz_path, allow_pickle=False) as data:
                stored = data["ids"]
        else:
            paths = chunk_paths(out_dir, index)
            if not all(path.exists() for path in paths):
                return False
            stored = np.load(paths[1], allow_pickle=False)
    except (OSError, ValueError, KeyError):
        return False
    if [str(uid) for uid in stored.tolist()] != list(expected_ids):
        return False
    return indexed is None or set(expected_ids) <= indexed


def write_chunk(out_dir: Path, index: int, staged_vols: Path, ids: np.ndarray, masks: np.ndarray, positions: np.ndarray, compress: bool = False) -> list[Path]:
    """出来上がった chunk を 1 つ公開し、staging のファイルを捨てる。

    volumes はすでに staging ファイルにあるので、小さな 3 つの配列を足して名前を置き直すだけ。ids を
    最後に rename するので中途半端な公開は起きない。``compress`` は 1 つの ``.npz`` に deflate する。
    """
    if compress:
        tmp_npz = chunk_npz_path(out_dir, index).with_name(f"chunk_{index:03d}.tmp.npz")
        volumes = np.load(staged_vols, mmap_mode="r")  # zip へ流し込むだけで、メモリには載せない
        np.savez_compressed(tmp_npz, vols=volumes, ids=ids, masks=masks, positions=positions)
        del volumes
        os.replace(tmp_npz, chunk_npz_path(out_dir, index))
        staged_vols.unlink(missing_ok=True)
        return [chunk_npz_path(out_dir, index)]

    vols_path, ids_path, masks_path, positions_path = chunk_paths(out_dir, index)
    tmp_masks, tmp_positions, tmp_ids = tmp_npy(masks_path), tmp_npy(positions_path), tmp_npy(ids_path)
    np.save(tmp_masks, masks)
    np.save(tmp_positions, positions)
    np.save(tmp_ids, ids)
    os.replace(staged_vols, vols_path)
    os.replace(tmp_masks, masks_path)
    os.replace(tmp_positions, positions_path)
    os.replace(tmp_ids, ids_path)  # 最後。これが在ることが chunk 完成の印
    return [vols_path, ids_path, masks_path, positions_path]


def open_cache_arrays(out_dir: Path, studies: list[str], n_slices: int, size: int, stack_offsets: np.ndarray | None = None) -> tuple[np.memmap, np.memmap, np.memmap]:
    """merged レイアウトの組み立て中の ``vols_padded`` / ``masks`` / ``positions`` を作る、または開き直す。

    ``ids.npy`` を最初に書き、それが行の順序になる。id が違うディレクトリを開き直すのは拒否する。
    ``stack_offsets`` を渡すと series レイアウトで、3 つを ``[T, ...]`` の 1 次元の並びにし ``stack_offsets.npy`` も書く。
    """
    ids_path = out_dir / "ids.npy"
    if ids_path.exists():
        stored = [str(uid) for uid in np.load(ids_path, allow_pickle=False).tolist()]
        if stored != studies:
            raise SystemExit(f"{ids_path} holds {len(stored)} studies in a different order than this run's {len(studies)}; build into a fresh --out")
    else:
        tmp_ids = tmp_npy(ids_path)
        np.save(tmp_ids, np.asarray(studies))
        os.replace(tmp_ids, ids_path)

    if stack_offsets is None:
        rows: tuple[int, ...] = (len(studies), n_slices)
    else:
        offsets_path = out_dir / STACK_OFFSETS
        if offsets_path.exists():
            if not np.array_equal(np.load(offsets_path), stack_offsets):
                raise SystemExit(f"{offsets_path} does not match this run's series table; build into a fresh --out")
        else:
            tmp_offsets = tmp_npy(offsets_path)
            np.save(tmp_offsets, stack_offsets.astype(np.int64))
            os.replace(tmp_offsets, offsets_path)
        rows = (int(stack_offsets[-1]),)
    shapes = {PADDED_VOLS: ((*rows, size, size), np.uint8), "masks.npy": (rows, np.uint8), "positions.npy": (rows, np.float32)}
    arrays: list[np.memmap] = []
    for name, (shape, dtype) in shapes.items():
        path = out_dir / name
        if path.exists():
            array = np.lib.format.open_memmap(path, mode="r+")
            if array.shape != shape or array.dtype != dtype:
                raise SystemExit(f"{path} is {array.dtype}{array.shape}, this run needs {np.dtype(dtype)}{shape}; build into a fresh --out")
        else:
            array = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
            if dtype == np.float32:
                array[:] = np.nan  # 「面が無い」であって「位置 0 mm」ではない
            array.flush()
        arrays.append(array)
    return arrays[0], arrays[1], arrays[2]


def pack_volumes(out_dir: Path) -> Path:
    """``vols_padded.npy`` の実在スライスだけを study 順に詰めて ``vols.npy`` と ``offsets.npy`` を書く。

    ``stack_offsets.npy`` があれば series レイアウトで、その範囲ごとに詰め、``series.csv`` を study 順に並べ直す。
    ``vols.npy`` を最後に置くので、それが在ることが完成の印。``vols_padded.npy`` は消さない。
    """
    padded = np.load(out_dir / PADDED_VOLS, mmap_mode="r")
    present = np.load(out_dir / "masks.npy") != 0
    if (out_dir / STACK_OFFSETS).exists():
        stack = np.load(out_dir / STACK_OFFSETS)
        if present.shape != padded.shape[:1] or int(stack[-1]) != len(present):
            raise ValueError(f"masks {present.shape} and {STACK_OFFSETS} (total {int(stack[-1])}) do not match {PADDED_VOLS} {padded.shape}")
        units = read_series_units(out_dir)  # unit の長さが stack_offsets と食い違えばここで落ちる
        tmp_units = (out_dir / SERIES_UNITS).with_name("series.tmp.csv")
        units.drop(columns="study_row").to_csv(tmp_units, index=False)
        os.replace(tmp_units, out_dir / SERIES_UNITS)
    else:
        if present.shape != padded.shape[:2]:
            raise ValueError(f"masks {present.shape} do not match {PADDED_VOLS} {padded.shape}")
        stack = np.arange(len(present) + 1, dtype=np.int64) * present.shape[1]
        padded = padded.reshape(-1, *padded.shape[2:])
        present = present.reshape(-1)
    offsets = np.concatenate([[0], np.cumsum([int(present[stack[row] : stack[row + 1]].sum()) for row in range(len(stack) - 1)])]).astype(np.int64)
    tmp_vols, tmp_offsets = tmp_npy(out_dir / "vols.npy"), tmp_npy(out_dir / "offsets.npy")
    packed = np.lib.format.open_memmap(tmp_vols, mode="w+", dtype=np.uint8, shape=(int(offsets[-1]), *padded.shape[1:]))
    for row in range(len(stack) - 1):
        packed[offsets[row] : offsets[row + 1]] = padded[stack[row] : stack[row + 1]][present[stack[row] : stack[row + 1]]]
    packed.flush()
    del packed
    np.save(tmp_offsets, offsets)
    os.replace(tmp_offsets, out_dir / "offsets.npy")
    os.replace(tmp_vols, out_dir / "vols.npy")
    return out_dir / "vols.npy"


def prune_csv(path: Path, drop: set[str]) -> int:
    """この run がこれから焼き直す study の行だけを落とす。選択の外にある行はそのまま残す。"""
    if not path.exists() or not drop:
        return 0
    try:
        frame = pd.read_csv(path, dtype=str)
    except (ValueError, pd.errors.EmptyDataError):
        path.unlink()
        return 0
    if ID_COLUMN not in frame.columns:
        return 0
    kept = frame[~frame[ID_COLUMN].astype(str).isin(drop)]
    dropped = len(frame) - len(kept)
    if dropped:
        kept.to_csv(path, index=False)
    return dropped


def make_pool(workers: int) -> Any:
    """fork 方式の worker プール。``--workers <= 1`` なら ``None``。"""
    if not workers or workers <= 1:
        return None
    import multiprocessing as mp

    # fork なら worker が import 済みのモジュールをそのまま引き継ぐ。
    context = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
    return context.Pool(workers, initializer=_init_worker)


def map_tasks(tasks: list[dict[str, Any]], pool: Any, consume: Callable[[dict[str, Any]], None]) -> None:
    results: Iterable[dict[str, Any]] = pool.imap_unordered(_process_study, tasks, chunksize=1) if pool is not None else (_process_study(t) for t in tasks)
    for result in results:
        consume(result)


def progress_line(done: int, total: int, started: float) -> str:
    elapsed = max(time.time() - started, 1e-6)
    rate = done / elapsed
    eta = (total - done) / rate if rate > 0 else float("inf")
    return f"[{done}/{total}] {rate:.2f} studies/s elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m"


def _census_study(task: dict[str, Any]) -> list[dict[str, Any]]:
    """worker 側の処理。*選ばれた* series ごとに 1 行、ファイル数・in-span 枚数・幾何を返す。画素は読まない。"""
    study_id = task["study_id"]
    study_dir = Path(task["study_dir"])
    span = task["span"]
    rows: list[dict[str, Any]] = []
    for assignment in assign_slots(task["rows"]):
        if not assignment.present:
            continue
        header = probe_series(study_dir / assignment.series_uid, uid=assignment.series_uid)
        n_files = int(header["n_files"])
        rows.append(
            {
                ID_COLUMN: study_id,
                SERIES_COLUMN: assignment.series_uid,
                "slot_index": assignment.slot_index,
                "slot_name": assignment.slot_name,
                "n_files": n_files,
                "in_span": span_count(n_files, span),
                "SliceThickness": header["SliceThickness"],
                "SpacingBetweenSlices": header["SpacingBetweenSlices"],
                "PixelSpacing_0": header["PixelSpacing_0"],
                "Rows": header["Rows"],
                "Columns": header["Columns"],
                "SeriesDescription": header["SeriesDescription"],
            }
        )
    return rows


def census(studies: list[str], series: pd.DataFrame, split_dir: Path, out_dir: Path, params: PreprocessParams, workers: int) -> int:
    """``out/series_census.csv`` を書き、Slot ごとの in-span 枚数の分布を表示する。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    by_study = {str(k): v for k, v in series.groupby(ID_COLUMN)}
    tasks = [{"study_id": study_id, "study_dir": str(split_dir / study_id), "rows": [] if by_study.get(study_id) is None else by_study[study_id].to_dict(orient="records"), "span": params.span} for study_id in studies]
    writer = AppendWriter(out_dir / "series_census.csv", CENSUS_FIELDS)
    collected: list[dict[str, Any]] = []
    started = time.time()
    pool = make_pool(workers)
    try:
        results = pool.imap_unordered(_census_study, tasks, chunksize=8) if pool is not None else (_census_study(t) for t in tasks)
        for done, rows in enumerate(results, start=1):
            for row in rows:
                writer.write(row)
                collected.append(row)
            if done % 200 == 0:
                print(progress_line(done, len(tasks), started), flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        writer.close()

    frame = pd.DataFrame(collected)
    print(f"census: {len(frame)} picked series over {len(studies)} studies, span={params.span}", flush=True)
    if frame.empty:
        return 0
    print("in-span slice count per slot:", flush=True)
    print(f"  {'slot':<14} {'n':>6} {'p50':>6} {'p75':>6} {'p90':>6} {'p95':>6} {'max':>6} {'mean':>7}", flush=True)
    for index, name in enumerate(SLOT_NAMES):
        block = frame[frame["slot_index"] == index]["in_span"]
        if block.empty:
            continue
        quantiles = block.quantile([0.5, 0.75, 0.9, 0.95]).tolist()
        print(f"  {index} {name:<12} {len(block):>6} " + " ".join(f"{q:>6.0f}" for q in quantiles) + f" {block.max():>6} {block.mean():>7.1f}", flush=True)
    caps = []
    for index in range(NUM_SLOTS):
        block = frame[frame["slot_index"] == index]["in_span"]
        p90 = float(block.quantile(0.9)) if len(block) else 0.0  # 誰も埋めない Slot の cap は 0
        caps.append(int(-(-p90 // 4) * 4))
    print(f"p90 caps rounded up to a multiple of 4: {caps} (total {sum(caps)})", flush=True)
    return 0


def dry_run(studies: list[str], series: pd.DataFrame, split_dir: Path, slots: list[Any]) -> dict[str, Any]:
    """Slot 割り当てだけを行う。各 Slot を持つ study が何件か、study ごとに Slot が何個かを出す。"""
    by_study = {str(k): v for k, v in series.groupby(ID_COLUMN)}
    have_dicoms = split_dir.is_dir()
    per_slot = Counter()
    per_count = Counter()
    n_series_total = n_empty = n_slices_total = 0
    for study_id in studies:
        rows = by_study.get(study_id)
        rows = [] if rows is None else rows.to_dict(orient="records")
        n_series_total += len(rows)
        counts = {str(r[SERIES_COLUMN]): count_dcm_files(split_dir / study_id / str(r[SERIES_COLUMN])) for r in rows} if have_dicoms else {}
        assignments = assign_slots(rows, counts, slots=slots)
        present = [a.slot_index for a in assignments if a.present]
        per_slot.update(present)
        per_count[len(present)] += 1
        if have_dicoms:
            n_empty += sum(1 for a in assignments if a.present and a.n_slices == 0)
            n_slices_total += sum(a.n_slices for a in assignments if a.present)

    total = max(len(studies), 1)
    print(f"studies={len(studies)} series={n_series_total}", flush=True)
    print("slot presence:", flush=True)
    for index, name in enumerate(SLOT_NAMES):
        n = per_slot[index]
        print(f"  {index} {name:<16} {n:>7} studies ({100.0 * n / total:5.1f}%)", flush=True)
    print("slots per study:", flush=True)
    for count in sorted(per_count):
        n = per_count[count]
        print(f"  {count} slot(s): {n:>7} studies ({100.0 * n / total:5.1f}%)", flush=True)
    if have_dicoms:
        print(f"picked series with no .dcm file: {n_empty} | slices in picked series: {n_slices_total}", flush=True)
    else:
        print(f"no DICOMs under {split_dir} - slot picking is CSV-only, so the counts above are still exact", flush=True)
    return {"per_slot": dict(per_slot), "per_count": dict(per_count), "n_studies": len(studies), "n_empty": n_empty}
