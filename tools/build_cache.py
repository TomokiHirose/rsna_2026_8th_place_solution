from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# BLAS のスレッド数を 1 にする。並列度は --workers が持つ。import した側に効かせないよう、コマンド起動時だけ。
if __name__ == "__main__":
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_var, "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import click  # noqa: E402
import numpy as np  # noqa: E402

from rsna_knee.data.cache_builder import (
    ID_COLUMN,
    INDEX_FIELDS,
    META_ROW_FIELDS,
    PADDED_VOLS,
    SERIES_INDEX_FIELDS,
    SERIES_UNIT_FIELDS,
    SERIES_UNITS,
    AppendWriter,
    census,
    check_recipe_matches,
    chunk_is_complete,
    dry_run,
    git_commit,
    indexed_studies,
    make_pool,
    map_tasks,
    open_cache_arrays,
    pack_subseries,
    pack_volumes,
    progress_line,
    prune_csv,
    read_split,
    select_studies,
    series_stack_offsets,
    tmp_npy,
    write_chunk,
    write_manifest,
)  # noqa: E402
from rsna_knee.data.dicom import PreprocessParams, resolve_slots  # noqa: E402
from rsna_knee.data.slots import DEFAULT_SLOT_TABLE, NUM_SLOTS, SLOT_TABLES, num_slices  # noqa: E402


def parse_shard(value: str | None) -> tuple[int, int]:
    if not value:
        return 0, 1
    try:
        index, total = (int(part) for part in value.split("/", 1))
    except ValueError as exc:
        raise SystemExit(f"--shard must look like 'i/n', got {value!r}") from exc
    if total < 1 or not (0 <= index < total):
        raise SystemExit(f"--shard out of range: {value!r}")
    return index, total


def parse_caps(value: str | None, slots: str) -> tuple[int, ...] | None:
    """``--caps 32,28,28,24,28`` を native の表にする。名前付きの表なら ``None``。"""
    if slots != "native":
        if value:
            raise SystemExit("--caps only applies to --slots native")
        return None
    if not value:
        raise SystemExit("--slots native needs --caps 'a,b,c,d,e'")
    try:
        caps = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise SystemExit(f"--caps must be {NUM_SLOTS} integers, got {value!r}") from exc
    if len(caps) != NUM_SLOTS or any(c < 0 for c in caps):
        raise SystemExit(f"--caps must be {NUM_SLOTS} non-negative integers, got {value!r}")
    return caps


def parse_span(value: str | None) -> tuple[float, float]:
    if not value:
        return PreprocessParams.span
    try:
        lo, hi = (float(part) for part in value.split(",", 1))
    except ValueError as exc:
        raise SystemExit(f"--span must look like 'lo,hi', got {value!r}") from exc
    return lo, hi


def load_standard(path: Path | None) -> tuple[tuple[float, ...], ...] | None:
    """``tools/fit_histogram_standard.py`` が書いた JSON の標準ランドマーク。"""
    if path is None:
        return None
    return tuple(tuple(float(v) for v in row) for row in json.loads(path.read_text())["standard"])


@click.command(help="Decode DICOM studies into a uint8 tensor cache.")
@click.option("--data-dir", required=True, type=click.Path(path_type=Path), help="Competition root (holds {split}.csv and {split}_series/).")
@click.option("--split", required=True, type=click.Choice(["train", "test"]), help="Which split to build.")
@click.option("--out", required=True, type=click.Path(path_type=Path), help="Output cache directory.")
@click.option("--shard", default=None, help="Shard selector 'i/n': keep studies whose index % n == i.")
@click.option("--limit", type=int, default=None, help="Process at most N studies (applied after sharding).")
@click.option("--workers", type=int, default=1, help="Worker processes (<=1 runs in-process).")
@click.option("--slots", default=DEFAULT_SLOT_TABLE, type=click.Choice([*sorted(SLOT_TABLES), "native"]), help="Slot table: 44 = 12/10/8/6/8, 64 = 18/14/12/8/12, native = every in-span slice capped by --caps.")
@click.option("--caps", default=None, help="Per-slot slice counts 'a,b,c,d,e' (required by --slots native).")
@click.option("--crop-mm", type=float, default=PreprocessParams.crop_mm, help="Physical centre-crop size in mm.")
@click.option("--anterior-shift-mm", type=float, default=PreprocessParams.anterior_shift_mm, help="Shift the crop centre this many mm toward the patient's anterior (in-plane projection; needs --resample exact).")
@click.option("--size", type=int, default=PreprocessParams.size, help="Output side length in pixels.")
@click.option("--span", default=None, help="Slice span as 'lo,hi' fractions (default 0.15,0.85 = the published corpus).")
@click.option("--resample", default=PreprocessParams.resample, type=click.Choice(["area", "exact"]), help="Crop + resize: area = integer centre box + INTER_AREA (the published recipe), exact = float centre box + PIL bicubic.")
@click.option("--long-series-sampling", default="uniform", type=click.Choice(["contiguous", "uniform"]), help="Native series above cap: uniform = linspace across the whole span (default); contiguous = central slices only (legacy).")
@click.option("--crop-pad/--no-crop-pad", default=False, help="Zero-pad when the crop box exceeds the image so mm per pixel stays fixed. Default shrinks the box to fit and magnifies it (team caches).")
@click.option("--subseries", is_flag=True, help="Also store shifted sub-series of series whose span holds >= 2x the cap (subseries.csv + sub_*.npy; --series-layout all --memmap with native slots only).")
@click.option("--histogram-standard", type=click.Path(path_type=Path), default=None, help="JSON from tools/fit_histogram_standard.py; replaces the percentile window with Nyul-Udupa histogram standardization.")
@click.option("--series-layout", default="slots", type=click.Choice(["slots", "all"]), help="slots = one series per fixed Slot; all = every series with a known plane, capped by its plane's first Slot (--memmap only).")
@click.option("--chunk-size", type=int, default=0, help="Studies per chunk file (the Kaggle shard layout); mutually exclusive with --memmap.")
@click.option("--memmap", is_flag=True, help="Write vols/ids/masks/positions/offsets (the merged layout, vols packed to present slices); for local builds.")
@click.option("--compress", is_flag=True, help="Write one deflated chunk_{k}.npz per chunk instead of four .npy files.")
@click.option("--staging-dir", type=click.Path(path_type=Path), default=None, help="Where the uncompressed chunk is assembled before --compress (default: --out).")
@click.option("--code-commit", default=None, help="Value for manifest.json's code_commit (default: git HEAD of this checkout).")
@click.option("--caps-rule", default=None, help="How --caps was chosen; recorded in manifest.json.")
@click.option("--census", "census_only", is_flag=True, help="Write series_census.csv (file / in-span counts per picked series) and stop.")
@click.option("--dry-run", is_flag=True, help="Only run slot assignment and print presence counts.")
@click.option("--progress-every", type=int, default=50, help="Print a progress line every N studies.")
def main(**options: Any) -> None:
    """Build one cache; the layout is exactly one of ``--memmap`` or ``--chunk-size N``."""
    raise SystemExit(build(SimpleNamespace(**options)))


def build(args: SimpleNamespace) -> int:
    split_dir = args.data_dir / f"{args.split}_series"
    studies, series = read_split(args.data_dir, args.split)
    studies = select_studies(studies, parse_shard(args.shard), args.limit)
    caps = parse_caps(args.caps, args.slots)
    standard = load_standard(args.histogram_standard)
    sampling = args.long_series_sampling
    if args.subseries and (args.series_layout != "all" or not args.memmap or args.slots != "native"):
        raise SystemExit("--subseries needs --series-layout all, --memmap and --slots native")
    if args.series_layout == "all" and (not args.memmap or args.chunk_size or standard is not None):
        raise SystemExit("--series-layout all needs --memmap and cannot be combined with --chunk-size or --histogram-standard")
    params = PreprocessParams(
        crop_mm=args.crop_mm,
        size=args.size,
        span=parse_span(args.span),
        slots=args.slots,
        caps=caps,
        native_spacing=args.slots == "native",
        resample=args.resample,
        long_series_sampling=sampling,
        crop_pad=args.crop_pad,
        subseries=args.subseries,
        histogram_standard=standard,
        series_layout=args.series_layout,
        anterior_shift_mm=args.anterior_shift_mm,
    )
    slots = resolve_slots(params)

    if args.dry_run:
        dry_run(studies, series, split_dir, slots)
        return 0
    if args.census_only:
        return census(studies, series, split_dir, args.out, params, args.workers)
    chunk_size = int(args.chunk_size or 0)
    if args.memmap == (chunk_size > 0):
        raise SystemExit("pick exactly one layout: --memmap (merged vols.npy, for local builds) or --chunk-size N (chunk files, for the Kaggle shard kernels)")
    n_slices = num_slices(slots)
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.memmap and (out_dir / "vols.npy").exists():
        raise SystemExit(f"{out_dir / 'vols.npy'} is already packed; build into a fresh --out")
    check_recipe_matches(out_dir, params, args.split)
    write_manifest(out_dir, params, args.split, args.shard, len(studies), git_commit(args.code_commit), args.caps_rule)

    by_study = {str(k): v for k, v in series.groupby(ID_COLUMN)}

    def make_task(study_id: str, **extra: Any) -> dict[str, Any]:
        return {"study_id": study_id, "study_dir": str(split_dir / study_id), "rows": [] if by_study.get(study_id) is None else by_study[study_id].to_dict(orient="records"), "params": params, **extra}

    staging_dir: Path = args.staging_dir or out_dir
    staging_dir.mkdir(parents=True, exist_ok=True)
    for stale in staging_dir.glob("chunk_*.tmp.npy"):  # 途中で殺された run の残骸
        stale.unlink(missing_ok=True)
    volumes = masks_array = positions_array = None
    todo_rows: list[tuple[int, str]] = []
    series_layout = params.series_layout == "all"
    # series レイアウトは study i の Stack を 1 次元の並びの [stack[i], stack[i + 1]) に置く。長さは series 表だけで決まる
    stack = series_stack_offsets(studies, series, params) if series_layout else None
    if args.memmap:
        volumes, masks_array, positions_array = open_cache_arrays(out_dir, studies, n_slices, params.size, stack)
        done = indexed_studies(out_dir)
        todo_rows = [(row, uid) for row, uid in enumerate(studies) if uid not in done]
        n_todo = len(todo_rows)
        prune_csv(out_dir / "series_meta.csv", {uid for _, uid in todo_rows})
        prune_csv(out_dir / SERIES_UNITS, {uid for _, uid in todo_rows})
        print(f"split={args.split} slots={args.slots} size={args.size} span={params.span} studies={len(studies)} cached={len(studies) - n_todo} to_do={n_todo} out={out_dir}/{PADDED_VOLS}", flush=True)
    else:
        groups = [studies[i : i + chunk_size] for i in range(0, len(studies), chunk_size)]
        indexed = indexed_studies(out_dir)
        finished = {k for k, group in enumerate(groups) if chunk_is_complete(out_dir, k, group, indexed)}
        todo_groups = [(k, group) for k, group in enumerate(groups) if k not in finished]
        n_todo = sum(len(group) for _, group in todo_groups)
        redo = {uid for _, group in todo_groups for uid in group}
        dropped = prune_csv(out_dir / "index.csv", redo) + prune_csv(out_dir / "series_meta.csv", redo)
        if dropped:
            print(f"resume: dropped {dropped} csv rows belonging to the {len(todo_groups)} chunks being rebuilt", flush=True)
        print(f"split={args.split} slots={args.slots} size={args.size} span={params.span} studies={len(studies)} chunks={len(groups)}x{chunk_size} done_chunks={len(finished)} to_do={n_todo} out={out_dir}", flush=True)

    index_writer = AppendWriter(out_dir / "index.csv", SERIES_INDEX_FIELDS if series_layout else INDEX_FIELDS)
    meta_writer = AppendWriter(out_dir / "series_meta.csv", META_ROW_FIELDS)
    unit_writer = AppendWriter(out_dir / SERIES_UNITS, SERIES_UNIT_FIELDS) if series_layout else None
    warnings_log = (out_dir / "warnings.log").open("a", encoding="utf-8")
    started = time.time()
    n_done = n_failed = n_warnings = 0
    pending: list[dict[str, Any]] = []

    def consume(result: dict[str, Any], buffer: list[dict[str, Any]] | None) -> None:
        """worker の結果 1 件を CSV に流す（chunk が確定するまでは ``buffer`` に貯める）。"""
        nonlocal n_failed, n_warnings
        if result["error"]:
            n_failed += 1
            warnings_log.write(f"study={result['study_id']} FAILED {result['error']}\n")
            warnings_log.flush()
            return
        if buffer is not None:
            buffer.append(result)
        else:
            for unit_row in result["unit_rows"]:  # index の行が完了の印なので、unit の行を先に書く
                unit_writer.write(unit_row)
            index_writer.write(result["index_row"])
            for meta_row in result["meta_rows"]:
                meta_writer.write(meta_row)
        for line in result["warnings"]:
            warnings_log.write(line + "\n")
        n_warnings += len(result["warnings"])
        if result["warnings"]:
            warnings_log.flush()

    def flush(buffer: list[dict[str, Any]]) -> None:
        for result in buffer:
            index_writer.write(result["index_row"])
            for meta_row in result["meta_rows"]:
                meta_writer.write(meta_row)
        buffer.clear()

    pool = make_pool(args.workers)
    try:
        if args.memmap:

            def collect_row(result: dict[str, Any]) -> None:
                nonlocal n_done
                if not result["error"]:
                    rows = result["position"] if stack is None else slice(int(stack[result["position"]]), int(stack[result["position"] + 1]))
                    masks_array[rows] = np.asarray(result["mask"], dtype=np.uint8)
                    positions_array[rows] = np.asarray(result["positions"], dtype=np.float32)
                consume(result, None)
                n_done += 1
                if args.progress_every and n_done % args.progress_every == 0:
                    masks_array.flush()
                    positions_array.flush()
                    print(progress_line(n_done, n_todo, started), flush=True)

            tasks = [make_task(uid, chunk_path=str(out_dir / PADDED_VOLS), position=row, stack=None if stack is None else (int(stack[row]), int(stack[row + 1]))) for row, uid in todo_rows]
            map_tasks(tasks, pool, collect_row)
            for array in (volumes, masks_array, positions_array):
                array.flush()
        else:
            for chunk_index, group in todo_groups:
                tmp_vols = tmp_npy(staging_dir / f"chunk_{chunk_index:03d}.npy")
                volumes = np.lib.format.open_memmap(tmp_vols, mode="w+", dtype=np.uint8, shape=(len(group), n_slices, params.size, params.size))
                volumes.flush()
                del volumes  # worker が同じファイルを自分で map する
                masks = np.zeros((len(group), n_slices), dtype=np.uint8)
                slice_positions = np.full((len(group), n_slices), np.nan, dtype=np.float32)
                tasks = [make_task(uid, chunk_path=str(tmp_vols), position=i) for i, uid in enumerate(group)]

                def collect(result: dict[str, Any], masks: np.ndarray = masks, slice_positions: np.ndarray = slice_positions) -> None:
                    nonlocal n_done
                    if not result["error"]:
                        masks[result["position"]] = np.asarray(result["mask"], dtype=np.uint8)
                        slice_positions[result["position"]] = np.asarray(result["positions"], dtype=np.float32)
                    consume(result, pending)
                    n_done += 1
                    if args.progress_every and n_done % args.progress_every == 0:
                        print(progress_line(n_done, n_todo, started), flush=True)

                map_tasks(tasks, pool, collect)
                flush(pending)
                written = write_chunk(out_dir, chunk_index, tmp_vols, np.asarray(group), masks, slice_positions, compress=args.compress)
                print(f"chunk {chunk_index:03d} written rows={len(group)} bytes={sum(p.stat().st_size for p in written)} {progress_line(n_done, n_todo, started)}", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        index_writer.close()
        meta_writer.close()
        if unit_writer is not None:
            unit_writer.close()
        warnings_log.close()

    print(f"done {progress_line(n_done, n_todo, started)} failed={n_failed} warnings={n_warnings}", flush=True)
    if n_failed:
        print(f"FAILED: {n_failed} studies could not be decoded -- see {out_dir / 'warnings.log'} and rerun to retry them", flush=True)
        return 1
    if args.memmap:  # 全 study が焼けてから実在スライスだけに詰め、組み立て用の padded を消す
        del volumes, masks_array, positions_array
        packed = pack_volumes(out_dir)
        (out_dir / PADDED_VOLS).unlink()
        print(f"packed {packed} bytes={packed.stat().st_size}", flush=True)
        if params.subseries:
            table = pack_subseries(out_dir)
            print(f"packed subseries rows={len(table)} studies={table[ID_COLUMN].nunique()} bytes={(out_dir / 'sub_vols.npy').stat().st_size}", flush=True)
    return 0


if __name__ == "__main__":
    main()
