from __future__ import annotations

import csv
import json
import math
import os
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from io import BytesIO
from pathlib import Path
from typing import Any, Literal, NamedTuple, TypeVar

import cv2
import numpy as np
import pydicom
from PIL import Image

try:  # pydicom >= 3
    from pydicom.pixels import apply_modality_lut
except ImportError:  # pragma: no cover - pydicom 2.x（Kaggle のイメージにはまだ載っていることがある）
    from pydicom.pixel_data_handlers.util import apply_modality_lut

from rsna_knee.data.dicom_fast import RawError, RawHeader, geometry_from_header, percentiles_from_counts, pixels_from_header, walk_header
from rsna_knee.data.slots import DEFAULT_SLOT_TABLE, Slot, SlotAssignment, _as_plane, get_slots, iter_series_rows, make_slots, num_slices, pick_series_for_slot, slot_offsets

__all__ = [
    "HEAD_BYTES",
    "HISTOGRAM_PERCENTILES",
    "IO_THREADS",
    "META_FIELDS",
    "DecodeError",
    "DecodedSlice",
    "PreprocessParams",
    "Series",
    "StudyTensor",
    "all_series_units",
    "apply_standard",
    "apply_window",
    "build_study",
    "build_study_variants",
    "count_dcm_files",
    "anterior_offset_px",
    "crop_box_exact",
    "crop_center_mm",
    "crop_resize_exact",
    "decode_slice_stored",
    "fit_histogram_standard",
    "list_dcm_files",
    "probe_series",
    "read_series",
    "resize",
    "resolve_slots",
    "select_slices",
    "slot_landmarks",
    "slot_window",
    "span_bounds",
    "span_count",
    "to_uint8",
]

#: ファイルを読むスレッド数。1 以下なら順に読む。notebook は ``dicom.IO_THREADS = n`` で変えられる。
IO_THREADS = 8
#: ヘッダだけ要るファイルで先頭から読むバイト数。足りなければ残りを読み足す。
HEAD_BYTES = 2048

with Path(__file__).with_name("corrupt_dicom_exclusions_v001.csv").open(newline="") as _registry:
    CORRUPT_SERIES = frozenset((row["study_instance_uid"], row["series_instance_uid"]) for row in csv.DictReader(_registry))

_T = TypeVar("_T")
_R = TypeVar("_R")


def _map_io(fn: Callable[[_T], _R], items: Iterable[_T]) -> list[_R]:
    """``[fn(x) for x in items]`` を、:data:`IO_THREADS` が 2 以上ならスレッドプールで。順序は保つ。

    ``fn`` は例外を自分で握って値にして返すこと（``executor.map`` は最初の例外で残りを捨てる）。
    """
    items = list(items)
    if IO_THREADS <= 1 or len(items) <= 1:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(max_workers=min(IO_THREADS, len(items))) as pool:
        return list(pool.map(fn, items))


#: ヒストグラム標準化のランドマーク。前景の 1 と 99 パーセンタイルが標準尺度の 0 と 1 で、間の 9 点が十分位。
HISTOGRAM_PERCENTILES: tuple[float, ...] = (1.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 99.0)


class DecodeError(RuntimeError):
    """1 枚のスライスを 2 次元の float 配列にできなかったときに送出する。"""


@dataclass(frozen=True)
class PreprocessParams:
    """DICOM から Stack を作るレシピ。native の cap 超過時は ``long_series_sampling`` で採取方式を選ぶ。"""

    crop_mm: float = 140.0
    size: int = 336
    span: tuple[float, float] = (0.15, 0.85)
    percentiles: tuple[float, float] = (2.0, 98.0)
    default_pixel_spacing: float = 0.5
    slots: str = DEFAULT_SLOT_TABLE
    caps: tuple[int, ...] | None = None
    native_spacing: bool = False
    # cap を超える series の採り方。"contiguous" は span の中央の連続 cap 枚（両端を捨てる）、"uniform" は span 全体から linspace で cap 枚（全範囲が残る）。
    long_series_sampling: Literal["contiguous", "uniform"] = "contiguous"
    # "area" は整数座標の中心ボックス + ``cv2.INTER_AREA``、"exact" は float 座標の中心ボックス + PIL bicubic。
    resample: str = "area"
    # True なら ``crop_mm`` のボックスが画像からはみ出す分を 0 で埋め、1 画素あたりの mm を保つ。False ははみ出す series を画像に収まるまで縮めた
    # ボックスで代用する（拡大になる）。このキーより前の cache と重みはすべて False で焼かれている。
    crop_pad: bool = False
    # Slot ごとの標準ランドマーク（:data:`HISTOGRAM_PERCENTILES` と同じ長さで 0 から 1）。None なら ``percentiles`` の輝度窓、
    # あればそれに代えてヒストグラム標準化（Nyúl–Udupa）を掛ける。
    histogram_standard: tuple[tuple[float, ...], ...] | None = None
    # "slots" は 5 つの Slot に 1 本ずつ、"all" は CSV の全 series を撮像面の最初の Slot の枚数上限で並べる。
    series_layout: Literal["slots", "all"] = "slots"
    # クロップの中心を患者の前方（LPS の -y）へずらす mm。面内に射影するので Axial は上下、Sagittal は左右に動き、
    # 前方が深さ方向の Coronal はほぼ動かない。``resample="exact"`` だけ。0 は画像中心のまま。
    anterior_shift_mm: float = 0.0
    # cache 作りだけの指定。True なら span 内が cap の 2 倍以上ある series から、互いに 1 枚ずつずれた sub-series（:func:`select_subseries`）も作り、
    # :attr:`StudyTensor.subseries` に載せる。Stack と推論の画素は変えない。native かつ series_layout="all" だけ。
    subseries: bool = False


def resolve_slots(params: PreprocessParams) -> list[Slot]:
    """:class:`PreprocessParams` が要求する Slot 表。``caps`` があれば ``slots`` より優先する。"""
    return make_slots(params.caps) if params.caps is not None else get_slots(params.slots)


_STR_TAGS = (
    "Manufacturer",
    "ManufacturerModelName",
    "SoftwareVersions",
    "ReceiveCoilName",
    "StationName",
    "InstitutionName",
    "PatientSex",
    "PatientAge",
    "BodyPartExamined",
    "Laterality",
    "SeriesDescription",
    "ProtocolName",
    "SequenceName",
    "ScanningSequence",
    "ScanOptions",
    "PhotometricInterpretation",
)
_FLOAT_TAGS = ("MagneticFieldStrength", "ImagingFrequency", "RepetitionTime", "EchoTime", "SliceThickness", "SpacingBetweenSlices")
_INT_TAGS = ("Rows", "Columns", "BitsStored")

#: ``series_meta.csv`` の列順。個々の series ではここの多くが ``None`` になる。
META_FIELDS: list[str] = [
    "Manufacturer",
    "ManufacturerModelName",
    "SoftwareVersions",
    "MagneticFieldStrength",
    "ImagingFrequency",
    "ReceiveCoilName",
    "StationName",
    "InstitutionName",
    "PatientSex",
    "PatientAge",
    "BodyPartExamined",
    "Laterality",
    "SeriesDescription",
    "ProtocolName",
    "SequenceName",
    "ScanningSequence",
    "ScanOptions",
    "RepetitionTime",
    "EchoTime",
    "SliceThickness",
    "SpacingBetweenSlices",
    "PixelSpacing_0",
    "PixelSpacing_1",
    "Rows",
    "Columns",
    "ImageOrientationPatient",
    "ImagePositionPatient",
    "TransferSyntaxUID",
    "PhotometricInterpretation",
    "BitsStored",
    "n_files",
    "sort_rule",
    "selected_indices",
]


def _multi_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) or type(value).__name__ == "MultiValue":
        return "|".join(str(x) for x in value)
    text = str(value).strip()
    return text or None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_seq(value: Any, length: int) -> list[float] | None:
    if value is None:
        return None
    try:
        out = [float(x) for x in value]
    except (TypeError, ValueError):
        return None
    return out if len(out) == length else None


def _empty_header(n_files: int, sort_rule: str) -> dict[str, Any]:
    return dict.fromkeys(META_FIELDS) | {"n_files": int(n_files), "sort_rule": sort_rule}


def _header_dict(ds: Any, n_files: int, sort_rule: str, first_ipp: Any = None) -> dict[str, Any]:
    """1 つの dataset から :data:`META_FIELDS` を抜き出す。無いタグは ``None`` になる。"""
    out: dict[str, Any] = dict.fromkeys(META_FIELDS)
    for tag in _STR_TAGS:
        out[tag] = _multi_str(getattr(ds, tag, None))
    for tag in _FLOAT_TAGS:
        out[tag] = _as_float(getattr(ds, tag, None))
    for tag in _INT_TAGS:
        out[tag] = _as_int(getattr(ds, tag, None))

    spacing = _float_seq(getattr(ds, "PixelSpacing", None), 2)
    out["PixelSpacing_0"] = spacing[0] if spacing else None
    out["PixelSpacing_1"] = spacing[1] if spacing else None

    iop = _float_seq(getattr(ds, "ImageOrientationPatient", None), 6)
    out["ImageOrientationPatient"] = "|".join(f"{v:g}" for v in iop) if iop else None

    ipp = _float_seq(first_ipp if first_ipp is not None else getattr(ds, "ImagePositionPatient", None), 3)
    out["ImagePositionPatient"] = "|".join(f"{v:g}" for v in ipp) if ipp else None

    meta = getattr(ds, "file_meta", None)
    out["TransferSyntaxUID"] = _multi_str(getattr(meta, "TransferSyntaxUID", None)) if meta is not None else None
    out["n_files"] = int(n_files)
    out["sort_rule"] = sort_rule
    return out


@dataclass
class Series:
    """DICOM の 1 series。並べ替え済みのファイルパス、スライスごとの幾何、スカラのヘッダタグを持つ。"""

    uid: str
    directory: Path
    paths: list[Path] = field(default_factory=list)
    positions: list[float] = field(default_factory=list)  # スライスごとの並べ替えキー
    pixel_spacings: list[float] = field(default_factory=list)  # スライスごとの PixelSpacing[0]（行方向）
    pixel_spacings_col: list[float] = field(default_factory=list)  # スライスごとの PixelSpacing[1]。タグが無ければ行方向と同じ
    orientations: list[list[float] | None] = field(default_factory=list)  # スライスごとの ImageOrientationPatient（6 値）。無ければ None
    sort_rules: list[str] = field(default_factory=list)  # 各スライスをどのキーで並べたか
    median_pixel_spacing: float = 0.5
    sort_rule: str = "empty"
    header: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.paths)


def list_dcm_files(directory: str | os.PathLike[str]) -> list[Path]:
    """series ディレクトリの ``*.dcm`` をファイル名順で返す。並べ替えの同着はこの順で決まる。"""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return [directory / name for name in sorted(e.name for e in os.scandir(directory) if e.is_file() and e.name.endswith(".dcm"))]


def count_dcm_files(directory: str | os.PathLike[str]) -> int:
    """series ディレクトリの ``.dcm`` ファイル数（存在しなければ 0）。"""
    return len(list_dcm_files(directory))


class _Record(NamedTuple):
    """スライス 1 枚分の並べ替えキーと幾何。``source`` はヘッダ dict を作る元、``ipp`` は ImagePositionPatient。"""

    position: float
    path: Path
    spacing: float
    rule: str
    source: Any
    spacing_col: float
    ipp: list[float] | None = None
    iop: list[float] | None = None


def _record_pydicom(path: Path, default_pixel_spacing: float) -> _Record:
    """スライス 1 枚のヘッダを pydicom で読む。読めなければ ``rule`` が ``"error"`` の :class:`_Record`。"""
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True)
        iop = getattr(ds, "ImageOrientationPatient", None)
        ipp = getattr(ds, "ImagePositionPatient", None)
        if iop is not None and ipp is not None and len(iop) == 6:
            row = np.array(iop[:3], float)
            col = np.array(iop[3:], float)
            position = float(np.dot(np.array(ipp, float), np.cross(row, col)))
            rule = "ipp_normal"
        else:
            position = float(getattr(ds, "InstanceNumber", 0) or 0)
            rule = "instance_number"
        spacing = getattr(ds, "PixelSpacing", None)
        spacing_col = float(spacing[1]) if spacing is not None and len(spacing) > 1 else None
        spacing = float(spacing[0]) if spacing is not None else default_pixel_spacing
        return _Record(position, path, spacing, rule, ds, spacing if spacing_col is None else spacing_col, None, [float(v) for v in iop] if iop is not None and len(iop) == 6 else None)
    except Exception:  # ヘッダが読めない場合。notebook はそのファイルを位置 0.0 のまま残す
        return _Record(0.0, path, default_pixel_spacing, "error", None, default_pixel_spacing)


def _read_header(path: Path) -> tuple[bytes, RawHeader] | None:
    """ファイル先頭の :data:`HEAD_BYTES` を読んで歩く（届かなければ残りも読む）。``None`` なら pydicom に任せる。"""
    try:
        with open(path, "rb") as fh:
            buf = fh.read(HEAD_BYTES)
            header = walk_header(buf)
            if header.pixel_offset is None:
                buf += fh.read()
                header = walk_header(buf)
    except Exception:
        return None
    return (buf, header) if header.pixel_offset is not None else None


def _record_from_raw(path: Path, buf: bytes, header: RawHeader, default_pixel_spacing: float) -> _Record | None:
    """生バイトのヘッダから :class:`_Record` を作る。pydicom と同じ結果を保証できない形なら ``None``。"""
    try:
        position, rule, spacing, spacing_col, ipp, iop = geometry_from_header(header, default_pixel_spacing)
    except Exception:
        return None
    return _Record(position, path, spacing, rule, buf, spacing_col, ipp, iop)


def _probe_dataset(source: Any) -> Any:
    """ヘッダ dict を作る dataset。生バイト経路のバイト列なら pydicom に *メモリ上で* 読ませる（ファイルは再び読まない）。"""
    if isinstance(source, bytes):
        return pydicom.dcmread(BytesIO(source), stop_before_pixels=True)
    return source


def _assemble_series(uid: str, directory: Path, records: list[_Record], default_pixel_spacing: float) -> Series:
    """スライスの :class:`_Record` を並べ替えて :class:`Series` にする。"""
    spacings = [rec.spacing for rec in records if rec.rule != "error"]
    records = sorted(records, key=lambda rec: rec.position)  # 安定ソート。同着はファイル名順のまま
    rules = sorted({rec.rule for rec in records})
    sort_rule = rules[0] if len(rules) == 1 else "mixed:" + "+".join(rules)
    median_spacing = float(np.median(spacings)) if spacings else default_pixel_spacing

    middle = records[len(records) // 2]
    probe = middle if middle.source is not None else next((rec for rec in records if rec.source is not None), None)
    first = next((rec for rec in records if rec.source is not None), None)
    first_ipp = None if first is None else first.ipp if isinstance(first.source, bytes) else getattr(first.source, "ImagePositionPatient", None)
    header = _header_dict(_probe_dataset(probe.source), len(records), sort_rule, first_ipp=first_ipp) if probe is not None else _empty_header(len(records), sort_rule)
    return Series(
        uid=uid,
        directory=directory,
        paths=[rec.path for rec in records],
        positions=[rec.position for rec in records],
        pixel_spacings=[rec.spacing for rec in records],
        pixel_spacings_col=[rec.spacing_col for rec in records],
        orientations=[rec.iop for rec in records],
        sort_rules=[rec.rule for rec in records],
        median_pixel_spacing=median_spacing,
        sort_rule=sort_rule,
        header=header,
    )


def read_series(directory: str | os.PathLike[str], uid: str | None = None, default_pixel_spacing: float = 0.5) -> Series:
    """全スライスのヘッダを読み、series を撮像順に並べて返す。

    並べ替えキーはスライスごとに、IOP（6 要素）と IPP が揃っていれば ``IPP . cross(IOP[:3], IOP[3:])``、
    無ければ ``InstanceNumber``、解釈できなければ 0.0。どれで並んだかは ``sort_rules`` に残る。
    """
    directory = Path(directory)
    uid = uid or directory.name
    paths = list_dcm_files(directory)
    if not paths:
        return Series(uid=uid, directory=directory, median_pixel_spacing=default_pixel_spacing, sort_rule="empty", header=_empty_header(0, "empty"))
    records: list[_Record] = []
    for path, head in zip(paths, _map_io(_read_header, paths)):
        record = _record_from_raw(path, head[0], head[1], default_pixel_spacing) if head is not None else None
        records.append(record if record is not None else _record_pydicom(path, default_pixel_spacing))
    return _assemble_series(uid, directory, records, default_pixel_spacing)


def probe_series(directory: str | os.PathLike[str], uid: str | None = None) -> dict[str, Any]:
    """デコードしない series のヘッダメタデータ。ヘッダを 1 枚読むだけ。"""
    directory = Path(directory)
    paths = list_dcm_files(directory)
    if not paths:
        return _empty_header(0, "empty")
    try:
        ds = pydicom.dcmread(str(paths[len(paths) // 2]), stop_before_pixels=True, force=True)
    except Exception:
        return _empty_header(len(paths), "not_read")
    return _header_dict(ds, len(paths), "not_read")


def span_bounds(n_slices: int, span: tuple[float, float] = (0.15, 0.85)) -> tuple[int, int]:
    """span が覆う添字の範囲（両端を含む）: ``int(n*lo)`` .. ``int(n*hi) - 1``。空の series では ``(0, -1)``。"""
    n = int(n_slices)
    if n <= 0:
        return 0, -1
    lo, hi = int(n * span[0]), int(n * span[1]) - 1
    return lo, min(max(hi, lo), n - 1)


def span_count(n_slices: int, span: tuple[float, float] = (0.15, 0.85)) -> int:
    """series のうち span の内側にあるスライスの枚数（空の series なら 0）。"""
    lo, hi = span_bounds(n_slices, span)
    return max(hi - lo + 1, 0)


def select_subseries(n_slices: int, cap: int, span: tuple[float, float]) -> list[list[int]]:
    """span 内が ``cap`` の 2 倍以上ある series の sub-series の添字。2 倍未満なら空。

    ``m = span 内の枚数 // cap`` 本。j 本目は span の先頭から j 枚目、j + m 枚目、... の列から、両端を含めて linspace で ``cap`` 枚を採る。
    どの sub-series も span のほぼ全体を覆い、互いに重ならない。
    """
    count = span_count(n_slices, span)
    m = count // cap if cap > 0 else 0
    if m < 2:
        return []
    lo, hi = span_bounds(n_slices, span)
    subseries = []
    for j in range(m):
        column = list(range(lo + j, hi + 1, m))
        subseries.append([column[int(i)] for i in np.linspace(0, len(column) - 1, cap).round().astype(int)])
    return subseries


def select_slices(n_slices: int, k: int, span: tuple[float, float] = (0.15, 0.85)) -> list[int]:
    """``lo, hi = int(n * span[0]), int(n * span[1]) - 1``（``hi = max(hi, lo)``）の間を ``np.linspace`` して
    偶数丸めし ``n - 1`` で頭打ちにした ``k`` 個の添字。重複は残すので短い series は同じスライスを 2 度返す。"""
    n = int(n_slices)
    if n <= 0 or k <= 0:
        return []
    if n == 1:
        return [0] * k
    lo, hi = int(n * span[0]), int(n * span[1]) - 1
    hi = max(hi, lo)
    picks = np.linspace(lo, hi, k).round().astype(int)
    return [int(min(int(p), n - 1)) for p in picks]


class DecodedSlice(NamedTuple):
    """デコードした 1 枚。``array`` が画素、``stored`` は unused bits 補正後の格納値（pydicom 経路なら ``None``）、
    ``rescale`` は ``(slope, intercept)``。``array == _rescale(stored, *rescale)`` が成り立つ。"""

    array: np.ndarray
    stored: np.ndarray | None
    rescale: tuple[float, float] | None


def _rescale(stored: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    """``apply_modality_lut`` の Rescale 経路: float64 で ``x * slope`` して ``+= intercept``、float32 に落とす。"""
    out = stored.astype(np.float64) * slope
    out += intercept
    return out.astype(np.float32)


def _read_file(path: Path) -> bytes | None:
    """ファイル全体を読む。読めなければ ``None``（デコード側が自分で読み直して :class:`DecodeError` にする）。"""
    try:
        return path.read_bytes()
    except OSError:
        return None


def _decode_slice_pydicom(path: str | os.PathLike[str], data: bytes | None = None) -> np.ndarray:
    """pydicom でスライス 1 枚を ``float32[H, W]`` にする。生バイト経路が扱えない形のフォールバック。

    ``apply_modality_lut`` が Rescale を適用し、``MONOCHROME1`` は ``max - x`` で反転する。
    """
    try:
        ds = pydicom.dcmread(str(path) if data is None else BytesIO(data))
        arr = np.asarray(apply_modality_lut(ds.pixel_array, ds)).astype(np.float32)
    except Exception as exc:  # デコーダのプラグインが無い、ファイルが途中で切れている、など
        raise DecodeError(f"{path}: {type(exc).__name__}: {exc}") from exc

    if arr.ndim == 3:  # RGB / パレット、あるいは multi-frame。1 面に落とす
        arr = arr.mean(axis=-1) if arr.shape[-1] in (3, 4) else arr[arr.shape[0] // 2]
    if arr.ndim != 2 or arr.size == 0:
        raise DecodeError(f"{path}: unsupported pixel array shape {arr.shape}")

    if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
        arr = float(arr.max()) - arr
    return np.ascontiguousarray(arr, dtype=np.float32)


def decode_slice_stored(path: str | os.PathLike[str], data: bytes | None = None) -> DecodedSlice:
    """1 枚のスライスを ``float32[H, W]`` にデコードし、格納値と Rescale も添えて返す。

    非圧縮 MONOCHROME2 は :mod:`rsna_knee.data.dicom_fast`、それ以外は :func:`_decode_slice_pydicom` で。
    """
    if data is None:
        try:
            data = Path(path).read_bytes()
        except Exception as exc:
            raise DecodeError(f"{path}: {type(exc).__name__}: {exc}") from exc
    try:
        stored, rescale = pixels_from_header(data, walk_header(data))
    except RawError:
        return DecodedSlice(_decode_slice_pydicom(path, data), None, None)
    array = stored.astype(np.float32) if rescale is None else _rescale(stored, *rescale)
    return DecodedSlice(array, stored, rescale)


def slot_window(arrays: list[np.ndarray], percentiles: tuple[float, float] = (2.0, 98.0)) -> tuple[float, float]:
    """1 つの Slot の輝度窓。採取したスライス *すべて* の生画素を連結したパーセンタイルで決める。空なら ``(0.0, 1.0)``。"""
    if not arrays:
        return 0.0, 1.0
    pixels = np.concatenate([np.asarray(a).ravel() for a in arrays])
    lo, hi = np.percentile(pixels, list(percentiles))
    return float(lo), float(hi)


def slot_landmarks(arrays: list[np.ndarray]) -> tuple[float, ...] | None:
    """1 つの Slot の前景（Slot の平均輝度より明るい画素）の :data:`HISTOGRAM_PERCENTILES`。前景が無ければ ``None``。"""
    if not arrays:
        return None
    pixels = np.concatenate([np.asarray(a, dtype=np.float32).ravel() for a in arrays])
    foreground = pixels[pixels > pixels.mean(dtype=np.float64)]
    if foreground.size == 0:
        return None
    return tuple(float(v) for v in np.percentile(foreground, list(HISTOGRAM_PERCENTILES)))


def _landmarks_from_counts(values: np.ndarray, counts: np.ndarray) -> tuple[float, ...] | None:
    """:func:`slot_landmarks` と同じランドマークを昇順の相異なる値と度数から求める。"""
    mean = float((values.astype(np.float64) * counts).sum() / counts.sum())
    foreground = values > mean
    if not foreground.any():
        return None
    return percentiles_from_counts(values[foreground], counts[foreground], HISTOGRAM_PERCENTILES)


def _slot_histogram(decoded: list[DecodedSlice]) -> tuple[np.ndarray, np.ndarray] | None:
    """Slot の全画素の度数表 ``(昇順の相異なる値, 度数)`` を 16 bit の格納値から作る。画素は並べ替えない。

    全スライスが格納値を持ち dtype と Rescale が揃っていなければ ``None``（画素から直接求める経路に任せる）。
    """
    if not decoded:
        return None
    first = decoded[0]
    if any(d.stored is None or d.stored.dtype != first.stored.dtype or d.rescale != first.rescale for d in decoded):
        return None
    counts = np.zeros(65536, dtype=np.int64)
    for d in decoded:
        counts += np.bincount(d.stored.reshape(-1).view(np.uint16), minlength=65536)
    codes = np.flatnonzero(counts)
    values = codes.astype(np.uint16).view(first.stored.dtype)  # uint16 のビット列を格納値の dtype に読み直す
    mapped = values.astype(np.float32) if first.rescale is None else _rescale(values, *first.rescale)
    if not np.isfinite(mapped).all():
        return None
    order = np.argsort(mapped, kind="stable")
    return mapped[order], counts[codes][order]


def apply_standard(img: np.ndarray, landmarks: Sequence[float], standard: Sequence[float]) -> np.ndarray:
    """ヒストグラム標準化。``landmarks`` を ``standard`` へ区分線形で写す。端の外側は ``standard`` の 0 と 1 に張り付く。"""
    return np.interp(np.asarray(img, dtype=np.float32), landmarks, standard).astype(np.float32)


def fit_histogram_standard(landmarks: Iterable[Sequence[tuple[float, ...] | None]], n_slots: int) -> tuple[tuple[float, ...], ...]:
    """Study ごと・Slot ごとの生ランドマークから標準ランドマークを作る。

    両端を 0 と 1 に線形で写したランドマークの位置を Slot ごとに平均する。両端が一致する Slot は数えない。
    """
    rows: list[list[np.ndarray]] = [[] for _ in range(n_slots)]
    for study in landmarks:
        for index, marks in enumerate(study):
            if marks is None or marks[-1] <= marks[0]:
                continue
            m = np.asarray(marks, dtype=np.float64)
            rows[index].append((m - m[0]) / (m[-1] - m[0]))
    empty = [index for index, row in enumerate(rows) if not row]
    if empty:
        raise ValueError(f"no usable landmarks for slots {empty}")
    return tuple(tuple(float(v) for v in np.mean(row, axis=0)) for row in rows)


def apply_window(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """``clip((x - lo) / (hi - lo + 1e-6), 0, 1)``。"""
    return np.clip((np.asarray(img, dtype=np.float32) - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)


#: ``crop_pad`` で 0 を足したあとの画像の 1 辺の上限（px）。140 mm で 0.017 mm/px に当たり、データの最小 0.073 mm/px より十分小さい。壊れた PixelSpacing で巨大な配列を作らないための上限。
MAX_PADDED_SIDE_PX = 8192


def _check_padded_side(side: float, spacing: float, crop_mm: float) -> None:
    """``crop_pad`` で作る画像の 1 辺が :data:`MAX_PADDED_SIDE_PX` を超えるなら例外にする。"""
    if side > MAX_PADDED_SIDE_PX:
        raise ValueError(f"crop_pad would build a {side:.0f} px side (crop_mm={crop_mm}, spacing={spacing}); PixelSpacing looks broken")


def crop_center_mm(img: np.ndarray, pixel_spacing: float | None, crop_mm: float = 140.0, pad: bool = False) -> tuple[np.ndarray, bool]:
    """``resample="area"`` の中心クロップ。``cpx = int(round(crop_mm / max(ps, 1e-3)))`` を ``min(H, W)`` で
    頭打ちにし、``(cropped, clamped)`` を返す。``clamped`` は短い方の辺いっぱいで代用したという意味。
    ``pad`` なら頭打ちにせず、足りない辺の両側に 0 を足して ``cpx`` 四方を切る（``clamped`` ははみ出したという意味）。"""
    h, w = img.shape[-2], img.shape[-1]
    ps = _as_float(pixel_spacing)
    if ps is None or not math.isfinite(ps) or ps <= 0:
        ps = 1e-3
    cpx = int(round(crop_mm / max(ps, 1e-3)))
    clamped = cpx > min(h, w)
    if clamped and pad:
        _check_padded_side(cpx, ps, crop_mm)
        pad_r, pad_c = max(cpx - h, 0), max(cpx - w, 0)
        img = np.pad(img, [(0, 0)] * (img.ndim - 2) + [(pad_r // 2, pad_r - pad_r // 2), (pad_c // 2, pad_c - pad_c // 2)])
        h, w = img.shape[-2], img.shape[-1]
    cpx = min(cpx, min(h, w))
    y0, x0 = (h - cpx) // 2, (w - cpx) // 2
    return img[..., y0 : y0 + cpx, x0 : x0 + cpx], clamped


def resize(img: np.ndarray, size: int = 336) -> np.ndarray:
    """float32 に対する ``cv2.resize(..., (size, size), interpolation=cv2.INTER_AREA)``。"""
    arr = np.ascontiguousarray(np.asarray(img, dtype=np.float32))
    return cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)


def _usable_spacing(value: float | None) -> float | None:
    """``value`` が有限かつ正の spacing なら float にして返す。そうでなければ ``None``。"""
    ps = _as_float(value)
    return ps if ps is not None and math.isfinite(ps) and ps > 0 else None


def crop_box_exact(h: int, w: int, spacing_row: float | None, spacing_col: float | None, crop_mm: float = 140.0, offset_px: tuple[float, float] = (0.0, 0.0), pad: bool = False) -> tuple[tuple[float, float, float, float], bool]:
    """:func:`crop_resize_exact` が切り出す連続ピクセル座標のボックス ``(x0, y0, x1, y1)``（画素の角が整数）と ``clamped``。

    元画素の座標を Stack の ``size x size`` へ写すとき（``(x - x0) / (x1 - x0) * size``）にも使う。``pad`` ならボックスを縮めず中心もずらしたままにするので、
    画像の外（負の座標や ``w``・``h`` を超える座標）に出ることがあり、``clamped`` はそのとき立つ。
    """
    ps_row = max(_usable_spacing(spacing_row) or 1e-3, 1e-3)
    ps_col = max(_usable_spacing(spacing_col) or ps_row, 1e-3)
    half_r = crop_mm / 2.0 / ps_row
    half_c = crop_mm / 2.0 / ps_col
    if pad:
        centre_r, centre_c = h / 2.0 + offset_px[0], w / 2.0 + offset_px[1]
        box = (centre_c - half_c, centre_r - half_r, centre_c + half_c, centre_r + half_r)
        return box, bool(box[0] < 0.0 or box[1] < 0.0 or box[2] > w or box[3] > h)
    clamped = bool(2.0 * half_r > h or 2.0 * half_c > w)
    if clamped:
        factor = min(h / (2.0 * half_r), w / (2.0 * half_c))
        half_r *= factor
        half_c *= factor
    centre_r = min(max(h / 2.0 + offset_px[0], half_r), h - half_r)
    centre_c = min(max(w / 2.0 + offset_px[1], half_c), w - half_c)
    # PIL は画像の外に出るボックスを拒む。clamp 後のボックスを浮動小数の丸め誤差から守る。
    return (max(centre_c - half_c, 0.0), max(centre_r - half_r, 0.0), min(centre_c + half_c, float(w)), min(centre_r + half_r, float(h))), clamped


def crop_resize_exact(img: np.ndarray, spacing_row: float | None, spacing_col: float | None, crop_mm: float = 140.0, size: int = 336, offset_px: tuple[float, float] = (0.0, 0.0), pad: bool = False) -> tuple[np.ndarray, bool]:
    """``resample="exact"``: ``crop_mm`` の中心クロップと ``size x size`` への再標本化を 1 パスで行う。

    ボックスを連続なピクセル座標（画像の中心は ``(H / 2, W / 2)``）に置き、半幅を *軸ごとに*
    ``crop_mm / 2 / spacing`` として PIL の bicubic で再標本化するので、中心は浮動小数の精度で合い、
    左右反転と可換で、``PixelSpacing`` が異方でも両軸ちょうど ``crop_mm`` を覆う。画像より大きいボックス
    は *両軸* を同じ係数で縮めて ``clamped`` を立てる。``pad`` なら縮めずに、はみ出す分を 0 で埋めて ``clamped`` を立てる（1 画素あたりの mm は全 series で同じ）。
    返すのは ``[0, 1]`` の ``float32[size, size]``。
    ``offset_px`` は中心を ``(行, 列)`` の画素だけずらす。``pad`` でなければ、ボックスが画像からはみ出す分はずらしを減らす。
    """
    if img.ndim != 2:
        raise ValueError(f"crop_resize_exact expects a 2-D slice, got shape {img.shape}")
    h, w = img.shape
    box, clamped = crop_box_exact(h, w, spacing_row, spacing_col, crop_mm, offset_px, pad)
    if pad and clamped:
        # PIL は画像の外に出るボックスを拒むので、はみ出す分だけ画像の周りに 0 を足してからボックスをずらす
        left, top = math.ceil(max(-box[0], 0.0)), math.ceil(max(-box[1], 0.0))
        right, bottom = math.ceil(max(box[2] - w, 0.0)), math.ceil(max(box[3] - h, 0.0))
        _check_padded_side(max(h + top + bottom, w + left + right), spacing_row, crop_mm)
        img = np.pad(np.asarray(img, dtype=np.float32), ((top, bottom), (left, right)))
        box = (box[0] + left, box[1] + top, min(box[2] + left, float(img.shape[1])), min(box[3] + top, float(img.shape[0])))
    image = Image.fromarray(np.ascontiguousarray(img, dtype=np.float32), mode="F")
    out = np.asarray(image.resize((int(size), int(size)), Image.BICUBIC, box=box), dtype=np.float32)
    return np.clip(out, 0.0, 1.0).astype(np.float32), clamped


def to_uint8(img: np.ndarray) -> np.ndarray:
    """``(x * 255).astype(uint8)``。丸めではなく切り捨て。"""
    return (np.asarray(img, dtype=np.float32) * 255).astype(np.uint8)


@dataclass
class SubSeries:
    """長い series 1 本から作った sub-series の 1 本（:func:`select_subseries`）。輝度窓とクロップは元の series と同じ。"""

    series_uid: str
    index: int  # 0 始まりの j
    count: int  # その series の sub-series の本数 m
    picks: list[int]  # series 内の添字（IPP の法線で並べた順）
    pixels: np.ndarray  # uint8[cap, size, size]
    mask: np.ndarray  # float32[cap]。デコードできたスライスが 1.0
    positions: np.ndarray  # float32[cap]。series の法線方向の mm。無ければ NaN


@dataclass
class StudyTensor:
    """1 Study のモデル入力と、それがどう作られたかの記録。"""

    study_id: str
    pixels: np.ndarray  # uint8[n_slices, size, size]。5 つの Slot か、series_layout="all" なら全 series を平坦化したもの
    mask: np.ndarray  # float32[n_slices]。実在の series から来たスライスが 1.0
    positions: np.ndarray  # float32[n_slices]。series の法線方向の mm。面が無いところは NaN
    assignments: list[SlotAssignment]
    warnings: list[str] = field(default_factory=list)
    series_headers: list[dict[str, Any]] = field(default_factory=list)
    # Slot ごとの生ランドマーク（:func:`slot_landmarks`）。欠けた Slot と前景の無い Slot は None
    landmarks: list[tuple[float, ...] | None] = field(default_factory=list)
    # ``PreprocessParams.subseries`` のときだけ。長い series の sub-series を Stack とは別に持つ
    subseries: list[SubSeries] = field(default_factory=list)


def build_study(study_dir: str | os.PathLike[str], series_rows: object, cfg: PreprocessParams | None = None, study_id: str | None = None) -> StudyTensor:
    """1 Study の ``uint8[n_slices, size, size]`` の Stack と、スライスごとの実在マスクを作る。

    Slot ごとに、series を選ぶ -> ``IPP . normal`` で並べる -> 設定に従って span の内側から採る ->
    ``apply_modality_lut`` -> Slot 単位のパーセンタイル窓かヒストグラム標準化 -> ``crop_mm`` の中心クロップ -> ``size x size``
    -> ``(x * 255).astype(uint8)``。マスクはデコード成功で 1。面の隣接は ``positions`` で見る。
    """
    return _build_study_shared(study_dir, series_rows, [cfg or PreprocessParams()], study_id)[0]


def _shares_decode(params: PreprocessParams, base: PreprocessParams) -> bool:
    """``params`` が ``base`` と読み出し・デコード・輝度窓を共有できるか。``size``・``resample``・``crop_pad`` だけが違うとき。"""
    return replace(params, size=base.size, resample=base.resample, crop_pad=base.crop_pad) == base


def build_study_variants(study_dir: str | os.PathLike[str], series_rows: object, params_list: Sequence[PreprocessParams], study_id: str | None = None) -> list[StudyTensor]:
    """同じ Study を ``params_list`` の各レシピで組み、同じ順の :class:`StudyTensor` を返す。

    ``size`` や ``resample`` だけ違う組なら読み出し・デコード・輝度窓を 1 度だけ行う。結果はどちらでも
    レシピごとに :func:`build_study` を呼んだものと同じ。
    """
    variants = list(params_list)
    if not variants:
        raise ValueError("params_list is empty")
    if all(_shares_decode(params, variants[0]) for params in variants[1:]):
        return _build_study_shared(study_dir, series_rows, variants, study_id)
    return [build_study(study_dir, series_rows, params, study_id=study_id) for params in variants]


def anterior_offset_px(iop: Sequence[float] | None, ps_row: float, ps_col: float, shift_mm: float) -> tuple[float, float] | None:
    """患者の前方（LPS の -y）へ ``shift_mm`` 動かす量を、面内の ``(行, 列)`` の画素にしたもの。IOP が無ければ None。

    IOP の前半 3 値は列番号が増える向き、後半 3 値は行番号が増える向き。前方ベクトルをそれぞれに射影する。
    """
    if iop is None:
        return None
    return (-shift_mm * float(iop[4]) / ps_row, -shift_mm * float(iop[1]) / ps_col)


def _crop_resample(windowed: np.ndarray, ps_row: float, ps_col: float, params: PreprocessParams, offset_px: tuple[float, float] = (0.0, 0.0)) -> tuple[np.ndarray, bool]:
    """窓掛け済みのスライス 1 枚 -> ``float32[size, size]``。``params.resample`` が経路を選ぶ。

    どちらも ``windowed`` を書き換えないので、同じスライスに複数のレシピを順に掛けられる。
    """
    if params.resample == "area":
        if params.anterior_shift_mm != 0.0:
            raise ValueError("anterior_shift_mm needs resample='exact'")
        cropped, clamped = crop_center_mm(windowed, ps_row, params.crop_mm, pad=params.crop_pad)
        return resize(cropped, params.size), clamped
    return crop_resize_exact(windowed, ps_row, ps_col, params.crop_mm, params.size, offset_px, pad=params.crop_pad)


def _add_subseries(series: Series, uid: str, cap: int, lo: float, hi: float, variants: list[PreprocessParams], out: list[list[SubSeries]], warnings: list[list[str]], prefix: str) -> None:
    """長い series の sub-series（:func:`select_subseries`）を、元の series と同じ輝度窓 ``[lo, hi]`` とクロップでレシピごとに作り ``out`` に足す。"""
    groups = select_subseries(len(series), cap, variants[0].span)
    if not groups:
        return
    paths = [series.paths[pick] for pick in sorted({pick for picks in groups for pick in picks})]
    blobs = dict(zip(paths, _map_io(_read_file, paths)))
    for j, picks in enumerate(groups):
        size_of = [params.size for params in variants]
        pixels = [np.zeros((cap, size, size), dtype=np.uint8) for size in size_of]
        mask = np.zeros(cap, dtype=np.float32)
        positions = np.full(cap, np.nan, dtype=np.float32)
        for k, pick in enumerate(picks):
            try:
                slice_ = decode_slice_stored(series.paths[pick], blobs[series.paths[pick]])
            except DecodeError as exc:
                for lines in warnings:
                    lines.append(f"{prefix} subseries={j} decode_failed slice={pick} file={series.paths[pick].name} err={str(exc)[:160]}")
                continue
            mask[k] = 1.0
            windowed = apply_window(slice_.array, lo, hi)
            spacing, spacing_col = series.pixel_spacings[pick], series.pixel_spacings_col[pick]
            ps_row = spacing if spacing > 0 else series.median_pixel_spacing
            ps_col = spacing_col if spacing_col > 0 else ps_row
            for v, params in enumerate(variants):
                shift = (0.0, 0.0)
                if params.anterior_shift_mm != 0.0:
                    shift = anterior_offset_px(series.orientations[pick], ps_row, ps_col, params.anterior_shift_mm) or (0.0, 0.0)
                resampled, _ = _crop_resample(windowed, ps_row, ps_col, params, shift)
                pixels[v][k] = to_uint8(resampled)
            if series.sort_rules[pick] == "ipp_normal":
                positions[k] = series.positions[pick]
        for v in range(len(variants)):
            out[v].append(SubSeries(series_uid=uid, index=j, count=len(groups), picks=list(picks), pixels=pixels[v], mask=mask.copy(), positions=positions.copy()))


def all_series_units(study_id: str, series_rows: object, slots: Sequence[Slot]) -> tuple[list[tuple[Slot, int, dict[str, Any]]], int]:
    """``series_layout="all"`` の unit の列 ``(撮像面の最初の Slot, Stack 上の先頭, series の行)`` と Stack 長。

    撮像面の分かる行を CSV の順に、:data:`CORRUPT_SERIES` を除いて並べる。各 unit は Slot の上限枚数と、その後ろの実在しない 1 枚を占める。
    """
    first_of_plane: dict[str, Slot] = {}
    for slot in slots:
        first_of_plane.setdefault(slot.plane.lower(), slot)
    units: list[tuple[Slot, int, dict[str, Any]]] = []
    n_slices = 0
    for row in iter_series_rows(series_rows):
        plane = _as_plane(row.get("Anatomical_Plane"))
        if plane is None or (study_id, str(row["SeriesInstanceUID"])) in CORRUPT_SERIES:
            continue
        units.append((first_of_plane[plane], n_slices, row))
        n_slices += first_of_plane[plane].n_slices + 1  # series の後ろに実在しない 1 枚を置き、窓が series をまたがないようにする
    return units, n_slices


def _build_study_shared(study_dir: str | os.PathLike[str], series_rows: object, variants: list[PreprocessParams], study_id: str | None) -> list[StudyTensor]:
    """読み出し・デコード・輝度窓を共有できるレシピの組に対する :func:`build_study` の本体。

    ``variants[0]`` が series の選択・並べ替え・採取・輝度窓を決める。組を確かめるのは呼び出し側。
    """
    base = variants[0]
    for params in variants:
        if params.long_series_sampling not in ("contiguous", "uniform"):
            raise ValueError(f"PreprocessParams.long_series_sampling must be 'contiguous' or 'uniform', got {params.long_series_sampling!r}")
        if params.resample not in ("area", "exact"):
            raise ValueError(f"PreprocessParams.resample must be 'area' or 'exact', got {params.resample!r}")
        if params.series_layout not in ("slots", "all"):
            raise ValueError(f"PreprocessParams.series_layout must be 'slots' or 'all', got {params.series_layout!r}")
        if params.subseries and (params.series_layout != "all" or not params.native_spacing):
            raise ValueError("PreprocessParams.subseries needs native_spacing and series_layout='all'")
    slots = resolve_slots(base)
    standards = base.histogram_standard
    if standards is not None and (len(standards) != len(slots) or any(len(row) != len(HISTOGRAM_PERCENTILES) or row[0] != 0.0 or row[-1] != 1.0 for row in standards)):
        raise ValueError(f"PreprocessParams.histogram_standard must hold {len(slots)} rows of {len(HISTOGRAM_PERCENTILES)} landmarks from 0 to 1, got {standards}")
    if base.series_layout == "all" and standards is not None:
        raise ValueError("series_layout='all' has no per-series histogram standard")
    study_dir = Path(study_dir)
    study_id = study_id or study_dir.name
    rows = list(iter_series_rows(series_rows))
    excluded = [row for row in rows if (study_id, str(row["SeriesInstanceUID"])) in CORRUPT_SERIES]
    rows = [row for row in rows if (study_id, str(row["SeriesInstanceUID"])) not in CORRUPT_SERIES]
    # (Slot, Stack 上の先頭, series の行)。"slots" は行を Slot ごとに選び、"all" は撮像面の分かる行をすべて CSV の順に並べる。
    units: list[tuple[Slot, int, dict[str, Any] | None]]
    if base.series_layout == "all":
        all_units, n_slices = all_series_units(study_id, rows, slots)
        units = [(slot, offset, row) for slot, offset, row in all_units]
    else:
        units = [(slot, offset, None) for slot, offset in zip(slots, slot_offsets(slots))]
        n_slices = num_slices(slots)

    pixels = [np.zeros((n_slices, params.size, params.size), dtype=np.uint8) for params in variants]
    positions = np.full(n_slices, np.nan, dtype=np.float32)
    present = np.zeros(n_slices, dtype=np.float32)
    used: set[str] = set()
    assignments: list[SlotAssignment] = []
    warnings: list[list[str]] = [[] for _ in variants]
    headers: list[dict[str, Any]] = []
    landmarks: list[tuple[float, ...] | None] = [None] * len(units)
    subseries: list[list[SubSeries]] = [[] for _ in variants]

    def warn_all(line: str) -> None:
        for lines in warnings:
            lines.append(line)

    for row in excluded:
        warn_all(f"study={study_id} series={row['SeriesInstanceUID']} quarantined_corrupt_series")

    for position, (slot, offset, row) in enumerate(units):
        if row is None:
            row = pick_series_for_slot(rows, slot.plane, slot.fluid, used)
        if row is None:
            assignments.append(SlotAssignment(slot.index, None, 0))
            warn_all(f"study={study_id} slot={slot.index} name={slot.name} missing")
            continue

        uid = str(row["SeriesInstanceUID"])
        used.add(uid)
        prefix = f"study={study_id} slot={slot.index} name={slot.name} series={uid}"
        series = read_series(study_dir / uid, uid=uid, default_pixel_spacing=base.default_pixel_spacing)
        assignments.append(SlotAssignment(slot.index, uid, len(series)))
        # Stack 上の行範囲と CSV の撮像面・液体強調フラグ。"all" の推論で窓の Slot を series ごとに決めるのに使う（:func:`series_row_slots`）。
        placement = {"stack_start": offset, "stack_stop": offset + slot.n_slices, "Anatomical_Plane": row.get("Anatomical_Plane"), "Fluid_Sensitive": row.get("Fluid_Sensitive")}
        headers.append({"StudyInstanceUID": study_id, "SeriesInstanceUID": uid, "slot_index": slot.index, "slot_name": slot.name, "assigned": 1, **placement, **series.header})
        if len(series) == 0:
            warn_all(f"{prefix} no_dcm_files -> slot zero-filled")
            continue
        if series.sort_rule != "ipp_normal":
            warn_all(f"{prefix} sort_rule={series.sort_rule}")

        wanted = slot.n_slices
        if base.native_spacing:  # span 内の面を cap まで採り、足りない分は padding にする
            wanted = min(slot.n_slices, span_count(len(series), base.span))
        picks = select_slices(len(series), wanted, base.span)
        if base.native_spacing and base.long_series_sampling == "contiguous" and picks:
            span_lo, span_hi = span_bounds(len(series), base.span)
            start = (span_lo + span_hi - (len(picks) - 1)) // 2
            picks = list(range(start, start + len(picks)))
        headers[-1]["selected_indices"] = json.dumps(picks)
        # 採るファイルを先に並列で読み切る（短い series は同じ添字を繰り返すので 1 ファイル 1 回）
        unique_paths = [series.paths[pick] for pick in sorted(set(picks))]
        blobs = dict(zip(unique_paths, _map_io(_read_file, unique_paths)))
        decoded: list[DecodedSlice | None] = []
        spacings: list[float] = []
        spacings_col: list[float] = []
        for pick in picks:
            try:
                decoded.append(decode_slice_stored(series.paths[pick], blobs[series.paths[pick]]))
                spacings.append(series.pixel_spacings[pick])
                spacings_col.append(series.pixel_spacings_col[pick])
            except DecodeError as exc:
                decoded.append(None)
                spacings.append(series.median_pixel_spacing)
                spacings_col.append(series.median_pixel_spacing)
                warn_all(f"{prefix} decode_failed slice={pick} file={series.paths[pick].name} err={str(exc)[:160]}")

        good = [d for d in decoded if d is not None]
        histogram = _slot_histogram(good)
        if histogram is None:
            lo, hi = slot_window([d.array for d in good], base.percentiles)
            landmarks[position] = slot_landmarks([d.array for d in good])
        else:
            lo, hi = percentiles_from_counts(*histogram, base.percentiles)
            landmarks[position] = _landmarks_from_counts(*histogram)
        standard = None if standards is None else standards[position]
        if standard is not None and good and landmarks[position] is None:
            warn_all(f"{prefix} no_foreground -> slot zero-filled")
        clamped = [0] * len(variants)
        for j, (pick, slice_, spacing, spacing_col) in enumerate(zip(picks, decoded, spacings, spacings_col)):  # len(picks) <= slot.n_slices
            if slice_ is None:  # デコードに失敗したスライスはゼロのまま。それでも場所は消費する
                continue
            present[offset + j] = 1.0
            if standard is None:
                windowed = apply_window(slice_.array, lo, hi)
            elif landmarks[position] is None:
                windowed = np.zeros(slice_.array.shape, dtype=np.float32)
            else:
                windowed = apply_standard(slice_.array, landmarks[position], standard)
            ps_row = spacing if spacing > 0 else series.median_pixel_spacing
            ps_col = spacing_col if spacing_col > 0 else ps_row
            for v, params in enumerate(variants):
                shift = (0.0, 0.0)
                if params.anterior_shift_mm != 0.0:
                    shifted = anterior_offset_px(series.orientations[pick], ps_row, ps_col, params.anterior_shift_mm)
                    if shifted is None:
                        warnings[v].append(f"{prefix} anterior_shift_skipped slice={pick} reason=no_orientation")
                    else:
                        shift = shifted
                resampled, was_clamped = _crop_resample(windowed, ps_row, ps_col, params, shift)
                clamped[v] += int(was_clamped)
                pixels[v][offset + j] = to_uint8(resampled)
            # mm なのは IPP を法線に射影したキーだけ。InstanceNumber は mm ではない
            if series.sort_rules[pick] == "ipp_normal":
                positions[offset + j] = series.positions[pick]
        for v, params in enumerate(variants):
            if clamped[v]:
                warnings[v].append(f"{prefix} {'crop_padded' if params.crop_pad else 'crop_clamped'} slices={clamped[v]}/{len(picks)} px={series.pixel_spacings[picks[0]]} crop_mm={params.crop_mm}")
            if base.native_spacing and len(picks) < slot.n_slices:
                warnings[v].append(f"{prefix} native_padded kept={len(picks)}/{slot.n_slices} n_files={len(series)}")
        if base.subseries:
            _add_subseries(series, uid, slot.n_slices, lo, hi, variants, subseries, warnings, prefix)

    for row in rows:  # どの Slot にも取られなかった series のメタデータ
        uid = row.get("SeriesInstanceUID")
        if uid is None or str(uid) in used:
            continue
        uid = str(uid)
        headers.append({"StudyInstanceUID": study_id, "SeriesInstanceUID": uid, "slot_index": None, "slot_name": None, "assigned": 0, **probe_series(study_dir / uid, uid=uid)})

    studies: list[StudyTensor] = []
    for v in range(len(variants)):
        mask = present.copy()
        variant_positions = positions.copy()
        variant_positions[mask == 0] = np.nan  # デコードできなかった面には位置も無い
        # レシピごとに別のオブジェクトを返す。呼び出し側が片方を書き換えても他方に漏れない
        record = dict(assignments=list(assignments), warnings=warnings[v], series_headers=[dict(h) for h in headers], landmarks=list(landmarks), subseries=subseries[v])
        studies.append(StudyTensor(study_id=study_id, pixels=pixels[v], mask=mask, positions=variant_positions, **record))
    return studies
