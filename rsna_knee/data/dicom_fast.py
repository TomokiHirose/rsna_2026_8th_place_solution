from __future__ import annotations

import struct
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

__all__ = ["RawError", "RawHeader", "geometry_from_header", "percentiles_from_counts", "pixels_from_header", "walk_header"]

EXPLICIT_VR_LITTLE_ENDIAN = b"1.2.840.10008.1.2.1"

#: 予約 2 バイト + 4 バイト長のヘッダを持つ VR。pydicom の ``EXPLICIT_VR_LENGTH_32`` と同じ集合
LONG_VRS = frozenset((b"OB", b"OD", b"OF", b"OL", b"OV", b"OW", b"SQ", b"SV", b"UC", b"UN", b"UR", b"UT", b"UV"))
#: DICOM PS3.5 の VR 全部。pydicom の ``STANDARD_VR`` と同じ集合。これ以外の 2 バイトは VR ではない
KNOWN_VRS = frozenset(b"AE AS AT CS DA DS DT FD FL IS LO LT OB OD OF OL OV OW PN SH SL SQ SS ST SV TM UC UI UL UN UR US UT UV".split())

TAG_INSTANCE_NUMBER = 0x00200013
TAG_IMAGE_POSITION = 0x00200032
TAG_IMAGE_ORIENTATION = 0x00200037
TAG_SAMPLES_PER_PIXEL = 0x00280002
TAG_PHOTOMETRIC = 0x00280004
TAG_NUMBER_OF_FRAMES = 0x00280008
TAG_ROWS = 0x00280010
TAG_COLUMNS = 0x00280011
TAG_PIXEL_SPACING = 0x00280030
TAG_BITS_ALLOCATED = 0x00280100
TAG_BITS_STORED = 0x00280101
TAG_PIXEL_REPRESENTATION = 0x00280103
TAG_RESCALE_INTERCEPT = 0x00281052
TAG_RESCALE_SLOPE = 0x00281053
TAG_MODALITY_LUT_SEQUENCE = 0x00283000
TAG_PIXEL_DATA = 0x7FE00010
TAG_TRANSFER_SYNTAX = 0x00020010

#: 歩きながら値を控えるタグ（並べ替えの幾何と、画素の形）。それ以外の element は長さだけ見て飛ばす
_GEOMETRY_TAGS = frozenset((TAG_INSTANCE_NUMBER, TAG_IMAGE_POSITION, TAG_IMAGE_ORIENTATION, TAG_PIXEL_SPACING))
_PIXEL_TAGS = frozenset((TAG_SAMPLES_PER_PIXEL, TAG_PHOTOMETRIC, TAG_NUMBER_OF_FRAMES, TAG_ROWS, TAG_COLUMNS, TAG_BITS_ALLOCATED, TAG_BITS_STORED, TAG_PIXEL_REPRESENTATION, TAG_RESCALE_INTERCEPT, TAG_RESCALE_SLOPE, TAG_MODALITY_LUT_SEQUENCE))
_WANTED = _GEOMETRY_TAGS | _PIXEL_TAGS

_unpack_hh = struct.Struct("<HH").unpack_from
_unpack_h = struct.Struct("<H").unpack_from
_unpack_i = struct.Struct("<I").unpack_from


class RawError(ValueError):
    """生バイト歩きが扱わない構造に出会った。呼び出し側はそのファイルを pydicom に落とす。"""


class RawHeader(NamedTuple):
    """Explicit VR Little Endian のファイル先頭を歩いた結果。

    ``elements`` は :data:`_WANTED` のタグだけを ``tag -> (VR, 値のバイト列)`` で持つ。``pixel_offset`` は
    PixelData の *値* の先頭 offset で、歩いたバッファに PixelData の element header が無ければ ``None``。
    """

    elements: dict[int, tuple[bytes, bytes]]
    pixel_offset: int | None
    pixel_length: int | None


def _element_header(buf: bytes, pos: int, end: int) -> tuple[int, bytes, int, int] | None:
    """``pos`` にある element の ``(tag, VR, 値の先頭 offset, 長さ)``。ヘッダがバッファに収まらなければ ``None``。"""
    if pos + 8 > end:
        return None
    group, elem = _unpack_hh(buf, pos)
    vr = buf[pos + 4 : pos + 6]
    if vr in LONG_VRS:
        if pos + 12 > end:
            return None
        return (group << 16) | elem, vr, pos + 12, _unpack_i(buf, pos + 8)[0]
    if vr not in KNOWN_VRS:
        raise RawError(f"unknown VR {vr!r} at offset {pos}")
    return (group << 16) | elem, vr, pos + 8, _unpack_h(buf, pos + 6)[0]


def walk_header(buf: bytes) -> RawHeader:
    """DICOM ファイルの先頭バイト列を PixelData（またはバッファの終わり）まで歩き、必要なタグを集める。

    file meta が group length で始まらない、Explicit VR Little Endian でない、未定義長、item delimiter、
    未知の VR は :class:`RawError`。値がバッファの外なら ``pixel_offset=None``（呼び出し側が読み足す）。
    """
    end = len(buf)
    if end < 144 or buf[128:132] != b"DICM":
        raise RawError("no DICM magic")
    if buf[132:140] != b"\x02\x00\x00\x00UL\x04\x00":
        raise RawError("file meta does not start with (0002,0000) UL")
    meta_end = 144 + _unpack_i(buf, 140)[0]
    if meta_end > end:
        return RawHeader({}, None, None)
    transfer_syntax = None
    pos = 144
    while pos < meta_end:
        header = _element_header(buf, pos, meta_end)
        if header is None:
            raise RawError("truncated file meta")
        tag, _vr, vstart, length = header
        if tag >> 16 != 0x0002 or length == 0xFFFFFFFF or vstart + length > meta_end:
            raise RawError("file meta is inconsistent with its group length")
        if tag == TAG_TRANSFER_SYNTAX:
            transfer_syntax = buf[vstart : vstart + length].rstrip(b"\x00 ")
        pos = vstart + length
    if transfer_syntax != EXPLICIT_VR_LITTLE_ENDIAN:
        raise RawError(f"transfer syntax {transfer_syntax!r} is not Explicit VR Little Endian")

    elements: dict[int, tuple[bytes, bytes]] = {}
    while True:
        header = _element_header(buf, pos, end)
        if header is None:
            return RawHeader(elements, None, None)
        tag, vr, vstart, length = header
        if tag == TAG_PIXEL_DATA:
            return RawHeader(elements, vstart, length)
        group = tag >> 16
        if group == 0x0002 or group == 0xFFFE or length == 0xFFFFFFFF:
            raise RawError(f"unsupported element {tag:08X} (length {length})")
        pos = vstart + length
        if tag in _WANTED:
            if pos > end:
                return RawHeader(elements, None, None)
            elements[tag] = (vr, buf[vstart:pos])


def _decimal_strings(elements: dict[int, tuple[bytes, bytes]], tag: int) -> list[float] | None:
    """DS の値を float の列にする（iso8859 decode → strip → ``\\`` で分割 → float）。

    タグが無ければ ``None``。VR が DS でない、空の要素がある、float にならないものは :class:`RawError`。
    """
    element = elements.get(tag)
    if element is None:
        return None
    vr, raw = element
    if vr != b"DS":
        raise RawError(f"tag {tag:08X} has VR {vr!r}, not DS")
    items = raw.decode("iso8859").strip().rstrip(" \x00").split("\\")
    if any(item.strip() == "" for item in items):
        raise RawError(f"tag {tag:08X} has an empty DS component")
    try:
        return [float(item) for item in items]
    except ValueError as exc:
        raise RawError(f"tag {tag:08X}: {exc}") from exc


def _integer_string(elements: dict[int, tuple[bytes, bytes]], tag: int) -> int | None:
    """IS の値を int にする（1 要素のプレーンな整数だけ。それ以外は :class:`RawError`）。タグが無ければ ``None``。"""
    element = elements.get(tag)
    if element is None:
        return None
    vr, raw = element
    if vr != b"IS":
        raise RawError(f"tag {tag:08X} has VR {vr!r}, not IS")
    items = raw.decode("iso8859").rstrip(" \x00").split("\\")
    if len(items) != 1:
        raise RawError(f"tag {tag:08X} is multi-valued")
    try:
        return int(items[0])
    except ValueError as exc:
        raise RawError(f"tag {tag:08X}: {exc}") from exc


def _unsigned_short(elements: dict[int, tuple[bytes, bytes]], tag: int) -> int:
    """US の値（1 要素）。無い、VR が違う、長さが 2 でないなら :class:`RawError`。"""
    element = elements.get(tag)
    if element is None:
        raise RawError(f"tag {tag:08X} is missing")
    vr, raw = element
    if vr != b"US" or len(raw) != 2:
        raise RawError(f"tag {tag:08X} is not a single US")
    return _unpack_h(raw, 0)[0]


def geometry_from_header(header: RawHeader, default_pixel_spacing: float) -> tuple[float, str, float, float, list[float] | None, list[float] | None]:
    """``(position, rule, spacing, spacing_col, ipp, iop)`` を生バイトから作る。``iop`` は 6 値のときだけ。

    IOP / PixelSpacing が 1 要素だけ、IPP が 3 要素でないなど、pydicom と同じ結果を保証できない形は
    :class:`RawError` にして、そのファイルを pydicom に任せる。
    """
    elements = header.elements
    iop = _decimal_strings(elements, TAG_IMAGE_ORIENTATION)
    ipp = _decimal_strings(elements, TAG_IMAGE_POSITION)
    spacing = _decimal_strings(elements, TAG_PIXEL_SPACING)
    if (iop is not None and len(iop) == 1) or (spacing is not None and len(spacing) == 1):
        raise RawError("single-valued IOP / PixelSpacing (pydicom raises on len())")
    if iop is not None and ipp is not None and len(iop) == 6:
        if len(ipp) != 3:
            raise RawError("IPP is not 3-valued")
        row = np.array(iop[:3], float)
        col = np.array(iop[3:], float)
        position = float(np.dot(np.array(ipp, float), np.cross(row, col)))
        rule = "ipp_normal"
    else:
        position = float(_integer_string(elements, TAG_INSTANCE_NUMBER) or 0)
        rule = "instance_number"
    spacing_col = float(spacing[1]) if spacing is not None else None
    spacing_row = float(spacing[0]) if spacing is not None else default_pixel_spacing
    return position, rule, spacing_row, spacing_row if spacing_col is None else spacing_col, ipp, [float(v) for v in iop] if iop is not None and len(iop) == 6 else None


def pixels_from_header(data: bytes, header: RawHeader) -> tuple[np.ndarray, tuple[float, float] | None]:
    """PixelData を格納値の ``int16 / uint16 [Rows, Columns]`` にし、``(RescaleSlope, RescaleIntercept)`` を添える。

    ``PixelRepresentation`` で符号を決めて ``np.frombuffer``、``BitsStored < BitsAllocated`` なら unused
    bits を補正する。受け付けるのは MONOCHROME2、SamplesPerPixel 1、NumberOfFrames 無しか 1、BitsAllocated
    16、ModalityLUTSequence 無し、長さが ``Rows * Columns * 2`` ちょうどのものだけで、他は :class:`RawError`。
    """
    elements = header.elements
    if header.pixel_offset is None or header.pixel_length is None:
        raise RawError("PixelData not found")
    photometric = elements.get(TAG_PHOTOMETRIC)
    if photometric is None or photometric[0] != b"CS" or photometric[1].rstrip(b"\x00 ") != b"MONOCHROME2":
        raise RawError("not MONOCHROME2")
    if TAG_MODALITY_LUT_SEQUENCE in elements:
        raise RawError("ModalityLUTSequence present")
    if _unsigned_short(elements, TAG_SAMPLES_PER_PIXEL) != 1:
        raise RawError("SamplesPerPixel != 1")
    if TAG_NUMBER_OF_FRAMES in elements and _integer_string(elements, TAG_NUMBER_OF_FRAMES) != 1:
        raise RawError("multi-frame")
    rows = _unsigned_short(elements, TAG_ROWS)
    cols = _unsigned_short(elements, TAG_COLUMNS)
    allocated = _unsigned_short(elements, TAG_BITS_ALLOCATED)
    stored = _unsigned_short(elements, TAG_BITS_STORED)
    representation = _unsigned_short(elements, TAG_PIXEL_REPRESENTATION)
    if allocated != 16 or not 1 <= stored <= 16 or representation not in (0, 1) or rows == 0 or cols == 0:
        raise RawError(f"unsupported pixel layout rows={rows} cols={cols} allocated={allocated} stored={stored} representation={representation}")
    nbytes = rows * cols * 2
    if header.pixel_length != nbytes or header.pixel_offset + nbytes > len(data):
        raise RawError(f"PixelData length {header.pixel_length} != {nbytes} or truncated")
    array = np.frombuffer(data, dtype=np.dtype("<i2" if representation == 1 else "<u2"), count=rows * cols, offset=header.pixel_offset).reshape(rows, cols)
    if stored < allocated:  # pydicom の correct_unused_bits=True と同じ 2 回のシフト
        array = array.copy()
        np.left_shift(array, allocated - stored, out=array)
        np.right_shift(array, allocated - stored, out=array)
    rescale = None
    if TAG_RESCALE_SLOPE in elements and TAG_RESCALE_INTERCEPT in elements:
        slope = _decimal_strings(elements, TAG_RESCALE_SLOPE)
        intercept = _decimal_strings(elements, TAG_RESCALE_INTERCEPT)
        if slope is None or intercept is None or len(slope) != 1 or len(intercept) != 1:
            raise RawError("multi-valued RescaleSlope / RescaleIntercept")
        rescale = (slope[0], intercept[0])
    return array, rescale


def percentiles_from_counts(values: np.ndarray, counts: np.ndarray, percentiles: Sequence[float]) -> tuple[float, ...]:
    """昇順の相異なる ``float32`` 値と度数から ``np.percentile(全画素, percentiles)`` と同じ値を返す。

    numpy の ``_quantile``（method="linear"）を式のまま写している。順序統計量は累積度数の
    ``searchsorted`` で引くので、画素を並べ替えない。``values`` は NaN も inf も含まないこと。
    """
    cumulative = np.cumsum(counts)
    n = int(cumulative[-1])
    q = np.true_divide(np.asarray(list(percentiles)), 100)
    virtual = (n - 1) * q
    previous = np.floor(virtual)
    following = previous + 1
    above = virtual >= n - 1
    previous[above] = -1
    following[above] = -1
    below = virtual < 0
    previous[below] = 0
    following[below] = 0
    previous = previous.astype(np.intp)
    following = following.astype(np.intp)
    gamma = virtual - previous

    def pick(ranks: np.ndarray) -> np.ndarray:
        return np.array([values[-1] if k == -1 else values[int(np.searchsorted(cumulative, k + 1, side="left"))] for k in ranks], dtype=np.float32)

    a = pick(previous)
    b = pick(following)
    diff = b - a
    result = a + diff * gamma
    np.subtract(b, diff * (1 - gamma), out=result, where=gamma >= 0.5)
    return tuple(float(v) for v in result)
