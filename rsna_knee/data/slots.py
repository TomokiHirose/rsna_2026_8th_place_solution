from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass

__all__ = [
    "DEFAULT_SLOT_TABLE",
    "NUM_SLICES",
    "NUM_SLOTS",
    "SLOTS",
    "SLOT_NAMES",
    "SLOT_OF_SLICE",
    "SLOT_OFFSETS",
    "SLOT_TABLES",
    "Slot",
    "SlotAssignment",
    "assign_slots",
    "get_slots",
    "iter_series_rows",
    "make_slots",
    "num_slices",
    "pick_series_for_slot",
    "series_row_slots",
    "slot_for_series",
    "slot_offsets",
    "slot_of_slice",
]


@dataclass(frozen=True)
class Slot:
    """固定された解剖学的 Slot 1 つと、その Slot が供給するスライス枚数。"""

    index: int
    name: str
    plane: str  # Sagittal | Coronal | Axial
    fluid: int  # 1 なら液体強調を優先、0 なら非液体強調を優先、-1 は優先なし
    n_slices: int


#: Slot 表ごとの採取枚数。並びは本パッケージが作るテンソルのスライス順そのものなので、並べ替えてはいけない。
#:
#:     slot            plane     fluid   "44"   "64"
#:     0  SAG_FLUID    Sagittal      1     12     18
#:     1  SAG_STRUCT   Sagittal      0     10     14
#:     2  COR_FLUID    Coronal       1      8     12
#:     3  COR_SECOND   Coronal       0      6      8
#:     4  AXIAL        Axial        -1      8     12
_SLOT_LAYOUT: list[tuple[int, str, str, int]] = [(0, "SAG_FLUID", "Sagittal", 1), (1, "SAG_STRUCT", "Sagittal", 0), (2, "COR_FLUID", "Coronal", 1), (3, "COR_SECOND", "Coronal", 0), (4, "AXIAL", "Axial", -1)]
_SLICE_COUNTS: dict[str, list[int]] = {"44": [12, 10, 8, 6, 8], "64": [18, 14, 12, 8, 12]}

SLOT_TABLES: dict[str, list[Slot]] = {name: [Slot(index, slot_name, plane, fluid, n) for (index, slot_name, plane, fluid), n in zip(_SLOT_LAYOUT, counts)] for name, counts in _SLICE_COUNTS.items()}

#: 何も指定されなかったときに使う表。公開コーパスと同じ 44 枚版。
DEFAULT_SLOT_TABLE: str = "44"

SLOTS: list[Slot] = SLOT_TABLES[DEFAULT_SLOT_TABLE]

NUM_SLOTS: int = len(SLOTS)
SLOT_NAMES: list[str] = [slot.name for slot in SLOTS]
NUM_SLICES: int = sum(slot.n_slices for slot in SLOTS)  # 44


def make_slots(counts: Sequence[int]) -> list[Slot]:
    """5 つの採取枚数から Slot 表を組み立てる（native レシピの Slot ごとの cap がこれ）。

    撮像面・優先する液体強調フラグ・名前は固定で、変わるのは Slot が供給する枚数だけ。
    """
    counts = [int(c) for c in counts]
    if len(counts) != len(_SLOT_LAYOUT) or any(c < 0 for c in counts):
        raise ValueError(f"expected {len(_SLOT_LAYOUT)} non-negative slice counts, got {counts}")
    return [Slot(index, name, plane, fluid, n) for (index, name, plane, fluid), n in zip(_SLOT_LAYOUT, counts)]


def get_slots(table: str | int | None = None) -> list[Slot]:
    """``table`` という名前の Slot 表を返す（``None`` なら :data:`SLOTS`、すなわち 44 枚版）。"""
    if table is None:
        return SLOTS
    key = str(table)
    if key not in SLOT_TABLES:
        raise KeyError(f"unknown slot table {table!r}; known: {sorted(SLOT_TABLES)}")
    return SLOT_TABLES[key]


def num_slices(slots: Sequence[Slot] | None = None) -> int:
    """Slot 表を平坦化したときのスライス総数。"""
    return sum(slot.n_slices for slot in (SLOTS if slots is None else slots))


def slot_offsets(slots: Sequence[Slot] | None = None) -> list[int]:
    """平坦化した Stack における各 Slot の先頭スライス番号。"""
    table = SLOTS if slots is None else list(slots)
    return [sum(s.n_slices for s in table[:i]) for i in range(len(table))]


def slot_of_slice(slots: Sequence[Slot] | None = None) -> list[int]:
    """``result[i]`` は平坦化スライス ``i`` を持つ Slot の番号。"""
    table = SLOTS if slots is None else slots
    return [slot.index for slot in table for _ in range(slot.n_slices)]


#: 平坦化した Stack における各 Slot の先頭スライス番号（44 の表）。
SLOT_OFFSETS: list[int] = slot_offsets(SLOTS)
#: ``SLOT_OF_SLICE[i]`` は平坦化スライス ``i`` を持つ Slot の番号（44 の表）。
SLOT_OF_SLICE: list[int] = slot_of_slice(SLOTS)


@dataclass(frozen=True)
class SlotAssignment:
    """ある Slot に選ばれた Series（``series_uid is None`` ならその Slot はゼロ埋め）。"""

    slot_index: int
    series_uid: str | None
    n_slices: int = 0  # 選ばれた series の .dcm ファイル数。不明・欠損なら 0

    @property
    def slot_name(self) -> str:
        return SLOT_NAMES[self.slot_index]  # Slot 名はどの表でも同じ

    @property
    def present(self) -> bool:
        return self.series_uid is not None


_PLANE_ALIASES = {"sagittal": "sagittal", "sag": "sagittal", "coronal": "coronal", "cor": "coronal", "axial": "axial", "ax": "axial", "tra": "axial", "transverse": "axial"}


def _as_plane(value: object) -> str | None:
    if value is None or (isinstance(value, float) and value != value):
        return None
    return _PLANE_ALIASES.get(str(value).strip().lower())


def _as_flag(value: object) -> int | None:
    """0/1 の CSV 列を int に寄せる。欠損や解釈できない値は ``None``（= 優先条件に一致しない）になる。"""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return None if value != value else int(value)
    try:
        return int(float(str(value).strip()))
    except ValueError:
        return None


def iter_series_rows(series_rows: object) -> Iterator[dict[str, object]]:
    """DataFrame でも mapping の list でも mapping の iterable でも受け取る。順序はどの Slot に入るかを決めるので保つ。"""
    to_dict = getattr(series_rows, "to_dict", None)
    if to_dict is not None and hasattr(series_rows, "columns"):  # pandas DataFrame
        yield from to_dict(orient="records")
        return
    for row in series_rows:  # type: ignore[union-attr]
        yield dict(row)


def _first(rows: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    return rows[0]


def pick_series_for_slot(rows: list[Mapping[str, object]], plane: str, fluid: int, used: set[str], choose: Callable[[Sequence[Mapping[str, object]]], Mapping[str, object]] = _first) -> Mapping[str, object] | None:
    """``plane`` の Slot を埋める series の行。無ければ ``None``。

    候補は、その撮像面を持ちまだどの Slot にも使われていない行で、CSV の並び順のまま。``fluid`` が 0 か 1
    で、その ``Fluid_Sensitive`` を持つ候補があればそこから、無ければ候補全体から ``choose``（既定は先頭）で 1 件選ぶ。
    """
    wanted = _PLANE_ALIASES[plane.lower()]
    candidates = [r for r in rows if _as_plane(r.get("Anatomical_Plane")) == wanted and str(r.get("SeriesInstanceUID")) not in used]
    if fluid in (0, 1):
        preferred = [r for r in candidates if _as_flag(r.get("Fluid_Sensitive")) == fluid]
        if preferred:
            return choose(preferred)
    return choose(candidates) if candidates else None


def slot_for_series(plane: object, fluid: object, slots: Sequence[Slot]) -> Slot:
    """series 自身の撮像面と ``Fluid_Sensitive`` が決める Slot。

    撮像面の Slot が 1 つならそれ、2 つ以上なら ``Slot.fluid`` がフラグと一致するもの。学習時の「希望のフラグの series が無ければ他の
    series で埋める」代替は行わず、series のフラグに従う。撮像面かフラグで 1 つに決まらなければ例外にする（Slot 0 などへ黙って寄せない）。
    """
    wanted = _as_plane(plane)
    candidates = [slot for slot in slots if _PLANE_ALIASES[slot.plane.lower()] == wanted]
    if not candidates:
        raise ValueError(f"no slot for Anatomical_Plane={plane!r}")
    if len(candidates) == 1:
        return candidates[0]
    flag = _as_flag(fluid)
    matched = [slot for slot in candidates if slot.fluid == flag]
    if len(matched) != 1:
        raise ValueError(f"Fluid_Sensitive={fluid!r} does not pick one of {[slot.name for slot in candidates]} for Anatomical_Plane={plane!r}")
    return matched[0]


def series_row_slots(units: Sequence[Mapping[str, object]], n_rows: int, slots: Sequence[Slot]) -> list[int]:
    """``series_layout="all"`` の Stack の行ごとの Slot 番号（どの series にも属さない行は -1）。

    ``units`` は Stack 上の行範囲 ``stack_start`` / ``stack_stop`` と ``Anatomical_Plane`` / ``Fluid_Sensitive`` を持つ series ごとの mapping。
    Slot は :func:`slot_for_series` で決める。
    """
    rows = [-1] * n_rows
    for unit in units:
        start, stop = int(unit["stack_start"]), int(unit["stack_stop"])
        rows[start:stop] = [slot_for_series(unit["Anatomical_Plane"], unit["Fluid_Sensitive"], slots).index] * (stop - start)
    return rows


def assign_slots(series_rows: object, n_slices: Mapping[str, int] | None = None, slots: Sequence[Slot] | None = None) -> list[SlotAssignment]:
    """1 Study の series 行を Slot 表に割り当て、Slot ごとに :class:`SlotAssignment` を 1 つ表の順で返す。

    表の先頭の Slot から順に選ぶので、1 つの series が 2 つ以上の Slot に使われることはない。選択は CSV
    だけで決まる。``n_slices``（``SeriesInstanceUID -> .dcm ファイル数``）は返り値への注記にしか使わない。
    """
    table = SLOTS if slots is None else slots
    rows = [r for r in iter_series_rows(series_rows) if r.get("SeriesInstanceUID") is not None]
    counts = n_slices or {}
    used: set[str] = set()
    assignments: list[SlotAssignment] = []
    for slot in table:
        row = pick_series_for_slot(rows, slot.plane, slot.fluid, used)
        if row is None:
            assignments.append(SlotAssignment(slot.index, None, 0))
            continue
        uid = str(row["SeriesInstanceUID"])
        used.add(uid)
        assignments.append(SlotAssignment(slot.index, uid, int(counts.get(uid, 0) or 0)))
    return assignments
