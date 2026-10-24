from __future__ import annotations

import json
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from rsna_knee.config import WindowConfig
from rsna_knee.config.config import AugConfig, DepthPriorConfig, UnionBagConfig
from rsna_knee.constants import LABELS, NUM_LABELS
from rsna_knee.data.cache_dataset import CACHE_IDS, CACHE_MASKS, CACHE_OFFSETS, CACHE_STACK_OFFSETS, CACHE_VOLS, build_uid_index, read_cache_manifest, read_series_units
from rsna_knee.data.depth_prior import load_depth_prior, normal_sign, pick_prior_windows, span_rows, study_laterality, window_depths, window_log_weights
from rsna_knee.data.slots import SLOT_NAMES, make_slots, pick_series_for_slot
from rsna_knee.data.union_bag import cap_windows, read_subseries, sample_versions, sample_windows, study_windows, version_pixels
from rsna_knee.models.windows import gather_windows, pick_window_runs, pick_windows, segment_window_indices, select_window_indices, slot_of_rows
from rsna_knee.modules.window_aug import window_augment
from rsna_knee.utils import get_logger

logger = get_logger(__name__)

__all__ = ["WindowStudyDataset", "load_subcase_sides", "load_subcases", "scale_stack", "translate_stack"]


def translate_stack(volume: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Stack 全体を ``(dy, dx)`` 画素だけ整数並進する。全スライス同じ量で、補間はしない。

    画素 ``(y, x)`` は ``(y + dy, x + dx)`` へ移り、空いた側は端の画素を複製して埋める。
    """
    if dy == 0 and dx == 0:
        return volume
    h, w = volume.shape[-2:]
    ay, ax = abs(dy), abs(dx)
    padded = np.pad(volume, ((0, 0), (ay, ay), (ax, ax)), mode="edge")
    return padded[:, ay - dy : ay - dy + h, ax - dx : ax - dx + w]


def scale_stack(volume: np.ndarray, factor: float) -> np.ndarray:
    """Stack 全体を画像中心のまわりに ``factor`` 倍に拡縮する。全スライス同じ倍率で、大きさは変えない。

    ``F.affine_grid`` + ``F.grid_sample``（bilinear、はみ出しは端の画素）を CPU で掛けて uint8 に丸め戻す。
    ``factor == 1.0`` は丸めまで含めて恒等。
    """
    x = torch.from_numpy(np.array(volume, dtype=np.float32))[None]  # [1, S, H, W]。S をチャネルに置いて格子を 1 枚で共有する
    theta = torch.tensor([[[1.0 / factor, 0.0, 0.0], [0.0, 1.0 / factor, 0.0]]])  # 出力座標 -> 入力座標の写像なので逆数
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    out = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return out[0].round_().clamp_(0.0, 255.0).to(torch.uint8).numpy()


def _read_units(cache_dir: Path) -> dict[int, list[dict[str, Any]]]:
    """``series.csv`` を cache の行ごとの unit の列（``unit`` 順）にする。行は :func:`pick_series_for_slot` が読むキーで持つ。"""
    frame = read_series_units(cache_dir).rename(columns={"plane": "Anatomical_Plane", "fluid": "Fluid_Sensitive"})[["study_row", "SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive", "stack_offset", "cap"]]
    units: dict[int, list[dict[str, Any]]] = {}
    for rec in frame.to_dict(orient="records"):
        units.setdefault(int(rec.pop("study_row")), []).append(rec)
    return units


def load_subcases(subcases_csv: str | Path) -> dict[str, tuple[str, frozenset[str]]]:
    """``data.subcases_csv`` を ``{子の ID: (親の StudyInstanceUID, 子の SeriesInstanceUID の集合)}`` にする。

    表は 1 series 1 行。``bilateral_knees/knee_series.csv`` の形（``child_study_id`` と、親の ``StudyInstanceUID``。``knee`` が left / right
    の行だけ使い、``unused`` は読まない）か、``StudyInstanceUID``（子の ID）・``ParentStudyInstanceUID`` を持つ形のどちらでも良い。
    子の ID が親や別の子の ID と重なる、1 つの子が複数の親を持つ、同じ series が 2 つの子に入る表はエラー。
    """
    frame = pd.read_csv(subcases_csv, dtype=str)
    if "child_study_id" in frame.columns:
        missing = [c for c in ("StudyInstanceUID", "SeriesInstanceUID", "child_study_id", "knee") if c not in frame.columns]
        if missing:
            raise ValueError(f"{subcases_csv}: missing columns {missing}")
        frame = frame[frame["knee"].isin(["left", "right"])].rename(columns={"StudyInstanceUID": "ParentStudyInstanceUID", "child_study_id": "StudyInstanceUID"})
        if frame["StudyInstanceUID"].isna().any():
            raise ValueError(f"{subcases_csv}: a left / right series has no child_study_id")
    missing = [c for c in ("StudyInstanceUID", "ParentStudyInstanceUID", "SeriesInstanceUID") if c not in frame.columns]
    if missing:
        raise ValueError(f"{subcases_csv}: missing columns {missing}")
    if frame["SeriesInstanceUID"].duplicated().any():
        raise ValueError(f"{subcases_csv}: a series belongs to more than one sub-case")
    out: dict[str, tuple[str, frozenset[str]]] = {}
    for child, group in frame.groupby("StudyInstanceUID", sort=True):
        parents = group["ParentStudyInstanceUID"].unique()
        if len(parents) != 1:
            raise ValueError(f"{subcases_csv}: sub-case {child} has parents {list(parents)}")
        out[str(child)] = (str(parents[0]), frozenset(group["SeriesInstanceUID"]))
    if set(out) & {parent for parent, _ in out.values()}:
        raise ValueError(f"{subcases_csv}: a sub-case id equals a parent id")
    return out


def load_subcase_sides(subcases_csv: str | Path) -> dict[str, int]:
    """``data.subcases_csv`` の ``knee`` 列から子ごとの左右（right = +1、left = -1。:func:`rsna_knee.data.depth_prior.study_laterality` と同じ符号）。

    ``knee`` 列が無い表なら空（深さ prior は子の series の ``Laterality`` で決める）。1 つの子に right と left が混ざる表はエラー。
    """
    frame = pd.read_csv(subcases_csv, dtype=str)
    if "knee" not in frame.columns:
        return {}
    child_column = "child_study_id" if "child_study_id" in frame.columns else "StudyInstanceUID"
    frame = frame[frame["knee"].isin(["left", "right"])]
    out: dict[str, int] = {}
    for child, knees in frame.groupby(child_column, sort=True)["knee"]:
        if knees.nunique() != 1:
            raise ValueError(f"{subcases_csv}: sub-case {child} has knees {sorted(knees.unique())}")
        out[str(child)] = 1 if knees.iloc[0] == "right" else -1
    return out


class WindowStudyDataset(Dataset):
    """Slot 内の Window Bag と所見ラベルを返す。学習は固定 K 窓、評価は最大 K 窓または全実在窓。

    series レイアウトの cache（manifest の ``series_layout == "all"``）では、Slot に入れる series を ``__getitem__`` ごとに選ぶ。
    学習は Slot の候補から無作為に 1 本、評価は :func:`rsna_knee.data.slots.assign_slots` と同じく先頭を取る。
    ``union_bag`` を渡すと Bag は全 series × 全版の窓（:class:`UnionBagConfig`、:func:`rsna_knee.data.union_bag.study_windows`）から組む。
    """

    def __init__(
        self,
        cache_dir: str | Path,
        study_ids: list[str],
        labels: pd.DataFrame | None = None,
        weights: np.ndarray | None = None,
        severity: np.ndarray | None = None,
        window: WindowConfig | None = None,
        img_size: int = 384,
        is_train: bool = False,
        brightness: float = 0.10,
        k: int | None = None,
        all_windows: bool = False,
        manifest: dict[str, Any] | None = None,
        translate_px: int = 0,
        scale_range: tuple[float, float] | None = None,
        scale_prob: float = 0.5,
        depth_flip: float = 0.0,
        window_aug: AugConfig | None = None,
        depth_prior: DepthPriorConfig | None = None,
        subseries: bool = False,
        union_bag: UnionBagConfig | None = None,
        subcases: dict[str, tuple[str, frozenset[str]]] | None = None,
        subcase_sides: dict[str, int] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.manifest = manifest or read_cache_manifest(self.cache_dir)
        self.slot_table = list(self.manifest["slot_table"])
        self.n_slices = int(self.manifest["n_slices"])
        self.window = window or WindowConfig()
        self.img_size = int(img_size)
        self.is_train = bool(is_train)
        self.brightness = float(brightness)
        self.translate_px = int(translate_px)
        self.scale_range = None if scale_range is None else (float(scale_range[0]), float(scale_range[1]))
        self.scale_prob = float(scale_prob)
        if depth_flip != 0:
            raise ValueError("depth_flip must be zero for slot-confined windows")
        # 窓を切った後の鏡像・Slot ごとの変換（学習時だけ）。
        self.window_aug = window_aug if is_train else None
        if all_windows and is_train:
            raise ValueError("training requires a fixed number of windows")
        # 学習中は k_train、検証中は k_val、採点時は k_eval。呼び出し側が特定の Bag サイズを求めるときは ``k`` で名指しする。
        self.k = None if all_windows else (int(k) if k is not None else (self.window.k_train if is_train else self.window.k_val))

        self.vols_path = self.cache_dir / CACHE_VOLS
        self.uid_index = build_uid_index([(self.vols_path, self.cache_dir / CACHE_IDS)])
        self.subcases = dict(subcases or {})
        self.subcase_sides = dict(subcase_sides or {})  # 子の左右（表の knee）。無い子は子の series の Laterality で決める
        # 子の ID は親の cache 行を指す。表に無い ID はそのまま cache を引く。
        self.cache_uid = {uid: self.subcases[uid][0] if uid in self.subcases else uid for uid in study_ids}
        missing = [uid for uid in study_ids if self.cache_uid[uid] not in self.uid_index]
        if missing:
            raise KeyError(f"{len(missing)} study ids are not in {self.cache_dir}, e.g. {missing[:3]}")
        self.study_ids = list(study_ids)

        self.masks = np.load(self.cache_dir / CACHE_MASKS, mmap_mode="r")
        self.offsets = np.load(self.cache_dir / CACHE_OFFSETS)
        self.series_layout = self.manifest.get("series_layout", "slots")
        if self.series_layout == "all":
            self.stack_offsets = np.load(self.cache_dir / CACHE_STACK_OFFSETS)
            self.units = _read_units(self.cache_dir)  # unit の長さの合計が stack_offsets と合うことはここで検められる
            counts = np.concatenate([[0], np.cumsum(np.asarray(self.masks) != 0)])
            present = counts[self.stack_offsets[1:]] - counts[self.stack_offsets[:-1]]
        elif self.series_layout == "slots":
            if any(uid in self.subcases for uid in self.study_ids):
                raise ValueError(f"{self.cache_dir}: sub-cases need a series layout cache (series_layout='all')")
            present = (np.asarray(self.masks) != 0).sum(axis=1)
        else:
            raise ValueError(f"{self.cache_dir}: unknown series_layout {self.series_layout!r}")
        if not np.array_equal(np.diff(self.offsets), present):
            raise ValueError(f"{self.cache_dir}: offsets do not match the present slices in masks")
        # 子の Slot 候補は親の unit のうち子の series だけ。子の series が親の unit に無ければエラー。
        self.subcase_units: dict[str, list[dict[str, Any]]] = {}
        for uid in self.study_ids:
            if uid not in self.subcases:
                continue
            parent, series = self.subcases[uid]
            kept = [unit for unit in self.units.get(self.uid_index[parent][1], []) if str(unit["SeriesInstanceUID"]) in series]
            absent = series - {str(unit["SeriesInstanceUID"]) for unit in kept}
            if absent:
                raise ValueError(f"{uid}: series {sorted(absent)} are not units of parent {parent}")
            self.subcase_units[uid] = kept
        if self.subcase_units:
            logger.info("sub-cases: %d knee-level studies read from %d parent rows", len(self.subcase_units), len({self.subcases[uid][0] for uid in self.subcase_units}))
        self.label_array = np.full((len(self.study_ids), NUM_LABELS), np.nan, dtype=np.float32) if labels is None else labels.reindex(self.study_ids)[LABELS].to_numpy(dtype=np.float32)
        self.weight_array = None if weights is None else np.asarray(weights, dtype=np.float32)
        if self.weight_array is not None and self.weight_array.shape != (len(self.study_ids), NUM_LABELS):
            raise ValueError(f"weights must be [{len(self.study_ids)}, {NUM_LABELS}], got {self.weight_array.shape}")
        self.severity_array = None if severity is None else np.asarray(severity, dtype=np.float32)
        if self.severity_array is not None and self.severity_array.shape != (len(self.study_ids), NUM_LABELS):
            raise ValueError(f"severity must be [{len(self.study_ids)}, {NUM_LABELS}], got {self.severity_array.shape}")
        # 長い series の sub-series。cache の行 -> {SeriesInstanceUID: (Stack 上の先頭, cap, [(start, stop), ...])}（:meth:`swap_subseries`）。
        self.subseries: dict[int, dict[str, tuple[int, int, list[tuple[int, int]]]]] | None = None
        if subseries:
            self.subseries = self._read_subseries()
        self.union_bag = union_bag
        if union_bag is not None:
            if self.series_layout != "all" or not self.manifest.get("subseries", False) or depth_prior is None or self.window.train_runs != "window" or subseries:
                raise ValueError(
                    f"union_bag needs a series_layout='all' cache + sub-series, depth_prior, train_runs='window', subseries=False: layout={self.series_layout!r} depth_prior={depth_prior is not None} runs={self.window.train_runs!r} sub={subseries}"
                )
            self.union_subs = read_subseries(self)
            self.sub_masks = np.load(self.cache_dir / "sub_masks.npy")
        self.depth_prior = depth_prior
        if depth_prior is not None:
            self._load_depth_prior(depth_prior)
        self._sub_vols: np.memmap | None = None
        self._vols: np.memmap | None = None
        logger.info("windows: %s slots=%s n_slices=%d studies=%d k=%s train=%s norm=%s", self.cache_dir.name, self.slot_table, self.n_slices, len(self.study_ids), self.k, self.is_train, self.window.norm)
        if self.is_train:
            logger.info("aug: brightness=%.3f translate_px=%d scale_range=%s scale_prob=%.2f window_aug=%s", self.brightness, self.translate_px, self.scale_range, self.scale_prob, None if self.window_aug is None else self.window_aug.model_dump())

    def __len__(self) -> int:
        return len(self.study_ids)

    def _read_subseries(self) -> dict[int, dict[str, tuple[int, int, list[tuple[int, int]]]]]:
        """``subseries.csv`` を読み、元の unit が全面実在し sub-series も全面実在するものだけを残す（実在の並びが同じなので画素を行ごとに差し替えられる）。"""
        if not self.is_train or self.series_layout != "all" or not self.manifest.get("subseries", False):
            raise ValueError("subseries needs a training dataset on a series_layout='all' cache built with build_cache.py --subseries")
        table = pd.read_csv(self.cache_dir / "subseries.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
        sub_masks = np.load(self.cache_dir / "sub_masks.npy")
        wanted = set(self.cache_uid.values())  # subseries.csv は cache の行（親）で引く
        out: dict[int, dict[str, tuple[int, int, list[tuple[int, int]]]]] = {}
        for rec in table.itertuples():
            if rec.StudyInstanceUID not in wanted:
                continue
            row = self.uid_index[rec.StudyInstanceUID][1]
            unit = self.units[row][int(rec.unit)]  # unit 順の列
            if unit["SeriesInstanceUID"] != rec.SeriesInstanceUID:
                raise ValueError(f"{self.cache_dir / 'subseries.csv'}: unit {rec.unit} of {rec.StudyInstanceUID} is not {rec.SeriesInstanceUID}")
            first, cap = int(unit["stack_offset"]), int(unit["cap"])
            base = np.asarray(self.masks[self.stack_offsets[row] + first : self.stack_offsets[row] + first + cap]) != 0
            if rec.stop - rec.start != cap or not base.all() or not sub_masks[rec.start : rec.stop].all():
                continue
            out.setdefault(row, {}).setdefault(str(rec.SeriesInstanceUID), (first, cap, []))[2].append((int(rec.start), int(rec.stop)))
        return out

    def pick_subseries(self, row: int, uids: list[str]) -> dict[str, tuple[int, int]]:
        """Slot に選んだ series ごとに、元の 1 本と sub-series の m + 1 通りから均等に 1 つ選ぶ。sub-series を選んだ series の ``sub_vols`` 上の ``(start, stop)`` を返す。"""
        choices = self.subseries.get(row, {}) if self.subseries is not None else {}
        swapped = {}
        for uid in uids:
            if uid in choices:
                spans = choices[uid][2]
                pick = random.randrange(len(spans) + 1)
                if pick > 0:
                    swapped[uid] = spans[pick - 1]
        return swapped

    def swap_subseries(self, row: int, volume: np.ndarray, presence: np.ndarray, swapped: dict[str, tuple[int, int]]) -> np.ndarray:
        """:meth:`pick_subseries` が sub-series を選んだ series の、Stack のその unit の行を差し替えた画素を返す。"""
        packed = np.cumsum(presence) - 1  # Stack の行 -> 詰めた行
        for uid, (start, stop) in swapped.items():
            if self._sub_vols is None:  # worker ごとに開く
                self._sub_vols = np.load(self.cache_dir / "sub_vols.npy", mmap_mode="r")
            first, cap, _ = self.subseries[row][uid]
            stack_rows = np.arange(first, first + cap)
            if not volume.flags.writeable:  # cache の memmap を書き換えない
                volume = volume.copy()
            volume[packed[stack_rows]] = self._sub_vols[start:stop]
        return volume

    def _volume(self) -> np.memmap:
        if self._vols is None:  # worker ごとに開く。プロセス間で共有しない
            self._vols = np.load(self.vols_path, mmap_mode="r")
        return self._vols

    def augment_stack(self, volume: np.ndarray) -> np.ndarray:
        """Stack 全体に共通の並進・拡縮を掛ける。無効な変換は乱数を引かない。"""
        if self.translate_px > 0:
            volume = translate_stack(volume, random.randint(-self.translate_px, self.translate_px), random.randint(-self.translate_px, self.translate_px))
        if self.scale_range is not None and random.random() < self.scale_prob:
            volume = scale_stack(volume, random.uniform(*self.scale_range))
        return volume

    def _load_depth_prior(self, depth_prior: DepthPriorConfig) -> None:
        """学習の窓の採取に使う Slot ごとの prior と、series の法線の符号・study の左右を読む。``union_bag`` の評価では窓の u を引くためだけに読む。"""
        if not (self.is_train or self.union_bag is not None) or self.series_layout != "all" or self.window.train_runs != "window":
            raise ValueError(f"data.depth_prior needs a training dataset on a series_layout 'all' cache with window.train_runs='window', got train={self.is_train} series_layout={self.series_layout!r} train_runs={self.window.train_runs!r}")
        self.prior = load_depth_prior(depth_prior.path)
        meta = pd.read_csv(self.cache_dir / "series_meta.csv", usecols=["StudyInstanceUID", "SeriesInstanceUID", "Laterality", "ImageOrientationPatient", "n_files", "selected_indices"], dtype=str)
        laterality = study_laterality(meta)
        self.prior_laterality = {self.uid_index[self.cache_uid[uid]][1]: laterality[self.cache_uid[uid]] for uid in self.study_ids if uid not in self.subcases}
        # 両膝の親の左右は混在で 0 になる。子は表の knee で決め、無ければ自分の series の Laterality で決める（DICOM の Laterality は
        # 両膝の検査で空が多い）。親の行は 2 つの子で共有されるので、行ではなく子の ID で持つ。
        self.subcase_laterality = {
            uid: self.subcase_sides[uid] if uid in self.subcase_sides else study_laterality(meta[meta["SeriesInstanceUID"].isin(self.subcases[uid][1])]).get(self.subcases[uid][0], 0) for uid in self.study_ids if uid in self.subcases
        }
        # 窓を出しうる series（スライスを採った series）ごとの法線の符号と、(Stack 行 j が span 内の何番目のスライスか, span 内の枚数 m)。
        sampled = meta[meta["selected_indices"].notna() & (meta["selected_indices"] != "[]")]
        self.prior_signs = {uid: normal_sign(iop) for uid, iop in zip(sampled["SeriesInstanceUID"], sampled["ImageOrientationPatient"])}
        span = tuple(self.manifest["span"])
        self.prior_rows = {uid: span_rows(json.loads(selected), int(n_files), span) for uid, n_files, selected in zip(sampled["SeriesInstanceUID"], sampled["n_files"], sampled["selected_indices"])}
        # sub-series の ``sub_vols`` 上の先頭 -> (Stack 行 j が span 内の何番目のスライスか, span 内の枚数 m)。
        self.prior_sub_rows: dict[int, tuple[np.ndarray, int]] = {}
        if self.subseries is not None or self.union_bag is not None:
            n_files = dict(zip(sampled["SeriesInstanceUID"], sampled["n_files"].astype(int)))
            table = pd.read_csv(self.cache_dir / "subseries.csv", dtype={"SeriesInstanceUID": str})
            for uid, start, selected in zip(table["SeriesInstanceUID"], table["start"], table["selected_indices"]):
                self.prior_sub_rows[int(start)] = span_rows(json.loads(selected), n_files[uid], span)
        logger.info("depth prior: %s alpha=%.3f slots=%s", depth_prior.path, depth_prior.alpha, sorted(self.prior))

    def _prior_units(self, row: int, owner: np.ndarray, local: np.ndarray, slot_ids: list[int], series: list[str], swapped: dict[str, tuple[int, int]], lat: int | None = None) -> Iterator[tuple[np.ndarray, str, np.ndarray, int, int, bool]]:
        """窓を持つ unit ごとに ``(その unit の窓 [N] の bool, Slot 名, 窓の中心の span 内の番号, span 内の枚数 m, 向き, 左右対称化するか)`` を出す。``local`` は窓の中心の unit 内の Stack 行。

        窓の深さは cache が採ったスライスの span 内の位置で、series の法線と study の左右で患者座標に揃える。左右が分からない Sagittal は prior を左右対称化する。
        ``swapped`` の series は、差し替える sub-series が採ったスライスの位置を使う。
        """
        lat = self.prior_laterality[row] if lat is None else lat  # 子の study は子の左右を渡す
        for s, (slot, uid) in enumerate(zip(slot_ids, series)):
            members = owner == s
            if members.any():
                sagittal = SLOT_NAMES[slot].startswith("SAG")
                orientation = self.prior_signs[uid] * (lat if sagittal and lat != 0 else 1)
                rows, m = self.prior_sub_rows[swapped[uid][0]] if uid in swapped else self.prior_rows[uid]
                yield members, SLOT_NAMES[slot], rows[local[members]], m, orientation, sagittal and lat == 0

    def prior_log_weights(self, row: int, owner: np.ndarray, local: np.ndarray, slot_ids: list[int], series: list[str], swapped: dict[str, tuple[int, int]], lat: int | None = None) -> np.ndarray:
        """各窓の採取重みの対数 ``[N]``（unit ごとに和 1）。窓の深さの取り方は :meth:`_prior_units`。"""
        log_q = np.empty(len(owner))
        for members, name, rows, m, orientation, symmetric in self._prior_units(row, owner, local, slot_ids, series, swapped, lat):
            log_q[members] = window_log_weights(self.prior[name], rows, m, orientation, symmetric, self.depth_prior.alpha)
        return log_q

    def prior_depths(self, row: int, owner: np.ndarray, local: np.ndarray, slot_ids: list[int], series: list[str], swapped: dict[str, tuple[int, int]], lat: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """各窓の符号つき正規化深さ u ``[N]`` と、prior を左右対称化して読むかの bool ``[N]``。採取重みと同じ深さ（:meth:`_prior_units`）。"""
        u = np.empty(len(owner), dtype=np.float32)
        symmetric = np.zeros(len(owner), dtype=bool)
        for members, _, rows, m, orientation, sym in self._prior_units(row, owner, local, slot_ids, series, swapped, lat):
            u[members] = window_depths(rows, m, orientation)
            symmetric[members] = sym
        return u, symmetric

    def series_window_indices(self, row: int, presence: np.ndarray, units: list[dict[str, Any]] | None = None, lat: int | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[int, int]], tuple[np.ndarray, np.ndarray] | None]:
        """Slot 表の順に unit を 1 本ずつ割り当て、その区間の窓の Stack 上の添字 ``[K, C]``、各窓の Slot 番号 ``[K]``、sub-series に差し替える series（:meth:`pick_subseries`）と、
        ``depth_prior`` があれば各窓の深さ（:meth:`prior_depths` の ``(u [K], 左右対称化 [K])``、無ければ ``None``）を返す。

        ``depth_prior`` が無ければ窓を採った後に、Bag に出てくる series の sub-series を出てくる順に選ぶ。
        あれば Slot の全 series の sub-series を窓より先に選び、学習の K 本は差し替えた後の位置で :func:`rsna_knee.data.depth_prior.pick_prior_windows` で採る。
        """
        units = self.units.get(row, []) if units is None else units  # 子の study は子の series だけ
        used: set[str] = set()
        segments, slot_ids, uids = [], [], []
        for slot in make_slots(self.slot_table):
            unit = pick_series_for_slot(units, slot.plane, slot.fluid, used, random.choice) if self.is_train else pick_series_for_slot(units, slot.plane, slot.fluid, used)
            if unit is None:
                continue
            used.add(str(unit["SeriesInstanceUID"]))
            segments.append((unit["stack_offset"], unit["stack_offset"] + unit["cap"]))
            slot_ids.append(slot.index)
            uids.append(str(unit["SeriesInstanceUID"]))
        index, owner = segment_window_indices(presence, segments, self.window.context)
        if self.depth_prior is None:
            chosen = pick_window_runs(index, owner, self.k) if self.is_train and self.window.train_runs == "slot" else pick_windows(len(index), self.k, self.is_train)
            return index[chosen], np.asarray(slot_ids)[owner[chosen]], self.pick_subseries(row, list(dict.fromkeys(uids[o] for o in owner[chosen]))), None
        swapped = self.pick_subseries(row, uids)
        local = index[:, self.window.context // 2] - np.asarray([start for start, _ in segments])[owner]
        chosen = pick_prior_windows(owner, self.k, self.prior_log_weights(row, owner, local, slot_ids, uids, swapped, lat))
        u, symmetric = self.prior_depths(row, owner, local, slot_ids, uids, swapped, lat)
        return index[chosen], np.asarray(slot_ids)[owner[chosen]], swapped, (u[chosen], symmetric[chosen])

    def union_item(self, idx: int) -> dict[str, Any]:
        """``union_bag`` の Bag。学習は版を :func:`sample_versions` で選んで :func:`sample_windows` で K 本、評価は全版の全窓を :func:`cap_windows` で頭打ちにする。"""
        uid = self.study_ids[idx]
        _, row = self.uid_index[self.cache_uid[uid]]
        series = self.subcases[uid][1] if uid in self.subcases else None
        meta = study_windows(self, row, self.union_subs.get(row, {}), self.sub_masks, series, self.subcase_laterality.get(uid))
        if self.is_train:
            members = np.flatnonzero(np.isin(meta["owner"], sample_versions(meta["groups"])))
            chosen = members[sample_windows(len(members), self.union_bag.k_min, self.union_bag.k_max)]
        else:
            chosen = cap_windows(meta["owner"], meta["version_slot"], self.union_bag.eval_cap)
        if self._sub_vols is None:  # worker ごとに開く
            self._sub_vols = np.load(self.cache_dir / "sub_vols.npy", mmap_mode="r")
        volume = version_pixels(self, row, meta["sources"], self._sub_vols, self.sub_masks)
        if self.is_train:
            volume = self.augment_stack(volume)
        slots, pos, rows = meta["version_slot"][meta["owner"][chosen]], meta["center"][chosen], meta["rows"][chosen]
        windows = gather_windows(volume, rows)
        item = self._finish_item(idx, windows, slots, pos, rows, (meta["u"][chosen], meta["u_sym"][chosen]))
        item["window_series"] = torch.from_numpy(meta["version_series"][meta["owner"][chosen]])
        return item

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.union_bag is not None:
            return self.union_item(idx)
        uid = self.study_ids[idx]
        _, row = self.uid_index[self.cache_uid[uid]]
        units, lat = self.subcase_units.get(uid), self.subcase_laterality.get(uid) if self.depth_prior is not None else None
        # 実在スライスだけを詰めた [P, R, R]。窓の添字は Stack 上で決め、詰めた行番号に引き直す。
        volume = np.asarray(self._volume()[self.offsets[row] : self.offsets[row + 1]])
        if volume.shape[1:] != (self.img_size, self.img_size):
            raise ValueError(f"{uid}: slices are {volume.shape[1:]}, expected {(self.img_size, self.img_size)}")
        presence = (self.masks[self.stack_offsets[row] : self.stack_offsets[row + 1]] if self.series_layout == "all" else self.masks[row]) != 0
        if self.subseries is not None:
            # 先に Bag の series を決め、その series の sub-series を選んでから Stack の変換を掛ける。sub-series が無い学習は従来の乱数の順のまま
            index, slots, swapped, depths = self.series_window_indices(row, presence, units, lat)
            volume = self.swap_subseries(row, volume, presence, swapped)
        if self.is_train:
            volume = self.augment_stack(volume)

        # [K, C, R, R]、値域 [0, 1]。正規化はモデル側が自分で掛ける。
        if self.series_layout == "all":
            if self.subseries is None:
                index, slots, _, depths = self.series_window_indices(row, presence, units, lat)
        else:
            depths = None
            index = select_window_indices(presence, self.slot_table, self.k, self.window.context, self.is_train, self.window.train_runs)
            slots = slot_of_rows(index[:, self.window.context // 2], self.slot_table)
        windows = gather_windows(volume, (np.cumsum(presence) - 1)[index])
        return self._finish_item(idx, windows, slots, index[:, self.window.context // 2].astype(np.int64), index.astype(np.int64), depths)

    def _finish_item(self, idx: int, windows: torch.Tensor, slots: np.ndarray, pos: np.ndarray, rows: np.ndarray, depths: tuple[np.ndarray, np.ndarray] | None) -> dict[str, Any]:
        """窓に鏡像・Slot ごとの変換と明るさを掛けて batch の dict にする。``pos`` は各窓の中心の行で、鏡像でスライス順を反転した窓は符号を反転し、隣の向きを揃える。
        ``rows`` は各チャネルの行 ``[K, C]`` で、スライス順を反転した窓は行も反転し、チャネルとの対応を保つ。
        """
        uid = self.study_ids[idx]
        label = torch.from_numpy(self.label_array[idx].copy())
        if self.window_aug is not None:
            windows, flipped = window_augment(windows, slots, self.slot_table, self.window_aug)
            pos = np.where(flipped, -pos, pos)
            rows = np.where(flipped[:, None], rows[:, ::-1], rows)
        if self.is_train and self.brightness > 0:
            # Bag 全体に 1 つの乗算ゲインを掛ける。
            gain = 1.0 + (random.random() - 0.5) * 2.0 * self.brightness
            windows = (windows * gain).clamp_(0.0, 1.0)
        # 窓は uint8 で運び、モデルが device 上で 255 で割る（float32 だと worker の共有メモリが 4 倍になる）。
        windows = (windows * 255.0).round_().clamp_(0.0, 255.0).to(torch.uint8)

        item: dict[str, Any] = {"windows": windows, "window_slot": torch.from_numpy(slots), "window_pos": torch.from_numpy(pos), "window_rows": torch.from_numpy(np.ascontiguousarray(rows)), "label": label, "study_id": uid}
        if self.weight_array is not None:
            item["weight"] = torch.from_numpy(self.weight_array[idx].copy())
        if self.severity_array is not None:
            item["severity"] = torch.from_numpy(self.severity_array[idx].copy())
        if depths is not None:
            # 窓の深さ u は解剖学的な位置なので、鏡像（窓内のスライス順の並べ替え）では変えない。
            item["window_u"] = torch.from_numpy(depths[0])
            item["window_u_sym"] = torch.from_numpy(depths[1])
        return item
