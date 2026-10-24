from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rsna_knee.config import Config
from rsna_knee.constants import NUM_LABELS
from rsna_knee.data.depth_prior import normal_sign, span_rows, study_laterality
from rsna_knee.data.slots import SLOT_NAMES, make_slots, pick_series_for_slot, series_row_slots, slot_for_series, slot_offsets
from rsna_knee.data.union_bag import cap_windows, version_depths, version_windows
from rsna_knee.models import WindowModel
from rsna_knee.models.factory import build_model

MODEL_FILENAME = "model.pt"
CONFIG_FILENAME = "config.json"


def load_model(artifact_dir: str | Path, map_location: str | torch.device = "cpu") -> WindowModel:
    """推論 artifact（``config.json`` + ``model.pt``）を eval モードのモデルとして読む。

    ``ModelConfig`` は未知のキーを拒むので、このビルドが解釈できない config はここで例外になる。
    """
    artifact_dir = Path(artifact_dir)
    cfg = Config(**json.loads((artifact_dir / CONFIG_FILENAME).read_text()))
    if cfg.model.kind == "vlm":
        cfg.model.vlm.source = artifact_dir / cfg.model.vlm.source
    model = build_model(cfg, pretrained=False)
    state = torch.load(artifact_dir / MODEL_FILENAME, map_location=map_location, weights_only=True)
    model.load_state_dict(state)
    return model.eval()


def canonical_device(device: str | torch.device) -> torch.device:
    """``cuda`` を ``cuda:<current>`` に解決する。パラメータの ``.device`` は常に添字付きなので、比較はこの形で行う。"""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def series_tta_stacks(series_layout: str, slot_table: list[int], pixels: np.ndarray, mask: np.ndarray, headers: list[dict], subseries: list, passes: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """series の TTA の回ごとの ``(pixels, mask)``。入力は ``series_layout="all"`` + ``subseries`` でデコードした Stack。

    各 series の選択肢は元の 1 本と sub-series（:attr:`StudyTensor.subseries`）。k 回目は:

    - ``series_layout="slots"``: Slot 表の順に、学習の seriesaug と同じ :func:`pick_series_for_slot` で候補の k 番目（候補数で割った余り）の series を選び、
      その series の選択肢 ``(k // 候補数) % 選択肢数`` の画素を Slot の行に写す。0 回目は各 Slot の先頭の候補・元の 1 本で、5 Slot のデコードと同じ Stack
    - ``series_layout="all"``: series はすべて並べたまま、sub-series を持つ series だけ選択肢 ``k % 選択肢数`` の画素に差し替える。0 回目は元の Stack

    回数は ``passes`` と、選択肢を一巡するのに要る回数の小さい方。
    """
    units = [h for h in headers if h.get("assigned", 1)]
    options: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for h in units:
        start, stop = int(h["stack_start"]), int(h["stack_stop"])
        options[str(h["SeriesInstanceUID"])] = [(pixels[start:stop], mask[start:stop])]
    for sub in sorted(subseries, key=lambda s: (s.series_uid, s.index)):
        options[sub.series_uid].append((sub.pixels, sub.mask))
    start_of = {str(h["SeriesInstanceUID"]): int(h["stack_start"]) for h in units}
    if series_layout == "all":
        needed = max((len(o) for o in options.values()), default=1)
        stacks = []
        for k in range(min(passes, needed)):
            out_pixels, out_mask = pixels.copy(), mask.copy()
            for uid, opts in options.items():
                if len(opts) > 1:
                    sub_pixels, sub_mask = opts[k % len(opts)]
                    out_pixels[start_of[uid] : start_of[uid] + len(sub_mask)] = sub_pixels
                    out_mask[start_of[uid] : start_of[uid] + len(sub_mask)] = sub_mask
            stacks.append((out_pixels, out_mask))
        return stacks
    slots = make_slots(slot_table)
    offsets = slot_offsets(slots)
    needed = 1
    stacks = []
    k = 0
    while k < min(passes, needed):
        out_pixels = np.zeros((sum(slot_table), *pixels.shape[1:]), dtype=pixels.dtype)
        out_mask = np.zeros(sum(slot_table), dtype=mask.dtype)
        used: set[str] = set()
        for slot, offset in zip(slots, offsets):
            seen: list[list[dict]] = []

            def choose(candidates, seen=seen):
                if k > 0:  # 2 回目以降は実在の面を持つ候補だけを回す（中身の無い series で Slot を空にしない）。0 回目は 5 Slot のデコードと同じ選び方
                    candidates = [r for r in candidates if options[str(r["SeriesInstanceUID"])][0][1].any()] or candidates
                seen.append(list(candidates))
                return candidates[k % len(candidates)]

            row = pick_series_for_slot(units, slot.plane, slot.fluid, used, choose)
            if row is None:
                continue
            uid = str(row["SeriesInstanceUID"])
            used.add(uid)
            opts = options[uid]
            n_candidates = len(seen[0])
            if k == 0:  # 一巡に要る回数は、各 Slot の「候補数 x 候補の選択肢数の最大」の最大
                needed = max(needed, n_candidates * max(len(options[str(r["SeriesInstanceUID"])]) for r in seen[0]))
            slot_pixels, slot_mask = opts[(k // n_candidates) % len(opts)]
            if len(slot_mask) != slot.n_slices:
                raise ValueError(f"series {uid} has {len(slot_mask)} rows but slot {slot.name} holds {slot.n_slices}; series TTA needs the plane caps of the slot table")
            out_pixels[offset : offset + slot.n_slices] = slot_pixels
            out_mask[offset : offset + slot.n_slices] = slot_mask
        stacks.append((out_pixels, out_mask))
        k += 1
    return stacks


@torch.inference_mode()
def predict_study(model: WindowModel, pixels: np.ndarray, mask: np.ndarray, device: str | torch.device = "cpu", tta_hflip: bool = False, series_headers: list[dict] | None = None, subseries: list | None = None) -> torch.Tensor:
    """Stack と Presence から Slot 内の窓を組み、CPU 上の所見確率を返す。モデルは指定 device に配置しておく。

    ``model.tta_series`` が 2 以上なら、``series_layout="all"`` + ``subseries`` でデコードした Stack から :func:`series_tta_stacks` で回ごとの Stack を組み、
    それぞれの確率を平均する（:func:`_predict_stacks_shared`。回をまたいで同じ窓は 1 度だけ encoder に通す）。
    ``model.union_cap`` があれば、同じデコードの全 series × 全版の窓を 1 つの Bag にして採点する（:func:`_predict_union`）。
    """
    if getattr(model, "union_cap", None) is not None:
        if getattr(model, "tta_series", 1) > 1 or model.tta_context_flip:
            raise ValueError("union_cap cannot be combined with tta_series or tta_context_flip")
        if series_headers is None or subseries is None:
            raise ValueError("union bags need series_headers and subseries (decode with series_layout='all' and subseries=True)")
        return _predict_union(model, union_versions(model.cfg.preprocess.slot_table, pixels, mask, series_headers, subseries), series_headers, device, tta_hflip)
    if getattr(model, "tta_series", 1) > 1:
        if series_headers is None or subseries is None:
            raise ValueError("series TTA needs series_headers and subseries (decode with series_layout='all' and subseries=True)")
        stacks = series_tta_stacks(model.series_layout, model.cfg.preprocess.slot_table, pixels, mask, series_headers, subseries, model.tta_series)
        return _predict_stacks_shared(model, stacks, device, tta_hflip, series_headers)
    return _predict_stack(model, pixels, mask, device, tta_hflip, series_headers)


def union_versions(slot_table: list[int], pixels: np.ndarray, mask: np.ndarray, headers: list[dict], subseries: list) -> list[tuple[int, np.ndarray, np.ndarray, int, dict, list[int]]]:
    """``series_layout="all"`` + ``subseries`` でデコードした study の版 ``(Slot 番号, 実在の印 [n], 実在スライスの画素 [p, R, R], series 番号, series の header, 採ったスライスの元の番号)``。

    並びは学習の :func:`rsna_knee.data.union_bag.study_windows` と同じで、series は Stack 上の順、series の中では元の版、sub-series の番号順。実在の無い版は除き、series 番号は版を持つ series だけに振る。
    """
    slots = make_slots(slot_table)
    subs: dict[str, list] = {}
    for sub in sorted(subseries, key=lambda s: (s.series_uid, s.index)):
        subs.setdefault(str(sub.series_uid), []).append(sub)
    versions = []
    series = 0
    for h in headers:
        if not h.get("assigned", 1):
            continue
        slot = slot_for_series(h["Anatomical_Plane"], h["Fluid_Sensitive"], slots).index
        start, stop = int(h["stack_start"]), int(h["stack_stop"])
        found = False
        for version_mask, version_pixels, picks in [(mask[start:stop], pixels[start:stop], h.get("selected_indices"))] + [(sub.mask, sub.pixels, sub.picks) for sub in subs.get(str(h["SeriesInstanceUID"]), [])]:
            present = np.asarray(version_mask) != 0
            if present.any():
                versions.append((slot, present, np.asarray(version_pixels)[present], series, h, json.loads(picks) if isinstance(picks, str) else picks))
                found = True
        series += found
    return versions


def union_depths(versions: list[tuple[int, np.ndarray, np.ndarray, int, dict, list[int]]], headers: list[dict], owner: np.ndarray, center: np.ndarray, span: tuple[float, float]) -> np.ndarray:
    """窓ごとの深さ u。学習の ``WindowStudyDataset.prior_depths`` と同じく、採ったスライスの span 内の位置を series の法線と study の左右（Sagittal だけ）で患者座標に揃える。"""
    # cache の series_meta.csv（空欄は欠損）と同じく、空の Laterality は欠損として study の左右を決める。
    sides = [h.get("Laterality") for h in headers]
    lat = study_laterality(pd.DataFrame({"StudyInstanceUID": "study", "Laterality": [side if side not in ("", None) else np.nan for side in sides]}))["study"]
    units = []
    for slot, _, _, _, h, picks in versions:
        orientation = normal_sign(h["ImageOrientationPatient"]) * (lat if SLOT_NAMES[slot].startswith("SAG") and lat != 0 else 1)
        units.append((*span_rows(picks, int(h["n_files"]), span), orientation))
    return version_depths(owner, center, units)


def _predict_union(model: WindowModel, versions: list[tuple[int, np.ndarray, np.ndarray, int, dict, list[int]]], headers: list[dict], device: str | torch.device, tta_hflip: bool) -> torch.Tensor:
    """版の全窓（``model.union_cap`` で頭打ち）を 1 つの Bag にして採点する。窓の切り方は :func:`rsna_knee.data.union_bag.version_windows`。

    ``model.series_neck`` には窓ごとの series 番号と深さ u（:func:`union_depths`）も渡す。
    """
    from rsna_knee.models.windows import gather_windows  # noqa: PLC0415 - import 階層を遅延させたまま保つ

    device = canonical_device(device)
    owner, center, rows = version_windows([version[1] for version in versions], model.cfg.window.context)
    version_slot = np.asarray([version[0] for version in versions], dtype=np.int64)
    chosen = cap_windows(owner, version_slot, model.union_cap)
    packed = np.concatenate([version[2] for version in versions])
    windows = gather_windows(packed, rows[chosen])[None].to(device)
    window_slot = torch.from_numpy(version_slot[owner[chosen]])[None].to(device)
    window_pos = torch.from_numpy(center[chosen])[None].to(device)
    window_rows = torch.from_numpy(rows[chosen])[None].to(device)
    neck: dict[str, torch.Tensor] = {}
    if model.series_neck is not None:
        version_series = np.asarray([version[3] for version in versions], dtype=np.int64)
        neck = {"window_series": torch.from_numpy(version_series[owner[chosen]])[None].to(device), "window_u": torch.from_numpy(union_depths(versions, headers, owner, center, tuple(model.cfg.preprocess.span))[chosen])[None].to(device)}
    views = [windows, windows.flip(-1)] if tta_hflip else [windows]
    return torch.stack([torch.sigmoid(model(view, window_slot=window_slot, window_pos=window_pos, window_rows=window_rows, **neck).float()) for view in views]).mean(dim=0)[0].cpu()


def _predict_stack(model: WindowModel, pixels: np.ndarray, mask: np.ndarray, device: str | torch.device, tta_hflip: bool, series_headers: list[dict] | None) -> torch.Tensor:
    """1 つの Stack を採点する :func:`predict_study` の本体。"""
    cfg = model.cfg
    device = canonical_device(device)
    param_device = next(model.parameters()).device
    if param_device != device:
        raise ValueError(f"model is on {param_device} but predict_study was asked for {device}; place the model once at load time")
    if pixels.shape[-1] != cfg.img_size or pixels.shape[-2] != cfg.img_size:
        raise ValueError(f"stack is {pixels.shape[-2]}x{pixels.shape[-1]} px but img_size is {cfg.img_size}; decode the study with the model's own recipe")
    if cfg.kind == "vlm":
        from rsna_knee.data.video import video_inputs

        if series_headers is None:
            raise ValueError("VLM inference requires series metadata")

        def score(array):
            inputs = video_inputs(model.processor, cfg, array, mask, series_headers)
            return torch.sigmoid(model.forward_batch({key: value.to(device) for key, value in inputs.items()}).float())[0].cpu()

        probs = score(pixels)
        return (probs + score(pixels[..., ::-1])) / 2 if tta_hflip else probs
    image, slots, pos, rows = (t.unsqueeze(0) for t in _stack_windows(model, pixels, mask, device, series_headers))
    views = [(view, view_rows) for view, view_rows in _views(model, image, rows, tta_hflip, window_axis=2)]
    probs = torch.stack([torch.sigmoid(model(view, window_slot=slots, window_pos=pos, window_rows=view_rows).float()) for view, view_rows in views]).mean(dim=0)
    return probs[0].cpu()


def _stack_windows(model: WindowModel, pixels: np.ndarray, mask: np.ndarray, device: torch.device, series_headers: list[dict] | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack の窓 ``[K, C, R, R]``、各窓の Slot 番号 ``[K]``、中心スライスの Stack 行番号 ``[K]``、各チャネルの Stack 行番号 ``[K, C]``（:func:`build_windows_on`）。"""
    from rsna_knee.models.windows import build_windows_on  # noqa: PLC0415 - import 階層を遅延させたまま保つ

    cfg = model.cfg
    # "all" の Stack は series の間に実在しない 1 枚を挟んでいるので、全体を 1 区間として実在の連続区間で窓を切る。
    slot_table = [len(mask)] if model.series_layout == "all" else cfg.preprocess.slot_table
    row_slots = None
    if model.series_layout == "all" and getattr(model, "spatial", None) is not None:
        # 空間 attention 枝は Slot ごとの query を読むので、1 区間にした "all" でも窓ごとに series の Slot を渡す。
        if series_headers is None:
            raise ValueError("model.spatial with series_layout='all' needs series_headers (stack_start / stack_stop / Anatomical_Plane / Fluid_Sensitive per series)")
        row_slots = series_row_slots([h for h in series_headers if h.get("assigned", 1)], len(mask), make_slots(cfg.preprocess.slot_table))
    return build_windows_on(pixels, mask, cfg.window.k_eval, cfg.img_size, cfg.window.context, device=device, slot_table=slot_table, row_slots=row_slots)


def _views(model: WindowModel, image: torch.Tensor, rows: torch.Tensor, tta_hflip: bool, window_axis: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """推論の向きの一覧 ``(窓, 各チャネルの Stack 行番号)``。``window_axis`` は窓のチャネル（スライス）の軸。

    窓内のスライス順の反転（context flip）は窓の Slot 内の位置を変えないので、Slot と中心の行はそのまま使う。各チャネルの行番号は順序ごと反転する。
    """
    views = [(image, rows), (image.flip(window_axis), rows.flip(-1))] if model.tta_context_flip else [(image, rows)]
    if tta_hflip:
        views += [(view.flip(-1), view_rows) for view, view_rows in views]
    return views


def _row_digests(pixels: np.ndarray, mask: np.ndarray, rows: np.ndarray) -> dict[int, bytes]:
    """窓が使う Stack 行ごとの、画素と実在の印のハッシュ。"""
    return {int(r): hashlib.blake2b(pixels[r].tobytes() + bytes([int(mask[r] != 0)]), digest_size=16).digest() for r in np.unique(rows)}


def _predict_stacks_shared(model: WindowModel, stacks: list[tuple[np.ndarray, np.ndarray]], device: str | torch.device, tta_hflip: bool, series_headers: list[dict] | None) -> torch.Tensor:
    """series の TTA の回ごとの Stack を採点し、回ごとの確率を平均した CPU 上の所見確率。:func:`_predict_stack` を回ごとに呼んで平均するのと同じ値。

    window モデルの encoder は窓ごとに独立なので、Slot 番号と構成スライスの画素が同じ窓は、回をまたいで 1 度だけ encoder に通す。
    回ごとには、その回の窓の並びに特徴を並べ直して集約と head（:meth:`WindowModel.head_from_features`）だけを走らせる。
    """
    if model.cfg.kind != "window":
        raise ValueError("series TTA is for the window model only")
    device = canonical_device(device)
    keys: dict[tuple, int] = {}
    unique_windows: list[torch.Tensor] = []
    unique_slots: list[int] = []
    passes = []
    for pixels, mask in stacks:
        image, slots, pos, rows = _stack_windows(model, pixels, mask, device, series_headers)
        rows_cpu, slots_cpu = rows.cpu().numpy(), slots.cpu().numpy()
        digest = _row_digests(pixels, mask, rows_cpu)
        index = []
        for w, (slot, window_rows) in enumerate(zip(slots_cpu.tolist(), rows_cpu)):
            key = (slot, *(digest[int(r)] for r in window_rows))
            if key not in keys:
                keys[key] = len(unique_windows)
                unique_windows.append(image[w])
                unique_slots.append(slot)
            index.append(keys[key])
        passes.append((torch.as_tensor(index, device=device), slots, pos, rows))
    windows = torch.stack(unique_windows)
    window_slots = torch.as_tensor(unique_slots, device=device)
    probs = torch.zeros(len(passes), NUM_LABELS)
    # 向きは :func:`_views` と同じ並び。各向きで窓のスライス順を反転したかを持ち、回ごとの行番号も同じく反転する。
    views = [(windows, False), (windows.flip(1), True)] if model.tta_context_flip else [(windows, False)]
    if tta_hflip:
        views += [(view.flip(-1), flipped) for view, flipped in views]
    for view, flipped in views:
        feats = model.encode(view[None], window_slot=window_slots[None])[0]
        for p, (index, slots, pos, rows) in enumerate(passes):
            view_rows = rows.flip(-1) if flipped else rows
            logits = model.head_from_features(feats[index][None], None, slots[None], pos[None], view_rows[None])
            probs[p] += torch.sigmoid(logits.float())[0].cpu()
    return (probs / len(views)).mean(dim=0)
