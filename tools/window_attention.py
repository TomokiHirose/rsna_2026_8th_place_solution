from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import click
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from rsna_knee.constants import LABELS, NUM_LABELS
from rsna_knee.data.depth_prior import normal_sign, study_laterality
from rsna_knee.data.slots import SLOT_NAMES, make_slots, pick_series_for_slot
from rsna_knee.inference import load_model
from rsna_knee.models.window_model import WindowModel
from rsna_knee.models.windows import gather_windows, segment_window_indices
from rsna_knee.modules.window_dataset import WindowStudyDataset


class AttentionStudies(Dataset):
    """All windows of a study with the evaluation-time series choice, plus slot, series, ordinal and depth u per window."""

    def __init__(self, ds: WindowStudyDataset, study_ids: list[str], laterality: dict[str, int], signs: dict[str, int], positions: np.ndarray) -> None:
        self.ds, self.study_ids, self.laterality, self.signs, self.positions = ds, study_ids, laterality, signs, positions

    def __len__(self) -> int:
        return len(self.study_ids)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
        ds, uid = self.ds, self.study_ids[i]
        _, row = ds.uid_index[uid]
        start = ds.stack_offsets[row]
        presence = ds.masks[start : ds.stack_offsets[row + 1]] != 0
        units = ds.units.get(row, [])
        used: set[str] = set()
        segments, slot_ids, series = [], [], []
        for slot in make_slots(ds.slot_table):
            unit = pick_series_for_slot(units, slot.plane, slot.fluid, used)
            if unit is None:
                continue
            used.add(str(unit["SeriesInstanceUID"]))
            segments.append((unit["stack_offset"], unit["stack_offset"] + unit["cap"]))
            slot_ids.append(slot.index)
            series.append(str(unit["SeriesInstanceUID"]))
        index, owner = segment_window_indices(presence, segments, ds.window.context)
        volume = np.asarray(ds._volume()[ds.offsets[row] : ds.offsets[row + 1]])
        windows = gather_windows(volume, (np.cumsum(presence) - 1)[index])
        center = index[:, ds.window.context // 2]
        frame = pd.DataFrame({"study_id": uid, "slot": np.asarray(slot_ids)[owner], "series_uid": np.asarray(series)[owner], "row": center})
        # Ordinal of the real slice within the unit: one window per real slice, so count windows in centre order
        frame["ordinal"] = frame.groupby("series_uid")["row"].rank(method="first").astype(int) - 1
        frame["n_real"] = frame.groupby("series_uid")["row"].transform("size")
        frame["u_stack"] = np.where(frame["n_real"] > 1, 2.0 * frame["ordinal"] / (frame["n_real"] - 1).clip(lower=1) - 1.0, 0.0)
        frame["sign"] = frame["series_uid"].map(self.signs)
        frame["lat"] = self.laterality.get(uid, 0)
        # Align to patient axes (sagittal: patient left +, coronal: posterior +, axial: head +); sagittal with known side: medial +
        sagittal = frame["slot"].map(lambda s: SLOT_NAMES[s].startswith("SAG")).to_numpy()
        frame["u"] = frame["sign"] * frame["u_stack"] * np.where(sagittal & (frame["lat"] != 0), frame["lat"], 1)
        frame["position_mm"] = self.positions[start + center]
        return windows, torch.from_numpy(np.asarray(slot_ids)[owner]), frame.drop(columns="row")


@torch.inference_mode()
def window_attention(model: WindowModel, windows: torch.Tensor, slots: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bag ``[K, C, R, R]`` -> GAP and spatial-branch window attention (each ``[K, 12]``, softmax over all windows) and study logits ``[12]``."""
    feats = model.encode(windows[None], None, slots[None])
    h = model.norm(feats[..., : model.feat_dim])
    gap = model.att(h).softmax(dim=1)
    sp = model.spatial_norm(feats[..., model.feat_dim :].unflatten(-1, (NUM_LABELS, model.feat_dim)))
    spatial = (model.spatial_att(sp) * model.spatial_w).sum(-1).softmax(dim=1)
    logits = model.head_from_features(feats, None, slots[None])
    return gap[0].cpu().numpy(), spatial[0].cpu().numpy(), logits[0].cpu().numpy()


@click.command(help="Dump the out-of-fold window attention of the teacher with slot and depth u per window (input of tools/fit_depth_prior.py)")
@click.option("--teacher", type=click.Path(path_type=Path), default=Path("output/exp242_coatnet_spatial_ema_teacher_start_v4"), help="training output of configs/depth_prior_teacher (fold<i>/model.pt)")
@click.option("--folds", type=click.Path(path_type=Path), default=Path("data/depth_prior/teacher_folds.csv"), help="each fold model scores its held-out studies")
@click.option("--cache", type=click.Path(path_type=Path), default=Path("data/cache/cache_native_384_series_all_exact_uniform_sub"))
@click.option("--out", type=click.Path(path_type=Path), default=Path("output/attention_prior/exp242"))
@click.option("--limit", type=int, default=None, help="max studies per fold (smoke test)")
def main(**options: Any) -> None:
    """Dump the teacher's window attention and study probabilities."""
    args = SimpleNamespace(**options)

    folds = pd.read_csv(args.folds)
    meta = pd.read_csv(args.cache / "series_meta.csv", usecols=["StudyInstanceUID", "SeriesInstanceUID", "Laterality", "ImageOrientationPatient"], dtype=str)
    laterality = study_laterality(meta)
    signs = {uid: normal_sign(iop) for uid, iop in zip(meta["SeriesInstanceUID"], meta["ImageOrientationPatient"])}
    positions = np.load(args.cache / "positions.npy", mmap_mode="r")
    args.out.mkdir(parents=True, exist_ok=True)
    frames, studies = [], []
    started = time.time()
    for fold in sorted(folds["fold"].unique()):
        ids = folds.loc[folds["fold"] == fold, "StudyInstanceUID"].tolist()[: args.limit]
        model = load_model(args.teacher / f"fold{fold}").cuda()
        if model.spatial is None or model.depth is not None or model.seq is not None or model.fsx is not None or model.pair is not None:
            raise ValueError("this dump reads the plain window model with the spatial branch only")
        model.infer_dtype, model.encode_chunk = torch.float16, 32
        ds = WindowStudyDataset(args.cache, ids, window=model.cfg.window, img_size=model.cfg.img_size, all_windows=True)
        loader = DataLoader(AttentionStudies(ds, ids, laterality, signs, positions), batch_size=None, num_workers=6)
        for i, (windows, slots, frame) in enumerate(loader):
            gap, spatial, logits = window_attention(model, windows.cuda(non_blocking=True), slots.cuda())
            frame.insert(1, "fold", fold)
            frame[[f"gap_{label}" for label in LABELS]] = gap
            frame[[f"sp_{label}" for label in LABELS]] = spatial
            frames.append(frame)
            studies.append({"StudyInstanceUID": frame["study_id"].iat[0], "fold": fold, "n_windows": len(frame), **dict(zip(LABELS, 1.0 / (1.0 + np.exp(-logits))))})
            if (i + 1) % 200 == 0:
                print(f"fold {fold}: {i + 1}/{len(ids)} {time.time() - started:.0f}s", flush=True)
    windows_path, study_path = args.out / "window_attention.csv.gz", args.out / "study_probs.csv"
    pd.concat(frames, ignore_index=True).to_csv(windows_path, index=False, float_format="%.6g")
    probs = pd.DataFrame(studies)
    probs.to_csv(study_path, index=False)
    print(f"wrote {windows_path} and {study_path}: {len(probs)} studies, {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
