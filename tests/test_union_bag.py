import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from rsna_knee.config.config import UnionBagConfig
from rsna_knee.constants import ID_COLUMN, LABELS
from rsna_knee.inference import load_model

from .conftest import REPO_ROOT
from .series_cache import EDGE, STUDIES, write_depth_prior, write_series_cache


def _train(tmp_path: Path, name: str, cfg: dict) -> Path:
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({**cfg, "experiment_name": name}))
    proc = subprocess.run([sys.executable, "-m", "rsna_knee.training", "--config", str(path), "--fold", "0"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return tmp_path / "output" / name / "fold0"


def _add_subseries(cache: Path, studies: dict[str, list[int]]) -> pd.DataFrame:
    """Add three sub-series to series 1 (Sagittal, 4 slices): fully present, missing its tail, and empty (dropped). Pixels are ``100 + 10 * sub + j``."""
    specs = [([3, 9, 15, 18], [1, 1, 1, 1]), ([2, 8, 14], [1, 1, 1, 0]), ([], [0, 0, 0, 0])]
    rows, vols, masks = [], [], []
    for uid, members in studies.items():
        if 1 in members:
            for sub, (selected, mask) in enumerate(specs, 1):
                start = len(masks)
                masks += mask
                vols += [np.full((EDGE, EDGE), 100 + 10 * sub + j, dtype=np.uint8) for j in range(len(mask))]
                rows.append({"StudyInstanceUID": uid, "SeriesInstanceUID": f"{uid}.1", "unit": members.index(1), "sub": sub, "start": start, "stop": start + len(mask), "selected_indices": json.dumps(selected)})
    np.save(cache / "sub_vols.npy", np.stack(vols))
    np.save(cache / "sub_masks.npy", np.asarray(masks, dtype=np.uint8))
    manifest = json.loads((cache / "manifest.json").read_text())
    (cache / "manifest.json").write_text(json.dumps({**manifest, "subseries": True}))
    table = pd.DataFrame(rows)
    table.to_csv(cache / "subseries.csv", index=False)
    return table


def test_online_union_bag_keeps_the_encoder_and_matches_the_submission_bag(tmp_path: Path) -> None:
    """Online union-bag training on a frozen encoder keeps the backbone and SAP (BatchNorm statistics too) and moves only the context branch and head.

    Union inference of the artifact from DICOM-style inputs gives the same probabilities as the cache evaluation Bag.
    """
    from rsna_knee.inference import predict_study
    from rsna_knee.modules.window_dataset import WindowStudyDataset

    studies = {f"e{i}": STUDIES["s0"] if i % 2 else STUDIES["s1"] for i in range(8)}
    cache = write_series_cache(tmp_path / "cache", studies)
    prior = write_depth_prior(cache, tmp_path / "prior.json", studies)
    table = _add_subseries(cache, studies)
    values = np.linspace(0.0, 1.0, len(studies))
    pd.DataFrame({ID_COLUMN: list(studies), **{name: values for name in LABELS}}).to_csv(tmp_path / "labels.csv", index=False)
    pd.DataFrame({ID_COLUMN: list(studies), "fold": [i // 2 % 2 for i in range(len(studies))]}).to_csv(tmp_path / "folds.csv", index=False)
    data = {
        "cache_dir": str(cache),
        "labels_csv": str(tmp_path / "labels.csv"),
        "folds_csv": str(tmp_path / "folds.csv"),
        "train_csv": str(tmp_path / "absent.csv"),
        "split": "folds",
        "batch_size": 2,
        "num_workers": 0,
        "pin_memory": False,
        "depth_prior": prior.model_dump(mode="json"),
    }
    model_cfg = {"backbone": "resnet18", "pretrained": False, "img_size": EDGE, "window": {"k_train": 3, "k_val": 3, "k_eval": None, "pool": "lse", "norm": "half"}, "spatial": {"enabled": True, "dim": 8, "att_hidden": 8}}
    train_cfg = {"epochs": 1, "precision": "32-true", "learning_rate": 0.01, "encoder_lr": 0.001, "rank_weight": 0.0, "rank_queue": 0, "window_loss_weight": 0.3}
    cfg = {"wandb_mode": "disabled", "output_dir": str(tmp_path / "output"), "data": data, "model": model_cfg, "train": train_cfg}
    stage1 = _train(tmp_path, "stage1", cfg)

    cfg["data"] = {**data, "union_bag": {"k_min": 4, "k_max": 16, "eval_cap": 14}}
    cfg["model"] = {**model_cfg, "freeze_encoder": True, "context": {"enabled": True, "dim": 8, "hidden": 8, "heads": 2}}
    cfg["train"] = {**train_cfg, "init_weights": str(stage1 / "model.pt"), "init_weights_skip": ["clsW", "clsb"]}
    stage2 = _train(tmp_path, "online", cfg)
    assert all((stage2 / name).exists() for name in ("model.pt", "config.json", "oof.csv", "oof_all.csv", "oof_union.csv"))
    before, after = torch.load(stage1 / "model.pt", weights_only=True), torch.load(stage2 / "model.pt", weights_only=True)
    frozen = [key for key in before if key.startswith(("backbone.", "spatial."))]
    assert any(key.endswith(("running_mean", "running_var", "num_batches_tracked")) for key in frozen)
    assert all(torch.equal(before[key], after[key]) for key in frozen)
    assert not torch.equal(before["clsW"], after["clsW"]) and any(key.startswith("context.") for key in after)

    # Submission path: the union bag built from the series_layout="all" Stack, series_headers and sub-series matches forward_batch on the cache evaluation Bag
    model = load_model(stage2).eval()
    model.union_cap = 14
    dataset = WindowStudyDataset(cache, ["e1"], window=model.cfg.window, img_size=EDGE, depth_prior=prior, union_bag=UnionBagConfig(eval_cap=14))
    item = dataset[0]
    assert item["windows"].shape[0] == 14
    _, row = dataset.uid_index["e1"]
    mask = np.asarray(dataset.masks[dataset.stack_offsets[row] : dataset.stack_offsets[row + 1]])
    stack = np.zeros((len(mask), EDGE, EDGE), dtype=np.uint8)
    stack[mask != 0] = np.load(cache / "vols.npy")[dataset.offsets[row] : dataset.offsets[row + 1]]
    units = dataset.units[row]
    headers = [{"SeriesInstanceUID": u["SeriesInstanceUID"], "stack_start": u["stack_offset"], "stack_stop": u["stack_offset"] + u["cap"], "Anatomical_Plane": u["Anatomical_Plane"], "Fluid_Sensitive": u["Fluid_Sensitive"]} for u in units]
    sub_vols, sub_masks = np.load(cache / "sub_vols.npy"), np.load(cache / "sub_masks.npy")
    subs = [
        SimpleNamespace(series_uid=rec.SeriesInstanceUID, index=int(rec.sub), mask=sub_masks[rec.start : rec.stop], pixels=sub_vols[rec.start : rec.stop], picks=json.loads(rec.selected_indices)) for rec in table[table[ID_COLUMN] == "e1"].itertuples()
    ]
    with torch.no_grad():
        expected = torch.sigmoid(model.forward_batch({key: item[key][None] for key in ("windows", "window_slot", "window_pos", "window_rows")}))[0]
    assert torch.allclose(predict_study(model, stack, mask, series_headers=headers, subseries=subs), expected, atol=1e-4)


def test_series_neck_trains_on_a_frozen_attention_encoder_and_matches_the_submission_bag(tmp_path: Path) -> None:
    """series_neck trained on a frozen attention encoder with the plane auxiliary head; union inference built from DICOM headers matches the cache evaluation Bag."""
    from rsna_knee.inference import predict_study
    from rsna_knee.modules.window_dataset import WindowStudyDataset

    studies = {f"e{i}": STUDIES["s0"] if i % 2 else STUDIES["s1"] for i in range(8)}
    cache = write_series_cache(tmp_path / "cache", studies)
    prior = write_depth_prior(cache, tmp_path / "prior.json", studies)
    meta = pd.read_csv(cache / "series_meta.csv", dtype=str, keep_default_na=False)
    meta.loc[meta["StudyInstanceUID"] == "e1", "Laterality"] = "R"  # flips the Sagittal direction by the study side
    meta.to_csv(cache / "series_meta.csv", index=False)
    table = _add_subseries(cache, studies)
    values = np.linspace(0.0, 1.0, len(studies))
    pd.DataFrame({ID_COLUMN: list(studies), **{name: values for name in LABELS}}).to_csv(tmp_path / "labels.csv", index=False)
    pd.DataFrame({ID_COLUMN: list(studies), "fold": [i // 2 % 2 for i in range(len(studies))]}).to_csv(tmp_path / "folds.csv", index=False)
    data = {"cache_dir": str(cache), "labels_csv": str(tmp_path / "labels.csv"), "folds_csv": str(tmp_path / "folds.csv"), "train_csv": str(tmp_path / "absent.csv"), "split": "folds", "batch_size": 2, "num_workers": 0, "pin_memory": False}
    model_cfg = {"backbone": "resnet18", "pretrained": False, "img_size": EDGE, "window": {"k_train": 3, "k_val": 3, "k_eval": None, "norm": "half"}, "spatial": {"enabled": True, "dim": 8, "att_hidden": 8}, "plane_aux": True}
    train_cfg = {"epochs": 1, "precision": "32-true", "learning_rate": 0.01, "encoder_lr": 0.001, "rank_weight": 0.0, "rank_queue": 0, "plane_aux_weight": 0.1}
    cfg = {"wandb_mode": "disabled", "output_dir": str(tmp_path / "output"), "data": data, "model": model_cfg, "train": train_cfg}
    stage1 = _train(tmp_path, "stage1", cfg)

    cfg["data"] = {**data, "depth_prior": prior.model_dump(mode="json"), "union_bag": {"k_min": 4, "k_max": 16, "eval_cap": 14}}
    cfg["model"] = {**model_cfg, "freeze_encoder": True, "series_neck": {"enabled": True, "gru_hidden": 8, "pos_hidden": 4, "att_hidden": 8}}
    cfg["train"] = {**train_cfg, "init_weights": str(stage1 / "model.pt"), "init_weights_skip": ["att.", "clsW", "clsb", "spatial_att.", "spatial_w", "planeW", "planeb"]}
    stage2 = _train(tmp_path, "neck", cfg)
    assert (stage2 / "oof.csv").exists() and (stage2 / "oof_last.csv").exists()
    before, after = torch.load(stage1 / "model.pt", weights_only=True), torch.load(stage2 / "model.pt", weights_only=True)
    assert all(torch.equal(before[key], after[key]) for key in before if key.startswith(("backbone.", "spatial.")))
    assert any(key.startswith("series_neck.gru.") for key in after) and "clsW" not in after

    model = load_model(stage2).eval()
    model.union_cap = 14
    dataset = WindowStudyDataset(cache, ["e1"], window=model.cfg.window, img_size=EDGE, depth_prior=prior, union_bag=UnionBagConfig(eval_cap=14))
    item = dataset[0]
    _, row = dataset.uid_index["e1"]
    mask = np.asarray(dataset.masks[dataset.stack_offsets[row] : dataset.stack_offsets[row + 1]])
    stack = np.zeros((len(mask), EDGE, EDGE), dtype=np.uint8)
    stack[mask != 0] = np.load(cache / "vols.npy")[dataset.offsets[row] : dataset.offsets[row + 1]]
    rows = meta[meta["StudyInstanceUID"] == "e1"].set_index("SeriesInstanceUID")
    placement = [
        {"SeriesInstanceUID": u["SeriesInstanceUID"], "stack_start": u["stack_offset"], "stack_stop": u["stack_offset"] + u["cap"], "Anatomical_Plane": u["Anatomical_Plane"], "Fluid_Sensitive": u["Fluid_Sensitive"]} for u in dataset.units[row]
    ]
    headers = [{**rows.loc[h["SeriesInstanceUID"]].to_dict(), **h} for h in placement]
    sub_vols, sub_masks = np.load(cache / "sub_vols.npy"), np.load(cache / "sub_masks.npy")
    subs = [
        SimpleNamespace(series_uid=rec.SeriesInstanceUID, index=int(rec.sub), mask=sub_masks[rec.start : rec.stop], pixels=sub_vols[rec.start : rec.stop], picks=json.loads(rec.selected_indices)) for rec in table[table[ID_COLUMN] == "e1"].itertuples()
    ]
    assert len(set(item["window_series"].tolist())) == len(headers) and item["window_u"].abs().max() <= 1
    with torch.no_grad():
        expected = torch.sigmoid(model.forward_batch({key: item[key][None] for key in ("windows", "window_slot", "window_pos", "window_rows", "window_series", "window_u")}))[0]
    assert torch.allclose(predict_study(model, stack, mask, series_headers=headers, subseries=subs), expected, atol=1e-4)
