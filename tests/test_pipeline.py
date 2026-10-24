import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from rsna_knee.constants import ID_COLUMN, LABELS
from rsna_knee.data.slots import SLOT_NAMES
from rsna_knee.experiments import ExperimentConfig
from tools import build_cache, kaggle_ops, select_weights

from .conftest import REPO_ROOT, invoke
from .synthetic_study import make_study

EDGE = 32
CAPS = "4,4,4,4,4"
# Five series per study; the first Sagittal series is long enough (>= 2 x cap in span) to get sub-series
SAGITTAL = [{"plane": "Sagittal", "fluid": 1, "n_slices": 10}, {"plane": "Sagittal", "fluid": 0, "n_slices": 5}]
SPECS = [*SAGITTAL, {"plane": "Coronal", "fluid": 1, "n_slices": 5}, {"plane": "Coronal", "fluid": 0, "n_slices": 4}, {"plane": "Axial", "fluid": 1, "n_slices": 5}]
TRAIN_IDS = [f"1.2.826.0.1.{1000 + i}" for i in range(8)]
TEST_IDS = ["1.2.826.0.1.2001", "1.2.826.0.1.2002"]
GOLD = TRAIN_IDS[:4]
STAGE1 = "pytest_base"
STAGE2 = "pytest_neck"
CONTEXT = "pytest_ctx"
BASE_LSE = "pytest_base_lse"


def _competition(root: Path) -> Path:
    """A competition directory laid out like the Kaggle input: train/test csv, series csv and DICOM folders."""
    for split, ids in (("train", TRAIN_IDS), ("test", TEST_IDS)):
        rows = []
        for seed, uid in enumerate(ids):
            rows += make_study(root / f"{split}_series", uid, SPECS if seed % 2 == 0 else SPECS[:3] + SPECS[4:], seed=seed + (100 if split == "test" else 0))
        pd.DataFrame(rows).to_csv(root / f"{split}_series.csv", index=False)
    train = pd.DataFrame({ID_COLUMN: TRAIN_IDS, "Report": ["text"] * len(TRAIN_IDS)})
    for j, name in enumerate(LABELS):
        train[name] = [float((i + j) % 2) if uid in GOLD else np.nan for i, uid in enumerate(TRAIN_IDS)]
    train.to_csv(root / "train.csv", index=False)
    pd.DataFrame({ID_COLUMN: TEST_IDS}).to_csv(root / "test.csv", index=False)
    pd.DataFrame({ID_COLUMN: TEST_IDS, **{name: 0.5 for name in LABELS}}).to_csv(root / "sample_submission.csv", index=False)
    return root


def _depth_prior(path: Path) -> Path:
    """A depth prior with one ACL component per Slot."""
    slots = {name: {"components": [{"finding": "ACL", "weight": 1.0, "mean": 0.5, "sigma": 0.2}], "findings": {}} for name in SLOT_NAMES}
    path.write_text(json.dumps({"version": 1, "truncate": [-1.0, 1.0], "sigma_floor": 0.08, "attention": "gap", "slots": slots}))
    return path


def _train(work: Path, name: str, data: dict, model: dict, train: dict) -> Path:
    """Run ``python -m rsna_knee.training`` in-process on one config and return the fold directory."""
    from rsna_knee.training import main

    cfg = {"experiment_name": name, "wandb_mode": "disabled", "output_dir": str(work / "output"), "data": data, "model": model, "train": train}
    path = work / f"{name}.json"
    path.write_text(json.dumps(cfg))
    result = invoke(main, ["--config", path, "--fold", 0])
    assert result.exit_code == 0, result.output
    return work / "output" / name / "fold0"


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    """Build the cache from DICOM, train both stages, pick the weights and fill a fake Kaggle input tree."""
    work = tmp_path_factory.mktemp("pipeline")
    comp = _competition(work / "comp")
    cache = work / "cache"
    recipe = ["--slots", "native", "--caps", CAPS, "--span", "0.06,0.94", "--size", EDGE, "--resample", "exact", "--long-series-sampling", "uniform", "--series-layout", "all", "--subseries", "--memmap"]
    common = ["--data-dir", comp, "--split", "train", *recipe, "--code-commit", "pytest"]
    result = invoke(build_cache.main, [*common, "--out", cache])
    assert result.exit_code == 0, result.output
    pd.DataFrame({ID_COLUMN: TRAIN_IDS, **{name: np.linspace(0.05, 0.95, len(TRAIN_IDS)) for name in LABELS}}).to_csv(work / "labels.csv", index=False)
    pd.DataFrame({ID_COLUMN: TRAIN_IDS, "fold": [i % 2 for i in range(len(TRAIN_IDS))]}).to_csv(work / "folds.csv", index=False)
    data = {"cache_dir": str(cache), "labels_csv": str(work / "labels.csv"), "train_csv": str(comp / "train.csv"), "folds_csv": str(work / "folds.csv"), "batch_size": 2, "num_workers": 0, "pin_memory": False}
    model = {"backbone": "resnet18", "pretrained": False, "img_size": EDGE, "window": {"k_train": 3, "k_val": 3, "k_eval": None, "norm": "half"}, "spatial": {"enabled": True, "dim": 8, "att_hidden": 8}, "plane_aux": True}
    train = {"epochs": 1, "precision": "32-true", "learning_rate": 0.01, "encoder_lr": 0.001, "rank_weight": 0.0, "rank_queue": 0, "plane_aux_weight": 0.1, "save_every_epoch": True, "ema_decay": 0.9}
    stage1 = _train(work, STAGE1, data, model, train)
    union = {**data, "depth_prior": {"path": str(_depth_prior(work / "prior.json")), "alpha": 1.0}, "union_bag": {"k_min": 4, "k_max": 16, "eval_cap": 14}}
    neck = {**model, "freeze_encoder": True, "series_neck": {"enabled": True, "gru_hidden": 8, "pos_hidden": 4, "att_hidden": 8}}
    skip = ["att.", "clsW", "clsb", "spatial_att.", "spatial_w", "planeW", "planeb"]
    stage2 = _train(work, STAGE2, union, neck, {**train, "learning_rate": 0.001, "encoder_lr": 0.0, "init_weights": str(stage1 / "checkpoints" / "epoch00_ema.pt"), "init_weights_skip": skip})
    lse = {**model, "window": {**model["window"], "pool": "lse"}, "plane_aux": False}
    lse_train = {**train, "plane_aux_weight": 0.0, "window_loss_weight": 0.3}
    base_lse = _train(work, BASE_LSE, {**data, "depth_prior": union["depth_prior"]}, lse, lse_train)
    contexts = {"context": {"enabled": True, "dim": 8, "hidden": 8, "heads": 2}, "slot_context": {"enabled": True, "dim": 8, "att_hidden": 8, "hidden": 8}}
    context = _train(work, CONTEXT, union, {**lse, "freeze_encoder": True, **contexts}, {**lse_train, "encoder_lr": 0.0, "init_weights": str(base_lse / "model.pt"), "init_weights_skip": ["clsW", "clsb"]})
    store = work / "submit"
    result = invoke(select_weights.main, [STAGE2, CONTEXT, "--min-epoch", 0, "--union-cap", 14, "--output-dir", work / "output", "--store", store])
    assert result.exit_code == 0, result.output
    kaggle_input = work / "kaggle" / "input"
    (kaggle_input / "competitions").mkdir(parents=True)
    (kaggle_input / "competitions" / "rsna-knee-abnormality-detection").symlink_to(comp)
    (kaggle_input / "datasets" / "OWNER").mkdir(parents=True)
    (kaggle_input / "datasets" / "OWNER" / "rsna-knee-8th-pkg").symlink_to(REPO_ROOT)
    (kaggle_input / "datasets" / "OWNER" / "rsna-knee-8th-model").symlink_to(store)
    return SimpleNamespace(work=work, comp=comp, cache=cache, stage1=stage1, stage2=stage2, base_lse=base_lse, context=context, store=store, kaggle_input=kaggle_input)


def test_cache_holds_every_series_and_sub_series(pipeline: SimpleNamespace) -> None:
    ids = [str(uid) for uid in np.load(pipeline.cache / "ids.npy")]
    assert ids == TRAIN_IDS
    series = pd.read_csv(pipeline.cache / "series.csv", dtype={ID_COLUMN: str})
    assert series.groupby(ID_COLUMN).size().loc[TRAIN_IDS].tolist() == [5, 4] * 4
    manifest = json.loads((pipeline.cache / "manifest.json").read_text())
    assert manifest["series_layout"] == "all" and manifest["subseries"] and manifest["size"] == EDGE
    assert len(pd.read_csv(pipeline.cache / "subseries.csv")) > 0
    masks, offsets = np.load(pipeline.cache / "masks.npy"), np.load(pipeline.cache / "offsets.npy")
    assert len(np.load(pipeline.cache / "vols.npy", mmap_mode="r")) == int(masks.sum()) == offsets[-1]


def test_stage1_writes_the_artifact_and_epoch_weights(pipeline: SimpleNamespace) -> None:
    assert all((pipeline.stage1 / name).exists() for name in ("model.pt", "config.json", "oof.csv", "checkpoints/epoch00_ema.pt"))
    oof = pd.read_csv(pipeline.stage1 / "oof.csv", dtype={"study_id": str})
    assert sorted(oof["study_id"]) == sorted(GOLD) and np.isfinite(oof[[f"pred_{name}" for name in LABELS]].to_numpy()).all()
    metrics = pd.read_csv(pipeline.stage1 / "version_0" / "metrics.csv")
    assert metrics["val/auc_macro"].notna().any()
    assert ExperimentConfig.model_validate_json((pipeline.stage1.parent / "inference.json").read_text()).model.preprocess.size == EDGE


def test_stage2_keeps_the_frozen_encoder_and_resets_the_head(pipeline: SimpleNamespace) -> None:
    before = torch.load(pipeline.stage1 / "checkpoints" / "epoch00_ema.pt", weights_only=True)
    after = torch.load(pipeline.stage2 / "model.pt", weights_only=True)
    assert all(torch.equal(before[key], after[key]) for key in before if key.startswith(("backbone.", "spatial.")))
    assert any(key.startswith("series_neck.gru.") for key in after) and "clsW" not in after
    context = torch.load(pipeline.context / "model.pt", weights_only=True)
    base = torch.load(pipeline.base_lse / "model.pt", weights_only=True)
    assert all(torch.equal(base[key], context[key]) for key in base if key.startswith(("backbone.", "spatial.")))
    assert any(key.startswith("context.") for key in context) and any(key.startswith("slot_context.") for key in context) and not torch.equal(context["clsW"], base["clsW"])


def test_select_weights_fills_the_store(pipeline: SimpleNamespace) -> None:
    for name in (STAGE2, CONTEXT):
        assert (pipeline.store / name / "fold0" / "model.pt").read_bytes() == (pipeline.work / "output" / name / "fold0" / "model.pt").read_bytes()
        assert ExperimentConfig.model_validate_json((pipeline.store / name / "inference.json").read_text()).union_cap == 14
        assert json.loads((pipeline.store / name / "selection.json").read_text())["folds"][0]["used_epoch"] == 0


def _run_notebook(kernel: str, pipeline: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, **overrides: object) -> tuple[pd.DataFrame, dict]:
    """Execute the code cells of ``kernels/<kernel>/inference.ipynb`` on the CPU with constants overridden after the constants cell."""
    import rsna_knee.predict as predict

    out = pipeline.work / f"kaggle_{kernel}"
    out.mkdir(exist_ok=True)
    monkeypatch.setenv("KAGGLE_INPUT", str(pipeline.kaggle_input))
    monkeypatch.setenv("KAGGLE_WORKING", str(out))
    monkeypatch.setattr(predict, "resolve_devices", lambda name, devices=None: [torch.device("cpu")])
    notebook = json.loads((REPO_ROOT / "kernels" / kernel / "inference.ipynb").read_text())
    scope: dict = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        exec(compile(source, f"{kernel}:{cell.get('id')}", "exec"), scope)
        if cell.get("id") == "constants":
            scope.update(WORKERS=0, PREFETCH=2, IO_THREADS=1, **overrides)
    return pd.read_csv(out / "submission.csv", dtype={ID_COLUMN: str}), scope


def _single(pipeline: SimpleNamespace, name: str) -> np.ndarray:
    """Probabilities of one experiment, scored study by study with ``build_study`` + ``predict_study`` (union bag at the store's cap)."""
    from rsna_knee.data.dicom import build_study
    from rsna_knee.inference import predict_study
    from rsna_knee.predict import PROB_HI, PROB_LO, load_experiments, read_split

    members = load_experiments(pipeline.store, [[name]])
    _, series_rows = read_split(pipeline.comp, "test")
    probs = []
    for uid in TEST_IDS:
        study = build_study(pipeline.comp / "test_series" / uid, series_rows[uid], members.group_params[0], study_id=uid)
        probs.append(predict_study(members.models[0], study.pixels, study.mask, series_headers=study.series_headers, subseries=study.subseries).numpy())
    return np.clip(np.stack(probs), PROB_LO, PROB_HI)


def test_notebook_a_matches_study_scoring_on_the_series_neck_model(pipeline: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    submission, scope = _run_notebook("A", pipeline, monkeypatch, Ensemble=[[STAGE2]], PART_WEIGHTS=None)
    assert scope["PKG_DIR"].name == "rsna-knee-8th-pkg" and scope["MODEL_ROOT"].name == "rsna-knee-8th-model"
    assert list(submission.columns) == [ID_COLUMN, *LABELS] and submission[ID_COLUMN].tolist() == TEST_IDS
    assert scope["members"].models[0].union_cap == 14
    np.testing.assert_allclose(submission[LABELS].to_numpy(), _single(pipeline, STAGE2), rtol=1e-5, atol=1e-6)


def test_notebook_b_blends_parts_by_weight(pipeline: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    submission, scope = _run_notebook("B", pipeline, monkeypatch, Ensemble=[[STAGE2], [CONTEXT]], PART_WEIGHTS=[0.25, 0.75])
    neck, context = _single(pipeline, STAGE2), _single(pipeline, CONTEXT)
    assert not np.allclose(neck, context)
    np.testing.assert_allclose(submission[LABELS].to_numpy(), 0.25 * neck + 0.75 * context, rtol=1e-5, atol=1e-6)
    scope["BLEND"] = "rank"
    notebook = json.loads((REPO_ROOT / "kernels" / "B" / "inference.ipynb").read_text())
    write = next("".join(cell["source"]) for cell in notebook["cells"] if cell.get("id") == "write")
    exec(compile(write, "B:write", "exec"), scope)
    from rsna_knee.predict import rank_average

    expected = 0.25 * rank_average(neck[:, None]) + 0.75 * rank_average(context[:, None])
    np.testing.assert_allclose(pd.read_csv(scope["OUT"])[LABELS].to_numpy(), expected, rtol=1e-6)


def test_kaggle_ops_model_build_only_stages_the_store(pipeline: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kaggle_ops, "STAGE_ROOT", pipeline.work / "stage")
    monkeypatch.setattr(kaggle_ops, "SUBMIT_STORE", pipeline.store)
    result = invoke(kaggle_ops.main, ["--owner", "someone", "model", "--build-only"])
    assert result.exit_code == 0
    staged = next((pipeline.work / "stage").glob("rsna-knee-8th-model-*"))
    assert json.loads((staged / "dataset-metadata.json").read_text())["id"] == "someone/rsna-knee-8th-model"
    for name in (STAGE2, CONTEXT):
        assert (staged / "payload" / name / "fold0" / "model.pt").samefile(pipeline.store / name / "fold0" / "model.pt")
        assert (staged / "payload" / name / "inference.json").is_file()
    assert "collected: 2 folds" in result.output
