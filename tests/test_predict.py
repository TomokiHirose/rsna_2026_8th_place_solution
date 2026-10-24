from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from rsna_knee.config import Config
from rsna_knee.constants import ID_COLUMN, LABELS
from rsna_knee.data.dicom import PreprocessParams, build_study
from rsna_knee.inference import CONFIG_FILENAME, MODEL_FILENAME, load_model, predict_study
from rsna_knee.models import build_window_model
from rsna_knee.models.windows import build_windows
from rsna_knee.predict import INFER_DTYPES, PROB_HI, PROB_LO, group_by_preprocess, load_members, load_models, main, predict_split, preprocess_params_for, read_split, submission_columns

from .conftest import invoke
from .synthetic_study import make_study

# Slot table the synthetic artifacts are fitted on (64 slices)
TABLE = [18, 14, 12, 8, 12]
EDGE = 32
# Default recipe of the synthetic artifacts; it goes into the config as the ``preprocess`` block
RECIPE = {"crop_mm": 140.0, "size": EDGE, "span": (0.02, 0.98), "percentiles": (2.0, 98.0), "slot_table": TABLE}

STUDY_A = "1.2.826.0.1.aaaa"
STUDY_B = "1.2.826.0.1.bbbb"


def _window_cfg(**model: object) -> Config:
    base = {"backbone": "resnet18", "pretrained": False, "img_size": EDGE, "window": {"k_train": 4, "k_val": 4, "k_eval": 6}, "preprocess": RECIPE}
    return Config(experiment_name="pytest-predict", wandb_mode="disabled", model={**base, **model})


def _artifact(root: Path, name: str, cfg: Config, seed: int | None = None) -> Path:
    """An artifact directory holding the small resnet18 of ``cfg``; ``seed`` varies the weights per member."""
    artifact = root / name
    artifact.mkdir(parents=True)
    if seed is not None:
        torch.manual_seed(seed)
    model = build_window_model(cfg, pretrained=False)
    torch.save(model.state_dict(), artifact / MODEL_FILENAME)
    (artifact / CONFIG_FILENAME).write_text(json.dumps(cfg.model_dump(mode="json"), indent=2))
    return artifact


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One artifact directory whose ``preprocess`` names the 32 px, 64-slice recipe."""
    return _artifact(tmp_path_factory.mktemp("model"), "fold0", _window_cfg())


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Two studies with different numbers of present slices."""
    root = tmp_path_factory.mktemp("input")
    split_dir = root / "test_series"

    rows = make_study(split_dir, STUDY_A, [{"plane": "Sagittal", "fluid": 1, "n_slices": 4}, {"plane": "Sagittal", "fluid": 0, "n_slices": 3}, {"plane": "Coronal", "fluid": 1, "n_slices": 3}, {"plane": "Axial", "fluid": 1, "n_slices": 3}], seed=11)
    rows += make_study(split_dir, STUDY_B, [{"plane": "Coronal", "fluid": 0, "n_slices": 3}, {"plane": "Sagittal", "fluid": 1, "n_slices": 4}], seed=12)
    studies = [STUDY_A, STUDY_B]
    pd.DataFrame(rows).to_csv(root / "test_series.csv", index=False)
    pd.DataFrame({ID_COLUMN: studies}).to_csv(root / "test.csv", index=False)
    sample = pd.DataFrame({ID_COLUMN: studies})
    for label in LABELS:
        sample[label] = 0.5
    sample.to_csv(root / "sample_submission.csv", index=False)
    return root


@pytest.fixture(scope="session")
def submission(data_dir: Path, model_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> tuple[pd.DataFrame, Path]:
    """One full ``predict_split`` run shared by the tests below."""
    out = tmp_path_factory.mktemp("run") / "submission.csv"
    frame = predict_split(data_dir=data_dir, split="test", model_dirs=[model_dir], out=out, device="cpu")
    return frame, out


def test_the_submission_has_the_sample_shape_and_scores_every_study(submission: tuple[pd.DataFrame, Path], data_dir: Path, model_dir: Path, tmp_path: Path) -> None:
    frame, out = submission
    expected = submission_columns(data_dir)
    assert expected == [ID_COLUMN, *LABELS]
    assert list(frame.columns) == expected and list(pd.read_csv(out).columns) == expected
    assert frame[ID_COLUMN].tolist() == [STUDY_A, STUDY_B] and frame[ID_COLUMN].is_unique

    values = frame[LABELS].to_numpy(dtype=float)
    assert np.isfinite(values).all()
    assert values.min() >= PROB_LO and values.max() <= PROB_HI  # clipped into the open interval
    pd.testing.assert_frame_equal(frame.reset_index(drop=True), pd.read_csv(out, dtype={ID_COLUMN: str}), check_exact=False, rtol=1e-6)

    # Only the studies scored under ``limit`` are in the submission
    limited = predict_split(data_dir=data_dir, split="test", model_dirs=[model_dir], out=tmp_path / "submission.csv", device="cpu", limit=1)
    assert limited[ID_COLUMN].tolist() == [STUDY_A]


def test_members_with_different_recipes_are_scored_as_the_plain_mean_over_all_members(data_dir: Path, tmp_path: Path) -> None:
    """Three members with two recipes: every study is scored and the ensemble is the mean over members, not over recipe groups."""
    members = [_artifact(tmp_path, "area_a", _window_cfg(), seed=1), _artifact(tmp_path, "area_b", _window_cfg(), seed=2), _artifact(tmp_path, "exact_c", _window_cfg(preprocess={**RECIPE, "crop_mm": 120.0, "resample": "exact"}), seed=3)]
    recipes = [preprocess_params_for(load_model(m)) for m in members]
    assert recipes[0] == recipes[1] != recipes[2]

    ensemble = predict_split(data_dir=data_dir, split="test", model_dirs=members, out=tmp_path / "ensemble.csv", device="cpu").set_index(ID_COLUMN)
    singles = [predict_split(data_dir=data_dir, split="test", model_dirs=[m], out=tmp_path / f"{m.name}.csv", device="cpu").set_index(ID_COLUMN) for m in members]

    expected = np.mean([s.loc[ensemble.index, LABELS].to_numpy(dtype=float) for s in singles], axis=0)
    np.testing.assert_allclose(ensemble[LABELS].to_numpy(dtype=float), expected, rtol=0, atol=1e-6)

    # Groups are keyed by value: two members with the same recipe fold into one group, in first-seen order
    loaded = [load_model(m) for m in members]
    groups = group_by_preprocess([loaded[0], loaded[2], loaded[1]])
    assert [params for params, _ in groups] == [preprocess_params_for(loaded[0]), preprocess_params_for(loaded[2])]
    assert [[m is g for m, g in zip(group, want)] for (_, group), want in zip(groups, [[loaded[0], loaded[1]], [loaded[2]]])] == [[True, True], [True]]
    with pytest.raises(ValueError, match="no models"):
        group_by_preprocess([])


def test_short_slots_allow_context_larger_than_total_slices(tmp_path: Path) -> None:
    short = _artifact(tmp_path, "short", _window_cfg(preprocess={**RECIPE, "slot_table": [1, 1, 1, 1, 1]}, window={"context": 7, "norm": "none", "k_eval": None}))
    model = load_models([short])[0]
    pixels = np.full((5, EDGE, EDGE), 128, dtype=np.uint8)
    probs = predict_study(model, pixels, np.ones(5, dtype=bool), device="cpu")
    assert probs.shape == (len(LABELS),) and torch.isfinite(probs).all()

    # Scoring refuses a Stack whose side differs from the artifact recipe instead of resizing it
    n_slices = sum(TABLE)
    model = load_models([_artifact(tmp_path, "plain", _window_cfg())])[0]
    with pytest.raises(ValueError, match="img_size"):
        predict_study(model, np.zeros((n_slices, EDGE * 2, EDGE * 2), dtype=np.uint8), np.ones(n_slices, dtype=np.float32), device="cpu")


def test_a_context_5_artifact_is_scored_with_5_slice_windows(tmp_path: Path) -> None:
    """The artifact config carries ``context``; loading and scoring both use 5-channel Bags."""
    artifact = _artifact(tmp_path, "ctx5", _window_cfg(window={"k_train": 4, "k_val": 4, "k_eval": 6, "context": 5, "norm": "mri"}))

    loaded = load_models([artifact])[0]
    assert loaded.cfg.window.context == 5 and loaded.pixel_mean.shape == (1, 5, 1, 1)
    pixels = np.random.default_rng(0).integers(0, 256, (sum(TABLE), EDGE, EDGE), dtype=np.uint8)
    mask = np.ones(sum(TABLE), dtype=np.float32)
    probs = predict_study(loaded, pixels, mask, device="cpu")
    assert probs.shape == (len(LABELS),) and bool(((probs >= 0) & (probs <= 1)).all())
    bag = build_windows(pixels, mask, 6, EDGE, context=5, slot_table=TABLE)
    assert bag.shape == (6, 5, EDGE, EDGE)
    with torch.no_grad():
        assert torch.allclose(probs, torch.sigmoid(loaded(bag[None]).float())[0], atol=1e-6)


def test_predict_split_agrees_with_the_single_study_entry_point(data_dir: Path, model_dir: Path, submission: tuple[pd.DataFrame, Path]) -> None:
    """The CLI and the notebook entry point ``predict_study`` share one path."""
    frame, _ = submission
    scored = frame[frame[ID_COLUMN] == STUDY_A][LABELS].to_numpy()[0]

    _, series_rows = read_split(data_dir, "test")
    params = PreprocessParams(caps=tuple(TABLE), size=EDGE, span=(0.02, 0.98), crop_mm=140.0, percentiles=(2.0, 98.0))
    study = build_study(data_dir / "test_series" / STUDY_A, series_rows[STUDY_A], params, study_id=STUDY_A)
    direct = predict_study(load_model(model_dir), study.pixels, study.mask, device="cpu").numpy()
    assert np.abs(np.clip(direct, PROB_LO, PROB_HI) - scored).max() < 1e-6


def test_the_cli_accepts_the_notebook_knobs_and_they_are_a_no_op_on_cpu(data_dir: Path, model_dir: Path, submission: tuple[pd.DataFrame, Path], tmp_path: Path) -> None:
    """``--infer-dtype float16 --encode-chunk 32`` is accepted and matches fp32 on the CPU (autocast only opens on CUDA, chunks do not mix windows)."""
    assert INFER_DTYPES["none"] is None and INFER_DTYPES["float16"] is torch.float16
    base = ["--data-dir", data_dir, "--split", "test", "--model-dir", model_dir, "--device", "cpu", "--workers", 0]
    assert invoke(main, [*base, "--out", tmp_path / "bad.csv", "--infer-dtype", "bfloat16"]).exit_code == 2
    out = tmp_path / "knobs.csv"
    result = invoke(main, [*base, "--out", out, "--infer-dtype", "float16", "--encode-chunk", 32])
    assert result.exit_code == 0, result.output
    plain, _ = submission
    got = pd.read_csv(out, dtype={ID_COLUMN: str})
    np.testing.assert_allclose(got[LABELS].to_numpy(dtype=float), plain[LABELS].to_numpy(dtype=float), rtol=0, atol=1e-6)


def test_load_members_groups_by_recipe_and_overrides_k_eval(tmp_path: Path) -> None:
    """Members with the same recipe share a group, and k_eval overrides match by run name."""
    a = _artifact(tmp_path, "a", _window_cfg(), seed=1)
    b = _artifact(tmp_path, "b", _window_cfg(), seed=2)
    c = _artifact(tmp_path, "c", _window_cfg(preprocess={**RECIPE, "crop_mm": 120.0}), seed=3)
    run_b = f"{tmp_path.name}/b"
    members = load_members([a, b, c], k_eval={run_b: 5}, encode_chunk=4)
    assert [m.k_eval for m in members.table().itertuples()] == [members.models[0].cfg.window.k_eval, 5, members.models[2].cfg.window.k_eval]
    assert members.group_members == [[0, 1], [2]]
    assert [params.crop_mm for params in members.group_params] == [140.0, 120.0]
    assert all(model.encode_chunk == 4 for model in members.models)
    assert list(members.table().columns) == ["run", "backbone", "img_size", "slices", "k_eval", "group", "weight"]
    # Runs are mixed equally; these 3 members are 3 runs, so each gets a third (shown to 4 decimals)
    assert members.weights == pytest.approx([1 / 3] * 3)
    assert [m.weight for m in members.table().itertuples()] == [0.3333] * 3


def test_all_missing_study_fails_instead_of_producing_predictions(model_dir: Path) -> None:
    model = load_model(model_dir)
    with pytest.raises(ValueError, match="no usable slices"):
        predict_study(model, np.zeros((sum(TABLE), EDGE, EDGE), dtype=np.uint8), np.zeros(sum(TABLE), dtype=bool), device="cpu")
