import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from rsna_knee.config.config import ModelConfig
from rsna_knee.constants import ID_COLUMN, LABELS
from rsna_knee.data.depth_prior import finding_density_table, interval_log_probs, load_depth_prior
from rsna_knee.models.window_model import WindowModel
from tools import fit_depth_prior, window_attention

from .conftest import invoke
from .series_cache import EDGE as SERIES_EDGE
from .series_cache import write_depth_prior, write_series_cache


def test_window_attention_dump_feeds_fit_depth_prior(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Dump the teacher's window attention on the CPU (``.cuda()`` is a no-op here), then fit the depth prior from the dump."""
    cache = write_series_cache(tmp_path / "cache")
    write_depth_prior(cache, tmp_path / "unused_prior.json")
    teacher = WindowModel(ModelConfig(backbone="resnet18", pretrained=False, img_size=SERIES_EDGE, spatial={"enabled": True, "dim": 8, "att_hidden": 8})).eval()
    monkeypatch.setattr(window_attention, "load_model", lambda path: teacher)
    monkeypatch.setattr(torch.nn.Module, "cuda", lambda self, *args, **kwargs: self)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
    pd.DataFrame({ID_COLUMN: ["s0", "s1"], "fold": [0, 1]}).to_csv(tmp_path / "folds.csv", index=False)
    result = invoke(window_attention.main, ["--teacher", tmp_path / "teacher", "--folds", tmp_path / "folds.csv", "--cache", cache, "--out", tmp_path / "dump"])
    assert result.exit_code == 0, result.output
    frame = pd.read_csv(tmp_path / "dump" / "window_attention.csv.gz")
    gap = frame[[f"gap_{name}" for name in LABELS]].to_numpy()
    assert np.isfinite(gap).all() and np.allclose(frame.assign(total=gap[:, 0]).groupby("study_id")["total"].sum(), 1.0, atol=1e-4)
    assert len(pd.read_csv(tmp_path / "dump" / "study_probs.csv")) == 2

    # Gold studies must be outside the dump; the two dumped studies get no gold labels
    pd.DataFrame({ID_COLUMN: ["g0", "s0"], **{name: [1.0, np.nan] for name in LABELS}}).to_csv(tmp_path / "train.csv", index=False)
    out = tmp_path / "prior" / "v1.json"
    args = ["--attention", tmp_path / "dump" / "window_attention.csv.gz", "--cache", cache, "--train-csv", tmp_path / "train.csv", "--out", out, "--max-k", 2, "--max-iter", 200]
    result = invoke(fit_depth_prior.main, args)
    assert result.exit_code == 0, result.output
    prior = json.loads(out.read_text())
    fitted = [c for slot in prior["slots"].values() for c in slot["components"]]
    assert fitted and all(np.isfinite([c["weight"], c["mean"], c["sigma"]]).all() for c in fitted)
    loaded = load_depth_prior(out)
    for components in loaded.values():
        assert components.shape[1] == 3 and np.isclose(components[:, 0].sum(), 1.0, atol=1e-6) and (components[:, 2] > 0).all()
        mass = np.exp(interval_log_probs(components, np.array([-1.0]), np.array([1.0])))
        assert np.allclose(mass, 1.0, atol=1e-6)
    table = finding_density_table(out, n_grid=41)
    assert table.max() <= 1.0 + 1e-6 and table.min() >= 0.0 and np.allclose(table[1], table[1][..., ::-1], atol=1e-6)
