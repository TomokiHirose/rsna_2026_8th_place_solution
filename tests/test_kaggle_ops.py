import json
from pathlib import Path

import pytest

from rsna_knee.config import Config
from rsna_knee.experiments import export_experiment_config
from tools import kaggle_ops

from .conftest import REPO_ROOT, invoke


def test_pkg_build_only_stages_the_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kaggle_ops, "STAGE_ROOT", tmp_path)
    result = invoke(kaggle_ops.main, ["--owner", "someone", "pkg", "--build-only"])
    assert result.exit_code == 0
    staged = next(tmp_path.glob("rsna-knee-8th-pkg-*"))
    assert json.loads((staged / "dataset-metadata.json").read_text())["id"] == "someone/rsna-knee-8th-pkg"
    package = staged / "payload" / "rsna_knee"
    assert (package / "inference.py").read_text() == (REPO_ROOT / "rsna_knee" / "inference.py").read_text()
    assert not list(package.rglob("__pycache__"))
    assert "collected:" in result.output


def test_model_build_only_fails_on_an_incomplete_fold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tiny_cfg: Config) -> None:
    store = tmp_path / "submit"
    export_experiment_config(store / "exp", tiny_cfg.model)
    (store / "exp" / "fold0").mkdir()
    (store / "exp" / "fold0" / "model.pt").write_bytes(b"weights")
    monkeypatch.setattr(kaggle_ops, "STAGE_ROOT", tmp_path / "stage")
    monkeypatch.setattr(kaggle_ops, "SUBMIT_STORE", store)
    result = invoke(kaggle_ops.main, ["--owner", "someone", "model", "--build-only"])
    assert result.exit_code == 1 and "missing artifact" in result.stderr


@pytest.mark.parametrize(("target", "slug"), [("A", "rsna-knee-8th-inference-a"), ("B", "rsna-knee-8th-inference-b"), ("C", "rsna-knee-8th-private-best-ensemble")])
def test_stage_kernel_fills_in_the_owner(target: str, slug: str, tmp_path: Path) -> None:
    kaggle_ops.stage_kernel(kaggle_ops.KERNEL_DIRS[target], tmp_path, "someone")
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert meta["id"] == kaggle_ops.kernel_id("someone", target) == f"someone/{slug}"
    assert meta["dataset_sources"] == ["someone/rsna-knee-8th-pkg", "someone/rsna-knee-8th-model"]
    notebook = json.loads((tmp_path / meta["code_file"]).read_text())
    source = "".join("".join(cell["source"]) for cell in notebook["cells"])
    assert 'OWNER = "someone"' in source and 'OWNER = "OWNER"' not in source


def test_stage_kernel_needs_exactly_one_placeholder(tmp_path: Path) -> None:
    source = tmp_path / "kernel"
    source.mkdir()
    meta = json.loads((kaggle_ops.KERNEL_DIRS["A"] / "kernel-metadata.json").read_text())
    (source / "kernel-metadata.json").write_text(json.dumps(meta))
    (source / meta["code_file"]).write_text(json.dumps({"cells": [{"cell_type": "code", "source": ["print(1)\n"]}]}))
    (tmp_path / "out").mkdir()
    with pytest.raises(kaggle_ops.OpsError, match="exactly one line"):
        kaggle_ops.stage_kernel(source, tmp_path / "out", "someone")
