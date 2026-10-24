import json
from pathlib import Path

import pandas as pd

from rsna_knee.experiments import ExperimentConfig, export_experiment_config
from tools.select_weights import choose, main

from .conftest import invoke


def _fold(path: Path, aucs: list[float]) -> None:
    """Write per-epoch gold AUCs to metrics.csv and place weight files whose contents identify them."""
    (path / "version_0").mkdir(parents=True)
    rows = [{"epoch": epoch, "val/auc_macro": auc} for epoch, auc in enumerate(aucs)] + [{"epoch": 0, "val/auc_macro": None}]
    pd.DataFrame(rows).to_csv(path / "version_0" / "metrics.csv", index=False)
    (path / "checkpoints").mkdir()
    for epoch in range(len(aucs)):
        (path / "checkpoints" / f"epoch{epoch:02d}_ema.pt").write_text(f"ema {epoch}")
    (path / "model.pt").write_text("gold best")
    (path / "config.json").write_text("{}")


def test_choose_uses_the_earliest_best_and_moves_early_bests_to_the_min_epoch() -> None:
    assert choose({0: 0.90, 1: 0.95, 2: 0.94, 3: 0.95, 4: 0.93}, 4) == (1, 4, "checkpoints/epoch04_ema.pt")
    assert choose({0: 0.90, 3: 0.92, 4: 0.93, 5: 0.94, 6: 0.94}, 4) == (5, 5, "model.pt")


def test_select_weights_fills_the_submit_store_per_fold(tmp_path: Path, tiny_cfg) -> None:
    """A fold whose best epoch is before 4 gets the epoch-4 EMA, others get model.pt; inference.json carries union_cap."""
    output = tmp_path / "output"
    export_experiment_config(output / "exp", tiny_cfg.model)
    _fold(output / "exp" / "fold0", [0.90, 0.95, 0.94, 0.93, 0.94, 0.93])
    _fold(output / "exp" / "fold1", [0.90, 0.91, 0.92, 0.93, 0.94, 0.95])
    store = tmp_path / "submit"
    assert invoke(main, ["exp", "--union-cap", 384, "--output-dir", output, "--store", store]).exit_code == 0
    assert (store / "exp" / "fold0" / "model.pt").read_text() == "ema 4"
    assert (store / "exp" / "fold1" / "model.pt").read_text() == "gold best"
    assert ExperimentConfig.model_validate_json((store / "exp" / "inference.json").read_text()).union_cap == 384
    selection = json.loads((store / "exp" / "selection.json").read_text())
    assert [(row["best_epoch"], row["used_epoch"]) for row in selection["folds"]] == [(1, 4), (5, 5)]
