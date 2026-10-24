import json
from pathlib import Path

import numpy as np
import pandas as pd

from rsna_knee.constants import ID_COLUMN, LABELS
from tools import make_folds

from .conftest import invoke


def test_make_folds_cli_groups_by_scanner(tmp_path: Path) -> None:
    """Twelve studies on six scanners, two duplicate reports dropped: every scanner lands in exactly one fold."""
    ids = [f"s{i:02d}" for i in range(12)]
    train = pd.DataFrame({ID_COLUMN: ids, "Report": [f"report {i % 10}" for i in range(12)]})
    for j, name in enumerate(LABELS):
        train[name] = [float((i + j) % 2) if i < 2 else np.nan for i in range(12)]
    train.to_csv(tmp_path / "train.csv", index=False)
    meta = pd.DataFrame({ID_COLUMN: ids, "Manufacturer": [f"m{i % 6}" for i in range(12)], "ManufacturerModelName": "x", "SoftwareVersions": "1", "ReceiveCoilName": "knee"})
    meta.to_csv(tmp_path / "series_meta.csv", index=False)
    pd.DataFrame({ID_COLUMN: ids, **{name: np.linspace(0, 1, 12) for name in LABELS}}).to_csv(tmp_path / "soft.csv", index=False)
    out = tmp_path / "folds" / "folds.csv"
    args = ["--train-csv", tmp_path / "train.csv", "--series-meta", tmp_path / "series_meta.csv", "--labels-csv", tmp_path / "soft.csv", "--out", out, "--folds", 2, "--dedup-reports"]
    result = invoke(make_folds.main, args)
    assert result.exit_code == 0, result.output
    folds = pd.read_csv(out)
    assert len(folds) == 10 and set(folds["fold"]) == {0, 1}
    scanner = meta.set_index(ID_COLUMN)["Manufacturer"]
    assert (folds.groupby(folds[ID_COLUMN].map(scanner))["fold"].nunique() == 1).all()
    assert json.loads(out.with_suffix(".json").read_text())["grouping"]["duplicate_reports_dropped"] == 2
