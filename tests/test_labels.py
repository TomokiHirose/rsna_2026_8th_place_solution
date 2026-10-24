from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rsna_knee.constants import ID_COLUMN, LABELS
from rsna_knee.modules.labels import SEVERITY_KEYS, UNKNOWN_VALUE, binarize, load_labels, load_severity

UIDS = ["s0", "s1", "s2"]


def write_llm(path: Path) -> Path:
    """Write soft labels with sentinels scattered over all three studies."""
    rows = {"s0": [UNKNOWN_VALUE] * 12, "s1": [0.9, UNKNOWN_VALUE, 0.02] + [0.4] * 9, "s2": [0.1] * 12}
    frame = pd.DataFrame([rows[uid] for uid in UIDS], columns=LABELS)
    frame.insert(0, ID_COLUMN, UIDS)
    # Shuffling the column order on disk must not change the output column order
    frame = frame[[ID_COLUMN, *reversed(LABELS)]]
    frame.to_csv(path, index=False)
    return path


def write_train(path: Path, gold_uid: str = "s0") -> Path:
    """Write a train.csv with one gold row; the others are empty, as in the real file."""
    frame = pd.DataFrame({ID_COLUMN: UIDS, "Report": ["text"] * 3})
    for i, name in enumerate(LABELS):
        frame[name] = [float(i % 2) if uid == gold_uid else np.nan for uid in UIDS]
    frame.to_csv(path, index=False)
    return path


def test_gold_overrides_llm_only_for_complete_rows(tmp_path: Path) -> None:
    plain = load_labels(write_llm(tmp_path / "llm.csv"))
    assert list(plain.columns) == LABELS  # output column order is fixed whatever the order on disk
    assert plain.index.tolist() == UIDS
    assert plain.index.name == ID_COLUMN
    assert (plain.dtypes == "float32").all()

    labels = load_labels(tmp_path / "llm.csv", write_train(tmp_path / "train.csv"))
    assert labels.loc["s0"].tolist() == [float(i % 2) for i in range(12)]
    assert labels.loc["s2"].tolist() == pytest.approx([0.1] * 12)

    # A row with only some of the 12 columns filled must not override anything
    train = pd.read_csv(tmp_path / "train.csv")
    train.loc[train[ID_COLUMN] == "s0", LABELS[0]] = np.nan
    train.to_csv(tmp_path / "partial.csv", index=False)
    assert load_labels(tmp_path / "llm.csv", tmp_path / "partial.csv").loc["s0"].tolist() == pytest.approx([UNKNOWN_VALUE] * 12)
    # A missing train.csv is treated the same way and does not fail
    assert load_labels(tmp_path / "llm.csv", tmp_path / "absent.csv").loc["s0"].tolist() == pytest.approx([UNKNOWN_VALUE] * 12)


def test_unknown_to_nan_skips_gold_rows(tmp_path: Path) -> None:
    llm, train = write_llm(tmp_path / "llm.csv"), write_train(tmp_path / "train.csv")
    labels = load_labels(llm, train, unknown_to_nan=True)
    # s0 is gold, so its 0/1 values stay even where they equal the sentinel
    assert not labels.loc["s0"].isna().any()
    assert bool(np.isnan(labels.loc["s1", LABELS[1]]))
    assert labels.loc["s1", LABELS[0]] == pytest.approx(0.9)
    assert not labels.loc["s2"].isna().any()
    assert int(labels.isna().to_numpy().sum()) == 1

    # The sentinel value is configurable; 0.1 makes the whole s2 row unknown
    other = load_labels(llm, None, unknown_to_nan=True, unknown_value=0.1)
    assert other.loc["s2"].isna().all()
    assert not other.loc["s0"].isna().any()


def test_empty_cells_follow_unknown_rule(tmp_path: Path) -> None:
    """Empty CSV cells (not mentioned by the report) follow the same rule as the sentinel."""
    frame = pd.read_csv(write_llm(tmp_path / "llm.csv"))
    frame.loc[frame[ID_COLUMN] == "s1", [LABELS[2], LABELS[3]]] = np.nan  # empty cells in a non-gold row
    frame.loc[frame[ID_COLUMN] == "s0", LABELS[5]] = np.nan  # an empty cell in the gold row; train.csv wins
    masked = tmp_path / "masked.csv"
    frame.to_csv(masked, index=False)
    train = write_train(tmp_path / "train.csv")

    filled = load_labels(masked, train, unknown_to_nan=False, unknown_value=0.05)
    assert not filled.isna().to_numpy().any()
    assert filled.loc["s1", LABELS[2]] == pytest.approx(0.05)
    assert filled.loc["s1", LABELS[3]] == pytest.approx(0.05)
    assert filled.loc["s1", LABELS[0]] == pytest.approx(0.9)
    assert filled.loc["s1", LABELS[1]] == pytest.approx(UNKNOWN_VALUE)  # the sentinel is not filled
    assert filled.loc["s0"].tolist() == [float(i % 2) for i in range(12)]
    assert (filled.dtypes == "float32").all()

    kept = load_labels(masked, train, unknown_to_nan=True)
    assert bool(np.isnan(kept.loc["s1", LABELS[2]]))
    assert bool(np.isnan(kept.loc["s1", LABELS[3]]))
    assert bool(np.isnan(kept.loc["s1", LABELS[1]]))  # the sentinel also becomes NaN
    assert kept.loc["s1", LABELS[0]] == pytest.approx(0.9)
    assert not kept.loc["s0"].isna().any()
    assert kept.loc["s0"].tolist() == [float(i % 2) for i in range(12)]
    assert int(kept.isna().to_numpy().sum()) == 3


def test_binarize_splits_at_half() -> None:
    out = binarize(np.array([[0.9, 0.5, 0.25, np.nan]]))
    assert out[0, 0] == 1.0
    assert np.isnan(out[0, 1])  # exactly one half is unknown
    assert out[0, 2] == 0.0
    assert np.isnan(out[0, 3])


def write_severity(path: Path) -> Path:
    """Long-format severities: all 12 findings for s0, only the first finding for s1."""
    keys = list(SEVERITY_KEYS)
    rows = [{ID_COLUMN: "s0", "target": key, "p_none": 1.0 - (i + 1) / 24, "severity_soft": (i + 1) / 24} for i, key in enumerate(keys)]
    rows.append({ID_COLUMN: "s1", "target": keys[0], "p_none": 0.0, "severity_soft": 1.0})
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_load_severity_pivots_the_long_table_into_label_order(tmp_path: Path) -> None:
    severity = load_severity(write_severity(tmp_path / "severity.csv"), ["s2", "s1", "s0"])
    assert severity.shape == (3, len(LABELS)) and severity.dtype == np.float32
    assert np.isnan(severity[0]).all()  # a study missing from the table
    assert severity[1, 0] == pytest.approx(1.0) and np.isnan(severity[1, 1:]).all()  # a missing (study, target) cell
    assert severity[2] == pytest.approx([(i + 1) / 24 for i in range(len(LABELS))])  # target keys follow the LABELS order
