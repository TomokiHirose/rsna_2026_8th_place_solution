from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rsna_knee.constants import LABELS
from rsna_knee.data.slots import NUM_SLICES
from tools import build_cache

from .conftest import invoke
from .synthetic_study import make_study

STUDY_A = "1.2.826.0.1.aaaa"
STUDY_B = "1.2.826.0.1.bbbb"
SIZE = 32


@pytest.fixture
def mini_dataset(tmp_path: Path) -> Path:
    """Two synthetic studies laid out like the competition data directory."""
    data_dir = tmp_path / "input"
    split_dir = data_dir / "train_series"
    rows = make_study(
        split_dir,
        STUDY_A,
        [
            {"plane": "Sagittal", "fluid": 1, "n_slices": 5, "pixel_spacing": 4.0},
            {"plane": "Sagittal", "fluid": 0, "n_slices": 4},
            {"plane": "Coronal", "fluid": 1, "n_slices": 4},
            {"plane": "Axial", "fluid": 1, "n_slices": 3},
            {"plane": "Axial", "fluid": 0, "n_slices": 3},  # there is only one axial Slot, so this series is never assigned
        ],
        seed=1,
    )
    rows += make_study(split_dir, STUDY_B, [{"plane": "Coronal", "fluid": 0, "n_slices": 4}], seed=2)
    pd.DataFrame(rows).to_csv(data_dir / "train_series.csv", index=False)

    labels = pd.DataFrame([{"StudyInstanceUID": STUDY_A, **{label: 1.0 for label in LABELS}}])
    studies = pd.DataFrame({"StudyInstanceUID": [STUDY_A, STUDY_B]})
    studies.to_csv(data_dir / "train.csv", index=False)
    labels.to_csv(data_dir / "labels.csv", index=False)
    return data_dir


def run_builder(data_dir: Path, out_dir: Path, *extra: str) -> int:
    """Build the merged layout (``--memmap``) used for training caches and return the exit code."""
    return invoke(build_cache.main, ["--data-dir", data_dir, "--split", "train", "--out", out_dir, "--size", SIZE, "--memmap", *extra]).exit_code


def test_build_cache_writes_vols_index_and_meta(mini_dataset: Path, tmp_path: Path) -> None:
    out = tmp_path / "cache"
    assert run_builder(mini_dataset, out, "--workers", "2") == 0

    assert [str(uid) for uid in np.load(out / "ids.npy")] == [STUDY_A, STUDY_B]  # train.csv order
    masks = np.load(out / "masks.npy")
    assert masks.shape == (2, NUM_SLICES) and masks.dtype == np.uint8
    # vols holds only present slices in study order; the padded assembly file is removed
    volumes = np.load(out / "vols.npy", mmap_mode="r")
    assert volumes.shape == (int(masks.sum()), SIZE, SIZE) and volumes.dtype == np.uint8
    assert np.load(out / "offsets.npy").tolist() == [0, int(masks[0].sum()), int(masks.sum())]
    assert (np.asarray(volumes).reshape(len(volumes), -1).max(axis=1) > 0).all()
    assert not (out / build_cache.PADDED_VOLS).exists()

    index = pd.read_csv(out / "index.csv", dtype={"mask": str})
    assert sorted(index["StudyInstanceUID"]) == sorted([STUDY_A, STUDY_B])
    row_a = index.set_index("StudyInstanceUID").loc[STUDY_A]
    assert row_a["n_series"] == 5 and row_a["n_slots"] == 4  # there is no second coronal series
    assert row_a["n_present"] == 44 - 6
    assert row_a["mask"].split("|") == ["1" * 12, "1" * 10, "1" * 8, "0" * 6, "1" * 8]
    assert row_a["slot0_n_slices"] == 5 and row_a["slot3_series"] != row_a["slot3_series"]  # NaN
    # A study whose only coronal series is not fluid-sensitive still fills the *first* coronal Slot
    assert index.set_index("StudyInstanceUID").loc[STUDY_B, "mask"].split("|") == ["0" * 12, "0" * 10, "1" * 8, "0" * 6, "0" * 8]

    meta = pd.read_csv(out / "series_meta.csv")
    assert len(meta) == 6  # every series of both studies, assigned or not
    assert set(meta["assigned"]) == {0, 1}  # the second axial series is read but not assigned
    assert meta["sort_rule"].isin(["ipp_normal", "not_read"]).all()
    assert meta["Rows"].eq(64).all() and meta["PixelSpacing_0"].isin([2.0, 4.0]).all()
    assert (out / "warnings.log").exists()


def test_shards_and_limit_partition_the_studies(mini_dataset: Path, tmp_path: Path) -> None:
    shard0, shard1 = tmp_path / "s0", tmp_path / "s1"
    run_builder(mini_dataset, shard0, "--shard", "0/2")
    run_builder(mini_dataset, shard1, "--shard", "1/2")
    ids0 = [str(uid) for uid in np.load(shard0 / "ids.npy")]
    ids1 = [str(uid) for uid in np.load(shard1 / "ids.npy")]
    assert ids0 == [STUDY_A] and ids1 == [STUDY_B]
    assert set(pd.read_csv(shard0 / "index.csv")["StudyInstanceUID"]) == {STUDY_A}

    run_builder(mini_dataset, tmp_path / "first", "--limit", "1")
    assert [str(uid) for uid in np.load(tmp_path / "first" / "ids.npy")] == [STUDY_A]
