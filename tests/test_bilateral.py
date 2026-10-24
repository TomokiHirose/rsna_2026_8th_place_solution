from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rsna_knee.data.bilateral import series_center_x, split_bilateral_studies
from rsna_knee.modules.window_dataset import load_subcases
from tools.split_bilateral_knees import main as split_cli

from .conftest import invoke

# IOP per plane: the Sagittal normal is -x; Coronal and Axial rows run along +x
IOP = {"Sagittal": "0|1|0|0|0|-1", "Coronal": "1|0|0|0|0|-1", "Axial": "1|0|0|0|1|0"}


def _row(study: str, series: str, plane: str, center_x: float, n: int = 21, step: float = 3.0, size: int = 256, spacing: float = 0.5) -> dict:
    """A ``series_meta.csv`` row of a series centred at x = ``center_x``."""
    half_plane = spacing * size / 2
    if plane == "Sagittal":
        x = center_x + step * (n - 1) / 2  # the normal is -x, so the first slice lies on the +x side of the centre
    else:
        x = center_x - half_plane  # rows run along +x, so the first pixel lies half a field of view on the -x side
    return {
        "StudyInstanceUID": study,
        "SeriesInstanceUID": series,
        "ImageOrientationPatient": IOP[plane],
        "ImagePositionPatient": f"{x}|-50|80",
        "PixelSpacing_0": spacing,
        "PixelSpacing_1": spacing,
        "Rows": size,
        "Columns": size,
        "n_files": n,
        "SpacingBetweenSlices": step,
        "SliceThickness": step,
    }


def _meta() -> pd.DataFrame:
    rows = []
    # Bilateral study b: 3 series on the right knee (x ~ -105) and 3 on the left knee (x ~ +115)
    for i, (plane, x) in enumerate([("Sagittal", -108), ("Coronal", -104), ("Axial", -106), ("Sagittal", 112), ("Coronal", 118), ("Axial", 115)]):
        rows.append(_row("b", f"b.{i}", plane, x))
    # Single-knee study s: centres drift a little (40 mm spread)
    for i, (plane, x) in enumerate([("Sagittal", 10), ("Coronal", 30), ("Axial", 50)]):
        rows.append(_row("s", f"s.{i}", plane, x))
    # Bilateral study u with one left-knee series: that series is unused and only the right knee becomes a child
    for i, (plane, x) in enumerate([("Sagittal", -100), ("Coronal", -98), ("Sagittal", 120)]):
        rows.append(_row("u", f"u.{i}", plane, x))
    return pd.DataFrame(rows)


def test_series_center_x_follows_the_geometry() -> None:
    meta = _meta()
    assert np.allclose(series_center_x(meta).to_numpy(), [-108, -104, -106, 112, 118, 115, 10, 30, 50, -100, -98, 120])
    broken = meta.head(1).assign(ImagePositionPatient=None)
    assert np.isnan(series_center_x(broken).iloc[0])


def test_split_finds_bilateral_studies_and_assigns_knees() -> None:
    table = split_bilateral_studies(_meta())
    assert set(table["StudyInstanceUID"]) == {"b", "u"}  # single-knee studies are not listed
    knees = dict(zip(table["SeriesInstanceUID"], zip(table["knee"], table["child_study_id"])))
    assert knees["b.0"] == knees["b.1"] == knees["b.2"] == ("right", "b-1")
    assert knees["b.3"] == knees["b.4"] == knees["b.5"] == ("left", "b-2")
    assert knees["u.0"] == knees["u.1"] == ("right", "u-1")
    assert knees["u.2"][0] == "unused" and pd.isna(knees["u.2"][1])
    # A threshold above the spread finds no bilateral study
    assert split_bilateral_studies(_meta(), min_spread_mm=300.0).empty


def test_split_stops_when_a_bilateral_series_has_no_position() -> None:
    meta = _meta()
    meta.loc[meta["SeriesInstanceUID"] == "b.4", "ImagePositionPatient"] = None
    with pytest.raises(ValueError, match="unknown position"):
        split_bilateral_studies(meta)
    meta = _meta()
    meta.loc[meta["SeriesInstanceUID"] == "s.1", "ImagePositionPatient"] = None  # a single-knee study does not stop the split
    assert set(split_bilateral_studies(meta)["StudyInstanceUID"]) == {"b", "u"}


def test_cli_writes_a_table_that_training_reads_and_compares_it(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    _meta().to_csv(cache / "series_meta.csv", index=False)
    out = tmp_path / "knee_series.csv"
    assert invoke(split_cli, ["--cache", cache, "--out", out]).exit_code == 0
    assert load_subcases(out) == {"b-1": ("b", frozenset({"b.0", "b.1", "b.2"})), "b-2": ("b", frozenset({"b.3", "b.4", "b.5"})), "u-1": ("u", frozenset({"u.0", "u.1"}))}
    assert invoke(split_cli, ["--cache", cache, "--out", tmp_path / "again.csv", "--compare", out]).exit_code == 0
    assert invoke(split_cli, ["--cache", cache, "--out", out, "--compare", out]).exit_code == 2  # refuses to compare a table with itself
    wrong = pd.read_csv(out)
    wrong.loc[wrong["SeriesInstanceUID"] == "b.0", "knee"] = "left"
    wrong.to_csv(tmp_path / "wrong.csv", index=False)
    assert invoke(split_cli, ["--cache", cache, "--out", tmp_path / "again2.csv", "--compare", tmp_path / "wrong.csv"]).exit_code == 1
