import json
from pathlib import Path

import numpy as np
import pandas as pd

from rsna_knee.config.config import DepthPriorConfig
from rsna_knee.data.slots import SLOT_NAMES

EDGE = 8
TABLE = [4, 4, 3, 3, 4]
# Each plane is capped at the slice count of its first Slot
CAP = {"Sagittal": 4, "Coronal": 3, "Axial": 4}
# (plane, fluid, Presence). Pixel j of series i is ``10 * i + j + 1``, so a window's pixels tell its series
SERIES = [("Sagittal", 1, [1, 1, 1]), ("Sagittal", 1, [1, 1, 1, 1]), ("Sagittal", 0, [1, 1]), ("Coronal", 1, [1, 1, 1]), ("Coronal", 0, [1, 0, 1]), ("Axial", -1, [1, 1, 1, 1]), ("Axial", -1, [1, 1])]
# Series indices of each study in CSV order
STUDIES = {"s0": [0, 1, 2, 3, 4, 5, 6], "s1": [5, 3, 0]}
MANIFEST = {"slots": str(sum(TABLE)), "slot_table": TABLE, "span": [0.02, 0.98], "size": EDGE, "crop_mm": 140.0, "long_series_sampling": "contiguous", "percentiles": [2.0, 98.0]}


def _slice(series: int, j: int) -> np.ndarray:
    return np.full((EDGE, EDGE), 10 * series + j + 1, dtype=np.uint8)


def write_series_cache(out: Path, studies: dict[str, list[int]] = STUDIES) -> Path:
    """Write a series-layout cache; each unit takes ``cap`` slices plus one absent separator slice."""
    out.mkdir(parents=True)
    vols, masks, rows, offsets, stack_offsets = [], [], [], [0], [0]
    for uid, members in studies.items():
        position = 0
        for unit, i in enumerate(members):
            plane, fluid, present = SERIES[i]
            cap = CAP[plane]
            mask = np.zeros(cap + 1, dtype=np.uint8)
            mask[: len(present)] = present
            vols += [_slice(i, j) for j in np.flatnonzero(mask)]
            masks.append(mask)
            rows.append({"StudyInstanceUID": uid, "SeriesInstanceUID": f"{uid}.{i}", "unit": unit, "plane": plane, "fluid": fluid, "stack_offset": position, "cap": cap, "n_present": int(mask.sum())})
            position += cap + 1
        offsets.append(len(vols))
        stack_offsets.append(stack_offsets[-1] + position)
    np.save(out / "vols.npy", np.stack(vols))
    np.save(out / "masks.npy", np.concatenate(masks))
    np.save(out / "positions.npy", np.zeros(stack_offsets[-1], dtype=np.float32))
    np.save(out / "offsets.npy", np.array(offsets, dtype=np.int64))
    np.save(out / "stack_offsets.npy", np.array(stack_offsets, dtype=np.int64))
    np.save(out / "ids.npy", np.array(list(studies), dtype="<U64"))
    pd.DataFrame(rows).to_csv(out / "series.csv", index=False)
    (out / "manifest.json").write_text(json.dumps({**MANIFEST, "series_layout": "all"}))
    return out


def write_depth_prior(cache: Path, prior_json: Path, studies: dict[str, list[int]] = STUDIES) -> DepthPriorConfig:
    """Write the cache's ``series_meta.csv`` and a prior with one ACL component per Slot.

    Series 1 samples 4 of 20 slices (19 in span); every other series holds its whole span.
    """
    iop = {"Sagittal": "0|1|0|0|0|-1", "Coronal": "1|0|0|0|0|-1", "Axial": "1|0|0|0|1|0"}
    selected = {i: [0, 6, 12, 18] if i == 1 else list(range(len(SERIES[i][2]))) for i in range(len(SERIES))}
    n_files = {i: 20 if i == 1 else len(SERIES[i][2]) + 1 for i in range(len(SERIES))}
    meta = [
        {"StudyInstanceUID": uid, "SeriesInstanceUID": f"{uid}.{i}", "Laterality": "L" if uid == "s0" else "", "ImageOrientationPatient": iop[SERIES[i][0]], "n_files": n_files[i], "selected_indices": json.dumps(selected[i])}
        for uid, members in studies.items()
        for i in members
    ]
    pd.DataFrame(meta).to_csv(cache / "series_meta.csv", index=False)
    slots = {name: {"components": [{"finding": "ACL", "weight": 1.0, "mean": 0.5, "sigma": 0.1}], "findings": {}} for name in SLOT_NAMES}
    prior_json.write_text(json.dumps({"version": 1, "truncate": [-1.0, 1.0], "sigma_floor": 0.08, "attention": "gap", "slots": slots}))
    return DepthPriorConfig(path=prior_json, alpha=1.0)
