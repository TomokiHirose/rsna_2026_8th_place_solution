import json
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import torch

from rsna_knee.config.config import ModelConfig
from rsna_knee.constants import ID_COLUMN, LABELS, NUM_LABELS
from tools import build_cache

from .conftest import invoke
from .synthetic_study import make_study


class _FakeProcessor:
    """Stands in for the Qwen processor: records the chat text and returns the frame count as a tensor."""

    def __init__(self) -> None:
        self.tokenizer = mock.MagicMock()
        self.tokenizer.apply_chat_template.side_effect = lambda messages, **_: json.dumps(messages)

    def __call__(self, text: list[str], videos: list[torch.Tensor], **_: object) -> dict:
        return {"input_ids": torch.tensor([[len(text[0])]]), "frames": torch.tensor([sum(len(video) for video in videos)]), "note": "dropped"}


VLM_RECIPE = {"crop_mm": 140.0, "size": 16, "span": (0.0, 1.0), "percentiles": (2.0, 98.0), "slot_table": [3, 3, 3, 3, 3], "native_spacing": True, "resample": "exact", "long_series_sampling": "uniform"}


@pytest.fixture(scope="module")
def slots_cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A slots-layout cache built from two synthetic DICOM studies with the VLM recipe (full span, native, uniform, exact)."""
    root = tmp_path_factory.mktemp("vlm")
    specs = [{"plane": "Sagittal", "fluid": 1, "n_slices": 5}, {"plane": "Coronal", "fluid": 0, "n_slices": 3}, {"plane": "Axial", "fluid": 1, "n_slices": 2}]
    rows = make_study(root / "comp" / "train_series", "1.2.3.1", specs, seed=0) + make_study(root / "comp" / "train_series", "1.2.3.2", specs[:2], seed=1)
    pd.DataFrame(rows).to_csv(root / "comp" / "train_series.csv", index=False)
    pd.DataFrame({ID_COLUMN: ["1.2.3.1", "1.2.3.2"]}).to_csv(root / "comp" / "train.csv", index=False)
    args = [
        "--data-dir",
        root / "comp",
        "--split",
        "train",
        "--out",
        root / "cache",
        "--slots",
        "native",
        "--caps",
        "3,3,3,3,3",
        "--span",
        "0,1",
        "--size",
        16,
        "--resample",
        "exact",
        "--long-series-sampling",
        "uniform",
        "--memmap",
        "--code-commit",
        "pytest",
    ]
    result = invoke(build_cache.main, args)
    assert result.exit_code == 0, result.output
    return root / "cache"


def test_video_dataset_turns_the_cache_into_processor_inputs(slots_cache: Path) -> None:
    """VideoStudyDataset + video_inputs: every real slice of every assigned series becomes one video frame."""
    import transformers

    from rsna_knee.data.video import metadata_value
    from rsna_knee.modules.video_dataset import VideoStudyDataset

    cfg = ModelConfig(kind="vlm", backbone="resnet18", pretrained=False, img_size=16, vlm={"source": "unused", "prompt": "Grade the knee."}, preprocess=VLM_RECIPE)
    labels = pd.DataFrame({name: [0.0, 1.0] for name in LABELS}, index=["1.2.3.1", "1.2.3.2"])
    dataset = VideoStudyDataset(slots_cache, ["1.2.3.1", "1.2.3.2"], labels=labels, weights=np.ones((2, NUM_LABELS)), img_size=16, all_windows=True, model=cfg)
    with mock.patch.object(transformers.AutoProcessor, "from_pretrained", return_value=_FakeProcessor()):
        items = [dataset[i] for i in range(len(dataset))]
    masks = np.load(slots_cache / "masks.npy")
    for item, mask in zip(items, masks):
        assert int(item["frames"]) == int((mask != 0).sum()) and "note" not in item
        assert item["label"].shape == (NUM_LABELS,) and item["weight"].shape == (NUM_LABELS,)
    assert metadata_value({"EchoTime": "12.50"}, "EchoTime") == "12.5" and metadata_value({}, "SliceThickness") == "unknown"
