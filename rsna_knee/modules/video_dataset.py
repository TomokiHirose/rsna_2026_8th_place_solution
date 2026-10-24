import random

import numpy as np
import pandas as pd
import torch

from rsna_knee.config import ModelConfig
from rsna_knee.constants import ID_COLUMN
from rsna_knee.data.video import video_inputs
from rsna_knee.modules.window_dataset import WindowStudyDataset


class VideoStudyDataset(WindowStudyDataset):
    """既存の Study cache を系列動画とメタデータとして読み込む。"""

    def __init__(self, *args, model: ModelConfig, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        recipe = model.preprocess
        if tuple(recipe.span) != (0.0, 1.0) or recipe.long_series_sampling != "uniform" or not recipe.native_spacing or recipe.resample != "exact":
            raise ValueError("VLM requires a full-span native uniform exact cache")
        if recipe.size != model.img_size:
            raise ValueError("VLM cache size must equal model.img_size")
        aug = self.window_aug
        if aug is not None and max(aug.mirror_prob, aug.affine_prob, aug.lowres_prob, aug.blur_prob, aug.noise_prob) > 0:
            raise ValueError("VLM does not apply window augmentation; set the window aug probabilities to zero")
        self.model_cfg = model
        rows = pd.read_csv(self.cache_dir / "series_meta.csv", keep_default_na=False).to_dict("records")
        self.headers = {}
        for row in rows:
            self.headers.setdefault(str(row[ID_COLUMN]), []).append(row)
        self.processor = None

    def __getitem__(self, idx: int) -> dict:
        from transformers import AutoProcessor

        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(self.model_cfg.vlm.source, local_files_only=True)
        uid = self.study_ids[idx]
        _, row = self.uid_index[uid]
        packed = np.asarray(self._volume()[self.offsets[row] : self.offsets[row + 1]])
        presence = self.masks[row] != 0
        if self.is_train:
            packed = self.augment_stack(packed)
            if self.brightness:
                gain = 1 + random.uniform(-self.brightness, self.brightness)
                packed = np.clip(packed.astype(np.float32) * gain, 0, 255).round().astype(np.uint8)
        # 詰めた実在スライスを Stack 上の位置へ戻す。欠損位置は 0 のまま。
        volume = np.zeros((len(presence), *packed.shape[1:]), dtype=np.uint8)
        volume[presence] = packed
        item = video_inputs(self.processor, self.model_cfg, volume, presence, self.headers[uid])
        item.update(label=torch.from_numpy(self.label_array[idx].copy()), study_id=uid)
        if self.weight_array is not None:
            item["weight"] = torch.from_numpy(self.weight_array[idx].copy())
        return item
