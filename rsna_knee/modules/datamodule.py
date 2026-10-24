from typing import Any

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

from rsna_knee.config import Config

#: batch の中で画素を持つキー。左右反転 TTA はこの tensor の最後の軸を反転する。
PIXEL_KEYS = ("windows",)
#: Bag の長さで並ぶキー。collate が Study ごとの窓数の違いを padding する。``window_u`` / ``window_u_sym``（窓の深さ）は深さ prior つきの学習 split と union bag が、``window_series``（窓の series 番号）は union bag だけが持つ。
BAG_KEYS = ("windows", "window_slot", "window_pos", "window_rows", "window_u", "window_u_sym", "window_series")
#: モデルに渡さないキー（target、識別子、損失の重み、重症度、窓ターゲットを引く左右対称化の印）。窓の深さ ``window_u`` は ``model.series_neck`` が読むので渡す。
TARGET_KEYS = ("label", "study_id", "weight", "severity", "window_u_sym")


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """可変長 Window と各窓の Slot 番号・中心スライスの行番号・各チャネルの行番号を padding して実在窓の mask を添え、ラベル・重み・重症度は Study 方向に積む。"""
    if "input_ids" in batch[0]:
        sequences = {"input_ids", "attention_mask", "mm_token_type_ids"}
        visual = {"pixel_values_videos", "video_grid_thw"}
        result = {}
        for key in batch[0]:
            values = [item[key] for item in batch]
            if key in sequences:
                result[key] = torch.nn.utils.rnn.pad_sequence([value.squeeze(0) for value in values], batch_first=True, padding_value=0)
            elif key in visual:
                result[key] = torch.cat(values)
            else:
                result[key] = torch.stack(values) if torch.is_tensor(values[0]) else values
        return result
    counts = [len(item["windows"]) for item in batch]
    if not counts or min(counts) < 1:
        raise ValueError("empty window bag")
    template = batch[0]
    result = {key: torch.stack([item[key] for item in batch]) if torch.is_tensor(template[key]) else [item[key] for item in batch] for key in template if key not in BAG_KEYS}
    for key in (key for key in BAG_KEYS if key in template):
        result[key] = torch.nn.utils.rnn.pad_sequence([item[key] for item in batch], batch_first=True)
    result["window_mask"] = torch.arange(max(counts))[None, :] < torch.tensor(counts)[:, None]
    return result


def model_inputs(batch: dict[str, Any], device: str | torch.device, hflip: bool = False) -> dict[str, torch.Tensor]:
    """batch のうちモデルが読む tensor（:data:`TARGET_KEYS` 以外）を ``device`` に送る。

    ``hflip`` なら画素の tensor（:data:`PIXEL_KEYS`）の最後の軸（左右）を反転する。学習ループは Lightning が
    batch ごと device に移すので使わず、``rsna_knee.training._window_oof`` が使う。
    """
    inputs: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if key in TARGET_KEYS:
            continue
        value = value.to(device)
        inputs[key] = value.flip(-1) if hflip and key in PIXEL_KEYS else value
    return inputs


class KneeDataModule(L.LightningDataModule):
    """組み立て済みの train / val データセットを ``cfg.data`` に従う DataLoader で包む。

    データセットはここで組まずに注入する。どの split がどの Study を供給するかは ``build_datasets`` が決める。
    """

    def __init__(self, cfg: Config, train_dataset: Dataset, val_dataset: Dataset) -> None:
        super().__init__()
        self.cfg = cfg
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def _loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset, batch_size=self.cfg.data.batch_size, shuffle=shuffle, num_workers=self.cfg.data.num_workers, pin_memory=self.cfg.data.pin_memory, collate_fn=collate_batch, drop_last=False, persistent_workers=self.cfg.data.num_workers > 0
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, shuffle=False)
