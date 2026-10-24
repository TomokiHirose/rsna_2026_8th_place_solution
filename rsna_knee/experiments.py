from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rsna_knee.config import ModelConfig

INFERENCE_FILENAME = "inference.json"


class ExperimentConfig(BaseModel):
    """experiment 内の全 fold が共有するモデルと推論実行の設定。"""

    model_config = ConfigDict(extra="forbid")

    model: ModelConfig
    encode_chunk: int = Field(default=32, gt=0)
    infer_dtype: Literal["float16", "float32"] = "float16"
    # "slots" は学習と同じ 5 Slot、"all" は推論時だけ全 series を撮像面ごとの枚数上限で並べる。
    series_layout: Literal["slots", "all"] = "slots"
    # True なら Window 内のスライス順を前後反転した入力の確率と平均する。
    tta_context_flip: bool = False
    # series の TTA の推論回数の上限。2 以上なら、Slot ごとに候補の series（学習の seriesaug と同じ選び方の k 番目）と、
    # その series の sub-series（元の 1 本 + m 本）を回して採点し平均する（:func:`rsna_knee.inference.series_tta_stacks`）。1 で無効。
    tta_series: int = Field(default=1, ge=1)
    # None でなければ、``series_layout="all"`` + sub-series のデコードで全 series × 全版の窓を 1 つの Bag にし、この窓数で Slot 比を保って頭打ちにして採点する
    # （:func:`rsna_knee.inference.predict_study`）。None なら従来の推論。
    union_cap: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _check_union(self) -> "ExperimentConfig":
        if self.union_cap is not None and (self.tta_series > 1 or self.tta_context_flip):
            raise ValueError("union_cap cannot be combined with tta_series or tta_context_flip")
        return self


def load_experiment_config(experiment_dir: Path) -> ExperimentConfig:
    """experiment の必須推論設定を読む。"""
    return ExperimentConfig.model_validate_json((experiment_dir / INFERENCE_FILENAME).read_text())


def export_experiment_config(experiment_dir: Path, model_config: ModelConfig) -> ExperimentConfig:
    """共有推論設定を初回だけ書き、既存設定のモデルが一致しなければ拒む。"""
    experiment_dir.mkdir(parents=True, exist_ok=True)
    destination = experiment_dir / INFERENCE_FILENAME
    if not destination.exists():
        config = ExperimentConfig(model=model_config)
        # 完成した JSON を上書きせず公開し、並列 fold の書き出しでも設定を共有する。
        with NamedTemporaryFile(mode="w", dir=experiment_dir, prefix=".inference-", suffix=".json", delete=False) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(config.model_dump_json(indent=2) + "\n")
        try:
            os.link(temporary_path, destination)
        except FileExistsError:
            pass
        finally:
            temporary_path.unlink()
    existing = load_experiment_config(experiment_dir)
    if existing.model != model_config:
        raise ValueError(f"{destination}: model configuration differs between folds")
    return existing
