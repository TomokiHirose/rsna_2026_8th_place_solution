from pathlib import Path
from typing import Any

import torch
from torch import nn

from rsna_knee.config import ModelConfig
from rsna_knee.constants import NUM_LABELS


class VlmMilModel(nn.Module):
    """Qwen の動画トークンを所見別 attention-MIL で集約する。"""

    def __init__(self, cfg: ModelConfig, pretrained: bool = True) -> None:
        super().__init__()
        from peft import LoraConfig, inject_adapter_in_model
        from transformers import AutoProcessor, Qwen3_5Config, Qwen3_5ForConditionalGeneration, Qwen3_5Model

        self.cfg = cfg
        self.series_layout = "slots"
        source = cfg.vlm.source
        self.processor = AutoProcessor.from_pretrained(source, local_files_only=True)
        if pretrained:
            self.backbone = Qwen3_5ForConditionalGeneration.from_pretrained(source, local_files_only=True, dtype=torch.bfloat16).model
        else:
            config = Qwen3_5Config.from_pretrained(source, local_files_only=True)
            self.backbone = Qwen3_5Model._from_config(config, dtype=torch.bfloat16)
        self.backbone.requires_grad_(False)
        targets = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj", "gate_proj", "up_proj", "down_proj"]
        adapter = LoraConfig(r=cfg.vlm.lora_rank, lora_alpha=cfg.vlm.lora_alpha, lora_dropout=cfg.vlm.lora_dropout, target_modules=targets, bias="none")
        inject_adapter_in_model(adapter, self.backbone.language_model)
        for parameter in self.backbone.language_model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        # 小さい学習率でも重みを更新できるよう、学習する視覚重みは fp32 で保持する。
        self.backbone.visual.float().requires_grad_(True)
        self.backbone.config.use_cache = False
        dim = self.backbone.config.text_config.hidden_size
        self.norm = nn.LayerNorm(dim)
        self.att = nn.Sequential(nn.Linear(dim, 256), nn.Tanh(), nn.Dropout(cfg.window.drop), nn.Linear(256, NUM_LABELS))
        self.clsW = nn.Parameter(torch.empty(NUM_LABELS, dim))
        self.clsb = nn.Parameter(torch.zeros(NUM_LABELS))
        nn.init.trunc_normal_(self.clsW, std=0.02)
        self.train()

    def set_grad_checkpointing(self, enable: bool = True) -> bool:
        """視覚と言語の両方で活性の再計算を有効にする。"""
        if enable:
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            self.backbone.gradient_checkpointing_disable()
        return True

    def forward_batch(self, batch: dict[str, Any]) -> torch.Tensor:
        """動画位置だけを所見別に集約し、12 logits を返す。"""
        keys = ("input_ids", "attention_mask", "pixel_values_videos", "video_grid_thw", "mm_token_type_ids")
        inputs = {key: batch[key] for key in keys}
        with torch.autocast(device_type=inputs["input_ids"].device.type, dtype=torch.bfloat16):
            hidden = self.backbone(**inputs, use_cache=False, return_dict=True).last_hidden_state
        video_mask = (inputs["input_ids"] == self.backbone.config.video_token_id) & inputs["attention_mask"].bool()
        features = [row[keep].float() for row, keep in zip(hidden, video_mask)]
        counts = torch.tensor([len(row) for row in features], device=hidden.device)
        if (counts == 0).any():
            raise ValueError("every study needs video tokens")
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            h = self.norm(nn.utils.rnn.pad_sequence(features, batch_first=True))
            mask = torch.arange(h.shape[1], device=h.device)[None] < counts[:, None]
            scores = self.att(h).masked_fill(~mask[..., None], -torch.inf)
            pooled = torch.einsum("bnc,bnd->bcd", scores.softmax(dim=1), h)
            return (pooled * self.clsW).sum(-1) + self.clsb

    def optimizer_groups(self, encoder_lr: float, head_lr: float) -> list[dict[str, Any]]:
        """視覚、merger、言語 LoRA、MIL の学習率を分ける。"""
        merger = list(self.backbone.visual.merger.parameters())
        merger_ids = {id(p) for p in merger}
        vision = [p for p in self.backbone.visual.parameters() if id(p) not in merger_ids and p.requires_grad]
        language = [p for p in self.backbone.language_model.parameters() if p.requires_grad]
        head = [p for name, p in self.named_parameters() if not name.startswith("backbone.") and p.requires_grad]
        return [{"params": vision, "lr": encoder_lr}, {"params": merger, "lr": self.cfg.vlm.merger_lr}, {"params": language, "lr": self.cfg.vlm.lora_lr}, {"params": head, "lr": head_lr}]

    def save_assets(self, directory: Path) -> None:
        """全重みと組み合わせてローカル復元する config と processor を保存する。"""
        self.backbone.config.save_pretrained(directory)
        self.processor.save_pretrained(directory)
