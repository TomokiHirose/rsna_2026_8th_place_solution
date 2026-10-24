from __future__ import annotations

from typing import Any

import timm
import torch
import torch.nn as nn

from rsna_knee.config import ModelConfig
from rsna_knee.constants import NUM_LABELS
from rsna_knee.models.spatial_branch import SpatialFindingPool
from rsna_knee.models.window_model import DEFAULT_ENCODE_CHUNK, pixel_stats

__all__ = ["BranchHead", "MultiBranchModel"]

#: multibranch の 3 head の名前（枝 0・枝 1・fusion の順）。log と OOF の列名に使う。
BRANCH_NAMES = ("b0", "b1", "fused")


class BranchHead(nn.Module):
    """GAP の窓特徴 ``[B, K, gap_dim]`` と SAP の所見別窓特徴 ``[B, K, 12, sap_dim]`` を、独立な 2 本の所見別 attention-MIL で集約し、連結を所見別の線形で読んで logits ``[B, 12]`` にする。"""

    def __init__(self, gap_dim: int, sap_dim: int, att_hidden: int, drop: float) -> None:
        super().__init__()
        self.gap_norm = nn.LayerNorm(gap_dim)
        self.gap_att = nn.Sequential(nn.Linear(gap_dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, NUM_LABELS))
        self.sap_norm = nn.LayerNorm(sap_dim)
        # SAP 側: 所見で共有する tanh 層と、所見ごとの読み出しベクトル。
        self.sap_att = nn.Sequential(nn.Linear(sap_dim, att_hidden), nn.Tanh(), nn.Dropout(drop))
        self.sap_w = nn.Parameter(nn.init.trunc_normal_(torch.zeros(NUM_LABELS, att_hidden), std=0.02))
        pooled_dim = gap_dim + sap_dim
        self.clsW = nn.Parameter(nn.init.trunc_normal_(torch.zeros(NUM_LABELS, pooled_dim), std=0.02))
        self.clsb = nn.Parameter(torch.zeros(NUM_LABELS))

    def forward(self, gap: torch.Tensor, sap: torch.Tensor, window_mask: torch.Tensor | None) -> torch.Tensor:
        h = self.gap_norm(gap)
        scores = self.gap_att(h)
        sp = self.sap_norm(sap)
        sp_scores = (self.sap_att(sp) * self.sap_w).sum(-1)
        if window_mask is not None:
            scores = scores.masked_fill(~window_mask[..., None], -torch.inf)
            sp_scores = sp_scores.masked_fill(~window_mask[..., None], -torch.inf)
        # 所見ごとに実在する窓だけで正規化する。
        pooled = torch.cat([torch.einsum("bkn,bkf->bnf", scores.softmax(dim=1), h), torch.einsum("bkn,bknf->bnf", sp_scores.softmax(dim=1), sp)], dim=-1)
        return (pooled * self.clsW).sum(-1) + self.clsb


class MultiBranchModel(nn.Module):
    """2 encoder の GAP / SAP 窓特徴を、枝 0・枝 1・fusion の 3 組の :class:`BranchHead` で読み、sigmoid 確率を ``blend`` で混ぜる。padding は encoder と集約から除く。"""

    def __init__(self, cfg: ModelConfig, pretrained: bool | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        window, mb = cfg.window, cfg.multibranch
        mean, std = pixel_stats(window.norm, window.context)
        weights = cfg.pretrained if pretrained is None else pretrained
        self.backbones = nn.ModuleList(timm.create_model(name, pretrained=weights, num_classes=0, in_chans=window.context, global_pool="avg") for name in mb.backbones)
        self.feat_dims: list[int] = [backbone.num_features for backbone in self.backbones]
        self.register_buffer("pixel_mean", mean)
        self.register_buffer("pixel_std", std)
        self.spatials = nn.ModuleList(SpatialFindingPool(dim, mb.spatial_dim) for dim in self.feat_dims)
        c0, c1 = self.feat_dims
        self.heads = nn.ModuleList(BranchHead(dim, dim, mb.att_hidden, window.drop) for dim in (c0, c1))
        d = mb.fusion_dim
        # fusion_dim があれば、fusion の入力を窓単位で非線形に D 次元へ射影する。
        self.fusion_gap_proj = None if d is None else nn.Sequential(nn.LayerNorm(c0 + c1), nn.Linear(c0 + c1, d), nn.GELU(), nn.Dropout(window.drop))
        self.fusion_sap_proj = None if d is None else nn.Sequential(nn.LayerNorm(c0 + c1), nn.Linear(c0 + c1, d), nn.GELU(), nn.Dropout(window.drop))
        fusion_in = c0 + c1 if d is None else d
        self.heads.append(BranchHead(fusion_in, fusion_in, mb.att_hidden, window.drop))
        self.register_buffer("blend", torch.tensor(mb.blend, dtype=torch.float32), persistent=False)
        if cfg.freeze_encoder:
            for p in self.backbones.parameters():
                p.requires_grad = False
        # 以下は :class:`WindowModel` と同じ意味の推論設定。
        self.encode_chunk: int | None = DEFAULT_ENCODE_CHUNK
        self.infer_dtype: torch.dtype | None = None
        self.series_layout: str = "slots"
        self.tta_context_flip: bool = False

    def set_grad_checkpointing(self, enable: bool = True) -> bool:
        """両 backbone に活性の再計算を掛ける。両方が対応していれば True。"""
        try:
            for backbone in self.backbones:
                backbone.set_grad_checkpointing(enable)
        except (AttributeError, NotImplementedError, TypeError):
            return False
        return True

    def optimizer_groups(self, encoder_lr: float | tuple[float, float], head_lr: float) -> list[dict[str, Any]]:
        """枝 0・枝 1 の backbone を ``encoder_lr``（1 つなら両方、2 つなら枝ごと）、それ以外を ``head_lr`` に置く AdamW の param group。"""
        lrs = (encoder_lr, encoder_lr) if isinstance(encoder_lr, float) else encoder_lr
        backbone_ids = {id(p) for p in self.backbones.parameters()}
        groups = [{"params": [p for p in backbone.parameters() if p.requires_grad], "lr": lr} for backbone, lr in zip(self.backbones, lrs)]
        head = [p for p in self.parameters() if id(p) not in backbone_ids and p.requires_grad]
        return [group for group in groups if group["params"]] + [{"params": head, "lr": head_lr}]

    def encode(self, windows: torch.Tensor, window_mask: torch.Tensor | None, window_slot: torch.Tensor) -> torch.Tensor:
        """``[B, K, C, R, R]`` -> 窓特徴 ``[B, K, D]``。列は ``g_0 (C_0) ‖ g_1 (C_1) ‖ s_0 (12 * C_0) ‖ s_1 (12 * C_1)``、``s_e`` は所見ごとに ``C_e`` 列ずつ並ぶ。padding 窓は 0。"""
        context = self.cfg.window.context
        if windows.ndim != 5 or windows.shape[2] != context:
            raise ValueError(f"expected windows [B, K, {context}, R, R], got {tuple(windows.shape)}")
        b, k = windows.shape[:2]
        flat, slots = windows.flatten(0, 1), window_slot.flatten()
        if window_mask is not None:
            if window_mask.dtype != torch.bool or window_mask.shape != (b, k) or not window_mask.any(dim=1).all():
                raise ValueError("window_mask must be boolean [B, K] with a real window in every bag")
            flat, slots = flat[window_mask.flatten()], slots[window_mask.flatten()]
        flat = (flat - self.pixel_mean) / self.pixel_std
        if self.infer_dtype is not None and not self.training and flat.is_cuda:
            with torch.autocast("cuda", dtype=self.infer_dtype):
                feats = self._encode_flat(flat, slots)
            feats = feats.float()  # ヘッドは fp32。autocast の外で戻す
        else:
            feats = self._encode_flat(flat, slots)
        if window_mask is not None:
            padded = feats.new_zeros((b * k, feats.shape[-1]))
            padded[window_mask.flatten()] = feats
            feats = padded
        return feats.view(b, k, -1)

    def _encode_flat(self, flat: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        """正規化済みの ``[N, C, R, R]`` -> ``[N, D]``。学習中は ``window.train_chunk`` 枚ずつ、それ以外は ``encode_chunk`` 枚ずつ通す。"""
        chunk = self.cfg.window.train_chunk if self.training else self.encode_chunk
        parts = zip(flat.split(chunk), slots.split(chunk)) if chunk and flat.shape[0] > chunk else [(flat, slots)]
        return torch.cat([self._encode_part(part, part_slots) for part, part_slots in parts], dim=0)

    def _encode_part(self, part: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        """各 backbone を 1 度だけ通し、最終特徴マップから GAP と SAP（所見 x Slot の query の位置平均を backbone の head に通したもの）を作る。"""
        n = part.shape[0]
        gaps, saps = [], []
        for backbone, spatial in zip(self.backbones, self.spatials):
            maps = backbone.forward_features(part)
            pooled = spatial(maps, slots)
            f = pooled.shape[-1]
            gaps.append(backbone.forward_head(maps))
            saps.append(backbone.forward_head(pooled.reshape(n * NUM_LABELS, f, 1, 1)).view(n, -1))
        return torch.cat(gaps + saps, dim=1)

    def heads_from_features(self, feats: torch.Tensor, window_mask: torch.Tensor | None) -> torch.Tensor:
        """``[B, K, D]`` の窓特徴 -> 3 head の logits ``[3, B, 12]``（枝 0・枝 1・fusion）。fusion は同じ窓の GAP と、同じ窓・同じ所見の SAP を channel 方向に連結して読む。"""
        c0, c1 = self.feat_dims
        g0, g1, s0, s1 = feats.split([c0, c1, NUM_LABELS * c0, NUM_LABELS * c1], dim=-1)
        s0, s1 = s0.unflatten(-1, (NUM_LABELS, c0)), s1.unflatten(-1, (NUM_LABELS, c1))
        logits = [self.heads[0](g0, s0, window_mask), self.heads[1](g1, s1, window_mask)]
        gap, sap = torch.cat([g0, g1], dim=-1), torch.cat([s0, s1], dim=-1)
        if self.fusion_gap_proj is not None:
            gap, sap = self.fusion_gap_proj(gap), self.fusion_sap_proj(sap)
        logits.append(self.heads[2](gap, sap, window_mask))
        return torch.stack(logits)

    def forward_branches(self, windows: torch.Tensor, window_mask: torch.Tensor | None, window_slot: torch.Tensor) -> torch.Tensor:
        """Window Bag の batch ``[B, K, C, R, R]`` -> 3 head の logits ``[3, B, 12]``。"""
        return self.heads_from_features(self.encode(windows, window_mask, window_slot), window_mask)

    def mix(self, logits_all: torch.Tensor) -> torch.Tensor:
        """3 head の logits ``[3, B, 12]`` -> ``blend`` で混ぜた sigmoid 確率の logit ``[B, 12]``。"""
        probs = torch.einsum("h,hbn->bn", self.blend, torch.sigmoid(logits_all.float()))
        return torch.logit(probs, eps=1e-6)

    def forward(self, windows: torch.Tensor, window_mask: torch.Tensor | None = None, window_slot: torch.Tensor | None = None, window_pos: torch.Tensor | None = None, window_rows: torch.Tensor | None = None) -> torch.Tensor:
        """:class:`WindowModel` と同じ署名で混合後の logits ``[B, 12]`` を返す。``window_pos`` / ``window_rows`` は読まない。"""
        if window_slot is None:
            raise ValueError("model.kind=multibranch needs window_slot")
        return self.mix(self.forward_branches(windows, window_mask, window_slot))

    def forward_batch(self, batch: dict[str, Any]) -> torch.Tensor:
        """dataset の batch に対する混合後の logits。"""
        return self(batch["windows"], batch.get("window_mask"), batch.get("window_slot"))

    def forward_batch_branches(self, batch: dict[str, Any]) -> torch.Tensor:
        """dataset の batch に対する 3 head の logits ``[3, B, 12]``。"""
        return self.forward_branches(batch["windows"], batch.get("window_mask"), batch["window_slot"])
