from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

import timm
import torch
import torch.nn as nn

from rsna_knee.config import Config, ModelConfig
from rsna_knee.constants import NUM_LABELS
from rsna_knee.data.slots import SLOTS
from rsna_knee.models.context_branch import ContextBranch
from rsna_knee.models.depth_branch import DepthBranch
from rsna_knee.models.fsx import FindingSlotFilm
from rsna_knee.models.pair_branch import PAIR_FINDINGS, PAIRS, PairBranch
from rsna_knee.models.seq_branch import SeqBranch
from rsna_knee.models.series_neck import SeriesNeck
from rsna_knee.models.slot_context_branch import SlotContextBranch
from rsna_knee.models.spatial_branch import SpatialFindingPool

__all__ = ["DEFAULT_ENCODE_CHUNK", "IMAGENET_MEAN", "IMAGENET_STD", "MRI_MEAN", "MRI_STD", "WindowModel", "build_window_model", "compile_backbone", "pixel_stats", "pool_window_logits"]

#: ImageNet の統計量（``norm="imagenet"``）。RGB の 3 組しか無いので context 3 専用。
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#: MRI の統計量（``norm="mri"``）。1 組を全チャネルに掛けるので context は何枚でもよい。
MRI_MEAN = 0.3504
MRI_STD = 0.3045

#: 学習中でないときに backbone へ 1 度に通す Window の数。``None`` なら Bag を一括で通す。
DEFAULT_ENCODE_CHUNK = 16


#: 撮像面の並び（Slot 表の初出順）と、Slot 番号 -> 撮像面の番号。``model.plane_aux`` の補助 head はこの順に並ぶ。
PLANES: list[str] = list(dict.fromkeys(slot.plane for slot in SLOTS))
PLANE_OF_SLOT: list[int] = [PLANES.index(slot.plane) for slot in SLOTS]


class WindowModel(nn.Module):
    """Slot 内の Window Bag を所見別 attention（``window.pool="attention"``）か窓 logit の LSE で集約し、12 所見の logits を返す。padding は encoder と集約から除く。"""

    def __init__(self, cfg: ModelConfig, pretrained: bool | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        window = cfg.window
        # 正規化は重みと一緒に運ぶ。backbone より先に作るのは、"imagenet" を context 3 以外に使う誤りを重みの取得より前に止めるため。
        mean, std = pixel_stats(window.norm, window.context)
        weights = cfg.pretrained if pretrained is None else pretrained
        self.backbone = timm.create_model(cfg.backbone, pretrained=weights, num_classes=0, in_chans=window.context, global_pool="avg")
        feat_dim: int = self.backbone.num_features
        self.feat_dim = feat_dim
        self.register_buffer("pixel_mean", mean)
        self.register_buffer("pixel_std", std)

        if cfg.fsx.enabled and cfg.depth.enabled:
            raise ValueError("model.fsx cannot run with model.depth (fsx does not read the depth features)")
        # 窓の 3 スライスを奥行き方向に混ぜて 1 本の特徴 z にする並列の枝。z は窓特徴に連結して head に渡す。無効なら作らない。
        # 融合では backbone の最後から 2 段目（解像度は最終段の 2 倍）の出力を枝に渡す。
        mid_channels = self.backbone.feature_info[-2]["num_chs"] if cfg.depth.fusion else None
        self.depth = DepthBranch(cfg.depth, mid_channels) if cfg.depth.enabled else None
        self.depth_norm = nn.LayerNorm(cfg.depth.width) if cfg.depth.enabled else None
        # 同一の重みで枝の寄与を測るとき True にし、LayerNorm 後の z をゼロにする。
        self.depth_zero: bool = False
        depth_dim = cfg.depth.width if cfg.depth.enabled else 0
        # GAP 側の窓特徴の幅（backbone の特徴と枝の z の連結）。窓方向の attention はこれを読む。
        self.gap_dim = feat_dim + depth_dim
        # 最終特徴マップを所見 x Slot の query で位置を重み付けして平均する枝。所見ごとの窓特徴を GAP とは別に集約して連結する。無効なら作らない。
        self.spatial = SpatialFindingPool(feat_dim, cfg.spatial.dim) if cfg.spatial.enabled else None
        self.spatial_norm = nn.LayerNorm(feat_dim) if cfg.spatial.enabled else None
        # ``model.series_neck`` は窓特徴より後ろ（窓と series の attention・head・補助 head）を自分で持つので、下の窓方向の attention と head は作らない。
        neck = cfg.series_neck.enabled
        attention = window.pool == "attention" and not neck
        # 枝の窓方向の attention: 所見で共有する tanh 層と、所見ごとの読み出しベクトル。窓 logit を集約する pool では作らない。
        self.spatial_att = nn.Sequential(nn.Linear(feat_dim, cfg.spatial.att_hidden), nn.Tanh(), nn.Dropout(window.drop)) if cfg.spatial.enabled and attention else None
        self.spatial_w = nn.Parameter(nn.init.trunc_normal_(torch.zeros(NUM_LABELS, cfg.spatial.att_hidden), std=0.02)) if cfg.spatial.enabled and attention else None
        # head が読む所見ごとの集約特徴の幅（GAP 側と、枝があればその所見ごとの集約の連結）。
        self.head_dim = self.gap_dim + (feat_dim if cfg.spatial.enabled else 0)
        # 窓間の文脈で条件付けた残差を所見ごとの窓特徴に足す枝。出力層がゼロなので学習開始時は枝なしと同じ。無効なら作らない。
        self.context = ContextBranch(cfg.context, self.gap_dim, self.head_dim) if cfg.context.enabled else None
        # depth 枝の stem 3 段目から窓内の二画像確認スコアを作る枝。スコアは窓特徴の末尾に付けて運び、head には渡さず半月板 2 所見の logits に足す。
        if cfg.pair.enabled and window.context != 3:
            raise ValueError(f"model.pair needs window.context=3 (it scores the 3 slice pairs), got {window.context}")
        self.pair = PairBranch(cfg.pair) if cfg.pair.enabled else None
        # 同一の重みで枝の寄与を測るとき True にし、logits への補正を外す。
        self.pair_zero: bool = False

        self.norm = nn.LayerNorm(feat_dim)
        # 所見別 attention-MIL の窓スコア。窓 logit を集約する pool では作らない。
        self.att = nn.Sequential(nn.Linear(self.gap_dim, 256), nn.Tanh(), nn.Dropout(window.drop), nn.Linear(256, NUM_LABELS)) if attention else None
        self.clsW = None if neck else nn.Parameter(nn.init.trunc_normal_(torch.zeros(NUM_LABELS, self.head_dim), std=0.02))
        self.clsb = None if neck else nn.Parameter(torch.zeros(NUM_LABELS))
        # Slot 内 -> Slot 間の文脈の残差を所見ごとの窓特徴に足し、文脈付きの窓 logit を別の head ``ctxW`` で読む枝。無効なら作らない。
        self.slot_context = SlotContextBranch(cfg.slot_context, self.head_dim) if cfg.slot_context.enabled else None
        self.ctxW = nn.Parameter(nn.init.trunc_normal_(torch.zeros(NUM_LABELS, self.head_dim), std=0.02)) if cfg.slot_context.enabled else None
        self.ctxb = nn.Parameter(torch.zeros(NUM_LABELS)) if cfg.slot_context.enabled else None
        if cfg.severity_head:
            # 同じ attention の pooled 特徴から所見ごとの重症度を当てる、もう 1 組の線形。学習の補助損失だけが読む。
            self.sevW = nn.Parameter(torch.zeros(NUM_LABELS, self.head_dim))
            self.sevb = nn.Parameter(torch.zeros(NUM_LABELS))
            nn.init.trunc_normal_(self.sevW, std=0.02)
        if depth_dim:
            # z 側の列をゼロから始めるので、学習開始時の logits は枝なしと一致する。
            with torch.no_grad():
                for weight in [*([self.att[0].weight] if attention else []), self.clsW, *([self.sevW] if cfg.severity_head else [])]:
                    weight[:, feat_dim : self.gap_dim] = 0
        if cfg.spatial.enabled and not neck:
            # 枝側の読み出しをゼロから始めるので、学習開始時の logits は枝なしと一致する。
            with torch.no_grad():
                for weight in [self.clsW, *([self.ctxW] if cfg.slot_context.enabled else []), *([self.sevW] if cfg.severity_head else [])]:
                    weight[:, self.gap_dim :] = 0
        # 撮像面ごとの補助 head。主と同じ pooled 特徴の形を面ごとの窓だけで作り、面ごとの別の線形で 12 所見を読む。無効なら作らない。
        self.planeW = nn.Parameter(nn.init.trunc_normal_(torch.zeros(len(PLANES), NUM_LABELS, self.head_dim), std=0.02)) if cfg.plane_aux and not neck else None
        self.planeb = nn.Parameter(torch.zeros(len(PLANES), NUM_LABELS)) if cfg.plane_aux and not neck else None
        self.register_buffer("plane_of_slot", torch.tensor(PLANE_OF_SLOT), persistent=False)
        # 所見 query と Slot 特徴の cross-attention で logits に上限付きの差分を足す head。無効なら作らない。
        self.fsx = FindingSlotFilm(cfg.fsx, feat_dim) if cfg.fsx.enabled else None
        # 同じ Slot で隣り合う窓の特徴を混ぜて窓特徴に足す残差枝。無効なら作らない。
        self.seq = SeqBranch(cfg.seq, feat_dim) if cfg.seq.enabled else None
        # 窓 -> series -> study の 2 段の集約（深さの順の双方向 GRU つき）。全 series の Bag の窓の series 番号と深さ u を要る。無効なら作らない。
        self.series_neck = SeriesNeck(cfg.series_neck, feat_dim, cfg.spatial.att_hidden, window.drop, PLANE_OF_SLOT) if neck else None

        if cfg.freeze_encoder:
            if self.depth is not None or self.pair is not None:
                raise ValueError("model.freeze_encoder computes the window features without gradients, so it cannot train model.depth or model.pair")
            for p in [*self.backbone.parameters(), *([] if self.spatial is None else self.spatial.parameters())]:
                p.requires_grad = False

        # 推論時に backbone へ 1 度に通す Window の数。backbone は Window 同士を混ぜないので、Bag を刻んでも変わるのはピークメモリだけ。
        self.encode_chunk: int | None = DEFAULT_ENCODE_CHUNK
        # 推論時に backbone を走らせる dtype。``None`` なら fp32（fitted 時と同じ算術）。``torch.float16`` なら
        # 学習中でなく入力が CUDA 上にあるときだけ autocast の下で走らせる。ヘッドは常に fp32。
        self.infer_dtype: torch.dtype | None = None
        # 推論時に ``self.backbone`` の代わりに窓特徴を計算するもの（:func:`compile_backbone` の産物）。
        self.backbone_runner: Callable[[torch.Tensor], torch.Tensor] | None = None
        # 推論時の Stack の並べ方（:class:`rsna_knee.data.dicom.PreprocessParams` の ``series_layout``）。experiment の推論設定が決める。
        self.series_layout: str = "slots"
        # 推論時に Window 内のスライス順を反転した入力とも採点して平均するか。experiment の推論設定が決める。
        self.tta_context_flip: bool = False
        # series の TTA の推論回数の上限（1 で無効）。experiment の推論設定が決める。
        self.tta_series: int = 1
        # union bag の推論の窓数の上限（None で従来の推論）。experiment の推論設定が決める。
        self.union_cap: int | None = None
        # Lightning は検証の前後に各 module の ``training`` を直接写し戻すので、固定する部分は作った時点から eval にしておく。
        self.train()

    def train(self, mode: bool = True) -> "WindowModel":
        """``freeze_encoder`` のとき、backbone と ``spatial`` は学習中も eval のまま（BatchNorm の統計と dropout を動かさない）。"""
        super().train(mode)
        if self.cfg.freeze_encoder:
            for module in [self.backbone, *([] if self.spatial is None else [self.spatial])]:
                module.eval()
        return self

    def set_grad_checkpointing(self, enable: bool = True) -> bool:
        """バックボーンに活性を保存させず再計算させる。対応しているかどうかを返す。"""
        try:
            self.backbone.set_grad_checkpointing(enable)
        except (AttributeError, NotImplementedError, TypeError):
            return False
        return True

    def encode(self, windows: torch.Tensor, window_mask: torch.Tensor | None = None, window_slot: torch.Tensor | None = None) -> torch.Tensor:
        """``[B, K, C, R, R]`` -> 窓ごとに 1 本の特徴 ``[B, K, D]``。Bag は batch 次元に畳み込む。``model.spatial`` は ``window_slot`` [B, K] を要る。"""
        context = self.cfg.window.context
        if windows.ndim != 5 or windows.shape[2] != context:
            raise ValueError(f"expected windows [B, K, {context}, R, R], got {tuple(windows.shape)}")
        b, k = windows.shape[:2]
        flat = windows.flatten(0, 1)
        if self.spatial is not None and window_slot is None:
            raise ValueError("model.spatial needs window_slot")
        slots = None if window_slot is None else window_slot.flatten()
        if window_mask is not None:
            if window_mask.dtype != torch.bool or window_mask.shape != (b, k) or not window_mask.any(dim=1).all():
                raise ValueError("window_mask must be boolean [B, K] with a real window in every bag")
            flat = flat[window_mask.flatten()]
            slots = None if slots is None else slots[window_mask.flatten()]
        if flat.dtype == torch.uint8:  # dataset は窓を uint8 で運ぶ
            flat = flat.float() / 255.0
        flat = (flat - self.pixel_mean) / self.pixel_std
        # 固定した encoder は勾配を流さないので活性を保存しない。
        with torch.no_grad() if self.cfg.freeze_encoder else contextlib.nullcontext():
            if self.infer_dtype is not None and not self.training and flat.is_cuda and self.backbone_runner is None:
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

    def _encode_flat(self, flat: torch.Tensor, slots: torch.Tensor | None = None) -> torch.Tensor:
        """正規化済みの ``[B * K, C, R, R]`` -> ``[B * K, D]``。学習中は ``window.train_chunk`` 枚ずつ、それ以外は ``encode_chunk`` 枚ずつ通す。

        ``model.depth`` が有効なら ``D`` は backbone の特徴と枝の ``z_raw`` の連結で、``feat_dim + depth.width``。
        ``model.depth.fusion`` なら backbone を ``forward_intermediates`` で 1 度だけ通し、最終出力から窓特徴を、中間特徴を枝に渡す。
        ``model.pair`` が有効なら、さらに窓のペアごとの二画像確認スコア ``3 * 2`` 列が末尾に付く。
        ``model.spatial`` が有効なら、GAP の特徴の後ろに所見ごとの枝の特徴 ``12 * feat_dim`` 列が付く（``slots`` は窓の Slot ``[B * K]``）。
        """
        chunk = self.cfg.window.train_chunk if self.training else self.encode_chunk
        if self.spatial is not None:
            if self.backbone_runner is not None:
                raise ValueError("model.spatial cannot run with a compiled backbone runner (it reads the feature map)")
            parts = zip(flat.split(chunk), slots.split(chunk)) if chunk and flat.shape[0] > chunk else [(flat, slots)]
            return torch.cat([self._encode_spatial(part, part_slots) for part, part_slots in parts], dim=0)
        if self.depth is not None and self.backbone_runner is not None:
            raise ValueError("model.depth cannot run with a compiled backbone runner (it wraps the backbone only)")
        backbone = self.backbone if self.training or self.backbone_runner is None else self.backbone_runner
        if self.depth is not None:
            depth, plain, pair, fusion = self.depth, self.backbone, self.pair, self.cfg.depth.fusion

            def with_depth(part: torch.Tensor) -> torch.Tensor:
                if fusion:
                    final, (backbone_mid,) = plain.forward_intermediates(part, indices=[len(plain.feature_info) - 2])
                    feat = plain.forward_head(final)
                else:
                    backbone_mid, feat = None, plain(part)
                z_raw, _, stem_mid = depth.forward_with_mid(part, backbone_mid)
                return torch.cat([feat, z_raw, *([] if pair is None else [pair.window_scores(stem_mid)])], dim=1)

            backbone = with_depth

        if chunk and flat.shape[0] > chunk:
            return torch.cat([backbone(part) for part in flat.split(chunk)], dim=0)
        return backbone(flat)

    def _encode_spatial(self, part: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        """窓 ``[n, C, R, R]`` -> ``[n, feat_dim + 12 * feat_dim]``。backbone は 1 度だけ通し、最終特徴マップを GAP と枝の両方が読む。枝の位置平均も backbone の head を通す。"""
        maps = self.backbone.forward_features(part)
        pooled = self.spatial(maps, slots)
        n, f = pooled.shape[0], pooled.shape[-1]
        return torch.cat([self.backbone.forward_head(maps), self.backbone.forward_head(pooled.reshape(n * NUM_LABELS, f, 1, 1)).view(n, -1)], dim=1)

    def _window_features(self, feats: torch.Tensor, window_mask: torch.Tensor | None, window_slot: torch.Tensor | None, window_pos: torch.Tensor | None) -> torch.Tensor:
        """``[B, K, D]`` の窓特徴 -> LayerNorm 後の窓特徴 ``h [B, K, gap_dim]``。``model.seq`` の差分と ``model.depth`` の ``z`` もここで入れる。"""
        h = self.norm(feats[..., : self.feat_dim])
        if self.seq is not None:
            if window_slot is None or window_pos is None:
                raise ValueError("model.seq needs window_slot and window_pos")
            h = h + self.seq(h, window_mask, window_slot, window_pos)
        if self.depth_norm is not None:
            z = self.depth_norm(feats[..., self.feat_dim : self.gap_dim])
            if self.depth_zero:
                z = torch.zeros_like(z)
            h = torch.cat([h, z], dim=-1)
        return h

    def window_logits_from_features(self, feats: torch.Tensor, window_mask: torch.Tensor | None = None, window_slot: torch.Tensor | None = None, window_pos: torch.Tensor | None = None) -> torch.Tensor:
        """``[B, K, D]`` の窓特徴 -> 推論と study 損失が読む窓 logit ``[B, K, 12]``（:meth:`window_logits_pair` の前半）。"""
        return self.window_logits_pair(feats, window_mask, window_slot, window_pos)[0]

    def window_logits_pair(self, feats: torch.Tensor, window_mask: torch.Tensor | None = None, window_slot: torch.Tensor | None = None, window_pos: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """``[B, K, D]`` の窓特徴 -> ``(推論と study 損失が読む窓 logit, 窓損失が掛かる窓 logit)``、どちらも ``[B, K, 12]``。

        ``model.spatial`` が有効なら、所見ごとの枝の窓特徴を GAP 側に連結し、``model.context`` の残差を足して ``clsW`` で読む（局所 logit）。
        ``model.slot_context`` が有効なら、前半はその残差もさらに足して ``ctxW`` で読み、後半は局所 logit。無効なら 2 つは同じもの。
        padding 窓の値は意味を持たない（集約と損失の側で ``window_mask`` が除く）。
        """
        if self.att is not None:
            raise ValueError("window logits need model.window.pool other than attention")
        h = self._window_features(feats, window_mask, window_slot, window_pos)
        if self.spatial is None:
            local = torch.einsum("bkd,nd->bkn", h, self.clsW) + self.clsb
            return local, local
        sp = self.spatial_norm(feats[..., self.feat_dim :].unflatten(-1, (NUM_LABELS, self.feat_dim)))
        per_finding = torch.cat([h[:, :, None, :].expand(-1, -1, NUM_LABELS, -1), sp], dim=-1)
        if self.context is not None:
            per_finding = per_finding + self.context(h, per_finding, window_mask)
        local = (per_finding * self.clsW).sum(-1) + self.clsb
        if self.slot_context is None:
            return local, local
        if window_slot is None:
            raise ValueError("model.slot_context needs window_slot")
        return ((per_finding + self.slot_context(per_finding, window_mask, window_slot)) * self.ctxW).sum(-1) + self.ctxb, local

    @torch.no_grad()
    def copy_head_to_context(self) -> None:
        """文脈付き head ``ctxW`` / ``ctxb`` を局所 head ``clsW`` / ``clsb`` の値で上書きする。"""
        self.ctxW.copy_(self.clsW)
        self.ctxb.copy_(self.clsb)

    def window_scores(self, feats: torch.Tensor, window_slot: torch.Tensor | None = None, window_pos: torch.Tensor | None = None, window_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """``[B, K, D]`` の窓特徴 -> ``(h [B, K, gap_dim], 所見ごとの窓スコア [B, K, 12], 枝の窓特徴 [B, K, 12, feat_dim], 枝の窓スコア [B, K, 12])``。

        スコアは softmax の前で、padding の mask も掛けていない。枝が無効なら後ろ 2 つは ``None``。``h`` は :meth:`_window_features`。
        """
        if self.att is None:
            raise ValueError(f"window_scores needs model.window.pool=attention, got {self.cfg.window.pool}")
        h = self._window_features(feats, window_mask, window_slot, window_pos)
        if self.spatial is None:
            return h, self.att(h), None, None
        sp = self.spatial_norm(feats[..., self.feat_dim :].unflatten(-1, (NUM_LABELS, self.feat_dim)))
        return h, self.att(h), sp, (self.spatial_att(sp) * self.spatial_w).sum(-1)

    def pool_scores(self, h: torch.Tensor, scores: torch.Tensor, sp: torch.Tensor | None, sp_scores: torch.Tensor | None, keep: torch.Tensor | None) -> torch.Tensor:
        """:meth:`window_scores` の出力を、``keep`` ``[B, K]`` の窓だけで所見ごとに softmax して集約した ``pooled [B, 12, head_dim]``。``keep`` が ``None`` なら全窓。

        枝があれば、所見ごとの枝の窓特徴を枝の窓スコアで同じ窓だけから集約し、GAP 側の集約に連結する。
        """
        if keep is not None:
            scores = scores.masked_fill(~keep[..., None], -torch.inf)
        pooled = torch.einsum("bkn,bkf->bnf", scores.softmax(dim=1), h)
        if sp is None:
            return pooled
        if keep is not None:
            sp_scores = sp_scores.masked_fill(~keep[..., None], -torch.inf)
        return torch.cat([pooled, torch.einsum("bkn,bknf->bnf", sp_scores.softmax(dim=1), sp)], dim=-1)

    def pool_features(self, feats: torch.Tensor, window_mask: torch.Tensor | None = None, window_slot: torch.Tensor | None = None, window_pos: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """``[B, K, D]`` の窓特徴 -> ``(h [B, K, gap_dim], pooled [B, 12, head_dim])``。``h`` は LayerNorm 後の窓特徴、``pooled`` は所見ごとの attention で実在する窓だけから集約したもの。"""
        h, scores, sp, sp_scores = self.window_scores(feats, window_slot, window_pos, window_mask)
        return h, self.pool_scores(h, scores, sp, sp_scores, window_mask)

    def _logits(self, h: torch.Tensor, pooled: torch.Tensor, window_mask: torch.Tensor | None, window_slot: torch.Tensor | None) -> torch.Tensor:
        """pooled 特徴 -> logits ``[B, 12]``。``model.fsx`` が有効なら Slot ごとの窓特徴で補正する。"""
        if self.fsx is None:
            return (pooled * self.clsW).sum(-1) + self.clsb
        if window_slot is None:
            raise ValueError("model.fsx needs window_slot")
        return self.fsx(pooled, h, window_mask, window_slot, self.clsW, self.clsb)[0]

    def _pair_correct(self, logits: torch.Tensor, feats: torch.Tensor, window_mask: torch.Tensor | None, window_rows: torch.Tensor | None) -> torch.Tensor:
        """``model.pair`` が有効なら、窓特徴の末尾のペアスコアを Study 内で集約して半月板 2 所見の logits に足す。"""
        if self.pair is None or self.pair_zero:
            return logits
        if window_rows is None:
            raise ValueError("model.pair needs window_rows")
        return self.pair.correct(logits, self.pair.aggregate(feats[..., self.gap_dim :], window_rows, window_mask))

    def _series_neck(
        self, feats: torch.Tensor, window_mask: torch.Tensor | None, window_slot: torch.Tensor | None, window_series: torch.Tensor | None, window_u: torch.Tensor | None, aux: bool
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """``model.series_neck`` の ``(logits, 補助 logits, 面に series があるか)``。LayerNorm 後の GAP 特徴と空間枝の特徴を渡す。"""
        if window_slot is None or window_series is None or window_u is None:
            raise ValueError("model.series_neck needs window_slot, window_series and window_u (the union bag)")
        sp = self.spatial_norm(feats[..., self.feat_dim :].unflatten(-1, (NUM_LABELS, self.feat_dim)))
        return self.series_neck(self.norm(feats[..., : self.feat_dim]), sp, window_mask, window_slot, window_series, window_u, aux=aux)

    def head_from_features(
        self,
        feats: torch.Tensor,
        window_mask: torch.Tensor | None = None,
        window_slot: torch.Tensor | None = None,
        window_pos: torch.Tensor | None = None,
        window_rows: torch.Tensor | None = None,
        window_series: torch.Tensor | None = None,
        window_u: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``[B, K, D]`` の窓特徴 -> logits ``[B, 12]``。``window_series`` / ``window_u`` ``[B, K]`` は ``model.series_neck`` だけが読む。"""
        if self.series_neck is not None:
            return self._series_neck(feats, window_mask, window_slot, window_series, window_u, aux=False)[0]
        if self.att is None:
            return self.pool_window_logits(self.window_logits_from_features(feats, window_mask, window_slot, window_pos), window_mask)
        logits = self._logits(*self.pool_features(feats, window_mask, window_slot, window_pos), window_mask, window_slot)
        return self._pair_correct(logits, feats, window_mask, window_rows)

    def pool_window_logits(self, window_logits: torch.Tensor, window_mask: torch.Tensor | None) -> torch.Tensor:
        """窓 logit ``[B, K, 12]`` を ``model.window.pool`` で study の logits ``[B, 12]`` に集約する。"""
        window = self.cfg.window
        return pool_window_logits(window_logits, window_mask, window.pool, window.pool_tau)

    def heads_from_features(
        self, feats: torch.Tensor, window_mask: torch.Tensor | None = None, window_slot: torch.Tensor | None = None, window_pos: torch.Tensor | None = None, window_rows: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``[B, K, D]`` の窓特徴 -> ``(logits, severity_logits)``、どちらも ``[B, 12]``。同じ pooled 特徴を 2 組の線形が読む。"""
        if not self.cfg.severity_head:
            raise ValueError("heads_from_features needs model.severity_head=true")
        h, pooled = self.pool_features(feats, window_mask, window_slot, window_pos)
        return self._pair_correct(self._logits(h, pooled, window_mask, window_slot), feats, window_mask, window_rows), (pooled * self.sevW).sum(-1) + self.sevb

    def forward(
        self,
        windows: torch.Tensor,
        window_mask: torch.Tensor | None = None,
        window_slot: torch.Tensor | None = None,
        window_pos: torch.Tensor | None = None,
        window_rows: torch.Tensor | None = None,
        window_series: torch.Tensor | None = None,
        window_u: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Window Bag の batch ``[B, K, C, R, R]`` に対する logits ``[B, 12]``。

        ``window_slot`` / ``window_pos`` ``[B, K]`` は各窓の Slot 番号と中心スライスの Stack 行番号、``window_rows`` ``[B, K, C]`` は各チャネルの Stack 行番号。
        ``window_series`` / ``window_u`` ``[B, K]`` は全 series の Bag での各窓の series 番号と深さで、``model.series_neck`` だけが読む。
        """
        return self.head_from_features(self.encode(windows, window_mask, window_slot), window_mask, window_slot, window_pos, window_rows, window_series, window_u)

    def forward_heads(
        self, windows: torch.Tensor, window_mask: torch.Tensor | None = None, window_slot: torch.Tensor | None = None, window_pos: torch.Tensor | None = None, window_rows: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Window Bag の batch に対する ``(logits, severity_logits)``。学習の補助損失用で、推論は :meth:`forward` を使う。"""
        return self.heads_from_features(self.encode(windows, window_mask, window_slot), window_mask, window_slot, window_pos, window_rows)

    def forward_plane_aux(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """dataset の batch -> ``(logits [B, 12], 撮像面ごとの補助 logits [B, 面, 12], 面に窓があるか [B, 面])``。

        主の logits は :meth:`forward_batch` と同じ。補助 head は主と同じ窓スコアを、その面の窓だけで softmax して集約し、面ごとの線形で読む。
        窓の無い面は全窓で集約して値を有限に保ち、``面に窓があるか`` で損失と記録から外す。
        """
        if not self.cfg.plane_aux:
            raise ValueError("forward_plane_aux needs model.plane_aux=true")
        window_mask, window_slot, window_pos, window_rows = batch.get("window_mask"), batch.get("window_slot"), batch.get("window_pos"), batch.get("window_rows")
        if self.series_neck is not None:
            return self._series_neck(self.encode(batch["windows"], window_mask, window_slot), window_mask, window_slot, batch.get("window_series"), batch.get("window_u"), aux=True)
        if window_slot is None:
            raise ValueError("model.plane_aux needs window_slot")
        feats = self.encode(batch["windows"], window_mask, window_slot)
        h, scores, sp, sp_scores = self.window_scores(feats, window_slot, window_pos, window_mask)
        logits = self._pair_correct(self._logits(h, self.pool_scores(h, scores, sp, sp_scores, window_mask), window_mask, window_slot), feats, window_mask, window_rows)
        valid = torch.ones_like(window_slot, dtype=torch.bool) if window_mask is None else window_mask
        plane = self.plane_of_slot[window_slot.clamp_min(0)]
        plane_logits, present = [], []
        for index in range(len(PLANES)):
            keep = valid & (plane == index)
            has = keep.any(dim=1)
            pooled = self.pool_scores(h, scores, sp, sp_scores, keep | (valid & ~has[:, None]))
            plane_logits.append((pooled * self.planeW[index]).sum(-1) + self.planeb[index])
            present.append(has)
        return logits, torch.stack(plane_logits, dim=1), torch.stack(present, dim=1)

    def forward_batch(self, batch: dict[str, Any]) -> torch.Tensor:
        """dataset の batch（``batch["windows"]``）に対する logits。"""
        return self(batch["windows"], batch.get("window_mask"), batch.get("window_slot"), batch.get("window_pos"), batch.get("window_rows"), batch.get("window_series"), batch.get("window_u"))

    def forward_batch_windows(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """dataset の batch に対する ``(logits [B, 12], 窓損失の掛かる窓 logit [B, K, 12])``。窓の補助損失用で、``model.window.pool`` が attention 以外のときだけ。

        ``model.slot_context`` が有効なら、窓損失は文脈の残差を足さない局所 logit（``clsW`` で読む）に掛ける。
        """
        window_mask, window_slot = batch.get("window_mask"), batch.get("window_slot")
        window_logits, local = self.window_logits_pair(self.encode(batch["windows"], window_mask, window_slot), window_mask, window_slot, batch.get("window_pos"))
        return self.pool_window_logits(window_logits, window_mask), local

    @torch.no_grad()
    def pair_diagnostics(self, windows: torch.Tensor, window_mask: torch.Tensor | None, window_rows: torch.Tensor) -> dict[str, torch.Tensor]:
        """診断用。Bag の全窓（padding も）を通し、所見マップ ``A [B, K, 3, 2, h, w]``、C の null 質量 ``null [B, K, 6, h, w]``、
        窓のペアごとの空間集約 ``window [B, K, 3, 2]``、Study のスカラー ``S [B, 2]`` を返す。
        """
        if self.pair is None:
            raise ValueError("pair_diagnostics needs model.pair.enabled=true")
        b, k = windows.shape[:2]
        flat = (windows.flatten(0, 1) - self.pixel_mean) / self.pixel_std
        a, null, u = self.pair.maps(self.depth.forward_with_mid(flat)[2])
        scores = self.pair.spatial(u).view(b, k, -1)
        return {"A": a.unflatten(0, (b, k)), "null": null.unflatten(0, (b, k)), "window": scores.unflatten(-1, (len(PAIRS), len(PAIR_FINDINGS))), "S": self.pair.aggregate(scores, window_rows, window_mask)}

    def fsx_attention(self, windows: torch.Tensor, window_mask: torch.Tensor | None, window_slot: torch.Tensor, window_pos: torch.Tensor | None = None) -> torch.Tensor:
        """診断用。所見 x Slot の attention を head で平均した ``[B, 12, S]``。欠けた Slot は 0。"""
        if self.fsx is None:
            raise ValueError("fsx_attention needs model.fsx.enabled=true")
        h, pooled = self.pool_features(self.encode(windows, window_mask, window_slot), window_mask, window_slot, window_pos)
        return self.fsx(pooled, h, window_mask, window_slot, self.clsW, self.clsb)[1].mean(dim=1)


def pool_window_logits(window_logits: torch.Tensor, window_mask: torch.Tensor | None, pool: str, tau: float) -> torch.Tensor:
    """窓 logit ``z [B, K, 12]`` を実在窓（``window_mask`` [B, K]、``None`` なら全窓）だけで集約した ``[B, 12]``。

    "lse" は ``tau * (logsumexp_k(z / tau) - log(n_real))``。
    """
    b, k = window_logits.shape[:2]
    real = torch.ones((b, k), dtype=torch.bool, device=window_logits.device) if window_mask is None else window_mask
    n_real = real.sum(dim=1, keepdim=True)  # [B, 1]
    masked = window_logits.masked_fill(~real[..., None], -torch.inf)
    if pool == "lse":
        return tau * (torch.logsumexp(masked / tau, dim=1) - torch.log(n_real.to(window_logits.dtype)))
    raise ValueError(f"unknown window pool {pool!r}")


def pixel_stats(norm: str, context: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``norm`` に応じた画素の ``(mean, std)`` を ``[1, C, 1, 1]``（``C = context``）で返す。"""
    if norm == "imagenet":
        if context != 3:
            raise ValueError(f"norm='imagenet' is for context 3 only (ImageNet has one mean/std per RGB channel), got context {context}; use norm='mri'")
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    elif norm == "mri":
        mean, std = (MRI_MEAN,) * context, (MRI_STD,) * context
    elif norm == "half":
        mean, std = (0.5,) * context, (0.5,) * context
    elif norm == "none":
        mean, std = (0.0,) * context, (1.0,) * context
    else:
        raise ValueError(f"unknown window norm {norm!r}")
    return torch.tensor(mean, dtype=torch.float32).view(1, context, 1, 1), torch.tensor(std, dtype=torch.float32).view(1, context, 1, 1)


def build_window_model(cfg: Config, pretrained: bool | None = None) -> WindowModel:
    """ルート config から :class:`WindowModel` を組み立てる（使うのは ``cfg.model`` だけ）。"""
    return WindowModel(cfg.model, pretrained=pretrained)


def compile_backbone(model: WindowModel, mode: str = "default") -> Callable[[torch.Tensor], torch.Tensor]:
    """device 配置済みの backbone を ``torch.compile(dynamic=False)`` で包み、runner に据えて返す。

    入力形状ごとにコンパイルする。``infer_dtype`` の autocast は runner 内で開く。
    """
    if model.cfg.kind != "window":
        raise ValueError(f"model.kind={model.cfg.kind} does not support window backbone compilation")
    if model.depth is not None or model.spatial is not None:
        raise ValueError("model.depth / model.spatial cannot be compiled with compile_backbone (it wraps the backbone only)")
    compiled = torch.compile(model.backbone, mode=mode, dynamic=False)

    def runner(flat: torch.Tensor) -> torch.Tensor:
        if model.infer_dtype is not None and flat.is_cuda:
            with torch.autocast("cuda", dtype=model.infer_dtype):
                return compiled(flat).float()
        return compiled(flat)

    model.backbone_runner = runner
    return runner
