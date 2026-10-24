from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from rsna_knee.config.config import DepthConfig

__all__ = ["MID_CHANNELS", "DepthBranch", "SliceStem"]

#: stem の 3 段目（H / 16）のチャネル数。
MID_CHANNELS = 128
#: ``SliceStem.layers`` のうち 3 段目までの層の数（Conv・BN・GELU が 3 組）。
_MID_LAYERS = 9


def _conv_bn_gelu(cin: int, cout: int, kernel: int, stride: int) -> list[nn.Module]:
    return [nn.Conv2d(cin, cout, kernel, stride=stride, padding=kernel // 2, bias=False), nn.BatchNorm2d(cout), nn.GELU()]


class SliceStem(nn.Module):
    """各スライス ``[M, 1, H, W]`` に同じ重みの Conv 列を掛け、``[M, width, H / stride, W / stride]`` にする（``stride`` は 32 か 16）。"""

    def __init__(self, width: int, stride: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(*_conv_bn_gelu(1, 32, 7, 4), *_conv_bn_gelu(32, 64, 3, 2), *_conv_bn_gelu(64, MID_CHANNELS, 3, 2), *_conv_bn_gelu(MID_CHANNELS, width, 3, stride // 16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    def forward_with_mid(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(出力 [M, width, H / stride, W / stride], 3 段目の出力 [M, 128, H / 16, W / 16])``。"""
        mid = self.layers[:_MID_LAYERS](x)
        return self.layers[_MID_LAYERS:](mid), mid


class DepthBranch(nn.Module):
    """窓の C スライス（``window.context``）を共有 stem で特徴マップにし、奥行き方向に混ぜた中央スライスの特徴を位置の attention で 1 本の ``z_raw`` に集約する。

    流れは stem（Conv → BN → GELU × 4）→ mixer（Conv3d → BN → GELU、または LSTM → BN）→ 中央スライス → 位置の attention で集約した ``z_raw``。
    ``cfg.fusion`` なら中央スライスの特徴に backbone の中間特徴 ``mid`` を連結し、3x3 Conv → BN → GELU で混ぜてから attention に渡す。
    区間端の窓は片側の隣接が無いので端画像を複製して入力する。
    """

    def __init__(self, cfg: DepthConfig, mid_channels: int | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        width = cfg.width
        # stride 16 なら最後の段で縮めず、中央面を backbone の最後から 2 段目と同じ解像度にする。
        self.stem = SliceStem(width, cfg.stride)
        if cfg.kind == "conv_depth":
            self.mixer: nn.Module = nn.Sequential(nn.Conv3d(width, width, (3, 1, 1), padding=(1, 0, 0), bias=False), nn.BatchNorm3d(width), nn.GELU())
        elif cfg.kind == "conv3d":
            self.mixer = nn.Sequential(nn.Conv3d(width, width, (3, 3, 3), padding=(1, 1, 1), bias=False), nn.BatchNorm3d(width), nn.GELU())
        else:
            self.mixer = nn.LSTM(width, width // 2, bidirectional=True, batch_first=True)
        # LSTM の出力には非線形が無いので、中央時点の特徴をここで正規化する。conv 系は mixer の中で正規化済み。
        self.out_norm = nn.BatchNorm2d(width) if isinstance(self.mixer, nn.LSTM) else None
        if cfg.fusion:
            if mid_channels is None:
                raise ValueError("model.depth.fusion needs the channel count of the backbone's intermediate feature map")
            # 中間特徴を width チャネル・中央面と同じ解像度にする（stride 32 なら 2x2 の平均で縮める）。正規化は連結後の GELU の直前で効かせるので非線形は置かない。
            self.mid_proj: nn.Module | None = nn.Sequential(nn.Conv2d(mid_channels, width, 1, bias=False), nn.BatchNorm2d(width), nn.AvgPool2d(2) if cfg.stride == 32 else nn.Identity())
            self.fuse: nn.Module | None = nn.Sequential(*_conv_bn_gelu(2 * width, width, 3, 1))
        else:
            self.mid_proj = self.fuse = None
        # 同一の重みで融合の片側の寄与を測るとき、"depth"（中央面）か "mid"（backbone の中間特徴）を連結の前にゼロにする。
        self.fusion_zero: Literal["depth", "mid"] | None = None
        # 中央スライスの特徴マップの位置ごとのスコア。位置方向の softmax で集約の重みにする。
        self.score = nn.Conv2d(width, 1, 1)

    def forward(self, flat: torch.Tensor, backbone_mid: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """正規化済みの窓 ``[N, C, H, W]`` -> ``(z_raw [N, width], beta [N, h * w])``。``beta`` は位置の attention の重み（診断用）。

        ``cfg.fusion`` なら ``backbone_mid`` は同じ窓の backbone 中間特徴 ``[N, C_mid, 2h, 2w]``。
        """
        return self.forward_with_mid(flat, backbone_mid)[:2]

    def forward_with_mid(self, flat: torch.Tensor, backbone_mid: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """:meth:`forward` の返り値に、stem 3 段目の画像別特徴 ``[N, C, 128, H / 16, W / 16]`` を添える。"""
        n, c, height, width_px = flat.shape
        x, mid = self.stem.forward_with_mid(flat.reshape(n * c, 1, height, width_px))
        h, w = x.shape[-2:]
        x = x.view(n, c, -1, h, w).transpose(1, 2)  # [N, width, C, h, w]
        if isinstance(self.mixer, nn.LSTM):
            seq = x.permute(0, 3, 4, 2, 1).reshape(n * h * w, c, -1)  # [N·h·w, C, width]
            center = self.mixer(seq)[0][:, c // 2]
            center = self.out_norm(center.view(n, h, w, -1).permute(0, 3, 1, 2))
        else:
            center = self.mixer(x)[:, :, c // 2]
        if self.fuse is not None:
            if backbone_mid is None:
                raise ValueError("model.depth.fusion needs the backbone's intermediate feature map")
            projected = self.mid_proj(backbone_mid)
            if self.fusion_zero == "depth":
                center = torch.zeros_like(center)
            elif self.fusion_zero == "mid":
                projected = torch.zeros_like(projected)
            elif self.fusion_zero is not None:
                raise ValueError(f"fusion_zero must be 'depth', 'mid' or None, got {self.fusion_zero!r}")
            center = self.fuse(torch.cat([center, projected], dim=1))
        beta = self.score(center).flatten(1).softmax(dim=1)  # [N, h * w]
        return torch.einsum("np,ncp->nc", beta, center.flatten(2)), beta, mid.view(n, c, *mid.shape[1:])
