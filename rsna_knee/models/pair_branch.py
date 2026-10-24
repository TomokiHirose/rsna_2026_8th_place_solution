from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rsna_knee.config.config import PairConfig
from rsna_knee.constants import LABELS
from rsna_knee.models.depth_branch import MID_CHANNELS

__all__ = ["PAIRS", "PAIR_FINDINGS", "PairBranch"]

#: 窓内で二画像確認に使うチャネルの組（無向）。
PAIRS = ((0, 1), (1, 2), (0, 2))
#: 補正する所見。
PAIR_FINDINGS = ("Medial Meniscus", "Lateral Meniscus")


def _lse_mean(x: torch.Tensor, tau: torch.Tensor, dim: int) -> torch.Tensor:
    """``(1 / tau) log mean exp(tau x)`` を ``dim`` 方向に取る。``tau`` は ``x`` の末尾の軸に掛かる。"""
    return (torch.logsumexp(x * tau, dim=dim) - math.log(x.shape[dim])) / tau


class PairBranch(nn.Module):
    """stem 3 段目の画像別特徴から、窓内の 2 画像が対応する局所領域に同じ所見の証拠を持つかを所見ごとのスカラーにする。

    A は画像ごとに独立な所見マップ、C は 5x5 近傍 + null の局所 cross-attention。``r = A_i * sum_q C A_j`` を両向きで平均し、
    空間方向と Study 内の有効なペアについて LSE 平均を取る。
    """

    def __init__(self, cfg: PairConfig) -> None:
        super().__init__()
        self.cfg = cfg
        n = len(PAIR_FINDINGS)
        self.evidence = nn.Sequential(nn.Conv2d(MID_CHANNELS, cfg.hidden, 3, padding=1, bias=False), nn.BatchNorm2d(cfg.hidden), nn.GELU(), nn.Conv2d(cfg.hidden, n, 1))
        nn.init.zeros_(self.evidence[-1].bias)  # A を 0.5 付近から始める
        self.query = nn.Conv2d(MID_CHANNELS, cfg.qk_dim, 1)
        self.key = nn.Conv2d(MID_CHANNELS, cfg.qk_dim, 1)
        self.null = nn.Linear(cfg.qk_dim, 1)
        self.rel_bias = nn.Parameter(torch.zeros(cfg.neighborhood**2))
        # 行 0 が空間方向、行 1 が Study 内のペア方向の温度。softplus を通して正にする。
        self.tau_raw = nn.Parameter(torch.full((2, n), math.log(math.expm1(cfg.tau_init))))
        self.weight = nn.Parameter(torch.full((n,), cfg.w_init))
        self.register_buffer("finding_index", torch.tensor([LABELS.index(name) for name in PAIR_FINDINGS]), persistent=False)

    def tau(self) -> torch.Tensor:
        """``[2, n]`` の温度（行 0 が空間、行 1 がペア）。"""
        return F.softplus(self.tau_raw)

    def maps(self, mid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``[N, 3, 128, h, w]`` -> ``(A [N, 3, n, h, w], null 質量 [N, 6, h, w], u [N, 3, n, h, w])``。

        null 質量の並びは ``PAIRS`` の順の i -> j、続いて同じ順の j -> i。``u`` は両向きの r の平均。
        """
        nw, c, _, h, w = mid.shape
        flat = mid.flatten(0, 1)
        a = torch.sigmoid(self.evidence(flat)).view(nw, c, -1, h, w)
        q = self.query(flat).view(nw, c, -1, h, w)
        k = self.key(flat).view(nw, c, -1, h, w)
        src = [i for i, _ in PAIRS] + [j for _, j in PAIRS]
        dst = [j for _, j in PAIRS] + [i for i, _ in PAIRS]
        qi, ai = q[:, src], a[:, src]
        radius = self.cfg.neighborhood // 2
        kp = F.pad(k[:, dst], (radius,) * 4)
        ap = F.pad(a[:, dst], (radius,) * 4)
        inside = F.pad(mid.new_ones(h, w), (radius,) * 4)
        offsets = [(dy, dx) for dy in range(2 * radius + 1) for dx in range(2 * radius + 1)]
        logits = torch.stack([(qi * kp[..., dy : dy + h, dx : dx + w]).sum(2) for dy, dx in offsets], dim=2) / math.sqrt(self.cfg.qk_dim) + self.rel_bias[:, None, None]
        outside = torch.stack([inside[dy : dy + h, dx : dx + w] == 0 for dy, dx in offsets])  # [25, h, w]
        logits = logits.masked_fill(outside, -torch.inf)
        null = self.null(qi.movedim(2, -1)).squeeze(-1)  # [N, 6, h, w]
        # null を含めた softmax。実位置の重みは再正規化しない。
        attn = torch.cat([logits, null[:, :, None]], dim=2).softmax(dim=2)
        matched = sum(attn[:, :, d, None] * ap[..., dy : dy + h, dx : dx + w] for d, (dy, dx) in enumerate(offsets))  # [N, 6, n, h, w]
        r = ai * matched
        return a, attn[:, :, -1], (r[:, : len(PAIRS)] + r[:, len(PAIRS) :]) / 2

    def spatial(self, u: torch.Tensor) -> torch.Tensor:
        """``u [N, 3, n, h, w]`` を空間方向に LSE 平均し、``[N, 3 * n]``（ペア優先の並び）にする。"""
        return _lse_mean(u.flatten(-2).transpose(-1, -2), self.tau()[0], dim=-2).flatten(1)

    def window_scores(self, mid: torch.Tensor) -> torch.Tensor:
        """``[N, 3, 128, h, w]`` -> 窓のペアごとの空間集約 ``[N, 3 * n]``。"""
        return self.spatial(self.maps(mid)[2])

    def aggregate(self, scores: torch.Tensor, window_rows: torch.Tensor, window_mask: torch.Tensor | None) -> torch.Tensor:
        """窓のペアのスコア ``[B, K, 3 * n]`` を、有効で重複を除いたペアについて LSE 平均し ``S [B, n]`` にする。

        行番号の等しい組（端の複製）と padding 窓は無効。同じ無向の行番号ペアは Bag 内で最初の 1 つだけ数える。有効なペアが無ければ 0。
        """
        b, k = scores.shape[:2]
        n = len(PAIR_FINDINGS)
        first = window_rows[..., [i for i, _ in PAIRS]]
        second = window_rows[..., [j for _, j in PAIRS]]
        lo, hi = torch.minimum(first, second).flatten(1), torch.maximum(first, second).flatten(1)  # [B, M]
        valid = lo != hi
        if window_mask is not None:
            valid = valid & window_mask[..., None].expand(b, k, len(PAIRS)).flatten(1)
        same = (lo[:, :, None] == lo[:, None, :]) & (hi[:, :, None] == hi[:, None, :]) & valid[:, None, :]
        earlier = torch.ones(lo.shape[1], lo.shape[1], dtype=torch.bool, device=lo.device).tril(-1)
        keep = valid & ~(same & earlier).any(dim=-1)
        found = keep.any(dim=1)
        tau = self.tau()[1]
        # 温度を掛けてから除く（-inf に掛けると温度の勾配が NaN になる）。有効なペアが無い Study は全要素で計算してから 0 に差し替え、-inf 同士の差で NaN を出さない。
        x = (scores.reshape(b, k * len(PAIRS), n).float() * tau).masked_fill(~(keep | ~found[:, None])[..., None], -torch.inf)
        count = keep.sum(dim=1).clamp(min=1)
        s = (torch.logsumexp(x, dim=1) - count.log()[:, None]) / tau
        return torch.where(found[:, None], s, torch.zeros_like(s))

    def correct(self, logits: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """半月板 2 所見の logits に ``w * S`` を足す（bias なし）。"""
        return logits.index_add(1, self.finding_index, (self.weight * s).to(logits.dtype))
