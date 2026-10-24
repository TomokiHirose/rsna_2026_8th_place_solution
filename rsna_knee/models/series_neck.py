from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from rsna_knee.config.config import SeriesNeckConfig
from rsna_knee.constants import NUM_LABELS
from rsna_knee.data.slots import NUM_SLOTS

__all__ = ["SeriesNeck"]


class SeriesNeck(nn.Module):
    """窓 -> series -> study の 2 段の集約。series の中は深さの順の双方向 GRU と所見ごとの attention、series の間は所見ごとの attention（Slot の埋め込みと Slot x 所見の定数つき）。

    GAP 側は GRU の出力 ``h [B, K, 2H]`` を、空間枝は所見ごとの窓特徴 ``sp [B, K, 12, F]`` をそれぞれのスコアで series の中だけで softmax して集約し、連結した ``v [B, S, 12, 2H + F]`` を
    series の間の softmax で混ぜて所見ごとの線形で読む。撮像面の補助 head は series の間の softmax をその面の series に絞る。
    """

    def __init__(self, cfg: SeriesNeckConfig, feat_dim: int, spatial_hidden: int, drop: float, plane_of_slot: list[int]) -> None:
        super().__init__()
        self.cfg = cfg
        width = 2 * cfg.gru_hidden
        self.dim = width + feat_dim
        # 深さ u・前の窓との差・次の窓との差・Slot の one-hot -> GAP 特徴に足す埋め込み。
        self.pos = nn.Sequential(nn.Linear(3 + NUM_SLOTS, cfg.pos_hidden), nn.GELU(), nn.Linear(cfg.pos_hidden, feat_dim))
        self.gru = nn.GRU(feat_dim, cfg.gru_hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(cfg.drop)
        # series の中の窓スコア。GAP 側は所見で共有する tanh 層 + 12 出力、空間枝は所見で共有する tanh 層 + 所見ごとの読み出し。
        self.att = nn.Sequential(nn.Linear(width, cfg.att_hidden), nn.Tanh(), nn.Dropout(drop), nn.Linear(cfg.att_hidden, NUM_LABELS))
        self.spatial_att = nn.Sequential(nn.Linear(feat_dim, spatial_hidden), nn.Tanh(), nn.Dropout(drop))
        self.spatial_w = nn.Parameter(nn.init.trunc_normal_(torch.zeros(NUM_LABELS, spatial_hidden), std=0.02))
        # series の間のスコア。読み出しと定数は 0 から始まるので、学習開始時は series の単純平均。
        self.series_in = nn.Linear(self.dim, cfg.att_hidden)
        self.slot_emb = nn.Parameter(nn.init.trunc_normal_(torch.zeros(NUM_SLOTS, cfg.att_hidden), std=0.02))
        self.series_drop = nn.Dropout(drop)
        self.series_w = nn.Parameter(torch.zeros(NUM_LABELS, cfg.att_hidden))
        self.series_b = nn.Parameter(torch.zeros(NUM_SLOTS, NUM_LABELS))
        self.clsW = nn.Parameter(nn.init.trunc_normal_(torch.zeros(NUM_LABELS, self.dim), std=0.02))
        self.clsb = nn.Parameter(torch.zeros(NUM_LABELS))
        self.num_planes = max(plane_of_slot) + 1
        self.planeW = nn.Parameter(nn.init.trunc_normal_(torch.zeros(self.num_planes, NUM_LABELS, self.dim), std=0.02))
        self.planeb = nn.Parameter(torch.zeros(self.num_planes, NUM_LABELS))
        self.register_buffer("plane_of_slot", torch.tensor(plane_of_slot), persistent=False)

    def encode_series(self, h: torch.Tensor, real: torch.Tensor, slot: torch.Tensor, series: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """``h [B, K, F]`` を (study, series) ごとに u の昇順に並べて GRU に通し、元の窓の位置に戻した ``[B, K, 2H]``。padding 窓は 0。"""
        b, k, f = h.shape
        n_series = int(series[real].max()) + 1
        index = torch.nonzero(real.flatten()).squeeze(1)
        group = (torch.arange(b, device=h.device)[:, None] * n_series + series).flatten()[index]
        depth = u.flatten()[index].float()
        by_depth = torch.argsort(depth, stable=True)
        order = by_depth[torch.argsort(group[by_depth], stable=True)]
        index, group, depth = index[order], group[order], depth[order]
        counts = torch.bincount(group, minlength=b * n_series)
        rank = torch.arange(len(group), device=h.device) - (torch.cumsum(counts, 0) - counts)[group]
        last = rank == counts[group] - 1
        before = torch.where(rank > 0, depth - depth.roll(1), 0.0)
        after = torch.where(last, 0.0, depth.roll(-1) - depth)
        onehot = F.one_hot(slot.flatten()[index], NUM_SLOTS).float()
        x = self.drop(h.flatten(0, 1)[index].float() + self.pos(torch.cat([depth[:, None], before[:, None], after[:, None], onehot], dim=1)))
        # 窓のある (study, series) だけを詰めた列に並べ、長さで pack して GRU に通す。
        present = counts > 0
        compact = torch.cumsum(present, 0) - 1
        rows = compact[group]
        padded = x.new_zeros((int(present.sum()), int(counts.max()), f))
        padded[rows, rank] = x
        packed = pack_padded_sequence(padded, counts[present].cpu(), batch_first=True, enforce_sorted=False)
        out, _ = pad_packed_sequence(self.gru(packed)[0], batch_first=True)
        flat = out.new_zeros((b * k, out.shape[-1]))
        flat[index] = self.drop(out[rows, rank])
        return flat.view(b, k, -1)

    def topk_mask(self, scores: torch.Tensor, member: torch.Tensor) -> torch.Tensor:
        """学習中に集約から外す窓 ``[B, K, 12]``。series ごと・所見ごとにスコア上位 ``ceil(topk_frac * n)`` 本（n が 1 なら 0 本、最大 n - 1 本）を候補にし、各候補を ``topk_drop`` で外す。"""
        ranked = scores.float()[:, :, None, :].masked_fill(~member[..., None], -torch.inf)  # [B, K, S, 12]
        order = torch.argsort(ranked, dim=1, descending=True)
        rank = torch.empty_like(order).scatter_(1, order, torch.arange(scores.shape[1], device=scores.device)[None, :, None, None].expand_as(order))
        n = member.sum(dim=1)  # [B, S]
        quota = torch.minimum(torch.ceil(n * self.cfg.topk_frac), n - 1).clamp_min(0)
        candidate = member[..., None] & (rank < quota[:, None, :, None])
        return (candidate & (torch.rand(candidate.shape, device=scores.device) < self.cfg.topk_drop)).any(dim=2)

    def forward(
        self, h: torch.Tensor, sp: torch.Tensor, window_mask: torch.Tensor | None, window_slot: torch.Tensor, window_series: torch.Tensor, window_u: torch.Tensor, aux: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """LayerNorm 後の GAP 特徴 ``h [B, K, F]`` と空間枝の特徴 ``sp [B, K, 12, F]`` -> ``(logits [B, 12], 撮像面ごとの補助 logits [B, 面, 12], 面に series があるか [B, 面])``。

        ``aux`` が偽なら後ろ 2 つは ``None``。fp32 で計算する。
        """
        real = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device) if window_mask is None else window_mask
        with torch.autocast(device_type=h.device.type, enabled=False):
            sp = sp.float()
            g = self.encode_series(h, real, window_slot, window_series, window_u)
            n_series = int(window_series[real].max()) + 1
            member = (window_series[..., None] == torch.arange(n_series, device=h.device)) & real[..., None]  # [B, K, S]
            present = member.any(dim=1)
            gap_scores, sp_scores = self.att(g), (self.spatial_att(sp) * self.spatial_w).sum(-1)
            if self.training and self.cfg.topk_frac > 0.0 and self.cfg.topk_drop > 0.0:
                gap_scores = gap_scores.masked_fill(self.topk_mask(gap_scores, member), -torch.inf)
                sp_scores = sp_scores.masked_fill(self.topk_mask(sp_scores, member), -torch.inf)

            def weights(scores: torch.Tensor) -> torch.Tensor:
                # 窓の無い series は全窓で一様にして有限に保ち、後で 0 を掛ける。
                logits = scores[:, :, None, :].masked_fill(~member[..., None], -torch.inf).masked_fill(~present[:, None, :, None], 0.0)
                return logits.softmax(dim=1) * present[:, None, :, None]

            v = torch.cat([torch.einsum("bksn,bkf->bsnf", weights(gap_scores), g), torch.einsum("bksn,bknf->bsnf", weights(sp_scores), sp)], dim=-1)  # [B, S, 12, D]
            series_slot = (member * window_slot[..., None]).amax(dim=1)  # [B, S]、series の窓はどれも同じ Slot
            hidden = self.series_drop(torch.tanh(self.series_in(v) + self.slot_emb[series_slot][:, :, None, :]))
            series_scores = (hidden * self.series_w).sum(-1) + self.series_b[series_slot]  # [B, S, 12]

            def pool(keep: torch.Tensor) -> torch.Tensor:
                beta = series_scores.masked_fill(~keep[..., None], -torch.inf).softmax(dim=1)
                return torch.einsum("bsn,bsnd->bnd", beta, v)

            logits = (pool(present) * self.clsW).sum(-1) + self.clsb
            if not aux:
                return logits, None, None
            plane = self.plane_of_slot[series_slot]
            plane_logits, has = [], []
            for index in range(self.num_planes):
                keep = present & (plane == index)
                found = keep.any(dim=1)
                # 面に series の無い study は全 series で集約して値を有限に保ち、``has`` で損失と記録から外す。
                plane_logits.append((pool(keep | (present & ~found[:, None])) * self.planeW[index]).sum(-1) + self.planeb[index])
                has.append(found)
            return logits, torch.stack(plane_logits, dim=1), torch.stack(has, dim=1)
