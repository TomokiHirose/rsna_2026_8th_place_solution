import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["masked_bce_with_logits", "pairwise_rank_loss", "pos_weight_from_targets", "soft_fcl_loss", "soft_lse_rank_loss", "soft_ral_loss", "window_bce_with_logits"]


def _masked_mean(loss: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
    """セル損失 ``[B, L]`` を ``NaN`` の target を除いて平均する。``weight`` があれば ``Σ w·loss·valid / Σ w·valid``。"""
    valid = ~torch.isnan(target)
    if not valid.any():
        return loss.sum() * 0.0
    loss = torch.where(valid, loss, torch.zeros_like(loss))
    if weight is None:
        return loss.sum() / valid.sum()
    cell = valid * weight.to(loss.dtype).to(loss.device)
    total = cell.sum()
    if not bool(total > 0):
        return loss.sum() * 0.0
    return (loss * cell).sum() / total


def _positive_weight(pos_weight: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor | float:
    return 1.0 if pos_weight is None else pos_weight.to(like.dtype).to(like.device)


def masked_bce_with_logits(logits: torch.Tensor, target: torch.Tensor, pos_weight: torch.Tensor | None = None, weight: torch.Tensor | None = None) -> torch.Tensor:
    """unknown (``NaN``) の target を無視する binary cross entropy。

    ``pos_weight`` は正例側に掛ける ``[L]`` の重み（``nn.BCEWithLogitsLoss`` と同じ）。``weight`` は
    セルごとの非負の重み ``[B, L]`` で、平均は ``Σ w·loss·valid / Σ w·valid`` になる。
    """
    safe_target = torch.nan_to_num(target, nan=0.0)
    positive = None if pos_weight is None else pos_weight.to(logits.dtype).to(logits.device)
    loss = F.binary_cross_entropy_with_logits(logits, safe_target, reduction="none", pos_weight=positive)
    return _masked_mean(loss, target, weight)


def window_bce_with_logits(window_logits: torch.Tensor, label: torch.Tensor, prior: torch.Tensor, window_mask: torch.Tensor, pos_weight: torch.Tensor | None = None) -> torch.Tensor:
    """窓 logit ``[B, K, L]`` と窓ターゲット ``label[:, None] * prior`` の BCE を、実在窓（``window_mask`` [B, K]）x ラベルが NaN でないセルで平均する。

    ``label`` は study の soft ラベル ``[B, L]``、``prior`` は窓ごとの所見別 prior 密度 ``[B, K, L]``（[0, 1]）。
    """
    target = (label[:, None, :] * prior).masked_fill(~window_mask[..., None], float("nan"))
    return masked_bce_with_logits(window_logits, target, pos_weight=pos_weight)


def ral_branches(logits: torch.Tensor, gamma_neg: float = 4.0, clip: float = 0.05, lamb: float = 1.5, epsilon_pos: float = 1.0, epsilon_pos_pow: float = -2.5) -> tuple[torch.Tensor, torch.Tensor]:
    """公式 Ralloss を ``y=1`` と ``y=0`` で別々に評価したセル損失 ``(L+, L-)``（reduction 前、符号は最小化向き）。

    ``gamma_pos=0``・``epsilon_neg=0``・focal 重みに勾配を流す公式の既定に固定している。
    """
    x = logits.float()
    p = torch.sigmoid(x)
    q = 1.0 - p
    positive = -(F.logsigmoid(x) + epsilon_pos * q + epsilon_pos_pow * 0.5 * q.pow(2))
    q_clip = (q + clip).clamp(max=1.0)
    negative = -torch.log(q_clip.clamp(min=1e-8)) * (lamb - p) * p.pow(2) * (lamb - q_clip) * (1.0 - q_clip).pow(gamma_neg)
    return positive, negative


def soft_ral_loss(logits: torch.Tensor, target: torch.Tensor, pos_weight: torch.Tensor | None = None, weight: torch.Tensor | None = None) -> torch.Tensor:
    """``y·L+ + (1-y)·L-`` の soft RAL。``pos_weight`` は陽性枝だけに掛け、平均は :func:`masked_bce_with_logits` と同じ。"""
    y = torch.nan_to_num(target.float(), nan=0.0)
    positive, negative = ral_branches(logits)
    loss = y * _positive_weight(pos_weight, positive) * positive + (1.0 - y) * negative
    return _masked_mean(loss, target, weight)


def soft_fcl_loss(logits: torch.Tensor, target: torch.Tensor, gamma: float, beta: float, pos_weight: torch.Tensor | None = None, weight: torch.Tensor | None = None) -> torch.Tensor:
    """``-y(1-p)^γ log p - (1-y)p^γ log(1-p) + β(p-y)^2`` の soft focal。

    ``pos_weight`` は陽性の focal 項だけに掛け、二乗誤差の anchor には掛けない。``γ=β=0`` で soft BCE。
    """
    x = logits.float()
    y = torch.nan_to_num(target.float(), nan=0.0)
    p = torch.sigmoid(x)
    focal = -y * _positive_weight(pos_weight, x) * (1.0 - p).pow(gamma) * F.logsigmoid(x) - (1.0 - y) * p.pow(gamma) * F.logsigmoid(-x)
    return _masked_mean(focal + beta * (p - y).pow(2), target, weight)


def _rank_terms(scores: torch.Tensor, labels: torch.Tensor, other_scores: torch.Tensor, other_labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``scores`` 側が上位であるべき対について、所見ごとの重み付き損失の和と重みの和（``[L]`` 2 本）。"""
    weight = torch.nan_to_num(labels[:, None, :] - other_labels[None, :, :], nan=0.0).clamp_min(0.0)
    loss = F.softplus(other_scores[None, :, :] - scores[:, None, :])
    return (weight * loss).sum(dim=(0, 1)), weight.sum(dim=(0, 1))


def pairwise_rank_loss(logits: torch.Tensor, target: torch.Tensor, ref_logits: torch.Tensor | None = None, ref_target: torch.Tensor | None = None) -> torch.Tensor:
    """所見ごとに study の対の順序を罰する RankNet 型の損失。

    「i が j より上位であるべき」対を数え上げ、``softplus(s_j - s_i)`` で ``s_i < s_j`` を罰する。対の
    重みは soft label の差 ``relu(y_i - y_j)`` で、どちらかが ``NaN`` なら 0。平均は所見ごとに取る
    （``Σ w·loss / Σ w``）ので、macro AUC と同じく所見を等しく重み付ける。

    ``ref_logits`` / ``ref_target``（``[M, L]`` 2 本）には直近の学習 batch を detach して溜めた queue を
    渡す。detach 済みなので勾配は現在の batch の ``logits`` 側にだけ流れる。揃えて渡すか両方 ``None``。
    対が 1 つも無ければ 0.0 を返す。
    """
    if (ref_logits is None) != (ref_target is None):
        raise ValueError("pairwise_rank_loss needs ref_logits and ref_target together, or neither")
    scores, labels = logits.float(), target.float()
    ref = None if ref_logits is None or ref_target is None else (ref_logits.float(), ref_target.float())
    # i in batch、j in batch + ref。batch 内の対はここで両向きとも出る。
    num, den = _rank_terms(scores, labels, scores if ref is None else torch.cat([scores, ref[0]]), labels if ref is None else torch.cat([labels, ref[1]]))
    if ref is not None:
        # ref が上位であるべき対（i in ref、j in batch）。現在の logits を下げる向きの勾配。
        ref_num, ref_den = _rank_terms(ref[0], ref[1], scores, labels)
        num, den = num + ref_num, den + ref_den
    paired = den > 0
    if not bool(paired.any()):
        return logits.sum() * 0.0
    return (num[paired] / den[paired]).mean()


def soft_lse_rank_loss(logits: torch.Tensor, target: torch.Tensor, ids: list[str], ref_logits: torch.Tensor | None = None, ref_target: torch.Tensor | None = None, ref_ids: list[str] | None = None, temperature: float = 1.0) -> torch.Tensor:
    """所見ごとに上位 anchor ``i`` の順位逆転を ``log(1 + (1/M_i) Σ_j q_ij exp((s_j - s_i)/T))`` で集約する損失。

    ``q_ij = relu(y_i - y_j)`` が正の対だけを数え、``M_i`` はその個数。対は少なくとも片端が現在の batch で、
    queue 同士の対・同じ study の対は除く。queue は study ID が現在の batch にあるものを除き、同じ ID は最新だけ
    残す。所見内で ``M_i > 0`` の anchor を平均し、対のある所見で平均する。対が無ければ 0.0。
    """
    scores, labels = logits.float(), target.float()
    names = list(ids)
    current = [True] * len(names)
    if ref_logits is not None:
        seen = set(names)
        keep = []
        for index in range(len(ref_ids) - 1, -1, -1):
            if ref_ids[index] not in seen:
                seen.add(ref_ids[index])
                keep.append(index)
        keep = keep[::-1]
        rows = torch.tensor(keep, dtype=torch.long, device=scores.device)
        scores = torch.cat([scores, ref_logits.float()[rows]])
        labels = torch.cat([labels, ref_target.float()[rows]])
        names += [ref_ids[index] for index in keep]
        current += [False] * len(keep)
    is_current = torch.tensor(current, device=scores.device)
    code = {name: index for index, name in enumerate(dict.fromkeys(names))}
    study = torch.tensor([code[name] for name in names], device=scores.device)
    allowed = (is_current[:, None] | is_current[None, :]) & (study[:, None] != study[None, :])
    q = torch.nan_to_num(labels[:, None, :] - labels[None, :, :], nan=0.0).clamp_min(0.0)
    eligible = allowed[:, :, None] & (q > 0)
    count = eligible.sum(dim=1)
    anchor = count > 0
    if not bool(anchor.any()):
        return logits.sum() * 0.0
    z = torch.log(q.clamp_min(1e-12)) + (scores[None, :, :] - scores[:, None, :]) / temperature - torch.log(count.clamp_min(1).float())[:, None, :]
    lse = torch.logsumexp(torch.where(eligible, z, torch.full_like(z, -1e4)), dim=1)
    per_anchor = torch.where(anchor, F.softplus(lse), torch.zeros_like(lse))
    paired = anchor.any(dim=0)
    return (per_anchor.sum(dim=0)[paired] / anchor.sum(dim=0)[paired]).mean()


def pos_weight_from_targets(targets: np.ndarray, lo: float = 0.03, hi: float = 0.7, max_weight: float = 10.0) -> torch.Tensor:
    """学習 target ``[N, L]`` から ``FloatTensor[L]`` の正例重み ``clip((1 - prev) / prev, 1, max_weight)`` を作る。

    ``prev`` はラベルごとの target の平均（``NaN`` は飛ばす）で、先に ``[lo, hi]`` にクリップする。``lo``
    の下限が無いとほとんど正例の出ない所見が数百倍の重みを得て損失を支配し、``1`` の下限が無いと
    ありふれた所見の正例が下げられてしまう。
    """
    prev = np.clip(np.nanmean(np.asarray(targets, dtype=np.float64), axis=0), lo, hi)
    return torch.tensor(np.clip((1.0 - prev) / prev, 1.0, max_weight), dtype=torch.float32)
