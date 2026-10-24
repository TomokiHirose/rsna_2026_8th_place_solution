import numpy as np
import torch
import torch.nn.functional as F

from rsna_knee.constants import LABELS, NUM_LABELS
from rsna_knee.modules import macro_auc, masked_bce_with_logits, pairwise_rank_loss
from rsna_knee.modules.losses import ral_branches, soft_fcl_loss, soft_lse_rank_loss, soft_ral_loss


def test_masked_bce_ignores_nan_targets_and_stays_differentiable() -> None:
    logits = torch.randn(4, NUM_LABELS)
    target = (torch.rand(4, NUM_LABELS) > 0.5).float()

    masked_target = target.clone()
    masked_target[0, 0] = float("nan")
    masked_target[2, 5] = float("nan")

    loss = masked_bce_with_logits(logits, masked_target)
    assert torch.isfinite(loss)

    valid = ~torch.isnan(masked_target)
    expected = F.binary_cross_entropy_with_logits(logits[valid], target[valid])
    assert torch.allclose(loss, expected, atol=1e-6)

    # Moving only an ignored position must not change the loss
    moved = masked_target.clone()
    logits2 = logits.clone()
    logits2[0, 0] += 100.0
    assert torch.allclose(masked_bce_with_logits(logits2, moved), loss, atol=1e-6)

    # All unknown gives 0 but must stay differentiable
    free = torch.randn(2, NUM_LABELS, requires_grad=True)
    degenerate = masked_bce_with_logits(free, torch.full((2, NUM_LABELS), float("nan")))
    assert float(degenerate.detach()) == 0.0
    degenerate.backward()


def test_macro_auc_excludes_single_class_columns_and_nan_targets() -> None:
    rng = np.random.default_rng(0)
    n = 50
    y_true = rng.integers(0, 2, size=(n, NUM_LABELS)).astype(float)
    y_true[:, 3] = 1.0  # single class, so the AUC is undefined
    y_prob = rng.random((n, NUM_LABELS))

    macro, per_label = macro_auc(y_true, y_prob)
    assert set(per_label) == set(LABELS)
    assert np.isnan(per_label[LABELS[3]])
    assert np.isfinite(macro)
    finite = [v for v in per_label.values() if np.isfinite(v)]
    assert len(finite) == NUM_LABELS - 1
    assert macro == float(np.mean(finite))

    # Unknown rows are dropped, so perfectly separated remaining rows give 1.0
    unknown = np.array([[1.0], [0.0], [np.nan]] * 4).repeat(NUM_LABELS, axis=1)
    assert macro_auc(unknown, np.tile(np.array([[0.9], [0.1], [0.5]] * 4), (1, NUM_LABELS)))[0] == 1.0


def test_pairwise_rank_loss_only_moves_the_current_batch() -> None:
    logits, target = torch.randn(4, NUM_LABELS, requires_grad=True), torch.rand(4, NUM_LABELS)
    # The queue holds detached logits and must not receive gradients
    ref_logits, ref_target = torch.randn(6, NUM_LABELS), torch.rand(6, NUM_LABELS)
    pairwise_rank_loss(logits, target, ref_logits, ref_target).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all() and float(logits.grad.abs().sum()) > 0.0
    assert ref_logits.grad is None


def _official_ralloss_cells(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The official Ralloss.forward (default arguments) before reduction."""
    p = torch.sigmoid(x)
    xs_neg = (1 - p + 0.05).clamp(max=1)
    los_pos = y * (torch.log(p.clamp(min=1e-8)) + 1.0 * (1 - p.clamp(min=1e-8)) - 2.5 * 0.5 * torch.pow(1 - p.clamp(min=1e-8), 2))
    los_neg = (1 - y) * torch.log(xs_neg.clamp(min=1e-8)) * (1.5 - p) * p**2 * (1.5 - xs_neg)
    pt = p * y + xs_neg * (1 - y)
    return -(los_pos + los_neg) * torch.pow(1 - pt, 4 * (1 - y))


def test_soft_ral_mixes_official_hard_branches_linearly() -> None:
    x = torch.linspace(-6, 6, 2 * NUM_LABELS).reshape(2, NUM_LABELS)
    positive, negative = ral_branches(x)
    assert torch.allclose(positive, _official_ralloss_cells(x, torch.ones_like(x)), atol=1e-5)
    assert torch.allclose(negative, _official_ralloss_cells(x, torch.zeros_like(x)), atol=1e-5)
    y = torch.rand(2, NUM_LABELS)
    assert torch.allclose(soft_ral_loss(x, y), (y * positive + (1 - y) * negative).mean(), atol=1e-6)


def test_soft_fcl_reduces_to_soft_bce_and_stays_finite() -> None:
    logits, target = torch.randn(4, NUM_LABELS) * 3, torch.rand(4, NUM_LABELS)
    target[1, 2] = float("nan")
    weight, pos_weight = torch.rand(4, NUM_LABELS), torch.rand(NUM_LABELS) + 1
    assert torch.allclose(soft_fcl_loss(logits, target, 0.0, 0.0, pos_weight=pos_weight, weight=weight), masked_bce_with_logits(logits, target, pos_weight=pos_weight, weight=weight), atol=1e-5)
    extreme = torch.tensor([[-80.0, 80.0] * (NUM_LABELS // 2)], requires_grad=True)
    soft_fcl_loss(extreme, torch.rand(1, NUM_LABELS), 2.0, 1.0).backward()
    assert torch.isfinite(extreme.grad).all()


def test_soft_lse_rank_orders_by_soft_gap_and_skips_ties() -> None:
    logits = torch.zeros(3, NUM_LABELS)
    ties = torch.full((3, NUM_LABELS), 0.4)
    assert float(soft_lse_rank_loss(logits, ties, ["a", "b", "c"])) == 0.0
    small, large = torch.tensor([[0.6], [0.4]]).repeat(1, NUM_LABELS), torch.tensor([[0.9], [0.1]]).repeat(1, NUM_LABELS)
    pair = torch.zeros(2, NUM_LABELS)
    assert soft_lse_rank_loss(pair, small, ["a", "b"]) < soft_lse_rank_loss(pair, large, ["a", "b"])
    inverted = torch.tensor([[-2.0], [2.0]]).repeat(1, NUM_LABELS)
    assert soft_lse_rank_loss(inverted, large, ["a", "b"]) > soft_lse_rank_loss(pair, large, ["a", "b"])


def test_soft_lse_rank_queue_gradients_and_duplicate_studies() -> None:
    logits = torch.zeros(1, NUM_LABELS, requires_grad=True)
    # A higher queued study pushes down the current lower study; queue-queue and same-study pairs are not counted
    ref_logits, ref_target = torch.zeros(3, NUM_LABELS, requires_grad=True), torch.tensor([[0.9], [0.0], [0.9]]).repeat(1, NUM_LABELS)
    soft_lse_rank_loss(logits, torch.full((1, NUM_LABELS), 0.1), ["cur"], ref_logits.detach(), ref_target, ["up", "low", "cur"]).backward()
    assert (logits.grad > 0).all() and ref_logits.grad is None
    only_self = soft_lse_rank_loss(torch.zeros(1, NUM_LABELS), torch.full((1, NUM_LABELS), 0.1), ["cur"], torch.zeros(1, NUM_LABELS), torch.full((1, NUM_LABELS), 0.9), ["cur"])
    assert float(only_self) == 0.0
