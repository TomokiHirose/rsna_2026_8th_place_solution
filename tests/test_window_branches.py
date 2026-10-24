import pytest
import torch

from rsna_knee.config.config import ModelConfig
from rsna_knee.constants import NUM_LABELS
from rsna_knee.data.slots import NUM_SLOTS
from rsna_knee.models.window_model import WindowModel

EDGE = 32
B, K = 2, 4

CASES = {
    "depth_pair": {"depth": {"enabled": True, "width": 16}, "pair": {"enabled": True, "qk_dim": 8, "hidden": 8}},
    "depth_lstm": {"depth": {"enabled": True, "width": 16, "kind": "lstm"}},
    "depth_conv3d_fusion16": {"depth": {"enabled": True, "width": 16, "kind": "conv3d", "fusion": True, "stride": 16}},
    "depth_fusion32": {"depth": {"enabled": True, "width": 16, "fusion": True}},
    "seq": {"seq": {"enabled": True, "width": 16, "layers": 1}},
    "fsx": {"fsx": {"enabled": True}},
}


def _bag(context: int = 3, edge: int = EDGE) -> dict[str, torch.Tensor]:
    """A random batch of two Bags; the last window of the second Bag is padding."""
    torch.manual_seed(0)
    mask = torch.ones(B, K, dtype=torch.bool)
    mask[1, -1] = False
    slot = torch.tensor([[0, 0, 1, 2], [3, 3, 4, 4]])
    pos = torch.tensor([[0, 1, 0, 0], [0, 1, 0, 1]])
    rows = torch.stack([pos - 1, pos, pos + 1], dim=-1).clamp_min(0)
    return {"windows": torch.rand(B, K, context, edge, edge), "window_mask": mask, "window_slot": slot, "window_pos": pos, "window_rows": rows}


def _model(name: str) -> WindowModel:
    torch.manual_seed(0)
    return WindowModel(ModelConfig(backbone="resnet18", pretrained=False, img_size=EDGE, **CASES[name]))


def _has_signal(params: list[torch.Tensor]) -> bool:
    grads = [p.grad for p in params]
    assert all(g is None or torch.isfinite(g).all() for g in grads)
    return any(g is not None and g.abs().sum() > 0 for g in grads)


@pytest.mark.parametrize("name", list(CASES))
def test_branch_trains_from_the_first_step(name: str) -> None:
    """One backward gives finite logits and a nonzero gradient where the branch is learnt at step 0."""
    model = _model(name)
    logits = model.forward_batch(_bag())
    assert logits.shape == (B, NUM_LABELS) and torch.isfinite(logits).all()
    logits.sum().backward()
    if model.depth is not None:
        # The depth columns of the readout start at zero, so only they move on the first step; the branch itself follows.
        assert model.clsW.grad[:, model.feat_dim : model.gap_dim].abs().sum() > 0
        # The pair branch reads the depth stem, so the stem learns at once only when pair is on.
        assert _has_signal(list(model.depth.stem.parameters())) == (model.pair is not None)
    for branch in (model.pair, model.seq, model.fsx):
        if branch is not None:
            assert _has_signal(list(branch.parameters()))


@pytest.mark.parametrize("name", [n for n in CASES if n.startswith("depth")])
def test_depth_branch_starts_as_a_no_op(name: str) -> None:
    """At initialisation zeroing the depth features (or one side of the fusion) leaves the logits unchanged."""
    model = _model(name).eval()
    batch = _bag()
    with torch.no_grad():
        base = model.forward_batch(batch)
        model.depth_zero = True
        assert torch.allclose(model.forward_batch(batch), base, atol=1e-5)
        model.depth_zero = False
        if model.depth.fuse is not None:
            for zero in ("depth", "mid"):
                model.depth.fusion_zero = zero
                assert torch.allclose(model.forward_batch(batch), base, atol=1e-5)
            model.depth.fusion_zero = "bad"
            with pytest.raises(ValueError):
                model.forward_batch(batch)


@pytest.mark.parametrize("name", list(CASES))
def test_padded_window_does_not_change_the_logits(name: str) -> None:
    """Changing the pixels of a padded window leaves both Bags' logits unchanged."""
    model = _model(name).eval()
    batch = _bag()
    noisy = dict(batch, windows=batch["windows"].clone())
    noisy["windows"][1, -1] = torch.randn_like(noisy["windows"][1, -1]) * 5
    with torch.no_grad():
        assert torch.allclose(model.forward_batch(batch), model.forward_batch(noisy), atol=1e-5)


def test_pair_and_fsx_diagnostics_have_the_documented_shapes() -> None:
    """``pair_diagnostics`` scores the two meniscus findings; ``fsx_attention`` is a finding x Slot map."""
    batch = _bag()
    with torch.no_grad():
        diag = _model("depth_pair").eval().pair_diagnostics(batch["windows"], batch["window_mask"], batch["window_rows"])
        assert diag["S"].shape == (B, 2) and torch.isfinite(diag["S"]).all()
        attention = _model("fsx").eval().fsx_attention(batch["windows"], batch["window_mask"], batch["window_slot"], batch["window_pos"])
    assert attention.shape == (B, NUM_LABELS, NUM_SLOTS) and torch.isfinite(attention).all()
