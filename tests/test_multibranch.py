import torch

from rsna_knee.config.config import ModelConfig
from rsna_knee.constants import NUM_LABELS
from rsna_knee.models.multibranch import BRANCH_NAMES, MultiBranchModel

from .test_window_branches import EDGE, B, _bag


def _model() -> MultiBranchModel:
    """Two tiny CNN encoders with the fusion projection."""
    torch.manual_seed(0)
    cfg = ModelConfig(kind="multibranch", backbone="resnet18", pretrained=False, img_size=EDGE, multibranch={"backbones": ("resnet18", "resnet10t"), "spatial_dim": 8, "att_hidden": 8, "fusion_dim": 16})
    return MultiBranchModel(cfg)


def test_every_head_and_encoder_learns_in_one_step() -> None:
    """Three finite heads; one AdamW step moves both encoders and the fusion projection."""
    model = _model()
    branches = model.forward_batch_branches(_bag())
    assert branches.shape == (len(BRANCH_NAMES), B, NUM_LABELS) and torch.isfinite(branches).all()
    groups = model.optimizer_groups((1e-4, 2e-4), 1e-3)
    assert [g["lr"] for g in groups][:2] == [1e-4, 2e-4]
    branches.sum().backward()
    before = [[p.detach().clone() for p in g["params"]] for g in groups]
    torch.optim.AdamW(groups).step()
    for group, old in zip(groups, before):
        assert any(not torch.equal(a, p) for a, p in zip(old, group["params"]))
    assert model.set_grad_checkpointing(False)


def test_blend_matches_mix_and_ignores_padding() -> None:
    """``forward_batch`` is ``mix`` of the three heads, and a padded window does not change it."""
    model = _model().eval()
    batch = _bag()
    noisy = dict(batch, windows=batch["windows"].clone())
    noisy["windows"][1, -1] = torch.randn_like(noisy["windows"][1, -1]) * 5
    with torch.no_grad():
        logits = model.forward_batch(batch)
        assert logits.shape == (B, NUM_LABELS) and torch.isfinite(logits).all()
        assert torch.allclose(model.mix(model.forward_batch_branches(batch)), logits, atol=1e-5)
        assert torch.allclose(model.forward_batch(noisy), logits, atol=1e-5)
