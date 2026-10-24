import torch

from rsna_knee.config import Config
from rsna_knee.constants import LABELS
from rsna_knee.models.window_model import PLANES, WindowModel

EDGE = 64


def _model() -> WindowModel:
    neck = {"enabled": True, "gru_hidden": 8, "pos_hidden": 4, "att_hidden": 8, "topk_frac": 0.5, "topk_drop": 0.9}
    cfg = Config(model={"backbone": "resnet18", "pretrained": False, "img_size": EDGE, "spatial": {"enabled": True, "dim": 8, "att_hidden": 8}, "plane_aux": True, "freeze_encoder": True, "series_neck": neck})
    return WindowModel(cfg.model, pretrained=False)


def test_series_neck_ignores_window_order_series_labels_and_padding() -> None:
    """In eval, reordering windows, relabelling series or adding padding leaves the logits unchanged (the GRU re-sorts each series by u).

    The auxiliary head is finite per plane, and a plane without series has ``present`` false.
    """
    torch.manual_seed(0)
    model = _model().eval()
    windows = torch.rand(2, 9, 3, EDGE, EDGE)
    mask = torch.tensor([[True] * 9, [True] * 6 + [False] * 3])
    series = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 2, 2], [1, 1, 0, 0, 0, 1, 0, 0, 0]])
    slot = torch.gather(torch.tensor([[0, 2, 4], [1, 3, 0]]), 1, series)  # study 1 has no Axial series
    u = torch.rand(2, 9) * 2 - 1
    batch = {"windows": windows, "window_mask": mask, "window_slot": slot, "window_series": series, "window_u": u}
    with torch.no_grad():
        logits = model.forward_batch(batch)
        perm = torch.tensor([3, 0, 8, 1, 5, 2, 7, 4, 6])
        shuffled = model.forward_batch({key: value[:, perm] for key, value in batch.items()})
        relabelled = model.forward_batch({**batch, "window_series": 2 - series})
        pad = {key: torch.cat([value, torch.zeros((2, 4, *value.shape[2:]), dtype=value.dtype)], dim=1) for key, value in batch.items()}
        padded = model.forward_batch(pad)
        main, plane_logits, present = model.forward_plane_aux(batch)
    assert all(torch.allclose(logits, other, atol=1e-5) for other in (shuffled, relabelled, padded, main))
    assert plane_logits.shape == (2, len(PLANES), len(LABELS)) and torch.isfinite(plane_logits).all()
    assert present.tolist() == [[True, True, True], [True, True, False]]


def test_series_neck_masks_top_windows_only_in_training_and_keeps_one_per_series() -> None:
    """Only in training, top-scoring windows are masked per series and finding; one-window series are kept and every series keeps a window.

    Gradients reach the GRU and the auxiliary head.
    """
    torch.manual_seed(0)
    model = _model()
    member = torch.tensor([[1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]]).bool().T[None]  # [1, K=6, S=3]: 4, 1 and 1 windows
    scores = torch.randn(1, 6, len(LABELS))
    dropped = torch.stack([model.series_neck.topk_mask(scores, member) for _ in range(50)])
    assert dropped[..., :4, :].any() and not dropped[..., 4:, :].any() and not dropped[..., :4, :].all(dim=-2).any()
    batch = {"windows": torch.rand(1, 6, 3, EDGE, EDGE), "window_slot": torch.tensor([[0, 0, 0, 0, 2, 4]]), "window_series": torch.tensor([[0, 0, 0, 0, 1, 2]]), "window_u": torch.linspace(-1, 1, 6)[None]}
    model.train()
    logits, plane_logits, _ = model.forward_plane_aux(batch)
    (logits.sum() + plane_logits.sum()).backward()
    assert model.series_neck.gru.weight_ih_l0.grad.abs().sum() > 0 and model.series_neck.planeW.grad.abs().sum() > 0 and model.backbone.conv1.weight.grad is None
