from pathlib import Path
from unittest import mock

import torch

from rsna_knee.config.config import ModelConfig
from rsna_knee.constants import NUM_LABELS

EDGE = 32


def _model_cfg(**extra: object) -> ModelConfig:
    return ModelConfig(backbone="resnet18", pretrained=False, img_size=EDGE, **extra)


def _tiny_qwen(directory: Path) -> Path:
    """Save a two-layer Qwen3.5 config (full attention only, so no linear-attention kernels are needed)."""
    from transformers import Qwen3_5Config

    vision = {"depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2, "out_hidden_size": 64, "patch_size": 4, "num_position_embeddings": 16}
    text = {"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2, "layer_types": ["full_attention", "full_attention"], "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32, "vocab_size": 300}
    Qwen3_5Config(vision_config=vision, text_config=text, video_token_id=250, image_token_id=251, vision_start_token_id=252, vision_end_token_id=253).save_pretrained(directory)
    return directory


def test_vlm_mil_forward_and_backward(tmp_path: Path) -> None:
    """Random-weight tiny Qwen3.5 (the processor is stubbed): one video of 4 merged tokens -> 12 finite logits."""
    import transformers

    from rsna_knee.models.vlm_mil import VlmMilModel

    cfg = _model_cfg(kind="vlm", vlm={"source": str(_tiny_qwen(tmp_path / "qwen")), "prompt": "p"})
    with mock.patch.object(transformers.AutoProcessor, "from_pretrained", return_value=mock.MagicMock()):
        model = VlmMilModel(cfg, pretrained=False)
    # One video: 2 frames -> 1 temporal patch, a 4x4 patch grid merged 2x2 -> 4 video tokens
    ids = torch.tensor([[1, 252, 250, 250, 250, 250, 253, 2]])
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "pixel_values_videos": torch.randn(16, 3 * 2 * 4 * 4), "video_grid_thw": torch.tensor([[1, 4, 4]]), "mm_token_type_ids": (ids == 250).long() * 2}
    logits = model.forward_batch(batch)
    assert logits.shape == (1, NUM_LABELS) and torch.isfinite(logits).all()
    logits.sum().backward()
    head_grads = [p.grad for group in model.optimizer_groups(1e-5, 1e-3)[2:] for p in group["params"]]
    assert any(g is not None and g.abs().sum() > 0 for g in head_grads)
    assert [len(group["params"]) > 0 for group in model.optimizer_groups(1e-5, 1e-3)] == [True] * 4
    assert model.set_grad_checkpointing(True) and model.set_grad_checkpointing(False)
    model.save_assets(tmp_path / "assets")
    assert (tmp_path / "assets" / "config.json").exists()
