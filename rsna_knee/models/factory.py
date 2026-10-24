from rsna_knee.config import Config
from rsna_knee.models.window_model import WindowModel


def build_model(cfg: Config, pretrained: bool | None = None):
    """設定のモデル種別に従って学習・推論のモデルを構築する。"""
    if cfg.model.kind == "vlm":
        from rsna_knee.models.vlm_mil import VlmMilModel

        return VlmMilModel(cfg.model, pretrained=cfg.model.pretrained if pretrained is None else pretrained)
    if cfg.model.kind == "multibranch":
        from rsna_knee.models.multibranch import MultiBranchModel

        return MultiBranchModel(cfg.model, pretrained=pretrained)
    return WindowModel(cfg.model, pretrained=pretrained)
