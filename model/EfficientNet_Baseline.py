from __future__ import annotations

import torch
from torch import nn
from torchvision import models


def build_efficientnet_v2_s(num_classes: int, freeze_backbone: bool = True, device: str | torch.device = "cpu") -> nn.Module:
    try:
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
    except TypeError:
        model = models.efficientnet_v2_s(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier[1].parameters():
            param.requires_grad = True
    model.to(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    return model


def build_model(num_classes: int, freeze_backbone: bool = True, device: str | torch.device = "cpu") -> nn.Module:
    return build_efficientnet_v2_s(num_classes=num_classes, freeze_backbone=freeze_backbone, device=device)


def build_optimizer(model: nn.Module, lr: float = 1e-3) -> torch.optim.Optimizer:
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(trainable_params, lr=lr)

LEGACY_BASE_MODEL = "efficientnet_v2_s"


def get_head_module_path(model: nn.Module | None = None) -> str:
    from model import _transfer_strategies as _ts

    target_model = model if model is not None else build_model(num_classes=101, device="cpu")
    return _ts.get_head_module_path(target_model, base_model=LEGACY_BASE_MODEL)


def get_feature_dim(model: nn.Module | None = None) -> int:
    from model import _transfer_strategies as _ts

    target_model = model if model is not None else build_model(num_classes=101, device="cpu")
    return int(_ts.get_feature_dim(target_model, base_model=LEGACY_BASE_MODEL))


def get_classifier_info(model: nn.Module | None = None) -> dict[str, object]:
    from model import _transfer_strategies as _ts

    target_model = model if model is not None else build_model(num_classes=101, device="cpu")
    payload = _ts.get_classifier_info(target_model, base_model=LEGACY_BASE_MODEL)
    payload.setdefault("source", "legacy")
    return payload


def replace_classifier_head(model: nn.Module, num_classes: int) -> nn.Module:
    from model import _transfer_strategies as _ts

    return _ts.replace_classifier_head(model, num_classes=int(num_classes), base_model=LEGACY_BASE_MODEL)

