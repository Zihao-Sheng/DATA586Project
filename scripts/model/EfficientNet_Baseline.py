from __future__ import annotations

import torch
from torch import nn
from torchvision import models


def build_efficientnet_v2_s(
    num_classes: int,
    freeze_backbone: bool = True,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Create an EfficientNetV2-S classifier with ImageNet pretrained weights."""
    try:
        model = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.DEFAULT
        )
    except TypeError:
        # older torchvision
        model = models.efficientnet_v2_s(pretrained=True)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier[1].parameters():
            param.requires_grad = True

    model.to(device)
    return model


def build_model(
    num_classes: int,
    freeze_backbone: bool = True,
    device: str | torch.device = "cpu",
) -> nn.Module:
    return build_efficientnet_v2_s(
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        device=device,
    )


def build_optimizer(
    model: nn.Module,
    lr: float = 1e-3,
) -> torch.optim.Optimizer:
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(trainable_params, lr=lr)