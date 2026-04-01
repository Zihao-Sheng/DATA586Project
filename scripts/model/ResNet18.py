from __future__ import annotations

import torch
from torch import nn
from torchvision import models


def build_resnet18(
    num_classes: int,
    freeze_backbone: bool = True,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Create a minimal ResNet18 classifier."""
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except TypeError:
        model = models.resnet18(pretrained=False)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

    model.to(device)
    return model


def build_model(
    num_classes: int,
    freeze_backbone: bool = True,
    device: str | torch.device = "cpu",
) -> nn.Module:
    return build_resnet18(
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        device=device,
    )


def build_optimizer(model: nn.Module, lr: float = 1e-3) -> torch.optim.Optimizer:
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(trainable_params, lr=lr)
