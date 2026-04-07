from __future__ import annotations

import torch
from torch import nn

from model._transfer_strategies import build_optimizer as _build_optimizer
from model._transfer_strategies import strategy_builder


def build_model(
    num_classes: int,
    freeze_backbone: bool = True,
    device: str | torch.device = "cpu",
) -> nn.Module:
    del freeze_backbone
    return strategy_builder("efficientnet", "bn_tuning")(num_classes, device)


def build_optimizer(model: nn.Module, lr: float = 1e-3) -> torch.optim.Optimizer:
    return _build_optimizer(model, lr=lr)
