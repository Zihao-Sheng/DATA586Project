from __future__ import annotations

import torch
from torch import nn

from model._transfer_strategies import build_optimizer as _build_optimizer
from model._transfer_strategies import strategy_builder


def build_model(num_classes: int, freeze_backbone: bool = False, device: str | torch.device = "cpu") -> nn.Module:
    del freeze_backbone
    return strategy_builder("efficientnet", "full_finetune")(num_classes, device)


def build_optimizer(model: nn.Module, lr: float = 1e-3) -> torch.optim.Optimizer:
    return _build_optimizer(model, lr=lr)

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

