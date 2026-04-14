from __future__ import annotations

import torch
from torch import nn

from model._transfer_strategies import (
    DoRALinear,
    apply_dora_recursively,
    build_optimizer as _build_optimizer,
    freeze_all,
    load_efficientnet_v2_s_classifier,
    strategy_builder,
)


GENERATED_SPEC = {'model_name': 'resnet18_lora_gen', 'base_model': 'resnet18', 'task_type': 'classification', 'method_type': 'lora', 'freeze_strategy': 'frozen_backbone_peft', 'train_bn': False, 'unfreeze_stages': [], 'peft_method': 'lora', 'peft_targets': {'feature_stages': [], 'layer_keys': ['layer4'], 'classifier': True}, 'peft_params': {'rank': 8, 'alpha': 16.0}, 'gradcam_target_hint': ['layer4'], 'pretrained': True, 'metadata_version': '1.1', 'generator_version': 'phase3_v1'}


def _resolved_device(device: str | torch.device) -> str | torch.device:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _strategy_tuple() -> tuple[str, str]:
    base_model = str(GENERATED_SPEC.get("base_model", "efficientnet_v2_s"))
    method_type = str(GENERATED_SPEC.get("method_type", "baseline"))
    backbone = "efficientnet" if base_model == "efficientnet_v2_s" else "resnet18"
    strategy_map = {
        "baseline": "linear_probe",
        "bn_tuning": "bn_tuning",
        "bn_last1": "bn_last1",
        "bn_last2": "bn_last2",
        "full_finetune": "full_finetune",
        "lora": "lora",
        "dora": "dora",
        "tsa": "tsa",
    }
    if method_type not in strategy_map:
        raise ValueError(f"Unsupported generated method_type: {method_type}")
    return backbone, strategy_map[method_type]


def build_model(num_classes: int, freeze_backbone: bool = True, device: str | torch.device = "cpu") -> nn.Module:
    del freeze_backbone
    resolved_device = _resolved_device(device)
    base_model = str(GENERATED_SPEC.get("base_model", "efficientnet_v2_s"))
    method_type = str(GENERATED_SPEC.get("method_type", "baseline"))
    pretrained = bool(GENERATED_SPEC.get("pretrained", True))

    if base_model == "efficientnet_v2_s" and method_type == "dora":
        feature_stages = []
        use_classifier = True
        rank = 8
        alpha = 16.0
        # Preserve handwritten DoRA-equivalent semantics by default; allow constrained custom targets when requested.
        if feature_stages == [6, 7] and use_classifier and rank == 8 and abs(alpha - 16.0) < 1e-9:
            backbone, strategy = _strategy_tuple()
            return strategy_builder(backbone, strategy)(num_classes, resolved_device, pretrained)

        model = load_efficientnet_v2_s_classifier(num_classes, pretrained=pretrained)
        freeze_all(model)
        for stage_idx in feature_stages:
            if 0 <= int(stage_idx) < len(model.features):
                apply_dora_recursively(model.features[int(stage_idx)])
        if use_classifier:
            model.classifier[1] = DoRALinear(model.classifier[1], rank=rank, alpha=alpha)
        model.to(resolved_device)
        return model

    backbone, strategy = _strategy_tuple()
    return strategy_builder(backbone, strategy)(num_classes, resolved_device, pretrained)


def build_optimizer(model: nn.Module, lr: float = 1e-3) -> torch.optim.Optimizer:
    return _build_optimizer(model, lr=lr)



def _classifier_base_model() -> str:
    return str(GENERATED_SPEC.get("base_model", ""))


def get_head_module_path(model: nn.Module | None = None) -> str:
    from model import _transfer_strategies as _ts

    target_model = model if model is not None else build_model(num_classes=101, device="cpu")
    return _ts.get_head_module_path(target_model, base_model=_classifier_base_model())


def get_feature_dim(model: nn.Module | None = None) -> int:
    from model import _transfer_strategies as _ts

    target_model = model if model is not None else build_model(num_classes=101, device="cpu")
    return int(_ts.get_feature_dim(target_model, base_model=_classifier_base_model()))


def get_classifier_info(model: nn.Module | None = None) -> dict[str, object]:
    from model import _transfer_strategies as _ts

    target_model = model if model is not None else build_model(num_classes=101, device="cpu")
    payload = _ts.get_classifier_info(target_model, base_model=_classifier_base_model())
    payload.setdefault("source", "generated_spec")
    payload.setdefault("model_name", str(GENERATED_SPEC.get("model_name", "")))
    return payload


def replace_classifier_head(model: nn.Module, num_classes: int) -> nn.Module:
    from model import _transfer_strategies as _ts

    return _ts.replace_classifier_head(model, num_classes=int(num_classes), base_model=_classifier_base_model())

def get_model_metadata() -> dict[str, object]:
    return dict(GENERATED_SPEC)


def get_capabilities() -> dict[str, bool]:
    method_type = str(GENERATED_SPEC.get("method_type", "baseline"))
    return {
        "supports_resume": True,
        "supports_gradcam": True,
        "supports_structure_editing": False,
        "supports_lora": method_type == "lora",
        "supports_dora": method_type == "dora",
        "supports_tsa": method_type == "tsa",
        "supports_bn_tuning": method_type in {"bn_tuning", "bn_last1", "bn_last2"},
        "supports_classifier_head_adaptation": True,
    }


def describe_model_structure() -> dict[str, object]:
    base_model = str(GENERATED_SPEC.get("base_model", "efficientnet_v2_s"))
    if base_model == "resnet18":
        return {
            "base_family": "resnet18",
            "feature_stages": ["conv1", "layer1", "layer2", "layer3", "layer4"],
            "classifier": "fc",
        }
    return {
        "base_family": "efficientnet_v2_s",
        "feature_stages": [f"features.{idx}" for idx in range(8)],
        "classifier": "classifier.1",
    }


def get_default_gradcam_targets() -> list[str]:
    return list(["layer4"])
