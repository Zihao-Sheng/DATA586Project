from __future__ import annotations

import torch
from torch import nn

from model._transfer_strategies import (
    build_optimizer as _build_optimizer,
    build_model_from_spec as _build_model_from_spec,
)


GENERATED_SPEC = {'model_name': 'efficient_net_bn_dora_agressive_last2_defroze', 'base_provider': 'torchvision', 'base_family': 'efficientnet', 'variant': 'efficientnet_v2_s', 'base_model': 'efficientnet_v2_s', 'task_type': 'classification', 'method_type': 'dora', 'freeze_strategy': 'bn_tuning_with_last_stages', 'train_bn': True, 'train_norm': False, 'unfreeze_stages': [6, 7], 'peft_method': 'dora', 'peft_targets': {'feature_stages': [6, 7], 'layer_keys': [], 'classifier': False}, 'peft_params': {'rank': 8, 'alpha': 16.0}, 'stage_lr_overrides': {}, 'gradcam_target_hint': ['features.7'], 'pretrained': True, 'metadata_version': '1.3', 'generator_version': 'phase_canvas_methods_v1', 'is_generated': True, 'source_of_truth': 'spec', 'spec_name': 'efficient_net_bn_dora_agressive_last2_defroze.json', 'spec_file': 'model_specs/efficient_net_bn_dora_agressive_last2_defroze.json', 'source_spec_file': 'model_specs/efficient_net_bn_dora_agressive_last2_defroze.json'}


def _resolved_device(device: str | torch.device) -> str | torch.device:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def build_model(num_classes: int, freeze_backbone: bool = True, device: str | torch.device = "cpu") -> nn.Module:
    del freeze_backbone
    resolved_device = _resolved_device(device)
    pretrained = bool(GENERATED_SPEC.get("pretrained", True))
    return _build_model_from_spec(dict(GENERATED_SPEC), num_classes=num_classes, device=resolved_device, pretrained=pretrained)


def build_optimizer(model: nn.Module, lr: float = 1e-3) -> torch.optim.Optimizer:
    return _build_optimizer(
        model,
        lr=lr,
        base_model=str(GENERATED_SPEC.get("base_model", "")),
        stage_lr_overrides=GENERATED_SPEC.get("stage_lr_overrides", {}),
    )



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
    metadata = dict(GENERATED_SPEC)
    metadata.setdefault("is_generated", True)
    metadata.setdefault("source_of_truth", "spec")
    metadata.setdefault("spec_name", f"{metadata.get('model_name', 'unknown')}.json")
    metadata.setdefault("spec_file", f"model_specs/{metadata.get('model_name', 'unknown')}.json")
    metadata.setdefault("source_spec_file", f"model_specs/{metadata.get('model_name', 'unknown')}.json")
    return metadata


def get_capabilities() -> dict[str, bool]:
    method_type = str(GENERATED_SPEC.get("method_type", "baseline"))
    return {
        "supports_resume": True,
        "supports_gradcam": True,
        "supports_structure_editing": True,
        "supports_lora": method_type == "lora",
        "supports_dora": method_type == "dora",
        "supports_tsa": method_type == "tsa",
        "supports_adapter": method_type == "adapter",
        "supports_bitfit": method_type == "bitfit",
        "supports_ssf": method_type == "ssf",
        "supports_bn_tuning": method_type in {"bn_tuning", "bn_last1", "bn_last2"},
        "supports_classifier_head_adaptation": True,
        "supports_norm_tuning": method_type == "norm_tuning",
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
    if base_model == "resnet50":
        return {
            "base_family": "resnet50",
            "feature_stages": ["conv1", "layer1", "layer2", "layer3", "layer4"],
            "classifier": "fc",
        }
    if base_model == "convnext_tiny":
        return {
            "base_family": "convnext_tiny",
            "feature_stages": ["stem", "stage1", "stage2", "stage3", "stage4"],
            "classifier": "classifier.2",
        }
    if base_model == "mobilenet_v3_large":
        return {
            "base_family": "mobilenet_v3_large",
            "feature_stages": ["stem", "stage1", "stage2", "stage3", "stage4"],
            "classifier": "classifier.3",
        }
    if base_model == "densenet121":
        return {
            "base_family": "densenet121",
            "feature_stages": ["stem", "denseblock1", "denseblock2", "denseblock3", "denseblock4"],
            "classifier": "classifier",
        }
    return {
        "base_family": "efficientnet_v2_s",
        "feature_stages": [f"features.{idx}" for idx in range(8)],
        "classifier": "classifier.1",
    }


def get_default_gradcam_targets() -> list[str]:
    return list(["features.7"])
