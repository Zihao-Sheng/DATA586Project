from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from core import runtime_paths

PROJECT_ROOT = runtime_paths.project_root()
MODEL_DIR = runtime_paths.model_dir()
SPEC_DIR = runtime_paths.model_specs_dir()

BASE_MODEL_ALIASES = {
    "torchvision/resnet18": "resnet18",
    "efficientnet": "efficientnet_v2_s",
    "efficientnet_v2_s": "efficientnet_v2_s",
    "torchvision/efficientnet_v2_s": "efficientnet_v2_s",
    "resnet": "resnet18",
    "resnet18": "resnet18",
    "resnet50": "resnet50",
    "torchvision/resnet50": "resnet50",
    "convnext_tiny": "convnext_tiny",
    "torchvision/convnext_tiny": "convnext_tiny",
    "mobilenet_v3_large": "mobilenet_v3_large",
    "torchvision/mobilenet_v3_large": "mobilenet_v3_large",
    "densenet121": "densenet121",
    "torchvision/densenet121": "densenet121",
}
SUPPORTED_BASE_MODELS = {
    "efficientnet_v2_s",
    "resnet18",
    "resnet50",
    "convnext_tiny",
    "mobilenet_v3_large",
    "densenet121",
}
SUPPORTED_METHODS_BY_BASE: dict[str, tuple[str, ...]] = {
    "efficientnet_v2_s": (
        "baseline",
        "bn_tuning",
        "bn_last1",
        "bn_last2",
        "norm_tuning",
        "full_finetune",
        "lora",
        "dora",
        "tsa",
        "adapter",
        "bitfit",
        "ssf",
    ),
    "resnet18": ("baseline", "bn_tuning", "norm_tuning", "full_finetune", "lora", "tsa", "adapter", "bitfit", "ssf"),
    "resnet50": ("baseline", "bn_tuning", "norm_tuning", "full_finetune", "lora", "tsa", "adapter", "bitfit", "ssf"),
    "convnext_tiny": ("baseline", "norm_tuning", "full_finetune", "lora", "tsa", "adapter", "bitfit", "ssf"),
    "mobilenet_v3_large": ("baseline", "bn_tuning", "norm_tuning", "full_finetune", "lora", "tsa", "adapter", "bitfit", "ssf"),
    "densenet121": ("baseline", "bn_tuning", "norm_tuning", "full_finetune", "lora", "tsa", "adapter", "bitfit", "ssf"),
}
SUPPORTED_FREEZE_STRATEGIES = {
    "linear_probe",
    "bn_tuning",
    "norm_tuning",
    "bn_tuning_with_last_stages",
    "frozen_backbone_peft",
    "bias_tuning",
    "full_finetune",
    "manual",
}
SUPPORTED_PEFT_METHODS = {None, "lora", "dora", "tsa", "adapter", "bitfit", "ssf"}

GENERATOR_VERSION = "phase_canvas_methods_v1"
SPEC_VERSION = "1.3"


def _base_family_for_model(base_model: str) -> str:
    if base_model.startswith("resnet"):
        return "resnet"
    if base_model.startswith("efficientnet"):
        return "efficientnet"
    if base_model.startswith("convnext"):
        return "convnext"
    if base_model.startswith("mobilenet"):
        return "mobilenet_v3"
    if base_model.startswith("densenet"):
        return "densenet"
    return "unknown"


@dataclass(frozen=True)
class CustomModelSpec:
    model_name: str
    base_provider: str
    base_family: str
    variant: str
    base_model: str
    task_type: str
    method_type: str
    freeze_strategy: str
    train_bn: bool
    train_norm: bool
    unfreeze_stages: list[int]
    peft_method: str | None
    peft_targets: dict[str, Any]
    peft_params: dict[str, Any]
    gradcam_target_hint: list[str]
    pretrained: bool = True
    metadata_version: str = SPEC_VERSION
    generator_version: str = GENERATOR_VERSION


@dataclass(frozen=True)
class GeneratedModelArtifacts:
    model_name: str
    model_file_path: Path
    spec_file_path: Path


def list_supported_base_models() -> list[str]:
    return sorted(SUPPORTED_BASE_MODELS)


def supported_methods_for_base(base_model: str) -> list[str]:
    normalized = _normalize_base_model(base_model)
    return list(SUPPORTED_METHODS_BY_BASE[normalized])


def default_spec_path_for_model_name(model_name: str) -> Path:
    return SPEC_DIR / f"{_normalize_model_name(model_name)}.json"


def list_saved_spec_files() -> list[Path]:
    if not SPEC_DIR.is_dir():
        return []
    return sorted(path for path in SPEC_DIR.glob("*.json") if path.is_file())


def resolve_spec_path_for_model_name(
    model_name: str,
    *,
    allow_legacy_mapping: bool = True,
    inspect_model_metadata: bool = True,
) -> Path:
    normalized_name = _normalize_model_name(model_name)
    candidates: list[Path] = [default_spec_path_for_model_name(normalized_name)]

    if allow_legacy_mapping:
        try:
            from core.model_registry import resolve_preferred_model_name

            preferred = resolve_preferred_model_name(normalized_name)
        except Exception:
            preferred = None
        if isinstance(preferred, str) and preferred and preferred != normalized_name:
            try:
                candidates.append(default_spec_path_for_model_name(preferred))
            except Exception:
                pass

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    if inspect_model_metadata:
        try:
            from core.model_registry import load_model_module

            module = load_model_module(normalized_name)
            metadata = module.get_model_metadata() if hasattr(module, "get_model_metadata") else {}
            if isinstance(metadata, dict):
                raw_spec_file = metadata.get("source_spec_file") or metadata.get("spec_file")
                if isinstance(raw_spec_file, str) and raw_spec_file.strip():
                    candidate = Path(raw_spec_file.strip()).expanduser()
                    if not candidate.is_absolute():
                        candidate = PROJECT_ROOT / candidate
                    candidate = candidate.resolve()
                    if candidate.is_file():
                        return candidate
        except Exception:
            pass

    attempted = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not resolve spec for model '{normalized_name}'. Tried:\n{attempted}"
    )


def load_spec_for_model_name(model_name: str) -> CustomModelSpec:
    spec_path = resolve_spec_path_for_model_name(model_name)
    return load_spec_file(spec_path)


def _normalize_model_name(name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_]", "_", str(name).strip()).strip("_").lower()
    normalized = re.sub(r"_+", "_", normalized)
    if not normalized or not re.match(r"^[a-z][a-z0-9_]*$", normalized):
        raise ValueError("Model Name must start with a letter and contain only letters, numbers, and underscores.")
    return normalized


def _normalize_base_model(base_model: str) -> str:
    normalized = str(base_model).strip().lower()
    canonical = BASE_MODEL_ALIASES.get(normalized)
    if canonical is None or canonical not in SUPPORTED_BASE_MODELS:
        raise ValueError(f"Unsupported base_model '{base_model}'. Supported: {', '.join(sorted(SUPPORTED_BASE_MODELS))}.")
    return canonical


def _parse_int_stage_list(raw_value: Any, *, min_value: int = 0, max_value: int = 7, field_name: str = "stages") -> list[int]:
    if isinstance(raw_value, list):
        values = raw_value
    else:
        text = str(raw_value).strip()
        if not text:
            return []
        values = [part.strip() for part in text.split(",")]
    stages: list[int] = []
    for item in values:
        try:
            stage = int(item)
        except Exception as exc:
            raise ValueError(f"Invalid {field_name} value '{item}'. Expected integer values.") from exc
        if stage < min_value or stage > max_value:
            raise ValueError(f"{field_name} values must be in range {min_value}..{max_value}.")
        stages.append(stage)
    return sorted(set(stages))


def _parse_string_list(raw_value: Any) -> list[str]:
    if isinstance(raw_value, list):
        values = raw_value
    else:
        text = str(raw_value).strip()
        if not text:
            return []
        values = [part.strip() for part in text.split(",")]
    items: list[str] = []
    for value in values:
        text = str(value).strip()
        if text:
            items.append(text)
    return items


def _default_gradcam_targets(base_model: str, method_type: str) -> list[str]:
    if base_model in {"resnet18", "resnet50"}:
        return ["layer4"]
    if base_model == "convnext_tiny":
        return ["features.7"]
    if method_type in {"bn_last1", "bn_last2"}:
        return ["features.7"]
    if base_model == "mobilenet_v3_large":
        return ["features.16"]
    if base_model == "densenet121":
        return ["features.denseblock4"]
    return ["features.7"]


def build_preset_spec(*, model_name: str, base_model: str, method_type: str) -> CustomModelSpec:
    normalized_name = _normalize_model_name(model_name)
    normalized_base = _normalize_base_model(base_model)
    normalized_method = str(method_type).strip().lower()
    allowed_methods = SUPPORTED_METHODS_BY_BASE[normalized_base]
    if normalized_method not in allowed_methods:
        raise ValueError(f"Method '{normalized_method}' is not supported for base_model '{normalized_base}'.")

    payload: dict[str, Any] = {
        "model_name": normalized_name,
        "base_provider": "torchvision",
        "base_family": _base_family_for_model(normalized_base),
        "variant": normalized_base,
        "base_model": normalized_base,
        "task_type": "classification",
        "method_type": normalized_method,
        "freeze_strategy": "manual",
        "train_bn": False,
        "train_norm": False,
        "unfreeze_stages": [],
        "peft_method": None,
        "peft_targets": {"feature_stages": [], "layer_keys": [], "classifier": False},
        "peft_params": {},
        "gradcam_target_hint": _default_gradcam_targets(normalized_base, normalized_method),
        "pretrained": True,
        "metadata_version": SPEC_VERSION,
        "generator_version": GENERATOR_VERSION,
    }

    if normalized_method == "baseline":
        payload["freeze_strategy"] = "linear_probe"
    elif normalized_method == "bn_tuning":
        payload["freeze_strategy"] = "bn_tuning"
        payload["train_bn"] = True
        payload["train_norm"] = True
    elif normalized_method == "norm_tuning":
        payload["freeze_strategy"] = "norm_tuning"
        payload["train_norm"] = True
    elif normalized_method == "bn_last1":
        payload["freeze_strategy"] = "bn_tuning_with_last_stages"
        payload["train_bn"] = True
        payload["train_norm"] = True
        payload["unfreeze_stages"] = [7]
    elif normalized_method == "bn_last2":
        payload["freeze_strategy"] = "bn_tuning_with_last_stages"
        payload["train_bn"] = True
        payload["train_norm"] = True
        payload["unfreeze_stages"] = [6, 7]
    elif normalized_method == "full_finetune":
        payload["freeze_strategy"] = "full_finetune"
    elif normalized_method == "lora":
        payload["freeze_strategy"] = "frozen_backbone_peft"
        payload["peft_method"] = "lora"
        if normalized_base == "efficientnet_v2_s":
            payload["peft_targets"] = {"feature_stages": [6, 7], "layer_keys": [], "classifier": True}
        elif normalized_base in {"resnet18", "resnet50"}:
            payload["peft_targets"] = {"feature_stages": [], "layer_keys": ["layer4"], "classifier": True}
        elif normalized_base == "convnext_tiny":
            payload["peft_targets"] = {"feature_stages": [], "layer_keys": ["stage4"], "classifier": True}
        elif normalized_base == "mobilenet_v3_large":
            payload["peft_targets"] = {"feature_stages": [], "layer_keys": ["stage4"], "classifier": True}
        elif normalized_base == "densenet121":
            payload["peft_targets"] = {"feature_stages": [], "layer_keys": ["denseblock4"], "classifier": True}
        else:
            payload["peft_targets"] = {"feature_stages": [], "layer_keys": [], "classifier": True}
        payload["peft_params"] = {"rank": 8, "alpha": 16.0}
    elif normalized_method == "dora":
        payload["freeze_strategy"] = "frozen_backbone_peft"
        payload["peft_method"] = "dora"
        payload["peft_targets"] = {"feature_stages": [6, 7], "layer_keys": [], "classifier": True}
        payload["peft_params"] = {"rank": 8, "alpha": 16.0}
    elif normalized_method == "tsa":
        payload["freeze_strategy"] = "frozen_backbone_peft"
        payload["peft_method"] = "tsa"
        if normalized_base == "efficientnet_v2_s":
            payload["peft_targets"] = {"feature_stages": [5, 6, 7], "layer_keys": [], "classifier": True}
        elif normalized_base in {"resnet18", "resnet50"}:
            payload["peft_targets"] = {"feature_stages": [], "layer_keys": ["layer3", "layer4"], "classifier": True}
        elif normalized_base == "convnext_tiny":
            payload["peft_targets"] = {"feature_stages": [], "layer_keys": ["stage3", "stage4"], "classifier": True}
        elif normalized_base == "mobilenet_v3_large":
            payload["peft_targets"] = {"feature_stages": [], "layer_keys": ["stage3", "stage4"], "classifier": True}
        elif normalized_base == "densenet121":
            payload["peft_targets"] = {"feature_stages": [], "layer_keys": ["denseblock3", "denseblock4"], "classifier": True}
        else:
            payload["peft_targets"] = {"feature_stages": [], "layer_keys": [], "classifier": True}
        payload["peft_params"] = {}
    elif normalized_method == "adapter":
        payload["freeze_strategy"] = "frozen_backbone_peft"
        payload["peft_method"] = "adapter"
        if normalized_base == "efficientnet_v2_s":
            payload["peft_targets"] = {"feature_stages": [5, 6, 7], "layer_keys": [], "classifier": True}
        elif normalized_base in {"resnet18", "resnet50"}:
            payload["peft_targets"] = {"feature_stages": [], "layer_keys": ["layer3", "layer4"], "classifier": True}
        elif normalized_base == "convnext_tiny":
            payload["peft_targets"] = {"feature_stages": [], "layer_keys": ["stage3", "stage4"], "classifier": True}
        elif normalized_base == "mobilenet_v3_large":
            payload["peft_targets"] = {"feature_stages": [], "layer_keys": ["stage3", "stage4"], "classifier": True}
        elif normalized_base == "densenet121":
            payload["peft_targets"] = {"feature_stages": [], "layer_keys": ["denseblock3", "denseblock4"], "classifier": True}
        payload["peft_params"] = {"bottleneck_dim": 32}
    elif normalized_method == "bitfit":
        payload["freeze_strategy"] = "bias_tuning"
        payload["peft_method"] = "bitfit"
        payload["peft_targets"] = {"feature_stages": [], "layer_keys": [], "classifier": True}
        payload["peft_params"] = {"scope": "all_bias"}
    elif normalized_method == "ssf":
        payload["freeze_strategy"] = "frozen_backbone_peft"
        payload["peft_method"] = "ssf"
        if normalized_base == "efficientnet_v2_s":
            payload["peft_targets"] = {"feature_stages": [5, 6, 7], "layer_keys": [], "classifier": True}
        elif normalized_base in {"resnet18", "resnet50"}:
            payload["peft_targets"] = {"feature_stages": [], "layer_keys": ["layer3", "layer4"], "classifier": True}
        elif normalized_base == "convnext_tiny":
            payload["peft_targets"] = {"feature_stages": [], "layer_keys": ["stage3", "stage4"], "classifier": True}
        elif normalized_base == "mobilenet_v3_large":
            payload["peft_targets"] = {"feature_stages": [], "layer_keys": ["stage3", "stage4"], "classifier": True}
        elif normalized_base == "densenet121":
            payload["peft_targets"] = {"feature_stages": [], "layer_keys": ["denseblock3", "denseblock4"], "classifier": True}
        payload["peft_params"] = {"init_scale": 1.0, "init_shift": 0.0}

    return spec_from_dict(payload)


def spec_to_dict(spec: CustomModelSpec) -> dict[str, Any]:
    return asdict(spec)


def spec_from_dict(payload: dict[str, Any]) -> CustomModelSpec:
    if not isinstance(payload, dict):
        raise ValueError("Spec payload must be an object/dict.")

    model_name = _normalize_model_name(payload.get("model_name", ""))
    base_model = _normalize_base_model(str(payload.get("base_model", "efficientnet_v2_s")))
    base_provider = str(payload.get("base_provider", "torchvision")).strip().lower() or "torchvision"
    if base_provider != "torchvision":
        raise ValueError("Current generator supports base_provider='torchvision' only.")
    base_family = str(payload.get("base_family", _base_family_for_model(base_model))).strip().lower() or _base_family_for_model(base_model)
    variant = str(payload.get("variant", base_model)).strip().lower() or base_model
    method_type = str(payload.get("method_type", "baseline")).strip().lower()
    if method_type not in SUPPORTED_METHODS_BY_BASE[base_model]:
        raise ValueError(f"Unsupported method_type '{method_type}' for base_model '{base_model}'.")

    freeze_strategy = str(payload.get("freeze_strategy", "manual")).strip().lower()
    if freeze_strategy not in SUPPORTED_FREEZE_STRATEGIES:
        raise ValueError(f"Unsupported freeze_strategy '{freeze_strategy}'.")

    task_type = str(payload.get("task_type", "classification")).strip().lower()
    if task_type != "classification":
        raise ValueError("Current generator supports task_type='classification' only.")

    train_bn = bool(payload.get("train_bn", False))
    train_norm = bool(payload.get("train_norm", train_bn))
    unfreeze_stages = _parse_int_stage_list(payload.get("unfreeze_stages", []), field_name="unfreeze_stages")

    raw_peft_method = payload.get("peft_method")
    peft_method = str(raw_peft_method).strip().lower() if raw_peft_method is not None and str(raw_peft_method).strip() else None
    if peft_method == "none":
        peft_method = None
    if peft_method not in SUPPORTED_PEFT_METHODS:
        raise ValueError(f"Unsupported peft_method '{peft_method}'.")

    expected_peft_method = {
        "lora": "lora",
        "dora": "dora",
        "tsa": "tsa",
        "adapter": "adapter",
        "bitfit": "bitfit",
        "ssf": "ssf",
    }.get(method_type)
    if expected_peft_method is None and peft_method is not None:
        raise ValueError(f"method_type='{method_type}' does not allow peft_method='{peft_method}'.")
    if expected_peft_method is not None and peft_method != expected_peft_method:
        raise ValueError(f"method_type='{method_type}' requires peft_method='{expected_peft_method}'.")
    if method_type == "dora" and base_model != "efficientnet_v2_s":
        raise ValueError("DoRA is currently supported for EfficientNet only.")

    raw_targets = payload.get("peft_targets")
    targets = raw_targets if isinstance(raw_targets, dict) else {}
    feature_stages = _parse_int_stage_list(targets.get("feature_stages", []), field_name="peft_targets.feature_stages")
    layer_keys = _parse_string_list(targets.get("layer_keys", []))
    classifier_target = bool(targets.get("classifier", False))
    peft_targets: dict[str, Any] = {
        "feature_stages": feature_stages,
        "layer_keys": layer_keys,
        "classifier": classifier_target,
    }

    raw_params = payload.get("peft_params")
    input_params = raw_params if isinstance(raw_params, dict) else {}
    peft_params: dict[str, Any] = {}
    if peft_method in {"lora", "dora"}:
        rank = int(input_params.get("rank", 8))
        alpha = float(input_params.get("alpha", 16.0))
        if rank <= 0:
            raise ValueError("peft_params.rank must be > 0.")
        if alpha <= 0:
            raise ValueError("peft_params.alpha must be > 0.")
        peft_params = {"rank": rank, "alpha": alpha}
    elif peft_method == "adapter":
        bottleneck_dim = int(input_params.get("bottleneck_dim", 32))
        if bottleneck_dim <= 0:
            raise ValueError("peft_params.bottleneck_dim must be > 0.")
        peft_params = {"bottleneck_dim": bottleneck_dim}
    elif peft_method == "bitfit":
        scope = str(input_params.get("scope", "all_bias")).strip().lower() or "all_bias"
        if scope not in {"all_bias", "norm_and_classifier_bias"}:
            raise ValueError("peft_params.scope must be one of: all_bias, norm_and_classifier_bias.")
        peft_params = {"scope": scope}
    elif peft_method == "ssf":
        init_scale = float(input_params.get("init_scale", 1.0))
        init_shift = float(input_params.get("init_shift", 0.0))
        peft_params = {"init_scale": init_scale, "init_shift": init_shift}

    if method_type == "dora":
        if base_model != "efficientnet_v2_s":
            raise ValueError("DoRA stage targets are only available for EfficientNet.")
        if not feature_stages and not classifier_target:
            raise ValueError("DoRA requires at least one feature stage or classifier target.")

    gradcam_target_hint = _parse_string_list(payload.get("gradcam_target_hint", []))
    if not gradcam_target_hint:
        gradcam_target_hint = _default_gradcam_targets(base_model, method_type)
    pretrained = bool(payload.get("pretrained", True))

    metadata_version = str(payload.get("metadata_version", SPEC_VERSION)).strip() or SPEC_VERSION
    generator_version = str(payload.get("generator_version", GENERATOR_VERSION)).strip() or GENERATOR_VERSION

    return CustomModelSpec(
        model_name=model_name,
        base_provider=base_provider,
        base_family=base_family,
        variant=variant,
        base_model=base_model,
        task_type=task_type,
        method_type=method_type,
        freeze_strategy=freeze_strategy,
        train_bn=train_bn,
        train_norm=train_norm,
        unfreeze_stages=unfreeze_stages,
        peft_method=peft_method,
        peft_targets=peft_targets,
        peft_params=peft_params,
        gradcam_target_hint=gradcam_target_hint,
        pretrained=pretrained,
        metadata_version=metadata_version,
        generator_version=generator_version,
    )


def load_spec_file(path: Path) -> CustomModelSpec:
    try:
        payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to read spec file: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Spec file must contain a JSON object.")
    return spec_from_dict(payload)


def save_spec_file(spec: CustomModelSpec, path: Path | None = None) -> Path:
    SPEC_DIR.mkdir(parents=True, exist_ok=True)
    target_path = path.expanduser().resolve() if path is not None else default_spec_path_for_model_name(spec.model_name).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(spec_to_dict(spec), indent=2, sort_keys=True), encoding="utf-8")
    return target_path


def build_phase1_spec(
    *,
    model_name: str,
    base_model: str,
    method_type: str,
    dora_feature_stages: str | list[int] | None = None,
    dora_include_classifier: bool = True,
    dora_rank: int = 8,
    dora_alpha: float = 16.0,
) -> CustomModelSpec:
    normalized_method = str(method_type).strip().lower()
    if normalized_method not in {"baseline", "bn_last1", "bn_last2", "dora"}:
        raise ValueError("build_phase1_spec supports baseline/bn_last1/bn_last2/dora only.")
    spec = build_preset_spec(model_name=model_name, base_model=base_model, method_type=normalized_method)
    if normalized_method != "dora":
        return spec

    payload = spec_to_dict(spec)
    payload["peft_targets"] = {
        "feature_stages": _parse_int_stage_list(dora_feature_stages if dora_feature_stages is not None else [6, 7], field_name="dora_feature_stages"),
        "layer_keys": [],
        "classifier": bool(dora_include_classifier),
    }
    payload["peft_params"] = {"rank": int(dora_rank), "alpha": float(dora_alpha)}
    feature_stages = payload["peft_targets"]["feature_stages"]
    payload["gradcam_target_hint"] = [f"features.{feature_stages[-1]}" if feature_stages else "features.7"]
    return spec_from_dict(payload)


def _render_model_template(spec: CustomModelSpec) -> str:
    spec_payload = asdict(spec)
    spec_payload.setdefault("is_generated", True)
    spec_payload.setdefault("source_of_truth", "spec")
    spec_payload.setdefault("spec_name", f"{spec.model_name}.json")
    spec_payload.setdefault("spec_file", f"model_specs/{spec.model_name}.json")
    spec_payload.setdefault("source_spec_file", f"model_specs/{spec.model_name}.json")
    spec_literal = repr(spec_payload)
    pretrained_literal = "True" if bool(spec.pretrained) else "False"
    gradcam_targets_literal = json.dumps(spec.gradcam_target_hint)

    return f'''from __future__ import annotations

import torch
from torch import nn

from model._transfer_strategies import (
    build_optimizer as _build_optimizer,
    build_model_from_spec as _build_model_from_spec,
)


GENERATED_SPEC = {spec_literal}


def _resolved_device(device: str | torch.device) -> str | torch.device:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def build_model(num_classes: int, freeze_backbone: bool = True, device: str | torch.device = "cpu") -> nn.Module:
    del freeze_backbone
    resolved_device = _resolved_device(device)
    pretrained = bool(GENERATED_SPEC.get("pretrained", {pretrained_literal}))
    return _build_model_from_spec(dict(GENERATED_SPEC), num_classes=num_classes, device=resolved_device, pretrained=pretrained)


def build_optimizer(model: nn.Module, lr: float = 1e-3) -> torch.optim.Optimizer:
    return _build_optimizer(model, lr=lr)


def get_model_metadata() -> dict[str, object]:
    metadata = dict(GENERATED_SPEC)
    metadata.setdefault("is_generated", True)
    metadata.setdefault("source_of_truth", "spec")
    metadata.setdefault("spec_name", f"{{metadata.get('model_name', 'unknown')}}.json")
    metadata.setdefault("spec_file", f"model_specs/{{metadata.get('model_name', 'unknown')}}.json")
    metadata.setdefault("source_spec_file", f"model_specs/{{metadata.get('model_name', 'unknown')}}.json")
    return metadata


def get_capabilities() -> dict[str, bool]:
    method_type = str(GENERATED_SPEC.get("method_type", "baseline"))
    return {{
        "supports_resume": True,
        "supports_gradcam": True,
        "supports_structure_editing": True,
        "supports_lora": method_type == "lora",
        "supports_dora": method_type == "dora",
        "supports_tsa": method_type == "tsa",
        "supports_adapter": method_type == "adapter",
        "supports_bitfit": method_type == "bitfit",
        "supports_ssf": method_type == "ssf",
        "supports_bn_tuning": method_type in {{"bn_tuning", "bn_last1", "bn_last2"}},
        "supports_norm_tuning": method_type == "norm_tuning",
    }}


def describe_model_structure() -> dict[str, object]:
    base_model = str(GENERATED_SPEC.get("base_model", "efficientnet_v2_s"))
    if base_model == "resnet18":
        return {{
            "base_family": "resnet18",
            "feature_stages": ["conv1", "layer1", "layer2", "layer3", "layer4"],
            "classifier": "fc",
        }}
    if base_model == "resnet50":
        return {{
            "base_family": "resnet50",
            "feature_stages": ["conv1", "layer1", "layer2", "layer3", "layer4"],
            "classifier": "fc",
        }}
    if base_model == "convnext_tiny":
        return {{
            "base_family": "convnext_tiny",
            "feature_stages": ["stem", "stage1", "stage2", "stage3", "stage4"],
            "classifier": "classifier.2",
        }}
    if base_model == "mobilenet_v3_large":
        return {{
            "base_family": "mobilenet_v3_large",
            "feature_stages": ["stem", "stage1", "stage2", "stage3", "stage4"],
            "classifier": "classifier.3",
        }}
    if base_model == "densenet121":
        return {{
            "base_family": "densenet121",
            "feature_stages": ["stem", "denseblock1", "denseblock2", "denseblock3", "denseblock4"],
            "classifier": "classifier",
        }}
    return {{
        "base_family": "efficientnet_v2_s",
        "feature_stages": [f"features.{{idx}}" for idx in range(8)],
        "classifier": "classifier.1",
    }}


def get_default_gradcam_targets() -> list[str]:
    return list({gradcam_targets_literal})
'''


def generate_custom_model(spec: CustomModelSpec, *, overwrite: bool = False) -> GeneratedModelArtifacts:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SPEC_DIR.mkdir(parents=True, exist_ok=True)

    model_file_path = MODEL_DIR / f"{spec.model_name}.py"
    spec_file_path = SPEC_DIR / f"{spec.model_name}.json"

    if model_file_path.exists() and not overwrite:
        raise FileExistsError(f"Model file already exists: {model_file_path}")

    model_file_path.write_text(_render_model_template(spec), encoding="utf-8")
    save_spec_file(spec, spec_file_path)

    return GeneratedModelArtifacts(
        model_name=spec.model_name,
        model_file_path=model_file_path,
        spec_file_path=spec_file_path,
    )
