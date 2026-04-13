from __future__ import annotations

import re
from pathlib import Path
from types import ModuleType
from typing import Any


SAFE_OPS_BY_STAGE_DEFAULT: dict[str, tuple[str, ...]] = {
    "fixed": (),
    "conservative": ("freeze", "unfreeze"),
    "bn": ("freeze", "unfreeze", "bn_tuning", "norm_tuning"),
    "peft": ("freeze", "unfreeze", "bn_tuning", "norm_tuning", "lora", "tsa", "adapter", "bitfit", "ssf"),
    "peft_dora": ("freeze", "unfreeze", "bn_tuning", "norm_tuning", "lora", "dora", "tsa", "adapter", "bitfit", "ssf"),
}


def _safe_ops_for_stage(
    *,
    has_bn: bool,
    has_conv: bool,
    has_linear: bool,
    allow_dora: bool,
    allow_peft: bool,
    conservative: bool,
) -> list[str]:
    if conservative:
        return list(SAFE_OPS_BY_STAGE_DEFAULT["conservative"])
    ops: list[str] = ["freeze", "unfreeze"]
    if has_bn:
        ops.append("bn_tuning")
    if has_bn or has_conv or has_linear:
        ops.append("norm_tuning")
    if allow_peft and (has_conv or has_linear):
        ops.extend(["lora", "tsa", "adapter", "bitfit", "ssf"])
        if allow_dora:
            ops.append("dora")
    return ops


def _stage(
    key: str,
    title: str,
    *,
    stage_type: str,
    source_module: str,
    module_family_hints: list[str],
    has_bn: bool,
    has_conv: bool,
    has_linear: bool,
    editable: bool,
    safe_operations: list[str],
    param_count: int | None = None,
) -> dict[str, Any]:
    return {
        "key": key,
        "title": title,
        "stage_type": stage_type,
        "source_module": source_module,
        "module_family_hints": module_family_hints,
        "param_count": param_count,
        "has_bn": bool(has_bn),
        "has_conv": bool(has_conv),
        "has_linear": bool(has_linear),
        "editable": bool(editable),
        "safe_operations": list(safe_operations),
    }


def _family_from_text(text: str) -> str:
    source = text.lower()
    if any(token in source for token in ("efficientnet_v2_s", "strategy_builder(\"efficientnet", "efficientnet")):
        return "efficientnet_v2_s"
    if any(token in source for token in ("convnext_tiny", "strategy_builder(\"convnext_tiny")):
        return "convnext_tiny"
    if any(token in source for token in ("mobilenet_v3_large", "strategy_builder(\"mobilenet_v3_large")):
        return "mobilenet_v3_large"
    if any(token in source for token in ("densenet121", "strategy_builder(\"densenet121")):
        return "densenet121"
    if any(token in source for token in ("resnet50", "strategy_builder(\"resnet50")):
        return "resnet50"
    if any(token in source for token in ("resnet18", "strategy_builder(\"resnet18", "layer4", ".fc")):
        return "resnet18"
    return "unknown"


def _feature_stage_count_from_text(text: str) -> int:
    matches = re.findall(r"features(?:\.|\[)(\d+)", text, flags=re.IGNORECASE)
    if not matches:
        return 8
    try:
        max_idx = max(int(raw) for raw in matches)
    except Exception:
        return 8
    return max(1, min(16, max_idx + 1))


def _explicit_structure_from_module(module: ModuleType) -> dict[str, Any] | None:
    if not hasattr(module, "describe_model_structure"):
        return None
    try:
        payload = module.describe_model_structure()
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None

    base_family = str(payload.get("base_family", "")).strip().lower()
    feature_stages = payload.get("feature_stages")
    classifier = str(payload.get("classifier", "classifier")).strip() or "classifier"
    if not isinstance(feature_stages, list):
        feature_stages = []

    if "resnet50" in base_family:
        normalized_base = "resnet50"
    elif "resnet" in base_family:
        normalized_base = "resnet18"
    elif "efficientnet" in base_family:
        normalized_base = "efficientnet_v2_s"
    elif "convnext" in base_family:
        normalized_base = "convnext_tiny"
    elif "mobilenet_v3_large" in base_family:
        normalized_base = "mobilenet_v3_large"
    elif "densenet121" in base_family or "densenet" in base_family:
        normalized_base = "densenet121"
    else:
        normalized_base = "unknown"

    stages: list[dict[str, Any]] = [
        _stage(
            "input",
            "Input",
            stage_type="input",
            source_module="",
            module_family_hints=["input"],
            has_bn=False,
            has_conv=False,
            has_linear=False,
            editable=False,
            safe_operations=[],
            param_count=0,
        )
    ]

    if normalized_base == "efficientnet_v2_s":
        stages.append(
            _stage(
                "stem",
                "Stem",
                stage_type="stem",
                source_module="features.0",
                module_family_hints=["conv", "bn", "mbconv"],
                has_bn=True,
                has_conv=True,
                has_linear=False,
                editable=True,
                safe_operations=_safe_ops_for_stage(
                    has_bn=True,
                    has_conv=True,
                    has_linear=False,
                    allow_dora=False,
                    allow_peft=False,
                    conservative=False,
                ),
            )
        )
        for idx, raw_stage in enumerate(feature_stages):
            stage_key = str(raw_stage).strip() or f"features.{idx}"
            title = f"Features[{idx}]"
            allow_peft = idx >= 5
            allow_dora = idx >= 6
            stages.append(
                _stage(
                    stage_key,
                    title,
                    stage_type="backbone_stage",
                    source_module=stage_key,
                    module_family_hints=["mbconv", "conv", "bn"],
                    has_bn=True,
                    has_conv=True,
                    has_linear=False,
                    editable=True,
                    safe_operations=_safe_ops_for_stage(
                        has_bn=True,
                        has_conv=True,
                        has_linear=False,
                        allow_dora=allow_dora,
                        allow_peft=allow_peft,
                        conservative=False,
                    ),
                )
            )
        stages.append(
            _stage(
                "classifier",
                "Classifier",
                stage_type="head",
                source_module=classifier,
                module_family_hints=["linear", "head"],
                has_bn=False,
                has_conv=False,
                has_linear=True,
                editable=True,
                safe_operations=_safe_ops_for_stage(
                    has_bn=False,
                    has_conv=False,
                    has_linear=True,
                    allow_dora=True,
                    allow_peft=True,
                    conservative=False,
                ),
            )
        )
    elif normalized_base in {"resnet18", "resnet50"}:
        stages.extend(
            [
                _stage(
                    "stem",
                    "Stem / Conv1",
                    stage_type="stem",
                    source_module="conv1",
                    module_family_hints=["conv", "bn"],
                    has_bn=True,
                    has_conv=True,
                    has_linear=False,
                    editable=True,
                    safe_operations=_safe_ops_for_stage(
                        has_bn=True,
                        has_conv=True,
                        has_linear=False,
                        allow_dora=False,
                        allow_peft=False,
                        conservative=False,
                    ),
                ),
                _stage(
                    "layer1",
                    "Layer1",
                    stage_type="backbone_stage",
                    source_module="layer1",
                    module_family_hints=["residual", "conv", "bn"],
                    has_bn=True,
                    has_conv=True,
                    has_linear=False,
                    editable=True,
                    safe_operations=_safe_ops_for_stage(
                        has_bn=True,
                        has_conv=True,
                        has_linear=False,
                        allow_dora=False,
                        allow_peft=False,
                        conservative=False,
                    ),
                ),
                _stage(
                    "layer2",
                    "Layer2",
                    stage_type="backbone_stage",
                    source_module="layer2",
                    module_family_hints=["residual", "conv", "bn"],
                    has_bn=True,
                    has_conv=True,
                    has_linear=False,
                    editable=True,
                    safe_operations=_safe_ops_for_stage(
                        has_bn=True,
                        has_conv=True,
                        has_linear=False,
                        allow_dora=False,
                        allow_peft=False,
                        conservative=False,
                    ),
                ),
                _stage(
                    "layer3",
                    "Layer3",
                    stage_type="backbone_stage",
                    source_module="layer3",
                    module_family_hints=["residual", "conv", "bn"],
                    has_bn=True,
                    has_conv=True,
                    has_linear=False,
                    editable=True,
                    safe_operations=_safe_ops_for_stage(
                        has_bn=True,
                        has_conv=True,
                        has_linear=False,
                        allow_dora=False,
                        allow_peft=True,
                        conservative=False,
                    ),
                ),
                _stage(
                    "layer4",
                    "Layer4",
                    stage_type="backbone_stage",
                    source_module="layer4",
                    module_family_hints=["residual", "conv", "bn"],
                    has_bn=True,
                    has_conv=True,
                    has_linear=False,
                    editable=True,
                    safe_operations=_safe_ops_for_stage(
                        has_bn=True,
                        has_conv=True,
                        has_linear=False,
                        allow_dora=False,
                        allow_peft=True,
                        conservative=False,
                    ),
                ),
                _stage(
                    "classifier",
                    "FC / Classifier",
                    stage_type="head",
                    source_module=classifier,
                    module_family_hints=["linear", "head"],
                    has_bn=False,
                    has_conv=False,
                    has_linear=True,
                    editable=True,
                    safe_operations=_safe_ops_for_stage(
                        has_bn=False,
                        has_conv=False,
                        has_linear=True,
                        allow_dora=False,
                        allow_peft=True,
                        conservative=False,
                    ),
                ),
            ]
        )
    elif normalized_base == "convnext_tiny":
        stages.extend(
            [
                _stage("stem", "Stem", stage_type="stem", source_module="features.0", module_family_hints=["conv"], has_bn=False, has_conv=True, has_linear=False, editable=True, safe_operations=_safe_ops_for_stage(has_bn=False, has_conv=True, has_linear=False, allow_dora=False, allow_peft=False, conservative=False)),
                _stage("stage1", "Stage1", stage_type="backbone_stage", source_module="features.1", module_family_hints=["convnext"], has_bn=False, has_conv=True, has_linear=False, editable=True, safe_operations=_safe_ops_for_stage(has_bn=False, has_conv=True, has_linear=False, allow_dora=False, allow_peft=False, conservative=False)),
                _stage("stage2", "Stage2", stage_type="backbone_stage", source_module="features.3", module_family_hints=["convnext"], has_bn=False, has_conv=True, has_linear=False, editable=True, safe_operations=_safe_ops_for_stage(has_bn=False, has_conv=True, has_linear=False, allow_dora=False, allow_peft=False, conservative=False)),
                _stage("stage3", "Stage3", stage_type="backbone_stage", source_module="features.5", module_family_hints=["convnext"], has_bn=False, has_conv=True, has_linear=False, editable=True, safe_operations=_safe_ops_for_stage(has_bn=False, has_conv=True, has_linear=False, allow_dora=False, allow_peft=True, conservative=False)),
                _stage("stage4", "Stage4", stage_type="backbone_stage", source_module="features.7", module_family_hints=["convnext"], has_bn=False, has_conv=True, has_linear=False, editable=True, safe_operations=_safe_ops_for_stage(has_bn=False, has_conv=True, has_linear=False, allow_dora=False, allow_peft=True, conservative=False)),
                _stage("classifier", "Classifier", stage_type="head", source_module=classifier, module_family_hints=["linear", "head"], has_bn=False, has_conv=False, has_linear=True, editable=True, safe_operations=_safe_ops_for_stage(has_bn=False, has_conv=False, has_linear=True, allow_dora=False, allow_peft=True, conservative=False)),
            ]
        )
    elif normalized_base == "mobilenet_v3_large":
        stages.extend(
            [
                _stage("stem", "Stem", stage_type="stem", source_module="features.0", module_family_hints=["conv", "bn"], has_bn=True, has_conv=True, has_linear=False, editable=True, safe_operations=_safe_ops_for_stage(has_bn=True, has_conv=True, has_linear=False, allow_dora=False, allow_peft=False, conservative=False)),
                _stage("stage1", "Feature Stage1", stage_type="backbone_stage", source_module="features.3", module_family_hints=["mobilenet_block", "conv", "bn"], has_bn=True, has_conv=True, has_linear=False, editable=True, safe_operations=_safe_ops_for_stage(has_bn=True, has_conv=True, has_linear=False, allow_dora=False, allow_peft=False, conservative=False)),
                _stage("stage2", "Feature Stage2", stage_type="backbone_stage", source_module="features.6", module_family_hints=["mobilenet_block", "conv", "bn"], has_bn=True, has_conv=True, has_linear=False, editable=True, safe_operations=_safe_ops_for_stage(has_bn=True, has_conv=True, has_linear=False, allow_dora=False, allow_peft=False, conservative=False)),
                _stage("stage3", "Feature Stage3", stage_type="backbone_stage", source_module="features.12", module_family_hints=["mobilenet_block", "conv", "bn"], has_bn=True, has_conv=True, has_linear=False, editable=True, safe_operations=_safe_ops_for_stage(has_bn=True, has_conv=True, has_linear=False, allow_dora=False, allow_peft=True, conservative=False)),
                _stage("stage4", "Feature Stage4", stage_type="backbone_stage", source_module="features.16", module_family_hints=["mobilenet_block", "conv", "bn"], has_bn=True, has_conv=True, has_linear=False, editable=True, safe_operations=_safe_ops_for_stage(has_bn=True, has_conv=True, has_linear=False, allow_dora=False, allow_peft=True, conservative=False)),
                _stage("classifier", "Classifier", stage_type="head", source_module=classifier, module_family_hints=["linear", "head"], has_bn=False, has_conv=False, has_linear=True, editable=True, safe_operations=_safe_ops_for_stage(has_bn=False, has_conv=False, has_linear=True, allow_dora=False, allow_peft=True, conservative=False)),
            ]
        )
    elif normalized_base == "densenet121":
        stages.extend(
            [
                _stage("stem", "Stem", stage_type="stem", source_module="features.conv0", module_family_hints=["conv", "bn"], has_bn=True, has_conv=True, has_linear=False, editable=True, safe_operations=_safe_ops_for_stage(has_bn=True, has_conv=True, has_linear=False, allow_dora=False, allow_peft=False, conservative=False)),
                _stage("denseblock1", "DenseBlock1", stage_type="backbone_stage", source_module="features.denseblock1", module_family_hints=["denseblock", "conv", "bn"], has_bn=True, has_conv=True, has_linear=False, editable=True, safe_operations=_safe_ops_for_stage(has_bn=True, has_conv=True, has_linear=False, allow_dora=False, allow_peft=False, conservative=False)),
                _stage("denseblock2", "DenseBlock2", stage_type="backbone_stage", source_module="features.denseblock2", module_family_hints=["denseblock", "conv", "bn"], has_bn=True, has_conv=True, has_linear=False, editable=True, safe_operations=_safe_ops_for_stage(has_bn=True, has_conv=True, has_linear=False, allow_dora=False, allow_peft=False, conservative=False)),
                _stage("denseblock3", "DenseBlock3", stage_type="backbone_stage", source_module="features.denseblock3", module_family_hints=["denseblock", "conv", "bn"], has_bn=True, has_conv=True, has_linear=False, editable=True, safe_operations=_safe_ops_for_stage(has_bn=True, has_conv=True, has_linear=False, allow_dora=False, allow_peft=True, conservative=False)),
                _stage("denseblock4", "DenseBlock4", stage_type="backbone_stage", source_module="features.denseblock4", module_family_hints=["denseblock", "conv", "bn"], has_bn=True, has_conv=True, has_linear=False, editable=True, safe_operations=_safe_ops_for_stage(has_bn=True, has_conv=True, has_linear=False, allow_dora=False, allow_peft=True, conservative=False)),
                _stage("classifier", "Classifier", stage_type="head", source_module=classifier, module_family_hints=["linear", "head"], has_bn=False, has_conv=False, has_linear=True, editable=True, safe_operations=_safe_ops_for_stage(has_bn=False, has_conv=False, has_linear=True, allow_dora=False, allow_peft=True, conservative=False)),
            ]
        )
    else:
        return None

    stages.append(
        _stage(
            "output",
            "Output",
            stage_type="output",
            source_module="",
            module_family_hints=["output"],
            has_bn=False,
            has_conv=False,
            has_linear=False,
            editable=False,
            safe_operations=[],
            param_count=0,
        )
    )
    return {
        "base_family": normalized_base,
        "structure_source": "explicit",
        "confidence": "high",
        "stages": stages,
    }


def _heuristic_structure_from_source(source_text: str) -> dict[str, Any]:
    family = _family_from_text(source_text)

    if family == "efficientnet_v2_s":
        feature_count = _feature_stage_count_from_text(source_text)
        stages: list[dict[str, Any]] = [
            _stage(
                "input",
                "Input",
                stage_type="input",
                source_module="",
                module_family_hints=["input"],
                has_bn=False,
                has_conv=False,
                has_linear=False,
                editable=False,
                safe_operations=[],
                param_count=0,
            ),
            _stage(
                "stem",
                "Stem",
                stage_type="stem",
                source_module="features.0",
                module_family_hints=["conv", "bn", "mbconv"],
                has_bn=True,
                has_conv=True,
                has_linear=False,
                editable=True,
                safe_operations=_safe_ops_for_stage(
                    has_bn=True,
                    has_conv=True,
                    has_linear=False,
                    allow_dora=False,
                    allow_peft=False,
                    conservative=False,
                ),
            ),
        ]
        for idx in range(feature_count):
            stage_key = f"features.{idx}"
            allow_peft = idx >= max(0, feature_count - 3)
            allow_dora = idx >= max(0, feature_count - 2)
            stages.append(
                _stage(
                    stage_key,
                    f"Features[{idx}]",
                    stage_type="backbone_stage",
                    source_module=stage_key,
                    module_family_hints=["mbconv", "conv", "bn"],
                    has_bn=True,
                    has_conv=True,
                    has_linear=False,
                    editable=True,
                    safe_operations=_safe_ops_for_stage(
                        has_bn=True,
                        has_conv=True,
                        has_linear=False,
                        allow_dora=allow_dora,
                        allow_peft=allow_peft,
                        conservative=False,
                    ),
                )
            )
        stages.extend(
            [
                _stage(
                    "classifier",
                    "Classifier",
                    stage_type="head",
                    source_module="classifier",
                    module_family_hints=["linear", "head"],
                    has_bn=False,
                    has_conv=False,
                    has_linear=True,
                    editable=True,
                    safe_operations=_safe_ops_for_stage(
                        has_bn=False,
                        has_conv=False,
                        has_linear=True,
                        allow_dora=True,
                        allow_peft=True,
                        conservative=False,
                    ),
                ),
                _stage(
                    "output",
                    "Output",
                    stage_type="output",
                    source_module="",
                    module_family_hints=["output"],
                    has_bn=False,
                    has_conv=False,
                    has_linear=False,
                    editable=False,
                    safe_operations=[],
                    param_count=0,
                ),
            ]
        )
        return {
            "base_family": "efficientnet_v2_s",
            "structure_source": "heuristic",
            "confidence": "medium",
            "stages": stages,
        }

    if family == "resnet18":
        stages = [
            _stage(
                "input",
                "Input",
                stage_type="input",
                source_module="",
                module_family_hints=["input"],
                has_bn=False,
                has_conv=False,
                has_linear=False,
                editable=False,
                safe_operations=[],
                param_count=0,
            ),
            _stage(
                "stem",
                "Stem / Conv1",
                stage_type="stem",
                source_module="conv1",
                module_family_hints=["conv", "bn"],
                has_bn=True,
                has_conv=True,
                has_linear=False,
                editable=True,
                safe_operations=_safe_ops_for_stage(
                    has_bn=True,
                    has_conv=True,
                    has_linear=False,
                    allow_dora=False,
                    allow_peft=False,
                    conservative=False,
                ),
            ),
            _stage(
                "layer1",
                "Layer1",
                stage_type="backbone_stage",
                source_module="layer1",
                module_family_hints=["residual", "conv", "bn"],
                has_bn=True,
                has_conv=True,
                has_linear=False,
                editable=True,
                safe_operations=_safe_ops_for_stage(
                    has_bn=True,
                    has_conv=True,
                    has_linear=False,
                    allow_dora=False,
                    allow_peft=False,
                    conservative=False,
                ),
            ),
            _stage(
                "layer2",
                "Layer2",
                stage_type="backbone_stage",
                source_module="layer2",
                module_family_hints=["residual", "conv", "bn"],
                has_bn=True,
                has_conv=True,
                has_linear=False,
                editable=True,
                safe_operations=_safe_ops_for_stage(
                    has_bn=True,
                    has_conv=True,
                    has_linear=False,
                    allow_dora=False,
                    allow_peft=False,
                    conservative=False,
                ),
            ),
            _stage(
                "layer3",
                "Layer3",
                stage_type="backbone_stage",
                source_module="layer3",
                module_family_hints=["residual", "conv", "bn"],
                has_bn=True,
                has_conv=True,
                has_linear=False,
                editable=True,
                safe_operations=_safe_ops_for_stage(
                    has_bn=True,
                    has_conv=True,
                    has_linear=False,
                    allow_dora=False,
                    allow_peft=True,
                    conservative=False,
                ),
            ),
            _stage(
                "layer4",
                "Layer4",
                stage_type="backbone_stage",
                source_module="layer4",
                module_family_hints=["residual", "conv", "bn"],
                has_bn=True,
                has_conv=True,
                has_linear=False,
                editable=True,
                safe_operations=_safe_ops_for_stage(
                    has_bn=True,
                    has_conv=True,
                    has_linear=False,
                    allow_dora=False,
                    allow_peft=True,
                    conservative=False,
                ),
            ),
            _stage(
                "classifier",
                "FC / Classifier",
                stage_type="head",
                source_module="fc",
                module_family_hints=["linear", "head"],
                has_bn=False,
                has_conv=False,
                has_linear=True,
                editable=True,
                safe_operations=_safe_ops_for_stage(
                    has_bn=False,
                    has_conv=False,
                    has_linear=True,
                    allow_dora=False,
                    allow_peft=True,
                    conservative=False,
                ),
            ),
            _stage(
                "output",
                "Output",
                stage_type="output",
                source_module="",
                module_family_hints=["output"],
                has_bn=False,
                has_conv=False,
                has_linear=False,
                editable=False,
                safe_operations=[],
                param_count=0,
            ),
        ]
        return {
            "base_family": "resnet18",
            "structure_source": "heuristic",
            "confidence": "medium",
            "stages": stages,
        }

    has_bn_token = "batchnorm" in source_text.lower() or " bn" in source_text.lower() or ".bn" in source_text.lower()
    stages = [
        _stage(
            "input",
            "Input",
            stage_type="input",
            source_module="",
            module_family_hints=["input"],
            has_bn=False,
            has_conv=False,
            has_linear=False,
            editable=False,
            safe_operations=[],
            param_count=0,
        ),
        _stage(
            "feature_extractor_stage1",
            "Feature Extractor Stage 1",
            stage_type="backbone_stage",
            source_module="features.0",
            module_family_hints=["feature_extractor"],
            has_bn=has_bn_token,
            has_conv=True,
            has_linear=False,
            editable=True,
            safe_operations=_safe_ops_for_stage(
                has_bn=has_bn_token,
                has_conv=True,
                has_linear=False,
                allow_dora=False,
                allow_peft=False,
                conservative=True,
            ),
        ),
        _stage(
            "feature_extractor_stage2",
            "Feature Extractor Stage 2",
            stage_type="backbone_stage",
            source_module="features.1",
            module_family_hints=["feature_extractor"],
            has_bn=has_bn_token,
            has_conv=True,
            has_linear=False,
            editable=True,
            safe_operations=_safe_ops_for_stage(
                has_bn=has_bn_token,
                has_conv=True,
                has_linear=False,
                allow_dora=False,
                allow_peft=False,
                conservative=True,
            ),
        ),
        _stage(
            "head",
            "Head / Classifier",
            stage_type="head",
            source_module="head",
            module_family_hints=["head", "linear"],
            has_bn=False,
            has_conv=False,
            has_linear=True,
            editable=True,
            safe_operations=_safe_ops_for_stage(
                has_bn=False,
                has_conv=False,
                has_linear=True,
                allow_dora=False,
                allow_peft=False,
                conservative=True,
            ),
        ),
        _stage(
            "output",
            "Output",
            stage_type="output",
            source_module="",
            module_family_hints=["output"],
            has_bn=False,
            has_conv=False,
            has_linear=False,
            editable=False,
            safe_operations=[],
            param_count=0,
        ),
    ]
    return {
        "base_family": "unknown",
        "structure_source": "fallback",
        "confidence": "low",
        "stages": stages,
    }


def describe_model_structure_for_canvas(
    *,
    model_name: str,
    module: ModuleType,
    module_path: Path,
) -> dict[str, Any]:
    explicit = _explicit_structure_from_module(module)
    if explicit is not None:
        return {
            "model_name": model_name,
            "base_family": explicit.get("base_family", "unknown"),
            "structure_source": explicit.get("structure_source", "explicit"),
            "confidence": explicit.get("confidence", "high"),
            "stages": explicit.get("stages", []),
        }

    try:
        source_text = module_path.read_text(encoding="utf-8")
    except Exception:
        source_text = ""
    heuristic = _heuristic_structure_from_source(source_text)
    return {
        "model_name": model_name,
        "base_family": heuristic.get("base_family", "unknown"),
        "structure_source": heuristic.get("structure_source", "fallback"),
        "confidence": heuristic.get("confidence", "low"),
        "stages": heuristic.get("stages", []),
    }
