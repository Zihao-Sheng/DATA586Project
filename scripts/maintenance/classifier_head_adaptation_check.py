from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(1, str(ROOT))

from core.custom_model_generator import build_preset_spec, generate_custom_model  # noqa: E402
from core.model_registry import load_model_module  # noqa: E402


GENERATED_CASES = [
    ("resnet18", "baseline"),
    ("resnet50", "baseline"),
    ("efficientnet_v2_s", "baseline"),
    ("convnext_tiny", "baseline"),
    ("mobilenet_v3_large", "baseline"),
    ("densenet121", "baseline"),
]

LEGACY_MODELS = [
    "EfficientNet_Baseline",
    "EfficientNet_BN_Last1",
    "EfficientNet_BN_Last2",
    "EfficientNet_BN_Tuning",
    "EfficientNet_DoRA",
    "EfficientNet_Full_Finetune",
    "EfficientNet_LoRA",
    "EfficientNet_TSA",
    "ResNet18_Baseline",
    "ResNet18_BN_Tuning",
    "ResNet18_Full_Finetune",
    "ResNet18_LoRA",
    "ResNet18_TSA",
]


def _validate_model_module(model_name: str) -> dict[str, object]:
    module = load_model_module(model_name)
    model = module.build_model(num_classes=101, freeze_backbone=True, device="cpu")
    if not hasattr(module, "get_classifier_info"):
        raise AssertionError(f"{model_name}: missing get_classifier_info")
    if not hasattr(module, "replace_classifier_head"):
        raise AssertionError(f"{model_name}: missing replace_classifier_head")
    if not hasattr(module, "get_feature_dim"):
        raise AssertionError(f"{model_name}: missing get_feature_dim")
    if not hasattr(module, "get_head_module_path"):
        raise AssertionError(f"{model_name}: missing get_head_module_path")

    before = module.get_classifier_info(model)
    feature_dim = int(module.get_feature_dim(model))
    if feature_dim != int(before.get("feature_dim", -1)):
        raise AssertionError(f"{model_name}: feature_dim mismatch")
    module.replace_classifier_head(model, 37)
    after = module.get_classifier_info(model)
    if int(after.get("num_classes", -1)) != 37:
        raise AssertionError(f"{model_name}: replacement output dim mismatch")
    return {
        "model_name": model_name,
        "head_path": before.get("head_module_path"),
        "feature_dim": feature_dim,
        "num_classes_before": int(before.get("num_classes", -1)),
        "num_classes_after": int(after.get("num_classes", -1)),
        "base_model": before.get("base_model"),
    }


def main() -> None:
    created_models: list[str] = []
    created_specs: list[Path] = []
    report: dict[str, object] = {"generated": [], "legacy": []}
    try:
        for base_model, method_type in GENERATED_CASES:
            model_name = f"zz_headcheck_{base_model}"
            spec = build_preset_spec(model_name=model_name, base_model=base_model, method_type=method_type)
            artifacts = generate_custom_model(spec, overwrite=True)
            created_models.append(artifacts.model_name)
            created_specs.append(artifacts.spec_file_path)
            report["generated"].append(_validate_model_module(artifacts.model_name))

        for model_name in LEGACY_MODELS:
            report["legacy"].append(_validate_model_module(model_name))

        print(json.dumps({"status": "ok", "report": report}, indent=2))
    finally:
        model_dir = PROJECT_ROOT / "model"
        for model_name in created_models:
            target = model_dir / f"{model_name}.py"
            if target.exists():
                target.unlink()
            cached = model_dir / "__pycache__"
            if cached.exists():
                for pyc in cached.glob(f"{model_name}*.pyc"):
                    pyc.unlink()
        for path in created_specs:
            if path.exists():
                path.unlink()


if __name__ == "__main__":
    main()
