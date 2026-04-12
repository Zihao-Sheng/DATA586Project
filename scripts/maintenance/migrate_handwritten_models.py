from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    insert_at = 1 if sys.path and sys.path[0] == str(PROJECT_ROOT) else 0
    sys.path.insert(insert_at, str(SCRIPTS_ROOT))

from core import custom_model_generator
from core.model_registry import discover_model_names, load_model_module


MODEL_DIR = PROJECT_ROOT / "model"
MODEL_SPECS_DIR = PROJECT_ROOT / "model_specs"
MIGRATION_MAP_PATH = MODEL_SPECS_DIR / "legacy_migration_map.json"
MIGRATION_REPORT_PATH = MODEL_SPECS_DIR / "legacy_migration_report.json"


@dataclass(frozen=True)
class MigrationTarget:
    handwritten_stem: str
    base_model: str
    method_type: str
    generated_model_name: str


MIGRATION_TARGETS: list[MigrationTarget] = [
    MigrationTarget("EfficientNet_Baseline", "efficientnet_v2_s", "baseline", "efficientnet_baseline_gen"),
    MigrationTarget("EfficientNet_BN_Last1", "efficientnet_v2_s", "bn_last1", "efficientnet_bn_last1_gen"),
    MigrationTarget("EfficientNet_BN_Last2", "efficientnet_v2_s", "bn_last2", "efficientnet_bn_last2_gen"),
    MigrationTarget("EfficientNet_BN_Tuning", "efficientnet_v2_s", "bn_tuning", "efficientnet_bn_tuning_gen"),
    MigrationTarget("EfficientNet_DoRA", "efficientnet_v2_s", "dora", "efficientnet_dora_gen"),
    MigrationTarget("EfficientNet_Full_Finetune", "efficientnet_v2_s", "full_finetune", "efficientnet_full_finetune_gen"),
    MigrationTarget("EfficientNet_LoRA", "efficientnet_v2_s", "lora", "efficientnet_lora_gen"),
    MigrationTarget("EfficientNet_TSA", "efficientnet_v2_s", "tsa", "efficientnet_tsa_gen"),
    MigrationTarget("ResNet18_Baseline", "resnet18", "baseline", "resnet18_baseline_gen"),
    MigrationTarget("ResNet18_BN_Tuning", "resnet18", "bn_tuning", "resnet18_bn_tuning_gen"),
    MigrationTarget("ResNet18_Full_Finetune", "resnet18", "full_finetune", "resnet18_full_finetune_gen"),
    MigrationTarget("ResNet18_LoRA", "resnet18", "lora", "resnet18_lora_gen"),
    MigrationTarget("ResNet18_TSA", "resnet18", "tsa", "resnet18_tsa_gen"),
]


def _load_handwritten_module(stem: str):
    module_path = MODEL_DIR / f"{stem}.py"
    spec = spec_from_file_location(f"legacy_{stem.lower()}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import handwritten model: {module_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _count_trainable_params(module, *, num_classes: int = 5) -> int:
    model = module.build_model(num_classes=num_classes, freeze_backbone=True, device="cpu")
    return sum(int(param.numel()) for param in model.parameters() if param.requires_grad)


def migrate() -> dict[str, Any]:
    MODEL_SPECS_DIR.mkdir(parents=True, exist_ok=True)

    mapping_records: list[dict[str, str]] = []
    report_items: list[dict[str, Any]] = []

    for target in MIGRATION_TARGETS:
        spec = custom_model_generator.build_preset_spec(
            model_name=target.generated_model_name,
            base_model=target.base_model,
            method_type=target.method_type,
        )
        artifacts = custom_model_generator.generate_custom_model(spec, overwrite=True)
        mapping_records.append(
            {
                "legacy_model": target.handwritten_stem,
                "generated_model": artifacts.model_name,
                "spec_file": str(artifacts.spec_file_path),
                "generated_file": str(artifacts.model_file_path),
                "base_model": target.base_model,
                "method_type": target.method_type,
            }
        )

        item: dict[str, Any] = {
            "legacy_model": target.handwritten_stem,
            "generated_model": artifacts.model_name,
            "base_model_expected": target.base_model,
            "method_type_expected": target.method_type,
            "checks": {},
        }

        checks = item["checks"]
        discovered = artifacts.model_name in discover_model_names()
        checks["discovered"] = discovered

        try:
            generated_module = load_model_module(artifacts.model_name)
            checks["load_generated_module"] = True
        except Exception as exc:
            checks["load_generated_module"] = False
            checks["load_generated_error"] = f"{type(exc).__name__}: {exc}"
            report_items.append(item)
            continue

        expected_interfaces = [
            "build_model",
            "build_optimizer",
            "get_model_metadata",
            "get_capabilities",
            "describe_model_structure",
            "get_default_gradcam_targets",
        ]
        checks["standard_interfaces"] = {
            name: bool(hasattr(generated_module, name))
            for name in expected_interfaces
        }
        checks["standard_interfaces_all_present"] = all(checks["standard_interfaces"].values())

        metadata = generated_module.get_model_metadata()
        checks["metadata_base_model_match"] = str(metadata.get("base_model")) == target.base_model
        checks["metadata_method_type_match"] = str(metadata.get("method_type")) == target.method_type
        checks["metadata_pretrained_present"] = "pretrained" in metadata

        capabilities = generated_module.get_capabilities()
        checks["capabilities_type_ok"] = isinstance(capabilities, dict)

        structure = generated_module.describe_model_structure()
        checks["structure_type_ok"] = isinstance(structure, dict)
        checks["structure_base_family"] = str(structure.get("base_family", ""))

        gradcam_targets = generated_module.get_default_gradcam_targets()
        checks["gradcam_targets_non_empty"] = isinstance(gradcam_targets, list) and len(gradcam_targets) > 0

        try:
            legacy_module = _load_handwritten_module(target.handwritten_stem)
            legacy_trainable = _count_trainable_params(legacy_module)
            generated_trainable = _count_trainable_params(generated_module)
            checks["trainable_params_legacy"] = legacy_trainable
            checks["trainable_params_generated"] = generated_trainable
            checks["trainable_params_match"] = legacy_trainable == generated_trainable
        except Exception as exc:
            checks["trainable_params_match"] = None
            checks["trainable_params_check_skipped_reason"] = f"{type(exc).__name__}: {exc}"

        report_items.append(item)

    map_payload = {
        "migration_version": "phase_migration_v1",
        "strategy": "side_by_side_generated_equivalents",
        "legacy_retained": True,
        "records": mapping_records,
    }
    MIGRATION_MAP_PATH.write_text(json.dumps(map_payload, indent=2, sort_keys=True), encoding="utf-8")

    summary = {
        "targets": len(MIGRATION_TARGETS),
        "generated_models": len(mapping_records),
        "all_discovered": all(item["checks"].get("discovered", False) for item in report_items),
        "all_interfaces_present": all(item["checks"].get("standard_interfaces_all_present", False) for item in report_items),
        "trainable_param_matches_known": sum(1 for item in report_items if item["checks"].get("trainable_params_match") is True),
        "trainable_param_checks_skipped": sum(1 for item in report_items if item["checks"].get("trainable_params_match") is None),
    }
    report_payload = {
        "migration_version": "phase_migration_v1",
        "summary": summary,
        "items": report_items,
    }
    MIGRATION_REPORT_PATH.write_text(json.dumps(report_payload, indent=2, sort_keys=True), encoding="utf-8")
    return report_payload


if __name__ == "__main__":
    report = migrate()
    print(json.dumps(report.get("summary", {}), indent=2, sort_keys=True))
