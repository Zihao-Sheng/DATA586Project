from __future__ import annotations

import json
from functools import lru_cache
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any

from core.model_structure_introspection import describe_model_structure_for_canvas as _describe_model_structure_for_canvas
from core import runtime_paths


MODEL_DIR = runtime_paths.model_dir()
MODEL_SPECS_DIR = runtime_paths.model_specs_dir()
LEGACY_MIGRATION_MAP_PATH = MODEL_SPECS_DIR / "legacy_migration_map.json"
IGNORED_MODEL_FILES = {"import_data.py", "__init__.py", "_transfer_strategies.py"}


def discover_model_names() -> list[str]:
    names: dict[str, str] = {}
    for path in sorted(MODEL_DIR.glob("*.py")):
        if path.name in IGNORED_MODEL_FILES or path.name.startswith("_"):
            continue
        key = path.stem.lower()
        names.setdefault(key, path.stem)
    return sorted(names.keys())


@lru_cache(maxsize=1)
def _load_legacy_migration_pairs() -> tuple[dict[str, str], dict[str, str]]:
    legacy_to_generated: dict[str, str] = {}
    generated_to_legacy: dict[str, str] = {}
    if not LEGACY_MIGRATION_MAP_PATH.is_file():
        return legacy_to_generated, generated_to_legacy
    try:
        payload = json.loads(LEGACY_MIGRATION_MAP_PATH.read_text(encoding="utf-8"))
    except Exception:
        return legacy_to_generated, generated_to_legacy
    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        return legacy_to_generated, generated_to_legacy
    for item in records:
        if not isinstance(item, dict):
            continue
        legacy_name = str(item.get("legacy_model", "")).strip().lower()
        generated_name = str(item.get("generated_model", "")).strip().lower()
        if not legacy_name or not generated_name:
            continue
        legacy_to_generated[legacy_name] = generated_name
        generated_to_legacy[generated_name] = legacy_name
    return legacy_to_generated, generated_to_legacy


def _canonical_name(name: str | None) -> str | None:
    if not isinstance(name, str):
        return None
    normalized = name.strip().lower()
    if not normalized:
        return None
    for model_name in discover_model_names():
        if model_name.lower() == normalized:
            return model_name
    return None


def equivalent_model_names(model_name: str | None) -> set[str]:
    canonical = _canonical_name(model_name)
    if canonical is None:
        return set()
    available = set(discover_model_names())
    legacy_to_generated, generated_to_legacy = _load_legacy_migration_pairs()
    names = {canonical}
    generated = legacy_to_generated.get(canonical)
    if generated and generated in available:
        names.add(generated)
    legacy = generated_to_legacy.get(canonical)
    if legacy and legacy in available:
        names.add(legacy)
    return names


def resolve_preferred_model_name(model_name: str | None) -> str | None:
    canonical = _canonical_name(model_name)
    if canonical is None:
        return None
    available = set(discover_model_names())
    legacy_to_generated, _ = _load_legacy_migration_pairs()
    mapped = legacy_to_generated.get(canonical)
    if mapped and mapped in available:
        return mapped
    return canonical


def describe_model_name(model_name: str | None) -> dict[str, Any]:
    canonical = _canonical_name(model_name)
    if canonical is None:
        return {
            "model_name": "",
            "exists": False,
            "source": "unknown",
            "is_generated": False,
            "is_legacy": False,
            "preferred_model_name": None,
            "equivalent_model_names": [],
            "display_label": "",
        }

    available = set(discover_model_names())
    legacy_to_generated, generated_to_legacy = _load_legacy_migration_pairs()
    spec_backed = (MODEL_SPECS_DIR / f"{canonical}.json").is_file()
    generated_by_name = canonical.endswith("_gen")
    is_generated = generated_by_name or spec_backed

    mapped_generated = legacy_to_generated.get(canonical)
    mapped_legacy = generated_to_legacy.get(canonical)
    is_legacy = mapped_generated is not None and mapped_generated in available

    preferred = resolve_preferred_model_name(canonical)
    equivalents = sorted(equivalent_model_names(canonical))
    source = "generated" if is_generated else ("legacy" if is_legacy else "handwritten")
    if source == "generated":
        display_label = f"{canonical} [Generated, Preferred]"
    elif source == "legacy" and preferred:
        display_label = f"{canonical} [Legacy, Fallback -> {preferred}]"
    else:
        display_label = canonical

    return {
        "model_name": canonical,
        "exists": True,
        "source": source,
        "is_generated": is_generated,
        "is_legacy": is_legacy,
        "preferred_model_name": preferred,
        "equivalent_model_names": equivalents,
        "legacy_equivalent": mapped_legacy if mapped_legacy in available else None,
        "generated_equivalent": mapped_generated if mapped_generated in available else None,
        "display_label": display_label,
    }


def discover_model_names_generated_first(*, include_legacy_fallback: bool = True) -> list[str]:
    available = discover_model_names()
    available_set = set(available)
    legacy_to_generated, _ = _load_legacy_migration_pairs()
    preferred: list[str] = []
    mapped_legacy: list[str] = []
    for model_name in available:
        mapped_generated = legacy_to_generated.get(model_name)
        if mapped_generated and mapped_generated in available_set:
            mapped_legacy.append(model_name)
        else:
            preferred.append(model_name)
    return preferred + mapped_legacy if include_legacy_fallback else preferred


def resolve_model_spec_path(model_name: str | None, *, allow_legacy_mapping: bool = True) -> Path | None:
    canonical = _canonical_name(model_name)
    if canonical is None:
        return None

    direct = MODEL_SPECS_DIR / f"{canonical}.json"
    if direct.is_file():
        return direct

    if allow_legacy_mapping:
        preferred = resolve_preferred_model_name(canonical)
        if isinstance(preferred, str) and preferred and preferred != canonical:
            preferred_path = MODEL_SPECS_DIR / f"{preferred}.json"
            if preferred_path.is_file():
                return preferred_path

    try:
        module = load_model_module(canonical)
        metadata = module.get_model_metadata() if hasattr(module, "get_model_metadata") else {}
    except Exception:
        metadata = {}
    if isinstance(metadata, dict):
        raw = metadata.get("source_spec_file") or metadata.get("spec_file")
        if isinstance(raw, str) and raw.strip():
            path = Path(raw.strip()).expanduser()
            if not path.is_absolute():
                path = MODEL_SPECS_DIR.parents[0] / path
            path = path.resolve()
            if path.is_file():
                return path
    return None


def discover_roundtrip_editable_models(*, include_legacy_fallback: bool = False) -> list[str]:
    names = discover_model_names_generated_first(include_legacy_fallback=include_legacy_fallback)
    return [name for name in names if resolve_model_spec_path(name, allow_legacy_mapping=True) is not None]


def model_module_name(model_name: str) -> str:
    normalized = model_name.lower()
    for path in sorted(MODEL_DIR.glob("*.py")):
        if path.name in IGNORED_MODEL_FILES or path.name.startswith("_"):
            continue
        if path.stem.lower() == normalized:
            return path.stem
    raise ValueError(f"Unsupported model: {model_name}")


def load_model_module(model_name: str) -> ModuleType:
    module_stem = model_module_name(model_name)
    module_path = MODEL_DIR / f"{module_stem}.py"
    spec = spec_from_file_location(f"shared_model_{module_stem.lower()}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load model module from {module_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "build_model") or not hasattr(module, "build_optimizer"):
        raise AttributeError(
            f"Model module '{module_stem}' must define both build_model and build_optimizer."
        )
    return module


def resolve_model_structure_for_canvas(model_name: str | None) -> dict[str, Any]:
    canonical = _canonical_name(model_name)
    if canonical is None:
        raise ValueError(f"Unknown model name: {model_name}")
    module_stem = model_module_name(canonical)
    module = load_model_module(canonical)
    module_path = MODEL_DIR / f"{module_stem}.py"
    return _describe_model_structure_for_canvas(
        model_name=canonical,
        module=module,
        module_path=module_path,
    )


@lru_cache(maxsize=256)
def model_metadata(model_name: str | None) -> dict[str, Any]:
    canonical = _canonical_name(model_name)
    if canonical is None:
        return {}
    try:
        module = load_model_module(canonical)
    except Exception:
        return {}
    try:
        payload = module.get_model_metadata() if hasattr(module, "get_model_metadata") else {}
    except Exception:
        payload = {}
    return payload if isinstance(payload, dict) else {}


def _spec_payload(model_name: str | None) -> dict[str, Any]:
    canonical = _canonical_name(model_name)
    if canonical is None:
        return {}
    spec_path = MODEL_SPECS_DIR / f"{canonical}.json"
    if not spec_path.is_file():
        return {}
    try:
        payload = json.loads(spec_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _infer_method_type(name: str) -> str:
    lowered = name.lower()
    mapping = (
        ("bn_last2", "bn_last2"),
        ("bn_last1", "bn_last1"),
        ("bn_tuning", "bn_tuning"),
        ("norm_tuning", "norm_tuning"),
        ("full_finetune", "full_finetune"),
        ("lora", "lora"),
        ("dora", "dora"),
        ("tsa", "tsa"),
        ("adapter", "adapter"),
        ("bitfit", "bitfit"),
        ("ssf", "ssf"),
        ("baseline", "baseline"),
    )
    for token, method in mapping:
        if token in lowered:
            return method
    return "unknown"


def _infer_family_variant(name: str) -> tuple[str, str]:
    lowered = name.lower()
    if "resnet50" in lowered:
        return "resnet", "resnet50"
    if "resnet18" in lowered:
        return "resnet", "resnet18"
    if "efficientnet_v2_s" in lowered:
        return "efficientnet", "efficientnet_v2_s"
    if "efficientnet" in lowered:
        return "efficientnet", "efficientnet_v2_s"
    if "convnext_tiny" in lowered:
        return "convnext", "convnext_tiny"
    if "mobilenet_v3_large" in lowered:
        return "mobilenet_v3", "mobilenet_v3_large"
    if "densenet121" in lowered:
        return "densenet", "densenet121"
    return "unknown", lowered


def model_catalog_entry(model_name: str | None) -> dict[str, Any]:
    base = describe_model_name(model_name)
    canonical = str(base.get("model_name", "")).strip()
    metadata = model_metadata(canonical)
    spec_payload = _spec_payload(canonical)

    provider = str(
        metadata.get("base_provider", spec_payload.get("base_provider", "torchvision"))
    ).strip().lower() or "torchvision"
    family = str(metadata.get("base_family", spec_payload.get("base_family", ""))).strip().lower()
    variant = str(metadata.get("variant", spec_payload.get("variant", ""))).strip().lower()
    method = str(metadata.get("method_type", spec_payload.get("method_type", ""))).strip().lower()
    pretrained_raw = metadata.get("pretrained")
    if not isinstance(pretrained_raw, bool):
        pretrained_raw = spec_payload.get("pretrained")
    pretrained = bool(pretrained_raw) if isinstance(pretrained_raw, bool) else None

    if not family or not variant:
        inferred_family, inferred_variant = _infer_family_variant(canonical)
        family = family or inferred_family
        variant = variant or inferred_variant
    if not method:
        method = _infer_method_type(canonical)

    return {
        **base,
        "provider": provider,
        "family": family,
        "variant": variant,
        "method_type": method,
        "pretrained": pretrained,
        "metadata": metadata,
    }


def model_display_label(model_name: str | None, *, include_name: bool = True) -> str:
    info = model_catalog_entry(model_name)
    name = str(info.get("model_name", "")).strip()
    source = str(info.get("source", "unknown"))
    provider = str(info.get("provider", "torchvision"))
    family = str(info.get("family", "unknown"))
    variant = str(info.get("variant", "unknown"))
    method = str(info.get("method_type", "unknown"))
    pretrained = info.get("pretrained")
    pre = "pretrained" if pretrained is True else ("scratch" if pretrained is False else "pretrained=?")
    src = "generated" if source == "generated" else ("legacy" if source == "legacy" else "handwritten")
    details = f"{provider}/{family}/{variant} | {method} | {pre} | {src}"
    return f"{name} [{details}]" if include_name else details


def sort_model_names_for_ui(model_names: list[str]) -> list[str]:
    source_order = {"generated": 0, "handwritten": 1, "legacy": 2, "unknown": 3}
    method_order = {
        "baseline": 0,
        "bn_tuning": 1,
        "norm_tuning": 2,
        "bn_last1": 3,
        "bn_last2": 4,
        "full_finetune": 5,
        "lora": 6,
        "dora": 7,
        "tsa": 8,
        "adapter": 9,
        "bitfit": 10,
        "ssf": 11,
        "unknown": 12,
    }

    def _key(name: str) -> tuple[Any, ...]:
        info = model_catalog_entry(name)
        return (
            source_order.get(str(info.get("source", "unknown")), 99),
            str(info.get("provider", "unknown")),
            str(info.get("family", "unknown")),
            str(info.get("variant", "unknown")),
            method_order.get(str(info.get("method_type", "unknown")), 99),
            str(info.get("model_name", name)).lower(),
        )

    return sorted(model_names, key=_key)

