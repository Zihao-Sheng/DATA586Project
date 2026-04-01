from __future__ import annotations

from importlib import import_module
from types import ModuleType
from pathlib import Path


MODEL_DIR = Path(__file__).resolve().parent / "model"
IGNORED_MODEL_FILES = {"import_data.py", "__init__.py"}


def discover_model_names() -> list[str]:
    names: dict[str, str] = {}
    for path in sorted(MODEL_DIR.glob("*.py")):
        if path.name in IGNORED_MODEL_FILES or path.name.startswith("_"):
            continue
        key = path.stem.lower()
        names.setdefault(key, path.stem)
    return sorted(names.keys())


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
    module = import_module(f"model.{module_stem}")
    if not hasattr(module, "build_model") or not hasattr(module, "build_optimizer"):
        raise AttributeError(
            f"Model module '{module_stem}' must define both build_model and build_optimizer."
        )
    return module

