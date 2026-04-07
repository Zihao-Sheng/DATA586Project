from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType


MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
IGNORED_MODEL_FILES = {"import_data.py", "__init__.py", "_transfer_strategies.py"}


def discover_model_names() -> list[str]:
    names: dict[str, str] = {}
    for path in sorted(MODEL_DIR.glob("*.py")):
        if path.name in IGNORED_MODEL_FILES or path.name.startswith("_"):
            continue
        names.setdefault(path.stem.lower(), path.stem)
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
