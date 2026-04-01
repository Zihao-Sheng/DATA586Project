import sys
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from core.model_registry import discover_model_names, load_model_module, model_module_name

__all__ = [
    "discover_model_names",
    "model_module_name",
    "load_model_module",
]
