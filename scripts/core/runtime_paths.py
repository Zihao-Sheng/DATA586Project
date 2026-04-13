from __future__ import annotations

import os
import sys
from pathlib import Path


_WORKING_FOLDERS = ("model", "model_specs", "checkpoints", "data", "logs")


def is_frozen_app() -> bool:
    return bool(getattr(sys, "frozen", False))


def project_root() -> Path:
    env_override = os.environ.get("DATA586_APP_ROOT", "").strip()
    if env_override:
        return Path(env_override).expanduser().resolve()
    if is_frozen_app():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def scripts_root() -> Path:
    return project_root() / "scripts"


def model_dir() -> Path:
    return project_root() / "model"


def model_specs_dir() -> Path:
    return project_root() / "model_specs"


def checkpoints_dir() -> Path:
    return project_root() / "checkpoints"


def data_dir() -> Path:
    return project_root() / "data"


def logs_dir() -> Path:
    return project_root() / "logs"


def ensure_working_folders() -> None:
    root = project_root()
    root.mkdir(parents=True, exist_ok=True)
    for name in _WORKING_FOLDERS:
        (root / name).mkdir(parents=True, exist_ok=True)

