from __future__ import annotations

import sys
import traceback
from datetime import datetime
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    scripts_root = project_root / "scripts"
    if str(scripts_root) not in sys.path:
        sys.path.insert(0, str(scripts_root))

    from app.training_gui import main as run_training_gui

    run_training_gui()


def report_failure(exc: BaseException) -> None:
    try:
        import ctypes

        project_root = Path(__file__).resolve().parents[2]
        log_dir = project_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "training_gui_launch_error.log"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        log_path.write_text(f"[{timestamp}]\n{message}\n", encoding="utf-8")
        ctypes.windll.user32.MessageBoxW(
            0,
            f"Training GUI failed to launch.\n\nSee:\n{log_path}",
            "Training Launcher Error",
            0x10,
        )
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        report_failure(exc)
        raise
