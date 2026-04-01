from pathlib import Path
import runpy


if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / "maintenance" / "ensure_packages_gui.pyw"
    runpy.run_path(str(target), run_name="__main__")
