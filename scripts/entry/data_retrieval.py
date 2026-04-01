import sys
from pathlib import Path
import runpy


if __name__ == "__main__":
    scripts_root = Path(__file__).resolve().parents[1]
    if str(scripts_root) not in sys.path:
        sys.path.insert(0, str(scripts_root))
    target = scripts_root / "pipeline" / "data_retrieval.py"
    runpy.run_path(str(target), run_name="__main__")
