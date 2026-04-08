from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from workflow.test_splits import evaluate_test_splits as workflow_evaluate_test_splits


def default_test_splits_root() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "test_splits"


def default_output_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "logs" / "test_split_evaluations"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint across every split in data/test_splits.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint to evaluate.")
    parser.add_argument("--model", default=None, help="Optional model name override.")
    parser.add_argument("--test-splits-root", type=Path, default=default_test_splits_root(), help="Root folder containing split directories.")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size.")
    parser.add_argument("--device", default="auto", help="Device override, for example cpu or cuda.")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir(), help="Directory to save csv/json outputs.")
    return parser.parse_args()


def evaluate_test_splits(
    *,
    checkpoint_path: Path,
    model_name: str | None,
    test_splits_root: Path,
    image_size: int,
    device: str,
    output_dir: Path,
    status_callback=None,
    progress_callback=None,
) -> tuple[dict[str, object], Path, Path]:
    return workflow_evaluate_test_splits(
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        test_splits_root=test_splits_root,
        image_size=image_size,
        device=device,
        output_dir=output_dir,
        status_callback=status_callback,
        progress_callback=progress_callback,
    )


def main() -> None:
    args = parse_args()
    payload, json_path, csv_path = evaluate_test_splits(
        checkpoint_path=args.checkpoint,
        model_name=args.model,
        test_splits_root=args.test_splits_root,
        image_size=args.image_size,
        device=args.device,
        output_dir=args.output_dir,
        status_callback=lambda message, indeterminate: print(message),
        progress_callback=lambda processed, total: print(f"PROGRESS {processed}/{total}"),
    )
    print(json.dumps(payload, indent=2))
    print(f"Saved JSON to: {json_path}")
    print(f"Saved CSV to: {csv_path}")


if __name__ == "__main__":
    main()
