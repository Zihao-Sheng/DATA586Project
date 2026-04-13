from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from core.model_registry import (
    discover_model_names,
    discover_model_names_generated_first,
    equivalent_model_names,
    load_model_module,
    resolve_preferred_model_name,
)


def default_checkpoint_path() -> Path:
    return Path(__file__).resolve().parents[2] / "checkpoints" / "resnet18" / "best.pth"


def default_checkpoint_root() -> Path:
    return Path(__file__).resolve().parents[2] / "checkpoints"


def discover_checkpoint_model_names(checkpoint_root: Path | None = None) -> list[str]:
    root = checkpoint_root.expanduser().resolve() if checkpoint_root is not None else default_checkpoint_root()
    if not root.is_dir():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def parse_args() -> argparse.Namespace:
    available_models = discover_model_names_generated_first(include_legacy_fallback=True)
    parser = argparse.ArgumentParser(description="Predict classes for one or more images.")
    parser.add_argument("image_paths", nargs="*", type=Path, help="Image paths to predict.")
    parser.add_argument(
        "--input-list",
        type=Path,
        default=None,
        help="Optional JSON file containing a list of image paths.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model type to load. If omitted, try to infer it from the checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=default_checkpoint_path(),
        help="Checkpoint path.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Input image size.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override, for example cpu or cuda.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional output JSON path. If omitted, no file is written.",
    )
    return parser.parse_args()


def normalize_model_name(model_name: str | None) -> str | None:
    if not isinstance(model_name, str):
        return None
    normalized = model_name.strip().lower()
    if not normalized:
        return None
    for available_model in discover_model_names():
        if available_model.lower() == normalized:
            return available_model
    return None


def guess_model_name_from_checkpoint_path(checkpoint_path: Path) -> str | None:
    path_text = " ".join(
        [
            checkpoint_path.name.lower(),
            checkpoint_path.stem.lower(),
            checkpoint_path.parent.name.lower(),
        ]
    )
    for candidate in discover_checkpoint_model_names():
        if candidate.lower() in path_text:
            return resolve_preferred_model_name(candidate) or candidate
    return None


def infer_model_name_from_state_dict(state_dict: object) -> str | None:
    if not isinstance(state_dict, dict):
        return None
    keys = [str(key) for key in state_dict.keys()]
    if any(key.startswith("features.") for key in keys):
        for candidate in discover_checkpoint_model_names():
            if "efficientnet" in candidate:
                return resolve_preferred_model_name(candidate) or candidate
    if any(key.startswith("layer1.") or key.startswith("conv1.") for key in keys):
        for candidate in discover_checkpoint_model_names():
            if "resnet18" in candidate:
                return resolve_preferred_model_name(candidate) or candidate
    return None


def inspect_prediction_checkpoint(checkpoint_path: Path) -> dict[str, object]:
    import torch

    resolved_path = checkpoint_path.expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {resolved_path}")

    try:
        checkpoint = torch.load(resolved_path, map_location="cpu")
    except Exception as exc:
        raise ValueError(f"Could not read checkpoint metadata: {resolved_path}\n{exc}") from exc

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint payload is not a dictionary: {resolved_path}")

    checkpoint_model_name = normalize_model_name(checkpoint.get("model_name"))
    checkpoint_model_preferred = resolve_preferred_model_name(checkpoint_model_name)
    path_guess_model_name = guess_model_name_from_checkpoint_path(resolved_path)
    state_dict_model_name = infer_model_name_from_state_dict(checkpoint.get("model_state_dict"))
    inferred_model_name = checkpoint_model_preferred or state_dict_model_name or path_guess_model_name
    class_to_idx = checkpoint.get("class_to_idx")
    num_classes = checkpoint.get("num_classes")
    return {
        "checkpoint_path": resolved_path,
        "checkpoint_model_name": checkpoint_model_name,
        "checkpoint_model_preferred": checkpoint_model_preferred,
        "path_guess_model_name": path_guess_model_name,
        "state_dict_model_name": state_dict_model_name,
        "inferred_model_name": inferred_model_name,
        "num_classes": int(num_classes) if isinstance(num_classes, int) else None,
        "class_count": len(class_to_idx) if isinstance(class_to_idx, dict) else None,
    }


def validate_prediction_checkpoint(checkpoint_path: Path, requested_model_name: str | None = None) -> dict[str, object]:
    metadata = inspect_prediction_checkpoint(checkpoint_path)
    requested = normalize_model_name(requested_model_name)
    checkpoint_model_name = metadata.get("checkpoint_model_name")
    inferred_model_name = metadata.get("inferred_model_name")

    if (
        isinstance(checkpoint_model_name, str)
        and isinstance(requested, str)
        and requested not in equivalent_model_names(checkpoint_model_name)
    ):
        raise ValueError(
            f"Checkpoint model mismatch: checkpoint is '{checkpoint_model_name}', but requested '{requested}'. "
            f"Choose the matching checkpoint for that model."
        )

    resolved_model_name = requested or (str(inferred_model_name) if isinstance(inferred_model_name, str) else None)
    if resolved_model_name is None:
        resolved_path = metadata.get("checkpoint_path")
        raise ValueError(f"Could not determine model type for checkpoint: {resolved_path}")
    resolved_model_name = resolve_preferred_model_name(resolved_model_name) or resolved_model_name

    return {
        **metadata,
        "resolved_model_name": resolved_model_name,
    }


def infer_model_name_from_checkpoint(checkpoint_path: Path) -> str | None:
    metadata = inspect_prediction_checkpoint(checkpoint_path)
    inferred = metadata.get("inferred_model_name")
    return str(inferred) if isinstance(inferred, str) else None


def load_model(checkpoint_path: Path, model_name: str, device: str):
    import torch

    resolved_device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
    metadata = validate_prediction_checkpoint(checkpoint_path, requested_model_name=model_name)
    requested_model_name = metadata["resolved_model_name"]
    class_to_idx = checkpoint["class_to_idx"]
    num_classes = checkpoint["num_classes"]

    model_module = load_model_module(str(requested_model_name))
    model = model_module.build_model(
        num_classes=num_classes,
        freeze_backbone=False,
        device=resolved_device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_to_idx


def build_transform(image_size: int):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


def supported_image_extensions() -> tuple[str, ...]:
    return (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def collect_image_paths_from_directories(directories: list[Path]) -> list[Path]:
    image_paths: list[Path] = []
    for directory in directories:
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in supported_image_extensions():
                image_paths.append(path.resolve())
    return image_paths


class ImagePathDataset:
    def __init__(self, image_paths: list[Path], transform) -> None:
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        from PIL import Image

        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image)
        return tensor, str(image_path)


def predict_image(
    model,
    image_path: Path,
    transform,
    idx_to_class: dict[int, str],
    device: str,
) -> dict[str, str | float]:
    import torch
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0, pred_idx].item())

    return {
        "image_path": str(image_path),
        "predicted_class": idx_to_class[pred_idx],
        "confidence": confidence,
    }


def predict_images_batch(
    model,
    image_paths: list[Path],
    transform,
    idx_to_class: dict[int, str],
    device: str,
    batch_size: int = 16,
    num_workers: int = 0,
    progress_callback=None,
) -> list[dict[str, str | float]]:
    import torch
    from torch.utils.data import DataLoader

    dataset = ImagePathDataset(image_paths, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    results: list[dict[str, str | float]] = []
    processed = 0
    total = len(dataset)

    with torch.no_grad():
        for tensors, batch_paths in dataloader:
            tensors = tensors.to(device)
            logits = model(tensors)
            probs = torch.softmax(logits, dim=1)
            pred_indices = torch.argmax(probs, dim=1)

            for batch_index, image_path in enumerate(batch_paths):
                pred_idx = int(pred_indices[batch_index].item())
                confidence = float(probs[batch_index, pred_idx].item())
                results.append(
                    {
                        "image_path": image_path,
                        "predicted_class": idx_to_class[pred_idx],
                        "confidence": confidence,
                    }
                )

            processed += len(batch_paths)
            if progress_callback is not None:
                progress_callback(processed, total)

    return results


def main() -> None:
    args = parse_args()
    import torch

    requested_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = requested_device if requested_device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = args.checkpoint.expanduser().resolve()

    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    model_name = args.model
    if model_name is not None:
        model_name = normalize_model_name(model_name)
    if model_name is None:
        model_name = infer_model_name_from_checkpoint(checkpoint_path)
    if model_name is None:
        raise SystemExit(f"Could not determine model type for checkpoint: {checkpoint_path}")

    model, class_to_idx = load_model(checkpoint_path, model_name, device)
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    transform = build_transform(args.image_size)

    raw_image_paths = list(args.image_paths)
    if args.input_list is not None:
        input_list_path = args.input_list.expanduser().resolve()
        if not input_list_path.is_file():
            raise SystemExit(f"Input list not found: {input_list_path}")
        try:
            loaded_paths = json.loads(input_list_path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            raise SystemExit(f"Could not read input list: {input_list_path}\n{exc}") from exc
        if not isinstance(loaded_paths, list):
            raise SystemExit(f"Input list must be a JSON array: {input_list_path}")
        raw_image_paths = [Path(str(item)) for item in loaded_paths]

    if not raw_image_paths:
        raise SystemExit("Provide one or more image paths, or use --input-list.")

    valid_image_paths: list[Path] = []
    for image_path in raw_image_paths:
        candidate_path = image_path.expanduser()
        try:
            resolved_path = candidate_path.resolve(strict=False)
        except Exception:
            resolved_path = candidate_path
        try:
            from PIL import Image

            with Image.open(resolved_path) as image:
                image.verify()
        except Exception as exc:
            print(f"Skipping unreadable image: {resolved_path} ({exc})")
            continue
        valid_image_paths.append(resolved_path)

    if not valid_image_paths:
        raise SystemExit("No readable images were provided.")

    results = predict_images_batch(
        model,
        valid_image_paths,
        transform,
        idx_to_class,
        device,
    )
    for result in results:
        print(
            f"{result['image_path']} -> {result['predicted_class']} "
            f"(confidence={result['confidence']:.4f})"
        )

    if args.output_path is not None:
        output_path = args.output_path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Saved predictions to: {output_path}")


if __name__ == "__main__":
    main()
