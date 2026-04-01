from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_registry import discover_model_names, load_model_module


def default_checkpoint_path() -> Path:
    return Path(__file__).resolve().parents[1] / "checkpoints" / "resnet18" / "best.pth"


def parse_args() -> argparse.Namespace:
    available_models = discover_model_names()
    parser = argparse.ArgumentParser(description="Predict classes for one or more images.")
    parser.add_argument("image_paths", nargs="+", type=Path, help="Image paths to predict.")
    parser.add_argument(
        "--model",
        default=available_models[0],
        choices=available_models,
        help="Model type to load.",
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


def load_model(checkpoint_path: Path, model_name: str, device: str):
    import torch

    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_to_idx = checkpoint["class_to_idx"]
    num_classes = checkpoint["num_classes"]

    model_module = load_model_module(model_name)
    model = model_module.build_model(
        num_classes=num_classes,
        freeze_backbone=False,
        device=device,
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

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = args.checkpoint.expanduser().resolve()

    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    model, class_to_idx = load_model(checkpoint_path, args.model, device)
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    transform = build_transform(args.image_size)

    valid_image_paths: list[Path] = []
    for image_path in args.image_paths:
        resolved_path = image_path.expanduser().resolve()
        if not resolved_path.is_file():
            print(f"Skipping missing image: {resolved_path}")
            continue
        valid_image_paths.append(resolved_path)

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
