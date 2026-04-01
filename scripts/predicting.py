from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_registry import discover_model_names, load_model_module


def default_checkpoint_path() -> Path:
    return Path(__file__).resolve().parents[1] / "checkpoints" / "resnet18_best.pth"


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

    results: list[dict[str, str | float]] = []
    for image_path in args.image_paths:
        resolved_path = image_path.expanduser().resolve()
        if not resolved_path.is_file():
            print(f"Skipping missing image: {resolved_path}")
            continue

        result = predict_image(model, resolved_path, transform, idx_to_class, device)
        results.append(result)
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
