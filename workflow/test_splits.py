from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from workflow.predicting import (
    build_transform,
    infer_model_name_from_checkpoint,
    load_model,
    predict_images_batch,
    supported_image_extensions,
)


def collect_split_image_paths(test_splits_root: Path) -> dict[str, list[Path]]:
    split_map: dict[str, list[Path]] = {}
    for split_dir in sorted(path for path in test_splits_root.iterdir() if path.is_dir()):
        image_paths = [path.resolve() for path in split_dir.rglob("*") if path.is_file() and path.suffix.lower() in supported_image_extensions()]
        split_map[split_dir.name] = sorted(image_paths)
    return split_map


def summarize_split_results(*, split_name: str, image_paths: list[Path], batch_results: list[dict[str, str | float]], class_to_idx: dict[str, int]) -> dict[str, object]:
    predicted_by_path = {str(result["image_path"]): result for result in batch_results}
    correct = 0
    evaluated = 0
    confidence_sum = 0.0
    skipped = 0
    for image_path in image_paths:
        actual_label = image_path.parent.name
        result = predicted_by_path.get(str(image_path))
        if actual_label not in class_to_idx or result is None:
            skipped += 1
            continue
        evaluated += 1
        predicted_label = str(result["predicted_class"])
        confidence_sum += float(result["confidence"])
        if predicted_label == actual_label:
            correct += 1
    accuracy = (correct / evaluated) if evaluated > 0 else 0.0
    avg_confidence = (confidence_sum / evaluated) if evaluated > 0 else 0.0
    return {
        "split": split_name,
        "total_images": len(image_paths),
        "evaluated_images": evaluated,
        "skipped_images": skipped,
        "correct_images": correct,
        "accuracy": accuracy,
        "avg_confidence": avg_confidence,
    }


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
    import torch

    resolved_checkpoint = checkpoint_path.expanduser().resolve()
    resolved_root = test_splits_root.expanduser().resolve()
    resolved_output_dir = output_dir.expanduser().resolve()
    if not resolved_checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {resolved_checkpoint}")
    if not resolved_root.is_dir():
        raise FileNotFoundError(f"Test splits root not found: {resolved_root}")
    resolved_device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    resolved_model_name = model_name or infer_model_name_from_checkpoint(resolved_checkpoint)
    if not resolved_model_name:
        raise ValueError(f"Could not determine model type for checkpoint: {resolved_checkpoint}")
    if status_callback is not None:
        status_callback(f"Loading {resolved_model_name} from checkpoint...", True)
    model, class_to_idx = load_model(resolved_checkpoint, resolved_model_name, resolved_device)
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    transform = build_transform(image_size)
    split_map = collect_split_image_paths(resolved_root)
    if not split_map:
        raise ValueError(f"No split directories with images were found in: {resolved_root}")
    split_summaries: list[dict[str, object]] = []
    total_images = sum(len(paths) for paths in split_map.values())
    processed_images = 0
    started_at = time.perf_counter()
    for split_name, image_paths in split_map.items():
        if status_callback is not None:
            status_callback(f"Evaluating split '{split_name}' ({len(image_paths)} image(s))...", False)
        batch_results = predict_images_batch(
            model,
            image_paths,
            transform,
            idx_to_class,
            resolved_device,
            batch_size=16,
            progress_callback=(lambda processed, total, base=processed_images: progress_callback(base + processed, total_images)) if progress_callback is not None else None,
        )
        processed_images += len(image_paths)
        if progress_callback is not None:
            progress_callback(processed_images, total_images)
        split_summaries.append(summarize_split_results(split_name=split_name, image_paths=image_paths, batch_results=batch_results, class_to_idx=class_to_idx))
    clean_accuracy = next((float(item["accuracy"]) for item in split_summaries if item["split"] == "clean"), 0.0)
    robustness_values = [float(item["accuracy"]) for item in split_summaries if item["split"] != "clean"]
    robustness_average = sum(robustness_values) / len(robustness_values) if robustness_values else 0.0
    total_seconds = time.perf_counter() - started_at
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"{resolved_model_name}_{timestamp}"
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    json_path = resolved_output_dir / f"{stem}.json"
    csv_path = resolved_output_dir / f"{stem}.csv"
    payload = {
        "checkpoint_path": str(resolved_checkpoint),
        "model_name": resolved_model_name,
        "test_splits_root": str(resolved_root),
        "image_size": image_size,
        "device": resolved_device,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_seconds": total_seconds,
        "clean_accuracy": clean_accuracy,
        "robustness_average": robustness_average,
        "splits": split_summaries,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "total_images", "evaluated_images", "skipped_images", "correct_images", "accuracy", "avg_confidence"])
        writer.writeheader()
        writer.writerows(split_summaries)
    return payload, json_path, csv_path
