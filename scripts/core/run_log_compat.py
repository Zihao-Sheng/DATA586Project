from __future__ import annotations

import json
from pathlib import Path


def normalize_run_log(data: dict, *, log_path: str | Path | None = None) -> dict:
    run = dict(data) if isinstance(data, dict) else {}
    if log_path is not None:
        run["_log_path"] = str(Path(log_path).expanduser().resolve())

    for key in ("args", "dataset", "model", "expected", "summary", "timing_summary", "analysis", "artifacts"):
        value = run.get(key)
        if not isinstance(value, dict):
            run[key] = {}

    args = run["args"]
    assert isinstance(args, dict)
    args.setdefault("optimizer", "adam")
    args.setdefault("scheduler", "none")
    args.setdefault("amp", False)
    args.setdefault("seed", 42)
    args.setdefault("train_transforms_preset", "baseline")
    args.setdefault("mild_blur_enabled", False)
    args.setdefault("mild_blur_prob", 0.0)
    augmentation_config = args.get("augmentation_config")
    if not isinstance(augmentation_config, dict):
        dataset_value = run.get("dataset")
        if isinstance(dataset_value, dict) and isinstance(dataset_value.get("augmentation_config"), dict):
            augmentation_config = dataset_value.get("augmentation_config")
    if not isinstance(augmentation_config, dict):
        augmentation_config = {}
    args["augmentation_config"] = augmentation_config
    args.setdefault("weight_decay", 0.0)
    args.setdefault("scheduler_t_max", None)
    args.setdefault("scheduler_step_size", None)
    args.setdefault("scheduler_gamma", None)
    args.setdefault("scheduler_patience", None)
    model = run["model"]
    assert isinstance(model, dict)
    model.setdefault("amp_enabled", bool(args.get("amp", False)))

    dataset = run["dataset"]
    assert isinstance(dataset, dict)
    dataset.setdefault("train_transforms_preset", str(args.get("train_transforms_preset", "baseline")))
    dataset.setdefault("mild_blur_enabled", bool(args.get("mild_blur_enabled", False)))
    dataset.setdefault("mild_blur_prob", float(args.get("mild_blur_prob", 0.0)))
    dataset.setdefault("augmentation_config", augmentation_config)

    epochs = run.get("epochs")
    run["epochs"] = epochs if isinstance(epochs, list) else []

    final_test = run.get("final_test")
    run["final_test"] = final_test if isinstance(final_test, dict) else {}

    return run


def load_run_log(path: Path) -> dict | None:
    try:
        data = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return normalize_run_log(data, log_path=path)


def safe_float(value) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def safe_int(value) -> int | None:
    return int(value) if isinstance(value, (int, float)) else None


def format_metric(value) -> str:
    numeric = safe_float(value)
    return f"{numeric:.4f}" if numeric is not None else "-"


def format_ratio(numerator, denominator) -> str:
    left = int(numerator) if isinstance(numerator, (int, float)) else 0
    right = int(denominator) if isinstance(denominator, (int, float)) else 0
    return f"{left}/{right}" if right > 0 else str(left)


def normalize_run_status(run: dict) -> str:
    status = str(run.get("status", "unknown"))
    return "incomplete_or_interrupted" if status == "running" else status


def infer_last_completed_epoch(run: dict) -> int:
    summary = run.get("summary") if isinstance(run.get("summary"), dict) else {}
    if isinstance(summary.get("last_completed_epoch"), (int, float)):
        return int(summary["last_completed_epoch"])
    epochs = run.get("epochs") if isinstance(run.get("epochs"), list) else []
    return len(epochs)


def infer_eval_name(run: dict) -> str:
    dataset = run.get("dataset") if isinstance(run.get("dataset"), dict) else {}
    if isinstance(dataset.get("eval_name"), str):
        return str(dataset["eval_name"])
    expected = run.get("expected") if isinstance(run.get("expected"), dict) else {}
    if "val_batches_per_epoch" in expected:
        return "val"
    if "test_batches_per_epoch" in expected:
        return "test"
    return "-"


def infer_best_eval_acc(run: dict) -> float | None:
    summary = run.get("summary") if isinstance(run.get("summary"), dict) else {}
    best = summary.get("best_eval_acc")
    if isinstance(best, (int, float)):
        return float(best)
    epochs = run.get("epochs") if isinstance(run.get("epochs"), list) else []
    best_value: float | None = None
    for epoch_record in epochs:
        if not isinstance(epoch_record, dict):
            continue
        for key, stage in epoch_record.items():
            if key in {"epoch", "lr", "best_eval_acc_after_epoch", "is_best_checkpoint"}:
                continue
            if isinstance(stage, dict) and isinstance(stage.get("acc"), (int, float)):
                value = float(stage["acc"])
                if best_value is None or value > best_value:
                    best_value = value
    return best_value


def extract_run_summary(run: dict) -> dict:
    normalized = normalize_run_log(run)
    return {
        "run_id": normalized.get("run_id", "unknown"),
        "status": normalize_run_status(normalized),
        "status_reason": normalized.get("status_reason", "-"),
        "start_time_utc": normalized.get("start_time_utc", "-"),
        "end_time_utc": normalized.get("end_time_utc", "-"),
        "last_completed_epoch": infer_last_completed_epoch(normalized),
        "best_eval_acc": infer_best_eval_acc(normalized),
        "eval_name": infer_eval_name(normalized),
        "args": normalized["args"],
        "dataset": normalized["dataset"],
        "model": normalized["model"],
        "expected": normalized["expected"],
        "summary": normalized["summary"],
        "timing_summary": normalized["timing_summary"],
        "analysis": normalized["analysis"],
        "artifacts": normalized["artifacts"],
        "final_test": normalized["final_test"],
        "epochs": normalized["epochs"],
    }


def run_display_name(run: dict, include_stage: str | None = None) -> str:
    args = run.get("args") if isinstance(run.get("args"), dict) else {}
    started = str(run.get("start_time_utc", "-"))[:10]
    model = str(args.get("model", "run"))
    checkpoint_name = Path(str(args.get("checkpoint_dir", "-"))).name
    base = f"{started} {model} ({checkpoint_name})"
    return f"{base} [{include_stage}]" if include_stage else base


def extract_analysis_block(run: dict, stage_name: str | None = None) -> dict | None:
    analysis = run.get("analysis") if isinstance(run.get("analysis"), dict) else {}
    if stage_name == "final_test":
        block = analysis.get("final_test")
        return block if isinstance(block, dict) else None
    if stage_name in {"val", "test"}:
        last_stage = analysis.get("last_eval_stage")
        if last_stage == stage_name:
            block = analysis.get("last_eval")
            return block if isinstance(block, dict) else None
        if stage_name == "test":
            block = analysis.get("final_test")
            if isinstance(block, dict):
                return block
    block = analysis.get("final_test")
    if isinstance(block, dict):
        return block
    block = analysis.get("last_eval")
    return block if isinstance(block, dict) else None


def summarize_error_block(analysis: dict | None, *, limit: int = 5) -> list[str]:
    if not isinstance(analysis, dict):
        return ["Error Analysis:", "- No per-class error summary recorded for this run."]
    lines = [
        "Error Analysis:",
        f"- total_examples: {analysis.get('total_examples', '-')}",
        f"- correct_examples: {analysis.get('correct_examples', '-')}",
        f"- misclassified_examples: {analysis.get('misclassified_examples', '-')}",
    ]
    top_pairs = analysis.get("top_misclassifications") if isinstance(analysis.get("top_misclassifications"), list) else []
    if top_pairs:
        lines.append("- top_confusions:")
        for item in top_pairs[:limit]:
            if not isinstance(item, dict):
                continue
            lines.append(
                "  "
                f"{item.get('true_label', '?')} -> {item.get('pred_label', '?')} "
                f"(count={item.get('count', '-')}, avg_conf={format_metric(item.get('avg_confidence'))})"
            )
    top_conf = analysis.get("top_confidence_errors") if isinstance(analysis.get("top_confidence_errors"), list) else []
    if top_conf:
        lines.append("- high_confidence_errors:")
        for item in top_conf[:limit]:
            if not isinstance(item, dict):
                continue
            lines.append(
                "  "
                f"{item.get('true_label', '?')} -> {item.get('pred_label', '?')} "
                f"(conf={format_metric(item.get('confidence'))})"
            )
    return lines


def timing_value_from_stage(stage: dict, timing_metric: str) -> float | None:
    timing = stage.get("timing", {}) if isinstance(stage, dict) else {}
    if not isinstance(timing, dict):
        return None
    if timing_metric == "total":
        return float(timing["total_seconds"]) if isinstance(timing.get("total_seconds"), (int, float)) else None
    if timing_metric == "pure":
        return float(timing["pure_seconds"]) if isinstance(timing.get("pure_seconds"), (int, float)) else None
    pure_seconds = timing.get("pure_seconds")
    batches = timing.get("batches")
    if isinstance(pure_seconds, (int, float)) and isinstance(batches, (int, float)) and float(batches) > 0:
        return float(pure_seconds) / float(batches)
    return None


def extract_epoch_metrics(run: dict, stage_name: str, value_kind: str, timing_metric: str | None = None) -> list[tuple[float, float]]:
    epochs = run.get("epochs") if isinstance(run.get("epochs"), list) else []
    stage_key = stage_name.lower()
    points: list[tuple[float, float]] = []
    for epoch_record in epochs:
        if not isinstance(epoch_record, dict):
            continue
        epoch_index = epoch_record.get("epoch")
        stage = epoch_record.get(stage_key)
        if not isinstance(epoch_index, (int, float)) or not isinstance(stage, dict):
            continue
        if value_kind == "accuracy":
            value = float(stage["acc"]) if isinstance(stage.get("acc"), (int, float)) else None
        else:
            value = timing_value_from_stage(stage, timing_metric or "total")
        if value is not None:
            points.append((float(epoch_index), float(value)))

    if stage_key == "test" and not points:
        final_test = run.get("final_test") if isinstance(run.get("final_test"), dict) else None
        if isinstance(final_test, dict):
            epoch_index = float(infer_last_completed_epoch(run))
            if value_kind == "accuracy":
                value = final_test.get("acc")
                if isinstance(value, (int, float)):
                    points.append((epoch_index, float(value)))
            else:
                value = timing_value_from_stage(final_test, timing_metric or "total")
                if value is not None:
                    points.append((epoch_index, value))
    return points
