from __future__ import annotations

import json
from pathlib import Path


def render_log_summary(log_paths: list[Path], view: str = "summary") -> str:
    runs = [_load_run(path) for path in log_paths]
    runs = [run for run in runs if run is not None]
    if not runs:
        return "No log runs found."

    normalized_view = view.strip().lower()
    if len(runs) >= 2 and normalized_view == "summary":
        return _render_compare_runs(runs)
    if len(runs) >= 2 and normalized_view in {"train", "val", "test"}:
        blocks: list[str] = []
        for run in runs:
            blocks.append(_run_display_name(run))
            blocks.append(_render_stage_epochs(run, normalized_view))
            blocks.append("")
        return "\n".join(blocks).strip()
    if normalized_view == "summary":
        return _render_run_summary(runs[0])
    return _render_stage_epochs(runs[0], normalized_view)


def _load_run(path: Path) -> dict | None:
    try:
        data = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data["_log_path"] = str(path.expanduser().resolve())
            return data
    except Exception:
        return None
    return None


def _normalize_run_status(run: dict) -> str:
    status = str(run.get("status", "unknown"))
    return "incomplete_or_interrupted" if status == "running" else status


def _safe_float(value) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _format_metric(value) -> str:
    numeric = _safe_float(value)
    return f"{numeric:.4f}" if numeric is not None else "-"


def _format_ratio(numerator, denominator) -> str:
    left = int(numerator) if isinstance(numerator, (int, float)) else 0
    right = int(denominator) if isinstance(denominator, (int, float)) else 0
    return f"{left}/{right}" if right > 0 else str(left)


def _infer_last_completed_epoch(run: dict) -> int:
    summary = run.get("summary") if isinstance(run.get("summary"), dict) else {}
    if isinstance(summary.get("last_completed_epoch"), (int, float)):
        return int(summary["last_completed_epoch"])
    epochs = run.get("epochs") if isinstance(run.get("epochs"), list) else []
    return len(epochs)


def _infer_eval_name(run: dict) -> str:
    dataset = run.get("dataset") if isinstance(run.get("dataset"), dict) else {}
    if isinstance(dataset.get("eval_name"), str):
        return str(dataset["eval_name"])
    expected = run.get("expected") if isinstance(run.get("expected"), dict) else {}
    if "val_batches_per_epoch" in expected:
        return "val"
    if "test_batches_per_epoch" in expected:
        return "test"
    return "-"


def _infer_best_eval_acc(run: dict) -> float | None:
    summary = run.get("summary") if isinstance(run.get("summary"), dict) else {}
    best = summary.get("best_eval_acc") if isinstance(summary, dict) else None
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


def _run_display_name(run: dict, include_stage: str | None = None) -> str:
    args = run.get("args") if isinstance(run.get("args"), dict) else {}
    started = str(run.get("start_time_utc", "-"))[:10]
    model = str(args.get("model", "run"))
    checkpoint_name = Path(str(args.get("checkpoint_dir", "-"))).name
    base = f"{started} {model} ({checkpoint_name})"
    return f"{base} [{include_stage}]" if include_stage else base


def _render_compare_runs(runs: list[dict]) -> str:
    header = (
        f"{'Started':<22} {'Model':<12} {'Status':<14} {'Progress':<9} "
        f"{'BestEval':<10} {'FinalTest':<10} {'Eval':<6} {'Batch':<6} {'LR':<10} {'Checkpoint'}"
    )
    separator = "-" * len(header)
    lines = [header, separator, "", "Average Timing Compare:"]
    for run in runs:
        args = run.get("args") if isinstance(run.get("args"), dict) else {}
        summary = run.get("summary") if isinstance(run.get("summary"), dict) else {}
        timing_summary = run.get("timing_summary") if isinstance(run.get("timing_summary"), dict) else {}
        stage_totals = timing_summary.get("stage_totals") if isinstance(timing_summary.get("stage_totals"), dict) else {}
        started = str(run.get("start_time_utc", "-"))[:19]
        model = str(args.get("model", "-"))[:12]
        status = _normalize_run_status(run)[:14]
        progress = _format_ratio(_infer_last_completed_epoch(run), args.get("planned_epochs_this_run"))
        best_eval = _format_metric(_infer_best_eval_acc(run))
        final_test = _format_metric(summary.get("final_test_acc"))
        eval_name = _infer_eval_name(run)[:6]
        batch_size = str(args.get("batch_size", "-"))[:6]
        lr = str(args.get("lr", "-"))[:10]
        checkpoint_name = Path(str(args.get("checkpoint_dir", "-"))).name[:20]
        lines.append(
            f"{started:<22} {model:<12} {status:<14} {progress:<9} "
            f"{best_eval:<10} {final_test:<10} {eval_name:<6} {batch_size:<6} {lr:<10} {checkpoint_name}"
        )
        train_stage = stage_totals.get("train") if isinstance(stage_totals.get("train"), dict) else {}
        test_stage = stage_totals.get("test") if isinstance(stage_totals.get("test"), dict) else {}
        train_batches = float(train_stage.get("batches", 0.0)) if isinstance(train_stage.get("batches"), (int, float)) else 0.0
        test_batches = float(test_stage.get("batches", 0.0)) if isinstance(test_stage.get("batches"), (int, float)) else 0.0
        train_avg_epoch = (
            float(train_stage.get("total_seconds", 0.0)) / max(float(_infer_last_completed_epoch(run)), 1.0)
            if _infer_last_completed_epoch(run) > 0 and isinstance(train_stage.get("total_seconds"), (int, float))
            else None
        )
        test_avg_epoch = (
            float(test_stage.get("total_seconds", 0.0)) / max(float(_infer_last_completed_epoch(run)), 1.0)
            if _infer_last_completed_epoch(run) > 0 and isinstance(test_stage.get("total_seconds"), (int, float))
            else None
        )
        train_avg_batch = (
            float(train_stage.get("pure_seconds", 0.0)) / train_batches
            if train_batches > 0 and isinstance(train_stage.get("pure_seconds"), (int, float))
            else None
        )
        test_avg_batch = (
            float(test_stage.get("pure_seconds", 0.0)) / test_batches
            if test_batches > 0 and isinstance(test_stage.get("pure_seconds"), (int, float))
            else None
        )
        lines.append(
            f"  avg_train_time_per_epoch={_format_metric(train_avg_epoch)}s, "
            f"avg_test_time_per_epoch={_format_metric(test_avg_epoch)}s, "
            f"avg_train_pure_per_batch={_format_metric(train_avg_batch)}s, "
            f"avg_test_pure_per_batch={_format_metric(test_avg_batch)}s"
        )
    return "\n".join(lines)


def _render_run_summary(run: dict) -> str:
    args = run.get("args") if isinstance(run.get("args"), dict) else {}
    dataset = run.get("dataset") if isinstance(run.get("dataset"), dict) else {}
    model_info = run.get("model") if isinstance(run.get("model"), dict) else {}
    expected = run.get("expected") if isinstance(run.get("expected"), dict) else {}
    epochs = run.get("epochs") if isinstance(run.get("epochs"), list) else []
    summary = run.get("summary") if isinstance(run.get("summary"), dict) else {}
    timing_summary = run.get("timing_summary") if isinstance(run.get("timing_summary"), dict) else {}

    planned_epochs = int(args.get("planned_epochs_this_run", 0)) if isinstance(args.get("planned_epochs_this_run"), (int, float)) else 0
    completed_epochs = len(epochs)
    progress_text = f"{completed_epochs}/{planned_epochs}" if planned_epochs > 0 else str(completed_epochs)

    lines = [
        f"Run ID: {run.get('run_id', 'unknown')}",
        f"Status: {_normalize_run_status(run)}",
        f"Status Reason: {run.get('status_reason', '-')}",
        f"Started (UTC): {run.get('start_time_utc', '-')}",
        f"Ended (UTC): {run.get('end_time_utc', '-')}",
        f"Model: {args.get('model', '-')}",
        f"Device: {args.get('device', '-')}",
        f"Command: {run.get('command', '-')}",
        f"Planned Epochs / Completed Epochs: {progress_text}",
        "",
        "Dataset Summary:",
        f"- data_root: {args.get('data_root', '-')}",
        f"- eval_name: {dataset.get('eval_name', '-')}",
        f"- num_classes: {dataset.get('num_classes', '-')}",
        f"- train_examples: {dataset.get('train_examples', '-')}",
        f"- eval_examples: {dataset.get('eval_examples', '-')}",
        f"- test_examples: {dataset.get('test_examples', '-')}",
        f"- validation_split: {dataset.get('use_validation_split', '-')}",
        f"- validation_proportion: {dataset.get('validation_proportion', '-')}",
        "",
        "Model Summary:",
        f"- total_params: {model_info.get('total_params', '-')}",
        f"- trainable_params: {model_info.get('trainable_params', '-')}",
        f"- frozen_params: {model_info.get('frozen_params', '-')}",
        f"- batch_size: {args.get('batch_size', '-')}",
        f"- lr: {args.get('lr', '-')}",
        f"- checkpoint_dir: {args.get('checkpoint_dir', '-')}",
        "",
        "Run Summary:",
        f"- best_eval_acc: {_format_metric(summary.get('best_eval_acc'))}",
        f"- best_eval_epoch: {summary.get('best_eval_epoch', '-')}",
        f"- last_completed_epoch: {summary.get('last_completed_epoch', '-')}",
        f"- last_eval_acc: {_format_metric(summary.get('last_eval_acc'))}",
        f"- last_eval_loss: {_format_metric(summary.get('last_eval_loss'))}",
        f"- final_test_acc: {_format_metric(summary.get('final_test_acc'))}",
        f"- final_test_loss: {_format_metric(summary.get('final_test_loss'))}",
        "",
        f"Expected Train Batches/Epoch: {expected.get('train_batches_per_epoch', '-')}",
        f"Expected Val Batches/Epoch: {expected.get('val_batches_per_epoch', '-')}",
        f"Expected Test Batches/Epoch: {expected.get('test_batches_per_epoch', '-')}",
        f"Expected Final Test Batches: {expected.get('final_test_batches', '-')}",
        f"Error Message: {run.get('error_message', '-')}",
        "",
        "Timing Summary:",
        f"- total_wall_time_seconds: {timing_summary.get('total_wall_time_seconds', '-')}",
        f"- total_pure_execution_time_seconds: {timing_summary.get('total_pure_execution_time_seconds', '-')}",
        f"- initialization_and_overhead_time_seconds: {timing_summary.get('initialization_and_overhead_time_seconds', '-')}",
    ]
    return "\n".join(lines)


def _render_stage_epochs(run: dict, stage_name: str) -> str:
    epochs = run.get("epochs") if isinstance(run.get("epochs"), list) else []
    stage_key = stage_name.lower()
    if stage_key == "test":
        final_test = run.get("final_test") if isinstance(run.get("final_test"), dict) else None
        if final_test:
            timing = final_test.get("timing", {})
            final_text = (
                f"Final test: loss={final_test.get('loss', '-')}, acc={final_test.get('acc', '-')}, "
                f"total_time={timing.get('total_seconds', '-')}, pure_time={timing.get('pure_seconds', '-')}, "
                f"batches={timing.get('batches', '-')}"
            )
            epoch_test_text = _render_stage_epochs({**run, "final_test": None}, "test")
            if epoch_test_text != "No test record in this run.":
                return final_text + "\n\nPer-epoch test:\n" + epoch_test_text
            return final_text

    if not epochs:
        return "No epoch records in this run."

    lines: list[str] = []
    for epoch_record in epochs:
        if not isinstance(epoch_record, dict):
            continue
        epoch_idx = epoch_record.get("epoch", "?")
        stage = epoch_record.get(stage_key)
        if not isinstance(stage, dict):
            continue
        timing = stage.get("timing", {})
        lr_text = _format_metric(epoch_record.get("lr"))
        best_text = _format_metric(epoch_record.get("best_eval_acc_after_epoch"))
        best_flag = "yes" if epoch_record.get("is_best_checkpoint") else "no"
        lines.append(
            f"Epoch {epoch_idx}: "
            f"loss={stage.get('loss', '-')}, acc={stage.get('acc', '-')}, "
            f"lr={lr_text}, best_eval_acc={best_text}, saved_best={best_flag}, "
            f"total_time={timing.get('total_seconds', '-')}, pure_time={timing.get('pure_seconds', '-')}, "
            f"batches={timing.get('batches', '-')}"
        )

    if not lines:
        return f"No {stage_key} records in this run."
    return "\n".join(lines)
