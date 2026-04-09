from __future__ import annotations

from pathlib import Path

from core.run_log_compat import (
    extract_epoch_metrics,
    extract_run_summary,
    format_metric,
    format_ratio,
    infer_eval_name,
    load_run_log,
    normalize_run_status,
    run_display_name,
)


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
            blocks.append(run_display_name(run))
            blocks.append(_render_stage_epochs(run, normalized_view))
            blocks.append("")
        return "\n".join(blocks).strip()
    if normalized_view == "summary":
        return _render_run_summary(runs[0])
    return _render_stage_epochs(runs[0], normalized_view)


def _load_run(path: Path) -> dict | None:
    return load_run_log(path)


def _render_compare_runs(runs: list[dict]) -> str:
    header = (
        f"{'Started':<22} {'Model':<12} {'Status':<14} {'Progress':<9} "
        f"{'BestEval':<10} {'FinalTest':<10} {'Eval':<6} {'Batch':<6} {'LR':<10} {'Checkpoint'}"
    )
    separator = "-" * len(header)
    lines = [header, separator, "", "Average Timing Compare:"]
    for run in runs:
        run_summary = extract_run_summary(run)
        args = run_summary["args"]
        summary = run_summary["summary"]
        timing_summary = run_summary["timing_summary"]
        stage_totals = timing_summary.get("stage_totals") if isinstance(timing_summary.get("stage_totals"), dict) else {}
        started = str(run_summary["start_time_utc"])[:19]
        model = str(args.get("model", "-"))[:12]
        status = str(run_summary["status"])[:14]
        progress = format_ratio(run_summary["last_completed_epoch"], args.get("planned_epochs_this_run"))
        best_eval = format_metric(run_summary["best_eval_acc"])
        final_test = format_metric(summary.get("final_test_acc"))
        eval_name = infer_eval_name(run)[:6]
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
            float(train_stage.get("total_seconds", 0.0)) / max(float(run_summary["last_completed_epoch"]), 1.0)
            if run_summary["last_completed_epoch"] > 0 and isinstance(train_stage.get("total_seconds"), (int, float))
            else None
        )
        test_avg_epoch = (
            float(test_stage.get("total_seconds", 0.0)) / max(float(run_summary["last_completed_epoch"]), 1.0)
            if run_summary["last_completed_epoch"] > 0 and isinstance(test_stage.get("total_seconds"), (int, float))
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
            f"  avg_train_time_per_epoch={format_metric(train_avg_epoch)}s, "
            f"avg_test_time_per_epoch={format_metric(test_avg_epoch)}s, "
            f"avg_train_pure_per_batch={format_metric(train_avg_batch)}s, "
            f"avg_test_pure_per_batch={format_metric(test_avg_batch)}s"
        )
    return "\n".join(lines)


def _render_run_summary(run: dict) -> str:
    run_summary = extract_run_summary(run)
    args = run_summary["args"]
    dataset = run_summary["dataset"]
    model_info = run_summary["model"]
    expected = run_summary["expected"]
    epochs = run_summary["epochs"]
    summary = run_summary["summary"]
    timing_summary = run_summary["timing_summary"]

    planned_epochs = int(args.get("planned_epochs_this_run", 0)) if isinstance(args.get("planned_epochs_this_run"), (int, float)) else 0
    completed_epochs = len(epochs)
    progress_text = f"{completed_epochs}/{planned_epochs}" if planned_epochs > 0 else str(completed_epochs)

    lines = [
        f"Run ID: {run_summary['run_id']}",
        f"Status: {normalize_run_status(run)}",
        f"Status Reason: {run_summary['status_reason']}",
        f"Started (UTC): {run_summary['start_time_utc']}",
        f"Ended (UTC): {run_summary['end_time_utc']}",
        f"Model: {args.get('model', '-')}",
        f"Device: {args.get('device', '-')}",
        f"Command: {run.get('command', '-')}",
        f"Planned Epochs / Completed Epochs: {progress_text}",
        "",
        "Dataset Summary:",
        f"- data_root: {args.get('data_root', '-')}",
        f"- eval_name: {dataset.get('eval_name', '-')}",
        f"- train_transforms_preset: {args.get('train_transforms_preset', dataset.get('train_transforms_preset', '-'))}",
        f"- mild_blur_enabled: {args.get('mild_blur_enabled', dataset.get('mild_blur_enabled', '-'))}",
        f"- mild_blur_prob: {args.get('mild_blur_prob', dataset.get('mild_blur_prob', '-'))}",
        f"- augmentation_config: {dataset.get('augmentation_config', args.get('augmentation_config', '-'))}",
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
        f"- optimizer: {args.get('optimizer', '-')}",
        f"- scheduler: {args.get('scheduler', '-')}",
        f"- amp_requested: {args.get('amp', '-')}",
        f"- amp_enabled: {model_info.get('amp_enabled', args.get('amp', '-'))}",
        f"- seed: {args.get('seed', '-')}",
        f"- checkpoint_dir: {args.get('checkpoint_dir', '-')}",
        "",
        "Run Summary:",
        f"- best_eval_acc: {format_metric(summary.get('best_eval_acc'))}",
        f"- best_eval_epoch: {summary.get('best_eval_epoch', '-')}",
        f"- last_completed_epoch: {summary.get('last_completed_epoch', '-')}",
        f"- last_eval_acc: {format_metric(summary.get('last_eval_acc'))}",
        f"- last_eval_loss: {format_metric(summary.get('last_eval_loss'))}",
        f"- final_test_acc: {format_metric(summary.get('final_test_acc'))}",
        f"- final_test_loss: {format_metric(summary.get('final_test_loss'))}",
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

    epochs = run.get("epochs") if isinstance(run.get("epochs"), list) else []
    if not epochs:
        return "No epoch records in this run."

    points = {int(epoch): value for epoch, value in extract_epoch_metrics(run, stage_key, "accuracy")}
    lines: list[str] = []
    for epoch_record in epochs:
        if not isinstance(epoch_record, dict):
            continue
        epoch_idx = epoch_record.get("epoch", "?")
        stage = epoch_record.get(stage_key)
        if not isinstance(stage, dict):
            continue
        timing = stage.get("timing", {})
        acc_value = points.get(int(epoch_idx)) if isinstance(epoch_idx, (int, float)) else None
        acc_text = stage.get("acc", "-") if acc_value is not None else "-"
        lr_text = format_metric(epoch_record.get("lr"))
        best_text = format_metric(epoch_record.get("best_eval_acc_after_epoch"))
        best_flag = "yes" if epoch_record.get("is_best_checkpoint") else "no"
        lines.append(
            f"Epoch {epoch_idx}: "
            f"loss={stage.get('loss', '-')}, acc={acc_text}, "
            f"lr={lr_text}, best_eval_acc={best_text}, saved_best={best_flag}, "
            f"total_time={timing.get('total_seconds', '-')}, pure_time={timing.get('pure_seconds', '-')}, "
            f"batches={timing.get('batches', '-')}"
        )

    if not lines:
        return f"No {stage_key} records in this run."
    return "\n".join(lines)
