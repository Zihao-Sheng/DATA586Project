from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
from workflow.data import data_import
from workflow.log_analysis import display_confusion_matrix, plot_efficiency_tradeoff
from workflow.model_registry import discover_model_names, load_model_module
from workflow.predicting import (
    build_transform,
    display_gradcam_comparison,
    load_model as load_prediction_model,
    predict_images_batch,
)
from workflow.test_splits import evaluate_test_splits
from workflow.training_core import (
    TrainingRunLogger,
    build_error_analysis,
    check_stop_requested,
    count_parameters,
    default_checkpoint_dir_for_model,
    file_signature,
)


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _resolved_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _workflow_output_dir() -> Path:
    return PROJECT_ROOT / "logs" / "workflow_runs"


def list_workflow_runs() -> list[dict[str, object]]:
    output_dir = _workflow_output_dir()
    if not output_dir.is_dir():
        return []

    rows: list[dict[str, object]] = []
    for path in sorted(output_dir.glob("*.json"), reverse=True):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(summary, dict):
            continue
        summary_block = summary.get("summary") if isinstance(summary.get("summary"), dict) else {}
        artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
        rows.append(
            {
                "workflow_id": summary.get("workflow_id"),
                "generated_at_utc": summary.get("generated_at_utc"),
                "model_name": summary_block.get("model_name"),
                "best_eval_acc": summary_block.get("best_eval_acc"),
                "final_test_acc": summary_block.get("final_test_acc"),
                "clean_accuracy": summary_block.get("clean_accuracy"),
                "robustness_average": summary_block.get("robustness_average"),
                "path": str(path.resolve()),
                "training_run_log": artifacts.get("training_run_log"),
                "test_split_json": artifacts.get("test_split_json"),
            }
        )
    return rows


class NotebookProgress:
    def __init__(self, title: str = "Workflow") -> None:
        self.title = title
        self._display = None
        self._enabled = False
        try:
            from IPython.display import HTML, display

            self._HTML = HTML
            self._display = display("", display_id=True)
            self._enabled = self._display is not None
        except Exception:
            self._HTML = None
            self._display = None
            self._enabled = False
        self.update(message="Ready", completed=0, total=1)

    def update(self, *, message: str, completed: int | float | None = None, total: int | float | None = None) -> None:
        if not self._enabled or self._display is None or self._HTML is None:
            return
        progress_html = ""
        if completed is not None and total is not None and total > 0:
            fraction = max(0.0, min(float(completed) / float(total), 1.0))
            percent = int(round(fraction * 100))
            progress_html = (
                "<div style='margin-top:8px;'>"
                "<div style='height:10px;border-radius:999px;background:#e2e8f0;overflow:hidden;'>"
                f"<div style='height:10px;width:{percent}%;background:linear-gradient(90deg,#2563eb,#38bdf8);'></div>"
                "</div>"
                f"<div style='margin-top:6px;font-size:12px;color:#475569;'>{completed}/{total} ({percent}%)</div>"
                "</div>"
            )
        html = (
            "<div style='border:1px solid #cbd5e1;border-radius:12px;padding:12px 14px;background:#f8fafc;"
            "font-family:Segoe UI,Arial,sans-serif;'>"
            f"<div style='font-weight:700;color:#0f172a;margin-bottom:4px;'>{self.title}</div>"
            f"<div style='color:#334155;font-size:13px;'>{message}</div>"
            f"{progress_html}"
            "</div>"
        )
        self._display.update(self._HTML(html))

    def clear(self) -> None:
        if not self._enabled or self._display is None or self._HTML is None:
            return
        self._display.update(self._HTML(""))


def _train_one_epoch_notebook(
    model,
    dataloader,
    loss_fn,
    optimizer,
    device: str,
    *,
    epoch: int,
    num_epochs: int,
    stop_file: Path | None,
    progress: NotebookProgress | None,
) -> tuple[float, float, dict[str, float]]:
    stage_total_start = time.perf_counter()
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    total_steps = len(dataloader)
    pure_start = time.perf_counter()

    for step_idx, (images, labels) in enumerate(dataloader, start=1):
        check_stop_requested(stop_file)
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        if progress is not None and total > 0:
            progress.update(
                message=(
                    f"Epoch {epoch}/{num_epochs} - Training "
                    f"(loss={total_loss / total:.4f}, acc={correct / total:.4f})"
                ),
                completed=step_idx,
                total=total_steps,
            )

    pure_seconds = time.perf_counter() - pure_start
    total_seconds = time.perf_counter() - stage_total_start
    avg_loss = (total_loss / total) if total > 0 else 0.0
    accuracy = (correct / total) if total > 0 else 0.0
    timing = {
        "total_seconds": total_seconds,
        "pure_seconds": pure_seconds,
        "batches": total_steps,
    }
    return avg_loss, accuracy, timing


def _evaluate_notebook(
    model,
    dataloader,
    loss_fn,
    device: str,
    *,
    class_names: list[str],
    stage_name: str,
    epoch: int | None,
    num_epochs: int | None,
    stop_file: Path | None,
    progress: NotebookProgress | None,
) -> tuple[float, float, dict[str, float], dict[str, object]]:
    stage_total_start = time.perf_counter()
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    pair_counts: dict[tuple[int, int], int] = {}
    pair_confidence_sums: dict[tuple[int, int], float] = {}
    top_confidence_errors: list[dict[str, object]] = []
    total_steps = len(dataloader)
    pure_start = time.perf_counter()

    with torch.no_grad():
        for step_idx, (images, labels) in enumerate(dataloader, start=1):
            check_stop_requested(stop_file)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            confidences = probs.gather(1, preds.unsqueeze(1)).squeeze(1)

            labels_cpu = labels.detach().cpu().tolist()
            preds_cpu = preds.detach().cpu().tolist()
            confidences_cpu = confidences.detach().cpu().tolist()
            for true_idx, pred_idx, confidence in zip(labels_cpu, preds_cpu, confidences_cpu):
                pair_key = (int(true_idx), int(pred_idx))
                pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1
                pair_confidence_sums[pair_key] = pair_confidence_sums.get(pair_key, 0.0) + float(confidence)
                if int(true_idx) != int(pred_idx):
                    top_confidence_errors.append(
                        {
                            "true_idx": int(true_idx),
                            "pred_idx": int(pred_idx),
                            "true_label": class_names[int(true_idx)] if int(true_idx) < len(class_names) else str(true_idx),
                            "pred_label": class_names[int(pred_idx)] if int(pred_idx) < len(class_names) else str(pred_idx),
                            "confidence": float(confidence),
                        }
                    )
            if progress is not None and total > 0:
                prefix = f"Epoch {epoch}/{num_epochs} - " if epoch is not None and num_epochs is not None else ""
                progress.update(
                    message=(
                        f"{prefix}{stage_name.capitalize()} "
                        f"(loss={total_loss / total:.4f}, acc={correct / total:.4f})"
                    ),
                    completed=step_idx,
                    total=total_steps,
                )

    pure_seconds = time.perf_counter() - pure_start
    total_seconds = time.perf_counter() - stage_total_start
    avg_loss = (total_loss / total) if total > 0 else 0.0
    accuracy = (correct / total) if total > 0 else 0.0
    timing = {
        "total_seconds": total_seconds,
        "pure_seconds": pure_seconds,
        "batches": total_steps,
    }
    analysis = build_error_analysis(
        pair_counts=pair_counts,
        pair_confidence_sums=pair_confidence_sums,
        top_confidence_errors=top_confidence_errors,
        class_names=class_names,
        total_examples=total,
        correct_examples=correct,
    )
    return avg_loss, accuracy, timing, analysis


@dataclass(slots=True)
class WorkflowConfig:
    model_name: str
    epochs: int = 3
    batch_size: int = 32
    num_workers: int = 4
    image_size: int = 224
    lr: float = 1e-3
    device: str = "auto"
    data_root: Path = PROJECT_ROOT / "data" / "food-101"
    checkpoint_dir: Path | None = None
    resume_path: Path | None = None
    use_validation_split: bool = True
    validation_proportion: float = 0.1
    split_seed: int = 42
    evaluate_test_splits_root: Path | None = PROJECT_ROOT / "data" / "test_splits"
    evaluate_test_splits_after_training: bool = True
    test_split_checkpoint: str = "best"
    progress_format: str = "tqdm"
    stop_file: Path | None = None
    freeze_backbone: bool | None = None

    def resolved_checkpoint_dir(self) -> Path:
        return (
            self.checkpoint_dir.expanduser().resolve()
            if self.checkpoint_dir is not None
            else default_checkpoint_dir_for_model(self.model_name)
        )

    def resolved_freeze_backbone(self) -> bool:
        if self.freeze_backbone is not None:
            return bool(self.freeze_backbone)
        return self.model_name.endswith("_baseline")


def run_experiment_workflow(config: WorkflowConfig) -> dict[str, object]:
    if config.model_name not in discover_model_names():
        raise ValueError(
            f"Unsupported model '{config.model_name}'. Available models: {', '.join(discover_model_names())}"
        )

    progress = NotebookProgress(title=f"Workflow: {config.model_name}")
    wall_total_start = time.perf_counter()
    resolved_device = _resolved_device(config.device)
    freeze_backbone = config.resolved_freeze_backbone()
    checkpoint_dir = config.resolved_checkpoint_dir()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / "best.pth"
    last_checkpoint_path = checkpoint_dir / "last.pth"
    stop_file = config.stop_file.expanduser().resolve() if config.stop_file is not None else None
    data_root = config.data_root.expanduser().resolve()

    train_loader, val_loader, test_loader, class_to_idx, num_classes = data_import(
        data_root=data_root,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        image_size=config.image_size,
        pin_memory=resolved_device.startswith("cuda"),
        use_validation_split=config.use_validation_split,
        validation_proportion=config.validation_proportion,
        split_seed=config.split_seed,
    )
    eval_loader = val_loader if config.use_validation_split else test_loader
    eval_name = "val" if config.use_validation_split else "test"
    class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]

    model_module = load_model_module(config.model_name)
    model = model_module.build_model(
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        device=resolved_device,
    )
    optimizer = model_module.build_optimizer(model, lr=config.lr)
    total_params, trainable_params = count_parameters(model)

    start_epoch = 0
    best_acc = -1.0
    if config.resume_path is not None:
        checkpoint = torch.load(config.resume_path.expanduser().resolve(), map_location=resolved_device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        best_acc = float(checkpoint.get("best_acc", -1.0))

    train_examples = len(train_loader.dataset)
    eval_examples = len(eval_loader.dataset)
    test_examples = len(test_loader.dataset)

    class ArgsProxy:
        pass

    args_proxy = ArgsProxy()
    args_proxy.data_root = data_root
    args_proxy.batch_size = config.batch_size
    args_proxy.num_workers = config.num_workers
    args_proxy.image_size = config.image_size
    args_proxy.lr = config.lr
    args_proxy.freeze_backbone = freeze_backbone
    args_proxy.use_validation_split = config.use_validation_split
    args_proxy.validation_proportion = config.validation_proportion
    args_proxy.resume = config.resume_path

    run_logger = TrainingRunLogger(
        checkpoint_dir=checkpoint_dir,
        best_checkpoint_path=best_checkpoint_path,
        last_checkpoint_path=last_checkpoint_path,
        args=args_proxy,
        model_name=config.model_name,
        device=resolved_device,
        start_epoch=start_epoch,
        num_epochs=config.epochs,
        eval_name=eval_name,
        train_batches=len(train_loader),
        eval_batches=len(eval_loader),
        test_batches=len(test_loader),
        train_examples=train_examples,
        eval_examples=eval_examples,
        test_examples=test_examples,
        num_classes=num_classes,
        class_names=class_names,
        total_params=total_params,
        trainable_params=trainable_params,
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        f"{eval_name}_loss": [],
        f"{eval_name}_acc": [],
        "test_loss": [],
        "test_acc": [],
    }
    stage_totals: dict[str, dict[str, float]] = {
        "train": {"total_seconds": 0.0, "pure_seconds": 0.0, "batches": 0.0},
        "val": {"total_seconds": 0.0, "pure_seconds": 0.0, "batches": 0.0},
        "test": {"total_seconds": 0.0, "pure_seconds": 0.0, "batches": 0.0},
    }
    final_test_loss: float | None = None
    final_test_acc: float | None = None
    final_test_analysis: dict[str, object] | None = None
    final_test_timing: dict[str, float] | None = None

    try:
        for epoch in range(start_epoch, config.epochs):
            check_stop_requested(stop_file)
            train_loss, train_acc, train_timing = _train_one_epoch_notebook(
                model,
                train_loader,
                nn.CrossEntropyLoss(),
                optimizer,
                resolved_device,
                epoch=epoch + 1,
                num_epochs=config.epochs,
                stop_file=stop_file,
                progress=progress,
            )
            eval_loss, eval_acc, eval_timing, eval_analysis = _evaluate_notebook(
                model,
                eval_loader,
                nn.CrossEntropyLoss(),
                resolved_device,
                class_names=class_names,
                epoch=epoch + 1,
                num_epochs=config.epochs,
                stage_name=eval_name,
                stop_file=stop_file,
                progress=progress,
            )

            history["train_loss"].append(float(train_loss))
            history["train_acc"].append(float(train_acc))
            history[f"{eval_name}_loss"].append(float(eval_loss))
            history[f"{eval_name}_acc"].append(float(eval_acc))

            stage_totals["train"]["total_seconds"] += float(train_timing["total_seconds"])
            stage_totals["train"]["pure_seconds"] += float(train_timing["pure_seconds"])
            stage_totals["train"]["batches"] += float(train_timing["batches"])
            stage_totals[eval_name]["total_seconds"] += float(eval_timing["total_seconds"])
            stage_totals[eval_name]["pure_seconds"] += float(eval_timing["pure_seconds"])
            stage_totals[eval_name]["batches"] += float(eval_timing["batches"])

            is_best_checkpoint = eval_acc > best_acc
            if is_best_checkpoint:
                best_acc = float(eval_acc)
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_name": config.model_name,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_acc": best_acc,
                        "num_classes": num_classes,
                        "class_to_idx": class_to_idx,
                        "use_validation_split": config.use_validation_split,
                        "validation_proportion": config.validation_proportion,
                    },
                    best_checkpoint_path,
                )
                run_logger.mark_best_checkpoint(epoch=epoch + 1, best_acc=best_acc, path=best_checkpoint_path)

            current_lr = float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else float(config.lr)
            run_logger.append_epoch(
                epoch=epoch + 1,
                train_loss=train_loss,
                train_acc=train_acc,
                train_timing=train_timing,
                eval_name=eval_name,
                eval_loss=eval_loss,
                eval_acc=eval_acc,
                eval_timing=eval_timing,
                eval_analysis=eval_analysis,
                lr=current_lr,
                best_acc_after_epoch=best_acc,
                is_best_checkpoint=is_best_checkpoint,
            )

        final_test_loss, final_test_acc, final_test_timing, final_test_analysis = _evaluate_notebook(
            model,
            test_loader,
            nn.CrossEntropyLoss(),
            resolved_device,
            class_names=class_names,
            stage_name="test",
            stop_file=stop_file,
            epoch=None,
            num_epochs=None,
            progress=progress,
        )
        history["test_loss"].append(float(final_test_loss))
        history["test_acc"].append(float(final_test_acc))
        stage_totals["test"]["total_seconds"] += float(final_test_timing["total_seconds"])
        stage_totals["test"]["pure_seconds"] += float(final_test_timing["pure_seconds"])
        stage_totals["test"]["batches"] += float(final_test_timing["batches"])
        run_logger.set_final_test(
            loss=final_test_loss,
            acc=final_test_acc,
            timing=final_test_timing,
            analysis=final_test_analysis,
        )

        torch.save(
            {
                "epoch": config.epochs,
                "model_name": config.model_name,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_acc,
                "num_classes": num_classes,
                "class_to_idx": class_to_idx,
                "use_validation_split": config.use_validation_split,
                "validation_proportion": config.validation_proportion,
            },
            last_checkpoint_path,
        )
        run_logger.mark_last_checkpoint(path=last_checkpoint_path)

        pure_execution_total = (
            stage_totals["train"]["pure_seconds"]
            + stage_totals["val"]["pure_seconds"]
            + stage_totals["test"]["pure_seconds"]
        )
        wall_total_elapsed = time.perf_counter() - wall_total_start
        init_and_overhead = max(wall_total_elapsed - pure_execution_total, 0.0)
        run_logger.finalize(
            status="completed",
            stage_totals=stage_totals,
            wall_total_elapsed=wall_total_elapsed,
            pure_execution_total=pure_execution_total,
            init_and_overhead=init_and_overhead,
            status_reason="completed_normally",
        )
    except KeyboardInterrupt:
        pure_execution_total = (
            stage_totals["train"]["pure_seconds"]
            + stage_totals["val"]["pure_seconds"]
            + stage_totals["test"]["pure_seconds"]
        )
        wall_total_elapsed = time.perf_counter() - wall_total_start
        init_and_overhead = max(wall_total_elapsed - pure_execution_total, 0.0)
        run_logger.finalize(
            status="interrupted",
            stage_totals=stage_totals,
            wall_total_elapsed=wall_total_elapsed,
            pure_execution_total=pure_execution_total,
            init_and_overhead=init_and_overhead,
            error_message="KeyboardInterrupt",
            status_reason="keyboard_interrupt",
        )
        raise
    except Exception as exc:
        progress.clear()
        pure_execution_total = (
            stage_totals["train"]["pure_seconds"]
            + stage_totals["val"]["pure_seconds"]
            + stage_totals["test"]["pure_seconds"]
        )
        wall_total_elapsed = time.perf_counter() - wall_total_start
        init_and_overhead = max(wall_total_elapsed - pure_execution_total, 0.0)
        run_logger.finalize(
            status="failed",
            stage_totals=stage_totals,
            wall_total_elapsed=wall_total_elapsed,
            pure_execution_total=pure_execution_total,
            init_and_overhead=init_and_overhead,
            error_message=f"{type(exc).__name__}: {exc}",
            status_reason=type(exc).__name__,
        )
        raise

    test_split_json_path: Path | None = None
    test_split_csv_path: Path | None = None
    test_split_payload: dict[str, object] | None = None
    if config.evaluate_test_splits_after_training and config.evaluate_test_splits_root is not None:
        selected_checkpoint = best_checkpoint_path if config.test_split_checkpoint == "best" else last_checkpoint_path
        test_split_payload, test_split_json_path, test_split_csv_path = evaluate_test_splits(
            checkpoint_path=selected_checkpoint,
            model_name=config.model_name,
            test_splits_root=config.evaluate_test_splits_root,
            image_size=config.image_size,
            device=resolved_device,
            output_dir=PROJECT_ROOT / "logs" / "test_split_evaluations",
            status_callback=lambda message, indeterminate: progress.update(message=message),
            progress_callback=lambda processed, total: progress.update(
                message="Evaluating test splits",
                completed=processed,
                total=total,
            ),
        )

    workflow_output_dir = _workflow_output_dir()
    workflow_output_dir.mkdir(parents=True, exist_ok=True)
    workflow_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    workflow_summary_path = workflow_output_dir / f"{workflow_id}.json"

    workflow_summary: dict[str, object] = {
        "workflow_id": workflow_id,
        "generated_at_utc": _now_iso_utc(),
        "config": {
            **asdict(config),
            "data_root": str(data_root),
            "checkpoint_dir": str(checkpoint_dir),
            "resume_path": str(config.resume_path.expanduser().resolve()) if config.resume_path is not None else None,
            "evaluate_test_splits_root": (
                str(config.evaluate_test_splits_root.expanduser().resolve())
                if config.evaluate_test_splits_root is not None
                else None
            ),
            "stop_file": str(stop_file) if stop_file is not None else None,
        },
        "artifacts": {
            "training_run_log": str(run_logger.path),
            "best_checkpoint": {
                "path": str(best_checkpoint_path),
                "signature": file_signature(best_checkpoint_path),
            },
            "last_checkpoint": {
                "path": str(last_checkpoint_path),
                "signature": file_signature(last_checkpoint_path),
            },
            "test_split_json": str(test_split_json_path) if test_split_json_path is not None else None,
            "test_split_csv": str(test_split_csv_path) if test_split_csv_path is not None else None,
        },
        "summary": {
            "model_name": config.model_name,
            "best_eval_acc": float(best_acc) if best_acc >= 0 else None,
            "final_test_acc": float(final_test_acc) if final_test_acc is not None else None,
            "final_test_loss": float(final_test_loss) if final_test_loss is not None else None,
            "clean_accuracy": (
                float(test_split_payload.get("clean_accuracy", 0.0))
                if isinstance(test_split_payload, dict)
                else None
            ),
            "robustness_average": (
                float(test_split_payload.get("robustness_average", 0.0))
                if isinstance(test_split_payload, dict)
                else None
            ),
            "total_params": int(total_params),
            "trainable_params": int(trainable_params),
        },
        "history": history,
        "final_test_analysis": final_test_analysis,
        "test_split_summary": test_split_payload,
    }
    workflow_summary_path.write_text(json.dumps(workflow_summary, indent=2), encoding="utf-8")
    progress.update(message="Workflow finished", completed=1, total=1)
    return workflow_summary


def load_workflow_summary(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def plot_training_history(
    workflow_summary_or_path: dict[str, object] | str | Path,
    metric: str = "accuracy",
) -> None:
    import matplotlib.pyplot as plt

    summary = (
        load_workflow_summary(workflow_summary_or_path)
        if isinstance(workflow_summary_or_path, (str, Path))
        else workflow_summary_or_path
    )
    history = summary.get("history") if isinstance(summary.get("history"), dict) else {}
    if metric == "loss":
        train_series = history.get("train_loss", [])
        eval_key = "val_loss" if history.get("val_loss") else "test_loss"
        eval_series = history.get(eval_key, [])
        y_label = "Loss"
        title = "Training Loss History"
    else:
        train_series = history.get("train_acc", [])
        eval_key = "val_acc" if history.get("val_acc") else "test_acc"
        eval_series = history.get(eval_key, [])
        y_label = "Accuracy"
        title = "Training Accuracy History"

    if not train_series and not eval_series:
        print("No training history available.")
        return

    plt.figure(figsize=(9, 5))
    if train_series:
        plt.plot(range(1, len(train_series) + 1), train_series, marker="o", label="Train")
    if eval_series:
        label = "Validation" if eval_key.startswith("val") else "Test"
        plt.plot(range(1, len(eval_series) + 1), eval_series, marker="s", label=label)
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.show()


def show_confusion_matrix(training_log_path: str | Path, view: str = "summary", top_k: int = 10) -> None:
    display_confusion_matrix([Path(training_log_path).expanduser().resolve()], view=view, top_k=top_k)


def plot_efficiency_compare(training_log_paths: list[str | Path], x_metric: str = "Train Wall Time") -> None:
    resolved = [Path(path).expanduser().resolve() for path in training_log_paths]
    plot_efficiency_tradeoff(resolved, x_metric=x_metric)


def compare_test_split_results(test_split_json_paths: list[str | Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in test_split_json_paths:
        payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        row: dict[str, object] = {
            "model_name": payload.get("model_name"),
            "clean_accuracy": payload.get("clean_accuracy"),
            "robustness_average": payload.get("robustness_average"),
            "total_seconds": payload.get("total_seconds"),
        }
        for split in payload.get("splits", []):
            if not isinstance(split, dict):
                continue
            row[f"split::{split.get('split')}"] = split.get("accuracy")
        rows.append(row)
    return rows


def plot_test_split_comparison(test_split_json_paths: list[str | Path]) -> list[dict[str, object]]:
    import matplotlib.pyplot as plt

    rows = compare_test_split_results(test_split_json_paths)
    if not rows:
        print("No test split summaries available.")
        return rows

    model_names = [str(row.get("model_name", f"model-{index+1}")) for index, row in enumerate(rows)]
    split_names = sorted(
        {
            key.split("::", 1)[1]
            for row in rows
            for key in row.keys()
            if isinstance(key, str) and key.startswith("split::")
        }
    )
    if not split_names:
        print("No split metrics available.")
        return rows

    plt.figure(figsize=(max(9, len(split_names) * 1.6), 5.5))
    for row, model_name in zip(rows, model_names):
        values = [float(row.get(f"split::{split_name}", 0.0) or 0.0) for split_name in split_names]
        plt.plot(split_names, values, marker="o", label=model_name)
    plt.ylabel("Accuracy")
    plt.title("Test Split Comparison")
    plt.xticks(rotation=30, ha="right")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return rows


def show_gradcam_compare(
    *,
    image_path: str | Path,
    model_specs: list[tuple[str, str | Path]],
    image_size: int = 224,
    device: str = "auto",
) -> None:
    progress = NotebookProgress(title="Grad-CAM Compare")
    progress.update(message="Generating Grad-CAM overlays")
    resolved_specs = [
        (str(model_name), Path(checkpoint_path).expanduser().resolve())
        for model_name, checkpoint_path in model_specs
    ]
    display_gradcam_comparison(
        image_path=Path(image_path).expanduser().resolve(),
        model_specs=resolved_specs,
        image_size=image_size,
        device=device,
    )
    progress.clear()


def show_prediction_compare(
    *,
    image_paths: list[str | Path],
    model_specs: list[tuple[str, str | Path]],
    image_size: int = 224,
    device: str = "auto",
) -> list[dict[str, object]]:
    from IPython.display import HTML, display

    progress = NotebookProgress(title="Prediction Compare")
    resolved_device = _resolved_device(device)
    resolved_images = [Path(path).expanduser().resolve() for path in image_paths]
    resolved_specs = [(str(model_name), Path(checkpoint_path).expanduser().resolve()) for model_name, checkpoint_path in model_specs]
    transform = build_transform(image_size)
    combined: dict[str, dict[str, object]] = {
        str(path): {"image_path": path, "comparisons": {}, "actual_label": path.parent.name if path.parent.name else None}
        for path in resolved_images
    }

    total_steps = max(len(resolved_images) * max(len(resolved_specs), 1), 1)
    completed_steps = 0
    for model_name, checkpoint_path in resolved_specs:
        progress.update(message=f"Loading {model_name}", completed=completed_steps, total=total_steps)
        model, class_to_idx = load_prediction_model(checkpoint_path, model_name, resolved_device)
        idx_to_class = {idx: name for name, idx in class_to_idx.items()}
        batch_results = predict_images_batch(
            model,
            resolved_images,
            transform,
            idx_to_class,
            resolved_device,
            batch_size=16,
            progress_callback=lambda processed, total, base=completed_steps: progress.update(
                message=f"Running {model_name}",
                completed=base + processed,
                total=total_steps,
            ),
        )
        completed_steps += len(resolved_images)
        for result in batch_results:
            resolved_image = Path(str(result["image_path"])).resolve()
            actual_label = resolved_image.parent.name if resolved_image.parent.name in class_to_idx else None
            row = combined[str(resolved_image)]
            comparisons = row["comparisons"]
            assert isinstance(comparisons, dict)
            comparisons[model_name] = {
                **result,
                "checkpoint_path": str(checkpoint_path),
                "actual_label": actual_label,
                "is_correct": None if actual_label is None else result["predicted_class"] == actual_label,
            }

    results = [combined[str(path)] for path in resolved_images]
    cards: list[str] = []
    for result in results:
        image_path = Path(str(result["image_path"]))
        actual_label = result.get("actual_label")
        actual_text = str(actual_label) if actual_label is not None else "Unknown"
        lines = []
        comparisons = result.get("comparisons") if isinstance(result.get("comparisons"), dict) else {}
        for model_name, _ in resolved_specs:
            entry = comparisons.get(model_name)
            if not isinstance(entry, dict):
                continue
            status = entry.get("is_correct")
            status_text = "Correct" if status is True else ("Wrong" if status is False else "Unknown")
            lines.append(
                f"<div style='font-size:12px;color:#334155;margin-top:4px;'><b>{model_name}</b>: "
                f"{entry.get('predicted_class', '-')} ({float(entry.get('confidence', 0.0)):.2%}, {status_text})</div>"
            )
        cards.append(
            f"""
            <div style="border:1px solid #cbd5e1;border-radius:14px;padding:12px;background:#ffffff;box-shadow:0 4px 14px rgba(15,23,42,.06);">
              <div style="font-weight:700;color:#0f172a;margin-bottom:6px;">{image_path.name}</div>
              <div style="color:#475569;font-size:13px;margin-bottom:6px;">Ground Truth: {actual_text}</div>
              {''.join(lines)}
            </div>
            """
        )
    display(HTML("<div style=\"display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px;\">" + "".join(cards) + "</div>"))
    progress.clear()
    return results


def show_workflow_summary_table(
    workflow_summaries_or_paths: list[dict[str, object] | str | Path] | dict[str, object] | str | Path,
) -> list[dict[str, object]]:
    from IPython.display import HTML, display

    items = workflow_summaries_or_paths
    if not isinstance(items, list):
        items = [items]
    rows: list[dict[str, object]] = []
    for item in items:
        summary = load_workflow_summary(item) if isinstance(item, (str, Path)) else item
        if not isinstance(summary, dict):
            continue
        summary_block = summary.get("summary") if isinstance(summary.get("summary"), dict) else {}
        artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
        rows.append(
            {
                "model_name": summary_block.get("model_name"),
                "best_eval_acc": summary_block.get("best_eval_acc"),
                "final_test_acc": summary_block.get("final_test_acc"),
                "clean_accuracy": summary_block.get("clean_accuracy"),
                "robustness_average": summary_block.get("robustness_average"),
                "trainable_params": summary_block.get("trainable_params"),
                "training_run_log": artifacts.get("training_run_log"),
                "test_split_json": artifacts.get("test_split_json"),
            }
        )

    if not rows:
        return rows

    headers = ["Model", "Best Eval", "Final Test", "Clean", "Robustness Avg", "Trainable Params"]
    body_rows = []
    for row in rows:
        body_rows.append(
            "<tr style='border-bottom:1px solid #e2e8f0;'>"
            f"<td style='padding:8px 10px;'>{row.get('model_name', '-')}</td>"
            f"<td style='padding:8px 10px;'>{_format_table_value(row.get('best_eval_acc'))}</td>"
            f"<td style='padding:8px 10px;'>{_format_table_value(row.get('final_test_acc'))}</td>"
            f"<td style='padding:8px 10px;'>{_format_table_value(row.get('clean_accuracy'))}</td>"
            f"<td style='padding:8px 10px;'>{_format_table_value(row.get('robustness_average'))}</td>"
            f"<td style='padding:8px 10px;'>{_format_table_int(row.get('trainable_params'))}</td>"
            "</tr>"
        )
    html = (
        "<table style='border-collapse:collapse;width:100%;font-family:Segoe UI,Arial,sans-serif;font-size:13px;'>"
        "<thead><tr>"
        + "".join(f"<th style='text-align:left;padding:8px 10px;border-bottom:2px solid #cbd5e1;background:#f8fafc;'>{header}</th>" for header in headers)
        + "</tr></thead><tbody>"
        + "".join(body_rows)
        + "</tbody></table>"
    )
    display(HTML(html))
    return rows


def _format_table_value(value) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return "-"


def _format_table_int(value) -> str:
    if isinstance(value, (int, float)):
        return f"{int(value):,}"
    return "-"
