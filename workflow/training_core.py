from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch.nn as nn


RUN_LOG_DIRNAME = "_run_logs"


def default_checkpoint_root() -> Path:
    return Path(__file__).resolve().parents[1] / "checkpoints"


def default_checkpoint_dir_for_model(model_name: str) -> Path:
    return default_checkpoint_root() / model_name


def now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def check_stop_requested(stop_file: Path | None) -> None:
    if stop_file is not None and stop_file.exists():
        raise KeyboardInterrupt


def file_signature(path: Path) -> dict[str, int | bool]:
    if not path.is_file():
        return {"exists": False}
    stat = path.stat()
    return {"exists": True, "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total_params, trainable_params


def build_error_analysis(
    *,
    pair_counts: dict[tuple[int, int], int],
    pair_confidence_sums: dict[tuple[int, int], float],
    top_confidence_errors: list[dict[str, object]],
    class_names: list[str],
    total_examples: int,
    correct_examples: int,
    top_pairs_limit: int = 30,
    top_confidence_limit: int = 20,
) -> dict[str, object]:
    confusion_pairs: list[dict[str, object]] = []
    for (true_idx, pred_idx), count in sorted(pair_counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1])):
        avg_conf = pair_confidence_sums[(true_idx, pred_idx)] / max(count, 1)
        confusion_pairs.append(
            {
                "true_idx": int(true_idx),
                "pred_idx": int(pred_idx),
                "true_label": class_names[true_idx],
                "pred_label": class_names[pred_idx],
                "count": int(count),
                "avg_confidence": float(avg_conf),
            }
        )
    misclassified_total = max(total_examples - correct_examples, 0)
    return {
        "total_examples": int(total_examples),
        "correct_examples": int(correct_examples),
        "misclassified_examples": int(misclassified_total),
        "accuracy": float(correct_examples / total_examples) if total_examples > 0 else 0.0,
        "class_names": list(class_names),
        "confusion_pairs": confusion_pairs,
        "top_misclassifications": confusion_pairs[:top_pairs_limit],
        "top_confidence_errors": sorted(
            top_confidence_errors,
            key=lambda item: (-float(item.get("confidence", 0.0)), str(item.get("true_label", "")), str(item.get("pred_label", ""))),
        )[:top_confidence_limit],
    }


class TrainingRunLogger:
    def __init__(
        self,
        *,
        checkpoint_dir: Path,
        best_checkpoint_path: Path,
        last_checkpoint_path: Path,
        args,
        model_name: str,
        device: str,
        start_epoch: int,
        num_epochs: int,
        eval_name: str,
        train_batches: int,
        eval_batches: int,
        test_batches: int,
        train_examples: int,
        eval_examples: int,
        test_examples: int,
        num_classes: int,
        class_names: list[str],
        total_params: int,
        trainable_params: int,
    ) -> None:
        run_logs_dir = checkpoint_dir / RUN_LOG_DIRNAME
        run_logs_dir.mkdir(parents=True, exist_ok=True)
        run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
        self.path = run_logs_dir / f"{run_id}.json"
        self.data: dict[str, object] = {
            "schema_version": 3,
            "run_id": run_id,
            "status": "running",
            "start_time_utc": now_iso_utc(),
            "end_time_utc": None,
            "error_message": None,
            "status_reason": None,
            "command": "workflow.run_experiment_workflow",
            "args": {
                "model": model_name,
                "data_root": str(args.data_root),
                "checkpoint_dir": str(checkpoint_dir),
                "epochs": int(num_epochs),
                "start_epoch": int(start_epoch),
                "planned_epochs_this_run": int(max(num_epochs - start_epoch, 0)),
                "batch_size": int(args.batch_size),
                "num_workers": int(args.num_workers),
                "image_size": int(args.image_size),
                "lr": float(args.lr),
                "device": device,
                "freeze_backbone": bool(args.freeze_backbone),
                "use_validation_split": bool(args.use_validation_split),
                "validation_proportion": float(args.validation_proportion),
                "resume": str(args.resume) if args.resume is not None else None,
            },
            "dataset": {
                "num_classes": int(num_classes),
                "class_count_from_mapping": int(num_classes),
                "class_names": list(class_names),
                "train_examples": int(train_examples),
                "eval_examples": int(eval_examples),
                "test_examples": int(test_examples),
                "eval_name": eval_name,
                "use_validation_split": bool(args.use_validation_split),
                "validation_proportion": float(args.validation_proportion),
            },
            "model": {
                "name": model_name,
                "total_params": int(total_params),
                "trainable_params": int(trainable_params),
                "frozen_params": int(total_params - trainable_params),
                "checkpoint_dir": str(checkpoint_dir),
                "best_checkpoint_path": str(best_checkpoint_path),
                "last_checkpoint_path": str(last_checkpoint_path),
            },
            "expected": {
                "train_batches_per_epoch": int(train_batches),
                f"{eval_name}_batches_per_epoch": int(eval_batches),
                "final_test_batches": int(test_batches),
            },
            "epochs": [],
            "final_test": None,
            "summary": {
                "best_eval_acc": None,
                "best_eval_epoch": None,
                "last_completed_epoch": int(start_epoch),
                "last_eval_acc": None,
                "last_eval_loss": None,
                "final_test_acc": None,
                "final_test_loss": None,
            },
            "timing_summary": None,
            "analysis": {
                "last_eval_stage": eval_name,
                "last_eval": None,
                "final_test": None,
            },
            "artifacts": {
                "best_checkpoint": {
                    "path": str(best_checkpoint_path),
                    "initial_signature": file_signature(best_checkpoint_path),
                    "final_signature": None,
                    "saved_epoch": None,
                    "saved_best_acc": None,
                },
                "last_checkpoint": {
                    "path": str(last_checkpoint_path),
                    "initial_signature": file_signature(last_checkpoint_path),
                    "final_signature": None,
                },
            },
        }
        self.write()

    def write(self) -> None:
        self.path.write_text(json.dumps(self.data, ensure_ascii=True, indent=2), encoding="utf-8")

    def append_epoch(self, **kwargs) -> None:
        epoch = int(kwargs["epoch"])
        eval_name = str(kwargs["eval_name"])
        epochs = self.data["epochs"]
        assert isinstance(epochs, list)
        epochs.append(
            {
                "epoch": epoch,
                "train": {"loss": float(kwargs["train_loss"]), "acc": float(kwargs["train_acc"]), "timing": kwargs["train_timing"]},
                eval_name: {"loss": float(kwargs["eval_loss"]), "acc": float(kwargs["eval_acc"]), "timing": kwargs["eval_timing"]},
                "lr": float(kwargs["lr"]),
                "best_eval_acc_after_epoch": float(kwargs["best_acc_after_epoch"]),
                "is_best_checkpoint": bool(kwargs["is_best_checkpoint"]),
            }
        )
        summary = self.data["summary"]
        assert isinstance(summary, dict)
        summary["last_completed_epoch"] = epoch
        summary["last_eval_acc"] = float(kwargs["eval_acc"])
        summary["last_eval_loss"] = float(kwargs["eval_loss"])
        if summary.get("best_eval_acc") is None or float(kwargs["best_acc_after_epoch"]) >= float(summary.get("best_eval_acc")):
            summary["best_eval_acc"] = float(kwargs["best_acc_after_epoch"])
            if kwargs["is_best_checkpoint"]:
                summary["best_eval_epoch"] = epoch
        analysis = self.data["analysis"]
        assert isinstance(analysis, dict)
        analysis["last_eval"] = kwargs["eval_analysis"]
        self.write()

    def mark_best_checkpoint(self, *, epoch: int, best_acc: float, path: Path) -> None:
        best = self.data["artifacts"]["best_checkpoint"]
        assert isinstance(best, dict)
        best["saved_epoch"] = int(epoch)
        best["saved_best_acc"] = float(best_acc)
        best["final_signature"] = file_signature(path)
        self.write()

    def mark_last_checkpoint(self, *, path: Path) -> None:
        last = self.data["artifacts"]["last_checkpoint"]
        assert isinstance(last, dict)
        last["final_signature"] = file_signature(path)
        self.write()

    def set_final_test(self, *, loss: float, acc: float, timing: dict[str, float], analysis: dict[str, object]) -> None:
        self.data["final_test"] = {"loss": float(loss), "acc": float(acc), "timing": timing, "analysis": analysis}
        summary = self.data["summary"]
        assert isinstance(summary, dict)
        summary["final_test_loss"] = float(loss)
        summary["final_test_acc"] = float(acc)
        analysis_root = self.data["analysis"]
        assert isinstance(analysis_root, dict)
        analysis_root["final_test"] = analysis
        self.write()

    def finalize(
        self,
        *,
        status: str,
        stage_totals: dict[str, dict[str, float]],
        wall_total_elapsed: float,
        pure_execution_total: float,
        init_and_overhead: float,
        error_message: str | None = None,
        status_reason: str | None = None,
    ) -> None:
        self.data["status"] = status
        self.data["end_time_utc"] = now_iso_utc()
        self.data["error_message"] = error_message
        self.data["status_reason"] = status_reason
        self.data["timing_summary"] = {
            "total_wall_time_seconds": float(wall_total_elapsed),
            "total_pure_execution_time_seconds": float(pure_execution_total),
            "initialization_and_overhead_time_seconds": float(init_and_overhead),
            "stage_totals": stage_totals,
        }
        self.write()
