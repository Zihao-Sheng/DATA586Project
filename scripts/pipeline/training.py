from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from model.import_data import data_import
from core.model_registry import discover_model_names, load_model_module

RUN_LOG_DIRNAME = "_run_logs"


def default_data_root() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "food-101"


def default_checkpoint_root() -> Path:
    return Path(__file__).resolve().parents[2] / "checkpoints"


def default_checkpoint_dir_for_model(model_name: str) -> Path:
    return default_checkpoint_root() / model_name


def parse_args() -> argparse.Namespace:
    available_models = discover_model_names()
    parser = argparse.ArgumentParser(description="Training entrypoint.")
    parser.add_argument(
        "--model",
        default=available_models[0],
        choices=available_models,
        help="Model trainer to run.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root(),
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory to save checkpoints. Defaults to checkpoints/<model>.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Input image size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override, for example cpu or cuda.",
    )
    parser.add_argument(
        "--freeze-backbone",
        dest="freeze_backbone",
        action="store_true",
        help="Freeze all backbone parameters except the classifier head.",
    )
    parser.add_argument(
        "--no-freeze-backbone",
        dest="freeze_backbone",
        action="store_false",
        help="Train the full backbone.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to checkpoint to resume training from.",
    )
    parser.add_argument(
        "--progress-format",
        default="tqdm",
        choices=["tqdm", "gui"],
        help="Progress output mode.",
    )
    parser.add_argument(
        "--use-validation-split",
        action="store_true",
        help="Split part of the training set into a validation set.",
    )
    parser.add_argument(
        "--validation-proportion",
        type=float,
        default=0.1,
        help="Proportion of the training set reserved for validation when enabled.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed used for train/validation splitting.",
    )
    parser.set_defaults(freeze_backbone=True)
    return parser.parse_args()


def now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_signature(path: Path) -> dict[str, int | bool]:
    if not path.is_file():
        return {"exists": False}
    stat = path.stat()
    return {
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


class TrainingRunLogger:
    def __init__(
        self,
        *,
        checkpoint_dir: Path,
        best_checkpoint_path: Path,
        last_checkpoint_path: Path,
        args: argparse.Namespace,
        model_name: str,
        device: str,
        start_epoch: int,
        num_epochs: int,
        eval_name: str,
        train_batches: int,
        eval_batches: int,
        test_batches: int,
    ) -> None:
        run_logs_dir = checkpoint_dir / RUN_LOG_DIRNAME
        run_logs_dir.mkdir(parents=True, exist_ok=True)
        run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
        self.path = run_logs_dir / f"{run_id}.json"
        self._finalized = False
        self.data: dict[str, object] = {
            "schema_version": 1,
            "run_id": run_id,
            "status": "running",
            "start_time_utc": now_iso_utc(),
            "end_time_utc": None,
            "error_message": None,
            "command": " ".join(sys.argv),
            "args": {
                "model": model_name,
                "data_root": str(args.data_root.expanduser().resolve()),
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
                "resume": str(args.resume.expanduser().resolve()) if args.resume is not None else None,
            },
            "expected": {
                "train_batches_per_epoch": int(train_batches),
                f"{eval_name}_batches_per_epoch": int(eval_batches),
                "final_test_batches": int(test_batches),
            },
            "epochs": [],
            "final_test": None,
            "timing_summary": None,
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

    def append_epoch(
        self,
        *,
        epoch: int,
        train_loss: float,
        train_acc: float,
        train_timing: dict[str, float],
        eval_name: str,
        eval_loss: float,
        eval_acc: float,
        eval_timing: dict[str, float],
    ) -> None:
        epochs = self.data["epochs"]
        assert isinstance(epochs, list)
        epochs.append(
            {
                "epoch": int(epoch),
                "train": {
                    "loss": float(train_loss),
                    "acc": float(train_acc),
                    "timing": train_timing,
                },
                eval_name: {
                    "loss": float(eval_loss),
                    "acc": float(eval_acc),
                    "timing": eval_timing,
                },
            }
        )
        self.write()

    def mark_best_checkpoint(self, *, epoch: int, best_acc: float, path: Path) -> None:
        artifacts = self.data["artifacts"]
        assert isinstance(artifacts, dict)
        best = artifacts["best_checkpoint"]
        assert isinstance(best, dict)
        best["saved_epoch"] = int(epoch)
        best["saved_best_acc"] = float(best_acc)
        best["final_signature"] = file_signature(path)
        self.write()

    def mark_last_checkpoint(self, *, path: Path) -> None:
        artifacts = self.data["artifacts"]
        assert isinstance(artifacts, dict)
        last = artifacts["last_checkpoint"]
        assert isinstance(last, dict)
        last["final_signature"] = file_signature(path)
        self.write()

    def set_final_test(self, *, loss: float, acc: float, timing: dict[str, float]) -> None:
        self.data["final_test"] = {
            "loss": float(loss),
            "acc": float(acc),
            "timing": timing,
        }
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
    ) -> None:
        self.data["status"] = status
        self.data["end_time_utc"] = now_iso_utc()
        self.data["error_message"] = error_message
        self.data["timing_summary"] = {
            "total_wall_time_seconds": float(wall_total_elapsed),
            "total_pure_execution_time_seconds": float(pure_execution_total),
            "initialization_and_overhead_time_seconds": float(init_and_overhead),
            "stage_totals": stage_totals,
        }
        self._finalized = True
        self.write()


def main() -> None:
    wall_total_start = time.perf_counter()
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = (
        args.checkpoint_dir.expanduser().resolve()
        if args.checkpoint_dir is not None
        else default_checkpoint_dir_for_model(args.model)
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / "best.pth"
    last_checkpoint_path = checkpoint_dir / "last.pth"
    start_epoch = 0

    data_root = args.data_root.expanduser().resolve()
    train_loader, val_loader, test_loader, class_to_idx, num_classes = data_import(
        data_root=data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        pin_memory=device.startswith("cuda"),
        use_validation_split=args.use_validation_split,
        validation_proportion=args.validation_proportion,
        split_seed=args.split_seed,
    )
    eval_loader = val_loader if args.use_validation_split else test_loader
    eval_name = "val" if args.use_validation_split else "test"

    model_module = load_model_module(args.model)
    model = model_module.build_model(
        num_classes=num_classes,
        freeze_backbone=args.freeze_backbone,
        device=device,
    )
    optimizer = model_module.build_optimizer(model, lr=args.lr)

    print(f"device: {device}")
    print(f"model: {args.model}")
    print(f"data_root: {data_root}")
    print(f"checkpoint_dir: {checkpoint_dir}")
    print(f"num_classes: {num_classes}")
    print(
        f"epochs: {args.epochs}, batch_size: {args.batch_size}, "
        f"num_workers: {args.num_workers}, image_size: {args.image_size}, lr: {args.lr}"
    )
    print(f"freeze_backbone: {args.freeze_backbone}")
    print(
        f"use_validation_split: {args.use_validation_split}, "
        f"validation_proportion: {args.validation_proportion}, split_seed: {args.split_seed}"
    )

    loss_fn = nn.CrossEntropyLoss()
    best_acc = -1.0

    fcn_history = {"train_loss": [], "train_acc": [], f"{eval_name}_loss": [], f"{eval_name}_acc": []}
    stage_totals: dict[str, dict[str, float]] = {
        "train": {"total_seconds": 0.0, "pure_seconds": 0.0, "batches": 0.0},
        "val": {"total_seconds": 0.0, "pure_seconds": 0.0, "batches": 0.0},
        "test": {"total_seconds": 0.0, "pure_seconds": 0.0, "batches": 0.0},
    }

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint.get("best_acc", -1.0)
        print(f"Resumed from checkpoint {args.resume} at epoch {start_epoch} with best_acc {best_acc:.4f}")

    num_epochs = args.epochs

    run_logger = TrainingRunLogger(
        checkpoint_dir=checkpoint_dir,
        best_checkpoint_path=best_checkpoint_path,
        last_checkpoint_path=last_checkpoint_path,
        args=args,
        model_name=args.model,
        device=device,
        start_epoch=start_epoch,
        num_epochs=num_epochs,
        eval_name=eval_name,
        train_batches=len(train_loader),
        eval_batches=len(eval_loader),
        test_batches=len(test_loader),
    )

    final_test_loss: float | None = None
    final_test_acc: float | None = None
    try:
        for epoch in range(start_epoch, num_epochs):
            train_loss, train_acc, train_timing = train_one_epoch(
                model,
                train_loader,
                loss_fn,
                optimizer,
                device,
                epoch=epoch + 1,
                num_epochs=num_epochs,
                progress_format=args.progress_format,
            )
            eval_loss, eval_acc, eval_timing = evaluate(
                model,
                eval_loader,
                loss_fn,
                device,
                epoch=epoch + 1,
                num_epochs=num_epochs,
                progress_format=args.progress_format,
                stage_name=eval_name,
            )

            stage_totals["train"]["total_seconds"] += train_timing["total_seconds"]
            stage_totals["train"]["pure_seconds"] += train_timing["pure_seconds"]
            stage_totals["train"]["batches"] += train_timing["batches"]
            stage_totals[eval_name]["total_seconds"] += eval_timing["total_seconds"]
            stage_totals[eval_name]["pure_seconds"] += eval_timing["pure_seconds"]
            stage_totals[eval_name]["batches"] += eval_timing["batches"]

            fcn_history["train_loss"].append(train_loss)
            fcn_history["train_acc"].append(train_acc)
            fcn_history[f"{eval_name}_loss"].append(eval_loss)
            fcn_history[f"{eval_name}_acc"].append(eval_acc)

            run_logger.append_epoch(
                epoch=epoch + 1,
                train_loss=train_loss,
                train_acc=train_acc,
                train_timing=train_timing,
                eval_name=eval_name,
                eval_loss=eval_loss,
                eval_acc=eval_acc,
                eval_timing=eval_timing,
            )

            if eval_acc > best_acc:
                best_acc = eval_acc
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_name": args.model,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_acc": best_acc,
                        "num_classes": num_classes,
                        "class_to_idx": class_to_idx,
                        "use_validation_split": args.use_validation_split,
                        "validation_proportion": args.validation_proportion,
                    },
                    best_checkpoint_path,
                )
                run_logger.mark_best_checkpoint(epoch=epoch + 1, best_acc=best_acc, path=best_checkpoint_path)

            train_avg_pure_per_batch = train_timing["pure_seconds"] / max(train_timing["batches"], 1)
            eval_avg_pure_per_batch = eval_timing["pure_seconds"] / max(eval_timing["batches"], 1)
            print(
                f"Epoch {epoch+1}: "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                f"train_total_time={train_timing['total_seconds']:.2f}s, "
                f"train_pure_time={train_timing['pure_seconds']:.2f}s, "
                f"train_avg_pure_per_batch={train_avg_pure_per_batch:.4f}s, "
                f"{eval_name}_loss={eval_loss:.4f}, {eval_name}_acc={eval_acc:.4f}, "
                f"{eval_name}_total_time={eval_timing['total_seconds']:.2f}s, "
                f"{eval_name}_pure_time={eval_timing['pure_seconds']:.2f}s, "
                f"{eval_name}_avg_pure_per_batch={eval_avg_pure_per_batch:.4f}s"
            )

        if args.use_validation_split:
            final_test_loss, final_test_acc, test_timing = evaluate(
                model,
                test_loader,
                loss_fn,
                device,
                progress_format=args.progress_format,
                stage_name="test",
            )
            stage_totals["test"]["total_seconds"] += test_timing["total_seconds"]
            stage_totals["test"]["pure_seconds"] += test_timing["pure_seconds"]
            stage_totals["test"]["batches"] += test_timing["batches"]
            run_logger.set_final_test(loss=final_test_loss, acc=final_test_acc, timing=test_timing)
            test_avg_pure_per_batch = test_timing["pure_seconds"] / max(test_timing["batches"], 1)
            print(
                f"Final test: test_loss={final_test_loss:.4f}, test_acc={final_test_acc:.4f}, "
                f"test_total_time={test_timing['total_seconds']:.2f}s, "
                f"test_pure_time={test_timing['pure_seconds']:.2f}s, "
                f"test_avg_pure_per_batch={test_avg_pure_per_batch:.4f}s"
            )

        pure_execution_total = (
            stage_totals["train"]["pure_seconds"]
            + stage_totals["val"]["pure_seconds"]
            + stage_totals["test"]["pure_seconds"]
        )
        wall_total_elapsed = time.perf_counter() - wall_total_start
        init_and_overhead = max(wall_total_elapsed - pure_execution_total, 0.0)

        print("\nTiming summary:")
        print(f"total_wall_time={wall_total_elapsed:.2f}s")
        print(f"total_pure_execution_time={pure_execution_total:.2f}s")
        print(f"initialization_and_overhead_time={init_and_overhead:.2f}s")
        for stage_name in ("train", "val", "test"):
            stage_total = stage_totals[stage_name]["total_seconds"]
            stage_pure = stage_totals[stage_name]["pure_seconds"]
            stage_batches = stage_totals[stage_name]["batches"]
            if stage_batches <= 0:
                continue
            stage_avg = stage_pure / stage_batches
            print(
                f"{stage_name}: total_time={stage_total:.2f}s, "
                f"pure_time={stage_pure:.2f}s, avg_pure_per_batch={stage_avg:.4f}s"
            )

        if final_test_loss is not None and final_test_acc is not None:
            print(f"final_test_loss={final_test_loss:.4f}, final_test_acc={final_test_acc:.4f}")

        torch.save(
            {
                "epoch": num_epochs,
                "model_name": args.model,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_acc,
                "num_classes": num_classes,
                "class_to_idx": class_to_idx,
                "use_validation_split": args.use_validation_split,
                "validation_proportion": args.validation_proportion,
            },
            last_checkpoint_path,
        )
        run_logger.mark_last_checkpoint(path=last_checkpoint_path)
        run_logger.finalize(
            status="completed",
            stage_totals=stage_totals,
            wall_total_elapsed=wall_total_elapsed,
            pure_execution_total=pure_execution_total,
            init_and_overhead=init_and_overhead,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
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
        )
        raise SystemExit(130)
    except Exception as exc:
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
        )
        raise
        

def emit_gui_progress(
    *,
    stage: str,
    epoch: int | None,
    num_epochs: int | None,
    step: int,
    total_steps: int,
    loss: float | None,
    acc: float | None,
) -> None:
    payload = {
        "type": "progress",
        "stage": stage,
        "epoch": epoch,
        "num_epochs": num_epochs,
        "step": step,
        "total_steps": total_steps,
        "loss": loss,
        "acc": acc,
    }
    print("GUI_PROGRESS " + json.dumps(payload), flush=True)


def train_one_epoch(
    model,
    dataloader,
    loss_fn,
    optimizer,
    device,
    epoch: int | None = None,
    num_epochs: int | None = None,
    progress_format: str = "tqdm",
):
    stage_total_start = time.perf_counter()
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    total_steps = len(dataloader)
    desc = f"Epoch {epoch}/{num_epochs} Train" if epoch is not None and num_epochs is not None else "Train"
    iterator = tqdm(dataloader, desc=desc, total=total_steps, leave=False) if progress_format == "tqdm" else dataloader

    pure_start = time.perf_counter()
    for step_idx, (images, labels) in enumerate(iterator, start=1):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs,labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        if total > 0:
            avg_loss = total_loss / total
            accuracy = correct / total
            if progress_format == "tqdm":
                iterator.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{accuracy:.4f}")
            else:
                emit_gui_progress(
                    stage="train",
                    epoch=epoch,
                    num_epochs=num_epochs,
                    step=step_idx,
                    total_steps=total_steps,
                    loss=avg_loss,
                    acc=accuracy,
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


def evaluate(
    model,
    dataloader,
    loss_fn,
    device,
    epoch: int | None = None,
    num_epochs: int | None = None,
    progress_format: str = "tqdm",
    stage_name: str = "eval",
):
    stage_total_start = time.perf_counter()
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    total_steps = len(dataloader)
    title = stage_name.capitalize()
    desc = f"Epoch {epoch}/{num_epochs} {title}" if epoch is not None and num_epochs is not None else title
    iterator = tqdm(dataloader, desc=desc, total=total_steps, leave=False) if progress_format == "tqdm" else dataloader
    pure_start = time.perf_counter()
    with torch.no_grad():
        for step_idx, (images, labels) in enumerate(iterator, start=1):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs,labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            if total > 0:
                avg_loss = total_loss / total
                accuracy = correct / total
                if progress_format == "tqdm":
                    iterator.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{accuracy:.4f}")
                else:
                    emit_gui_progress(
                        stage=stage_name,
                        epoch=epoch,
                        num_epochs=num_epochs,
                        step=step_idx,
                        total_steps=total_steps,
                        loss=avg_loss,
                        acc=accuracy,
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

if __name__ == "__main__":
    main()
