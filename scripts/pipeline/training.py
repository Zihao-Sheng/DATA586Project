from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from model.import_data import data_import
from core.model_registry import discover_model_names, load_model_module


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


def main() -> None:
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

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint.get("best_acc", -1.0)
        print(f"Resumed from checkpoint {args.resume} at epoch {start_epoch} with best_acc {best_acc:.4f}")
    num_epochs = args.epochs
    
    for epoch in range(start_epoch, num_epochs):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            epoch=epoch + 1,
            num_epochs=num_epochs,
            progress_format=args.progress_format,
        )
        eval_loss, eval_acc = evaluate(
            model,
            eval_loader,
            loss_fn,
            device,
            epoch=epoch + 1,
            num_epochs=num_epochs,
            progress_format=args.progress_format,
            stage_name=eval_name,
        )

        fcn_history["train_loss"].append(train_loss)
        fcn_history["train_acc"].append(train_acc)
        fcn_history[f"{eval_name}_loss"].append(eval_loss)
        fcn_history[f"{eval_name}_acc"].append(eval_acc)

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

        print(f"Epoch {epoch+1}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"{eval_name}_loss={eval_loss:.4f}, {eval_name}_acc={eval_acc:.4f}")

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
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    total_steps = len(dataloader)
    desc = f"Epoch {epoch}/{num_epochs} Train" if epoch is not None and num_epochs is not None else "Train"
    iterator = tqdm(dataloader, desc=desc, total=total_steps, leave=False) if progress_format == "tqdm" else dataloader

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

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


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
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    total_steps = len(dataloader)
    title = stage_name.capitalize()
    desc = f"Epoch {epoch}/{num_epochs} {title}" if epoch is not None and num_epochs is not None else title
    iterator = tqdm(dataloader, desc=desc, total=total_steps, leave=False) if progress_format == "tqdm" else dataloader
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

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

if __name__ == "__main__":
    main()
