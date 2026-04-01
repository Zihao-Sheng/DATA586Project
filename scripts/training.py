from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from model.import_data import data_import
from model.ResNet18 import build_optimizer, build_resnet18


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir = Path(__file__).resolve().parents[1] / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader, class_to_idx, num_classes = data_import(data_root='../data/food-101')

    model = build_resnet18(
        num_classes=num_classes,
        freeze_backbone=True,
        device=device,
    )
    optimizer = build_optimizer(model)

    print(f"device: {device}")
    print(f"num_classes: {num_classes}")

    loss_fn = nn.CrossEntropyLoss()
    best_acc = -1.0

    fcn_history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    num_epochs = 3
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, epoch=epoch + 1, num_epochs=num_epochs
        )
        test_loss, test_acc = evaluate(
            model, test_loader, loss_fn, device, epoch=epoch + 1, num_epochs=num_epochs
        )

        fcn_history["train_loss"].append(train_loss)
        fcn_history["train_acc"].append(train_acc)
        fcn_history["test_loss"].append(test_loss)
        fcn_history["test_acc"].append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "num_classes": num_classes,
                    "class_to_idx": class_to_idx,
                },
                checkpoint_dir / "resnet18_best.pth",
            )

        print(f"Epoch {epoch+1}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")

    torch.save(
        {
            "epoch": num_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_acc": best_acc,
            "num_classes": num_classes,
            "class_to_idx": class_to_idx,
        },
        checkpoint_dir / "resnet18_last.pth",
    )
        

def train_one_epoch(model, dataloader, loss_fn, optimizer, device, epoch: int | None = None, num_epochs: int | None = None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    desc = f"Epoch {epoch}/{num_epochs} Train" if epoch is not None and num_epochs is not None else "Train"
    pbar = tqdm(dataloader, desc=desc, total=len(dataloader), leave=False)
    for images, labels in pbar:
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
            pbar.set_postfix(loss=f"{total_loss / total:.4f}", acc=f"{correct / total:.4f}")

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, dataloader, loss_fn, device, epoch: int | None = None, num_epochs: int | None = None):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    desc = f"Epoch {epoch}/{num_epochs} Eval" if epoch is not None and num_epochs is not None else "Eval"
    pbar = tqdm(dataloader, desc=desc, total=len(dataloader), leave=False)
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs,labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            if total > 0:
                pbar.set_postfix(loss=f"{total_loss / total:.4f}", acc=f"{correct / total:.4f}")

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

if __name__ == "__main__":
    main()
