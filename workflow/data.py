from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def default_data_root() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "food-101"


class Food101Dataset(Dataset):
    def __init__(self, samples: list[list[Path | int]], transform=None) -> None:
        super().__init__()
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def class_reader(data_root: Path | None = None) -> tuple[list[str], dict[str, int]]:
    root = default_data_root() if data_root is None else Path(data_root)
    class_path = root / "meta" / "classes.txt"
    with class_path.open("r", encoding="utf-8") as handle:
        classes = [line.strip() for line in handle if line.strip()]
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    return classes, class_to_idx


def read_split(data_root: Path | None = None, split_name: str = "train", class_to_idx: dict[str, int] | None = None) -> list[list[Path | int]]:
    root = default_data_root() if data_root is None else Path(data_root)
    if class_to_idx is None:
        _, class_to_idx = class_reader(root)
    file_name = "train.json" if split_name == "train" else "test.json"
    file_path = root / "meta" / file_name
    split_list: list[list[Path | int]] = []
    with file_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    for class_name, foods in data.items():
        label = class_to_idx[class_name]
        for image in foods:
            split_list.append([root / "images" / f"{image}.jpg", label])
    return split_list


def split_train_validation(
    samples: list[list[Path | int]],
    validation_proportion: float = 0.1,
    split_seed: int = 42,
) -> tuple[list[list[Path | int]], list[list[Path | int]]]:
    if not 0.0 < validation_proportion < 1.0:
        raise ValueError("validation_proportion must be between 0 and 1.")
    import random
    rng = random.Random(split_seed)
    grouped: dict[int, list[list[Path | int]]] = {}
    for sample in samples:
        grouped.setdefault(int(sample[1]), []).append(sample)
    train_split: list[list[Path | int]] = []
    val_split: list[list[Path | int]] = []
    for items in grouped.values():
        shuffled = list(items)
        rng.shuffle(shuffled)
        if len(shuffled) <= 1:
            train_split.extend(shuffled)
            continue
        val_count = max(1, int(len(shuffled) * validation_proportion))
        val_count = min(val_count, len(shuffled) - 1)
        val_split.extend(shuffled[:val_count])
        train_split.extend(shuffled[val_count:])
    rng.shuffle(train_split)
    rng.shuffle(val_split)
    return train_split, val_split


def build_transforms(image_size: int = 224):
    transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    return transform, transform


def data_import(
    data_root: Path | None = None,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    image_size: int = 224,
    use_validation_split: bool = False,
    validation_proportion: float = 0.1,
    split_seed: int = 42,
):
    root = default_data_root() if data_root is None else Path(data_root)
    classes, class_to_idx = class_reader(root)
    train_samples = read_split(root, "train", class_to_idx)
    test_samples = read_split(root, "test", class_to_idx)
    val_samples: list[list[Path | int]] = []
    if use_validation_split:
        train_samples, val_samples = split_train_validation(train_samples, validation_proportion=validation_proportion, split_seed=split_seed)
    train_transform, test_transform = build_transforms(image_size=image_size)
    train_dataset = Food101Dataset(train_samples, transform=train_transform)
    val_dataset = Food101Dataset(val_samples, transform=test_transform) if use_validation_split else None
    test_dataset = Food101Dataset(test_samples, transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False) if use_validation_split else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)
    return train_loader, val_loader, test_loader, class_to_idx, len(classes)
