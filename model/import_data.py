import json
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def _default_data_root():
    return Path(__file__).resolve().parents[1] / "data" / "food-101"


class Food101Dataset(Dataset):
    def __init__(self, samples, transform=None):
        super().__init__()
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def data_import(
    data_root=None,
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    image_size=224,
    use_validation_split=False,
    validation_proportion=0.1,
    split_seed=42,
):
    if data_root is None:
        data_root = _default_data_root()
    classes, class_to_idx = class_reader(data_root)
    train_samples = read_split(data_root, split_name="train", class_to_idx=class_to_idx)
    test_samples = read_split(data_root, split_name="test", class_to_idx=class_to_idx)
    val_samples = []
    if use_validation_split:
        train_samples, val_samples = split_train_validation(
            train_samples,
            validation_proportion=validation_proportion,
            split_seed=split_seed,
        )
    train_transform, test_transform = build_transforms(image_size=image_size)
    train_dataset = Food101Dataset(train_samples, transform=train_transform)
    val_dataset = Food101Dataset(val_samples, transform=test_transform) if use_validation_split else None
    test_dataset = Food101Dataset(test_samples, transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    val_loader = (
        DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)
        if use_validation_split
        else None
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)
    return train_loader, val_loader, test_loader, class_to_idx, len(classes)


def class_reader(data_root=None):
    if data_root is None:
        data_root = _default_data_root()
    data_root = Path(data_root)
    class_path = data_root / "meta" / "classes.txt"
    with class_path.open("r", encoding="utf-8") as handle:
        classes = [line.strip() for line in handle if line.strip()]
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    return classes, class_to_idx


def read_split(data_root=None, split_name="train", class_to_idx=None):
    if data_root is None:
        data_root = _default_data_root()
    split_list = []
    file_name = "train.json" if split_name == "train" else "test.json"
    data_root = Path(data_root)
    file_path = data_root / "meta" / file_name
    with file_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    for name, foods in data.items():
        idx = class_to_idx[name]
        for image in foods:
            split_list.append([data_root / "images" / f"{image}.jpg", idx])
    return split_list


def split_train_validation(samples, validation_proportion=0.1, split_seed=42):
    if not 0.0 < validation_proportion < 1.0:
        raise ValueError("validation_proportion must be between 0 and 1.")
    import random

    rng = random.Random(split_seed)
    grouped = {}
    for sample in samples:
        grouped.setdefault(sample[1], []).append(sample)

    train_split = []
    val_split = []
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


def build_transforms(image_size=224):
    transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    return transform, transform
