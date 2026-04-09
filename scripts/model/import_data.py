import sys
from pathlib import Path
import json
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torchvision.transforms import InterpolationMode

def _default_data_root():
    return Path(__file__).resolve().parents[2] / 'data' / 'food-101'

class Food101Dataset(Dataset):
    def __init__(self, samples, transform=None):
        super().__init__()
        self.samples=samples
        self.transform=transform
    def __len__(self):
        return len(self.samples)
    def __getitem__(self,idx):
        path,label=self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image=self.transform(image)
        
        return image,label



def data_import(data_root=None,
                batch_size=32,
                num_workers=4,
                pin_memory=True,
                image_size=224,
                train_transforms_preset='baseline',
                mild_blur_enabled=False,
                mild_blur_prob=0.10,
                custom_augmentation=None,
                use_validation_split=False,
                validation_proportion=0.1,
                split_seed=42):
    if data_root is None:
        data_root = _default_data_root()
    classes,class_to_idx=class_reader(data_root)
    train_samples=read_split(data_root,split_name='train',class_to_idx=class_to_idx)
    test_samples=read_split(data_root,split_name='test',class_to_idx=class_to_idx)
    val_samples = []
    if use_validation_split:
        train_samples, val_samples = split_train_validation(
            train_samples,
            validation_proportion=validation_proportion,
            split_seed=split_seed,
        )
    validate_samples(train_samples)
    if use_validation_split:
        validate_samples(val_samples)
    validate_samples(test_samples)
    train_transform,test_transform=build_transforms(
        image_size=image_size,
        train_preset=train_transforms_preset,
        mild_blur_enabled=mild_blur_enabled,
        mild_blur_prob=mild_blur_prob,
        custom_augmentation=custom_augmentation,
    )
    train_dataset=Food101Dataset(train_samples,transform=train_transform)
    val_dataset=Food101Dataset(val_samples,transform=test_transform) if use_validation_split else None
    test_dataset=Food101Dataset(test_samples,transform=test_transform)
    train_loader=DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )
    val_loader=DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    ) if use_validation_split else None
    test_loader=DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )
    return train_loader,val_loader,test_loader,class_to_idx,len(classes)


def class_reader(data_root=None):
    if data_root is None:
        data_root = _default_data_root()
    data_root = Path(data_root)
    classes = []
    class_path = data_root / 'meta' / 'classes.txt'
    with class_path.open('r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
        classes.extend(lines)
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    return classes, class_to_idx


def read_split(data_root=None, split_name='train', class_to_idx=None):
    if data_root is None:
        data_root = _default_data_root()
    split_list=[]
    if split_name=='train':
        file='train.json'
    else:
        file='test.json'
    data_root=Path(data_root)
    file_path=data_root/'meta'/file
    with file_path.open('r',encoding='utf-8') as f:
        data=json.load(f)
        for name, food in data.items():
            idx=class_to_idx[name]
            for image in food:
                image=image+'.jpg'
                image_path=data_root/'images'/image
                split_list.append([image_path,idx])
    return split_list


def validate_samples(samples):
    total=len(samples)
    missing=0
    for item in samples:
        if item[0].exists()==False:
            print (item[0])
            missing+=1
    print('Validated Rate:',100*(total-missing)/total,'%')
    return missing


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
    for label, items in grouped.items():
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


class RandomDownsampleUpsample:
    def __init__(
        self,
        *,
        min_scale: float = 0.25,
        max_scale: float = 0.65,
        probability: float = 0.5,
        interpolation_down: InterpolationMode = InterpolationMode.BILINEAR,
        interpolation_up: InterpolationMode = InterpolationMode.BICUBIC,
    ):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.probability = probability
        self.interpolation_down = interpolation_down
        self.interpolation_up = interpolation_up

    def __call__(self, image):
        import random

        if random.random() > self.probability:
            return image
        width, height = image.size
        if width <= 1 or height <= 1:
            return image
        scale = random.uniform(self.min_scale, self.max_scale)
        down_width = max(1, int(round(width * scale)))
        down_height = max(1, int(round(height * scale)))
        image = transforms.functional.resize(image, (down_height, down_width), interpolation=self.interpolation_down)
        image = transforms.functional.resize(image, (height, width), interpolation=self.interpolation_up)
        return image


def build_eval_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])


def _enabled_config(config, key):
    section = config.get(key) if isinstance(config, dict) else None
    return section if isinstance(section, dict) else {}


def build_train_transform(image_size=224, preset='baseline', mild_blur_enabled=False, mild_blur_prob=0.10, custom_augmentation=None):
    normalized_preset = str(preset).strip().lower()
    resize = transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR)
    baseline = transforms.Compose([
        resize,
        transforms.ToTensor(),
    ])

    if normalized_preset == 'custom':
        config = custom_augmentation if isinstance(custom_augmentation, dict) else {}
        transform_steps = [
            transforms.RandomResizedCrop(image_size, scale=(0.70, 1.0), ratio=(0.85, 1.15), interpolation=InterpolationMode.BILINEAR),
        ]
        if bool(_enabled_config(config, "horizontal_flip").get("enabled")):
            transform_steps.append(transforms.RandomHorizontalFlip(p=0.5))
        downsample = _enabled_config(config, "downsample")
        if downsample.get("enabled"):
            min_scale = float(downsample.get("min_scale", 0.18))
            max_scale = float(downsample.get("max_scale", 0.55))
            if not (0.0 < min_scale <= max_scale <= 1.0):
                raise ValueError("custom downsample min/max scale must satisfy 0 < min <= max <= 1.")
            transform_steps.append(
                transforms.RandomApply(
                    [
                        RandomDownsampleUpsample(
                            min_scale=min_scale,
                            max_scale=max_scale,
                            probability=1.0,
                            interpolation_down=InterpolationMode.BOX,
                            interpolation_up=InterpolationMode.BICUBIC,
                        )
                    ],
                    p=float(downsample.get("probability", 0.65)),
                )
            )
        if bool(_enabled_config(config, "color_jitter").get("enabled")):
            transform_steps.append(transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.08, hue=0.02))
        blur = _enabled_config(config, "mild_blur")
        if blur.get("enabled"):
            transform_steps.append(
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))], p=float(blur.get("probability", 0.10)))
            )
        transform_steps.append(transforms.ToTensor())
        erasing = _enabled_config(config, "random_erasing")
        if erasing.get("enabled"):
            transform_steps.append(
                transforms.RandomErasing(
                    p=float(erasing.get("probability", 0.08)),
                    scale=(0.02, 0.08),
                    ratio=(0.5, 2.0),
                    value='random',
                )
            )
    elif normalized_preset == 'baseline':
        transform_steps = list(baseline.transforms)
    elif normalized_preset == 'standard':
        transform_steps = [
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.9, 1.1), interpolation=InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.10, hue=0.02),
            transforms.ToTensor(),
        ]
    elif normalized_preset == 'robust':
        transform_steps = [
            transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0), ratio=(0.8, 1.2), interpolation=InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.15, hue=0.03),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.6))], p=0.18),
            transforms.RandomAutocontrast(p=0.08),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.10, scale=(0.02, 0.10), ratio=(0.3, 3.3), value='random'),
        ]
    elif normalized_preset == 'downsample_focus':
        transform_steps = [
            transforms.RandomResizedCrop(image_size, scale=(0.70, 1.0), ratio=(0.85, 1.15), interpolation=InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [
                    RandomDownsampleUpsample(
                        min_scale=0.18,
                        max_scale=0.55,
                        probability=1.0,
                        interpolation_down=InterpolationMode.BOX,
                        interpolation_up=InterpolationMode.BICUBIC,
                    )
                ],
                p=0.65,
            ),
            transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.08, hue=0.02),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.08, scale=(0.02, 0.08), ratio=(0.5, 2.0), value='random'),
        ]
    else:
        raise ValueError(f"Unsupported train transform preset: {preset}")
    if normalized_preset != 'custom' and mild_blur_enabled:
        if not 0.0 < float(mild_blur_prob) <= 1.0:
            raise ValueError("mild_blur_prob must be between 0 and 1 when mild blur is enabled.")
        insert_at = max(len(transform_steps) - 2, 0)
        transform_steps.insert(
            insert_at,
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))], p=float(mild_blur_prob)),
        )
    return transforms.Compose(transform_steps)


def build_transforms(image_size=224, train_preset='baseline', mild_blur_enabled=False, mild_blur_prob=0.10, custom_augmentation=None):
    train_transforms = build_train_transform(
        image_size=image_size,
        preset=train_preset,
        mild_blur_enabled=mild_blur_enabled,
        mild_blur_prob=mild_blur_prob,
        custom_augmentation=custom_augmentation,
    )
    test_transforms = build_eval_transform(image_size=image_size)
    return train_transforms, test_transforms
