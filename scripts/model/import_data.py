import sys
from pathlib import Path
import json
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

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
                image_size=224):
    if data_root is None:
        data_root = _default_data_root()
    classes,class_to_idx=class_reader(data_root)
    train_samples=read_split(data_root,split_name='train',class_to_idx=class_to_idx)
    test_samples=read_split(data_root,split_name='test',class_to_idx=class_to_idx)
    validate_samples(train_samples)
    validate_samples(test_samples)
    train_transform,test_transform=build_transforms(image_size=image_size)
    train_dataset=Food101Dataset(train_samples,transform=train_transform)
    test_dataset=Food101Dataset(test_samples,transform=test_transform)
    train_loader=DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )
    test_loader=DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )
    return train_loader,test_loader,class_to_idx,len(classes)


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


def build_transforms(image_size=224):
    train_transforms=transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
    ])
    test_transforms=transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
    ])
    return train_transforms,test_transforms
