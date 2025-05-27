
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset

def collate_fn(batch):
    pixels = torch.stack([example['pixel_values'] for example in batch])
    labels = torch.tensor([example['label'] for example in batch], dtype=torch.long)
    return {
        'pixel_values': pixels,
        'labels': labels,
    }

def get_dataloaders(
    data_dir: str = '/remote-home1/lzjjin/project/fudan-course/DATA130051.01/PJ2/PJ2-1/data/101_ObjectCategories',
    batch_size: int = 32,
    image_size: int = 224,
    split_ratio=(0.8, 0.2),
    seed: int = 42,
    augment: bool = True,
):

    ds = load_dataset("imagefolder", data_dir=data_dir)

    if augment:
        tfms = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        tfms = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def transform_fn(example):
        example['pixel_values'] = [tfms(img) for img in example['image']]
        return example

    ds = ds.with_transform(transform_fn)
    full_ds = ds['train']
    split = full_ds.train_test_split(test_size=1 - split_ratio[0], seed=seed)
    train_ds, val_ds = split['train'], split['test']

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader
