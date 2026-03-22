"""
Shared data loading utilities.
Transforms and loader creation - used by all steps.
"""
import numpy as np
import pandas as pd
import random
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset import AlzDataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(img_size=224):
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.ElasticTransform(alpha=50, sigma=5, p=0.2),
        A.GaussNoise(var_limit=(5, 25), p=0.2),
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    return train_transform, val_transform


def load_splits(splits_dir):
    splits_dir = Path(splits_dir)
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")
    test_df = pd.read_csv(splits_dir / "test.csv")
    return train_df, val_df, test_df


def create_loaders(train_df, val_df, test_df, train_transform, val_transform,
                   batch_size=128, num_workers=8, sampling="shuffle"):
    """
    Create DataLoaders with specified sampling strategy.

    sampling options:
      "shuffle"   - plain shuffle, no weighting (clean baseline)
      "weighted"  - WeightedRandomSampler (oversample minority)
    """
    train_dataset = AlzDataset(train_df["path"].tolist(), train_df["label"].tolist(), train_transform)
    val_dataset = AlzDataset(val_df["path"].tolist(), val_df["label"].tolist(), val_transform)
    test_dataset = AlzDataset(test_df["path"].tolist(), test_df["label"].tolist(), val_transform)

    if sampling == "weighted":
        labels = train_df["label"].values
        class_counts = np.bincount(labels)
        sample_weights = (1.0 / class_counts)[labels]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                                  num_workers=num_workers, pin_memory=True, drop_last=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True, drop_last=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


def create_balanced_df(train_df, class_names, target_per_class=2000, seed=42):
    """
    Create balanced training DataFrame.
    Undersample majority, oversample minority (with replacement).
    Returns NEW DataFrame - does not modify original.
    """
    random.seed(seed)
    balanced_rows = []

    for label, cls in enumerate(class_names):
        class_df = train_df[train_df["class_name"] == cls]
        n = len(class_df)

        if n >= target_per_class:
            sampled = class_df.sample(n=target_per_class, random_state=seed)
        else:
            # Oversample with replacement
            sampled = class_df.sample(n=target_per_class, replace=True, random_state=seed)

        balanced_rows.append(sampled)

    return pd.concat(balanced_rows, ignore_index=True)
