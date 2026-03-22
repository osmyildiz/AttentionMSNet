"""PyTorch Dataset for Alzheimer MRI classification."""
import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class AlzDataset(Dataset):
    """Alzheimer's Disease MRI Dataset.
    
    Loads images from directory structure:
        data/raw/
            Non Demented/
            Very Mild Demented/
            Mild Demented/
            Moderate Demented/
    """
    CLASS_MAP = {
        "Non Demented": 0, "NonDemented": 0,
        "Very Mild Demented": 1, "VeryMildDemented": 1,
        "Mild Demented": 2, "MildDemented": 2,
        "Moderate Demented": 3, "ModerateDemented": 3,
    }
    CLASS_NAMES = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = np.array(img)
        label = self.labels[idx]

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]

        return img, label

    @classmethod
    def from_directory(cls, root_dir, transform=None):
        """Create dataset from directory structure."""
        image_paths = []
        labels = []

        for class_name in sorted(os.listdir(root_dir)):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            label = cls.CLASS_MAP.get(class_name)
            if label is None:
                continue
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(class_dir, fname))
                    labels.append(label)

        return cls(image_paths, labels, transform)

    @classmethod
    def from_csv(cls, csv_path, transform=None):
        """Create dataset from CSV with columns: path, label."""
        df = pd.read_csv(csv_path)
        return cls(df["path"].tolist(), df["label"].tolist(), transform)
