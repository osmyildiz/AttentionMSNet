"""Subject-level stratified data splitting.
CRITICAL: Prevents data leakage by ensuring all images from 
the same subject stay in the same split.
"""
import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit


def extract_subject_id(filepath):
    """Extract subject ID from filename.
    Heuristic: Use the filename prefix before any slice/augmentation suffix.
    Adjust this function based on actual filename patterns in the dataset.
    """
    fname = Path(filepath).stem
    # Try to extract base subject ID (remove slice numbers, augmentation markers)
    # Common patterns: "OAS1_0001_MR1_mpr-1_100" -> "OAS1_0001"
    # Or simple: "mild_123" -> "mild_123"
    # For OASIS-derived Kaggle: filenames may be like "26 (100).jpg"
    match = re.match(r'^(\d+)', fname)
    if match:
        return match.group(1)
    # Fallback: use full filename as ID (each image = unique subject)
    return fname


def create_subject_level_splits(data_dir, output_dir, train_ratio=0.7, 
                                 val_ratio=0.15, test_ratio=0.15, seed=42):
    """Create train/val/test splits at subject level.
    
    Args:
        data_dir: Path to raw data directory
        output_dir: Path to save split CSVs
        train_ratio, val_ratio, test_ratio: Split ratios
        seed: Random seed
    
    Returns:
        dict with train/val/test DataFrames
    """
    CLASS_MAP = {
        "Non Demented": 0, "NonDemented": 0,
        "Very Mild Demented": 1, "VeryMildDemented": 1,
        "Mild Demented": 2, "MildDemented": 2,
        "Moderate Demented": 3, "ModerateDemented": 3,
    }

    records = []
    for class_name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        label = CLASS_MAP.get(class_name)
        if label is None:
            continue
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                fpath = os.path.join(class_dir, fname)
                subject_id = extract_subject_id(fpath)
                records.append({
                    "path": fpath,
                    "label": label,
                    "class_name": class_name,
                    "subject_id": f"{class_name}_{subject_id}",
                })

    df = pd.DataFrame(records)
    print(f"Total images: {len(df)}")
    print(f"Unique subjects: {df['subject_id'].nunique()}")
    print(f"Class distribution:\n{df['label'].value_counts().sort_index()}")

    # Get unique subjects with their labels
    subjects = df.groupby("subject_id")["label"].first().reset_index()

    # First split: train+val vs test
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(subjects, subjects["label"]))
    
    trainval_subjects = subjects.iloc[trainval_idx]
    test_subjects = subjects.iloc[test_idx]

    # Second split: train vs val
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio_adjusted, random_state=seed)
    train_idx, val_idx = next(sss2.split(trainval_subjects, trainval_subjects["label"]))
    
    train_subjects = trainval_subjects.iloc[train_idx]
    val_subjects = trainval_subjects.iloc[val_idx]

    # Map back to images
    splits = {}
    for name, subj_df in [("train", train_subjects), ("val", val_subjects), ("test", test_subjects)]:
        split_df = df[df["subject_id"].isin(subj_df["subject_id"])]
        splits[name] = split_df
        
        os.makedirs(output_dir, exist_ok=True)
        split_df.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)
        print(f"{name}: {len(split_df)} images, {split_df['subject_id'].nunique()} subjects")

    return splits
