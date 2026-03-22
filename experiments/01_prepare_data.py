"""
Step 01: Data Preparation (Binary + 10-Fold Subject-Level CV)
==============================================================
Scans raw data, extracts real OASIS subject IDs, creates binary labels.
Saves full DataFrame + subject-level fold indices for 10-fold CV.
"""
import os, sys, re, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.helpers import set_seed, load_config, ensure_dirs

print("=" * 60)
print("  Step 01: Data Preparation (Binary + 10-Fold CV)")
print("=" * 60)

cfg = load_config("configs/config.yaml")
set_seed(cfg["project"]["seed"])

DATA_DIR = Path(cfg["data"]["raw_dir"])
SPLITS_DIR = Path(cfg["data"]["splits_dir"])
ensure_dirs(str(SPLITS_DIR))

# ── Binary: Non-Demented (0) vs Demented (1) ──
FOLDER_TO_CLASS = {
    "Non Demented": "Non-Demented",
    "Very mild Dementia": "Demented",
    "Mild Dementia": "Demented",
    "Moderate Dementia": "Demented",
}
CLASS_MAP = {"Non-Demented": 0, "Demented": 1}
CLASS_NAMES = list(CLASS_MAP.keys())
SCAN_FOLDERS = list(FOLDER_TO_CLASS.keys())

# ── Scan images ──
records = []
for folder_name in SCAN_FOLDERS:
    class_dir = DATA_DIR / folder_name
    if not class_dir.exists():
        print(f"WARNING: {class_dir} not found!")
        continue
    mapped_class = FOLDER_TO_CLASS[folder_name]
    images = [f for f in class_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
    for img_path in images:
        stem = img_path.stem
        match = re.search(r"(OAS1_\d{4})", stem)
        if match:
            subj_id = match.group(1)
        else:
            match2 = re.match(r"^(\d+)", stem)
            subj_id = match2.group(1) if match2 else stem
        records.append({
            "path": str(img_path),
            "filename": img_path.name,
            "class_name": mapped_class,
            "label": CLASS_MAP[mapped_class],
            "subject_id": subj_id,
        })

df = pd.DataFrame(records)
print(f"\nTotal images:    {len(df):,}")
print(f"Unique subjects: {df['subject_id'].nunique():,}")
print(f"\nClass distribution (images):")
for cls in CLASS_NAMES:
    n = len(df[df["class_name"] == cls])
    print(f"  {cls:15s}  {n:>7,}  ({n/len(df)*100:.1f}%)")

# ── Subject-level info ──
subjects = df.groupby("subject_id").agg(
    label=("label", "first"),
    n_images=("path", "count")
).reset_index()

print(f"\nClass distribution (subjects):")
for cls in CLASS_NAMES:
    lbl = CLASS_MAP[cls]
    n = len(subjects[subjects["label"] == lbl])
    print(f"  {cls:15s}  {n:>4} subjects")

# ── 10-Fold Stratified CV (subject-level) ──
K = cfg["training"]["n_folds"]
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

print(f"\n{K}-Fold Subject-Level CV:")
fold_info = []
for fold, (train_idx, test_idx) in enumerate(skf.split(subjects, subjects["label"])):
    train_subjects_df = subjects.iloc[train_idx]
    test_subjs = set(subjects.iloc[test_idx]["subject_id"])

    # Split train into train/val (90/10 within train)
    val_idx_inner = train_subjects_df.groupby("label", group_keys=False).apply(
        lambda x: x.sample(frac=0.1, random_state=42+fold)
    ).index
    val_subjs = set(subjects.loc[val_idx_inner, "subject_id"])
    train_subjs = set(train_subjects_df["subject_id"]) - val_subjs

    # Verify no overlap
    assert len(train_subjs & val_subjs) == 0
    assert len(train_subjs & test_subjs) == 0
    assert len(val_subjs & test_subjs) == 0

    train_imgs = len(df[df["subject_id"].isin(train_subjs)])
    val_imgs = len(df[df["subject_id"].isin(val_subjs)])
    test_imgs = len(df[df["subject_id"].isin(test_subjs)])
    test_dem = len(df[(df["subject_id"].isin(test_subjs)) & (df["label"]==1)])

    print(f"  Fold {fold+1:>2}: Train {len(train_subjs):>3} subj ({train_imgs:>5} img) | "
          f"Val {len(val_subjs):>3} subj ({val_imgs:>5} img) | "
          f"Test {len(test_subjs):>3} subj ({test_imgs:>5} img, {test_dem} Dem)")

    fold_info.append({
        "fold": fold,
        "train_subjects": list(train_subjs),
        "val_subjects": list(val_subjs),
        "test_subjects": list(test_subjs),
    })

# Save
df.to_csv(SPLITS_DIR / "full_data.csv", index=False)
with open(SPLITS_DIR / "cv_folds.json", "w") as f:
    json.dump(fold_info, f, indent=2)

print(f"\nSaved: {SPLITS_DIR}/full_data.csv")
print(f"Saved: {SPLITS_DIR}/cv_folds.json")
print("\nStep 01 DONE.")
