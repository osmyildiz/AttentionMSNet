"""
Step 10: Swin Multi-Scale Feature Extraction & Validation
==========================================================
Extract multi-scale features from Swin-Tiny to verify the pipeline works.
Uses existing data splits from step_01_prepare_data.py.

This step:
  1. Loads full_data.csv and cv_folds.json (from step_01)
  2. Loads Swin-Tiny pretrained model
  3. Extracts features from stages 2, 3, 4 for a sample batch
  4. Validates shapes and saves a sample .npz for verification
  5. Quick sanity check: frozen Swin + LogReg (like Tarik's approach)
"""

import os, sys, json, time
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import DataLoader
from transformers import SwinModel, AutoImageProcessor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Project imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.helpers import set_seed, load_config, ensure_dirs
from scripts.data_utils import get_transforms, create_loaders
from src.data.dataset import AlzDataset

print("=" * 60)
print("  Step 10: Swin Multi-Scale Feature Extraction")
print("=" * 60)

# ── Config ──
cfg = load_config("configs/config.yaml")
SEED = cfg["project"]["seed"]
set_seed(SEED)

SPLITS_DIR = Path(cfg["data"]["splits_dir"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = cfg["data"]["image_size"]
BATCH_SIZE = 64  # Swin inference only, no grads needed

print(f"Device: {DEVICE}")
print(f"Image size: {IMG_SIZE}")

# ── Load data ──
df = pd.read_csv(SPLITS_DIR / "full_data.csv")
with open(SPLITS_DIR / "cv_folds.json") as f:
    folds = json.load(f)

print(f"Total images: {len(df):,}")
print(f"Folds: {len(folds)}")
print(f"Classes: {df['class_name'].value_counts().to_dict()}")

# ── Load Swin-Tiny ──
print("\nLoading Swin-Tiny pretrained model...")
MODEL_NAME = "microsoft/swin-tiny-patch4-window7-224"
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
swin = SwinModel.from_pretrained(MODEL_NAME).to(DEVICE)
swin.eval()

total_params = sum(p.numel() for p in swin.parameters())
print(f"Swin-Tiny loaded: {total_params:,} parameters")

# ── Feature extraction function ──
@torch.no_grad()
def extract_swin_features(image_paths, labels, batch_size=64):
    """
    Extract multi-scale features from Swin-Tiny.
    Returns:
        pooler: (N, 768) — pooler output (Tarik's approach)
        stage2: (N, 192) — stage 2 global avg pool
        stage3: (N, 384) — stage 3 global avg pool
        stage4: (N, 768) — stage 4 global avg pool
        y: (N,) labels
    """
    all_pooler, all_s2, all_s3, all_s4, all_y = [], [], [], [], []

    n_batches = (len(image_paths) + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(image_paths))
        batch_paths = image_paths[start:end]
        batch_labels = labels[start:end]

        # Load images
        images = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            images.append(img)

        # Process with Swin processor
        inputs = processor(images=images, return_tensors="pt").to(DEVICE)

        # Forward with hidden states
        outputs = swin(**inputs, output_hidden_states=True, return_dict=True)

        # hidden_states[0] = embeddings, [1]=stage1, [2]=stage2, [3]=stage3, [4]=stage4
        hs = outputs.hidden_states

        # Global average pooling per stage
        s2 = hs[2].mean(dim=1).cpu().numpy()  # (B, 192)
        s3 = hs[3].mean(dim=1).cpu().numpy()  # (B, 384)
        s4 = hs[4].mean(dim=1).cpu().numpy()  # (B, 768)
        pooler = outputs.pooler_output.cpu().numpy()  # (B, 768)

        all_pooler.append(pooler)
        all_s2.append(s2)
        all_s3.append(s3)
        all_s4.append(s4)
        all_y.append(np.array(batch_labels))

        if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
            print(f"  Batch {batch_idx+1}/{n_batches}")

    return (
        np.vstack(all_pooler),
        np.vstack(all_s2),
        np.vstack(all_s3),
        np.vstack(all_s4),
        np.concatenate(all_y),
    )


# ═══════════════════════════════════════════════════════════════
# PART A: Validate multi-scale extraction on fold 0
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART A: Multi-scale feature shape validation (Fold 0)")
print("=" * 60)

fold0 = folds[0]
train_subjs = set(fold0["train_subjects"])
test_subjs = set(fold0["test_subjects"])

train_df = df[df["subject_id"].isin(train_subjs)].reset_index(drop=True)
test_df = df[df["subject_id"].isin(test_subjs)].reset_index(drop=True)

print(f"Train: {len(train_df)} images, Test: {len(test_df)} images")

# Extract a small sample first for validation
sample_n = min(32, len(test_df))
sample_df = test_df.head(sample_n)

print(f"\nExtracting features for {sample_n} sample images...")
pooler, s2, s3, s4, y = extract_swin_features(
    sample_df["path"].tolist(),
    sample_df["label"].tolist(),
    batch_size=16,
)

print(f"\nFeature shapes:")
print(f"  pooler_output: {pooler.shape}  (Tarik's approach)")
print(f"  stage2 (GAP):  {s2.shape}  — spatial details")
print(f"  stage3 (GAP):  {s3.shape}  — mid-level")
print(f"  stage4 (GAP):  {s4.shape}  — high-level")
print(f"  labels:        {y.shape}")

# Save sample for verification
ensure_dirs("outputs/swin_features")
np.savez(
    "outputs/swin_features/sample_validation.npz",
    pooler=pooler, stage2=s2, stage3=s3, stage4=s4, labels=y,
)
print(f"Saved: outputs/swin_features/sample_validation.npz")


# ═══════════════════════════════════════════════════════════════
# PART B: Sanity check — Frozen Swin + LogReg (Tarik-style)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART B: Sanity Check — Swin pooler + LogReg (Fold 0)")
print("=" * 60)

print("\nExtracting FULL train features...")
t0 = time.time()
pooler_train, s2_train, s3_train, s4_train, y_train = extract_swin_features(
    train_df["path"].tolist(),
    train_df["label"].tolist(),
    batch_size=BATCH_SIZE,
)
print(f"Train extraction: {time.time()-t0:.1f}s")

print("Extracting FULL test features...")
t0 = time.time()
pooler_test, s2_test, s3_test, s4_test, y_test = extract_swin_features(
    test_df["path"].tolist(),
    test_df["label"].tolist(),
    batch_size=BATCH_SIZE,
)
print(f"Test extraction: {time.time()-t0:.1f}s")

# Save fold 0 features
np.savez_compressed(
    "outputs/swin_features/fold0_features.npz",
    pooler_train=pooler_train, pooler_test=pooler_test,
    s2_train=s2_train, s2_test=s2_test,
    s3_train=s3_train, s3_test=s3_test,
    s4_train=s4_train, s4_test=s4_test,
    y_train=y_train, y_test=y_test,
)
print("Saved: outputs/swin_features/fold0_features.npz")

# Logistic Regression baselines
print("\n--- LogReg Baselines (Fold 0) ---")

configs = {
    "pooler_only (Tarik)": pooler_train,
    "stage2 (192-d)":      s2_train,
    "stage3 (384-d)":      s3_train,
    "stage4 (768-d)":      s4_train,
    "concat_s234 (1344-d)": np.hstack([s2_train, s3_train, s4_train]),
    "concat_all (2112-d)":  np.hstack([pooler_train, s2_train, s3_train, s4_train]),
}

test_configs = {
    "pooler_only (Tarik)": pooler_test,
    "stage2 (192-d)":      s2_test,
    "stage3 (384-d)":      s3_test,
    "stage4 (768-d)":      s4_test,
    "concat_s234 (1344-d)": np.hstack([s2_test, s3_test, s4_test]),
    "concat_all (2112-d)":  np.hstack([pooler_test, s2_test, s3_test, s4_test]),
}

results = []
for name, X_tr in configs.items():
    X_te = test_configs[name]
    lr = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
    lr.fit(X_tr, y_train)
    y_pred = lr.predict(X_te)
    acc = accuracy_score(y_test, y_pred)
    results.append({"feature": name, "accuracy": acc, "dim": X_tr.shape[1]})
    print(f"  {name:30s}  acc={acc:.4f}  dim={X_tr.shape[1]}")

# Save results table
results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
results_df.to_csv("outputs/tables/step10_swin_logreg_baseline.csv", index=False)
print(f"\nSaved: outputs/tables/step10_swin_logreg_baseline.csv")

print("\n" + "=" * 60)
print("Best feature config:", results_df.iloc[0]["feature"])
print(f"Best accuracy: {results_df.iloc[0]['accuracy']:.4f}")
print("=" * 60)

# Show classification report for best
best_name = results_df.iloc[0]["feature"]
X_tr_best = configs[best_name]
X_te_best = test_configs[best_name]
lr = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
lr.fit(X_tr_best, y_train)
y_pred = lr.predict(X_te_best)
class_names = cfg["data"]["class_names"]
print(f"\nClassification Report ({best_name}):")
print(classification_report(y_test, y_pred, target_names=class_names))

print("\nStep 10 DONE.")
print("Next: step_11_swin_attention.py (end-to-end training with CBAM + focal loss)")
