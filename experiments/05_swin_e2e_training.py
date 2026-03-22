"""
Step 11: SwinAttentionNet — 10-Fold Subject-Level CV Training
==============================================================
End-to-end training: Swin-Tiny + CBAM + Multi-Scale Fusion + Focal Loss

Uses:
  - data splits from step_01 (cv_folds.json)
  - SwinAttentionNet model from src/models/swin_attention.py
  - Weighted random sampling for class imbalance
  - Cosine annealing LR scheduler
  - Early stopping on validation loss
"""

import os, sys, json, time, copy
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoImageProcessor
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)

# Project imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.helpers import set_seed, load_config, ensure_dirs
from src.models.swin_attention import SwinAttentionNet, FocalLoss

print("=" * 60)
print("  Step 11: SwinAttentionNet 10-Fold CV Training")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════

cfg = load_config("configs/config.yaml")
SEED = cfg["project"]["seed"]
set_seed(SEED)

SPLITS_DIR = Path(cfg["data"]["splits_dir"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = cfg["data"]["image_size"]
NUM_CLASSES = cfg["data"]["num_classes"]
CLASS_NAMES = cfg["data"]["class_names"]

# Training hyperparams
EPOCHS = 50
BATCH_SIZE = 32  # Swin is bigger than EfficientNet, smaller batch
LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 10
FOCAL_GAMMA = 2.0
FREEZE_STAGES = (0, 1)  # Freeze Swin stages 0 and 1
DROPOUT = 0.4
FUSION_DIM = 256
N_FOLDS = cfg["training"]["n_folds"]

print(f"Device: {DEVICE}")
print(f"Classes: {NUM_CLASSES} — {CLASS_NAMES}")
print(f"Epochs: {EPOCHS}, Batch: {BATCH_SIZE}, LR: {LR}")
print(f"Focal gamma: {FOCAL_GAMMA}")
print(f"Frozen Swin stages: {FREEZE_STAGES}")


# ═══════════════════════════════════════════════════════════════
# Dataset (uses Swin processor instead of albumentations)
# ═══════════════════════════════════════════════════════════════

class SwinAlzDataset(Dataset):
    """Dataset that uses HuggingFace processor for Swin input."""
    def __init__(self, paths, labels, processor, augment=False):
        self.paths = paths
        self.labels = labels
        self.processor = processor
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")

        # Simple augmentation for training (Swin processor handles resize+normalize)
        if self.augment:
            import torchvision.transforms as T
            aug = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(15),
                T.ColorJitter(brightness=0.2, contrast=0.2),
            ])
            img = aug(img)

        inputs = self.processor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)  # (3, 224, 224)
        label = self.labels[idx]
        return pixel_values, label


# ═══════════════════════════════════════════════════════════════
# Training utilities
# ═══════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        logits, _ = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits, _ = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


# ═══════════════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════════════

df = pd.read_csv(SPLITS_DIR / "full_data.csv")
with open(SPLITS_DIR / "cv_folds.json") as f:
    folds = json.load(f)

print(f"\nTotal images: {len(df):,}")
print(f"Class distribution: {df['class_name'].value_counts().to_dict()}")

# Load Swin processor (for dataset)
processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

# Compute class weights for focal loss (inverse frequency)
class_counts = df["label"].value_counts().sort_index().values.astype(float)
class_weights = (1.0 / class_counts)
class_weights = class_weights / class_weights.sum() * len(class_weights)
print(f"Focal loss class weights: {class_weights}")


# ═══════════════════════════════════════════════════════════════
# K-Fold Training
# ═══════════════════════════════════════════════════════════════

ensure_dirs("outputs/models", "outputs/tables", "outputs/figures")

all_fold_results = []

for fold_idx, fold_data in enumerate(folds):
    fold_num = fold_idx + 1
    print(f"\n{'='*60}")
    print(f"  FOLD {fold_num}/{N_FOLDS}")
    print(f"{'='*60}")

    # Split subjects
    train_subjs = set(fold_data["train_subjects"])
    val_subjs = set(fold_data["val_subjects"])
    test_subjs = set(fold_data["test_subjects"])

    train_df = df[df["subject_id"].isin(train_subjs)].reset_index(drop=True)
    val_df = df[df["subject_id"].isin(val_subjs)].reset_index(drop=True)
    test_df = df[df["subject_id"].isin(test_subjs)].reset_index(drop=True)

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Datasets
    train_dataset = SwinAlzDataset(
        train_df["path"].tolist(), train_df["label"].tolist(),
        processor, augment=True,
    )
    val_dataset = SwinAlzDataset(
        val_df["path"].tolist(), val_df["label"].tolist(),
        processor, augment=False,
    )
    test_dataset = SwinAlzDataset(
        test_df["path"].tolist(), test_df["label"].tolist(),
        processor, augment=False,
    )

    # Weighted sampler for training
    train_labels = train_df["label"].values
    sample_class_counts = np.bincount(train_labels, minlength=NUM_CLASSES).astype(float)
    sample_weights = (1.0 / sample_class_counts)[train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # Model
    model = SwinAttentionNet(
        num_classes=NUM_CLASSES,
        freeze_stages=FREEZE_STAGES,
        fusion_dim=FUSION_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)

    if fold_idx == 0:
        total_p = sum(p.numel() for p in model.parameters())
        train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model: {total_p:,} total params, {train_p:,} trainable")

    # Loss, optimizer, scheduler
    criterion = FocalLoss(alpha=class_weights.tolist(), gamma=FOCAL_GAMMA)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # Training loop
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        elapsed = time.time() - t0

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} | "
                  f"Train loss={train_loss:.4f} acc={train_acc:.4f} | "
                  f"Val loss={val_loss:.4f} acc={val_acc:.4f} | "
                  f"{elapsed:.1f}s")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Load best model and evaluate on test
    model.load_state_dict(best_model_state)
    test_loss, test_acc, y_pred, y_true = evaluate(model, test_loader, criterion, DEVICE)

    # Metrics
    f1 = f1_score(y_true, y_pred, average="weighted")
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n  Fold {fold_num} TEST: acc={test_acc:.4f} f1={f1:.4f} prec={precision:.4f} rec={recall:.4f}")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))

    # Fusion weights
    fw = model.get_fusion_weights()
    print(f"  Fusion weights: {fw}")

    # Save fold result
    fold_result = {
        "fold": fold_num,
        "test_acc": test_acc,
        "test_f1": f1,
        "test_precision": precision,
        "test_recall": recall,
        "test_loss": test_loss,
        "best_val_loss": best_val_loss,
        **{f"fw_{k}": v for k, v in fw.items()},
    }
    all_fold_results.append(fold_result)

    # Save best model
    torch.save(best_model_state, f"outputs/models/swin_attention_fold{fold_num}.pt")

    # Clean up GPU memory
    del model, optimizer, scheduler, criterion
    torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  FINAL RESULTS — SwinAttentionNet 10-Fold CV")
print("=" * 60)

results_df = pd.DataFrame(all_fold_results)
print(results_df.to_string(index=False))

print(f"\n--- Average ± Std ---")
for metric in ["test_acc", "test_f1", "test_precision", "test_recall"]:
    mean = results_df[metric].mean()
    std = results_df[metric].std()
    print(f"  {metric:20s}: {mean:.4f} ± {std:.4f}")

# Fusion weight averages
for col in results_df.columns:
    if col.startswith("fw_"):
        print(f"  {col:20s}: {results_df[col].mean():.4f}")

results_df.to_csv("outputs/tables/step11_swin_attention_cv.csv", index=False)
print(f"\nSaved: outputs/tables/step11_swin_attention_cv.csv")

print("\nStep 11 DONE.")
print("Next: step_12_swin_gradcam.py (Grad-CAM++ explainability)")
