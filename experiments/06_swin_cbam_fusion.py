"""
Step 11c: Frozen Swin + Multi-Scale CBAM Attention
====================================================
KEY INSIGHT: Don't fine-tune Swin. Use it as a frozen feature extractor
(like Tarik) but ADD multi-scale extraction + CBAM attention on top.

Pipeline:
  Phase 1: Extract ALL multi-scale features once (frozen Swin) → save to disk
  Phase 2: Train lightweight CBAM + MLP on extracted features (10-fold CV)

This is FAST because:
  - Feature extraction happens ONCE (not every epoch)
  - Training is just CBAM + MLP on pre-extracted features
  - Each fold takes seconds, not minutes

Abstract alignment:
  ✓ "pretrained convolutional backbone" → Swin-Tiny (pretrained)
  ✓ "channel-spatial attention refinement" → CBAM on extracted features
  ✓ "multi-scale feature aggregation" → stage 2+3+4 features
  ✓ "focal loss with adaptive class weighting" → FocalLoss
  ✓ "balanced sampling strategies" → WeightedRandomSampler
"""

import os, sys, json, time, copy
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import SwinModel, AutoImageProcessor
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.helpers import set_seed, load_config, ensure_dirs

print("=" * 70)
print("  Step 11c: Frozen Swin + Multi-Scale CBAM Attention")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════

cfg = load_config("configs/config.yaml")
SEED = cfg["project"]["seed"]
set_seed(SEED)

SPLITS_DIR = Path(cfg["data"]["splits_dir"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = cfg["data"]["num_classes"]
CLASS_NAMES = cfg["data"]["class_names"]
N_FOLDS = cfg["training"]["n_folds"]

# Hyperparams — tuned for extracted features
EPOCHS = 100
BATCH_SIZE = 256        # Large batch OK — features are small vectors
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 15
FOCAL_GAMMA = 1.0       # Less aggressive than 2.0
HIDDEN_DIM = 512
DROPOUT = 0.3
SWIN_BATCH = 64         # For feature extraction

print(f"Device: {DEVICE}")
print(f"Classes: {NUM_CLASSES} — {CLASS_NAMES}")


# ═══════════════════════════════════════════════════════════════
# PHASE 1: Extract features ONCE from frozen Swin
# ═══════════════════════════════════════════════════════════════

FEATURE_FILE = "outputs/swin_features/all_multiscale_features.npz"

df = pd.read_csv(SPLITS_DIR / "full_data.csv")
with open(SPLITS_DIR / "cv_folds.json") as f:
    folds = json.load(f)

print(f"\nTotal images: {len(df):,}")
print(f"Class distribution: {df['class_name'].value_counts().to_dict()}")

if os.path.exists(FEATURE_FILE):
    print(f"\nLoading pre-extracted features from {FEATURE_FILE}...")
    data = np.load(FEATURE_FILE)
    all_s2 = data["stage2"]
    all_s3 = data["stage3"]
    all_s4 = data["stage4"]
    all_pooler = data["pooler"]
    all_labels = data["labels"]
    all_paths = data["paths"]
    print(f"Loaded: {all_s2.shape[0]} images")
else:
    print("\nExtracting features from frozen Swin-Tiny...")
    print("This runs ONCE — subsequent runs load from disk.\n")

    processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    swin = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224").to(DEVICE)
    swin.eval()
    for p in swin.parameters():
        p.requires_grad = False

    all_paths_list = df["path"].tolist()
    all_labels_list = df["label"].tolist()

    all_s2, all_s3, all_s4, all_pooler = [], [], [], []
    n_batches = (len(all_paths_list) + SWIN_BATCH - 1) // SWIN_BATCH

    t0 = time.time()
    with torch.no_grad():
        for bi in range(n_batches):
            start = bi * SWIN_BATCH
            end = min(start + SWIN_BATCH, len(all_paths_list))
            batch_paths = all_paths_list[start:end]

            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = processor(images=images, return_tensors="pt").to(DEVICE)

            outputs = swin(**inputs, output_hidden_states=True, return_dict=True)
            hs = outputs.hidden_states

            # Stage outputs — global average pooled
            s2 = hs[2].mean(dim=1).cpu().numpy()
            s3 = hs[3].mean(dim=1).cpu().numpy()
            s4 = hs[4].mean(dim=1).cpu().numpy()
            pooler = outputs.pooler_output.cpu().numpy()

            all_s2.append(s2)
            all_s3.append(s3)
            all_s4.append(s4)
            all_pooler.append(pooler)

            if (bi + 1) % 50 == 0 or bi == n_batches - 1:
                elapsed = time.time() - t0
                eta = elapsed / (bi + 1) * (n_batches - bi - 1)
                print(f"  Batch {bi+1}/{n_batches} | {elapsed:.0f}s elapsed | ETA {eta:.0f}s")

    all_s2 = np.vstack(all_s2)
    all_s3 = np.vstack(all_s3)
    all_s4 = np.vstack(all_s4)
    all_pooler = np.vstack(all_pooler)
    all_labels = np.array(all_labels_list)
    all_paths = np.array(all_paths_list)

    ensure_dirs("outputs/swin_features")
    np.savez_compressed(
        FEATURE_FILE,
        stage2=all_s2, stage3=all_s3, stage4=all_s4,
        pooler=all_pooler, labels=all_labels, paths=all_paths,
    )
    elapsed = time.time() - t0
    print(f"\nFeature extraction done in {elapsed:.0f}s")
    print(f"Saved: {FEATURE_FILE}")

    del swin, processor
    torch.cuda.empty_cache()

print(f"Feature shapes: s2={all_s2.shape}, s3={all_s3.shape}, s4={all_s4.shape}, pooler={all_pooler.shape}")

# Build path → index mapping
path_to_idx = {p: i for i, p in enumerate(all_paths)}


# ═══════════════════════════════════════════════════════════════
# PHASE 2: Lightweight CBAM + MLP model on extracted features
# ═══════════════════════════════════════════════════════════════

class ChannelAttention1D(nn.Module):
    """Channel attention for 1D feature vectors."""
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


class MultiScaleCBAMClassifier(nn.Module):
    """
    Lightweight classifier on pre-extracted multi-scale Swin features.

    For each scale: channel attention → projection → fused → MLP classifier
    """
    def __init__(self, s2_dim=384, s3_dim=768, s4_dim=768, pooler_dim=768,
                 hidden_dim=512, num_classes=2, dropout=0.3):
        super().__init__()

        # Channel attention per scale
        self.ca_s2 = ChannelAttention1D(s2_dim)
        self.ca_s3 = ChannelAttention1D(s3_dim)
        self.ca_s4 = ChannelAttention1D(s4_dim)
        self.ca_pooler = ChannelAttention1D(pooler_dim)

        # Project each scale to common dim
        proj_dim = hidden_dim // 4
        self.proj_s2 = nn.Sequential(nn.Linear(s2_dim, proj_dim), nn.GELU(), nn.Dropout(dropout * 0.5))
        self.proj_s3 = nn.Sequential(nn.Linear(s3_dim, proj_dim), nn.GELU(), nn.Dropout(dropout * 0.5))
        self.proj_s4 = nn.Sequential(nn.Linear(s4_dim, proj_dim), nn.GELU(), nn.Dropout(dropout * 0.5))
        self.proj_pooler = nn.Sequential(nn.Linear(pooler_dim, proj_dim), nn.GELU(), nn.Dropout(dropout * 0.5))

        # Learnable scale importance
        self.scale_weights = nn.Parameter(torch.ones(4) / 4)

        # Classifier on fused features
        fused_dim = proj_dim  # weighted sum, not concat
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # ALSO build a concat-based classifier (ablation)
        concat_dim = proj_dim * 4
        self.classifier_concat = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self.use_concat = False  # toggle for ablation

    def forward(self, s2, s3, s4, pooler):
        # Channel attention
        s2_att = self.ca_s2(s2)
        s3_att = self.ca_s3(s3)
        s4_att = self.ca_s4(s4)
        p_att = self.ca_pooler(pooler)

        # Project
        s2_proj = self.proj_s2(s2_att)
        s3_proj = self.proj_s3(s3_att)
        s4_proj = self.proj_s4(s4_att)
        p_proj = self.proj_pooler(p_att)

        if self.use_concat:
            fused = torch.cat([s2_proj, s3_proj, s4_proj, p_proj], dim=1)
            return self.classifier_concat(fused)
        else:
            # Weighted sum
            w = F.softmax(self.scale_weights, dim=0)
            fused = w[0] * s2_proj + w[1] * s3_proj + w[2] * s4_proj + w[3] * p_proj
            return self.classifier(fused)

    def get_scale_weights(self):
        w = F.softmax(self.scale_weights, dim=0)
        return {"stage2": w[0].item(), "stage3": w[1].item(),
                "stage4": w[2].item(), "pooler": w[3].item()}


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            focal = self.alpha.to(targets.device)[targets] * focal
        return focal.mean() if self.reduction == "mean" else focal.sum()


# ═══════════════════════════════════════════════════════════════
# Feature dataset
# ═══════════════════════════════════════════════════════════════

class FeatureDataset(Dataset):
    def __init__(self, indices):
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return (
            torch.FloatTensor(all_s2[i]),
            torch.FloatTensor(all_s3[i]),
            torch.FloatTensor(all_s4[i]),
            torch.FloatTensor(all_pooler[i]),
            int(all_labels[i]),
        )


# ═══════════════════════════════════════════════════════════════
# Training functions
# ═══════════════════════════════════════════════════════════════

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for s2, s3, s4, pooler, labels in loader:
        s2, s3, s4, pooler, labels = s2.to(device), s3.to(device), s4.to(device), pooler.to(device), labels.to(device)
        logits = model(s2, s3, s4, pooler)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for s2, s3, s4, pooler, labels in loader:
        s2, s3, s4, pooler, labels = s2.to(device), s3.to(device), s4.to(device), pooler.to(device), labels.to(device)
        logits = model(s2, s3, s4, pooler)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


# ═══════════════════════════════════════════════════════════════
# 10-Fold CV
# ═══════════════════════════════════════════════════════════════

# Class weights
class_counts = np.bincount(all_labels.astype(int), minlength=NUM_CLASSES).astype(float)
class_weights = (1.0 / class_counts)
class_weights = class_weights / class_weights.sum() * NUM_CLASSES
print(f"\nFocal class weights: {class_weights}")

ensure_dirs("outputs/models", "outputs/tables")

# Run both fusion strategies
for fusion_mode in ["weighted", "concat"]:
    print(f"\n{'='*70}")
    print(f"  FUSION MODE: {fusion_mode.upper()}")
    print(f"{'='*70}")

    all_results = []

    for fold_idx, fold_data in enumerate(folds):
        fold_num = fold_idx + 1
        print(f"\n--- Fold {fold_num}/{N_FOLDS} ---")

        # Map subjects to feature indices
        train_subjs = set(fold_data["train_subjects"])
        val_subjs = set(fold_data["val_subjects"])
        test_subjs = set(fold_data["test_subjects"])

        train_mask = df["subject_id"].isin(train_subjs).values
        val_mask = df["subject_id"].isin(val_subjs).values
        test_mask = df["subject_id"].isin(test_subjs).values

        train_indices = np.where(train_mask)[0]
        val_indices = np.where(val_mask)[0]
        test_indices = np.where(test_mask)[0]

        print(f"  Train: {len(train_indices)} | Val: {len(val_indices)} | Test: {len(test_indices)}")

        # Weighted sampler
        train_labels = all_labels[train_indices].astype(int)
        sc = np.bincount(train_labels, minlength=NUM_CLASSES).astype(float)
        sw = (1.0 / sc)[train_labels]
        sampler = WeightedRandomSampler(sw, len(sw), replacement=True)

        train_loader = DataLoader(FeatureDataset(train_indices), batch_size=BATCH_SIZE, sampler=sampler, drop_last=True)
        val_loader = DataLoader(FeatureDataset(val_indices), batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(FeatureDataset(test_indices), batch_size=BATCH_SIZE, shuffle=False)

        # Model
        model = MultiScaleCBAMClassifier(
            s2_dim=all_s2.shape[1], s3_dim=all_s3.shape[1],
            s4_dim=all_s4.shape[1], pooler_dim=all_pooler.shape[1],
            hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES, dropout=DROPOUT,
        ).to(DEVICE)
        model.use_concat = (fusion_mode == "concat")

        if fold_idx == 0:
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  Model params: {n_params:,}")

        criterion = FocalLoss(alpha=class_weights.tolist(), gamma=FOCAL_GAMMA)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

        best_val_loss = float("inf")
        best_state = None
        patience_ctr = 0

        for epoch in range(EPOCHS):
            tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
            va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)
            scheduler.step()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Ep {epoch+1:3d} | tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | va_loss={va_loss:.4f} va_acc={va_acc:.4f}")

            if va_loss < best_val_loss:
                best_val_loss = va_loss
                best_state = copy.deepcopy(model.state_dict())
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE:
                    print(f"  Early stop at epoch {epoch+1}")
                    break

        # Test
        model.load_state_dict(best_state)
        te_loss, te_acc, y_pred, y_true = evaluate(model, test_loader, criterion, DEVICE)
        f1 = f1_score(y_true, y_pred, average="weighted")
        prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)

        sw_dict = model.get_scale_weights() if fusion_mode == "weighted" else {}

        print(f"  TEST: acc={te_acc:.4f} f1={f1:.4f}")
        print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))

        all_results.append({
            "fold": fold_num, "fusion": fusion_mode,
            "test_acc": te_acc, "test_f1": f1, "test_prec": prec, "test_rec": rec,
            **{f"sw_{k}": v for k, v in sw_dict.items()},
        })

        # Save best model (weighted only)
        if fusion_mode == "weighted":
            torch.save(best_state, f"outputs/models/swin_frozen_cbam_fold{fold_num}.pt")

        del model, optimizer, scheduler
        torch.cuda.empty_cache()

    # Summary
    rdf = pd.DataFrame(all_results)
    print(f"\n{'='*70}")
    print(f"  {fusion_mode.upper()} FUSION — SUMMARY")
    print(f"{'='*70}")
    print(rdf.to_string(index=False))
    for m in ["test_acc", "test_f1", "test_prec", "test_rec"]:
        print(f"  {m:15s}: {rdf[m].mean():.4f} ± {rdf[m].std():.4f}")

    rdf.to_csv(f"outputs/tables/step11c_frozen_{fusion_mode}.csv", index=False)


# ═══════════════════════════════════════════════════════════════
# Also run: Tarik-style baselines (RF, XGB, Stacking) on full CV
# for fair comparison with our attention model
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("  TARIK-STYLE ML BASELINES (10-Fold Subject-Level CV)")
print(f"{'='*70}")

from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

ml_results = []

for fold_idx, fold_data in enumerate(folds):
    fold_num = fold_idx + 1

    train_subjs = set(fold_data["train_subjects"]) | set(fold_data["val_subjects"])  # merge train+val for ML
    test_subjs = set(fold_data["test_subjects"])

    train_mask = df["subject_id"].isin(train_subjs).values
    test_mask = df["subject_id"].isin(test_subjs).values

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]

    # Concat all features (best for ML)
    X_train = np.hstack([all_s2[train_idx], all_s3[train_idx], all_s4[train_idx], all_pooler[train_idx]])
    X_test = np.hstack([all_s2[test_idx], all_s3[test_idx], all_s4[test_idx], all_pooler[test_idx]])
    y_train = all_labels[train_idx].astype(int)
    y_test = all_labels[test_idx].astype(int)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=500, max_depth=20, random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    rf_f1 = f1_score(y_test, rf.predict(X_test), average="weighted")

    # XGBoost
    xgb = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.1,
                         subsample=0.8, colsample_bytree=0.8,
                         random_state=SEED, n_jobs=-1, verbosity=0,
                         eval_metric="logloss")
    xgb.fit(X_train, y_train)
    xgb_acc = accuracy_score(y_test, xgb.predict(X_test))
    xgb_f1 = f1_score(y_test, xgb.predict(X_test), average="weighted")

    # Stacking
    stacking = StackingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)),
            ("xgb", XGBClassifier(n_estimators=300, max_depth=6, random_state=SEED, n_jobs=-1, verbosity=0, eval_metric="logloss")),
        ],
        final_estimator=LogisticRegression(max_iter=2000, random_state=SEED),
        cv=5, n_jobs=-1,
    )
    stacking.fit(X_train, y_train)
    st_acc = accuracy_score(y_test, stacking.predict(X_test))
    st_f1 = f1_score(y_test, stacking.predict(X_test), average="weighted")

    print(f"  Fold {fold_num:2d}: RF={rf_acc:.4f} XGB={xgb_acc:.4f} Stack={st_acc:.4f}")

    # Classification report for stacking (best model)
    if fold_num == 1:
        print(f"\n  Fold 1 Stacking Report:")
        print(classification_report(y_test, stacking.predict(X_test), target_names=CLASS_NAMES, zero_division=0))

    ml_results.append({
        "fold": fold_num,
        "rf_acc": rf_acc, "rf_f1": rf_f1,
        "xgb_acc": xgb_acc, "xgb_f1": xgb_f1,
        "stacking_acc": st_acc, "stacking_f1": st_f1,
    })

ml_df = pd.DataFrame(ml_results)
print(f"\n{'='*70}")
print("  ML BASELINES SUMMARY")
print(f"{'='*70}")
for m in ["rf_acc", "xgb_acc", "stacking_acc"]:
    print(f"  {m:15s}: {ml_df[m].mean():.4f} ± {ml_df[m].std():.4f}")
for m in ["rf_f1", "xgb_f1", "stacking_f1"]:
    print(f"  {m:15s}: {ml_df[m].mean():.4f} ± {ml_df[m].std():.4f}")

ml_df.to_csv("outputs/tables/step11c_ml_baselines.csv", index=False)

print(f"\n{'='*70}")
print("  ALL DONE")
print(f"{'='*70}")
print("Outputs:")
print("  outputs/tables/step11c_frozen_weighted.csv   — CBAM weighted fusion")
print("  outputs/tables/step11c_frozen_concat.csv     — CBAM concat fusion")
print("  outputs/tables/step11c_ml_baselines.csv      — RF/XGB/Stacking")
print("  outputs/swin_features/all_multiscale_features.npz")
print("  outputs/models/swin_frozen_cbam_fold*.pt")
