"""
Step 02: 10-Fold Subject-Level CV with Ablation
================================================
Trains multiple configurations across 10 folds.
Reports mean ± std for all metrics.

Configurations:
  1. EfficientNet-B3 (CE) — clean baseline
  2. EfficientNet-B3 (CE + ClassWeights) — imbalance handling
  3. EfficientNet-B3 + CBAM (CE + ClassWeights) — attention
  4. EfficientNet-B3 + CBAM + MultiScale (CE + ClassWeights) — full model
"""
import os, sys, json, time, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.helpers import set_seed, load_config, get_device, ensure_dirs
from src.models.hybrid_model import HybridAttentionNet
from src.data.dataset import AlzDataset
from scripts.data_utils import get_transforms

# ── Config ──
cfg = load_config("configs/config.yaml")
set_seed(cfg["project"]["seed"])
device = get_device()
CLASS_NAMES = cfg["data"]["class_names"]
NUM_CLASSES = cfg["data"]["num_classes"]
SPLITS_DIR = cfg["data"]["splits_dir"]

print("=" * 70)
print("  Step 02: 10-Fold Subject-Level CV with Ablation")
print("=" * 70)
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Classes: {CLASS_NAMES}")

# ── Load data ──
full_df = pd.read_csv(os.path.join(SPLITS_DIR, "full_data.csv"))
with open(os.path.join(SPLITS_DIR, "cv_folds.json")) as f:
    folds = json.load(f)

print(f"Total images: {len(full_df):,}")
print(f"Folds: {len(folds)}")

train_transform, val_transform = get_transforms(224)

# ── Model configs to evaluate ──
MODEL_CONFIGS = [
    {"name": "EfficientNet-B3 (CE)",
     "backbone": "efficientnet_b3", "attention": False, "multiscale": False,
     "loss": "ce", "class_weights": False},
    {"name": "EfficientNet-B3 (CE+CW)",
     "backbone": "efficientnet_b3", "attention": False, "multiscale": False,
     "loss": "ce", "class_weights": True},
    {"name": "EfficientNet-B3+CBAM (CE+CW)",
     "backbone": "efficientnet_b3", "attention": True, "multiscale": False,
     "loss": "ce", "class_weights": True},
    {"name": "EfficientNet-B3+CBAM+MS (CE+CW)",
     "backbone": "efficientnet_b3", "attention": True, "multiscale": True,
     "loss": "ce", "class_weights": True},
]


def get_class_weights(labels):
    """Compute inverse frequency class weights."""
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * NUM_CLASSES
    return torch.FloatTensor(weights).to(device)


def train_one_fold(model, train_loader, val_loader, criterion,
                   num_epochs=50, lr=3e-4, weight_decay=1e-3, patience=10):
    """Train one fold, return best model state and best val metrics."""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_f1, best_state = 0.0, None
    patience_counter = 0

    for epoch in range(num_epochs):
        # Train
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Validate
        model.eval()
        val_preds, val_labels, val_probs = [], [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                val_preds.extend(probs.argmax(1).cpu().numpy())
                val_labels.extend(labels.numpy())
                val_probs.extend(probs.cpu().numpy())

        val_f1 = f1_score(val_labels, val_preds, average="macro")
        scheduler.step()

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    model.load_state_dict(best_state)
    return model, best_f1


def evaluate_model(model, test_loader):
    """Evaluate model on test set, return metrics dict."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            all_preds.extend(probs.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    try:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    except:
        auc = 0.0

    # Sensitivity (recall for Demented=1) and Specificity (recall for Non-Demented=0)
    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True)
    sensitivity = report["Demented"]["recall"]
    specificity = report["Non-Demented"]["recall"]

    return {
        "accuracy": acc,
        "f1_macro": f1,
        "auc_roc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


# ── Main CV Loop ──
all_results = []

for cfg_idx, mcfg in enumerate(MODEL_CONFIGS):
    print(f"\n{'='*70}")
    print(f"  Config {cfg_idx+1}/{len(MODEL_CONFIGS)}: {mcfg['name']}")
    print(f"{'='*70}")

    fold_metrics = []
    t_start = time.time()

    for fold_idx, fold in enumerate(folds):
        set_seed(42 + fold_idx)

        # Get fold data
        train_subjs = set(fold["train_subjects"])
        val_subjs = set(fold["val_subjects"])
        test_subjs = set(fold["test_subjects"])

        train_df = full_df[full_df["subject_id"].isin(train_subjs)]
        val_df = full_df[full_df["subject_id"].isin(val_subjs)]
        test_df = full_df[full_df["subject_id"].isin(test_subjs)]

        # Create loaders
        train_ds = AlzDataset(train_df["path"].tolist(), train_df["label"].tolist(), train_transform)
        val_ds = AlzDataset(val_df["path"].tolist(), val_df["label"].tolist(), val_transform)
        test_ds = AlzDataset(test_df["path"].tolist(), test_df["label"].tolist(), val_transform)

        train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

        # Build model
        model = HybridAttentionNet(
            backbone_name=mcfg["backbone"],
            num_classes=NUM_CLASSES,
            pretrained=True,
            dropout=0.4,
            use_attention=mcfg["attention"],
            use_multiscale=mcfg["multiscale"],
        ).to(device)

        # Loss
        if mcfg["class_weights"]:
            weights = get_class_weights(train_df["label"].values)
            criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            criterion = nn.CrossEntropyLoss()

        # Train
        model, best_val_f1 = train_one_fold(
            model, train_loader, val_loader, criterion,
            num_epochs=50, lr=3e-4, weight_decay=1e-3, patience=10
        )

        # Test
        metrics = evaluate_model(model, test_loader)
        fold_metrics.append(metrics)

        print(f"  Fold {fold_idx+1:>2}: Acc={metrics['accuracy']:.4f} "
              f"F1={metrics['f1_macro']:.4f} AUC={metrics['auc_roc']:.4f} "
              f"Sens={metrics['sensitivity']:.4f} Spec={metrics['specificity']:.4f}")

        # Save best fold model (fold 0)
        if fold_idx == 0:
            safe = mcfg["name"].lower().replace(" ", "_").replace("+", "_").replace("(", "").replace(")", "")
            torch.save(model.state_dict(), f"outputs/models/{safe}_fold0.pth")

        del model, train_loader, val_loader, test_loader
        torch.cuda.empty_cache()

    elapsed = time.time() - t_start

    # Aggregate
    metrics_keys = ["accuracy", "f1_macro", "auc_roc", "sensitivity", "specificity"]
    means = {k: np.mean([m[k] for m in fold_metrics]) for k in metrics_keys}
    stds = {k: np.std([m[k] for m in fold_metrics]) for k in metrics_keys}

    print(f"\n  {'─'*60}")
    print(f"  {mcfg['name']} — 10-Fold CV Results:")
    print(f"  Accuracy:    {means['accuracy']:.4f} ± {stds['accuracy']:.4f}")
    print(f"  F1 (Macro):  {means['f1_macro']:.4f} ± {stds['f1_macro']:.4f}")
    print(f"  AUC-ROC:     {means['auc_roc']:.4f} ± {stds['auc_roc']:.4f}")
    print(f"  Sensitivity: {means['sensitivity']:.4f} ± {stds['sensitivity']:.4f}")
    print(f"  Specificity: {means['specificity']:.4f} ± {stds['specificity']:.4f}")
    print(f"  Time: {elapsed/60:.1f} min")

    result_row = {"Config": mcfg["name"]}
    for k in metrics_keys:
        result_row[f"{k}_mean"] = f"{means[k]:.4f}"
        result_row[f"{k}_std"] = f"{stds[k]:.4f}"
    result_row["fold_details"] = fold_metrics
    all_results.append(result_row)

# ── Save Results ──
ensure_dirs("outputs/tables")
summary_rows = []
for r in all_results:
    row = {k: v for k, v in r.items() if k != "fold_details"}
    summary_rows.append(row)

df_results = pd.DataFrame(summary_rows)
df_results.to_csv("outputs/tables/cv_results.csv", index=False)

# Save detailed per-fold
detailed = []
for r in all_results:
    for i, fm in enumerate(r["fold_details"]):
        row = {"Config": r["Config"], "Fold": i+1}
        row.update(fm)
        detailed.append(row)
pd.DataFrame(detailed).to_csv("outputs/tables/cv_results_per_fold.csv", index=False)

print(f"\n{'='*70}")
print("  FINAL 10-FOLD CV SUMMARY")
print(f"{'='*70}")
print(df_results.to_string(index=False))
print(f"\nSaved: outputs/tables/cv_results.csv")
print(f"Saved: outputs/tables/cv_results_per_fold.csv")
print("\nStep 02 DONE.")
