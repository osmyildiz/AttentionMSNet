"""
Step 03: Architecture Ablation
===============================
PURPOSE: Isolate the contribution of each ARCHITECTURAL component.

FIXED from Step 02c: best imbalance strategy (loss + sampling)
VARIABLE: architecture only

Experiments:
  A) CNN backbone only           (no attention, no multiscale)
  B) CNN + CBAM                  (attention only)
  C) CNN + MultiScale            (fusion only)
  D) CNN + CBAM + MultiScale     (full hybrid)

This produces TABLE 3 in the paper: "Architectural Contribution Analysis"
"""
import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.helpers import set_seed, load_config, get_device, ensure_dirs
from src.training.losses import FocalLoss
from src.evaluation.metrics import print_results
from src.models.hybrid_model import HybridAttentionNet
from scripts.train_utils import train_model, evaluate
from scripts.data_utils import get_transforms, load_splits, create_loaders

print("=" * 70)
print("  Step 03: Architecture Ablation")
print("  FIXED loss/sampling from Step 02c, VARIABLE architecture")
print("=" * 70)

cfg = load_config("configs/config.yaml")
set_seed(cfg["project"]["seed"])
device = get_device()

NUM_CLASSES = cfg["data"]["num_classes"]
CLASS_NAMES = cfg["data"]["class_names"]
BATCH_SIZE = cfg["training"]["batch_size"]
NUM_EPOCHS = cfg["training"]["epochs"]
LR = cfg["training"]["lr"]
WEIGHT_DECAY = cfg["training"]["weight_decay"]
PATIENCE = cfg["training"]["early_stopping_patience"]
NUM_WORKERS = cfg["training"].get("num_workers", 8)
DROPOUT = cfg["model"]["dropout"]
BACKBONE = cfg["model"]["backbone"]

ensure_dirs("outputs/models", "outputs/tables", "logs")

train_transform, val_transform = get_transforms(cfg["data"]["image_size"])
train_df, val_df, test_df = load_splits(cfg["data"]["splits_dir"])

# ── Determine best imbalance strategy from 02c ──
try:
    prev = pd.read_csv("outputs/tables/step02c_imbalance_ablation.csv")
    best_strategy = prev.loc[prev["F1_Macro"].astype(float).idxmax(), "Strategy"]
    print(f"Best imbalance strategy from Step 02c: {best_strategy}")
except FileNotFoundError:
    best_strategy = "Focal"
    print(f"Step 02c results not found, defaulting to: {best_strategy}")

# ── Configure loss and sampling based on best strategy ──
train_labels = train_df["label"].values
class_counts = np.bincount(train_labels)
cw = torch.tensor([len(train_labels) / (NUM_CLASSES * c) for c in class_counts], dtype=torch.float32).to(device)

if "WeightedSampler" in best_strategy:
    sampling = "weighted"
else:
    sampling = "shuffle"

if "Focal" in best_strategy and "ClassWeights" in best_strategy:
    criterion = FocalLoss(alpha=0.25, gamma=2.0, weight=cw)
elif "Focal" in best_strategy:
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
elif "ClassWeights" in best_strategy:
    criterion = nn.CrossEntropyLoss(weight=cw)
else:
    criterion = nn.CrossEntropyLoss()

print(f"FIXED Loss: {criterion.__class__.__name__}")
print(f"FIXED Sampling: {sampling}")

train_loader, val_loader, test_loader = create_loaders(
    train_df, val_df, test_df, train_transform, val_transform,
    batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, sampling=sampling
)

# ── Architecture configs ──
arch_configs = [
    {"name": "CNN_only",            "use_attention": False, "use_multiscale": False},
    {"name": "CNN+CBAM",            "use_attention": True,  "use_multiscale": False},
    {"name": "CNN+MultiScale",      "use_attention": False, "use_multiscale": True},
    {"name": "CNN+CBAM+MultiScale", "use_attention": True,  "use_multiscale": True},
]

# ── Run ablation ──
results = {}
for ac in arch_configs:
    name = ac["name"]
    print(f"\n{'='*70}")
    print(f"  Architecture: {BACKBONE} + {name}")
    print(f"  Attention: {ac['use_attention']}, MultiScale: {ac['use_multiscale']}")
    print(f"{'='*70}")

    model = HybridAttentionNet(
        backbone_name=BACKBONE,
        num_classes=NUM_CLASSES,
        pretrained=True,
        dropout=DROPOUT,
        use_attention=ac["use_attention"],
        use_multiscale=ac["use_multiscale"],
    )

    model, _, _ = train_model(
        model, f"arch_{name}", train_loader, val_loader, criterion, device,
        CLASS_NAMES, num_epochs=NUM_EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
        patience=PATIENCE, log_dir="logs"
    )

    safe = name.lower().replace("+", "_")
    torch.save(model.state_dict(), f"outputs/models/arch_{safe}.pth")

    test_metrics, _ = evaluate(model, test_loader, criterion, device, CLASS_NAMES)
    print_results(test_metrics, title=f"{name} (Test)")
    results[name] = test_metrics

    del model
    torch.cuda.empty_cache()

# ── Summary Table ──
rows = []
for name, m in results.items():
    report = m["classification_report"]
    rows.append({
        "Architecture": name,
        "Imbalance": best_strategy,
        "Accuracy": f'{m["accuracy"]:.4f}',
        "F1_Macro": f'{m["f1_macro"]:.4f}',
        "AUC_ROC": f'{m["auc_roc"]:.4f}' if m.get("auc_roc") else "N/A",
        "MildMod_Recall": f'{report["Mild+Moderate Dementia"]["recall"]:.4f}',
        "MildMod_F1": f'{report["Mild+Moderate Dementia"]["f1-score"]:.4f}',
    })

comp_df = pd.DataFrame(rows)
comp_df.to_csv("outputs/tables/step03_architecture_ablation.csv", index=False)

print("\n" + "=" * 90)
print("  ARCHITECTURE ABLATION")
print("=" * 90)
print(comp_df.to_string(index=False))
print("=" * 90)
print("\nStep 03 DONE.")
