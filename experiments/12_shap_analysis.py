"""
Step 06: SHAP Feature Analysis
================================
PURPOSE: Analyze WHICH features drive classification decisions.
Complements Grad-CAM++ (spatial) with feature-level explanation.

Uses XGBoost/LightGBM from Step 04 (tree-based SHAP is exact and fast).

Outputs:
  - SHAP summary (beeswarm) plot
  - Per-class SHAP force plots
  - Feature importance ranking
  - SHAP interaction analysis

No training - analysis only. Runs fast (~10-20 min).
"""
import os, sys, json, pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.helpers import set_seed, load_config, get_device, ensure_dirs
from src.models.hybrid_model import HybridAttentionNet
from src.data.dataset import AlzDataset
from src.ensemble.ensemble import extract_features
from scripts.data_utils import get_transforms

print("=" * 70)
print("  Step 06: SHAP Feature Analysis")
print("=" * 70)

cfg = load_config("configs/config.yaml")
set_seed(cfg["project"]["seed"])
device = get_device()

NUM_CLASSES = cfg["data"]["num_classes"]
CLASS_NAMES = cfg["data"]["class_names"]
IMG_SIZE = cfg["data"]["image_size"]
BATCH_SIZE = cfg["training"]["batch_size"]
NUM_WORKERS = cfg["training"].get("num_workers", 0)
BACKBONE = cfg["model"]["backbone"]
DROPOUT = cfg["model"]["dropout"]

ensure_dirs("outputs/shap", "outputs/figures")

try:
    import shap
    print(f"SHAP version: {shap.__version__}")
except ImportError:
    print("ERROR: shap not installed. Run: pip install shap")
    sys.exit(1)

try:
    import xgboost as xgb
    from xgboost import XGBClassifier
    print(f"XGBoost version: {xgb.__version__}")
except ImportError:
    print("ERROR: xgboost not installed. Run: pip install xgboost")
    sys.exit(1)


# ══════════════════════════════════════════════════════════
# 1. Load Model & Extract Features
# ══════════════════════════════════════════════════════════

model_path = "outputs/models/arch_cnn_cbam_multiscale.pth"
if not os.path.exists(model_path):
    for alt in ["outputs/models/C_CNN_CBAM_MultiScale.pth"]:
        if os.path.exists(alt):
            model_path = alt
            break
    else:
        print("ERROR: No trained hybrid model found.")
        sys.exit(1)

print(f"Loading model: {model_path}")
model = HybridAttentionNet(
    backbone_name=BACKBONE,
    num_classes=NUM_CLASSES,
    pretrained=False,
    dropout=DROPOUT,
    use_attention=True,
    use_multiscale=True,
).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

_, val_transform = get_transforms(IMG_SIZE)

SPLITS_DIR = Path(cfg["data"]["splits_dir"])
train_df = pd.read_csv(SPLITS_DIR / "train.csv")
test_df = pd.read_csv(SPLITS_DIR / "test.csv")

# Use a subset of train for SHAP background (full set too slow)
train_subset = train_df.groupby("label").apply(
    lambda x: x.sample(n=min(200, len(x)), random_state=42)
).reset_index(drop=True)

train_sub_dataset = AlzDataset(train_subset["path"].tolist(), train_subset["label"].tolist(), val_transform)
test_dataset = AlzDataset(test_df["path"].tolist(), test_df["label"].tolist(), val_transform)

train_sub_loader = DataLoader(train_sub_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

print("Extracting features...")
X_train_sub, y_train_sub = extract_features(model, train_sub_loader, device)
X_test, y_test = extract_features(model, test_loader, device)
print(f"  Train subset: {X_train_sub.shape}")
print(f"  Test: {X_test.shape}")


# ══════════════════════════════════════════════════════════
# 2. Train XGBoost for SHAP (TreeExplainer is exact)
# ══════════════════════════════════════════════════════════

print("\nTraining XGBoost for SHAP analysis...")
xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    objective="multi:softprob",
    num_class=NUM_CLASSES,
    eval_metric="mlogloss",
    random_state=42,
    use_label_encoder=False,
    n_jobs=-1,
)
xgb_model.fit(X_train_sub, y_train_sub, verbose=False)
print("XGBoost trained.")


# ══════════════════════════════════════════════════════════
# 3. SHAP TreeExplainer
# ══════════════════════════════════════════════════════════

print("\nComputing SHAP values (this may take a few minutes)...")
explainer = shap.TreeExplainer(xgb_model)

# Use a test subset for visualization (full test too dense)
test_sample_idx = []
for label in range(NUM_CLASSES):
    cls_idx = np.where(y_test == label)[0]
    n_sample = min(50, len(cls_idx))
    test_sample_idx.extend(np.random.choice(cls_idx, n_sample, replace=False))
test_sample_idx = sorted(test_sample_idx)

X_test_sample = X_test[test_sample_idx]
y_test_sample = y_test[test_sample_idx]

shap_values = explainer.shap_values(X_test_sample)
print(f"SHAP values computed for {len(test_sample_idx)} test samples.")

# shap_values is a list of [n_samples, n_features] arrays, one per class
if isinstance(shap_values, list):
    n_features = shap_values[0].shape[1]
else:
    n_features = shap_values.shape[2]

print(f"Feature dimension: {n_features}")


# ══════════════════════════════════════════════════════════
# 4. Figure 1: SHAP Summary (Beeswarm) - Per Class
# ══════════════════════════════════════════════════════════

print("\nGenerating SHAP figures...")

fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(NUM_CLASSES * 5, 6))
for cls_idx, cls_name in enumerate(CLASS_NAMES):
    plt.sca(axes[cls_idx])
    if isinstance(shap_values, list):
        sv = shap_values[cls_idx]
    else:
        sv = shap_values[:, :, cls_idx]
    shap.summary_plot(sv, X_test_sample, max_display=15, show=False,
                      plot_size=None, color_bar=False)
    axes[cls_idx].set_title(cls_name, fontsize=10, fontweight="bold")
    if cls_idx > 0:
        axes[cls_idx].set_ylabel("")

plt.suptitle("SHAP Feature Importance by Class", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/figures/shap_summary_per_class.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: outputs/figures/shap_summary_per_class.png")


# ══════════════════════════════════════════════════════════
# 5. Figure 2: Global Feature Importance
# ══════════════════════════════════════════════════════════

if isinstance(shap_values, list):
    mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
else:
    mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))

top_k = 30
top_indices = np.argsort(mean_abs_shap)[-top_k:][::-1]
top_importances = mean_abs_shap[top_indices]

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(range(top_k), top_importances[::-1], color="#3498db", alpha=0.8)
ax.set_yticks(range(top_k))
ax.set_yticklabels([f"Feature {i}" for i in top_indices[::-1]], fontsize=8)
ax.set_xlabel("Mean |SHAP Value|")
ax.set_title(f"Top {top_k} Most Important Deep Features", fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/figures/shap_global_importance.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: outputs/figures/shap_global_importance.png")


# ══════════════════════════════════════════════════════════
# 6. Figure 3: Class Discrimination Analysis
# ══════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(12, 6))
colors = ["#2ecc71", "#f39c12", "#e74c3c", "#9b59b6"]

x = np.arange(top_k)
width = 0.2

for cls_idx, (cls_name, color) in enumerate(zip(CLASS_NAMES, colors)):
    if isinstance(shap_values, list):
        cls_importance = np.abs(shap_values[cls_idx]).mean(axis=0)[top_indices[:top_k]]
    else:
        cls_importance = np.abs(shap_values[:, :, cls_idx]).mean(axis=0)[top_indices[:top_k]]
    ax.bar(x + cls_idx * width, cls_importance, width, label=cls_name, color=color, alpha=0.8)

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels([f"F{i}" for i in top_indices[:top_k]], fontsize=7, rotation=45)
ax.set_ylabel("Mean |SHAP Value|")
ax.set_title("Feature Importance by Class (Top 30 Features)", fontweight="bold")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("outputs/figures/shap_class_discrimination.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: outputs/figures/shap_class_discrimination.png")


# ══════════════════════════════════════════════════════════
# 7. Feature Importance Table
# ══════════════════════════════════════════════════════════

importance_rows = []
for rank, idx in enumerate(top_indices[:50]):
    row = {"Rank": rank + 1, "Feature_Index": int(idx), "Global_Importance": f"{mean_abs_shap[idx]:.6f}"}
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        if isinstance(shap_values, list):
            cls_imp = np.abs(shap_values[cls_idx])[:, idx].mean()
        else:
            cls_imp = np.abs(shap_values[:, idx, cls_idx]).mean()
        row[f"{cls_name}_Importance"] = f"{cls_imp:.6f}"
    importance_rows.append(row)

imp_df = pd.DataFrame(importance_rows)
imp_df.to_csv("outputs/tables/step06_feature_importance.csv", index=False)
print("  Saved: outputs/tables/step06_feature_importance.csv")


# ══════════════════════════════════════════════════════════
# 8. Summary
# ══════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  SHAP Analysis Summary")
print("=" * 60)
print(f"  Total features analyzed: {n_features}")
print(f"  Test samples analyzed: {len(test_sample_idx)}")
print(f"  Top feature (global): Feature {top_indices[0]} (importance: {top_importances[0]:.6f})")

# Per-class top features
for cls_idx, cls_name in enumerate(CLASS_NAMES):
    if isinstance(shap_values, list):
        cls_top = np.abs(shap_values[cls_idx]).mean(axis=0).argmax()
    else:
        cls_top = np.abs(shap_values[:, :, cls_idx]).mean(axis=0).argmax()
    print(f"  Top feature for {cls_name:25s}: Feature {cls_top}")

print("\nOutputs:")
print("  outputs/figures/shap_summary_per_class.png     (paper Figure)")
print("  outputs/figures/shap_global_importance.png      (paper Figure)")
print("  outputs/figures/shap_class_discrimination.png   (paper Figure)")
print("  outputs/tables/step06_feature_importance.csv    (supplementary)")
print("\nStep 06 DONE.")
