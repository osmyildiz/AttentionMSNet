import functools, builtins
builtins.print = functools.partial(print, flush=True)

import os, sys, json, time, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

sys.path.insert(0, '.')
from src.utils.helpers import set_seed, load_config, get_device
from src.models.hybrid_model import HybridAttentionNet
from src.data.dataset import AlzDataset
from scripts.data_utils import get_transforms

cfg = load_config("configs/config.yaml")
device = get_device()
CLASS_NAMES = cfg["data"]["class_names"]
NUM_CLASSES = cfg["data"]["num_classes"]

print("=" * 70)
print("  IMAGE-LEVEL 10-Fold CV (Leakage Experiment)")
print("=" * 70)
print(f"GPU: {torch.cuda.get_device_name(0)}")

full_df = pd.read_csv("data/splits/full_data.csv")
print(f"Total: {len(full_df)} images")

train_transform, val_transform = get_transforms(224)

def get_class_weights(labels):
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    w = 1.0 / (counts + 1e-6)
    w = w / w.sum() * NUM_CLASSES
    return torch.FloatTensor(w).to(device)

# IMAGE-LEVEL 10-fold (slices randomly split, NO subject isolation)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
all_indices = np.arange(len(full_df))
all_labels = full_df["label"].values

fold_results = []

for fold, (train_val_idx, test_idx) in enumerate(skf.split(all_indices, all_labels)):
    print(f"\n--- Fold {fold+1}/10 ---")
    set_seed(42 + fold)

    # Split train_val into train/val (90/10)
    np.random.seed(42 + fold)
    val_size = len(train_val_idx) // 10
    perm = np.random.permutation(len(train_val_idx))
    val_idx = train_val_idx[perm[:val_size]]
    train_idx = train_val_idx[perm[val_size:]]

    train_df = full_df.iloc[train_idx]
    val_df = full_df.iloc[val_idx]
    test_df = full_df.iloc[test_idx]

    # Check overlap
    train_subjs = set(train_df["subject_id"])
    test_subjs = set(test_df["subject_id"])
    overlap = len(train_subjs & test_subjs)
    print(f"  Subject overlap: {overlap}/{len(test_subjs)} ({overlap/max(len(test_subjs),1)*100:.0f}%)")
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    train_ds = AlzDataset(train_df["path"].tolist(), train_df["label"].tolist(), train_transform)
    val_ds = AlzDataset(val_df["path"].tolist(), val_df["label"].tolist(), val_transform)
    test_ds = AlzDataset(test_df["path"].tolist(), test_df["label"].tolist(), val_transform)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

    model = HybridAttentionNet("efficientnet_b3", NUM_CLASSES, True, 0.4, True, True).to(device)
    weights = get_class_weights(train_df["label"].values)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    best_f1, best_state, pat = 0.0, None, 0
    for epoch in range(50):
        model.train()
        for img, lbl in train_loader:
            img, lbl = img.to(device), lbl.to(device)
            optimizer.zero_grad()
            loss = criterion(model(img), lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        vp, vl = [], []
        with torch.no_grad():
            for img, lbl in val_loader:
                out = model(img.to(device))
                vp.extend(out.argmax(1).cpu().numpy())
                vl.extend(lbl.numpy())
        vf1 = f1_score(vl, vp, average="macro")
        scheduler.step()

        if vf1 > best_f1:
            best_f1 = vf1
            best_state = copy.deepcopy(model.state_dict())
            pat = 0
        else:
            pat += 1

        if epoch % 5 == 0 or pat == 0:
            print(f"    E{epoch+1:>2}/50 VaF1:{vf1:.4f}{' *BEST*' if pat==0 else ''}")

        if pat >= 10:
            print(f"    >> Early stop at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    model.eval()
    ap, al, aprob = [], [], []
    with torch.no_grad():
        for img, lbl in test_loader:
            out = model(img.to(device))
            p = torch.softmax(out, dim=1)
            ap.extend(p.argmax(1).cpu().numpy())
            al.extend(lbl.numpy())
            aprob.extend(p.cpu().numpy())

    ap, al, aprob = np.array(ap), np.array(al), np.array(aprob)
    acc = accuracy_score(al, ap)
    f1 = f1_score(al, ap, average="macro")
    try: auc = roc_auc_score(al, aprob[:,1])
    except: auc = 0.0
    rpt = classification_report(al, ap, target_names=CLASS_NAMES, output_dict=True)
    sens = rpt["Demented"]["recall"]
    spec = rpt["Non-Demented"]["recall"]

    print(f"  Fold {fold+1} TEST: Acc={acc:.4f} F1={f1:.4f} AUC={auc:.4f} Sens={sens:.4f} Spec={spec:.4f}")
    fold_results.append({"fold": fold+1, "acc": acc, "f1": f1, "auc": auc, "sens": sens, "spec": spec})

    del model; torch.cuda.empty_cache()

# Summary
print(f"\n{'='*70}")
print("  IMAGE-LEVEL 10-FOLD CV SUMMARY (WITH LEAKAGE)")
print(f"{'='*70}")
df = pd.DataFrame(fold_results)
for col in ["acc", "f1", "auc", "sens", "spec"]:
    print(f"  {col:6s}: {df[col].mean():.4f} +/- {df[col].std():.4f}")

df.to_csv("outputs/tables/leakage_10fold_imagelevel.csv", index=False)
print("\nSaved: outputs/tables/leakage_10fold_imagelevel.csv")
print("DONE")
