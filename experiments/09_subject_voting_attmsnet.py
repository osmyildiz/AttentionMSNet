"""
Subject-Level Voting: Train AttentionMS-Net per fold, then aggregate
slice predictions per subject. Full 10-fold CV.
"""
import functools, builtins
builtins.print = functools.partial(print, flush=True)

import os, sys, json, copy, time
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
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
print("  AttentionMS-Net + Subject-Level Voting (10-fold CV)")
print("=" * 70)
print(f"GPU: {torch.cuda.get_device_name(0)}")

full_df = pd.read_csv("data/splits/full_data.csv")
with open("data/splits/cv_folds.json") as f:
    folds = json.load(f)

train_transform, val_transform = get_transforms(224)

def get_class_weights(labels):
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    w = 1.0 / (counts + 1e-6)
    w = w / w.sum() * NUM_CLASSES
    return torch.FloatTensor(w).to(device)

slice_results = []
subject_results = []

for fi, fold in enumerate(folds):
    set_seed(42 + fi)
    print(f"\n{'='*60}")
    print(f"  Fold {fi+1}/10")
    print(f"{'='*60}")
    
    train_subjs = set(fold["train_subjects"])
    val_subjs = set(fold["val_subjects"])
    test_subjs = set(fold["test_subjects"])
    
    train_df = full_df[full_df["subject_id"].isin(train_subjs)]
    val_df = full_df[full_df["subject_id"].isin(val_subjs)]
    test_df = full_df[full_df["subject_id"].isin(test_subjs)].copy()
    
    print(f"  Train:{len(train_df)} Val:{len(val_df)} Test:{len(test_df)} ({test_df['subject_id'].nunique()} subjects)")
    
    train_ds = AlzDataset(train_df["path"].tolist(), train_df["label"].tolist(), train_transform)
    val_ds = AlzDataset(val_df["path"].tolist(), val_df["label"].tolist(), val_transform)
    test_ds = AlzDataset(test_df["path"].tolist(), test_df["label"].tolist(), val_transform)
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
    
    # Train
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
            best_f1 = vf1; best_state = copy.deepcopy(model.state_dict()); pat = 0
        else:
            pat += 1
        if pat >= 10:
            print(f"  Early stop at epoch {epoch+1}")
            break
    
    # Evaluate
    model.load_state_dict(best_state)
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for img, lbl in test_loader:
            out = model(img.to(device))
            p = torch.softmax(out, dim=1)
            all_probs.extend(p.cpu().numpy())
            all_labels.extend(lbl.numpy())
    
    all_probs = np.array(all_probs)
    all_preds = all_probs.argmax(1)
    all_labels = np.array(all_labels)
    
    # Slice-level metrics
    s_acc = accuracy_score(all_labels, all_preds)
    s_f1 = f1_score(all_labels, all_preds, average="macro")
    s_auc = roc_auc_score(all_labels, all_probs[:, 1])
    s_rpt = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True)
    s_sens = s_rpt["Demented"]["recall"]
    s_spec = s_rpt["Non-Demented"]["recall"]
    
    # Subject-level voting
    test_df = test_df.reset_index(drop=True)
    test_df["prob_1"] = all_probs[:, 1]
    
    subj_agg = test_df.groupby("subject_id").agg(
        label=("label", "first"),
        mean_prob=("prob_1", "mean"),
    ).reset_index()
    
    subj_preds = (subj_agg["mean_prob"] > 0.5).astype(int).values
    subj_labels = subj_agg["label"].values
    subj_probs = subj_agg["mean_prob"].values
    
    j_acc = accuracy_score(subj_labels, subj_preds)
    j_f1 = f1_score(subj_labels, subj_preds, average="macro")
    j_auc = roc_auc_score(subj_labels, subj_probs)
    j_rpt = classification_report(subj_labels, subj_preds, target_names=CLASS_NAMES, output_dict=True)
    j_sens = j_rpt["Demented"]["recall"]
    j_spec = j_rpt["Non-Demented"]["recall"]
    
    print(f"  Slice:   Acc={s_acc:.4f} F1={s_f1:.4f} AUC={s_auc:.4f} Sens={s_sens:.4f} Spec={s_spec:.4f}")
    print(f"  Subject: Acc={j_acc:.4f} F1={j_f1:.4f} AUC={j_auc:.4f} Sens={j_sens:.4f} Spec={j_spec:.4f}")
    
    slice_results.append({"acc": s_acc, "f1": s_f1, "auc": s_auc, "sens": s_sens, "spec": s_spec})
    subject_results.append({"acc": j_acc, "f1": j_f1, "auc": j_auc, "sens": j_sens, "spec": j_spec})
    
    del model; torch.cuda.empty_cache()

# Summary
print(f"\n{'='*70}")
print("  FINAL COMPARISON (10-fold CV)")
print(f"{'='*70}")

df_s = pd.DataFrame(slice_results)
df_j = pd.DataFrame(subject_results)

print(f"\n{'Metric':<8} {'Slice-Level':>16} {'Subject-Vote':>16} {'Diff':>10}")
print("-" * 52)
for col in ["acc", "f1", "auc", "sens", "spec"]:
    sm, ss = df_s[col].mean(), df_s[col].std()
    jm, js = df_j[col].mean(), df_j[col].std()
    print(f"{col:<8} {sm:.4f}+/-{ss:.4f}  {jm:.4f}+/-{js:.4f}  {jm-sm:+.4f}")

# Save
df_s.to_csv("outputs/tables/attmsnet_slice_level.csv", index=False)
df_j.to_csv("outputs/tables/attmsnet_subject_voting.csv", index=False)
print("\nSaved: outputs/tables/attmsnet_slice_level.csv")
print("Saved: outputs/tables/attmsnet_subject_voting.csv")
print("DONE")
