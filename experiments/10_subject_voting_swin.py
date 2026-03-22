"""
Subject-Level Voting for Swin+ML pipeline.
Uses existing full metrics results + raw predictions.
Retrains RF/XGB/Stacking per fold, gets probabilities, aggregates per subject.
"""
import functools, builtins
builtins.print = functools.partial(print, flush=True)

import os, sys, json, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

sys.path.insert(0, '.')
from src.utils.helpers import set_seed, load_config

cfg = load_config("configs/config.yaml")
CLASS_NAMES = cfg["data"]["class_names"]
NUM_CLASSES = cfg["data"]["num_classes"]

print("=" * 70)
print("  Swin+ML Subject-Level Voting (10-fold CV)")
print("=" * 70)

data = np.load("outputs/swin_features/all_multiscale_features.npz")
all_feat = np.concatenate([data["stage2"], data["stage3"], data["stage4"], data["pooler"]], axis=1)
all_labels = data["labels"]
print(f"Features: {all_feat.shape}")

full_df = pd.read_csv("data/splits/full_data.csv")
with open("data/splits/cv_folds.json") as f:
    folds = json.load(f)

subject_ids = full_df["subject_id"].values

def get_class_weights(labels):
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    w = 1.0 / (counts + 1e-6)
    w = w / w.sum() * NUM_CLASSES
    return w

models_config = {
    "RF": lambda: RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
    "XGB": lambda: XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42,
                                  use_label_encoder=False, eval_metric="logloss", n_jobs=-1),
}

# Results storage
slice_results = {m: [] for m in ["RF", "XGB", "Stacking"]}
subject_results = {m: [] for m in ["RF", "XGB", "Stacking"]}

for fi, fold in enumerate(folds):
    set_seed(42 + fi)
    
    train_mask = np.array([sid in set(fold["train_subjects"]) for sid in subject_ids])
    val_mask = np.array([sid in set(fold["val_subjects"]) for sid in subject_ids])
    test_mask = np.array([sid in set(fold["test_subjects"]) for sid in subject_ids])
    
    X_train = all_feat[train_mask | val_mask]
    y_train = all_labels[train_mask | val_mask]
    X_test = all_feat[test_mask]
    y_test = all_labels[test_mask]
    test_subjects = subject_ids[test_mask]
    
    print(f"\n--- Fold {fi+1}/10 --- Train:{len(X_train)} Test:{len(X_test)} ({len(set(test_subjects))} subjects)")
    
    # Train models and get probabilities
    trained = {}
    for name, make_clf in models_config.items():
        clf = make_clf()
        clf.fit(X_train, y_train)
        trained[name] = clf
    
    # Stacking
    base = [
        ("rf", RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)),
        ("xgb", XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42,
                              use_label_encoder=False, eval_metric="logloss", n_jobs=-1)),
    ]
    stack = StackingClassifier(estimators=base, final_estimator=LogisticRegression(max_iter=1000), cv=3, n_jobs=-1)
    stack.fit(X_train, y_train)
    trained["Stacking"] = stack
    
    for name, clf in trained.items():
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)
        
        # Slice-level
        s_acc = accuracy_score(y_test, preds)
        s_f1 = f1_score(y_test, preds, average="macro")
        s_auc = roc_auc_score(y_test, probs[:, 1])
        s_rpt = classification_report(y_test, preds, target_names=CLASS_NAMES, output_dict=True)
        s_sens = s_rpt["Demented"]["recall"]
        s_spec = s_rpt["Non-Demented"]["recall"]
        
        # Subject-level voting
        df_test = pd.DataFrame({
            "subject_id": test_subjects,
            "label": y_test,
            "prob_1": probs[:, 1],
        })
        subj_agg = df_test.groupby("subject_id").agg(
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
        
        print(f"  {name:10s} Slice: Acc={s_acc:.4f} AUC={s_auc:.4f} Sens={s_sens:.4f} | Subj: Acc={j_acc:.4f} AUC={j_auc:.4f} Sens={j_sens:.4f}")
        
        slice_results[name].append({"acc": s_acc, "f1": s_f1, "auc": s_auc, "sens": s_sens, "spec": s_spec})
        subject_results[name].append({"acc": j_acc, "f1": j_f1, "auc": j_auc, "sens": j_sens, "spec": j_spec})

# Summary
print(f"\n{'='*70}")
print("  SUMMARY: Slice vs Subject-Level")
print(f"{'='*70}")

for name in ["RF", "XGB", "Stacking"]:
    df_s = pd.DataFrame(slice_results[name])
    df_j = pd.DataFrame(subject_results[name])
    print(f"\n{name}:")
    print(f"  {'Metric':<8} {'Slice':>16} {'Subject':>16} {'Diff':>8}")
    for col in ["acc", "f1", "auc", "sens", "spec"]:
        sm, ss = df_s[col].mean(), df_s[col].std()
        jm, js = df_j[col].mean(), df_j[col].std()
        print(f"  {col:<8} {sm:.4f}+/-{ss:.4f}  {jm:.4f}+/-{js:.4f}  {jm-sm:+.4f}")

# Save
all_data = []
for name in ["RF", "XGB", "Stacking"]:
    df_s = pd.DataFrame(slice_results[name])
    df_j = pd.DataFrame(subject_results[name])
    for col in ["acc", "f1", "auc", "sens", "spec"]:
        all_data.append({"model": name, "metric": col,
                        "slice_mean": f"{df_s[col].mean():.4f}", "slice_std": f"{df_s[col].std():.4f}",
                        "subj_mean": f"{df_j[col].mean():.4f}", "subj_std": f"{df_j[col].std():.4f}"})
pd.DataFrame(all_data).to_csv("outputs/tables/swin_ml_subject_voting.csv", index=False)
print("\nSaved: outputs/tables/swin_ml_subject_voting.csv")
print("DONE")
