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

print("=" * 70)
print("  Swin ML Baselines — Full Metrics (AUC, Sens, Spec)")
print("=" * 70)

# Load features
data = np.load("outputs/swin_features/all_multiscale_features.npz")
all_feat = np.concatenate([data["stage2"], data["stage3"], data["stage4"], data["pooler"]], axis=1)
all_labels = data["labels"]
print(f"Features: {all_feat.shape}, Labels: {all_labels.shape}")

full_df = pd.read_csv("data/splits/full_data.csv")
with open("data/splits/cv_folds.json") as f:
    folds = json.load(f)

subject_ids = full_df["subject_id"].values
results = []

for fi, fold in enumerate(folds):
    set_seed(42 + fi)
    train_mask = np.array([sid in set(fold["train_subjects"]) for sid in subject_ids])
    val_mask = np.array([sid in set(fold["val_subjects"]) for sid in subject_ids])
    test_mask = np.array([sid in set(fold["test_subjects"]) for sid in subject_ids])
    
    X_train = all_feat[train_mask | val_mask]
    y_train = all_labels[train_mask | val_mask]
    X_test = all_feat[test_mask]
    y_test = all_labels[test_mask]
    
    print(f"\n--- Fold {fi+1}/10 --- Train:{len(X_train)} Test:{len(X_test)}")
    
    row = {"fold": fi+1}
    
    for name, clf in [
        ("rf", RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)),
        ("xgb", XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, 
                              use_label_encoder=False, eval_metric="logloss", n_jobs=-1)),
    ]:
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)
        
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        auc = roc_auc_score(y_test, probs[:, 1])
        rpt = classification_report(y_test, preds, target_names=CLASS_NAMES, output_dict=True)
        sens = rpt["Demented"]["recall"]
        spec = rpt["Non-Demented"]["recall"]
        
        row[f"{name}_acc"] = acc
        row[f"{name}_f1"] = f1
        row[f"{name}_auc"] = auc
        row[f"{name}_sens"] = sens
        row[f"{name}_spec"] = spec
        print(f"  {name}: Acc={acc:.4f} F1={f1:.4f} AUC={auc:.4f} Sens={sens:.4f} Spec={spec:.4f}")
    
    # Stacking
    base = [
        ("rf", RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)),
        ("xgb", XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42,
                              use_label_encoder=False, eval_metric="logloss", n_jobs=-1)),
    ]
    stack = StackingClassifier(estimators=base, final_estimator=LogisticRegression(max_iter=1000), cv=3, n_jobs=-1)
    stack.fit(X_train, y_train)
    preds = stack.predict(X_test)
    probs = stack.predict_proba(X_test)
    
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    auc = roc_auc_score(y_test, probs[:, 1])
    rpt = classification_report(y_test, preds, target_names=CLASS_NAMES, output_dict=True)
    sens = rpt["Demented"]["recall"]
    spec = rpt["Non-Demented"]["recall"]
    
    row["stacking_acc"] = acc
    row["stacking_f1"] = f1
    row["stacking_auc"] = auc
    row["stacking_sens"] = sens
    row["stacking_spec"] = spec
    print(f"  stacking: Acc={acc:.4f} F1={f1:.4f} AUC={auc:.4f} Sens={sens:.4f} Spec={spec:.4f}")
    
    results.append(row)

df = pd.DataFrame(results)
df.to_csv("outputs/tables/step11c_ml_full_metrics.csv", index=False)

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
for model in ["rf", "xgb", "stacking"]:
    print(f"\n{model.upper()}:")
    for metric in ["acc", "f1", "auc", "sens", "spec"]:
        col = f"{model}_{metric}"
        print(f"  {metric:5s}: {df[col].mean():.4f} +/- {df[col].std():.4f}")

print(f"\nSaved: outputs/tables/step11c_ml_full_metrics.csv")
print("DONE")
