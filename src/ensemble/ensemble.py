"""Ensemble classification: Deep features + XGBoost/LightGBM."""
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def extract_features(model, dataloader, device):
    """Extract deep features from the model's penultimate layer.
    
    Args:
        model: HybridAttentionNet with extract_features method
        dataloader: PyTorch DataLoader
        device: torch device
    
    Returns:
        features: np.ndarray (N, feature_dim)
        labels: np.ndarray (N,)
    """
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            feats = model.extract_features(images)
            all_features.append(feats.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_features), np.concatenate(all_labels)


def train_xgboost(X_train, y_train, X_val, y_val, n_estimators=300):
    """Train XGBoost on deep features."""
    clf = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.1,
        objective='multi:softprob',
        num_class=4,
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42,
    )
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return clf


def train_lightgbm(X_train, y_train, X_val, y_val, n_estimators=300):
    """Train LightGBM on deep features."""
    clf = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        num_leaves=31,
        learning_rate=0.1,
        objective='multiclass',
        num_class=4,
        random_state=42,
        verbose=-1,
    )
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    return clf


def soft_voting_ensemble(cnn_probs, xgb_probs, lgb_probs, 
                          weights=(0.4, 0.3, 0.3)):
    """Soft voting ensemble of CNN + XGBoost + LightGBM."""
    w = np.array(weights)
    combined = w[0] * cnn_probs + w[1] * xgb_probs + w[2] * lgb_probs
    return np.argmax(combined, axis=1), combined
