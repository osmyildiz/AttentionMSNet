"""
SHAP analysis wrapper for feature-level model interpretation.

Computes SHAP values on extracted deep features to identify
which feature dimensions most influence classification decisions.
"""

import numpy as np
import shap
import matplotlib.pyplot as plt


def compute_shap_values(model, X_train, X_test, feature_names=None, max_samples=500):
    """Compute SHAP values for a sklearn-compatible classifier.
    
    Uses KernelExplainer for model-agnostic SHAP computation
    on extracted feature vectors.
    
    Args:
        model: Trained sklearn classifier (RF, XGB, Stacking, etc.)
        X_train: Training features for background distribution
        X_test: Test features to explain
        feature_names: Optional list of feature names
        max_samples: Max background samples (default 500)
        
    Returns:
        shap_values: SHAP values array
        explainer: SHAP explainer object
    """
    # Subsample background for efficiency
    if len(X_train) > max_samples:
        idx = np.random.choice(len(X_train), max_samples, replace=False)
        background = X_train[idx]
    else:
        background = X_train
    
    explainer = shap.KernelExplainer(model.predict_proba, background)
    shap_values = explainer.shap_values(X_test)
    
    return shap_values, explainer


def plot_shap_per_class(shap_values, X_test, class_names, top_k=15, save_path=None):
    """Plot SHAP summary for each class side by side.
    
    Shows top-k most important features for each class,
    revealing which features drive predictions in opposite
    directions for different diagnoses.
    
    Args:
        shap_values: List of SHAP value arrays (one per class)
        X_test: Test feature matrix
        class_names: List of class names
        top_k: Number of top features to display
        save_path: Optional path to save figure
    """
    n_classes = len(class_names)
    fig, axes = plt.subplots(1, n_classes, figsize=(8 * n_classes, 10))
    
    if n_classes == 1:
        axes = [axes]
    
    for i, (ax, name) in enumerate(zip(axes, class_names)):
        # Get mean absolute SHAP values per feature
        mean_abs = np.abs(shap_values[i]).mean(axis=0)
        top_indices = np.argsort(mean_abs)[-top_k:][::-1]
        
        feature_labels = [f"Feature {j}" for j in top_indices]
        
        plt.sca(ax)
        shap.summary_plot(
            shap_values[i][:, top_indices],
            X_test[:, top_indices],
            feature_names=feature_labels,
            show=False,
            plot_size=None,
        )
        ax.set_title(name, fontsize=14, fontweight='bold')
    
    plt.suptitle("SHAP Feature Importance by Class", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def get_discriminative_features(shap_values, class_names, top_k=10):
    """Identify features with opposing effects across classes.
    
    Features that push toward one class and away from another
    are the most discriminative and clinically interesting.
    
    Args:
        shap_values: List of SHAP value arrays
        class_names: List of class names  
        top_k: Number of top features to return
        
    Returns:
        discriminative: List of (feature_idx, class0_effect, class1_effect) tuples
    """
    if len(shap_values) < 2:
        return []
    
    mean_0 = shap_values[0].mean(axis=0)
    mean_1 = shap_values[1].mean(axis=0)
    
    # Discrimination score: features with large opposing effects
    disc_score = np.abs(mean_0 - mean_1)
    top_idx = np.argsort(disc_score)[-top_k:][::-1]
    
    results = []
    for idx in top_idx:
        results.append({
            'feature_idx': int(idx),
            f'{class_names[0]}_effect': float(mean_0[idx]),
            f'{class_names[1]}_effect': float(mean_1[idx]),
            'discrimination_score': float(disc_score[idx]),
        })
    
    return results
