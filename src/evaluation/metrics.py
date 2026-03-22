"""Evaluation metrics for Alzheimer classification."""
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)


def compute_metrics(y_true, y_pred, y_prob=None, class_names=None):
    """Compute all metrics in one call.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (N, num_classes) for AUC
        class_names: List of class names
    
    Returns:
        dict with all metrics
    """
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

    if y_prob is not None:
        try:
            results["auc_roc"] = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro"
            )
        except ValueError:
            results["auc_roc"] = None

    if class_names:
        labels = list(range(len(class_names)))
        results["classification_report"] = classification_report(
            y_true, y_pred, labels=labels, target_names=class_names,
            output_dict=True, zero_division=0
        )

    return results


def print_results(results, title="Results"):
    """Pretty print evaluation results."""
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    print(f"  Accuracy:     {results['accuracy']:.4f}")
    print(f"  F1 (Macro):   {results['f1_macro']:.4f}")
    print(f"  F1 (Weighted):{results['f1_weighted']:.4f}")
    if results.get("auc_roc"):
        print(f"  AUC-ROC:      {results['auc_roc']:.4f}")
    print(f"{'='*50}\n")
