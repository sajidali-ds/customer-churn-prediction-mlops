"""Evaluation metrics. Accuracy alone is misleading on an imbalanced churn
dataset (~20% positive class), so this reports precision/recall/F1/ROC-AUC
and a confusion matrix as well."""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils import save_json


def compute_metrics(y_test, y_prob, y_pred=None) -> dict:
    """Shared metric computation so the ANN and every classical ML model in
    compare_models.py are scored with identical logic — otherwise a
    comparison table is not a fair comparison."""
    if y_pred is None:
        y_pred = (y_prob >= 0.5).astype(int)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def evaluate_and_save(model, X_test, y_test, config: dict) -> dict:
    y_prob = model.predict(X_test).ravel()
    metrics = compute_metrics(y_test, y_prob)

    y_pred = (y_prob >= 0.5).astype(int)
    print(classification_report(y_test, y_pred, target_names=["Stayed", "Churned"]))
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

    save_json(metrics, config["artifacts"]["metrics_path"])
    return metrics
