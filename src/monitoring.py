"""
Lightweight production monitoring: logs every prediction the app makes,
and detects data drift by comparing the distribution of incoming requests
against the distribution the model was trained on (a Kolmogorov-Smirnov
test per numeric feature). No external monitoring service required —
everything is stored as flat files under artifacts/, so it works the same
locally or in a container.
"""
import csv
import datetime
import json
import os

import numpy as np
from scipy.stats import ks_2samp

NUMERIC_FEATURES = [
    "CreditScore", "Age", "Tenure", "Balance",
    "NumOfProducts", "EstimatedSalary",
]

REFERENCE_STATS_PATH = "artifacts/reference_stats.json"
PREDICTION_LOG_PATH = "artifacts/prediction_log.csv"


def save_reference_stats(X_train, config: dict) -> None:
    """Called once at training time. Stores a sample of the training
    feature distribution so later predictions can be compared against it."""
    stats = {}
    for col in NUMERIC_FEATURES:
        if col in X_train.columns:
            # Keep a compact sample (max 2000 points) rather than the full
            # training set, to keep the reference file small.
            values = X_train[col].sample(min(2000, len(X_train)), random_state=42).tolist()
            stats[col] = values
    with open(REFERENCE_STATS_PATH, "w") as f:
        json.dump(stats, f)


def log_prediction(customer: dict, churn_probability: float) -> None:
    """Appends one row per prediction: timestamp, raw inputs, output."""
    os.makedirs(os.path.dirname(PREDICTION_LOG_PATH), exist_ok=True)
    file_exists = os.path.exists(PREDICTION_LOG_PATH)

    row = dict(customer)
    row["churn_probability"] = churn_probability
    row["timestamp"] = datetime.datetime.now().isoformat()

    with open(PREDICTION_LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def check_drift(p_value_threshold: float = 0.05) -> dict:
    """Compares logged prediction inputs against the training reference
    distribution for each numeric feature using a two-sample KS test.

    A low p-value (< threshold) means the incoming data likely comes from a
    different distribution than training data -- a signal the model may
    need retraining. Returns per-feature results plus an overall flag.
    """
    if not os.path.exists(REFERENCE_STATS_PATH):
        return {"error": "No reference_stats.json found. Run training first."}
    if not os.path.exists(PREDICTION_LOG_PATH):
        return {"error": "No predictions logged yet."}

    with open(REFERENCE_STATS_PATH) as f:
        reference = json.load(f)

    import pandas as pd
    logs = pd.read_csv(PREDICTION_LOG_PATH)

    results = {}
    any_drift = False
    for col, ref_values in reference.items():
        if col not in logs.columns or len(logs) < 10:
            continue
        stat, p_value = ks_2samp(ref_values, logs[col].dropna())
        drifted = bool(p_value < p_value_threshold)
        any_drift = any_drift or drifted
        results[col] = {"ks_statistic": float(stat), "p_value": float(p_value), "drift_detected": drifted}

    return {"n_predictions_checked": len(logs), "any_drift_detected": any_drift, "features": results}
