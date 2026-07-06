"""
Compare the ANN against classical ML classification algorithms on the exact
same preprocessed train/test split, using the exact same metrics.

Run with:  python -m src.compare_models

Must be run AFTER src/train.py, since it reuses the fitted encoders/scaler
saved by training (so every model sees identical features) and reads the
ANN's own test metrics from artifacts/metrics.json to include it fairly in
the same table.
"""
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.data_preprocessing import (
    load_artifacts,
    load_raw_data,
    split_features_target,
    train_val_test_split,
    transform_features,
)
from src.evaluate import compute_metrics
from src.utils import ensure_dir, load_config

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=15),
    "Naive Bayes": GaussianNB(),
    "SVM (RBF kernel)": SVC(probability=True, random_state=42),
}

try:
    from xgboost import XGBClassifier

    MODELS["XGBoost"] = XGBClassifier(
        eval_metric="logloss", random_state=42, n_estimators=300
    )
except ImportError:
    print("xgboost not installed, skipping (pip install xgboost to include it).")


def get_same_split_as_training(config: dict):
    """Reproduces the identical train/val/test split train.py used, using
    the encoders/scaler already fitted and saved by training — so classical
    ML models here see the exact same features and split as the ANN did."""
    df = load_raw_data(config["data"]["raw_path"])
    X, y = split_features_target(df, config)

    gender_encoder, geo_encoder, scaler, feature_columns = load_artifacts(config)
    X_encoded = transform_features(X, gender_encoder, geo_encoder, feature_columns)

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X_encoded, y, config
    )
    # Classical ML models don't need a separate early-stopping validation
    # set, so fold val back into train for a slightly larger training set.
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])

    X_train_scaled = scaler.transform(X_train_full)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train_full, y_test


def run_comparison():
    config = load_config("config.yaml")
    ensure_dir(config["artifacts"]["dir"])

    X_train, X_test, y_train, y_test = get_same_split_as_training(config)

    results = []

    for name, clf in MODELS.items():
        print(f"Training {name}...")
        start = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - start

        y_prob = clf.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_prob)
        metrics["model"] = name
        metrics["train_time_sec"] = round(train_time, 3)
        results.append(metrics)

    # Bring in the ANN's own test metrics computed by train.py, so it's
    # scored on the identical test set with the identical function.
    try:
        with open(config["artifacts"]["metrics_path"]) as f:
            ann_metrics = json.load(f)
        ann_metrics["model"] = "ANN (Keras)"
        ann_metrics["train_time_sec"] = None
        results.append(ann_metrics)
    except FileNotFoundError:
        print("No artifacts/metrics.json found — run `python -m src.train` first "
              "to include the ANN in the comparison.")

    df = pd.DataFrame(results).drop(columns=["confusion_matrix"], errors="ignore")
    df = df[["model", "accuracy", "precision", "recall", "f1", "roc_auc", "train_time_sec"]]
    df = df.sort_values("roc_auc", ascending=False).reset_index(drop=True)

    csv_path = f"{config['artifacts']['dir']}/model_comparison.csv"
    df.to_csv(csv_path, index=False)
    print("\n" + df.to_string(index=False))
    print(f"\nSaved comparison table to {csv_path}")

    plot_comparison(df, config)
    return df


def plot_comparison(df: pd.DataFrame, config: dict):
    metrics_to_plot = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    fig, ax = plt.subplots(figsize=(11, 6))
    df.set_index("model")[metrics_to_plot].plot(kind="bar", ax=ax)
    ax.set_title("Classification Algorithm Comparison — Customer Churn")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    png_path = f"{config['artifacts']['dir']}/model_comparison.png"
    plt.savefig(png_path, dpi=150)
    print(f"Saved comparison chart to {png_path}")


if __name__ == "__main__":
    run_comparison()
