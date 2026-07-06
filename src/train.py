"""
End-to-end training pipeline.

Run with:  python -m src.train
"""
import datetime
import os

import numpy as np
import mlflow
import mlflow.keras
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from src.data_preprocessing import (
    fit_encoders,
    load_raw_data,
    save_artifacts,
    split_features_target,
    train_val_test_split,
    transform_features,
)
from src.evaluate import evaluate_and_save
from src.model import build_model
from src.monitoring import save_reference_stats
from src.utils import ensure_dir, load_config, save_json, set_seed


def main():
    config = load_config("config.yaml")
    set_seed(config["data"]["random_state"])
    ensure_dir(config["artifacts"]["dir"])
    ensure_dir(config["logging"]["log_dir"])

    mlflow.set_tracking_uri(config.get("mlflow", {}).get("tracking_uri", "mlruns"))
    mlflow.set_experiment(config.get("mlflow", {}).get("experiment_name", "churn-prediction"))

    with mlflow.start_run():
        mlflow.log_params({
            "hidden_layers": config["model"]["hidden_layers"],
            "dropout_rate": config["model"]["dropout_rate"],
            "learning_rate": config["model"]["learning_rate"],
            "batch_size": config["model"]["batch_size"],
            "epochs": config["model"]["epochs"],
            "class_weighting": config["model"]["class_weighting"],
        })

        print("Loading data...")
        df = load_raw_data(config["data"]["raw_path"])
        X, y = split_features_target(df, config)

        print("Fitting encoders...")
        gender_encoder, geo_encoder = fit_encoders(X, config)
        X_encoded = transform_features(X, gender_encoder, geo_encoder)
        feature_columns = X_encoded.columns.tolist()

        print("Splitting train/val/test...")
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
            X_encoded, y, config
        )

        print("Fitting scaler on training data only (avoids leakage)...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        save_artifacts(gender_encoder, geo_encoder, scaler, feature_columns, config)
        save_reference_stats(X_train, config)

        class_weight = None
        if config["model"]["class_weighting"]:
            weights = compute_class_weight(
                class_weight="balanced", classes=np.unique(y_train), y=y_train
            )
            class_weight = {i: w for i, w in enumerate(weights)}
            print(f"Using class weights: {class_weight}")

        print("Building model...")
        model = build_model(input_dim=X_train_scaled.shape[1], config=config)
        model.summary()

        log_dir = os.path.join(
            config["logging"]["log_dir"], datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config["model"]["early_stopping_patience"],
                restore_best_weights=True,
            ),
            keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
            ),
        ]

        print("Training...")
        history = model.fit(
            X_train_scaled,
            y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=config["model"]["epochs"],
            batch_size=config["model"]["batch_size"],
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=2,
        )

        model.save(config["artifacts"]["model_path"])
        save_json(history.history, config["artifacts"]["history_path"])
        print(f"Model saved to {config['artifacts']['model_path']}")
        mlflow.log_param("epochs_trained", len(history.history["loss"]))

        print("Evaluating on held-out test set...")
        metrics = evaluate_and_save(model, X_test_scaled, y_test, config)
        mlflow.log_metrics({
            "test_accuracy": metrics["accuracy"],
            "test_precision": metrics["precision"],
            "test_recall": metrics["recall"],
            "test_f1": metrics["f1"],
            "test_roc_auc": metrics["roc_auc"],
        })

        # Log + version the model in the MLflow Model Registry.
        mlflow.keras.log_model(model, artifact_path="model", registered_model_name="churn-ann")
        for path in [
            config["artifacts"]["scaler_path"],
            config["artifacts"]["gender_encoder_path"],
            config["artifacts"]["geo_encoder_path"],
            config["artifacts"]["feature_columns_path"],
        ]:
            mlflow.log_artifact(path, artifact_path="preprocessing")

        print(f"MLflow run logged. View with: mlflow ui  (run ID: {mlflow.active_run().info.run_id})")


if __name__ == "__main__":
    main()
