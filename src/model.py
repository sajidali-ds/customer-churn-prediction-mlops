"""ANN model definition, kept separate from training loop so it can be
unit-tested and reused (e.g. for hyperparameter search) independently."""
from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_dim: int, config: dict) -> keras.Model:
    mcfg = config["model"]
    model = keras.Sequential(name="churn_ann")
    model.add(layers.Input(shape=(input_dim,)))

    for units in mcfg["hidden_layers"]:
        model.add(layers.Dense(units, activation=mcfg["activation"]))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(mcfg["dropout_rate"]))

    model.add(layers.Dense(1, activation=mcfg["output_activation"]))

    optimizer = keras.optimizers.Adam(learning_rate=mcfg["learning_rate"])
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model
