"""
Single-customer inference, reusing the exact preprocessing pipeline from
training. Can be run standalone:  python -m src.predict
"""
import pandas as pd
from tensorflow import keras

from src.data_preprocessing import load_artifacts, transform_features
from src.monitoring import log_prediction
from src.utils import load_config


class ChurnPredictor:
    """Loads the trained model + preprocessing artifacts once, then serves
    predictions cheaply. Use one instance per process (e.g. one per
    Streamlit session) rather than reloading files on every call."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.model = keras.models.load_model(self.config["artifacts"]["model_path"])
        (
            self.gender_encoder,
            self.geo_encoder,
            self.scaler,
            self.feature_columns,
        ) = load_artifacts(self.config)

    def predict(self, customer: dict) -> dict:
        """customer: dict with raw feature keys, e.g.
        {"CreditScore": 600, "Geography": "France", "Gender": "Male",
         "Age": 40, "Tenure": 3, "Balance": 60000, "NumOfProducts": 2,
         "HasCrCard": 1, "IsActiveMember": 1, "EstimatedSalary": 50000}
        """
        df = pd.DataFrame([customer])
        X = transform_features(df, self.gender_encoder, self.geo_encoder, self.feature_columns)
        X_scaled = self.scaler.transform(X)
        probability = float(self.model.predict(X_scaled, verbose=0)[0][0])
        log_prediction(customer, probability)
        return {
            "churn_probability": probability,
            "will_churn": probability >= 0.5,
        }


if __name__ == "__main__":
    sample_customer = {
        "CreditScore": 600,
        "Geography": "France",
        "Gender": "Male",
        "Age": 40,
        "Tenure": 3,
        "Balance": 60000,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 50000,
    }
    predictor = ChurnPredictor()
    result = predictor.predict(sample_customer)
    print(result)
