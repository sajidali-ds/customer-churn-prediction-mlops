import pandas as pd
import pytest

from src.data_preprocessing import fit_encoders, transform_features
from src.model import build_model
from src.utils import load_config


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "CreditScore": [600, 700, 650],
            "Geography": ["France", "Germany", "Spain"],
            "Gender": ["Male", "Female", "Male"],
            "Age": [40, 35, 50],
            "Tenure": [3, 5, 1],
            "Balance": [60000.0, 0.0, 120000.0],
            "NumOfProducts": [2, 1, 3],
            "HasCrCard": [1, 0, 1],
            "IsActiveMember": [1, 1, 0],
            "EstimatedSalary": [50000.0, 60000.0, 70000.0],
        }
    )


def test_encoders_fit_and_transform(sample_df):
    config = {"data": {"drop_cols": []}}
    gender_encoder, geo_encoder = fit_encoders(sample_df, config)
    transformed = transform_features(sample_df, gender_encoder, geo_encoder)

    # Geography should be one-hot expanded, Gender should be numeric.
    assert "Geography" not in transformed.columns
    assert transformed["Gender"].dtype != object
    assert any(col.startswith("Geography_") for col in transformed.columns)
    assert len(transformed) == len(sample_df)


def test_transform_respects_feature_column_order(sample_df):
    config = {"data": {"drop_cols": []}}
    gender_encoder, geo_encoder = fit_encoders(sample_df, config)
    full = transform_features(sample_df, gender_encoder, geo_encoder)
    feature_columns = full.columns.tolist()

    single_row = sample_df.iloc[[0]]
    transformed_single = transform_features(
        single_row, gender_encoder, geo_encoder, feature_columns
    )
    assert transformed_single.columns.tolist() == feature_columns


def test_build_model_output_shape():
    config = load_config("config.yaml")
    model = build_model(input_dim=12, config=config)
    assert model.output_shape == (None, 1)
    assert model.layers[-1].activation.__name__ == "sigmoid"


def test_check_drift_handles_missing_files(tmp_path, monkeypatch):
    from src import monitoring

    monkeypatch.chdir(tmp_path)
    result = monitoring.check_drift()
    assert "error" in result
