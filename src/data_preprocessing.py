"""
Data loading and preprocessing.

Design goal: the exact same transformation code path is used at training
time and at inference time (train.py and predict.py / app.py both call
`transform_features`). In the original project the Streamlit app re-implemented
encoding by hand, which is a common source of train/serve skew bugs.
"""
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def split_features_target(df: pd.DataFrame, config: dict):
    df = df.drop(columns=config["data"]["drop_cols"], errors="ignore")
    y = df[config["data"]["target_col"]]
    X = df.drop(columns=[config["data"]["target_col"]])
    return X, y


def fit_encoders(X: pd.DataFrame, config: dict):
    """Fit a LabelEncoder for Gender and a OneHotEncoder for Geography.

    Returns fitted encoders; does not mutate X.
    """
    gender_encoder = LabelEncoder()
    gender_encoder.fit(X["Gender"])

    geo_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    geo_encoder.fit(X[["Geography"]])

    return gender_encoder, geo_encoder


def transform_features(X: pd.DataFrame, gender_encoder: LabelEncoder,
                        geo_encoder: OneHotEncoder, feature_columns=None) -> pd.DataFrame:
    """Apply fitted encoders to raw feature dataframe and return a numeric
    dataframe with consistent column order (feature_columns), ready for
    scaling. Works for a single-row inference dataframe or a full batch.
    """
    X = X.copy()
    X["Gender"] = gender_encoder.transform(X["Gender"])

    geo_encoded = geo_encoder.transform(X[["Geography"]])
    geo_cols = geo_encoder.get_feature_names_out(["Geography"])
    geo_df = pd.DataFrame(geo_encoded, columns=geo_cols, index=X.index)

    X = X.drop(columns=["Geography"])
    X = pd.concat([X, geo_df], axis=1)

    if feature_columns is not None:
        # Guarantees identical column order/set at train and inference time.
        for col in feature_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_columns]

    return X


def train_val_test_split(X, y, config: dict):
    test_size = config["data"]["test_size"]
    val_size = config["data"]["val_size"]
    random_state = config["data"]["random_state"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), random_state=random_state, stratify=y
    )
    relative_val = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - relative_val), random_state=random_state, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_artifacts(gender_encoder, geo_encoder, scaler, feature_columns, config: dict):
    joblib.dump(gender_encoder, config["artifacts"]["gender_encoder_path"])
    joblib.dump(geo_encoder, config["artifacts"]["geo_encoder_path"])
    joblib.dump(scaler, config["artifacts"]["scaler_path"])
    joblib.dump(list(feature_columns), config["artifacts"]["feature_columns_path"])


def load_artifacts(config: dict):
    gender_encoder = joblib.load(config["artifacts"]["gender_encoder_path"])
    geo_encoder = joblib.load(config["artifacts"]["geo_encoder_path"])
    scaler = joblib.load(config["artifacts"]["scaler_path"])
    feature_columns = joblib.load(config["artifacts"]["feature_columns_path"])
    return gender_encoder, geo_encoder, scaler, feature_columns
