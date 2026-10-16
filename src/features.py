"""Preprocessing + model pipeline construction (shared by training and serving)."""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config

FEATURES = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), config.NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_FEATURES),
        ]
    )


def build_pipeline(model) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])
