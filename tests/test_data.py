import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import clean


@pytest.fixture
def raw_df():
    return pd.DataFrame(
        {
            "age": [63, 67, 67, 63],
            "sex": [1, 1, 1, 1],
            "cp": [1, 4, 4, 1],
            "trestbps": [145, 160, 120, 145],
            "chol": [233, 286, 229, 233],
            "fbs": [1, 0, 0, 1],
            "restecg": [2, 2, 2, 2],
            "thalach": [150, 108, 129, 150],
            "exang": [0, 1, 1, 0],
            "oldpeak": [2.3, 1.5, 2.6, 2.3],
            "slope": [3, 2, 2, 3],
            "ca": [0.0, 3.0, np.nan, 0.0],
            "thal": [6.0, 3.0, 7.0, 6.0],
            "target": [0, 2, 4, 0],
        }
    )


def test_clean_removes_duplicates(raw_df):
    assert len(clean(raw_df)) == 3


def test_clean_target_is_binary(raw_df):
    assert set(clean(raw_df)[config.TARGET].unique()) <= {0, 1}
    assert clean(raw_df)[config.TARGET].tolist() == [0, 1, 1]


def test_clean_fills_missing_values(raw_df):
    assert clean(raw_df).isna().sum().sum() == 0


def test_clean_does_not_mutate_input(raw_df):
    clean(raw_df)
    assert raw_df["ca"].isna().sum() == 1
