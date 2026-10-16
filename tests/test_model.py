import pytest
from sklearn.linear_model import LogisticRegression

from src.data import load_clean
from src.features import FEATURES, build_pipeline, build_preprocessor
from src.predict import predict
from src import config


@pytest.fixture(scope="module")
def data():
    df = load_clean()
    return df[FEATURES].astype(float), df[config.TARGET]


def test_dataset_shape(data):
    X, y = data
    assert X.shape[1] == 13
    assert len(X) == len(y) > 250


def test_preprocessor_output_has_no_nans(data):
    X, _ = data
    out = build_preprocessor().fit_transform(X)
    assert out.shape[0] == len(X)
    assert not (out != out).any()


def test_pipeline_fits_and_predicts(data):
    X, y = data
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    assert pipe.score(X, y) > 0.8


def test_predict_returns_valid_output(data):
    X, y = data
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    result = predict(pipe, X.head(2).to_dict("records"))
    assert len(result) == 2
    for r in result:
        assert r["prediction"] in (0, 1)
        assert 0.5 <= r["confidence"] <= 1.0


def test_pipeline_handles_unseen_category(data):
    X, y = data
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    row = X.head(1).copy()
    row["cp"] = 99.0
    assert pipe.predict(row).shape == (1,)
