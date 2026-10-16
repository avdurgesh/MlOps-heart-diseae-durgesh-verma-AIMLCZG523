"""Load the saved pipeline and run predictions."""
import joblib
import pandas as pd

from src import config
from src.features import FEATURES


def load_model(path=config.MODEL_PATH):
    return joblib.load(path)


def predict(model, records: list[dict]) -> list[dict]:
    df = pd.DataFrame(records)[FEATURES].astype(float)
    proba = model.predict_proba(df)[:, 1]
    return [
        {"prediction": int(p >= 0.5), "confidence": float(max(p, 1 - p)), "probability_disease": float(p)}
        for p in proba
    ]


if __name__ == "__main__":
    sample = {
        "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
        "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6,
    }
    print(predict(load_model(), [sample]))
