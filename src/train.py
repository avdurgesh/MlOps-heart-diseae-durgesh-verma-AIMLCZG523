"""Train, tune and track classifiers with MLflow; save the best pipeline."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import joblib  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)

from src import config  # noqa: E402
from src.data import load_clean  # noqa: E402
from src.features import FEATURES, build_pipeline  # noqa: E402

EXPERIMENT = "heart-disease-classification"
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
SCORING = ["accuracy", "precision", "recall", "roc_auc"]

CANDIDATES = {
    "logistic_regression": (
        LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE),
        {"model__C": [0.01, 0.1, 1, 10]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=config.RANDOM_STATE),
        {
            "model__n_estimators": [100, 300],
            "model__max_depth": [3, 5, None],
            "model__min_samples_leaf": [1, 3],
        },
    ),
}


def _log_plots(pipe, X_test, y_test, name: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=ax)
    ax.set_title(f"ROC curve - {name}")
    mlflow.log_figure(fig, "roc_curve.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test, ax=ax)
    ax.set_title(f"Confusion matrix - {name}")
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close(fig)


def train() -> dict:
    df = load_clean()
    X, y = df[FEATURES].astype(float), df[config.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=config.RANDOM_STATE
    )

    mlflow.set_tracking_uri((config.ROOT / "mlruns").as_uri())
    mlflow.set_experiment(EXPERIMENT)

    results, best = {}, None
    for name, (model, grid) in CANDIDATES.items():
        with mlflow.start_run(run_name=name):
            search = GridSearchCV(
                build_pipeline(model), grid, cv=CV, scoring="roc_auc", n_jobs=-1
            )
            search.fit(X_train, y_train)
            pipe = search.best_estimator_

            cv_scores = cross_validate(pipe, X_train, y_train, cv=CV, scoring=SCORING)
            cv_metrics = {f"cv_{m}": float(cv_scores[f"test_{m}"].mean()) for m in SCORING}

            pred = pipe.predict(X_test)
            proba = pipe.predict_proba(X_test)[:, 1]
            test_metrics = {
                "test_accuracy": accuracy_score(y_test, pred),
                "test_precision": precision_score(y_test, pred),
                "test_recall": recall_score(y_test, pred),
                "test_roc_auc": roc_auc_score(y_test, proba),
            }

            mlflow.log_params({k.replace("model__", ""): v for k, v in search.best_params_.items()})
            mlflow.log_param("model_type", name)
            mlflow.log_param("n_features", len(FEATURES))
            mlflow.log_metrics({**cv_metrics, **test_metrics})
            _log_plots(pipe, X_test, y_test, name)
            mlflow.sklearn.log_model(pipe, artifact_path="model", input_example=X_train.head(3))

            results[name] = {**cv_metrics, **test_metrics, "params": search.best_params_}
            print(f"{name}: " + ", ".join(f"{k}={v:.3f}" for k, v in {**cv_metrics, **test_metrics}.items()))
            if best is None or cv_metrics["cv_roc_auc"] > results[best]["cv_roc_auc"]:
                best = name
                best_pipe = pipe

    config.MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best_pipe, config.MODEL_PATH)
    summary = {"best_model": best, "features": FEATURES, "results": results}
    (config.MODEL_DIR / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"Best model: {best} -> saved to {config.MODEL_PATH}")
    return summary


if __name__ == "__main__":
    train()
