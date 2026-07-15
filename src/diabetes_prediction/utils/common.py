from datetime import datetime
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from diabetes_prediction.config import settings
from diabetes_prediction.pipeline import build_pipeline


def get_data(split_name: str = "train"):
    files = {
        "train": settings.TRAIN_FILE,
        "validation": settings.VAL_FILE,
        "test": settings.TEST_FILE,
    }
    file = "train.csv"
    name = split_name.lower()
    if name in files:
        file = files[name]

    path = settings.DATA_DIR / "prepared" / file
    if not path.exists():
        raise FileNotFoundError("Dataset not found. Please run ``ingest.py`` first.")

    df = pd.read_csv(path)
    x = df.drop(settings.TARGET, axis=1)
    y = df[settings.TARGET]
    return x, y


def evaluate_model(model, run_name, model_name=None) -> dict[str, Any]:
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("run_id", run.info.run_id)

        if not model_name:
            model_name = type(model).__name__
        x, y = get_data()
        cv = StratifiedKFold(
            n_splits=10, shuffle=True, random_state=settings.RANDOM_STATE
        )
        scoring = ["recall", "precision", "f1"]

        transform = build_pipeline()
        pipeline = Pipeline([("transformer", transform), ("estimator", model)])

        start = datetime.now()
        cv_results = cross_validate(pipeline, x, y, scoring=scoring, cv=cv, n_jobs=-1)
        end = datetime.now()

        metrics = {}
        cv_params = {
            "model_type": model_name,
            "cv_splits": 10,
            "cv_shuffle": True,
            "cv_random_state": settings.RANDOM_STATE,
            "cv_duration": str(end - start),
        }

        for metric in scoring:
            metrics[f"cv_{metric}_mean"] = round(cv_results[f"test_{metric}"].mean(), 4)
            metrics[f"cv_{metric}_std"] = round(cv_results[f"test_{metric}"].std(), 4)

        mlflow.log_metrics(metrics)
        mlflow.log_params(cv_params)
        mlflow.end_run()
        return metrics


def predict_with_threshold(model, x, threshold):
    probabilities = model.predict_proba(x)[:, 1]
    return np.array(probabilities >= threshold, dtype=np.int8)
