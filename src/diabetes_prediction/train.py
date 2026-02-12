import os
import json
from pathlib import Path

import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline

from .config import config
from .pipeline import pipeline
from .utils import get_data, evaluate_model


def save_artifacts(model, metrics):
    if not os.path.exists(config.MODEL_PATH):
        os.mkdir(config.MODEL_PATH)
    path = Path(config.MODEL_PATH) / "model.joblib"
    joblib.dump(model, path)

    if not os.path.exists(config.METRICS_PATH):
        os.mkdir(config.METRICS_PATH)
    path = Path(config.METRICS_PATH) / "metrics.json"
    with open(path, "w") as file:
        json.dump(metrics, file)


def train():
    x_train, y_train = get_data("train")
    full_pipeline = make_pipeline(
        pipeline,
        GaussianNB()
    )

    full_pipeline.fit(x_train, y_train)
    metrics = evaluate_model(
        full_pipeline, x_train, y_train, "train"
    )
    save_artifacts(full_pipeline, metrics)


if __name__ == "__main__":
    train()
