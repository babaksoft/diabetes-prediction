import os
import json
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from .config import config
from .pipeline import pipeline
from .utils import feature_target_split, evaluate_model


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


def train(data_path):
    x_train, y_train = feature_target_split(data_path)
    full_pipeline = make_pipeline(
        pipeline,
        LogisticRegression()
    )

    full_pipeline.fit(x_train, y_train)
    metrics = evaluate_model(
        full_pipeline, x_train, y_train, "train"
    )
    save_artifacts(full_pipeline, metrics)


def main():
    data_path = Path(config.DATA_PATH) / "prepared" / config.TRAIN_FILE
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            "Train dataset not found. Please run ingest.py before training."
        )

    train(data_path)


if __name__ == "__main__":
    main()
