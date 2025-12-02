import os
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import make_pipeline

from config import config
from pipeline import pipeline


def save_artifacts(model, metrics):
    path = Path(config.MODEL_PATH) / "model.joblib"
    joblib.dump(model, path)
    path = Path(config.METRICS_PATH) / "metrics.json"
    with open(path, "w") as file:
        json.dump(metrics, file)


def train(data_path):
    data = pd.read_csv(data_path)
    x_train = data.drop(config.TARGET, axis=1)
    y_train = data[config.TARGET]
    full_pipeline = make_pipeline(
        pipeline,
        LogisticRegression()
    )

    full_pipeline.fit(x_train, y_train)
    y_predict = full_pipeline.predict(x_train)

    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_train, y_predict, average="binary"
    )
    metrics = {
        "Precision (train)": round(precision, 4),
        "Recall (train)": round(recall, 4),
        "F1 score (train)": round(f1_score, 4)
    }
    save_artifacts(full_pipeline, metrics)


def main():
    path = Path(config.DATA_PATH) / "prepared" / config.TRAIN_FILE
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Train dataset not found. Please run ingest.py before training."
        )

    train(path)


if __name__ == "__main__":
    main()
