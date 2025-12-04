import os
import json
from pathlib import Path

import joblib

from config import config
import utils


def save_metrics(metrics):
    if not os.path.exists(config.METRICS_PATH):
        os.mkdir(config.METRICS_PATH)
    path = Path(config.METRICS_PATH) / "test_metrics.json"
    with open(path, "w") as file:
        json.dump(metrics, file)


def evaluate(model_path, data_path):
    x_test, y_test = utils.feature_target_split(data_path)
    pipeline = joblib.load(model_path)
    metrics = utils.evaluate_model(
        pipeline, x_test, y_test, "test"
    )
    save_metrics(metrics)


def main():
    data_path = Path(config.DATA_PATH) / "prepared" / config.TEST_FILE
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            "Test dataset not found. Please run ingest.py before evaluating."
        )
    model_path = Path(config.MODEL_PATH) / "model.joblib"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Trained model not found. Please run train.py before evaluating."
        )

    evaluate(model_path, data_path)


if __name__ == "__main__":
    main()
