import os
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from config import config


def save_metrics(metrics):
    path = Path(config.METRICS_PATH) / "test_metrics.json"
    with open(path, "w") as file:
        json.dump(metrics, file)


def evaluate(model_path, data_path):
    data = pd.read_csv(data_path)
    x_test = data.drop(config.TARGET, axis=1)
    y_test = data[config.TARGET]
    pipeline = joblib.load(model_path)

    y_predict = pipeline.predict(x_test)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_test, y_predict, average="binary"
    )
    metrics_ = {
        "Precision (test)": round(precision, 4),
        "Recall (test)": round(recall, 4),
        "F1 score (test)": round(f1_score, 4)
    }
    save_metrics(metrics_)


def main():
    data_path_ = Path(config.DATA_PATH) / "prepared" / config.TEST_FILE
    if not os.path.exists(data_path_):
        raise FileNotFoundError(
            "Test dataset not found. Please run ingest.py before evaluating."
        )
    model_path_ = Path(config.MODEL_PATH) / "model.joblib"
    if not os.path.exists(model_path_):
        raise FileNotFoundError(
            "Trained model not found. Please run train.py before evaluating."
        )

    evaluate(model_path_, data_path_)


if "__main__" == __name__:
    main()
