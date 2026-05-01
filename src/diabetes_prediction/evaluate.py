import json
import os
from pathlib import Path

from diabetes_prediction.config import config
from diabetes_prediction.utils import (
    get_data,
    get_metrics,
    load_model,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
)


def save_metrics(metrics, model_type=None):
    if not os.path.exists(config.METRICS_PATH):
        os.mkdir(config.METRICS_PATH)

    suffix = f"_{model_type}" if model_type is not None else ""
    path = Path(config.METRICS_PATH) / f"metrics{suffix}.json"
    with open(path, "w") as file:
        json.dump(metrics, file)


def save_artifacts(model, x, y):
    plot_confusion_matrix(model, x, y, mode="triage")
    plot_confusion_matrix(model, x, y, mode="balanced")
    plot_roc_curve(model, x, y)
    plot_pr_curve(model, x, y)


def evaluate():
    x_test, y_test = get_data("test")
    pipeline = load_model()

    metrics = get_metrics(pipeline, x_test, y_test, config.TRIAGE_THRESHOLD)
    save_metrics(metrics, model_type="triage")

    metrics = get_metrics(pipeline, x_test, y_test, config.BALANCED_THRESHOLD)
    save_metrics(metrics, model_type="balanced")
    save_artifacts(pipeline, x_test, y_test)


def main():
    data_path = Path(config.DATA_PATH) / "prepared" / config.TEST_FILE
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            "Test dataset not found. Please run ingest.py before evaluating."
        )

    evaluate()
    print("[INFO] Metrics successfully saved in /metrics folder.")


if __name__ == "__main__":
    main()
