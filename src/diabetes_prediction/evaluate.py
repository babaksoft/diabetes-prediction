import os
import json
from pathlib import Path

from .config import config
from .utils import get_data, get_metrics, load_model


def save_metrics(metrics, path):
    if not os.path.exists(config.METRICS_PATH):
        os.mkdir(config.METRICS_PATH)
    with open(path, "w") as file:
        json.dump(metrics, file)


def evaluate():
    x_test, y_test = get_data("test")
    pipeline = load_model()

    path = Path(config.METRICS_PATH) / "metrics_triage.json"
    metrics = get_metrics(
        pipeline, x_test, y_test, config.TRIAGE_THRESHOLD
    )
    save_metrics(metrics, path)

    path = Path(config.METRICS_PATH) / "metrics_balanced.json"
    metrics = get_metrics(
        pipeline, x_test, y_test, config.BALANCED_THRESHOLD
    )
    save_metrics(metrics, path)


def main():
    data_path = Path(config.DATA_PATH) / "prepared" / config.TEST_FILE
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            "Test dataset not found. Please run ingest.py before evaluating."
        )

    evaluate()


if __name__ == "__main__":
    main()
