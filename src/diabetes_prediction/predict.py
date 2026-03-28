import os
from pathlib import Path

import numpy as np
import pandas as pd

from .config import config
from .utils import predict_with_threshold, load_model, load_local_model


def get_prediction_results(model, input_data, mode="triage"):
    data = pd.DataFrame(input_data)

    threshold = (
        config.TRIAGE_THRESHOLD if mode == "triage" else config.BALANCED_THRESHOLD
    )
    prediction = predict_with_threshold(model, data, threshold)
    output = np.where(prediction == 1, "Yes", "No").tolist()
    results = {"predictions": output}
    return results


def make_local_prediction(input_data, mode="triage"):
    pipeline = load_local_model()
    return get_prediction_results(pipeline, input_data, mode)


def make_prediction(input_data, mode="triage"):
    pipeline = load_model()
    return get_prediction_results(pipeline, input_data, mode)


def predict(data_path, mode="triage"):
    df_test = pd.read_csv(data_path)
    input_data = df_test.drop(config.TARGET, axis=1).iloc[:1]
    print(input_data)
    return make_prediction(input_data, mode)


def main():
    data_path = Path(config.DATA_PATH) / "prepared" / config.TEST_FILE
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            "Test dataset not found. Please run ingest.py before predicting."
        )

    # Mode can be read from command-line arguments (use triage by default)
    prediction = predict(data_path)
    print(prediction)


if __name__ == "__main__":
    main()
