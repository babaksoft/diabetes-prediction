import os
from pathlib import Path

import numpy as np
import pandas as pd

from .config import config
from .utils import predict_with_threshold, load_model


def make_prediction(input_data):
    pipeline = load_model()
    data = pd.DataFrame(input_data)

    prediction = predict_with_threshold(
        pipeline, data, config.TRIAGE_THRESHOLD
    )
    output = np.where(prediction==1, "Yes", "No").tolist()
    results = {"prediction": output}
    return results


def predict(data_path):
    df_test = pd.read_csv(data_path)
    input_data = df_test.drop(config.TARGET, axis=1).iloc[:1]
    print(input_data)
    return make_prediction(input_data)


def main():
    data_path = Path(config.DATA_PATH) / "prepared" / config.TEST_FILE
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            "Test dataset not found. Please run ingest.py before predicting."
        )

    prediction = predict(data_path)
    print(prediction)


if __name__ == '__main__':
    main()
