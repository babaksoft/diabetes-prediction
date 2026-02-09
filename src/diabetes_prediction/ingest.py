import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow

from .config import config


def ingest(raw_path, to_dir):
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Data Ingestion")
    with mlflow.start_run(run_name="Drop duplicates") as run:
        mlflow.set_tag("run_id", run.info.run_id)
        df = pd.read_csv(raw_path)
        metrics = {
            "dataset_size": len(df),
            "duplicate_count": df.duplicated().sum()
        }
        mlflow.log_metrics(metrics)

    rs = config.RANDOM_STATE
    df = df.drop_duplicates()
    metrics = {
        "dataset_size": len(df),
        "train_test_split": config.TRAIN_TEST_SPLIT,
        "test_val_split": config.TEST_VAL_SPLIT,
        "random_state": rs
    }

    with mlflow.start_run(run_name="Split Dataset") as run:
        mlflow.set_tag("run_id", run.info.run_id)
        df_train, df_test = train_test_split(
            df, test_size=config.TRAIN_TEST_SPLIT,
            stratify=df[config.TARGET], random_state=rs
        )
        df_test, df_val = train_test_split(
            df_test, test_size=config.TEST_VAL_SPLIT,
            stratify=df_test[config.TARGET], random_state=rs)

        metrics["train_size"] = len(df_train)
        metrics["val_size"] = len(df_val)
        metrics["test_size"] = len(df_test)

        df_train.to_csv(to_dir / config.TRAIN_FILE, header=True, index=False)
        df_val.to_csv(to_dir / config.VAL_FILE, header=True, index=False)
        df_test.to_csv(to_dir / config.TEST_FILE, header=True, index=False)
        mlflow.log_metrics(metrics)


def main():
    raw_path = Path(config.DATA_PATH) / "raw" / config.RAW_FILE
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            "Raw dataset not found. You may need to reinstall this package."
        )

    to_dir = Path(config.DATA_PATH) / "prepared"
    if not os.path.exists(to_dir):
        os.mkdir(to_dir)
    if os.path.exists(to_dir / config.TRAIN_FILE) or \
        os.path.exists(to_dir / config.VAL_FILE) or \
        os.path.exists(to_dir / config.TEST_FILE):
        print("[INFO] Dataset is already ingested.")
        return

    ingest(raw_path, to_dir)
    print("[INFO] Raw dataset successfully ingested.")


if __name__ == '__main__':
    main()
