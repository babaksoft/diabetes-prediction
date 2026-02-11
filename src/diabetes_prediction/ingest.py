import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow

from .config import config
from .utils import fix_label_noise


def fix_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    with mlflow.start_run(run_name="Drop duplicates") as run:
        mlflow.set_tag("run_id", run.info.run_id)
        metrics = {
            "dataset_size": len(df),
            "duplicate_count": df.duplicated().sum()
        }
        mlflow.log_metrics(metrics)
        mlflow.end_run()

        return df.drop_duplicates()


def fix_label_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    with mlflow.start_run(run_name="Fix label noise") as run:
        mlflow.set_tag("run_id", run.info.run_id)
        df_clean = fix_label_noise(df, config.TARGET)
        metrics = {
            "dataset_size": len(df),
            "noisy_label_count": len(df) - len(df_clean)
        }
        mlflow.log_metrics(metrics)
        mlflow.end_run()

        return df_clean


def ingest(raw_path, to_dir):
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Data Ingestion (final)")
    mlflow.set_experiment_tag(
        "reason",
        "First ingestion experiment didn't take care of label noise."
    )
    df = pd.read_csv(raw_path)
    df = fix_duplicates(df)
    df = fix_label_conflicts(df)

    rs = config.RANDOM_STATE
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
        mlflow.end_run()


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
