import logging
import os

import mlflow
import pandas as pd

from diabetes_prediction.config import settings
from diabetes_prediction.config.logging import configure_logging
from diabetes_prediction.utils.ingest import (
    fix_duplicates,
    fix_label_conflicts,
    split_data,
    track_data_splits,
    track_duplicates,
    track_label_conflicts,
)

logger = logging.getLogger(__name__)


def ingest(mlflow_tracking: bool = settings.MLFLOW_TRACKING) -> None:
    raw_path = settings.DATA_DIR / "raw" / settings.RAW_FILE
    if not raw_path.exists():
        raise FileNotFoundError(
            "Raw dataset not found. Please run ``dvc pull`` and try again."
        )

    to_dir = settings.DATA_DIR / "prepared"
    to_dir.mkdir(exist_ok=True)
    os.environ["DP_INGEST_DIR"] = str(to_dir)

    if (
        os.path.exists(to_dir / settings.TRAIN_FILE)
        or os.path.exists(to_dir / settings.VAL_FILE)
        or os.path.exists(to_dir / settings.TEST_FILE)
    ):
        logger.info("Data is already ingested.")
        return

    df = pd.read_csv(raw_path)

    if mlflow_tracking:
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment("Data Ingestion")

        df = track_duplicates(df)
        df = track_label_conflicts(df)
        track_data_splits(df)
    else:
        df, _ = fix_duplicates(df)
        df, _ = fix_label_conflicts(df)
        _ = split_data(df)

    logger.info("Data ingestion completed.")


if __name__ == "__main__":
    configure_logging()
    ingest()
