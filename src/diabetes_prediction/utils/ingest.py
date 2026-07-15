import logging
import os
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

from diabetes_prediction.config import settings

logger = logging.getLogger(__name__)


def get_metrics(
    raw: pd.DataFrame,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, Any]:
    metrics = {
        "dataset_size": len(raw),
        "train_split": settings.TRAIN_SPLIT,
        "val_split": settings.VAL_SPLIT,
        "test_split": settings.TEST_SPLIT,
        "random_state": settings.RANDOM_STATE,
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
    }

    y_full = raw[settings.TARGET]
    y_train = train[settings.TARGET]
    y_val = val[settings.TARGET]
    y_test = test[settings.TARGET]

    pos_full = y_full.value_counts(normalize=True)[1]
    pos_train = y_train.value_counts(normalize=True)[1]
    pos_val = y_val.value_counts(normalize=True)[1]
    pos_test = y_test.value_counts(normalize=True)[1]

    metrics.update(
        {
            "positive_ratio_full": round(float(pos_full), 6),
            "positive_ratio_train": round(float(pos_train), 6),
            "positive_ratio_val": round(float(pos_val), 6),
            "positive_ratio_test": round(float(pos_test), 6),
            "positive_count_train": y_train.value_counts()[1],
            "negative_count_train": y_train.value_counts()[0],
        }
    )

    return metrics


def fix_label_noise(data: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df: pd.DataFrame = data.copy()
    features = list(df.columns.drop(target_col))

    # Step 1 : Find feature groups with >1 unique label
    # Group by all feature columns
    grouped = df.groupby(features)[target_col].nunique()

    # Groups where label count > 1
    conflicting_groups = grouped[grouped > 1]

    # Step 2 : Extract all conflicting rows
    # Identify feature combinations with conflicts
    conflict_keys = conflicting_groups.index

    # Convert multi-index to DataFrame for merging
    df_conflict = pd.DataFrame(list(conflict_keys), columns=features)

    # Merge back to original df
    df.merge(df_conflict, on=features, how="inner")

    # Step 3 : Remove conflicting rows
    # Remove all rows that belong to conflicting groups
    df_clean = df.merge(df_conflict, on=features, how="left", indicator=True)
    df_clean = df_clean[df_clean["_merge"] == "left_only"].drop(columns="_merge")

    return df_clean


def fix_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    duplicate_count = df.duplicated().sum()
    duplicate_percent = duplicate_count / len(df) * 100
    metrics = {
        "dataset_size": len(df),
        "duplicate_count": duplicate_count,
        "duplicate_percent": round(duplicate_percent, 3),
        "cleaned_size": len(df) - duplicate_count,
    }

    logger.info(
        "Removed duplicates: dataset_size=%d duplicate_count=%d "
        "duplicate_percent=%.3f cleaned_size=%d",
        metrics["dataset_size"],
        metrics["duplicate_count"],
        metrics["duplicate_percent"],
        metrics["cleaned_size"],
    )

    return df.drop_duplicates(), metrics


def track_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    with mlflow.start_run(run_name="Drop duplicates") as run:
        mlflow.set_tag("run_id", run.info.run_id)

        df_cleaned, metrics = fix_duplicates(df)

        mlflow.log_metrics(metrics)

    return df_cleaned


def fix_label_conflicts(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    df_cleaned = fix_label_noise(df, settings.TARGET)
    noisy_label_count = len(df) - len(df_cleaned)
    noisy_label_percent = noisy_label_count / len(df) * 100
    metrics = {
        "dataset_size": len(df),
        "noisy_label_count": noisy_label_count,
        "noisy_label_percent": round(noisy_label_percent, 3),
        "cleaned_size": len(df_cleaned),
    }

    logger.info(
        "Removed label noise: dataset_size=%d noisy_label_count=%d "
        "noisy_label_percent=%.3f cleaned_size=%d",
        metrics["dataset_size"],
        metrics["noisy_label_count"],
        metrics["noisy_label_percent"],
        metrics["cleaned_size"],
    )

    return df_cleaned, metrics


def track_label_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    with mlflow.start_run(run_name="Fix label noise") as run:
        mlflow.set_tag("run_id", run.info.run_id)

        df_cleaned, metrics = fix_label_conflicts(df)

        mlflow.log_metrics(metrics)

    return df_cleaned


def split_data(df: pd.DataFrame) -> dict[str, Any]:
    rs = settings.RANDOM_STATE
    df_train, df_test = train_test_split(
        df,
        train_size=settings.TRAIN_SPLIT,
        stratify=df[settings.TARGET],
        random_state=rs,
    )
    df_test, df_val = train_test_split(
        df_test,
        test_size=settings.TEST_VAL_SPLIT,
        stratify=df_test[settings.TARGET],
        random_state=rs,
    )

    metrics = get_metrics(df, df_train, df_val, df_test)
    logger.info("Performed train/validation/test split on cleaned dataset.")
    logger.info(
        "Split statistics: "
        "dataset_size=%d train_split=%d%% val_split=%d%% test_split=%d%% "
        "random_state=%d train_size=%d val_size=%d test_size=%d ",
        metrics["dataset_size"],
        int(metrics["train_split"] * 100),
        int(metrics["val_split"] * 100),
        int(metrics["test_split"] * 100),
        metrics["random_state"],
        metrics["train_size"],
        metrics["val_size"],
        metrics["test_size"],
    )

    logger.info(
        "Label statistics: "
        "positive_ratio_full=%.6f positive_ratio_train=%.6f "
        "positive_ratio_val=%.6f positive_ratio_test=%.6f "
        "positive_count_train=%d negative_count_train=%d",
        metrics["positive_ratio_full"],
        metrics["positive_ratio_train"],
        metrics["positive_ratio_val"],
        metrics["positive_ratio_test"],
        metrics["positive_count_train"],
        metrics["negative_count_train"],
    )

    to_dir = Path(os.environ["DP_INGEST_DIR"])
    df_train.to_csv(to_dir / settings.TRAIN_FILE, header=True, index=False)
    df_val.to_csv(to_dir / settings.VAL_FILE, header=True, index=False)
    df_test.to_csv(to_dir / settings.TEST_FILE, header=True, index=False)

    return metrics


def track_data_splits(df: pd.DataFrame) -> None:
    with mlflow.start_run(run_name="Split Dataset") as run:
        mlflow.set_tag("run_id", run.info.run_id)

        metrics = split_data(df)
        ingest_dir = os.environ["DP_INGEST_DIR"]

        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(local_dir=ingest_dir)
