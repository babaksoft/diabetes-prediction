import os
from pathlib import Path
from datetime import datetime
from typing import Any

import pandas as pd
import mlflow
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from ..config import config
from ..pipeline import build_pipeline


def eval_baseline(data_path) -> dict[str, Any]:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Baseline Model")
    with mlflow.start_run(run_name="CV Result") as run:
        mlflow.set_tag("run_id", run.info.run_id)

        df = pd.read_csv(data_path)
        x_train = df.drop(config.TARGET, axis=1)
        y_train = df[config.TARGET]
        cv = StratifiedKFold(
            n_splits=10, shuffle=True,
            random_state=config.RANDOM_STATE
        )
        scoring = ["recall", "precision", "f1"]

        transform = build_pipeline()
        pipeline = Pipeline([
            ("transformer", transform),
            ("estimator", GaussianNB())
        ])

        start = datetime.now()
        cv_results = cross_validate(
            pipeline, x_train, y_train,
            scoring=scoring, cv=cv, n_jobs=-1
        )
        end = datetime.now()

        metrics = {}
        cv_params = {
            "model_type": "GaussianNB",
            "cv_splits": 10,
            "shuffle": True,
            "random_state": config.RANDOM_STATE,
            "cv_duration": str(end - start)
        }

        for metric in scoring:
            metrics[f"cv_{metric}_mean"] = round(
                cv_results[f"test_{metric}"].mean(), 4)
            metrics[f"cv_{metric}_std"] = round(
                cv_results[f"test_{metric}"].std(), 4)

        mlflow.log_metrics(metrics)
        mlflow.log_params(cv_params)
        mlflow.end_run()
        return metrics


def main():
    train_path = Path(config.DATA_PATH) / "prepared" / config.TRAIN_FILE
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            "Train dataset not found. Please run ingest.py before training.")

    metrics = eval_baseline(train_path)
    print("\nBaseline model metrics - Summary :\n")
    print(f"Recall = {metrics['cv_recall_mean']} +/- {metrics['cv_recall_std']}")
    print(f"Precision = {metrics['cv_precision_mean']} +/- {metrics['cv_precision_std']}")
    print(f"F1 Score = {metrics['cv_f1_mean']} +/- {metrics['cv_f1_std']}")


if __name__ == "__main__":
    main()