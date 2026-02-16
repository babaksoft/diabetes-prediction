import os
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import (
    precision_recall_curve, precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from .config import config
from .pipeline import build_pipeline


def get_data(split_name: str = "train"):
    files = {
        "train": config.TRAIN_FILE,
        "validation": config.VAL_FILE,
        "test": config.TEST_FILE
    }
    file = "train.csv"
    name = split_name.lower()
    if name in files:
        file = files[name]

    path = Path(config.DATA_PATH) / "prepared" / file
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Dataset not found. Please run ingest.py first.")

    df = pd.read_csv(path)
    x = df.drop(config.TARGET, axis=1)
    y = df[config.TARGET]
    return x, y


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


def evaluate_model(model, run_name, model_name=None) -> dict[str, Any]:
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("run_id", run.info.run_id)

        if not model_name:
            model_name = type(model).__name__
        x, y = get_data()
        cv = StratifiedKFold(
            n_splits=10, shuffle=True,
            random_state=config.RANDOM_STATE)
        scoring = ["recall", "precision", "f1"]

        transform = build_pipeline()
        pipeline = Pipeline([
            ("transformer", transform),
            ("estimator", model)
        ])

        start = datetime.now()
        cv_results = cross_validate(
            pipeline, x, y, scoring=scoring, cv=cv, n_jobs=-1)
        end = datetime.now()

        metrics = {}
        cv_params = {
            "model_type": model_name,
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


def predict_with_threshold(model, x, threshold):
    probabilities = model.predict_proba(x)[:, 1]
    return np.array(probabilities > threshold, dtype=np.int8)


def get_metrics(model, x, y, threshold, prefix=None):
    prefix = f"{prefix}_" if prefix is not None else ""
    y_predict = predict_with_threshold(model, x, threshold)
    precision, recall, fscore, _ = precision_recall_fscore_support(
        y, y_predict, average="binary"
    )
    return {
        f"{prefix}precision": round(precision, 4),
        f"{prefix}recall": round(recall, 4),
        f"{prefix}f1": round(fscore, 4)
    }


def save_model(model, model_name, x):
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    signature = infer_signature(x, model.predict(x))
    mlflow.sklearn.log_model(
        sk_model=model,
        registered_model_name=model_name,
        name="model",
        signature=signature
    )

    if not os.path.exists(config.MODEL_PATH):
        os.mkdir(config.MODEL_PATH)
    path = Path(config.MODEL_PATH) / "hgb_pipeline.joblib"
    joblib.dump(model, path)


def load_model():
    path = Path(config.MODEL_PATH) / "hgb_pipeline.joblib"
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Trained model not found. Please run train.py before predicting."
        )
    pipeline = joblib.load(path)
    return pipeline


# Plot side-by-side train/validation metrics in a bar chart
def plot_metrics(train, val, metric, names):
    fig, ax = plt.subplots()
    bar_width = 0.35
    index = np.arange(len(names))

    ax.bar(index, train, bar_width, label="train")
    ax.bar(index + bar_width, val, bar_width, label="validation")

    ax.set_xlabel("Trained Model")
    ax.set_ylabel(metric)
    ax.set_title(f"Train vs. Validation {metric}")
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(names)

    ax.legend(loc="best")
    fig.tight_layout()

    path = config.METRICS_PATH / f"Train_Val_{metric}.png"
    plt.savefig(path)
    plt.close()


# Plot confusion matrix using a default matplotlib colormap
def plot_confusion_matrix(model, x, y, cmap="summer", normalize=None):
    y_predict = model.predict(x)
    cm = confusion_matrix(y, y_predict, labels=model.classes_, normalize=normalize)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

    cm_display.plot(cmap=cmap)
    path = config.METRICS_PATH / "CM.png"
    plt.savefig(path)
    plt.close()


# Plot ROC curve for given model and train/val/test data
def plot_roc_curve(model, x, y):
    roc_display = RocCurveDisplay.from_estimator(model, x, y)

    roc_display.plot()
    path = config.METRICS_PATH / "ROC.png"
    plt.savefig(path)
    plt.close()


# Plot Precision-Recall (PR) curve
def plot_pr_curve(model, x, y):
    predictions = model.predict(x)
    precision, recall, _ = precision_recall_curve(y, predictions)
    pr_display = PrecisionRecallDisplay(precision=precision, recall=recall)

    pr_display.plot()
    path = config.METRICS_PATH / "PR.png"
    plt.savefig(path)
    plt.close()


# Log model attributes in current MLFlow experiment
def mlflow_register(
        model, model_name: str, x_train, run_name: str, metrics: dict[str, float]
):
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("project", config.PROJECT_NAME)
        mlflow.set_tag("run_id", run.info.run_id)
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)

        signature = infer_signature(x_train, model.predict(x_train))
        mlflow.sklearn.log_model(model, name=model_name, signature=signature)

        mlflow.end_run()
