import json
import logging
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline

from diabetes_prediction.config import settings
from diabetes_prediction.utils.common import predict_with_threshold

logger = logging.getLogger(__name__)


def get_metrics(
    model: Any,
    x: pd.DataFrame,
    y: pd.Series,
    threshold: float,
    prefix: str | None = None,
) -> dict[str, float]:
    y_predict = predict_with_threshold(model, x, threshold)
    precision, recall, fscore, _ = precision_recall_fscore_support(
        y, y_predict, average="binary"
    )

    prefix = f"{prefix}_" if prefix is not None else ""
    metrics = {
        f"{prefix}precision": round(precision, 4),
        f"{prefix}recall": round(recall, 4),
        f"{prefix}f1": round(fscore, 4),
    }

    logger.info(
        "Calculated metrics: %s",
        " ".join(f"{key}={val}" for key, val in metrics.items()),
    )

    return metrics


def save_metrics(metrics: dict[str, float], mode: str | None = None) -> None:
    settings.METRICS_DIR.mkdir(exist_ok=True)

    suffix = f"_{mode}" if mode is not None else ""
    path = settings.METRICS_DIR / f"metrics{suffix}.json"
    with open(path, "w") as file:
        json.dump(metrics, file)

    logger.info("Model metrics (mode=%s) saved to: %s", mode, str(path))


# Plot confusion matrix using a default matplotlib colormap
def plot_confusion_matrix(
    model: Pipeline,
    x: pd.DataFrame,
    y: pd.Series,
    mode: str = "triage",
    normalize: bool | None = None,
) -> None:
    cmap = "summer"
    threshold = (
        settings.TRIAGE_THRESHOLD if mode == "triage" else settings.BALANCED_THRESHOLD
    )
    y_predict = predict_with_threshold(model, x, threshold)
    estimator = model.named_steps["estimator"]
    cm = confusion_matrix(y, y_predict, labels=estimator.classes_, normalize=normalize)
    cm_display = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=estimator.classes_
    )

    cm_display.plot(cmap=cmap)
    path = settings.METRICS_DIR / f"confusion_matrix_{mode}.png"
    plt.savefig(path)
    plt.close()

    logger.info("Confusion matrix (mode=%s) saved to: %s", mode, str(path))


# Plot ROC curve for given model and train/val/test data
def plot_roc_curve(model: Any, x: pd.DataFrame, y: pd.Series) -> None:
    y_scores = model.predict_proba(x)[:, 1]
    roc_display = RocCurveDisplay.from_predictions(y, y_scores)

    roc_display.plot()
    path = settings.METRICS_DIR / "roc_curve.png"
    plt.savefig(path)
    plt.close()

    logger.info("ROC curve saved to: %s", str(path))


# Plot Precision-Recall (PR) curve
def plot_pr_curve(model: Any, x: pd.DataFrame, y: pd.Series) -> None:
    y_scores = model.predict_proba(x)[:, 1]
    precision, recall, _ = precision_recall_curve(y, y_scores)
    pr_display = PrecisionRecallDisplay(precision=precision, recall=recall)

    pr_display.plot()
    path = settings.METRICS_DIR / "pr_curve.png"
    plt.savefig(path)
    plt.close()

    logger.info("PR curve saved to: %s", str(path))


def save_artifacts(model: Any, x: pd.DataFrame, y: pd.Series, mode: str) -> None:
    plot_confusion_matrix(model, x, y, mode)
    plot_roc_curve(model, x, y)
    plot_pr_curve(model, x, y)
