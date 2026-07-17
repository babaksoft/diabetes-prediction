import json
import logging
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import shap.plots as plots
from shap import Explanation, TreeExplainer
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


def get_shap_values(pipeline: Pipeline, x: pd.DataFrame) -> Explanation:
    transformer = pipeline.named_steps["transformer"]
    feature_names = transformer.get_feature_names_out()
    x_trans = pd.DataFrame(
        transformer.transform(x),
        columns=feature_names,
        index=x.index,
    )

    rename = {
        "categorical__gender_Female": "Gender: Female",
        "categorical__gender_Male": "Gender: Male",
        "categorical__gender_Other": "Gender: Other",
        "categorical__smoking_history_No Info": "Smoking: No Info",
        "categorical__smoking_history_current": "Smoking: Current",
        "categorical__smoking_history_ever": "Smoking: Ever",
        "categorical__smoking_history_former": "Smoking: Former",
        "categorical__smoking_history_never": "Smoking: Never",
        "categorical__smoking_history_not current": "Smoking: Not Current",
        "numerical__age": "Age",
        "numerical__bmi": "BMI",
        "numerical__HbA1c_level": "HbA1c",
        "numerical__blood_glucose_level": "Blood Glucose",
        "binary__hypertension": "Hypertension",
        "binary__heart_disease": "Heart Disease",
    }
    x_trans.rename(columns=rename, inplace=True)
    explainer = TreeExplainer(pipeline.named_steps["estimator"])

    return explainer(x_trans)


def save_shap_plots(pipeline: Pipeline, x: pd.DataFrame):
    shap_values = get_shap_values(pipeline, x)

    # Summary / Feature importance plot
    path = settings.METRICS_DIR / "shap_feature_importance.png"

    plt.figure(figsize=(8, 6))
    plots.bar(shap_values, show=False)
    plt.tight_layout()

    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("SHAP feature importance plot saved to: %s", str(path))

    # Beeswarm plot
    path = settings.METRICS_DIR / "shap_beeswarm.png"

    plt.figure(figsize=(10, 8))
    plots.beeswarm(shap_values, show=False)
    plt.tight_layout()

    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("SHAP beeswarm plot saved to: %s", str(path))

    # Waterfall plot on single data point (index=349)
    index = 349
    path = settings.METRICS_DIR / "shap_waterfall.png"

    plt.figure(figsize=(10, 6))
    plots.waterfall(shap_values[index], show=False)
    plt.tight_layout()

    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(
        "SHAP waterfall plot on test instance (index=%d) saved to: %s",
        index,
        str(path),
    )
