import json
import os
import tarfile
from pathlib import Path

# As matplotlib may NOT be included in processing container, install if necessary
try:
    import matplotlib.pyplot as plt
except ImportError:
    import subprocess
    import sys

    print("Installing dependencies...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "matplotlib==3.10.9"]
    )
    import matplotlib.pyplot as plt

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
)


def predict_with_threshold(model, x, threshold):
    probabilities = model.predict_proba(x)[:, 1]
    return np.array(probabilities > threshold, dtype=np.int8)


def get_metrics(model, x, y, threshold):
    y_predict = predict_with_threshold(model, x, threshold)
    precision, recall, fscore, _ = precision_recall_fscore_support(
        y, y_predict, average="binary"
    )
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(fscore, 4),
        "threshold": round(threshold, 4),
    }


# Plot confusion matrix using a default matplotlib colormap
def plot_confusion_matrix(model, x, y, thresholds_, out_dir: Path, mode="triage"):
    cmap = "summer"
    threshold = thresholds_[mode]
    y_predict = predict_with_threshold(model, x, threshold)
    estimator = model.named_steps["estimator"]
    cm = confusion_matrix(y, y_predict, labels=estimator.classes_, normalize=None)
    cm_display = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=estimator.classes_
    )

    cm_display.plot(cmap=cmap)
    path = out_dir / f"confusion_matrix_{mode}.png"
    plt.savefig(path)
    plt.close()


# Plot ROC curve for given model and train/val/test data
def plot_roc_curve(model, x, y, out_dir: Path):
    y_scores = model.predict_proba(x)[:, 1]
    roc_display = RocCurveDisplay.from_predictions(y, y_scores)

    roc_display.plot()
    path = out_dir / "roc_curve.png"
    plt.savefig(path)
    plt.close()


# Plot Precision-Recall (PR) curve
def plot_pr_curve(model, x, y, out_dir: Path):
    y_scores = model.predict_proba(x)[:, 1]
    precision, recall, _ = precision_recall_curve(y, y_scores)
    pr_display = PrecisionRecallDisplay(precision=precision, recall=recall)

    pr_display.plot()
    path = out_dir / "pr_curve.png"
    plt.savefig(path)
    plt.close()


def get_model_artifacts():
    model_dir = Path("/opt/ml/processing/model")
    extract_dir = model_dir / "extracted"
    os.makedirs(extract_dir, exist_ok=True)

    print("Extracting model from path :", model_dir / "model.tar.gz")
    with tarfile.open(model_dir / "model.tar.gz") as tar:
        tar.extractall(path=extract_dir)
    print("Extracted model contents :", os.listdir(extract_dir))

    print("Loading model...")
    model = joblib.load(extract_dir / "model.joblib")

    with open(extract_dir / "thresholds.json") as file:
        thresholds_ = json.load(file)
    with open(extract_dir / "metadata.json") as file:
        metadata_ = json.load(file)
    return model, thresholds_, metadata_


def get_test_data(metadata_):
    target = metadata_["target"]
    test_path = os.path.join("/opt/ml/processing/test", "test.csv")
    df = pd.read_csv(test_path)
    x = df.drop(target, axis=1)
    y = df[target]
    return x, y


def evaluate_model(model, thresholds_, metadata_, x, y):
    print("Saving model metrics...")
    eval_path = Path("/opt/ml/processing/evaluation")
    metrics = {
        "triage": get_metrics(model, x, y, thresholds_["triage"]),
        "balanced": get_metrics(model, x, y, thresholds_["balanced"]),
        "model_version": metadata_["model_version"],
    }
    with open(eval_path / "evaluation.json", "w") as file:
        json.dump(metrics, file)

    print("Saving model performance plots...")
    plot_confusion_matrix(model, x, y, thresholds_, eval_path, mode="triage")
    plot_confusion_matrix(model, x, y, thresholds_, eval_path, mode="balanced")
    plot_roc_curve(model, x, y, eval_path)
    plot_pr_curve(model, x, y, eval_path)


if __name__ == "__main__":
    pipeline, thresholds, metadata = get_model_artifacts()
    x_test, y_test = get_test_data(metadata)
    evaluate_model(pipeline, thresholds, metadata, x_test, y_test)
