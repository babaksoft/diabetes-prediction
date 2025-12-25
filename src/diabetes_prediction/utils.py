import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import \
    precision_recall_fscore_support, precision_recall_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from sklearn.model_selection import cross_validate
import mlflow
from mlflow.models import infer_signature

from .config import config


def make_label(label: str, label_type: str | None = None) -> str:
    return f"{label} ({label_type})" if label_type else label


def feature_target_split(csv_path):
    data = pd.read_csv(csv_path)
    x = data.drop(config.TARGET, axis=1)
    y = data[config.TARGET]
    return x,y


def evaluate_model(model, x, y, name: str | None = None):
    y_predict = model.predict(x)
    precision, recall, fscore, _ = precision_recall_fscore_support(
        y, y_predict, average="binary"
    )

    metrics = {
        make_label("Precision", name): round(precision, 4),
        make_label("Recall", name): round(recall, 4),
        make_label("F1 score", name): round(fscore, 4),
    }
    return metrics


# Train a model and return train/validation metrics
def train_and_validate(model, x_train, y_train, cv=5):
    model.fit(x_train, y_train)

    # Training metrics
    y_predict = model.predict(x_train)
    precision, recall, fscore, _ = precision_recall_fscore_support(
        y_train, y_predict, average="binary"
    )

    # Validation metrics
    scoring = ["precision", "recall", "f1"]
    scores = cross_validate(
        model, x_train, y_train, scoring=scoring, cv=cv
    )

    return pd.Series({
        "Train Recall": recall,
        "Val Recall": scores["test_recall"].mean(),
        "Train Precision": precision,
        "Val Precision": scores["test_precision"].mean(),
        "Train FScore": fscore,
        "Val FScore": scores["test_f1"].mean(),
    })


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
