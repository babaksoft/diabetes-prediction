import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import \
    recall_score, precision_recall_fscore_support, precision_recall_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from sklearn.model_selection import cross_val_score
import mlflow

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


# Train a model and return train/validation Recall
def train_and_validate(model, x_train, y_train, cv=5):
    model.fit(x_train, y_train)

    # Training Recall
    y_predict = model.predict(x_train)
    train_recall = recall_score(y_train, y_predict)

    # Validation Recall
    val_recall = cross_val_score(
        model, x_train, y_train, scoring="recall", cv=cv
    )

    return train_recall, val_recall.mean()


# Plot side-by-side train/validation metrics in a bar chart
def plot_metrics(train, val, names):
    fig, ax = plt.subplots()
    bar_width = 0.35
    index = np.arange(len(names))

    ax.bar(index, train, bar_width, label="train")
    ax.bar(index + bar_width, val, bar_width, label="validation")

    ax.set_xlabel("Trained Model")
    ax.set_ylabel("Recall")
    ax.set_title("Train vs. Validation Recall")
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(names)

    ax.legend(loc="best")
    fig.tight_layout()

    path = config.METRICS_PATH / "Train_Val_Recall.png"
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
        model, model_name: str, run_name: str, metrics: dict[str, float]
):
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("project", config.PROJECT_NAME)
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name=model_name)

        mlflow.end_run()
