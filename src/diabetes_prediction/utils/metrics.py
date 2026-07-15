import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
)

from diabetes_prediction.config import settings
from diabetes_prediction.utils.common import predict_with_threshold


def get_metrics(model, x, y, threshold, prefix=None):
    prefix = f"{prefix}_" if prefix is not None else ""
    y_predict = predict_with_threshold(model, x, threshold)
    precision, recall, fscore, _ = precision_recall_fscore_support(
        y, y_predict, average="binary"
    )
    return {
        f"{prefix}precision": round(precision, 4),
        f"{prefix}recall": round(recall, 4),
        f"{prefix}f1": round(fscore, 4),
    }


# Plot confusion matrix using a default matplotlib colormap
def plot_confusion_matrix(model, x, y, mode="triage", normalize=None):
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


# Plot ROC curve for given model and train/val/test data
def plot_roc_curve(model, x, y):
    y_scores = model.predict_proba(x)[:, 1]
    roc_display = RocCurveDisplay.from_predictions(y, y_scores)

    roc_display.plot()
    path = settings.METRICS_DIR / "roc_curve.png"
    plt.savefig(path)
    plt.close()


# Plot Precision-Recall (PR) curve
def plot_pr_curve(model, x, y):
    y_scores = model.predict_proba(x)[:, 1]
    precision, recall, _ = precision_recall_curve(y, y_scores)
    pr_display = PrecisionRecallDisplay(precision=precision, recall=recall)

    pr_display.plot()
    path = settings.METRICS_DIR / "pr_curve.png"
    plt.savefig(path)
    plt.close()
