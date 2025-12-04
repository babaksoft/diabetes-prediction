import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from config import config


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

    make_label = lambda label, label_type:\
        f"{label} ({label_type})" if label_type else label
    metrics = {
        make_label("Precision", name): round(precision, 4),
        make_label("Recall", name): round(recall, 4),
        make_label("F1 score", name): round(fscore, 4),
    }
    return metrics
