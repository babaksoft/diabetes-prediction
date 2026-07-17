import mlflow
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from diabetes_prediction.config import settings
from diabetes_prediction.pipeline import build_pipeline
from diabetes_prediction.utils.common import load_data


def analyze_thresholds_triage(name, model, x, y):
    # Model evaluation on validation set
    y_scores = model.predict_proba(x)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y, y_scores)

    # Based on our business requirements for triage model :
    # Recall >= 0.96, Precision >= 0.25
    valid_idx = np.where(
        (recalls >= settings.MIN_RECALL) & (precisions >= settings.MIN_PRECISION)
    )[
        0
    ]  # This returns a one-element tuple of indices

    best_idx = -1
    if len(valid_idx) > 0:
        triage_supported = True
        best_idx = valid_idx[-1]
    else:
        triage_supported = False

    recall = round(recalls[best_idx], 4)
    precision = round(precisions[best_idx], 4)
    threshold = round(thresholds[best_idx], 4)

    print(f"\n{name} scores (adjusted) :")
    print(f"Recall = {recall}")
    print(f"Precision = {precision}")
    print(f"Threshold = {threshold}")

    return precision, recall, threshold, triage_supported


def analyze_thresholds_balanced(name, model, x, y):
    # Model evaluation on validation set
    y_scores = model.predict_proba(x)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y, y_scores)

    # Discard last precision/recall item that doesn't have a threshold
    precisions = precisions[:-1]
    recalls = recalls[:-1]

    f1 = np.where(
        (precisions + recalls) == 0,
        0,
        2 * precisions * recalls / (precisions + recalls),
    )
    max_f1_idx = f1.argmax()

    max_f1 = round(f1[max_f1_idx], 4)
    threshold = round(thresholds[max_f1_idx], 4)

    print(f"\n{name} model :")
    print(f"Max F1 = {max_f1}")
    print(f"Threshold = {threshold}")

    return max_f1, threshold


def get_baseline_model():
    transform = build_pipeline()
    pipeline = Pipeline([("transformer", transform), ("estimator", GaussianNB())])

    x, y = load_data("train")
    pipeline.fit(x, y)
    return pipeline


def get_boosting_model():
    # Copied from MLflow experiment
    params = {
        "max_depth": 5,
        "max_features": 0.8,
        "max_iter": 100,
        "validation_fraction": 0.15,
        "class_weight": "balanced",
        "random_state": settings.RANDOM_STATE,
    }
    model = HistGradientBoostingClassifier().set_params(**params)
    transform = build_pipeline()
    pipeline = Pipeline([("transform", transform), ("estimator", model)])

    x, y = load_data("train")
    pipeline.fit(x, y)
    return pipeline


def get_logistic_model():
    # Copied from MLflow experiment
    params = {
        "l1_ratio": 0.2,
        "solver": "saga",
        "C": 5.0,
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": settings.RANDOM_STATE,
    }
    model = LogisticRegression().set_params(**params)
    transform = build_pipeline()
    pipeline = Pipeline([("transform", transform), ("estimator", model)])

    x, y = load_data("train")
    pipeline.fit(x, y)
    return pipeline


def get_triage_params(name, alias, pipeline, x, y):
    precision, recall, threshold, supported = analyze_thresholds_triage(
        name, pipeline, x, y
    )
    if supported:
        params = {
            f"{alias}_type": type(pipeline.named_steps["estimator"]).__name__,
            f"{alias}_recall": recall,
            f"{alias}_precision": precision,
            f"{alias}_threshold": threshold,
        }
    else:
        params = {
            f"{alias}_type": type(pipeline.named_steps["estimator"]).__name__,
            f"{alias}_triage_supported": supported,
        }

    return params


def log_triage_run(x, y):
    with mlflow.start_run(run_name="Triage") as run:
        mlflow.set_tag("run_id", run.info.run_id)
        params = {}

        model = get_baseline_model()
        params.update(get_triage_params("GNB", "baseline", model, x, y))

        model = get_boosting_model()
        params.update(get_triage_params("Hist-GB", "model1", model, x, y))

        model = get_logistic_model()
        params.update(get_triage_params("LR", "model2", model, x, y))

        mlflow.log_params(params)
        mlflow.end_run()


def log_balanced_run(x, y):
    with mlflow.start_run(run_name="Balanced") as run:
        mlflow.set_tag("run_id", run.info.run_id)
        params = {}

        model = get_baseline_model()
        f1, threshold = analyze_thresholds_balanced("GNB", model, x, y)
        params.update(
            {
                "baseline_type": type(model.named_steps["estimator"]).__name__,
                "baseline_f1": f1,
                "baseline_threshold": threshold,
            }
        )

        model = get_boosting_model()
        f1, threshold = analyze_thresholds_balanced("Hist-GB", model, x, y)
        params.update(
            {
                "model1_type": type(model.named_steps["estimator"]).__name__,
                "model1_f1": f1,
                "model1_threshold": threshold,
            }
        )

        model = get_logistic_model()
        f1, threshold = analyze_thresholds_balanced("LR", model, x, y)
        params.update(
            {
                "model2_type": type(model.named_steps["estimator"]).__name__,
                "model2_f1": f1,
                "model2_threshold": threshold,
            }
        )

        mlflow.log_params(params)
        mlflow.end_run()


def analyze_thresholds():
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name="Model Selection")
    x_val, y_val = load_data("validation")
    log_triage_run(x_val, y_val)
    log_balanced_run(x_val, y_val)


if __name__ == "__main__":
    analyze_thresholds()
