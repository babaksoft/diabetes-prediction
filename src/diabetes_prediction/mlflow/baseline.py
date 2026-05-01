from typing import Any

import mlflow
from sklearn.naive_bayes import GaussianNB

from diabetes_prediction.config import config
from diabetes_prediction.utils import evaluate_model


def evaluate_baseline() -> dict[str, Any]:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Baseline Model")
    model = GaussianNB()
    return evaluate_model(model=model, run_name="main")


def main():
    metrics = evaluate_baseline()
    print("\nBaseline model metrics - Summary :\n")
    print(f"Recall = {metrics['cv_recall_mean']} ± {metrics['cv_recall_std']}")
    print(f"Precision = {metrics['cv_precision_mean']} ± {metrics['cv_precision_std']}")
    print(f"F1 Score = {metrics['cv_f1_mean']} ± {metrics['cv_f1_std']}")


if __name__ == "__main__":
    main()
