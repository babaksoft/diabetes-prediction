import mlflow
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from ..config import config
from ..pipeline import build_pipeline
from ..utils import get_data, get_metrics


def get_model():
    # Copied from MLflow experiment
    params = {
        "max_depth": 5,
        "max_features": 0.8,
        "max_iter": 100,
        "validation_fraction": 0.15,
        "class_weight": "balanced",
        "random_state": config.RANDOM_STATE,
    }
    model = HistGradientBoostingClassifier().set_params(**params)
    transformer = build_pipeline()
    pipeline = Pipeline([("transformer", transformer), ("estimator", model)])
    return pipeline


def evaluate():
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name="Final Evaluation")
    with mlflow.start_run(run_name="main") as run:
        mlflow.set_tag("run_id", run.info.run_id)
        mlflow.sklearn.autolog()
        metrics = {}

        x_train, y_train = get_data("train")
        x_val, y_val = get_data("validation")
        x = pd.concat([x_train, x_val], axis=0)
        y = pd.concat([y_train, y_val], axis=0)
        pipeline = get_model()
        pipeline.fit(x, y)

        x_test, y_test = get_data("test")
        metrics["triage_threshold"] = config.TRIAGE_THRESHOLD
        metrics.update(
            get_metrics(
                pipeline, x_test, y_test, config.TRIAGE_THRESHOLD, prefix="triage"
            )
        )

        metrics["balanced_threshold"] = config.BALANCED_THRESHOLD
        metrics.update(
            get_metrics(
                pipeline, x_test, y_test, config.BALANCED_THRESHOLD, prefix="balanced"
            )
        )

        mlflow.log_metrics(metrics)
        mlflow.end_run()


if __name__ == "__main__":
    evaluate()
    print("[INFO] Final evaluation results were successfully logged to MLflow.")
