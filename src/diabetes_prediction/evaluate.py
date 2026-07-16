import logging

from diabetes_prediction.config import settings
from diabetes_prediction.config.logging import configure_logging
from diabetes_prediction.utils.common import get_data
from diabetes_prediction.utils.metrics import (
    get_metrics,
    save_artifacts,
    save_metrics,
)
from diabetes_prediction.utils.model import (
    load_local_model,
    load_model,
)

logger = logging.getLogger(__name__)


def evaluate() -> None:
    x_test, y_test = get_data("test")
    pipeline = load_model() if settings.MLFLOW_TRACKING else load_local_model()

    logger.info(
        "Evaluating model: mode=triage threshold=%.4f", settings.TRIAGE_THRESHOLD
    )

    metrics = get_metrics(pipeline, x_test, y_test, settings.TRIAGE_THRESHOLD)
    save_metrics(metrics, mode="triage")
    save_artifacts(pipeline, x_test, y_test, mode="triage")

    logger.info(
        "Evaluating model: mode=balanced threshold=%.4f", settings.BALANCED_THRESHOLD
    )

    metrics = get_metrics(pipeline, x_test, y_test, settings.BALANCED_THRESHOLD)
    save_metrics(metrics, mode="balanced")
    save_artifacts(pipeline, x_test, y_test, mode="balanced")


if __name__ == "__main__":
    configure_logging()
    evaluate()
