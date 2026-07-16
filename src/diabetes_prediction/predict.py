import logging
from typing import Any

import pandas as pd

from diabetes_prediction.config import settings
from diabetes_prediction.config.logging import configure_logging
from diabetes_prediction.utils.common import get_data
from diabetes_prediction.utils.model import (
    load_local_model,
    load_model,
)

logger = logging.getLogger(__name__)


def get_predictions(
    model: Any,
    x_batch: pd.DataFrame,
    mode: str,
    labels: bool = True,
) -> dict[str, Any]:
    threshold = (
        settings.TRIAGE_THRESHOLD if mode == "triage" else settings.BALANCED_THRESHOLD
    )

    probabilities = model.predict_proba(x_batch)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    if labels:
        predictions = [
            "Positive" if pred == 1 else "Negative" for pred in predictions.ravel()
        ]
    else:
        predictions = predictions.ravel().tolist()
    rounded_probs = [round(float(prob), 6) for prob in probabilities.ravel()]

    return {
        "mode": mode,
        "predictions": predictions,
        "probabilities": rounded_probs,
        "threshold": threshold,
    }


def local_predict(x_batch: pd.DataFrame, mode: str) -> dict[str, Any]:
    pipeline = load_local_model()
    return get_predictions(pipeline, x_batch, mode)


def mlflow_predict(x_batch: pd.DataFrame, mode: str) -> dict[str, Any]:
    pipeline = load_model()
    return get_predictions(pipeline, x_batch, mode)


def show_predictions(trues: pd.Series, output: dict[str, Any]) -> None:
    print("\nGround truth labels :")
    print(trues)
    print("\nPredictions :")
    preds = output["predictions"]
    probs = output["probabilities"]
    result = []
    pos_count = 0
    correct_count = 0
    for pred, prob, true in zip(preds, probs, trues):
        truth = "Positive" if true == 1 else "Negative"
        prediction = "Positive" if pred == 1 else "Negative"
        checked = "✓" if prediction == truth else "✗"
        result.append(f"[{checked}] {prediction}\t(p={round(prob, 4)})")
        if pred == 1:
            pos_count += 1
        if checked == "✓":
            correct_count += 1
    print("\n".join(result))
    print("\nSummary :")
    print(f"Correct : {correct_count}/{len(preds)}")
    print(f"Predicted positives : {pos_count}")
    print("\n")


def demo_predictions(
    model: Any, x_batch_: pd.DataFrame, y_batch_: pd.Series, mode: str
) -> None:
    threshold = (
        settings.TRIAGE_THRESHOLD if mode == "triage" else settings.BALANCED_THRESHOLD
    )

    print(f"\nBatch inference (mode={mode}, threshold={threshold})")
    print("---------------------------------------------------\n")
    print("Input batch :\n", x_batch_)
    output = get_predictions(model, x_batch_, mode, labels=False)
    show_predictions(y_batch_, output)


def predict() -> None:
    x, y = get_data("test")
    x_batch = x.iloc[345:350, :]
    y_batch = y.iloc[345:350]

    pipeline = load_model() if settings.MLFLOW_TRACKING else load_local_model()
    demo_predictions(pipeline, x_batch, y_batch, mode="triage")
    demo_predictions(pipeline, x_batch, y_batch, mode="balanced")
    print("Inference demo successfully completed.")


if __name__ == "__main__":
    configure_logging()
    predict()
