from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI

from diabetes_prediction.api.diabetes import Diabetes

TRIAGE_THRESHOLD = 0.3631
BALANCED_THRESHOLD = 0.8871


def load_model():
    curr_dir = Path(__file__).resolve().parent
    path = curr_dir / "hgb_pipeline.joblib"
    if not path.exists():
        raise FileNotFoundError("Model file not found.")

    return joblib.load(path)


app = FastAPI(
    title="Diabetes Risk Prediction API",
    summary="Predicts diabetes risk using two operation modes: "
    "triage (high recall) and balanced (high precision).",
    description="For more information, consult project documentation at: "
    "`https://github.com/babaksoft/diabetes-prediction/blob/master/README.md`",
)
model = load_model()


def get_predictions(
    input: list[Diabetes], threshold: float, mode: str
) -> dict[str, Any]:
    if input is None:
        raise ValueError("Input data cannot be None.")

    if not input:
        return {}

    df_input = pd.DataFrame([pd.Series(item.model_dump()) for item in input])
    probabilities = model.predict_proba(df_input)[:, 1]
    predictions = np.array(probabilities >= threshold, dtype=np.int8)

    return {
        "mode": mode,
        "predictions": [
            "Positive" if pred == 1 else "Negative" for pred in predictions
        ],
        "probabilities": [round(prob, 4) for prob in probabilities],
        "threshold": threshold,
    }


@app.get("/")
async def index():
    return {
        "name": "Diabetes Risk Prediction (v0.1)",
        "description": "Predicts diabetes risk using strict business policies. "
        "For usage hints and examples, please consult API "
        "documentation at `/docs`.",
    }


@app.post("/predict")
async def predict(input: list[Diabetes], mode: str = "triage"):
    threshold = TRIAGE_THRESHOLD if mode == "triage" else BALANCED_THRESHOLD
    return get_predictions(input, threshold, mode)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
