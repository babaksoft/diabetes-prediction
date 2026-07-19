from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import prometheus_client
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

from diabetes_prediction.api.diabetes import Diabetes
from diabetes_prediction.api.instrument import (
    OUTCOME_COUNTER,
    PREDICT_LATENCY_HIST,
    PREDICTION_SAMPLES_COUNTER,
    REQUEST_COUNTER,
)

TRIAGE_THRESHOLD = 0.3631
BALANCED_THRESHOLD = 0.8871


def load_model():
    curr_dir = Path(__file__).resolve().parent
    path = curr_dir / "hgb_pipeline.joblib"
    if not path.exists():
        raise FileNotFoundError("Model file not found.")

    return joblib.load(path)


@asynccontextmanager
async def lifespan(api: FastAPI):
    app.state.model = load_model()
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Diabetes Risk Prediction API",
    summary="Predicts diabetes risk using two operation modes: "
    "triage (high recall) and balanced (high precision).",
    description="For more information, consult project documentation at: "
    "`https://github.com/babaksoft/diabetes-prediction/blob/master/README.md`",
)


def get_predictions(
    input: list[Diabetes], threshold: float, mode: str
) -> dict[str, Any]:
    if input is None:
        raise ValueError("Input data cannot be None.")

    if not input:
        return {}

    model = app.state.model
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
        "For usage hints and examples, please consult API documentation at `/docs`.",
    }


@app.get("/health")
async def check_health():
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    return {
        "status": "healthy",
        "model": "loaded",
        "version": "v1",
    }


@app.post("/predict")
async def predict(input: list[Diabetes], mode: str = "triage"):
    REQUEST_COUNTER.labels(mode=mode).inc()
    PREDICTION_SAMPLES_COUNTER.labels(mode=mode).inc(len(input))

    threshold = TRIAGE_THRESHOLD if mode == "triage" else BALANCED_THRESHOLD
    with PREDICT_LATENCY_HIST.time():
        result = get_predictions(input, threshold, mode)

    positive_count = sum(pred == "Positive" for pred in result["predictions"])
    negative_count = len(result["predictions"]) - positive_count
    OUTCOME_COUNTER.labels(
        mode=mode,
        prediction="Positive",
    ).inc(positive_count)
    OUTCOME_COUNTER.labels(
        mode=mode,
        prediction="Negative",
    ).inc(negative_count)

    return result


@app.get("/metrics")
async def get_metrics():
    return Response(
        content=prometheus_client.generate_latest(),
        media_type=prometheus_client.CONTENT_TYPE_LATEST,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
