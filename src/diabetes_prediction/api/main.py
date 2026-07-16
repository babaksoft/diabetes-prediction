from pathlib import Path

import joblib
import numpy as np
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
    "`https://github.com/babaksoft/diabetes-prediction/README.md`",
)
model = load_model()


def predict_with_threshold(x, threshold):
    probabilities = model.predict_proba(x)[:, 1]
    return np.array(probabilities > threshold, dtype=np.int8)


@app.get("/")
async def index():
    info = {
        "name": "Diabetes Risk Prediction (v0.1)",
        "description": "Predicts diabetes risk using strict business policies. "
        "For usage hints and examples, please consult API "
        "documentation at `/docs`.",
    }
    return info


@app.post("/predict")
async def predict(diabetes: Diabetes, mode: str = "triage"):
    threshold = TRIAGE_THRESHOLD if mode == "triage" else BALANCED_THRESHOLD
    data = diabetes.as_dataframe()
    prediction = predict_with_threshold(data, threshold)
    output = np.where(prediction == 1, "Positive", "Negative").tolist()
    return {"prediction": output[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
