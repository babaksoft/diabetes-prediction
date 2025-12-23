import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI
import uvicorn
import joblib

from .diabetes import Diabetes
from ..config import config


def load_model():
    path = Path(config.MODEL_PATH) / "model.joblib"
    if not os.path.exists(path):
        raise FileNotFoundError("Model file not found.")
    return joblib.load(path)


app = FastAPI()
model = load_model()

@app.get("/")
async  def index():
    info = {
        "name": "Diabetes Prediction App (v0.1)",
        "description": "Predicts the onset of diabetes. For usage hints and examples, "
                       "please consult API documentation at '/docs'."
    }
    return info


@app.post("/predict")
async def predict(diabetes: Diabetes):
    data = diabetes.as_data_point()
    prediction = model.predict(data)
    output = np.where(prediction==1, "Diabetes", "No Diabetes").tolist()
    return {"prediction": output[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
