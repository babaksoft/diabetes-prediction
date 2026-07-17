import logging
from typing import Any

import joblib
import mlflow
import pandas as pd
from mlflow.models import infer_signature

from diabetes_prediction.config import settings

logger = logging.getLogger(__name__)


def save_model(model: Any, x: pd.DataFrame, model_name: str = "HGBClassifier") -> None:
    if settings.MLFLOW_TRACKING:
        logger.info("Registering model: model_name=%s", model_name)

        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        signature = infer_signature(x, model.predict(x))
        mlflow.sklearn.log_model(
            sk_model=model,
            registered_model_name=model_name,
            name="model",
            signature=signature,
        )

        logger.info("Registered model in MLflow Registry.")

    settings.MODEL_DIR.mkdir(exist_ok=True)
    local_path = settings.MODEL_DIR / "hgb_pipeline.joblib"
    joblib.dump(model, local_path)

    logger.info("Saved model to local path: %s", local_path)


def load_model(model_name: str = "HGBClassifier", stage: str = "None") -> Any:
    logger.info("Loading model: model_name=%s stage=%s", model_name, stage)

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

    mlflow_path = settings.MODEL_DIR / "mlflow"
    mlflow_path.mkdir(exist_ok=True)

    model_uri = f"models:/{model_name}/{stage}"
    model = mlflow.sklearn.load_model(model_uri=model_uri, dst_path=mlflow_path)

    logger.info(
        "Loaded '%s' from MLflow Registry: model_uri=%s",
        model_name,
        model_uri,
    )

    return model


def load_local_model() -> Any:
    try:
        path = settings.MODEL_DIR / "hgb_pipeline.joblib"
        model = joblib.load(path)

        logger.info("Loaded model from: %s", str(path))

        return model
    except FileNotFoundError:
        logger.exception("Model not found. Please run ``train.py`` first.")

        raise
