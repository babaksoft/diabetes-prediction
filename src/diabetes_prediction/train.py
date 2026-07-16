import logging
from datetime import datetime

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from diabetes_prediction.config import settings
from diabetes_prediction.config.logging import configure_logging
from diabetes_prediction.pipeline import build_pipeline
from diabetes_prediction.utils.common import get_data
from diabetes_prediction.utils.model import save_model

logger = logging.getLogger(__name__)


def train():
    x_train, y_train = get_data("train")
    x_val, y_val = get_data("validation")
    x = pd.concat([x_train, x_val], axis=0)
    y = pd.concat([y_train, y_val], axis=0)

    logger.info("Prepared train data for final model (train+val).")

    hgb_clf = HistGradientBoostingClassifier(
        class_weight="balanced",
        max_depth=5,
        max_features=0.8,
        max_iter=100,
        validation_fraction=0.15,
        random_state=settings.RANDOM_STATE,
    )
    transformer = build_pipeline()
    pipeline = Pipeline([("transformer", transformer), ("estimator", hgb_clf)])

    logger.info("Model: %s", type(hgb_clf).__name__)
    logger.info("Parameters: %s", str(hgb_clf.get_params()))

    start = datetime.now()
    pipeline.fit(x, y)
    elapsed = datetime.now() - start

    logger.info("Model training completed at %.3f second(s).", elapsed.total_seconds())

    save_model(pipeline, x, "HGBClassifier")


if __name__ == "__main__":
    configure_logging()
    train()
