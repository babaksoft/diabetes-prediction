import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from .config import config
from .pipeline import build_pipeline
from .utils import get_data, save_model


def train():
    x_train, y_train = get_data("train")
    x_val, y_val = get_data("validation")
    x = pd.concat([x_train, x_val], axis=0)
    y = pd.concat([y_train, y_val], axis=0)
    hgb_clf = HistGradientBoostingClassifier(
        class_weight="balanced",
        max_depth=5,
        max_features=0.8,
        max_iter=100,
        validation_fraction=0.15,
        random_state=config.RANDOM_STATE
    )
    transformer = build_pipeline()
    pipeline = Pipeline([
        ("transformer", transformer),
        ("estimator", hgb_clf)
    ])
    pipeline.fit(x, y)
    save_model(pipeline, x, "HGBClassifier")


if __name__ == "__main__":
    train()
    print("[INFO] Model successfully registered in MLflow Model Registry.")
