import argparse
from datetime import datetime
import joblib
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 147
NUM_FEATURES = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
CAT_FEATURES = ["gender", "smoking_history"]
BIN_FEATURES = ["hypertension", "heart_disease"]
TARGET = "diabetes"
TRIAGE_THRESHOLD = 0.3631
BALANCED_THRESHOLD = 0.8871


def get_args() -> argparse.Namespace:
    print("Extracting arguments...")
    parser = argparse.ArgumentParser()

    # Add model parameters
    parser.add_argument("--class-weight", type=str, default="balanced")
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--max-features", type=float, default=0.8)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)

    # Add data and model directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--val", type=str, default=os.environ.get("SM_CHANNEL_VAL"))
    parser.add_argument("--train-file", type=str, default="train.csv")
    parser.add_argument("--val-file", type=str, default="validation.csv")

    args_ = parser.parse_args()
    return args_


def get_pipeline() -> ColumnTransformer:
    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")
    )
    num_pipeline = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler())

    # In both binary features, most frequent value is 0
    bin_transform = SimpleImputer(strategy="constant", fill_value=np.int64(0.0))

    return ColumnTransformer(
        [
            ("categorical", cat_pipeline, CAT_FEATURES),
            ("numerical", num_pipeline, NUM_FEATURES),
            ("binary", bin_transform, BIN_FEATURES),
        ]
    )


def save_artifacts(args_, features_, target):
    print("Saving artifacts...")
    thresholds = {
        "triage": TRIAGE_THRESHOLD,
        "balanced": BALANCED_THRESHOLD,
    }
    with open(os.path.join(args_.model_dir, "thresholds.json"), "w") as file:
        json.dump(thresholds, file)

    metadata = {
        "model_type": "HistGradientBoostingClassifier",
        "model_version": "v1",
        "training_timestamp": datetime.now().isoformat(),
        "trained_on": "train+val",
        "input_features": NUM_FEATURES + CAT_FEATURES + BIN_FEATURES,
        "transformed_features": features_,
        "target": target,
        "class_weight": args_.class_weight,
        "max_depth": args_.max_depth,
        "max_features": args_.max_features,
        "max_iter": args_.max_iter,
        "validation_fraction": args_.validation_fraction,
        "random_state": args_.random_state,
        "data_version": "v1",
        "framework": "scikit-learn",
        "framework_version": "1.4-2",
    }
    with open(os.path.join(args_.model_dir, "metadata.json"), "w") as file:
        json.dump(metadata, file)


if __name__ == "__main__":
    args = get_args()
    transformer = get_pipeline()

    print("Reading data...")
    train_path = str(os.path.join(args.train, args.train_file))
    val_path = str(os.path.join(args.val, args.val_file))
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    print("Building train dataset...")
    x_train = df_train.drop(TARGET, axis=1)
    x_val = df_val.drop(TARGET, axis=1)
    y_train = df_train[TARGET]
    y_val = df_val[TARGET]
    x = pd.concat([x_train, x_val], axis=0)
    y = pd.concat([y_train, y_val], axis=0)

    print("Training model...")
    hgb_clf = HistGradientBoostingClassifier(
        class_weight=args.class_weight,
        max_depth=args.max_depth,
        max_features=args.max_features,
        max_iter=args.max_iter,
        validation_fraction=args.validation_fraction,
        random_state=args.random_state,
    )
    model = Pipeline([("transformer", transformer), ("estimator", hgb_clf)])
    model.fit(x, y)

    features = model.named_steps["transformer"].get_feature_names_out()
    features = [str(f) for f in features]
    save_artifacts(args, features, TARGET)
    path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, path)
    print("Model saved to :", path)
