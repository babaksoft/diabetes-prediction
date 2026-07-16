import logging
from pathlib import Path

# Global config
RANDOM_STATE = 147
PROJECT_NAME = "diabetes-prediction"
MLFLOW_TRACKING = False  # Only used in reproducible training demo
MLFLOW_TRACKING_URI = "http://localhost:5000/"
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
LOG_LEVEL = logging.INFO

# Path config
DATA_DIR = PACKAGE_ROOT / "data"
MODEL_DIR = PACKAGE_ROOT / "model"
METRICS_DIR = PACKAGE_ROOT / "metrics"

# Data ingestion config
RAW_FILE = "diabetes_prediction.csv"
TRAIN_FILE = "train.csv"
VAL_FILE = "validation.csv"
TEST_FILE = "test.csv"
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
TEST_VAL_SPLIT = 0.5  # Used during test/val split

# Preprocessing pipeline config
TARGET = "diabetes"
NUMERICAL_FEATURES = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
CATEGORICAL_FEATURES = ["gender", "smoking_history"]
BINARY_FEATURES = ["hypertension", "heart_disease"]

# Model selection config
MIN_RECALL = 0.96
MIN_PRECISION = 0.25
MIN_F1_SCORE = 0.68

# Inference config (first data point in test set)
TRIAGE_THRESHOLD = 0.3631
BALANCED_THRESHOLD = 0.8871
TEST_INSTANCE = {
    "gender": "Female",
    "age": 29.0,
    "hypertension": 0,
    "heart_disease": 0,
    "smoking_history": "never",
    "bmi": 24.89,
    "HbA1c_level": 5.8,
    "blood_glucose_level": 130,
}
