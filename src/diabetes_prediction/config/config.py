from pathlib import Path


PROJECT_NAME = "diabetes-prediction"
PACKAGE_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PACKAGE_ROOT / "data"
MODEL_PATH = PACKAGE_ROOT / "model"
METRICS_PATH = PACKAGE_ROOT / "metrics"

RAW_FILE = "diabetes_prediction.csv"
TRAIN_FILE = "train.csv"
VALIDATION_FILE = "validation.csv"
TEST_FILE = "test.csv"

TARGET = "diabetes"

NUMERICAL_FEATURES = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]

CATEGORICAL_FEATURES = ["gender", "smoking_history"]

# Binary (0/1) features
BINARY_FEATURES = ["hypertension", "heart_disease"]

RANDOM_STATE = 147
