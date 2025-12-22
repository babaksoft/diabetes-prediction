from pathlib import Path


RANDOM_STATE = 147

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

# First data point in test set
TEST_INSTANCE = {
    "gender": "Male",
    "age": 24.0,
    "hypertension": 0,
    "heart_disease": 0,
    "smoking_history": "No Info",
    "bmi": 28.07,
    "HbA1c_level": 6.2,
    "blood_glucose_level": 155,
}
