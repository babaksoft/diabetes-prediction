from pathlib import Path


# Global config
RANDOM_STATE = 147
PROJECT_NAME = "diabetes-prediction"
MLFLOW_TRACKING_URI = "http://localhost:5000/"
PACKAGE_ROOT = Path(__file__).resolve().parent.parent

# Path config
DATA_PATH = PACKAGE_ROOT / "data"
MODEL_PATH = PACKAGE_ROOT / "model"
METRICS_PATH = PACKAGE_ROOT / "metrics"

# Data ingestion config
RAW_FILE = "diabetes_prediction.csv"
TRAIN_FILE = "train.csv"
VAL_FILE = "validation.csv"
TEST_FILE = "test.csv"
TRAIN_TEST_SPLIT = 0.2 # Used during train/test split
TEST_VAL_SPLIT = 0.5 # Used during test/val split

# Preprocessing pipeline config
TARGET = "diabetes"
NUMERICAL_FEATURES = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
CATEGORICAL_FEATURES = ["gender", "smoking_history"]
BINARY_FEATURES = ["hypertension", "heart_disease"]

# Prediction config (first data point in test set)
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
