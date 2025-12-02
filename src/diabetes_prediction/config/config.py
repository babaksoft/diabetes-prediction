from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent.parent

DATA_PATH = PACKAGE_ROOT / "data"
MODEL_PATH = PACKAGE_ROOT / "model"

RAW_FILE = "diabetes_prediction.csv"
TRAIN_FILE = "train.csv"
VALIDATION_FILE = "validation.csv"
TEST_FILE = "test.csv"

TARGET = "diabetes"

# TODO: Set names of features used for training
# Note: probably not required as there are only 9-10 features
FEATURES = []

# TODO: Set names of numerical features
NUMERICAL_FEATURES = []

# TODO: Set names of categorical features
CATEGORICAL_FEATURES = []

FEATURES_TO_ENCODE = []

# Features for log transformation
LOG_FEATURES = []
