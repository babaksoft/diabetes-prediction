import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import config


def build_pipeline() -> ColumnTransformer:
    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore")
    )
    num_pipeline = make_pipeline(
        SimpleImputer(strategy="mean"),
        StandardScaler()
    )

    # In both binary features, most frequent value is 0
    bin_transform = SimpleImputer(
        strategy="constant", fill_value=np.int64(0.0)
    )

    return  ColumnTransformer([
        ("categorical", cat_pipeline, config.CATEGORICAL_FEATURES),
        ("numerical", num_pipeline, config.NUMERICAL_FEATURES),
        ("binary", bin_transform, config.BINARY_FEATURES)
    ])

pipeline = build_pipeline()
