import pandas as pd
import pytest

from diabetes_prediction.config import settings
from diabetes_prediction.predict import local_predict


@pytest.fixture
def triage_prediction():
    """Use triage operation mode to make inference for a single record"""

    ser_data = pd.Series(settings.TEST_INSTANCE)
    df_data = pd.DataFrame([ser_data])
    result = local_predict(x_batch=df_data, mode="triage")
    return result["predictions"][0]


@pytest.fixture
def balanced_prediction():
    """Use balanced operation mode to make inference for a single record"""

    ser_data = pd.Series(settings.TEST_INSTANCE)
    df_data = pd.DataFrame([ser_data])
    result = local_predict(x_batch=df_data, mode="balanced")
    return result["predictions"][0]


def test_triage_prediction_not_none(triage_prediction):
    """Prediction for test instance in triage mode must not be None."""

    assert triage_prediction is not None


def test_triage_prediction_dtype(triage_prediction):
    """Prediction for test instance in triage mode must be a string value."""

    assert isinstance(triage_prediction, str)


def test_triage_prediction_output(triage_prediction):
    """Prediction for test instance in triage mode must be 'Negative'."""

    assert triage_prediction == "Negative"


def test_balanced_prediction_not_none(balanced_prediction):
    """Prediction for test instance in balanced mode must not be None."""

    assert balanced_prediction is not None


def test_balanced_prediction_dtype(balanced_prediction):
    """Prediction for test instance in balanced mode must be a string value."""

    assert isinstance(balanced_prediction, str)


def test_balanced_prediction_output(balanced_prediction):
    """Prediction for test instance in balanced mode must be 'Negative'."""

    assert balanced_prediction == "Negative"
