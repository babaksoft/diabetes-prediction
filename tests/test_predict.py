import pandas as pd
import pytest

from diabetes_prediction.config import config
from diabetes_prediction.predict import make_local_prediction


@pytest.fixture
def triage_prediction():
    """ This function will use triage operation mode to make inference
     for a single record"""
    single_test = pd.Series(config.TEST_INSTANCE)
    result = make_local_prediction(
        input_data=[single_test], mode="triage"
    )
    return result


@pytest.fixture
def balanced_prediction():
    """ This function will use balanced operation mode to make inference
     for a single record"""
    single_test = pd.Series(config.TEST_INSTANCE)
    result = make_local_prediction(
        input_data=[single_test], mode="balanced"
    )
    return result


def test_triage_prediction_not_none(triage_prediction):
    """ This function will check if result of prediction is not None"""
    assert triage_prediction is not None


def test_triage_prediction_dtype(triage_prediction):
    """ This function will check if data type of prediction result
     is str i.e. string """
    assert isinstance(triage_prediction.get("predictions")[0], str)


def test_triage_prediction_output(triage_prediction):
    """ This function will check if result of prediction is No """
    # Correct prediction for the first test data point is "No diabetes"
    assert triage_prediction.get("predictions")[0] == "No"


def test_balanced_prediction_not_none(balanced_prediction):
    """ This function will check if result of prediction is not None"""
    assert balanced_prediction is not None


def test_balanced_prediction_dtype(balanced_prediction):
    """ This function will check if data type of result of prediction is str i.e. string """
    assert isinstance(balanced_prediction.get("predictions")[0], str)


def test_balanced_prediction_output(balanced_prediction):
    """ This function will check if result of prediction is No """
    # Correct prediction for the first test data point is "No diabetes"
    assert balanced_prediction.get("predictions")[0] == "No"
