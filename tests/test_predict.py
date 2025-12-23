import pandas as pd
import pytest

from diabetes_prediction.config import config
from diabetes_prediction.predict import make_prediction


@pytest.fixture
def single_prediction():
    """ This function will predict the result for a single record"""
    single_test = pd.Series(config.TEST_INSTANCE)
    result = make_prediction([single_test])
    return result

def test_single_prediction_not_none(single_prediction):
    """ This function will check if result of prediction is not None"""
    assert single_prediction is not None

def test_single_prediction_dtype(single_prediction):
    """ This function will check if data type of result of prediction is str i.e. string """
    assert isinstance(single_prediction.get("prediction")[0], str)

def test_single_prediction_output(single_prediction):
    """ This function will check if result of prediction is No """
    # Correct prediction for the first test data point is "No diabetes"
    assert single_prediction.get("prediction")[0] == "No"
