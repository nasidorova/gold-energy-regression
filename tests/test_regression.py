import numpy as np
import pandas as pd
import pytest
from src.regression import prepare_series, evaluate


def example():
    return pd.DataFrame({"date": pd.date_range("2024-01-01", periods=100), "value": np.arange(100.0)})


def test_lags_do_not_use_current_target():
    prepared = prepare_series(example(), "date", "value")
    assert prepared.iloc[0]["value"] == 7
    assert prepared.iloc[0]["lag_1"] == 6
    assert prepared.iloc[0]["lag_7"] == 0


def test_bad_columns_rejected():
    with pytest.raises(ValueError):
        prepare_series(example(), "date", "missing")


def test_evaluation_baseline():
    prepared = prepare_series(example(), "date", "value")
    results, predictions, test, selected = evaluate(prepared, "value")
    assert len(results) == 4
    assert np.allclose(predictions["Previous observation"], test["lag_1"])
    assert selected in {"Gradient Boosting", "AdaBoost", "Ridge"}
