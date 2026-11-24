"""Chronological, one-step-ahead regression experiments.

All lag features use *previous observed rows*. The evaluation assumes the
previous day's observation becomes available before the next prediction.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score


def prepare_series(frame: pd.DataFrame, date_col: str, target_col: str) -> pd.DataFrame:
    if date_col not in frame or target_col not in frame:
        raise ValueError(f"Expected columns: {date_col!r}, {target_col!r}")
    data = frame[[date_col, target_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data[target_col] = pd.to_numeric(data[target_col], errors="coerce")
    data = data.dropna().sort_values(date_col).drop_duplicates(date_col, keep="last")
    if len(data) < 45:
        raise ValueError("At least 45 valid dated observations are required")
    data = data.reset_index(drop=True)
    data["month"] = data[date_col].dt.month
    data["day_of_week"] = data[date_col].dt.dayofweek
    data["lag_1"] = data[target_col].shift(1)
    data["lag_2"] = data[target_col].shift(2)
    data["lag_7"] = data[target_col].shift(7)
    data["lag_diff_1"] = data["lag_1"] - data["lag_2"]
    return data.dropna().reset_index(drop=True)


FEATURES = ["month", "day_of_week", "lag_1", "lag_2", "lag_7", "lag_diff_1"]


def evaluate(data: pd.DataFrame, target_col: str, test_fraction: float = .2):
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between zero and one")
    cutoff = int(len(data) * (1 - test_fraction))
    train, test = data.iloc[:cutoff], data.iloc[cutoff:]
    if len(train) < 30 or len(test) < 2:
        raise ValueError("Not enough training/test rows")
    x_train, y_train = train[FEATURES], train[target_col]
    x_test, y_test = test[FEATURES], test[target_col]
    models = {
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    }
    cv = TimeSeriesSplit(n_splits=3)
    rows, predictions = [], {"Previous observation": test["lag_1"].to_numpy()}
    for name, estimator in models.items():
        cv_mae = -cross_val_score(estimator, x_train, y_train, cv=cv, scoring="neg_mean_absolute_error").mean()
        model = clone(estimator).fit(x_train, y_train)
        predictions[name] = model.predict(x_test)
        rows.append({"model": name, "cv_mae": cv_mae})
    rows.append({"model": "Previous observation", "cv_mae": np.nan})
    results = pd.DataFrame(rows)
    for i, row in results.iterrows():
        pred = predictions[row["model"]]
        results.loc[i, "test_mae"] = mean_absolute_error(y_test, pred)
        results.loc[i, "test_rmse"] = np.sqrt(mean_squared_error(y_test, pred))
        results.loc[i, "test_r2"] = r2_score(y_test, pred)
    best_model = results.dropna(subset=["cv_mae"]).sort_values("cv_mae").iloc[0]["model"]
    return results, predictions, test, best_model
