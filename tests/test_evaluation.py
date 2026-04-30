"""Tests for src.evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation import evaluate_regression_model, regression_metrics


def test_regression_metrics_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    metrics = regression_metrics(y, y)
    assert metrics["MAE"] == 0
    assert metrics["MSE"] == 0
    assert metrics["RMSE"] == 0
    assert metrics["R2"] == 1.0


def test_regression_metrics_keys():
    y = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.1, 1.9, 3.2])
    metrics = regression_metrics(y, pred)
    assert set(metrics.keys()) == {"MAE", "MSE", "RMSE", "R2"}
    assert metrics["RMSE"] == np.sqrt(metrics["MSE"])


def test_evaluate_regression_model_shape(synthetic_sales_df):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

    df = synthetic_sales_df.dropna()
    X = df[["Item_MRP", "Item_Weight", "Item_Visibility"]].to_numpy()
    y = df["Item_Outlet_Sales"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    model = LinearRegression().fit(X_train, y_train)
    df_metrics = evaluate_regression_model("LR", model, X_train, y_train, X_test, y_test)

    assert isinstance(df_metrics, pd.DataFrame)
    assert set(df_metrics["split"]) == {"Train", "Test"}
    assert {"MAE", "MSE", "RMSE", "R2"}.issubset(df_metrics.columns)
