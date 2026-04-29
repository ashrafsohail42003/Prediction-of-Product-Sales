"""Regression evaluation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Calculate common regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_true, y_pred),
    }


def evaluate_regression_model(name: str, model, X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """Evaluate a fitted regression model on training and testing data."""
    rows = []
    for split, X, y in [
        ("Train", X_train, y_train),
        ("Test", X_test, y_test),
    ]:
        preds = model.predict(X)
        metrics = regression_metrics(y, preds)
        rows.append({"model": name, "split": split, **metrics})
    return pd.DataFrame(rows)
