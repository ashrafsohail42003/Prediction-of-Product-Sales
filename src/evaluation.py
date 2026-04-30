from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(y_true, y_pred),
    }


def evaluate_regression_model(
    name: str,
    model: Any,
    X_train: Any,
    y_train: Any,
    X_test: Any,
    y_test: Any,
) -> pd.DataFrame:
    rows = []
    for split_name, X, y in (("Train", X_train, y_train), ("Test", X_test, y_test)):
        preds = model.predict(X)
        rows.append({"model": name, "split": split_name, **regression_metrics(y, preds)})
    return pd.DataFrame(rows)
