from __future__ import annotations
from typing import Any
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from .config import DROP_FEATURES, RANDOM_STATE, TARGET, TEST_SIZE
from .preprocessing import make_preprocessor
def split_features_target(
    df: pd.DataFrame,
    *,
    target: str = TARGET,
    drop_features: tuple[str, ...] = DROP_FEATURES,
) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target, *drop_features])
    y = df[target]
    return X, y

def make_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def build_linear_regression_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(scale_numeric=True)),
            ("model", LinearRegression()),
        ]
    )


def build_random_forest_pipeline(**model_params: Any) -> Pipeline:
    params: dict[str, Any] = {"random_state": RANDOM_STATE, "n_jobs": 1}
    params.update(model_params)
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(scale_numeric=False)),
            ("model", RandomForestRegressor(**params)),
        ]
    )


def tune_random_forest(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> GridSearchCV:
    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [5, None],
        "model__min_samples_leaf": [1, 3],
    }
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="r2",
        cv=3,
        n_jobs=1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    return search
