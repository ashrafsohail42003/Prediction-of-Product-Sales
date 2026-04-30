"""sklearn preprocessing factory.

A single entry point — :func:`build_preprocessor` — returns a fitted-ready
:class:`ColumnTransformer` that:

1. Median-imputes numeric columns and standardises them.
2. Mode-imputes categorical columns and one-hot encodes them.

Imputation is applied **after** ``train_test_split`` at fit time, which is the
correct ordering to avoid leaking test-set statistics into training.
"""

from __future__ import annotations

from typing import Iterable

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def _numeric_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def _categorical_pipeline() -> Pipeline:
    # sklearn >= 1.2 introduced `sparse_output`; we keep the encoder dense so
    # downstream DataFrame inspection stays ergonomic.
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )


def build_preprocessor(
    numeric_features: Iterable[str] = NUMERIC_FEATURES,
    categorical_features: Iterable[str] = CATEGORICAL_FEATURES,
) -> ColumnTransformer:
    """Build the project's canonical :class:`ColumnTransformer`.

    Parameters
    ----------
    numeric_features, categorical_features:
        Column-name iterables; defaults come from :mod:`sales_prediction.config`.
    """

    return ColumnTransformer(
        transformers=[
            ("num", _numeric_pipeline(), list(numeric_features)),
            ("cat", _categorical_pipeline(), list(categorical_features)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
