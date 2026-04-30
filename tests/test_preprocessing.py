"""Tests for src.preprocessing."""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer

from src.preprocessing import make_preprocessor


def test_make_preprocessor_returns_column_transformer():
    pre = make_preprocessor()
    assert isinstance(pre, ColumnTransformer)


def test_preprocessor_imputes_and_encodes(synthetic_sales_df):
    df = synthetic_sales_df.drop(columns=["Item_Outlet_Sales"])
    pre = make_preprocessor(scale_numeric=True)
    transformed = pre.fit_transform(df)

    # Output should be 2D dense numeric (no NaNs after imputation).
    assert isinstance(transformed, np.ndarray)
    assert transformed.ndim == 2
    assert not np.isnan(transformed).any()
    assert transformed.shape[0] == len(df)


def test_preprocessor_handles_unknown_categories(synthetic_sales_df):
    df = synthetic_sales_df.drop(columns=["Item_Outlet_Sales"])
    pre = make_preprocessor()
    pre.fit(df)

    unseen = df.head(3).copy()
    unseen["Outlet_Type"] = "BrandNewType"
    # Should NOT raise -- handle_unknown="ignore"
    out = pre.transform(unseen)
    assert out.shape[0] == 3


def test_preprocessor_scaling_changes_numeric_columns(synthetic_sales_df):
    df = synthetic_sales_df.drop(columns=["Item_Outlet_Sales"])
    unscaled = make_preprocessor(scale_numeric=False).fit_transform(df)
    scaled = make_preprocessor(scale_numeric=True).fit_transform(df)
    # Scaled output should have ~zero mean for the first numeric column slot.
    assert not np.allclose(unscaled[:, 0].mean(), 0, atol=1e-1)
    assert np.isclose(scaled[:, 0].mean(), 0, atol=1e-6)
