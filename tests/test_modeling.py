"""Tests for src.modeling."""

from __future__ import annotations

from sklearn.pipeline import Pipeline

from src.modeling import (
    build_linear_regression_pipeline,
    build_random_forest_pipeline,
    make_train_test_split,
    split_features_target,
)


def test_split_features_target_drops_id_and_target(synthetic_sales_df):
    X, y = split_features_target(synthetic_sales_df)
    assert "Item_Outlet_Sales" not in X.columns
    assert "Item_Identifier" not in X.columns
    assert y.name == "Item_Outlet_Sales"


def test_train_test_split_is_reproducible(synthetic_sales_df):
    X, y = split_features_target(synthetic_sales_df)
    a = make_train_test_split(X, y, test_size=0.25, random_state=42)
    b = make_train_test_split(X, y, test_size=0.25, random_state=42)
    # Same indices on both runs.
    assert (a[0].index == b[0].index).all()
    assert (a[1].index == b[1].index).all()


def test_linear_pipeline_fits_and_predicts(synthetic_sales_df):
    X, y = split_features_target(synthetic_sales_df)
    X_train, X_test, y_train, _ = make_train_test_split(X, y)
    pipeline = build_linear_regression_pipeline()
    assert isinstance(pipeline, Pipeline)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    assert preds.shape[0] == X_test.shape[0]


def test_random_forest_pipeline_fits_and_predicts(synthetic_sales_df):
    X, y = split_features_target(synthetic_sales_df)
    X_train, X_test, y_train, _ = make_train_test_split(X, y)
    pipeline = build_random_forest_pipeline(n_estimators=10, max_depth=4)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    assert preds.shape[0] == X_test.shape[0]
