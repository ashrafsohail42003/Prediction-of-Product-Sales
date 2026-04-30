"""Tests for src.data_loader."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import (
    FAT_CONTENT_MAP,
    basic_cleaning,
    fix_inconsistent_categories,
    missing_value_report,
    restore_placeholders_to_null,
    summarize_numeric_columns,
)


def test_fix_inconsistent_categories_normalizes_fat_content(synthetic_sales_df):
    cleaned = fix_inconsistent_categories(synthetic_sales_df)
    assert set(cleaned["Item_Fat_Content"].unique()).issubset({"Low Fat", "Regular"})


def test_fat_content_map_is_complete():
    # Every value in the map should normalize to one of the canonical labels.
    assert set(FAT_CONTENT_MAP.values()) == {"Low Fat", "Regular"}


def test_basic_cleaning_drops_duplicates(synthetic_sales_df):
    duplicated = pd.concat([synthetic_sales_df, synthetic_sales_df.head(3)], ignore_index=True)
    cleaned = basic_cleaning(duplicated, fill_placeholders=False)
    assert len(cleaned) == len(synthetic_sales_df)


def test_basic_cleaning_with_placeholders_removes_nulls(synthetic_sales_df):
    cleaned = basic_cleaning(synthetic_sales_df, fill_placeholders=True)
    assert cleaned.isna().sum().sum() == 0


def test_restore_placeholders_round_trip(synthetic_sales_df):
    placeholders = basic_cleaning(synthetic_sales_df, fill_placeholders=True)
    restored = restore_placeholders_to_null(placeholders)
    # Restored should have nulls again wherever we originally had them.
    assert restored.isna().sum().sum() > 0


def test_missing_value_report_returns_only_missing_columns(synthetic_sales_df):
    report = missing_value_report(synthetic_sales_df)
    assert "Item_Weight" in report.index
    assert (report["missing_count"] > 0).all()


def test_summarize_numeric_columns_has_expected_stats(synthetic_sales_df):
    summary = summarize_numeric_columns(synthetic_sales_df)
    assert {"min", "max", "mean"}.issubset(summary.columns)
    assert "Item_MRP" in summary.index
    # Sanity: numeric columns must appear, non-numeric ones must not.
    assert "Item_Type" not in summary.index
    assert np.isfinite(summary.loc["Item_MRP", "mean"])
