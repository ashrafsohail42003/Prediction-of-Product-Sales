"""Data loading and cleaning helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MISSING_CATEGORY, MISSING_NUMERIC, PROCESSED_DATA_PATH, RAW_DATA_PATH


FAT_CONTENT_MAP = {
    "LF": "Low Fat",
    "low fat": "Low Fat",
    "Low Fat": "Low Fat",
    "reg": "Regular",
    "Regular": "Regular",
}


def load_raw_data(path=RAW_DATA_PATH) -> pd.DataFrame:
    """Load the original sales prediction data."""
    return pd.read_csv(path)


def fix_inconsistent_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize known inconsistent categorical labels."""
    cleaned = df.copy()
    object_cols = cleaned.select_dtypes(include="object").columns
    cleaned[object_cols] = cleaned[object_cols].apply(lambda col: col.str.strip())

    if "Item_Fat_Content" in cleaned.columns:
        cleaned["Item_Fat_Content"] = cleaned["Item_Fat_Content"].replace(FAT_CONTENT_MAP)

    return cleaned


def basic_cleaning(
    df: pd.DataFrame,
    *,
    fill_placeholders: bool = False,
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """Clean duplicates and categories, optionally filling missing values with placeholders."""
    cleaned = df.copy()

    if drop_duplicates:
        cleaned = cleaned.drop_duplicates()

    cleaned = fix_inconsistent_categories(cleaned)

    if fill_placeholders:
        numeric_cols = cleaned.select_dtypes(include=np.number).columns
        categorical_cols = cleaned.select_dtypes(exclude=np.number).columns
        cleaned[numeric_cols] = cleaned[numeric_cols].fillna(MISSING_NUMERIC)
        cleaned[categorical_cols] = cleaned[categorical_cols].fillna(MISSING_CATEGORY)

    return cleaned


def restore_placeholders_to_null(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the Part 2 placeholder values back to nulls for feature inspection."""
    restored = df.copy()
    restored = restored.replace({MISSING_CATEGORY: np.nan, MISSING_NUMERIC: np.nan})
    return restored


def save_processed_snapshot(df: pd.DataFrame, path=PROCESSED_DATA_PATH) -> None:
    """Save a cleaned CSV snapshot for reproducible EDA."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing counts and percentages by column."""
    missing = df.isna().sum()
    report = pd.DataFrame(
        {
            "missing_count": missing,
            "missing_percent": (missing / len(df) * 100).round(2),
        }
    )
    return report.query("missing_count > 0").sort_values("missing_count", ascending=False)


def summarize_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return min, max, and mean for numerical columns."""
    numeric_cols = df.select_dtypes(include=np.number).columns
    return df[numeric_cols].agg(["min", "max", "mean"]).T.round(3)
