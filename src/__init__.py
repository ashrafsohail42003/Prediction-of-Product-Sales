"""Reusable helpers for the retail sales prediction project.

This package exposes a small, stable public API so notebooks and downstream
consumers do not need to know about internal module layout.
"""

from __future__ import annotations

from .config import (
    BEST_MODEL_PATH,
    DATA_DICTIONARY,
    FIGURES_DIR,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET,
    ensure_project_dirs,
)
from .data_loader import (
    basic_cleaning,
    load_raw_data,
    missing_value_report,
    restore_placeholders_to_null,
    save_processed_snapshot,
    summarize_numeric_columns,
)
from .evaluation import evaluate_regression_model, regression_metrics
from .modeling import (
    build_linear_regression_pipeline,
    build_random_forest_pipeline,
    make_train_test_split,
    split_features_target,
    tune_random_forest,
)
from .preprocessing import make_preprocessor
from .project_pipeline import run_pipeline

__all__ = [
    "BEST_MODEL_PATH",
    "DATA_DICTIONARY",
    "FIGURES_DIR",
    "PROCESSED_DATA_PATH",
    "RANDOM_STATE",
    "RAW_DATA_PATH",
    "TARGET",
    "basic_cleaning",
    "build_linear_regression_pipeline",
    "build_random_forest_pipeline",
    "ensure_project_dirs",
    "evaluate_regression_model",
    "load_raw_data",
    "make_preprocessor",
    "make_train_test_split",
    "missing_value_report",
    "regression_metrics",
    "restore_placeholders_to_null",
    "run_pipeline",
    "save_processed_snapshot",
    "split_features_target",
    "summarize_numeric_columns",
    "tune_random_forest",
]

__version__ = "1.0.0"
