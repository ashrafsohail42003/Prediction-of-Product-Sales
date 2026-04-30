from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from .config import (
    BEST_MODEL_PATH,
    FIGURES_DIR,
    MODELS_DIR,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET,
    ensure_project_dirs,
)
from .data_loader import basic_cleaning, load_raw_data, save_processed_snapshot
from .evaluation import evaluate_regression_model
from .logging_config import configure_logging
from .modeling import (
    build_linear_regression_pipeline,
    build_random_forest_pipeline,
    make_train_test_split,
    split_features_target,
    tune_random_forest,
)
from .visualization import (
    plot_correlation_heatmap,
    plot_linear_regression_coefficients,
    plot_model_comparison,
    plot_outlet_type_vs_sales,
    plot_rf_feature_importance,
    save_figure,
    set_plot_style,
)

logger = logging.getLogger(__name__)


def _save_core_eda_figures(df: pd.DataFrame) -> None:
    set_plot_style()

    plot_correlation_heatmap(df, FIGURES_DIR / "correlation_heatmap.png")
    plot_outlet_type_vs_sales(df, FIGURES_DIR / "outlet_type_vs_sales.png")

    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="Item_MRP", bins=30, kde=True, color="#3a7ca5")
    plt.title("Distribution of Item MRP")
    plt.xlabel("Maximum Retail Price")
    plt.ylabel("Count")
    save_figure(FIGURES_DIR / "item_mrp_distribution.png")

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Item_Outlet_Sales", color="#86b6a8")
    plt.title("Distribution of Item Outlet Sales")
    plt.xlabel("Item Outlet Sales")
    save_figure(FIGURES_DIR / "sales_boxplot.png")

    plt.figure(figsize=(9, 6))
    order = df["Item_Type"].value_counts().index
    sns.countplot(data=df, y="Item_Type", order=order, color="#7d5a5a")
    plt.title("Item Type Frequency")
    plt.xlabel("Count")
    plt.ylabel("")
    save_figure(FIGURES_DIR / "item_type_countplot.png")


def run_pipeline() -> dict[str, Any]:

    configure_logging()
    ensure_project_dirs()

    logger.info("Loading raw data from %s", PROCESSED_DATA_PATH.parent.parent / "raw")
    raw_df = load_raw_data()
    logger.info("Raw shape: %s rows x %s columns", *raw_df.shape)

    cleaned_df = basic_cleaning(raw_df, fill_placeholders=False)
    save_processed_snapshot(cleaned_df, PROCESSED_DATA_PATH)
    logger.info("Saved cleaned snapshot to %s", PROCESSED_DATA_PATH)

    logger.info("Generating EDA figures...")
    _save_core_eda_figures(cleaned_df)

    X, y = split_features_target(cleaned_df)
    X_train, X_test, y_train, y_test = make_train_test_split(X, y)
    logger.info("Train/Test sizes: %s / %s", len(X_train), len(X_test))

    logger.info("Training Linear Regression...")
    linear_model = build_linear_regression_pipeline()
    linear_model.fit(X_train, y_train)

    logger.info("Training default Random Forest...")
    default_rf = build_random_forest_pipeline(n_estimators=200)
    default_rf.fit(X_train, y_train)

    logger.info("Tuning Random Forest with GridSearchCV...")
    search = tune_random_forest(build_random_forest_pipeline(), X_train, y_train)
    best_params = {
        key.replace("model__", ""): value for key, value in search.best_params_.items()
    }
    logger.info("Best CV params: %s (CV R2=%.4f)", best_params, search.best_score_)

    best_rf = build_random_forest_pipeline(**best_params)
    best_rf.fit(X_train, y_train)

    metrics_df = pd.concat(
        [
            evaluate_regression_model(
                "Linear Regression", linear_model, X_train, y_train, X_test, y_test
            ),
            evaluate_regression_model(
                "Random Forest", default_rf, X_train, y_train, X_test, y_test
            ),
            evaluate_regression_model(
                "Tuned Random Forest", best_rf, X_train, y_train, X_test, y_test
            ),
        ],
        ignore_index=True,
    )
    metrics_df = metrics_df.round({"MAE": 2, "MSE": 2, "RMSE": 2, "R2": 4})

    metrics_path: Path = REPORTS_DIR / "model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info("Saved metrics to %s", metrics_path)

    plot_model_comparison(metrics_df, FIGURES_DIR / "model_comparison.png")
    coef_df = plot_linear_regression_coefficients(
        linear_model, FIGURES_DIR / "linreg_coefficients.png"
    )
    importance_df = plot_rf_feature_importance(
        best_rf, FIGURES_DIR / "rf_feature_importance.png"
    )
    coef_df.head(20).to_csv(REPORTS_DIR / "linear_coefficients.csv", index=False)
    importance_df.head(20).to_csv(REPORTS_DIR / "rf_feature_importance.csv", index=False)

    joblib.dump(best_rf, BEST_MODEL_PATH)
    logger.info("Persisted best model to %s", BEST_MODEL_PATH)

    test_scores = metrics_df.query("split == 'Test'").sort_values("R2", ascending=False)
    recommended_model = test_scores.iloc[0]["model"]

    summary: dict[str, Any] = {
        "rows": int(raw_df.shape[0]),
        "columns": int(raw_df.shape[1]),
        "duplicates": int(raw_df.duplicated().sum()),
        "missing_values": raw_df.isna().sum().loc[lambda s: s > 0].to_dict(),
        "cleaned_processed_path": str(PROCESSED_DATA_PATH),
        "best_grid_params": best_params,
        "cv_best_r2": round(float(search.best_score_), 4),
        "recommended_model": recommended_model,
        "test_metrics": test_scores.to_dict(orient="records"),
        "top_linear_coefficients": coef_df.head(3).to_dict(orient="records"),
        "top_rf_features": importance_df.head(5).to_dict(orient="records"),
        "random_state": RANDOM_STATE,
        "target": TARGET,
        "models_dir": str(MODELS_DIR),
    }

    summary_path = REPORTS_DIR / "project_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote project summary to %s", summary_path)

    return summary


if __name__ == "__main__":  
    print(json.dumps(run_pipeline(), indent=2))
