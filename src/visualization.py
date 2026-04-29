"""Reusable plotting helpers for EDA, model evaluation, and interpretation."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def set_plot_style() -> None:
    """Apply a consistent reporting style."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 220
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelsize"] = 10


def save_figure(path) -> None:
    """Save the current matplotlib figure and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, path) -> None:
    """Save a heatmap of numeric feature correlations."""
    set_plot_style()
    corr = df.select_dtypes(include="number").corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, linewidths=0.5)
    plt.title("Correlation Heatmap of Numeric Features")
    save_figure(path)


def plot_outlet_type_vs_sales(df: pd.DataFrame, path) -> None:
    """Save a bar plot of average sales by outlet type."""
    set_plot_style()
    order = (
        df.groupby("Outlet_Type")["Item_Outlet_Sales"]
        .mean()
        .sort_values(ascending=False)
        .index
    )
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=df, x="Outlet_Type", y="Item_Outlet_Sales", order=order, errorbar=None)
    ax.set_title("Average Sales by Outlet Type")
    ax.set_xlabel("Outlet Type")
    ax.set_ylabel("Average Item Outlet Sales")
    plt.xticks(rotation=20, ha="right")
    save_figure(path)


def plot_model_comparison(metrics_df: pd.DataFrame, path) -> None:
    """Save a side-by-side comparison of test model performance."""
    set_plot_style()
    test_metrics = metrics_df.query("split == 'Test'").copy()
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=test_metrics, x="model", y="R2", hue="model", dodge=False, legend=False)
    ax.set_title("Test R-Squared by Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Test R-Squared")
    ax.set_ylim(0, max(0.75, test_metrics["R2"].max() + 0.05))
    plt.xticks(rotation=15, ha="right")
    save_figure(path)


def _model_feature_frame(pipeline, values, value_name: str) -> pd.DataFrame:
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    return pd.DataFrame({"feature": feature_names, value_name: values})


def plot_linear_regression_coefficients(pipeline, path, *, top_n: int = 20) -> pd.DataFrame:
    """Save the strongest linear regression coefficients by absolute value."""
    set_plot_style()
    coefs = pipeline.named_steps["model"].coef_
    coef_df = _model_feature_frame(pipeline, coefs, "coefficient")
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    top = coef_df.sort_values("abs_coefficient", ascending=False).head(top_n)

    plt.figure(figsize=(9, 7))
    ax = sns.barplot(data=top, y="feature", x="coefficient", hue="coefficient", palette="vlag", legend=False)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Most Influential Linear Regression Coefficients")
    ax.set_xlabel("Coefficient")
    ax.set_ylabel("")
    save_figure(path)
    return coef_df.sort_values("abs_coefficient", ascending=False)


def plot_rf_feature_importance(pipeline, path, *, top_n: int = 20) -> pd.DataFrame:
    """Save the strongest random forest feature importances."""
    set_plot_style()
    importances = pipeline.named_steps["model"].feature_importances_
    importance_df = _model_feature_frame(pipeline, importances, "importance")
    top = importance_df.sort_values("importance", ascending=False).head(top_n)

    plt.figure(figsize=(9, 7))
    ax = sns.barplot(data=top, y="feature", x="importance", color="#2f7f8f")
    ax.set_title("Random Forest Feature Importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    save_figure(path)
    return importance_df.sort_values("importance", ascending=False)
