"""Reusable EDA plot helpers.

Matplotlib/seaborn primitives that the notebook (and any downstream analysis
scripts) can import instead of re-implementing boilerplate. All functions
return the created ``Figure`` so callers can choose to ``.savefig(...)`` them.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# A single place to tune the project's plot look-and-feel.
DEFAULT_PALETTE = "viridis"
DEFAULT_CONTEXT = "notebook"
DEFAULT_STYLE = "whitegrid"


def set_plot_style(
    *,
    context: str = DEFAULT_CONTEXT,
    style: str = DEFAULT_STYLE,
    palette: str = DEFAULT_PALETTE,
) -> None:
    """Apply the project-wide seaborn / matplotlib look."""

    sns.set_theme(context=context, style=style, palette=palette)
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 140,
            "figure.autolayout": True,
            "axes.titleweight": "semibold",
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "legend.frameon": False,
        }
    )


# ---------------------------------------------------------------------------
# Univariate helpers
# ---------------------------------------------------------------------------
def explore_numeric(
    df: pd.DataFrame, column: str, *, bins: int | str = "auto"
) -> plt.Figure:
    """Plot a histogram (with KDE) and a boxplot side-by-side for *column*."""

    fig, axes = plt.subplots(2, 1, figsize=(6.5, 5.5), sharex=True)
    sns.histplot(data=df, x=column, bins=bins, kde=True, ax=axes[0])
    sns.boxplot(data=df, x=column, ax=axes[1])
    axes[0].set_title(f"Distribution of {column}")
    axes[1].set_title("")
    axes[1].set_xlabel(column)

    null_count = int(df[column].isna().sum())
    null_pct = null_count / len(df) * 100
    fig.suptitle("")
    fig.text(
        0.99,
        0.01,
        f"NaNs: {null_count} ({null_pct:.2f}%)",
        ha="right",
        va="bottom",
        fontsize=9,
        color="dimgray",
    )
    return fig


def explore_categorical(
    df: pd.DataFrame, column: str, *, top_n: int | None = 20
) -> plt.Figure:
    """Plot a countplot (ordered desc) for categorical *column*."""

    order = df[column].value_counts(dropna=True).index
    if top_n is not None and len(order) > top_n:
        order = order[:top_n]

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=df, x=column, order=order, ax=ax)
    ax.set_title(f"Countplot of {column}")
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")

    null_count = int(df[column].isna().sum())
    null_pct = null_count / len(df) * 100
    unique = df[column].nunique(dropna=True)
    ax.text(
        0.99,
        0.97,
        f"NaNs: {null_count} ({null_pct:.2f}%) · Unique: {unique}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="dimgray",
    )
    return fig


# ---------------------------------------------------------------------------
# Multivariate helpers
# ---------------------------------------------------------------------------
def plot_grid(
    df: pd.DataFrame,
    columns: Sequence[str],
    plot_fn,
    *,
    ncols: int = 3,
    figsize_per_cell: tuple[float, float] = (5.0, 3.5),
) -> plt.Figure:
    """Arrange univariate plots of *columns* on a ``ncols``-wide grid."""

    n = len(columns)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_cell[0] * ncols, figsize_per_cell[1] * nrows),
    )
    axes = axes.flatten() if nrows * ncols > 1 else [axes]
    for ax, col in zip(axes, columns):
        plot_fn(df, col, ax)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def correlation_heatmap(
    df: pd.DataFrame,
    *,
    numeric_only: bool = True,
    cmap: str = "coolwarm",
) -> plt.Figure:
    """Heatmap of the Pearson correlation between numeric columns."""

    corr = df.corr(numeric_only=numeric_only)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cmap=cmap,
        square=True,
        cbar_kws={"shrink": 0.75},
        ax=ax,
    )
    ax.set_title("Correlation heatmap (numeric features)")
    return fig


def target_vs_feature(
    df: pd.DataFrame,
    feature: str,
    target: str,
    *,
    dtype_hint: str | None = None,
) -> plt.Figure:
    """Plot *feature* against *target*, choosing a sensible chart by dtype."""

    kind = dtype_hint or (
        "numeric" if pd.api.types.is_numeric_dtype(df[feature]) else "categorical"
    )
    fig, ax = plt.subplots(figsize=(7, 4))

    if kind == "numeric":
        sns.scatterplot(data=df, x=feature, y=target, alpha=0.35, s=14, ax=ax)
        sns.regplot(
            data=df,
            x=feature,
            y=target,
            scatter=False,
            color="black",
            line_kws={"linewidth": 1.2},
            ax=ax,
        )
    else:
        order = df.groupby(feature)[target].median().sort_values(ascending=False).index
        sns.boxplot(data=df, x=feature, y=target, order=order, ax=ax)
        ax.tick_params(axis="x", rotation=45)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")

    ax.set_title(f"{target} vs {feature}")
    return fig


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------
def null_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-column null counts and percentages, sorted descending."""

    total = df.isna().sum()
    pct = (total / len(df) * 100).round(2)
    return (
        pd.DataFrame({"null_count": total, "null_pct": pct})
        .sort_values("null_count", ascending=False)
    )


def summary_stats(df: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Return ``min / max / mean / median / std`` for numeric columns."""

    numeric = df.select_dtypes(include="number")
    if columns is not None:
        numeric = numeric[list(columns)]
    stats = numeric.agg(["min", "max", "mean", "median", "std"]).T
    return stats.round(3)
