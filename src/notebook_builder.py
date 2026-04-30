from __future__ import annotations
import argparse
import textwrap
from pathlib import Path
import nbformat as nbf
import pandas as pd
from .config import DATA_DICTIONARY, RAW_DATA_PATH, ROOT_DIR, TARGET
from .data_loader import basic_cleaning


NOTEBOOK_DIR = ROOT_DIR / "notebooks"


def md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(source: str):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip())


def notebook(cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    return nb


def setup_cell():
    return code(
        """
        from pathlib import Path
        import sys

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        from IPython.display import Image, display

        PROJECT_ROOT = Path.cwd()
        if PROJECT_ROOT.name == "notebooks":
            PROJECT_ROOT = PROJECT_ROOT.parent
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.append(str(PROJECT_ROOT))

        from src.config import (
            BEST_MODEL_PATH,
            DATA_DICTIONARY,
            FIGURES_DIR,
            PROCESSED_DATA_PATH,
            RANDOM_STATE,
            RAW_DATA_PATH,
            TARGET,
        )
        from src.data_loader import (
            basic_cleaning,
            load_raw_data,
            missing_value_report,
            restore_placeholders_to_null,
            save_processed_snapshot,
            summarize_numeric_columns,
        )
        from src.evaluation import evaluate_regression_model
        from src.modeling import (
            build_linear_regression_pipeline,
            build_random_forest_pipeline,
            make_train_test_split,
            split_features_target,
            tune_random_forest,
        )
        from src.preprocessing import make_preprocessor
        from src.visualization import (
            plot_correlation_heatmap,
            plot_linear_regression_coefficients,
            plot_model_comparison,
            plot_outlet_type_vs_sales,
            plot_rf_feature_importance,
            save_figure,
            set_plot_style,
        )

        pd.set_option("display.max_columns", 100)
        set_plot_style()
        """
    )


def feature_type(df: pd.DataFrame, feature: str) -> str:
    if feature in {"Outlet_Size", "Outlet_Location_Type"}:
        return "Ordinal categorical"
    if pd.api.types.is_numeric_dtype(df[feature]):
        return "Numeric"
    return "Categorical (nominal)"


def null_action(feature: str, dtype_label: str, null_count: int) -> str:
    if null_count == 0:
        return "No missing-value action is needed."
    if "Numeric" in dtype_label:
        return "Impute with the median after the train-test split."
    return "Impute with the most frequent category after the train-test split."


def expected_predictor(feature: str) -> str:
    strong = {"Item_MRP", "Outlet_Type", "Outlet_Size", "Outlet_Location_Type"}
    moderate = {"Item_Visibility", "Item_Type", "Outlet_Identifier", "Outlet_Establishment_Year"}
    weak = {"Item_Fat_Content", "Item_Weight"}
    if feature == "Item_Identifier":
        return "No; this is a product ID and should not be treated as a general business driver."
    if feature in strong:
        return "Yes; this feature has a clear business relationship to sales."
    if feature in moderate:
        return "Possibly; it may capture store or product context that affects demand."
    if feature in weak:
        return "Only weakly; the business connection is plausible but likely indirect."
    return "Possibly."


def apparent_signal(df: pd.DataFrame, feature: str) -> str:
    target = df[TARGET]
    if feature == TARGET:
        return "This is the target, so it is not used as a predictor."
    if feature == "Item_Identifier":
        return "The raw averages vary by product ID, but this is high-cardinality and not reliable for generalization."
    if pd.api.types.is_numeric_dtype(df[feature]):
        corr = df[[feature, TARGET]].corr(numeric_only=True).iloc[0, 1]
        strength = "strong" if abs(corr) >= 0.30 else "moderate" if abs(corr) >= 0.10 else "weak"
        return f"The correlation with sales is {corr:.3f}, which suggests a {strength} linear signal."
    means = df.groupby(feature, dropna=False)[TARGET].mean()
    ratio = (means.max() - means.min()) / target.mean()
    strength = "strong" if ratio >= 0.50 else "moderate" if ratio >= 0.20 else "weak"
    return (
        f"Average sales vary by about {ratio:.1%} of the overall mean across categories, "
        f"suggesting a {strength} signal."
    )


def feature_answer_markdown(df: pd.DataFrame, feature: str) -> str:
    null_count = int(df[feature].isna().sum())
    null_pct = null_count / len(df) * 100
    cardinality = int(df[feature].nunique(dropna=True))
    top_share = df[feature].value_counts(normalize=True, dropna=False).iloc[0]
    const_text = (
        "constant"
        if cardinality <= 1
        else "quasi-constant"
        if top_share >= 0.95
        else "not constant or quasi-constant"
    )
    dtype_label = feature_type(df, feature)
    high_cardinality = "Yes" if cardinality > 10 else "No"
    business_exclusion = (
        "Exclude from modeling because it is a high-cardinality identifier."
        if feature == "Item_Identifier"
        else "Keep unless model validation shows it hurts generalization."
    )

    return f"""
    Feature inspection notes:

    - Type: {dtype_label}.
    - Null values: {null_count:,} ({null_pct:.2f}%); action: {null_action(feature, dtype_label, null_count)}
    - Constant or quasi-constant: {const_text}.
    - Cardinality: {cardinality:,}; high cardinality (>10): {high_cardinality}.
    - Known before target: Yes, this is known before the item sales are measured.
    - Business exclusion: {business_exclusion}

    Predictor notes:

    - Business expectation: {expected_predictor(feature)}
    - Visual/statistical signal: {apparent_signal(df, feature)}
    """


def build_eda_notebook() -> None:
    raw_df = pd.read_csv(RAW_DATA_PATH)
    clean_df = basic_cleaning(raw_df, fill_placeholders=False)
    features = [col for col in clean_df.columns if col != TARGET]

    cells = [
        md(
            """
            # Retail Sales Prediction: EDA and Feature Inspection

            This notebook covers the loading, cleaning, exploratory visuals, and feature inspection workflow for the retail sales prediction project.
            """
        ),
        setup_cell(),
        md("## Loading Data"),
        code(
            """
            df = pd.read_csv(RAW_DATA_PATH)
            df.info()
            display(df.head())
            """
        ),
        md("## Data Cleaning"),
        code(
            """
            print(f"Rows: {df.shape[0]:,}")
            print(f"Columns: {df.shape[1]:,}")
            display(df.dtypes.to_frame("dtype"))
            """
        ),
        code(
            """
            print(f"Duplicate rows: {df.duplicated().sum():,}")
            display(missing_value_report(df))
            """
        ),
        code(
            """
            print("Original Item_Fat_Content values:")
            print(sorted(df["Item_Fat_Content"].dropna().unique()))

            clean_df = basic_cleaning(df, fill_placeholders=False)

            print("\\nCleaned Item_Fat_Content values:")
            print(sorted(clean_df["Item_Fat_Content"].dropna().unique()))
            """
        ),
        code(
            """
            placeholder_df = basic_cleaning(df, fill_placeholders=True)
            display(placeholder_df.isna().sum().to_frame("missing_after_placeholder"))
            display(summarize_numeric_columns(clean_df))

            save_processed_snapshot(clean_df, PROCESSED_DATA_PATH)
            print(f"Saved cleaned snapshot to: {PROCESSED_DATA_PATH}")
            """
        ),
        md(
            """
            ## Exploratory Visuals

            The following visuals satisfy the EDA requirements: histograms for numeric distributions, boxplots for statistical summaries, countplots for categorical frequencies, and a heatmap for correlations.
            """
        ),
        code(
            """
            numeric_cols = clean_df.select_dtypes(include="number").columns
            clean_df[numeric_cols].hist(figsize=(12, 8), bins=30)
            plt.suptitle("Numeric Feature Distributions", y=1.02)
            plt.tight_layout()
            plt.show()
            """
        ),
        code(
            """
            plt.figure(figsize=(10, 5))
            sns.boxplot(data=clean_df[["Item_Weight", "Item_Visibility", "Item_MRP", "Item_Outlet_Sales"]])
            plt.title("Boxplots for Numeric Features")
            plt.xticks(rotation=20, ha="right")
            plt.show()
            """
        ),
        code(
            """
            plt.figure(figsize=(9, 6))
            sns.countplot(data=clean_df, y="Item_Type", order=clean_df["Item_Type"].value_counts().index)
            plt.title("Frequency of Item Types")
            plt.xlabel("Count")
            plt.ylabel("")
            plt.show()
            """
        ),
        code(
            """
            plot_correlation_heatmap(clean_df, FIGURES_DIR / "correlation_heatmap.png")
            display(Image(filename=str(FIGURES_DIR / "correlation_heatmap.png")))
            """
        ),
        code(
            """
            plot_outlet_type_vs_sales(clean_df, FIGURES_DIR / "outlet_type_vs_sales.png")
            display(Image(filename=str(FIGURES_DIR / "outlet_type_vs_sales.png")))
            """
        ),
        md(
            """
            ## Feature Inspection

            The placeholders used for the Part 2 cleaning check are converted back to null values here so feature inspection can report the real missing-value frequency.
            """
        ),
        code(
            """
            feature_inspection_df = restore_placeholders_to_null(placeholder_df)
            display(missing_value_report(feature_inspection_df))
            """
        ),
        code(
            """
            def plot_univariate_feature(data, feature):
                plt.figure(figsize=(8, 4.5))
                if pd.api.types.is_numeric_dtype(data[feature]):
                    sns.histplot(data=data, x=feature, bins=30, kde=True)
                    plt.ylabel("Count")
                else:
                    order = data[feature].value_counts(dropna=False).head(20).index
                    sns.countplot(data=data, y=feature, order=order)
                    plt.xlabel("Count")
                    plt.ylabel("")
                plt.title(f"Distribution of {feature}")
                plt.tight_layout()
                plt.show()


            def plot_feature_vs_target(data, feature, target=TARGET):
                plt.figure(figsize=(8, 4.5))
                if pd.api.types.is_numeric_dtype(data[feature]):
                    sns.scatterplot(data=data, x=feature, y=target, alpha=0.35, edgecolor=None)
                else:
                    order = (
                        data.groupby(feature, dropna=False)[target]
                        .mean()
                        .sort_values(ascending=False)
                        .head(20)
                        .index
                    )
                    sns.barplot(data=data, y=feature, x=target, order=order, errorbar=None)
                    plt.ylabel("")
                plt.title(f"{feature} vs. {target}")
                plt.tight_layout()
                plt.show()
            """
        ),
    ]

    for feature in features:
        cells.extend(
            [
                md(f"### {feature}\n\nDefinition: {DATA_DICTIONARY[feature]}"),
                code(f'plot_univariate_feature(feature_inspection_df, "{feature}")'),
                md(feature_answer_markdown(clean_df, feature)),
                code(f'plot_feature_vs_target(feature_inspection_df, "{feature}")'),
                md(
                    f"""
                    Based on the business context, {expected_predictor(feature)}

                    Based on the visualization, {apparent_signal(clean_df, feature)}
                    """
                ),
            ]
        )

    path = NOTEBOOK_DIR / "01_eda.ipynb"
    nbf.write(notebook(cells), path)


def build_modeling_notebook() -> None:
    metrics_path = ROOT_DIR / "reports" / "model_metrics.csv"
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()

    def metric(model: str, split: str, name: str) -> float:
        if metrics.empty:
            return float("nan")
        row = metrics.query("model == @model and split == @split").iloc[0]
        return row[name]

    cells = [
        md(
            """
            # Retail Sales Prediction: Modeling and Interpretation

            This notebook loads a fresh copy of the raw dataset, performs pre-split cleaning only where safe, builds preprocessing pipelines, compares models, tunes a Random Forest, and interprets the final model.
            """
        ),
        setup_cell(),
        md("## Fresh Data Load"),
        code(
            """
            raw_df = pd.read_csv(RAW_DATA_PATH)
            print(raw_df.shape)
            display(raw_df.head())
            """
        ),
        md(
            """
            ## Pre-Split Cleaning

            To avoid data leakage, only duplicate removal and category standardization happen before splitting. Missing-value imputation is handled inside the scikit-learn preprocessing pipeline after the train-test split.
            """
        ),
        code(
            """
            model_df = basic_cleaning(raw_df, fill_placeholders=False)

            print(f"Duplicate rows after cleaning: {model_df.duplicated().sum():,}")
            print(sorted(model_df["Item_Fat_Content"].dropna().unique()))
            display(missing_value_report(model_df))
            """
        ),
        md("## Feature and Target Split"),
        code(
            """
            X, y = split_features_target(model_df)
            X_train, X_test, y_train, y_test = make_train_test_split(X, y)

            print(f"X_train: {X_train.shape}")
            print(f"X_test: {X_test.shape}")
            print(f"y_train: {y_train.shape}")
            print(f"y_test: {y_test.shape}")
            """
        ),
        md("## Preprocessing Object"),
        code(
            """
            preprocessor = make_preprocessor(scale_numeric=True)
            preprocessor
            """
        ),
        md("## Linear Regression"),
        code(
            """
            linear_model = build_linear_regression_pipeline()
            linear_model.fit(X_train, y_train)

            linear_metrics = evaluate_regression_model(
                "Linear Regression", linear_model, X_train, y_train, X_test, y_test
            )
            display(linear_metrics.round(4))
            """
        ),
        md(
            f"""
            Linear Regression has train R-squared {metric("Linear Regression", "Train", "R2"):.4f} and test R-squared {metric("Linear Regression", "Test", "R2"):.4f}. The two scores are close, so this model is not heavily overfit, but it underfits some of the nonlinear relationships in the data.
            """
        ),
        md("## Default Random Forest"),
        code(
            """
            default_rf = build_random_forest_pipeline(n_estimators=200)
            default_rf.fit(X_train, y_train)

            rf_metrics = evaluate_regression_model(
                "Random Forest", default_rf, X_train, y_train, X_test, y_test
            )
            display(rf_metrics.round(4))
            """
        ),
        md(
            f"""
            The default Random Forest has train R-squared {metric("Random Forest", "Train", "R2"):.4f} and test R-squared {metric("Random Forest", "Test", "R2"):.4f}. This large gap indicates overfitting. Its test score is also lower than Linear Regression, so tuning is needed before recommending a tree-based model.
            """
        ),
        md("## Tuned Random Forest"),
        code(
            """
            search = tune_random_forest(build_random_forest_pipeline(), X_train, y_train)
            print("Best parameters:", search.best_params_)
            print(f"Best cross-validation R2: {search.best_score_:.4f}")

            best_params = {key.replace("model__", ""): value for key, value in search.best_params_.items()}
            tuned_rf = build_random_forest_pipeline(**best_params)
            tuned_rf.fit(X_train, y_train)

            tuned_metrics = evaluate_regression_model(
                "Tuned Random Forest", tuned_rf, X_train, y_train, X_test, y_test
            )
            display(tuned_metrics.round(4))
            """
        ),
        code(
            """
            metrics_df = pd.concat([linear_metrics, rf_metrics, tuned_metrics], ignore_index=True)
            metrics_df = metrics_df.round({"MAE": 2, "MSE": 2, "RMSE": 2, "R2": 4})
            display(metrics_df)

            metrics_df.to_csv(PROJECT_ROOT / "reports" / "model_metrics.csv", index=False)
            plot_model_comparison(metrics_df, FIGURES_DIR / "model_comparison.png")
            display(Image(filename=str(FIGURES_DIR / "model_comparison.png")))
            """
        ),
        md(
            f"""
            The tuned Random Forest improved over the default Random Forest on the test set ({metric("Tuned Random Forest", "Test", "R2"):.4f} vs. {metric("Random Forest", "Test", "R2"):.4f}). The tuned model is the recommended model because it has the best test R-squared and the lowest RMSE among the compared models.

            For a non-technical stakeholder: the tuned model explains about {metric("Tuned Random Forest", "Test", "R2") * 100:.1f}% of the variation in item sales on unseen data. I would also communicate RMSE (${metric("Tuned Random Forest", "Test", "RMSE"):,.0f}) because it describes the typical prediction error in the same sales units stakeholders understand.
            """
        ),
        md("## Linear Regression Coefficients"),
        code(
            """
            coef_df = plot_linear_regression_coefficients(
                linear_model, FIGURES_DIR / "linreg_coefficients.png"
            )
            display(Image(filename=str(FIGURES_DIR / "linreg_coefficients.png")))
            display(coef_df.head(10))
            """
        ),
        md(
            """
            Top coefficient interpretation:

            - `Outlet_Type_Grocery Store` has the strongest negative coefficient, meaning grocery store outlets are associated with much lower predicted sales than the baseline store type.
            - `Item_MRP` has a strong positive coefficient, meaning higher-priced items tend to generate higher sales.
            - `Outlet_Identifier_OUT027` has a strong positive coefficient, suggesting that this specific outlet performs above the baseline after preprocessing.
            """
        ),
        md("## Tree-Based Feature Importances"),
        code(
            """
            importance_df = plot_rf_feature_importance(
                tuned_rf, FIGURES_DIR / "rf_feature_importance.png"
            )
            display(Image(filename=str(FIGURES_DIR / "rf_feature_importance.png")))
            display(importance_df.head(10))
            """
        ),
        md(
            """
            Top Random Forest feature interpretation:

            - `Item_MRP` is the most important feature, showing that listed price is the strongest driver of predicted sales.
            - `Outlet_Type_Grocery Store` is highly important, reinforcing that grocery outlets behave differently from supermarkets.
            - `Outlet_Type_Supermarket Type3`, `Outlet_Identifier_OUT027`, and `Outlet_Establishment_Year` indicate that store format and store identity/history matter for sales prediction.
            """
        ),
        md("## Save Final Model"),
        code(
            """
            import joblib

            joblib.dump(tuned_rf, BEST_MODEL_PATH)
            print(f"Saved final tuned model to: {BEST_MODEL_PATH}")
            """
        ),
        md(
            """
            ## Stakeholder Recommendation

            I recommend using the tuned Random Forest for sales forecasting because it provides the strongest test-set performance while reducing the overfitting seen in the default forest. From a business perspective, the retailer should pay close attention to item price, store format, and high-performing outlet patterns when planning assortment, placement, and sales forecasts.
            """
        ),
    ]

    path = NOTEBOOK_DIR / "02_modeling.ipynb"
    nbf.write(notebook(cells), path)


def execute_notebook(path: Path) -> None:
    from nbclient import NotebookClient

    nb = nbf.read(path, as_version=4)
    client = NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(ROOT_DIR)}})
    client.execute()
    nbf.write(nb, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Execute notebooks after creating them.")
    args = parser.parse_args()

    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    build_eda_notebook()
    build_modeling_notebook()

    if args.execute:
        execute_notebook(NOTEBOOK_DIR / "01_eda.ipynb")
        execute_notebook(NOTEBOOK_DIR / "02_modeling.ipynb")


if __name__ == "__main__":
    main()
