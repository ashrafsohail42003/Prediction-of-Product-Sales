# Prediction of Product Sales

[![CI](https://github.com/ashrafsohail42003/Prediction-of-Product-Sales/actions/workflows/ci.yml/badge.svg)](https://github.com/ashrafsohail42003/Prediction-of-Product-Sales/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5%2B-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

End-to-end machine-learning project that predicts item-level sales for a retail
chain. Goal: help the retailer understand which product and store characteristics
most strongly drive sales, and provide a regression model that forecasts
`Item_Outlet_Sales` on unseen data.

The dataset has 8,523 rows and 12 columns. Target: `Item_Outlet_Sales`.

The project is structured as a production-style Python project — modular `src/`
package, reproducible notebook build, pinned modern dependencies — to double as
a software-engineering sample alongside the data-science deliverable.

---

## Highlights

- **Reproducible pipeline** with a single command (`python -m src.cli pipeline`).
- **No data leakage** — duplicates and category fixes happen pre-split; missing-value
  imputation lives inside the `sklearn` `Pipeline` so it runs *after* the train-test split.
- **Three models compared** — Linear Regression, default Random Forest, and a
  GridSearchCV-tuned Random Forest.
- **Tested & linted** — `pytest` suite for the data, preprocessing, evaluation,
  and modeling layers, plus `ruff` + `black` enforced in CI.
- **Programmatic notebooks** — both notebooks are generated from
  `src/notebook_builder.py`, so they stay in sync with the codebase.

## Business problem

Retail sales are influenced by both product attributes and outlet characteristics.
This analysis identifies the most useful predictors and benchmarks regression
models that can forecast item-level outlet sales.

## Key insights

### Outlet type is strongly related to sales
![Average sales by outlet type](reports/figures/outlet_type_vs_sales.png)

Supermarket formats — especially higher-tier supermarkets — produce significantly
higher average sales than grocery outlets.

### Item price is the strongest model driver
![Random forest feature importances](reports/figures/rf_feature_importance.png)

The tuned Random Forest ranked `Item_MRP` as the most important predictor, followed
by outlet-format indicators (grocery store, supermarket type).

## Model performance

The recommended model is the **tuned Random Forest Regressor**, which produced the
best test-set performance among the compared models.

| Model | Test R² | Test RMSE | Test MAE |
| --- | ---: | ---: | ---: |
| Tuned Random Forest | 0.6026 | 1,047.09 | 728.36 |
| Linear Regression   | 0.5671 | 1,092.86 | 804.12 |
| Default Random Forest | 0.5588 | 1,103.32 | 768.82 |

For a non-technical audience: the tuned model explains roughly **60%** of the
variation in sales on unseen data. The RMSE of about **1,047 sales units** means
predictions are typically off by that amount — useful as a planning tool, with
clear room to improve once richer signals (promotions, inventory, time series)
are available.

![Model comparison](reports/figures/model_comparison.png)

## Model interpretation

### Linear Regression coefficients
![Linear regression coefficients](reports/figures/linreg_coefficients.png)

The largest linear effects: grocery-store outlets are associated with lower
predicted sales, while higher `Item_MRP` and the OUT027 outlet indicator are
associated with higher predicted sales.

### Random Forest feature importance
| Rank | Feature | Interpretation |
| ---: | --- | --- |
| 1 | `Item_MRP` | Listed item price is the strongest signal for sales. |
| 2 | `Outlet_Type_Grocery Store` | Grocery stores behave differently from supermarket formats. |
| 3 | `Outlet_Type_Supermarket Type3` | Store format meaningfully shifts predicted sales. |
| 4 | `Outlet_Identifier_OUT027` | This outlet has a strong individual sales pattern. |
| 5 | `Outlet_Establishment_Year` | Store age contributes a smaller but useful signal. |

## Recommendations

Focus forecasting and merchandising review on **item pricing**, **outlet format**,
and **high-performing individual outlets**. Grocery stores likely need a separate
forecasting strategy from supermarkets because their behavior differs substantially.
Future iterations should incorporate promotion calendars, inventory availability,
local demographics, and time-based sales history.

---

## Quickstart with Anaconda

The recommended environment is conda. After installing
[Anaconda](https://www.anaconda.com/download) or
[Miniconda](https://docs.conda.io/projects/miniconda/en/latest/), open the
**Anaconda Prompt** (Windows) or any terminal (macOS / Linux):

```bash
# 1) Clone the repository
git clone https://github.com/ashrafsohail42003/Prediction-of-Product-Sales.git
cd Prediction-of-Product-Sales

# 2) Create the conda environment from environment.yml
conda env create -f environment.yml

# 3) Activate it
conda activate retail-sales-prediction

# 4) (Optional) install the package itself in editable mode
pip install -e ".[dev]"

# 5) Place the raw dataset at data/raw/sales_predictions.csv
#    (data/raw/ is git-ignored on purpose)

# 6) Run the full pipeline
python -m src.cli pipeline -v
```

Outputs land in:

- `reports/figures/*.png` — EDA + interpretation plots used in this README
- `reports/model_metrics.csv` — train/test metrics for every model
- `reports/project_summary.json` — machine-readable run summary
- `models/best_rf_model.joblib` — the persisted best model

### Working in JupyterLab

```bash
conda activate retail-sales-prediction
jupyter lab
```

Open `notebooks/01_eda.ipynb` and `notebooks/02_modeling.ipynb`. They are
regenerated from `src/notebook_builder.py` if you ever want to rebuild them:

```bash
python -m src.cli notebooks            # regenerate
python -m src.cli notebooks --execute  # regenerate AND execute
```

### Updating the environment later

```bash
conda env update -f environment.yml --prune
```

### Removing the environment

```bash
conda deactivate
conda env remove -n retail-sales-prediction
```

## Alternative: pip / venv

If you do not want conda:

```bash
python -m venv .venv
.\.venv\Scripts\activate           # Windows
source .venv/bin/activate          # macOS / Linux
pip install -r requirements-dev.txt
python -m src.cli pipeline -v
```

## Repository structure

```text
.
├── data/
│   ├── raw/              Original CSV (git-ignored)
│   └── processed/        Cleaned EDA snapshot
├── notebooks/
│   ├── 01_eda.ipynb      Loading, cleaning, EDA, feature inspection
│   └── 02_modeling.ipynb Preprocessing, modeling, tuning, interpretation
├── src/
│   ├── cli.py            Command-line entry point
│   ├── config.py         Paths, constants, RANDOM_STATE
│   ├── data_loader.py    Loading + cleaning helpers
│   ├── preprocessing.py  ColumnTransformer factory (post-split imputation)
│   ├── modeling.py       Pipelines + GridSearchCV
│   ├── evaluation.py     Regression metrics
│   ├── visualization.py  Reusable plotting helpers
│   ├── project_pipeline.py  End-to-end runner with logging
│   ├── notebook_builder.py  Programmatic notebook generation
│   └── logging_config.py
├── tests/                Pytest suite
├── reports/
│   ├── figures/          Exported visualizations
│   ├── model_metrics.csv
│   ├── linear_coefficients.csv
│   ├── rf_feature_importance.csv
│   └── project_summary.json
├── models/               Persisted models (git-ignored)
├── .github/workflows/    CI
├── environment.yml       conda environment
├── pyproject.toml        Packaging + tooling config
├── requirements*.txt     pip dependency lists
├── Makefile              Convenience targets
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

## Reproducibility & determinism

- All sources of randomness (train-test split, Random Forest, GridSearchCV) read
  the same seed from `src.config.RANDOM_STATE` (= 42).
- The pipeline writes a deterministic JSON summary (`reports/project_summary.json`)
  containing the chosen hyper-parameters, CV score, recommended model, top-5
  feature importances, and the top linear coefficients — useful for diffing
  between runs.

## Author

**Ashraf Sohail Alkahlout** — Computer Engineering graduate, M.Sc. candidate in
Machine Learning at the Islamic University of Gaza. Project demonstrates
end-to-end data-science engineering: cleaning, EDA, leakage-free preprocessing,
model comparison, hyperparameter tuning, and interpretation.

## License

[MIT](LICENSE)
