# Contributing

Thanks for taking a look at this project. The notes below cover the local setup,
testing, and the conventions used in this repository.

## Local setup

The recommended workflow uses **conda** (Anaconda or Miniconda):

```bash
conda env create -f environment.yml
conda activate retail-sales-prediction
pip install -e ".[dev]"
```

Place the original dataset at `data/raw/sales_predictions.csv`.

## Running things

```bash
# Full ML pipeline (CLI):
python -m src.cli pipeline -v

# Rebuild notebooks from src/notebook_builder.py:
python -m src.cli notebooks
python -m src.cli notebooks --execute   # runs them too

# Tests + coverage:
pytest --cov=src --cov-report=term-missing

# Lint + format:
ruff check src tests
black src tests
```

A `Makefile` wraps the same commands (`make test`, `make lint`, `make pipeline`).

## Conventions

- **Reproducibility:** all randomness is seeded from `src.config.RANDOM_STATE`.
- **No data leakage:** missing-value imputation is implemented inside the
  scikit-learn `Pipeline` so it always runs *after* the train-test split.
- **No hardcoded paths:** anything path-related lives in `src/config.py`.
- **Public API:** consumers should import from `src` (not from internal modules)
  whenever possible — see `src/__init__.py`.
- **Code style:** ruff + black, 100-char line length, Python 3.10+.

## Pull requests

1. Branch from `master`.
2. Run `make lint` and `make test` before pushing.
3. Update or add tests when changing behavior in `src/`.
4. Keep notebooks reproducible — they are regenerated from `notebook_builder.py`.
