.PHONY: help env install install-dev pipeline notebooks lint format test coverage clean

PYTHON ?= python

help:
	@echo "Targets:"
	@echo "  env          Create the conda environment from environment.yml"
	@echo "  install      Install runtime dependencies"
	@echo "  install-dev  Install dev dependencies (pytest, ruff, black)"
	@echo "  pipeline     Run the full training/eval pipeline (CLI)"
	@echo "  notebooks    Rebuild the EDA + modeling notebooks"
	@echo "  lint         Run ruff + black --check"
	@echo "  format       Run ruff --fix + black"
	@echo "  test         Run the pytest suite"
	@echo "  coverage     Run tests with coverage report"
	@echo "  clean        Remove caches and generated artifacts"

env:
	conda env create -f environment.yml

install:
	$(PYTHON) -m pip install -r requirements.txt

install-dev:
	$(PYTHON) -m pip install -r requirements-dev.txt

pipeline:
	$(PYTHON) -m src.cli pipeline

notebooks:
	$(PYTHON) -m src.notebook_builder

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m black --check src tests

format:
	$(PYTHON) -m ruff check --fix src tests
	$(PYTHON) -m black src tests

test:
	$(PYTHON) -m pytest

coverage:
	$(PYTHON) -m pytest --cov=src --cov-report=term-missing --cov-report=html

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -prune -exec rm -rf {} +
