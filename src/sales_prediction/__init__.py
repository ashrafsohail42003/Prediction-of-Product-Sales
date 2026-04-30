"""sales_prediction — end-to-end retail product-sales analysis toolkit.

This package exposes a small, testable set of utilities that are consumed by
the project notebook. The public surface is intentionally narrow:

* :mod:`sales_prediction.config` — dataset metadata and column groupings
* :mod:`sales_prediction.data` — dataset download / loading helpers
* :mod:`sales_prediction.cleaning` — category standardization & null masking
* :mod:`sales_prediction.eda` — reusable univariate / bivariate plotters
* :mod:`sales_prediction.features` — deterministic feature engineering
* :mod:`sales_prediction.preprocessing` — sklearn ColumnTransformer factory

Keeping this logic in a package (rather than inlined in the notebook only)
lets us unit-test it and reuse it across downstream modelling experiments.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "__version__",
]
