"""Deterministic feature engineering.

Kept tiny: we only derive ``Outlet_Age`` at the moment. The function is still
worth extracting so that notebook / preprocessing code both agree on the
*exact* definition (and reference year) of the feature.
"""

from __future__ import annotations

import pandas as pd

from .config import REFERENCE_YEAR


def add_outlet_age(
    df: pd.DataFrame,
    *,
    reference_year: int = REFERENCE_YEAR,
    source_column: str = "Outlet_Establishment_Year",
    target_column: str = "Outlet_Age",
) -> pd.DataFrame:
    """Append a ``target_column`` = ``reference_year - source_column``.

    The reference year is pinned (see :mod:`sales_prediction.config`) so
    retraining months apart yields identical features — i.e. we avoid the
    silent feature drift that ``datetime.now().year`` would introduce.
    """

    if source_column not in df.columns:
        raise KeyError(f"{source_column!r} not found; cannot derive {target_column!r}")

    out = df.copy()
    out[target_column] = (reference_year - out[source_column]).astype("int16")
    return out
