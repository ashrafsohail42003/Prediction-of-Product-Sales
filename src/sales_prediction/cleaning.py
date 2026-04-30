"""Data-cleaning routines.

All functions are written to be *pure* — they return a new DataFrame and never
mutate the input — which keeps notebook cells replayable without side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .config import FAT_CONTENT_ALIASES


# ---------------------------------------------------------------------------
# Category standardisation
# ---------------------------------------------------------------------------
def standardize_fat_content(
    df: pd.DataFrame, column: str = "Item_Fat_Content"
) -> pd.DataFrame:
    """Collapse `{LF, low fat, Low Fat}` and `{reg, Regular}` onto canonical labels."""

    out = df.copy()
    out[column] = out[column].replace(FAT_CONTENT_ALIASES)
    return out


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows and reset the index."""

    return df.drop_duplicates().reset_index(drop=True)


def drop_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Drop the given columns, ignoring any that are not present."""

    keep = [c for c in columns if c in df.columns]
    return df.drop(columns=keep)


# ---------------------------------------------------------------------------
# Placeholder / null-mask bookkeeping
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class NullMask:
    """Snapshot of which rows were null in a given column before placeholder filling.

    We capture this so that Part 4 of the project (feature inspection) can
    reliably revert placeholders back to ``NaN`` — surgical replacement is
    always safer than ``df.replace(placeholder, np.nan)`` because the latter
    would also overwrite any organically-occurring values that happen to equal
    the placeholder.
    """

    column: str
    mask: pd.Series  # boolean indexer aligned to the DataFrame's index

    def restore(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of *df* with the captured cells set back to ``NaN``."""
        out = df.copy()
        if self.column not in out.columns:
            return out
        # Ensure the mask aligns with the provided frame (tolerate reindexing).
        aligned = self.mask.reindex(out.index, fill_value=False)
        if out[self.column].dtype.kind in {"O", "U", "S"}:
            out.loc[aligned, self.column] = np.nan
        else:
            out.loc[aligned, self.column] = np.nan
        return out


def fill_with_placeholder(
    df: pd.DataFrame,
    column: str,
    placeholder: object,
) -> tuple[pd.DataFrame, NullMask]:
    """Fill NaNs in *column* with *placeholder* and return a restoration token.

    The returned :class:`NullMask` can be passed to :meth:`NullMask.restore`
    later to revert exactly those cells back to ``NaN``.
    """

    if column not in df.columns:
        raise KeyError(f"Column {column!r} not found in DataFrame")

    mask = df[column].isna().copy()
    out = df.copy()
    out[column] = out[column].fillna(placeholder)
    return out, NullMask(column=column, mask=mask)
