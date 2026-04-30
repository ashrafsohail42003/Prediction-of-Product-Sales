"""Project-wide constants.

Centralising these values avoids magic strings scattered across the codebase
and makes it trivial to swap the data source or retrain on a new schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
# Google Drive file id for `sales_predictions_2023.csv`.
DATASET_GDRIVE_FILE_ID: Final[str] = "1syH81TVrbBsdymLT_jl2JIf6IjPXtSQw"
DATASET_DIRECT_URL: Final[str] = (
    f"https://drive.google.com/uc?export=download&id={DATASET_GDRIVE_FILE_ID}"
)
DATASET_FILENAME: Final[str] = "sales_predictions_2023.csv"

# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------
# Resolve the repo root relative to this file: <repo>/src/sales_prediction/config.py
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DATA_DIR: Final[Path] = REPO_ROOT / "data"
RAW_DIR: Final[Path] = DATA_DIR / "raw"
FIGURES_DIR: Final[Path] = REPO_ROOT / "reports" / "figures"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE: Final[int] = 42
TEST_SIZE: Final[float] = 0.20

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
TARGET: Final[str] = "Item_Outlet_Sales"
ID_COLUMNS: Final[tuple[str, ...]] = ("Item_Identifier",)

NUMERIC_FEATURES: Final[tuple[str, ...]] = (
    "Item_Weight",
    "Item_Visibility",
    "Item_MRP",
    "Outlet_Age",
)
CATEGORICAL_FEATURES: Final[tuple[str, ...]] = (
    "Item_Fat_Content",
    "Item_Type",
    "Outlet_Identifier",
    "Outlet_Size",
    "Outlet_Location_Type",
    "Outlet_Type",
)

# ---------------------------------------------------------------------------
# Domain knowledge
# ---------------------------------------------------------------------------
# Reference year used when deriving `Outlet_Age`. We deliberately freeze this
# value so that reruns remain reproducible (feature drift would otherwise occur
# as real-time clocks advance).
REFERENCE_YEAR: Final[int] = 2013

# Mapping used to collapse the noisy `Item_Fat_Content` variants into the two
# canonical labels defined by the business.
FAT_CONTENT_ALIASES: Final[dict[str, str]] = {
    "LF": "Low Fat",
    "low fat": "Low Fat",
    "reg": "Regular",
}


@dataclass(frozen=True, slots=True)
class SchemaMetadata:
    """Lightweight container that mirrors the official data dictionary."""

    name: str
    description: str


DATA_DICTIONARY: Final[dict[str, SchemaMetadata]] = {
    "Item_Identifier": SchemaMetadata(
        "Item_Identifier", "Unique product ID."
    ),
    "Item_Weight": SchemaMetadata(
        "Item_Weight", "Weight of the product."
    ),
    "Item_Fat_Content": SchemaMetadata(
        "Item_Fat_Content", "Whether the product is low-fat or regular."
    ),
    "Item_Visibility": SchemaMetadata(
        "Item_Visibility",
        "Percentage of total display area of all products in a store allocated to this product.",
    ),
    "Item_Type": SchemaMetadata(
        "Item_Type", "Category the product belongs to."
    ),
    "Item_MRP": SchemaMetadata(
        "Item_MRP", "Maximum Retail Price (list price) of the product."
    ),
    "Outlet_Identifier": SchemaMetadata(
        "Outlet_Identifier", "Unique store ID."
    ),
    "Outlet_Establishment_Year": SchemaMetadata(
        "Outlet_Establishment_Year", "Year in which the store was established."
    ),
    "Outlet_Size": SchemaMetadata(
        "Outlet_Size", "Size of the store in terms of ground area covered."
    ),
    "Outlet_Location_Type": SchemaMetadata(
        "Outlet_Location_Type", "Type of area in which the store is located."
    ),
    "Outlet_Type": SchemaMetadata(
        "Outlet_Type", "Whether the outlet is a grocery store or some sort of supermarket."
    ),
    "Item_Outlet_Sales": SchemaMetadata(
        "Item_Outlet_Sales", "Sales of the product in the particular store (target)."
    ),
}
