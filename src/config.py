from __future__ import annotations

from pathlib import Path
from typing import Final
RANDOM_STATE: Final[int] = 42
TEST_SIZE: Final[float] = 0.25
TARGET: Final[str] = "Item_Outlet_Sales"
# Features dropped before model training (high-cardinality identifiers, etc.)
DROP_FEATURES: Final[tuple[str, ...]] = ("Item_Identifier",)
MISSING_CATEGORY: Final[str] = "Missing"
MISSING_NUMERIC: Final[int] = -1
ROOT_DIR: Final[Path] = Path(__file__).resolve().parents[1]

DATA_DIR: Final[Path] = ROOT_DIR / "data"
RAW_DATA_PATH: Final[Path] = DATA_DIR / "raw" / "sales_predictions.csv"
PROCESSED_DATA_PATH: Final[Path] = DATA_DIR / "processed" / "cleaned_sales.csv"

REPORTS_DIR: Final[Path] = ROOT_DIR / "reports"
FIGURES_DIR: Final[Path] = REPORTS_DIR / "figures"

MODELS_DIR: Final[Path] = ROOT_DIR / "models"
BEST_MODEL_PATH: Final[Path] = MODELS_DIR / "best_rf_model.joblib"
DATA_DICTIONARY: Final[dict[str, str]] = {
    "Item_Identifier": "Unique product ID",
    "Item_Weight": "Weight of product",
    "Item_Fat_Content": "Whether the product is low fat or regular",
    "Item_Visibility": (
        "The percentage of total display area of all products in a store "
        "allocated to the particular product"
    ),
    "Item_Type": "The category to which the product belongs",
    "Item_MRP": "Maximum Retail Price (list price) of the product",
    "Outlet_Identifier": "Unique store ID",
    "Outlet_Establishment_Year": "The year in which the store was established",
    "Outlet_Size": "The size of the store in terms of ground area covered",
    "Outlet_Location_Type": "The type of area in which the store is located",
    "Outlet_Type": "Whether the outlet is a grocery store or some sort of supermarket",
    "Item_Outlet_Sales": (
        "Sales of the product in the particular store. This is the target "
        "variable to be predicted."
    ),
}


def ensure_project_dirs() -> None:
    """Create the project's output directories idempotently."""
    for path in (
        DATA_DIR / "raw",
        DATA_DIR / "processed",
        FIGURES_DIR,
        MODELS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
