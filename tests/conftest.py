
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def synthetic_sales_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 60
    df = pd.DataFrame(
        {
            "Item_Identifier": [f"FD{i:03d}" for i in range(n)],
            "Item_Weight": rng.normal(12, 4, size=n),
            # Inconsistent categories on purpose -- cleaning should normalize them.
            "Item_Fat_Content": rng.choice(
                ["Low Fat", "low fat", "LF", "Regular", "reg"], size=n
            ),
            "Item_Visibility": rng.uniform(0, 0.3, size=n),
            "Item_Type": rng.choice(
                ["Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables"], size=n
            ),
            "Item_MRP": rng.uniform(30, 270, size=n),
            "Outlet_Identifier": rng.choice(
                ["OUT010", "OUT013", "OUT027", "OUT035"], size=n
            ),
            "Outlet_Establishment_Year": rng.choice(
                [1985, 1997, 1999, 2002, 2009], size=n
            ),
            "Outlet_Size": rng.choice(["Small", "Medium", "High", None], size=n),
            "Outlet_Location_Type": rng.choice(
                ["Tier 1", "Tier 2", "Tier 3"], size=n
            ),
            "Outlet_Type": rng.choice(
                ["Grocery Store", "Supermarket Type1", "Supermarket Type3"], size=n
            ),
            "Item_Outlet_Sales": rng.uniform(50, 8000, size=n),
        }
    )
    # Inject some nulls to exercise imputation.
    df.loc[0:5, "Item_Weight"] = np.nan
    return df
