"""Order-composition, product and payment features (FEATURE_SPEC section 4)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def group_category(category: pd.Series, groups: dict[str, list[str]]) -> pd.Series:
    """Map English category names to coarse groups.

    Missing category -> 'unknown' (missing product data can itself be informative);
    a category not listed in the mapping -> 'other'.
    """
    lookup = {cat: grp for grp, cats in groups.items() for cat in cats}
    mapped = category.map(lookup)
    return mapped.where(category.notna(), "unknown").fillna("other")


def add_product_features(df: pd.DataFrame, category_groups: dict[str, list[str]]) -> pd.DataFrame:
    """Return composition/product/payment feature columns for `df` (same index)."""
    out = pd.DataFrame(index=df.index)
    out["n_items"] = df["n_items"].astype(int)
    out["total_price"] = df["total_price"]
    out["total_freight"] = df["total_freight"]
    # Freight relative to price: cheap goods with expensive shipping are typically bulky/remote.
    out["freight_ratio"] = (df["total_freight"] / df["total_price"]).where(df["total_price"] > 0)
    out["total_weight_kg"] = df["total_weight_g"] / 1000
    out["total_volume_l"] = df["total_volume_cm3"] / 1000
    out["max_item_weight_kg"] = df["max_item_weight_g"] / 1000
    out["category_group"] = group_category(df["top_item_category"], category_groups)
    out["product_photos_qty_mean"] = df["product_photos_qty_mean"]
    out["payment_type"] = df["payment_type_dominant"].fillna("unknown")
    out["payment_installments_max"] = df["payment_installments_max"].astype(float)
    return out.replace([np.inf, -np.inf], np.nan)
