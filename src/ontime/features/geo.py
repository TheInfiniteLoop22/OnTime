"""Logistics-geometry features (FEATURE_SPEC section 1). Pure functions on a DataFrame.

All inputs are known at order approval: zip centroids of the primary seller and the customer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088

# IBGE macro-regions. Why: states are too granular for rare ones; regions are a stable grouping
# that captures the big north/north-east vs south-east logistics divide.
STATE_REGION: dict[str, str] = {
    **dict.fromkeys(["AC", "AP", "AM", "PA", "RO", "RR", "TO"], "N"),
    **dict.fromkeys(["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"], "NE"),
    **dict.fromkeys(["DF", "GO", "MT", "MS"], "CO"),
    **dict.fromkeys(["ES", "MG", "RJ", "SP"], "SE"),
    **dict.fromkeys(["PR", "RS", "SC"], "S"),
}


def haversine_km(
    lat1: pd.Series | np.ndarray | float,
    lng1: pd.Series | np.ndarray | float,
    lat2: pd.Series | np.ndarray | float,
    lng2: pd.Series | np.ndarray | float,
) -> np.ndarray:
    """Great-circle distance in km between two points (vectorized).

    Why haversine: straight-line distance on a sphere is a good, cheap proxy for shipping
    distance when we only know zip-prefix centroids, not road networks.
    """
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(lng2) - np.radians(lng1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def add_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the geo feature columns for `df` (same index).

    Requires: seller/customer lat, lng, state, *_geo_imputed, n_distinct_sellers.
    """
    out = pd.DataFrame(index=df.index)
    out["distance_km"] = haversine_km(
        df["seller_lat"], df["seller_lng"], df["customer_lat"], df["customer_lng"]
    )
    out["log_distance_km"] = np.log1p(out["distance_km"])
    out["same_state"] = (df["seller_state"] == df["customer_state"]).astype(int)
    out["customer_region"] = df["customer_state"].map(STATE_REGION).fillna("unknown")
    out["seller_region"] = df["seller_state"].map(STATE_REGION).fillna("unknown")
    out["geo_imputed"] = (df["seller_geo_imputed"] | df["customer_geo_imputed"]).astype(int)
    out["n_distinct_sellers"] = df["n_distinct_sellers"].astype(int)
    return out
