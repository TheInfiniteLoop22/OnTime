"""Geographic feature helpers (pure functions, unit-testable on tiny frames)."""

from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088


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
