# watch_imminent.py
from __future__ import annotations

# --- path bootstrap: allows both flat and src/ layouts ---
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
for p in (ROOT, SRC):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
# --- end bootstrap ---

from datetime import datetime, timedelta
import numpy as np

from wetterdienst.provider.dwd.radar import (
    DwdRadarValues,
    DwdRadarParameter,
    DwdRadarDataSubset,
)
from wetterdienst import Period

# Prefer src.config.Cfg; fall back to local config.py
try:
    from src.config import Cfg
except ModuleNotFoundError:
    from config import Cfg


def _resolve_reflectivity_param() -> DwdRadarParameter:
    """
    Find a reflectivity parameter that exists in this wetterdienst version.
    Tries a list of known names, then falls back to the first enum containing 'REFLECT'.
    """
    candidates = [
        "RADAR_REFLECTIVITY",  # newer names in some versions
        "COMPOSITE_REFLECTIVITY",
        "REFLECTIVITY",
        "RX",                  # classic composite reflectivity
    ]
    for name in candidates:
        param = getattr(DwdRadarParameter, name, None)
        if param is not None:
            return param

    # Fallback: pick first enum whose name contains REFLECT
    for p in DwdRadarParameter:
        if "REFLECT" in p.name:
            return p

    # As a last resort, try RX if present at all
    if hasattr(DwdRadarParameter, "RX"):
        return DwdRadarParameter.RX

    raise RuntimeError(
        f"Could not find a reflectivity parameter in DwdRadarParameter. "
        f"Available: {[p.name for p in DwdRadarParameter]}"
    )


def _resolve_period_5min():
    """
    Return a 5-minute period/time resolution value that this wetterdienst version accepts.
    Many versions accept the enum Period.MINUTE_5; others accept the string 'minute_5'.
    """
    return getattr(Period, "MINUTE_5", "minute_5")


def _resolve_subset_germany():
    """
    Return a national composite subset/dataset enum that exists in this version.
    """
    for name in ("GERMANY", "NATIONAL", "COMPOSITE"):
        s = getattr(DwdRadarDataSubset, name, None)
        if s is not None:
            return s
    # If nothing matches, just raise with helpful info
    raise RuntimeError(
        f"Could not resolve a national composite subset. "
        f"Available: {[s.name for s in DwdRadarDataSubset]}"
    )


def fetch_radar_last_hour():
    """
    Fetch last hour of DWD composite reflectivity at ~5-min resolution.
    Returns: xarray.DataArray with dims ["time", "y", "x"] in dBZ.
    """
    end = datetime.utcnow().replace(second=0, microsecond=0)
    start = end - timedelta(minutes=60)

    param = _resolve_reflectivity_param()
    subset = _resolve_subset_germany()
    period_5 = _resolve_period_5min()

    # Some wetterdienst versions take (start_date/end_date, subset, period),
    # others accept strings for period. This call covers both.
    values = DwdRadarValues(
        parameter=param,
        start_date=start,
        end_date=end,
        subset=subset,
        period=period_5,
    )

    ds = values.to_xarray()  # Dataset with variable "value"
    da = ds["value"].transpose("time", "y", "x").astype(float)
    return da


def reflectivity_to_rainrate(Z: np.ndarray) -> np.ndarray:
    """Marshall–Palmer Z=200*R^1.6  ->  R=(Z/200)^(1/1.6). Z in dBZ."""
    Z_lin = 10.0 ** (Z / 10.0)
    R = (Z_lin / 200.0) ** (1.0 / 1.6)
    R[np.isnan(R)] = 0.0
    return R


def berlin_point_index(da, lat: float, lon: float) -> tuple[int, int]:
    """Nearest grid index to (lat, lon). If coords missing, use grid center."""
    if "latitude" in da.coords and "longitude" in da.coords:
        j = int(np.abs(da.coords["latitude"].values - lat).argmin())
        i = int(np.abs(da.coords["longitude"].values - lon).argmin())
    else:
        j = int(da.shape[1] // 2)
        i = int(da.shape[2] // 2)
    return j, i
