# watch_imminent.py
from __future__ import annotations

# --- path bootstrap (works for both src/ and flat layouts) ---
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


def _enum_member(enum_cls, names_in_priority):
    """Return the first enum member (not string) whose name matches any in names_in_priority."""
    members = getattr(enum_cls, "__members__", {})
    for name in names_in_priority:
        if name in members:
            return members[name]
    return None


def _resolve_reflectivity_param() -> DwdRadarParameter:
    """
    Pick a reflectivity parameter valid for this wetterdienst version.
    We prefer RX (composite reflectivity), then other known names.
    """
    # Prefer RX first to avoid invalid names on older versions
    pref = ["RX", "REFLECTIVITY", "COMPOSITE_REFLECTIVITY", "RADAR_REFLECTIVITY"]
    param = _enum_member(DwdRadarParameter, pref)
    if param is not None:
        return param

    # Fallback: any member containing "REFLECT"
    for name, member in DwdRadarParameter.__members__.items():
        if "REFLECT" in name:
            return member

    raise RuntimeError(
        "Could not find a reflectivity parameter in DwdRadarParameter. "
        f"Available: {list(DwdRadarParameter.__members__.keys())}"
    )


def _resolve_period_5min():
    """
    Return a 5-minute period value accepted by this version.
    Some versions use Period.MINUTE_5; others accept 'minute_5' (string).
    """
    members = getattr(Period, "__members__", {})
    if "MINUTE_5" in members:
        return members["MINUTE_5"]
    return "minute_5"


def _resolve_subset_germany():
    """
    Return a national composite subset accepted by this version.
    Names vary across versions.
    """
    pref = ["GERMANY", "NATIONAL", "COMPOSITE"]
    subset = _enum_member(DwdRadarDataSubset, pref)
    if subset is not None:
        return subset
    raise RuntimeError(
        "Could not resolve a national composite subset. "
        f"Available: {list(DwdRadarDataSubset.__members__.keys())}"
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

    values = DwdRadarValues(
        parameter=param,     # enum, not string
        start_date=start,
        end_date=end,
        subset=subset,       # enum
        period=period_5,     # enum or 'minute_5' string (both supported)
    )

    ds = values.to_xarray()         # Dataset with variable "value"
    da = ds["value"].transpose("time", "y", "x").astype(float)
    return da


def reflectivity_to_rainrate(Z: np.ndarray) -> np.ndarray:
    """Marshall–Palmer: Z=200*R^1.6  ->  R=(Z/200)^(1/1.6). Z in dBZ."""
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
