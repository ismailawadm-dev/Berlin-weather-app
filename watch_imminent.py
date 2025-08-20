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

# Prefer src.config.Cfg; fall back to local config.py (keeps same behavior as streamlit_app)
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
    Prefer RX (composite reflectivity), then other known names.
    """
    preferred = ["RX", "REFLECTIVITY", "COMPOSITE_REFLECTIVITY", "RADAR_REFLECTIVITY"]
    param = _enum_member(DwdRadarParameter, preferred)
    if param is not None:
        return param

    # Fallback: any member containing "REFLECT"
    for name, member in DwdRadarParameter.__members__.items():
        if "REFLECT" in name:
            return member

    raise RuntimeError(
        "No reflectivity-like parameter found in DwdRadarParameter. "
        f"Available: {list(DwdRadarParameter.__members__.keys())}"
    )


def _resolve_period_5min():
    """
    Return a 5-minute period value accepted by this version.
    Some versions use Period.MINUTE_5; others accept the string 'minute_5'.
    """
    members = getattr(Period, "__members__", {})
    if "MINUTE_5" in members:
        return members["MINUTE_5"]
    return "minute_5"  # string fallback accepted by many versions


def _resolve_subset():
    """
    Return a national/composite subset if present; otherwise be liberal:
    accept SIMPLE or POLARIMETRIC if that is what this version exposes.
    If nothing looks right, return None and we'll try constructing without subset.
    """
    members = getattr(DwdRadarDataSubset, "__members__", {})
    for name in ["GERMANY", "NATIONAL", "COMPOSITE", "SIMPLE", "POLARIMETRIC"]:
        if name in members:
            return members[name]
    return None  # we'll try without subset entirely


def fetch_radar_last_hour():
    """
    Fetch last hour of DWD composite reflectivity at ~5-min resolution.
    Returns: xarray.DataArray with dims ["time", "y", "x"] in dBZ.
    """
    end = datetime.utcnow().replace(second=0, microsecond=0)
    start = end - timedelta(minutes=60)

    param = _resolve_reflectivity_param()
    period_5 = _resolve_period_5min()
    subset = _resolve_subset()

    # Try with subset (if we found one), then without subset.
    attempts = []
    base_kwargs = dict(
        parameter=param,
        start_date=start,
        end_date=end,
        period=period_5,
    )
    if subset is not None:
        attempts.append({**base_kwargs, "subset": subset})
    attempts.append(base_kwargs)  # try without subset too

    last_err = None
    for kwargs in attempts:
        try:
            values = DwdRadarValues(**kwargs)
            ds = values.to_xarray()  # Dataset with variable "value"
            da = ds["value"].transpose("time", "y", "x").astype(float)
            return da
        except Exception as e:
            last_err = e
            continue

    # If we get here, all attempts failed; surface a helpful error
    raise RuntimeError(
        "Could not fetch DWD radar reflectivity; "
        f"last error was: {last_err!r}; "
        f"param={getattr(param,'name',param)!r}, "
        f"subset_tried={getattr(subset,'name',subset)!r}, "
        f"period={getattr(period_5,'name',period_5)!r}, "
        f"available_subsets={list(getattr(DwdRadarDataSubset,'__members__',{}).keys())}"
    )


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
