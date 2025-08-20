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
# Newer API sometimes exposes a dedicated period enum:
try:
    from wetterdienst.provider.dwd.radar import DwdRadarPeriod  # may exist
except Exception:
    DwdRadarPeriod = None

# Older API exposes Period in top-level package:
try:
    from wetterdienst import Period  # may exist
except Exception:
    Period = None

# Prefer src.config.Cfg; fallback to local config.py
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


def _resolve_reflectivity_param():
    """
    Pick a reflectivity-like parameter valid for this wetterdienst version.
    Prefer RX (composite reflectivity), then others containing REFLECT.
    """
    pref = ["RX", "REFLECTIVITY", "COMPOSITE_REFLECTIVITY", "RADAR_REFLECTIVITY", "HG_REFLECTIVITY"]
    m = _enum_member(DwdRadarParameter, pref)
    if m:
        return m
    # Fallback: any member containing "REFLECT"
    for name, member in DwdRadarParameter.__members__.items():
        if "REFLECT" in name:
            return member
    # Absolute last resort: first enum member
    return list(DwdRadarParameter.__members__.values())[0]


def _resolve_subset():
    """
    Choose a subset that exists in this build.
    Some versions only expose SIMPLE / POLARIMETRIC; accept those too.
    """
    return _enum_member(DwdRadarDataSubset, ["GERMANY", "NATIONAL", "COMPOSITE", "SIMPLE", "POLARIMETRIC"])


def _resolve_period_5min():
    """
    Return a 5-minute period enum if available (new or old API).
    If none can be found, return None so we can try calling without a period.
    """
    # Newer API enum?
    if DwdRadarPeriod is not None:
        pref = ["MINUTE_5", "MIN_5", "FIVE_MINUTES", "PT5M", "P5M"]
        m = _enum_member(DwdRadarPeriod, pref)
        if m:
            return m
        # Heuristic: any member name containing both "MIN" and "5"
        for name, member in DwdRadarPeriod.__members__.items():
            if "MIN" in name and "5" in name:
                return member

    # Older API enum?
    if Period is not None:
        m = _enum_member(Period, ["MINUTE_5", "MIN_5", "FIVE_MINUTES"])
        if m:
            return m

    # Nothing definite
    return None


def fetch_radar_last_hour():
    """
    Fetch last hour of DWD composite reflectivity at ~5-min resolution.
    Returns: xarray.DataArray with dims ["time", "y", "x"] in dBZ.
    """
    end = datetime.utcnow().replace(second=0, microsecond=0)
    start = end - timedelta(minutes=60)

    param = _resolve_reflectivity_param()
    subset = _resolve_subset()
    period5 = _resolve_period_5min()

    # Try combinations from most specific to least; some versions reject
    # unknown/irrelevant kwargs, so we also try without them.
    base = dict(parameter=param, start_date=start, end_date=end)
    attempts = []
    if subset is not None and period5 is not None:
        attempts.append({**base, "subset": subset, "period": period5})
    if subset is not None:
        attempts.append({**base, "subset": subset})
    if period5 is not None:
        attempts.append({**base, "period": period5})
    attempts.append(base)  # bare-minimum call

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

    raise RuntimeError(
        "Could not fetch DWD radar reflectivity; "
        f"last error={last_err!r}; "
        f"param={getattr(param,'name',param)!r}, "
        f"subset={getattr(subset,'name',subset)!r}, "
        f"period={getattr(period5,'name',period5)!r}, "
        f"available_periods="
        f"{list(getattr(DwdRadarPeriod,'__members__',{}).keys()) or list(getattr(Period,'__members__',{}).keys())}"
    )


def reflectivity_to_rainrate(Z: np.ndarray) -> np.ndarray:
    """Marshall–Palmer: Z = 200 * R^1.6  =>  R = (Z/200)^(1/1.6). Z in dBZ."""
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
