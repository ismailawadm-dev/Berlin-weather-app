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
import io
from typing import Optional, Iterable

import numpy as np
import xarray as xr

from wetterdienst.provider.dwd.radar import (
    DwdRadarValues,
    DwdRadarParameter,
    DwdRadarDataSubset,
)

# Some builds have a dedicated radar period enum, others reuse top-level Period
try:
    from wetterdienst.provider.dwd.radar import DwdRadarPeriod  # may not exist
except Exception:
    DwdRadarPeriod = None

try:
    from wetterdienst import Period  # may not exist or may only have RECENT/HISTORICAL
except Exception:
    Period = None

# Prefer src.config.Cfg; fallback to local config.py
try:
    from src.config import Cfg
except ModuleNotFoundError:
    from config import Cfg


def _enum_member(enum_cls, names_in_priority: Iterable[str]):
    members = getattr(enum_cls, "__members__", {})
    for name in names_in_priority:
        if name in members:
            return members[name]
    return None


def _resolve_reflectivity_param():
    pref = ["RX", "COMPOSITE_REFLECTIVITY", "RADAR_REFLECTIVITY", "REFLECTIVITY", "HG_REFLECTIVITY"]
    m = _enum_member(DwdRadarParameter, pref)
    if m:
        return m
    for name, member in DwdRadarParameter.__members__.items():
        if "REFLECT" in name:
            return member
    return list(DwdRadarParameter.__members__.values())[0]


def _resolve_subset():
    return _enum_member(DwdRadarDataSubset, ["GERMANY", "NATIONAL", "COMPOSITE", "SIMPLE", "POLARIMETRIC"])


def _resolve_period_5min():
    if DwdRadarPeriod is not None:
        pref = ["MINUTE_5", "MIN_5", "FIVE_MINUTES", "PT5M", "P5M"]
        m = _enum_member(DwdRadarPeriod, pref)
        if m:
            return m
        for name, member in DwdRadarPeriod.__members__.items():
            if "MIN" in name and "5" in name:
                return member
    if Period is not None:
        m = _enum_member(Period, ["MINUTE_5", "MIN_5", "FIVE_MINUTES"])
        if m:
            return m
    return None


def _resolve_recent_period():
    if DwdRadarPeriod is not None:
        m = _enum_member(DwdRadarPeriod, ["RECENT"])
        if m:
            return m
    if Period is not None:
        m = _enum_member(Period, ["RECENT"])
        if m:
            return m
    return None


def _values_to_dataarray(values: DwdRadarValues) -> xr.DataArray:
    # Fast path
    if hasattr(values, "to_xarray"):
        ds = values.to_xarray()
        da = ds["value"] if "value" in ds else next(iter(ds.data_vars.values()))
        dims = list(da.dims)
        if "time" in dims:
            spatial = [d for d in dims if d != "time"]
            if len(spatial) >= 2:
                da = da.transpose("time", spatial[-2], spatial[-1])
        return da.astype(float)

    # Fallback: iterate query() items
    arrays = []
    for item in values.query():
        ds = None
        if hasattr(item, "to_xarray"):
            try:
                ds = item.to_xarray()
            except Exception:
                ds = None
        if ds is None:
            try:
                with item.open() as fobj:
                    try:
                        ds = xr.open_dataset(fobj)
                    except Exception:
                        data = fobj.read()
                        ds = xr.open_dataset(io.BytesIO(data))
            except Exception:
                ds = None
        if ds is None:
            continue

        if "value" in ds:
            da_i = ds["value"]
        elif len(ds.data_vars):
            da_i = next(iter(ds.data_vars.values()))
        else:
            continue

        if "time" not in da_i.dims:
            da_i = da_i.expand_dims("time")
        arrays.append(da_i)

    if not arrays:
        raise RuntimeError("No radar items returned or they could not be decoded.")

    da = xr.concat(arrays, dim="time")
    dims = list(da.dims)
    if "time" in dims:
        spatial = [d for d in dims if d != "time"]
        if len(spatial) >= 2:
            da = da.transpose("time", spatial[-2], spatial[-1])
    return da.astype(float)


def fetch_radar_last_hour():
    end = datetime.utcnow().replace(second=0, microsecond=0)
    start = end - timedelta(minutes=60)

    param = _resolve_reflectivity_param()
    subset = _resolve_subset()
    period5 = _resolve_period_5min()
    period_recent = _resolve_recent_period()

    base = dict(parameter=param, start_date=start, end_date=end)

    attempts = []
    if subset is not None and period5 is not None:
        attempts.append({**base, "subset": subset, "period": period5})
    if subset is not None and period_recent is not None:
        attempts.append({**base, "subset": subset, "period": period_recent})
    if subset is not None:
        attempts.append({**base, "subset": subset})
    if period5 is not None:
        attempts.append({**base, "period": period5})
    if period_recent is not None:
        attempts.append({**base, "period": period_recent})
    attempts.append(base)

    last_err: Optional[Exception] = None
    for kwargs in attempts:
        try:
            values = DwdRadarValues(**kwargs)
            da = _values_to_dataarray(values)
            return da.transpose("time", ..., ...)
        except Exception as e:
            last_err = e
            continue

    avail_periods = (
        list(getattr(DwdRadarPeriod, "__members__", {}).keys())
        or list(getattr(Period, "__members__", {}).keys())
    )
    raise RuntimeError(
        "Could not fetch DWD radar reflectivity; "
        f"last error={last_err!r}; "
        f"param={getattr(param,'name',param)!r}, "
        f"subset={getattr(subset,'name',subset)!r}, "
        f"period5={getattr(period5,'name',period5)!r}, "
        f"recent={getattr(period_recent,'name',period_recent)!r}, "
        f"available_periods={avail_periods}"
    )


def reflectivity_to_rainrate(Z: np.ndarray) -> np.ndarray:
    Z_lin = 10.0 ** (Z / 10.0)
    R = (Z_lin / 200.0) ** (1.0 / 1.6)
    R[np.isnan(R)] = 0.0
    return R


def berlin_point_index(da, lat: float, lon: float) -> tuple[int, int]:
    if "latitude" in da.coords and "longitude" in da.coords:
        j = int(np.abs(da.coords["latitude"].values - lat).argmin())
        i = int(np.abs(da.coords["longitude"].values - lon).argmin())
    else:
        j = int(da.shape[1] // 2)
        i = int(da.shape[2] // 2)
    return j, i
