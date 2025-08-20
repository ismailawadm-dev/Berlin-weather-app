# watch_imminent.py
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
for p in (ROOT, SRC):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

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

try:
    from wetterdienst.provider.dwd.radar import DwdRadarPeriod  # may not exist
except Exception:
    DwdRadarPeriod = None

try:
    from wetterdienst import Period  # may not exist
except Exception:
    Period = None

try:
    from src.config import Cfg
except ModuleNotFoundError:
    from config import Cfg


def _enum_member(enum_cls, names: Iterable[str]):
    members = getattr(enum_cls, "__members__", {})
    for n in names:
        if n in members:
            return members[n]
    return None


def _resolve_reflectivity_param():
    names = list(DwdRadarParameter.__members__.keys())
    # We strongly prefer the composite reflectivity RX
    if "RX" in names:
        return DwdRadarParameter["RX"]
    # fallbacks that often exist in other builds
    for n in ["COMPOSITE_REFLECTIVITY", "REFLECTIVITY", "RADAR_REFLECTIVITY"]:
        if n in names:
            return DwdRadarParameter[n]
    # last resort: raise with a helpful message
    raise RuntimeError(
        f"No known reflectivity parameter found. Available: {names}"
    )


def _resolve_subset():
    # National composite in different builds
    return _enum_member(DwdRadarDataSubset, ["GERMANY", "NATIONAL", "COMPOSITE", "SIMPLE"])


def _resolve_period_5min():
    if DwdRadarPeriod is not None:
        m = _enum_member(DwdRadarPeriod, ["MINUTE_5", "MIN_5", "FIVE_MINUTES"])
        if m:
            return m
    if Period is not None:
        m = _enum_member(Period, ["MINUTE_5", "MIN_5", "FIVE_MINUTES"])
        if m:
            return m
    return None


def _resolve_recent():
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
    if hasattr(values, "to_xarray"):
        ds = values.to_xarray()
        da = ds["value"] if "value" in ds else next(iter(ds.data_vars.values()))
        dims = list(da.dims)
        if "time" in dims:
            spatial = [d for d in dims if d != "time"]
            if len(spatial) >= 2:
                da = da.transpose("time", spatial[-2], spatial[-1])
        return da.astype(float)

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

    param = _resolve_reflectivity_param()          # try RX first
    subset = _resolve_subset()                     # national/simple/composite
    p5    = _resolve_period_5min()                 # 5-min period if available
    prec  = _resolve_recent()                      # RECENT period as fallback

    base = dict(parameter=param, start_date=start, end_date=end)
    attempts = []

    # Order of attempts: (subset + 5min) → (subset + RECENT) → subset only → periods → bare
    if subset and p5:
        attempts.append({**base, "subset": subset, "period": p5})
    if subset and prec:
        attempts.append({**base, "subset": subset, "period": prec})
    if subset:
        attempts.append({**base, "subset": subset})
    if p5:
        attempts.append({**base, "period": p5})
    if prec:
        attempts.append({**base, "period": prec})
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
        f"last error={last_err!r}; param={getattr(param,'name',param)!r}, "
        f"subset={getattr(subset,'name',subset)!r}, period5={getattr(p5,'name',p5)!r}, "
        f"recent={getattr(prec,'name',prec)!r}, available_periods={avail_periods}"
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
