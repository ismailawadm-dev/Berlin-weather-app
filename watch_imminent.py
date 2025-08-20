# watch_imminent.py
from __future__ import annotations

# --- path bootstrap (works whether you run via Streamlit or directly) ---
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
for p in (ROOT, SRC):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
# --- end bootstrap ---

import json
import platform
import subprocess
from datetime import datetime, timedelta, timezone
from typing import List

import numpy as np
import pandas as pd
import xarray as xr

from wetterdienst.provider.dwd.radar import (
    DwdRadarValues,
    DwdRadarParameter,
    DwdRadarDataSubset,
)
# DwdRadarPeriod is optional in some builds; guard the import.
try:
    from wetterdienst.provider.dwd.radar import DwdRadarPeriod  # type: ignore
except Exception:
    DwdRadarPeriod = None  # type: ignore[assignment]

STATE_FILE = ".alert_state.json"


# --------------------------- notifications (optional) ---------------------------

def desktop_notify(title: str, message: str) -> None:
    """Best-effort desktop popup (macOS/Linux). Silently ignored if unavailable."""
    os_name = platform.system()
    try:
        if os_name == "Darwin":
            subprocess.run(
                ["osascript", "-e", f'display notification "{message}" with title "{title}"'],
                check=False,
            )
        elif os_name == "Linux":
            subprocess.run(["notify-send", title, message], check=False)
    except Exception:
        pass


def slack_notify(webhook_url: str | None, title: str, message: str) -> None:
    """Send a simple Slack message via incoming webhook."""
    if not webhook_url:
        return
    try:
        import requests
        requests.post(webhook_url, json={"text": f"*{title}*\n{message}"}, timeout=5)
    except Exception:
        pass


# ----------------------- wetterdienst compatibility helpers --------------------

def _resolve_reflectivity_param():
    """
    Pick a reflectivity product supported by the installed wetterdienst build.

    Priority:
      1) RX (classic national composite) if available
      2) Common alternatives found in recent builds (HG/PG/RW/…)
      3) Any enum member that contains 'REFLECTIVITY' but not 'VELOCITY'
    """
    members = getattr(DwdRadarParameter, "__members__", {})

    if "RX" in members:
        return members["RX"]

    for name in [
        "HG_REFLECTIVITY", "PG_REFLECTIVITY", "RW_REFLECTIVITY",
        "WN_REFLECTIVITY", "SF_REFLECTIVITY", "RE_REFLECTIVITY",
        "RQ_REFLECTIVITY", "RY_REFLECTIVITY", "RV_REFLECTIVITY",
        "DX_REFLECTIVITY", "PF_REFLECTIVITY",
        "PX_REFLECTIVITY", "PX250_REFLECTIVITY", "PZ_CAPPI",
    ]:
        if name in members:
            return members[name]

    for name, val in members.items():
        if "REFLECTIVITY" in name and "VELOCITY" not in name:
            return val

    raise RuntimeError(
        f"No known reflectivity parameter found. Available: {list(members.keys())}"
    )


def _resolve_subset():
    """
    Choose a valid national composite subset for the current build.
    Many builds expose 'SIMPLE' (and sometimes 'POLARIMETRIC').
    """
    members = getattr(DwdRadarDataSubset, "__members__", {})
    # Prefer SIMPLE if present; otherwise pick anything that sounds like composite.
    for preferred in ["COMPOSITE", "SIMPLE"]:
        if preferred in members:
            return members[preferred]
    # Fallback: first available
    if members:
        return next(iter(members.values()))
    raise RuntimeError("No DwdRadarDataSubset members available.")


def _resolve_recent_period():
    """
    Return the 'RECENT' period enum if available, otherwise a string fallback.
    Some wetterdienst versions require DwdRadarPeriod.RECENT, others accept 'recent'.
    """
    if DwdRadarPeriod is not None:
        members = getattr(DwdRadarPeriod, "__members__", {})
        if "RECENT" in members:
            return members["RECENT"]
        # Some builds only expose HISTORICAL/RECENT; if RECENT missing, string works.
    return "recent"


# --------------------------- public functions used by app ----------------------

def fetch_radar_last_hour() -> xr.DataArray:
    """
    Fetch last hour of DWD reflectivity at ~5-min steps as xarray DataArray [time,y,x] (float).
    Handles wetterdienst API differences across versions.
    """
    end = datetime.utcnow().replace(tzinfo=timezone.utc, second=0, microsecond=0)
    start = end - timedelta(minutes=60)

    param = _resolve_reflectivity_param()
    subset = _resolve_subset()
    recent = _resolve_recent_period()

    # Construct the query. Different wetterdienst versions accept different argument names.
    # We'll try the modern names first (start_time/end_time). If that fails, we retry with
    # start_date/end_date.
    last_err: Exception | None = None
    values_obj = None
    for kwargs in (
        dict(parameter=param, start_time=start, end_time=end, subset=subset, period=recent),
        dict(parameter=param, start_date=start, end_date=end, subset=subset, period=recent),
    ):
        try:
            values_obj = DwdRadarValues(**kwargs)  # type: ignore[arg-type]
            break
        except Exception as e:
            last_err = e
            continue
    if values_obj is None:
        raise RuntimeError(
            f"Could not create DwdRadarValues; last error was: {last_err!r}"
        )

    # Query items; convert each to xarray (item has to_xarray in most builds).
    items = []
    try:
        items = list(values_obj.query())
    except Exception as e:
        last_err = e

    if not items:
        raise RuntimeError(
            f"No radar items returned or they could not be decoded "
            f"(last error={last_err!r})."
        )

    frames: List[xr.DataArray] = []
    for item in items:
        try:
            # Most builds: item.to_xarray() -> Dataset with a single data var (reflectivity)
            ds = item.to_xarray()  # type: ignore[attr-defined]
            # Pick the first 2D/3D var
            varname = next((v for v in ds.data_vars if ds[v].ndim >= 2), None)
            if varname is None:
                continue
            da = ds[varname]
            # Ensure we have a 'time' dimension: if missing, synthesize from item metadata
            if "time" not in da.dims:
                # Try a few common attribute names; if not present, infer by order at 5-min steps.
                t = getattr(item, "when", None) or getattr(item, "valid_time", None)
                if t is None:
                    # Fallback: append a dummy timestamp; we'll sort later.
                    t = end
                da = da.expand_dims(time=[pd.to_datetime(t)])
            # Normalize dtype/dims
            if {"y", "x"}.issubset(set(da.dims)):
                pass
            elif {"latitude", "longitude"}.issubset(set(da.dims)):
                da = da.rename({"latitude": "y", "longitude": "x"})
            elif da.ndim == 2:
                da = da.rename({da.dims[-2]: "y", da.dims[-1]: "x"})
            frames.append(da.astype(float))
        except Exception as e:
            last_err = e
            continue

    if not frames:
        raise RuntimeError(
            "Could not decode any radar frames to xarray "
            f"(last error={last_err!r})."
        )

    da_all = xr.concat(frames, dim="time").sortby("time")
    # Make missing into 0.0 after conversion: we keep NaNs (handled later in conversion to rainrate)
    return da_all


def reflectivity_to_rainrate(Z: np.ndarray) -> np.ndarray:
    """
    Marshall–Palmer:  Z = 200 * R^1.6   ->   R = (Z/200)^(1/1.6)
    Input Z is reflectivity in dBZ. Output R is mm/h.
    """
    Z_lin = 10.0 ** (Z / 10.0)
    R = (Z_lin / 200.0) ** (1.0 / 1.6)
    R[np.isnan(R)] = 0.0
    return R


def berlin_point_index(da: xr.DataArray, lat: float, lon: float) -> tuple[int, int]:
    """Nearest grid index to (lat, lon). If coords missing, use grid center."""
    if "latitude" in da.coords and "longitude" in da.coords:
        j = int(np.abs(da.coords["latitude"].values - lat).argmin())
        i = int(np.abs(da.coords["longitude"].values - lon).argmin())
    elif "y" in da.dims and "x" in da.dims:
        j = int(da.sizes.get("y", 1) // 2)
        i = int(da.sizes.get("x", 1) // 2)
    else:
        j = int(da.shape[-2] // 2)
        i = int(da.shape[-1] // 2)
    return j, i
