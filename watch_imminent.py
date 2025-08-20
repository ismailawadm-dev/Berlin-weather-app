# watch_imminent.py
from __future__ import annotations

# --- path bootstrap (works whether you run this file directly or via Streamlit) ---
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
for p in (ROOT, SRC):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
# --- end bootstrap ---

import time
import json
import platform
import subprocess
from datetime import datetime, timedelta

import numpy as np
import pysteps as ps
from wetterdienst.provider.dwd.radar import (
    DwdRadarValues,
    DwdRadarParameter,
    DwdRadarDataSubset,
)
from wetterdienst import Period

# Prefer src.config.Cfg if project uses a src/ layout; fall back to local config.py
try:
    from src.config import Cfg
except ModuleNotFoundError:
    from config import Cfg

STATE_FILE = ".alert_state.json"


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
    import requests  # local import to avoid a hard dependency if unused

    payload = {"text": f"*{title}*\n{message}"}
    try:
        requests.post(webhook_url, json=payload, timeout=5)
    except Exception:
        pass


# ---------- Wetterdienst compatibility helpers ----------

def _pick_rx_param() -> DwdRadarParameter:
    """Find an RX/reflectivity enum across wetterdienst versions."""
    candidates = (
        "RX",
        "RADAR_RX",
        "REFLECTIVITY",
        "RX_REFLECTIVITY",
        "PPI_RX",
        "RADAR_REFLECTIVITY",
    )
    for name in candidates:
        if hasattr(DwdRadarParameter, name):
            return getattr(DwdRadarParameter, name)
    return list(DwdRadarParameter)[0]  # last-resort fallback


def _pick_subset():
    """Find a suitable national/composite subset across versions; return None if not available."""
    candidates = (
        "GERMANY",
        "NATIONAL",
        "COMPOSITE",
        "WHOLE",
        "DE",
    )
    for name in candidates:
        if hasattr(DwdRadarDataSubset, name):
            return getattr(DwdRadarDataSubset, name)
    try:
        return list(DwdRadarDataSubset)[0]  # at least return a valid member if the enum exists
    except Exception:
        return None  # some versions might not require/offer subset at all


def fetch_radar_last_hour():
    """
    Fetch last hour of DWD reflectivity at 5-min steps as an xarray DataArray [time, y, x] in dBZ.
    Works across wetterdienst versions by probing enum names.
    """
    end = datetime.utcnow().replace(second=0, microsecond=0)
    start = end - timedelta(minutes=60)

    param = _pick_rx_param()
    subset = _pick_subset()

    # Build kwargs depending on what's available
    kwargs = dict(parameter=param, start_date=start, end_date=end, period=Period.MINUTE_5)
    if subset is not None:
        kwargs["subset"] = subset

    try:
        values = DwdRadarValues(**kwargs)
    except TypeError:
        # Some older versions may not accept "subset" or "period" as kwargs.
        # Try progressively simpler signatures.
        kwargs.pop("subset", None)
        try:
            values = DwdRadarValues(**kwargs)
        except TypeError:
            values = DwdRadarValues(parameter=param, start_date=start, end_date=end)

    ds = values.to_xarray()  # Dataset with a single data var (often "value")
    var_name = "value" if "value" in ds.data_vars else list(ds.data_vars)[0]
    return ds[var_name].transpose("time", "y", "x").astype(float)


# ---------- Conversions & utilities ----------

def reflectivity_to_rainrate(Z: np.ndarray) -> np.ndarray:
    """Marshall–Palmer: Z=200*R^1.6  ->  R=(Z/200)^(1/1.6). Z is in dBZ."""
    Z_lin = 10.0 ** (Z / 10.0)
    R = (Z_lin / 200.0) ** (1.0 / 1.6)
    R[np.isnan(R)] = 0.0
    return R


def berlin_point_index(da, lat: float, lon: float) -> tuple[int, int]:
    """Find nearest grid index to (lat,lon). If coords missing, use grid center."""
    if "latitude" in da.coords and "longitude" in da.coords:
        j = int(np.abs(da.coords["latitude"].values - lat).argmin())
        i = int(np.abs(da.coords["longitude"].values - lon).argmin())
    else:
        j = int(da.shape[1] // 2)
        i = int(da.shape[2] // 2)
    return j, i


# ---------- Main alert loop (optional when running file directly) ----------

def main_loop() -> None:
    cfg = Cfg("config.yaml")
    a = cfg["alerts"]

    if not a.get("enabled", False):
        print("Alerts disabled in config.")
        return

    lat = a.get("lat", cfg["location"]["lat"])
    lon = a.get("lon", cfg["location"]["lon"])
    lead_min = int(a.get("lead_minutes", 10))
    thr = float(a.get("rainrate_threshold_mmph", 0.2))
    cooldown = int(a.get("cooldown_minutes", 30))
    slack_url = a.get("channels", {}).get("slack_webhook", "")
    use_popup = a.get("channels", {}).get("desktop_notify", True)
    check_interval = int(a.get("check_interval_sec", 180))

    # Load last alert time (for cooldown)
    try:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
    except Exception:
        state = {"last_alert_ts": "1970-01-01T00:00:00Z"}

    while True:
        try:
            rx = fetch_radar_last_hour()  # DataArray [time,y,x] in dBZ
            R = reflectivity_to_rainrate(rx.values)  # mm/h
            oflow = ps.motion.get_method("lucaskanade")(R)
            extrap = ps.extrapolation.get_method("semilagrangian")
            steps = max(1, int(round(lead_min / 5)))
            Rf = extrap(R[-12:], oflow, steps)

            j, i = berlin_point_index(rx, lat, lon)
            rain_future = float(Rf[steps - 1, j, i])

            now = datetime.utcnow()
            last = datetime.fromisoformat(state["last_alert_ts"].replace("Z", "+00:00"))
            cooled_down = (now - last) > timedelta(minutes=cooldown)

            if rain_future >= thr and cooled_down:
                title = f"Rain in ~{l
