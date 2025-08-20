# watch_imminent.py
from __future__ import annotations

# -------- path bootstrap (works when run directly or via Streamlit) ----------
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
for p in (ROOT, SRC):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
# -----------------------------------------------------------------------------

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
)
from wetterdienst import Period

# Prefer src.config.Cfg if a src/ layout exists; otherwise use local config.py
try:
    from src.config import Cfg
except ModuleNotFoundError:
    from config import Cfg

STATE_FILE = ".alert_state.json"


def desktop_notify(title: str, message: str) -> None:
    """Best-effort desktop popup (macOS/Linux)."""
    os_name = platform.system()
    try:
        if os_name == "Darwin":
            # macOS Notification Center
            subprocess.run(
                ["osascript", "-e", f'display notification "{message}" with title "{title}"'],
                check=False,
            )
        elif os_name == "Linux":
            subprocess.run(["notify-send", title, message], check=False)
    except Exception:
        pass  # non-fatal


def slack_notify(webhook_url: str | None, title: str, message: str) -> None:
    """Send a simple Slack message via incoming webhook (optional)."""
    if not webhook_url:
        return
    try:
        import requests
        requests.post(webhook_url, json={"text": f"*{title}*\n{message}"}, timeout=5)
    except Exception:
        pass  # non-fatal


def _first_var_name(ds):
    """Return the first data variable name from an xarray Dataset."""
    return next(iter(ds.data_vars))


def fetch_radar_last_hour():
    """
    Fetch the last hour of DWD RX reflectivity at 5-minute steps as an
    xarray DataArray with dims [time, y, x], values in dBZ.

    NOTE: We do NOT pass a 'subset' argument to avoid enum differences
    across wetterdienst versions. The default covers Germany.
    """
    end = datetime.utcnow().replace(second=0, microsecond=0)
    start = end - timedelta(minutes=60)

    values = DwdRadarValues(
        parameter=DwdRadarParameter.RX,
        start_date=start,
        end_date=end,
        period=Period.MINUTE_5,
    )
    ds = values.to_xarray()  # Dataset, variable name can vary
    var = "value" if "value" in ds.variables else _first_var_name(ds)
    da = ds[var].transpose("time", "y", "x").astype(float)
    return da


def reflectivity_to_rainrate(Z: np.ndarray) -> np.ndarray:
    """Marshall–Palmer: Z=200*R^1.6  ->  R=(Z/200)^(1/1.6). Z in dBZ."""
    Z_lin = 10.0 ** (Z / 10.0)
    R = (Z_lin / 200.0) ** (1.0 / 1.6)
    R[np.isnan(R)] = 0.0
    return R


def berlin_point_index(da, lat: float, lon: float) -> tuple[int, int]:
    """Find nearest grid index to (lat,lon). If coords missing, use grid center."""
    if hasattr(da, "coords") and "latitude" in da.coords and "longitude" in da.coords:
        j = int(np.abs(np.asarray(da.coords["latitude"].values) - lat).argmin())
        i = int(np.abs(np.asarray(da.coords["longitude"].values) - lon).argmin())
    else:
        j = int(da.shape[1] // 2)
        i = int(da.shape[2] // 2)
    return j, i


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

    # cooldown state
    try:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
    except Exception:
        state = {"last_alert_ts": "1970-01-01T00:00:00Z"}

    while True:
        try:
            # 1) fetch & nowcast
            rx = fetch_radar_last_hour()           # DataArray [time,y,x] in dBZ
            R = reflectivity_to_rainrate(rx.values)  # mm/h

            oflow = ps.motion.get_method("lucaskanade")(R)
            extrap = ps.extrapolation.get_method("semilagrangian")
            steps = max(1, int(round(lead_min / 5)))
            Rf = extrap(R[-12:], oflow, steps)     # last hour -> next steps

            # 2) probe Berlin
            j, i = berlin_point_index(rx, lat, lon)
            rain_future = float(Rf[steps - 1, j, i])

            # 3) cooldown gating + notify
            now = datetime.utcnow()
            last = datetime.fromisoformat(state["last_alert_ts"].replace("Z", "+00:00"))
            cooled_down = (now - last) > timedelta(minutes=cooldown)

            if rain_future >= thr and cooled_down:
                title = f"Rain in ~{lead_min} minutes (Berlin)"
                msg = f"Forecast rain rate ≥ {thr:.2f} mm/h. Prepare couriers & ETAs."
                if use_popup:
                    desktop_notify(title, msg)
                slack_notify(slack_url, title, msg)
                state["last_alert_ts"] = now.isoformat() + "Z"
                try:
                    with open(STATE_FILE, "w") as f:
                        json.dump(state, f)
                except Exception:
                    pass

        except Exception as e:
            print("Alert loop error:", e)

        time.sleep(check_interval)


if __name__ == "__main__":
    main_loop()
