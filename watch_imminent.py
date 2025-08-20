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
from wetterdienst.provider.dwd.radar import DwdRadarValues  # note: no enums imported
from wetterdienst import Period

# Prefer src.config.Cfg if a src/ layout exists; otherwise use local config.py
try:
    from src.config import Cfg
except ModuleNotFoundError:
    from config import Cfg

STATE_FILE = ".alert_state.json"


def desktop_notify(title: str, message: str) -> None:
    os_name = platform.system()
    try:
        if os_name == "Darwin":
            subprocess.run(
                ["osascript", "-e", f'display notification \"{message}\" with title \"{title}\"'],
                check=False,
            )
        elif os_name == "Linux":
            subprocess.run(["notify-send", title, message], check=False)
    except Exception:
        pass  # best-effort


def slack_notify(webhook_url: str | None, title: str, message: str) -> None:
    if not webhook_url:
        return
    try:
        import requests
        requests.post(webhook_url, json={"text": f"*{title}*\n{message}"}, timeout=5)
    except Exception:
        pass  # best-effort


def _first_var_name(ds):
    return next(iter(ds.data_vars))


def fetch_radar_last_hour(debug: bool = False):
    """
    Fetch last hour of DWD composite reflectivity (5-min) as DataArray [time,y,x] in dBZ.

    We avoid fragile enums and try a few parameter/subset spellings that differ
    across wetterdienst versions.
    """
    end = datetime.utcnow().replace(second=0, microsecond=0)
    start = end - timedelta(minutes=60)

    # Try multiple spellings for backwards/forwards compatibility
    param_opts = ("rx", "RX", "reflectivity", "RADAR_REFLECTIVITY")
    subset_opts = ("germany", "composite", None)  # some versions use "germany", others "composite"
    period_opts = (Period.MINUTE_5, "minute_5")   # accept enum or string

    last_err = None
    for param in param_opts:
        for subset in subset_opts:
            for period in period_opts:
                try:
                    kwargs = dict(parameter=param, start_date=start, end_date=end, period=period)
                    if subset is not None:
                        kwargs["subset"] = subset
                    values = DwdRadarValues(**kwargs)
                    ds = values.to_xarray()
                    var = "value" if "value" in ds.variables else _first_var_name(ds)
                    da = ds[var].transpose("time", "y", "x").astype(float)
                    if debug:
                        print(f"[watch_imminent] DWD radar OK with param={param}, subset={subset}, period={period}")
                    return da
                except Exception as e:
                    last_err = e
                    continue

    raise RuntimeError(f"Could not fetch DWD radar reflectivity; last error was: {last_err!r}")


def reflectivity_to_rainrate(Z: np.ndarray) -> np.ndarray:
    Z_lin = 10.0 ** (Z / 10.0)
    R = (Z_lin / 200.0) ** (1.0 / 1.6)
    R[np.isnan(R)] = 0.0
    return R


def berlin_point_index(da, lat: float, lon: float) -> tuple[int, int]:
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

    try:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
    except Exception:
        state = {"last_alert_ts": "1970-01-01T00:00:00Z"}

    while True:
        try:
            rx = fetch_radar_last_hour()
            R = reflectivity_to_rainrate(rx.values)

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
