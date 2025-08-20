# watch_imminent.py
from __future__ import annotations

# ---- path bootstrap (works when run directly or via Streamlit) --------------
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
from wetterdienst.provider.dwd.radar import DwdRadarValues  # keep it minimal

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
                ["osascript", "-e", f'display notification "{message}" with title "{title}"'],
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


def _try_make_values(kwargs):
    """
    Helper: build DwdRadarValues with kwargs; ignore unknown kwargs (TypeError).
    Return xarray Dataset if ok, else None.
    """
    try:
        v = DwdRadarValues(**kwargs)
        return v.to_xarray()
    except TypeError:
        return None
    except Exception as e:
        raise e


def fetch_radar_last_hour():
    """
    Fetch last hour of DWD composite reflectivity (5-min) as DataArray [time,y,x] in dBZ.

    We avoid fragile Enum imports and try several keyword/argument variants that
    changed across wetterdienst versions.
    """
    end = datetime.utcnow().replace(second=0, microsecond=0)
    start = end - timedelta(minutes=60)

    # Parameter name conventions across versions
    param_variants = (
        {"parameter": "rx"},
        {"parameter": "RX"},
        {"parameter": "reflectivity"},
        {"parameter": "RADAR_REFLECTIVITY"},
        # older signatures where 'composite' is the parameter and 'product' selects rx
        {"parameter": "composite", "product": "rx"},
    )

    # Composite grid naming
    subset_variants = (
        {"subset": "germany"},
        {"subset": "composite"},
        {"site": "germany"},  # some versions use 'site' instead of 'subset'
        {},
    )

    # Time keys and resolution spelling
    timekey_variants = (
        {"start_date": start, "end_date": end},
        {"start_time": start, "end_time": end},
    )
    tres_variants = (
        {"time_resolution": "minute_5"},
        {"time_resolution": "minute5"},
        {"time_resolution": "5_minutes"},
        {"time_resolution": "5min"},
        {},  # some versions infer it
    )
    period_variants = (
        {"period": "recent"},
        {"period": "latest"},
        {"period": "historical"},
        {},  # no period
    )

    last_err = None

    for p in param_variants:
        for s in subset_variants:
            for t in timekey_variants:
                for tr in tres_variants:
                    for per in period_variants:
                        kwargs = {}
                        kwargs.update(p)
                        kwargs.update(s)
                        kwargs.update(t)
                        kwargs.update(tr)
                        kwargs.update(per)
                        try:
                            ds = _try_make_values(kwargs)
                            if ds is None:
                                continue
                            var = "value" if "value" in ds.variables else _first_var_name(ds)
                            da = ds[var].transpose("time", "y", "x").astype(float)
                            return da
                        except Exception as e:
                            last_err = e
                            continue

    raise RuntimeError(f"Could not fetch DWD radar reflectivity; last error was: {last_err!r}")


def reflectivity_to_rainrate(Z: np.ndarray) -> np.ndarray:
    # Marshall–Palmer: Z=200*R^1.6  ->  R=(Z/200)^(1/1.6). Z in dBZ.
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

            from datetime import datetime, timedelta
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
