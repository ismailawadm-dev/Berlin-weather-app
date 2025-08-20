from __future__ import annotations
import os, time, json, platform, subprocess
from datetime import datetime, timedelta
import numpy as np
import xarray as xr
import pysteps as ps
from wetterdienst.provider.dwd.radar import DwdRadarValues, DwdRadarParameter, DwdRadarDataSubset
from wetterdienst import Period

from ..config import Cfg

STATE_FILE = ".alert_state.json"

def desktop_notify(title, message):
    sys = platform.system()
    try:
        if sys == "Darwin":
            subprocess.run(["osascript", "-e", f'display notification "{message}" with title "{title}"'], check=False)
        elif sys == "Linux":
            subprocess.run(["notify-send", title, message], check=False)
    except Exception:
        pass

def slack_notify(webhook_url, title, message):
    import requests
    if not webhook_url:
        return
    payload = {"text": f"*{title}*\n{message}"}
    try:
        requests.post(webhook_url, json=payload, timeout=5)
    except Exception:
        pass

def fetch_radar_last_hour():
    end = datetime.utcnow().replace(second=0, microsecond=0)
    start = end - timedelta(minutes=60)
    values = DwdRadarValues(
        parameter=DwdRadarParameter.RX,
        start_date=start,
        end_date=end,
        subset=DwdRadarDataSubset.GERMANY,
        period=Period.MINUTE_5,
    )
    ds = values.to_xarray()
    return ds["value"].transpose("time","y","x").astype(float)

def reflectivity_to_rainrate(Z):
    Z_lin = 10 ** (Z / 10.0)
    R = (Z_lin / 200.0) ** (1.0/1.6)
    R[np.isnan(R)] = 0.0
    return R

def berlin_point_index(da, lat, lon):
    if "latitude" in da.coords and "longitude" in da.coords:
        j = np.abs(da.coords["latitude"].values - lat).argmin()
        i = np.abs(da.coords["longitude"].values - lon).argmin()
    else:
        j = da.shape[1]//2; i = da.shape[2]//2
    return int(j), int(i)

def main_loop():
    cfg = Cfg()
    a = cfg["alerts"]
    if not a["enabled"]:
        print("Alerts disabled in config.")
        return
    lat = a.get("lat", cfg["location"]["lat"]); lon = a.get("lon", cfg["location"]["lon"])
    lead_min = int(a.get("lead_minutes",10))
    thr = float(a.get("rainrate_threshold_mmph",0.2))
    cooldown = int(a.get("cooldown_minutes",30))
    slack_url = a.get("channels",{}).get("slack_webhook","")
    use_popup = a.get("channels",{}).get("desktop_notify", True)

    try:
        with open(STATE_FILE,"r") as f:
            state = json.load(f)
    except Exception:
        state = {"last_alert_ts": "1970-01-01T00:00:00Z"}

    while True:
        try:
            rx = fetch_radar_last_hour()
            R = reflectivity_to_rainrate(rx.values)
            oflow = ps.motion.get_method("lucaskanade")(R)
            extrap = ps.extrapolation.get_method("semilagrangian")
            steps = max(1, int(round(lead_min/5)))
            Rf = extrap(R[-12:], oflow, steps)

            j, i = berlin_point_index(rx, cfg["location"]["lat"], cfg["location"]["lon"])
            rain10 = float(Rf[steps-1, j, i])

            now = datetime.utcnow()
            from datetime import timezone
            last = datetime.fromisoformat(state["last_alert_ts"].replace("Z","+00:00"))
            if rain10 >= thr and (now - last) > timedelta(minutes=cooldown):
                title = "Rain in ~{} minutes (Berlin)".format(lead_min)
                msg = f"Forecast rain rate ≥ {thr} mm/h. Prepare couriers & ETAs."
                if use_popup:
                    desktop_notify(title, msg)
                slack_notify(slack_url, title, msg)
                state["last_alert_ts"] = now.isoformat()+"Z"
                with open(STATE_FILE,"w") as f:
                    json.dump(state,f)
        except Exception as e:
            print("Alert loop error:", e)
        time.sleep(int(a.get("check_interval_sec",180)))

if __name__ == "__main__":
    main_loop()
