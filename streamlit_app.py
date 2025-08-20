# --- path bootstrap (must be first) ---
import importlib.util, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
for p in (ROOT, SRC):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
# --- end bootstrap ---

import datetime as dt
import pandas as pd
import streamlit as st
import numpy as np
import subprocess

# Config (src/ or flat)
try:
    from src.config import Cfg
except ModuleNotFoundError:
    from config import Cfg

# Optional GEFS (herbie is not on Cloud)
try:
    from gefs import GEFSDownloader
except Exception:
    GEFSDownloader = None  # type: ignore

# Features
from make_features import summarize_ensemble

# Models (src/ or flat)
try:
    from src.prob_model import ProbModel
except ModuleNotFoundError:
    from prob_model import ProbModel
try:
    from src.quant_model import QuantileModel
except ModuleNotFoundError:
    from quant_model import QuantileModel
try:
    from src.calib import Calibrator
except ModuleNotFoundError:
    from calib import Calibrator

# Alerts / imminent radar helpers (src/ or flat)
try:
    from src.alerts.watch_imminent import (
        fetch_radar_last_hour,
        reflectivity_to_rainrate,
        berlin_point_index,
    )
except Exception:
    from watch_imminent import (
        fetch_radar_last_hour,
        reflectivity_to_rainrate,
        berlin_point_index,
    )

st.set_page_config(page_title="Berlin Rain Planning", page_icon="☔", layout="centered")
st.title("Berlin Rain Planning ☔")
st.caption("Week-2 risk + 60-minute imminent rain indicator")

cfg = Cfg("config.yaml")
lat = cfg["location"]["lat"]
lon = cfg["location"]["lon"]

st.sidebar.header("Controls")
sel_date = st.sidebar.date_input("Target date", dt.date.today())
runhour  = st.sidebar.selectbox("Model run hour (UTC)", [0, 6, 12, 18], index=0)
lead_minutes = st.sidebar.slider("Imminent alert horizon (min)", 10, 60, 60, 5)
rain_thr = st.sidebar.slider("Rain rate threshold (mm/h)", 0.1, 3.0, 0.6, 0.1)

# Try to load models (if you previously trained locally & committed them)
try:
    pm  = ProbModel.load(cfg.models_dir / "prob_model.lgbm.joblib")
    cal = Calibrator.load(cfg.models_dir / "calibrator.joblib")
    qm  = QuantileModel.load(cfg.models_dir / "quant_model.lgbm.joblib")
    models_loaded = True
except Exception:
    models_loaded = False

# --------------------- 1) Training (safe on Cloud) ---------------------
st.subheader("1) Day prediction (risk tiers)")
if not models_loaded:
    st.info(
        "Models not trained yet. You can click **Train models** to run the training "
        "entry point. On Streamlit Cloud (no `herbie`), training will skip gracefully. "
        "For real training, run locally, then commit the model files."
    )

    if st.button("Train models (3 years)"):
        with st.spinner("Launching trainer…"):
            mod = "src.train" if importlib.util.find_spec("src.train") else "train"
            proc = subprocess.run(
                [sys.executable, "-m", mod, "--start", "2022-01-01", "--end", dt.date.today().isoformat()],
                text=True,
                capture_output=True,
                check=False,
            )
        if proc.returncode == 0:
            st.success("Training finished (or was skipped if `herbie` isn’t available).")
        else:
            st.error("Training failed. See details below.")
        with st.expander("stdout"):
            st.code(proc.stdout or "(no output)")
        with st.expander("stderr"):
            st.code(proc.stderr or "(no errors)")

else:
    # --------------------- 2) Day prediction (uses models) ---------------------
    if GEFSDownloader is None:
        st.warning(
            "GEFS download is disabled in this environment (no `herbie`). "
            "Day prediction section is skipped."
        )
    else:
        try:
            dl = GEFSDownloader(cfg)
            ds = dl.stack_members(
                sel_date.isoformat(),
                runhour,
                members=cfg["features"]["members"],
                leads=cfg["features"]["leads_hours"],
                bbox=tuple(cfg["location"]["bbox"]),
            )
            X = summarize_ensemble(ds, lat, lon)
            raw_p = pm.predict_proba(X)
            p_cal = cal.predict(raw_p)
            q = qm.predict(X)

            out = X.copy()
            out["PoP"]   = p_cal
            out["P50mm"] = q[0.5]
            out["P75mm"] = q[0.75]
            out["P90mm"] = q[0.9]

            daily = out.resample("1D").agg(
                {"PoP": "mean", "P50mm": "sum", "P75mm": "sum", "P90mm": "sum"}
            )
            daily["Risk"] = pd.cut(
                daily["PoP"], [0, 0.3, 0.6, 1.0], labels=["Low", "Med", "High"], include_lowest=True
            )

            st.write("**Daily summary**")
            st.dataframe(
                daily.style.format({"PoP": "{:.0%}", "P50mm": "{:.1f}", "P75mm": "{:.1f}", "P90mm": "{:.1f}"})
            )
            st.write("**3-hour bins**")
            st.dataframe(
                out[["PoP", "P50mm", "P75mm", "P90mm"]].style.format(
                    {"PoP": "{:.0%}", "P50mm": "{:.1f}", "P75mm": "{:.1f}", "P90mm": "{:.1f}"}
                )
            )
        except Exception as e:
            st.error(f"Day prediction failed: {e}")

st.markdown("---")

# --------------------- 3) Imminent rain (next up to 60 min) ---------------------
st.subheader("2) Imminent rain (next 10–60 min)")
if st.button("Check now"):
    with st.spinner("Fetching DWD radar and nowcasting…"):
        try:
            rx = fetch_radar_last_hour()  # DataArray [time,y,x] of dBZ
            R = reflectivity_to_rainrate(rx.values)
            import pysteps as ps
            oflow  = ps.motion.get_method("lucaskanade")(R)
            extrap = ps.extrapolation.get_method("semilagrangian")
            steps  = max(1, int(round(lead_minutes / 5)))
            Rf = extrap(R[-12:], oflow, steps)

            j, i = berlin_point_index(rx, lat, lon)
            rain_now    = float(R[-1, j, i])
            rain_future = float(Rf[steps - 1, j, i])

            st.metric("Rain now (mm/h)", f"{rain_now:.2f}")
            st.metric(f"Rain in +{lead_minutes} min (mm/h)", f"{rain_future:.2f}")
            if rain_future >= rain_thr:
                st.success("Likely rain within the selected horizon. Prepare couriers/ETAs.")
            else:
                st.info("No significant rain expected in that window at Berlin center.")
        except Exception as e:
            st.warning(
                "Live radar isn’t available from this environment right now "
                "(provider returned no recent items / incompatible parameters)."
            )
            st.caption(f"Details: {e}")
