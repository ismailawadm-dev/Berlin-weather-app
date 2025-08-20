# --- path bootstrap (must be first) ---
import importlib.util, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
for p in (ROOT, SRC):
    p = str(p)
    if p not in sys.path:
        sys.path.insert(0, p)
# --- end bootstrap ---

import datetime as dt
import pandas as pd
import streamlit as st
import numpy as np

# Config (with fallback to flat layout)
try:
    from src.config import Cfg
except ModuleNotFoundError:
    from config import Cfg

# GEFS live download (optional on Cloud)
try:
    from gefs import GEFSDownloader
except Exception:
    GEFSDownloader = None

from make_features import summarize_ensemble

# Models (support both src/ and flat)
try:
    from src.modeling.prob_model import ProbModel
except ModuleNotFoundError:
    from prob_model import ProbModel

try:
    from src.modeling.quant_model import QuantileModel
except ModuleNotFoundError:
    from quant_model import QuantileModel

try:
    from src.modeling.calib import Calibrator
except ModuleNotFoundError:
    from calib import Calibrator

# Imminent-rain tools (robust fallback)
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

# ---------- Sidebar ----------
st.sidebar.header("Controls")
sel_date = st.sidebar.date_input("Target date", dt.date.today())
runhour = st.sidebar.selectbox("Model run hour (UTC)", [0, 6, 12, 18], index=0)
lead_minutes = st.sidebar.slider("Imminent alert horizon (min)", 10, 60, 60, 5)
rain_thr = st.sidebar.slider("Rain rate threshold (mm/h)", 0.1, 2.0, 0.6, 0.1)

# ---------- Load models ----------
try:
    pm = ProbModel.load(cfg.models_dir / "prob_model.lgbm.joblib")
    cal = Calibrator.load(cfg.models_dir / "calibrator.joblib")
    qm = QuantileModel.load(cfg.models_dir / "quant_model.lgbm.joblib")
    models_loaded = True
except Exception:
    models_loaded = False

# ===============================================================
# 1) Day prediction (risk tiers) — prefers precomputed file
# ===============================================================
st.subheader("1) Day prediction (risk tiers)")

precomp_path = Path("data") / "latest.parquet"
if models_loaded and precomp_path.exists():
    try:
        df = pd.read_parquet(precomp_path)
        # Expect 'valid' in the table written by CI
        df["valid"] = pd.to_datetime(df["valid"])
        df = df.set_index("valid").sort_index()

        # Filter by the chosen date
        day_mask = (df.index.date == sel_date)
        day_df = df.loc[day_mask]

        if day_df.empty:
            st.info("No precomputed bins for this date yet. Try a nearby date or later.")
        else:
            daily = day_df.resample("1D").agg(
                {"PoP": "mean", "P50mm": "sum", "P75mm": "sum", "P90mm": "sum"}
            )
            daily["Risk"] = pd.cut(
                daily["PoP"],
                [0, 0.3, 0.6, 1.0],
                labels=["Low", "Med", "High"],
                include_lowest=True,
            )

            st.write("**Daily summary** (precomputed)")
            st.dataframe(
                daily.style.format(
                    {"PoP": "{:.0%}", "P50mm": "{:.1f}", "P75mm": "{:.1f}", "P90mm": "{:.1f}"}
                )
            )
            st.write("**3-hour bins** (precomputed)")
            st.dataframe(
                day_df[["PoP", "P50mm", "P75mm", "P90mm"]].style.format(
                    {"PoP": "{:.0%}", "P50mm": "{:.1f}", "P75mm": "{:.1f}", "P90mm": "{:.1f}"}
                )
            )
    except Exception as e:
        st.warning(f"Precomputed forecast could not be read ({e}).")

elif not models_loaded:
    st.info(
        "Models not trained yet. Click **Train models** to build them locally (recommended), "
        "then commit the resulting files into the repo so CI can generate daily forecasts."
    )
else:
    # Fallback to live GEFS (only works if herbie/cfgrib are available)
    if GEFSDownloader is None:
        st.warning(
            "Live GEFS download is disabled here. Enable CI precomputation (recommended) "
            "or run locally with herbie/cfgrib to generate predictions."
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
            out["PoP"] = p_cal
            out["P50mm"] = q[0.5]
            out["P75mm"] = q[0.75]
            out["P90mm"] = q[0.9]
            daily = out.resample("1D").agg(
                {"PoP": "mean", "P50mm": "sum", "P75mm": "sum", "P90mm": "sum"}
            )
            daily["Risk"] = pd.cut(
                daily["PoP"],
                [0, 0.3, 0.6, 1.0],
                labels=["Low", "Med", "High"],
                include_lowest=True,
            )
            st.write("**Daily summary**")
            st.dataframe(
                daily.style.format(
                    {"PoP": "{:.0%}", "P50mm": "{:.1f}", "P75mm": "{:.1f}", "P90mm": "{:.1f}"}
                )
            )
            st.write("**3-hour bins**")
            st.dataframe(
                out[["PoP", "P50mm", "P75mm", "P90mm"]].style.format(
                    {"PoP": "{:.0%}", "P50mm": "{:.1f}", "P75mm": "{:.1f}", "P90mm": "{:.1f}"}
                )
            )
        except Exception as e:
            st.warning(f"Could not compute live forecast here ({e}).")

# ===============================================================
# 2) Imminent rain (next 60 min)
# ===============================================================
st.markdown("---")
st.subheader("2) Imminent rain (next 60 min)")

if st.button("Check now"):
    with st.spinner("Fetching DWD radar and nowcasting…"):
        try:
            rx = fetch_radar_last_hour()  # DataArray [time,y,x] in dBZ or mm/h (polyfilled)
            R = reflectivity_to_rainrate(rx.values)
            import pysteps as ps

            oflow = ps.motion.get_method("lucaskanade")(R)
            extrap = ps.extrapolation.get_method("semilagrangian")

            steps = max(1, int(round(lead_minutes / 5)))
            Rf = extrap(R[-12:], oflow, steps)  # last hour → 12 steps of 5 min

            j, i = berlin_point_index(rx, lat, lon)
            rain_now = float(R[-1, j, i])
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
                f"(error: {e})."
            )

# ===============================================================
# Training button (optional)
# ===============================================================
st.markdown("---")
st.subheader("Model training (optional)")

st.info(
    "If you want to retrain models locally, run `python -m train --start 2022-01-01 --end YYYY-MM-DD`, "
    "then commit the three model files to the repo so CI can publish forecasts."
)

if st.button("Train models (3 years)"):
    with st.spinner("Training… this can take several minutes on first run"):
        import subprocess
        mod = "src.train" if importlib.util.find_spec("src.train") else "train"
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    mod,
                    "--start",
                    "2022-01-01",
                    "--end",
                    dt.date.today().isoformat(),
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            st.error("Training finished (or was skipped if 'herbie' isn’t available). See logs below.")
            with st.expander("stderr"):
                st.code(e.stderr or "No stderr captured.")
        else:
            st.success("Training finished. Commit model files so CI can use them.")
