# streamlit_app.py
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

# Config (works whether config lives at repo root or under src/)
try:
    from src.config import Cfg
except ModuleNotFoundError:
    from config import Cfg

# Optional GEFS downloader (herbie might not be available on Cloud)
try:
    from gefs import GEFSDownloader
except Exception:
    GEFSDownloader = None

# Feature engineering
from make_features import summarize_ensemble

# Models (fallback if they're not under src/modeling/)
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

# Imminent-rain helpers (robust import)
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

# ------------------------------------------------------------------

st.set_page_config(page_title="Berlin Rain Planning", page_icon="☔", layout="centered")
st.title("Berlin Rain Planning ☔")
st.caption("Week-2 risk + up to 60-minute imminent rain indicator")

cfg = Cfg("config.yaml")
lat = cfg["location"]["lat"]
lon = cfg["location"]["lon"]

# Sidebar controls
st.sidebar.header("Controls")
sel_date = st.sidebar.date_input("Target date", dt.date.today())
runhour = st.sidebar.selectbox("Model run hour (UTC)", [0, 6, 12, 18], index=0)
lead_minutes = st.sidebar.slider("Imminent alert horizon (min)", 10, 60, 60, 5)
rain_thr = st.sidebar.slider("Rain rate threshold (mm/h)", 0.1, 2.0, 0.6, 0.1)

# Load models if present
try:
    pm = ProbModel.load(cfg.models_dir / "prob_model.lgbm.joblib")
    cal = Calibrator.load(cfg.models_dir / "calibrator.joblib")
    qm = QuantileModel.load(cfg.models_dir / "quant_model.lgbm.joblib")
    models_loaded = True
except Exception:
    models_loaded = False

# ------------------------------------------------------------------
# 1) Day prediction
# ------------------------------------------------------------------
st.subheader("1) Day prediction (risk tiers)")

if not models_loaded:
    st.info(
        "Models not trained yet. Use the **Quick test** first to ensure the pipeline "
        "runs here, then do the full 3-year training."
    )

    # --- Quick sanity run: last 7 days ---
    if st.button("Quick test (last 7 days)"):
        with st.spinner("Running a tiny 7-day training to verify the pipeline…"):
            import subprocess, shlex

            mod = "src.train" if importlib.util.find_spec("src.train") else "train"
            end = dt.date.today()
            start = end - dt.timedelta(days=7)
            cmd = [sys.executable, "-m", mod, "--start", start.isoformat(), "--end", end.isoformat()]
            st.write("Command:", " ".join(shlex.quote(c) for c in cmd))
            res = subprocess.run(cmd, text=True, capture_output=True)
            st.write("Exit code:", res.returncode)
            if res.stdout:
                st.subheader("stdout")
                st.code(res.stdout, language="bash")
            if res.stderr:
                st.subheader("stderr")
                st.code(res.stderr, language="bash")
            if res.returncode != 0:
                st.error("Quick test FAILED. See logs above and fix the cause, then try again.")
                st.stop()
            st.success("Quick test finished without a non-zero exit code. You can now try the full training.")
            st.experimental_rerun()

    # --- Full training: ~3 years ---
    if st.button("Train models (3 years)"):
        with st.spinner("Training… this can take several minutes on first run"):
            import subprocess, shlex

            mod = "src.train" if importlib.util.find_spec("src.train") else "train"
            cmd = [sys.executable, "-m", mod, "--start", "2022-01-01", "--end", dt.date.today().isoformat()]
            st.write("Command:", " ".join(shlex.quote(c) for c in cmd))
            res = subprocess.run(cmd, text=True, capture_output=True)
            st.write("Exit code:", res.returncode)
            if res.stdout:
                st.subheader("stdout")
                st.code(res.stdout, language="bash")
            if res.stderr:
                st.subheader("stderr")
                st.code(res.stderr, language="bash")
            if res.returncode != 0:
                st.error("Training FAILED. See logs above for the real error.")
                st.stop()
        st.success("Training finished successfully.")
        st.experimental_rerun()

else:
    # Inference path
    if GEFSDownloader is None:
        st.warning(
            "GEFS download is disabled in this environment, so the day-prediction section is skipped."
        )
    else:
        try:
            with st.spinner("Downloading ensemble and building features…"):
                dl = GEFSDownloader(cfg)
                ds = dl.stack_members(
                    sel_date.isoformat(),
                    runhour,
                    members=cfg["features"]["members"],
                    leads=cfg["features"]["leads_hours"],
                    bbox=tuple(cfg["location"]["bbox"]),
                )
                X = summarize_ensemble(ds, lat, lon)

            with st.spinner("Running probabilistic and quantile models…"):
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
                daily["PoP"], [0, 0.3, 0.6, 1.0], labels=["Low", "Med", "High"], include_lowest=True
            )

            st.write("**Daily summary**")
            st.dataframe(daily.style.format({"PoP": "{:.0%}", "P50mm": "{:.1f}", "P75mm": "{:.1f}", "P90mm": "{:.1f}"}))

            st.write("**3-hour bins**")
            st.dataframe(
                out[["PoP", "P50mm", "P75mm", "P90mm"]].style.format(
                    {"PoP": "{:.0%}", "P50mm": "{:.1f}", "P75mm": "{:.1f}", "P90mm": "{:.1f}"}
                )
            )
        except Exception as e:
            st.error(f"Day prediction failed: {e!r}")

# ------------------------------------------------------------------
# 2) Imminent rain (up to 60 minutes)
# ------------------------------------------------------------------
st.markdown("---")
st.subheader("2) Imminent rain (next 10–60 min)")

if st.button("Check now"):
    with st.spinner("Fetching DWD radar and nowcasting…"):
        try:
            rx = fetch_radar_last_hour()  # dBZ xarray DataArray [time,y,x]
            R = reflectivity_to_rainrate(rx.values)  # mm/h

            import pysteps as ps

            oflow = ps.motion.get_method("lucaskanade")(R)
            extrap = ps.extrapolation.get_method("semilagrangian")

            # 5-min frames -> steps for requested horizon (e.g. 60 min -> 12 steps)
            steps = max(1, int(round(lead_minutes / 5)))
            # use last hour (12 x 5-min frames)
            Rf = extrap(R[-12:], oflow, steps)

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
                f"(data source returned no recent items / incompatible parameters).\n\n"
                f"Details: {e!r}"
            )
