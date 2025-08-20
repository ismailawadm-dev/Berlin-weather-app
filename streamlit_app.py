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

from __future__ import annotations
import datetime as dt
import pandas as pd
import numpy as np
import streamlit as st

# Config
try:
    from src.config import Cfg
except ModuleNotFoundError:
    from config import Cfg

# Optional live GEFS (may be None on Cloud)
try:
    from gefs import GEFSDownloader
except Exception:
    GEFSDownloader = None

# Models
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

# Features
from make_features import summarize_ensemble

# Imminent rain tools (will gracefully degrade if radar is unavailable)
try:
    # If you kept it under src/alerts/
    from src.alerts.watch_imminent import (
        fetch_radar_last_hour,
        reflectivity_to_rainrate,
        berlin_point_index,
    )
except Exception:
    # If the file is in repo root as watch_imminent.py
    try:
        from watch_imminent import (
            fetch_radar_last_hour,
            reflectivity_to_rainrate,
            berlin_point_index,
        )
    except Exception:
        fetch_radar_last_hour = None
        reflectivity_to_rainrate = None
        berlin_point_index = None


st.set_page_config(page_title="Berlin Rain Planning", page_icon="☔", layout="centered")
st.title("Berlin Rain Planning ☔")
st.caption("Week-2 risk (live GEFS when available; falls back to cached forecast) + 60-minute imminent rain indicator")

cfg = Cfg("config.yaml")
lat = cfg["location"]["lat"]
lon = cfg["location"]["lon"]

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("Controls")
sel_date = st.sidebar.date_input("Target date", dt.date.today())
runhour = st.sidebar.selectbox("Model run hour (UTC)", [0, 6, 12, 18], index=0)
lead_minutes = st.sidebar.slider("Imminent alert horizon (min)", 10, 60, 60, 5)
rain_thr = st.sidebar.slider("Rain rate threshold (mm/h)", 0.1, 2.0, 0.6, 0.1)

# ----------------------------
# Load models (from committed files)
# ----------------------------
def _load_models():
    try:
        pm = ProbModel.load(cfg.models_dir / "prob_model.lgbm.joblib")
        cal = Calibrator.load(cfg.models_dir / "calibrator.joblib")
        qm = QuantileModel.load(cfg.models_dir / "quant_model.lgbm.joblib")
        return pm, cal, qm
    except Exception:
        return None, None, None

pm, cal, qm = _load_models()
models_loaded = all(x is not None for x in (pm, cal, qm))

# ----------------------------
# Training button (skips on Cloud)
# ----------------------------
st.subheader("1) Day prediction (risk tiers)")
if not models_loaded:
    st.info("Models not trained yet. Click **Train models** (skips on Cloud). For real training, run locally then commit the model files.")
    if st.button("Train models (3 years)"):
        with st.spinner("Training…"):
            import subprocess
            mod = "src.train" if importlib.util.find_spec("src.train") else "train"
            try:
                subprocess.run(
                    [sys.executable, "-m", mod, "--start", "2022-01-01", "--end", dt.date.today().isoformat()],
                    check=True,
                )
                st.success("Training finished. Commit model files so the app can use them.")
            except subprocess.CalledProcessError:
                st.warning("Training was skipped or failed in this environment. Train locally, then commit models/*.joblib.")
else:
    # ----------------------------
    # Live prediction or cached fallback
    # ----------------------------
    def _predict_from_features(X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (3h bins dataframe, daily dataframe)."""
        raw_p = pm.predict_proba(X)
        p_cal = cal.predict(raw_p)
        q = qm.predict(X)
        out = X.copy()
        out["PoP"] = p_cal
        out["P50mm"] = q[0.5]
        out["P75mm"] = q[0.75]
        out["P90mm"] = q[0.9]
        daily = (
            out.resample("1D").agg({"PoP": "mean", "P50mm": "sum", "P75mm": "sum", "P90mm": "sum"})
        )
        daily["Risk"] = pd.cut(daily["PoP"], [0, 0.3, 0.6, 1.0], labels=["Low", "Med", "High"], include_lowest=True)
        return out, daily

    def _load_cached_for_date(date_obj: dt.date):
        cache_path = ROOT / "data" / "forecast_cache.parquet"
        if not cache_path.exists():
            return None
        df = pd.read_parquet(cache_path)
        # date column is stored as YYYY-MM-DD
        df["date"] = pd.to_datetime(df["date"]).dt.date
        sel = df[df["date"] == date_obj].copy()
        if sel.empty:
            return None
        sel = sel.set_index(pd.to_datetime(sel["bin_start"])).sort_index()
        # rebuild daily the same way
        daily = (
            sel.resample("1D").agg({"PoP": "mean", "P50mm": "sum", "P75mm": "sum", "P90mm": "sum"})
        )
        daily["Risk"] = pd.cut(daily["PoP"], [0, 0.3, 0.6, 1.0], labels=["Low", "Med", "High"], include_lowest=True)
        return sel, daily

    def _try_live_or_cached():
        # Try live GEFS
        if GEFSDownloader is not None:
            try:
                dl = GEFSDownloader(cfg)
                ds = dl.stack_members(
                    sel_date.isoformat(),
                    int(runhour),
                    members=cfg["features"]["members"],
                    leads=cfg["features"]["leads_hours"],
                    bbox=tuple(cfg["location"]["bbox"]),
                )
                X = summarize_ensemble(ds, lat, lon)
                bins, daily = _predict_from_features(X)
                st.success("Live GEFS fetched successfully.")
                return bins, daily, "live"
            except Exception as e:
                st.info(f"Live GEFS unavailable here ({type(e).__name__}). Falling back to cached forecast…")

        # Cached fallback
        cached = _load_cached_for_date(sel_date)
        if cached is None:
            st.warning(
                "No cached forecast found for that date. Generate it with "
                "`scripts/fetch_and_cache.py` locally or enable the provided GitHub Action."
            )
            return None, None, "none"
        st.info("Showing cached forecast.")
        return cached[0], cached[1], "cached"

    bins, daily, mode = _try_live_or_cached()

    if mode != "none":
        st.write("**Daily summary**")
        st.dataframe(daily.style.format({"PoP":"{:.0%}","P50mm":"{:.1f}","P75mm":"{:.1f}","P90mm":"{:.1f}"}))
        st.write("**3-hour bins**")
        st.dataframe(bins[["PoP","P50mm","P75mm","P90mm"]].style.format(
            {"PoP":"{:.0%}","P50mm":"{:.1f}","P75mm":"{:.1f}","P90mm":"{:.1f}"}
        ))

# ----------------------------
# Imminent rain (60 min) – graceful when radar isn't available
# ----------------------------
st.markdown("---")
st.subheader("2) Imminent rain (next 60 min)")
if st.button("Check now"):
    with st.spinner("Fetching DWD radar and nowcasting…"):
        if not all([fetch_radar_last_hour, reflectivity_to_rainrate, berlin_point_index]):
            st.info(
                "Live radar isn’t available from this environment right now "
                "(wetterdienst/decoding not present)."
            )
        else:
            try:
                rx = fetch_radar_last_hour()  # dBZ xarray DataArray
                if rx is None or getattr(rx, "values", None) is None:
                    raise RuntimeError("No radar items returned or could not decode frames.")
                R = reflectivity_to_rainrate(rx.values)
                import pysteps as ps
                oflow = ps.motion.get_method("lucaskanade")(R)
                extrap = ps.extrapolation.get_method("semilagrangian")
                steps = max(1, int(round(60/5)))  # 60 min horizon in 5-min steps
                Rf = extrap(R[-12:], oflow, steps)  # last 60 minutes as baseline window
                j, i = berlin_point_index(rx, lat, lon)
                rain_now = float(R[-1, j, i])
                rain_future = float(Rf[steps-1, j, i])
                st.metric("Rain now (mm/h)", f"{rain_now:.2f}")
                st.metric("Rain in +60 min (mm/h)", f"{rain_future:.2f}")
                if rain_future >= rain_thr:
                    st.success("Likely rain within the next 60 minutes. Prepare couriers/ETAs.")
                else:
                    st.info("All clear — no significant rain expected in the next 60 minutes near Berlin.")
            except Exception as e:
                st.info(
                    "Live radar isn’t available from this environment right now "
                    "(wetterdienst returned no recent items / incompatible parameters)."
                )
                st.caption(f"Details: {type(e).__name__}: {e}")
