# scripts/make_daily_forecast.py
from __future__ import annotations
import datetime as dt
from pathlib import Path

import pandas as pd

# local imports (already in your repo)
from config import Cfg
from make_features import summarize_ensemble
from prob_model import ProbModel
from quant_model import QuantileModel
from calib import Calibrator
from gefs import GEFSDownloader


def _have_models(models_dir: Path) -> bool:
    req = [
        models_dir / "prob_model.lgbm.joblib",
        models_dir / "calibrator.joblib",
        models_dir / "quant_model.lgbm.joblib",
    ]
    return all(p.exists() for p in req)


def _nearest_cycle_utc() -> int:
    # 0/6/12/18 UTC — choose the last completed cycle
    h = dt.datetime.utcnow().hour
    for cand in (18, 12, 6, 0):
        if h >= cand:
            return cand
    return 18


def main() -> None:
    cfg = Cfg("config.yaml")
    models_dir = cfg.models_dir if hasattr(cfg, "models_dir") else Path("models")

    if not _have_models(models_dir):
        print("[forecast] Models are missing. Skipping forecast generation.")
        return

    # Location + GEFS request window from config
    lat = cfg["location"]["lat"]
    lon = cfg["location"]["lon"]
    bbox = tuple(cfg["location"]["bbox"])
    leads = cfg["features"]["leads_hours"]
    members = cfg["features"]["members"]

    # Choose target run (today with last completed cycle)
    run_date = dt.date.today().isoformat()
    run_hour = _nearest_cycle_utc()

    # Download + stack members -> xarray Dataset
    dl = GEFSDownloader(cfg)
    ds = dl.stack_members(
        run_date,
        run_hour,
        members=members,
        leads=leads,
        bbox=bbox,
    )

    # Build 3-h features at the point
    X = summarize_ensemble(ds, lat, lon)

    # Load trained models
    pm = ProbModel.load(models_dir / "prob_model.lgbm.joblib")
    cal = Calibrator.load(models_dir / "calibrator.joblib")
    qm = QuantileModel.load(models_dir / "quant_model.lgbm.joblib")

    # Predict
    raw_p = pm.predict_proba(X)
    p_cal = cal.predict(raw_p)
    q = qm.predict(X)

    out = X.copy()
    out["PoP"] = p_cal
    out["P50mm"] = q[0.5]
    out["P75mm"] = q[0.75]
    out["P90mm"] = q[0.9]

    # Daily rollup for convenience
    daily = out.resample("1D").agg(
        {"PoP": "mean", "P50mm": "sum", "P75mm": "sum", "P90mm": "sum"}
    )
    daily["date"] = daily.index.date

    # Save outputs into /data
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    ts = dt.datetime.utcnow().strftime("%Y%m%d%H")

    # Full 3-hour table
    out_reset = out.reset_index()  # keep 'valid' timestamp as a column
    out_reset.to_parquet(data_dir / f"forecast_{ts}.parquet", index=False)
    out_reset.to_parquet(data_dir / "latest.parquet", index=False)

    # Daily summary table (optional; handy for debugging)
    daily.to_csv(data_dir / f"forecast_daily_{ts}.csv", index=False)
    daily.to_csv(data_dir / "latest_daily.csv", index=False)

    print(f"[forecast] Wrote data/latest.parquet and data/latest_daily.csv at {ts}Z")


if __name__ == "__main__":
    main()
