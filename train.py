from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from .config import Cfg
from .data_sources.gefs import GEFSDownloader
from .data_sources.radolan import fetch_radolan_rw, point_series
from .features.make_features import summarize_ensemble
from .modeling.prob_model import ProbModel
from .modeling.quant_model import QuantileModel
from .modeling.calib import Calibrator
from .utils.io import ensure_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--runhour", type=int, default=0)
    args = ap.parse_args()

    cfg = Cfg()
    lat = cfg["location"]["lat"]; lon = cfg["location"]["lon"]
    bbox = cfg["location"]["bbox"]
    leads = cfg["features"]["leads_hours"]
    members = cfg["features"]["members"]

    print("Downloading RADOLAN labels…")
    da_rw = fetch_radolan_rw(args.start, args.end, bbox=(bbox[0], bbox[1], bbox[2], bbox[3]), cache_dir=Path(cfg.cache)/"radolan")
    s_point = point_series(da_rw, lat, lon)
    s_point = s_point.resample("3H").sum()
    y_bins = s_point.index

    dl = GEFSDownloader(cfg)
    frames = []
    for dt in pd.date_range(args.start, args.end, freq="1D"):
        try:
            ds = dl.stack_members(dt.strftime("%Y-%m-%d"), args.runhour, members=members, leads=leads, bbox=tuple(bbox))
            fe = summarize_ensemble(ds, lat, lon)
            frames.append(fe)
        except Exception as e:
            print("Skip", dt, e)
    if not frames:
        raise SystemExit("No GEFS data fetched; check connectivity or date range.")
    X = pd.concat(frames, axis=0).sort_index()

    X = X.loc[X.index.intersection(y_bins)]
    y = (s_point.reindex(X.index, method="nearest") >= cfg["training"]["rain_threshold_mm"]) * 1
    y_amt = s_point.reindex(X.index, method="nearest").fillna(0.0)

    ensure_dir(cfg.models_dir)
    print("Fitting probability model…")
    pm = ProbModel(cfg["models"]["prob"]["params"]).fit(X, y)
    raw_p = pm.predict_proba(X)

    print("Calibrating…")
    cal = Calibrator(cfg["calibration"]["method"]).fit(raw_p, y)
    raw_brier = float(((raw_p - y)**2).mean())
    cal_brier = float(((cal.predict(raw_p) - y)**2).mean())
    print("Brier score raw -> calib:", round(raw_brier,3), "->", round(cal_brier,3))

    print("Fitting quantile model…")
    qm = QuantileModel(cfg["models"]["quant"]["quantiles"], cfg["models"]["quant"]["params"]).fit(X, y_amt)

    pm.save(cfg.models_dir/"prob_model.lgbm.joblib")
    cal.save(cfg.models_dir/"calibrator.joblib")
    qm.save(cfg.models_dir/"quant_model.lgbm.joblib")
    print("Saved models to", cfg.models_dir)

if __name__ == "__main__":
    main()
