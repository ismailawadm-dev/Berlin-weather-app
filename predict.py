from __future__ import annotations
import argparse
import pandas as pd

from .config import Cfg
from .data_sources.gefs import GEFSDownloader
from .features.make_features import summarize_ensemble
from .modeling.prob_model import ProbModel
from .modeling.quant_model import QuantileModel
from .modeling.calib import Calibrator

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="Target date YYYY-MM-DD")
    ap.add_argument("--runhour", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = Cfg()
    lat = cfg["location"]["lat"]; lon = cfg["location"]["lon"]
    bbox = cfg["location"]["bbox"]
    leads = cfg["features"]["leads_hours"]
    members = cfg["features"]["members"]

    dl = GEFSDownloader(cfg)
    ds = dl.stack_members(args.date, args.runhour, members=members, leads=leads, bbox=tuple(bbox))
    X = summarize_ensemble(ds, lat, lon)

    pm = ProbModel.load(cfg.models_dir/"prob_model.lgbm.joblib")
    cal = Calibrator.load(cfg.models_dir/"calibrator.joblib")
    qm = QuantileModel.load(cfg.models_dir/"quant_model.lgbm.joblib")

    raw_p = pm.predict_proba(X)
    p_cal = cal.predict(raw_p)
    q = qm.predict(X)

    out = X.copy()
    out["pop_raw"] = raw_p
    out["pop_cal"] = p_cal
    out["mm_p50"] = q[0.5]
    out["mm_p75"] = q[0.75]
    out["mm_p90"] = q[0.9]

    daily = out.resample("1D").agg({"pop_cal":"mean", "mm_p50":"sum", "mm_p75":"sum", "mm_p90":"sum"})
    daily["risk_tier"] = pd.cut(daily["pop_cal"], bins=[0,0.3,0.6,1.0], labels=["low","med","high"], include_lowest=True)

    out.to_csv(args.out.replace(".csv","_3h.csv"))
    daily.to_csv(args.out.replace(".csv","_daily.csv"))
    print("Wrote:", args.out.replace(".csv","_3h.csv"), "and", args.out.replace(".csv","_daily.csv"))

if __name__ == "__main__":
    main()
