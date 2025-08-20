from __future__ import annotations
import argparse, json
import pandas as pd
from pathlib import Path

def find_threshold_for_precision(df, target_precision=0.99):
    df = df.dropna(subset=["pop_cal","y_actual"]).copy()
    best = None
    for thr in [i/100 for i in range(50,100)]:
        y_pred = (df["pop_cal"] >= thr).astype(int)
        tp = ((y_pred==1) & (df["y_actual"]==1)).sum()
        fp = ((y_pred==1) & (df["y_actual"]==0)).sum()
        if tp+fp == 0: 
            continue
        prec = tp/(tp+fp)
        rec = tp/(df["y_actual"]==1).sum()
        if prec >= target_precision:
            best = {"thr": thr, "precision": float(prec), "recall": float(rec)}
            break
    return best

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--precision", type=float, default=0.99)
    args = ap.parse_args()
    df = pd.read_csv(args.in_csv)
    best = find_threshold_for_precision(df, args.precision)
    if not best:
        raise SystemExit("No threshold found for requested precision. Try lower precision.")
    Path(args.out_json).write_text(json.dumps(best, indent=2))
    print("Saved:", args.out_json, best)
