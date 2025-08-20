# --- path bootstrap (put at line 1 of make_features.py) ---
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent        # project root (folder with streamlit_app.py)
SRC  = ROOT / "src"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# primary import (package style); fallback to plain 'utils' if needed
try:
    from src.utils.timealign import to_3h_bins
except ModuleNotFoundError:
    from utils.timealign import to_3h_bins
# --- end bootstrap ---
from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr
# make_features.py
from src.utils.timealign import to_3h_bins
def summarize_ensemble(ds: xr.Dataset, lat, lon) -> pd.DataFrame:
    point = ds.sel(latitude=lat, longitude=lon, method="nearest")
    df_list = []
    for f in point["lead"].values:
        valid_time = pd.to_datetime(point["time"].values[0]) + pd.to_timedelta(int(f), unit="h")
        pm = point.sel(lead=f)
        # Try APCP as total precip; depending on mapping it may be named 'tp'
        apcp_name = "APCP" if "APCP" in pm else ("tp" if "tp" in pm else list(pm.data_vars)[0])
        apcp = pm[apcp_name].values
        def stats(arr):
            return {
                "mean": float(np.nanmean(arr)),
                "std": float(np.nanstd(arr)),
                "p90": float(np.nanquantile(arr, 0.9)),
                "p10": float(np.nanquantile(arr, 0.1)),
            }
        s_ap = stats(apcp)
        prob_gt01 = float((np.array(apcp) >= 0.1).mean())
        row = {
            "valid": valid_time,
            "apcp_mean": s_ap["mean"],
            "apcp_std": s_ap["std"],
            "apcp_p90": s_ap["p90"],
            "apcp_prob_gt01": prob_gt01,
        }
        df_list.append(row)
    df = pd.DataFrame(df_list).set_index("valid").sort_index()
    df["bin"] = to_3h_bins(df.index)
    agg = df.groupby("bin").agg({
        "apcp_mean": "sum",
        "apcp_std": "mean",
        "apcp_p90": "sum",
        "apcp_prob_gt01": "mean",
    })
    return agg
