from __future__ import annotations
import argparse
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
import pysteps as ps
from wetterdienst.provider.dwd.radar import DwdRadarParameter, DwdRadarValues, DwdRadarDataSubset
from wetterdienst import Period

from .config import Cfg
from .utils.io import ensure_dir

def fetch_latest_radolan(hours=2):
    end = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(hours=hours)
    values = DwdRadarValues(
        parameter=DwdRadarParameter.RW,
        start_date=start,
        end_date=end,
        subset=DwdRadarDataSubset.GERMANY,
        period=Period.HOURLY,
    )
    ds = values.to_xarray()
    return ds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--hours", type=int, default=2)
    args = ap.parse_args()

    cfg = Cfg()
    ensure_dir(cfg.cache/"nowcast")

    ds = fetch_latest_radolan(hours=args.hours)
    R, metadata = ps.io.importers.import_xarray(ds["value"].transpose("time","y","x"))
    oflow_method = ps.motion.get_method("lucaskanade")
    V = oflow_method(R)
    extrap_method = ps.extrapolation.get_method("semilagrangian")
    R_f = extrap_method(R[-3:], V, 12)
    out = xr.DataArray(R_f, dims=("t","y","x"))
    out.to_dataset(name="rw_nowcast_mmph").to_zarr(args.out, mode="w")
    print("Wrote nowcast to", args.out)

if __name__ == "__main__":
    main()
