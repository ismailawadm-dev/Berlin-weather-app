from __future__ import annotations
from pathlib import Path
import xarray as xr
from herbie import Herbie

from ..config import Cfg
from ..utils.io import ensure_dir

VARS_DEFAULT = ["APCP", "PWAT", "CAPE", "RH:700 mb", "HGT:500 mb", "UGRD:850 mb", "VGRD:850 mb", "TMP:2 m above ground"]

class GEFSDownloader:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        ensure_dir(cfg.cache / "gefs")

    def fetch(self, run_date, run_hour, member, fxx, bbox=None, vars=None) -> xr.Dataset:
        if vars is None:
            vars = VARS_DEFAULT
        h = Herbie(
            model="gefs",
            product="atmos",
            member=member,
            date=run_date,
            grib=0.5,
            hour=run_hour,
            fxx=fxx,
        )
        ds = h.xarray(vars=vars, remove_grib=True)
        if bbox:
            lon0, lat0, lon1, lat1 = bbox
            ds = ds.sel(latitude=slice(lat1, lat0), longitude=slice(lon0, lon1))
        return ds

    def stack_members(self, date, hour, members, leads, bbox=None, vars=None) -> xr.Dataset:
        dsets = []
        for m in members:
            for f in leads:
                try:
                    d = self.fetch(date, hour, m, f, bbox=bbox, vars=vars)
                    d = d.assign_coords({"member": m, "lead": f}).expand_dims(["member","lead"])
                    dsets.append(d)
                except Exception as e:
                    print(f"Warn: skip {m} f{f}: {e}")
        return xr.combine_by_coords(dsets, combine_attrs="override")
