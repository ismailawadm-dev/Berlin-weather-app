from __future__ import annotations
from pathlib import Path
import pandas as pd
import xarray as xr
from wetterdienst.provider.dwd.radar import DwdRadarValues, DwdRadarParameter, DwdRadarDataSubset
from wetterdienst import Period

from ..utils.io import ensure_dir

def fetch_radolan_rw(start, end, bbox, cache_dir: Path) -> xr.DataArray:
    ensure_dir(cache_dir)
    values = DwdRadarValues(
        parameter=DwdRadarParameter.RW,
        start_date=start,
        end_date=end,
        subset=DwdRadarDataSubset.GERMANY,
        period=Period.HOURLY,
    )
    ds = values.to_xarray()
    lon0, lat0, lon1, lat1 = bbox
    ds = ds.sel(longitude=slice(lon0, lon1), latitude=slice(lat0, lat1))
    return ds["value"].rename("radolan_mm")

def point_series(da: xr.DataArray, lat, lon) -> pd.Series:
    s = da.sel(latitude=lat, longitude=lon, method="nearest").to_series()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    return s
