# gefs.py
from __future__ import annotations

# Keep import safe when 'herbie' is not installed (Cloud).
try:
    from herbie import Herbie  # type: ignore
except Exception:
    Herbie = None  # sentinel

if Herbie is None:
    # Export a sentinel so callers can check and skip downloads.
    GEFSDownloader = None  # type: ignore
else:
    # Implement only in environments where herbie is installed.
    import xarray as xr

    class GEFSDownloader:
        def __init__(self, cfg):
            self.cfg = cfg

        def stack_members(
            self,
            date_str: str,
            runhour: int,
            members,
            leads,
            bbox,
        ) -> xr.Dataset:
            # Put your real Herbie-based fetching here for local training.
            raise NotImplementedError(
                "GEFSDownloader requires 'herbie' and is intended for local runs."
            )
