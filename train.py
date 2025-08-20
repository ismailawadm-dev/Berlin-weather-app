# train.py
from __future__ import annotations

# ---------------- path bootstrap (works with/without src/) ----------------
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
for p in (ROOT, SRC):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# ---------------- imports that work in both layouts ----------------
try:
    from src.config import Cfg
except ModuleNotFoundError:
    from config import Cfg

try:
    from src.prob_model import ProbModel  # noqa: F401
except ModuleNotFoundError:
    from prob_model import ProbModel  # noqa: F401
try:
    from src.quant_model import QuantileModel  # noqa: F401
except ModuleNotFoundError:
    from quant_model import QuantileModel  # noqa: F401
try:
    from src.calib import Calibrator  # noqa: F401
except ModuleNotFoundError:
    from calib import Calibrator  # noqa: F401

# GEFS downloader optional (herbie not on Cloud)
try:
    from gefs import GEFSDownloader
except Exception:
    GEFSDownloader = None  # type: ignore

import argparse
import logging


def train_model(start: str, end: str, out_dir: str | None = None) -> None:
    cfg = Cfg("config.yaml")
    logging.basicConfig(level=logging.INFO, format="[train] %(message)s")
    log = logging.getLogger("train")

    log.info(f"Requested training window: {start} → {end}")

    # If your real pipeline requires GEFS, skip politely when unavailable.
    if GEFSDownloader is None:
        log.warning(
            "GEFSDownloader is unavailable (no 'herbie'). Training is disabled on Cloud.\n"
            "Run training locally (install 'herbie'), then commit the model files."
        )
        return

    # TODO: Replace the block below with your actual training logic.
    log.info("GEFSDownloader available in this environment. Insert real training here.")
    # Example placeholders (do not run on Cloud):
    # from make_features import summarize_ensemble
    # dl = GEFSDownloader(cfg)
    # ... build dataset & fit models ...
    # ProbModel(...).save(cfg.models_dir / "prob_model.lgbm.joblib")
    # Calibrator(...).save(cfg.models_dir / "calibrator.joblib")
    # QuantileModel(...).save(cfg.models_dir / "quant_model.lgbm.joblib")


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--out", default=None, help="Optional output directory")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_model(args.start, args.end, args.out)
