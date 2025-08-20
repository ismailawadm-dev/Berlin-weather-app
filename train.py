# train.py
from __future__ import annotations

# -------------------------------------------------------------------
# 1) Path bootstrap so imports work from Streamlit or CLI
# -------------------------------------------------------------------
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
for p in (ROOT, SRC):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# -------------------------------------------------------------------
# 2) Imports that work in both layouts (src/ … or flat repo)
# -------------------------------------------------------------------
try:
    from src.config import Cfg
except ModuleNotFoundError:
    from config import Cfg

# If your training code imports other project modules, prefer this pattern:
try:
    from src.prob_model import ProbModel
    from src.quant_model import QuantileModel
    from src.calib import Calibrator
    from src.gefs import GEFSDownloader
    from src.make_features import summarize_ensemble
except ModuleNotFoundError:
    from prob_model import ProbModel
    from quant_model import QuantileModel
    from calib import Calibrator
    from gefs import GEFSDownloader
    from make_features import summarize_ensemble

# -------------------------------------------------------------------
# 3) >>> PASTE YOUR EXISTING TRAINING LOGIC HERE <<<
#     (All the functions/classes you already had – unchanged.)
#
#     If your old file had something like:
#        - def train_model(start, end, out_dir=None): ...
#        - or a main() that parses --start/--end and does the work
#     keep it exactly as-is. The only thing that changed is the imports.
# -------------------------------------------------------------------

# Example minimal structure if you didn't have one;
# delete this block if your file already defines the real logic.
if "train_model" not in globals():
    import argparse
    from datetime import date

    def train_model(start: str, end: str, out_dir: str | None = None) -> None:
        """
        Placeholder: replace with your real training routine.
        This exists only so the script can run if you hadn't defined one.
        """
        cfg = Cfg("config.yaml")
        print(f"[train.py] Training stub from {start} to {end}.")
        print("Replace this function with your existing training code.")

    def _parse_args():
        ap = argparse.ArgumentParser()
        ap.add_argument("--start", required=True, help="YYYY-MM-DD")
        ap.add_argument("--end", required=True, help="YYYY-MM-DD")
        ap.add_argument("--out", default=None, help="Optional output dir")
        return ap.parse_args()

# -------------------------------------------------------------------
# 4) CLI entry point so `python -m train --start ... --end ...` works
# -------------------------------------------------------------------
if __name__ == "__main__":
    # If your original file already had its own argparse/main,
    # keep it and remove this block. Otherwise this gives you a safe default.
    if " _parse_args" in globals():
        args = _parse_args()
        train_model(args.start, args.end, args.out)
    else:
        # Fallback: try to mimic your old interface
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("--start", required=True)
        ap.add_argument("--end", required=True)
        ap.add_argument("--out", default=None)
        args = ap.parse_args()
        train_model(args.start, args.end, args.out)
