"""Evaluate the trained forecaster at multiple validity horizons and check
how close the residual distribution is to normal."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isro_gnss.config import load_config
from isro_gnss.evaluate import run_evaluation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_evaluation(cfg)
