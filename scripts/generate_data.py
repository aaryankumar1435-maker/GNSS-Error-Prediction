"""Generate (or regenerate) the synthetic GNSS clock/ephemeris error dataset."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isro_gnss.config import load_config
from isro_gnss.data_gen import generate_dataset, save_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = generate_dataset(
        n_geo=cfg.data.n_geo,
        n_meo=cfg.data.n_meo,
        train_days=cfg.data.train_days,
        test_days=cfg.data.test_days,
        step_minutes=cfg.data.step_minutes,
        seed=cfg.data.seed,
    )
    save_dataset(df, cfg.data.csv_path)
    print(f"Wrote {len(df)} rows for {df['sat_id'].nunique()} satellites to {cfg.data.csv_path}")
