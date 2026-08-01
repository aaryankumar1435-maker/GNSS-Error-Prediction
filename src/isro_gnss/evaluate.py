from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader

from .config import Config
from .dataset import Scaler, WindowDataset, build_series, make_test_indices
from .models import RNNForecaster
from .train import load_or_generate_data
from .utils import get_device


def _normality_stats(residuals: np.ndarray) -> dict:
    sample = residuals
    if len(sample) > 5000:
        rng = np.random.default_rng(0)
        sample = rng.choice(residuals, size=5000, replace=False)
    shapiro_stat, shapiro_p = stats.shapiro(sample)
    jb_stat, jb_p = stats.jarque_bera(residuals)
    return dict(
        mean=float(np.mean(residuals)),
        std=float(np.std(residuals)),
        skewness=float(stats.skew(residuals)),
        kurtosis_excess=float(stats.kurtosis(residuals)),
        shapiro_stat=float(shapiro_stat),
        shapiro_p=float(shapiro_p),
        jarque_bera_stat=float(jb_stat),
        jarque_bera_p=float(jb_p),
    )


def _make_plots(errors: np.ndarray, channels: list[str], horizons_minutes: list[int],
                 step_minutes: int, horizon_steps: int, plots_dir: str) -> None:
    for ci, ch in enumerate(channels):
        residuals = errors[:, :, ci].reshape(-1)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(residuals, bins=40, density=True, alpha=0.7, label="residuals")
        mu, sigma = float(np.mean(residuals)), float(np.std(residuals))
        xs = np.linspace(residuals.min(), residuals.max(), 200)
        ax.plot(xs, stats.norm.pdf(xs, mu, sigma), "r-", label="normal fit")
        ax.set_title(f"{ch}: residual distribution")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"{ch}_hist.png"))
        plt.close(fig)

        fig = plt.figure(figsize=(5, 5))
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title(f"{ch}: QQ plot vs normal")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{ch}_qq.png"))
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for ci, ch in enumerate(channels):
        rmses = []
        for minutes in horizons_minutes:
            step_idx = min(max(minutes // step_minutes - 1, 0), horizon_steps - 1)
            e = errors[:, step_idx, ci]
            rmses.append(float(np.sqrt(np.mean(e ** 2))))
        ax.plot(horizons_minutes, rmses, marker="o", label=ch)
    ax.set_xlabel("validity horizon (minutes)")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE vs prediction horizon")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "rmse_vs_horizon.png"))
    plt.close(fig)


def run_evaluation(cfg: Config) -> dict:
    device = get_device()
    ckpt = torch.load(cfg.train.checkpoint_path, map_location=device, weights_only=False)
    channels = ckpt["channels"]
    lookback, horizon = ckpt["lookback_steps"], ckpt["horizon_steps"]
    step_minutes = ckpt["step_minutes"]

    df = load_or_generate_data(cfg)
    series, sat_id_to_idx, sat_ids, sat_types = build_series(df, channels)
    if sat_id_to_idx != ckpt["sat_id_to_idx"]:
        raise ValueError("Satellite ordering in data does not match the training checkpoint")

    scaler = Scaler().load_state_dict(ckpt["scaler_state"])
    scaled = scaler.transform(series)

    n_sats, n_total_steps = series.shape[0], series.shape[1]
    test_idx = make_test_indices(n_sats, n_total_steps, lookback, horizon)
    test_loader = DataLoader(WindowDataset(scaled, test_idx, lookback, horizon),
                              batch_size=cfg.train.batch_size, shuffle=False)

    model = RNNForecaster(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    preds_scaled, targets_scaled = [], []
    with torch.no_grad():
        for x, sat_idx, y in test_loader:
            pred = model(x.to(device), sat_idx.to(device)).cpu().numpy()
            preds_scaled.append(pred)
            targets_scaled.append(y.numpy())
    preds_scaled = np.concatenate(preds_scaled, axis=0)
    targets_scaled = np.concatenate(targets_scaled, axis=0)

    preds = scaler.inverse_transform(preds_scaled)
    targets = scaler.inverse_transform(targets_scaled)
    errors = preds - targets  # (n_sats, horizon, n_channels)

    metrics: dict = {"per_horizon": {}, "normality": {}, "overall": {}}
    for minutes in cfg.eval.horizons_minutes:
        step_idx = min(max(minutes // step_minutes - 1, 0), horizon - 1)
        err_h = errors[:, step_idx, :]
        metrics["per_horizon"][str(minutes)] = {
            ch: dict(
                rmse=float(np.sqrt(np.mean(err_h[:, ci] ** 2))),
                mae=float(np.mean(np.abs(err_h[:, ci]))),
                bias=float(np.mean(err_h[:, ci])),
            )
            for ci, ch in enumerate(channels)
        }

    for ci, ch in enumerate(channels):
        residuals = errors[:, :, ci].reshape(-1)
        metrics["normality"][ch] = _normality_stats(residuals)
        metrics["overall"][ch] = dict(
            rmse=float(np.sqrt(np.mean(residuals ** 2))),
            mae=float(np.mean(np.abs(residuals))),
        )

    os.makedirs(cfg.eval.plots_dir, exist_ok=True)
    _make_plots(errors, channels, list(cfg.eval.horizons_minutes), step_minutes, horizon, cfg.eval.plots_dir)

    os.makedirs(os.path.dirname(cfg.eval.metrics_path) or ".", exist_ok=True)
    with open(cfg.eval.metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {cfg.eval.metrics_path}")
    print(f"Saved plots to {cfg.eval.plots_dir}")
    return metrics
