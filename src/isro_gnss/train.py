from __future__ import annotations

import copy
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import Config
from .data_gen import generate_dataset, save_dataset
from .dataset import Scaler, WindowDataset, build_series, make_train_val_indices
from .models import RNNForecaster
from .utils import get_device, set_seed


def load_or_generate_data(cfg: Config) -> pd.DataFrame:
    csv_path = cfg.data.csv_path
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = generate_dataset(
        n_geo=cfg.data.n_geo,
        n_meo=cfg.data.n_meo,
        train_days=cfg.data.train_days,
        test_days=cfg.data.test_days,
        step_minutes=cfg.data.step_minutes,
        seed=cfg.data.seed,
    )
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    save_dataset(df, csv_path)
    return df


def run_training(cfg: Config) -> dict:
    set_seed(cfg.data.seed)
    device = get_device()

    df = load_or_generate_data(cfg)
    channels = list(cfg.dataset.channels)
    series, sat_id_to_idx, sat_ids, sat_types = build_series(df, channels)

    steps_per_day = int(24 * 60 / cfg.data.step_minutes)
    n_train_steps = cfg.data.train_days * steps_per_day
    lookback, horizon = cfg.dataset.lookback_steps, cfg.dataset.horizon_steps

    scaler = Scaler().fit(series[:, :n_train_steps, :])
    scaled = scaler.transform(series)

    n_sats = series.shape[0]
    train_idx, val_idx = make_train_val_indices(
        n_sats, n_train_steps, lookback, horizon, cfg.train.val_fraction
    )
    train_ds = WindowDataset(scaled, train_idx, lookback, horizon)
    val_ds = WindowDataset(scaled, val_idx, lookback, horizon)
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False)

    model = RNNForecaster(
        n_channels=len(channels),
        horizon_steps=horizon,
        n_satellites=n_sats,
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        sat_embed_dim=cfg.model.sat_embed_dim,
        rnn_type=cfg.model.rnn_type,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    loss_fn = torch.nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience_left = cfg.train.early_stopping_patience
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(cfg.train.epochs):
        model.train()
        train_losses = []
        for x, sat_idx, y in train_loader:
            x, sat_idx, y = x.to(device), sat_idx.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x, sat_idx)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, sat_idx, y in val_loader:
                x, sat_idx, y = x.to(device), sat_idx.to(device), y.to(device)
                pred = model(x, sat_idx)
                val_losses.append(loss_fn(pred, y).item())

        train_loss, val_loss = float(np.mean(train_losses)), float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"epoch {epoch + 1:3d}/{cfg.train.epochs}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_left = cfg.train.early_stopping_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch + 1} (best val_loss={best_val_loss:.5f})")
                break

    model.load_state_dict(best_state)

    checkpoint = {
        "model_state": model.state_dict(),
        "model_config": dict(
            n_channels=len(channels),
            horizon_steps=horizon,
            n_satellites=n_sats,
            hidden_size=cfg.model.hidden_size,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            sat_embed_dim=cfg.model.sat_embed_dim,
            rnn_type=cfg.model.rnn_type,
        ),
        "scaler_state": scaler.state_dict(),
        "sat_id_to_idx": sat_id_to_idx,
        "sat_ids": sat_ids,
        "sat_types": sat_types,
        "channels": channels,
        "lookback_steps": lookback,
        "horizon_steps": horizon,
        "step_minutes": cfg.data.step_minutes,
        "best_val_loss": best_val_loss,
        "history": history,
    }
    ckpt_path = cfg.train.checkpoint_path
    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
    torch.save(checkpoint, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path} (best_val_loss={best_val_loss:.5f})")
    return checkpoint
