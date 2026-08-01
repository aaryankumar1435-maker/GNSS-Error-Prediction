"""Windowed lookback/horizon dataset built from the long-format error CSV."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class Scaler:
    """Per-channel standardization fit on training data only."""

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, arr: np.ndarray) -> "Scaler":
        # arr: (n_sats, n_steps, n_channels)
        self.mean = arr.mean(axis=(0, 1))
        std = arr.std(axis=(0, 1))
        std[std < 1e-6] = 1e-6
        self.std = std
        return self

    def transform(self, arr: np.ndarray) -> np.ndarray:
        return (arr - self.mean) / self.std

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        return arr * self.std + self.mean

    def state_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    def load_state_dict(self, state: dict) -> "Scaler":
        self.mean = np.array(state["mean"], dtype=np.float32)
        self.std = np.array(state["std"], dtype=np.float32)
        return self


def build_series(df: pd.DataFrame, channels: Sequence[str]):
    """Pivot the long-format dataframe into a (n_sats, n_steps, n_channels) array."""
    sat_ids = sorted(df["sat_id"].unique())
    sat_id_to_idx = {s: i for i, s in enumerate(sat_ids)}
    sat_types = []
    series = []
    for s in sat_ids:
        sub = df[df["sat_id"] == s].sort_values("timestamp")
        series.append(sub[list(channels)].to_numpy(dtype=np.float32))
        sat_types.append(sub["sat_type"].iloc[0])
    series = np.stack(series, axis=0)
    return series, sat_id_to_idx, sat_ids, sat_types


def make_train_val_indices(n_sats: int, n_train_steps: int, lookback: int, horizon: int,
                            val_fraction: float):
    max_start = n_train_steps - lookback - horizon
    if max_start < 0:
        raise ValueError("train_days too short for the configured lookback + horizon")
    starts = np.arange(0, max_start + 1)
    n_val = max(1, int(len(starts) * val_fraction))
    train_starts, val_starts = starts[: len(starts) - n_val], starts[len(starts) - n_val:]
    train_idx = [(s, int(st)) for s in range(n_sats) for st in train_starts]
    val_idx = [(s, int(st)) for s in range(n_sats) for st in val_starts]
    return train_idx, val_idx


def make_test_indices(n_sats: int, n_total_steps: int, lookback: int, horizon: int):
    start = n_total_steps - lookback - horizon
    if start < 0:
        raise ValueError("Not enough steps for one test window")
    return [(s, start) for s in range(n_sats)]


class WindowDataset(Dataset):
    def __init__(self, series: np.ndarray, index: list[tuple[int, int]], lookback: int, horizon: int):
        self.series = series
        self.index = index
        self.lookback = lookback
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        sat_row, start = self.index[i]
        x = self.series[sat_row, start: start + self.lookback]
        y = self.series[sat_row, start + self.lookback: start + self.lookback + self.horizon]
        return torch.from_numpy(x), torch.tensor(sat_row, dtype=torch.long), torch.from_numpy(y)
