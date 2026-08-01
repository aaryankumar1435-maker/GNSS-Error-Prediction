"""Synthetic generator for satellite clock & ephemeris upload-vs-model errors.

Models the real dynamic the challenge is about: a ground upload resets the
broadcast (uploaded) parameters close to the truth, then the gap between the
uploaded values and the ICD-propagated model grows until the next upload
(sawtooth), riding on top of an orbital-period periodic term and noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CHANNELS = ["clock_error_ns", "eph_radial_m", "eph_along_m", "eph_cross_m"]

# (growth_rate_mean, growth_rate_std), noise_std, (periodic_amp_mean, periodic_amp_std), upload_interval_minutes
_GEO_PARAMS = {
    "clock_error_ns": dict(growth=(0.08, 0.02), noise=0.15, amp=(0.6, 0.2), upload_min=120),
    "eph_radial_m": dict(growth=(0.004, 0.001), noise=0.01, amp=(0.05, 0.02), upload_min=120),
    "eph_along_m": dict(growth=(0.02, 0.005), noise=0.03, amp=(0.3, 0.1), upload_min=120),
    "eph_cross_m": dict(growth=(0.003, 0.001), noise=0.01, amp=(0.04, 0.015), upload_min=120),
}
_MEO_PARAMS = {
    "clock_error_ns": dict(growth=(0.15, 0.03), noise=0.2, amp=(0.8, 0.25), upload_min=60),
    "eph_radial_m": dict(growth=(0.008, 0.002), noise=0.015, amp=(0.08, 0.03), upload_min=60),
    "eph_along_m": dict(growth=(0.05, 0.01), noise=0.05, amp=(0.5, 0.15), upload_min=60),
    "eph_cross_m": dict(growth=(0.006, 0.0015), noise=0.012, amp=(0.06, 0.02), upload_min=60),
}
_ORBITAL_PERIOD_MIN = {"GEO": 1436.0, "MEO": 718.0}
_PARAMS_BY_TYPE = {"GEO": _GEO_PARAMS, "MEO": _MEO_PARAMS}


def _simulate_channel(n_steps: int, step_minutes: int, rng: np.random.Generator,
                       params: dict, orbital_period_min: float) -> np.ndarray:
    growth_mean, growth_std = params["growth"]
    growth_rate = max(rng.normal(growth_mean, growth_std), growth_mean * 0.2)
    upload_min = params["upload_min"]
    amp_mean, amp_std = params["amp"]
    amplitude = max(rng.normal(amp_mean, amp_std), 0.0)
    phase = rng.uniform(0, 2 * np.pi)
    noise_std = params["noise"]

    t_minutes = np.arange(n_steps) * step_minutes
    tau = t_minutes % upload_min  # minutes elapsed since the last upload -> sawtooth reset

    growth_term = growth_rate * tau + 0.15 * growth_rate * (tau ** 2) / upload_min
    periodic_term = amplitude * np.sin(2 * np.pi * t_minutes / orbital_period_min + phase)
    noise_term = rng.normal(0.0, noise_std, size=n_steps)
    residual_after_upload = rng.normal(0.0, noise_std * 0.5, size=n_steps)

    return growth_term + periodic_term + noise_term + residual_after_upload


def generate_dataset(n_geo: int, n_meo: int, train_days: int, test_days: int,
                      step_minutes: int, seed: int,
                      start_time: str = "2025-01-01T00:00:00") -> pd.DataFrame:
    total_days = train_days + test_days
    n_steps = int(total_days * 24 * 60 / step_minutes)
    timestamps = pd.date_range(start=start_time, periods=n_steps, freq=f"{step_minutes}min")

    sat_specs = [("GEO", i) for i in range(1, n_geo + 1)] + [("MEO", i) for i in range(1, n_meo + 1)]

    frames = []
    for sat_type, idx in sat_specs:
        sat_id = f"{sat_type[0]}{idx:02d}"
        type_offset = 0 if sat_type == "GEO" else 5000
        sat_rng = np.random.default_rng(seed + type_offset + idx)
        orbital_period = _ORBITAL_PERIOD_MIN[sat_type]
        params_table = _PARAMS_BY_TYPE[sat_type]

        channel_data = {
            ch: _simulate_channel(n_steps, step_minutes, sat_rng, params_table[ch], orbital_period)
            for ch in CHANNELS
        }
        df = pd.DataFrame(channel_data)
        df.insert(0, "timestamp", timestamps)
        df.insert(1, "sat_id", sat_id)
        df.insert(2, "sat_type", sat_type)
        frames.append(df)

    full = pd.concat(frames, ignore_index=True)
    return full.sort_values(["sat_id", "timestamp"]).reset_index(drop=True)


def save_dataset(df: pd.DataFrame, csv_path: str) -> None:
    df.to_csv(csv_path, index=False)
