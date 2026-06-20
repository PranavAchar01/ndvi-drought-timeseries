"""Generate and load NDVI (Normalized Difference Vegetation Index) time series.

NDVI ranges roughly [-0.2, 0.9] for land; healthy vegetation is high. We model a
monthly series as: baseline + linear trend + annual seasonality + drought dips +
noise, clipped to a physical range. Real MODIS/Landsat NDVI exports (a date
column + a value column) load via ``load_ndvi_csv``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_ndvi(
    start: str = "2005-01-01",
    periods: int = 240,                 # 20 years monthly
    freq: str = "MS",
    base: float = 0.55,
    trend_per_year: float = -0.004,     # gradual browning
    seasonal_amp: float = 0.18,
    drought_periods: tuple = ((60, 8, 0.20), (150, 6, 0.28)),  # (start_idx, len, depth)
    noise: float = 0.02,
    seed: int = 0,
) -> pd.Series:
    """Return a monthly NDVI ``pd.Series`` indexed by date.

    ``drought_periods`` is a list of (start_index, length_months, depth) triples
    that subtract a smooth dip from the signal to mimic vegetation stress.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=periods, freq=freq)
    t = np.arange(periods)
    years = t / 12.0
    month = idx.month.to_numpy()

    trend = trend_per_year * years
    # Peak greenness in northern-hemisphere summer (month ~7)
    seasonal = seasonal_amp * np.sin(2 * np.pi * (month - 4) / 12.0)
    signal = base + trend + seasonal

    for start_idx, length, depth in drought_periods:
        if start_idx >= periods:
            continue
        end_idx = min(start_idx + length, periods)
        span = end_idx - start_idx
        if span <= 0:
            continue
        ramp = np.resize(np.hanning(max(span, 2)), span)
        signal[start_idx:end_idx] -= depth * ramp

    signal = signal + rng.normal(0, noise, size=periods)
    signal = np.clip(signal, -0.2, 0.95)
    return pd.Series(signal, index=idx, name="ndvi")


def load_ndvi_csv(path: str, date_col: str = "date", value_col: str = "ndvi") -> pd.Series:
    """Load an NDVI CSV into a date-indexed Series, sorted and de-duplicated."""
    df = pd.read_csv(path)
    if date_col not in df or value_col not in df:
        raise KeyError(f"CSV must contain {date_col!r} and {value_col!r} columns.")
    s = pd.Series(df[value_col].to_numpy(dtype=float),
                  index=pd.to_datetime(df[date_col]), name="ndvi")
    return s[~s.index.duplicated(keep="first")].sort_index()
