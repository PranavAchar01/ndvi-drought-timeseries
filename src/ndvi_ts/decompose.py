"""Seasonal-trend decomposition for monthly NDVI without heavy dependencies."""
from __future__ import annotations

import numpy as np
import pandas as pd


def climatology(s: pd.Series) -> pd.Series:
    """Mean value per calendar month (the seasonal 'normal')."""
    return s.groupby(s.index.month).mean()


def _centered_moving_average(s: pd.Series, window: int = 12) -> pd.Series:
    """Centered MA; for even windows use the 2x(window) convention."""
    if window % 2 == 0:
        ma = s.rolling(window, center=True).mean()
        ma = ma.rolling(2, center=True).mean().shift(-1)
    else:
        ma = s.rolling(window, center=True).mean()
    return ma


def seasonal_decompose(s: pd.Series, period: int = 12) -> pd.DataFrame:
    """Additive decomposition -> DataFrame[observed, trend, seasonal, residual].

    Classical method: trend via centered moving average, seasonal as the mean
    detrended value per period position (normalized to sum ~0), residual is the
    remainder. Mirrors statsmodels' classical decomposition without the dep.
    """
    s = s.astype(float)
    trend = _centered_moving_average(s, period)
    detrended = s - trend
    pos = np.arange(len(s)) % period
    seasonal_means = pd.Series(detrended.to_numpy(), index=pos).groupby(level=0).mean()
    seasonal_means = seasonal_means - seasonal_means.mean()
    seasonal = pd.Series(seasonal_means.reindex(pos).to_numpy(), index=s.index)
    residual = s - trend - seasonal
    return pd.DataFrame({"observed": s, "trend": trend,
                         "seasonal": seasonal, "residual": residual})
