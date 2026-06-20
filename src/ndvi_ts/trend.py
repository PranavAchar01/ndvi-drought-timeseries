"""Non-parametric trend detection: Mann-Kendall test + Theil-Sen slope.

These are the standard tools for monotonic trend in environmental time series
because they tolerate non-normal residuals and outliers far better than OLS.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class MKResult:
    trend: str          # "increasing" | "decreasing" | "no trend"
    s: float            # Mann-Kendall S statistic
    z: float            # normal-approx test statistic
    p: float            # two-sided p-value
    tau: float          # Kendall's tau


def mann_kendall(x, alpha: float = 0.05) -> MKResult:
    """Mann-Kendall monotonic trend test with tie-corrected variance."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n < 4:
        raise ValueError("Mann-Kendall needs at least 4 points.")

    # S statistic
    s = 0
    for k in range(n - 1):
        s += np.sum(np.sign(x[k + 1:] - x[k]))

    # Tie-corrected variance
    _, counts = np.unique(x, return_counts=True)
    tie = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie) / 18.0

    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0

    p = 2 * (1 - stats.norm.cdf(abs(z)))
    tau = stats.kendalltau(np.arange(n), x).statistic

    if p < alpha and z > 0:
        trend = "increasing"
    elif p < alpha and z < 0:
        trend = "decreasing"
    else:
        trend = "no trend"
    return MKResult(trend=trend, s=float(s), z=float(z), p=float(p), tau=float(tau))


def theil_sen_slope(x, time_step: float = 1.0):
    """Theil-Sen median-of-slopes estimate. Returns (slope, intercept, lo, hi)."""
    x = np.asarray(x, dtype=float)
    t = np.arange(x.size) * time_step
    slope, intercept, lo, hi = stats.theilslopes(x, t)
    return float(slope), float(intercept), float(lo), float(hi)
