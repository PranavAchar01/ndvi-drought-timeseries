import numpy as np
from ndvi_ts.trend import mann_kendall, theil_sen_slope


def test_mk_detects_increasing():
    x = np.arange(50) + np.random.default_rng(0).normal(0, 0.5, 50)
    r = mann_kendall(x)
    assert r.trend == "increasing" and r.p < 0.05


def test_mk_detects_decreasing():
    x = -np.arange(50) + np.random.default_rng(1).normal(0, 0.5, 50)
    assert mann_kendall(x).trend == "decreasing"


def test_mk_no_trend_on_noise():
    x = np.random.default_rng(2).normal(0, 1, 80)
    assert mann_kendall(x).trend == "no trend"


def test_theil_sen_recovers_slope():
    t = np.arange(100)
    x = 0.3 * t + 2 + np.random.default_rng(3).normal(0, 1, 100)
    slope, intercept, lo, hi = theil_sen_slope(x)
    assert abs(slope - 0.3) < 0.05
    assert lo <= slope <= hi
