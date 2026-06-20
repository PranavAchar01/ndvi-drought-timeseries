import numpy as np
import pandas as pd
from ndvi_ts.data import make_synthetic_ndvi
from ndvi_ts.decompose import seasonal_decompose, climatology
from ndvi_ts.anomaly import (standardized_anomaly, vegetation_condition_index,
                             detect_drought_events)


def test_synthetic_shape_and_range():
    s = make_synthetic_ndvi(periods=120)
    assert len(s) == 120
    assert s.min() >= -0.2 and s.max() <= 0.95


def test_decompose_reconstructs():
    s = make_synthetic_ndvi(periods=120)
    d = seasonal_decompose(s)
    recon = d["trend"] + d["seasonal"] + d["residual"]
    ok = recon.dropna()
    assert np.allclose(ok.values, s.loc[ok.index].values, atol=1e-9)


def test_climatology_has_12_months():
    s = make_synthetic_ndvi(periods=120)
    assert len(climatology(s)) == 12


def test_vci_bounds():
    s = make_synthetic_ndvi(periods=240)
    vci = vegetation_condition_index(s).dropna()
    assert vci.min() >= -1e-6 and vci.max() <= 100 + 1e-6


def test_anomaly_zero_mean():
    s = make_synthetic_ndvi(periods=240)
    a = standardized_anomaly(s).dropna()
    assert abs(a.mean()) < 0.2


def test_detect_known_drought():
    # Drought injected around indices 60 and 150 in the default generator
    s = make_synthetic_ndvi(periods=240, seed=0)
    events = detect_drought_events(standardized_anomaly(s), threshold=-1.0)
    assert len(events) >= 1
    assert all(e["duration_months"] >= 2 for e in events)
