"""Drought-relevant anomaly metrics derived from NDVI time series."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List

import numpy as np
import pandas as pd


def standardized_anomaly(s: pd.Series) -> pd.Series:
    """Per-calendar-month z-score: (x - clim_mean) / clim_std.

    Negative values mean vegetation is below its seasonal normal — a drought
    signal that removes the seasonal cycle so anomalies are comparable in any
    month."""
    by_month = s.groupby(s.index.month)
    mean = by_month.transform("mean")
    std = by_month.transform("std").replace(0, np.nan)
    return ((s - mean) / std).rename("anomaly")


def vegetation_condition_index(s: pd.Series) -> pd.Series:
    """VCI = 100 * (NDVI - NDVI_min) / (NDVI_max - NDVI_min), per calendar month.

    A widely used drought index (Kogan, 1995): 0 = worst observed condition,
    100 = best. VCI < 35 commonly flags vegetation drought stress."""
    g = s.groupby(s.index.month)
    mn = g.transform("min")
    mx = g.transform("max")
    rng = (mx - mn).replace(0, np.nan)
    return (100 * (s - mn) / rng).rename("vci")


@dataclass
class DroughtEvent:
    start: str
    end: str
    duration_months: int
    min_anomaly: float
    mean_anomaly: float


def detect_drought_events(anomaly: pd.Series, threshold: float = -1.0,
                          min_duration: int = 2) -> List[dict]:
    """Find runs where standardized anomaly stays <= threshold.

    Returns a list of event dicts (start, end, duration, severity). This is a
    run-length scan over the boolean drought mask."""
    mask = (anomaly <= threshold).to_numpy()
    events: List[dict] = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            if (j - i + 1) >= min_duration:
                seg = anomaly.iloc[i:j + 1]
                events.append(asdict(DroughtEvent(
                    start=str(anomaly.index[i].date()),
                    end=str(anomaly.index[j].date()),
                    duration_months=int(j - i + 1),
                    min_anomaly=float(seg.min()),
                    mean_anomaly=float(seg.mean()),
                )))
            i = j + 1
        else:
            i += 1
    return events
