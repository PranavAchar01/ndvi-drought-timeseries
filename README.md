# NDVI Drought Time-Series Toolkit

Trend, seasonality, and drought-anomaly analysis for **vegetation index time series** (NDVI), the kind produced by MODIS, Landsat, and Sentinel-2 surface-reflectance products. Built for **geospatial time-series analysis** of the Earth's water/vegetation cycle.

Given a date-indexed NDVI series it answers three questions that matter for drought and ecosystem monitoring:

1. **Is vegetation trending up or down?** — Mann-Kendall test + Theil-Sen slope (robust, non-parametric).
2. **What is the seasonal cycle, and what's left after removing it?** — classical seasonal-trend decomposition.
3. **When was vegetation under stress?** — standardized anomalies, the Vegetation Condition Index (VCI), and automatic drought-event detection.

## Why these methods

For environmental series, residuals are rarely normal and outliers are common, so **Mann-Kendall** and **Theil-Sen** are the field-standard trend tools. **VCI** (Kogan, 1995) and per-month **standardized anomalies** remove the seasonal cycle so a "low" reading in winter is comparable to one in summer — essential for honest drought flagging.

## Quickstart

```bash
pip install -e ".[dev,plot]"

# Analyze a real NDVI export (CSV with date,ndvi columns)
ndvi-analyze --csv my_ndvi.csv --plot report.png

# Or run on 20 years of synthetic data with injected droughts
ndvi-analyze --plot report.png
```

Example output:

```json
{
  "trend": "decreasing",
  "mann_kendall": {"z": -3.9, "p": 0.0001, "tau": -0.18},
  "theil_sen_slope_per_year": -0.0041,
  "min_vci": 2.7,
  "drought_events": [
    {"start": "2010-01-01", "end": "2010-08-01", "duration_months": 8,
     "min_anomaly": -2.3, "mean_anomaly": -1.6}
  ]
}
```

## Input format

A CSV with a date column and a value column:

```
date,ndvi
2005-01-01,0.54
2005-02-01,0.56
...
```

Pass custom column names to `load_ndvi_csv(path, date_col=..., value_col=...)`.

## API

```python
from ndvi_ts import (make_synthetic_ndvi, mann_kendall, theil_sen_slope,
                     seasonal_decompose, standardized_anomaly,
                     vegetation_condition_index, detect_drought_events)

s = make_synthetic_ndvi()
mk = mann_kendall(s.to_numpy())            # -> MKResult(trend, z, p, tau)
dec = seasonal_decompose(s)                # observed/trend/seasonal/residual
events = detect_drought_events(standardized_anomaly(s), threshold=-1.0)
```

## Tests

```bash
pytest -q
```

Tests verify trend recovery on known slopes, decomposition reconstruction, VCI/anomaly bounds, and detection of injected drought events.

## License

MIT — see [LICENSE](LICENSE).
