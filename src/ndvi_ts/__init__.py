"""NDVI vegetation time-series analysis: trend, seasonality, drought anomalies."""
from .data import make_synthetic_ndvi, load_ndvi_csv
from .trend import mann_kendall, theil_sen_slope, MKResult
from .decompose import climatology, seasonal_decompose
from .anomaly import standardized_anomaly, vegetation_condition_index, detect_drought_events

__all__ = [
    "make_synthetic_ndvi", "load_ndvi_csv",
    "mann_kendall", "theil_sen_slope", "MKResult",
    "climatology", "seasonal_decompose",
    "standardized_anomaly", "vegetation_condition_index", "detect_drought_events",
]
__version__ = "0.1.0"
