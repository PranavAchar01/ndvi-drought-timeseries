"""Generate 20 years of NDVI, analyze trend + drought, and save a figure."""
import json
from ndvi_ts.data import make_synthetic_ndvi
from ndvi_ts.cli import analyze

s = make_synthetic_ndvi()
print(json.dumps(analyze(s), indent=2))
