"""CLI: run a full NDVI trend + drought analysis on real or synthetic data."""
from __future__ import annotations

import argparse
import json

from .data import make_synthetic_ndvi, load_ndvi_csv
from .trend import mann_kendall, theil_sen_slope
from .decompose import seasonal_decompose
from .anomaly import standardized_anomaly, vegetation_condition_index, detect_drought_events


def analyze(s, threshold: float = -1.0):
    mk = mann_kendall(s.to_numpy())
    slope, intercept, lo, hi = theil_sen_slope(s.to_numpy())
    anom = standardized_anomaly(s)
    vci = vegetation_condition_index(s)
    events = detect_drought_events(anom, threshold=threshold)
    return {
        "n_obs": int(s.size),
        "span": [str(s.index.min().date()), str(s.index.max().date())],
        "mean_ndvi": round(float(s.mean()), 4),
        "trend": mk.trend,
        "mann_kendall": {"z": round(mk.z, 3), "p": round(mk.p, 5), "tau": round(mk.tau, 3)},
        "theil_sen_slope_per_month": round(slope, 6),
        "theil_sen_slope_per_year": round(slope * 12, 6),
        "min_vci": round(float(vci.min()), 1),
        "drought_events": events,
    }


def main():
    p = argparse.ArgumentParser(description="NDVI trend & drought analysis")
    p.add_argument("--csv", default=None, help="CSV with date,ndvi columns")
    p.add_argument("--threshold", type=float, default=-1.0,
                   help="Standardized-anomaly threshold for drought (default -1.0)")
    p.add_argument("--plot", default=None, help="Optional output PNG path")
    args = p.parse_args()

    s = load_ndvi_csv(args.csv) if args.csv else make_synthetic_ndvi()
    report = analyze(s, args.threshold)
    print(json.dumps(report, indent=2))

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        dec = seasonal_decompose(s)
        anom = standardized_anomaly(s)
        fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        ax[0].plot(s.index, s.values, lw=1); ax[0].plot(dec.index, dec["trend"], "r", lw=2)
        ax[0].set_ylabel("NDVI"); ax[0].set_title("NDVI with Theil-Sen / MA trend")
        ax[1].plot(dec.index, dec["seasonal"]); ax[1].set_ylabel("Seasonal")
        ax[2].axhline(args.threshold, color="r", ls="--")
        ax[2].fill_between(anom.index, anom.values, args.threshold,
                           where=(anom.values <= args.threshold), color="orange", alpha=.6)
        ax[2].plot(anom.index, anom.values, lw=1); ax[2].set_ylabel("Std. anomaly")
        fig.tight_layout(); fig.savefig(args.plot, dpi=120)
        print(f"Saved plot -> {args.plot}")


if __name__ == "__main__":
    main()
