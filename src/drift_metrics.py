"""
PSI and KS-test implementations for distribution drift detection.

PSI (Population Stability Index):
  - Manual implementation using decile binning
  - Cross-checked against scipy's ks_2samp for the KS side

Thresholds (industry standard, from credit scoring):
  PSI < 0.10  : no meaningful shift
  PSI 0.10-0.20: moderate shift, monitor closely
  PSI > 0.20  : significant shift, alert
"""
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index.

    PSI = Σ (actual% − expected%) × ln(actual% / expected%)

    Bins are computed from the reference distribution (decile breakpoints).
    A small epsilon prevents log(0) on empty bins.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]

    if len(reference) == 0 or len(current) == 0:
        return np.nan

    # Build bin edges from reference deciles
    percentiles = np.linspace(0, 100, bins + 1)
    breakpoints = np.nanpercentile(reference, percentiles)
    # Collapse duplicate edges (happens with discrete/sparse data)
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return np.nan
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    ref_counts = np.histogram(reference, bins=breakpoints)[0].astype(float)
    cur_counts = np.histogram(current, bins=breakpoints)[0].astype(float)

    # ponytail: epsilon avoids log(0); adds negligible bias on large samples
    eps = 1e-6
    ref_pct = ref_counts / ref_counts.sum() + eps
    cur_pct = cur_counts / cur_counts.sum() + eps

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def ks_test(reference: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
    """
    Two-sample KS test. Returns (statistic, p_value).
    Wraps scipy.stats.ks_2samp — verified to match it exactly (see tests).
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    return ks_2samp(reference, current)


def null_rate(series: pd.Series) -> float:
    return float(series.isna().mean())


def monitor_window(
    reference_df: pd.DataFrame,
    window_df: pd.DataFrame,
    feature_cols: list,
    model,
    fill_values: pd.Series,
) -> Dict:
    """
    Compute PSI, KS, null rates, and prediction/error drift for one window.

    Returns:
        feature_psi   : {col: float}
        feature_ks    : {col: {"statistic": float, "p_value": float}}
        null_rates    : {col: float}
        pred_psi      : float  — distribution of model outputs
        error_psi     : float  — distribution of |predicted − actual| (concept drift signal)
        window_rmse   : float
    """
    result: Dict = {"feature_psi": {}, "feature_ks": {}, "null_rates": {}}

    for col in feature_cols:
        ref_vals = reference_df[col].values
        cur_vals = window_df[col].values
        result["feature_psi"][col] = psi(ref_vals, cur_vals)
        stat, pval = ks_test(ref_vals, cur_vals)
        result["feature_ks"][col] = {"statistic": float(stat), "p_value": float(pval)}
        result["null_rates"][col] = null_rate(window_df[col])

    X_ref = reference_df[feature_cols].fillna(fill_values)
    X_cur = window_df[feature_cols].fillna(fill_values)
    ref_preds = model.predict(X_ref)
    cur_preds = model.predict(X_cur)
    result["pred_psi"] = psi(ref_preds, cur_preds)

    if "rating" in window_df.columns and "rating" in reference_df.columns:
        ref_errors = np.abs(ref_preds - reference_df["rating"].values)
        cur_errors = np.abs(cur_preds - window_df["rating"].values)
        result["error_psi"] = psi(ref_errors, cur_errors)
        result["window_rmse"] = float(np.sqrt(np.mean((cur_preds - window_df["rating"].values) ** 2)))

    return result
