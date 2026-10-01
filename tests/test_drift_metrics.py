"""
Correctness tests for PSI and KS implementations.
All PSI cases are verifiable by hand or against scipy.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from drift_metrics import ks_test, psi


# ── PSI ──────────────────────────────────────────────────────────────────────

def test_psi_zero_on_identical():
    """PSI = 0 when actual distribution equals expected exactly."""
    rng = np.random.RandomState(42)
    ref = rng.normal(3.5, 0.5, 10_000)
    result = psi(ref, ref)
    assert result < 1e-4, f"Expected ~0, got {result}"


def test_psi_positive_on_any_shift():
    rng = np.random.RandomState(1)
    ref = rng.normal(0, 1, 5_000)
    slightly_shifted = rng.normal(0.3, 1, 5_000)
    assert psi(ref, slightly_shifted) > 0


def test_psi_above_threshold_on_large_shift():
    """PSI > 0.2 (critical) on a known large distributional shift."""
    rng = np.random.RandomState(7)
    ref = rng.normal(3.5, 0.5, 10_000)
    shifted = rng.normal(5.5, 0.4, 10_000)  # mean shift of 2, beyond rating scale edge
    result = psi(ref, shifted)
    assert result > 0.2, f"Expected PSI > 0.2, got {result:.4f}"


def test_psi_monotone_with_shift_magnitude():
    """Larger shift → larger PSI (monotone relationship)."""
    rng = np.random.RandomState(99)
    ref = rng.normal(0, 1, 8_000)
    small = rng.normal(0.4, 1, 8_000)
    large = rng.normal(1.5, 1, 8_000)
    assert psi(ref, small) < psi(ref, large), "PSI should grow with shift magnitude"


def test_psi_symmetric_approximation():
    """
    PSI is not symmetric (it's KL-based), but swapping reference/current
    should both produce non-zero values when distributions differ.
    """
    rng = np.random.RandomState(5)
    a = rng.normal(0, 1, 5_000)
    b = rng.normal(1, 1, 5_000)
    assert psi(a, b) > 0
    assert psi(b, a) > 0


def test_psi_nan_on_empty_input():
    result = psi(np.array([]), np.array([1.0, 2.0]))
    assert np.isnan(result)


# ── KS test ──────────────────────────────────────────────────────────────────

def test_ks_matches_scipy_exactly():
    """Our ks_test must be a thin wrapper — statistic and p-value identical to scipy."""
    rng = np.random.RandomState(42)
    a = rng.normal(0, 1, 1_000)
    b = rng.normal(0.5, 1.2, 1_000)
    our_stat, our_pval = ks_test(a, b)
    sci_stat, sci_pval = ks_2samp(a, b)
    assert abs(our_stat - sci_stat) < 1e-12
    assert abs(our_pval - sci_pval) < 1e-12


def test_ks_high_pvalue_on_similar_distributions():
    """Pre-injection: p-value > 0.05 when distributions are drawn from the same population."""
    rng = np.random.RandomState(3)
    ref = rng.normal(3.5, 0.5, 800)
    current = rng.normal(3.5, 0.5, 800)
    _, pval = ks_test(ref, current)
    assert pval > 0.05, f"Expected p > 0.05 for similar distributions, got {pval:.4f}"


def test_ks_low_pvalue_on_different_distributions():
    """Post-injection: p-value < 0.05 when distributions are meaningfully different."""
    rng = np.random.RandomState(3)
    ref = rng.normal(3.5, 0.5, 800)
    drifted = rng.normal(5.0, 0.3, 800)
    _, pval = ks_test(ref, drifted)
    assert pval < 0.05, f"Expected p < 0.05 for different distributions, got {pval:.4f}"


def test_ks_ignores_nan():
    rng = np.random.RandomState(10)
    a = rng.normal(0, 1, 500)
    b = rng.normal(0, 1, 500)
    b_with_nans = b.copy()
    b_with_nans[:50] = np.nan
    stat_clean, pval_clean = ks_test(a, b)
    stat_nan, pval_nan = ks_test(a, b_with_nans)
    # Both should complete without error; results differ due to smaller sample
    assert not np.isnan(stat_nan)


# ── Known hand-computed PSI case ──────────────────────────────────────────────

def test_psi_hand_computed_case():
    """
    Verify PSI against a hand-computable case:
    reference = uniform on {1,2,3,4,5}, each 20%
    current   = {1:5%, 2:10%, 3:15%, 4:30%, 5:40%}
    Using 5 equal-mass bins:
      bin contributions ≈ sum((act-exp)*ln(act/exp))
    """
    # Generate approximate samples matching target distributions
    ref = np.array([1]*2000 + [2]*2000 + [3]*2000 + [4]*2000 + [5]*2000, dtype=float)
    cur = np.array([1]*500 + [2]*1000 + [3]*1500 + [4]*3000 + [5]*4000, dtype=float)
    # Manually: exp_pct per bin ≈ 0.2; act_pcts ≈ 0.05, 0.10, 0.15, 0.30, 0.40
    # PSI ≈ (0.05-0.2)*ln(0.05/0.2) + (0.10-0.2)*ln(0.10/0.2) + ...
    #      ≈ (-0.15)*(-1.386) + (-0.10)*(-0.693) + (-0.05)*(-0.288) + (0.10)*(0.405) + (0.20)*(0.693)
    #      ≈ 0.208 + 0.069 + 0.014 + 0.041 + 0.139 ≈ 0.471
    result = psi(ref, cur, bins=5)
    # With epsilon the result is slightly different; allow ±0.15 tolerance
    assert 0.3 < result < 0.7, f"Hand-computed PSI expected ~0.47, got {result:.4f}"
