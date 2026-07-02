"""Regression tests for analysis/trace_analysis.py (count-rate diagnostics).

This module had NO test coverage. The functions assess DLS data quality from an
intensity trace: summary stats, robust baseline, shot-noise outlier flagging,
running average, blocking (correlation) analysis, the count-rate histogram +
Fano factor, and the Augmented Dickey-Fuller stationarity test.

Ground truth here is INDEPENDENT of the program: Poisson shot noise has Fano
factor 1 and std = sqrt(mean); a random walk has a unit root (non-stationary);
an AR(1) process is positively correlated; spikes sit at indices we choose. So a
pass means the diagnostic agrees with probability theory, not with a forward
model of its own.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from analysis import trace_analysis as T

from fixtures import synthetic_trace as ST

MEAN = 10_000.0   # cps; with dt = 1 s, sqrt(mean) = 100 cps shot-noise sigma


# ------------------------------------------------------- summary statistics ---

def test_trace_statistics_recovers_poisson_mean_and_sigma():
    tr = ST.poisson_trace(MEAN, n=4000, seed=1)
    s = T.compute_trace_statistics(tr)
    assert s.n_points == 4000
    assert s.mean_cps == pytest.approx(MEAN, rel=0.01)          # law of large numbers
    assert s.std_cps == pytest.approx(math.sqrt(MEAN), rel=0.06)  # Poisson: sigma = sqrt(mean)
    assert s.minimum_cps <= s.mean_cps <= s.maximum_cps
    assert s.duration_s == pytest.approx(3999.0, rel=1e-9)      # (n-1) * dt


def test_cv_matches_shot_noise():
    tr = ST.poisson_trace(MEAN, n=4000, seed=2)
    s = T.compute_trace_statistics(tr)
    # CV = sigma/mean = 1/sqrt(mean) for a Poisson process.
    assert s.cv == pytest.approx(1.0 / math.sqrt(MEAN), rel=0.08)


# --------------------------------------------------------------- baseline ---

def test_baseline_percentile_robust_to_spikes():
    clean = ST.poisson_trace(MEAN, n=2000, seed=3)
    spiked, _ = ST.poisson_trace_with_spikes(MEAN, n=2000, seed=3)
    b_clean = T.identify_baseline(clean, method="percentile", parameter=25.0)
    b_spiked = T.identify_baseline(spiked, method="percentile", parameter=25.0)
    # A lower-percentile baseline barely moves when upward dust spikes are added.
    assert b_spiked.baseline_cps == pytest.approx(b_clean.baseline_cps, rel=0.02)
    # 25th percentile of a symmetric distribution sits below the mean.
    assert b_clean.baseline_cps < MEAN


def test_baseline_sigma_clip_near_mean_and_uses_fewer_points():
    tr = ST.poisson_trace_with_spikes(MEAN, n=2000, seed=4)[0]
    b = T.identify_baseline(tr, method="sigma_clip", parameter=3.0)
    assert b.method == "sigma_clip"
    assert b.baseline_cps == pytest.approx(MEAN, rel=0.02)   # spikes clipped away
    assert b.n_points_used < 2000                            # some points rejected


def test_baseline_rejects_bad_inputs():
    tr = ST.poisson_trace(MEAN, n=200, seed=5)
    with pytest.raises(ValueError):
        T.identify_baseline(tr, method="percentile", parameter=150.0)
    with pytest.raises(ValueError):
        T.identify_baseline(tr, method="nonsense")


# ---------------------------------------------------------- outlier flags ---

def test_flag_outliers_recovers_exact_spike_indices():
    tr, spikes = ST.poisson_trace_with_spikes(MEAN, n=2000, spike_indices=(250, 900, 1600),
                                              spike_multiple=4.0, seed=6)
    flags = T.flag_outliers(tr, k=5.0)   # 5-sigma envelope: base ~never exceeds, spikes always do
    flagged = set(np.flatnonzero(flags.flagged_mask).tolist())
    assert flagged == set(spikes.tolist())
    assert flags.n_flagged == len(spikes)


def test_flag_outliers_bad_k_raises():
    tr = ST.poisson_trace(MEAN, n=200, seed=7)
    with pytest.raises(ValueError):
        T.flag_outliers(tr, k=0.0)


# ---------------------------------------------------------- normalization ---

def test_normalize_trace_centres_near_one():
    tr = ST.poisson_trace(MEAN, n=2000, seed=8)
    nz = T.normalize_trace(tr)
    # Exact identity: normalized = count_rates / baseline (uses the ACTUAL mean).
    assert nz.normalized.mean() == pytest.approx(
        tr.count_rates_cps.mean() / nz.baseline_cps, rel=1e-9)
    # baseline is the 25th percentile (< mean), so the normalized mean sits above 1.
    assert nz.normalized.mean() > 1.0


# ------------------------------------------------------- running average ---

def test_running_average_flat_trace():
    tr = ST.poisson_trace(MEAN, n=1000, seed=9)
    ra = T.running_average(tr, window_points=51)
    assert ra.running_mean.shape == tr.count_rates_cps.shape
    # A wide window over flat data averages close to the global mean everywhere.
    assert np.all(np.abs(ra.running_mean - MEAN) < 5 * math.sqrt(MEAN))
    assert ra.running_mean.mean() == pytest.approx(MEAN, rel=0.01)


def test_running_average_requires_exactly_one_window():
    tr = ST.poisson_trace(MEAN, n=200, seed=10)
    with pytest.raises(ValueError):
        T.running_average(tr)                                   # neither
    with pytest.raises(ValueError):
        T.running_average(tr, window_s=10.0, window_points=5)   # both


# --------------------------------------------------------- block variance ---

def test_block_variance_uncorrelated_poisson_no_correlations():
    tr = ST.poisson_trace(MEAN, n=4000, seed=11)
    bv = T.block_variance(tr)
    # White noise: SE independent of block size -> ratio ~1, no correlations flagged.
    assert bv.correlations_detected is False
    assert bv.se_ratio == pytest.approx(1.0, abs=0.5)


def test_block_variance_ar1_flags_correlations():
    tr = ST.ar1_trace(MEAN, n=4000, phi=0.9, seed=12)
    bv = T.block_variance(tr)
    # Positive AR(1) correlations make the blocked SE rise with block size.
    assert bv.correlations_detected is True
    assert bv.se_ratio > 1.5


def test_block_variance_too_short_raises():
    short = ST.poisson_trace(MEAN, n=5, seed=13)
    with pytest.raises(ValueError):
        T.block_variance(short)


# ------------------------------------------------- histogram + Fano factor ---

def test_histogram_fano_factor_is_one_for_poisson():
    tr = ST.poisson_trace(MEAN, n=6000, seed=14)
    h = T.fit_count_rate_histogram(tr, integration_time_s=1.0)
    # THE independent truth: variance/mean in count space = 1 for a Poisson process.
    assert h.fano_factor == pytest.approx(1.0, abs=0.15)
    assert h.mean_counts == pytest.approx(MEAN, rel=0.01)
    assert h.shot_noise_cv == pytest.approx(1.0 / math.sqrt(MEAN), rel=0.01)


def test_histogram_no_divide_by_zero_and_finite_chi2():
    # Regression for the empty-bin chi^2 guard (a documented past RuntimeWarning):
    # with expected==0 bins excluded, the reduced chi^2 stays finite / None, never inf.
    tr = ST.poisson_trace(MEAN, n=3000, seed=15)
    h = T.fit_count_rate_histogram(tr, distribution="both", integration_time_s=1.0)
    for chi2 in (h.gaussian_chi2_reduced, h.poisson_chi2_reduced):
        assert chi2 is None or math.isfinite(chi2)


def test_histogram_bad_arguments_raise():
    tr = ST.poisson_trace(MEAN, n=500, seed=16)
    with pytest.raises(ValueError):
        T.fit_count_rate_histogram(tr, distribution="weibull")
    with pytest.raises(ValueError):
        T.fit_count_rate_histogram(tr, bin_method="scott")


# --------------------------------------------------- stationarity (ADF) ---

def test_adf_flags_stationary_poisson():
    tr = ST.poisson_trace(MEAN, n=1500, seed=17)
    r = T.test_stationarity_adf(tr)
    # White Poisson noise is stationary: ADF rejects the unit-root null (p small).
    assert r.is_stationary is True
    assert r.p_value < 0.05


def test_adf_flags_nonstationary_random_walk():
    tr = ST.random_walk_trace(MEAN, n=1500, seed=18)
    r = T.test_stationarity_adf(tr)
    # A random walk has a unit root: ADF fails to reject -> non-stationary.
    assert r.is_stationary is False
    assert r.p_value >= 0.05
