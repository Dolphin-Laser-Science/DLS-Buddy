"""Synthetic count-rate trace builders for the trace-diagnostics tests.

These build :class:`core.data_models.IntensityTrace` objects whose ground truth is
*independent* of the code under test — Poisson shot noise (Fano factor = 1 exactly),
spikes at known indices, a random walk (non-stationary by construction), and an
AR(1) process (positively correlated by construction). That independence is the
point: `analysis/trace_analysis.py` should recover facts that come from probability
theory, not from the program's own forward model.

Sampling convention: uniform 1 s spacing (``dt_s = 1.0``) so that a count *rate*
in cps equals the integer photon *count* per sample — which makes the Poisson /
Fano bookkeeping exact and the histogram fit's count-space statistics trivial to
reason about. All randomness is seeded (deterministic).
"""
from __future__ import annotations

import numpy as np

from core.data_models import IntensityTrace


def _times(n, dt_s):
    return np.arange(n, dtype=float) * dt_s


def poisson_trace(mean_cps=10_000.0, n=2000, dt_s=1.0, seed=0):
    """A flat, stationary, shot-noise-limited trace.

    Counts per sample ~ Poisson(lambda), lambda = mean_cps * dt_s; the rate is
    counts / dt_s. With dt_s = 1 the Fano factor (var/mean in count space) is 1
    to sampling accuracy and std_cps ~ sqrt(mean_cps).
    """
    rng = np.random.RandomState(seed)
    lam = mean_cps * dt_s
    counts = rng.poisson(lam, size=n).astype(float)
    return IntensityTrace(times_s=_times(n, dt_s), count_rates_cps=counts / dt_s)


def poisson_trace_with_spikes(mean_cps=10_000.0, n=2000, spike_indices=(250, 900, 1600),
                              spike_multiple=4.0, dt_s=1.0, seed=0):
    """A flat Poisson trace with dust-like spikes at KNOWN indices.

    Returns ``(trace, spike_indices)``. Each spike sample is set to
    ``spike_multiple * mean_cps`` — far outside the mean +/- k*sqrt(mean) shot-noise
    envelope for any reasonable k, so an outlier flagger must recover exactly
    these indices and no others.
    """
    base = poisson_trace(mean_cps, n, dt_s, seed)
    y = base.count_rates_cps.copy()
    idx = np.asarray(sorted(set(int(i) for i in spike_indices)), dtype=int)
    y[idx] = spike_multiple * mean_cps
    return IntensityTrace(times_s=base.times_s.copy(), count_rates_cps=y), idx


def random_walk_trace(mean_cps=10_000.0, n=2000, step_cps=50.0, dt_s=1.0, seed=0):
    """A drifting, NON-stationary trace (a bounded random walk).

    A cumulative sum of zero-mean steps has a unit root, so the Augmented
    Dickey-Fuller test should FAIL to reject non-stationarity (is_stationary=False).
    Kept positive by construction so it reads as a physical count rate.
    """
    rng = np.random.RandomState(seed)
    walk = np.cumsum(rng.normal(0.0, step_cps, size=n))
    y = np.abs(mean_cps + walk) + 1.0
    return IntensityTrace(times_s=_times(n, dt_s), count_rates_cps=y)


def ar1_trace(mean_cps=10_000.0, n=2000, phi=0.9, noise_cps=100.0, dt_s=1.0, seed=0):
    """A stationary but POSITIVELY CORRELATED trace (AR(1), 0 < phi < 1).

    x[t] = mean + phi*(x[t-1] - mean) + eps. Its slow-mode correlations make the
    blocking standard error rise with block size (correlations_detected=True),
    unlike white Poisson noise. Stays stationary (|phi| < 1).
    """
    rng = np.random.RandomState(seed)
    eps = rng.normal(0.0, noise_cps, size=n)
    x = np.empty(n, dtype=float)
    x[0] = mean_cps
    for t in range(1, n):
        x[t] = mean_cps + phi * (x[t - 1] - mean_cps) + eps[t]
    return IntensityTrace(times_s=_times(n, dt_s), count_rates_cps=x)
