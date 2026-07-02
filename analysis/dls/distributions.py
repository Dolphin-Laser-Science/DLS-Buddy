"""
analysis/dls/distributions.py
=============================

Distribution methods: recover a full decay-rate / size distribution rather than a
few parameters.

The field ACF is discretised on a grid of decay rates Gamma_n (equivalently
hydrodynamic radii Rh_n):

    |g1(tau_m)|^2  ~  sum_n  x_n exp(-2 Gamma_n tau_m)        (A x)_m

(the factor of 2 follows the Siegert relation, Chu 1991; the cross-terms-negligible
approximation, Liénard et al. 2022 Eq. 11, lets |g1|^2 be written as a single sum
of exponentials in 2 Gamma). The transfer matrix is A[m, n] = exp(-2 Gamma_n
tau_m). We solve for the non-negative weights x_n.

  NNLS      : min ||A x - y||^2          s.t. x >= 0          (no smoothing)
  CONTIN    : min ||A x - y||^2 + alpha^2 ||L x||^2   s.t. x >= 0
              with L the second-difference operator (Provencher's regularizor).
  Lognormal : a single-mode parametric distribution fit through the same A.

Solver note (a deliberate departure from the Salazar et al. 2023 GUI): we do NOT
use SLSQP with a sum(x)=1 equality constraint. Instead we use the standard
augmented-NNLS formulation -- stack [A; alpha L] over [y; 0] and call
scipy.optimize.nnls -- which is faster, more robust, and avoids forcing the
distribution to integrate to exactly 1 when the data do not support it. The
reported distribution is normalised to sum 1 afterwards, for display only.

The data y fed to the solver is the baseline-subtracted, beta-normalised
correlogram, y = (g2 - 1 - baseline) / beta, so that y(0) ~ 1 and the recovered
weights are an intensity-weighted distribution. beta and baseline are estimated by
default (see _estimate_beta / _estimate_baseline in _common) but may be overridden.

This module also owns the distribution post-processing shared by the GUI: the
Rh<->Gamma axis toggle (distribution_axis) and peak detection
(find_distribution_peaks).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy import optimize

from core.data_models import DLSMeasurement
from analysis.dls._common import (
    _apply_tau_window,
    _build_rh_gamma_grid,
    _estimate_baseline,
    _estimate_beta,
    _measurement_q_m,
    _require_viscosity,
    _rms_error,
)
from analysis.dls.cumulants import fit_cumulants


@dataclass
class DistributionResult:
    """A decay-rate / size distribution from NNLS or CONTIN."""
    method: str                       # 'nnls' or 'contin'
    alpha: Optional[float]            # regularisation parameter (None for NNLS)
    # the grid (paired: rh_grid_nm[i] corresponds to gamma_grid_s_inv[i])
    rh_grid_nm: np.ndarray
    gamma_grid_s_inv: np.ndarray
    weights: np.ndarray               # normalised intensity weights (sum = 1)
    # summary statistics of the distribution
    mean_rh_nm: float                 # intensity-weighted mean Rh
    mean_gamma_s_inv: float           # intensity-weighted mean decay rate
    peak_rh_nm: float                 # Rh at the largest weight
    # quality / context
    beta: float
    beta_estimated: bool
    baseline: float
    baseline_estimated: bool
    q_m_inv: float
    fit_tau_s: np.ndarray             # delay times used (after windowing)
    fitted_g2m1: np.ndarray           # beta * (A x) reconstructed on fit_tau_s
    residuals: np.ndarray             # (g2-1) data - fitted, on fit_tau_s
    rms_error: float
    residual_norm: float              # ||A x - y||^2 (normalised-space residual)
    solution_norm: float              # ||x||^2
    n_skipped: int = 0                # leading channels dropped (skip_initial_channels)


@dataclass
class LCurveResult:
    """The alpha sweep used to choose CONTIN's regularisation parameter."""
    alphas: np.ndarray
    residual_norms: np.ndarray        # ||A x - y||^2 for each alpha
    solution_norms: np.ndarray        # ||x||^2 for each alpha
    optimal_alpha: float
    optimal_index: int


@dataclass
class ContinResult:
    """CONTIN result: the chosen distribution plus the L-curve it came from."""
    distribution: DistributionResult
    lcurve: LCurveResult
    alpha_was_user_supplied: bool


# ---------------------------------------------------------------------------
# Solver internals
# ---------------------------------------------------------------------------

def _second_difference_operator(n: int) -> np.ndarray:
    """The (n-2) x n second-difference matrix L (rows [.. 1 -2 1 ..])."""
    L = np.zeros((max(n - 2, 0), n))
    for i in range(n - 2):
        L[i, i] = 1.0
        L[i, i + 1] = -2.0
        L[i, i + 2] = 1.0
    return L


def _solve_distribution(
    A: np.ndarray,
    y: np.ndarray,
    alpha: float,
    L: Optional[np.ndarray],
) -> np.ndarray:
    """Solve a non-negative (optionally regularised) least-squares problem.

    Minimises ||A x - y||^2 + alpha^2 ||L x||^2 subject to x >= 0 by stacking
    [A; alpha L] over [y; 0] and calling scipy.optimize.nnls. For NNLS, pass
    alpha = 0 (or L = None); the augmentation then vanishes.
    """
    if alpha > 0 and L is not None and L.shape[0] > 0:
        A_aug = np.vstack([A, alpha * L])
        y_aug = np.concatenate([y, np.zeros(L.shape[0])])
    else:
        A_aug, y_aug = A, y
    x, _ = optimize.nnls(A_aug, y_aug)
    return x


def _distribution_summary(
    x: np.ndarray,
    A: np.ndarray,
    y: np.ndarray,
    rh_grid_nm: np.ndarray,
    gamma_grid: np.ndarray,
    beta: float,
    baseline: float,
    g2m1_data: np.ndarray,
    tau: np.ndarray,
    q_m_inv: float,
    method: str,
    alpha: Optional[float],
    beta_estimated: bool,
    baseline_estimated: bool,
    n_skipped: int = 0,
) -> DistributionResult:
    """Assemble a DistributionResult from a raw solver solution x."""
    total = x.sum()
    weights = x / total if total > 0 else x
    # intensity-weighted means over the distribution
    mean_gamma = float(np.sum(weights * gamma_grid))
    mean_rh = float(np.sum(weights * rh_grid_nm))
    peak_rh = float(rh_grid_nm[int(np.argmax(weights))]) if weights.size else float('nan')
    # reconstruct g2-1 in measured space: beta*(A x) + baseline
    fitted_norm = A @ x
    fitted_g2m1 = beta * fitted_norm + baseline
    residuals = g2m1_data - fitted_g2m1
    residual_norm = float(np.sum((A @ x - y) ** 2))
    solution_norm = float(np.sum(x ** 2))
    return DistributionResult(
        method=method, alpha=alpha,
        rh_grid_nm=rh_grid_nm, gamma_grid_s_inv=gamma_grid, weights=weights,
        mean_rh_nm=mean_rh, mean_gamma_s_inv=mean_gamma, peak_rh_nm=peak_rh,
        beta=beta, beta_estimated=beta_estimated,
        baseline=baseline, baseline_estimated=baseline_estimated,
        q_m_inv=q_m_inv, fit_tau_s=tau, fitted_g2m1=fitted_g2m1,
        residuals=residuals, rms_error=_rms_error(residuals),
        residual_norm=residual_norm, solution_norm=solution_norm,
        n_skipped=int(n_skipped or 0),
    )


def _prepare_distribution_inputs(
    measurement, tau_min_s, tau_max_s, beta, baseline,
    rh_min_nm, rh_max_nm, n_grid, skip_initial_channels=0,
):
    """Shared setup for NNLS/CONTIN: window, baseline, beta, grid, A, y."""
    tau, g2m1 = _apply_tau_window(
        measurement.delay_times_s, measurement.correlogram,
        tau_min_s, tau_max_s, min_points=4,
        skip_initial_channels=skip_initial_channels,
    )
    baseline_estimated = baseline is None
    if baseline is None:
        baseline = _estimate_baseline(tau, g2m1)
    beta_estimated = beta is None
    if beta is None:
        beta = _estimate_beta(tau, g2m1, baseline)
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta!r}.")

    q = _measurement_q_m(measurement)
    eta = _require_viscosity(measurement)   # distributions are reported in Rh
    rh_grid, gamma_grid = _build_rh_gamma_grid(
        rh_min_nm, rh_max_nm, n_grid, q, measurement.temperature_K, eta,
    )
    # transfer matrix A[m,n] = exp(-2 Gamma_n tau_m); data y = (g2-1-baseline)/beta
    A = np.exp(-2.0 * np.outer(tau, gamma_grid))
    y = (g2m1 - baseline) / beta
    return (tau, g2m1, baseline, baseline_estimated, beta, beta_estimated,
            q, rh_grid, gamma_grid, A, y)


# ---------------------------------------------------------------------------
# NNLS
# ---------------------------------------------------------------------------

def fit_nnls(
    measurement: DLSMeasurement,
    rh_min_nm: float = 1.0,
    rh_max_nm: float = 1000.0,
    n_grid: int = 100,
    beta: Optional[float] = None,
    baseline: Optional[float] = None,
    tau_min_s: Optional[float] = None,
    tau_max_s: Optional[float] = None,
    skip_initial_channels: int = 0,
) -> DistributionResult:
    """Recover a decay-rate distribution by non-negative least squares.

    Solves min ||A x - y||^2 subject to x >= 0, with no smoothing. NNLS is the
    unregularised special case (alpha = 0) of CONTIN. It can resolve clearly
    separated modes but, lacking a smoothness constraint, tends to produce spiky
    distributions that are sensitive to noise; CONTIN is usually preferable for
    real data.

    Parameters
    ----------
    measurement : DLSMeasurement
    rh_min_nm, rh_max_nm : float
        Hydrodynamic-radius grid limits (nm). Default 1 to 1000 nm, log-spaced.
    n_grid : int
        Number of grid points (default 100).
    beta : float, optional
        Coherence factor for normalisation. Estimated from the data if omitted.
    baseline : float, optional
        Residual baseline offset to subtract. Estimated from the long-delay tail
        if omitted.
    tau_min_s, tau_max_s : float, optional
        Inclusive delay-time window. Default uses all points.

    Returns
    -------
    DistributionResult
    """
    (tau, g2m1, baseline, baseline_est, beta, beta_est,
     q, rh_grid, gamma_grid, A, y) = _prepare_distribution_inputs(
        measurement, tau_min_s, tau_max_s, beta, baseline,
        rh_min_nm, rh_max_nm, n_grid,
        skip_initial_channels=skip_initial_channels)

    x = _solve_distribution(A, y, alpha=0.0, L=None)
    return _distribution_summary(
        x, A, y, rh_grid, gamma_grid, beta, baseline, g2m1, tau, q,
        method='nnls', alpha=None,
        beta_estimated=beta_est, baseline_estimated=baseline_est,
        n_skipped=skip_initial_channels)


# ---------------------------------------------------------------------------
# Lognormal (single-mode parametric distribution)
# ---------------------------------------------------------------------------

def fit_lognormal(
    measurement: DLSMeasurement,
    rh_min_nm: float = 1.0,
    rh_max_nm: float = 1000.0,
    n_grid: int = 100,
    beta: Optional[float] = None,
    baseline: Optional[float] = None,
    tau_min_s: Optional[float] = None,
    tau_max_s: Optional[float] = None,
    skip_initial_channels: int = 0,
) -> DistributionResult:
    """Fit a single-mode LOGNORMAL Rh distribution to g2-1.

    A parametric distribution method: it assumes the intensity-weighted size
    distribution is lognormal in Rh and fits its median Rh and log-width, using
    the SAME forward model as NNLS/CONTIN (A[m,n] = exp(-2 Gamma_n tau_m)), so the
    result drops straight into the distribution plot/summary. On the log-spaced Rh
    grid the discrete lognormal weight is w_i proportional to
    exp(-(ln Rh_i - mu)^2 / 2 sigma^2) (the 1/Rh of the pdf cancels the log-grid
    spacing); mu = ln(median Rh), sigma is the log-width (polydispersity).

    Unlike NNLS/CONTIN it cannot resolve multiple modes, but it is robust to noise
    and always yields a smooth, single-peaked distribution -- a good default when
    the sample is known to be unimodal. Returns a DistributionResult
    (method='lognormal').
    """
    (tau, g2m1, baseline, baseline_est, beta, beta_est,
     q, rh_grid, gamma_grid, A, y) = _prepare_distribution_inputs(
        measurement, tau_min_s, tau_max_s, beta, baseline,
        rh_min_nm, rh_max_nm, n_grid,
        skip_initial_channels=skip_initial_channels)

    ln_rh = np.log(rh_grid)

    def _weights(mu: float, sigma: float) -> np.ndarray:
        w = np.exp(-0.5 * ((ln_rh - mu) / sigma) ** 2)
        s = w.sum()
        return w / s if s > 0 else w

    def model(_x, amp, mu, sigma):
        return amp * (A @ _weights(mu, sigma))

    # Seed the median from a quick cumulant Rh when it falls on the grid.
    mu0 = 0.5 * (ln_rh[0] + ln_rh[-1])
    try:
        cum = fit_cumulants(measurement, order=2,
                            tau_min_s=tau_min_s, tau_max_s=tau_max_s,
                            skip_initial_channels=skip_initial_channels)
        if np.isfinite(cum.rh_nm) and rh_grid[0] <= cum.rh_nm <= rh_grid[-1]:
            mu0 = math.log(cum.rh_nm)
    except Exception:
        pass

    p0 = [1.0, mu0, 0.3]
    bounds = ([1e-6, ln_rh[0], 1e-2], [np.inf, ln_rh[-1], 3.0])
    try:
        popt, _ = optimize.curve_fit(
            model, np.arange(tau.size), y, p0=p0, bounds=bounds, maxfev=10000)
        amp, mu, sigma = float(popt[0]), float(popt[1]), float(popt[2])
    except (RuntimeError, ValueError):
        amp, mu, sigma = 1.0, mu0, 0.3

    x = amp * _weights(mu, sigma)
    return _distribution_summary(
        x, A, y, rh_grid, gamma_grid, beta, baseline, g2m1, tau, q,
        method='lognormal', alpha=None,
        beta_estimated=beta_est, baseline_estimated=baseline_est,
        n_skipped=skip_initial_channels)


# ---------------------------------------------------------------------------
# CONTIN with L-curve alpha selection
# ---------------------------------------------------------------------------

def _lcurve_corner(alphas, residual_norms, solution_norms) -> int:
    """Pick the L-curve corner index (Salazar et al. 2023 GUI method).

    Each axis (residual norm, solution norm) is taken to log10, then linearly
    rescaled to [-10, 10]. The chosen alpha is the one whose (rescaled-log
    residual, rescaled-log solution) point is closest to the origin of that
    scaled square -- the elbow that balances fit quality against distribution
    complexity. Robust to the absolute scales of the two norms.
    """
    res = np.asarray(residual_norms, dtype=float)
    sol = np.asarray(solution_norms, dtype=float)
    # guard against non-positive values before log
    res = np.clip(res, 1e-300, None)
    sol = np.clip(sol, 1e-300, None)
    res_log = np.log10(res)
    sol_log = np.log10(sol)

    def rescale(v):
        vmin, vmax = v.min(), v.max()
        if vmax == vmin:
            return np.zeros_like(v)
        return (20.0 / (vmax - vmin)) * (v - 0.5 * (vmin + vmax))  # -> [-10, 10]

    res_s = rescale(res_log)
    sol_s = rescale(sol_log)
    dist = np.sqrt(res_s ** 2 + sol_s ** 2)
    return int(np.argmin(dist))


def fit_contin(
    measurement: DLSMeasurement,
    rh_min_nm: float = 1.0,
    rh_max_nm: float = 1000.0,
    n_grid: int = 100,
    alpha: Optional[float] = None,
    alpha_min: float = 1e-6,
    alpha_max: float = 1e2,
    n_alpha: int = 20,
    beta: Optional[float] = None,
    baseline: Optional[float] = None,
    tau_min_s: Optional[float] = None,
    tau_max_s: Optional[float] = None,
    skip_initial_channels: int = 0,
) -> ContinResult:
    """Recover a smoothed decay-rate distribution by regularised inversion.

    Solves min ||A x - y||^2 + alpha^2 ||L x||^2 subject to x >= 0, with L the
    second-difference operator (Provencher 1982). The regularisation parameter
    alpha trades fit quality against distribution smoothness.

    If alpha is None (default), an L-curve sweep over [alpha_min, alpha_max] is
    run and the corner is chosen automatically; the full sweep is returned so the
    choice can be inspected and overridden. If alpha is given, that value is used
    directly and a single-point "L-curve" is returned for consistency.

    Parameters
    ----------
    measurement : DLSMeasurement
    rh_min_nm, rh_max_nm, n_grid : grid specification (default 1-1000 nm, 100 pts)
    alpha : float, optional
        Fixed regularisation parameter. If omitted, chosen by the L-curve.
    alpha_min, alpha_max, n_alpha :
        Log-spaced alpha sweep for the L-curve (default 1e-6 to 1e2, 20 points).
    beta, baseline : float, optional
        Coherence factor and baseline; estimated from the data if omitted.
    tau_min_s, tau_max_s : float, optional
        Inclusive delay-time window. Default uses all points.

    Returns
    -------
    ContinResult
        .distribution is the chosen DistributionResult; .lcurve holds the sweep.
    """
    (tau, g2m1, baseline, baseline_est, beta, beta_est,
     q, rh_grid, gamma_grid, A, y) = _prepare_distribution_inputs(
        measurement, tau_min_s, tau_max_s, beta, baseline,
        rh_min_nm, rh_max_nm, n_grid,
        skip_initial_channels=skip_initial_channels)

    L = _second_difference_operator(n_grid)

    if alpha is not None:
        # User-fixed alpha: solve once.
        x = _solve_distribution(A, y, alpha=alpha, L=L)
        dist = _distribution_summary(
            x, A, y, rh_grid, gamma_grid, beta, baseline, g2m1, tau, q,
            method='contin', alpha=alpha,
            beta_estimated=beta_est, baseline_estimated=baseline_est,
            n_skipped=skip_initial_channels)
        lcurve = LCurveResult(
            alphas=np.array([alpha]),
            residual_norms=np.array([dist.residual_norm]),
            solution_norms=np.array([dist.solution_norm]),
            optimal_alpha=alpha, optimal_index=0)
        return ContinResult(distribution=dist, lcurve=lcurve,
                            alpha_was_user_supplied=True)

    # L-curve sweep
    alphas = np.geomspace(alpha_min, alpha_max, n_alpha)
    residual_norms = np.empty(n_alpha)
    solution_norms = np.empty(n_alpha)
    solutions = []
    for i, a in enumerate(alphas):
        x = _solve_distribution(A, y, alpha=a, L=L)
        solutions.append(x)
        residual_norms[i] = np.sum((A @ x - y) ** 2)
        solution_norms[i] = np.sum(x ** 2)

    opt_idx = _lcurve_corner(alphas, residual_norms, solution_norms)
    opt_alpha = float(alphas[opt_idx])
    x_opt = solutions[opt_idx]

    dist = _distribution_summary(
        x_opt, A, y, rh_grid, gamma_grid, beta, baseline, g2m1, tau, q,
        method='contin', alpha=opt_alpha,
        beta_estimated=beta_est, baseline_estimated=baseline_est,
        n_skipped=skip_initial_channels)
    lcurve = LCurveResult(
        alphas=alphas, residual_norms=residual_norms,
        solution_norms=solution_norms, optimal_alpha=opt_alpha,
        optimal_index=opt_idx)
    return ContinResult(distribution=dist, lcurve=lcurve,
                        alpha_was_user_supplied=False)


# ---------------------------------------------------------------------------
# Rh <-> Gamma axis helper (for the visualisation toggle and general use)
# ---------------------------------------------------------------------------

def distribution_axis(distribution: DistributionResult, axis: str = 'rh'):
    """Return (x_values, weights, x_label) for plotting a DistributionResult.

    Supports the Rh <-> Gamma visualisation toggle. The same weights are plotted
    against either the hydrodynamic-radius grid or the decay-rate grid; only the
    x-axis changes. Values are returned sorted ascending in the chosen axis.

    Parameters
    ----------
    distribution : DistributionResult
    axis : str
        'rh' (default) -> x is hydrodynamic radius (nm).
        'gamma'        -> x is decay rate (s^-1).

    Returns
    -------
    (x_values, weights, x_label)
    """
    if axis == 'rh':
        x = distribution.rh_grid_nm
        label = 'Hydrodynamic radius Rh (nm)'
    elif axis == 'gamma':
        x = distribution.gamma_grid_s_inv
        label = 'Decay rate Gamma (s^-1)'
    else:
        raise ValueError(f"axis must be 'rh' or 'gamma', got {axis!r}.")
    order = np.argsort(x)
    return x[order], distribution.weights[order], label


# ---------------------------------------------------------------------------
# Distribution peak detection (for multi-population samples)
# ---------------------------------------------------------------------------

@dataclass
class DistributionPeak:
    """One resolved population in a CONTIN/NNLS size distribution.

    `weight_fraction` is the fraction of the total (intensity-weighted) area under
    that peak's lobe -- so the peaks of a distribution sum to ~1. It is an
    intensity fraction, not a number or mass fraction (large particles scatter far
    more). `is_dominant` marks the largest-area peak.
    """
    rh_nm: float                      # Rh at the peak maximum
    gamma_s_inv: float                # decay rate at the peak maximum
    weight_fraction: float            # intensity-weighted area of the peak's lobe
    is_dominant: bool


def find_distribution_peaks(
    distribution: DistributionResult,
    min_weight_fraction: float = 0.05,
) -> List[DistributionPeak]:
    """Resolve the distinct populations (peaks) in a size distribution.

    Local maxima of the (intensity-weighted) distribution are found on the Rh grid;
    each peak's lobe runs to the local minima on either side, and the lobe's summed
    weight is its intensity fraction. Peaks contributing less than
    `min_weight_fraction` of the total are dropped as noise. The returned peaks are
    sorted by Rh ascending; exactly one is flagged `is_dominant` (the largest area).

    A monomodal distribution returns a single peak. This is the basis for offering a
    CONTIN/NNLS peak Rh as a rho source and for letting the user pick a peak when a
    sample is multi-population.
    """
    rh = np.asarray(distribution.rh_grid_nm, dtype=float)
    w = np.asarray(distribution.weights, dtype=float)
    gamma = np.asarray(distribution.gamma_grid_s_inv, dtype=float)
    if rh.size == 0:
        return []
    order = np.argsort(rh)
    rh, w, gamma = rh[order], w[order], gamma[order]
    total = float(w.sum())
    if not (total > 0):
        return []

    n = w.size
    # local maxima, including the endpoints if they rise above their neighbour
    maxima: List[int] = []
    for i in range(n):
        left_ok = (i == 0) or (w[i] >= w[i - 1])
        right_ok = (i == n - 1) or (w[i] >= w[i + 1])
        strictly = ((i > 0 and w[i] > w[i - 1]) or (i < n - 1 and w[i] > w[i + 1]))
        if left_ok and right_ok and strictly and w[i] > 0:
            maxima.append(i)
    if not maxima:
        maxima = [int(np.argmax(w))]

    # lobe boundaries: the local minimum between each pair of adjacent maxima
    bounds = [0]
    for a, b in zip(maxima[:-1], maxima[1:], strict=True):
        bounds.append(a + int(np.argmin(w[a:b + 1])))
    bounds.append(n)

    peaks: List[DistributionPeak] = []
    for k, peak_i in enumerate(maxima):
        lo, hi = bounds[k], bounds[k + 1]
        frac = float(w[lo:hi].sum() / total)
        if frac < min_weight_fraction:
            continue
        peaks.append(DistributionPeak(
            rh_nm=float(rh[peak_i]), gamma_s_inv=float(gamma[peak_i]),
            weight_fraction=frac, is_dominant=False))
    if not peaks:
        return []
    dominant = max(range(len(peaks)), key=lambda j: peaks[j].weight_fraction)
    peaks[dominant].is_dominant = True
    return peaks
