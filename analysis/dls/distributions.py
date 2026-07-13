"""
analysis/dls/distributions.py
=============================

Distribution methods: recover a full decay-rate / size distribution rather than a
few parameters.

We invert the FIELD ACF g1 (not |g1|^2) on a grid of decay rates Gamma_n
(equivalently hydrodynamic radii Rh_n):

    g1(tau_m)  ~  sum_n  x_n exp(-Gamma_n tau_m)             (A x)_m

with a single-Gamma kernel A[m, n] = exp(-Gamma_n tau_m). This is the accepted-
standard DLS distribution inversion (Berne & Pecora 2000 Sec. 8.11 + App. 4.C):
g1 is recovered from the measured correlogram through the Siegert relation
g2 - 1 = beta |g1|^2 (Chu 1991), so the fit lives in g1-space and keeps the full
cross-term structure of a multimodal g1. (The older diagonal form
|g1|^2 ~ sum_n x_n exp(-2 Gamma_n tau_m) is the "cross-terms-negligible"
approximation, Liénard et al. 2022 Eq. 11 -- exact only for a single narrow mode;
applied to a multimodal sample it fabricates a phantom peak at the mean decay
rate.) We solve for the non-negative weights
x_n, which are the intensity-weighted contributions (g1 amplitudes normalize with
the scattered intensity).

  NNLS      : min ||M^^.5 (A x - y)||^2          s.t. x >= 0     (no smoothing)
  CONTIN    : min ||M^^.5 (A x - y)||^2 + alpha^2 ||L x||^2   s.t. x >= 0
              with L the second-difference operator (Provencher's regularizor).
  Lognormal : a single-mode parametric distribution fit through the same A.

Statistically-weighted residuals (the M above). Recovering g1 = sqrt((g2-1)/beta)
rectifies zero-mean long-lag noise into a positive pedestal that an UNWEIGHTED
non-negative fit would absorb as spurious large-Rh weight. We therefore weight the
residual by the delta-method inverse-variance of the recovered g1: propagating a
(locally uniform) g2-1 noise through the sqrt gives Var(g1) ~ Var(g2-1)/(4 beta^2
g1^2), so the per-lag weight is w(tau) proportional to g1(tau)^2 -- long-lag /
clipped channels get ~0 weight, which suppresses the pedestal AND keeps the
Provencher F-test residual (Provencher 1982a Eq. 3.9 is the weighted V) honest.
The unknown noise scale cancels in the argmin and in the F-test's (V-V0)/V0 ratio,
so the RELATIVE weight is parameter-free (no new inputs, no per-run tuning). The
weight is a precision map that improves the FIT only; no statistical +/- is ever
reported from a single correlogram (invariant 8) -- Schätzel 1990 (the correlated,
non-uniform correlator noise this approximates) is exactly why a single-shot ± is
still deferred to replicate averaging.

Solver note (a deliberate departure from the Salazar et al. 2023 GUI): we do NOT
use SLSQP with a sum(x)=1 equality constraint. Instead we use the standard
augmented-NNLS formulation -- stack [M^.5 A; alpha L] over [M^.5 y; 0] and call
scipy.optimize.nnls -- which is faster, more robust, and avoids forcing the
distribution to integrate to exactly 1 when the data do not support it. The
reported distribution is normalized to sum 1 afterwards, for display only.

The data y fed to the solver is the recovered field ACF, y = sign(u) sqrt(|u|) with
u = (g2 - 1 - baseline) / beta, so that y(0) ~ 1 and the recovered weights are an
intensity-weighted distribution. The signed (not clipped) square root keeps the
long-lag noise zero-mean so no positive g1 pedestal is rectified into the fit. beta
and baseline are estimated by default (see _estimate_beta / _estimate_baseline in
_common) but may be overridden.

This module also owns the distribution post-processing shared by the GUI: the
Rh<->Gamma axis toggle (distribution_axis) and peak detection
(find_distribution_peaks).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy import optimize, special
from scipy.interpolate import CubicSpline

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
    alpha: Optional[float]            # regularization parameter (None for NNLS)
    # the grid (paired: rh_grid_nm[i] corresponds to gamma_grid_s_inv[i])
    rh_grid_nm: np.ndarray
    gamma_grid_s_inv: np.ndarray
    weights: np.ndarray               # normalized intensity weights (sum = 1)
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
    fitted_g2m1: np.ndarray           # beta * (A x)^2 + baseline reconstructed on fit_tau_s
    residuals: np.ndarray             # (g2-1) data - fitted, on fit_tau_s
    rms_error: float
    residual_norm: float              # weighted ||M^.5 (A x - y)||^2 (g1-space residual)
    solution_norm: float              # ||x||^2
    n_skipped: int = 0                # leading channels dropped (skip_initial_channels)


@dataclass
class LCurveResult:
    """The alpha sweep used to choose CONTIN's regularization parameter.

    Despite the name (kept for backward compatibility), this holds the sweep for
    ANY of the three alpha-selection methods — GCV, the L-curve corner, or the
    Provencher F-test. `dof_eff` is populated for GCV and the F-test (both need the
    Tikhonov hat-trace); `ftest_fc` only for the F-test; `gcv` only for GCV.
    """
    alphas: np.ndarray
    residual_norms: np.ndarray        # weighted ||M^.5 (A x - y)||^2 for each alpha
    solution_norms: np.ndarray        # ||L x||^2 (regularization seminorm = L-curve axis)
    optimal_alpha: float
    optimal_index: int
    dof_eff: Optional[np.ndarray] = None    # Tikhonov effective DOF per alpha (GCV + F-test)
    ftest_fc: Optional[np.ndarray] = None   # cumulative F ("probability to reject") per alpha
    gcv: Optional[np.ndarray] = None        # GCV score V(alpha) per alpha (minimized)


@dataclass
class ContinResult:
    """CONTIN result: the chosen distribution plus the sweep it came from.

    `alpha_selection_method` records how the automatic alpha was chosen
    ('gcv' | 'lcurve' | 'ftest'), and `ftest_prob_reject` the F-test level when
    applicable, so a distribution is never ambiguous about how its regularization
    was picked. When alpha was user-supplied, the method is reported as 'user'.

    `alpha_at_ceiling` flags that the selected alpha landed on the HIGH-alpha end of
    the sweep. For GCV this is the documented flat-minimum failure mode (Hansen 1998
    p.185) and signals possible over-regularization (widen the range or inspect the
    L-curve); a low-alpha (floor) pick is legitimate on clean data and is NOT flagged.
    """
    distribution: DistributionResult
    lcurve: LCurveResult
    alpha_was_user_supplied: bool
    alpha_selection_method: str = 'lcurve'    # 'gcv' | 'lcurve' | 'ftest' | 'user'
    ftest_prob_reject: Optional[float] = None
    alpha_at_ceiling: bool = False


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


def _apply_weight(A: np.ndarray, y: np.ndarray,
                  w: Optional[np.ndarray]) -> tuple:
    """Row-scale (A, y) by sqrt(w) so an ordinary LS solve minimizes the WEIGHTED
    residual sum(w (A x - y)^2). w = None leaves the system unweighted (M = I)."""
    if w is None:
        return A, y
    sw = np.sqrt(w)
    return A * sw[:, None], y * sw


def _solve_distribution(
    A: np.ndarray,
    y: np.ndarray,
    w: Optional[np.ndarray],
    alpha: float,
    L: Optional[np.ndarray],
) -> np.ndarray:
    """Solve a non-negative, statistically-weighted, optionally-regularized problem.

    Minimizes ||M^.5 (A x - y)||^2 + alpha^2 ||L x||^2 subject to x >= 0, where M is
    the diagonal per-lag weight w (delta-method inverse-variance of the recovered
    g1). The data rows are pre-scaled by sqrt(w) and the (unweighted) regularizer is
    stacked below: [sqrt(w) A; alpha L] over [sqrt(w) y; 0], solved by
    scipy.optimize.nnls. For NNLS pass alpha = 0 (or L = None); the augmentation then
    vanishes. Pass w = None for an unweighted solve.
    """
    Aw, yw = _apply_weight(A, y, w)
    if alpha > 0 and L is not None and L.shape[0] > 0:
        A_aug = np.vstack([Aw, alpha * L])
        y_aug = np.concatenate([yw, np.zeros(L.shape[0])])
    else:
        A_aug, y_aug = Aw, yw
    x, _ = optimize.nnls(A_aug, y_aug)
    return x


def _distribution_summary(
    x: np.ndarray,
    A: np.ndarray,
    y: np.ndarray,
    w: Optional[np.ndarray],
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
    # A x is the recovered field ACF g1; map back to measured space via the Siegert
    # relation g2 - 1 = beta |g1|^2 (+ residual baseline).
    g1_fit = A @ x
    fitted_g2m1 = beta * g1_fit ** 2 + baseline
    residuals = g2m1_data - fitted_g2m1
    # the WEIGHTED g1-space residual the F-test / L-curve read (M = w, or I if None)
    resid_g1 = g1_fit - y
    wv = resid_g1 ** 2 if w is None else w * resid_g1 ** 2
    residual_norm = float(np.sum(wv))
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
    # g1-space inversion (Berne & Pecora 2000): single-Gamma kernel
    # A[m,n] = exp(-Gamma_n tau_m); recover the field ACF from the Siegert relation
    # g2 - 1 = beta |g1|^2, i.e. g1 = sqrt((g2-1-baseline)/beta).
    A = np.exp(-np.outer(tau, gamma_grid))
    u = (g2m1 - baseline) / beta                       # the measured |g1|^2 estimate
    # SIGN-PRESERVING recovery: in the long-lag noise floor u is zero-mean and dips
    # negative; a hard clip-at-0 would keep only the positive excursions and rectify
    # that noise into a POSITIVE g1 pedestal, which a non-negative fit absorbs as
    # spurious large-Rh weight (the noisy-unimodal mean-Rh regression, S158). Taking
    # the signed square root keeps the noise zero-mean, so no pedestal forms; the
    # fitted model A x (a true non-negative g1) still only chases the real signal.
    y = np.sign(u) * np.sqrt(np.abs(u))
    # delta-method inverse-variance weight of the recovered g1: propagating a
    # (locally uniform) g2-1 noise through the sqrt gives Var(g1) prop 1/|u|, so
    # w prop |u| = y^2. The overall scale cancels in the argmin / F-test ratio, so the
    # relative weight is parameter-free; noise-floor channels (|u| -> 0) get ~0 weight.
    # (Note sqrt(w) y = u exactly, so the weighted fit targets the same unbiased
    # (g2-1)/beta as the legacy diagonal kernel, but through a cross-term-preserving
    # g1 design -- noise-robust AND phantom-free.)
    w = y ** 2
    return (tau, g2m1, baseline, baseline_estimated, beta, beta_estimated,
            q, rh_grid, gamma_grid, A, y, w)


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
    unregularized special case (alpha = 0) of CONTIN. It can resolve clearly
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
        Coherence factor for normalization. Estimated from the data if omitted.
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
     q, rh_grid, gamma_grid, A, y, w) = _prepare_distribution_inputs(
        measurement, tau_min_s, tau_max_s, beta, baseline,
        rh_min_nm, rh_max_nm, n_grid,
        skip_initial_channels=skip_initial_channels)

    x = _solve_distribution(A, y, w, alpha=0.0, L=None)
    return _distribution_summary(
        x, A, y, w, rh_grid, gamma_grid, beta, baseline, g2m1, tau, q,
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
    the SAME forward model as NNLS/CONTIN (the g1-space kernel A[m,n] =
    exp(-Gamma_n tau_m)), so the result drops straight into the distribution
    plot/summary. On the log-spaced Rh
    grid the discrete lognormal weight is w_i proportional to
    exp(-(ln Rh_i - mu)^2 / 2 sigma^2) (the 1/Rh of the pdf cancels the log-grid
    spacing); mu = ln(median Rh), sigma is the log-width (polydispersity).

    Unlike NNLS/CONTIN it cannot resolve multiple modes, but it is robust to noise
    and always yields a smooth, single-peaked distribution -- a good default when
    the sample is known to be unimodal. Returns a DistributionResult
    (method='lognormal').
    """
    (tau, g2m1, baseline, baseline_est, beta, beta_est,
     q, rh_grid, gamma_grid, A, y, w) = _prepare_distribution_inputs(
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
    # weight the fit by the same delta-method w: curve_fit minimizes
    # sum(((model - y)/sigma)^2), so sigma = 1/sqrt(w) reproduces sum(w (model-y)^2).
    # Clip so clipped-to-zero channels get a large (not infinite) sigma, i.e. ~0 weight.
    sigma_w = 1.0 / np.sqrt(np.clip(w, 1e-12, None))
    try:
        popt, _ = optimize.curve_fit(
            model, np.arange(tau.size), y, p0=p0, bounds=bounds, maxfev=10000,
            sigma=sigma_w, absolute_sigma=False)
        amp, mu, sigma = float(popt[0]), float(popt[1]), float(popt[2])
    except (RuntimeError, ValueError):
        amp, mu, sigma = 1.0, mu0, 0.3

    x = amp * _weights(mu, sigma)
    return _distribution_summary(
        x, A, y, w, rh_grid, gamma_grid, beta, baseline, g2m1, tau, q,
        method='lognormal', alpha=None,
        beta_estimated=beta_est, baseline_estimated=baseline_est,
        n_skipped=skip_initial_channels)


# ---------------------------------------------------------------------------
# CONTIN with L-curve alpha selection
# ---------------------------------------------------------------------------

def _regularization_seminorm_sq(L: Optional[np.ndarray], x: np.ndarray) -> float:
    """||L x||^2 -- the Tikhonov regularization seminorm, the L-curve's solution axis.

    Hansen's L-curve for GENERAL-FORM Tikhonov (regularizer alpha^2 ||L x||^2, here L
    the second-difference operator) plots log||A x - y|| against log||L x|| -- the
    seminorm of the term actually being penalised, NOT the standard-form ||x||. Using
    ||x|| would compute the corner on the wrong axis (Hansen 1998 Eq. 7.24 is stated for
    the general-form curve). Falls back to ||x||^2 only if L is degenerate (no rows).
    """
    if L is not None and L.shape[0] > 0:
        return float(np.sum((L @ x) ** 2))
    return float(np.sum(x ** 2))


def _lcurve_corner(alphas, residual_norms, solution_norms) -> int:
    """Pick the L-curve corner: the point of MAXIMUM CURVATURE of the log-log curve.

    This is Hansen's canonical L-curve criterion (Hansen 1998 Eq. 7.24), located by
    the Hansen & O'Leary (1993) algorithm -- fit a cubic spline through the discrete
    L-curve points, take the spline's analytic curvature, and pick the global maximum.
    The curve is plotted in log-log coordinates (mandatory, Hansen 1998 p.188):

        zeta = log||A x - y||,   eta = log||L x||,
        kappa = (zeta' eta'' - zeta'' eta') / (zeta'^2 + eta'^2)^(3/2)         (7.24)

    parameterized by t = log10(alpha) (monotone: the L-curve is traversed
    monotonically as alpha grows). `residual_norms`/`solution_norms` are stored as
    SQUARED norms, so zeta/eta take 0.5*log10 to recover the log of the norm itself.
    The dense-grid argmax is snapped back to the nearest SOLVED sweep alpha (we only
    have distributions at the sweep points).

    (This REPLACES the earlier Salazar et al. 2023 box-rescale heuristic -- a
    published but non-canonical corner. A deliberately hand-rolled finite-difference
    argmax is avoided: it is fragile on the staircase L-curve, which is exactly the
    empirical robustification the spline algorithm exists to replace.)
    """
    a = np.asarray(alphas, dtype=float)
    res = np.clip(np.asarray(residual_norms, dtype=float), 1e-300, None)
    sol = np.clip(np.asarray(solution_norms, dtype=float), 1e-300, None)
    # log of the NORM (not the squared norm the sweep stores) -> factor 0.5.
    zeta = 0.5 * np.log10(res)
    eta = 0.5 * np.log10(sol)
    t = np.log10(a)
    # A cubic spline needs >= 4 STRICTLY INCREASING abscissae. A single- or few-point
    # sweep, or equal/near-duplicate bounds (alpha_min == alpha_max makes t constant),
    # would make CubicSpline raise -- fall back to the discrete curvature instead of
    # crashing the whole CONTIN fit.
    if a.size < 4 or not np.all(np.diff(t) > 0):
        return _discrete_curvature_argmax(zeta, eta)
    # NATURAL boundary conditions (second derivative = 0 at the sweep ends): the L-curve
    # corner is an INTERIOR feature, and a free (not-a-knot) spline overshoots at the
    # endpoints -- its second derivative blows up there, injecting spurious curvature
    # that grows toward the boundary and would pin the corner to the sweep ceiling
    # (an artifact, not a real corner). Natural BCs force kappa -> 0 at the ends,
    # suppressing that overshoot while leaving the genuine interior bend untouched.
    spline_z = CubicSpline(t, zeta, bc_type='natural')
    spline_e = CubicSpline(t, eta, bc_type='natural')
    td = np.linspace(t[0], t[-1], 512)
    zp, zpp = spline_z(td, 1), spline_z(td, 2)
    ep, epp = spline_e(td, 1), spline_e(td, 2)
    grad2 = zp ** 2 + ep ** 2                 # squared tangent magnitude
    gmax = float(grad2.max())
    if gmax <= 0.0:
        # a perfectly flat L-curve (every solve returned the same norms) carries no
        # corner information; pick the least-regularized end rather than divide by zero.
        return 0
    # Guard the L-curve's flat plateaus. For our ill-conditioned kernel the solution
    # has already converged at small alpha (residual and solution norm both plateau),
    # so the curve is stationary there and Eq. (7.24) is a 0/0 singularity -- an
    # interpolating-spline artifact that spikes to spurious huge curvature, NEVER a
    # real corner (a corner requires the curve to actually be turning). Restrict the
    # global max to points whose tangent magnitude is a non-negligible fraction of the
    # curve's own maximum -- a scale-free numerical guard, not a tuned threshold --
    # and compute curvature ONLY there, so the flat-region 0/0 division never runs.
    valid = grad2 > 1e-3 * gmax
    kappa = np.full(td.size, -np.inf)
    kappa[valid] = ((zp[valid] * epp[valid] - zpp[valid] * ep[valid])
                    / np.power(grad2[valid], 1.5))
    t_star = td[int(np.argmax(kappa))]
    return int(np.argmin(np.abs(t - t_star)))


def _discrete_curvature_argmax(zeta, eta) -> int:
    """Fallback corner for a sweep too short (or too degenerate) for a cubic spline.

    A finite-difference curvature of the discrete (zeta, eta) points -- used ONLY in
    the degenerate short-sweep/equal-bounds guard, never on a normal sweep. Guards the
    same 0/0 singularity (coincident points give a zero tangent) as the spline path.
    """
    if zeta.size < 3:
        return 0
    zp, ep = np.gradient(zeta), np.gradient(eta)
    zpp, epp = np.gradient(zp), np.gradient(ep)
    grad2 = zp ** 2 + ep ** 2
    if not np.any(grad2 > 0.0):
        return 0
    kappa = np.where(grad2 > 0.0,
                     (zp * epp - zpp * ep) / np.power(np.clip(grad2, 1e-300, None), 1.5),
                     -np.inf)
    return int(np.argmax(kappa))


def _gcv_corner(residual_norms, dof_eff, n_data: int):
    """Generalized Cross-Validation alpha selection (Golub, Heath & Wahba 1979).

    Minimizes the GCV functional (their Eq. 1.4), which for our weighted Tikhonov
    fit and with Tr(I - A(alpha)) = n - dof(alpha) reduces to

        GCV(alpha) = n * V(alpha) / (n - dof(alpha))^2                          (1.4)

    where n = n_data is the number of lag channels, V(alpha) the WEIGHTED residual
    ||M^.5 (A x - y)||^2 (`residual_norms`), and dof(alpha) the effective degrees of
    freedom = trace of the Tikhonov hat matrix (`_tikhonov_effective_dof`). The
    n-scaling cancels for the minimizer, so the absolute noise level is irrelevant --
    GCV is sigma^2-free and asymptotically optimal (Wahba). The chosen alpha is the
    argmin. Returns (index, gcv_array). The gcv_array is returned for the result/plot.

    Guarded against a non-positive (n - dof): dof is the Tikhonov hat-trace, a mild
    upper bound on the free-parameter count that for the ill-posed Laplace kernel
    stays well below n, but it is clipped to (0, n) defensively.
    """
    v = np.asarray(residual_norms, dtype=float)
    dof = np.clip(np.asarray(dof_eff, dtype=float), 1e-6, n_data - 1e-6)
    gcv = n_data * v / (n_data - dof) ** 2
    return int(np.argmin(gcv)), gcv


def _tikhonov_effective_dof(A: np.ndarray, L: np.ndarray, alpha: float,
                            AtA: Optional[np.ndarray] = None) -> float:
    """Effective degrees of freedom of the Tikhonov solution at this alpha.

    The regularized least-squares solution has the linear "hat" (influence) matrix
    H = A (A^T A + alpha^2 L^T L)^-1 A^T; its trace is the effective number of free
    parameters -- the honest degrees of freedom, NOT the raw grid size (Hansen 1998;
    the number of grid points is "to a large extent arbitrary", Provencher 1982). It
    decreases smoothly from ~rank(A) toward 0 as alpha grows (more smoothing = fewer
    effective parameters). tr(H) = tr((A^T A + alpha^2 L^T L)^-1 A^T A). For the
    ill-conditioned Laplace-inversion kernel A this stays well below the data count,
    so the F-test degrees of freedom are always well defined. The non-negativity
    constraint can only lower it further, so this is a mild upper bound.
    """
    if AtA is None:
        AtA = A.T @ A
    M = AtA + (alpha ** 2) * (L.T @ L)
    try:
        sol = np.linalg.solve(M, AtA)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(M, AtA, rcond=None)[0]
    return float(np.clip(np.trace(sol), 1e-6, None))


def _ftest_corner(residual_norms, dof_eff, n_data: int,
                  prob_reject: float = 0.5):
    """Provencher's F-test ("probability to reject") alpha selection.

    Implements the original criterion of Provencher (1982a, "A constrained
    regularization method...", Comput. Phys. Commun. 27:213), Eqs. (3.23)-(3.24).
    Among the sweep of increasingly-smoothed solutions, judge how significant each
    solution's residual increase over the least-squares reference is, and pick the one
    at the chosen probability level. For each alpha the F-statistic is

        F(alpha) = [ (V(alpha) - V0) / V0 ] * (Ny - NDF0) / NDF0             (Eq. 3.24)

    with V(alpha) = ||A x - y||^2, and V0 the MINIMUM residual over the sweep, at the
    reference alpha_0 (the least-regularized / least-squares end). Ny is the number of
    data points and NDF0 = NDF(alpha_0) the effective degrees of freedom AT that
    reference -- Provencher's NDF = sum_j s_j^2/(s_j^2 + alpha^2) (Eqs. 3.15-3.16),
    which is exactly the trace of the Tikhonov hat matrix (`_tikhonov_effective_dof`),
    NOT the raw grid size. NDF0 is a single fixed value (taken at alpha_0), used in
    BOTH the scaling factor and the F-distribution -- not the per-alpha DOF. F is
    F-distributed with (NDF0, Ny - NDF0) degrees of freedom, so its cumulative value

        fc(alpha) = P[F(alpha); NDF0, Ny - NDF0]                             (Eq. 3.23)

    is Provencher's PROB1, the "probability to reject": ~0 where the regularizer barely
    changes the fit (rough, under-smoothed) and ~1 where it changes it a lot (Provencher:
    "only when PROB1 > ~0.9 are there significant grounds to suspect alpha may be too
    large"). The chosen solution is the one whose fc is closest to `prob_reject`
    (Provencher's default 0.5). A HIGHER prob_reject tolerates more fit degradation ->
    selects a LARGER alpha (smoother, more parsimonious); a LOWER one a smaller alpha
    (rougher). For an ill-posed Laplace kernel NDF0 is small (Provencher: "surprisingly
    small even with accurate data"), so the F-test dof are always well defined.

    Returns (index, fc_array). fc_array (PROB1 per alpha) is returned for the result/plot.

    Caveat (documented in the guide): the F-test assumes independent residuals, but a
    single correlogram's lag channels are correlated (Schaetzel 1990), so Ny overstates
    the independent information and the level is a guide, not an exact test -- part of
    why the L-curve remains the default.
    """
    v = np.asarray(residual_norms, dtype=float)
    dof = np.asarray(dof_eff, dtype=float)
    ny = float(n_data)
    # Reference alpha_0 = the solution with the MINIMUM residual (least-squares end;
    # normally the smallest alpha, but argmin also covers Provencher's note that a
    # numerical instability can push it to a slightly larger alpha).
    ref = int(np.argmin(v))
    v0 = float(v[ref])
    ndf0 = float(np.clip(dof[ref], 1e-6, ny - 1e-6))   # NDF(alpha_0), fixed
    frac_increase = np.clip((v - v0) / v0 if v0 > 0 else np.zeros_like(v), 0.0, None)
    f_stat = frac_increase * (ny - ndf0) / ndf0
    fc = special.fdtr(ndf0, ny - ndf0, f_stat)         # F CDF, dof (NDF0, Ny-NDF0)
    idx = int(np.argmin(np.abs(fc - prob_reject)))
    return idx, fc


def fit_contin(
    measurement: DLSMeasurement,
    rh_min_nm: float = 1.0,
    rh_max_nm: float = 1000.0,
    n_grid: int = 100,
    alpha: Optional[float] = None,
    alpha_min: float = 1e-6,
    alpha_max: float = 1e2,
    n_alpha: int = 40,
    beta: Optional[float] = None,
    baseline: Optional[float] = None,
    tau_min_s: Optional[float] = None,
    tau_max_s: Optional[float] = None,
    skip_initial_channels: int = 0,
    alpha_method: str = 'gcv',
    ftest_prob_reject: float = 0.5,
) -> ContinResult:
    """Recover a smoothed decay-rate distribution by regularized inversion.

    Solves min ||A x - y||^2 + alpha^2 ||L x||^2 subject to x >= 0, with L the
    second-difference operator (Provencher 1982). The regularization parameter
    alpha trades fit quality against distribution smoothness.

    If alpha is None (default), a sweep over [alpha_min, alpha_max] is run and alpha
    is chosen automatically by `alpha_method`; the full sweep is returned so the
    choice can be inspected and overridden. If alpha is given, that value is used
    directly and a single-point sweep is returned for consistency.

    Three automatic selectors share the same sweep (only the selection differs):
      * 'gcv' (default): Generalized Cross-Validation (Golub, Heath & Wahba 1979) --
        minimizes n V(alpha)/(n - dof(alpha))^2, sigma^2-free and asymptotically
        optimal, the robust general choice for this ill-conditioned kernel.
      * 'lcurve': the L-curve corner = point of maximum curvature of the log-log
        residual-vs-solution-norm curve (Hansen 1998 Eq. 7.24; Hansen & O'Leary 1993).
      * 'ftest': Provencher's original F-test / "probability to reject" criterion
        (Provencher 1982a; Scotti et al. 2015), picking the solution whose residual
        increase over the least-regularized fit sits at the `ftest_prob_reject`
        significance level (default 0.5). Higher -> smoother, lower -> rougher.

    Parameters
    ----------
    measurement : DLSMeasurement
    rh_min_nm, rh_max_nm, n_grid : grid specification (default 1-1000 nm, 100 pts)
    alpha : float, optional
        Fixed regularization parameter. If omitted, chosen by `alpha_method`.
    alpha_min, alpha_max, n_alpha :
        Log-spaced alpha sweep (default 1e-6 to 1e2, 40 points). The 40-point sweep
        gives the L-curve spline a stable curvature and lets GCV resolve its interior
        minimum; the range is unchanged from earlier versions.
    beta, baseline : float, optional
        Coherence factor and baseline; estimated from the data if omitted.
    tau_min_s, tau_max_s : float, optional
        Inclusive delay-time window. Default uses all points.
    alpha_method : str
        'gcv' (default), 'lcurve', or 'ftest'. Ignored if `alpha` is given.
    ftest_prob_reject : float
        F-test significance level (Provencher default 0.5). Only used for 'ftest'.

    Returns
    -------
    ContinResult
        .distribution is the chosen DistributionResult; .lcurve holds the sweep;
        .alpha_selection_method records which selector chose alpha;
        .alpha_at_ceiling flags a high-alpha-end pick (GCV over-regularization guard).
    """
    if alpha_method not in ('gcv', 'lcurve', 'ftest'):
        raise ValueError(
            f"alpha_method must be 'gcv', 'lcurve', or 'ftest', got {alpha_method!r}.")
    (tau, g2m1, baseline, baseline_est, beta, beta_est,
     q, rh_grid, gamma_grid, A, y, w) = _prepare_distribution_inputs(
        measurement, tau_min_s, tau_max_s, beta, baseline,
        rh_min_nm, rh_max_nm, n_grid,
        skip_initial_channels=skip_initial_channels)

    L = _second_difference_operator(n_grid)

    if alpha is not None:
        # User-fixed alpha: solve once.
        x = _solve_distribution(A, y, w, alpha=alpha, L=L)
        dist = _distribution_summary(
            x, A, y, w, rh_grid, gamma_grid, beta, baseline, g2m1, tau, q,
            method='contin', alpha=alpha,
            beta_estimated=beta_est, baseline_estimated=baseline_est,
            n_skipped=skip_initial_channels)
        lcurve = LCurveResult(
            alphas=np.array([alpha]),
            residual_norms=np.array([dist.residual_norm]),
            solution_norms=np.array([_regularization_seminorm_sq(L, x)]),
            optimal_alpha=alpha, optimal_index=0)
        return ContinResult(distribution=dist, lcurve=lcurve,
                            alpha_was_user_supplied=True,
                            alpha_selection_method='user')

    # Sweep (shared by all three selectors -- only the selection function differs).
    alphas = np.geomspace(alpha_min, alpha_max, n_alpha)
    residual_norms = np.empty(n_alpha)
    solution_norms = np.empty(n_alpha)
    solutions = []
    for i, a in enumerate(alphas):
        x = _solve_distribution(A, y, w, alpha=a, L=L)
        solutions.append(x)
        residual_norms[i] = np.sum(w * (A @ x - y) ** 2)   # weighted V(alpha)
        solution_norms[i] = _regularization_seminorm_sq(L, x)   # ||L x||^2 (L-curve axis)

    dof_eff = ftest_fc = gcv_curve = None
    if alpha_method in ('gcv', 'ftest'):
        # Both GCV and the F-test need the Tikhonov hat-trace, which must reflect the
        # WEIGHTED design (the same M^.5 A the solve minimizes over) so it agrees with
        # the weighted V(alpha) (Provencher 1982a Eq. 3.9; Golub 1979 Eq. 1.4).
        Aw, _yw = _apply_weight(A, y, w)
        AtA = Aw.T @ Aw
        dof_eff = np.array([_tikhonov_effective_dof(Aw, L, a, AtA) for a in alphas])
    if alpha_method == 'gcv':
        opt_idx, gcv_curve = _gcv_corner(residual_norms, dof_eff, n_data=int(y.size))
    elif alpha_method == 'ftest':
        opt_idx, ftest_fc = _ftest_corner(
            residual_norms, dof_eff, n_data=int(y.size),
            prob_reject=ftest_prob_reject)
    else:
        opt_idx = _lcurve_corner(alphas, residual_norms, solution_norms)
    opt_alpha = float(alphas[opt_idx])
    x_opt = solutions[opt_idx]
    # High-alpha-end pick: over-regularization signal (GCV flat-minimum failure mode,
    # Hansen 1998 p.185). A floor (low-alpha) pick is legitimate on clean data and is
    # deliberately NOT flagged. Generic over selectors (cheap, correct for all three).
    alpha_at_ceiling = bool(opt_idx == n_alpha - 1)

    dist = _distribution_summary(
        x_opt, A, y, w, rh_grid, gamma_grid, beta, baseline, g2m1, tau, q,
        method='contin', alpha=opt_alpha,
        beta_estimated=beta_est, baseline_estimated=baseline_est,
        n_skipped=skip_initial_channels)
    lcurve = LCurveResult(
        alphas=alphas, residual_norms=residual_norms,
        solution_norms=solution_norms, optimal_alpha=opt_alpha,
        optimal_index=opt_idx, dof_eff=dof_eff, ftest_fc=ftest_fc, gcv=gcv_curve)
    return ContinResult(distribution=dist, lcurve=lcurve,
                        alpha_was_user_supplied=False,
                        alpha_selection_method=alpha_method,
                        ftest_prob_reject=(ftest_prob_reject
                                           if alpha_method == 'ftest' else None),
                        alpha_at_ceiling=alpha_at_ceiling)


# ---------------------------------------------------------------------------
# Rh <-> Gamma axis helper (for the visualization toggle and general use)
# ---------------------------------------------------------------------------

def distribution_axis(distribution: DistributionResult, axis: str = 'rh'):
    """Return (x_values, weights, x_label) for plotting a DistributionResult.

    Supports the Rh <-> Gamma visualization toggle. The same weights are plotted
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
    # Local maxima, including the endpoints if they rise above their neighbor. A
    # flat-topped peak (a run of consecutive EQUAL heights, e.g. [...,1,1,...]) is
    # ONE population, not two: scan by runs of equal weight and report the run's
    # center, so an exact plateau isn't split into two spurious peaks.
    maxima: List[int] = []
    i = 0
    while i < n:
        if not (w[i] > 0):
            i += 1
            continue
        j = i
        while j + 1 < n and w[j + 1] == w[i]:   # extend the equal-height run [i..j]
            j += 1
        left_ok = (i == 0) or (w[i] >= w[i - 1])
        right_ok = (j == n - 1) or (w[j] >= w[j + 1])
        strictly = ((i > 0 and w[i] > w[i - 1]) or (j < n - 1 and w[j] > w[j + 1]))
        if left_ok and right_ok and strictly:
            maxima.append((i + j) // 2)         # center of the plateau (grid index)
        i = j + 1
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
