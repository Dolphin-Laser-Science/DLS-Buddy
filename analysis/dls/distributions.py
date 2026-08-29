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
The weight's overall scale cancels in the NNLS argmin and at any FIXED alpha, so
the relative weight needs no new inputs. It does NOT cancel in the alpha
selection: rescaling w by c reparametrizes the Tikhonov objective as
alpha -> alpha/sqrt(c), so the selected alpha slides by sqrt(c) across the fixed
[alpha_min, alpha_max] sweep -- and w = |g2-1-baseline|/beta makes the per-run
beta exactly such a rescale (measured: one 1.60x sweep step per beta-halving,
recovered peak unchanged). The automatic selectors re-minimize and track the
shift, so the recovered distribution is near-invariant until the tracked optimum
reaches a sweep bound (flagged at either end: alpha_at_ceiling / alpha_at_floor);
a USER-FIXED alpha does not compensate at all. Any substituted weight must be scale-anchored for the same
reason. The weight is a precision map that improves the FIT only; no statistical +/- is ever
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
    # False only when a nonlinear fit (lognormal) failed to converge and the curve is
    # the seed estimate, not a fit. The linear solvers (NNLS/CONTIN) always deliver
    # their solution, so they leave the default. Consumers must never present the
    # summary values, peaks, or a rho source from a success=False result as fitted.
    success: bool = True


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
    L-curve). `alpha_at_floor` is its twin for the LOW end: the selector wanted a
    smaller alpha than the range allows -- at a low floor (the default 1e-4) that is
    essentially unregularized NNLS, whose peak structure may be inversion noise rather
    than populations (the "extra peaks" of an unregularized fit, CONTIN Users Manual
    §3.2.3); at a user-raised floor it is only the least-smoothed answer the range
    permitted. Both are set from the same selected sweep index, so they are exclusive
    whenever the sweep has two or more points (a one-point sweep is both edges); a
    user-supplied alpha sets neither (there is no sweep). Both flag, neither
    withholds: the distribution is always returned (CLAUDE.md invariant 4,
    flag-not-refuse).

    `alpha_reliable` / `reliability_note` are the family pair every parametric result
    carries (CumulantResult's shape): `alpha_reliable` is False iff either edge flag is
    set, and `reliability_note` is the ONE sentence every surface shows for it --
    ALPHA_FLOOR_NOTE / ALPHA_CEILING_NOTE, both joined with '; ' on a one-point sweep,
    '' when interior or user-fixed. Every surface reads this string (pinned by
    tests/test_workspace.py::test_contin_edge_note_is_one_string_on_every_surface).
    """
    distribution: DistributionResult
    lcurve: LCurveResult
    alpha_was_user_supplied: bool
    alpha_selection_method: str = 'lcurve'    # 'gcv' | 'lcurve' | 'ftest' | 'user'
    ftest_prob_reject: Optional[float] = None
    alpha_at_ceiling: bool = False
    alpha_at_floor: bool = False
    alpha_reliable: bool = True
    reliability_note: str = ''


# The sweep-edge wording, in its one home. The floor phrase is sized to the predicate:
# the index says only that the selector wanted past the bound; what the solution then IS
# depends on where the user put the range (unregularized at the default floor, not at
# alpha = 30), so the floor's consequence lives in the how-to help, conditioned on the
# floor's value. The ceiling phrase keeps its hedged consequence ("possible
# over-regularization"), which holds at any ceiling.
ALPHA_FLOOR_NOTE = ('CONTIN α at the sweep floor — the selector wanted a smaller α than '
                    'the range allows')
ALPHA_CEILING_NOTE = 'CONTIN α at the sweep ceiling — possible over-regularization'


def _edge_note_for(at_floor: bool, at_ceiling: bool) -> str:
    """The reliability note for a selected-alpha position: '' when interior, one phrase
    at an edge, both (floor first) on a one-point sweep, which is both edges."""
    parts = ([ALPHA_FLOOR_NOTE] if at_floor else []) + ([ALPHA_CEILING_NOTE] if at_ceiling else [])
    return '; '.join(parts)


# ---------------------------------------------------------------------------
# Solver internals
# ---------------------------------------------------------------------------

# Iteration budget for scipy.optimize.nnls, as a multiple of the number of unknowns.
# scipy's own default is 3n, which one committed real correlogram exceeds at one alpha of
# its 40-point sweep (a 298x100 augmented system: 300 raises, 1000 converges, 10000 gives
# the identical vector) -- and an uncaught raise there cost the user the whole
# distribution. A budget, not a tolerance (invariant 4: no physics threshold hides here):
# nnls stops on optimality, so raising the cap never changes a solve that already
# converged. Serves every _solve_distribution caller (fit_contin's sweep and fit_nnls).
NNLS_MAXITER_FACTOR = 10


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
    # NNLS_MAXITER_FACTOR is an iteration BUDGET, not a tolerance: nnls terminates on
    # optimality, so any converged solve is unchanged by a larger cap, and the cap only
    # decides whether a slow solve finishes or raises. See the constant's comment.
    x, _ = optimize.nnls(A_aug, y_aug, maxiter=NNLS_MAXITER_FACTOR * A_aug.shape[1])
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
    success: bool = True,
) -> DistributionResult:
    """Assemble a DistributionResult from a raw solver solution x."""
    total = x.sum()
    # When the solution cannot be normalized, WITHHOLD the whole summary rather than
    # returning the unnormalized vector. Three reasons it has to be NaN and not `x`:
    # `weights` is documented as "normalized intensity weights (sum = 1)", so handing back
    # an unnormalized vector breaks that contract silently; `mean_rh` computed from it is
    # meaningless; and -- the one that actually bit -- `peak_rh = rh_grid[argmax(weights)]`
    # is UNCHANGED by any guard on the mean, so an all-equal degenerate vector still
    # reports the smallest grid point as "the peak", a perfectly plausible small size.
    # Guarding the divisor alone fixed the mean and left the peak lying (spec-03 review).
    if math.isfinite(total) and total > 0:
        weights = x / total
    else:
        weights = np.full_like(np.asarray(x, dtype=float), np.nan)
    # intensity-weighted means over the distribution
    mean_gamma = float(np.sum(weights * gamma_grid))
    mean_rh = float(np.sum(weights * rh_grid_nm))
    peak_rh = (float(rh_grid_nm[int(np.nanargmax(weights))])
               if weights.size and np.any(np.isfinite(weights)) else float('nan'))
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
        success=success,
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
    # isfinite as well as > 0: `beta <= 0` admits nan AND +inf, and every line below
    # divides by beta -- an infinite one silently yields a signal of zeros, a NaN one a
    # signal of NaNs, and neither reaches the user as an error. `_estimate_beta` now
    # guarantees a positive finite value, but a USER-supplied beta arrives unchecked
    # (2026-07-31 spec-01).
    if not (math.isfinite(beta) and beta > 0):
        raise ValueError(f"beta must be a positive finite number, got {beta!r}.")

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
    # spurious large-Rh weight (the noisy-unimodal mean-Rh regression). Taking
    # the signed square root keeps the noise zero-mean, so no pedestal forms; the
    # fitted model A x (a true non-negative g1) still only chases the real signal.
    y = np.sign(u) * np.sqrt(np.abs(u))
    # Delta-method inverse-variance weight of the recovered g1: a (locally uniform)
    # g2-1 noise through the sqrt gives Var(g1) prop 1/|u|, so w prop |u| = y^2. The
    # overall scale cancels at any FIXED alpha (and for NNLS) but NOT in the alpha
    # selection: w carries 1/beta, and rescaling w by c slides the selected alpha by
    # sqrt(c) across the fixed sweep (module docstring); noise-floor channels
    # (|u| -> 0) get ~0 weight. (Note sqrt(w) y = u exactly, so the weighted fit
    # targets the same unbiased (g2-1)/beta as the legacy diagonal kernel, but
    # through a cross-term-preserving g1 design -- noise-robust AND phantom-free.)
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
        return w / s if (math.isfinite(s) and s > 0) else w

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

    converged = True
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
    except (RuntimeError, ValueError, TypeError):
        # Same exception family the exponential fitters catch around curve_fit. The
        # seed shape stands in as a "roughly here, unconverged" hint; the flag
        # keeps it from ever reading as a fit downstream.
        amp, mu, sigma = 1.0, mu0, 0.3
        converged = False

    x = amp * _weights(mu, sigma)
    return _distribution_summary(
        x, A, y, w, rh_grid, gamma_grid, beta, baseline, g2m1, tau, q,
        method='lognormal', alpha=None,
        beta_estimated=beta_est, baseline_estimated=baseline_est,
        n_skipped=skip_initial_channels, success=converged)


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
    # A cubic spline needs >= 4 STRICTLY INCREASING, FINITE abscissae. A single- or
    # few-point sweep, or equal/near-duplicate bounds (alpha_min == alpha_max makes t
    # constant), would make CubicSpline raise -- fall back to the discrete curvature
    # instead of crashing the whole CONTIN fit. Finiteness is checked on t ITSELF, not
    # on its diffs: an alpha of 0 or +inf puts a -inf/+inf in t, whose diff is +inf,
    # which passes a bare "strictly increasing" test and then poisons the whole spline
    # to NaN (a corner silently pinned to the least-regularized end).
    if a.size < 4 or not (np.all(np.isfinite(t)) and np.all(np.diff(t) > 0)):
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
    if not (math.isfinite(gmax) and gmax > 0.0):
        # a perfectly flat L-curve (every solve returned the same norms) carries no
        # corner information; pick the least-regularized end rather than divide by zero.
        # A non-finite gmax (a NaN/inf norm anywhere on the curve) carries none either,
        # and the bare `gmax <= 0.0` form let it through -- NaN compares False.
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
    # Usable points are those with a FINITE, non-zero tangent: a bare `grad2 > 0.0`
    # compares False for NaN (fine) but True for +inf, and an infinite lane makes the
    # curvature NaN, which np.argmax would then select as the corner.
    usable = np.isfinite(grad2) & (grad2 > 0.0)
    if not np.any(usable):
        return 0
    kappa = np.where(usable,
                     (zp * epp - zpp * ep) / np.power(np.clip(grad2, 1e-300, None), 1.5),
                     -np.inf)
    return int(np.argmax(kappa))


def _masked(a: np.ndarray) -> np.ndarray:
    """A copy of `a` with every non-finite entry replaced by +inf, so it can never win an
    `argmin`.

    `np.argmin` returns the INDEX OF A NaN when one is present, so an alpha whose criterion
    could not be computed is exactly the alpha a bare `argmin` selects -- a silent wrong
    selection, strictly worse than the withhold it came from. `np.clip` does not help:
    it passes NaN straight through.

    One helper shared by both alpha selectors AND by both of their fallback branches --
    masking a selector's main objective while leaving its last-resort `argmin` bare is the
    same defect one branch down. `_discrete_curvature_argmax` uses the same
    never-wins-sentinel idea with `-inf`, since it maximizes.
    """
    a = np.asarray(a, dtype=float)
    return np.where(np.isfinite(a), a, np.inf)


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
    # An unanswerable alpha must not WIN the selection. `np.argmin` returns the index of a
    # NaN, and `np.clip` passes NaN straight through, so an alpha whose hat-trace could
    # not be computed was exactly the alpha this returned -- a silent wrong selection,
    # strictly worse than the withhold it came from. Masked to a never-wins +inf, the same
    # form `_discrete_curvature_argmax` uses.
    if not np.any(np.isfinite(gcv)):
        # No alpha in the sweep has a usable GCV value, so the criterion did not answer.
        # Fall back to the minimum finite RESIDUAL -- the same reference the F-test uses --
        # rather than to index 0 by accident. (Index 0 IS the smallest alpha for the
        # ascending geomspace the caller builds, but that is a property of the caller, not
        # of this function, and the two must not agree only by coincidence.) If even the
        # residuals are all non-finite the masked argmin ties at index 0, which is the one
        # case where there is genuinely no basis to prefer any alpha.
        return int(np.argmin(_masked(v))), gcv
    return int(np.argmin(_masked(gcv))), gcv


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

    Returns NaN when the normal matrix is not finite. `np.linalg.solve` is an LU
    factorization, not an SVD, so a non-finite entry does NOT make it raise -- it returns
    entries that are not a solution (measured: a mix of NaN and zeros, matrix-dependent),
    the `except LinAlgError` never fires, and the garbage trace was clipped into a
    plausible-looking number. Both `A.T @ A` and `alpha**2` can saturate here on finite
    input, since neither the design nor the alpha sweep has a magnitude bound.
    """
    if AtA is None:
        AtA = A.T @ A
    # `alpha * alpha`, not `alpha ** 2`: the sweep passes numpy floats (which saturate),
    # but the signature accepts a PYTHON float, and there `**` RAISES OverflowError --
    # before any guard below can run. See analysis/dls/_common.py::_d_from_gamma.
    M = AtA + (alpha * alpha) * (L.T @ L)
    if not (np.all(np.isfinite(M)) and np.all(np.isfinite(AtA))):
        return float('nan')
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
    # numerical instability can push it to a slightly larger alpha), chosen among alphas
    # whose effective dof is USABLE. NDF0 is taken from the reference and appears in both
    # the scaling factor and the F-distribution, so a reference whose hat-trace could not be
    # computed makes every fc NaN -- and since `np.argmin` returns the index of a NaN, the
    # criterion would then select that same failed alpha. Masking the objective alone cannot
    # fix it: the poison enters through the reference.
    dof_usable = np.isfinite(dof)
    ref_pool = np.where(np.isfinite(v) & dof_usable, v, np.inf)
    # The last-resort branch is masked too: `np.argmin` returns the index of a NaN, so a
    # bare `argmin(v)` here would select an alpha whose RESIDUAL is NaN.
    ref = int(np.argmin(ref_pool if np.any(np.isfinite(ref_pool)) else _masked(v)))
    v0 = float(v[ref])
    ndf0 = float(np.clip(dof[ref], 1e-6, ny - 1e-6))   # NDF(alpha_0), fixed
    # An unusable reference residual yields NaN, NOT zeros. Substituting zeros turns the
    # poison into a perfectly finite F-statistic of 0 for every alpha, so `fc` becomes a
    # fabricated flat curve, `objective` is 0.5 everywhere, the "criterion did not answer"
    # branch below never fires, and the tie resolves to index 0 -- the "index 0 by
    # accident" the sibling selector explicitly disowns -- and the fabricated curve would
    # also be stored on `LCurveResult.ftest_fc` and plotted as a real PROB1.
    frac_increase = np.clip((v - v0) / v0 if (math.isfinite(v0) and v0 > 0)
                            else np.full_like(v, np.nan), 0.0, None)
    f_stat = frac_increase * (ny - ndf0) / ndf0
    fc = special.fdtr(ndf0, ny - ndf0, f_stat)         # F CDF, dof (NDF0, Ny-NDF0)
    # As in `_gcv_corner`: a NaN would WIN this argmin, so an alpha whose effective dof
    # could not be computed would be the one selected. Masked to a never-wins +inf. If
    # NDF0 itself is unanswerable every fc is NaN, and the fallback is the reference alpha
    # (the least-squares end this criterion is defined against), not index 0 by accident.
    objective = np.abs(fc - prob_reject)
    if not np.any(np.isfinite(objective)):
        return ref, fc
    return int(np.argmin(_masked(objective))), fc


def fit_contin(
    measurement: DLSMeasurement,
    rh_min_nm: float = 1.0,
    rh_max_nm: float = 1000.0,
    n_grid: int = 100,
    alpha: Optional[float] = None,
    alpha_min: float = 1e-4,
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
    second-difference operator (Provencher 1982a Eq. 3.10; implemented as the CONTIN
    program of Provencher 1982). The regularization parameter
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
        Log-spaced alpha sweep (default 1e-4 to 1e2, 40 points). The 40-point sweep
        gives the L-curve spline a stable curvature and lets GCV resolve its interior
        minimum. The floor is a documented default (CLAUDE.md invariant 4), derived
        rather than picked: CONTIN's own sweep is scale-relative, from machine
        precision times the largest singular value up to that value (CONTIN Users
        Manual §4.1.3), and its "alpha small enough" test is that the smallest-alpha
        solution's residual agrees with the unregularized one in the first few figures
        (§3.6.5 point 1). 1e-4 is the highest floor at which that still holds on the
        committed reference correlograms (measured 2026-08-24 over n_grid 15-1000: the
        residual at 1e-4 exceeds the unregularized one by <= 1e-5 relative on every
        cell, while at 1e-3 regularization is already acting), so a floor pick is the
        unregularized reference by construction and is flagged as such. A finer Rh grid
        (n_grid) is what drives the F-test and GCV onto the floor; the flag says when
        it happened. Pinned by tests/test_dls.py::test_alpha_floor_is_the_unregularized_reference
        and the floor-flag tests beside it.
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
        .alpha_at_ceiling flags a high-alpha-end pick (GCV over-regularization guard);
        .alpha_at_floor flags a low-alpha-end pick (the selector wanted a smaller
        alpha than the range allows); .alpha_reliable is False at either edge and
        .reliability_note carries the one sentence every surface shows for it.
    """
    if alpha_method not in ('gcv', 'lcurve', 'ftest'):
        raise ValueError(
            f"alpha_method must be 'gcv', 'lcurve', or 'ftest', got {alpha_method!r}.")
    # Both tails (rule 12): a bare `alpha > 0` admits +inf (an infinite regularizer)
    # and lets NaN through, so the guard names finiteness as the siblings do.
    if alpha is not None and not (math.isfinite(alpha) and alpha > 0):
        raise ValueError(
            f"alpha must be a finite number > 0, got {alpha!r}. A non-positive alpha "
            f"disables the Tikhonov regularization entirely (see fit_nnls for that "
            f"unregularized case).")
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
    # The bounds are the fixed alpha's family -- and the two the GUI exposes as free
    # text -- so they take the same finite-and-positive guard: a NaN or negative bound
    # would otherwise ride np.geomspace onto the result as the recorded alpha, under
    # the sweep-edge note (a proxy, not the failed precondition). An inverted pair is
    # not refused: it sweeps descending, a shape the edge-note surfaces are pinned on.
    if not (math.isfinite(alpha_min) and math.isfinite(alpha_max)
            and alpha_min > 0 and alpha_max > 0):
        raise ValueError(
            f"alpha_min / alpha_max must be finite and > 0, got ({alpha_min!r}, "
            f"{alpha_max!r}): the alpha sweep cannot be built from them.")
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
    # Sweep-edge picks, one predicate each on the SAME selected index (so the two can
    # never both fire and no second threshold exists). Ceiling: over-regularization
    # signal (GCV flat-minimum failure mode, Hansen 1998 p.185). Floor: essentially
    # unregularized NNLS at the default floor (see the fit_contin docstring on
    # alpha_min for what reaches it), whose extra peaks are inversion noise. Generic
    # over selectors.
    alpha_at_ceiling = bool(opt_idx == n_alpha - 1)
    alpha_at_floor = bool(opt_idx == 0)
    reliability_note = _edge_note_for(alpha_at_floor, alpha_at_ceiling)

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
                        alpha_at_ceiling=alpha_at_ceiling,
                        alpha_at_floor=alpha_at_floor,
                        alpha_reliable=not reliability_note,
                        reliability_note=reliability_note)


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

    `rh_lo_nm` / `rh_hi_nm` are the Rh values at the first and last grid points of
    the peak's lobe (the integration bounds the fraction was summed over --
    inclusive on both ends). NaN when the producing path predates the bounds
    (additive fields, defaulted).
    """
    rh_nm: float                      # Rh at the peak maximum
    gamma_s_inv: float                # decay rate at the peak maximum
    weight_fraction: float            # intensity-weighted area of the peak's lobe
    is_dominant: bool
    rh_lo_nm: float = float('nan')    # Rh at the lobe's first grid point
    rh_hi_nm: float = float('nan')    # Rh at the lobe's last grid point


def scan_distribution_peaks(rh_grid_nm, weights, min_weight_fraction: float = 0.05):
    """Resolve the peaks of a weight-vs-Rh distribution, weighting-agnostic.

    Returns a list of ``(peak_index, weight_fraction, lo_idx, hi_idx)`` on the Rh-SORTED
    grid, so the caller can build peak objects in whatever weighting the ``weights`` carry
    (the same scan serves the intensity distribution AND its reweighted volume/surface/
    number views — one definition, per the fix-the-family rule). ``peak_index`` indexes
    the Rh-ascending arrays returned in ``rh_sorted``; ``lo_idx``/``hi_idx`` are the
    peak's lobe bounds as a half-open slice ``[lo_idx, hi_idx)`` of those arrays — the
    exact range the fraction was summed over (mode-resolved scattering integrates a
    lobe between these bounds, so they are returned rather than discarded).

    Returns ``(peaks, rh_sorted, w_sorted)``: peaks is ``[(idx, frac, lo, hi), ...]``
    (empty if the distribution is degenerate), rh_sorted / w_sorted the ascending-Rh
    arrays the indices refer to.
    """
    rh = np.asarray(rh_grid_nm, dtype=float)
    w = np.asarray(weights, dtype=float)
    if rh.size == 0:
        return [], rh, w
    order = np.argsort(rh)
    rh, w = rh[order], w[order]
    total = float(w.sum())
    if not (math.isfinite(total) and total > 0):
        return [], rh, w

    n = w.size
    # Local maxima, including the endpoints if they rise above their neighbor. A
    # flat-topped peak (a run of consecutive EQUAL heights, e.g. [...,1,1,...]) is
    # ONE population, not two: scan by runs of equal weight and report the run's
    # center, so an exact plateau isn't split into two spurious peaks.
    maxima: List[int] = []
    i = 0
    while i < n:
        if not (math.isfinite(w[i]) and w[i] > 0):
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

    peaks = []
    for k, peak_i in enumerate(maxima):
        lo, hi = bounds[k], bounds[k + 1]
        frac = float(w[lo:hi].sum() / total)
        if frac < min_weight_fraction:
            continue
        peaks.append((peak_i, frac, lo, hi))
    return peaks, rh, w


def validate_integration_bounds(bounds_nm):
    """Validate manual integration intervals; returns them as float pairs.

    THE one predicate home (rule 12: one boundary, not two to keep aligned):
    both `integrate_between` (at use) and the controller's `set_mode_bounds`
    (at user-input time) call this, so what the store accepts and what the
    integration refuses cannot drift apart. Raises ValueError on a non-finite
    or inverted interval, and on overlap -- including TOUCHING intervals
    (hi_a == lo_b), because bounds are inclusive both ends, so even touching
    would count a shared grid point twice.
    """
    pairs = [(float(lo), float(hi)) for lo, hi in bounds_nm]
    for lo, hi in pairs:
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            raise ValueError(
                f'integration bounds must be finite with lo < hi, got ({lo}, {hi}).')
    for (lo_a, hi_a), (lo_b, hi_b) in zip(sorted(pairs), sorted(pairs)[1:], strict=False):
        if hi_a >= lo_b:
            raise ValueError(
                f'integration bounds overlap: ({lo_a}, {hi_a}) and ({lo_b}, {hi_b}) — '
                'overlapping intervals would count the shared weight twice.')
    return pairs


def integrate_between(rh_grid_nm, weights, bounds_nm):
    """Integrate a weight-vs-Rh distribution over user-set Rh intervals, weighting-agnostic.

    The manual-bounds counterpart of `scan_distribution_peaks`' automatic lobes: when
    overlapping populations defeat the local-minimum split, the user supplies explicit
    ``bounds_nm = [(lo_nm, hi_nm), ...]`` and each interval's summed weight becomes that
    mode's fraction of the total. Works on any weighting (intensity or reweighted views),
    exactly like the scan — one definition for the whole distribution family.

    Interval semantics: grid points with ``lo_nm <= rh <= hi_nm`` belong to the interval
    (inclusive both ends — the values are user-facing Rh, not slice indices). Intervals
    must be well-formed (``lo < hi``) and must not overlap each other: a split that counts
    a grid point twice writes fractions that sum past 1 — a false record, refused with a
    ``ValueError``. An interval that captures NO weight is a *noted* result (fraction 0.0,
    ``rh_at_max_nm`` NaN), not an error: an empty window is honest information about where
    the user cut.

    Returns a list of ``(rh_at_max_nm, weight_fraction, lo_idx, hi_idx)`` in the order of
    ``bounds_nm``, on the Rh-ascending grid — the same tuple shape as the scan, with the
    peak position taken as the weight maximum inside the interval. ``lo_idx``/``hi_idx``
    are the half-open slice of the sorted arrays. Fractions are relative to the FULL
    distribution total (not the covered subset), so a partial cover discloses itself as
    fractions summing below 1. Returns ``(intervals, rh_sorted, w_sorted)``; empty
    ``intervals`` if the distribution is degenerate (empty grid or non-positive total).
    """
    rh = np.asarray(rh_grid_nm, dtype=float)
    w = np.asarray(weights, dtype=float)
    pairs = validate_integration_bounds(bounds_nm)
    order = np.argsort(rh)
    rh, w = rh[order], w[order]
    if rh.size == 0:
        return [], rh, w
    total = float(w.sum())
    if not (math.isfinite(total) and total > 0):
        return [], rh, w
    intervals = []
    for lo_nm, hi_nm in pairs:
        inside = np.nonzero((rh >= lo_nm) & (rh <= hi_nm))[0]
        if inside.size == 0:
            intervals.append((float('nan'), 0.0, 0, 0))
            continue
        lo_idx, hi_idx = int(inside[0]), int(inside[-1]) + 1
        seg = w[lo_idx:hi_idx]      # NaN weights never reach here: a NaN total is degenerate above
        frac = float(seg.sum() / total)
        if frac <= 0.0:             # grid points but no weight — noted, not an error
            intervals.append((float('nan'), 0.0, lo_idx, hi_idx))
            continue
        peak_i = lo_idx + int(np.argmax(seg))
        intervals.append((float(rh[peak_i]), frac, lo_idx, hi_idx))
    return intervals, rh, w


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

    A success=False result has NO peaks: its curve is the seed estimate, and a peak
    of a failed fit is a fit-like number no caller may present. The refusal lives
    here, in the one scan every consumer routes through, so no call site needs its
    own gate.
    """
    if getattr(distribution, 'success', True) is False:
        return []
    gamma = np.asarray(distribution.gamma_grid_s_inv, dtype=float)
    rh_in = np.asarray(distribution.rh_grid_nm, dtype=float)
    # A gamma grid whose length disagrees with the Rh grid is a malformed/degenerate result;
    # the pre-refactor code co-sorted the two and raised IndexError there (caught upstream as
    # "no peaks"). Preserve that: return [] rather than emitting NaN-gamma peaks that could
    # surface a NaN-decay-rate population in the table or as a rho source.
    if gamma.size != rh_in.size:
        return []
    scanned, rh, _w = scan_distribution_peaks(
        rh_in, distribution.weights, min_weight_fraction)
    if not scanned:
        return []
    # Re-sort gamma to match the Rh-ascending arrays the scan returned.
    gamma = gamma[np.argsort(rh_in)]
    peaks: List[DistributionPeak] = []
    for peak_i, frac, lo, hi in scanned:
        peaks.append(DistributionPeak(
            rh_nm=float(rh[peak_i]), gamma_s_inv=float(gamma[peak_i]),
            weight_fraction=frac, is_dominant=False,
            rh_lo_nm=float(rh[lo]), rh_hi_nm=float(rh[hi - 1])))
    if not peaks:
        return []
    dominant = max(range(len(peaks)), key=lambda j: peaks[j].weight_fraction)
    peaks[dominant].is_dominant = True
    return peaks
