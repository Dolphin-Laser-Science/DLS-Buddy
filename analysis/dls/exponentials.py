"""
analysis/dls/exponentials.py
============================

Parametric exponential fits of g2(tau) - 1, all carrying the Siegert factor of 2
explicitly (the Siegert relation g2 - 1 = beta |g1|^2; Chu 1991) and reporting
Gamma as the physical g1 decay rate:

  - fit_single_exponential : g2 - 1 = beta exp(-2 Gamma tau)  (monodisperse)
  - fit_double_exponential : two discrete modes via the full Siegert relation
  - fit_kww                : Kohlrausch-Williams-Watts stretched exponential

Each is seeded from a second-order cumulant fit (computed internally if not
supplied, on the same delay window) and returns plot-ready arrays.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import optimize, special

from core.data_models import DLSMeasurement
from analysis.dls._common import (
    _apply_tau_window,
    _decay_rate_to_rh_nm,
    _measurement_q_m,
    _require_viscosity,
    _rms_error,
)
from analysis.dls.cumulants import CumulantResult, fit_cumulants


@dataclass
class ExponentialMode:
    """One decay mode of an exponential fit."""
    amplitude_fraction: float         # relative g1 amplitude (modes sum to 1)
    gamma_s_inv: float
    d_m2_s: float
    rh_nm: float


@dataclass
class SingleExponentialResult:
    beta: float
    mode: ExponentialMode
    q_m_inv: float
    fit_tau_s: np.ndarray             # delay times actually fitted (after windowing)
    fitted_g2m1: np.ndarray           # model over fit_tau_s
    residuals: np.ndarray
    rms_error: float
    success: bool


@dataclass
class DoubleExponentialResult:
    beta: float
    mode1: ExponentialMode            # by convention the faster mode (larger Gamma)
    mode2: ExponentialMode            # the slower mode (smaller Gamma)
    q_m_inv: float
    fit_tau_s: np.ndarray             # delay times actually fitted (after windowing)
    fitted_g2m1: np.ndarray
    residuals: np.ndarray
    rms_error: float
    success: bool


@dataclass
class KWWResult:
    """Kohlrausch-Williams-Watts (stretched-exponential) fit."""
    beta: float                       # coherence factor (optical intercept)
    tau_c_s: float                    # characteristic relaxation time of g1
    stretch: float                    # stretch exponent in (0, 1]  (distinct from beta)
    mean_tau_s: float                 # <tau> = (tau_c/stretch) Gamma(1/stretch)
    # sizes from the two natural rate definitions
    gamma_from_tau_c_s_inv: float     # 1 / tau_c
    rh_from_tau_c_nm: float
    gamma_from_mean_tau_s_inv: float  # 1 / <tau>
    rh_from_mean_tau_nm: float
    q_m_inv: float
    fit_tau_s: np.ndarray             # delay times actually fitted (after windowing)
    fitted_g2m1: np.ndarray
    residuals: np.ndarray
    rms_error: float
    success: bool


# ===========================================================================
# Single exponential
# ===========================================================================

def _single_exp_model(tau, beta, gamma):
    """g2 - 1 = beta exp(-2 Gamma tau)."""
    return beta * np.exp(-2.0 * gamma * tau)


def fit_single_exponential(
    measurement: DLSMeasurement,
    seed_cumulant: Optional[CumulantResult] = None,
    tau_min_s: Optional[float] = None,
    tau_max_s: Optional[float] = None,
    skip_initial_channels: int = 0,
) -> SingleExponentialResult:
    """Fit g2(tau) - 1 = beta exp(-2 Gamma tau) by nonlinear least squares.

    Initial guesses are taken from a cumulant fit (computed internally if not
    supplied). The single exponential is the trivial monodisperse baseline; for a
    clean monomodal sample its Gamma should match the first cumulant.

    Parameters
    ----------
    measurement : DLSMeasurement
    seed_cumulant : CumulantResult, optional
        A cumulant fit to seed initial guesses. Computed internally if omitted
        (using the same delay window).
    tau_min_s, tau_max_s : float, optional
        Inclusive delay-time window. Default (None, None) uses all points.

    Returns
    -------
    SingleExponentialResult
    """
    tau, g2m1 = _apply_tau_window(
        measurement.delay_times_s, measurement.correlogram,
        tau_min_s, tau_max_s, min_points=2,
        skip_initial_channels=skip_initial_channels,
    )
    if seed_cumulant is None:
        seed_cumulant = fit_cumulants(measurement, order=2,
                                      tau_min_s=tau_min_s, tau_max_s=tau_max_s,
                                      skip_initial_channels=skip_initial_channels)

    q = _measurement_q_m(measurement)
    p0 = [max(seed_cumulant.beta, 1e-6), max(seed_cumulant.gamma_s_inv, 1e-3)]
    bounds = ([0.0, 0.0], [np.inf, np.inf])

    success = True
    try:
        popt, _ = optimize.curve_fit(
            _single_exp_model, tau, g2m1, p0=p0, bounds=bounds, maxfev=20000
        )
        beta, gamma = float(popt[0]), float(popt[1])
    except (RuntimeError, ValueError, TypeError):
        # D2: on failure NaN the physical params (not the seed p0), so a caller that
        # forgets to check `success` can't read the starting cumulant estimate as a
        # fit. NaN propagates to Gamma/D/Rh and the fitted curve/residuals below.
        success = False
        beta, gamma = float('nan'), float('nan')

    fitted = _single_exp_model(tau, beta, gamma)
    residuals = g2m1 - fitted

    d = gamma / q ** 2 if gamma > 0 else float('nan')
    try:
        eta = _require_viscosity(measurement)
        rh = _decay_rate_to_rh_nm(gamma, q, measurement.temperature_K, eta)
    except ValueError:
        rh = float('nan')

    mode = ExponentialMode(amplitude_fraction=1.0, gamma_s_inv=gamma,
                           d_m2_s=d, rh_nm=rh)
    return SingleExponentialResult(
        beta=beta, mode=mode, q_m_inv=q, fit_tau_s=tau,
        fitted_g2m1=fitted, residuals=residuals,
        rms_error=_rms_error(residuals), success=success,
    )


# ===========================================================================
# Double exponential
# ===========================================================================

def _double_exp_model(tau, beta, f1, gamma1, gamma2):
    """g2 - 1 = beta [ f1 exp(-G1 tau) + (1-f1) exp(-G2 tau) ]^2 (full Siegert)."""
    g1 = f1 * np.exp(-gamma1 * tau) + (1.0 - f1) * np.exp(-gamma2 * tau)
    return beta * g1 ** 2


def fit_double_exponential(
    measurement: DLSMeasurement,
    seed_cumulant: Optional[CumulantResult] = None,
    tau_min_s: Optional[float] = None,
    tau_max_s: Optional[float] = None,
    skip_initial_channels: int = 0,
) -> DoubleExponentialResult:
    """Fit two discrete decay modes via the full Siegert relation.

        g2 - 1 = beta [ f1 exp(-G1 tau) + (1-f1) exp(-G2 tau) ]^2

    The squared (rather than additive) form keeps the cross term, which is the
    physically correct Siegert expansion of two modes. Initial guesses follow the
    Brookhaven manual prescription: from the second-order cumulant, split into
    G1 = Gamma + sqrt(mu2) and G2 = Gamma - sqrt(mu2) with equal amplitudes.

    Modes are returned ordered so mode1 is the faster (larger Gamma).

    Parameters
    ----------
    measurement : DLSMeasurement
    seed_cumulant : CumulantResult, optional
    tau_min_s, tau_max_s : float, optional
        Inclusive delay-time window. Default (None, None) uses all points.

    Returns
    -------
    DoubleExponentialResult
    """
    tau, g2m1 = _apply_tau_window(
        measurement.delay_times_s, measurement.correlogram,
        tau_min_s, tau_max_s, min_points=4,
        skip_initial_channels=skip_initial_channels,
    )
    if seed_cumulant is None:
        seed_cumulant = fit_cumulants(measurement, order=2,
                                      tau_min_s=tau_min_s, tau_max_s=tau_max_s,
                                      skip_initial_channels=skip_initial_channels)

    q = _measurement_q_m(measurement)
    gamma0 = max(seed_cumulant.gamma_s_inv, 1e-3)
    spread = math.sqrt(max(seed_cumulant.mu2_s_inv2, 0.0))
    if spread <= 0:
        spread = 0.5 * gamma0   # fall back to a moderate split if monodisperse seed
    g1_guess = gamma0 + spread
    g2_guess = max(gamma0 - spread, gamma0 * 0.1)
    p0 = [max(seed_cumulant.beta, 1e-6), 0.5, g1_guess, g2_guess]
    bounds = ([0.0, 0.0, 0.0, 0.0], [np.inf, 1.0, np.inf, np.inf])

    success = True
    try:
        popt, _ = optimize.curve_fit(
            _double_exp_model, tau, g2m1, p0=p0, bounds=bounds, maxfev=40000
        )
        beta, f1, gamma1, gamma2 = (float(popt[0]), float(popt[1]),
                                    float(popt[2]), float(popt[3]))
    except (RuntimeError, ValueError, TypeError):
        success = False
        beta, f1, gamma1, gamma2 = (float('nan'),) * 4   # D2: NaN, not the seed p0

    fitted = _double_exp_model(tau, beta, f1, gamma1, gamma2)
    residuals = g2m1 - fitted

    try:
        eta = _require_viscosity(measurement)
        have_eta = True
    except ValueError:
        have_eta = False
        eta = None

    def make_mode(frac, gamma):
        d = gamma / q ** 2 if gamma > 0 else float('nan')
        # `gamma > 0` also excludes NaN (a failed fit, D2), so _decay_rate_to_rh_nm
        # is never handed a non-positive rate it would reject.
        rh = (_decay_rate_to_rh_nm(gamma, q, measurement.temperature_K, eta)
              if (have_eta and gamma > 0) else float('nan'))
        return ExponentialMode(amplitude_fraction=frac, gamma_s_inv=gamma,
                               d_m2_s=d, rh_nm=rh)

    # Order so mode1 is the faster (larger Gamma) mode.
    if gamma1 >= gamma2:
        mode1 = make_mode(f1, gamma1)
        mode2 = make_mode(1.0 - f1, gamma2)
    else:
        mode1 = make_mode(1.0 - f1, gamma2)
        mode2 = make_mode(f1, gamma1)

    return DoubleExponentialResult(
        beta=beta, mode1=mode1, mode2=mode2, q_m_inv=q, fit_tau_s=tau,
        fitted_g2m1=fitted, residuals=residuals,
        rms_error=_rms_error(residuals), success=success,
    )


# ===========================================================================
# Kohlrausch-Williams-Watts (stretched exponential)
# ===========================================================================

def _kww_model(tau, beta, tau_c, stretch):
    """g2 - 1 = beta exp(-2 (tau/tau_c)^stretch)."""
    return beta * np.exp(-2.0 * (tau / tau_c) ** stretch)


def fit_kww(
    measurement: DLSMeasurement,
    seed_cumulant: Optional[CumulantResult] = None,
    tau_min_s: Optional[float] = None,
    tau_max_s: Optional[float] = None,
    skip_initial_channels: int = 0,
) -> KWWResult:
    """Fit a Kohlrausch-Williams-Watts (stretched-exponential) model.

        g1(tau) = exp(-(tau/tau_c)^stretch),  so
        g2 - 1  = beta exp(-2 (tau/tau_c)^stretch)

    The stretch exponent (0 < stretch <= 1) captures a continuous spread of
    relaxation times; stretch = 1 recovers a single exponential. The mean
    relaxation time follows from the gamma function:

        <tau> = (tau_c / stretch) * Gamma(1 / stretch)

    Both the characteristic time tau_c and the mean time <tau> are reported, each
    with its corresponding hydrodynamic radius, since the literature uses both.

    Parameters
    ----------
    measurement : DLSMeasurement
    seed_cumulant : CumulantResult, optional
    tau_min_s, tau_max_s : float, optional
        Inclusive delay-time window. Default (None, None) uses all points.

    Returns
    -------
    KWWResult
    """
    tau, g2m1 = _apply_tau_window(
        measurement.delay_times_s, measurement.correlogram,
        tau_min_s, tau_max_s, min_points=3,
        skip_initial_channels=skip_initial_channels,
    )
    if seed_cumulant is None:
        seed_cumulant = fit_cumulants(measurement, order=2,
                                      tau_min_s=tau_min_s, tau_max_s=tau_max_s,
                                      skip_initial_channels=skip_initial_channels)

    q = _measurement_q_m(measurement)
    gamma0 = max(seed_cumulant.gamma_s_inv, 1e-3)
    tau_c0 = 1.0 / gamma0
    p0 = [max(seed_cumulant.beta, 1e-6), tau_c0, 1.0]
    # stretch strictly in (0, 1]; keep a tiny lower bound away from 0.
    bounds = ([0.0, 0.0, 1e-3], [np.inf, np.inf, 1.0])

    success = True
    try:
        popt, _ = optimize.curve_fit(
            _kww_model, tau, g2m1, p0=p0, bounds=bounds, maxfev=40000
        )
        beta, tau_c, stretch = float(popt[0]), float(popt[1]), float(popt[2])
    except (RuntimeError, ValueError, TypeError):
        success = False
        beta, tau_c, stretch = (float('nan'),) * 3       # D2: NaN, not the seed p0

    fitted = _kww_model(tau, beta, tau_c, stretch)
    residuals = g2m1 - fitted

    mean_tau = (tau_c / stretch) * special.gamma(1.0 / stretch)
    gamma_tau_c = 1.0 / tau_c if tau_c > 0 else float('nan')
    gamma_mean = 1.0 / mean_tau if mean_tau > 0 else float('nan')

    try:
        eta = _require_viscosity(measurement)
        rh_tau_c = _decay_rate_to_rh_nm(gamma_tau_c, q, measurement.temperature_K, eta)
        rh_mean = _decay_rate_to_rh_nm(gamma_mean, q, measurement.temperature_K, eta)
    except ValueError:
        rh_tau_c = float('nan')
        rh_mean = float('nan')

    return KWWResult(
        beta=beta, tau_c_s=tau_c, stretch=stretch, mean_tau_s=mean_tau,
        gamma_from_tau_c_s_inv=gamma_tau_c, rh_from_tau_c_nm=rh_tau_c,
        gamma_from_mean_tau_s_inv=gamma_mean, rh_from_mean_tau_nm=rh_mean,
        q_m_inv=q, fit_tau_s=tau, fitted_g2m1=fitted, residuals=residuals,
        rms_error=_rms_error(residuals), success=success,
    )
