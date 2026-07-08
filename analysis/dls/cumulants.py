"""
analysis/dls/cumulants.py
=========================

Cumulant-expansion analysis of g2(tau) - 1 (Koppel 1972; Frisken 2001).

Two fitting methods share one ``CumulantResult``:

  - 'linear'    (Koppel 1972): weighted log-polynomial fit of ln(g2 - 1) over the
                decay region; closed-form, zero-baseline, uses the amplitude cutoff.
  - 'nonlinear' (Frisken 2001): direct g2-1 fit by nonlinear least squares with a
                FLOATING baseline (moments-about-the-mean form); robust to baseline
                drift and noisy/low-count data. This is the user-facing default
                (seeded from Settings by the controller); the engine default stays
                'linear' so direct callers and existing tests are unchanged.

In the Siegert convention
    ln(g2 - 1) = ln(beta) - 2 Gamma tau + mu2 tau^2 - (mu3/3) tau^3 + ...
so the polynomial coefficients map directly to the cumulants. PDI = mu2 / Gamma^2;
the expansion is considered reliable only for PDI <= CUMULANT_PDI_VALIDITY_LIMIT.

Per invariant 8 neither method reports a +/- from a single correlogram (its lag
channels are correlated, Schaetzel 1990); replicate averaging is the uncertainty
source.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import optimize

from core.data_models import DLSMeasurement
from analysis.dls._common import (
    _apply_tau_window,
    _decay_rate_to_rh_nm,
    _measurement_q_m,
    _require_viscosity,
    _rms_error,
)


# The cumulant expansion is a small-tau (low-polydispersity) approximation; it is
# generally considered unreliable above this polydispersity index.
CUMULANT_PDI_VALIDITY_LIMIT = 0.3


@dataclass
class CumulantResult:
    """Cumulant-expansion fit of g2(tau) - 1."""
    order: int
    beta: float                       # coherence factor (intercept), exp(c0)
    gamma_s_inv: float                # first cumulant = mean g1 decay rate
    mu2_s_inv2: float                 # second cumulant (variance of Gamma)
    mu3_s_inv3: Optional[float]       # third cumulant (order 3 only), else None
    pdi: float                        # mu2 / gamma^2
    pdi_valid: bool                   # pdi <= CUMULANT_PDI_VALIDITY_LIMIT
    # derived size (intensity-weighted, apparent at this q)
    d_m2_s: float
    rh_nm: float
    q_m_inv: float
    # fit bookkeeping
    fit_cutoff: float                 # points kept where g2m1 > cutoff*intercept
    n_points_used: int
    coefficients: np.ndarray          # polynomial coeffs [c0, c1, c2, ...]
    fit_tau_s: np.ndarray             # the tau values actually fitted
    fitted_g2m1: np.ndarray           # model g2-1 at fit_tau_s
    residuals: np.ndarray             # data - model over fit_tau_s
    rms_error: float                  # RMS residual over the cutoff-masked (high-
    #                                   amplitude, short-lag) support — the SAME for
    #                                   'linear' and 'nonlinear', so it is comparable
    #                                   across methods (may differ from RMS of the
    #                                   full `residuals` array on the nonlinear path)
    n_skipped: int = 0                # leading channels dropped (skip_initial_channels)
    method: str = 'linear'            # 'linear' (Koppel 1972) or 'nonlinear' (Frisken 2001)
    baseline: float = 0.0             # fitted floating baseline B (nonlinear); 0 for linear
    success: bool = True              # nonlinear convergence; True for linear (always) and
                                      # for a nonlinear fit that fell back to linear -> False


def fit_cumulants(
    measurement: DLSMeasurement,
    order: int = 2,
    fit_cutoff: float = 0.1,
    tau_min_s: Optional[float] = None,
    tau_max_s: Optional[float] = None,
    skip_initial_channels: int = 0,
    method: str = 'linear',
) -> CumulantResult:
    """Fit g2(tau) - 1 by the method of cumulants.

    Two fitting methods, selected by `method`:

    - 'linear' (Koppel 1972, the engine default): fit a weighted polynomial to
      ln(g2 - 1) over the decay region (overflow-free, closed-form). Assumes a
      zero baseline and uses the amplitude cutoff.
    - 'nonlinear' (Frisken 2001): fit g2 - 1 directly by nonlinear least squares
      with a FLOATING baseline (moments-about-the-mean form), using the whole
      windowed correlogram (no amplitude cutoff). More robust to baseline drift
      and noisy/low-count data. (See the module docstring for why the engine
      default is 'linear' while the GUI seeds 'nonlinear'.)

    Both return the same CumulantResult. Per invariant 8 neither reports a +/-
    from a single correlogram (lag channels are correlated, Schaetzel 1990);
    replicate averaging is the uncertainty source.

    Parameters
    ----------
    measurement : DLSMeasurement
    order : int
        Cumulant order: 1, 2 (default), or 3.
    fit_cutoff : float
        Linear method only: keep points where (g2-1) > fit_cutoff * intercept
        (default 0.1). Ignored by the nonlinear method.
    tau_min_s, tau_max_s : float, optional
        Inclusive delay-time window (applied first). Default uses all points.
    skip_initial_channels : int
        Drop the first N channels by index before windowing (default 0); see
        _apply_tau_window. Composes with tau_min_s as an intersection.
    method : str
        'linear' (default) or 'nonlinear'.

    Returns
    -------
    CumulantResult

    Raises
    ------
    ValueError
        If order is not 1/2/3, method is unknown, or too few points survive.
    """
    if order not in (1, 2, 3):
        raise ValueError(f"order must be 1, 2, or 3, got {order!r}.")
    if method == 'nonlinear':
        return _fit_cumulants_nonlinear(
            measurement, order, fit_cutoff, tau_min_s, tau_max_s, skip_initial_channels)
    if method == 'linear':
        return _fit_cumulants_linear(
            measurement, order, fit_cutoff, tau_min_s, tau_max_s, skip_initial_channels)
    raise ValueError(f"method must be 'linear' or 'nonlinear', got {method!r}.")


def _fit_cumulants_linear(
    measurement: DLSMeasurement,
    order: int,
    fit_cutoff: float,
    tau_min_s: Optional[float],
    tau_max_s: Optional[float],
    skip_initial_channels: int,
) -> CumulantResult:
    """Linear/log cumulant (Koppel 1972): weighted polynomial fit of ln(g2-1).

    In the Siegert convention
        ln(g2 - 1) = ln(beta) - 2 Gamma tau + mu2 tau^2 - (mu3/3) tau^3 + ...
    so the polynomial coefficients map directly to the cumulants. Points are
    weighted by g2 - 1, and only points above `fit_cutoff` times the intercept
    are used (the noisy baseline tail breaks the logarithm).
    """
    tau, g2m1 = _apply_tau_window(
        measurement.delay_times_s, measurement.correlogram,
        tau_min_s, tau_max_s, min_points=order + 1,
        skip_initial_channels=skip_initial_channels,
    )

    intercept0 = float(g2m1.max())
    mask = (g2m1 > fit_cutoff * intercept0) & (g2m1 > 0)
    t = tau[mask]
    y = g2m1[mask]
    if t.size < order + 1:
        raise ValueError(
            f"Only {t.size} points survive the cutoff; need at least "
            f"{order + 1} for an order-{order} fit. Try a smaller fit_cutoff."
        )

    logy = np.log(y)
    # np.polyfit weights multiply the residuals; weight by signal amplitude.
    coeffs_high_first = np.polyfit(t, logy, order, w=y)
    c = coeffs_high_first[::-1]   # [c0, c1, c2, (c3)]

    beta = math.exp(c[0])
    gamma = -c[1] / 2.0
    mu2 = c[2] if order >= 2 else 0.0
    mu3 = (-3.0 * c[3]) if order >= 3 else None
    pdi = mu2 / gamma ** 2 if gamma != 0 else float('nan')

    q = _measurement_q_m(measurement)
    d = gamma / q ** 2 if gamma > 0 else float('nan')
    try:
        eta = _require_viscosity(measurement)
        rh = _decay_rate_to_rh_nm(gamma, q, measurement.temperature_K, eta)
    except ValueError:
        rh = float('nan')

    # Model evaluated over the fitted region, for plotting and residuals. The fitted
    # region IS the cutoff-masked (high-amplitude, short-lag) subset — so rms_error
    # below is over that support, matching the nonlinear path's rms (D3: comparable
    # goodness-of-fit across methods).
    model_log = np.polyval(coeffs_high_first, t)
    fitted = np.exp(model_log)
    residuals = y - fitted

    return CumulantResult(
        order=order,
        beta=beta,
        gamma_s_inv=gamma,
        mu2_s_inv2=mu2,
        mu3_s_inv3=mu3,
        pdi=pdi,
        pdi_valid=bool(pdi <= CUMULANT_PDI_VALIDITY_LIMIT) if math.isfinite(pdi) else False,
        d_m2_s=d,
        rh_nm=rh,
        q_m_inv=q,
        fit_cutoff=fit_cutoff,
        n_points_used=int(t.size),
        coefficients=c,
        fit_tau_s=t,
        fitted_g2m1=fitted,
        residuals=residuals,
        rms_error=_rms_error(residuals),
        n_skipped=int(skip_initial_channels or 0),
        method='linear',
        baseline=0.0,
        success=True,
    )


def _frisken_model(tau, B, beta, gamma, mu2, mu3):
    """Frisken (2001) Eq. 23 cumulant model (moments about the mean):

        g2-1 = B + beta * [ exp(-gamma tau) (1 + (mu2/2) tau^2 - (mu3/6) tau^3) ]^2

    The square is the Siegert relation; expanding about the mean rate (rather than
    about tau=0 as the log-polynomial does) keeps the fit stable at large tau.
    """
    inner = np.exp(-gamma * tau) * (1.0 + 0.5 * mu2 * tau ** 2 - (mu3 / 6.0) * tau ** 3)
    return B + beta * inner * inner


def _fit_cumulants_nonlinear(
    measurement: DLSMeasurement,
    order: int,
    fit_cutoff: float,
    tau_min_s: Optional[float],
    tau_max_s: Optional[float],
    skip_initial_channels: int,
) -> CumulantResult:
    """Nonlinear cumulant (Frisken 2001): direct g2-1 fit with a floating baseline.

    Fits the whole windowed correlogram by weighted nonlinear least squares
    (weights proportional to signal amplitude, matching the linear method). The
    floating baseline B and coherence beta are free parameters. On non-convergence
    it falls back to the linear result flagged success=False, so the caller always
    gets a usable number. No SE is reported from one correlogram (invariant 8).
    """
    tau, g2m1 = _apply_tau_window(
        measurement.delay_times_s, measurement.correlogram,
        tau_min_s, tau_max_s, min_points=order + 3,
        skip_initial_channels=skip_initial_channels,
    )
    q = _measurement_q_m(measurement)

    # Seed beta/gamma/mu2 from a quick linear fit on the same window; baseline
    # from the long-delay tail (nominally ~0 after the parser's subtraction).
    try:
        seed = _fit_cumulants_linear(
            measurement, order, fit_cutoff, tau_min_s, tau_max_s, skip_initial_channels)
        beta0 = max(seed.beta, 1e-6)
        gamma0 = max(seed.gamma_s_inv, 1e-3)
        mu2_0 = max(seed.mu2_s_inv2, 0.0)
        mu3_0 = seed.mu3_s_inv3 or 0.0
    except Exception:
        beta0, gamma0, mu2_0, mu3_0 = float(g2m1.max()), 1.0, 0.0, 0.0
    tail = g2m1[-max(2, g2m1.size // 4):]
    B0 = float(np.mean(tail))
    # The baseline is small drift around the (already ~0) long-time value, NOT a
    # free amplitude sink: bound it near the tail so the fit cannot absorb the
    # decay into B and chase a short-lag artefact (which would give a spurious
    # Rh ~ 0).
    dB = max(5.0 * float(np.std(tail)), 0.05)
    B_lo, B_hi = B0 - dB, B0 + dB

    # weights proportional to signal amplitude (down-weight the noisy tail)
    w = np.clip(g2m1 - B0, 1e-6, None)
    sigma = 1.0 / np.sqrt(w)

    if order == 1:
        f = lambda t, B, be, ga: _frisken_model(t, B, be, ga, 0.0, 0.0)
        p0 = [B0, beta0, gamma0]
        lo = [B_lo, 1e-12, 1e-9]
        hi = [B_hi, np.inf, np.inf]
    elif order == 2:
        f = lambda t, B, be, ga, m2: _frisken_model(t, B, be, ga, m2, 0.0)
        p0 = [B0, beta0, gamma0, mu2_0]
        lo = [B_lo, 1e-12, 1e-9, 0.0]
        hi = [B_hi, np.inf, np.inf, np.inf]
    else:
        f = _frisken_model
        p0 = [B0, beta0, gamma0, mu2_0, mu3_0]
        lo = [B_lo, 1e-12, 1e-9, 0.0, -np.inf]
        hi = [B_hi, np.inf, np.inf, np.inf, np.inf]

    ok = True
    B = beta = gamma = mu2 = 0.0
    mu3 = None
    rh = d = float('nan')
    try:
        popt, _ = optimize.curve_fit(
            f, tau, g2m1, p0=p0, sigma=sigma, absolute_sigma=False,
            bounds=(lo, hi), maxfev=10000)
    except (RuntimeError, ValueError, TypeError):
        ok = False
    if ok:
        B = float(popt[0]); beta = float(popt[1]); gamma = float(popt[2])
        mu2 = float(popt[3]) if order >= 2 else 0.0
        mu3 = float(popt[4]) if order >= 3 else None
        d = gamma / q ** 2 if gamma > 0 else float('nan')
        try:
            eta = _require_viscosity(measurement)
            rh = _decay_rate_to_rh_nm(gamma, q, measurement.temperature_K, eta)
        except ValueError:
            rh = float('nan')
        # Reject degenerate solutions: non-finite, collapsed coherence, or an
        # unphysically tiny Rh (the fit chased a short-lag artefact into Gamma).
        if not (math.isfinite(gamma) and gamma > 0 and math.isfinite(beta) and beta > 0):
            ok = False
        elif math.isfinite(rh) and rh < 0.2:
            ok = False
    if not ok:
        # D2 (owner decision 2026-07-07): a failed nonlinear fit returns NaN physical
        # outputs, NOT a linear-fit substitute — so a caller can't read a fit-like
        # number from a failed fit. (Previously it fell back to the linear fit flagged
        # success=False; that labelled fallback was dropped in favour of an honest NaN.)
        nan = float('nan')
        return CumulantResult(
            order=order, beta=nan, gamma_s_inv=nan, mu2_s_inv2=nan, mu3_s_inv3=None,
            pdi=nan, pdi_valid=False, d_m2_s=nan, rh_nm=nan, q_m_inv=q,
            fit_cutoff=fit_cutoff, n_points_used=int(tau.size),
            coefficients=np.full(order + 1, nan),
            fit_tau_s=tau, fitted_g2m1=np.full(tau.shape, nan),
            residuals=np.full(tau.shape, nan), rms_error=nan,
            n_skipped=int(skip_initial_channels or 0),
            method='nonlinear', baseline=nan, success=False)

    pdi = mu2 / gamma ** 2 if gamma != 0 else float('nan')
    fitted = f(tau, *popt)
    residuals = g2m1 - fitted
    # D3: report rms_error over the SAME cutoff-masked (high-amplitude, short-lag)
    # support the linear path uses, so the two methods' rms are directly comparable.
    # (The full window is NOT a valid shared support: the linear method's log-poly
    # model diverges at long lag for order>=2, so its rms can only be defined on the
    # short-lag region — the nonlinear rms is restricted to match. The `residuals`
    # field stays the full-window array for plotting.)
    intercept0 = float(g2m1.max())
    rms_mask = (g2m1 > fit_cutoff * intercept0) & (g2m1 > 0)
    rms = _rms_error(residuals[rms_mask]) if rms_mask.any() else _rms_error(residuals)
    # coefficients parity with the linear path: [c0=ln beta, c1=-2 gamma, c2=mu2, ...]
    coeffs = [math.log(beta), -2.0 * gamma]
    if order >= 2:
        coeffs.append(mu2)
    if order >= 3:
        coeffs.append(-(mu3 or 0.0) / 3.0)

    return CumulantResult(
        order=order,
        beta=beta,
        gamma_s_inv=gamma,
        mu2_s_inv2=mu2,
        mu3_s_inv3=mu3,
        pdi=pdi,
        pdi_valid=bool(pdi <= CUMULANT_PDI_VALIDITY_LIMIT) if math.isfinite(pdi) else False,
        d_m2_s=d,
        rh_nm=rh,
        q_m_inv=q,
        fit_cutoff=fit_cutoff,
        n_points_used=int(tau.size),
        coefficients=np.asarray(coeffs, dtype=float),
        fit_tau_s=tau,
        fitted_g2m1=fitted,
        residuals=residuals,
        rms_error=rms,
        n_skipped=int(skip_initial_channels or 0),
        method='nonlinear',
        baseline=B,
        success=True,
    )
