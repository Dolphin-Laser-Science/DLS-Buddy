"""
analysis/dls/exponentials.py
============================

Parametric exponential fits of g2(tau) - 1, all carrying the Siegert factor of 2
explicitly (the Siegert relation g2 - 1 = beta |g1|^2; Chu 1991) and reporting
Gamma as the physical g1 decay rate:

  - fit_single_exponential : g2 - 1 = beta exp(-2 Gamma tau)  (monodisperse)
  - fit_double_exponential : two discrete modes via the full Siegert relation
  - fit_kww                : Kohlrausch-Williams-Watts stretched exponential

Each is seeded from a second-order cumulant fit, computed internally on the same
delay window, and returns plot-ready arrays.

The seed is deliberately NOT a parameter. A caller-supplied seed was removed once no
caller was found to pass one, and the obvious reason to add one back
-- reusing a cumulant fit the controller already ran -- would feed these fits a
DIFFERENT starting vector, because that fit uses the user's `cumulant_order` and
`cumulant_method` settings while the seed below is fixed at order 2. On a
nonlinear least-squares fit a different starting vector is a different converged
answer, with nothing to flag it.
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
    _d_from_gamma,
    _decay_completeness,
    _decay_completeness_check,
    MAX_RETAINED_ACF_RATIO,
    _observed_decay_check,
    _lowest_observed_acf_ratio,
    _decay_rate_to_rh_nm,
    _measurement_q_m,
    MIN_DECAY_COMPLETENESS,
    _require_viscosity,
    _result_flags,
    NO_VISCOSITY_NOTE,
    _rms_error,
    _seed_or_floor,
)
from analysis.dls.cumulants import (
    CUMULANT_MAX_RH_NM, CUMULANT_MIN_RH_NM, _size_range_check, fit_cumulants)


@dataclass
class ExponentialMode:
    """One decay mode of an exponential fit.

    Carries its OWN reliability verdict, not the fit's. A two-mode fit routinely pairs a
    healthy fast mode with a slow one that never finished decaying inside the lag window,
    and one flag on the result cannot say that -- it either clears both or condemns both.
    Same shape as `ModePoint`'s `low_amplitude` / `amplitude_floor` pair in
    `analysis/dls/modes.py`: the verdict and the number it was reached from travel together.
    """
    amplitude_fraction: float         # relative g1 amplitude (modes sum to 1)
    gamma_s_inv: float
    d_m2_s: float
    rh_nm: float
    # Why THIS mode's rate/size is unreliable, on the same rh_reliable / reliability_note
    # pair every other DLS result carries, so display and export read one shape.
    rh_reliable: bool = True
    reliability_note: str = ''
    # Gamma * (lag span) -- the relaxation times the measurement covered. The measured
    # quantity the flag above was reached from, recorded so a surface can state it.
    decay_completeness: float = float('nan')
    # WHICH cause the note carries, because the two differ in what they invalidate: an
    # undecayed mode's RATE is an extrapolation, while a missing viscosity leaves the rate
    # exact and only withholds the radius. A consumer that must caveat rate columns cannot
    # tell them apart from the note text without re-deriving the threshold (rule 12's trap),
    # so the fitter states it. Mirrors `ModePoint.decay_incomplete`.
    decay_incomplete: bool = False


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
    # Why a NaN Rh is blank (e.g. no viscosity), on the same rh_reliable /
    # reliability_note pair the cumulant results carry, so every display and
    # export surface reads one shape across the parametric family.
    rh_reliable: bool = True
    reliability_note: str = ''


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
    # Result-level roll-up: False when EITHER mode is unreliable, notes joined (see
    # _result_flags). Per-cell display and export read the per-mode fields; this exists so a
    # surface that has not been taught about modes over-flags rather than missing a flag.
    rh_reliable: bool = True
    reliability_note: str = ''


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
    # KWW has no ExponentialMode to hang a verdict on, so the roll-up IS the flag here --
    # and it is ONE verdict for both Rh channels, since they report one decay two ways. The
    # lag-window question is asked of the DATA (ISO's retained-range rule in its data form),
    # not of tau_c; see fit_kww for why a rate-based test cannot work for this model.
    rh_reliable: bool = True
    reliability_note: str = ''
    # Which cause, for the same reason ExponentialMode carries it: an undecayed correlogram
    # makes tau_c and <tau> themselves extrapolations, a missing viscosity does not.
    decay_incomplete: bool = False


def _tau_span_s(tau) -> float:
    """The fitted lag SPAN, `tau[-1] - tau[0]` -- what `_decay_completeness` measures over.

    The span rather than the last lag, because ISO 22412 A.1.2 states its retained-range
    rule relative to the first retained point and because a user who narrows `tau_min_s` has
    measured a decay across the window, not since zero. On a real log lag axis the two agree
    to ~1e-3; on a narrowed window they do not, and the span is the honest one.
    """
    return float(tau[-1] - tau[0]) if len(tau) >= 2 else float('nan')


def _mode_flags(gamma_s_inv: float, tau_span_s: float, min_decay_completeness: float,
                have_eta: bool, rh_nm: float = float('nan'),
                max_rh_nm: float = CUMULANT_MAX_RH_NM,
                min_rh_nm: float = CUMULANT_MIN_RH_NM) -> tuple:
    """(completeness, rh_reliable, reliability_note, decay_incomplete) for ONE mode.

    The single place the reasons an exponential Rh is unreliable are combined, so the
    three fitters cannot drift apart on it (rule 10) -- the exponential counterpart of
    `cumulants._cumulant_flags`, rolled up by the shared `_result_flags`. Three
    preconditions, ALL emitted when they apply, undecayed first: they are independent
    facts about different things -- this mode's Gamma is an extrapolation past the end
    of the data; its size is outside the range DLS can establish (the technique's ISO
    ceiling, or the artifact floor -- `_size_range_check`, the same two-sided check the
    cumulant applies, since the sedimentation limit is a limit of the technique and an
    arm that omitted it reported a 28 um radius clean beside a flagged cumulant); no
    viscosity was available to turn a Gamma into a radius -- and none implies another.
    An earlier version let the undecayed note REPLACE the viscosity one on the theory
    that it was the stronger statement; that silently withdrew "no viscosity;
    Stokes-Einstein needs it" from a measurement that had both, so a blank Rh stated a
    reason that was not why it was blank.
    """
    completeness = _decay_completeness(gamma_s_inv, tau_span_s)
    decayed_ok, note = _decay_completeness_check(completeness, min_decay_completeness)
    # The size verdict is asked only of a rate the measurement observed: an undecayed
    # mode's Rh is an extrapolation, not a size, so judging it against the technique's
    # range would flag a number for being large when the truth is that it is not a
    # measurement at all -- the one precedence here, because the size verdict DERIVES
    # from the rate the undecayed note already disowns (unlike the viscosity note).
    _size_ok, size_note = (_size_range_check(rh_nm, max_rh_nm, min_rh_nm)
                           if decayed_ok else (True, ''))
    reliable, joined = _result_flags(note, size_note, '' if have_eta else NO_VISCOSITY_NOTE)
    return completeness, reliable, joined, not decayed_ok


# ===========================================================================
# Single exponential
# ===========================================================================

def _single_exp_model(tau, beta, gamma):
    """g2 - 1 = beta exp(-2 Gamma tau)."""
    return beta * np.exp(-2.0 * gamma * tau)


def fit_single_exponential(
    measurement: DLSMeasurement,
    tau_min_s: Optional[float] = None,
    tau_max_s: Optional[float] = None,
    skip_initial_channels: int = 0,
    min_decay_completeness: float = MIN_DECAY_COMPLETENESS,
    max_rh_nm: float = CUMULANT_MAX_RH_NM,
    min_rh_nm: float = CUMULANT_MIN_RH_NM,
) -> SingleExponentialResult:
    """Fit g2(tau) - 1 = beta exp(-2 Gamma tau) by nonlinear least squares.

    Initial guesses are taken from a cumulant fit computed internally on the same
    delay window (see the module docstring on why the seed is not a parameter).
    The single exponential is the trivial monodisperse baseline; for a clean
    monomodal sample its Gamma should match the first cumulant.

    Parameters
    ----------
    measurement : DLSMeasurement
    tau_min_s, tau_max_s : float, optional
        Inclusive delay-time window. Default (None, None) uses all points.
    min_decay_completeness : float
        Gamma * (lag span) below which a mode is FLAGGED (not rejected) as not having
        finished decaying inside the lag window; default MIN_DECAY_COMPLETENESS. Raise it
        to demand a more complete decay, lower it deliberately if you are reading rates
        from a deliberately truncated window -- see _decay_completeness_check.
    max_rh_nm, min_rh_nm : float
        The size range DLS can establish, outside which the mode's Rh is FLAGGED (not
        rejected); defaults CUMULANT_MAX_RH_NM / CUMULANT_MIN_RH_NM, the same two-sided
        `_size_range_check` the cumulant applies.

    Returns
    -------
    SingleExponentialResult
    """
    tau, g2m1 = _apply_tau_window(
        measurement.delay_times_s, measurement.correlogram,
        tau_min_s, tau_max_s, min_points=2,
        skip_initial_channels=skip_initial_channels,
    )
    seed_cumulant = fit_cumulants(measurement, order=2,
                                  tau_min_s=tau_min_s, tau_max_s=tau_max_s,
                                  skip_initial_channels=skip_initial_channels)

    q = _measurement_q_m(measurement)
    p0 = [_seed_or_floor(seed_cumulant.beta, 1e-6), _seed_or_floor(seed_cumulant.gamma_s_inv, 1e-3)]
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

    d = _d_from_gamma(gamma, q)
    visc_note = ''
    try:
        eta = _require_viscosity(measurement)
        rh = _decay_rate_to_rh_nm(gamma, q, measurement.temperature_K, eta)
    except ValueError:
        rh = float('nan')
        visc_note = NO_VISCOSITY_NOTE

    completeness, mode_reliable, mode_note, undecayed = _mode_flags(
        gamma, _tau_span_s(tau), min_decay_completeness, not visc_note,
        rh, max_rh_nm, min_rh_nm)
    mode = ExponentialMode(amplitude_fraction=1.0, gamma_s_inv=gamma,
                           d_m2_s=d, rh_nm=rh,
                           rh_reliable=mode_reliable, reliability_note=mode_note,
                           decay_completeness=completeness, decay_incomplete=undecayed)
    rh_reliable, reliability_note = _result_flags(mode_note)
    return SingleExponentialResult(
        beta=beta, mode=mode, q_m_inv=q, fit_tau_s=tau,
        fitted_g2m1=fitted, residuals=residuals,
        rms_error=_rms_error(residuals), success=success,
        rh_reliable=rh_reliable, reliability_note=reliability_note,
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
    tau_min_s: Optional[float] = None,
    tau_max_s: Optional[float] = None,
    skip_initial_channels: int = 0,
    min_decay_completeness: float = MIN_DECAY_COMPLETENESS,
    max_rh_nm: float = CUMULANT_MAX_RH_NM,
    min_rh_nm: float = CUMULANT_MIN_RH_NM,
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
    tau_min_s, tau_max_s : float, optional
        Inclusive delay-time window. Default (None, None) uses all points.
    min_decay_completeness : float
        Gamma * (lag span) below which a mode is FLAGGED (not rejected) as not having
        finished decaying inside the lag window; default MIN_DECAY_COMPLETENESS. Raise it
        to demand a more complete decay, lower it deliberately if you are reading rates
        from a deliberately truncated window -- see _decay_completeness_check.
    max_rh_nm, min_rh_nm : float
        The size range DLS can establish, outside which a mode's Rh is FLAGGED (not
        rejected), per mode; defaults CUMULANT_MAX_RH_NM / CUMULANT_MIN_RH_NM. A fast
        mode below the floor is the usual signature of afterpulsing absorbed into the
        fit -- drop the leading channels rather than read it as a size.

    Returns
    -------
    DoubleExponentialResult
    """
    tau, g2m1 = _apply_tau_window(
        measurement.delay_times_s, measurement.correlogram,
        tau_min_s, tau_max_s, min_points=4,
        skip_initial_channels=skip_initial_channels,
    )
    seed_cumulant = fit_cumulants(measurement, order=2,
                                  tau_min_s=tau_min_s, tau_max_s=tau_max_s,
                                  skip_initial_channels=skip_initial_channels)

    q = _measurement_q_m(measurement)
    gamma0 = _seed_or_floor(seed_cumulant.gamma_s_inv, 1e-3)
    spread = math.sqrt(_seed_or_floor(seed_cumulant.mu2_s_inv2, 0.0))
    if spread <= 0:
        spread = 0.5 * gamma0   # fall back to a moderate split if monodisperse seed
    g1_guess = gamma0 + spread
    g2_guess = max(gamma0 - spread, gamma0 * 0.1)
    p0 = [_seed_or_floor(seed_cumulant.beta, 1e-6), 0.5, g1_guess, g2_guess]
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

    tau_span = _tau_span_s(tau)

    def make_mode(frac, gamma):
        # `gamma > 0` excludes NaN (a failed fit, D2) but ADMITS +inf, which would give a
        # finite, plausible Rh = 0.0 nm out of Stokes-Einstein -- hence isfinite too
        # (2026-07-31 spec-01). _decay_rate_to_rh_nm now guards this itself; the gate here
        # keeps `d` and `rh` withholding together.
        ok = math.isfinite(gamma) and gamma > 0
        d = _d_from_gamma(gamma, q)
        rh = (_decay_rate_to_rh_nm(gamma, q, measurement.temperature_K, eta)
              if (have_eta and ok) else float('nan'))
        # Per MODE, not per fit: this fit's whole reason to exist is that its two modes can
        # differ, and on real data the common case is a measured fast mode beside a slow one
        # that never decayed. A single flag here would clear both or condemn both.
        completeness, reliable, note, undecayed = _mode_flags(
            gamma, tau_span, min_decay_completeness, have_eta, rh, max_rh_nm, min_rh_nm)
        return ExponentialMode(amplitude_fraction=frac, gamma_s_inv=gamma,
                               d_m2_s=d, rh_nm=rh,
                               rh_reliable=reliable, reliability_note=note,
                               decay_completeness=completeness,
                               decay_incomplete=undecayed)

    # Order so mode1 is the faster (larger Gamma) mode.
    if gamma1 >= gamma2:
        mode1 = make_mode(f1, gamma1)
        mode2 = make_mode(1.0 - f1, gamma2)
    else:
        mode1 = make_mode(1.0 - f1, gamma2)
        mode2 = make_mode(f1, gamma1)

    rh_reliable, reliability_note = _result_flags(mode1.reliability_note,
                                                  mode2.reliability_note)
    return DoubleExponentialResult(
        beta=beta, mode1=mode1, mode2=mode2, q_m_inv=q, fit_tau_s=tau,
        fitted_g2m1=fitted, residuals=residuals,
        rms_error=_rms_error(residuals), success=success,
        rh_reliable=rh_reliable, reliability_note=reliability_note,
    )


# ===========================================================================
# Kohlrausch-Williams-Watts (stretched exponential)
# ===========================================================================

def _kww_model(tau, beta, tau_c, stretch):
    """g2 - 1 = beta exp(-2 (tau/tau_c)^stretch)."""
    return beta * np.exp(-2.0 * (tau / tau_c) ** stretch)


def fit_kww(
    measurement: DLSMeasurement,
    tau_min_s: Optional[float] = None,
    tau_max_s: Optional[float] = None,
    skip_initial_channels: int = 0,
    max_retained_acf_ratio: float = MAX_RETAINED_ACF_RATIO,
    max_rh_nm: float = CUMULANT_MAX_RH_NM,
    min_rh_nm: float = CUMULANT_MIN_RH_NM,
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
    tau_min_s, tau_max_s : float, optional
        Inclusive delay-time window. Default (None, None) uses all points.
    max_retained_acf_ratio : float
        Largest normalized-ACF value reached ANYWHERE in the fitted window -- the minimum
        over it, ISO 22412 A.1.2's "at least one value smaller than" -- that still counts as
        a decay the measurement observed; default MAX_RETAINED_ACF_RATIO, which is ISO's
        1/50 (Xu 2000 p. 241 gives an independent, stricter 0.005 that selects the same runs
        on the committed data). Above it the fit is FLAGGED, not rejected -- see
        _observed_decay_check. Unlike the exponential arms this is a data-level test, not a
        rate test, because a stretched exponential hides an incomplete decay in its stretch
        exponent rather than in tau_c.
    max_rh_nm, min_rh_nm : float
        The size range DLS can establish, outside which a reported Rh is FLAGGED (not
        rejected); defaults CUMULANT_MAX_RH_NM / CUMULANT_MIN_RH_NM. Both Rh channels are
        checked and the one verdict covers both, as for the lag-window criterion.

    Returns
    -------
    KWWResult
    """
    tau, g2m1 = _apply_tau_window(
        measurement.delay_times_s, measurement.correlogram,
        tau_min_s, tau_max_s, min_points=3,
        skip_initial_channels=skip_initial_channels,
    )
    seed_cumulant = fit_cumulants(measurement, order=2,
                                  tau_min_s=tau_min_s, tau_max_s=tau_max_s,
                                  skip_initial_channels=skip_initial_channels)

    q = _measurement_q_m(measurement)
    gamma0 = _seed_or_floor(seed_cumulant.gamma_s_inv, 1e-3)
    tau_c0 = 1.0 / gamma0
    p0 = [_seed_or_floor(seed_cumulant.beta, 1e-6), tau_c0, 1.0]
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

    _mean_tau_raw = (tau_c / stretch) * special.gamma(1.0 / stretch)
    # A small `stretch` (the fit's own lower bound reaches 1e-3) drives 1/stretch past
    # ~171.6, where special.gamma saturates to +inf -- gate the reported field the same
    # way its own derived channels below already gate on it, rather than storing +inf.
    mean_tau = _mean_tau_raw if math.isfinite(_mean_tau_raw) else float('nan')
    # isfinite as well as > 0, for the same reason as the sibling gates: a non-finite
    # tau_c/mean_tau must yield NaN, not the 0.0 that 1/inf would hand downstream as a
    # legitimate-looking rate (2026-07-31 spec-01).
    gamma_tau_c = 1.0 / tau_c if (math.isfinite(tau_c) and tau_c > 0) else float('nan')
    gamma_mean = (1.0 / mean_tau if (math.isfinite(mean_tau) and mean_tau > 0)
                  else float('nan'))

    visc_note = ''
    try:
        eta = _require_viscosity(measurement)
        rh_tau_c = _decay_rate_to_rh_nm(gamma_tau_c, q, measurement.temperature_K, eta)
        rh_mean = _decay_rate_to_rh_nm(gamma_mean, q, measurement.temperature_K, eta)
    except ValueError:
        rh_tau_c = float('nan')
        rh_mean = float('nan')
        visc_note = NO_VISCOSITY_NOTE

    # ONE verdict, read off the DATA rather than off tau_c -- and shared by both Rh
    # channels, because a KWW has one decay and the two channels are two ways of reporting
    # it, not two decays. A rate-based test cannot work here: a stretched exponential
    # absorbs an undecayed mode into the STRETCH exponent while tau_c stays short (measured
    # on sample 20: s falls to 0.21-0.68, and Gamma*delta_tau on 1/tau_c clears 4 of 4
    # correlograms that plainly never decayed). ISO's retained-range rule, in its data form,
    # asks the MEASURED `g2m1`, never `fitted + residuals` -- algebraically the same, but
    # NaN on a curve_fit failure, silencing the criterion exactly when the fit is worst.
    decay_ratio = _lowest_observed_acf_ratio(g2m1)
    decay_ok, decay_note = _observed_decay_check(decay_ratio, max_retained_acf_ratio)
    # The size range is asked of BOTH channels and the verdict is one, like the decay
    # verdict: two readings of one decay. `_result_flags` collapses a note both share.
    # Asked only of a decay the measurement observed -- the precedence the cumulant and
    # single/double-exponential arms apply: the size of an unobserved decay is not a size.
    if decay_ok:
        _ok_c, size_note_c = _size_range_check(rh_tau_c, max_rh_nm, min_rh_nm)
        _ok_m, size_note_m = _size_range_check(rh_mean, max_rh_nm, min_rh_nm)
    else:
        size_note_c = size_note_m = ''
    rh_reliable, reliability_note = _result_flags(
        decay_note, size_note_c, size_note_m, visc_note)

    return KWWResult(
        beta=beta, tau_c_s=tau_c, stretch=stretch, mean_tau_s=mean_tau,
        gamma_from_tau_c_s_inv=gamma_tau_c, rh_from_tau_c_nm=rh_tau_c,
        gamma_from_mean_tau_s_inv=gamma_mean, rh_from_mean_tau_nm=rh_mean,
        q_m_inv=q, fit_tau_s=tau, fitted_g2m1=fitted, residuals=residuals,
        rms_error=_rms_error(residuals), success=success,
        rh_reliable=rh_reliable, reliability_note=reliability_note,
        decay_incomplete=not decay_ok,
    )
