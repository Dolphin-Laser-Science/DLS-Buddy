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
from analysis import uncertainty as unc
from analysis.dls._common import (
    _apply_tau_window,
    _d_from_gamma,
    _decay_rate_to_rh_nm,
    _lowest_observed_acf_ratio,
    MAX_RETAINED_ACF_RATIO,
    _measurement_q_m,
    _observed_decay_check,
    _require_viscosity,
    _result_flags,
    NO_VISCOSITY_NOTE,
    _rms_error,
    _seed_or_floor,
)


# The cumulant expansion is a small-tau (low-polydispersity) approximation; it is
# generally considered unreliable above this polydispersity index.
CUMULANT_PDI_VALIDITY_LIMIT = 0.3

# Minimum physically meaningful hydrodynamic radius (nm) for a DLS fit: below it the fit
# has chased a short-lag artifact into Gamma rather than measured a particle (a bare
# solvent molecule is ~0.1 nm). The NONLINEAR cumulant rejects such a solution outright
# (owner decision D2 -- the free floating-baseline fit is the one that chases artifacts,
# and no measurement stands behind the number); every other arm FLAGS the value through
# `_size_range_check`'s floor (invariant 4, flag-not-refuse). A documented module
# constant per invariant 3, mirroring CUMULANT_PDI_VALIDITY_LIMIT.
CUMULANT_MIN_RH_NM = 0.2

# Maximum hydrodynamic RADIUS (nm) a cumulant fit reports without a reliability flag, the
# ceiling pairing with the floor above. ISO 22412:2017 sec. 5, p. 4: "Sedimentation of
# particles, dependent on their density, sets an upper limit to the particle size that can
# be assessed by the technique; typically, the upper limit is much less than 10 um." A fit
# returning more has extrapolated a decay that never happened inside the lag window. The
# standard's quantity is a DIAMETER (its symbol table, p. 3, defines x as the "hydrodynamic
# diameter of a particle") and we report a RADIUS, so 10 um becomes 5 um = 5e3 nm here --
# which is why this is not 1e4. Exceeding it does not invalidate the fit.
CUMULANT_MAX_RH_NM = 5.0e3


def _size_range_check(rh_nm: float, max_rh_nm: float,
                      min_rh_nm: float = 0.0) -> tuple[bool, str]:
    """(rh_reliable, reliability_note) for a fitted Rh against the technique's range.

    TWO-SIDED (rule 12's two-sided clause), and FLAGS on both tails. Above the ceiling
    there may well be a real, very large scatterer, and refusing to show the user a
    number they legitimately measured would be the wrong call; below the floor the fit
    has chased a short-lag artifact, and the user needs to be told that the number is
    not a particle size and what removes the artifact. Either way the value is reported
    carrying its flag -- invariant 4's flag-not-refuse rule. The one deliberate
    exception is the NONLINEAR cumulant, which rejects a sub-floor solution outright
    before this check runs (owner decision D2; see CUMULANT_MIN_RH_NM) -- so the floor
    here is what the linear cumulant and the exponential arms apply, and it never fires
    on the nonlinear path. The default floor of 0.0 keeps a caller that asks only the
    ceiling question unchanged.

    Shared by every DLS arm so they cannot drift apart (rule 10): both cumulant paths,
    the single/double-exponential modes and the KWW channels all read this one shape.
    The ISO sedimentation ceiling is a limit of the TECHNIQUE, not of the cumulant
    method (ISO 22412:2017 sec. 5 speaks of what "can be assessed by the technique"), so
    an arm that omitted it reported a 28 um radius as reliable where its cumulant
    neighbour flagged the same number.

    A non-finite Rh is left unflagged HERE: `success` carries the failed-fit case, and
    the one non-finite cause that deserves its own words — a missing viscosity — is
    flagged by the caller (NO_VISCOSITY_NOTE), after this check, so the range note and
    the viscosity note cannot both claim the field.
    """
    if not math.isfinite(rh_nm):
        return True, ''
    if rh_nm > max_rh_nm:
        return False, (
            f'Rh = {rh_nm:.3g} nm is above the ~{max_rh_nm:.3g} nm radius '
            f'({2 * max_rh_nm:.3g} nm diameter) upper limit of DLS '
            f'(ISO 22412:2017 sec. 5), beyond which sedimentation dominates Brownian '
            f'motion. A size this large usually means the correlogram did not decay '
            f'within the lag window -- check for a static or aggregated sample, and '
            f'measure a longer lag range before reading this as a particle size.'
        )
    if rh_nm < min_rh_nm:
        return False, (
            f'Rh = {rh_nm:.3g} nm is below the {min_rh_nm:.3g} nm floor of a physically '
            f'meaningful hydrodynamic radius (a bare solvent molecule is ~0.1 nm): the '
            f'fit has chased a short-lag artifact -- detector afterpulsing or correlator '
            f'dead time in the first channels -- into the decay rate rather than measured '
            f'a particle. Drop the leading channels (skip initial channels) and refit.'
        )
    return True, ''


def _pdi_check(pdi: float) -> tuple[bool, str]:
    """(rh_reliable, reliability_note) for the cumulant expansion's polydispersity
    precondition -- the "cumulant PDI > 0.3 -> unreliable" rule (CLAUDE.md, physics that
    must be enforced in code), given its note so the flag names the precondition
    (invariant 4 clause 1) and travels with the size wherever `rh_reliable` does.

    Exactly the rule as stated -- ABOVE the limit -- and nothing wider; this reads the
    PDI alone. The PDI cell's own verdict (`pdi_valid`) also rejects an unmeasured PDI
    (order-1 fit, NaN: not a failed precondition, an order-1 Rh is perfectly good), a
    negative one (mu2 < 0, which a monodisperse sample's fit returns as a numerical ~0
    with either sign -- measured -1e-15 on a clean synthetic exponential, so folding it
    here flagged healthy sizes), and a non-positive Gamma (which carries its own,
    stronger note). Those stay that cell's verdict; only the polydispersity limit is a
    precondition of the SIZE.
    """
    if math.isfinite(pdi) and pdi > CUMULANT_PDI_VALIDITY_LIMIT:
        return False, (
            f'PDI = {pdi:.3g} is above the {CUMULANT_PDI_VALIDITY_LIMIT:.3g} validity '
            f'limit of the cumulant expansion, a low-polydispersity approximation, so '
            f'this z-average size is unreliable for a sample this polydisperse -- prefer '
            f'a distribution method (NNLS / CONTIN).'
        )
    return True, ''


def _nonpositive_gamma_check(gamma_s_inv: float) -> tuple[bool, str]:
    """(rh_reliable, reliability_note) for a CONVERGED fit whose decay rate is not
    positive -- a finite Gamma <= 0, which the linear (Koppel) path returns whenever the
    log-amplitude rises with lag (a weak scatterer on a drifting or over-subtracted
    baseline, a blank run). The nonlinear path rejects the same solution outright
    (owner decision D2); here the closed-form coefficients are a legitimate outcome of
    the data, so the fit stays `success=True` and the RESULT is flagged: `_d_from_gamma`
    already withholds D and Rh from such a Gamma, and without this the withheld cells
    travelled with a clean flag and an empty note. A non-finite Gamma is a failed fit
    and is `success`'s to carry, so it is left unflagged here.
    """
    if not math.isfinite(gamma_s_inv) or gamma_s_inv > 0:
        return True, ''
    return False, (
        f'The fitted decay rate is not positive (Gamma = {gamma_s_inv:.3g} 1/s): the '
        f'correlogram\'s amplitude does not fall with lag, so no diffusion coefficient or '
        f'size can be formed from this fit. Usually a weak scatterer on a noisy or '
        f'drifting baseline -- check the correlogram and the delay window.'
    )


def _cumulant_flags(rh_nm: float, gamma_s_inv: float, g2m1_windowed: np.ndarray,
                    pdi: float, pdi_valid: bool, visc_note: str,
                    max_rh_nm: float, min_rh_nm: float,
                    max_retained_acf_ratio: float) -> tuple[bool, str, bool]:
    """(rh_reliable, reliability_note, decay_incomplete) for one cumulant fit.

    The single place the cumulant's preconditions are combined, shared by both fit
    paths so they cannot drift apart (rule 10) -- the cumulant counterpart of
    `exponentials._mode_flags`. Five verdicts, each from its own predicate, rolled up by
    `_result_flags` (disjunction; every live note kept, in this order):

      1. a non-positive decay rate (`_nonpositive_gamma_check`) -- the strongest, since
         there is no rate to judge anything else by;
      2. an undecayed lag window (`_observed_decay_check`) -- the ISO 22412:2017 Annex
         A.1.2 retained-range criterion in its DATA form, see below;
      3. the technique's size range, both tails (`_size_range_check`) -- asked only
         when the decay was observed, since the size of an undecayed decay is not one;
      4. the expansion's polydispersity limit (`_pdi_check`);
      5. a missing viscosity (NO_VISCOSITY_NOTE, from the caller).

    WHY THE DATA FORM, NOT THE RATE FORM, FOR THE CUMULANT (canonical-method anchor,
    CLAUDE.md). ISO states its criterion FOR the cumulant fit's retained range -- "at
    least one value smaller than [G2(tau_1) - A]/50" -- which is exactly the data-level
    question `_lowest_observed_acf_ratio` asks, so this is the standard's own form for
    this arm, not a rewrite of it. The rate form Gamma * delta_tau (the exponential
    arms' predicate, the same constant) was measured and rejected for the cumulant: on
    the committed aggregating sample (DLS-SLS-Pairs sample 20, 8 angles x 2 methods) the
    cumulant's MEAN rate is dominated by the fast component, so Gamma * delta_tau read
    21-540 relaxation times "covered" on every cell while the correlogram plainly never
    reached 1/50 of its initial amplitude -- the rate form flagged 0 of 16 cells, the
    data form 16 of 16, and 0 of the 364 healthy cells (2026-08-26). The same reason the
    KWW arm asks the data: a model that absorbs an undecayed component into a parameter
    other than its rate (here mu2 and the truncated expansion; for Frisken also the
    floating baseline, which pins to a tail that is still signal) cannot be judged by
    that rate. The two forms agree on a monodisperse exponential, where the question is
    the same either way.

    The question is asked of the WINDOWED correlogram (after `_apply_tau_window`, before
    the linear path's amplitude cutoff) -- a deliberate deviation from the Annex's letter
    (the canonical-method anchor's class 2, benchmarked): ISO prescribes the criterion on
    the RETAINED range of the cumulant fit, and this program's linear path retains only
    the points above `fit_cutoff` (0.1 of the intercept), a range that by construction
    can never reach the 1/50 mark, so asked of the retained points the criterion could
    never pass. Asked of the window the fit was given, it measures what the Annex means
    to measure -- did the measurement see the decay complete -- and the measurement on
    the committed corpus is the benchmark: 0 of 364 healthy cells flagged, 16 of 16 of
    the aggregating sample.
    """
    _gamma_ok, gamma_note = _nonpositive_gamma_check(gamma_s_inv)
    decay_ok, decay_note = _observed_decay_check(
        _lowest_observed_acf_ratio(g2m1_windowed), max_retained_acf_ratio)
    # The size verdict is asked only of a decay the measurement observed (the one
    # precedence, shared with `exponentials._mode_flags`): an undecayed correlogram's Rh
    # is an extrapolation, not a size, and the size note's own wording used to be the
    # PROXY for this condition -- now that the condition is named, the proxy stands
    # down. Every other verdict is independent and is kept beside it.
    _size_ok, size_note = (_size_range_check(rh_nm, max_rh_nm, min_rh_nm)
                           if decay_ok else (True, ''))
    _pdi_ok, pdi_note = _pdi_check(pdi)
    rh_reliable, reliability_note = _result_flags(
        gamma_note, decay_note, size_note, pdi_note, visc_note)
    return rh_reliable, reliability_note, not decay_ok


def _pdi_and_validity(order: int, mu2: float, gamma: float) -> tuple[float, bool]:
    """Polydispersity index PDI = mu2 / gamma^2 and its reliability flag.

    An order-1 cumulant fits ONLY the mean decay rate Gamma -- it has no curvature
    (mu2) term -- so the polydispersity is UNMEASURED, not zero. Report PDI = NaN
    (not 0.0) so it is never mis-stated as a confirmed-monodisperse PDI = 0.000
    (invariant-8 spirit: omit when none is honest). For order >= 2, PDI
    is meaningful; pdi_valid requires it finite, non-negative (a negative mu2 from an
    over-subtracted baseline or a negative Gamma is physically impossible), and at or
    below the validity limit. NaN correctly fails that test, so order-1 is never
    marked reliable. Both fit paths (linear, nonlinear) share this to stay in lockstep
    (rule 10).

    Gamma is tested for FINITENESS, not only positivity. `+inf` passes a bare `gamma > 0`,
    and mu2/inf**2 is a perfectly finite 0.0 -- so the `math.isfinite(pdi)` conjunct could
    not see it and an infinite decay rate was reported as PDI = 0.000, flagged RELIABLE.
    A non-finite rate must be caught on the rate itself; there is no NaN downstream to
    notice (2026-07-31 spec-01).

    The square goes through `power_or_nan` for the reason its own docstring gives: the
    predicate names `gamma` while the division names `gamma ** 2`, and on a Python float
    `**` RAISES at the overflow tail (Gamma ~ 1e200 gave `OverflowError`) and underflows
    to 0.0 at the other, where `gamma != 0` still passes and the division raises
    `ZeroDivisionError`. A NaN divisor gives the NaN PDI this function already treats as
    "unmeasured", so no new branch is needed.
    """
    if order < 2:
        pdi = float('nan')
    else:
        pdi = (mu2 / unc.power_or_nan(gamma, 2)
               if (math.isfinite(gamma) and gamma != 0) else float('nan'))
    valid = bool(math.isfinite(pdi) and 0.0 <= pdi <= CUMULANT_PDI_VALIDITY_LIMIT
                 and math.isfinite(gamma) and gamma > 0)
    return pdi, valid


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
    success: bool = True              # fit produced usable coefficients. False for a
                                      # non-converged nonlinear fit, and for a linear fit
                                      # whose coefficients came back non-finite
    # "Did this method's preconditions hold for this fit?" (CLAUDE.md invariant 4): False
    # when the correlogram never decayed inside the lag window, the size is outside the
    # range DLS can establish, the PDI is outside the expansion's validity, the decay rate
    # came back non-positive, or no viscosity was available -- each named on the note
    # (see _cumulant_flags). The value is still reported: this flags it, it does not NaN it.
    rh_reliable: bool = True
    reliability_note: str = ''        # why rh is unreliable; '' when it is not flagged
    # The lag-window verdict on its own, because it invalidates something the other
    # causes do not: an undecayed correlogram makes Gamma ITSELF an extrapolation, where
    # an out-of-range size or a missing viscosity leaves the rate exact. Same field, same
    # meaning, as on ExponentialMode / KWWResult / ModePoint.
    decay_incomplete: bool = False


def fit_cumulants(
    measurement: DLSMeasurement,
    order: int = 2,
    fit_cutoff: float = 0.1,
    tau_min_s: Optional[float] = None,
    tau_max_s: Optional[float] = None,
    skip_initial_channels: int = 0,
    method: str = 'linear',
    max_rh_nm: float = CUMULANT_MAX_RH_NM,
    max_retained_acf_ratio: float = MAX_RETAINED_ACF_RATIO,
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
    max_rh_nm : float
        Rh above which the result is FLAGGED (not rejected) as outside the size range
        DLS can establish; default CUMULANT_MAX_RH_NM. Raise it deliberately if you are
        studying genuinely very large scatterers and understand the sedimentation
        caveat -- see _size_range_check.
    max_retained_acf_ratio : float
        The lag-window precondition: the windowed correlogram must fall, at some lag,
        below this fraction of its starting amplitude, or the fit is FLAGGED (not
        rejected) as an extrapolation past the end of the measurement. Default
        MAX_RETAINED_ACF_RATIO (ISO 22412:2017 Annex A.1.2's 1/50); see
        `_common._observed_decay_check`, and `_cumulant_flags` for why the cumulant asks
        the data rather than its own rate.

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
            measurement, order, fit_cutoff, tau_min_s, tau_max_s, skip_initial_channels,
            max_rh_nm, max_retained_acf_ratio)
    if method == 'linear':
        return _fit_cumulants_linear(
            measurement, order, fit_cutoff, tau_min_s, tau_max_s, skip_initial_channels,
            max_rh_nm, max_retained_acf_ratio)
    raise ValueError(f"method must be 'linear' or 'nonlinear', got {method!r}.")


def _fit_cumulants_linear(
    measurement: DLSMeasurement,
    order: int,
    fit_cutoff: float,
    tau_min_s: Optional[float],
    tau_max_s: Optional[float],
    skip_initial_channels: int,
    max_rh_nm: float = CUMULANT_MAX_RH_NM,
    max_retained_acf_ratio: float = MAX_RETAINED_ACF_RATIO,
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
    # Weights multiply the residuals; weight by signal amplitude. Routed through
    # polyfit_or_nan because the WEIGHT is a third non-finite channel that makes the bare
    # np.polyfit raise (unlike the response, which merely returns NaN).
    coeffs_high_first = unc.polyfit_or_nan(t, logy, order, w=y)
    c = coeffs_high_first[::-1]   # [c0, c1, c2, (c3)]

    # A withheld fit must not travel as a successful one. `success` was documented "True
    # for linear (always)" because this path could not previously RETURN without
    # coefficients -- it raised instead. Now that it degrades to NaN, the flag has a state
    # to describe, and leaving it True would label the row plain "linear" instead of
    # "linear (failed)" beside a line of n/a (gui/dls_module.py:1167).
    fit_ok = bool(np.all(np.isfinite(coeffs_high_first)))

    # `math.exp` overflows above ~709.78 and RAISES, and `fit_ok` cannot see it: it tests
    # the COEFFICIENTS, and a badly-scaled log-intercept is perfectly finite. The other
    # tail is quieter and needs the same answer -- below ~-745 the exponential underflows
    # to a finite 0.0, which sails past any `except` and would hand every consumer a
    # divide-by-beta. So the predicate is on the RESULT, exactly as in `_estimate_beta`,
    # which carries this form for the same expression. The handler assigns NaN rather than
    # inf because the predicate below is what decides either way.
    try:
        beta = math.exp(c[0]) if fit_ok else float('nan')
    except OverflowError:
        beta = float('nan')
    if not (math.isfinite(beta) and beta > 0):
        beta = float('nan')
    gamma = -c[1] / 2.0
    mu2 = c[2] if order >= 2 else 0.0
    mu3 = (-3.0 * c[3]) if order >= 3 else None
    pdi, pdi_valid = _pdi_and_validity(order, mu2, gamma)

    q = _measurement_q_m(measurement)
    d = _d_from_gamma(gamma, q)
    visc_note = ''
    try:
        eta = _require_viscosity(measurement)
        rh = _decay_rate_to_rh_nm(gamma, q, measurement.temperature_K, eta)
    except ValueError:
        rh = float('nan')
        visc_note = NO_VISCOSITY_NOTE

    # Model evaluated over the fitted region, for plotting and residuals. The fitted
    # region IS the cutoff-masked (high-amplitude, short-lag) subset — so rms_error
    # below is over that support, matching the nonlinear path's rms (so goodness-of-fit
    # is comparable across methods).
    model_log = np.polyval(coeffs_high_first, t)
    fitted = np.exp(model_log)
    residuals = y - fitted

    # Preconditions are judged on the WINDOWED correlogram `g2m1`, not on the cutoff
    # subset `y` the regression saw: ISO's retained-range rule binds the data the fit
    # was given, and the amplitude cutoff is this method's own choice of what to
    # regress (see _cumulant_flags). The floor is a FLAG here, where the nonlinear path
    # rejects (owner decision D2, CUMULANT_MIN_RH_NM).
    rh_reliable, reliability_note, decay_incomplete = _cumulant_flags(
        rh, gamma, g2m1, pdi, pdi_valid, visc_note,
        max_rh_nm, CUMULANT_MIN_RH_NM, max_retained_acf_ratio)

    return CumulantResult(
        order=order,
        beta=beta,
        gamma_s_inv=gamma,
        mu2_s_inv2=mu2,
        mu3_s_inv3=mu3,
        pdi=pdi,
        pdi_valid=pdi_valid,
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
        success=fit_ok,
        rh_reliable=rh_reliable,
        reliability_note=reliability_note,
        decay_incomplete=decay_incomplete,
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
    max_rh_nm: float = CUMULANT_MAX_RH_NM,
    max_retained_acf_ratio: float = MAX_RETAINED_ACF_RATIO,
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
        # `_seed_or_floor`, not `max(...)`: `max(nan, x)` returns nan, and the linear fit
        # this seeds from is exactly the one that can withhold beta.
        beta0 = _seed_or_floor(seed.beta, 1e-6)
        gamma0 = _seed_or_floor(seed.gamma_s_inv, 1e-3)
        mu2_0 = _seed_or_floor(seed.mu2_s_inv2, 0.0)
        mu3_0 = seed.mu3_s_inv3 or 0.0
    except Exception:
        beta0, gamma0, mu2_0, mu3_0 = float(g2m1.max()), 1.0, 0.0, 0.0
    tail = g2m1[-max(2, g2m1.size // 4):]
    B0 = float(np.mean(tail))
    # The baseline is small drift around the (already ~0) long-time value, NOT a
    # free amplitude sink: bound it near the tail so the fit cannot absorb the
    # decay into B and chase a short-lag artifact (which would give a spurious
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
        d = _d_from_gamma(gamma, q)
        visc_note = ''
        try:
            eta = _require_viscosity(measurement)
            rh = _decay_rate_to_rh_nm(gamma, q, measurement.temperature_K, eta)
        except ValueError:
            rh = float('nan')
            visc_note = NO_VISCOSITY_NOTE
        # Reject degenerate solutions: non-finite, collapsed coherence, or an
        # unphysically tiny Rh (the fit chased a short-lag artifact into Gamma).
        if not (math.isfinite(gamma) and gamma > 0 and math.isfinite(beta) and beta > 0):
            ok = False
        elif math.isfinite(rh) and rh < CUMULANT_MIN_RH_NM:
            ok = False
    if not ok:
        # D2 (owner decision 2026-07-07): a failed nonlinear fit returns NaN physical
        # outputs, NOT a linear-fit substitute — so a caller can't read a fit-like
        # number from a failed fit. (Previously it fell back to the linear fit flagged
        # success=False; that labeled fallback was dropped in favor of an honest NaN.)
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

    pdi, pdi_valid = _pdi_and_validity(order, mu2, gamma)
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

    # The floor is passed for shape parity with the linear path; it cannot fire here,
    # since a sub-floor solution was rejected above (owner decision D2). The undecayed
    # question is asked of the data, not of the fitted Gamma -- a Frisken fit on an
    # undecayed correlogram pins its baseline to a tail that is still signal and
    # absorbs the decay into B, so its Gamma is the wrong number to judge by
    # (measured: an 800 nm scatterer truncated at 0.1 relaxation times fits 242 nm).
    rh_reliable, reliability_note, decay_incomplete = _cumulant_flags(
        rh, gamma, g2m1, pdi, pdi_valid, visc_note,
        max_rh_nm, CUMULANT_MIN_RH_NM, max_retained_acf_ratio)

    return CumulantResult(
        order=order,
        beta=beta,
        gamma_s_inv=gamma,
        mu2_s_inv2=mu2,
        mu3_s_inv3=mu3,
        pdi=pdi,
        pdi_valid=pdi_valid,
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
        rh_reliable=rh_reliable,
        reliability_note=reliability_note,
        decay_incomplete=decay_incomplete,
    )
