"""
analysis/dls/_common.py
=======================

Shared helpers used by two or more DLS method families (cumulants, exponentials,
distributions, angular, replicate). Kept private to the package; the public API is
re-exported from ``analysis/dls/__init__.py``.

These are the low-level primitives every fit shares: the scattering vector for a
measurement, the viscosity guard, the decay-rate -> Rh conversion, the RMS-error
helper, the delay-time window (skip + tau bounds), and the distribution-method
baseline/beta/grid builders. They take data + parameters and return plain values
or arrays; none plot, write files, or mutate inputs.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from core.data_models import DLSMeasurement
from physics.constants import scattering_vector_q_m, stokes_einstein_rh
from analysis import uncertainty as unc


def _measurement_q_m(measurement: DLSMeasurement) -> float:
    """Scattering vector q (m^-1) for a measurement, from its angle/wavelength/n."""
    return scattering_vector_q_m(
        measurement.angle_deg,
        measurement.wavelength_nm,
        measurement.solvent_refractive_index,
    )


# The reason string every degrading fit attaches to its blank Rh when viscosity is
# absent, so the NaN says why on each display/export surface. One home for the wording,
# shared by the cumulant and exponential arms so they cannot drift apart.
NO_VISCOSITY_NOTE = (
    "Rh is not computed: this measurement has no viscosity, which Stokes-Einstein "
    "needs. Set viscosity on the measurement to get Rh; the decay rate and "
    "diffusion coefficient do not need it."
)


def _require_viscosity(measurement: DLSMeasurement) -> float:
    """Return the measurement viscosity (Pa.s), or raise if it is missing.

    Viscosity is needed only to convert a decay rate to a hydrodynamic radius
    (Stokes-Einstein). Decay rates and diffusion coefficients do not need it.
    """
    if measurement.viscosity_Pa_s is None:
        raise ValueError(
            "This measurement has no viscosity, which is required to compute a "
            "hydrodynamic radius (Rh). Set viscosity_Pa_s on the measurement, or "
            "use the decay rate / diffusion coefficient outputs, which do not "
            "need viscosity."
        )
    return measurement.viscosity_Pa_s


# The fraction of its initial amplitude that a correlogram's RETAINED fit range must fall
# to for ISO 22412:2017 to call the delay range correct. Annex A.1.2 (printed p. 20) fixes
# the range kept for the cumulant fit as [G2(tau_1) - A] > [G2(tau_j) - A] > a hundredth of
# [G2(tau_1) - A], "with at least one value smaller than" a FIFTIETH of it. The 1/50 is the
# binding half -- the point the range must REACH, where the /100 only bounds where it may
# stop. The "shall be discarded" clause printed next in that paragraph governs the
# POSITIVITY requirement that follows, not this reach, for which ISO states no consequence.
_ISO_RETAINED_RANGE_DECAY_FACTOR = 50.0

# Smallest decay completeness -- Gamma * (lag span), the relaxation times the measured lag
# range covers -- at which a mode's rate is a MEASUREMENT rather than an extrapolation past
# the end of the data. A documented, overridable default per invariant 4; the criterion it
# comes from, and how far it is ISO's rather than ours, are in _decay_completeness_check.
#
# Written as its ISO relation, not as the literal 1.96, so the relationship is what the code
# asserts (the CUMULANT_MAX_RH_NM == 1e4/2 precedent): [G2(tau) - A] goes as
# |g1|^2 = exp(-2 Gamma tau), so "reach 1/50 of the initial amplitude" is 2 G dt >= ln(50).
MIN_DECAY_COMPLETENESS = math.log(_ISO_RETAINED_RANGE_DECAY_FACTOR) / 2.0


# The DATA-LEVEL form of the very same ISO criterion MIN_DECAY_COMPLETENESS states as a
# rate: the retained range must reach 1/50 of its initial amplitude. One constant behind
# both, so the two arms cannot drift to different standards -- the rate form assumes an
# exponential decay law, the data form assumes nothing, and it is one sentence either way.
# Xu (2000) p. 241 gives an independent, stricter version; measured on the committed Pairs
# census the two select the SAME cells, so nothing rests on choosing between them. See
# `_lowest_observed_acf_ratio` and the Theory guide for both, and for Xu's other half --
# a first lag too long to catch the fastest motion, deliberately out of scope here.
MAX_RETAINED_ACF_RATIO = 1.0 / _ISO_RETAINED_RANGE_DECAY_FACTOR


# The tolerances of the through-origin verdict two fits share -- Gamma vs q^2 across
# angles, and one picked mode family: `is_diffusive` is R^2 >= DIFFUSIVE_R2_MIN AND
# |intercept|/mean(Gamma) <= DIFFUSIVE_INTERCEPT_REL_MAX, so translational diffusion
# (Gamma = D q^2, a line through the origin) means the same thing on both surfaces.
# General statistical criteria per invariant 4 -- nothing instrument-, solvent- or
# polymer-specific -- and overridable per call: each fit takes them as parameters and
# echoes what it used onto its result.
DIFFUSIVE_R2_MIN = 0.98
DIFFUSIVE_INTERCEPT_REL_MAX = 0.1


def _lowest_observed_acf_ratio(g2m1) -> float:
    """The LOWEST normalized ACF reached anywhere in the fitted window, over its first value.

    ISO 22412 A.1.2 asks for "at least ONE value smaller than [G2(tau_1) - A]/50" -- a
    minimum over the retained range, not the value at the end. That distinction is the whole
    behaviour of this helper, and reading it as the last point instead was measured wrong:
    a healthy correlogram's tail is noise scattered about zero, so whichever channel happens
    to land last can sit back above the threshold long after the decay finished. On the
    committed Pairs census the last-point reading flagged 35 of 168 KWW fits across 14 of 21
    samples -- including 10 nm gold and Dextran(488k), which plainly decayed -- while the
    minimum flags 8 of 168, all of them the one aggregating sample, at every angle, with no
    healthy sample touched. The faithful reading is also the robust one.

    It is likewise threshold-insensitive where the last-point reading was not: ISO's 1/50
    and Xu's stricter 0.005 (Xu 2000 p. 241) select the SAME eight cells, so the criterion
    does not rest on picking between two sources' numbers.

    Model-free by construction, which is why the KWW arm uses it: it asks the correlogram
    rather than a fitted decay law, so no per-model algebra enters and a model that absorbs
    an undecayed mode into a SHAPE parameter cannot hide it (a stretched exponential does
    exactly that -- on sample 20 the stretch exponent falls to 0.21-0.68 while tau_c stays
    short, and every tau_c-based measure clears all eight).

    g2-1 is proportional to |g1|^2 by the Siegert relation, so this ratio is the normalized
    ACF's with beta cancelling. Dividing by the first retained point rather than assuming
    |g1(tau_1)|^2 = 1 is the conservative direction: a window starting partway down the
    decay inflates the ratio and so flags sooner, never later.

    Withholds (NaN) when there is no ratio to take: fewer than two finite points, or a
    non-finite/non-positive first point. A NEGATIVE result is legal and means the signal
    reached the noise floor -- the check below reads it as a pass, which is correct.

    KNOWN GAP. ISO states the reach on [G2(tau) - A]; this reads `g2 - 1`, which is meant to
    be baseline-removed already. A residual drift offset larger than the threshold therefore
    reads as an incomplete decay. Measured: no false flag on the 160 healthy committed runs
    (typical margin ~100x), reproduced at ~2 % of beta on a constructed one. Subtracting a
    tail-estimated baseline is NOT the fix -- on a genuinely undecayed correlogram the tail
    is still signal, so it would erase the flag on the case this exists for; the fix is a
    floating baseline on the fit (a recorded follow-up, not yet built).
    """
    y = np.asarray(g2m1, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 2:
        return float('nan')
    y0 = float(y[0])
    if not (math.isfinite(y0) and y0 > 0):
        return float('nan')
    ratio = float(np.min(y)) / y0
    if math.isfinite(ratio):
        return ratio
    # Both tails of an overflowed quotient keep their sign: +inf reads as the strongest
    # 'did not decay', -inf as the documented pass (the signal crossed the noise floor).
    return math.copysign(float('inf'), ratio)


def _observed_decay_check(
    ratio: float,
    max_ratio: float = MAX_RETAINED_ACF_RATIO,
) -> Tuple[bool, str]:
    """(rh_reliable, reliability_note) for one fit's `_lowest_observed_acf_ratio`.

    Same shape and same disposition as `_decay_completeness_check` -- flags, never rejects
    -- so every arm reads one shape whichever question it had to ask. A NaN ratio is left
    unflagged for the same reason a NaN completeness is: `success` already carries a failed
    fit and there is nothing here to judge.

    NOTE THE POLARITY, which is INVERTED from its sibling and was wrong here first.
    In `_decay_completeness_check` a non-finite value means +inf relaxation times
    covered, so passing it is right. Here the quantity is what is LEFT at the end, so
    +inf means the signal GREW instead of decaying -- the most emphatic failure there
    is. Only NaN is unanswerable; a copied `not math.isfinite(...)` cleared the worst
    case in the file, and only the two-sided smoke test caught it.
    """
    if math.isnan(ratio) or ratio <= max_ratio:
        return True, ''
    return False, (
        f'This correlogram did not finish decaying inside its lag window: at no measured lag '
        f'does it fall below {ratio:.3g} of its starting amplitude, against the '
        f'{max_ratio:.3g} that ISO 22412:2017 (Annex A.1.2) requires a retained fit range '
        f'to reach. The relaxation time, and any size derived from '
        f'it, is an extrapolation past the end of the measurement rather than a measured '
        f'value. Measure a longer lag range; a static or aggregated sample is the usual '
        f'cause.'
    )


def _decay_completeness(gamma_s_inv: float, tau_span_s: float) -> float:
    """Gamma * delta_tau: how many relaxation times the measured lag range covers.

    ONE computation, so the predicate below has nothing of its own left to mis-size
    (rule 12). `tau_span_s` is the SPAN of the fitted window, `fit_tau_s[-1] - fit_tau_s[0]`,
    not its last value: ISO's rule is stated relative to the first RETAINED point tau_1, and
    on a user-narrowed window the decay actually observed is the one across the window. On a
    real log lag axis tau_1 << tau_max, so the two readings agree to ~1e-3.

    Two-sided, and both tails are honest verdicts rather than errors (rule 12's
    two-sided clause):

        1e300 * 1e300  ->  inf   a hugely complete decay; passes the check
        1e-300 * 1e-300 -> 0.0   no decay at all; fails it

    so the product is returned as it comes out, with no result guard to get backwards. The
    square trap does not arise -- this is a product of two independent quantities, never
    `x ** 2` -- but the same rule applies to the operands: a non-finite or non-positive
    Gamma or span has no completeness to report and returns NaN, which the predicate reads
    as "unanswerable, do not flag" the way `_size_range_check` reads a non-finite Rh.
    """
    if not (math.isfinite(gamma_s_inv) and gamma_s_inv > 0):
        return float('nan')
    if not (math.isfinite(tau_span_s) and tau_span_s > 0):
        return float('nan')
    return gamma_s_inv * tau_span_s


def _decay_completeness_check(
    completeness: float,
    min_completeness: float = MIN_DECAY_COMPLETENESS,
) -> Tuple[bool, str]:
    """(rh_reliable, reliability_note) for one mode's `_decay_completeness`.

    A pure predicate on an ALREADY-COMPUTED quantity, deliberately: the guard and the
    operation it protects are then the same number, so there is no second expression to size
    against the first (rule 12's "one computation" remedy). Callers compute the completeness
    once, record it on the mode, and pass it here.

    FLAGS rather than rejects, matching `cumulants._size_range_check`'s shape so every arm
    reads one shape -- and for the same reason: the fitted rate is real output of a real
    correlogram, and what the user needs is to be told it is an extrapolation, not to have
    it withheld (invariant 4's flag-not-refuse rule: a computed value is always reported,
    carrying its flag).

    CANONICAL-METHOD ANCHOR (CLAUDE.md). The criterion and its scale are ISO 22412:2017's,
    not this program's: Annex A.1.2 (printed p. 20) fixes the retained fit range and asks
    it to reach 1/50 of the initial amplitude (stating no consequence for failing to, which
    is why FLAGGING rather than discarding deviates from nothing here), and
    Sec. 10.2.1 a) (printed p. 13) states the qualitative half -- a valid result needs a
    delay-time range that "covers the set of decay times". What the standard states for
    the correlogram's WHOLE retained range, against the single mean rate a cumulant
    returns, this applies PER MODE. That extension is ours, and it is made because a
    two-mode fit's slow mode can be undecayed beside a perfectly measured fast one, which
    a single whole-correlogram verdict cannot express. Not a house invention; equally, not
    something ISO prints per mode.

    A NaN completeness is left unflagged: `success` already carries a failed fit, exactly as
    `_size_range_check` leaves a non-finite Rh alone. An infinite one passes -- it is a decay
    that finished many times over.

    NOTE PRECEDENCE. Where a caller has both this note and `NO_VISCOSITY_NOTE` to attach,
    THIS one wins. It is the stronger statement: a missing viscosity says only that Gamma
    could not be turned into a radius, while an undecayed mode says the Gamma itself is
    extrapolated. Both set `rh_reliable = False`, so the flag never under-reports either way
    -- only the sentence differs.
    """
    if not math.isfinite(completeness) or completeness >= min_completeness:
        return True, ''
    return False, (
        f'This decay did not finish inside the lag window: the measured lag range '
        f'spans only {completeness:.3g} of its relaxation time 1/Gamma, against the '
        f'{min_completeness:.3g} that ISO 22412:2017 (Annex A.1.2) requires of a retained '
        f'fit range -- its rule that the range must reach 1/{_ISO_RETAINED_RANGE_DECAY_FACTOR:.0f} '
        f'of the initial amplitude is 2*Gamma*delta_tau = ln '
        f'{_ISO_RETAINED_RANGE_DECAY_FACTOR:.0f}. The decay rate of this mode, and any size '
        f'derived from it, is an extrapolation past the end of the measurement rather than '
        f'a measured value. Measure a longer lag range; a static or aggregated sample is '
        f'the usual cause.'
    )


def _result_flags(*notes: str) -> tuple:
    """Roll several precondition verdicts up to one (rh_reliable, reliability_note).

    A DISJUNCTION: the result is unreliable when ANY of its notes is live. That direction
    is chosen deliberately -- a display or export surface that reads only the rolled-up
    flag then over-flags a healthy neighbouring mode rather than clearing an unreliable
    one, and a missing flag is the failure that matters. Per-cell surfaces read the
    per-mode field.

    Every arm combines its verdicts through this one helper -- the exponential fits roll
    their modes up with it, the cumulant fit joins its own preconditions with it, and
    the angular fits fold the per-point verdicts of the measurements they pooled -- so
    no arm can drift to its own precedence rule. ALL applicable notes are kept, in the
    order given, undecayed-type causes first by convention: they are independent facts
    about different things (this rate is an extrapolation; no viscosity was available to
    turn a rate into a radius), and neither implies the other. An earlier version let
    the stronger note REPLACE the viscosity one; that silently withdrew "no viscosity"
    from a measurement that had both, so a blank Rh stated a reason that was not why it
    was blank.

    Duplicates collapse: a cause shared by two modes (a missing viscosity) would
    otherwise print its sentence twice in one note.
    """
    live = [n for n in notes if n]
    return (not live), ' '.join(dict.fromkeys(live))


def _d_from_gamma(gamma_s_inv: float, q_m_inv: float) -> float:
    """Translational diffusion coefficient D = Gamma / q^2, or NaN when it is unanswerable.

    The one place that division is written, because it needs TWO guards and every inline
    copy of it had only one. `Gamma` must be finite and positive -- a failed fit gives NaN
    (owner decision D2) -- and `q**2` must be finite and non-zero, which is a separate
    question about the OPTICS, not the fit: an extreme wavelength or refractive index
    makes q underflow, and `q**2` reaches exactly 0.0 while q itself is still non-zero.

    Written inline, this expression tends to guard `gamma` and not `q` -- which raises a
    bare `ZeroDivisionError` at one tail and returns `D = +inf` with `success = True` at
    the other, a wrong number carrying a good flag. The guard names one operand while the
    division names two (CLAUDE.md rule 12), and that is why this helper exists rather than
    another inline predicate.

    **The square is written `q * q`, never `q ** 2`, and that is not a style choice.** On
    a PYTHON float, `**` RAISES `OverflowError` where `*` saturates to `inf`:

        1e300 * 1e300  ->  inf          math.isfinite(inf) is False, so the guard fires
        1e300 ** 2     ->  OverflowError    raised INSIDE the guard, before it can run

    So the first version of this guard closed the underflow tail it was written for and
    opened a crash on the overflow tail, in the guard expression itself. (NumPy is not
    affected -- `np.float64(1e300) ** 2` is `inf` -- so array guards elsewhere are fine;
    this is a Python-scalar trap only.) A guard on a computed quantity is TWO-SIDED and
    must be checked at 1e+300 as well as 1e-300; writing the square with `**` is what
    violating that looks like.

    **The last guard is on the RESULT, and neither input guard can replace it.** Two
    positive finite inputs still make an unusable quotient at the ends of the range:

        _d_from_gamma(2e3,    2e-153)  ->  q2 = 4e-306, finite and positive  ->  D = inf
        _d_from_gamma(2e-297, 2e+15)   ->  both finite and positive          ->  D = 0.0

    The first is exactly the `D = +inf` this helper's second paragraph says it exists to
    prevent, riding out to `d_m2_s` on CumulantResult/ExponentialMode; the second reaches
    `stokes_einstein_rh`, which REFUSES a non-positive D, so a withhold turns into a
    `ValueError` out of the public `angular.gamma_to_rh_nm`. Guarding the two operands
    answers "can this division be performed", which is not the question -- the question is
    "is the answer a diffusion coefficient", and only the quotient can answer that
    (CLAUDE.md rule 12).
    """
    if not (math.isfinite(gamma_s_inv) and gamma_s_inv > 0):
        return float('nan')
    q2 = q_m_inv * q_m_inv          # `*` saturates to inf; `**` would raise -- see above
    if not (math.isfinite(q2) and q2 > 0):
        return float('nan')
    d_m2_s = gamma_s_inv / q2
    return d_m2_s if (math.isfinite(d_m2_s) and d_m2_s > 0) else float('nan')


def _seed_or_floor(value: float, floor: float) -> float:
    """A finite starting value for a fit's `p0`: `value` if usable, else `floor`.

    THE fix for a NaN-blind idiom that appeared eight times across the DLS fitters as
    `max(seed.<field>, <floor>)`. That does not do what it looks like: `max(nan, x)`
    returns **nan**, because the comparison is False and `max` keeps its first argument.

    It is load-bearing because the cumulant seed's fields are WITHHOLDABLE -- an
    unrepresentable log-intercept yields `beta = nan` while `gamma` and `Rh` stay perfectly
    valid. The NaN then reaches the starting vector of the
    single-exponential, double-exponential, KWW and nonlinear-cumulant fits, and
    `curve_fit` failed on a correlogram the data supported. The degradation was invisible
    because it happened to raise a `ValueError` those callers already catch, so a withheld
    SEED read as a failed FIT.

    One helper for the whole idiom rather than per-field guards: guarding the field a
    finding names leaves its siblings untouched on the very same lines, which is a change
    list written from the finding rather than from a scan (rules 10 and 11).

    A seed is a starting point, never a reported value: every caller returns the FITTED
    parameter on success and NaNs the physical results on failure, so a floor used here
    cannot surface as a physical claim (invariant 3 is not engaged).
    """
    value = float(value)
    return max(value, floor) if math.isfinite(value) else floor


def _decay_rate_to_rh_nm(
    gamma_s_inv: float,
    q_m_inv: float,
    temperature_K: float,
    viscosity_Pa_s: float,
) -> float:
    """Convert a g1 decay rate Gamma (s^-1) to a hydrodynamic radius (nm).

    D = Gamma / q^2 ;  Rh = kB T / (6 pi eta D).

    The single conversion used by the cumulant and every exponential path, so it is the
    highest-leverage gate of its family. Gamma must be FINITE as well as positive:
    `gamma <= 0` rejects NaN but admits `+inf`, and an infinite Gamma gives an infinite D,
    which Stokes-Einstein turns into a finite, plausible-looking Rh = 0.0 nm carrying no ±
    to warn anyone (2026-07-31 spec-01). A non-finite q is caught for the same reason --
    it makes D non-finite by the division rather than by the numerator.

    Both guards come from `_d_from_gamma`, which this delegates to. They were duplicated
    here, and the duplicate then had to be corrected twice in a row (the q**2 underflow,
    then the q**2 OVERFLOW that the first correction introduced). Two copies of a guard
    is two chances to size it wrong; one delegation is none (rule 12).

    The delegated guard covers `d == 0.0` as well as a non-finite D, which this line's own
    `isfinite` did not: `stokes_einstein_rh` refuses a non-POSITIVE D, so an underflowed
    quotient raised `ValueError` here instead of withholding, breaking the "answer or NaN,
    never raise" contract at the public `angular.gamma_to_rh_nm`. Sizing this predicate to
    the one downstream is what rule 12 asks for; the test that the two agree is the ladder
    in tests/test_positivity_guard_family.py.
    """
    d_m2_s = _d_from_gamma(gamma_s_inv, q_m_inv)
    if not (math.isfinite(d_m2_s) and d_m2_s > 0):
        return float('nan')
    rh_m = stokes_einstein_rh(d_m2_s, temperature_K, viscosity_Pa_s)
    return rh_m * 1e9


def _rms_error(residuals: np.ndarray) -> float:
    """Root-mean-square of an array of residuals."""
    if residuals.size == 0:
        return float('nan')
    return float(np.sqrt(np.mean(residuals ** 2)))


def _apply_tau_window(
    tau: np.ndarray,
    g2m1: np.ndarray,
    tau_min_s: Optional[float],
    tau_max_s: Optional[float],
    min_points: int,
    skip_initial_channels: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Restrict (tau, g2-1) to the analysis window used by every fit.

    Two complementary restrictions, applied in this order:

      1. `skip_initial_channels` -- drop the first N correlator channels by INDEX
         (the shortest, finest-spaced lags). These leading channels are where
         detector afterpulsing and correlator dead-time artifacts live, regardless
         of their absolute lag time, so this is an instrumental cleanup, not a
         physical window. Default 0 leaves the correlogram untouched.
      2. `[tau_min_s, tau_max_s]` -- the user's physical delay-time window
         (inclusive). Either bound may be None to leave that side open.

    The two compose as an intersection: the effective first fitted point is the
    later of "channel N" and "first channel with tau >= tau_min_s". The skip is
    applied FIRST so that a spuriously inflated first channel can never distort a
    downstream intercept/amplitude-cutoff or beta estimate. This whole step runs
    before any method-specific selection (e.g. the cumulant amplitude cutoff).

    Parameters
    ----------
    tau, g2m1 : np.ndarray
        Delay times (s, ascending) and correlogram (g2 - 1).
    tau_min_s, tau_max_s : float or None
        Inclusive lower/upper delay bounds. None leaves that side open.
    min_points : int
        Minimum number of surviving points the calling model needs.
    skip_initial_channels : int
        Number of leading channels to drop by index (default 0). Must be a
        non-negative integer smaller than the number of channels.

    Returns
    -------
    (tau_windowed, g2m1_windowed)

    Raises
    ------
    ValueError
        If skip_initial_channels is invalid, tau_min_s >= tau_max_s, or fewer
        than min_points survive.
    """
    # 1. Leading-channel skip (by index), validated.
    if skip_initial_channels is None:
        skip_initial_channels = 0
    if (isinstance(skip_initial_channels, bool)
            or not isinstance(skip_initial_channels, (int, np.integer))
            or skip_initial_channels < 0):
        raise ValueError(
            f"skip_initial_channels must be a non-negative integer, "
            f"got {skip_initial_channels!r}."
        )
    skip = int(skip_initial_channels)
    if skip >= tau.size:
        raise ValueError(
            f"skip_initial_channels ({skip}) removes the whole correlogram "
            f"({tau.size} channels). Reduce it."
        )
    if skip:
        tau = tau[skip:]
        g2m1 = g2m1[skip:]

    # 2. Physical delay-time window (inclusive).
    if (tau_min_s is not None and tau_max_s is not None
            and tau_min_s >= tau_max_s):
        raise ValueError(
            f"tau_min_s ({tau_min_s}) must be < tau_max_s ({tau_max_s})."
        )
    mask = np.ones(tau.shape, dtype=bool)
    if tau_min_s is not None:
        mask &= tau >= tau_min_s
    if tau_max_s is not None:
        mask &= tau <= tau_max_s
    t = tau[mask]
    y = g2m1[mask]
    if t.size < min_points:
        raise ValueError(
            f"Only {t.size} points remain after skipping the first {skip} "
            f"channel(s) and applying the delay window [{tau_min_s}, {tau_max_s}] s; "
            f"this fit needs at least {min_points}. Reduce skip_initial_channels "
            f"or widen the window."
        )
    return t, y


# ---------------------------------------------------------------------------
# beta / baseline estimation (distribution methods)
# ---------------------------------------------------------------------------

def _estimate_baseline(tau: np.ndarray, g2m1: np.ndarray,
                       tail_fraction: float = 0.25) -> float:
    """Estimate a residual baseline offset from the long-delay tail.

    The correlogram is nominally g2 - 1 (baseline already removed, so ~0 at long
    tau), but real data drift leaves a small offset. We estimate it as the mean
    of the last `tail_fraction` of the (time-ordered) points, where the signal
    has fully decayed.
    """
    n = g2m1.size
    k = max(1, int(round(tail_fraction * n)))
    return float(np.mean(g2m1[-k:]))


def baseline_over_window(tau: np.ndarray, g2m1: np.ndarray,
                         lo_s: float, hi_s: float) -> Optional[float]:
    """The residual baseline offset measured over the user's own tau window: the mean
    of g2 - 1 across ``lo_s <= tau <= hi_s``, or None when the window selects no point.

    The sibling of `_estimate_baseline`, which takes the last `tail_fraction` of the
    points by INDEX. This one takes a window in SECONDS because that is what the user
    drags on the correlogram, and the two agree only on a monotone tau axis -- they
    can differ by a point, which is why the widget's own copy of this arithmetic was a
    second definition of a physics-affecting quantity rather than a display detail
    (invariant 5: the engine owns the number, the widget owns the handles). The
    endpoints are ordered here, so a window dragged right-to-left measures the same
    baseline as one dragged left-to-right.
    """
    lo, hi = sorted((float(lo_s), float(hi_s)))
    mask = (tau >= lo) & (tau <= hi)
    if not np.any(mask):
        return None
    return float(np.mean(g2m1[mask]))


def _estimate_beta(tau: np.ndarray, g2m1: np.ndarray,
                   baseline: float) -> float:
    """Estimate the coherence factor beta (intercept) for normalization.

    Uses a short second-order cumulant-style log-polynomial extrapolation of the
    baseline-subtracted signal to tau = 0. Robust and cheap; the user can always
    override beta explicitly. Falls back to the max if the extrapolation fails.

    Guaranteed to return a positive FINITE beta. Every consumer divides by it
    (`u = (g2m1 - baseline) / beta`), so a NaN beta would hand NNLS/CONTIN an all-NaN
    signal and an infinite one an all-zero signal -- both silent, neither an error.
    """
    y = g2m1 - baseline
    # The fallback floor is computed from the same array as the fit, so it needs the same
    # finiteness test: `y.max()` is nan/+inf exactly when y is, and returning it would
    # trade a crash for a poisoned beta (2026-07-31 spec-01, review finding).
    max_y = float(y.max()) if y.size else float('nan')
    intercept0 = max_y if math.isfinite(max_y) else 0.0
    mask = (y > 0.1 * intercept0) & (y > 0)
    if mask.sum() >= 3:
        t = tau[mask]
        coeffs = unc.polyfit_or_nan(t, np.log(y[mask]), 2, w=y[mask])
        # The finiteness test has to be on the COEFFICIENTS, not left to the `except`.
        # This site's degradation used to be driven by `np.polyfit` RAISING; routing it
        # through a helper that returns NaN instead would have made the handler dead code
        # and turned the floored intercept below into a NaN beta -- a fix silently
        # deleting the caller's fallback (2026-07-31 spec-01).
        if np.all(np.isfinite(coeffs)):
            # Test the RESULT, not the failure mode. `except OverflowError` covers only
            # the exp OVERFLOW tail (intercept > ~709); the UNDERFLOW tail
            # (intercept < ~-745) returns a finite 0.0, which sails past any except
            # clause and past the floor below -- so the function returned beta = 0.0
            # while its own docstring promised a positive finite value, and the
            # distribution layer then raised about a beta the user never supplied.
            # One predicate on the answer covers both tails.
            try:
                candidate = float(math.exp(np.polyval(coeffs, 0.0)))
            except OverflowError:
                candidate = float('inf')
            if math.isfinite(candidate) and candidate > 0:
                return candidate
    return max(intercept0, 1e-9)


def _build_rh_gamma_grid(
    rh_min_nm: float,
    rh_max_nm: float,
    n_grid: int,
    q_m_inv: float,
    temperature_K: float,
    viscosity_Pa_s: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build paired log-spaced Rh (nm) and Gamma (s^-1) grids.

    Rh is log-spaced from rh_min_nm to rh_max_nm. Each Rh maps to a decay rate
    via Gamma = D q^2 with D = kB T / (6 pi eta Rh). Because Gamma is inversely
    proportional to Rh, the gamma grid is descending where rh ascends; both are
    returned in Rh-ascending order.
    """
    from physics.constants import stokes_einstein_diffusion_coefficient
    rh_grid_nm = np.geomspace(rh_min_nm, rh_max_nm, n_grid)
    if rh_grid_nm.size == 0:
        return rh_grid_nm, np.array([])
    # D = kB T / (6π η Rh) ∝ 1/Rh, so compute D once (the physics function guards
    # scalars, so it can't take the array) at the smallest Rh and scale the rest
    # exactly — vectorizing the former per-point Python loop without duplicating the
    # Stokes-Einstein formula into the analysis layer.
    d0 = stokes_einstein_diffusion_coefficient(
        rh_grid_nm[0] * 1e-9, temperature_K, viscosity_Pa_s)
    # `q_m_inv * q_m_inv`, never `** 2`: `**` binds tighter than `*`, so the square is
    # evaluated as a PYTHON float before any array arithmetic, and there it RAISES
    # OverflowError at q ~ 1e155 rather than saturating (measured, out of the NNLS/CONTIN
    # grid builder). Same trap as `_d_from_gamma`'s note above, but on a FACTOR of a `*`
    # rather than the divisor of a `/`.
    gamma_grid = d0 * (rh_grid_nm[0] / rh_grid_nm) * (q_m_inv * q_m_inv)
    return rh_grid_nm, gamma_grid
