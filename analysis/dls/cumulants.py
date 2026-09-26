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

THE RESIDUAL WEIGHT is a named, recorded choice on both arms, and it travels with the
number on `CumulantResult.weighting`. The linear arm has one, the program's form of the
weight ISO 22412:2017 Annex A.1.2 prescribes below Formula (A.18), printed p. 20: the
standard asks for a normalized weight that accounts for the log transformation and prints
no form, and each squared residual of ln(g2-1) weighted by (g2-1)^2 is the program's own
derivation of it. The nonlinear arm has two: the sourced per-channel
variance of Biganzoli and Ferri (2018) Eq. (9), exact in Gamma, evaluated on a fitted
single exponential ('biganzoli-full' / 'biganzoli-classical'), and the instrument's own
reported per-channel SDs ('instrument', Frisken 2001's route). What each one IS, and
which divergences it carries, is on `fit_cumulants`' `weighting` parameter and on
`_fit_cumulants_nonlinear`.

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
    _live_channels,
    channel_sample_times,
    _d_from_gamma,
    _decay_rate_to_rh_nm,
    _keyed_notes,
    _lowest_observed_acf_ratio,
    MAX_RETAINED_ACF_RATIO,
    _measurement_q_m,
    _observed_decay_check,
    _pinned_offset,
    _require_viscosity,
    _result_flags,
    NO_VISCOSITY_NOTE,
    _rms_error,
    _biganzoli_sd,
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

# Amplitude at which the linear cumulant's retained RANGE stops, as a fraction of the
# windowed correlogram's peak. NAMED rather than left a bare default because the value is
# user-settable and so has three homes -- this one, `SettingsState.cumulant_fit_cutoff`
# and `angular.py`'s pass-through -- and three homes for one number is how they drift.
# CANONICAL-METHOD ANCHOR (CLAUDE.md): a BENCHMARKED DEVIATION. ISO 22412:2017 Annex A.1.2
# prescribes 1/100; this is 1/10, after Mailer, Clegg & Pusey (2015) Sec. 3 p. 7-8. The
# full statement, including which A.1.2 clauses are implemented, is on the `fit_cutoff`
# parameter of `fit_cumulants` and in the Theory-and-Equations-Guide.
CUMULANT_FIT_CUTOFF_DEFAULT = 0.1

# The residual weightings a caller may REQUEST on the nonlinear arm, each one a statement
# about what a residual is measured in: 'biganzoli', the sourced per-channel SD of
# Biganzoli and Ferri (2018) Eq. (9), exact in Gamma, evaluated on a fitted single
# exponential (Koppel 1972); 'instrument', the instrument's own per-channel SDs, supplied
# by the caller (Frisken 2001); 'uniform', no weight (sigma_k = 1) -- a labelled opt-in,
# never a candidate for the default. The amplitude weight evaluated on the noisy
# observation, printed by no source, is not among them: the Theory-and-Equations-Guide's
# "Which weight is the default" says why it went.
NONLINEAR_WEIGHTINGS = ('biganzoli', 'instrument', 'uniform')

# The linear arm's default weighting: the program's form of the weight ISO 22412:2017
# Annex A.1.2 prescribes below Formula (A.18) for the LINEARIZED fit (the standard prints no
# form; (g2-1)^2 on each squared residual of ln(g2-1) is the program's derivation); and
# the arm's roster, which adds the same unweighted opt-in on its own residual, ln(g2-1).
LINEAR_WEIGHTING = 'iso-a18'
LINEAR_WEIGHTINGS = (LINEAR_WEIGHTING, 'uniform')

# What `CumulantResult.weighting` may hold. The request token 'biganzoli' splits in two on
# the way to the result, because the same sample is weighted differently on an instrument
# that reports a mean count rate (Eq. (9)'s two shot-noise brackets enter) and one that
# cannot (the classical infinite-count-rate limit), and a result must say which it got.
# 'uniform' is recorded on either arm, and also where the default's sourced weight could
# not be formed and the fit fell back to it (see `_fit_cumulants_nonlinear`).
RECORDED_WEIGHTINGS = frozenset({
    'biganzoli-full', 'biganzoli-classical', 'instrument', 'iso-a18', 'uniform',
})

# The nonlinear arm's default weighting, chosen HERE and nowhere else: `weighting=None`
# on that arm means this constant, so the program-wide default is one line to move and
# every result records which value produced it. The sourced variance is the default
# (owner-ruled 2026-09-08; the evidence -- the in-domain mu2 bias removed, the weighted
# residuals white -- is in the Theory-and-Equations-Guide's "Which weight is the default").
# The
# one request resolves to the classical limit on an instrument that reports no mean count
# rate, and the result records which form it got -- instrument-agnostic by construction.
CUMULANT_NONLINEAR_WEIGHTING_DEFAULT = 'biganzoli'

# The recorded value(s) each REQUEST token resolves to ('biganzoli' splits by whether a
# count rate entered) and, driven from the default, the set the EXPORT IS SILENT ON.
# Owner-ruled 2026-09-08, this comment its normative home: the program's default
# weighting, whatever it is, is not labelled in the export -- the guide describes it, as
# it describes HC3 and 'calibrated' -- while a weighting the user opted into is named
# wherever its numbers travel. The linear arm's one weighting is silent too: a label
# with no alternative says nothing. `exporting.export._weighting_note` reads this set.
_RECORDED_FORMS = {
    'biganzoli': ('biganzoli-full', 'biganzoli-classical'),
    'instrument': ('instrument',),
    'uniform': ('uniform',),
    LINEAR_WEIGHTING: (LINEAR_WEIGHTING,),
}
DEFAULT_RECORDED_WEIGHTINGS = (frozenset(_RECORDED_FORMS[CUMULANT_NONLINEAR_WEIGHTING_DEFAULT])
                               | {LINEAR_WEIGHTING})

# Below what DIMENSIONLESS value a fitted moment is called BOUND-RESTING: the stored PDI
# mu2/Gamma^2 and the fitted m4 = mu4/Gamma^4 are each compared against it. Owner-ruled
# 2026-09-08, this comment its normative home: the note is a REPORT that the optimiser
# came to rest at the wall, and the wall is tested on the dimensionless moment against
# 1e-6 -- above both heights at which the optimiser stops (a moment seeded on its 0.0
# bound is parked at scipy's 1e-10 feasibility nudge; one driven there stops by xtol a
# little above) and four orders below the smallest polydispersity a correlogram resolves.
# `_moment_bound_note` says why the dimensionless form and not the fitted parameter.
MOMENT_BOUND_REL = 1e-6

# Floor of the nonlinear (Frisken) cumulant's floating-baseline bound, as a FRACTION of the
# windowed correlogram's peak amplitude: B may move at least this far from the tail mean
# even when the tail's own scatter is smaller (a noiseless synthetic tail has none). It
# replaced an absolute +/-0.05 that was routinely tens of times wider than the data's
# scatter and let B absorb a real baseline error silently -- see
# `_frisken_baseline_bound`. A documented module constant per invariant 3.
FRISKEN_BASELINE_BOUND_REL_FLOOR = 1e-3

# How close to an edge of `[B0 - dB, B0 + dB]` the fitted floating baseline must come
# before the fit is called BOUND-ACTIVE (see `_bound_active_note`). A NUMERICAL closeness
# tolerance sized to the solver, not a physical threshold: `curve_fit` stops on its own
# convergence tolerances rather than landing on a bound to the last bit, so a saturated
# fit rests a few times 1e-8 of the half-width inside the edge and an ulp-scale test
# would never fire.
FRISKEN_BOUND_ACTIVE_REL_TOL = 1e-6


def _bound_active_note(b_fitted: float, b0: float, d_b: float) -> str:
    """The passive note for a floating baseline that came to rest ON its bound, or ''.

    ONE predicate on the quantities the bound already computed -- the centre `b0` and the
    half-width `d_b` handed to `curve_fit` as `[b0 - d_b, b0 + d_b]` -- so there is no
    second expression to mis-size against the constraint it describes (rule 12's
    one-computation form). Two-sided by construction: `abs(b - b0)` reaches `d_b` at
    either edge, and the note does not say which, because both mean the same thing --
    the optimizer wanted a baseline the bound would not let it have.

    It is a NOTE, not a flag (invariant 4 clause 2, owner-ruled 2026-09-03): a
    bound-active B is a symptom of where the tail was measured, not a failed precondition
    of the cumulant expansion, and the polydispersity verdict already flags the fits this
    was measured to co-occur with. Same one-string discipline as the CONTIN sweep-edge
    notes (`distributions._ALPHA_STATE_NOTES`), which this mirrors.

    A non-finite input yields no note: every comparison against NaN is False, so the
    predicate below is written to answer "provably at the bound", never "not provably
    interior".

    WHERE IT DOES NOT FIRE, measured over the committed 90 degree corpus: the distance
    from the edge is a continuum, not a gap -- three fits sit exactly on it, two more
    4.4e-6 and 5.2e-6 away (five times this tolerance, so unmarked), and the next is at
    2e-3. And a baseline region on a quiet stretch of the DECAY leaves B interior at a
    high centre, so the largest size errors a misplaced region produces carry no note at
    all. Both are the open owner fork on this method's sheet, not settled here.
    """
    if not (math.isfinite(b_fitted) and math.isfinite(b0) and math.isfinite(d_b)
            and d_b > 0.0):
        return ''
    if abs(b_fitted - b0) < d_b * (1.0 - FRISKEN_BOUND_ACTIVE_REL_TOL):
        return ''
    return (
        f'The floating baseline came to rest on its bound (B = {b_fitted:.4g}, the '
        f'largest move of {d_b:.3g} the bound allows from the tail mean {b0:.4g}), so '
        f'the decay rate is set by the edge of that bound rather than by an interior '
        f'best fit. Either the lags the baseline was measured over still hold signal, '
        f'or the baseline region sits inside the decay -- check the baseline region and '
        f'measure a longer delay range.'
    )


def _moment_bound_note(pdi: float, m4: Optional[float], order: int) -> tuple:
    """`(target, note)` pairs for a fitted MOMENT that came to rest on its lower bound of 0.

    Pairs rather than bare strings because the two notes are about DIFFERENT quantities
    and the caller cannot tell them apart otherwise: both fire conditionally, so a
    one-element return could be either, and a reader that put the mu4 sentence on the PDI
    would be stating something false about a measured number. The targets name result
    attributes and are consumed by `_keyed_notes`; see its docstring for the convention.

    ONE predicate per moment on the DIMENSIONLESS quantity the reader sees: `pdi` is the
    stored mu2/Gamma^2 (the same `_pdi_and_validity` value the result carries -- one
    computation, rule 12), `m4` the fitted mu4/Gamma^4 at order 4 (the parameter itself,
    dimensionless by the model's layout; None below order 4). Both are compared against
    `MOMENT_BOUND_REL`, whose comment is the normative home of the threshold and of the
    owner's 2026-09-08 ruling that put the test on the dimensionless moment.

    WHY THE STORED PDI AND NOT THE FITTED PARAMETER. mu2 is now fitted dimensionless
    (`m2 = mu2 / Gamma^2`, `_frisken_model_scaled_moments`), so the stored PDI equals the
    fitted m2 to rounding and the two select the same fits; the predicate stays on the
    stored PDI so that the value the reader sees and the value tested are one computation.
    The threshold is
    a bound-DETECTION height, not a tolerance on the moment; an `== 0.0` predicate in its
    place would be blind to the whole parked case, since scipy's
    `make_strictly_feasible(x0, lb, ub, rstep=1e-10)` moves a moment seeded on its bound
    to 1e-10 before the fit starts.

    IT FIRES ON BOTH CAUSES, deliberately. A moment DRIVEN onto the bound and one PARKED
    on the seed it started at mean the same thing to a reader: the fit found no interior
    optimum for that moment, so the number reported is the constraint rather than a
    measurement. The parked case is not exotic -- it is reachable on every noiseless
    monodisperse synthetic, where mu2 is seeded at 0 and nothing moves it off.

    They are NOTES, not flags (invariant 4 clause 1, CLAUDE.md -- a flag answers whether a
    precondition held; this docstring is the ruling's home together with it): `rh_reliable`
    and `pdi_valid` are untouched, because
    no precondition of the cumulant expansion failed. The measurement is fine, the size is
    fine, and the polydispersity is simply below what this correlogram's noise resolves --
    which is what the sentence says, in place of a verdict it has no grounds for.

    A non-finite entry yields no note, as in `_bound_active_note`: every comparison against
    NaN is False, so the predicate answers "provably at the bound", never "not provably
    interior".
    """
    # The wording states the FACT and points to the guide, never a cause (owner-ruled
    # 2026-09-08; `ui_style_guide.md` R5.7 is the pattern's home): more than one thing
    # drives a moment onto its bound -- a polydispersity below what the correlogram
    # resolves, an expansion that does not describe the sample at this order -- and the
    # fit cannot tell them apart, so the note does not pretend to.
    notes = []
    if order >= 2 and math.isfinite(pdi) and pdi <= MOMENT_BOUND_REL:
        # Targets `pdi`, not `mu2_s_inv2`: the predicate is on the stored PDI and the
        # sentence is about the PDI shown, so the note lands on the number it describes.
        notes.append((
            'pdi',
            'The second moment mu2 came to rest on its lower bound (0), so the PDI shown '
            'is that bound, not a measurement; the Theory-and-Equations-Guide sec. 3.2.7 '
            'explains what can cause this.'
        ))
    if order == 4 and m4 is not None and math.isfinite(m4) and m4 <= MOMENT_BOUND_REL:
        notes.append((
            'mu4_s_inv4',
            'The fourth moment mu4 came to rest on its lower bound (0), so the mu4 shown '
            'is that bound, not a moment; the Theory-and-Equations-Guide sec. 3.2.7 '
            'explains what can cause this.'
        ))
    return tuple(notes)


def _dead_channel_note(n_dead: int, n_windowed: int) -> str:
    """The passive note for channels that collected NO coincidence, or ''.

    An intensity autocorrelation cannot be negative, so `g2 <= 0` -- equivalently
    `g2 - 1 <= -1`, the form the internal data model stores -- is not a small number: it
    is a channel that recorded nothing at all. Those channels cannot enter the linear
    cumulant's log-polynomial fit and are removed with the rest of the non-positive
    signal, silently, leaving a perfectly good size for the head of the correlogram and
    no statement that part of the window measured nothing.

    WHY THE PREDICATE IS `g2 <= 0` AND NOT "the positivity mask dropped something"
    (measured 2026-09-03 over the 352 committed real correlograms, both formats): a
    non-positive `g2 - 1` is the ORDINARY state of a decayed tail -- the median
    correlogram has 24 % of its channels below zero and only 10 of 352 have none -- so a
    note on that condition would fire on 97 % of linear fits and say nothing. A channel
    at `g2 <= 0` is rare and specific: 6 of the same 352. The distinction is physical, not
    a tuned threshold, which is what keeps this inside invariant 4's "general
    light-scattering criteria" rule; there is no constant here to tune.

    A NOTE, not a flag (invariant 4 clause 2; owner-ruled 2026-09-03 on the committed
    `corr-dead-tail-channels.csv`): the precondition the cumulant needs is a decay inside
    the lag window, and the surviving channels had one -- the 121 nm it reports is right
    for them. What was missing was any statement that a sixth of the window measured
    nothing. The DLS side of `sls._nonpositive_ordinate_note`, with the same shape (count,
    total, the cause named as a property of the data, the usual reason named).

    EVERY fitter says it, and every fitter REMOVES the channels -- the mask is
    `_common._live_channels`, read by both cumulant arms directly and by the single, double,
    KWW, NNLS, CONTIN and lognormal fits through `_common._apply_tau_window_live` (one rule,
    owner-ruled 2026-09-08). The linear path cannot take
    the logarithm of a dead channel; the nonlinear path once fitted the whole window and
    reached the same effective answer by accident -- its legacy amplitude weight put a
    dead channel at the weight floor -- but a channel with no coincidence has no estimate
    and no variance either, so under a weight stated from the photon statistics
    (`_sourced_sigma`) it would enter at full weight and drag the baseline to -1 (measured
    2026-09-08 on the committed dead-tail correlogram: B = -1.04, beta = 1.71, a 2 um size
    for a 121 nm sample). Removal is the only treatment that is right under every weight;
    the fitted curve and the residuals still span the whole window for the plot.
    """
    if n_dead <= 0:
        return ''
    return (
        f'{n_dead} of {n_windowed} channels in the delay window collected no coincidence '
        f'(g2 = 0, so g2-1 = -1 exactly), so they could not be fitted and were dropped; '
        f'the size reported here is measured on the channels that remain, usually the '
        f'shorter lags. A dead tail usually means the sample '
        f'scattered too weakly for the longest lags of this delay range -- check the '
        f'correlogram, and consider a shorter delay range or a longer run.'
    )


def _tail_for_bound(tau: np.ndarray, g2m1: np.ndarray,
                    baseline_window) -> np.ndarray:
    """The channels the nonlinear cumulant measures its floating baseline over.

    `baseline_window` None -> the engine's own INDEX tail, the last quarter of the
    windowed lags (at least two channels), unchanged and the default on every path that
    does not name a window. A `(lo_s, hi_s)` pair -> the channels whose lag lies inside
    it, so the user's baseline region names where the tail is measured rather than
    supplying a value to subtract (owner-ruled 2026-09-03; the region a user never set
    never reaches here -- see `gui/dls_module.py::_AnalysisRegion`).

    A window holding fewer than two channels is REFUSED, naming the window: a one-channel
    "tail" has no scatter, so `5*std` collapses to zero and the bound would silently fall
    back to its relative floor as though the user's region had been honoured. Widening it
    for the user would be the same silence one step further away (the package-1 D2 shape
    for an unusable input: refuse and name it, never repair it).

    The endpoints are ordered here, as `_common.baseline_over_window` orders its own, so a
    region dragged right-to-left names the same channels as one dragged left-to-right.
    """
    if baseline_window is None:
        return g2m1[-max(2, g2m1.size // 4):]
    lo, hi = sorted((float(baseline_window[0]), float(baseline_window[1])))
    mask = (tau >= lo) & (tau <= hi)
    n = int(np.count_nonzero(mask))
    if n < 2:
        raise ValueError(
            f"The baseline region [{lo:.4g}, {hi:.4g}] s holds {n} of the {tau.size} "
            f"live delay channels this fit was given; the nonlinear cumulant measures its "
            f"floating baseline over that region and needs at least 2. The count is over "
            f"the channels inside the delay window that collected any coincidence -- not "
            f"the whole correlogram, and not channels at g2 = 0, which every cumulant "
            f"drops -- so a region lying outside that window, on a dead tail, or outside "
            f"this measurement's lag range altogether is empty here however wide it looks. "
            f"Widen the region, move it onto live channels inside the delay window, or "
            f"restore both to their defaults."
        )
    return g2m1[mask]


def _frisken_baseline_bound(g2m1_windowed: np.ndarray, tail: np.ndarray,
                            rel_floor: float) -> float:
    """Half-width of the floating baseline's bound, `B in [B0 - dB, B0 + dB]`, for the
    nonlinear cumulant -- or NaN when no positive, finite bound can be formed.

    The bound exists so the fit cannot absorb the decay into B (which would give a spurious
    Rh ~ 0). Its width is the LARGER of the tail's own scatter, `5 * std(tail)`, and a floor
    relative to the correlogram's peak, `rel_floor * max(g2m1)`, and it is CAPPED at that
    peak -- the baseline may never roam farther than the correlogram's own amplitude. The
    width and the cap are sized from the same array the fit sees, so the guard and the
    operation it protects are one computation, and both tails of the float range are
    survived: on a noiseless tail (`std -> 0`) the floor keeps the bound open, and on a
    tail whose variance overflows (`std -> inf`) the cap keeps it finite.

    The floor is RELATIVE, not absolute, because an absolute floor is sized to no
    correlogram: on a low-amplitude one it is far wider than the tail's scatter, and the fit
    then places B well above the data's own baseline and shifts Gamma with it (the worked
    case is pinned in the test suite). Frisken (2001) floats B to DETECT small deviations from
    a correctly determined baseline; Koppel (1972, Eq. 19) and ISO 22412:2017 Annex A.1.2
    both treat the baseline as a determined quantity and a large residual as a
    data-quality signal, so a bound wide enough to swallow one silently defeats both.

    Why an INDEX-defined tail (the last quarter of the windowed LIVE lags -- the arm
    hands this helper the channels that collected coincidence) and not the
    Gamma-relative one: ISO 22412:2017 Annex A.1.2 defines the tail-estimated baseline at
    delay times `tau >> 25 / Gamma`. A Gamma-relative tail needs the very Gamma this fit
    is being run to obtain, and on a correlogram that has not decayed an index tail is
    still measuring signal, so `std(tail)` there prices decay as noise and widens the bound
    where it should tighten. Kept as a flagged convenience simplification of the standard
    (the canonical-method anchor's third category, CLAUDE.md): the undecayed case is
    flagged separately by the retained-range criterion (`_cumulant_flags`), and the fit's
    Gamma on such a correlogram is not read as a size either way.
    """
    peak = float(np.max(g2m1_windowed))
    # A non-finite peak is refused up front: every comparison against NaN is False, so
    # `min`/`max` would silently hand back the other operand and neither the floor nor
    # the cap would be in force while the returned width still read as finite.
    if not (math.isfinite(peak) and peak > 0.0):
        return float('nan')
    spread = 5.0 * float(np.std(tail))
    d_b = min(max(spread, rel_floor * peak), peak)
    return d_b if (math.isfinite(d_b) and d_b > 0.0) else float('nan')


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
            f'within the delay window -- check for a static or aggregated sample, and '
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
            f'this cumulant size is unreliable for a sample this polydisperse -- prefer '
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

    The question is asked of the LIVE channels of the WINDOWED correlogram (after
    `_apply_tau_window` and `_live_channels`, before the linear path's amplitude cutoff;
    a dead channel is no observation of the decay and must not pass the criterion for a
    fit that never saw it) -- a deliberate deviation from the Annex's letter
    (the canonical-method anchor's class 2, benchmarked): ISO prescribes the criterion on
    the RETAINED range of the cumulant fit.

    WHY THE WINDOW AND NOT THE RETAINED RANGE. Because the LAG WINDOW is the frame the
    precondition is stated in. CLAUDE.md invariant 4 clause 1 names this criterion
    directly -- "a mode's decay completes inside the lag window -- the `Gamma*dtau`
    criterion" -- so the window is the quantity the precondition is ABOUT, and the
    amplitude cutoff is a truncation applied within it: a different quantity, at a
    different stage, answering a different question. Asked of the retained range the
    criterion would not be a stricter version of itself; it would be a criterion about
    something else.

    NOT rested on the broader claim that a verdict must not move with any analysis
    setting (round 1 finding 4). That claim is false of this very code -- the window is
    built by `_apply_tau_window` from `tau_min_s`/`tau_max_s` and
    `skip_initial_channels`, all user-set -- and it collides with invariant 4 clause 2,
    which makes a user's windowing and sub-selection the user's own. The narrow reason
    above survives both clauses; the broad one survived neither.

    (The justification here was once arithmetic -- that a range bounded below at 0.1 of
    the intercept 'by construction can never reach the 1/50 mark', 0.02 < 0.1. True of
    the shipped default, false at the Annex's own 0.01, so it stopped being a reason the
    moment the cutoff became settable. Recorded rather than deleted because a
    justification a control can falsify is worth recognising by shape.)

    The measurement on the committed corpus is the benchmark either way: 0 of 364 healthy
    cells flagged, 16 of 16 of the aggregating sample.
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


# The invariant STEM of the retained-range note, named rather than proxied (invariant 4
# clause 1). A stem and not the whole note since round 2 finding 1: the note is now
# parametrised on the surviving point count and the order, because the sentence that used
# to follow -- "the measurement shows no decay over its lag window" -- asserted a cause the
# range's size cannot establish. Consumers and tests match on this prefix so the note and
# they cannot drift apart, while the numbers stay free to vary.
NO_RETAINED_RANGE_NOTE_STEM = (
    'No contiguous range of the correlogram long enough for an order-'
)


def _retained_range_too_short_note(n_points: int, order: int, fit_cutoff: float) -> str:
    """Why the retained range could not carry an order-`order` fit, without guessing why.

    ROUND 2 FINDING 1, and the fix is not the one the finding proposed. The old text said
    flatly that 'the measurement shows no decay over its lag window'. That is FALSE for a
    correlogram which decays fine but whose retained range is merely short -- reachable at
    the shipped default cutoff, where a 3-channel range fits at order 2 and fails at
    order 3, and the note reaches the screen behind a warning glyph.

    The finding proposed splitting on `t.size == 0` vs `0 < t.size < order + 1`. MEASURED,
    THAT DOES NOT SEPARATE THE TWO CAUSES: pure uncorrelated noise -- the genuinely-no-decay
    case -- retains 1 to 6 channels across seeds, overlapping the short-but-real range
    completely. A size boundary would have told a user with a dead sample to lower their
    cutoff. There is no cheap predicate here, so this note does not claim to know which
    cause applies; it states what is TRUE in both (the count, and what the order needs) and
    names both remedies. Invariant 4 clause 1 wants the failed precondition named, not a
    proxy -- and 'fewer points survived than this order needs' IS the precondition, where
    'no decay' was an inference about the sample.
    """
    return (
        f'No contiguous range of the correlogram long enough for an order-{order} fit '
        f'stays above the amplitude cutoff: {n_points} point(s) survive at a cutoff of '
        f'{fit_cutoff:g} of the windowed peak, and at least {order + 1} are needed, so '
        f'there is nothing for the cumulant expansion to regress. Either the measurement '
        f'shows little or no decay over its delay window -- check the sample, the count '
        f'rate and the lag range -- or the range is real but too short for this order, in '
        f'which case a shallower cutoff or a lower order will fit it.'
    )


def _retained_range_mask(signal: np.ndarray, fit_cutoff: float) -> np.ndarray:
    """ISO 22412:2017 Annex A.1.2's retained set: a contiguous RANGE, not a filter.

    A.1.2 (prose prescription, printed p. 20) states the retained values as
    "a range in [G2(tau_j) - A] of [G2(tau_1) - A] > [G2(tau_j) - A] > {[G2(tau_1) - A]/100}"
    -- an interval, terminated where the decay crosses the threshold. A correlogram
    decays monotonically only in EXPECTATION, so once the underlying decay approaches
    the threshold, noise carries individual channels above and below it and a
    per-channel predicate keeps only the ones that landed above: a rectified,
    upward-biased tail that the log-linear regression then fits as a slow decay mode.
    Gamma_bar and mu2 both fall and Rh = kT/(6 pi eta D) rises without bound. Measured
    2026-09-05 over the cutoff grid, full decay at cutoff 1/100: median |Rh - truth|/truth
    20.131 under the per-channel predicate against 0.032 under this one, the whole
    difference being ~21 rectified tail channels out of ~148.

    Two sources describe the same rule as a TERMINATION rather than a filter, and they
    are what this rests on -- A.1.2's own sentence can also be read as a two-sided band
    in amplitude, on which reading contiguity in tau is an inference:
    Schaetzel (1990) p. 293, "the commonly used countermeasure is to assume that all
    chi_i vanish for all i >= k, where k is the smallest positive integer with a
    normalized measured correlation less than zero" -- stated there for NOISE ESTIMATION
    on the amplitude estimator at a ZERO threshold, so it is a precedent for the shape
    (terminate, do not filter) and not a prescription for this fit's retained range; and Mailer, Clegg & Pusey (2015)
    Sec. 3, p. 7-8, "the data were TRUNCATED WHEN g(2)(tau) - 1 DROPPED TO 10% of its
    initial value".

    The interval is anchored on the PEAK channel and grown outward until the threshold
    is crossed in each direction -- not taken as the leading run from channel 0. The
    threshold is measured from `signal.max()`, so the peak is the one channel guaranteed
    to satisfy it, and the interval containing it is therefore never empty for a positive
    peak. A leading-run reading is equivalent whenever the correlogram's head is clean,
    and on the committed corpus it is not always: `09 - PS latex NPs (700nm)/DLS - 120
    deg.dat` dips at channel 1 (0.019 against a 0.032 threshold) before peaking at
    channel 5, and a leading run would retain ONE channel and turn a fitted measurement
    into a too-few-points refusal. That is ISO clause 3's discard behaviour arriving by
    the back door, and it collides with the owner's flag-not-refuse ruling (2026-08-24).
    Measured over the 190 committed Brookhaven correlograms at the shipped cutoff: both
    readings differ from the shipped mask on the same 10 files; the leading run refuses 1
    of them, this form refuses none.

    `signal > 0` is NOT algebraically dead here even though `signal > fit_cutoff *
    signal.max()` implies it for every positive peak (measured 2026-09-05: 0 of 162 grid
    cells where it changes the mask). It is what makes a NON-POSITIVE-peak window return
    an empty range: with a peak <= 0 the threshold is itself <= 0, the peak fails its own
    threshold, and without this conjunct the interval would retain the peak channel alone
    and hand a non-positive value to `np.log`. Under the superseded per-channel predicate
    the conjunct emptied the mask directly; under a range it is load-bearing in a second
    way, so the "delete it and fall to the too-few-points refusal" option is no longer
    equivalent (rule 12: the guard and the operation it protects are now one computation).
    """
    above = (signal > fit_cutoff * float(signal.max())) & (signal > 0)
    peak = int(np.argmax(signal))
    mask = np.zeros(signal.shape, dtype=bool)
    if above[peak]:
        crossings = np.flatnonzero(~above)
        before, after = crossings[crossings < peak], crossings[crossings > peak]
        lo = int(before[-1]) + 1 if before.size else 0
        hi = int(after[0]) if after.size else int(signal.size)
        mask[lo:hi] = True
    return mask


@dataclass
class CumulantResult:
    """Cumulant-expansion fit of g2(tau) - 1.

    THE HIGHER MOMENTS (`mu3_s_inv3`, `mu4_s_inv4`) are present from the order that fits
    them and None below it. Both are CENTRAL MOMENTS of the decay-rate distribution: mu3 is
    also the third cumulant, since cumulants and central moments differ only from order 4
    (Mailer, Clegg & Pusey 2015, terminology note, preprint p. 6). mu4 is the fourth central
    moment -- the quantity Frisken's nonlinear model fits directly (Mailer Eq. 13, p. 5) --
    and NOT the fourth cumulant kappa4 = mu4 - 3 mu2^2 that the linear log-polynomial's
    quartic coefficient yields (Mailer Eq. 12, same page); the linear arm converts, so the
    two arms report one quantity under one name (`_fit_cumulants_linear`).

    REPORTED BARE, WITH NO VALIDITY COMPANION (owner-ruled 2026-09-05; this docstring is
    the ruling's normative home, CLAUDE.md invariant 4). Mailer sec. 4.3-4.4 measure the
    uncertainty of a fitted mu3 or mu4 as comparable to its own value at every
    polydispersity -- and that is the BEST case: on this program's planted benchmarks the
    nonlinear arm's mu4 often rests on its zero bound at PDI <~ 0.1, and the linear arm's
    converted mu4 is off by decades and often negative below PDI ~ 0.6 (a
    log-polynomial's quartic coefficient amplifies the tail noise the exponential
    prefactor damps; the Theory-and-Equations-Guide, "Orders 3 and 4: the higher moments"). So
    their unreliability is a constant property of the
    method, not a precondition that held or failed on THIS run, and a flag is reserved for
    the latter. The caution lives in the readout tooltips and the
    Theory-and-Equations-Guide's "Orders 3 and 4: the higher moments";
    the values are diagnostics for comparing fits of one sample, never a measurement of the
    distribution's shape, and a fitted 0.0 at order 4 is the bound, not a moment.
    """
    order: int
    beta: float                       # coherence factor (intercept), exp(c0)
    gamma_s_inv: float                # first cumulant = mean g1 decay rate
    mu2_s_inv2: float                 # second cumulant (variance of Gamma)
    # The higher moments: central moments, order-gated, reported bare -- the class
    # docstring is the normative home of what they are and why no flag accompanies them.
    mu3_s_inv3: Optional[float]       # third central moment == third cumulant (order >= 3)
    mu4_s_inv4: Optional[float]       # fourth CENTRAL MOMENT, not kappa4 (order >= 4)
    pdi: float                        # mu2 / gamma^2
    pdi_valid: bool                   # pdi <= CUMULANT_PDI_VALIDITY_LIMIT
    # derived size (intensity-weighted, apparent at this q)
    d_m2_s: float
    rh_nm: float
    q_m_inv: float
    # fit bookkeeping
    fit_cutoff: float                 # bounds the retained RANGE; see fit_cumulants
    n_points_used: int
    coefficients: np.ndarray          # polynomial coeffs [c0, c1, c2, ...]
    fit_tau_s: np.ndarray             # the tau values actually fitted
    fitted_g2m1: np.ndarray           # model g2-1 at fit_tau_s
    residuals: np.ndarray             # data - model over fit_tau_s
    rms_error: float                  # RMS residual over the retained range (high-
    #                                   amplitude, short-lag) support — the SAME for
    #                                   'linear' and 'nonlinear', so it is comparable
    #                                   across methods (may differ from RMS of the
    #                                   full `residuals` array on the nonlinear path)
    # WHICH WEIGHT BUILT THIS FIT'S RESIDUALS, one of `RECORDED_WEIGHTINGS`; required, since a
    # weight moves every number here (the `se_estimator` pattern, invariant 8 clause A).
    # 'biganzoli-…' says whether a committed count rate entered Eq. (9); 'uniform' is no
    # weight, asked for or the flagged fallback (`_fit_cumulants_nonlinear`).
    weighting: str
    # WHICH per-channel sample time the sourced weight was evaluated on: 'instrument'
    # (the parser recorded the correlator's own, `DLSMeasurement.sample_times_s`),
    # 'axis' (a stand-in derived from the delay axis, `channel_sample_times`), or ''
    # when this fit read none (the linear arm; the 'instrument' and 'uniform' weightings).
    # Required and beside `weighting` for the same reason: on a correlator whose sampling
    # time is not its lag spacing the stand-in is off by decades on most channels, so a
    # number weighted on it never travels without saying so -- the exporter names the
    # axis-derived case.
    sample_time_provenance: str
    # WHICH form of Eq. (9)'s 1/(M - k) the sourced weight was evaluated on: 'finite' (the
    # 'biganzoli' weighting on a measurement carrying `DLSMeasurement.duration_s`, so
    # 1/(M - k) = Delta_t_k / (T - tau_k)), 'infinite' (none carried: the T -> infinity form,
    # a named practicality divergence), or '' when this fit read no duration (the linear arm;
    # the 'instrument' and 'uniform' weightings). Beside `sample_time_provenance`, since the two forms weight
    # a long-lag channel differently, but a RECORD for a caller holding the result: unlike the
    # axis-derived sample time no export column or note names it -- the exported weighting
    # says which variance ran; this says only which form of its 1/(M - k).
    run_duration_form: str
    n_skipped: int = 0                # leading channels dropped (skip_initial_channels)
    method: str = 'linear'            # 'linear' (Koppel 1972) or 'nonlinear' (Frisken 2001)
    baseline: float = 0.0             # the baseline the fit used: the fitted floating B
    #                                   (nonlinear), or the caller's pinned offset subtracted
    #                                   before the log (linear; 0.0 when none was supplied)
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
    # PASSIVE notices about how this fit was obtained -- channels dropped before it, a
    # floating baseline that came to rest on its bound. They are NOT reliability verdicts:
    # `rh_reliable` answers "did the method's preconditions hold" (invariant 4 clause 2)
    # and neither of these says they did not, so the flag is untouched and these travel
    # beside it. Same field name, type and tier as AngularResult / ConcentrationResult
    # carry in this package, and the exporter relays them through `_analysis_notes`.
    notes: tuple = ()
    # What each note is ABOUT, index-aligned with `notes`: a result attribute name, or ''
    # for a note about the whole fit. Built WITH the notes in one step
    # (`_common._keyed_notes`), so a display surface can put a passive notice on the cell
    # it explains instead of on the fit. Empty when a producer wrote notes without
    # targets; a reader meeting unequal lengths treats every note as fit-level.
    note_targets: tuple = ()


def _resolve_weighting(method: str, weighting: Optional[str]) -> Optional[str]:
    """The residual weighting a cumulant fit of `method` will run under, as a REQUEST token.

    `weighting=None` is the ARM's default -- `CUMULANT_NONLINEAR_WEIGHTING_DEFAULT` on the
    nonlinear arm, `LINEAR_WEIGHTING` on the linear -- and any other value is returned as
    given once it has passed the refusals below. The one resolution, shared by
    `fit_cumulants` and by the pooled fits that echo what their per-point cumulants were
    asked for (`angular.py`), so the echo and the fit cannot read two rules.

    Refused, naming the roster of the arm in question: a token on no arm's roster, and a
    token that belongs to the OTHER arm. Each cross-arm refusal is driven from the roster of
    the arm it protects (`LINEAR_WEIGHTINGS`, `NONLINEAR_WEIGHTINGS`), never from the other
    arm's tuple, so a token that joins both rosters ('uniform') is refused by neither.
    A `method` that is neither arm is returned unresolved; `fit_cumulants` refuses it.
    """
    known = set(NONLINEAR_WEIGHTINGS) | set(LINEAR_WEIGHTINGS)
    if weighting is not None and weighting not in known:
        raise ValueError(
            f"weighting must be one of {sorted(known)}, got {weighting!r}. The nonlinear "
            f"arm takes {sorted(NONLINEAR_WEIGHTINGS)} and the linear arm "
            f"{sorted(LINEAR_WEIGHTINGS)}; pass None for the arm's own default.")
    if method == 'nonlinear':
        resolved = CUMULANT_NONLINEAR_WEIGHTING_DEFAULT if weighting is None else weighting
        if resolved not in NONLINEAR_WEIGHTINGS:
            raise ValueError(
                f"weighting={weighting!r} is not on the nonlinear arm's roster, which fits "
                f"g2-1 directly and takes one of {sorted(NONLINEAR_WEIGHTINGS)}, or None "
                f"for its default. {LINEAR_WEIGHTING!r} is the LINEARIZED fit's weight "
                f"(ISO 22412:2017 Annex A.1.2); pass method='linear' to use it.")
        return resolved
    if method == 'linear':
        resolved = LINEAR_WEIGHTING if weighting is None else weighting
        if resolved not in LINEAR_WEIGHTINGS:
            raise ValueError(
                f"weighting={weighting!r} belongs to the nonlinear arm; the linear "
                f"(log-polynomial) fit takes one of {sorted(LINEAR_WEIGHTINGS)}, or None "
                f"for ISO 22412:2017 Annex A.1.2's `w = y`. Pass method='nonlinear' to use "
                f"it.")
        return resolved
    return weighting


def fit_cumulants(
    measurement: DLSMeasurement,
    order: int = 2,
    fit_cutoff: float = CUMULANT_FIT_CUTOFF_DEFAULT,
    tau_min_s: Optional[float] = None,
    tau_max_s: Optional[float] = None,
    skip_initial_channels: int = 0,
    method: str = 'linear',
    max_rh_nm: float = CUMULANT_MAX_RH_NM,
    max_retained_acf_ratio: float = MAX_RETAINED_ACF_RATIO,
    baseline_bound_rel_floor: float = FRISKEN_BASELINE_BOUND_REL_FLOOR,
    baseline: Optional[float] = None,
    baseline_window: Optional[tuple] = None,
    weighting: Optional[str] = None,
    channel_sigma: Optional[np.ndarray] = None,
) -> CumulantResult:
    """Fit g2(tau) - 1 by the method of cumulants.

    Two fitting methods, selected by `method`:

    - 'linear' (Koppel 1972, the engine default): fit a weighted polynomial to
      ln(g2 - 1) over the decay region (overflow-free, closed-form). Assumes a
      zero baseline and uses the amplitude cutoff.
    - 'nonlinear' (Frisken 2001): fit g2 - 1 directly by nonlinear least squares
      with a FLOATING baseline (moments-about-the-mean form), fitting the whole
      windowed correlogram -- the amplitude cutoff does not select the channels it
      REGRESSES, though it does reach this arm two other ways (see `fit_cutoff`).
      More robust to baseline drift and noisy/low-count data. (See the module
      docstring for why the engine default is 'linear' while the GUI seeds
      'nonlinear'.)

    Both return the same CumulantResult. Per invariant 8 neither reports a +/-
    from a single correlogram (lag channels are correlated, Schaetzel 1990);
    replicate averaging is the uncertainty source.

    Parameters
    ----------
    measurement : DLSMeasurement
    order : int
        Cumulant order: 1, 2 (default), 3 or 4. Orders 3 and 4 add the third and fourth
        central moments (`mu3_s_inv3`, `mu4_s_inv4`); which order suits which sample, and
        why the higher moments are diagnostics rather than measurements, is on
        `CumulantResult` and in the Theory-and-Equations-Guide's "Orders 3 and 4: the
        higher moments".
    fit_cutoff : float
        The amplitude threshold that bounds the RETAINED RANGE (default
        CUMULANT_FIT_CUTOFF_DEFAULT, 0.1), and a
        range is what it selects on both arms -- not a per-channel filter. ISO
        22412:2017 Annex A.1.2 (prose prescription, printed p. 20) states the retained
        values as "a range in [G2(tau_j) - A]", and Schaetzel (1990) p. 293 and Mailer,
        Clegg & Pusey (2015) Sec. 3 p. 7-8 both describe the same rule as a truncation.
        The range is the contiguous run of channels above `fit_cutoff * peak` that
        contains the peak; see `_retained_range_mask`.
        Per arm, and this parameter is NOT inert on either one. The LINEAR method
        regresses that range -- its only use of the cutoff. The NONLINEAR method does not
        select the channels it REGRESSES this way (it fits every live channel of the
        windowed correlogram), but reads the cutoff twice all the same:
          * `rms_error` is reported over the same retained range, so the two methods'
            goodness-of-fit numbers are comparable (decision D3). Measured over the 190
            committed DLS-SLS-Pairs runs, 1/10 -> 1/100 moves it on 188 of them (median
            +0.4 %, range -34 % to +248 %).
          * the SEED is a linear fit taken at this same cutoff, supplying `curve_fit`'s
            starting vector `beta0/gamma0` and `m2_0`/`m3_0`/`m4_0`, the rescaled
            moments -- see `_frisken_model_scaled_moments` -- under bounds. Measured
            over 288 grid cells the resulting Rh spread is median 0.0000 / p95 0.0002 /
            max 0.0670 -- real, but nearly always inert (measured under the legacy
            weight). Under the
            sourced weight the cutoff also seeds the order-1 first pass the variance is
            evaluated at, and on the widest planted case -- a bimodal four decades wide,
            outside the expansion -- Gamma moves with it at the low orders by percents
            rather than the parts per million it moves elsewhere (measured under the
            Gamma << 1 closed form, 2026-09-08; the pin the suite carries is the COUNT of
            such draws per case, re-measured under the exact form, not these percentages).
        Both are pinned (`test_the_nonlinear_arms_rms_error_moves_with_the_cutoff_on_a_`
        `committed_run`, `test_the_nonlinear_arm_seeds_itself_at_the_callers_cutoff`).
        CANONICAL-METHOD ANCHOR (CLAUDE.md): the default 1/10 is a BENCHMARKED DEVIATION
        from ISO 22412:2017 Annex A.1.2, whose canonical figure is 1/100 -- not a
        convenience simplification. Its benchmark is Mailer, Clegg & Pusey (2015) Sec. 3
        p. 7-8, whose linear cumulant analysis truncates at exactly 10 % of the initial
        amplitude in a comparison against distributions of known width; the 190-run corpus
        measurement in the Theory guide corroborates it and is not the warrant.
        The deviation is bounded by what else of A.1.2 is implemented. Its 1/50
        requirement IS (`MAX_RETAINED_ACF_RATIO` and `MIN_DECAY_COMPLETENESS` in
        `_common.py`, the data form and the rate form of the same clause). Two are NOT:
        the >= 20 retained points, and the all-retained-values-positive clause as the
        Annex states it, whose remedy is to discard the MEASUREMENT and so collides with
        this program's flag-not-refuse policy (CLAUDE.md invariant 4). A.1.2 reaches 1/100
        only on measurements it has already refused; this program refuses none, which is
        why the Annex's own figure is offered rather than defaulted to.
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
    baseline_bound_rel_floor : float
        Nonlinear method only: the floor of the floating baseline's bound as a fraction
        of the windowed correlogram's peak (default FRISKEN_BASELINE_BOUND_REL_FLOOR);
        see `_frisken_baseline_bound`. Ignored by the linear method.
    baseline : float, optional
        A baseline the caller determined (the Correlogram tab's baseline region). The
        linear method subtracts it from g2 - 1 before taking the log -- ISO 22412:2017
        Annex A.1.2's own linearization, Formula (A.16), whose baseline A is determined
        beforehand -- and records it on the result. The NONLINEAR method ignores it
        except through its linear seed: nothing is subtracted there, because Frisken's B
        floats. A caller-supplied VALUE was tried as that bound's centre and measured to move
        nothing when the region measures the tail and to hand back a green wrong size
        when it does not (a centre above the windowed data pins B above every fitted
        point), so `baseline` never reaches it. The bound's centre is always the mean of
        the tail slice -- which `baseline_window` below can redefine, and that is the
        owner-ruled route by which the user's region reaches this arm. Default None: the
        linear method subtracts nothing.
    baseline_window : (float, float), optional
        Nonlinear method only, and a DIFFERENT quantity from `baseline`: a delay window
        `(lo_s, hi_s)` in seconds naming WHERE the floating baseline's tail is measured.
        It is never subtracted. The channels inside it replace the index tail as the
        slice `_frisken_baseline_bound` is sized from, so B still floats within five
        times that slice's own scatter (floored at `baseline_bound_rel_floor` of the
        peak, capped at the peak) about its mean. The fit's SHAPE is unchanged -- the
        model, the bounds' form and the cutoff are all as before -- but two things move
        with the slice, because both are built from its mean `B0`: the bound's CENTRE and
        the initial guess `p0[0]`, of the order-n fit and, under the sourced weight, of
        the first pass its variance is evaluated at. That is the whole reach of this
        kwarg, and it is why a region placed off the tail can move the reported size.
        Default None: the index tail, exactly as before. Two kwargs and not one because
        the two meanings are different -- a value to subtract cannot be turned back into
        a window, and each fitter reads only the one it understands. A window holding
        fewer than two channels is refused, naming the window (see `_tail_for_bound`).
        CANONICAL-METHOD ANCHOR (CLAUDE.md): the index-tail default is the flagged
        convenience simplification of ISO 22412:2017 Annex A.1.2 documented on
        `_frisken_baseline_bound`; the user-window form is a HOUSE EXTENSION of Frisken
        2001, which floats B about "the baseline" without prescribing where it is
        measured -- never a claim to be the standard.
    weighting : str, optional
        WHAT A RESIDUAL IS MEASURED IN, and the one thing on this arm that is recorded on
        the result as provenance (`CumulantResult.weighting`). Default None means the
        ARM's default: `CUMULANT_NONLINEAR_WEIGHTING_DEFAULT` on the nonlinear arm -- the
        one place that choice is made -- and `LINEAR_WEIGHTING` on the linear arm. A token
        belonging to the other arm is refused rather than ignored (`_resolve_weighting`).
        The nonlinear arm's roster, in full on `_fit_cumulants_nonlinear`:
          * 'biganzoli' -- the sourced per-channel variance, Biganzoli and Ferri (2018)
            Eq. (9) with Eq. (10), printed p. 29378, exact in Gamma, evaluated on a
            fitted single exponential per Koppel (1972) printed p. 4818, through
            `_biganzoli_sd`. Recorded as 'biganzoli-full' when the measurement carries a
            mean count rate and 'biganzoli-classical' when it does not.
          * 'instrument' -- Frisken (2001)'s own route, printed p. 4089: the weights come
            from the instrument's reported per-channel standard deviations, which this
            program cannot read from any file, so `channel_sigma` supplies them.
          * 'uniform' -- no weight: unweighted least squares on g2-1. A flagged
            simplification offered as a labelled opt-in (it is the absence of a noise
            model, and the sourced variance stays the default); no first pass runs.
        Every fit in the arm runs the trust region in Jacobian-scaled units whatever the
        weighting (`_fit_cumulants_nonlinear`, THE SOLVER'S SCALE), so two weightings
        differ in the weight alone. Under the DEFAULT request, a sourced weight that Eq. (9)
        refuses to form falls back to 'uniform', recorded, flagged and named in the note.
        The linear arm's default, 'iso-a18', is the program's form of the weight ISO
        22412:2017 Annex A.1.2 prescribes below Formula (A.18), printed p. 20, which prints
        no form: (g2-1)^2 on each squared residual; its 'uniform' is the same opt-in on ITS residual,
        unweighted least squares on ln(g2-1). Neither is changed by any of the above.
    channel_sigma : np.ndarray, optional
        Per-channel standard deviations for `weighting='instrument'`, on the FULL delay
        axis `measurement.delay_times_s` -- the engine cuts them to the fitted channels
        with the same `_apply_tau_window` call the data goes through, so the two name one
        channel set. Required by that weighting and refused under every other one, since a
        weight silently discarded is worse than one refused. Every entry must be finite and
        strictly positive: only ratios matter (`absolute_sigma=False`), so the units are
        free, but a zero or a NaN is not a standard deviation.

    The three instrument inputs the sourced weight reads are NOT parameters here: they
    travel on the measurement, so every fit of one run reads the same inputs and no
    caller can drop one. `measurement.mean_count_rate_cps` (set by
    `LoadedMeasurement.build()` from the committed value; None when nothing usable was
    committed) turns Eq. (9)'s two shot-noise brackets on under `weighting='biganzoli'`
    (the per-block substitution n_bar_k = <c.r.> * Delta_t_k of Biganzoli and Ferri
    2018, printed p. 29381) and so selects 'biganzoli-full' over 'biganzoli-classical';
    None is the classical infinite-count-rate limit -- the honest fallback for a run
    with no rate, not a guess. `measurement.sample_times_s`, when a parser recorded one,
    is the per-channel sample time the same equation is evaluated on; otherwise a
    stand-in is derived from the delay axis, and `CumulantResult.sample_time_provenance`
    records which. `measurement.duration_s`, when the file states the run duration T,
    enters Eq. (9)'s 1/(M - k) as Delta_t_k / (T - tau_k); None is the T -> infinity
    form, and `CumulantResult.run_duration_form` records which. The weightings that do
    not read them ('instrument', 'uniform', the whole linear arm) leave all three untouched.

    Returns
    -------
    CumulantResult

    Raises
    ------
    ValueError
        If order is not 1/2/3/4, `fit_cutoff` is outside (0, 1), method is unknown, or
        too few points survive; and for every unusable combination of `weighting` and
        `channel_sigma`, raised before either arm is dispatched and naming the parameter
        to change.
    """
    # Orders 1-4. Order 4 is where the literature stops: Mailer, Clegg & Pusey (2015)
    # p. 14 report that even fourth-order fits do not recover mu3 or mu4 reliably, so no
    # fifth moment is fittable and none is offered.
    if order not in (1, 2, 3, 4):
        raise ValueError(f"order must be 1, 2, 3, or 4, got {order!r}.")
    # Refused, never clamped, and refused HERE rather than at the widget: the engine is
    # the one boundary every caller of a now user-settable value crosses. A cutoff at or
    # above 1 retains nothing but the peak (the threshold is `cutoff * peak`); one at or
    # below 0 makes the range meaningless and, at 0 exactly, silently retains every
    # positive channel -- the per-channel reading this range rule replaced, back through
    # the front door. The predicate reads the RAW parameter, not the product, so there is
    # no gap for a float tail to land in (CLAUDE.md rule 12).
    if not (isinstance(fit_cutoff, (int, float)) and not isinstance(fit_cutoff, bool)
            and math.isfinite(fit_cutoff) and 0.0 < float(fit_cutoff) < 1.0):
        raise ValueError(
            f"fit_cutoff must be a finite fraction strictly between 0 and 1, got "
            f"{fit_cutoff!r}. It is the amplitude at which the retained range stops, as "
            f"a fraction of the correlogram's peak; ISO 22412:2017 Annex A.1.2 "
            f"prescribes 0.01 and this program defaults to 0.1.")
    # The weighting parameters are refused HERE, before either arm is dispatched, for the
    # reason the cutoff is: this is the one boundary every caller crosses, and a weight is
    # not a thing to repair on the caller's behalf. `weighting=None` resolves to the ARM's
    # default first, so every check below reads the value the fit will actually use.
    resolved = _resolve_weighting(method, weighting)
    if channel_sigma is not None and resolved != 'instrument':
        raise ValueError(
            f"channel_sigma was supplied but weighting={resolved!r} does not read it, so "
            f"the standard deviations would be silently discarded. Pass "
            f"weighting='instrument' to weight by them, or drop channel_sigma.")
    if resolved == 'instrument':
        if channel_sigma is None:
            raise ValueError(
                "weighting='instrument' weights by the instrument's own reported "
                "per-channel standard deviations (Frisken 2001), which no parser here "
                "reads from a file: supply them as channel_sigma on the full delay axis, "
                "or choose another weighting.")
        sig = np.asarray(channel_sigma, dtype=float)
        n_axis = int(np.asarray(measurement.delay_times_s).size)
        if sig.ndim != 1 or sig.size != n_axis:
            raise ValueError(
                f"channel_sigma must be a 1-D array with one entry per delay channel of "
                f"this measurement ({n_axis}), got shape {sig.shape}. It is given on the "
                f"FULL delay axis and windowed inside the fit exactly as the correlogram "
                f"is, so a pre-windowed array is the wrong length here.")
        if not (np.all(np.isfinite(sig)) and np.all(sig > 0.0)):
            raise ValueError(
                "channel_sigma must be finite and strictly positive on every channel: a "
                "zero, a negative value or a NaN is not a standard deviation, and only "
                "ratios between the entries reach the fit. Drop the affected channels "
                "with the delay window, or supply a usable value for them.")
    if method == 'nonlinear':
        return _fit_cumulants_nonlinear(
            measurement, order, fit_cutoff, tau_min_s, tau_max_s, skip_initial_channels,
            max_rh_nm, max_retained_acf_ratio, baseline_bound_rel_floor, baseline,
            baseline_window, resolved, channel_sigma)
    if method == 'linear':
        return _fit_cumulants_linear(
            measurement, order, fit_cutoff, tau_min_s, tau_max_s, skip_initial_channels,
            max_rh_nm, max_retained_acf_ratio, baseline, resolved)
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
    baseline: Optional[float] = None,
    weighting: str = LINEAR_WEIGHTING,
) -> CumulantResult:
    """Linear/log cumulant (Koppel 1972): weighted polynomial fit of ln(g2-1).

    In the Siegert convention
        ln(g2 - 1 - baseline) = ln(beta) - 2 Gamma tau + mu2 tau^2 - (mu3/3) tau^3 + ...
    so the polynomial coefficients map directly to the cumulants. By default each residual
    is multiplied by its point's g2 - 1, so its square carries (g2 - 1)^2, the inverse
    variance of ln(g2 - 1) under a noise constant across channels -- the program's own
    derivation of the normalized weight ISO 22412:2017 Annex A.1.2 prescribes below
    Formula (A.18), printed p. 20, which prints no form; recorded on the result as
    `LINEAR_WEIGHTING`; under
    `weighting='uniform'` they are not weighted at all (the helper's own `w=None`), and
    the result records that instead -- and the regressed set is the
    contiguous RANGE bounded below by
    `fit_cutoff` times the windowed peak -- not every point above it, which on a noisy
    tail keeps the upward excursions and regresses the rectification as a slow mode. See
    `_retained_range_mask`. A caller-supplied `baseline` is subtracted before the log
    (ISO 22412:2017 Formula A.16, the baseline determined beforehand); with none the
    baseline is zero, as Koppel's normalized signal is by construction.

    WHEN NOTHING SURVIVES, two different situations reach the same point and get
    different answers (round 1 finding 8, owner-ruled 2026-09-05):

      - The CALLER's baseline sits at or above the windowed peak, so the subtraction
        empties the range on its own and no cutoff can rescue it. That is an INPUT ERROR
        the user can act on, so it RAISES with a message naming the remedy.
      - Otherwise the measurement simply has no contiguous range above the cutoff -- a
        correlogram with no decay (pure noise, a solvent blank). That is a PROPERTY OF
        THE DATA, which is what a reliability flag exists to report, so it returns the
        D2-shaped NaN result its sibling `_fit_cumulants_nonlinear` already returns for
        its own unfittable case: success=False, NaN physical outputs, and a note naming
        the precondition that failed (invariant 4 clause 1).

    Withholding the second case by exception instead made every consumer carry its own
    `try`, and the pooled Gamma-vs-q^2 and D-vs-c fits -- whose documented contract is to
    DROP an unusable point and disclose it -- aborted entirely on one bad angle.

    THE FOURTH-ORDER COEFFICIENT, taken from the page and not re-derived here. ln(g2-1) is
    ln beta + 2 ln g1, and the cumulant expansion of ln g1 carries kappa4/4! on tau^4 --
    Frisken 2001 Eq. 16 (printed p. 4088); Mailer, Clegg & Pusey 2015 Eq. 12 (preprint
    p. 5), whose tau^4 coefficient is (mu4 - 3 mu2^2)/4!. So c4 = 2 kappa4/24 = kappa4/12,
    and what this regression yields at fourth order is the fourth CUMULANT. The result
    field is the fourth CENTRAL MOMENT -- the quantity the nonlinear arm fits directly
    (Mailer Eq. 13, same page) -- so it is converted, mu4 = kappa4 + 3 mu2^2 (the two
    differ only from order 4 on: the terminology note, preprint p. 6), and the two arms
    report one quantity under one name. Left unconverted they would disagree by the factor
    kappa4/mu4 = 2 sigma^2/(1 + 2 sigma^2) on a Schulz distribution (Mailer Eqs. 18-19)
    with no error and no flag -- the invariant-7 conflation class on a new field. The
    converted value is not bounded (mu2 is not either, on this arm) and under realistic
    noise is mostly noise itself -- see `CumulantResult`.
    """
    if weighting not in LINEAR_WEIGHTINGS:
        # `fit_cumulants` refuses every other token before this arm is dispatched; a private
        # caller that bypassed it gets the same refusal rather than a mis-recorded weight.
        raise ValueError(
            f"weighting must be one of {sorted(LINEAR_WEIGHTINGS)} on the linear arm, got "
            f"{weighting!r}.")
    tau, g2m1 = _apply_tau_window(
        measurement.delay_times_s, measurement.correlogram,
        tau_min_s, tau_max_s, min_points=order + 1,
        skip_initial_channels=skip_initial_channels,
    )
    offset = _pinned_offset(baseline)
    signal = g2m1 - offset

    # A RANGE, not a per-channel filter -- ISO 22412:2017 A.1.2's own word, and the
    # termination Schaetzel (1990) p. 293 and Mailer (2015) Sec. 3 p. 7-8 both describe.
    # See `_retained_range_mask` for the mechanism, the measurement and why the interval
    # is anchored on the peak.
    mask = _retained_range_mask(signal, fit_cutoff)
    # Counted on the RAW windowed g2-1, never on the offset-subtracted `signal`: a dead
    # channel is a property of the measurement (g2 = 0 -- nothing was counted), not of
    # what a caller chose to subtract afterwards. Counted before the amplitude cutoff is
    # folded in, and reported on its own: the channels the cutoff discards are this
    # method's own choice of what to regress, and joining the two into one number would
    # put the routine under the same word as the pathological.
    alive = _live_channels(g2m1)
    n_dead = int(np.count_nonzero(~alive))
    t = tau[mask]
    y = signal[mask]
    if t.size < order + 1:
        # An input error RAISES; a property of the data is FLAGGED -- this function's
        # docstring says which is which and why. Asked as `baseline is not None`, never
        # as `offset >= peak` alone: with no baseline supplied `offset` is 0.0, so a
        # non-positive-peak window would be blamed on a baseline the caller never set.
        peak_raw = float(g2m1.max())
        if baseline is not None and offset >= peak_raw:
            raise ValueError(
                f"Only {t.size} points survive the cutoff; need at least "
                f"{order + 1} for an order-{order} fit. The supplied baseline "
                f"({offset:.3g}) is at or above the windowed correlogram's peak "
                f"({peak_raw:.3g}), so no point survives the subtraction: move the "
                f"baseline region onto the tail."
            )
        # The note is PARAMETRISED on what actually happened rather than asserting a
        # cause, because the two causes that reach here are not separable by the range's
        # size -- see `_retained_range_too_short_note`. `decay_incomplete` stays True: it
        # is the honest reading of 'no range long enough survived', and the alternative
        # (keying it off the size) would carry the same false precision the note dropped.
        nan = float('nan')
        notes, note_targets = _keyed_notes(
            ('', _dead_channel_note(n_dead, int(g2m1.size))))
        return CumulantResult(
            order=order, beta=nan, gamma_s_inv=nan, mu2_s_inv2=nan, mu3_s_inv3=None,
            mu4_s_inv4=None,
            pdi=nan, pdi_valid=False, d_m2_s=nan, rh_nm=nan,
            q_m_inv=_measurement_q_m(measurement),
            fit_cutoff=fit_cutoff, n_points_used=int(t.size),
            coefficients=np.full(order + 1, nan),
            # The WINDOWED axis, not the (possibly empty) retained one, so this failure
            # exports the same SHAPE as `_fit_cumulants_nonlinear`'s D2 failure: both keep
            # the delay axis and NaN the curve. Handing back `t` here wrote zero-row curve
            # columns beside length-1 scalars, so Origin got a structurally different file
            # for the two arms on the same failure class. `n_points_used` still reports the
            # RETAINED count -- that is the diagnostic number, and it stays honest at 0.
            fit_tau_s=tau, fitted_g2m1=np.full(tau.shape, nan),
            residuals=np.full(tau.shape, nan), rms_error=nan,
            weighting=weighting,
            sample_time_provenance='',    # the linear arm reads no sample time
            run_duration_form='',         # nor a run duration
            n_skipped=int(skip_initial_channels or 0),
            method='linear', baseline=offset, success=False,
            rh_reliable=False,
            reliability_note=_retained_range_too_short_note(
                int(t.size), order, fit_cutoff),
            decay_incomplete=True,
            notes=notes, note_targets=note_targets,
        )

    logy = np.log(y)
    # Weights multiply the residuals (so the squared residual carries y^2); by default
    # weight by signal amplitude, and under
    # 'uniform' by nothing (`w=None`, the helper's own default). Routed through
    # polyfit_or_nan because the WEIGHT is a third non-finite channel that makes the bare
    # np.polyfit raise (unlike the response, which merely returns NaN).
    coeffs_high_first = unc.polyfit_or_nan(t, logy, order,
                                           w=None if weighting == 'uniform' else y)
    c = coeffs_high_first[::-1]   # [c0, c1, c2, (c3, c4)]

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
    # THE ORDER-4 CONVERSION: c4 is the fourth CUMULANT / 12, the field is the fourth
    # CENTRAL MOMENT -- the page loci and the reason are in this function's docstring
    # ("The fourth-order coefficient"). `mu2 * mu2`, never `mu2 ** 2` (CLAUDE.md rule 12:
    # a Python-float `**` raises where `*` saturates).
    mu4 = (12.0 * c[4] + 3.0 * mu2 * mu2) if order >= 4 else None
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
    fitted = np.exp(model_log) + offset
    residuals = g2m1[mask] - fitted

    # Preconditions are judged on the WINDOWED correlogram `g2m1`, not on the cutoff
    # subset `y` the regression saw: ISO's retained-range rule binds the data the fit
    # was given, and the amplitude cutoff is this method's own choice of what to
    # regress (see _cumulant_flags). The floor is a FLAG here, where the nonlinear path
    # rejects (owner decision D2, CUMULANT_MIN_RH_NM). The LIVE channels of that window:
    # a dead channel carries no observation of the decay, so it cannot testify that the
    # signal reached its floor (round 4 finding 1 -- twelve trailing dead channels turned
    # an undecayed, 4x-wrong fit green).
    rh_reliable, reliability_note, decay_incomplete = _cumulant_flags(
        rh, gamma, g2m1[alive], pdi, pdi_valid, visc_note,
        max_rh_nm, CUMULANT_MIN_RH_NM, max_retained_acf_ratio)

    notes, note_targets = _keyed_notes(
        ('', _dead_channel_note(n_dead, int(g2m1.size))))
    return CumulantResult(
        order=order,
        beta=beta,
        gamma_s_inv=gamma,
        mu2_s_inv2=mu2,
        mu3_s_inv3=mu3,
        mu4_s_inv4=mu4,
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
        weighting=weighting,
        sample_time_provenance='',    # the linear arm reads no sample time
        run_duration_form='',         # nor a run duration
        n_skipped=int(skip_initial_channels or 0),
        method='linear',
        baseline=offset,
        success=fit_ok,
        rh_reliable=rh_reliable,
        reliability_note=reliability_note,
        decay_incomplete=decay_incomplete,
        notes=notes, note_targets=note_targets,
    )


def _frisken_model(tau, B, beta, gamma, mu2, mu3, mu4):
    """Frisken (2001) Eq. 23 cumulant model (moments about the mean), carried to the
    fourth moment as Mailer, Clegg & Pusey (2015) Eq. 13 (preprint p. 5) print it:

        g2-1 = B + beta * [ exp(-gamma tau)
                            (1 + (mu2/2) tau^2 - (mu3/6) tau^3 + (mu4/24) tau^4) ]^2

    The square is the Siegert relation; expanding about the mean rate (rather than
    about tau=0 as the log-polynomial does) keeps the fit stable at large tau. Frisken's
    own Eq. 22-23 (printed p. 4089) print the series to mu3 and an ellipsis; the tau^4
    coefficient mu4/4! is read from Mailer Eq. 13, which restates Frisken's form one
    order further, and it is the fourth CENTRAL MOMENT -- not the cumulant kappa4 that
    the log-polynomial's quartic coefficient carries (Mailer Eq. 12; the two differ only
    from order 4 on, terminology note p. 6). Every moment is a required argument: a
    caller states the truncation it wants by passing a literal 0.0 for the moments above
    it, so no order is ever evaluated silently.
    """
    inner = np.exp(-gamma * tau) * (1.0 + 0.5 * mu2 * tau ** 2 - (mu3 / 6.0) * tau ** 3
                                    + (mu4 / 24.0) * tau ** 4)
    return B + beta * inner * inner


def _frisken_model_scaled_moments(tau, B, beta, gamma, m2, m3, m4):
    """The SAME model, with ALL THREE moments carried dimensionless,
    ``m2 = mu2 / Gamma^2``, ``m3 = mu3 / Gamma^3`` and ``m4 = mu4 / Gamma^4``.

    The fitted MODEL does not change -- this is an exact algebraic reparametrisation of
    `_frisken_model`, which it calls -- but which vector the optimiser searches does, and
    that is the point. The same argument covers every moment, each higher one worse: the
    natural scale of mu_n is Gamma^n, so mu4's sits four decades past mu3's.

    THE DEFECT. `curve_fit` has no analytic derivatives, so it estimates each one by a
    finite difference of step ``sqrt(eps) * max(1, |p|)``. With ``tau ~ 1/Gamma``, the term
    ``(mu3/6) tau^3`` is O(1) only when ``mu3 ~ Gamma^3`` -- on the committed corpus
    ``mu3 ~ 1e11``. A parameter of that natural magnitude sitting NEAR ZERO is perturbed by
    1.49e-8 ABSOLUTE, the model moves by ~1e-20, and the difference rounds to exactly 0.0:
    the Jacobian column is dead and the optimiser concludes the parameter does not affect
    the fit. Measured 2026-09-06 over the 190 committed runs, order 3: with mu3 seeded at
    zero the column was identically zero on 129 and the fit returned mu3 unchanged on 127;
    the fitted mu3 depended on its seed on 143 unconditionally.

    WHY RESCALING FIXES IT -- and NOT for the reason it is tempting to give. The step does
    NOT become relative: scipy's rule is ``rel_step * max(1, |x|)``, so it stays 1.49e-8
    ABSOLUTE for every ``|p| <= 1``, which is the whole range this rescaling targets.
    Verified on the pinned scipy (1.17.1): ``_compute_absolute_step`` returns 1.49012e-08
    identically at x = 0.0, 1e-10, 1e-3, 0.3 and 1.0, and turns relative only above x ~ 1e7.
    What changes is the SENSITIVITY, not the step: after rescaling ``d(model)/d(m3)`` is
    O(1) (peak ~0.22), so the same 1.49e-8 absolute perturbation moves ``g2-1`` by ~5e-10 --
    about 1e7 times the ULP of an O(1) correlation value -- instead of ~1e-20, which is
    below it. A dead column becomes a live one because the model now responds to the step,
    not because the step grew. (An earlier draft of this docstring claimed the step becomes
    relative. It does not, and a maintainer who believed it would wrongly conclude that
    pushing any parameter below 1 makes it safe.)

    This is the canonical remedy for a badly-scaled problem, not a local trick: Dennis &
    Schnabel (1996) sec. 7.1 (printed pp. 155-157) prescribes rescaling the variables --
    "change their units" (p. 156) -- so that each transformed variable has magnitude
    "about 1" (p. 157). It is applied one stage EARLIER than scipy's ``x_scale``, which
    conditions the trust region and the norms but not the difference estimate: a Jacobian
    column that has already rounded to zero stays zero however the step is subsequently
    scaled, and ``x_scale='jac'`` was measured ineffective against THIS defect (120 of 190
    still frozen) even though it is this codebase's SLS remedy. It is the remedy for the
    OTHER half -- the step -- and since 2026-09-08 every fit in the nonlinear arm runs under
    it for exactly that reason (`_fit_cumulants_nonlinear`, THE SOLVER'S SCALE). It is also the
    form the DLS literature writes these moments in (Mailer et al. 2015 Eq. 18).

    mu2 TOO: it is carried dimensionless like m3 and m4, the one convention Dennis &
    Schnabel (1996) sec. 7.1 prescribe for every variable. The two parametrisations share
    one optimum -- refitted at `ftol=xtol=gtol=1e-14` they agree in Gamma to rounding --
    and the rescaled search stops nearer it.

    Gamma is the LIVE fitted parameter, not the seed, so the conversion is exact at every
    step of the search rather than only at the solution. At order 1 no moment is fitted and
    the moment arguments are literal 0.0, so the model reduces to `_frisken_model` exactly
    -- with one caveat worth stating rather than leaving to be discovered:
    ``0.0 * power_or_nan(...)`` is NaN once ``Gamma^n`` stops being representable
    (measured: ``power_or_nan(1e103, 3)`` is already NaN, and ``Gamma^4`` reaches that
    tail sooner -- ``power_or_nan(1e77, 4)`` is finite, ``power_or_nan(1e78, 4)`` NaN), so
    every order carries a Gamma coupling at that UPPER tail (the lower one, Gamma^4
    underflowing, sits near 1e-77, sixty-eight decades below Gamma's 1e-9 bound).
    It is a documented tail, not a live path, because Gamma is a decay rate: the fastest
    DLS relaxation any correlator resolves is ~1e8 1/s (a 25 ns first channel), seventy
    decades short of 1e78, and the optimiser's Gamma is seeded from the data, so no
    realistic or adversarial correlogram drives it there.
    """
    return _frisken_model(tau, B, beta, gamma,
                          m2 * unc.power_or_nan(gamma, 2),
                          m3 * unc.power_or_nan(gamma, 3),
                          m4 * unc.power_or_nan(gamma, 4))


def _sourced_sigma(tau_w: np.ndarray, dt_w: np.ndarray, gamma_bar: float, beta: float,
                   mean_count_rate_cps: Optional[float],
                   duration_s: Optional[float]) -> np.ndarray:
    """The nonlinear cumulant's SOURCED residual weight: the relative per-channel SD of
    Biganzoli and Ferri (2018) Eq. (9) with Eq. (10), printed p. 29378, exact in Gamma,
    evaluated on a single exponential.

    A thin, testable wrapper over `_biganzoli_sd` -- the ONE implementation of that equation
    in the tree, which the synthetic generator's two 'biganzoli_*' noise models also draw
    through, so a planted correlogram is an oracle for the weight built from it: exact on
    every channel of the multi-tau builder's layout, the paper's (see
    `channel_sample_times` for the one counted exception). Nothing about the equation is
    restated here; `_biganzoli_sd` is its home.

    `gamma_bar` and `beta` are Koppel's evaluation point (1972, the prose prescription
    after Eq. 26, printed p. 4818): the variance belongs to the BEST-FIT single exponential,
    never to the noisy observation it weights. `dt_w` is the per-channel sample time -- the
    instrument's own when the parser recorded one (`DLSMeasurement.sample_times_s`), else
    read off the delay axis by `channel_sample_times`. `mean_count_rate_cps` is None for
    the classical infinite-count-rate limit or a committed rate, which adds Eq. (9)'s two
    shot-noise brackets through the per-block substitution n_bar_k = <c.r.> * Delta_t_k
    (printed p. 29381). `duration_s` is the run duration T, which enters Eq. (9)'s
    1/(M - k) as Delta_t_k / (T - tau_k), or None for the T -> infinity form. Every array
    is already cut to the fitted channels; this function does no windowing of its own.

    REFUSES a sigma with any non-finite or non-positive entry, naming the inputs. The guard
    is on the RESULT rather than on the inputs that produce it, because saturation is a
    property of the product (rule 12): `curve_fit` needs every entry finite and strictly
    positive, and that is exactly the predicate. The caller falls back to the unweighted fit
    on the refusal (`_fit_cumulants_nonlinear`, "A weight that cannot be formed").
    """
    sigma = _biganzoli_sd(tau_w, dt_w, gamma_bar, beta,
                          mean_count_rate_cps=mean_count_rate_cps, duration_s=duration_s)
    if not (np.all(np.isfinite(sigma)) and np.all(sigma > 0.0)):
        raise ValueError(
            f"the sourced residual weight is not usable on every fitted channel: "
            f"Biganzoli and Ferri Eq. (9) at gamma_bar = {gamma_bar!r} 1/s, "
            f"beta = {beta!r}, mean_count_rate_cps = {mean_count_rate_cps!r} and "
            f"duration_s = {duration_s!r} gives a per-channel SD with a non-finite or "
            f"non-positive entry, and a weight must be finite and strictly positive on "
            f"every channel it weights."
        )
    return sigma


def _fit_cumulants_nonlinear(
    measurement: DLSMeasurement,
    order: int,
    fit_cutoff: float,
    tau_min_s: Optional[float],
    tau_max_s: Optional[float],
    skip_initial_channels: int,
    max_rh_nm: float = CUMULANT_MAX_RH_NM,
    max_retained_acf_ratio: float = MAX_RETAINED_ACF_RATIO,
    baseline_bound_rel_floor: float = FRISKEN_BASELINE_BOUND_REL_FLOOR,
    baseline: Optional[float] = None,
    baseline_window: Optional[tuple] = None,
    weighting: str = CUMULANT_NONLINEAR_WEIGHTING_DEFAULT,
    channel_sigma: Optional[np.ndarray] = None,
) -> CumulantResult:
    """Nonlinear cumulant (Frisken 2001): direct g2-1 fit with a floating baseline.

    Fits every live channel of the windowed correlogram by weighted nonlinear least
    squares (a channel at g2-1 = -1 collected no coincidence and is dropped, with a note
    -- `_dead_channel_note`; the fitted curve is still drawn across it). The
    floating baseline B and coherence beta are free parameters; B is bounded about
    the mean of a TAIL slice by `_frisken_baseline_bound` (a caller's `baseline`
    reaches the linear seed only, never the bound's centre). Which channels that slice
    holds is the one thing `baseline_window` changes: the index tail with none supplied,
    the user's baseline region with one -- see `_tail_for_bound`. On non-convergence the
    result carries NaN physical outputs and success=False (owner decision D2); no SE is
    reported from one correlogram (invariant 8).

    THE RESIDUAL WEIGHT, by `weighting`, and it is recorded on the result. Only RATIOS of
    sigma reach the fit (`absolute_sigma=False`), so a relative per-channel shape is all
    either of the two has to supply -- which is also why a factor common to every channel
    cancels out of the sourced form, while the count rate and the run duration's
    per-channel remainder T - tau_k do not.

    'biganzoli' is the SOURCED variance: Biganzoli and Ferri (2018) Eq. (9) with Eq. (10),
    printed p. 29378, exact in Gamma, through `_biganzoli_sd` -- the one implementation of
    that equation in the tree, which the synthetic generator's two 'biganzoli_*' noise
    models also draw through, so a planted correlogram is an oracle for the weight built
    from it, exact on every channel of the multi-tau builder's layout, the paper's (see
    `channel_sample_times` for the one counted exception). It is evaluated on a FITTED
    SINGLE EXPONENTIAL, never on the noisy observation it weights: a first pass fits this
    same model at order 1 under uniform weights, and its Gamma and beta are the evaluation
    point Koppel (1972) prescribes in the prose after his Eq. 26, printed p. 4818. The
    per-channel sample time Eq. (9) needs is the INSTRUMENT'S OWN when the parser
    recorded one (`measurement.sample_times_s` -- the Brookhaven .dat states one per
    correlator section, and on that correlator the lag spacing is not the sample time),
    else a stand-in read off the delay axis by `channel_sample_times`; which one ran is
    recorded as `sample_time_provenance` ('instrument' / 'axis') beside the weighting,
    because the sourced weight on a stand-in is exact in a quantity the correlator may
    not have used. The measurement's `mean_count_rate_cps` (the committed value, via
    `LoadedMeasurement.build()`) adds Eq. (9)'s two shot-noise brackets through the
    per-block substitution n_bar_k = <c.r.> * Delta_t_k of Biganzoli and Ferri (2018),
    printed p. 29381, and is recorded as 'biganzoli-full'; without one the classical
    infinite-count-rate limit is used, recorded as 'biganzoli-classical'. The
    measurement's `duration_s`, when the file states the run duration T, enters Eq. (9)'s
    1/(M - k) as Delta_t_k / (T - tau_k); without one the T -> infinity form is used, and
    which ran is recorded as `run_duration_form` ('finite' / 'infinite').
    CANONICAL-METHOD ANCHOR (CLAUDE.md): the canonical form, diagonal -- four named
    practicality divergences. (i) Diagonal weighting, no whitening: the exact covariance
    across multi-tau stages has no in-library source (the paper restricts its Eq. (8) to
    same-stage pairs, pp. 29381 and 29387). (ii) The classical limit when no count rate is
    committed. (iii) The T -> infinity form when the measurement carries no run duration,
    recorded as `run_duration_form`. (iv) The large-Gamma asymptotic branch inside
    `_biganzoli_sd`. (ii)-(iv) are each a limit or a rewriting of Eq. (9), none a
    different model; Schaetzel (1990) Eq. (49), printed p. 296, is the Gamma << 1 limit
    the exact form reduces to. And one PREMISE, disclosed rather than chosen: the
    estimator Eq. (9) describes assumes symmetric normalization (`_biganzoli_sd` records
    where), and no parser here records which normalization an instrument used.

    'instrument' is Frisken's own route, printed p. 4089 ("weights were calculated from
    standard deviations provided by the ALV-Laser software"): the weights are the
    instrument's reported per-channel standard deviations, which no file this program
    reads carries, so the caller supplies them through `channel_sigma`.

    'uniform' is no weight at all: the order-n fit runs under `sigma=None`, unweighted least
    squares on g2-1, and no first pass runs, since nothing is evaluated. A flagged
    simplification (the absence of a noise model) offered as a labelled opt-in; the sourced
    variance stays the default, and nothing ranks the two.

    A WEIGHT THAT CANNOT BE FORMED, two cases (this paragraph is the rule's home).
    (1) NO FIT CAN RUN: too few live channels, or a first pass that fails or
    returns a non-finite Gamma or beta -- there is no evaluation point for the variance and
    nothing to fall back from, so the D2 return below reports it and still names the
    weighting it could not build. (2) EQ. (9) REFUSES ITS INPUTS: `_sourced_sigma` raises
    (its refusals are `_biganzoli_sd`'s, and its own of an unusable SD) after a first pass
    that did fit. The fit then FALLS BACK to
    'uniform' rather than withholding: the unweighted order-n fit runs, the result records
    'uniform' (so the export names it), the note names the refusal, and `rh_reliable` is
    False, because the default method's precondition -- a weight formed from the
    measurement's own inputs -- failed on this run (invariant 4: flag, don't refuse). The
    fallback reads no sample time and no run duration, so both provenance fields are ''.

    THE SOLVER'S SCALE IS ONE RULE (owner-ruled 2026-09-08; this paragraph is the ruling's
    home). Every `curve_fit` in this arm -- the first pass and the order-n fit, under every
    weighting and at every order -- runs the trust-region method with ``x_scale='jac'``: the
    trust region measured in the Jacobian's column norms, updated as the search proceeds --
    More (1978) sec. 6, Eq. (6.3) printed p. 112, ``d_i^(k) = max(d_i^(k-1), ||d_i F(x_k)||)``
    with ``D_k = diag(d_i^(k))`` (his Eq. (6.1), p. 111), the adaptive choice his results
    support over the initial and the un-smoothed ones. It is the diagonal scaling matrix
    Dennis & Schnabel (1996) sec. 7.1 (printed pp. 155-157) prescribe for a badly-scaled
    problem, set from the Jacobian rather than from user-supplied typical magnitudes, and
    the choice `analysis/sls.py` makes at every one of its sites. In raw units the
    six-parameter search under a sourced sigma crawled: a moment in its own units
    (natural scale Gamma^2) sat beside B ~ 1e-3 and the O(1) m3, m4, and on the 190
    committed corpus runs two
    order-4 fits exhausted the 10,000-evaluation budget (~4.8 s each, then the D2 failure)
    and three more needed 5,300-5,700 evaluations, against ~9 under the since-retired
    amplitude weight;
    scaled, no cell of the 570 needs more than a few dozen (measured 2026-09-08). This is
    the scaling `_frisken_model_scaled_moments` records as
    ineffective against the DEAD-COLUMN defect -- a derivative that has rounded to zero is
    zero at any scale -- which the dimensionless m3, m4 fixed; the crawl is the other half,
    the STEP, which is exactly what D_x conditions. WHAT THE SCALE IS NOT: a better
    optimiser. On an ill-determined fit -- a wide sample at an order the expansion does not
    describe -- the cost surface has more than one valley, and the scale changes which one
    the search descends into. Lower / higher / tied cost against the raw search, measured
    2026-09-08: 17 / 3 / 173 under the sourced weighting and 55 / 11 / 227 under the
    since-retired amplitude weight, of 570 corpus cells each; 0 / 3 / 44 and 2 / 0 / 69 of 153 planted-registry draws.
    The one material loss is the tail-heavy PDI-0.11 case at order 3, whose registered draw
    now parks mu2 on its bound, noted, where the raw search resolved 0.133 against a planted
    0.114. No exception is carved for it, for any weighting, or for the orders that
    happened not to crawl on the corpus: one rule.

    THE MOMENTS ARE FITTED DIMENSIONLESSLY (`m_n = mu_n / Gamma^n`, n = 2, 3, 4) and
    converted back with the FITTED Gamma before they are reported --
    `_frisken_model_scaled_moments` holds the reason and the citation (and the history of
    mu2's own units). The reported quantities, the bounds' meaning and the model are
    unchanged; what changes is that a moment near zero
    is REACHABLE, because the model's sensitivity to a parameter of order 1 is itself of
    order 1, where its sensitivity to a parameter of natural magnitude `Gamma^n` sitting
    near zero is below the ULP of the correlation value.

    ORDER 4 fits the fourth CENTRAL MOMENT mu4 directly (Mailer, Clegg & Pusey 2015
    Eq. 13, preprint p. 5); `CumulantResult.mu4_s_inv4` carries exactly that quantity,
    and the linear arm converts its own kappa4 to match. Its lower bound is a STATED
    choice, not an inherited one: mu4 = <(Gamma - Gbar)^4> over a distribution is an even
    central moment and so non-negative by construction, on the same ground as mu2 >= 0
    (Frisken 2001 printed p. 4089: "the first two cumulants must be positive, but the third
    cumulant can be positive or negative"). The tighter mu4 >= mu2^2 (Jensen) is not a
    box bound and is not imposed. Mailer sec. 4.7 (preprint p. 13) measures fourth-order
    fits as "not useful" at noise s = 1e-2 -- several percent in Gamma, +/-0.20 in mu2 --
    while second-order fits still gave Gamma to ~2 %; and their step (iv), p. 14, says
    that below mu2/Gbar^2 ~ 0.1 the second-order fit is the MORE precise. Neither is a
    per-run precondition this arm can test, so neither is a flag (invariant 4); both are
    stated in the order control's help and the Theory guide, which is where a constant
    property of a method belongs (owner-ruled 2026-09-05, see `CumulantResult`).
    """
    tau, g2m1 = _apply_tau_window(
        measurement.delay_times_s, measurement.correlogram,
        tau_min_s, tau_max_s, min_points=order + 3,
        skip_initial_channels=skip_initial_channels,
    )
    q = _measurement_q_m(measurement)
    # Dead channels (g2 = 0, stored as g2-1 = -1: no coincidence at all) leave the FIT --
    # the tail statistic, both passes and every weight -- and stay in the window for the
    # plotted curve, the residuals and the note that counts them. `_dead_channel_note`
    # records why: a channel with no estimate has no variance a weight could be stated
    # from, and fitted at full weight it drags the baseline to -1.
    alive = _live_channels(g2m1)
    tau_fit, g_fit = tau[alive], g2m1[alive]
    # Rule 12, sized to the operation: `_apply_tau_window` counted its `min_points` on the
    # raw window, but everything below runs on the live channels, so the same minimum is
    # re-derived on THEM. Too few live channels is the D2 failure return, never a bare
    # numpy error from an empty tail or a curve_fit with more parameters than points.
    enough_alive = tau_fit.size >= order + 3

    # Seed beta/gamma/mu2 from a quick linear fit on the same window AND AT THE CALLER'S
    # `fit_cutoff` -- the second of the two ways the cutoff reaches this arm, stated here
    # because the parameter doc once claimed it reached neither. It is `curve_fit`'s
    # starting vector under bounds, so it is not decorative: measured over 288 grid cells
    # the Rh spread it produces is median 0.0000 / max 0.0670 under the legacy weight;
    # under the sourced weight it also seeds the first pass, and the one planted case that
    # moves with it (a bimodal four decades wide, at order 3) is pinned in the test suite.
    # Baseline from the long-delay tail (nominally ~0 after the parser's subtraction).
    try:
        seed = _fit_cumulants_linear(
            measurement, order, fit_cutoff, tau_min_s, tau_max_s, skip_initial_channels,
            baseline=baseline)
        # `_seed_or_floor`, not `max(...)`: `max(nan, x)` returns nan, and the linear fit
        # this seeds from is exactly the one that can withhold beta.
        beta0 = _seed_or_floor(seed.beta, 1e-6)
        gamma0 = _seed_or_floor(seed.gamma_s_inv, 1e-3)
        mu2_0 = _seed_or_floor(seed.mu2_s_inv2, 0.0)
        # Neither `_seed_or_floor` nor `or 0.0`. `or 0.0` is NaN-blind (NaN is truthy), so
        # a withheld linear mu3 would reach `curve_fit`'s p0 and fail the fit. But the
        # helper the three siblings use is a FLOOR, and mu3 is the one seed here that is
        # legitimately SIGNED -- its bound below is -inf, where mu2's is 0.0. Flooring it
        # would clamp a negative seed to zero, the one value this fit cannot move off
        # (the difference step there is absolute, far below mu3's ~Gamma^3 scale), so the
        # floor helper would manufacture a frozen parameter. Test the two real properties.
        mu3_seed = seed.mu3_s_inv3
        mu3_0 = (float(mu3_seed)
                 if mu3_seed is not None and math.isfinite(mu3_seed) else 0.0)
        # mu4 IS floored, like mu2 and unlike mu3: it is bounded below at 0.0 (see the
        # branch table), so a negative linear-arm seed -- possible, since the converted
        # kappa4 + 3 mu2^2 inherits the log-polynomial's noise -- would start the search
        # outside its own feasible set. The seed is the linear arm's already-CONVERTED
        # central moment, so seed and result are one quantity.
        mu4_seed = seed.mu4_s_inv4
        mu4_0 = _seed_or_floor(float(mu4_seed), 0.0) if mu4_seed is not None else 0.0
    except Exception:  # noqa: BLE001 — seed fallback below
        beta0, gamma0, mu2_0, mu3_0, mu4_0 = float(g2m1.max()), 1.0, 0.0, 0.0, 0.0
    # mu2, mu3 and mu4 are carried dimensionless -- see `_frisken_model_scaled_moments`.
    # The seed converts here with the SEED Gamma and the result converts back below with
    # the FITTED one, both through the `power_or_nan` the model itself uses, so seed and
    # model are one computation with no second predicate to mis-size (rule 12). m2 is
    # floored at its 0.0 bound, as mu2_0 is; `_seed_or_floor` is deliberately NOT used on
    # m3, for the signed reason at `mu3_0` above. An unusable power yields NaN, which maps
    # to a 0.0 start -- safe now, and the one value the raw mu3 could not move off.
    m2_0 = _seed_or_floor(mu2_0 / unc.power_or_nan(gamma0, 2), 0.0)
    _m3_0 = mu3_0 / unc.power_or_nan(gamma0, 3)
    m3_0 = _m3_0 if math.isfinite(_m3_0) else 0.0
    _m4_0 = mu4_0 / unc.power_or_nan(gamma0, 4)
    m4_0 = _m4_0 if math.isfinite(_m4_0) else 0.0
    _pinned_offset(baseline)   # refused up front if non-finite, as on every other arm
    if enough_alive:
        tail = _tail_for_bound(tau_fit, g_fit, baseline_window)
        B0 = float(np.mean(tail))
    else:
        # No bound can be formed on too few live channels; NaN routes to the D2 return
        # exactly as a bound that could not be formed does below.
        tail = g_fit
        B0 = 0.0
    # The baseline is small drift around the (already ~0) long-time value, NOT a
    # free amplitude sink: bound it near the tail so the fit cannot absorb the
    # decay into B and chase a short-lag artifact (which would give a spurious
    # Rh ~ 0). The half-width is sized to the data's own scale -- see
    # _frisken_baseline_bound for why, and for the index-tail deviation from ISO.
    dB = (_frisken_baseline_bound(g_fit, tail, baseline_bound_rel_floor) if enough_alive
          else float('nan'))
    B_lo, B_hi = B0 - dB, B0 + dB

    # THE RESIDUAL WEIGHT, by the resolved `weighting` -- the one place this arm decides
    # what a residual is measured in. `sigma = None` is the UNWEIGHTED fit, asked for
    # ('uniform') or fallen back to; a weight that could not be formed so that no fit can
    # run is `weight_failed`, which the `ok` line below turns into the D2 failure. Two
    # names for two things, so the predicate and the operation it guards read one flag
    # (rule 12). What each branch IS, and the sources
    # behind it, are in this function's docstring. The recorded name is resolved first, so
    # every return path of this function carries the same provenance string.
    mean_count_rate_cps = measurement.mean_count_rate_cps
    duration_s = measurement.duration_s
    recorded = weighting
    sample_time_provenance = ''
    run_duration_form = ''
    weight_failed = False       # no weight AND no fit: the D2 return
    weight_refusal = ''         # Eq. (9)'s own refusal, when it refused; the fallback names it
    if weighting == 'biganzoli':
        recorded = ('biganzoli-full' if mean_count_rate_cps is not None
                    else 'biganzoli-classical')
        # WHICH per-channel sample time and which form of 1/(M - k) Eq. (9) is evaluated
        # on, resolved here so every return path carries them: the instrument's own sample
        # time when the parser recorded one, else the stand-in derived from the delay
        # axis; the finite run duration when the measurement carries one, else T -> inf.
        sample_time_provenance = ('instrument' if measurement.sample_times_s is not None
                                  else 'axis')
        run_duration_form = 'finite' if duration_s is not None else 'infinite'
    # THE SOLVER'S SCALE IS ONE RULE: every `curve_fit` below -- the first pass and the
    # order-n fit, under every weighting and at every order -- runs the trust region in
    # the Jacobian's column norms (`x_scale='jac'`: More 1978 Eq. (6.3), p. 112 -- the
    # diagonal scaling Dennis & Schnabel 1996 sec. 7.1 prescribe, set from the Jacobian;
    # the house SLS choice at every one of its sites), because in raw units the
    # six-parameter search under a sourced sigma crawled to the evaluation ceiling. No
    # exception for any weighting or for the orders that did not crawl (owner-ruled
    # 2026-09-08; the measurement and the ruling's home are this function's docstring).
    if not enough_alive:
        sigma = None            # nothing to weight; the `ok` line below reports the fit
        weight_failed = True
    elif weighting == 'uniform':
        sigma = None            # no weight, by request: no first pass, nothing evaluated
    elif weighting == 'instrument':
        # Windowed through the SAME `_apply_tau_window` call the correlogram went through,
        # so sigma and g2m1 name one channel set rather than two that happen to match.
        _tau_sig, sigma = _apply_tau_window(
            measurement.delay_times_s, np.asarray(channel_sigma, dtype=float),
            tau_min_s, tau_max_s, min_points=order + 3,
            skip_initial_channels=skip_initial_channels,
        )
        sigma = sigma[alive]
    elif weighting == 'biganzoli':
        # Koppel's evaluation point, literally: an order-1 fit of this same model under
        # UNIFORM weights, whose Gamma and beta the variance is then evaluated at. ONE
        # pass -- the order-n fit below runs once under the resulting sigma, seeded from
        # the linear seed exactly as it is on every other weighting. No fixed point.
        dt_full = (measurement.sample_times_s if measurement.sample_times_s is not None
                   else channel_sample_times(measurement.delay_times_s))
        _tau_dt, dt = _apply_tau_window(
            measurement.delay_times_s, dt_full,
            tau_min_s, tau_max_s, min_points=order + 3,
            skip_initial_channels=skip_initial_channels,
        )
        dt = dt[alive]
        stage = 'first pass'
        try:
            popt1, _ = optimize.curve_fit(
                lambda t, B, be, ga: _frisken_model_scaled_moments(
                    t, B, be, ga, 0.0, 0.0, 0.0),
                tau_fit, g_fit, p0=[B0, beta0, gamma0], sigma=None,
                bounds=([B_lo, 1e-12, 1e-9], [B_hi, np.inf, np.inf]), maxfev=10000,
                x_scale='jac')
            gamma1, beta1 = float(popt1[2]), float(popt1[1])
            if not (math.isfinite(gamma1) and math.isfinite(beta1)):
                raise ValueError('the first pass returned a non-finite Gamma or beta')
            stage = 'weight'
            sigma = _sourced_sigma(tau_fit, dt, gamma1, beta1, mean_count_rate_cps,
                                   duration_s)
        except (RuntimeError, ValueError, TypeError) as exc:
            sigma = None
            if stage == 'weight':
                # Eq. (9) REFUSED its inputs after a first pass that fitted: the FALLBACK
                # (this function's docstring, "A weight that cannot be formed", case 2). The
                # order-n fit runs unweighted and records it; the refusal names the input to
                # change, so the note carries it (invariant 4 clause 1: name the failed
                # precondition, never a proxy). No sample time or duration was used.
                weight_refusal = str(exc)
                recorded = 'uniform'
                sample_time_provenance = ''
                run_duration_form = ''
            else:
                # The first pass did not fit: no evaluation point, nothing to fall back from.
                weight_failed = True
    else:
        # `fit_cumulants` refuses every other token before this arm is dispatched; a
        # private caller that bypassed it gets the same refusal rather than an unbound sigma.
        raise ValueError(
            f"weighting must be one of {sorted(NONLINEAR_WEIGHTINGS)} on the nonlinear "
            f"arm, got {weighting!r}.")

    # The bounds are stated on the DIMENSIONLESS moments and mean exactly what they meant
    # on the dimensional ones: Gamma is bounded strictly positive on the line above, so
    # `m2 >= 0` and `mu2 >= 0` cut the same feasible set, as do `m4 >= 0` and `mu4 >= 0`
    # and the two-sided infinities on the signed m3. Order 1 fits no moment at all and is
    # therefore bit-identical to the previous parametrisation; orders 2 and 3 pass a
    # literal 0.0 for every moment above them, so each order's model is exactly the
    # truncation it names. The mu4 >= 0 bound is a stated choice, argued in this
    # function's docstring.
    if order == 1:
        f = lambda t, B, be, ga: _frisken_model_scaled_moments(t, B, be, ga, 0.0, 0.0, 0.0)
        p0 = [B0, beta0, gamma0]
        lo = [B_lo, 1e-12, 1e-9]
        hi = [B_hi, np.inf, np.inf]
    elif order == 2:
        f = lambda t, B, be, ga, m2: _frisken_model_scaled_moments(t, B, be, ga, m2, 0.0, 0.0)
        p0 = [B0, beta0, gamma0, m2_0]
        lo = [B_lo, 1e-12, 1e-9, 0.0]
        hi = [B_hi, np.inf, np.inf, np.inf]
    elif order == 3:
        f = lambda t, B, be, ga, m2, m3: _frisken_model_scaled_moments(t, B, be, ga, m2, m3, 0.0)
        p0 = [B0, beta0, gamma0, m2_0, m3_0]
        lo = [B_lo, 1e-12, 1e-9, 0.0, -np.inf]
        hi = [B_hi, np.inf, np.inf, np.inf, np.inf]
    else:
        f = _frisken_model_scaled_moments
        p0 = [B0, beta0, gamma0, m2_0, m3_0, m4_0]
        lo = [B_lo, 1e-12, 1e-9, 0.0, -np.inf, 0.0]
        hi = [B_hi, np.inf, np.inf, np.inf, np.inf, np.inf]

    # A bound that could not be formed (NaN: a non-positive or non-finite half-width) is a
    # failed fit, not a free B -- curve_fit would raise on it anyway, but the intent is
    # stated rather than left to the exception. A weight that could not be formed so that
    # no fit can run is the same kind of thing and gets the same answer: a fit nobody can
    # run, reported as one. Read off `weight_failed`, never off `sigma is None`, which is
    # also the unweighted fit asked for or fallen back to.
    ok = math.isfinite(dB) and not weight_failed
    # Built here, not at the successful return, because the D2 failure path returns too
    # and a dead tail is exactly what makes a fit fail (rule 10: the sibling constructor
    # in the same function). The BOUND-active note is deliberately not on that path --
    # there is no fitted B to be at a bound -- so only this one is shared.
    dead_note = _dead_channel_note(int(np.count_nonzero(~alive)), int(g2m1.size))
    B = beta = gamma = mu2 = 0.0
    mu3 = mu4 = None
    rh = d = float('nan')
    if ok:
        try:
            popt, _ = optimize.curve_fit(
                f, tau_fit, g_fit, p0=p0, sigma=sigma, absolute_sigma=False,
                bounds=(lo, hi), maxfev=10000, x_scale='jac')
        except (RuntimeError, ValueError, TypeError):
            ok = False
    if ok:
        B = float(popt[0]); beta = float(popt[1]); gamma = float(popt[2])
        # mu2, mu3 and mu4 come back to physical units with the FITTED Gamma -- the exact
        # inverse of their rescaling, since the model used that same live Gamma at every
        # step.
        mu2 = float(popt[3]) * unc.power_or_nan(gamma, 2) if order >= 2 else 0.0
        mu3 = float(popt[4]) * unc.power_or_nan(gamma, 3) if order >= 3 else None
        mu4 = float(popt[5]) * unc.power_or_nan(gamma, 4) if order >= 4 else None
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
        # The mu2/mu3/mu4 conjuncts came WITH THE RESCALING and are not decoration: each
        # is a PRODUCT (m_n * Gamma^n) rather than a fitted parameter, and a product of two
        # finite numbers can overflow where a value `curve_fit` returns never can.
        if not (math.isfinite(gamma) and gamma > 0 and math.isfinite(beta) and beta > 0
                and math.isfinite(mu2)
                and (mu3 is None or math.isfinite(mu3))
                and (mu4 is None or math.isfinite(mu4))):
            ok = False
        elif math.isfinite(rh) and rh < CUMULANT_MIN_RH_NM:
            ok = False
    if not ok:
        # D2 (owner decision 2026-07-07): a failed nonlinear fit returns NaN physical
        # outputs, NOT a linear-fit substitute — so a caller can't read a fit-like
        # number from a failed fit. (Previously it fell back to the linear fit flagged
        # success=False; that labeled fallback was dropped in favor of an honest NaN.)
        nan = float('nan')
        notes, note_targets = _keyed_notes(('', dead_note))
        return CumulantResult(
            order=order, beta=nan, gamma_s_inv=nan, mu2_s_inv2=nan, mu3_s_inv3=None,
            mu4_s_inv4=None,
            pdi=nan, pdi_valid=False, d_m2_s=nan, rh_nm=nan, q_m_inv=q,
            fit_cutoff=fit_cutoff, n_points_used=int(tau_fit.size),
            coefficients=np.full(order + 1, nan),
            fit_tau_s=tau, fitted_g2m1=np.full(tau.shape, nan),
            residuals=np.full(tau.shape, nan), rms_error=nan,
            weighting=recorded,
            sample_time_provenance=sample_time_provenance,
            run_duration_form=run_duration_form,
            n_skipped=int(skip_initial_channels or 0),
            method='nonlinear', baseline=nan, success=False,
            # The flag travels with the withheld number, as on the linear arm's no-range
            # return: a NaN Rh under a clean flag silences the DLS tab's warning line and
            # the export's Rh comment (invariant 4 clause 1 -- name the failed precondition).
            rh_reliable=False,
            reliability_note=(
                (f'Only {tau_fit.size} of the {tau.size} channels in the delay window '
                 f'collected any coincidence, fewer than the {order + 3} an order-{order} '
                 f'fit needs, so the nonlinear cumulant did not fit; its values are '
                 f'withheld rather than replaced by a substitute. Widen the delay window '
                 f'past the dead channels, or use a longer run.')
                if not enough_alive else
                (f'The nonlinear cumulant could not form its residual weight: '
                 f'{weight_refusal} The unweighted fit it fell back to did not converge to '
                 f'a usable solution either, so its values are withheld rather than replaced '
                 f'by a substitute. Check the correlogram, the delay window and the baseline '
                 f'region, or use the linear method.')
                if weight_refusal else
                'The nonlinear cumulant fit did not converge to a usable solution (the fit '
                'failed, its residual weight could not be formed, or the solution was '
                'degenerate), so its values are withheld rather than replaced by a '
                'substitute. Check the correlogram, the delay window and the baseline '
                'region, or use the linear method.'),
            notes=notes, note_targets=note_targets)

    pdi, pdi_valid = _pdi_and_validity(order, mu2, gamma)
    fitted = f(tau, *popt)
    residuals = g2m1 - fitted
    # D3: rms_error over the SAME retained range the linear path regresses, from ONE
    # computation on the SAME signal -- baseline-subtracted when the caller pinned one, so
    # both arms anchor on one peak at one absolute threshold and the two rms values stay
    # comparable. Measured before that subtraction: at a pinned 0.02 the linear arm kept
    # 116 channels and this one 118. The full window is not a valid shared support (the
    # log-poly model diverges at long lag for order>=2); `residuals` stays full-window for
    # plotting. No fitted value moves here -- rms_error is goodness-of-fit -- but a user
    # reads it, and `_pinned_offset` is 0.0 when no baseline was supplied.
    rms_mask = _retained_range_mask(g2m1 - _pinned_offset(baseline), fit_cutoff)
    rms = _rms_error(residuals[rms_mask]) if rms_mask.any() else _rms_error(residuals)
    # coefficients parity with the linear path: [c0=ln beta, c1=-2 gamma, c2=mu2,
    # c3=-mu3/3, c4=kappa4/12] -- the log-polynomial's coefficients, so the quartic one
    # is the CUMULANT (mu4 - 3 mu2^2)/12, the inverse of the linear arm's conversion.
    coeffs = [math.log(beta), -2.0 * gamma]
    if order >= 2:
        coeffs.append(mu2)
    if order >= 3:
        coeffs.append(-(mu3 or 0.0) / 3.0)
    if order >= 4:
        coeffs.append(((mu4 or 0.0) - 3.0 * mu2 * mu2) / 12.0)

    # The floor is passed for shape parity with the linear path; it cannot fire here,
    # since a sub-floor solution was rejected above (owner decision D2). The undecayed
    # question is asked of the data, not of the fitted Gamma -- a Frisken fit on an
    # undecayed correlogram pins its baseline to a tail that is still signal and
    # absorbs the decay into B, so its Gamma is the wrong number to judge by
    # (measured: an 800 nm scatterer truncated at 0.1 relaxation times fits 242 nm).
    # Asked of the LIVE channels -- the ones the fit saw -- never of a dead tail, which
    # would read as the signal reaching its floor (round 4 finding 1).
    rh_reliable, reliability_note, decay_incomplete = _cumulant_flags(
        rh, gamma, g_fit, pdi, pdi_valid, visc_note,
        max_rh_nm, CUMULANT_MIN_RH_NM, max_retained_acf_ratio)
    if weight_refusal:
        # The fallback's verdict, first: the default method's precondition -- a weight
        # formed from this measurement's own inputs -- failed, and the note names the input
        # Eq. (9) refused. Joined through the one roll-up, so every other verdict stays.
        rh_reliable, reliability_note = _result_flags(
            f'The nonlinear cumulant could not form its residual weight, so this fit is '
            f'unweighted (uniform least squares) instead of the default sourced variance: '
            f'{weight_refusal} The values are reported, fitted without the noise model the '
            f'default method prescribes. Use the linear method for a weighted fit of these '
            f'inputs: its weight reads none of them.',
            reliability_note)

    # The one arm that can carry several passive notes about several different
    # quantities, and so the reason the targets exist at all: a bound-active baseline,
    # dead channels, and either moment resting on its bound can all be true of one fit.
    notes, note_targets = _keyed_notes(
        ('baseline', _bound_active_note(B, B0, dB)),
        ('', dead_note),
        *_moment_bound_note(pdi, popt[5] if order == 4 else None, order),
    )
    return CumulantResult(
        order=order,
        beta=beta,
        gamma_s_inv=gamma,
        mu2_s_inv2=mu2,
        mu3_s_inv3=mu3,
        mu4_s_inv4=mu4,
        pdi=pdi,
        pdi_valid=pdi_valid,
        d_m2_s=d,
        rh_nm=rh,
        q_m_inv=q,
        fit_cutoff=fit_cutoff,
        n_points_used=int(tau_fit.size),
        coefficients=np.asarray(coeffs, dtype=float),
        fit_tau_s=tau,
        fitted_g2m1=fitted,
        residuals=residuals,
        rms_error=rms,
        weighting=recorded,
        sample_time_provenance=sample_time_provenance,
        run_duration_form=run_duration_form,
        n_skipped=int(skip_initial_channels or 0),
        method='nonlinear',
        baseline=B,
        success=True,
        rh_reliable=rh_reliable,
        reliability_note=reliability_note,
        decay_incomplete=decay_incomplete,
        notes=notes, note_targets=note_targets,
    )
