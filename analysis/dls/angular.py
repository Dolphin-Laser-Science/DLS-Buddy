"""
analysis/dls/angular.py
=======================

Shared multi-angle / multi-concentration front-end. These functions combine the
single-measurement fits (cumulant / single-exponential) across angle (to test
whether motion is diffusive and extract D) and across concentration (to extrapolate
to infinite dilution). They also provide the public Rh <-> Gamma converters used by
the visualization layer.

Uncertainty (invariant 8): each angle / each concentration is an independent
measurement, so the regression standard errors here ARE defensible (statistical
only, excluding calibration systematics) -- unlike a single correlogram. They are
computed via analysis/uncertainty.py.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, replace
from typing import List, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from core.data_models import (DLSMeasurement, casefold_identity_name,
                              round_temperature_for_identity)
from core.note_registry import DECAY_BEARING_CODES
from core.notes import WHERE, WHERE_UNIT, Note, RefusedInputs, find_notes, render
from analysis import uncertainty as unc
from physics.constants import stokes_einstein_diffusion_coefficient, stokes_einstein_rh
from analysis.dls._common import (
    _all_notes, d_from_gamma, _decay_rate_to_rh_nm, _lifted, _measurement_q_m,
    _no_viscosity_note, _not_diffusive_note, _result_flags, DIFFUSIVE_INTERCEPT_REL_MAX,
    DIFFUSIVE_R2_MIN)
from analysis.dls.cumulants import (
    CUMULANT_FIT_CUTOFF_DEFAULT, CUMULANT_MAX_RH_NM, CUMULANT_MIN_RH_NM,
    _resolve_weighting, size_range_check, fit_cumulants)
from analysis.dls.exponentials import fit_single_exponential


@dataclass
class GammaQ2Result:
    """Multi-angle Gamma vs q^2 analysis for one sample (one concentration).

    For purely translational diffusion, Gamma = D q^2: a straight line through
    the origin, with the apparent diffusion coefficient D_app = Gamma/q^2
    independent of angle. Deviations are diagnostic:
      - a nonzero intercept (unconstrained fit) indicates a non-diffusive
        contribution or a slow relaxation mode;
      - an upward trend in D_app with q^2 indicates internal/segmental modes
        (q Rg > 1);
      - low R^2 indicates the data are simply not linear in q^2.
    """
    angles_deg: np.ndarray            # sorted by q^2 ascending
    q_m_inv: np.ndarray
    q2_m2: np.ndarray                 # q^2 for each angle (m^-2)
    gamma_s_inv: np.ndarray           # decay rate at each angle
    d_app_m2_s: np.ndarray            # Gamma/q^2 at each angle
    # through-origin fit (the translational D)
    d_m2_s: float                     # slope of Gamma = D q^2 through origin
    rh_nm: float                      # Rh from D
    # linearity / deviation diagnostics
    r_squared: float                  # R^2 of the unconstrained linear fit
    intercept_s_inv: float            # intercept of the unconstrained fit
    intercept_relative: float         # intercept / mean(Gamma)
    d_app_trend_rel: float            # (slope of D_app vs q^2) * q2_range / mean(D_app)
    is_diffusive: bool                # passes the general linearity criteria
    # context
    gamma_source: str
    temperature_K: float
    r2_threshold: float
    intercept_rel_threshold: float
    # statistical (regression) standard errors (each angle is an independent point)
    d_se: Optional[float] = None
    rh_se: Optional[float] = None
    se_estimator: str = 'hc3'             # covariance estimator behind the SEs
    # The cumulant inputs that produced the per-point Gamma, echoed for the same reason
    # `gamma_source` is: this Rh is promotable, and both of these move it. Optional with
    # defaults, unlike `gamma_source`, because a 'single'-sourced fit reads neither and a
    # required field would force every such site to pass a value it did not use.
    cumulant_method: Optional[str] = None
    cumulant_fit_cutoff: Optional[float] = None
    # The residual weighting the per-point cumulants were ASKED for, resolved to the arm's
    # request token ('uniform', or the arm's default when none was named), for the same
    # reason as the pair above; None under a 'single' source. And the points -- angles in
    # deg, or concentrations in g/mL -- whose default weight could not be formed and so
    # fell back to 'uniform' on their own (`_fit_cumulants_nonlinear`), so a pooled number
    # built partly from unweighted points never travels without saying which.
    cumulant_weighting: Optional[str] = None
    cumulant_weighting_fallbacks: tuple = ()
    # The cumulant ORDER every per-point fit ran at, echoed with the three inputs above for
    # their reason: it moves the per-point Gamma, so it moves this promotable Rh. None
    # under a 'single' source, which fits no cumulant.
    cumulant_order: Optional[int] = None
    # What each per-point cumulant RECORDED -- the residual weighting that actually ran
    # (`CumulantResult.weighting`) and the sample time its sourced weight was evaluated on
    # (`CumulantResult.sample_time_provenance`) -- aligned with the sorted per-point arrays,
    # '' under a 'single' source. The request above says what was asked for; these say
    # what ran, so a pooled export can name an axis-derived stand-in as the per-fit export
    # does (the `cumulant_weighting_fallbacks` shape, one level wider).
    point_weightings: tuple = ()
    point_sample_time_provenance: tuple = ()
    # every eligible measurement's per-point Γ/q²/D_app + quality tag, attached by the
    # controller run so the GUI table/graying come from the run.
    all_points: Optional[list] = None
    # Every note of the fit: the preconditions `rh_reliable` rests on, then the records
    # (e.g. a mixed sample identity -- the clause the module also emits via warnings.warn,
    # keep-both -- the dropped angles, the per-angle fits' own records).
    notes: Tuple[Note, ...] = ()
    # "Did the preconditions hold?" for the pooled Rh, the flag every per-measurement DLS
    # result carries, so it travels WITH the value to the Cross-Sample promotion instead
    # of stopping at the per-angle fits: a per-angle verdict (undecayed window, size
    # range, PDI, no viscosity) is rolled up here, and the pooled Rh is checked against
    # the technique's size range itself. Without it a flagged tier-1 cumulant Rh was
    # laundered into an unflagged q->0 Rh.
    rh_reliable: bool = True
    # Always True: extrapolated over angle only, the Rh holds one concentration's
    # interactions (invariant 7). `core.notes.qualifiers` reads it.
    is_apparent: bool = True


@dataclass
class ConcentrationExtrapolationResult:
    """Extrapolation of the apparent diffusion coefficient to infinite dilution.

    D(c) = D0 (1 + kD c + ...).  D0 gives the infinite-dilution Rh; kD is the
    diffusion interaction parameter (positive in good solvents, negative in poor).
    """
    concentrations_g_per_mL: np.ndarray   # sorted ascending; ALL supplied, incl. dropped
    d_values_m2_s: np.ndarray             # NaN where no usable D could be formed (dropped)
    d0_m2_s: float                    # intercept: D at c -> 0
    rh0_nm: float                     # Rh from D0
    kd_mL_per_g: float                # slope / D0
    slope: float                      # dD/dc  (m^2/s per g/mL)
    r_squared: float
    temperature_K: float
    n_concentrations: int             # distinct concentrations FITTED (dropped ones excluded)
    # Which fitter produced the per-point Gamma ('cumulant' | 'single'). REQUIRED, exactly
    # as on the sibling GammaQ2Result: with a default, a return site that forgot to pass it
    # would silently claim 'cumulant', and the controller invalidates cumulant-sourced
    # results on a method switch -- so the quiet failure is a result cleared, or kept, on a
    # provenance nobody wrote. A D0/Rh0 must not travel without the input that produced it.
    gamma_source: str
    # statistical (regression) standard errors (each concentration independent)
    d0_se: Optional[float] = None
    rh0_se: Optional[float] = None
    kd_se: Optional[float] = None
    se_estimator: str = 'hc3'             # covariance estimator behind the SEs
    # The cumulant inputs that produced the per-point Gamma, echoed for the same reason
    # `gamma_source` is: this Rh is promotable, and both of these move it. Optional with
    # defaults, unlike `gamma_source`, because a 'single'-sourced fit reads neither and a
    # required field would force every such site to pass a value it did not use.
    cumulant_method: Optional[str] = None
    cumulant_fit_cutoff: Optional[float] = None
    # The residual weighting the per-point cumulants were ASKED for, resolved to the arm's
    # request token ('uniform', or the arm's default when none was named), for the same
    # reason as the pair above; None under a 'single' source. And the points -- angles in
    # deg, or concentrations in g/mL -- whose default weight could not be formed and so
    # fell back to 'uniform' on their own (`_fit_cumulants_nonlinear`), so a pooled number
    # built partly from unweighted points never travels without saying which.
    cumulant_weighting: Optional[str] = None
    cumulant_weighting_fallbacks: tuple = ()
    # The cumulant ORDER every per-point fit ran at; see `GammaQ2Result.cumulant_order`.
    cumulant_order: Optional[int] = None
    # What each per-point cumulant recorded; see `GammaQ2Result.point_weightings`.
    point_weightings: tuple = ()
    point_sample_time_provenance: tuple = ()
    # Reliability of the 1/coefficient kD = slope/D0 (the DLS sibling of the SLS
    # Mw = 1/intercept family; rule 10). kD sign-flips when D0 <= 0 -- a WELL-conditioned
    # fit (a steep +slope extrapolating D0 below zero) can still produce a finite,
    # false-confident kD ±, which `well_conditioned` alone does NOT catch -- so kD is
    # gated on a D0 > 0 sign guard as well as the numerical-health flag, and on its
    # inputs' decay rates and a finite value (`extrapolate_diffusion_vs_concentration`). The
    # value is still returned (flag, don't NaN the whole result); a precondition note says
    # why.
    kd_reliable: bool = True
    well_conditioned: bool = True         # numerical-health of the [1, c] design
    cond: float = float('nan')            # condition number of the D-vs-c design
    # The Rh0 precondition flag, on the same name every DLS Rh carries (GammaQ2Result,
    # CumulantResult, the exponential results) so the promotion path reads one shape.
    rh_reliable: bool = True
    # every eligible measurement's per-point D_app + quality tag, attached by the
    # controller run so the GUI table/graying come from the run.
    all_points: Optional[list] = None
    # Every note of the fit: the kD causes (sign flip / ill-conditioned) and the Rh0 causes
    # (a rolled-up per-point verdict, the size range of the extrapolated Rh0), each about
    # the attributes it fails, then the records (e.g. a mixed system -- the clause the
    # module also emits via warnings.warn, keep-both).
    notes: Tuple[Note, ...] = ()
    # Always False: D0 is extrapolated to infinite dilution, the thermodynamic Rh0
    # (invariant 7). `core.notes.qualifiers` reads it.
    is_apparent: bool = False


# ---------------------------------------------------------------------------
# Rh <-> Gamma converters (for the visualization toggle and general use)
# ---------------------------------------------------------------------------

def gamma_to_rh_nm(gamma_s_inv, q_m_inv, temperature_K, viscosity_Pa_s):
    """Public converter: g1 decay rate Gamma (s^-1) -> hydrodynamic radius (nm)."""
    return _decay_rate_to_rh_nm(gamma_s_inv, q_m_inv, temperature_K, viscosity_Pa_s)


def rh_nm_to_gamma(rh_nm, q_m_inv, temperature_K, viscosity_Pa_s):
    """Public converter: hydrodynamic radius (nm) -> g1 decay rate Gamma (s^-1).

    Gamma = D q^2 with D = kB T / (6 pi eta Rh).
    """
    # The withhold contract is this function's own: an unusable Rh yields NaN, it does not
    # raise. `rh_nm <= 0` rejected NaN but admitted +inf, which used to give D = 0.0 and so
    # Gamma = 0.0, and `stokes_einstein_diffusion_coefficient` refuses a non-finite Rh, so
    # it would raise instead. The finiteness test keeps the contract at this layer.
    if not (math.isfinite(rh_nm) and rh_nm > 0):
        return float('nan')
    d = stokes_einstein_diffusion_coefficient(rh_nm * 1e-9, temperature_K, viscosity_Pa_s)
    # The contract is enforced on the RESULT, exactly as in the inverse `d_from_gamma`.
    # The guard above is about `rh_nm`; the expression is `d * q ** 2`, and that mismatch
    # breaks the contract at BOTH ends -- a Python-float `**` raises OverflowError at
    # q ~ 1e155, and the square underflows to 0.0 at q ~ 1e-170, returning a decay rate of
    # exactly zero: a plausible wrong number where NaN was promised.
    gamma = d * (q_m_inv * q_m_inv)            # `*` saturates to inf; `**` would RAISE
    return gamma if (math.isfinite(gamma) and gamma > 0) else float('nan')


# ---------------------------------------------------------------------------
# Gamma vs q^2 (multi-angle)
# ---------------------------------------------------------------------------

#: The per-measurement fit inputs `analyze_gamma_q2` and
#: `extrapolate_diffusion_vs_concentration` forward to one measurement's own fit: the
#: delay window, and the baseline the caller measured on THAT correlogram -- a value to
#: subtract (`baseline`), or, for the nonlinear cumulant, the lag span its floating
#: baseline is measured over (`baseline_window`). Any other key is refused, so a
#: per-measurement dict cannot quietly override an input the whole pool shares.
PER_MEASUREMENT_FIT_KEYS = frozenset({'tau_min_s', 'tau_max_s', 'baseline',
                                      'baseline_window'})


def _per_measurement_fit_kwargs(measurements, fit_kwargs, gamma_source,
                                cumulant_method='linear') -> list:
    """One dict of fit inputs per measurement (empty when none were given), refusing a
    list that does not align with the measurements or a key outside
    `PER_MEASUREMENT_FIT_KEYS`.

    Each fit is sent only the baseline input it reads, the GUI's own rule for the same
    fits (`_AnalysisRegion.parametric_fit_kwargs`): the nonlinear cumulant takes the span
    its floating baseline is measured over (`baseline_window`), never a value, which would
    only steer its starting point; the linear cumulant and the single exponential take a
    value (`baseline`), never a span, which `fit_cumulants` would silently ignore on the
    linear arm. A pairing a fit does not read is refused rather than dropped, so a caller
    that routes one method's input to another fails loudly instead of fitting without
    the baseline the user placed. Aligned by position, as `measurements` is the one order
    the caller and this module share."""
    reads_window = gamma_source == 'cumulant' and cumulant_method == 'nonlinear'
    if fit_kwargs is None:
        return [{} for _ in measurements]
    fit_kwargs = list(fit_kwargs)
    if len(fit_kwargs) != len(measurements):
        raise ValueError(
            f"fit_kwargs has {len(fit_kwargs)} entries for {len(measurements)} "
            f"measurements; give one (possibly empty) dict per measurement.")
    out = []
    for kw in fit_kwargs:
        kw = dict(kw or {})
        unknown = set(kw) - PER_MEASUREMENT_FIT_KEYS
        if unknown:
            raise ValueError(
                f"fit_kwargs may carry only {sorted(PER_MEASUREMENT_FIT_KEYS)} per "
                f"measurement, got {sorted(unknown)}.")
        if 'baseline_window' in kw and not reads_window:
            raise ValueError(
                "baseline_window is the nonlinear cumulant's input -- the lag span its "
                "floating baseline is measured over; this fit takes a baseline value "
                "(`baseline`) instead.")
        if 'baseline' in kw and reads_window:
            raise ValueError(
                "the nonlinear cumulant takes the lag span its floating baseline is "
                "measured over (`baseline_window`), not a baseline value, which would "
                "only steer its starting point.")
        out.append(kw)
    return out


class PointFit(NamedTuple):
    """One measurement's per-point fit, as the pooled engines and the controller's
    per-point table read it: the rate, its verdict and the precondition notes it rests on,
    whether the default weight fell back, the per-fit RECORD the pooled result relays --
    the fit's record notes, the weighting that ran and the sample time it was evaluated on
    ('' for each under a single-exponential source, which records no weighting) -- and the
    two values the table tags a point by: the fit's PDI (NaN under a single-exponential
    source, which reports none) and whether it converged (the fit's `success`)."""
    gamma: float
    rh_reliable: bool
    preconditions: Tuple[Note, ...]
    fell_back: bool
    notes: Tuple[Note, ...] = ()
    weighting: str = ''
    sample_time_provenance: str = ''
    pdi: float = float('nan')
    success: bool = True


def _records(notes) -> Tuple[Note, ...]:
    """The notes of `notes` that are not preconditions: what a per-point fit records about
    how it was obtained, which the pooled result relays."""
    preconditions = find_notes(notes, kind='precondition')
    return tuple(n for n in notes if n not in preconditions)


def fit_point(measurement, *, gamma_source, tau_min_s, tau_max_s,
              skip_initial_channels=0, cumulant_method='linear',
              cumulant_fit_cutoff=CUMULANT_FIT_CUTOFF_DEFAULT,
              cumulant_weighting=None, cumulant_order=2, fit_kw=None):
    """A `PointFit` -- (Gamma, rh_reliable, its precondition notes, fell_back), the fit's
    record, its PDI and its convergence -- from one measurement's per-point fit: the fit
    `analyze_gamma_q2` and `extrapolate_diffusion_vs_concentration` run on each point, and
    that a caller may run once and hand them (`point_fits`).

    `fit_kw` is this measurement's own inputs (`PER_MEASUREMENT_FIT_KEYS`), laid over the
    shared window: a window key in it wins for this measurement, and a baseline key is
    handed to the fit as it stands. `cumulant_order` is the order of a cumulant fit.

    `fell_back` is True when the cumulant was asked for a weighting other than 'uniform'
    and recorded 'uniform': the default's sourced weight could not be formed on this run
    and the fit fell back on its own. The vocabulary block of `cumulants.py`
    (`RECORDED_WEIGHTINGS`) names the two routes to a recorded 'uniform' -- asked for, or
    fallen back to -- and this test excludes the first. Always False under a 'single' source.

    The verdict travels with the rate: reading `.gamma_s_inv` alone dropped every
    per-point precondition verdict (undecayed window, size range, PDI, no viscosity)
    on the way to the pooled fit, so a multi-angle Rh built from fits every one of which
    was flagged reached the caller clean. Both sources carry the pair on the same names.

    Every place that shows or uses a cumulant fit uses the same weighting and the same
    instrument inputs (owner-ruled 2026-09-08; this docstring is the rule's home). The
    inputs the nonlinear cumulant's default residual weight reads -- the committed mean
    count rate and the parser's per-channel sample times -- travel ON the measurement
    (`DLSMeasurement.mean_count_rate_cps` / `.sample_times_s`, set by
    `LoadedMeasurement.build()`), so a per-point cumulant here reads the SAME weight inputs
    as the DLS tab's fit of that run by construction: nothing is threaded for them, and no
    caller can drop one. The WEIGHTING is threaded (`cumulant_weighting`, the user's
    Settings choice, as the method and cutoff are), and so are the order, the window and
    the baseline (`cumulant_order`, `fit_kw`): the per-point fit is the tab's own fit of
    that run only when the caller passes the tab's values for those.
    """
    kw = dict(tau_min_s=tau_min_s, tau_max_s=tau_max_s)
    kw.update(fit_kw or {})
    if gamma_source == 'cumulant':
        # This site READS THE USER'S CUTOFF AND WEIGHTING, unlike the four seed sites
        # elsewhere in `analysis/`. The reason is not the call's shape but whose fit it is:
        # this is the same cumulant the user just configured, run on the same
        # measurement, and its Gamma is reported to them. A seed inside another method
        # is an implementation detail of that method and must not move when a
        # display-side setting does; a user-facing cumulant must.
        fit = fit_cumulants(measurement, order=cumulant_order,
                            skip_initial_channels=skip_initial_channels,
                            method=cumulant_method,
                            fit_cutoff=cumulant_fit_cutoff,
                            weighting=cumulant_weighting, **kw)
        fell_back = fit.weighting == 'uniform' and cumulant_weighting != 'uniform'
        return PointFit(fit.gamma_s_inv, fit.rh_reliable,
                        find_notes(fit, kind='precondition'), fell_back,
                        _records(fit.notes), fit.weighting, fit.sample_time_provenance,
                        fit.pdi, fit.success)
    elif gamma_source == 'single':
        fit = fit_single_exponential(
            measurement, skip_initial_channels=skip_initial_channels, **kw)
        return PointFit(fit.mode.gamma_s_inv, fit.mode.rh_reliable,
                        find_notes(fit.mode, kind='precondition'), False,
                        _records(fit.notes), success=fit.success)
    else:
        raise ValueError(
            f"gamma_source must be 'cumulant' or 'single', got {gamma_source!r}."
        )


def _fit_every_point(measurements, per_kw, where, **fit_args) -> List[PointFit]:
    """`fit_point` over the pooled set, naming the point when a fit REFUSES
    its inputs: a per-measurement input the fit cannot take (a baseline span holding too
    few of its live channels, a window that leaves too few points) would otherwise abort
    the whole pooled fit with a message that names no run. The refusal still aborts it --
    the inputs are the caller's, and a pool fitted around a silently dropped run is not
    what was asked for -- but it says which run. `where(m)` names a point ("at 30 deg").
    A refusal that names its cause (`RefusedInputs`) is re-raised as one, carrying its
    notes, so a caller relaying it names the cause; any other stays a plain ValueError."""
    fits = []
    for m, kw in zip(measurements, per_kw, strict=True):
        try:
            fits.append(fit_point(m, **fit_args, fit_kw=kw))
        except RefusedInputs as exc:
            raise RefusedInputs(
                exc.notes, f'The fit {where(m)} refused its inputs: {exc}') from exc
        except ValueError as exc:
            raise ValueError(f'The fit {where(m)} refused its inputs: {exc}') from exc
    return fits


def _given_point_fits(measurements, point_fits, fit_kwargs) -> List[PointFit]:
    """A caller's `point_fits`, one per measurement, in place of `_fit_every_point`:
    refused together with `fit_kwargs`, which only a fit run here would read, and when
    they do not align with `measurements`."""
    if fit_kwargs is not None:
        raise ValueError(
            "point_fits and fit_kwargs were both given: the per-point fits are already "
            "run, so no per-measurement fit inputs remain to apply; give one or the "
            "other.")
    return [f for _m, f in zip(measurements, point_fits, strict=True)]


def _relayed_point_notes(points: Sequence[float], fits: Sequence[PointFit], attr: str,
                         unit: str) -> Tuple[Note, ...]:
    """Each per-point fit's RECORD notes, each distinct one once, as notes about `attr`,
    the pooled result's per-point array they explain, with the points that carried it in
    its `where` (their coordinates in `unit`; `core.notes.render` prints them in front) --
    the relay `run_mode_map` makes of a run's notes. A pooled result is the only surface
    that shows its Rh, so a note about how a point was fitted (a floating baseline resting
    on its bound, µ₂ at its bound, dead channels) must reach it, and say where it was
    found. `points` and `fits` are aligned (the caller's sorted order)."""
    merged: dict = {}
    for x, fit in zip(points, fits, strict=True):
        for n in _lifted(fit.notes, attr):
            entry = merged.setdefault((n.code, n.values), [n, []])
            if float(x) not in entry[1]:
                entry[1].append(float(x))
    return tuple(replace(n, values={**n.values, WHERE: tuple(where), WHERE_UNIT: unit})
                 for n, where in merged.values())


def _decay_bearing(preconditions: Tuple[Note, ...]) -> bool:
    """Whether a flagged point's preconditions include one that bears on its decay rate
    (`core.note_registry.DECAY_BEARING_CODES`, the one roster every reader of the rule
    reads): such a point's Gamma, and so its D, is itself in doubt, where a point lacking
    only a viscosity has an exact Gamma and D and no radius."""
    return any(n.code in DECAY_BEARING_CODES for n in preconditions)


def _pooled_point_notes(point_preconditions: Sequence[Tuple[Note, ...]], inputs: str,
                        rh_attr: str, stated_at_pool: bool = False, *,
                        points: Sequence[float] = (), unit: str = '',
                        decay_about: Tuple[str, ...] = ()) -> Tuple[Note, ...]:
    """The flagged per-point verdicts of a pooled fit rolled up to the result, about the
    pooled Rh `rh_attr`: how many of the fitted points were flagged for a cause bearing on
    their decay rate (`inputs_flagged_decay`, which fails a line's kD as well as its Rh)
    and how many for any other cause (`inputs_flagged`), each with its points in its
    `where` -- a point counts once, in the first when it carries both -- and the first
    flagged point's own preconditions (the reason, with that point in their `where`, so a
    per-point cause never reads as the pooled fit's), since per-point notes differ in
    their values and a full list would bury the reader. `points` holds each fitted point's
    coordinate in `unit`, aligned with `point_preconditions`, as `_relayed_point_notes`
    takes them. `decay_about` names the other verdicts' attributes the decay count fails
    (a line's kD), so the count is about every value it fails, as the result's notes
    contract says. The per-point fits themselves carry every note. Empty when no fitted
    point is flagged."""
    # The missing viscosity the caller states ONCE at the pooled level (`stated_at_pool`,
    # when the pooled Rh is blank for that reason) is left out of the per-point notes so
    # it is not stated twice. Left out only then: a point lacking viscosity in a set whose
    # others have it keeps its note here, or the roll-up would launder exactly that flag.
    if stated_at_pool:
        point_preconditions = [tuple(n for n in pre if n.code != 'no_viscosity')
                               for pre in point_preconditions]
    flagged = [(float(x), pre) for x, pre in zip(points, point_preconditions, strict=True)
               if pre]
    if not flagged:
        return ()
    first_at, first = flagged[0]
    decay = tuple(x for x, pre in flagged if _decay_bearing(pre))
    other = tuple(x for x, pre in flagged if not _decay_bearing(pre))
    counts = []
    if decay:
        counts.append(Note('inputs_flagged_decay', (rh_attr, *decay_about), {
            'k': len(decay), 'n': len(point_preconditions), 'inputs': inputs,
            WHERE: decay, WHERE_UNIT: unit}))
    if other:
        counts.append(Note('inputs_flagged', (rh_attr,), {
            'k': len(other), 'n': len(point_preconditions), 'inputs': inputs,
            WHERE: other, WHERE_UNIT: unit}))
    return (*counts,
            *(replace(n, values={**n.values, WHERE: (first_at,), WHERE_UNIT: unit})
              for n in _lifted(first, rh_attr)))


def analyze_gamma_q2(
    measurements: List[DLSMeasurement],
    gamma_source: str = 'cumulant',
    tau_min_s: Optional[float] = None,
    tau_max_s: Optional[float] = None,
    r2_threshold: float = DIFFUSIVE_R2_MIN,
    intercept_rel_threshold: float = DIFFUSIVE_INTERCEPT_REL_MAX,
    skip_initial_channels: int = 0,
    cumulant_method: str = 'linear',
    cumulant_fit_cutoff: float = CUMULANT_FIT_CUTOFF_DEFAULT,
    estimator: str = 'hc3',
    cumulant_weighting: Optional[str] = None,
    cumulant_order: int = 2,
    fit_kwargs: Optional[Sequence[Optional[Mapping]]] = None,
    point_fits: Optional[Sequence[PointFit]] = None,
) -> GammaQ2Result:
    """Analyze Gamma vs q^2 across angles for one sample.

    Fits a decay rate at each angle, then examines its dependence on q^2.
    Translational diffusion gives Gamma = D q^2 (a line through the origin); the
    through-origin slope is the diffusion coefficient. Linearity diagnostics
    flag non-diffusive behavior.

    Parameters
    ----------
    measurements : list of DLSMeasurement
        Two or more measurements of the SAME sample (same polymer, solvent,
        concentration, temperature) at DIFFERENT angles.
    gamma_source : str
        How to get Gamma at each angle: 'cumulant' (first cumulant, default) or
        'single' (single-exponential fit).
    tau_min_s, tau_max_s : float, optional
        Delay-time window passed through to the per-angle fits.
    r2_threshold : float
        Minimum R^2 for the is_diffusive flag (default DIFFUSIVE_R2_MIN,
        0.98). A general statistical criterion, not a system-specific tuning
        value.
    intercept_rel_threshold : float
        Maximum |intercept|/mean(Gamma) for the is_diffusive flag (default
        DIFFUSIVE_INTERCEPT_REL_MAX, 0.1).
    cumulant_weighting : str, optional
        The residual weighting of every per-angle cumulant, passed to `fit_cumulants` as
        `cumulant_method` and `cumulant_fit_cutoff` are; None is the arm's default. An angle
        whose default weight cannot be formed falls back to 'uniform' on its own, and the
        result records which (`cumulant_weighting_fallbacks`, and a record on `notes`).
    cumulant_order : int
        The order of every per-angle cumulant fit (default 2), echoed on the result as
        `cumulant_order`. Unused under a 'single' source.
    fit_kwargs : sequence of dict, optional
        One dict per measurement, aligned with `measurements`, of that measurement's OWN
        fit inputs (`PER_MEASUREMENT_FIT_KEYS`): each angle's baseline is measured on its
        own correlogram, and the nonlinear cumulant takes the span itself, so no single
        value could stand for all of them. A window key in a dict overrides
        `tau_min_s`/`tau_max_s` for that measurement. None (the default) is every
        measurement's fit on the shared inputs alone, exactly as before this parameter.
    point_fits : sequence of PointFit, optional
        One per measurement, aligned with `measurements`: each point's fit already run by
        `fit_point` on the inputs this call names, used in place of fitting it again.
        The caller vouches for those inputs, since the result echoes this call's
        cumulant inputs; refused together with `fit_kwargs`, which only a fit run here
        would read.

    Returns
    -------
    GammaQ2Result

    Raises
    ------
    ValueError
        If fewer than two measurements, or they are not at distinct angles; or if
        `point_fits` is given together with `fit_kwargs`, or does not align with
        `measurements`.
    """
    if len(measurements) < 2:
        raise ValueError(
            f"Gamma vs q^2 needs at least two angles, got {len(measurements)}."
        )

    # Check they are the same sample (sample_key ignores angle, so all should match).
    # Combination doctrine rule 2 (disclosure), core/data_models.py: an identity
    # mix computes and notes; the note travels to the export Comments.
    notes: list = []       # the result's records (keep-both with the warning)
    keys = {m.sample_key for m in measurements}
    if len(keys) > 1:
        # Single source of the message -> both the note and the warning.
        mixed = Note('mixed_sample_identity', (), {
            'fields': ('polymer', 'solvent', 'concentration', 'temperature')})
        notes.append(mixed)
        warnings.warn(render(mixed, surface='export'), UserWarning, stacklevel=2)

    angles = np.array([m.angle_deg for m in measurements], dtype=float)
    if np.unique(angles).size < 2:
        raise ValueError(
            "Gamma vs q^2 needs measurements at distinct angles; the angles "
            "supplied are not all different."
        )

    q = np.array([_measurement_q_m(m) for m in measurements])
    q2 = q ** 2
    # A non-finite scattering vector is a PARAMETER error, not a fit failure: q =
    # 4*pi*n/lambda*sin(theta/2), so it can only come from a refractive index or a
    # wavelength that has gone non-finite. Raise here, beside the other input errors and
    # before the fits, rather than degrade to a table of NaNs -- the user can act on this
    # one, and only a clear message tells them where to look. (Guarding here also means
    # `linear_fit_through_origin` below cannot silently withhold first and make the error
    # arrive one operation late.)
    if not np.all(np.isfinite(q2)):
        raise ValueError(
            "The scattering vector q is not finite for at least one measurement, so "
            "Gamma vs q^2 cannot be fit. q = 4*pi*n/lambda*sin(theta/2) -- check the "
            "solvent refractive index and the wavelength on these measurements."
        )
    weighting_echo = (_resolve_weighting(cumulant_method, cumulant_weighting)
                      if gamma_source == 'cumulant' else None)
    order_echo = int(cumulant_order) if gamma_source == 'cumulant' else None
    if point_fits is not None:
        fits = _given_point_fits(measurements, point_fits, fit_kwargs)
    else:
        per_kw = _per_measurement_fit_kwargs(measurements, fit_kwargs, gamma_source,
                                             cumulant_method)
        fits = _fit_every_point(measurements, per_kw, lambda m: f'at {m.angle_deg:g} deg',
                                gamma_source=gamma_source, tau_min_s=tau_min_s,
                                tau_max_s=tau_max_s,
                                skip_initial_channels=skip_initial_channels,
                                cumulant_method=cumulant_method,
                                cumulant_fit_cutoff=cumulant_fit_cutoff,
                                cumulant_weighting=cumulant_weighting,
                                cumulant_order=cumulant_order)
    gamma = np.array([f.gamma for f in fits], dtype=float)
    point_notes = [f.preconditions if not f.rh_reliable else () for f in fits]

    # sort by q^2 ascending for clean output
    order = np.argsort(q2)
    angles, q, q2, gamma = angles[order], q[order], q2[order], gamma[order]
    point_notes = [point_notes[i] for i in order]
    sorted_fits = [fits[i] for i in order]
    fallbacks = tuple(float(angles[k]) for k, f in enumerate(sorted_fits) if f.fell_back)
    notes.extend(_relayed_point_notes(angles, sorted_fits, 'gamma_s_inv', 'deg'))
    record = dict(point_weightings=tuple(f.weighting for f in sorted_fits),
                  point_sample_time_provenance=tuple(
                      f.sample_time_provenance for f in sorted_fits))

    # A measurement whose decay rate is unusable -- non-positive (a weak, noisy or
    # rising correlogram) or non-finite (a failed fit) -- is DROPPED from the line and
    # disclosed on `notes`, as `extrapolate_diffusion_vs_concentration` does (fitted, a
    # negative Gamma biased D by +18 % on Rh; a NaN one NaN'd the result with notes=()).
    # The returned arrays keep EVERY angle, a dropped one carrying NaN in Gamma and D_app
    # like the sibling's dropped concentration, so the plot marks the gap and the export
    # keeps the row -- an omitted angle must never look unmeasured. The fit reads the
    # usable subset only.
    usable = np.isfinite(gamma) & (gamma > 0)
    if not usable.all():
        notes.append(Note('unusable_rate_dropped', ('gamma_s_inv',), {
            'n_dropped': int((~usable).sum()), 'n': int(angles.size), 'points': 'angles',
            'quantity': 'Γ'}))
    gamma = np.where(usable, gamma, np.nan)
    fitted_notes = [n for n, u in zip(point_notes, usable, strict=True) if u]
    temperature_K = measurements[0].temperature_K
    eta = measurements[0].viscosity_Pa_s

    n_usable = int(np.unique(angles[usable]).size)
    if n_usable < 2:
        # Fewer than two distinct angles survived, so there is no line to fit. Withheld
        # rather than raised, as in the sibling: the INPUT was valid and the data failed.
        too_few = Note('too_few_points', ('d_m2_s', 'rh_nm'),
                       {'n': n_usable, 'points': 'angles', 'need': 2})
        nan = float('nan')
        return GammaQ2Result(
            angles_deg=angles, q_m_inv=q, q2_m2=q2, gamma_s_inv=gamma,
            d_app_m2_s=gamma / q2,
            d_m2_s=nan, rh_nm=nan, r_squared=nan, intercept_s_inv=nan,
            intercept_relative=nan, d_app_trend_rel=nan, is_diffusive=False,
            gamma_source=gamma_source, temperature_K=temperature_K,
            r2_threshold=r2_threshold, intercept_rel_threshold=intercept_rel_threshold,
            d_se=None, rh_se=None, se_estimator=estimator,
            notes=_all_notes(too_few, notes),
            cumulant_method=(cumulant_method if gamma_source == 'cumulant' else None),
            cumulant_fit_cutoff=(cumulant_fit_cutoff if gamma_source == 'cumulant'
                                 else None),
            cumulant_weighting=weighting_echo, cumulant_weighting_fallbacks=fallbacks,
            cumulant_order=order_echo, **record,
            rh_reliable=False,
        )
    d_app = gamma / q2
    q2_fit, g_fit, d_app_fit = q2[usable], gamma[usable], d_app[usable]

    # Through-origin slope D (Gamma = D q^2) with its standard error (each angle is
    # an independent measurement, so the regression SE is defensible).
    d, d_se = unc.linear_fit_through_origin(q2_fit, g_fit, estimator)
    d = float(d)

    # Unconstrained linear fit for the intercept and R^2.
    slope_u, intercept_u = unc.polyfit_or_nan(q2_fit, g_fit, 1)
    gamma_line = slope_u * q2_fit + intercept_u
    ss_res = float(np.sum((g_fit - gamma_line) ** 2))
    ss_tot = float(np.sum((g_fit - g_fit.mean()) ** 2))
    r2 = unc.r_squared_or_nan(ss_res, ss_tot)
    _gbar = float(g_fit.mean())
    intercept_rel = (intercept_u / _gbar
                     if (math.isfinite(_gbar) and _gbar != 0) else float('nan'))

    # q-dependence of D_app (internal-mode indicator), expressed relatively.
    if q2_fit.max() > q2_fit.min():
        dapp_slope = unc.polyfit_or_nan(q2_fit, d_app_fit, 1)[0]
        dapp_trend_rel = float(dapp_slope * (q2_fit.max() - q2_fit.min())
                               / d_app_fit.mean())
    else:
        dapp_trend_rel = float('nan')

    # `math.isfinite(d)` as well as `d > 0`: `+inf` passes a bare `> 0`, and
    # stokes_einstein_rh(inf, ...) used to return a finite, plausible-looking Rh = 0.0 nm
    # with no ± to warn anyone. The physics function now refuses a non-finite D outright,
    # so this gate is defence in depth -- but it has to be here too, or the refusal would
    # surface as an uncaught ValueError instead of the NaN this branch exists to give.
    if eta is not None and math.isfinite(d) and d > 0:
        rh = stokes_einstein_rh(d, temperature_K, eta) * 1e9
    else:
        rh = float('nan')

    # Rh ~ 1/D, so the fractional SE carries over (sigma_Rh/Rh = sigma_D/D).
    rh_se = unc.power_law_se(rh, d, d_se, -1) if math.isfinite(rh) else None

    is_diffusive = bool(
        math.isfinite(r2) and r2 >= r2_threshold
        and abs(intercept_rel) <= intercept_rel_threshold
    )
    diffusive_note = None if is_diffusive else _not_diffusive_note(r2, intercept_rel)

    # The pooled verdict: the per-angle preconditions of the fits this line rests on
    # (rolled up -- a flagged Gamma makes the slope through it suspect whichever angle
    # it sits at), the technique's size range applied to the pooled Rh itself, and the
    # missing-viscosity cause of a blank Rh. `_result_flags`, the disjunction every arm
    # uses, so a flag at one angle is never cleared by seven clean neighbours.
    _size_ok, size_note = size_range_check(rh, CUMULANT_MAX_RH_NM, CUMULANT_MIN_RH_NM)
    pool_visc = _no_viscosity_note() if eta is None else None
    rh_reliable, preconditions = _result_flags(
        _pooled_point_notes(fitted_notes, 'angles', 'rh_nm', pool_visc is not None,
                            points=angles[usable], unit='deg'),
        size_note, pool_visc)

    return GammaQ2Result(
        angles_deg=angles, q_m_inv=q, q2_m2=q2, gamma_s_inv=gamma,
        d_app_m2_s=d_app, d_m2_s=d, rh_nm=rh,
        r_squared=r2, intercept_s_inv=float(intercept_u),
        intercept_relative=float(intercept_rel),
        d_app_trend_rel=dapp_trend_rel, is_diffusive=is_diffusive,
        gamma_source=gamma_source, temperature_K=temperature_K,
        r2_threshold=r2_threshold, intercept_rel_threshold=intercept_rel_threshold,
        d_se=unc.se_or_none(d_se), rh_se=unc.se_or_none(rh_se),
        se_estimator=estimator,
        cumulant_method=(cumulant_method if gamma_source == 'cumulant' else None),
        cumulant_fit_cutoff=(cumulant_fit_cutoff if gamma_source == 'cumulant'
                             else None),
        cumulant_weighting=weighting_echo, cumulant_weighting_fallbacks=fallbacks,
        cumulant_order=order_echo, **record,
        notes=_all_notes(preconditions, diffusive_note, notes),
        rh_reliable=rh_reliable,
    )


# ---------------------------------------------------------------------------
# Concentration extrapolation
# ---------------------------------------------------------------------------

def extrapolate_diffusion_vs_concentration(
    measurements: List[DLSMeasurement],
    gamma_source: str = 'cumulant',
    tau_min_s: Optional[float] = None,
    tau_max_s: Optional[float] = None,
    skip_initial_channels: int = 0,
    cumulant_method: str = 'linear',
    cumulant_fit_cutoff: float = CUMULANT_FIT_CUTOFF_DEFAULT,
    estimator: str = 'hc3',
    cond_limit: float = unc.COND_LIMIT,
    cumulant_weighting: Optional[str] = None,
    cumulant_order: int = 2,
    fit_kwargs: Optional[Sequence[Optional[Mapping]]] = None,
    point_fits: Optional[Sequence[PointFit]] = None,
) -> ConcentrationExtrapolationResult:
    """Extrapolate the apparent diffusion coefficient to infinite dilution.

    For each measurement, D_app = Gamma / q^2 is computed (Gamma from the chosen
    source). D_app is then fit linearly against concentration and extrapolated to
    c -> 0:

        D(c) = D0 (1 + kD c)

    giving the infinite-dilution diffusion coefficient D0 (and thus Rh0) and the
    diffusion interaction parameter kD = slope / D0.

    The measurements should be the same polymer/solvent/temperature at different
    concentrations. They may be at a single common angle (the usual case) or each
    reduced via Gamma/q^2; since D_app is angle-independent for pure diffusion,
    either works, but mixing angles will fold any q-dependence into the result.

    A measurement that yields no usable D_app (a non-positive or non-finite Gamma --
    a cumulant fit on a weak, noisy correlogram does this) is DROPPED from the fit and
    the drop is recorded on `notes`; the remaining concentrations are extrapolated
    normally. `d_values_m2_s` still carries a NaN in its place -- which the plotting layer
    marks with a dotted vertical line, since a NaN in a scatter draws nothing and an
    invisible omission is what this policy must not look like -- and `n_concentrations`
    counts only what was fitted. If fewer than two distinct concentrations survive there
    is nothing to extrapolate, and the whole result is withheld (NaN) with a
    `too_few_points` precondition note saying so.

    Parameters
    ----------
    measurements : list of DLSMeasurement
        Two or more measurements at different concentrations.
    gamma_source : str
        'cumulant' (default) or 'single'.
    tau_min_s, tau_max_s : float, optional
        Delay-time window passed to the per-measurement fits.
    cumulant_weighting : str, optional
        The residual weighting of every per-concentration cumulant; see
        `analyze_gamma_q2`, whose parameter this mirrors, fallback record included.
    cumulant_order, fit_kwargs, point_fits
        The per-point cumulant order, each measurement's own fit inputs and its fit
        already run; see `analyze_gamma_q2`, whose parameters these mirror.

    Returns
    -------
    ConcentrationExtrapolationResult

    Raises
    ------
    ValueError
        If fewer than two distinct concentrations are supplied; or if `point_fits` is
        given together with `fit_kwargs`, or does not align with `measurements`.
    """
    if len(measurements) < 2:
        raise ValueError(
            f"Concentration extrapolation needs at least two measurements, got "
            f"{len(measurements)}."
        )

    # Warn if polymer/solvent/temperature are not consistent.
    # Combination doctrine rule 2 (disclosure), core/data_models.py: an identity
    # mix computes and notes; the note travels to the export Comments. Compared
    # through the SHARED identity forms, never raw values -- a raw compare flagged
    # a 'PVP' + 'pvp ' ladder the workspace itself grouped as one sample.
    notes: list = []       # the result's records (keep-both with the warning)
    ids = {(casefold_identity_name(m.polymer_name),
            casefold_identity_name(m.solvent_name),
            round_temperature_for_identity(m.temperature_K))
           for m in measurements}
    if len(ids) > 1:
        # Single source of the message -> both the note and the warning.
        mixed = Note('mixed_sample_identity', (),
                     {'fields': ('polymer', 'solvent', 'temperature')})
        notes.append(mixed)
        warnings.warn(render(mixed, surface='export'), UserWarning, stacklevel=2)

    conc = np.array([m.concentration_g_per_mL for m in measurements], dtype=float)
    if np.unique(conc).size < 2:
        raise ValueError(
            "Concentration extrapolation needs at least two distinct "
            "concentrations; the values supplied are not all different."
        )

    # `d_from_gamma`, not an inline `gamma / q ** 2`: written inline this carries both
    # float-boundary defects the helper exists to close -- a Python-float `**` RAISES
    # OverflowError at q ~ 1e155, and the square underflows to 0.0 at q ~ 1e-170 for a
    # ZeroDivisionError. (The helper is not yet the quotient's single home;
    # `analyze_gamma_q2` above still computes it inline.)
    weighting_echo = (_resolve_weighting(cumulant_method, cumulant_weighting)
                      if gamma_source == 'cumulant' else None)
    order_echo = int(cumulant_order) if gamma_source == 'cumulant' else None
    if point_fits is not None:
        fits = _given_point_fits(measurements, point_fits, fit_kwargs)
    else:
        per_kw = _per_measurement_fit_kwargs(measurements, fit_kwargs, gamma_source,
                                             cumulant_method)
        fits = _fit_every_point(
            measurements, per_kw,
            # A run with no concentration is named "n/a" (the measurement's own repr), so
            # its refusal still surfaces as the named ValueError rather than a format
            # TypeError.
            lambda m: (f'at {m.concentration_g_per_mL:.4g} g/mL'
                       if m.concentration_g_per_mL is not None else 'at c = n/a')
                      + f', {m.angle_deg:g} deg',
            gamma_source=gamma_source, tau_min_s=tau_min_s, tau_max_s=tau_max_s,
            skip_initial_channels=skip_initial_channels, cumulant_method=cumulant_method,
            cumulant_fit_cutoff=cumulant_fit_cutoff, cumulant_weighting=cumulant_weighting,
            cumulant_order=cumulant_order)
    d_app = np.array([d_from_gamma(f.gamma, _measurement_q_m(m))
                      for f, m in zip(fits, measurements, strict=True)])
    # The per-point verdict rides beside its D (see `fit_point`); it is
    # rolled up over the FITTED points below, once the drop has decided which those are.
    point_notes = [f.preconditions if not f.rh_reliable else () for f in fits]

    order = np.argsort(conc)
    conc, d_app = conc[order], d_app[order]
    point_notes = [point_notes[i] for i in order]
    sorted_fits = [fits[i] for i in order]
    fallbacks = tuple(float(conc[k]) for k, f in enumerate(sorted_fits) if f.fell_back)
    notes.extend(_relayed_point_notes(conc, sorted_fits, 'd_values_m2_s', 'g/mL'))
    record = dict(point_weightings=tuple(f.weighting for f in sorted_fits),
                  point_sample_time_provenance=tuple(
                      f.sample_time_provenance for f in sorted_fits))

    # A measurement whose D_app is unanswerable is DROPPED from the fit, not allowed to NaN
    # the whole extrapolation. `d_from_gamma` returns NaN for a non-positive or non-finite
    # Gamma -- a cumulant fit on a weak, noisy correlogram does that in ordinary conditions
    # -- and one such point would otherwise NaN d0/kD/Rh0/R2 together. The surviving
    # concentrations still answer the question, so they are fitted and the drop is disclosed
    # on `notes` (the same drop-non-finite precedent as `replicate_mean_se`). The full
    # arrays are still returned so the plot shows the gap.
    usable = np.isfinite(d_app)
    if not usable.all():
        # The note names no cause because `d_from_gamma` withholds for several -- a
        # non-positive/non-finite Gamma, a q outside the representable range, a quotient
        # that over/underflows -- and naming one would assert a cause that may not have
        # occurred.
        notes.append(Note('unusable_rate_dropped', ('d_values_m2_s',), {
            'n_dropped': int((~usable).sum()), 'n': int(conc.size),
            'points': 'concentrations', 'quantity': 'D'}))
    conc_fit, d_fit = conc[usable], d_app[usable]

    n_usable = int(np.unique(conc_fit).size)
    if n_usable < 2:
        # Fewer than two distinct concentrations survived, so there is nothing to
        # extrapolate. Withheld rather than raised: the INPUT was valid (the guards above
        # passed) and it is the data that failed, which is a flag-don't-crash case. Note
        # `linear_fit` would not raise here either -- on one point it returns a slope and
        # `well_conditioned = True` -- so the fit is skipped, not fitted-and-discarded.
        too_few = Note('too_few_points', ('d0_m2_s', 'rh0_nm', 'kd_mL_per_g'),
                       {'n': n_usable, 'points': 'concentrations', 'need': 2})
        return ConcentrationExtrapolationResult(
            concentrations_g_per_mL=conc, d_values_m2_s=d_app,
            d0_m2_s=float('nan'), rh0_nm=float('nan'), kd_mL_per_g=float('nan'),
            slope=float('nan'), r_squared=float('nan'),
            temperature_K=measurements[0].temperature_K,
            # ZERO, not the survivor count: the field means concentrations FITTED, and no
            # fit ran. Reporting `1` here put "concs fitted: 1" on a GUI label for a design
            # that never existed -- the same shape as the reliability note naming a cause
            # that did not occur.
            n_concentrations=0, gamma_source=gamma_source,
            d0_se=None, rh0_se=None, kd_se=None, se_estimator=estimator,
            cumulant_method=(cumulant_method if gamma_source == 'cumulant' else None),
            cumulant_fit_cutoff=(cumulant_fit_cutoff if gamma_source == 'cumulant'
                                 else None),
            cumulant_weighting=weighting_echo, cumulant_weighting_fallbacks=fallbacks,
            cumulant_order=order_echo, **record,
            kd_reliable=False, well_conditioned=False, cond=float('inf'),
            rh_reliable=False,
            notes=_all_notes(too_few, notes),
        )
    point_notes = [n for n, u in zip(point_notes, usable, strict=True) if u]

    lf = unc.linear_fit(conc_fit, d_fit, estimator, cond_limit)  # D(c)=D0+slope*c; cov [intercept, slope]
    slope, intercept = lf.slope, lf.intercept
    d0 = float(intercept)
    r2 = lf.r_squared
    # kD = slope/D0 is a 1/coefficient quantity: guard on D0 > 0 (not just != 0). A
    # non-positive infinite-dilution D0 is unphysical AND makes kD sign-flip (a good-
    # solvent +slope misread as poor-solvent), so we do not report kD/kD_se from it --
    # mirroring the rh0 guard just below. ONE hoisted predicate gates every use of D0
    # below, and it tests FINITENESS as well: a bare `d0 > 0` admits +inf, which makes
    # kD = slope/inf = 0.0 -- a finite value that then passes `isfinite(kd)` and is
    # flagged reliable with an empty note.
    d0_ok = math.isfinite(d0) and d0 > 0
    kd = float(slope / d0) if d0_ok else float('nan')

    # Statistical SEs: D0 from the intercept; kD = slope/D0 through the 2x2 cov.
    d0_se = unc.se_or_none(lf.intercept_se)
    kd_se = None
    if d0_ok and math.isfinite(lf.intercept_se):
        # `power_or_nan`, not `d0 ** 2`: `d0_ok` is a predicate about d0 while the
        # division is by its SQUARE, so it cannot answer the question. A Python-float `**`
        # RAISES at the overflow tail rather than saturating, so this would crash at
        # d0 = 1e200 and hit ZeroDivisionError at 1e-200, out of an SE path whose whole
        # contract is to withhold.
        kd_se = unc.se_or_none(unc.propagate(
            [-slope / unc.power_or_nan(d0, 2), 1.0 / d0], lf.cov))  # [intercept, slope]

    # Reliability of kD: a finite value from a D0 > 0, well-conditioned fit over inputs
    # whose decay rates hold, each failed precondition on its own note (invariant 4
    # clause 1) and the value still shown. The D0 <= 0 sign flip is NOT a conditioning
    # failure (the design can be perfectly well-conditioned), so it has its own guard. An
    # input flagged for a cause bearing on its decay rate (`DECAY_BEARING_CODES`) fails
    # kD as it fails Rh0, named by `inputs_flagged_decay` in the Rh0 roll-up below; a
    # cause that does not (a missing viscosity) leaves kD standing.
    decay_flagged = any(_decay_bearing(pre) for pre in point_notes)
    kd_reliable = bool(d0_ok and lf.well_conditioned and math.isfinite(kd)
                       and not decay_flagged)
    # The note splits `d0_ok` back into its two halves because they are different
    # failures: a fit that returned no D0 at all must not be reported as a non-positive
    # D0, a cause that did not occur. Only the note splits -- one predicate still gates
    # every USE of D0. A kD past the float range from a usable D0 is named on its own.
    d0_about = ('d0_m2_s', 'kd_mL_per_g', 'rh0_nm')
    if not math.isfinite(d0):
        d0_note = Note('d0_not_finite', d0_about)
    elif not d0_ok:
        d0_note = Note('d0_nonpositive', d0_about, {'d0': d0, 'limit': 0.0})
    else:
        d0_note = None
    cond_note = (None if lf.well_conditioned else
                 Note('ill_conditioned', ('kd_mL_per_g',),
                      {'cond': lf.cond, 'limit': lf.cond_limit}))
    overflow_note = (Note('kd_not_finite', ('kd_mL_per_g',))
                     if d0_ok and not math.isfinite(kd) else None)

    temperature_K = measurements[0].temperature_K
    eta = measurements[0].viscosity_Pa_s
    if eta is not None and d0_ok:                    # d0_ok: see the kD block above
        rh0 = stokes_einstein_rh(d0, temperature_K, eta) * 1e9
    else:
        rh0 = float('nan')
    rh0_se = unc.power_law_se(rh0, d0, d0_se, -1) if math.isfinite(rh0) else None

    # The Rh0 verdict, the same roll-up `analyze_gamma_q2` makes: the per-point
    # preconditions of the fits the line rests on, the technique's size range applied
    # to the extrapolated Rh0, and the missing-viscosity cause of a blank Rh0. Joined
    # onto the kD note above -- one note field, two flags (see the dataclass).
    _size_ok, size_note = size_range_check(rh0, CUMULANT_MAX_RH_NM, CUMULANT_MIN_RH_NM,
                                            about=('rh0_nm',))
    pool_visc = _no_viscosity_note(('rh0_nm',)) if eta is None else None
    # The D0 cause belongs to Rh0 too: `rh0 = f(d0)` is withheld by the SAME `d0_ok`
    # predicate, so a verdict built without it reports True beside a NaN (invariant 4
    # clause 1 -- the reported value must be one the method can produce). It is the note
    # the kD cause already carries; `_result_flags` keeps it once on the join below.
    rh_reliable, rh_notes = _result_flags(
        _pooled_point_notes(point_notes, 'concentrations', 'rh0_nm', pool_visc is not None,
                            points=conc[usable], unit='g/mL',
                            decay_about=('kd_mL_per_g',)),
        size_note, pool_visc, d0_note)
    _joined_ok, preconditions = _result_flags(d0_note, cond_note, overflow_note, rh_notes)

    return ConcentrationExtrapolationResult(
        concentrations_g_per_mL=conc, d_values_m2_s=d_app,
        d0_m2_s=d0, rh0_nm=rh0, kd_mL_per_g=kd, slope=float(slope),
        r_squared=r2, temperature_K=temperature_K,
        n_concentrations=int(np.unique(conc_fit).size),   # concentrations FITTED, not supplied
        gamma_source=gamma_source,
        d0_se=d0_se, rh0_se=unc.se_or_none(rh0_se), kd_se=kd_se,
        se_estimator=estimator,
        cumulant_method=(cumulant_method if gamma_source == 'cumulant' else None),
        cumulant_fit_cutoff=(cumulant_fit_cutoff if gamma_source == 'cumulant'
                             else None),
        cumulant_weighting=weighting_echo, cumulant_weighting_fallbacks=fallbacks,
        cumulant_order=order_echo, **record,
        kd_reliable=kd_reliable, well_conditioned=lf.well_conditioned,
        cond=lf.cond,
        rh_reliable=rh_reliable,
        notes=_all_notes(preconditions, notes),
    )
