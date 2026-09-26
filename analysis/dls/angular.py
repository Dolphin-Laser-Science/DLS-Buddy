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
from dataclasses import dataclass
from typing import List, Mapping, NamedTuple, Optional, Sequence

import numpy as np

from core.data_models import (DLSMeasurement, casefold_identity_name,
                              round_temperature_for_identity)
from analysis import uncertainty as unc
from physics.constants import stokes_einstein_rh
from analysis.dls._common import (
    _d_from_gamma, _decay_rate_to_rh_nm, _measurement_q_m, _result_flags,
    DIFFUSIVE_INTERCEPT_REL_MAX, DIFFUSIVE_R2_MIN, NO_VISCOSITY_NOTE)
from analysis.dls.cumulants import (
    CUMULANT_FIT_CUTOFF_DEFAULT, CUMULANT_MAX_RH_NM, CUMULANT_MIN_RH_NM,
    _resolve_weighting, _size_range_check, fit_cumulants)
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
    # passive, glyph-less notices for the GUI ⓘ tier (e.g. a mixed sample identity)
    # -- the same text the module also emits via warnings.warn (keep-both).
    notes: tuple = ()
    # "Did the preconditions hold?" for the pooled Rh, on the same pair every
    # per-measurement DLS result carries, so the flag travels WITH the value to the
    # Cross-Sample promotion instead of stopping at the per-angle fits: a per-angle
    # verdict (undecayed window, size range, PDI, no viscosity) is rolled up here, and
    # the pooled Rh is checked against the technique's size range itself. Without it a
    # flagged tier-1 cumulant Rh was laundered into an unflagged q->0 Rh.
    rh_reliable: bool = True
    reliability_note: str = ''


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
    # gated on BOTH a D0 > 0 sign guard and the numerical-health flag. The value is still
    # returned (flag, don't NaN the whole result); `reliability_note` says why.
    kd_reliable: bool = True
    well_conditioned: bool = True         # numerical-health of the [1, c] design
    cond: float = float('nan')            # condition number of the D-vs-c design
    # ONE note field for two flags. `reliability_note` carries the kD reason (sign flip /
    # ill-conditioned) AND the Rh0 reason (a rolled-up per-point verdict, the size range
    # of the extrapolated Rh0), joined -- the two flags below say which applies. The
    # per-point verdicts bear on both, since D0 and kD come from the same line.
    reliability_note: str = ''
    # The Rh0 precondition flag, on the same name every DLS Rh carries (GammaQ2Result,
    # CumulantResult, the exponential results) so the promotion path reads one shape.
    rh_reliable: bool = True
    # every eligible measurement's per-point D_app + quality tag, attached by the
    # controller run so the GUI table/graying come from the run.
    all_points: Optional[list] = None
    # passive, glyph-less notices for the GUI ⓘ tier (e.g. a mixed system) -- the
    # same text the module also emits via warnings.warn (keep-both).
    notes: tuple = ()


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
    from physics.constants import stokes_einstein_diffusion_coefficient
    # The withhold contract is this function's own: an unusable Rh yields NaN, it does not
    # raise. `rh_nm <= 0` rejected NaN but admitted +inf, which used to give D = 0.0 and so
    # Gamma = 0.0, and `stokes_einstein_diffusion_coefficient` refuses a non-finite Rh, so
    # it would raise instead. The finiteness test keeps the contract at this layer.
    if not (math.isfinite(rh_nm) and rh_nm > 0):
        return float('nan')
    d = stokes_einstein_diffusion_coefficient(rh_nm * 1e-9, temperature_K, viscosity_Pa_s)
    # The contract is enforced on the RESULT, exactly as in the inverse `_d_from_gamma`.
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


class _PointFit(NamedTuple):
    """One measurement's per-point fit, as the pooled engines read it: the rate, its
    verdict, whether the default weight fell back, and the per-fit RECORD the pooled
    result relays -- the fit's passive notes, the weighting that ran and the sample time
    it was evaluated on ('' for each under a single-exponential source, which records no
    weighting)."""
    gamma: float
    rh_reliable: bool
    reliability_note: str
    fell_back: bool
    notes: tuple = ()
    weighting: str = ''
    sample_time_provenance: str = ''


def _gamma_for_measurement(measurement, gamma_source, tau_min_s, tau_max_s,
                           skip_initial_channels=0, cumulant_method='linear',
                           cumulant_fit_cutoff=CUMULANT_FIT_CUTOFF_DEFAULT,
                           cumulant_weighting=None, cumulant_order=2, fit_kw=None):
    """A `_PointFit` -- (Gamma, rh_reliable, reliability_note, fell_back) and the fit's
    record -- from one measurement's per-point fit.

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
        return _PointFit(fit.gamma_s_inv, fit.rh_reliable, fit.reliability_note, fell_back,
                         tuple(fit.notes), fit.weighting, fit.sample_time_provenance)
    elif gamma_source == 'single':
        fit = fit_single_exponential(
            measurement, skip_initial_channels=skip_initial_channels, **kw)
        return _PointFit(fit.mode.gamma_s_inv, fit.mode.rh_reliable,
                         fit.mode.reliability_note, False, tuple(fit.notes))
    else:
        raise ValueError(
            f"gamma_source must be 'cumulant' or 'single', got {gamma_source!r}."
        )


def _fit_every_point(measurements, per_kw, where, gamma_source, tau_min_s, tau_max_s,
                     *args) -> List[_PointFit]:
    """`_gamma_for_measurement` over the pooled set, naming the point when a fit REFUSES
    its inputs: a per-measurement input the fit cannot take (a baseline span holding too
    few of its live channels, a window that leaves too few points) would otherwise abort
    the whole pooled fit with a message that names no run. The refusal still aborts it --
    the inputs are the caller's, and a pool fitted around a silently dropped run is not
    what was asked for -- but it says which run. `where(m)` names a point ("at 30 deg")."""
    fits = []
    for m, kw in zip(measurements, per_kw, strict=True):
        try:
            fits.append(_gamma_for_measurement(m, gamma_source, tau_min_s, tau_max_s,
                                               *args, kw))
        except ValueError as exc:
            raise ValueError(f'The fit {where(m)} refused its inputs: {exc}') from exc
    return fits


def _relayed_point_notes(points: Sequence[float], fits: Sequence[_PointFit],
                         unit: str) -> List[str]:
    """Each per-point fit's PASSIVE notes, once per distinct note with the points that
    carried it named in front -- the shape `run_mode_map` relays a run's notes in and
    `extract_modes` an inversion's. A pooled result is the only surface that shows its
    Rh, so a note about how a point was fitted (a floating baseline resting on its bound,
    µ₂ at its bound, dead channels) must reach it; word-for-word repeats are named once.
    `points` and `fits` are aligned (the caller's sorted order)."""
    noted: dict = {}
    for x, fit in zip(points, fits, strict=True):
        for note in fit.notes:
            if note:
                noted.setdefault(note, []).append(f'{x:.4g}')
    return [f"{', '.join(dict.fromkeys(where))}{unit}: {note}"
            for note, where in noted.items()]


def _pooled_point_note(angles_or_concs: np.ndarray, notes: List[str], axis: str,
                       unit: str, stated_at_pool: str = '') -> str:
    """One sentence rolling the flagged per-point verdicts of a pooled fit up to the
    result: WHICH points were flagged (the fact the pooled surface can state), and the
    first such note verbatim (the reason), since per-point notes differ in their
    numbers and a full list would bury the reader. The per-point fits themselves carry
    every note. Empty when no fitted point is flagged."""
    # A sentence the caller states ONCE at the pooled level (`stated_at_pool`: the
    # missing-viscosity note, when the pooled Rh is blank for that reason) is stripped
    # from the per-point notes so it does not print twice inside the "at 30 deg: ..."
    # prefix. Stripped only then: a point lacking viscosity in a set whose others have
    # it keeps its note here, or the roll-up would launder exactly that flag.
    if stated_at_pool:
        notes = [n.replace(stated_at_pool, '').strip() for n in notes]
    flagged = [(x, n) for x, n in zip(angles_or_concs, notes, strict=True) if n]
    if not flagged:
        return ''
    where = ', '.join(f'{x:.4g}{unit}' for x, _ in flagged)
    return (f'{len(flagged)} of {len(notes)} fitted {axis} carry a flagged per-point fit '
            f'({where}); at {flagged[0][0]:.4g}{unit}: {flagged[0][1]}')


def _fallback_note(fallbacks: tuple, unit: str) -> str:
    """The passive sentence naming the points whose default weight fell back to 'uniform'
    (`_gamma_for_measurement`), or '' when none did. Each such point's own fit note names
    the input Eq. (9) refused; this says only where, so the pooled result states it once."""
    if not fallbacks:
        return ''
    where = ', '.join(f'{x:.4g}' for x in fallbacks)
    return (f'The sourced residual weight could not be formed at {where}{unit}, so '
            f'{"that point" if len(fallbacks) == 1 else "those points"} fell back to the '
            f'unweighted (uniform) cumulant fit; each run\'s own fit note names the refusal.')


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
        result records which (`cumulant_weighting_fallbacks`, and a sentence on `notes`).
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

    Returns
    -------
    GammaQ2Result

    Raises
    ------
    ValueError
        If fewer than two measurements, or they are not at distinct angles.
    """
    if len(measurements) < 2:
        raise ValueError(
            f"Gamma vs q^2 needs at least two angles, got {len(measurements)}."
        )

    # Check they are the same sample (sample_key ignores angle, so all should match).
    # Combination doctrine rule 2 (disclosure), core/data_models.py: an identity
    # mix computes and notes; the note travels to the export Comments.
    notes: list = []       # passive ⓘ notices surfaced on the result (keep-both)
    keys = {m.sample_key for m in measurements}
    if len(keys) > 1:
        # Single source of the message -> both the note and the warning.
        msg = (
            "Measurements do not all share the same sample identity "
            "(polymer/solvent/concentration/temperature). Γ vs q² assumes one "
            "sample across angles."
        )
        notes.append(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)

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
    per_kw = _per_measurement_fit_kwargs(measurements, fit_kwargs, gamma_source,
                                         cumulant_method)
    fits = _fit_every_point(measurements, per_kw, lambda m: f'at {m.angle_deg:g} deg',
                            gamma_source, tau_min_s, tau_max_s, skip_initial_channels,
                            cumulant_method, cumulant_fit_cutoff, cumulant_weighting,
                            cumulant_order)
    gamma = np.array([f.gamma for f in fits], dtype=float)
    point_notes = [f.reliability_note if not f.rh_reliable else '' for f in fits]

    # sort by q^2 ascending for clean output
    order = np.argsort(q2)
    angles, q, q2, gamma = angles[order], q[order], q2[order], gamma[order]
    point_notes = [point_notes[i] for i in order]
    sorted_fits = [fits[i] for i in order]
    fallbacks = tuple(float(angles[k]) for k, f in enumerate(sorted_fits) if f.fell_back)
    if fallbacks:
        notes.append(_fallback_note(fallbacks, ' deg'))
    notes.extend(_relayed_point_notes(angles, sorted_fits, ' deg'))
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
        dropped = ', '.join(f'{a:.4g}' for a in angles[~usable])
        notes.append(
            f"Dropped {int((~usable).sum())} of {angles.size} measurement(s) from the "
            f"Gamma-vs-q^2 fit (theta = {dropped} deg): no usable decay rate (a "
            f"non-positive or non-finite Gamma). Check those correlograms."
        )
    gamma = np.where(usable, gamma, np.nan)
    fitted_notes = [n for n, u in zip(point_notes, usable, strict=True) if u]
    temperature_K = measurements[0].temperature_K
    eta = measurements[0].viscosity_Pa_s

    if np.unique(angles[usable]).size < 2:
        # Fewer than two distinct angles survived, so there is no line to fit. Withheld
        # rather than raised, as in the sibling: the INPUT was valid and the data failed.
        notes.append(
            "Fewer than two angles yielded a usable decay rate, so no Gamma-vs-q^2 line "
            "was fitted."
        )
        nan = float('nan')
        return GammaQ2Result(
            angles_deg=angles, q_m_inv=q, q2_m2=q2, gamma_s_inv=gamma,
            d_app_m2_s=gamma / q2,
            d_m2_s=nan, rh_nm=nan, r_squared=nan, intercept_s_inv=nan,
            intercept_relative=nan, d_app_trend_rel=nan, is_diffusive=False,
            gamma_source=gamma_source, temperature_K=temperature_K,
            r2_threshold=r2_threshold, intercept_rel_threshold=intercept_rel_threshold,
            d_se=None, rh_se=None, se_estimator=estimator, notes=tuple(notes),
            cumulant_method=(cumulant_method if gamma_source == 'cumulant' else None),
            cumulant_fit_cutoff=(cumulant_fit_cutoff if gamma_source == 'cumulant'
                                 else None),
            cumulant_weighting=weighting_echo, cumulant_weighting_fallbacks=fallbacks,
            cumulant_order=order_echo, **record,
            rh_reliable=False,
            reliability_note=(
                "D and Rh could not be computed: fewer than two measurements yielded a "
                "usable decay rate, so there was no line to fit -- see the notes for "
                "which angles were dropped."),
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

    # The pooled verdict: the per-angle preconditions of the fits this line rests on
    # (rolled up -- a flagged Gamma makes the slope through it suspect whichever angle
    # it sits at), the technique's size range applied to the pooled Rh itself, and the
    # missing-viscosity cause of a blank Rh. `_result_flags`, the disjunction every arm
    # uses, so a flag at one angle is never cleared by seven clean neighbours.
    _size_ok, size_note = _size_range_check(rh, CUMULANT_MAX_RH_NM, CUMULANT_MIN_RH_NM)
    pool_visc = NO_VISCOSITY_NOTE if eta is None else ''
    rh_reliable, reliability_note = _result_flags(
        _pooled_point_note(angles[usable], fitted_notes, 'angles', ' deg', pool_visc),
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
        notes=tuple(notes),
        rh_reliable=rh_reliable, reliability_note=reliability_note,
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
    `reliability_note` saying so.

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
    cumulant_order, fit_kwargs
        The per-point cumulant order and each measurement's own fit inputs; see
        `analyze_gamma_q2`, whose parameters these mirror.

    Returns
    -------
    ConcentrationExtrapolationResult

    Raises
    ------
    ValueError
        If fewer than two distinct concentrations are supplied.
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
    notes: list = []       # passive ⓘ notices surfaced on the result (keep-both)
    ids = {(casefold_identity_name(m.polymer_name),
            casefold_identity_name(m.solvent_name),
            round_temperature_for_identity(m.temperature_K))
           for m in measurements}
    if len(ids) > 1:
        # Single source of the message -> both the note and the warning.
        msg = (
            "Measurements do not all share polymer/solvent/temperature. "
            "Concentration extrapolation assumes one system."
        )
        notes.append(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)

    conc = np.array([m.concentration_g_per_mL for m in measurements], dtype=float)
    if np.unique(conc).size < 2:
        raise ValueError(
            "Concentration extrapolation needs at least two distinct "
            "concentrations; the values supplied are not all different."
        )

    # `_d_from_gamma`, not an inline `gamma / q ** 2`: written inline this carries both
    # float-boundary defects the helper exists to close -- a Python-float `**` RAISES
    # OverflowError at q ~ 1e155, and the square underflows to 0.0 at q ~ 1e-170 for a
    # ZeroDivisionError. (The helper is not yet the quotient's single home;
    # `analyze_gamma_q2` above still computes it inline.)
    weighting_echo = (_resolve_weighting(cumulant_method, cumulant_weighting)
                      if gamma_source == 'cumulant' else None)
    order_echo = int(cumulant_order) if gamma_source == 'cumulant' else None
    per_kw = _per_measurement_fit_kwargs(measurements, fit_kwargs, gamma_source,
                                         cumulant_method)
    fits = _fit_every_point(
        measurements, per_kw,
        # A run with no concentration is named "n/a" (the measurement's own repr), so its
        # refusal still surfaces as the named ValueError rather than a format TypeError.
        lambda m: (f'at {m.concentration_g_per_mL:.4g} g/mL'
                   if m.concentration_g_per_mL is not None else 'at c = n/a')
                  + f', {m.angle_deg:g} deg',
        gamma_source, tau_min_s, tau_max_s, skip_initial_channels, cumulant_method,
        cumulant_fit_cutoff, cumulant_weighting, cumulant_order)
    d_app = np.array([_d_from_gamma(f.gamma, _measurement_q_m(m))
                      for f, m in zip(fits, measurements, strict=True)])
    # The per-point verdict rides beside its D (see `_gamma_for_measurement`); it is
    # rolled up over the FITTED points below, once the drop has decided which those are.
    point_notes = [f.reliability_note if not f.rh_reliable else '' for f in fits]

    order = np.argsort(conc)
    conc, d_app = conc[order], d_app[order]
    point_notes = [point_notes[i] for i in order]
    sorted_fits = [fits[i] for i in order]
    fallbacks = tuple(float(conc[k]) for k, f in enumerate(sorted_fits) if f.fell_back)
    if fallbacks:
        notes.append(_fallback_note(fallbacks, ' g/mL'))
    notes.extend(_relayed_point_notes(conc, sorted_fits, ' g/mL'))
    record = dict(point_weightings=tuple(f.weighting for f in sorted_fits),
                  point_sample_time_provenance=tuple(
                      f.sample_time_provenance for f in sorted_fits))

    # A measurement whose D_app is unanswerable is DROPPED from the fit, not allowed to NaN
    # the whole extrapolation. `_d_from_gamma` returns NaN for a non-positive or non-finite
    # Gamma -- a cumulant fit on a weak, noisy correlogram does that in ordinary conditions
    # -- and one such point would otherwise NaN d0/kD/Rh0/R2 together. The surviving
    # concentrations still answer the question, so they are fitted and the drop is disclosed
    # on `notes` (the same drop-non-finite precedent as `replicate_mean_se`). The full
    # arrays are still returned so the plot shows the gap.
    usable = np.isfinite(d_app)
    if not usable.all():
        dropped = ', '.join(f'{c:.4g}' for c in conc[~usable])
        # The reason is a DISJUNCTION because `_d_from_gamma` withholds for several
        # causes -- a non-positive/non-finite Gamma, a q outside the representable range,
        # a quotient that over/underflows -- and naming one would assert a cause that may
        # not have occurred.
        notes.append(
            f"Dropped {int((~usable).sum())} of {conc.size} measurement(s) from the "
            f"D-vs-c fit (c = {dropped} g/mL): no usable D could be formed (a "
            f"non-positive decay rate, or a q outside the representable range). "
            f"Check those correlograms."
        )
    conc_fit, d_fit = conc[usable], d_app[usable]

    if np.unique(conc_fit).size < 2:
        # Fewer than two distinct concentrations survived, so there is nothing to
        # extrapolate. Withheld rather than raised: the INPUT was valid (the guards above
        # passed) and it is the data that failed, which is a flag-don't-crash case. Note
        # `linear_fit` would not raise here either -- on one point it returns a slope and
        # `well_conditioned = True` -- so the fit is skipped, not fitted-and-discarded.
        notes.append(
            "Fewer than two concentrations yielded a usable D, so no extrapolation to "
            "infinite dilution was attempted."
        )
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
            reliability_note=(
                "D0 could not be computed: fewer than two measurements yielded a usable "
                "decay rate, so there was nothing to extrapolate -- see the notes for "
                "which concentrations were dropped."),
            rh_reliable=False,
            notes=tuple(notes),
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

    # Reliability of kD: a finite value from a D0 > 0, well-conditioned fit. When False
    # the value is still shown (flag, don't NaN) with the reason on `reliability_note`.
    # The D0 <= 0 sign flip is NOT a conditioning failure (the design can be perfectly
    # well-conditioned), so it needs its own guard -- distinct from `well_conditioned`.
    kd_reliable = bool(d0_ok and lf.well_conditioned and math.isfinite(kd))
    # The note splits `d0_ok` back into its two halves because they are different
    # failures: a fit that returned no D0 at all must not be reported as "D0 <= 0 ...
    # extrapolated non-physical", a cause that did not occur. Only the message splits --
    # one predicate still gates every USE of D0.
    if not math.isfinite(d0):
        reliability_note = (
            "D0 could not be computed: the D-vs-c fit returned no finite intercept, so "
            "no infinite-dilution value (and no kD) is reported -- this is a failed fit, "
            "not a non-physical extrapolation.")
    elif not d0_ok:
        reliability_note = (
            "D0 <= 0: the infinite-dilution diffusion coefficient extrapolated "
            "non-physical, so kD = slope/D0 sign-flips and its ± is not defensible -- "
            "add lower concentrations closer to c -> 0.")
    elif not lf.well_conditioned:
        reliability_note = (
            f"ill-conditioned D-vs-c design (cond={lf.cond:.1e}); kD numerically "
            "unreliable -- add concentrations or widen their spread.")
    else:
        reliability_note = ''

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
    _size_ok, size_note = _size_range_check(rh0, CUMULANT_MAX_RH_NM, CUMULANT_MIN_RH_NM)
    pool_visc = NO_VISCOSITY_NOTE if eta is None else ''
    # The D0 cause belongs to Rh0 too: `rh0 = f(d0)` is withheld by the SAME `d0_ok`
    # predicate, so a verdict built without it reports True beside a NaN (invariant 4
    # clause 1 -- the reported value must be one the method can produce). The string is
    # the one the kD note already carries; `_result_flags` de-duplicates it on the join
    # below, so the sentence still prints once.
    d0_cause = reliability_note if not d0_ok else ''
    rh_reliable, rh_note = _result_flags(
        _pooled_point_note(conc_fit, point_notes, 'concentrations', ' g/mL', pool_visc),
        size_note, pool_visc, d0_cause)
    _joined_ok, reliability_note = _result_flags(reliability_note, rh_note)

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
        cond=lf.cond, reliability_note=reliability_note,
        rh_reliable=rh_reliable,
        notes=tuple(notes),
    )
