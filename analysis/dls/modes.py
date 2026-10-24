"""
analysis/dls/modes.py
=====================

Mode-resolved dynamics. Three pure steps:

  1. `extract_modes`      -- every relaxation mode of ONE measurement, as
                             pickable `ModePoint`s (the rows of the "mode map",
                             the (q^2, Gamma) scatter of Sedlak & Amis 1992
                             Fig. 3); `extract_mode_runs` runs it over a
                             map's runs, tagging each point with its run.
  2. `fit_mode_family`    -- Gamma vs q^2 for ONE user-picked mode family:
                             through-origin D (+ apparent Rh, always computed,
                             always labeled) and the slow-mode deviation form
                             Gamma = a q^2 + b q^4 (Sedlak & Amis 1992a,
                             J. Chem. Phys. 96, 817 -- p. 819).
  3. `amplitude_vs_q`     -- the fast/slow amplitude report A_f/A_s vs
                             sin^2(theta/2) (their Fig. 4 axis) with the
                             fast-amplitude angle-flatness diagnostic.

NO association machinery lives here (owner-ruled 2026-08-14; this docstring is
that ruling's normative home): which points form a family is the USER's pick,
made in the GUI and carried by the controller as (angle, Gamma) VALUE pairs;
these functions only extract and fit what they are handed. The through-origin
("diffusive") verdict is a tolerance-carrying LABEL on the result, never an
eligibility gate -- every family's Rh is computed and flagged apparent +
mode-resolved (owner-ruled 2026-08-14; a slow mode's Stokes-Einstein size is
wanted even when the mode is not a particle -- the guide carries Sedlak 1996's
caution).

Uncertainty (invariant 8): a mode's amplitude and per-point Gamma come from ONE
correlogram, so they carry no +-. The per-family fits combine independent
angles, so their regression SEs are defensible (statistical only), exactly as
in analyze_gamma_q2; the coefficient SEs are linear pass-throughs of the
validated HC3/OLS covariances and Rh's SE is the validated reciprocal
(power_law_se, exponent -1) -- both inherit their Monte-Carlo validation, and
no new nonlinear transform is introduced.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, replace
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from core.data_models import DLSMeasurement, round_angle_for_identity
from core.note_registry import REGISTRY
from core.notes import WHERE, WHERE_UNIT, Note, RefusedInputs, find_notes, has_note, render
from analysis import uncertainty as unc
from physics.constants import stokes_einstein_rh
from analysis.dls._common import (
    _all_notes,
    d_from_gamma,
    _decay_completeness,
    _decay_completeness_check,
    _fitted_tau_span,
    _grid_end_check,
    _measurement_q_m,
    _no_viscosity_note,
    _not_diffusive_note,
    _require_viscosity,
    _result_flags,
    DIFFUSIVE_INTERCEPT_REL_MAX,
    DIFFUSIVE_R2_MIN,
    MAX_RETAINED_ACF_RATIO,
    MIN_DECAY_COMPLETENESS,
)
from analysis.dls.cumulants import CUMULANT_MAX_RH_NM, CUMULANT_MIN_RH_NM, size_range_check
from analysis.dls.exponentials import fit_double_exponential
from analysis.dls.distributions import (
    _fit_decay_check, find_distribution_peaks, fit_contin, fit_nnls, integrate_between,
    lobe_geometric_mean)


# The mode-amplitude resolvability floor: below ~2 % of the total scattered intensity a
# mode's contribution should not be evaluated (Sedlak 2001, Physical Chemistry of
# Polyelectrolytes ch. 1, pp. 7-8, one of his two requirements for evaluating a mode; the
# dashed-line caveat of Sedlak & Amis 1992a p. 820). A documented default (invariant 4).
# It gates a NOTE on a point, never its existence: a sub-floor 'double' mode, or a manual
# integration interval ('contin' / 'nnls', not floored), still appears and is pickable. On
# the automatic lobes the same value is `extract_modes`' default `min_weight_fraction`, so
# a sub-floor lobe is dropped before it becomes a point unless the caller lowers that.
DEFAULT_AMPLITUDE_FLOOR = 0.02

MODE_SOURCES = ('double', 'contin', 'nnls')
# The fit each source runs, as the `fit_refused_inputs` note names it.
_SOURCE_FIT_NAMES = {'double': 'double-exponential', 'contin': 'CONTIN', 'nnls': 'NNLS'}


@dataclass(frozen=True)
class ModePoint:
    """One extracted relaxation mode of one measurement -- a mode-map point.

    Frozen: a pick target that cannot mutate is the structural half of the
    snapshot ruling (owner, 2026-08-14) -- user picks are matched against
    these by (angle_deg, gamma_s_inv) VALUE, so a re-extraction that moves a
    mode's Gamma orphans the pick rather than silently re-aiming it.

    `amplitude_fraction` is the mode's relative g1 amplitude (double-
    exponential source) or its intensity-weighted lobe area (distribution
    sources); either way the modes of one run sum to ~1 and it is an INTENSITY
    fraction, never a mass/number fraction. `rh_nm` is the mode's apparent,
    mode-resolved Stokes-Einstein size (NaN when viscosity is missing) --
    computed by the source fitters through the one Gamma->Rh home.

    `item_id` and `polarization_state` are provenance its caller reads and
    `extract_mode_runs` attaches (`dataclasses.replace`); `extract_modes` leaves
    the defaults. `item_id` is NEVER a pick key.

    `alpha`/`alpha_selection_method` record the CONTIN regularization the
    amplitudes came from (None/'' for double/NNLS), and `gcv_gamma` the
    robust-GCV parameter beside them (None unless the selector was GCV: 'gcv'
    names a family, so the pair (method, γ) is the provenance) — partition
    provenance for mode-resolved static intensities, captured here because the
    inversion result is dropped after extraction. `rh_min_nm`/`rh_max_nm`/`n_grid` record
    the size grid that inversion ran on for the same reason: it is a user
    setting, so two runs of one measurement can resolve differently, and the
    grid that produced a mode is otherwise unrecoverable from the mode. They
    are read back off the grid the fit actually used, not the grid requested,
    so they describe the inversion rather than the intent (defaults for the
    parametric source, which has no grid).
    `weighting`/`sample_time_provenance`/`run_duration_form` are captured for
    exactly the same reason and in the same way — read OFF THE FIT, not off the
    request — and carry the same meanings as on `DistributionResult`: which
    residual weight built the inversion these amplitudes came from, and what
    that weight read. A weight moves the recovered distribution, the selected
    α and therefore every lobe area on this map, so a mode says which form
    produced it. All three default to '' for the parametric source, which has
    no inversion behind it. `bounds_*` record the integration
    lobe the amplitude was summed over (distribution sources; NaN/'' for the
    parametric double-exponential): 'auto' = the scan's own local-minimum
    lobes, 'manual' = user-set bounds. All additive with defaults — picks are
    (angle, Gamma) value-keyed, so nothing re-keys.

    `mean_gamma_s_inv` / `mean_rh_nm` are the mode's POSITION: for a CONTIN or NNLS
    source its lobe's geometric mean (`lobe_geometric_mean`, the automatic lobes and the
    manual intervals alike), for the double-exponential source the mode's own Gamma and
    Rh, which have no lobe. ONE Gamma per point drives every derived quantity: `d_m2_s`
    and the decay-completeness verdict derive from the mean, the map plots it, and
    `fit_mode_family` regresses on it (`plotted_gamma_s_inv`). `gamma_s_inv` / `rh_nm`
    stay the node with the most weight, and `gamma_s_inv` stays the PICK KEY: the node is
    the grid index of the extracted point, the better identity, and the mean a property
    derived from it, so a saved pick still matches and no migration exists. The one cost,
    disclosed: a pick is keyed to a Gamma the user does not see plotted (the plotted point
    sits at the mean, about half a grid step from the node for a narrow lobe and a step or
    more for a broad, overlapping or grid-end one). NaN when the producing path predates
    them; `plotted_gamma_s_inv` then reads the node.
    """
    angle_deg: float
    q_m_inv: float
    q2_m2: float
    gamma_s_inv: float
    amplitude_fraction: float
    d_m2_s: float                     # Gamma/q^2 via d_from_gamma (NaN-capable)
    rh_nm: float                      # apparent, mode-resolved (NaN-capable)
    source_method: str                # 'double' | 'contin' | 'nnls'
    mode_index: int                   # rank within its run, fast->slow (0 = fastest)
    n_modes_in_run: int
    low_amplitude: bool               # amplitude_fraction < amplitude_floor (a NOTE gate)
    amplitude_floor: float            # the documented floor the flag used
    item_id: str = ''
    # The lag-window verdict, the same NOTE-gate shape as low_amplitude/amplitude_floor
    # above: the condition, the measured quantity it came from, and the documented
    # threshold it was judged against. Set for EVERY source -- a mode that never finished
    # decaying is the same defect whether it came from the parametric fit or from a
    # distribution peak, and one rate predicate covers all three (rule 12). True when
    # decay_completeness < min_decay_completeness, or, for a distribution source, when
    # the correlogram itself never decayed (`distributions._fit_decay_check`), so a
    # point can carry the flag with its own completeness above the threshold.
    decay_incomplete: bool = False
    decay_completeness: float = float('nan')      # Gamma * (fitted lag span)
    min_decay_completeness: float = MIN_DECAY_COMPLETENESS
    # The grid-end verdict beside it, the same note-gate shape: the condition here, the
    # measured quantity the node `rh_nm`, the threshold the grid's ends `rh_min_nm` /
    # `rh_max_nm` below. True when the point's lobe (or manual interval) has its maximum
    # on the grid's first or last node (`_common._grid_end_check`); False for the
    # parametric source, which has no grid, and on a point built before the field existed.
    grid_end: bool = False
    polarization_state: Optional[str] = None   # 'VV' | 'VH' | None (unpolarized class)
    alpha: Optional[float] = None     # CONTIN regularization (None for double/NNLS)
    alpha_selection_method: str = ''  # 'gcv' | 'lcurve' | 'ftest' | 'user' | '' (non-CONTIN)
    gcv_gamma: Optional[float] = None  # robust-GCV gamma beside 'gcv' (None otherwise)
    rh_min_nm: float = float('nan')   # the size grid the inversion ran on
    rh_max_nm: float = float('nan')   # (NaN/0 for the parametric double-exponential)
    n_grid: int = 0
    weighting: str = ''               # the inversion's recorded residual-weighting form
    sample_time_provenance: str = ''  # 'instrument' | 'axis' | '' (what that weight read)
    run_duration_form: str = ''       # 'finite' | 'infinite' | '' ('' for the parametric
    #                                   source, which runs no inversion at all)
    bounds_lo_rh_nm: float = float('nan')   # integration-lobe bounds (distribution sources)
    bounds_hi_rh_nm: float = float('nan')
    bounds_source: str = ''           # 'auto' | 'manual' | '' (parametric)
    mean_gamma_s_inv: float = float('nan')  # the mode's position: the lobe's geometric
    mean_rh_nm: float = float('nan')        # mean (distribution), its own (double)


def plotted_gamma_s_inv(point) -> float:
    """The Gamma every derived quantity of a mode point reads: its position, the lobe's
    geometric mean (`ModePoint.mean_gamma_s_inv`), or the node `gamma_s_inv` for a point
    that carries no mean -- a mode recorded, or a double built, before the field existed.
    Takes a `ModePoint` or a recorded snapshot of one (the controller's
    `RecordedModePoint`), read by attribute."""
    mean = getattr(point, 'mean_gamma_s_inv', None)
    if mean is not None and math.isfinite(mean) and mean > 0:
        return float(mean)
    return float(point.gamma_s_inv)


def _point_rh_nm(point) -> float:
    """The size at a mode point's position, `plotted_gamma_s_inv`'s companion: the lobe's
    geometric mean `mean_rh_nm`, or the node `rh_nm` for a point that carries no mean. NaN
    when the point has no size (no viscosity), which the size range reads as unanswerable."""
    mean = getattr(point, 'mean_rh_nm', None)
    if mean is not None and math.isfinite(mean) and mean > 0:
        return float(mean)
    return float(point.rh_nm)


@dataclass
class ModeExtraction:
    """Every usable mode of one measurement, fast->slow."""
    modes: tuple                      # ModePoint, Gamma descending
    source_method: str
    success: bool
    # Every note of the extraction: why it failed (`success` False), the inversion's own
    # notes relayed, and the records of what was skipped or set by hand.
    notes: Tuple[Note, ...] = ()
    # The residual weighting the inversion recorded (`DistributionResult.weighting`), the
    # one its points carry; '' for the parametric source, which runs no inversion.
    weighting: str = ''


@dataclass
class ModeMapRun:
    """One sample's mode map: every extracted mode of every eligible run.

    Assembled by the controller (which walks the sample, reads each run's item_id /
    polarization_state and pools `extract_mode_runs`' points); cached as the
    substrate the user picks families from.
    `source_method` is recorded on the result, not the cache key (the
    gamma_source precedent).
    """
    points: tuple                     # ModePoint, all runs pooled
    source_method: str
    notes: Tuple[Note, ...] = ()


@dataclass
class ModeFamilyFitResult:
    """Gamma vs q^2 for one user-picked mode family.

    Same criteria and verdict field name as the sibling GammaQ2Result (rule
    10): `is_diffusive` = R^2 >= r2_threshold AND |intercept|/mean(Gamma) <=
    intercept_rel_threshold, with both thresholds echoed onto the result so the
    verdict travels with what produced it. The verdict is a LABEL -- it gates
    nothing; Rh is computed whenever D is usable (owner-ruled 2026-08-14).

    The q^2+q^4 deviation form Gamma = a q^2 + b q^4 (Sedlak & Amis 1992a
    p. 819) is the slow-mode diagnostic: a is the q->0 apparent diffusion
    coefficient of the mode, b the leading deviation. It needs >= 3 distinct
    angles (two parameters, no intercept); with fewer it is withheld (NaN) with
    a note, never raised -- the through-origin fit is the primary result.
    """
    # snapshot of what was fitted (sorted q^2 ascending)
    angles_deg: np.ndarray
    q_m_inv: np.ndarray
    q2_m2: np.ndarray
    gamma_s_inv: np.ndarray
    d_app_m2_s: np.ndarray            # per-point Gamma/q^2
    amplitude_fractions: np.ndarray   # per-point mode amplitude (no +-; invariant 8)
    # through-origin fit (Gamma = D q^2)
    d_m2_s: float
    rh_nm: float                      # APPARENT, MODE-RESOLVED; always computed when D usable
    # linearity / origin diagnostics (verdict + its thresholds, the house shape)
    r_squared: float                  # R^2 of the unconstrained linear fit
    intercept_s_inv: float
    intercept_relative: float         # intercept / mean(Gamma)
    is_diffusive: bool                # the tolerance-carrying LABEL, never a gate
    r2_threshold: float
    intercept_rel_threshold: float
    # slow-mode deviation form Gamma = a q^2 + b q^4
    a_m2_s: float                     # NaN + note when < 3 distinct angles
    b_m4_s: float
    q4_r_squared: float
    q4_cond: float
    q4_well_conditioned: bool
    # context
    source_method: str                # extraction source of the picked points
    temperature_K: float
    # statistical (regression) standard errors -- independent angles, defensible
    d_se: Optional[float] = None
    rh_se: Optional[float] = None
    a_se: Optional[float] = None
    b_se: Optional[float] = None
    se_estimator: str = 'hc3'
    n_low_amplitude: int = 0          # picked points under their amplitude floor
    n_decay_incomplete: int = 0       # picked points that never finished decaying
    # Every note of the fit: the q^2+q^4 form's preconditions, then the records of the
    # pick set (keep-both with a warning where noted)
    notes: Tuple[Note, ...] = ()
    # Always True: one mode's Rh at one concentration (invariant 7). `core.notes.qualifiers`
    # reads it, and names the value mode-resolved by the class (MODE_FAMILY_CLASSES).
    is_apparent: bool = True
    # "Did the preconditions hold?" for the family's D and Rh, the flag every DLS Rh
    # carries, rolled up over the picked points the way `GammaQ2Result` rolls up its
    # angles (`_result_flags`, the any-one rule): an undecayed point, a point at the Rh
    # grid's end, the family Rh past the technique's size range, no viscosity. The
    # amplitude floor is a record, never this verdict (guide sec. 3.5.4).
    rh_reliable: bool = True


@dataclass
class AmplitudeRatioResult:
    """The fast/slow amplitude report: A_f/A_s vs sin^2(theta/2).

    The axis is Sedlak & Amis 1992a Fig. 4 (p. 820). `fast_flat` is the
    angle-flatness diagnostic -- fast-mode amplitudes are expected q-flat in
    the two-mode phenomenology -- expressed as (max-min)/mean of A_f against a
    DOCUMENTED house threshold (no printed criterion exists in the held
    sources; the sources state flatness qualitatively, so the default is a
    disclosed house parameter, invariant 4). It gates a note, never a refusal.

    No SEs anywhere: every amplitude is a single-correlogram quantity
    (invariant 8).
    """
    angles_deg: np.ndarray            # shared angles only, ascending
    sin2_half_theta: np.ndarray
    fast_fraction: np.ndarray
    slow_fraction: np.ndarray
    ratio_f_over_s: np.ndarray        # NaN where unanswerable (A_s = 0)
    fast_flat: bool                   # label; False when unanswerable (a note says why)
    fast_rel_span: float              # (max - min)/mean of A_f; NaN when < 2 angles
    flatness_rel_threshold: float
    n_shared_angles: int
    notes: Tuple[Note, ...] = ()      # the flatness preconditions, then the records


# ---------------------------------------------------------------------------
# 1. Per-measurement mode extraction
# ---------------------------------------------------------------------------

def _refusal_notes(exc: ValueError, source: str, measurement) -> Tuple[Note, ...]:
    """The notes an extraction carries when its fitter refused the correlogram.

    A `RefusedInputs` names its cause as notes, relayed here about the whole extraction
    (they were about the refusing fit's own attributes). The proxy `fit_refused_inputs`
    stays where no cause names the refusal: a plain `ValueError`, or relayed causes none
    of which fails `success` itself (a missing viscosity fails only `rh_reliable`, on
    fits that ran), each read off its registry row. A missing viscosity is also named by
    the inversion's own test of it (`_require_viscosity`), whether or not the fitter
    raised it as a cause; the double-exponential source degrades without one rather than
    refusing."""
    causes = exc.notes if isinstance(exc, RefusedInputs) else ()
    relayed = tuple(replace(n, about=()) for n in causes)
    named = any('success' in REGISTRY[n.code].fails
                for n in relayed if n.code in REGISTRY)
    proxy = (None if named
             else Note('fit_refused_inputs', (), {'fit': _SOURCE_FIT_NAMES[source]}))
    viscosity = None
    if source in ('contin', 'nnls'):
        try:
            _require_viscosity(measurement)
        except ValueError:
            viscosity = _no_viscosity_note(())
    return _all_notes(proxy, relayed, viscosity)


def extract_modes(
    measurement: DLSMeasurement,
    source: str = 'double',
    *,
    min_weight_fraction: float = DEFAULT_AMPLITUDE_FLOOR,
    amplitude_floor: float = DEFAULT_AMPLITUDE_FLOOR,
    tau_min_s: Optional[float] = None,
    tau_max_s: Optional[float] = None,
    skip_initial_channels: int = 0,
    manual_bounds_rh_nm: Optional[Sequence[Sequence[float]]] = None,
    min_decay_completeness: float = MIN_DECAY_COMPLETENESS,
    baseline: Optional[float] = None,
    max_retained_acf_ratio: float = MAX_RETAINED_ACF_RATIO,
    **dist_kw,
) -> ModeExtraction:
    """Extract every usable relaxation mode of one measurement.

    Reuses the shipped fitters -- nothing is recomputed here: 'double' takes
    the two `ExponentialMode`s of `fit_double_exponential`; 'contin'/'nnls'
    run the inversion and take its `find_distribution_peaks` populations
    (`**dist_kw` forwards grid/alpha/beta options; `baseline` below reaches every
    source). Rh on each mode
    comes from those fitters' own Gamma->Rh conversion (the single home).

    `min_weight_fraction` is the distribution-peak noise threshold; its
    default here is the amplitude floor (0.02) rather than the distribution
    tab's 0.05, because the mode map exists to SHOW weak modes down to the
    resolvability limit -- the ~2 % of the total intensity Sedlak (2001, pp. 7-8)
    requires of a mode before its contribution is evaluated; treating a lobe
    below it as inversion noise is this program's reading, not his -- so
    dropping it is extraction
    granularity, not a gate on a result. Modes above the floor appear
    unflagged; a double-exponential mode below it appears WITH the
    `low_amplitude` note-flag (that path has no noise threshold).

    Inputs not prescribed by the source: dropping a distribution lobe below
    `min_weight_fraction` (default 0.02) as inversion noise, this program's reading of
    Sedlak's requirement (above), and the inputs the underlying fitters' own lines name
    (`fit_double_exponential`, `find_distribution_peaks`); prescribed: `amplitude_floor`
    (default 0.02), Sedlak 2001's requirement that a mode's amplitude not fall below
    approximately 2 % of the total scattered intensity before its contribution is
    evaluated (pp. 7-8), and the decay-completeness criterion as
    `fit_double_exponential` states it.

    `manual_bounds_rh_nm` -- user-set integration intervals `[(lo, hi), ...]`
    in Rh (nm), replacing the automatic lobe split when overlapping peaks
    defeat it (distribution sources ONLY; 'double' raises -- the parametric
    amplitudes come from the model, not from cutting a spectrum, and that IS
    the recommended route for genuine overlap). Each interval becomes one
    mode; the bounds are recorded on the points as provenance and a manual
    extraction always carries the wide-separation caveat note. Malformed or
    overlapping intervals raise (`integrate_between`'s contract -- a split
    that double-counts weight writes a false record).

    `baseline` is a residual baseline offset the caller measured on this
    correlogram, handed to whichever fitter the source names -- the double
    exponential subtracts it as its pinned baseline, the inversions subtract it
    before inverting -- exactly as each fitter takes it when called directly. It is
    named here rather than left in `**dist_kw` because the 'double' branch forwards
    none of those. None (the default) is each fitter's own default.

    `min_decay_completeness` is the lag-window criterion: a mode whose
    Gamma * (fitted lag span) falls below it never finished decaying inside the
    measured range, so its rate is an extrapolation rather than a measurement.
    Applied to EVERY source from one predicate, because the defect is a property
    of a rate against a lag window and not of how the rate was obtained; the
    'double' points therefore agree with `fit_double_exponential`'s own per-mode
    verdict by construction. A distribution source's points also take the
    correlogram's own decay (`distributions._fit_decay_check` at
    `max_retained_acf_ratio`), as its peaks do: an inversion's lobe passes its rate
    test on a correlogram that never decayed, and that fact fails every lobe of the
    fit. It gates a note, never a refusal.

    A distribution source's own notes are relayed onto the extraction -- that the
    weight could not be formed, dead channels, CONTIN's alpha preconditions --
    because the inversion result is dropped here and a note left on it would reach
    no surface. Whether the weight ran without a committed count rate is the
    recorded form each point carries (`ModePoint.weighting`).

    Failure-tolerant by contract: a fit failure OR an unfittable correlogram
    (the fitters' own ValueError) returns `success=False` with empty modes and
    the reason as a note -- one weird run must not kill a multi-angle map, and
    the controller relays the note. Only an invalid `source`, manual bounds on
    'double', bad manual bounds, or a `baseline_window` (which no source reads) raise
    (caller bugs, not data conditions).
    """
    if source not in MODE_SOURCES:
        raise ValueError(
            f"source must be one of {MODE_SOURCES}, got {source!r}."
        )
    # Truthiness, not `is not None`: an empty interval list means "no manual
    # bounds" everywhere — steering onto the manual path with zero intervals
    # would extract nothing and still claim manual bounds were used.
    if manual_bounds_rh_nm and source == 'double':
        raise ValueError(
            "Manual integration bounds apply to distribution sources only; "
            "the double-exponential's amplitudes come from the model, not "
            "from integrating a spectrum. Use 'contin' or 'nnls', or the "
            "parametric route itself for genuinely overlapping modes."
        )
    # No source here measures a floating baseline over a lag span: the double exponential
    # would drop the key (its branch forwards none of `**dist_kw`) and CONTIN/NNLS would
    # raise a TypeError. Refused by name, so a caller's region is never silently not applied.
    if 'baseline_window' in dist_kw:
        raise ValueError(
            "extract_modes takes a baseline value (`baseline`); none of its sources "
            "measures a floating baseline over a lag span, so `baseline_window` is "
            "refused.")

    notes: List[Note] = []
    q = _measurement_q_m(measurement)
    q2 = q * q                    # `*` saturates to inf; `**` would raise (rule 12)
    if not (math.isfinite(q2) and q2 > 0):
        # A non-finite/zero q^2 is a PARAMETER error (refractive index /
        # wavelength / angle), same disclosure as the analyze_gamma_q2 gate --
        # but per-measurement extraction is failure-tolerant (the map shows
        # every OTHER run), so this withholds with a note instead of raising.
        return ModeExtraction(
            modes=(), source_method=source, success=False,
            notes=(Note('q_unusable', (), {'q2': float(q2)}),))

    # (gamma, amplitude, rh, bounds, bounds_source, mean_gamma, mean_rh, grid_end) per mode,
    # unordered: the node, then the position (the lobe's geometric mean; a double mode's
    # own), then whether the lobe's maximum is a grid-end node (never, for a double). The
    # fitter call is wrapped: an unfittable CORRELOGRAM raises ValueError out of the fits
    # (e.g. the cumulant seed's "0 points survive the cutoff" on a flat trace), a data
    # failure to disclose per run, not an error to propagate -- one weird run must not kill
    # a multi-angle map; the message travels as the note. Manual-bounds integration runs
    # OUTSIDE the wrap: its ValueError is an input-contract violation, not a data failure.
    _no_bounds = (float('nan'), float('nan'))
    raw: List[tuple] = []
    dist = None
    alpha: Optional[float] = None
    alpha_method = ''
    gcv_gamma: Optional[float] = None
    rh_lo = rh_hi = float('nan')
    n_grid_used = 0
    weighting = sample_time_provenance = run_duration_form = ''
    # The fitted lag SPAN, read off whichever fit produced the modes -- never off the
    # measurement, so a user's tau window and the skipped leading channels are already
    # in it (every fit result carries `fit_tau_s`, its window after both).
    tau_span = float('nan')
    try:
        if source == 'double':
            res = fit_double_exponential(
                measurement, tau_min_s=tau_min_s, tau_max_s=tau_max_s,
                skip_initial_channels=skip_initial_channels, baseline=baseline)
            if not res.success:
                return ModeExtraction(
                    modes=(), source_method=source, success=False,
                    notes=(Note('fit_not_converged', ()),))
            # NO all-modes-undecayed withhold here, owner-ruled 2026-09-06. One returned
            # no modes at all when both were flagged `decay_incomplete`, contradicting
            # flag-not-refuse -- and "both modes undecayed" is equally true of a genuinely
            # slow sample, which is what this map is FOR. Both now travel, flagged, and
            # every surface already tells them apart: the point carries
            # `decay_completeness`, the map draws such modes as a labelled square, and the
            # export caveats Gamma and D_app.
            tau_span = _fitted_tau_span(res.fit_tau_s)
            for m in (res.mode1, res.mode2):
                raw.append((m.gamma_s_inv, m.amplitude_fraction, m.rh_nm,
                            _no_bounds, '', m.gamma_s_inv, m.rh_nm, False))
        else:
            fitter = fit_contin if source == 'contin' else fit_nnls
            res = fitter(measurement, tau_min_s=tau_min_s, tau_max_s=tau_max_s,
                         skip_initial_channels=skip_initial_channels, baseline=baseline,
                         **dist_kw)
            dist = res.distribution if source == 'contin' else res
            tau_span = _fitted_tau_span(dist.fit_tau_s)
            # Partition provenance: the regularization the amplitudes came
            # from would otherwise be dropped with the inversion result here
            # (mode-resolved static intensities record it on the snapshot).
            alpha = dist.alpha
            # Same reason, same place: read the grid off the fit rather than
            # off the request, so it reports what the amplitudes came from
            # even when the caller passed no grid at all.
            _grid = np.asarray(dist.rh_grid_nm, dtype=float)
            if _grid.size:
                rh_lo = float(np.min(_grid))
                rh_hi = float(np.max(_grid))
                n_grid_used = int(_grid.size)
            # Same reason again: which residual weight built this inversion, and what
            # that weight read. Off the FIT, like alpha and the grid above -- a caller
            # may have passed no `weighting` at all, and the engine's own default and
            # its fallback are only visible on the result.
            weighting = dist.weighting
            sample_time_provenance = dist.sample_time_provenance
            run_duration_form = dist.run_duration_form
            # And its NOTES, relayed. The inversion result is dropped here, so a note
            # left on it would never reach the map: the note a weight that could not be
            # formed carries says why a point records the form it does, and CONTIN's
            # alpha preconditions are caveats the mode amplitudes inherit.
            notes.extend(dist.notes)
            if source == 'contin':
                alpha_method = ('user' if res.alpha_was_user_supplied
                                else res.alpha_selection_method)
                gcv_gamma = res.gcv_gamma       # None unless the selector was GCV
                notes.extend(find_notes(res, kind='precondition'))
    except ValueError as exc:
        return ModeExtraction(modes=(), source_method=source, success=False,
                              notes=_refusal_notes(exc, source, measurement))

    if dist is not None:
        if manual_bounds_rh_nm:
            rh_grid = np.asarray(dist.rh_grid_nm, dtype=float)
            gamma_grid = np.asarray(dist.gamma_grid_s_inv, dtype=float)
            if gamma_grid.size != rh_grid.size:
                return ModeExtraction(     # the find_distribution_peaks guard
                    modes=(), source_method=source, success=False,
                    notes=(Note('grid_malformed', ()),), weighting=weighting)
            intervals, rh_s, w_s = integrate_between(
                rh_grid, dist.weights, manual_bounds_rh_nm)
            gamma_s = gamma_grid[np.argsort(rh_grid)]
            n_empty = 0
            for rh_max, frac, lo, hi in intervals:
                if not math.isfinite(rh_max):
                    n_empty += 1
                    continue
                peak_i = lo + int(np.argmax(w_s[lo:hi]))
                # The interval's position, and its grid-end verdict, by the one computation
                # each the automatic lobes use.
                mean = lobe_geometric_mean(rh_s, w_s, lo, hi, gamma_s)
                at_end = not _grid_end_check(peak_i, rh_s.size, float(rh_s[peak_i]))[0]
                raw.append((float(gamma_s[peak_i]), frac, float(rh_max),
                            (float(rh_s[lo]), float(rh_s[hi - 1])), 'manual',
                            mean.gamma_s_inv, mean.rh_nm, at_end))
            if n_empty:
                notes.append(Note('empty_manual_interval', (), {'n': n_empty}))
            # The standing rail (owner, 2026-08-14): a manual split exists
            # because peaks overlap, and overlap violates the wide-separation
            # validity condition -- the caveat travels unconditionally.
            notes.append(Note('manual_bounds', ()))
        else:
            peaks = find_distribution_peaks(
                dist, min_weight_fraction, min_decay_completeness=min_decay_completeness,
                max_retained_acf_ratio=max_retained_acf_ratio)
            for pk in peaks:
                raw.append((pk.gamma_s_inv, pk.weight_fraction, pk.rh_nm,
                            (pk.rh_lo_nm, pk.rh_hi_nm), 'auto',
                            pk.mean_gamma_s_inv, pk.mean_rh_nm, has_note(pk, 'grid_end')))

    # A mode whose Gamma is not a positive finite number cannot be placed on
    # the map or picked by value; skip it with a note (disclosed, not silent).
    usable = [t for t in raw if math.isfinite(t[0]) and t[0] > 0
              and math.isfinite(t[5]) and t[5] > 0]
    if len(usable) < len(raw):
        notes.append(Note('unusable_rate_dropped', ('modes',), {
            'n_dropped': len(raw) - len(usable), 'n': len(raw), 'points': 'modes',
            'quantity': 'Γ'}))
    if not usable:
        return ModeExtraction(modes=(), source_method=source, success=False,
                              notes=_all_notes(Note('no_modes', ()), notes),
                              weighting=weighting)

    usable.sort(key=lambda t: -t[5])          # fast -> slow (Gamma descending)
    n = len(usable)
    # A distribution source's correlogram decay, asked once of the fit, fails every one of
    # its points (a recommendation the owner accepted, 2026-10-17); the parametric source's
    # verdict is its own rate test, `fit_double_exponential`'s.
    data_ok = (_fit_decay_check(dist, max_retained_acf_ratio)[0] if dist is not None
               else True)

    def _decay_fields(gamma: float) -> dict:
        # The completeness is computed ONCE and both the flag and the recorded value come
        # off that one number, so there is no second expression left to size against the
        # first (rule 12). Two calls here would be two chances to pass different arguments.
        completeness = _decay_completeness(gamma, tau_span)
        reliable, _note = _decay_completeness_check(completeness,
                                                    min_decay_completeness)
        return dict(decay_incomplete=not (reliable and data_ok),
                    decay_completeness=completeness,
                    min_decay_completeness=float(min_decay_completeness))

    modes = tuple(
        ModePoint(
            angle_deg=float(measurement.angle_deg),
            q_m_inv=float(q), q2_m2=float(q2),
            gamma_s_inv=float(g),
            amplitude_fraction=float(a),
            d_m2_s=d_from_gamma(mg, q),       # from the position, as everything derived
            rh_nm=float(r),
            source_method=source, mode_index=i, n_modes_in_run=n,
            low_amplitude=bool(a < amplitude_floor),
            amplitude_floor=float(amplitude_floor),
            **_decay_fields(mg),
            alpha=alpha, alpha_selection_method=alpha_method, gcv_gamma=gcv_gamma,
            rh_min_nm=rh_lo, rh_max_nm=rh_hi, n_grid=n_grid_used,
            weighting=weighting,
            sample_time_provenance=sample_time_provenance,
            run_duration_form=run_duration_form,
            bounds_lo_rh_nm=float(blo), bounds_hi_rh_nm=float(bhi),
            bounds_source=bsrc,
            mean_gamma_s_inv=float(mg), mean_rh_nm=float(mr),
            grid_end=bool(at_end),
        )
        for i, (g, a, r, (blo, bhi), bsrc, mg, mr, at_end) in enumerate(usable)
    )
    return ModeExtraction(modes=modes, source_method=source, success=True,
                          notes=_all_notes(notes), weighting=weighting)


@dataclass(frozen=True)
class ModeRunOutcome:
    """One run's mode extraction, as `extract_mode_runs` returns it: the `ModeExtraction`
    with its points tagged with the run's provenance, or None when the extraction
    refused its inputs (a `ValueError`)."""
    extraction: Optional[ModeExtraction]
    # The refusal's own notes (`core.notes.RefusedInputs.notes`); () for a refusal that
    # names no cause, and for a run that was extracted.
    refusal: Tuple[Note, ...] = ()


def extract_mode_runs(
    runs: Sequence[Tuple[DLSMeasurement, Mapping[str, Any],
                         Optional[Sequence[Sequence[float]]], Any, Any]],
    source: str,
    **kw,
) -> Tuple[ModeRunOutcome, ...]:
    """Every mode of every run of a mode map, one `extract_modes` call per run.

    Each run is (measurement, its own keyword arguments, its manual integration bounds,
    item_id, polarization state). A run's keyword arguments are laid over the shared
    `**kw`, and the two provenance values are opaque here: each point the run yields
    carries them as `ModePoint.item_id` and `ModePoint.polarization_state`.

    Returns one outcome per run, in the order given. An extraction that refuses its
    inputs (a `ValueError`) leaves its run with no points and its refusal's notes, so
    one weird run never kills the map; any other exception propagates.
    """
    out: List[ModeRunOutcome] = []
    for m, run_kw, bounds, item_id, polarization_state in runs:
        # Manual integration bounds apply to distribution sources only: the parametric
        # 'double' would refuse them, so they stay with the caller, dormant, until a
        # distribution run.
        mb = bounds if source != 'double' else None
        try:
            ext = extract_modes(m, source, manual_bounds_rh_nm=mb, **{**kw, **run_kw})
        except ValueError as exc:
            out.append(ModeRunOutcome(
                extraction=None,
                refusal=tuple(exc.notes) if isinstance(exc, RefusedInputs) else ()))
            continue
        tagged = tuple(replace(pt, item_id=item_id, polarization_state=polarization_state)
                       for pt in ext.modes)
        out.append(ModeRunOutcome(extraction=replace(ext, modes=tagged)))
    return tuple(out)


# ---------------------------------------------------------------------------
# 2. Per-family Gamma vs q^2 fit
# ---------------------------------------------------------------------------

def fit_mode_family(
    points: Sequence[ModePoint],
    *,
    temperature_K: float,
    viscosity_Pa_s: Optional[float],
    r2_threshold: float = DIFFUSIVE_R2_MIN,
    intercept_rel_threshold: float = DIFFUSIVE_INTERCEPT_REL_MAX,
    estimator: str = 'hc3',
) -> ModeFamilyFitResult:
    """Fit one picked mode family's Gamma vs q^2.

    Through-origin slope D (Gamma = D q^2) with its regression SE, an
    unconstrained fit for the intercept/R^2 diagnostics, and the q^2+q^4
    deviation form -- the same criteria, thresholds and verdict name as
    `analyze_gamma_q2` (rule 10). The diffusive verdict is a LABEL; Rh is
    computed whenever D is usable and is APPARENT + MODE-RESOLVED by
    construction (the caller flags it so on every surface).

    Parameters
    ----------
    points : sequence of ModePoint
        The user's picked family: two or more modes at distinct angles.
        Which points form a family is the user's decision -- this function
        never associates.
    temperature_K, viscosity_Pa_s
        For Stokes-Einstein. viscosity may be None -> Rh is NaN (the decay
        rate and D never need it).
    r2_threshold, intercept_rel_threshold
        The through-origin verdict's documented tolerances (defaults
        DIFFUSIVE_R2_MIN (0.98) and DIFFUSIVE_INTERCEPT_REL_MAX (0.1) --
        general statistical criteria, echoed onto the result).

    Raises
    ------
    ValueError
        Fewer than two points, not at distinct angles, or a non-finite q^2
        (a parameter error, disclosed exactly as in analyze_gamma_q2).
    """
    if len(points) < 2:
        raise ValueError(
            f"A mode-family fit needs at least two picked modes at distinct "
            f"angles, got {len(points)}."
        )

    notes: List[Note] = []            # the records of the pick set

    angles = np.array([p.angle_deg for p in points], dtype=float)
    if np.unique(angles).size < 2:
        raise ValueError(
            "A mode-family fit needs picked modes at distinct angles; the "
            "picked points are not at different angles."
        )

    q = np.array([p.q_m_inv for p in points], dtype=float)
    q2 = np.array([p.q2_m2 for p in points], dtype=float)
    if not np.all(np.isfinite(q2)):
        raise ValueError(
            "The scattering vector q is not finite for at least one picked "
            "mode, so Gamma vs q^2 cannot be fit. q = 4*pi*n/lambda*"
            "sin(theta/2) -- check the solvent refractive index and the "
            "wavelength on the source measurements."
        )
    # Each point's POSITION (the lobe's geometric mean for a distribution source), the
    # Gamma its `d_m2_s` was derived from -- one Gamma per point (`plotted_gamma_s_inv`).
    gamma = np.array([plotted_gamma_s_inv(p) for p in points], dtype=float)
    amps = np.array([p.amplitude_fraction for p in points], dtype=float)
    d_app = np.array([p.d_m2_s for p in points], dtype=float)

    # Disclosures on the pick set (doctrine rule 2: compute and note).
    methods = {p.source_method for p in points}
    if len(methods) > 1:
        # Single source of the message -> both the note and the warning
        # (keep-both), the analyze_gamma_q2 identity-mix shape.
        mixed = Note('mixed_extraction_methods', (),
                     {'methods': tuple(sorted(_SOURCE_FIT_NAMES[m] for m in methods))})
        notes.append(mixed)
        warnings.warn(render(mixed, surface='export'), UserWarning, stacklevel=2)
    ids = [p.item_id for p in points if p.item_id]
    same_run = len(ids) != len(set(ids))
    if same_run:
        # Part of the residual spread is then WITHIN one correlogram, whose lag channels
        # are correlated, so an independent-points SE would under-report (below).
        notes.append(Note('se_withheld_same_run', ('d_se', 'rh_se', 'a_se', 'b_se')))
    n_low = int(sum(1 for p in points if p.low_amplitude))
    if n_low:
        floor = max(p.amplitude_floor for p in points)
        notes.append(Note('low_amplitude_points', ('amplitude_fractions',), {
            'n': n_low, 'total': len(points), 'floor': float(floor)}))
    # The point verdicts the family's Rh rests on, each its own count note (a mode can be
    # strong and undecayed, or weak and fully decayed): about the decay rates they judge
    # and every value fitted from them, so the precondition reaches each export cell.
    point_about = ('gamma_s_inv', 'd_m2_s', 'rh_nm', 'a_m2_s', 'b_m4_s',
                   'd_se', 'rh_se', 'a_se', 'b_se')
    n_undecayed = int(sum(1 for p in points if p.decay_incomplete))
    undecayed_note = None
    if n_undecayed:
        undecayed_note = Note('undecayed_points', point_about, {
            'n': n_undecayed, 'total': len(points)})
        notes.append(undecayed_note)
    n_grid_end = int(sum(1 for p in points if getattr(p, 'grid_end', False)))
    grid_end_note = None
    if n_grid_end:
        grid_end_note = Note('grid_end_points', point_about,
                             {'n': n_grid_end, 'total': len(points)})
        notes.append(grid_end_note)
    # A point outside the technique's size range fails the family's Rh as an angle's does
    # Gamma vs q^2's (`angular._pooled_point_notes`, the shape: the count, then the first
    # point's own reason where it was found), the size asked only of a decay the point
    # observed (the one precedence of `cumulants._cumulant_flags`), on the point's
    # position (a recommendation the owner accepted, 2026-10-17). Every such cause bears
    # on the decay rate (`DECAY_BEARING_CODES`), so the count is the decay-bearing one.
    out_of_range = [(float(a), n) for a, p in zip(angles, points, strict=True)
                    if not p.decay_incomplete
                    for n in (size_range_check(_point_rh_nm(p), CUMULANT_MAX_RH_NM,
                                                CUMULANT_MIN_RH_NM)[1],) if n is not None]
    range_notes: Tuple[Note, ...] = ()
    if out_of_range:
        first_at, first = out_of_range[0]
        range_notes = (
            Note('inputs_flagged_decay', ('rh_nm',), {
                'k': len(out_of_range), 'n': len(points), 'inputs': 'mode points',
                WHERE: tuple(a for a, _n in out_of_range), WHERE_UNIT: 'deg'}),
            replace(first, values={**first.values, WHERE: (first_at,), WHERE_UNIT: 'deg'}))
    notes.extend(range_notes)

    order = np.argsort(q2)
    angles, q, q2, gamma = angles[order], q[order], q2[order], gamma[order]
    amps, d_app = amps[order], d_app[order]

    # Through-origin slope D with its SE (independent angles -> defensible).
    d, d_se = unc.linear_fit_through_origin(q2, gamma, estimator)
    d = float(d)

    # Unconstrained fit for the intercept and R^2 (same math as the sibling).
    slope_u, intercept_u = unc.polyfit_or_nan(q2, gamma, 1)
    gamma_fit = slope_u * q2 + intercept_u
    ss_res = float(np.sum((gamma - gamma_fit) ** 2))
    ss_tot = float(np.sum((gamma - gamma.mean()) ** 2))
    r2 = unc.r_squared_or_nan(ss_res, ss_tot)
    _gbar = float(gamma.mean())
    intercept_rel = (intercept_u / _gbar
                     if (math.isfinite(_gbar) and _gbar != 0) else float('nan'))

    is_diffusive = bool(
        math.isfinite(r2) and r2 >= r2_threshold
        and abs(intercept_rel) <= intercept_rel_threshold
    )

    # The q^2+q^4 deviation form: Gamma = a q^2 + b q^4, no constant column
    # (the printed form has none -- Sedlak & Amis 1992a p. 819). Two free
    # parameters, so it needs >= 3 distinct abscissae; with fewer it is
    # WITHHELD with a note (the through-origin fit above is the primary
    # result, so the secondary form degrades, it does not raise).
    a = b = q4_r2 = q4_cond = float('nan')
    a_se = b_se = None
    q4_ok = False
    q4_note = None
    # `q2.max() > 0` beside the count: q2 is already finite (raised on above),
    # but an all-zero abscissa would make the normalization below divide by
    # zero -- the same degenerate design linear_fit_through_origin withholds on.
    if np.unique(q2).size >= 3 and q2.max() > 0:
        # Fit on the NORMALIZED abscissa u = q^2 / max(q^2), not on [q^2, q^4]
        # directly: with q^2 ~ 1e14 m^-2 the raw columns differ in scale by
        # ~1e14, putting the design's small singular value at lstsq's rcond
        # truncation boundary and making `well_conditioned` fire on every real
        # angle set. Gamma = alpha u + beta u^2 is an EXACT reparameterization
        # (a = alpha/s, b = beta/s^2, s = max(q^2)); the linear rescale carries
        # the SEs, inheriting the covariance's validation. `q4_cond` reports
        # the scaled design actually fitted -- the honest health number.
        s = float(q2.max())
        u = q2 / s
        X = np.column_stack([u, u * u])
        mf = unc.multilinear_fit(X, gamma, estimator)
        a, b = float(mf.coeffs[0]) / s, float(mf.coeffs[1]) / (s * s)
        q4_r2 = mf.r_squared
        q4_cond = mf.cond
        q4_ok = bool(mf.well_conditioned)
        # Unit-jacobian propagate for a single coefficient's SE off the 2x2
        # covariance -- the public route (the form-factor Rg SE precedent),
        # not the private _diag_se -- then the exact rescale.
        alpha_se = unc.se_or_none(unc.propagate([1.0, 0.0], mf.cov))
        beta_se = unc.se_or_none(unc.propagate([0.0, 1.0], mf.cov))
        a_se = alpha_se / s if alpha_se is not None else None
        b_se = beta_se / (s * s) if beta_se is not None else None
        if not q4_ok:
            q4_note = Note('ill_conditioned', ('a_m2_s', 'b_m4_s'),
                           {'cond': q4_cond, 'limit': mf.cond_limit})
    else:
        q4_note = Note('q4_not_fitted', ('a_m2_s', 'b_m4_s'),
                       {'n': int(np.unique(q2).size), 'need': 3})

    # Apparent, mode-resolved Rh -- ALWAYS computed when D is usable (the
    # verdict above never gates it; owner-ruled 2026-08-14). isfinite as well
    # as > 0: +inf passes a bare > 0 and stokes_einstein_rh refuses it loudly,
    # where this path's contract is NaN (the analyze_gamma_q2 gate, rule 10).
    if viscosity_Pa_s is not None and math.isfinite(d) and d > 0:
        rh = stokes_einstein_rh(d, temperature_K, viscosity_Pa_s) * 1e9
    else:
        rh = float('nan')
    rh_se = unc.power_law_se(rh, d, d_se, -1) if math.isfinite(rh) else None

    # The family's Rh verdict, `analyze_gamma_q2`'s roll-up: the point verdicts above, the
    # technique's size range on the family Rh itself and the viscosity a blank Rh lacks.
    size_note = size_range_check(rh, CUMULANT_MAX_RH_NM, CUMULANT_MIN_RH_NM)[1]
    visc_note = _no_viscosity_note() if viscosity_Pa_s is None else None
    rh_reliable, _rh_notes = _result_flags(undecayed_note, grid_end_note, *range_notes,
                                           size_note, visc_note)
    notes.extend(n for n in (size_note, visc_note) if n is not None)
    if not is_diffusive:
        notes.append(_not_diffusive_note(r2, intercept_rel))

    # Invariant 8: the SEs above are
    # licensed by the points being INDEPENDENT angles. Two modes of one
    # correlogram in the set break that -- the spread is partly within one
    # run's correlated lag channels -- so every regression SE is withheld
    # (the values still report; only the +- is disqualified).
    if same_run:
        d_se = rh_se = a_se = b_se = None

    return ModeFamilyFitResult(
        angles_deg=angles, q_m_inv=q, q2_m2=q2, gamma_s_inv=gamma,
        d_app_m2_s=d_app, amplitude_fractions=amps,
        d_m2_s=d, rh_nm=rh,
        r_squared=r2, intercept_s_inv=float(intercept_u),
        intercept_relative=float(intercept_rel),
        is_diffusive=is_diffusive,
        r2_threshold=r2_threshold,
        intercept_rel_threshold=intercept_rel_threshold,
        a_m2_s=a, b_m4_s=b, q4_r_squared=q4_r2, q4_cond=q4_cond,
        q4_well_conditioned=q4_ok,
        source_method=(sorted(methods)[0] if len(methods) == 1 else 'mixed'),
        temperature_K=temperature_K,
        d_se=unc.se_or_none(d_se), rh_se=unc.se_or_none(rh_se),
        a_se=a_se, b_se=b_se,
        se_estimator=estimator,
        n_low_amplitude=n_low,
        n_decay_incomplete=n_undecayed,
        notes=_all_notes(q4_note, notes),
        rh_reliable=rh_reliable,
    )


# ---------------------------------------------------------------------------
# 3. Fast/slow amplitude-vs-q report
# ---------------------------------------------------------------------------

def amplitude_vs_q(
    fast_points: Sequence[ModePoint],
    slow_points: Sequence[ModePoint],
    *,
    flatness_rel_threshold: float = 0.2,
) -> AmplitudeRatioResult:
    """The fast/slow amplitude report: A_f/A_s vs sin^2(theta/2).

    Angles are matched through the shared identity form
    (`round_angle_for_identity`), never raw float equality. An angle bucket
    where either family holds MORE than one point is skipped with a note
    naming it (no invented averaging); an angle present in only one family is
    skipped and counted in a note. With no shared angles the result is empty
    with a note -- withheld, not raised (an ordinary picking state, not an
    input error).

    `fast_flat` is (max-min)/mean of A_f <= `flatness_rel_threshold` -- a
    DISCLOSED house default (no printed criterion in the held sources), a
    diagnostic that gates a note only. It is False with a note when fewer
    than two shared angles make flatness unanswerable (the is_diffusive
    convention for an unanswerable verdict).
    """
    notes: List[Note] = []            # the records of the pairing

    def _by_angle(pts: Sequence[ModePoint], label: str):
        buckets: dict = {}
        for p in pts:
            buckets.setdefault(round_angle_for_identity(p.angle_deg),
                               []).append(p)
        multi = sorted(a for a, lst in buckets.items() if len(lst) > 1)
        if multi:
            # Skipped there: no averaging is invented.
            notes.append(Note('multiple_points_at_angle', (), {
                'family': label, 'angles': tuple(float(a) for a in multi)}))
        return {a: lst[0] for a, lst in buckets.items() if len(lst) == 1}

    fast = _by_angle(fast_points, 'fast')
    slow = _by_angle(slow_points, 'slow')

    candidates = sorted(set(fast) & set(slow))
    n_only = len(set(fast) ^ set(slow))
    if n_only:
        notes.append(Note('angles_in_one_family', (), {'n': n_only}))
    # A_f/A_s is defined WITHIN one decomposition: the modes of one run sum
    # to ~1, so a pair drawn from two different runs (or two polarization
    # states -- different scattering channels) is a ratio of two unrelated
    # normalizations (invariant 7).
    # Both provenance fields already travel on every ModePoint.
    shared, mismatched = [], []
    for a in candidates:
        f, s = fast[a], slow[a]
        if (f.item_id != s.item_id
                or f.polarization_state != s.polarization_state):
            mismatched.append(a)
        else:
            shared.append(a)
    if mismatched:
        # A_f/A_s is only defined within one decomposition (one run, one polarization).
        notes.append(Note('mixed_runs_at_angle', ('ratio_f_over_s',), {
            'angles': tuple(float(a) for a in mismatched)}))
    if not shared:
        return AmplitudeRatioResult(
            angles_deg=np.array([]), sin2_half_theta=np.array([]),
            fast_fraction=np.array([]), slow_fraction=np.array([]),
            ratio_f_over_s=np.array([]),
            fast_flat=False, fast_rel_span=float('nan'),
            flatness_rel_threshold=flatness_rel_threshold,
            n_shared_angles=0,
            notes=_all_notes(Note('too_few_shared_angles', ('fast_flat',),
                                  {'n': 0, 'need': 2}), notes),
        )

    angles = np.array(shared, dtype=float)
    a_f = np.array([fast[a].amplitude_fraction for a in shared], dtype=float)
    a_s = np.array([slow[a].amplitude_fraction for a in shared], dtype=float)
    sin2 = np.sin(np.radians(angles) / 2.0) ** 2

    # A_f/A_s, withheld (NaN) where unanswerable rather than emitting inf
    # from a zero or non-finite slow amplitude.
    answerable = np.isfinite(a_f) & np.isfinite(a_s) & (a_s > 0)
    ratio = np.full(angles.shape, float('nan'))
    ratio[answerable] = a_f[answerable] / a_s[answerable]
    if not answerable.all():
        # The slow-mode amplitude there is zero or not finite.
        notes.append(Note('ratio_withheld', ('ratio_f_over_s',),
                          {'n': int((~answerable).sum())}))

    # Fast-amplitude angle-flatness: (max-min)/mean against the documented
    # threshold. Unanswerable (< 2 shared angles, or a degenerate mean) ->
    # False with a note, the is_diffusive convention.
    finite_f = a_f[np.isfinite(a_f)]
    mean_f = float(finite_f.mean()) if finite_f.size else float('nan')
    if finite_f.size >= 2 and math.isfinite(mean_f) and mean_f > 0:
        rel_span = float((finite_f.max() - finite_f.min()) / mean_f)
        flat = bool(rel_span <= flatness_rel_threshold)
        flat_note = (None if flat else Note('fast_amplitude_not_flat', ('fast_flat',), {
            'span': rel_span, 'limit': float(flatness_rel_threshold)}))
    else:
        rel_span = float('nan')
        flat = False
        # The usable shared angles: finite fast amplitudes with a positive mean.
        n_usable = int(finite_f.size) if (math.isfinite(mean_f) and mean_f > 0) else 0
        flat_note = Note('too_few_shared_angles', ('fast_flat',),
                         {'n': n_usable, 'need': 2})

    return AmplitudeRatioResult(
        angles_deg=angles, sin2_half_theta=sin2,
        fast_fraction=a_f, slow_fraction=a_s, ratio_f_over_s=ratio,
        fast_flat=flat, fast_rel_span=rel_span,
        flatness_rel_threshold=flatness_rel_threshold,
        n_shared_angles=int(angles.size),
        notes=_all_notes(flat_note, notes),
    )
