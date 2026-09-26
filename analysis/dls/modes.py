"""
analysis/dls/modes.py
=====================

Mode-resolved dynamics. Three pure steps:

  1. `extract_modes`      -- every relaxation mode of ONE measurement, as
                             pickable `ModePoint`s (the rows of the "mode map",
                             the (q^2, Gamma) scatter of Sedlak & Amis 1992
                             Fig. 3).
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
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from core.data_models import DLSMeasurement, round_angle_for_identity
from analysis import uncertainty as unc
from physics.constants import stokes_einstein_rh
from analysis.dls._common import (
    _d_from_gamma,
    _decay_completeness,
    _decay_completeness_check,
    _measurement_q_m,
    DIFFUSIVE_INTERCEPT_REL_MAX,
    DIFFUSIVE_R2_MIN,
    MIN_DECAY_COMPLETENESS,
)
from analysis.dls.exponentials import fit_double_exponential
from analysis.dls.distributions import (
    find_distribution_peaks, fit_contin, fit_nnls, integrate_between)


# The mode-amplitude resolvability floor: below ~2 % of the total scattered
# intensity a mode's parameters are imprecise (Sedlak 2001, Physical Chemistry
# of Polyelectrolytes ch. 1, pp. 7-8; the dashed-line caveat of Sedlak & Amis
# 1992a p. 820). A documented default, not a magic number (invariant 4); it
# gates a NOTE on the point, never its existence -- a sub-floor mode still
# appears on the map and is still pickable.
DEFAULT_AMPLITUDE_FLOOR = 0.02

MODE_SOURCES = ('double', 'contin', 'nnls')


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

    `item_id` and `polarization_state` are provenance attached by the
    controller (`dataclasses.replace`); the analysis layer leaves the defaults.
    `item_id` is NEVER a pick key.

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
    """
    angle_deg: float
    q_m_inv: float
    q2_m2: float
    gamma_s_inv: float
    amplitude_fraction: float
    d_m2_s: float                     # Gamma/q^2 via _d_from_gamma (NaN-capable)
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
    # distribution peak, and one predicate covers all three (rule 12).
    decay_incomplete: bool = False    # decay_completeness < min_decay_completeness
    decay_completeness: float = float('nan')      # Gamma * (fitted lag span)
    min_decay_completeness: float = MIN_DECAY_COMPLETENESS
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


@dataclass
class ModeExtraction:
    """Every usable mode of one measurement, fast->slow."""
    modes: tuple                      # ModePoint, Gamma descending
    source_method: str
    success: bool
    notes: tuple = ()


@dataclass
class ModeMapRun:
    """One sample's mode map: every extracted mode of every eligible run.

    Assembled by the controller (which walks the sample and attaches item_id /
    polarization_state); cached as the substrate the user picks families from.
    `source_method` is recorded on the result, not the cache key (the
    gamma_source precedent).
    """
    points: tuple                     # ModePoint, all runs pooled
    source_method: str
    notes: tuple = ()


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
    # passive, glyph-less notices for the GUI's info tier (keep-both where noted)
    notes: tuple = ()


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
    fast_flat: bool                   # verdict; False when unanswerable (note says why)
    fast_rel_span: float              # (max - min)/mean of A_f; NaN when < 2 angles
    flatness_rel_threshold: float
    n_shared_angles: int
    notes: tuple = ()


def _fitted_tau_span(fit_tau_s) -> float:
    """The lag span of a fit's own windowed tau array, or NaN when there is none.

    Mirrors `exponentials._tau_span_s`; kept here rather than imported because this module
    reads the array off a RESULT (which may be empty on a degenerate fit) while that one
    holds the array it just windowed. Both feed the one `_decay_completeness`.
    """
    tau = np.asarray(fit_tau_s, dtype=float)
    return float(tau[-1] - tau[0]) if tau.size >= 2 else float('nan')


# ---------------------------------------------------------------------------
# 1. Per-measurement mode extraction
# ---------------------------------------------------------------------------

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
    resolvability limit -- below that limit a lobe is indistinguishable from
    inversion noise (Sedlak 2001 pp. 7-8), so dropping it is extraction
    granularity, not a gate on a result. Modes above the floor appear
    unflagged; a double-exponential mode below it appears WITH the
    `low_amplitude` note-flag (that path has no noise threshold).

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
    verdict by construction. It gates a note, never a refusal.

    A distribution source's own PASSIVE notes are relayed verbatim onto the
    extraction -- that its residual weight ran without a committed count rate,
    or that the weight could not be formed and which input refused -- because
    the inversion result is dropped here and a note left on it would reach no
    surface. They are the notes that say why a point records the form it does.

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

    notes: List[str] = []
    q = _measurement_q_m(measurement)
    q2 = q * q                    # `*` saturates to inf; `**` would raise (rule 12)
    if not (math.isfinite(q2) and q2 > 0):
        # A non-finite/zero q^2 is a PARAMETER error (refractive index /
        # wavelength / angle), same disclosure as the analyze_gamma_q2 gate --
        # but per-measurement extraction is failure-tolerant (the map shows
        # every OTHER run), so this withholds with a note instead of raising.
        return ModeExtraction(
            modes=(), source_method=source, success=False,
            notes=("The scattering vector q is not usable for this measurement "
                   "(q = 4*pi*n/lambda*sin(theta/2)) -- check the solvent "
                   "refractive index and the wavelength.",))

    # (gamma, amplitude, rh, (bounds_lo, bounds_hi), bounds_source) tuples from
    # the chosen source, unordered. The fitter call is wrapped: an unfittable
    # CORRELOGRAM raises ValueError out of the underlying fits (e.g. the
    # cumulant seed's "0 points survive the cutoff" on a flat trace), and for a
    # per-measurement map extraction that is a data failure to disclose, not an
    # error to propagate -- one weird run must not kill a multi-angle map. The
    # message travels as the note. Manual-bounds integration runs OUTSIDE the
    # wrap: its ValueError is an input-contract violation, not a data failure.
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
                    notes=("Double-exponential fit failed; no modes extracted "
                           "from this measurement.",))
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
                            _no_bounds, ''))
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
            # And its PASSIVE NOTES, relayed verbatim. The inversion result is dropped
            # here, so a note left on it would never reach the map: the no-count-rate
            # note and the note a weight that could not be formed carries are the two
            # that say why a point records the form it does.
            notes.extend(dist.notes)
            if source == 'contin':
                alpha_method = ('user' if res.alpha_was_user_supplied
                                else res.alpha_selection_method)
                gcv_gamma = res.gcv_gamma       # None unless the selector was GCV
                if res.reliability_note:
                    # The one sentence every surface shows (ContinResult carries it).
                    notes.append(res.reliability_note
                                 + '; the mode amplitudes inherit that caveat.')
    except ValueError as exc:
        return ModeExtraction(
            modes=(), source_method=source, success=False,
            notes=(f"No modes could be extracted from this measurement: "
                   f"{exc}",))

    if dist is not None:
        if manual_bounds_rh_nm:
            rh_grid = np.asarray(dist.rh_grid_nm, dtype=float)
            gamma_grid = np.asarray(dist.gamma_grid_s_inv, dtype=float)
            if gamma_grid.size != rh_grid.size:
                return ModeExtraction(     # the find_distribution_peaks guard
                    modes=(), source_method=source, success=False,
                    notes=("The distribution's decay-rate grid is malformed; "
                           "no modes extracted.",))
            intervals, rh_s, w_s = integrate_between(
                rh_grid, dist.weights, manual_bounds_rh_nm)
            gamma_s = gamma_grid[np.argsort(rh_grid)]
            n_empty = 0
            for rh_max, frac, lo, hi in intervals:
                if not math.isfinite(rh_max):
                    n_empty += 1
                    continue
                peak_i = lo + int(np.argmax(w_s[lo:hi]))
                raw.append((float(gamma_s[peak_i]), frac, float(rh_max),
                            (float(rh_s[lo]), float(rh_s[hi - 1])), 'manual'))
            if n_empty:
                notes.append(
                    f"{n_empty} manual integration interval(s) captured no "
                    f"weight and produced no mode."
                )
            # The standing rail (owner, 2026-08-14): a manual split exists
            # because peaks overlap, and overlap violates the wide-separation
            # validity condition -- the caveat travels unconditionally.
            notes.append(
                "Manual integration bounds were used for this run's mode "
                "amplitudes. Overlapping peaks violate the wide-separation "
                "validity condition of the mode decomposition; for genuinely "
                "overlapping modes the parametric (double-exponential) route "
                "is the better separator."
            )
        else:
            peaks = find_distribution_peaks(dist, min_weight_fraction)
            for pk in peaks:
                raw.append((pk.gamma_s_inv, pk.weight_fraction, pk.rh_nm,
                            (pk.rh_lo_nm, pk.rh_hi_nm), 'auto'))

    # A mode whose Gamma is not a positive finite number cannot be placed on
    # the map or picked by value; skip it with a note (disclosed, not silent).
    usable = [t for t in raw if math.isfinite(t[0]) and t[0] > 0]
    if len(usable) < len(raw):
        notes.append(
            f"Skipped {len(raw) - len(usable)} mode(s) with no usable decay "
            f"rate (non-positive or non-finite Gamma)."
        )
    if not usable:
        notes.append("No usable modes were extracted from this measurement.")
        return ModeExtraction(modes=(), source_method=source, success=False,
                              notes=tuple(notes))

    usable.sort(key=lambda t: -t[0])          # fast -> slow (Gamma descending)
    n = len(usable)

    def _decay_fields(gamma: float) -> dict:
        # The completeness is computed ONCE and both the flag and the recorded value come
        # off that one number, so there is no second expression left to size against the
        # first (rule 12). Two calls here would be two chances to pass different arguments.
        completeness = _decay_completeness(gamma, tau_span)
        reliable, _note = _decay_completeness_check(completeness,
                                                    min_decay_completeness)
        return dict(decay_incomplete=not reliable,
                    decay_completeness=completeness,
                    min_decay_completeness=float(min_decay_completeness))

    modes = tuple(
        ModePoint(
            angle_deg=float(measurement.angle_deg),
            q_m_inv=float(q), q2_m2=float(q2),
            gamma_s_inv=float(g),
            amplitude_fraction=float(a),
            d_m2_s=_d_from_gamma(g, q),
            rh_nm=float(r),
            source_method=source, mode_index=i, n_modes_in_run=n,
            low_amplitude=bool(a < amplitude_floor),
            amplitude_floor=float(amplitude_floor),
            **_decay_fields(g),
            alpha=alpha, alpha_selection_method=alpha_method, gcv_gamma=gcv_gamma,
            rh_min_nm=rh_lo, rh_max_nm=rh_hi, n_grid=n_grid_used,
            weighting=weighting,
            sample_time_provenance=sample_time_provenance,
            run_duration_form=run_duration_form,
            bounds_lo_rh_nm=float(blo), bounds_hi_rh_nm=float(bhi),
            bounds_source=bsrc,
        )
        for i, (g, a, r, (blo, bhi), bsrc) in enumerate(usable)
    )
    return ModeExtraction(modes=modes, source_method=source, success=True,
                          notes=tuple(notes))


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

    notes: List[str] = []

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
    gamma = np.array([p.gamma_s_inv for p in points], dtype=float)
    amps = np.array([p.amplitude_fraction for p in points], dtype=float)
    d_app = np.array([p.d_m2_s for p in points], dtype=float)

    # Disclosures on the pick set (doctrine rule 2: compute and note).
    methods = {p.source_method for p in points}
    if len(methods) > 1:
        # Single source of the message -> both the note and the warning
        # (keep-both), the analyze_gamma_q2 identity-mix shape.
        msg = ("The picked modes come from different extraction methods "
               f"({', '.join(sorted(methods))}). A mode family should be "
               "picked from one method's map.")
        notes.append(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)
    ids = [p.item_id for p in points if p.item_id]
    same_run = len(ids) != len(set(ids))
    if same_run:
        notes.append(
            "Two or more picked modes belong to the same measurement -- a "
            "mode family normally takes at most one mode per run. The "
            "regression SEs are withheld: part of the residual spread is "
            "then WITHIN one correlogram, whose lag channels are correlated, "
            "so an independent-points SE would under-report."
        )
    n_low = int(sum(1 for p in points if p.low_amplitude))
    if n_low:
        floor = max(p.amplitude_floor for p in points)
        notes.append(
            f"{n_low} of {len(points)} picked mode(s) fall below the "
            f"{floor:.0%} amplitude floor; their decay rates and amplitudes "
            f"are imprecise."
        )
    n_undecayed = int(sum(1 for p in points if p.decay_incomplete))
    if n_undecayed:
        # A separate judgement from the amplitude floor, so it gets its own line rather
        # than sharing one: a mode can be strong and undecayed, or weak and fully decayed.
        floor = max(p.min_decay_completeness for p in points)
        notes.append(
            f"{n_undecayed} of {len(points)} picked mode(s) did not finish "
            f"decaying inside their delay window (Gamma * lag span below "
            f"{floor:.3g}); their decay rates are extrapolations past the end "
            f"of the measurement, so the fitted D and Rh inherit that."
        )

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
            notes.append(
                f"The q^2+q^4 design is ill-conditioned (cond={q4_cond:.1e}); "
                f"a and b are numerically unreliable -- widen the angle range."
            )
    else:
        notes.append(
            "The q^2+q^4 deviation form needs at least three distinct "
            "angles; it was not fitted."
        )

    # Apparent, mode-resolved Rh -- ALWAYS computed when D is usable (the
    # verdict above never gates it; owner-ruled 2026-08-14). isfinite as well
    # as > 0: +inf passes a bare > 0 and stokes_einstein_rh refuses it loudly,
    # where this path's contract is NaN (the analyze_gamma_q2 gate, rule 10).
    if viscosity_Pa_s is not None and math.isfinite(d) and d > 0:
        rh = stokes_einstein_rh(d, temperature_K, viscosity_Pa_s) * 1e9
    else:
        rh = float('nan')
    rh_se = unc.power_law_se(rh, d, d_se, -1) if math.isfinite(rh) else None

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
        notes=tuple(notes),
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
    notes: List[str] = []

    def _by_angle(pts: Sequence[ModePoint], label: str):
        buckets: dict = {}
        for p in pts:
            buckets.setdefault(round_angle_for_identity(p.angle_deg),
                               []).append(p)
        multi = sorted(a for a, lst in buckets.items() if len(lst) > 1)
        if multi:
            angles_txt = ', '.join(f'{a:g}' for a in multi)
            notes.append(
                f"Skipped angle(s) {angles_txt} deg: the {label} family holds "
                f"more than one picked mode there (no averaging is invented)."
            )
        return {a: lst[0] for a, lst in buckets.items() if len(lst) == 1}

    fast = _by_angle(fast_points, 'fast')
    slow = _by_angle(slow_points, 'slow')

    candidates = sorted(set(fast) & set(slow))
    n_only = len(set(fast) ^ set(slow))
    if n_only:
        notes.append(
            f"{n_only} angle(s) present in only one family were skipped."
        )
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
        angles_txt = ', '.join(f'{a:g}' for a in mismatched)
        notes.append(
            f"Skipped angle(s) {angles_txt} deg: the fast and slow modes "
            f"there do not come from the same run and polarization state, "
            f"and A_f/A_s is only defined within one decomposition."
        )
    if not shared:
        notes.append(
            "The fast and slow families share no same-run angles; the "
            "amplitude report is empty."
        )
        return AmplitudeRatioResult(
            angles_deg=np.array([]), sin2_half_theta=np.array([]),
            fast_fraction=np.array([]), slow_fraction=np.array([]),
            ratio_f_over_s=np.array([]),
            fast_flat=False, fast_rel_span=float('nan'),
            flatness_rel_threshold=flatness_rel_threshold,
            n_shared_angles=0, notes=tuple(notes),
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
        notes.append(
            f"A_f/A_s withheld at {int((~answerable).sum())} angle(s): the "
            f"slow-mode amplitude there is zero or not finite."
        )

    # Fast-amplitude angle-flatness: (max-min)/mean against the documented
    # threshold. Unanswerable (< 2 shared angles, or a degenerate mean) ->
    # False with a note, the is_diffusive convention.
    finite_f = a_f[np.isfinite(a_f)]
    mean_f = float(finite_f.mean()) if finite_f.size else float('nan')
    if finite_f.size >= 2 and math.isfinite(mean_f) and mean_f > 0:
        rel_span = float((finite_f.max() - finite_f.min()) / mean_f)
        flat = bool(rel_span <= flatness_rel_threshold)
        if not flat:
            notes.append(
                f"The fast-mode amplitude varies by "
                f"{rel_span:.0%} of its mean across angles (threshold "
                f"{flatness_rel_threshold:.0%}); in the two-mode "
                f"phenomenology A_f is expected q-flat -- see the guide."
            )
    else:
        rel_span = float('nan')
        flat = False
        notes.append(
            "Fast-amplitude angle-flatness could not be assessed (fewer than "
            "two shared angles with a usable fast amplitude)."
        )

    return AmplitudeRatioResult(
        angles_deg=angles, sin2_half_theta=sin2,
        fast_fraction=a_f, slow_fraction=a_s, ratio_f_over_s=ratio,
        fast_flat=flat, fast_rel_span=rel_span,
        flatness_rel_threshold=flatness_rel_threshold,
        n_shared_angles=int(angles.size),
        notes=tuple(notes),
    )
