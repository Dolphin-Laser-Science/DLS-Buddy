"""
analysis/mode_partition.py
==========================

Mode-resolved static intensities: the Sedlak partition.

A DLS mode decomposition assigns each relaxation mode an intensity fraction
A_i(theta) (the modes of one run sum to ~1). Sedlak (1996, J. Chem. Phys.
105:10123, Eqs. 2-3, p. 10125) partitions the solvent-subtracted,
standard-normalized excess intensity by exactly those fractions -- so each
mode's share of the STATIC scattering becomes its own angular series,
analyzable on its own (his Eq. 7, p. 10130: a Guinier-type fit on the
slow-mode amplitude alone gives that mode's apparent size). This module is
the pure arithmetic of that method:

  1. `partition_rayleigh`   -- split one concentration's excess Rayleigh
                               ratio by per-angle mode amplitudes:
                               dR_i(theta) = A_i(theta) * dR(theta).
  2. `amplitude_guinier`    -- Sedlak Eq. 7 on one mode's series, in the
                               honest linear form 1/A vs q^2 (the intercept
                               IS the extrapolated A(0) the ratio needs).

Why the partition happens on the EXCESS RAYLEIGH side, never on raw
intensities: the mode amplitudes decompose the polymer solution's correlated
(excess) scattering -- solvent scatter is not mode-partitionable. Partitioning
a raw intensity would allocate A_i * I_solvent to each mode and the pipeline's
full-blank subtraction would then over-subtract each mode by
(1 - A_i) * I_solvent, angle-dependently. On the Rayleigh side the operation
is exact and unit-clean (a dimensionless fraction times cm^-1), and the whole
upstream pipeline -- dark and solvent subtraction, sin(theta), transmittance,
k_c and the (n/n_cal)^2 correction, the data mask -- is reused unchanged.

Each per-mode series is an ordinary `RayleighRatioResult` clone (the class
every SLS fitter consumes -- rule 10: one class, not a sibling to re-teach
seven fitters about), so per-mode Guinier/Debye/Zimm run on the partition
output directly, and the partition record travels in `derived_provenance`
into the export Comments exactly as trace-derivation provenance does.

Doctrine notes (core/data_models.py): the partition never refuses -- a
per-mode static result is degraded-apparent, never a category error (rule 1's
own fenced counter-example). Closure deficits, separation, low amplitudes and
unmatched angles are all NOTES that travel with the result (rule 2). N modes
are partitioned by the same closure as two (Sedlak 2001 Eq. 11's n-mode
form); nothing here is two-mode-specific.

Uncertainty (invariant 8): per-mode intensities carry NO +- -- each is one
ill-posed inversion's amplitude times one intensity (the NNLS/CONTIN stance).
The amplitude-Guinier Rg carries the regression SE across independent angles
(statistical only, EXCLUDING inversion systematics -- alpha, bounds), withheld
when two fitted points share a source run, exactly as `fit_mode_family` treats
the same amplitude class. Its Rg SE is delta-method through the same
slope/intercept jacobian as the Debye Rg -- but the transform is NONLINEAR
(the root of a coefficient ratio), so it carries its OWN Monte-Carlo gate in
the amplitudes' regime rather than inheriting the Debye validation:
measured CONSERVATIVE-safe (meets-or-exceeds the sampling spread, never
under-reporting) across 3-20 % amplitude scatter -- the
`test_amplitude_guinier_se_conservative_montecarlo` gate in
tests/test_mode_partition.py, the calibration-free-A2 precedent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

import numpy as np

from core.data_models import round_angle_for_identity
from analysis import uncertainty as unc
from analysis.sls import RayleighRatioResult

__all__ = ['PartitionModeInput', 'PartitionedMode', 'ModePartitionResult',
           'AmplitudeGuinierResult', 'partition_rayleigh', 'amplitude_guinier',
           'DEFAULT_CLOSURE_TOL', 'DEFAULT_SEPARATION_RATIO_MIN']

# Closure disclosure threshold: the picked modes' fractions should sum to ~1
# at every matched angle (Sedlak's closure I/I_B = sum A_i holds when the set
# is exhaustive). A deviation beyond this fraction gates a NOTE -- a
# documented default, not a magic number (invariant 4).
DEFAULT_CLOSURE_TOL = 0.05

# Mode-separation disclosure threshold: the decomposition is meaningful when
# the modes are well separated (Sedlak's own stated condition -- QUALITATIVE
# in every held source, so this numeric default is a disclosed HOUSE
# convenience, invariant 4). Expressed as the minimum ratio of adjacent
# modes' decay rates at a shared angle; gates a NOTE, never a refusal.
DEFAULT_SEPARATION_RATIO_MIN = 5.0


@dataclass(frozen=True)
class PartitionModeInput:
    """One mode family's per-angle amplitudes, as primitives.

    The controller unpacks a recorded mode set into these (one per family,
    one entry per point); the engine never sees workspace objects -- the
    `DerivedIntensityInput` precedent. Parallel tuples, one entry per
    recorded point of this family.
    """
    label: str
    role: str                              # '' | 'fast' | 'slow'
    angles_deg: Tuple[float, ...]
    amplitude_fractions: Tuple[float, ...]
    gamma_s_inv: Tuple[float, ...]
    item_ids: Tuple[str, ...]
    low_amplitude: Tuple[bool, ...]
    bounds_sources: Tuple[str, ...] = ()   # 'auto' | 'manual' | '' per point
    # Per point: did this mode finish decaying inside its lag window? Optional with the
    # same `bounds_sources` shape (checked only when supplied) so an existing caller is
    # not broken by its arrival. It reaches the same kind of disclosure `low_amplitude`
    # does: an amplitude read off a decay the measurement never saw is imprecise for the
    # same reason a sub-floor one is, and the partitioned intensity inherits it.
    decay_incomplete: Tuple[bool, ...] = ()


@dataclass
class PartitionedMode:
    """One mode's share of the static series: an ordinary Rayleigh result."""
    label: str
    role: str
    rayleigh: RayleighRatioResult          # dR_i = A_i * dR (NaN where unmatched)


@dataclass
class ModePartitionResult:
    """The Sedlak partition of one concentration's excess Rayleigh ratio.

    `closure_fraction` is the per-angle sum of the picked modes' amplitudes
    (NaN where no mode matched, or where the matched modes come from
    different runs -- a closure is only meaningful within ONE decomposition).
    `well_separated` is a LABEL against the documented house threshold; it
    gates a note, never a refusal (doctrine rule 2).
    """
    angles_deg: np.ndarray                 # the SLS series' angles, ascending
    excess_total_cm_inv: np.ndarray        # the unpartitioned dR (what sums back)
    closure_fraction: np.ndarray           # sum of matched A_i per angle (NaN-capable)
    per_mode: Tuple[PartitionedMode, ...]
    n_matched_angles: int                  # angles with at least one mode amplitude
    n_unmatched_angles: int
    closure_tol: float
    separation_ratio_min: float
    min_separation_ratio: float            # min adjacent-Gamma ratio over shared angles
    well_separated: bool                   # LABEL -- gates a note, never a refusal
    notes: tuple = ()


@dataclass
class AmplitudeGuinierResult:
    """Sedlak Eq. 7 on one mode's partitioned series: apparent mode size.

    Fitted as 1/A vs q^2 (an exact rearrangement of A(0)/A(q) = 1 +
    Rg^2 q^2 / 3): the intercept is 1/A(0) -- the extrapolated normalization
    obtained from the fit itself, never from a hand-picked lowest angle --
    and Rg^2 = 3 * slope / intercept, where every intensity SCALE cancels
    (calibration and dn/dc do not enter). What does NOT cancel is an ADDITIVE
    term: a solvent blank that was not measured leaves a residue in the parent
    dR which survives the partition and biases the slope, whether or not that
    residue varies with angle -- hence `blank_measured` below, the one input
    absence this Rg does not shrug off. APPARENT + MODE-RESOLVED by
    construction (invariant 7; Sedlak prints the 'apparent' caveat himself,
    2001 p. 42): the number tracks relative changes; its absolute value is
    resolution-limited.

    `rg_resolution_floor_nm` is recomputed from the FITTED q-window (never
    copied from Sedlak's optics): below it the model's total predicted
    relative change of 1/A across the window is under `min_relative_change`
    -- the order of the per-point scatter ill-posed amplitudes carry -- so
    only relative changes are meaningful. A documented, overridable
    parameter (invariant 4); `below_resolution_floor` gates a note, the
    value still reports.
    """
    q2_nm2: np.ndarray                     # the fitted abscissa (kept points)
    inv_amplitude: np.ndarray              # the fitted ordinate 1/A
    slope: float
    intercept: float
    rg_app_nm: float                       # apparent, mode-resolved (NaN-capable)
    amplitude_q0: float                    # A(0) = 1/intercept (Eq. 7's normalization)
    r_squared: float
    n_angles: int
    qrg_max: float                         # sqrt(q2_max) * Rg (NaN-capable)
    guinier_valid: bool                    # small-qRg expansion validity (label)
    qrg_max_valid: float                   # the echoed threshold
    rg_resolution_floor_nm: float
    below_resolution_floor: bool           # LABEL, gates a note
    min_relative_change: float             # the floor's delta, echoed
    is_apparent: bool                      # always True (structural)
    mode_resolved: bool                    # always True (structural)
    rg_se: Optional[float] = None
    se_estimator: str = 'hc3'
    well_conditioned: bool = True
    cond: float = float('nan')
    # The 1/coefficient family flag (rule 10): Rg^2 = 3*slope/intercept divides
    # by a fitted coefficient, so a degenerate design (or no extractable size)
    # drops this flag with the reason in reliability_note -- the value still
    # reports (flag, don't NaN).
    rg_reliable: bool = True
    # Was the parent series' solvent blank MEASURED at every angle? This fit takes bare
    # arrays and no RayleighRatioResult, so the caller fills it from the Rayleigh
    # results the amplitudes came from. It is the one input absence this Rg does not
    # shrug off (the docstring says why), so it gates `rg_reliable`. Normative home:
    # sls.RayleighRatioResult.blank_measured.
    blank_measured: bool = True
    reliability_note: str = ''
    derived_provenance: str = ''           # partition record (export family)
    notes: tuple = ()


#: The precondition sentence for a mode-resolved size fitted from a series whose solvent
#: blank was never measured. A literal here rather than a copied record: this result carries
#: no `blank_provenance` field to copy one into (see `amplitude_guinier`).
_NO_BLANK_NOTE = ("solvent blank not measured for the parent series: the partitioned "
                  "amplitudes inherit whatever the assumed blank left in dR, so this "
                  "apparent size is not reliable")


def _join_provenance(partition: str, existing: str) -> str:
    """Two independent facts about a number's origin never shadow each other:
    a partitioned point can ALSO be trace-derived (the `_join_comments`
    principle, applied at the source)."""
    parts = [p for p in (existing.strip(), partition.strip()) if p]
    return '; '.join(parts)


def partition_rayleigh(
    rayleigh: RayleighRatioResult,
    modes: Sequence[PartitionModeInput],
    *,
    provenance: str = '',
    closure_tol: float = DEFAULT_CLOSURE_TOL,
    separation_ratio_min: float = DEFAULT_SEPARATION_RATIO_MIN,
) -> ModePartitionResult:
    """Partition one concentration's excess Rayleigh ratio by mode amplitudes.

    dR_i(theta) = A_i(theta) * dR(theta) per mode, with the Zimm ordinate
    recomputed as K c / dR_i. Angles are matched through the shared identity
    form (`round_angle_for_identity`), never raw float equality. Per mode, an
    SLS angle with no recorded amplitude is NaN in that mode's series
    (excluded from downstream fits by their own finite filters -- an invented
    amplitude would be fabricated data, so nothing is interpolated), and an
    angle where the family holds MORE than one recorded point is skipped with
    a note naming it (no invented averaging -- the `amplitude_vs_q`
    precedent).

    Closure: the fractions are used AS MEASURED, never renormalized --
    renormalizing a subset would silently reassign an excluded mode's
    intensity to the picked ones (a fabricated number, where the deficit is
    an honest disclosed one). When the picked set is exhaustive the closure
    holds exactly: sum_i dR_i = dR at every matched angle.

    Raises ValueError only on empty inputs or mismatched tuple lengths
    (caller bugs); every data condition is a note.
    """
    if not modes:
        raise ValueError('partition_rayleigh needs at least one mode family.')
    for m in modes:
        n = len(m.angles_deg)
        if (len(m.amplitude_fractions) != n or len(m.gamma_s_inv) != n
                or len(m.item_ids) != n or len(m.low_amplitude) != n
                or (m.bounds_sources and len(m.bounds_sources) != n)
                or (m.decay_incomplete and len(m.decay_incomplete) != n)):
            raise ValueError(
                f"Mode family {m.label!r}: parallel tuples disagree in length."
            )

    notes: List[str] = []
    angles = np.asarray(rayleigh.angles_deg, dtype=float)
    dR = np.asarray(rayleigh.excess_rayleigh_cm_inv, dtype=float)
    n_angles = angles.size

    # Per mode: rounded angle -> (amplitude, gamma, item_id, low, undecayed),
    # single-point buckets only (multiplicity skipped with a note, never averaged).
    per_mode_maps = []
    for m in modes:
        buckets: dict = {}
        for j, a in enumerate(m.angles_deg):
            buckets.setdefault(round_angle_for_identity(a), []).append(j)
        multi = sorted(a for a, js in buckets.items() if len(js) > 1)
        if multi:
            angles_txt = ', '.join(f'{a:g}' for a in multi)
            notes.append(
                f"Skipped angle(s) {angles_txt} deg for mode {m.label!r}: more "
                f"than one recorded point there (no averaging is invented)."
            )
        per_mode_maps.append({
            a: (float(m.amplitude_fractions[js[0]]),
                float(m.gamma_s_inv[js[0]]), m.item_ids[js[0]],
                bool(m.low_amplitude[js[0]]),
                bool(m.decay_incomplete[js[0]]) if m.decay_incomplete else False)
            for a, js in buckets.items() if len(js) == 1})

    closure = np.full(n_angles, float('nan'))
    per_mode_dR = [np.full(n_angles, float('nan')) for _ in modes]
    sep_ratios: List[float] = []
    n_mixed_run = 0
    n_low = 0
    n_undecayed = 0
    for k in range(n_angles):
        key = round_angle_for_identity(angles[k])
        found = [(i, per_mode_maps[i][key]) for i in range(len(modes))
                 if key in per_mode_maps[i]]
        for i, (amp, _g, _iid, low, undecayed) in found:
            per_mode_dR[i][k] = amp * dR[k]
            if low:
                n_low += 1          # counts points a partition was BUILT on
            if undecayed:
                n_undecayed += 1    # same counting rule, separate judgement
        if not found:
            continue
        ids = {iid for _i, (_a, _g, iid, _low, _u) in found if iid}
        if len(ids) > 1:
            n_mixed_run += 1        # amplitudes valid per se; the SUM is not
        else:
            closure[k] = float(sum(amp for _i, (amp, _g, _iid, _low, _u)
                                   in found))
            # The separation ratio, like the closure, is only defined WITHIN
            # one decomposition -- decay rates from different runs never
            # coexisted in one spectrum, so they feed no verdict.
            gammas = sorted(g for _i, (_a, g, _iid, _low, _u) in found
                            if math.isfinite(g) and g > 0)
            for g_lo, g_hi in zip(gammas, gammas[1:], strict=False):
                if g_lo > 0:
                    sep_ratios.append(g_hi / g_lo)
    if n_mixed_run:
        notes.append(
            f"Closure withheld at {n_mixed_run} angle(s): the matched modes "
            f"there come from different runs, and a closure is only defined "
            f"within one decomposition."
        )
    # Low-amplitude disclosure -- counted over the points the partition was
    # actually built on (matched, single-bucket), never over every recorded
    # point (the note claimed
    # imprecision for intensities that were never built).
    if n_low:
        notes.append(
            f"{n_low} of the mode point(s) used sit below their amplitude "
            f"floor; the partitioned intensities built on them are imprecise."
        )
    if n_undecayed:
        notes.append(
            f"{n_undecayed} of the mode point(s) used never finished decaying "
            f"inside their delay window; their amplitudes come from a decay the "
            f"measurement did not observe, so the partitioned intensities "
            f"built on them are unreliable."
        )

    matched = [k for k in range(n_angles)
               if any(math.isfinite(per_mode_dR[i][k]) for i in range(len(modes)))]
    n_matched = len(matched)
    n_unmatched = n_angles - n_matched
    for i, m in enumerate(modes):
        n_missing = int(np.sum(~np.isfinite(per_mode_dR[i])))
        if n_missing:
            notes.append(
                f"Mode {m.label!r}: {n_missing} of {n_angles} angle(s) have no "
                f"recorded amplitude and were excluded from its series."
            )

    # Closure disclosure -- deficit (modes not in the set) and excess
    # (double-picked weight) are different stories, so two wordings.
    fin = np.isfinite(closure)
    if np.any(fin):
        deficit = float(np.nanmax(1.0 - closure[fin])) if np.any(
            closure[fin] < 1.0 - closure_tol) else 0.0
        excess = float(np.nanmax(closure[fin] - 1.0)) if np.any(
            closure[fin] > 1.0 + closure_tol) else 0.0
        if deficit > 0.0:
            notes.append(
                f"Up to {deficit:.0%} of the excess intensity at some angles "
                f"is carried by modes not in this recorded set (the fractions "
                f"are used as measured, never renormalized)."
            )
        if excess > 0.0:
            notes.append(
                f"The picked modes' fractions sum to more than 1 by up to "
                f"{excess:.0%} at some angles -- the same intensity may be "
                f"counted in two picked modes."
            )

    # Separation label: min adjacent-Gamma ratio over shared angles, against
    # the documented house threshold. Vacuously well-separated with a single
    # matched mode everywhere (nothing to separate). Manual bounds force the
    # note regardless (the standing rail: a manual split exists because the
    # peaks overlap).
    min_sep = float(min(sep_ratios)) if sep_ratios else float('nan')
    well_sep = bool(math.isnan(min_sep) or min_sep >= separation_ratio_min)
    if not well_sep:
        notes.append(
            f"The picked modes' decay rates come within a factor "
            f"{min_sep:.1f} of each other at some angle (threshold "
            f"{separation_ratio_min:g}); the mode decomposition -- and this "
            f"partition with it -- assumes well-separated modes."
        )
    if any('manual' in (m.bounds_sources or ()) for m in modes):
        notes.append(
            "Manual integration bounds contributed to these amplitudes; "
            "overlapping peaks violate the wide-separation validity condition "
            "of the partition."
        )

    # The per-mode Rayleigh clones: same class every SLS fitter consumes.
    k_const = rayleigh.optical_constant_K
    c = rayleigh.concentration_g_per_mL
    per_mode = []
    for i, m in enumerate(modes):
        dR_i = per_mode_dR[i]
        with np.errstate(divide='ignore', invalid='ignore'):
            kc_i = k_const * c / dR_i
        kc_i = np.where(np.isfinite(kc_i), kc_i, float('nan'))
        mode_prov = _join_provenance(provenance, rayleigh.derived_provenance)
        per_mode.append(PartitionedMode(
            label=m.label, role=m.role,
            rayleigh=replace(
                rayleigh,
                excess_rayleigh_cm_inv=dR_i,
                kc_over_dR_mol_per_g=kc_i,
                derived_provenance=mode_prov,
            )))

    return ModePartitionResult(
        angles_deg=angles,
        excess_total_cm_inv=dR,
        closure_fraction=closure,
        per_mode=tuple(per_mode),
        n_matched_angles=n_matched,
        n_unmatched_angles=n_unmatched,
        closure_tol=closure_tol,
        separation_ratio_min=separation_ratio_min,
        min_separation_ratio=min_sep,
        well_separated=well_sep,
        notes=tuple(notes),
    )


def amplitude_guinier(
    q2_nm2,
    amplitudes,
    *,
    source_item_ids: Optional[Sequence[str]] = None,
    estimator: str = 'hc3',
    min_relative_change: float = 0.05,
    qrg_max_valid: float = 1.3,
    cond_limit: float = unc.COND_LIMIT,
    derived_provenance: str = '',
    blank_measured: bool = True,
) -> AmplitudeGuinierResult:
    """Sedlak Eq. 7 on one mode's amplitude series: apparent mode size.

    `amplitudes` is the mode's partitioned excess series dR_i(theta) (any
    common scale -- the scale cancels in slope/intercept). Non-positive or
    non-finite amplitudes are excluded with a note (1/A needs A > 0; a
    negative partitioned point means the parent dR itself went negative
    there).

    Raises ValueError with fewer than two kept points at distinct q^2 (the
    `debye_analysis` contract -- a fit that cannot be formed is a caller
    condition, not a degradable result).
    """
    q2 = np.asarray(q2_nm2, dtype=float)
    amp = np.asarray(amplitudes, dtype=float)
    if q2.shape != amp.shape:
        raise ValueError('q2_nm2 and amplitudes disagree in length.')
    ids = (list(source_item_ids) if source_item_ids is not None
           else [''] * q2.size)
    if len(ids) != q2.size:
        raise ValueError('source_item_ids disagrees in length with the data.')

    notes: List[str] = []
    good = np.isfinite(q2) & np.isfinite(amp) & (amp > 0)
    n_dropped = int(np.sum(~good))
    if n_dropped:
        notes.append(
            f"{n_dropped} point(s) with a non-positive or non-finite "
            f"amplitude were excluded (1/A needs A > 0)."
        )
    q2k = q2[good]
    ampk = amp[good]
    ids_kept = [i for i, g in zip(ids, good, strict=True) if g]
    if np.unique(q2k).size < 2:
        raise ValueError(
            'The amplitude-Guinier fit needs at least two usable points at '
            'distinct q^2.'
        )

    y = 1.0 / ampk
    fit = unc.linear_fit(q2k, y, estimator, cond_limit)  # cov [intercept, slope]
    slope, intercept = float(fit.slope), float(fit.intercept)

    # Rg^2 = 3 * slope / intercept, guarded exactly as the Debye sibling: the
    # intercept 1/A(0) is physically positive, and a non-positive ratio means
    # the amplitude does not decrease with q -- no size, disclosed.
    ratio = (slope / intercept
             if (math.isfinite(intercept) and intercept != 0) else float('nan'))
    rg = (math.sqrt(3.0 * ratio)
          if (math.isfinite(ratio) and ratio > 0
              and math.isfinite(intercept) and intercept > 0) else float('nan'))
    amp_q0 = (1.0 / intercept
              if (math.isfinite(intercept) and intercept > 0) else float('nan'))
    no_size_note = ''
    if not math.isfinite(rg):
        no_size_note = (
            "No apparent size could be extracted: the mode's amplitude does "
            "not decrease with q (non-positive slope/intercept ratio)."
        )
        notes.append(no_size_note)

    # Regression SE (statistical only), delta-method through the same
    # slope/intercept jacobian as the Debye Rg -- but Rg = sqrt(3*slope/
    # intercept) is NONLINEAR, so it does not inherit the Debye MC validation:
    # its own gate (test_amplitude_guinier_se_conservative_montecarlo) runs in
    # the amplitudes' regime and measures it conservative-safe. Withheld when
    # two fitted points share a source run: the spread is then partly WITHIN
    # one correlogram's correlated lag channels, so an independent-points SE
    # would under-report (the fit_mode_family stance on this same class).
    rg_se = None
    nonempty = [i for i in ids_kept if i]
    same_run = len(nonempty) != len(set(nonempty))
    if (math.isfinite(fit.intercept_se) and math.isfinite(rg) and rg > 0
            and slope != 0 and intercept != 0):
        jac = [-rg / (2.0 * intercept), rg / (2.0 * slope)]
        rg_se = unc.se_or_none(unc.propagate(jac, fit.cov))
    if same_run:
        rg_se = None
        notes.append(
            "Two or more fitted points come from the same run; the "
            "regression SE is withheld (part of the spread is within one "
            "correlogram, whose lag channels are correlated)."
        )
    elif rg_se is not None:
        notes.append(
            "The Rg SE is the regression SE across independent angles -- "
            "statistical only, excluding the inversion's own systematics "
            "(regularization, integration bounds)."
        )

    # Resolution floor from the FITTED q-window (invariant 4: documented,
    # overridable; never Sedlak's copied 25 nm). Below it the model's total
    # predicted relative change of 1/A across the window,
    # (Rg^2 / 3) * (q2_max - q2_min), is under `min_relative_change`.
    span = float(q2k.max() - q2k.min())     # > 0: >= 2 distinct q^2 enforced
    floor = math.sqrt(3.0 * min_relative_change / span)
    below = bool(math.isfinite(rg) and rg < floor)
    if below:
        notes.append(
            f"The apparent Rg is below the ~{floor:.0f} nm resolution floor "
            f"of this q-range -- treat it as relative-changes-only."
        )

    qrg_max = (math.sqrt(float(q2k.max())) * rg
               if math.isfinite(rg) else float('nan'))
    valid = bool(math.isfinite(qrg_max) and qrg_max <= qrg_max_valid)
    if math.isfinite(qrg_max) and not valid:
        notes.append(
            f"q*Rg reaches {qrg_max:.2f} (validity limit {qrg_max_valid:g}); "
            f"the small-qRg expansion behind Eq. 7 is strained at the high "
            f"angles."
        )

    # This fit's own words, not the Zimm/Debye sibling's: it has one concentration
    # and yields only an apparent Rg, so the note names that quantity and the one
    # remedy the method has (a wider angular spread), never Mw/A2 or more
    # concentrations.
    note = ('' if fit.well_conditioned else
            f"ill-conditioned fit design (cond={fit.cond:.1e}); the apparent Rg is "
            f"numerically unreliable -- widen the angular spread")
    # A size the amplitudes cannot yield is a failed precondition too, so the flag's
    # own note names it rather than leaving `rg_reliable=False` unexplained.
    if no_size_note:
        note = ' '.join(x for x in (note, no_size_note) if x)
    # The strained-regime verdict is a failed precondition of the METHOD, exactly as on
    # `sls.guinier_analysis` (which cites this class as its pattern source): it drops
    # `rg_reliable` and names itself, rather than riding only in `notes`.
    if math.isfinite(qrg_max) and not valid:
        note = ' '.join(x for x in (note, notes[-1]) if x)
    # The blank names itself HERE, not in a `blank_provenance` record, because this result
    # has none: it takes bare arrays and no RayleighRatioResult, so there is no sentence to
    # copy. Without this the flag went down with nothing to explain it, and the
    # mode-partition exporter -- which composes its Comments from `reliability_note` --
    # flagged a size and said why on no surface at all (round d0ce f1). Invariant 4
    # clause 1: a False flag names its failed precondition.
    if not blank_measured:
        note = ' '.join(x for x in (note, _NO_BLANK_NOTE) if x)

    return AmplitudeGuinierResult(
        q2_nm2=q2k, inv_amplitude=y,
        slope=slope, intercept=intercept,
        rg_app_nm=float(rg), amplitude_q0=float(amp_q0),
        r_squared=float(fit.r_squared), n_angles=int(q2k.size),
        qrg_max=float(qrg_max), guinier_valid=valid,
        qrg_max_valid=qrg_max_valid,
        rg_resolution_floor_nm=float(floor),
        below_resolution_floor=below,
        min_relative_change=min_relative_change,
        is_apparent=True, mode_resolved=True,
        rg_se=rg_se, se_estimator=estimator,
        well_conditioned=fit.well_conditioned, cond=fit.cond,
        rg_reliable=bool(math.isfinite(rg) and rg > 0 and fit.well_conditioned
                         and valid and blank_measured),
        blank_measured=bool(blank_measured),
        reliability_note=note,
        derived_provenance=derived_provenance,
        notes=tuple(notes),
    )
