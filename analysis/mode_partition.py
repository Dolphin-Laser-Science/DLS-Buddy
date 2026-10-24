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

The per-mode preconditions are counted per mode and per angle. The amplitude
floor (cited at `analysis.dls.modes.extract_modes`) and the lag window's (a mode
that never finished decaying) are each judged on the points one mode's share was
built on. A fraction is a mode's share of ONE decomposition at its angle, so a
mode that did not decay there sits, extrapolated, in every fraction's
denominator: every mode's point at that angle fails, the undecayed mode's as
undecayed and its decayed neighbours' as sharing the decomposition. A mode whose
lobe sits on the Rh grid's end fails its angle by the same rule
(`analysis.sls.ModePointCounts` is its home). Each per-mode
clone carries its counts (`ModePointCounts`) for the fits that read them; which fits
fold them is pinned by
`tests/test_mode_partition.py::test_the_planted_truncation_fails_every_mode_at_its_angles`.

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
from core.notes import Note
from analysis import uncertainty as unc
from analysis.sls import (QRG_MAX_VALID_DEFAULT, ModePointCounts, RayleighRatioResult,
                          mode_point_notes)

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
    # Per point: is this mode's lobe maximum on the Rh grid's first or last node? The same
    # optional shape; the rule it feeds is `analysis.sls.ModePointCounts`'.
    grid_end: Tuple[bool, ...] = ()


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
    gates a note, never a refusal (doctrine rule 2). `notes` holds every note of
    the partition as `core.notes.Note`.
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
    notes: Tuple[Note, ...] = ()


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
    parameter (invariant 4); `below_resolution_floor` is read as a qualifier
    (`core.notes.qualifiers`), the value still reports.

    `notes` holds every note of the fit as `core.notes.Note`, its preconditions the
    unmeasured blank's included -- this result has no `blank_provenance` for an exporter
    to name the blank from.
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
    # drops this flag with the reason in a precondition note -- the value still
    # reports (flag, don't NaN).
    rg_reliable: bool = True
    # Was the parent series' solvent blank MEASURED at every angle? This fit takes bare
    # arrays and no RayleighRatioResult, so the caller fills it from the Rayleigh
    # results the amplitudes came from. It is the one input absence this Rg does not
    # shrug off (the docstring says why), so it gates `rg_reliable`. Normative home:
    # sls.RayleighRatioResult.blank_measured.
    blank_measured: bool = True
    derived_provenance: str = ''           # partition record (export family)
    notes: Tuple[Note, ...] = ()


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

    The per-mode preconditions are counted per mode over the points its share was
    built on, under the per-angle rule (the module docstring): each mode's undecayed,
    grid-end and sub-floor points are notes here, and each clone carries the counts
    (`RayleighRatioResult.mode_counts`), the angles where an undecayed or grid-end
    neighbour shared its decomposition among them, for the fits that read them.

    Inputs not prescribed by the source: `separation_ratio_min` (default 5.0),
    which gates the `well_separated` verdict and its note, and `closure_tol`
    (default 0.05), which gates a record.
    Sedlak states the separation requirement only qualitatively (2001, pp. 7-8:
    peaks "should not overlap completely"), and the held resolution theory of
    the inversion gives a limit that depends on the signal-to-noise ratio, not
    a partition threshold.

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
                or (m.decay_incomplete and len(m.decay_incomplete) != n)
                or (m.grid_end and len(m.grid_end) != n)):
            raise ValueError(
                f"Mode family {m.label!r}: parallel tuples disagree in length."
            )

    notes: List[Note] = []
    angles = np.asarray(rayleigh.angles_deg, dtype=float)
    dR = np.asarray(rayleigh.excess_rayleigh_cm_inv, dtype=float)
    n_angles = angles.size

    # Per mode: rounded angle -> (amplitude, gamma, item_id, low, undecayed, at the grid
    # end), single-point buckets only (multiplicity skipped with a note, never averaged).
    per_mode_maps = []
    for m in modes:
        buckets: dict = {}
        for j, a in enumerate(m.angles_deg):
            buckets.setdefault(round_angle_for_identity(a), []).append(j)
        multi = sorted(a for a, js in buckets.items() if len(js) > 1)
        if multi:
            # More than one recorded point at an angle: skipped, no averaging invented.
            notes.append(Note('multiple_points_at_angle', (),
                              {'family': m.label, 'angles': tuple(float(a) for a in multi)}))
        per_mode_maps.append({
            a: (float(m.amplitude_fractions[js[0]]),
                float(m.gamma_s_inv[js[0]]), m.item_ids[js[0]],
                bool(m.low_amplitude[js[0]]),
                bool(m.decay_incomplete[js[0]]) if m.decay_incomplete else False,
                bool(m.grid_end[js[0]]) if m.grid_end else False)
            for a, js in buckets.items() if len(js) == 1})

    closure = np.full(n_angles, float('nan'))
    per_mode_dR = [np.full(n_angles, float('nan')) for _ in modes]
    sep_ratios: List[float] = []
    mixed_run_angles: List[float] = []
    # Per mode, over the points its share was built on: [built, undecayed, low, spoiled,
    # at the grid end, sharing a decomposition with a mode at the grid end].
    counts = [[0, 0, 0, 0, 0, 0] for _ in modes]
    for k in range(n_angles):
        key = round_angle_for_identity(angles[k])
        found = [(i, per_mode_maps[i][key]) for i in range(len(modes))
                 if key in per_mode_maps[i]]
        ids = {iid for _i, (_a, _g, iid, _low, _u, _e) in found if iid}
        # One decomposition per run at this angle (the closure's own condition): an
        # undecayed mode reaches every fraction's denominator in ITS run, a decayed
        # neighbour's included, and no other run's. With one run at the angle, every mode
        # shares it; with several, a mode shares only the modes of its own named run. A
        # mode at the grid's end spoils its neighbours by the same rule.
        undecayed_runs = [iid for _i, (_a, _g, iid, _low, u, _e) in found if u]
        grid_end_runs = [iid for _i, (_a, _g, iid, _low, _u, e) in found if e]
        for i, (amp, _g, iid_i, low, undecayed, at_end) in found:
            per_mode_dR[i][k] = amp * dR[k]
            if not math.isfinite(per_mode_dR[i][k]):
                continue            # no intensity built here (a masked angle)
            shared_undecayed = any(len(ids) <= 1 or (iid_u and iid_u == iid_i)
                                   for iid_u in undecayed_runs)
            shared_grid_end = any(len(ids) <= 1 or (iid_e and iid_e == iid_i)
                                  for iid_e in grid_end_runs)
            counts[i][0] += 1
            counts[i][1] += bool(undecayed)
            counts[i][2] += bool(low)
            counts[i][3] += bool(shared_undecayed and not undecayed)
            counts[i][4] += bool(at_end)
            counts[i][5] += bool(shared_grid_end and not at_end)
        if not found:
            continue
        if len(ids) > 1:
            mixed_run_angles.append(float(angles[k]))   # amplitudes valid; the SUM is not
        else:
            closure[k] = float(sum(amp for _i, (amp, _g, _iid, _low, _u, _e)
                                   in found))
            # The separation ratio, like the closure, is only defined WITHIN
            # one decomposition -- decay rates from different runs never
            # coexisted in one spectrum, so they feed no verdict.
            gammas = sorted(g for _i, (_a, g, _iid, _low, _u, _e) in found
                            if math.isfinite(g) and g > 0)
            for g_lo, g_hi in zip(gammas, gammas[1:], strict=False):
                if g_lo > 0:
                    sep_ratios.append(g_hi / g_lo)
    if mixed_run_angles:
        # Closure withheld there: a closure is only defined within one decomposition.
        notes.append(Note('mixed_runs_at_angle', ('closure_fraction',),
                          {'angles': tuple(mixed_run_angles)}))
    # Each mode's counts, over the points its share was actually built on (matched,
    # single-bucket, a finite intensity), never over every recorded point: its own
    # undecayed, grid-end and sub-floor points named here; the neighbours such a mode
    # spoiled are named by the fits that read the clone's counts (`sls.mode_point_notes`).
    mode_counts = [ModePointCounts(m.label, *c) for m, c in zip(modes, counts, strict=True)]
    for mc in mode_counts:
        notes.extend(n for n in mode_point_notes(mc, about=('per_mode',))
                     if n.code not in ('partition_angle_spoiled', 'partition_angle_grid_end'))

    matched = [k for k in range(n_angles)
               if any(math.isfinite(per_mode_dR[i][k]) for i in range(len(modes)))]
    n_matched = len(matched)
    n_unmatched = n_angles - n_matched
    for i, m in enumerate(modes):
        n_missing = int(np.sum(~np.isfinite(per_mode_dR[i])))
        if n_missing:
            notes.append(Note('mode_amplitude_missing', ('per_mode',),
                              {'label': m.label, 'n': n_missing, 'total': n_angles}))

    # Closure disclosure -- deficit (modes not in the set) and excess
    # (double-picked weight) are different stories, so two wordings.
    fin = np.isfinite(closure)
    if np.any(fin):
        deficit = float(np.nanmax(1.0 - closure[fin])) if np.any(
            closure[fin] < 1.0 - closure_tol) else 0.0
        excess = float(np.nanmax(closure[fin] - 1.0)) if np.any(
            closure[fin] > 1.0 + closure_tol) else 0.0
        # Two facts, not one: modes missing from the set, and weight picked twice. The
        # fractions are used as measured, never renormalized.
        if deficit > 0.0:
            notes.append(Note('closure_deficit', ('closure_fraction',), {'deficit': deficit}))
        if excess > 0.0:
            notes.append(Note('closure_excess', ('closure_fraction',),
                              {'limit': 1.0, 'excess': excess}))

    # Separation label: min adjacent-Gamma ratio over shared angles, against
    # the documented house threshold. Vacuously well-separated with a single
    # matched mode everywhere (nothing to separate). Manual bounds force the
    # note regardless (the standing rail: a manual split exists because the
    # peaks overlap).
    min_sep = float(min(sep_ratios)) if sep_ratios else float('nan')
    well_sep = bool(math.isnan(min_sep) or min_sep >= separation_ratio_min)
    if not well_sep:
        notes.append(Note('modes_not_separated', (),
                          {'ratio': min_sep, 'limit': float(separation_ratio_min)}))
    if any('manual' in (m.bounds_sources or ()) for m in modes):
        notes.append(Note('manual_bounds'))

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
                mode_resolved=True,
                mode_counts=mode_counts[i],
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
    qrg_max_valid: float = QRG_MAX_VALID_DEFAULT,
    cond_limit: float = unc.COND_LIMIT,
    derived_provenance: str = '',
    blank_measured: bool = True,
    mode_counts: Optional[ModePointCounts] = None,
) -> AmplitudeGuinierResult:
    """Sedlak Eq. 7 on one mode's amplitude series: apparent mode size.

    `amplitudes` is the mode's partitioned excess series dR_i(theta) (any
    common scale -- the scale cancels in slope/intercept). Non-positive or
    non-finite amplitudes are excluded with a note (1/A needs A > 0; a
    negative partitioned point means the parent dR itself went negative
    there).

    `mode_counts` is the per-mode clone's counts (`RayleighRatioResult.mode_counts`),
    filled by the caller as `blank_measured` is, since this fit takes bare arrays: an
    undecayed, grid-end, sub-floor or spoiled point names itself and fails `rg_reliable`.

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

    notes: List[Note] = []
    good = np.isfinite(q2) & np.isfinite(amp) & (amp > 0)
    n_dropped = int(np.sum(~good))
    if n_dropped:
        # 1/A needs A > 0.
        notes.append(Note('nonpositive_points_dropped', ('rg_app_nm',),
                          {'n': n_dropped, 'ordinate': 'or non-finite amplitude'}))
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
        notes.append(Note('se_withheld_same_run', ('rg_se',)))
    # A reported Rg SE excludes the inversion's own systematics: a qualifier, read off
    # `mode_resolved` and `rg_se` (core.notes.qualifiers), never a note.

    # Resolution floor from the FITTED q-window (invariant 4: documented,
    # overridable; never Sedlak's copied 25 nm). Below it the model's total
    # predicted relative change of 1/A across the window,
    # (Rg^2 / 3) * (q2_max - q2_min), is under `min_relative_change`.
    span = float(q2k.max() - q2k.min())     # > 0: >= 2 distinct q^2 enforced
    floor = math.sqrt(3.0 * min_relative_change / span)
    # Below the floor only relative changes are meaningful: a qualifier, read off
    # `below_resolution_floor` (core.notes.qualifiers), never a note.
    below = bool(math.isfinite(rg) and rg < floor)

    qrg_max = (math.sqrt(float(q2k.max())) * rg
               if math.isfinite(rg) else float('nan'))
    valid = bool(math.isfinite(qrg_max) and qrg_max <= qrg_max_valid)

    # The size's failed preconditions, each named (invariant 4 clause 1). The design's
    # conditioning: this fit has one concentration and yields only an apparent Rg, so the
    # note is about that quantity alone, never Mw/A2.
    if not fit.well_conditioned:
        notes.append(Note('ill_conditioned', ('rg_app_nm',),
                          {'cond': float(fit.cond), 'limit': float(cond_limit)}))
    # A size the amplitudes cannot yield: the amplitude does not decrease with q (the
    # slope/intercept ratio is not positive), or the intercept 1/A(0) is not.
    if not math.isfinite(rg):
        if math.isfinite(intercept) and intercept > 0:
            values = {'quantity': 'slope/intercept', 'value': float(ratio), 'unit': 'nm²'}
        else:
            values = {'quantity': '1/A(0)', 'value': float(intercept),
                      'unit': '(1/amplitude)'}
        notes.append(Note('no_rg', ('rg_app_nm',), {**values, 'required': 'positive'}))
    # The strained-regime verdict is a failed precondition of the METHOD, exactly as on
    # `sls.guinier_analysis` (which cites this class as its pattern source): it drops
    # `rg_reliable` and names itself.
    if math.isfinite(qrg_max) and not valid:
        notes.append(Note('qrg_above_limit', ('rg_app_nm',),
                          {'qrg': float(qrg_max), 'limit': float(qrg_max_valid),
                           'regime': 'amplitude-Guinier'}))
    # The blank names itself HERE, not through a `blank_provenance` record, because this
    # result has none: it takes bare arrays and no RayleighRatioResult. Without it the flag
    # went down with nothing to explain it, and the mode-partition exporter -- which
    # composes its Comments from the fit's notes -- flagged a size and said why on no
    # surface at all (round d0ce f1). Invariant 4 clause 1.
    if not blank_measured:
        notes.append(Note('blank_unmeasured', ('rg_app_nm',)))
    partition = mode_point_notes(mode_counts, about=('rg_app_nm',)) if mode_counts else ()
    notes += partition

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
                         and valid and blank_measured and not partition),
        blank_measured=bool(blank_measured),
        derived_provenance=derived_provenance,
        notes=tuple(notes),
    )
