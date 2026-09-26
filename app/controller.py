"""
app/controller.py
=================

The controller: the framework-agnostic layer between the (future) GUI and the
analysis engine. It owns the `Workspace`, mediates parameter edits with
commit/working semantics, orchestrates analysis runs, and reads/writes session
files. It imports the engine (that is its job); it imports NOTHING from any GUI
toolkit, so the same controller works under PySide6, under a script, or in tests.

Why this layer exists
---------------------
The architectural invariant is that no analysis logic lives in the GUI. The
controller is how that is enforced: widgets call controller methods like
`run_zimm(sample_id)` and display what comes back; they never call the engine
directly. Keeping the controller pure-Python also means the GUI framework can be
swapped later by rewriting only the widget shell.

Commit / working model
----------------------
Each measurement holds a working parameter set (what the user is editing) and a
committed set (what the last analysis used). The controller exposes:
  - set_param / set_calibration_field : edit the working set
  - is_dirty / dirty_keys             : has anything changed since the last commit?
  - commit                            : adopt working -> committed, then regroup
                                        (also pushes an undo restore point)
  - undo                              : step back to the previously applied state
                                        (or first discard un-applied edits)
  - undo_to_committed                 : discard edits (working <- committed)
  - reset_working_params              : blank entered params (kept in working)
Analysis always runs on committed parameters, so a half-edited table never
silently changes a result. The GUI highlights dirty fields and shows a pending-
update indicator from these queries; the analysis itself only runs on commit.

Threading
---------
Enabled (2026-07): the GUI dispatches every heavy `run_*` call to a single worker
thread via `gui/worker.py`, so a CONTIN sweep / DDLS Monte-Carlo / replicate
average no longer freezes the window. The controller itself stays thread-agnostic
and Qt-free — each expensive call is one method with no widget interaction — and
the GUI enforces one analysis in flight at a time (it never reads the mutated
result state until the worker hands it back on the main thread).

Change history
--------------
2026-06-13  controller.py v1: workspace management, commit/working parameter
            state, calibration state, DLS + SLS analysis orchestration, JSON
            session save/load (self-contained + reload-from-source).
"""

from __future__ import annotations

import inspect
import json
import math
import os
import tempfile
import warnings
from datetime import datetime
from dataclasses import dataclass, field, fields, replace
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from core.workspace import (
    Workspace, LoadedMeasurement, LoadedTrace, Sample, MeasurementResultRow, SampleRhRow,
    blank_label,
    set_result_key, replicate_rh_key, extrapolation_result_key, DELIBERATE_TIERS,
    RESULT_QUANTITIES, RESULT_VALUE_FIELDS, RESULT_QUANTITY_TITLES,
    reset_result_quantity, RegroupLoss,
    _DLS_PARAM_KEYS, _SLS_PARAM_KEYS, _PROVENANCE_KEYS, _absent,
)
from core.naming import (measurement_display_name, measurement_name,
                         sample_display_name, source_file_name)
from core.data_models import (_normalize_chain_type, casefold_identity_name,
                              polarization_state as _polarization_state,
                              round_angle_for_identity,
                              round_concentration_for_identity,
                              round_temperature_for_identity, usable_rate)
from physics import solvents as solvents_lib
from app.settings import SettingsState
from app.analysis_region import (REGION_LABEL, REGION_NOTE_PREFIX, RegionState,
                                 describe_region, region_fit_kwargs)
from analysis import dls as dls_engine
from analysis.dls.cumulants import _pdi_check as _dls_pdi_check
from analysis.dls.cumulants import _size_range_check as _dls_size_range_check
from analysis.dls import DEFAULT_AMPLITUDE_FLOOR, MIN_DECAY_COMPLETENESS
from analysis import sls as sls_engine
from analysis import form_factor as ff
from analysis import depolarization as depol_engine
from analysis import utilities as util
from analysis import trace_analysis as trace_engine
from analysis import derived_sls
from analysis import mode_partition
from analysis import synthetic_dataset as synth
from analysis import optical_model as optical
from analysis.utilities import (
    compute_rho, ResultCandidate, select_default_candidate,
    fit_power_law, ScalingResult,
)
from analysis import uncertainty as unc
from physics import constants as phys
from exporting import export as exporter


# Trace-analysis defaults. These used to be persisted settings; per feedback
# 2026-06-26 #6 the user-facing ones (outlier k) moved into the Traces tab as a
# session field, and the rest are fixed engine fallbacks here (they were never
# per-run controls).
_TRACE_BASELINE_METHOD = 'percentile'
_TRACE_BASELINE_PARAMETER = 25.0
_TRACE_OUTLIER_K = trace_engine.OUTLIER_K_DEFAULT   # seeded, never restated
_TRACE_BLOCKVAR_THRESHOLD = trace_engine.BLOCKVAR_THRESHOLD_DEFAULT  # seeded, never restated
_TRACE_ADF_SIGNIFICANCE = 0.05

# THE single source for which trace diagnostics can export their own data, and
# what each is called. The Traces tab seeds its Diagnostic combo from this rather
# than restating the names, so a view can never appear on the tab with no export
# behind it -- the same one-source discipline the outlier k follows above.
# Insertion order is the order the combo offers them; 'running_average' is
# exported but is an overlay rather than a combo entry, so it comes last.
TRACE_DIAGNOSTIC_VIEWS = {
    'histogram': 'Count-rate histogram',
    'block_variance': 'Block variance',
    'allan': 'Allan deviation',
    'psd': 'Power spectral density',
    'periodogram': 'Periodogram + false alarm',
    'running_average': 'Running average',
}

# The subset the Diagnostic combo offers as sub-plots (running average is drawn
# as an overlay on the main trace axes instead).
TRACE_DIAGNOSTIC_PLOT_VIEWS = tuple(
    k for k in TRACE_DIAGNOSTIC_VIEWS if k != 'running_average')


def _value_in(value: Optional[float], excluded) -> bool:
    """True if `value` matches any entry in `excluded` (float-tolerant). Used to drop
    user-excluded outlier angles/concentrations from a multi-point fit (#9)."""
    if value is None or not excluded:
        return False
    return any(np.isclose(float(value), float(e)) for e in excluded)


def _in_axis_terms(build, kw):
    """Run a multi-angle synthetic build, translating ONE refusal into the lag-axis terms.

    `synthetic_dataset._tuple_multi_angle`'s pattern, one layer up: when the caller passed
    no `run_duration_s` on a multi-tau axis (the Utilities generator offers no run duration,
    so the set runs for the builder's default), a lag axis that outlasts the run is refused
    in the multi-tau settings the caller CAN change. The predicate stays in
    `build_multi_angle_dls`; only the vocabulary moves.
    """
    try:
        return build()
    except ValueError as exc:
        if ('run_duration_s' in kw or kw.get('grid') != 'multitau'
                or 'does not outlast' not in str(exc)):
            raise
        run_s = inspect.signature(synth.build_multi_angle_dls).parameters[
            'run_duration_s'].default
        raise ValueError(
            f"The lag axis runs past the synthetic set's {run_s:g} s run, and no lag can "
            f"outlast the run that measured it. Shorten the axis: fewer Blocks (octaves), "
            f"a smaller Δt₀, or fewer Channels/octave.") from exc


def _refuse_an_average_as_a_repeat(lms) -> None:
    """Raise if a selection handed to any averaging action holds a replicate average.

    An average is not a repeat, so it never enters another average (combination doctrine
    rule 1, core/data_models.py, which is this refusal's normative home). Counted beside its
    own sources it adds no scatter but raises N; beside independent runs it counts as one
    repeat while carrying a fraction of one run's variance. Either way the replicate +/-
    under-reports. Keyed on the item's own tag, never on its sources, so no source-id
    bookkeeping can go stale.
    """
    n_avg = sum(1 for lm in lms if lm.derived_kind == 'replicate_average')
    if n_avg:
        raise ValueError(
            f"The selection holds {n_avg} averaged measurement(s). An average is not a "
            f"repeat measurement, so it cannot enter another average: counted as one it "
            f"understates the replicate ±. Select the source runs instead of the average "
            f"(fit the average on its own to use its cleaner curve).")


def _ddls_paired_angles(angle_geometry_pairs) -> set:
    """The angle buckets holding BOTH a VV and a VH -- THE DDLS pairing rule.

    Shared by run_ddls (the fit) and ddls_correlogram_summary (the UI preview), so
    the preview can never disagree with the fit about which angles pair. Takes
    (angle_deg, geometry-label) pairs; angles bucket through
    round_angle_for_identity and labels through _polarization_state.
    """
    by_angle: Dict[float, set] = {}
    for angle, geom in angle_geometry_pairs:
        by_angle.setdefault(round_angle_for_identity(angle),
                            set()).add(_polarization_state(geom))
    return {a for a, states in by_angle.items()
            if 'VV' in states and 'VH' in states}


def _fmt_deg(angle_deg: Optional[float]) -> str:
    """Angle for a provenance label: '90°' (or '?°' if unknown)."""
    if angle_deg is None:
        return '?°'
    return f"{float(angle_deg):g}°"


def _fmt_conc(conc_g_per_mL: Optional[float]) -> str:
    """Concentration for a provenance label, in mg/mL ('0.5 mg/mL')."""
    if conc_g_per_mL is None:
        return 'c = ?'
    return f"{float(conc_g_per_mL) * 1e3:g} mg/mL"


def _fmt_r2(r2: float) -> str:
    """R^2 for a provenance label; 'n/a' for a degenerate (non-finite) fit."""
    return f"{r2:.3f}" if np.isfinite(r2) else 'n/a'


# Human-readable name for each form-factor model, for the model-dependent flag/label
# (ui_style_guide §9 terminology). Keyed by the ff model constant.
_FORM_FACTOR_LABELS = {
    ff.SPHERE: 'sphere', ff.DEBYE: 'Gaussian coil', ff.ELLIPSOID: 'ellipsoid',
    ff.MIE: 'Mie sphere', ff.CORE_SHELL: 'core-shell',
}

# Agree-on-Rg tolerance for the review-01 "size determined, shape ambiguous" consensus candidate
# (Part C): a degenerate fit's per-solution Rg values agree when their relative spread is within
# _FF_CONSENSUS_RG_REL OR their absolute spread is within _FF_CONSENSUS_RG_ABS_NM (the floor keeps
# the relative test from being needlessly harsh at small sizes). Owner-set defaults (2026-07-21):
# core-shell 36.8/37.3/37.3/37.3 -> agree; ellipsoid 101/129/175 -> do not. The consensus reports
# the MEAN of the solution Rg values. Documented in the Advanced Guide.
_FF_CONSENSUS_RG_REL = 0.10
_FF_CONSENSUS_RG_ABS_NM = 1.0
# Safety cap on the number of per-solution degenerate candidates offered (owner-set): real
# ellipsoid/core-shell degenerate sets are 3-4; a larger set is a malformed fit, so offer the
# top-N by fit quality with a "N of M shown" note rather than flooding the picker.
_FF_MAX_SOLUTIONS_OFFERED = 8


def _rg_values_agree(rgs) -> bool:
    """Whether a degenerate fit's per-solution Rg values agree closely enough to call the SIZE
    determined even though the shape/radii are not (review-01 Part C). True when the peak spread is
    within _FF_CONSENSUS_RG_REL of the mean OR within _FF_CONSENSUS_RG_ABS_NM absolute. Needs >= 2
    finite positive values; a single value is trivially 'determined' (True)."""
    vals = [float(r) for r in rgs if r is not None and np.isfinite(r) and r > 0]
    if len(vals) < 2:
        return len(vals) == 1
    spread = max(vals) - min(vals)
    mean = sum(vals) / len(vals)
    return spread <= _FF_CONSENSUS_RG_ABS_NM or (mean > 0 and spread / mean <= _FF_CONSENSUS_RG_REL)


def _cap_solutions(sols: list) -> tuple:
    """Apply the offered-solution cap (review-01): if a degenerate set exceeds
    _FF_MAX_SOLUTIONS_OFFERED, keep the top-N by fit quality (R²). Returns
    ``(kept, n_total, was_capped)``. A well-formed set (<= the cap) is returned unchanged."""
    n_total = len(sols)
    if n_total <= _FF_MAX_SOLUTIONS_OFFERED:
        return sols, n_total, False
    kept = sorted(
        sols,
        key=lambda s: (float(getattr(s, 'r_squared', float('-inf')))
                       if np.isfinite(getattr(s, 'r_squared', float('nan')))
                       else float('-inf')),
        reverse=True)[:_FF_MAX_SOLUTIONS_OFFERED]
    return kept, n_total, True


def _ff_solution_descriptor(model: str, sol) -> str:
    """The discriminating shape parameters of one degenerate form-factor solution, so the user can
    map an offered solution to their external (EM/SAXS/…) picture (G3). Ellipsoid -> prolate/oblate
    + axial ratio + Rg; core-shell -> core/outer radii + Rg. '' for a non-degenerate solution or a
    shape with no distinguishing parameters (sphere/coil/Mie are never degenerate)."""
    if not getattr(sol, 'degenerate', False):
        return ''
    rg = getattr(sol, 'rg_nm', float('nan'))
    if model == ff.ELLIPSOID:
        kind = 'prolate' if getattr(sol, 'is_prolate', False) else 'oblate'
        eps = getattr(sol, 'axial_ratio', float('nan'))
        # Honesty (invariant 8): only present ε as a resolved discriminator when the fit says the
        # axial ratio is identifiable. When it is not (low qRg -> axial_ratio_reliable False, SE
        # withheld), flag the value as unresolved rather than stating a shape verdict the data
        # cannot support. The branch's own ε is still shown (it distinguishes this solution).
        if getattr(sol, 'axial_ratio_reliable', True):
            return f"{kind} (ε≈{eps:.2g}, Rg {rg:.0f} nm)"
        return f"{kind} (ε≈{eps:.2g} — axial ratio unresolved, Rg {rg:.0f} nm)"
    if model == ff.CORE_SHELL:
        rc = getattr(sol, 'r_core_nm', float('nan'))
        ro = getattr(sol, 'r_outer_nm', float('nan'))
        return f"core {rc:.0f} nm / outer {ro:.0f} nm (Rg {rg:.0f} nm)"
    return f"Rg {rg:.0f} nm"


def _ff_source_id(model: str, descriptor: str) -> str:
    """The identity of ONE offered form-factor solution.

    A degenerate fit reports several solutions of the same model (`degenerate = len(accepted)
    > 1` in both multi-solution fitters), so the model name alone names all of them at once.
    The descriptor -- the shape parameters that distinguish the branches, and '' for a
    non-degenerate fit -- is what makes the key unique, so it is appended exactly when there
    is more than one solution to tell apart.

    Measured: the collision is currently UNREACHABLE through the one consumer that compares
    keys, because every member of a degenerate set carries `auto_selectable=False`, so
    `select_default_candidate` returns None and only an explicit pick can store one -- which
    `_is_current` excludes before it compares anything. Separated anyway rather than left
    resting on an unrelated gate: this key is an identity now.

    It separates the branches the DISPLAY separates, which is not the same as unique. The
    descriptor is display prose and rounds (`{eps:.2g}`, `{rg:.0f}`), so two branches
    differing below the rounding still mint one id. Left as-is while the collision is
    unreachable; a caller that ever compares these keys outside `_is_current` needs a key
    built from the unrounded parameters instead."""
    return f'ff_{model}@{descriptor}' if descriptor else f'ff_{model}'


def _per_series_source_id(kind: str, index: int, conc: float) -> str:
    """The identity of ONE per-concentration SLS fit (Debye / Guinier).

    Keyed on POSITION in the series, not on the concentration alone. `masked_rayleigh`
    returns one result per SLS measurement, so an ordinary repeat run at one concentration
    yields two results that `f'{kind}@{conc}'` named jointly -- while their labels, which
    carry the angle count and R2, named one each. A key coarser than the label it replaced
    is not an identity, and the picker would preselect whichever came last.

    The position is stable within a refresh: the Rg and Mw builders enumerate the same
    `nonzero` list, in load order. Masking a concentration or removing a measurement shifts
    it, so a stored key stops matching -- which fails closed (nothing preselected), the safe
    direction, and honest, because the candidate set really did change."""
    return f'{kind}@{index}:{conc}'


# The candidate `kind` of a single-angle ladder fit (the fixed-angle Debye plot over a
# concentration series). Spelled once: the Mw and A2 builders, the picker's grouping table
# in gui/cross_module.py and the tests all name the same token.
_LADDER_KIND = 'sls_ladder'


def _ladder_source_id(angle_deg: float) -> str:
    """The identity of ONE single-angle ladder fit: its ANGLE.

    A sample may carry several ladders at once -- one per angle the user ran -- and each is
    offered as its own explicit pick (owner-ruled 2026-09-03), so the angle is exactly what
    tells them apart. It is also the whole of the stored result's identity within a
    (sample, fraction): `results[('single_angle_ladder', sid, angle, fraction)]`.

    `:g` matches `_fmt_deg`, so the id and the label name the same angle in the same
    spelling; a ladder re-run at the same angle mints the same id, which is the point (the
    user's pick survives the re-run)."""
    return f'ladder@{float(angle_deg):g}'


#: The suffix naming a tier-1 Rh candidate that is the user's OWN stored Correlogram
#: cumulant fit, as `gamma_q2@run` names a stored Gamma-vs-q^2 run.
_STORED_FIT_SUFFIX = '@run'


def _stored_fit_source_id(item_id: str) -> str:
    """The identity of the tier-1 candidate read from the user's own Correlogram fit of
    `item_id`, told apart from the candidate re-derived at the Settings order (whose id
    is the bare item_id) because the two are different fits: a change of the Settings
    order invalidates a pick of the second and never of the first, which ran at the
    Correlogram tab's own order box."""
    return f'{item_id}{_STORED_FIT_SUFFIX}'


def _is_stored_fit_source_id(source_id) -> bool:
    return isinstance(source_id, str) and source_id.endswith(_STORED_FIT_SUFFIX)


def _ladder_candidate_label(res) -> str:
    """The picker/export provenance label for one single-angle ladder fit — SINGLE source
    shared by the Mw and A2 candidate builders (the `_ff_candidate_label` pattern: one physical
    fit must not get two divergent labels).

    It names the ANGLE because at one angle the q → 0 extrapolation was never done
    (invariant 7): the Mw is Mw·P(q) and the slope is 2·A2 only where P(q) ≈ 1, so "apparent"
    here is a property of the angle the user chose, and a reader of the label — in the picker,
    on `SampleResult.mw_label`, or in the Cross-Sample export's source column — must be able to
    see which angle that was."""
    return (f"Single-angle ladder at θ = {_fmt_deg(getattr(res, 'angle_deg', None))}, "
            f"{int(getattr(res, 'n_conc', 0))} concentrations "
            f"(apparent{_scale_caveat(res)}, R² = {_fmt_r2(float(res.r_squared))})")


def _ff_candidate_label(name: str, sol, degenerate: bool, descriptor: str, r2: float) -> str:
    """The picker/export provenance label for one form-factor solution — SINGLE source shared by the
    Rg and Mw candidate builders (rule 10: one physical fit must not get two divergent labels). A
    degenerate solution reads by its distinguishing shape parameters + 'user-selected among N
    degenerate solutions' (G2, neutral); a resolved single fit keeps the model-dependent-only text."""
    regime = 'apparent' if sol.is_apparent else 'thermodynamic'
    if degenerate:
        n = int(getattr(sol, 'n_solutions', 1) or 1)
        return (f"{name.capitalize()} P(q) {descriptor} — user-selected among {n} degenerate "
                f"solutions ({regime}, R² = {_fmt_r2(r2)}) — model-dependent ({name} P(q))")
    return (f"{name.capitalize()} P(q) fit over {sol.n_angles} angles "
            f"({regime}, R² = {_fmt_r2(r2)}) — model-dependent ({name} P(q))")


def _ff_solution_ok(solution) -> bool:
    """Shared Rg/Mw precondition for one form-factor solution: it converged. Generic over the
    whole family (rule 11) -- reads only shared attributes.

    Degenerate solutions (an ellipsoid/core-shell fit the data cannot resolve to one shape) are
    NO LONGER excluded here (review-01): each is offered as an explicit, fully-flagged pickable
    candidate (`degenerate=True`, `auto_selectable=False`) -- guided manual entry among the
    physically-admissible fits, never the tab's silent default. A FormFactorResult (sphere/coil/
    Mie) has no `degenerate` field, so it stays a single non-degenerate always-offered solution."""
    return bool(getattr(solution, 'converged', False))


def _ff_rg_offered(solution) -> bool:
    """Whether a form-factor solution offers an honest, model-dependent Rg candidate.

    On top of `_ff_solution_ok`: a finite Rg, and a size the fit can stand behind --
    `rg_reliable` is False when the coated Rg is contrast-matched/hollow (nan) and, on
    EVERY form-factor result, when the fitted size collapsed below
    `analysis.sls.FORM_FACTOR_MIN_SIZE_NM`. That second cause is why this reads
    `rg_reliable` and not `form_factor_valid`: the latter is ALSO False past a form-factor
    minimum, where the amplitude trades off against the zero location and the size is
    sound, so gating here on it would withhold good Rg values. The getattr default is now
    a fallback for a foreign result type, not a description of the family."""
    if not _ff_solution_ok(solution):
        return False
    rg = getattr(solution, 'rg_nm', float('nan'))
    if rg is None or not np.isfinite(rg):
        return False
    return bool(getattr(solution, 'rg_reliable', True))


def _scale_caveat(res) -> str:
    """A result label's scale caveat, from BOTH scale flags; '' when the scale is known.

    An Mw or A2 can be on an arbitrary scale for two independent reasons -- no calibration
    constant (dR arbitrary) or no dn/dc (K = 1, so Kc/dR arbitrary) -- and both can hold at
    once. THE one body for every label that states the caveat: the offered CANDIDATE labels
    and the labels the Run writes onto `SampleResult`, which the Cross-Sample source picker
    shows as the value's provenance. Those two drifted apart once already -- the candidate
    labels learned the dn/dc cause while the written-back ones kept saying "UNCALIBRATED"
    for a run that was calibrated -- because a label is a COPY of the reason, not a read of
    the flag, so it survives every flag-level fix. Reads the flags off the result rather
    than taking bools, so a third scale flag cannot be added without passing through here.
    """
    parts = []
    if not getattr(res, 'calibrated', True):
        parts.append('UNCALIBRATED')
    if not getattr(res, 'dn_dc_supplied', True):
        parts.append('no usable dn/dc')
    return f"; {' + '.join(parts)} — arbitrary scale" if parts else ''


def _ff_mw_offered(solution) -> bool:
    """Whether a form-factor solution offers a model-dependent Mw candidate: `_ff_solution_ok`
    plus a finite Mw from a calibrated, reliable fit (Mw is scale-dependent -- needs
    calibration, exactly like Zimm's). Independent of Rg-definedness (a core-shell with an
    undefined Rg can still have a valid Mw)."""
    if not _ff_solution_ok(solution):
        return False
    mw = getattr(solution, 'mw_g_per_mol', float('nan'))
    if mw is None or not np.isfinite(mw):
        return False
    return bool(getattr(solution, 'calibrated', False)
                and getattr(solution, 'mw_reliable', False))


def _result_model_dependent(res, quantity: str) -> bool:
    """Whether a SampleResult contributes a MODEL-DEPENDENT (form-factor) value to a cross-sample
    analysis of `quantity`: its Mw is a form-factor Mw (feeds every Mw axis), or — on the 'rg'
    axis — its Rg is a form-factor Rg. Rh (DLS) and A2 are never model-dependent. Single-sourced
    so the scaling caption/export and the architecture export/caption apply ONE definition of the
    honesty axis (rule 10 — a change here reaches every surface at once)."""
    md = bool(getattr(res, 'mw_model_dependent', False))
    if quantity == 'rg':
        md = md or bool(getattr(res, 'rg_model_dependent', False))
    return md


def _result_degenerate(res, quantity: str) -> bool:
    """Whether a SampleResult contributes a DEGENERATE form-factor value to a cross-sample analysis
    of `quantity`: its Mw came from a degenerate fit (feeds every Mw axis), or -- on the 'rg' axis --
    its Rg did. The sibling of `_result_model_dependent` (rule 10): a degenerate value is always also
    model-dependent, but flagged one tier stronger (the shape assumption is one the data cannot pin
    down). Rh (DLS) and A2 are never degenerate. Single-sourced so the scaling caption/export and the
    architecture export/caption apply ONE definition (a change here reaches every surface at once)."""
    dg = bool(getattr(res, 'mw_degenerate', False))
    if quantity == 'rg':
        dg = dg or bool(getattr(res, 'rg_degenerate', False))
    return dg


def _scaling_point_source(res, quantity: str) -> str:
    """The plain-language provenance of ONE plotted scaling point: where its Mw came from,
    and where the plotted y value (Rg / Rh / A2) came from.

    The stored `<q>_label` verbatim, never a re-derivation — the same rule the picker and the
    architecture export's `Rg source` / `Rh source` columns follow, and the reason a
    single-angle ladder's angle reaches the file at all: the angle lives in the label
    (`_ladder_candidate_label`), not in a flag. A value with no label yet (nothing selected,
    or a pre-key session) names its source tier instead of printing an empty cell, so the
    column never reads as "no provenance" when the real answer is "hand-entered".
    """
    y_field = {'rg': 'rg', 'rh': 'rh', 'a2': 'a2'}[quantity]
    y_name = {'rg': 'Rg', 'rh': 'Rh', 'a2': 'A2'}[quantity]

    def _one(prefix: str, q: str) -> str:
        label = getattr(res, f'{q}_label', '') or ''
        return f"{prefix}: {label or (getattr(res, f'{q}_source', '') or 'unknown')}"

    return f"{_one('Mw', 'mw')}; {_one(y_name, y_field)}"


def _result_arbitrary_scale(res, quantity: str) -> bool:
    """Whether a SampleResult contributes an ARBITRARY-SCALE value to a cross-sample analysis of
    `quantity`: its Mw is on an arbitrary scale (Mw is the X axis of EVERY scaling plot, so its
    flag feeds them all), or -- on the 'a2' axis -- its plotted A2 is. Rg and Rh carry no scale
    flags at all: both are slope/intercept ratios in which the unknown scale cancels, so neither
    can put its own axis on an arbitrary scale.

    The OR is load-bearing, not a convenience: replacing the Mw term with A2's on the A2 axis
    would drop the caveat from an uncalibrated Mw plotted against a trusted A2 -- which is
    exactly what a one-line "use A2's flag on the A2 axis" fix does.

    Third sibling of `_result_model_dependent` / `_result_degenerate` and single-sourced for the
    same reason (rule 10): the scaling caption and the scaling export apply ONE definition of
    "a plotted value whose scale is unknown", so a change here reaches both at once. Calls the
    SampleResult accessors rather than re-spelling the predicate -- `is_arbitrary_scale` has
    exactly one body, gated by tests/test_arbitrary_scale_predicate_family.py.
    """
    arb = bool(res.mw_is_arbitrary_scale())
    if quantity == 'a2':
        arb = arb or bool(res.a2_is_arbitrary_scale())
    return arb


# Parameters shared by every measurement in a sample (entered once, propagated).
# Concentration and angle are deliberately excluded -- they are per-measurement
# axes within a sample, not shared identity.
_SHARED_PARAM_KEYS = (
    'polymer_name', 'solvent_name', 'temperature_K', 'wavelength_nm',
    'solvent_refractive_index', 'viscosity_Pa_s', 'dn_dc_mL_per_g',
    # chain_type (linear/branched) is a sample-level label, entered once and
    # propagated like polymer_name; it feeds the Cross-Sample g/h readouts and is
    # deliberately NOT part of sample_group_key (a label, not a grouping key).
    'chain_type',
    # The typed 90 deg solvent intensity: a property of the SOLVENT, so it is entered
    # once per sample exactly as dn/dc is, not per concentration. SLS-only -- it is
    # absent from _DLS_PARAM_KEYS, and `set_shared_param` writes a key only when the
    # sibling's own kind roster allows it, so DLS members skip it.
    'solvent_blank_intensity_90deg',
)

# Parameter keys the Reset button preserves: the scattering
# angle is a structural fact read from the instrument file, not a typed parameter, so
# wiping it would lose which angle a correlogram belongs to. Everything else the user
# entered is cleared.
_RESET_PRESERVE_KEYS = ('angle_deg',)


# Which fitted parameters the replicate derived-results averaging reports per
# method, and which one (if any) is the single Rh written back to the sample.
# Each spec is (list of (display_name, getter), primary_rh_name); primary_rh_name
# is None for multi-mode methods (double-exp), which are reported only.
_DLS_PARAM_SPECS = {
    'cumulant': ([
        ('Rh', lambda r: r.rh_nm),
        ('Γ', lambda r: r.gamma_s_inv),
        ('PDI', lambda r: r.pdi),
    ], 'Rh'),
    'single': ([
        ('Rh', lambda r: r.mode.rh_nm),
        ('Γ', lambda r: r.mode.gamma_s_inv),
        ('β', lambda r: r.beta),
    ], 'Rh'),
    'double': ([
        ('Rh (fast)', lambda r: r.mode1.rh_nm),
        ('Rh (slow)', lambda r: r.mode2.rh_nm),
        ('Γ (fast)', lambda r: r.mode1.gamma_s_inv),
        ('Γ (slow)', lambda r: r.mode2.gamma_s_inv),
    ], None),
    'kww': ([
        ('Rh (τc)', lambda r: r.rh_from_tau_c_nm),
        ('Rh (⟨τ⟩)', lambda r: r.rh_from_mean_tau_nm),
        ('τc', lambda r: r.tau_c_s),
        ('stretch', lambda r: r.stretch),
    ], 'Rh (τc)'),
}

_DLS_PARAM_UNITS = {
    'Rh': 'nm', 'Rh (fast)': 'nm', 'Rh (slow)': 'nm',
    'Rh (τc)': 'nm', 'Rh (⟨τ⟩)': 'nm',
    'Γ': 's⁻¹', 'Γ (fast)': 's⁻¹', 'Γ (slow)': 's⁻¹',
    'PDI': '', 'β': '', 'τc': 's', 'stretch': '',
}

# Per-method FIT DETAILS -- every quantity of a parametric fit that is not already a
# headline column -- as `(storage key, display name, getter, unit, note-target key)`. Read
# by BOTH the snapshot writer and the Summary columns, so a quantity cannot be saved without
# being shown or shown without being saved. `_collect_fit_details` below is the contract's
# normative home: what each element means, why this is a sibling of `_DLS_PARAM_SPECS`
# rather than part of it, and what is stored versus rendered.
_DLS_DETAIL_SPECS = {
    'cumulant': [
        ('gamma', 'Γ', lambda r: r.gamma_s_inv, 's⁻¹', 'gamma_s_inv'),
        ('mu2', 'μ₂', lambda r: r.mu2_s_inv2, 's⁻²', 'mu2_s_inv2'),
        ('mu3', 'μ₃', lambda r: r.mu3_s_inv3, 's⁻³', 'mu3_s_inv3'),
        ('mu4', 'μ₄', lambda r: r.mu4_s_inv4, 's⁻⁴', 'mu4_s_inv4'),
        ('order', 'order', lambda r: r.order, '', 'order'),
        ('method', 'method', lambda r: r.method, '', 'method'),
        # `baseline_b` and not `baseline`, and the distinction is load-bearing rather than
        # cosmetic: on the nonlinear arm this is a FITTED parameter, while every entry
        # below is the caller's PINNED offset handed back unchanged. They are different
        # quantities that share a word, they behave differently under a solver, and a test
        # tolerance keys on this one alone (`tests/test_session_roundtrip_family.py`).
        ('baseline_b', 'baseline B', lambda r: r.baseline, '', 'baseline'),
    ],
    'single': [
        ('gamma', 'Γ', lambda r: r.mode.gamma_s_inv, 's⁻¹', None),
        ('beta', 'β', lambda r: r.beta, '', 'beta'),
        ('baseline', 'baseline', lambda r: r.baseline, '', 'baseline'),
        ('converged', 'converged', lambda r: r.success, '', 'success'),
    ],
    'double': [
        ('frac_fast', 'fast frac', lambda r: r.mode1.amplitude_fraction, '', None),
        ('frac_slow', 'slow frac', lambda r: r.mode2.amplitude_fraction, '', None),
        ('baseline', 'baseline', lambda r: r.baseline, '', 'baseline'),
        ('converged', 'converged', lambda r: r.success, '', 'success'),
    ],
    'kww': [
        ('stretch', 'stretch s', lambda r: r.stretch, '', 'stretch'),
        ('tau_c', 'τ<sub>c</sub>', lambda r: r.tau_c_s, 's', 'tau_c_s'),
        ('mean_tau', '⟨τ⟩', lambda r: r.mean_tau_s, 's', 'mean_tau_s'),
        # The SECOND size this arm reports. One decay, two ways of reporting it: the
        # Summary headline carries the tau_c size, so without this one the mean-tau size
        # is on no durable surface at all and a reloaded session cannot show it.
        ('rh_mean_tau', 'R<sub>h</sub>(⟨τ⟩)', lambda r: r.rh_from_mean_tau_nm,
         'nm, apparent', None),
        ('baseline', 'baseline', lambda r: r.baseline, '', 'baseline'),
        ('converged', 'converged', lambda r: r.success, '', 'success'),
    ],
}


def _collect_fit_details(method: str, res) -> Dict[str, Any]:
    """The detail quantities of one parametric fit, keyed by their display names.

    This docstring is the normative home of `_DLS_DETAIL_SPECS`'s contract.

    EACH ENTRY is `(storage key, display name, getter, unit, note-target key)`.

    THE STORAGE KEY is what `fit_details` is keyed by in the session, and it is a stable
    ASCII token that is NOT the display name. The two were one string at first, which
    welded the session format to a user-visible label: renaming a column -- the kind of
    edit this project's quality bar licenses as cheap and unreviewed -- would have made
    every saved value for that quantity unreachable, falling back to a blank cell with no
    error and no version bump to notice. Display names are now free to change; storage
    keys are not, and `tests/test_session_roundtrip_family.py` pins the roster by
    identity so a change to one has to be deliberate. (No migration was owed: the
    display-name form existed only between two commits on one unreleased version, so no
    session outside this repository was ever written with it.)

    A SIBLING of `_DLS_PARAM_SPECS` rather than an extension of it, deliberately: that
    dict's getters are fed into the replicate-averaging path's numeric mean, and `method`
    and `converged` here are a string and a bool. Same shape, different consumer.

    THE FOURTH ELEMENT is the RESULT ATTRIBUTE the column shows, which is how a passive
    note reaches the cell it explains: `_keyed_notes` (analysis/dls/_common.py) tags each
    note with the attribute it is about, and a note whose tag matches this key lands on
    this column. `None` means no note can target it -- the value is nested (an exponential
    mode's Gamma) and no note builder names it. A note tagged '' (about the whole fit, as
    the dead-channel note is), or tagged an attribute no column shows, falls back to the
    method's headline cell: never dropped, and never guessed onto a neighbour.

    WHAT IS STORED is the fit result's own value -- the raw `method` string, the raw
    `success` bool -- never the painted '... (failed)' or 'yes'/'no'. Rendering is the
    table's job, which is what lets a test assert the saved row equals the result.

    Driven entirely by `_DLS_DETAIL_SPECS`, so adding a quantity there is the only edit
    needed to have it saved: there is no second list here to keep in step. A method with
    no detail spec (every distribution method -- their cells are resolved peaks, which
    have no fixed parameter set) collects nothing and gets an empty mapping.

    A getter that raises is a quantity this result does not carry -- a shape the caller
    should never produce, since the spec is written against the arm's own result class,
    but a snapshot writer is the wrong place to lose a whole row of good numbers over one
    missing attribute. The quantity is left out and the rest are stored.
    """
    out: Dict[str, Any] = {}
    for key, _name, getter, _unit, _target in _DLS_DETAIL_SPECS.get(method, ()):
        try:
            value = getter(res)
        except AttributeError:
            continue
        # numpy scalars serialize as their own type through the session's JSON writer;
        # store the plain Python value so a reloaded row compares equal to the fit.
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        out[key] = value
    return out


def _collect_passive_notes(res) -> Dict[str, List[str]]:
    """A fit's passive notices, grouped by the result attribute each is about.

    Reads the engine's index-aligned `notes` / `note_targets` pair and turns it into the
    mapping the durable row stores (`MeasurementResultRow.passive_notes`), which is the
    shape the Summary table reads: it asks a column "what notes are about you?" and never
    has to index two tuples itself.

    UNEQUAL LENGTHS ARE TREATED AS ALL-FIT-LEVEL, never as a partial alignment. A result
    built by a producer that has not been taught targets carries none, and guessing that
    the first N notes line up with the N targets would put a note on a number it is not
    about -- which is worse than the '' fallback, where the note still reaches the reader
    on the fit's headline cell. `_common._keyed_notes` is why this stays a defensive
    branch rather than a live one: a producer cannot build a misaligned pair through it.
    """
    notes = tuple(getattr(res, 'notes', ()) or ())
    targets = tuple(getattr(res, 'note_targets', ()) or ())
    if len(targets) != len(notes):
        targets = ('',) * len(notes)
    out: Dict[str, List[str]] = {}
    # strict=True asserts what the branch above just guaranteed: after it the two are
    # equal in length on every path, so a raise here would mean that normalization had
    # stopped working -- which is worth hearing about rather than silently truncating a
    # note off the end.
    for target, note in zip(targets, notes, strict=True):
        out.setdefault(target, []).append(note)
    return out


def _relayed_run_notes(named) -> List[str]:
    """Each run's PASSIVE fit notes for a result built from several runs: `named` is
    [(run name, its notes)], and each distinct note is returned once with the runs that
    carried it named in front ("A, B: <note>"), in first-seen order, a name once each.
    A caller passes the notes `_collect_passive_notes` records on a single run's Summary
    row, flattened across the columns they are about.

    `analysis/dls/angular.py::_relayed_point_notes`' shape, naming runs where that one
    numbers points. A replicate average or a DDLS result is the only surface that shows its
    number, so a note about how a run was fitted -- a floating baseline or µ2 resting on its
    bound, dead channels -- must reach it; the flags are the result's own and untouched."""
    noted: Dict[str, List[str]] = {}
    for name, notes in named:
        for note in notes:
            if note:
                noted.setdefault(note, []).append(name)
    return [f"{', '.join(dict.fromkeys(names))}: {note}" for note, names in noted.items()]

# Distribution methods yield a VARIABLE number of size-distribution peaks (not a
# fixed parameter set), so replicate averaging aligns their peaks positionally
# (Rh-ascending) instead of by name. They are report-only (no single Rh written).
_DLS_DISTRIBUTION_METHODS = ('nnls', 'contin', 'lognormal')

#: Per-point `quality` tags (from `_dls_run_points`) that are still handed to a
#: sample-level fit. ONE definition, used at every eligibility site: sites that disagree
#: remove a point BEFORE the engine sees it, making the engine's drop disclosure (the
#: dotted plot marker, the result note, the export Comments cell) unreachable from the GUI.
#: `'nonpositive_gamma'` and `'high_pdi'` are FLAGGED-BUT-PASSED: the fit converged, and
#: the ENGINE decides -- both lines drop a non-positive Γ and say so on `notes`, and both
#: carry a high-PDI point's verdict on `rh_reliable`. `'fit_failed'` / `'nonpositive_q'`
#: stop here, because there is no Γ to hand on at all.
FIT_ELIGIBLE_QUALITIES = ('ok', 'high_pdi', 'nonpositive_gamma')


#: The Settings cumulant residual-weighting POLICY tokens and the engine request each maps
#: to: 'sourced' is `weighting=None`, each arm's own sourced default; 'uniform' is the
#: unweighted opt-in on either arm. A policy, never an engine token, so a later weighting
#: joins here without migrating a persisted setting.
CUMULANT_WEIGHTING_POLICIES = {'sourced': None, 'uniform': 'uniform'}


def _cumulant_weighting_request(policy: str) -> Optional[str]:
    """The `fit_cumulants` weighting request for a Settings policy token.

    Takes the TOKEN rather than reading the settings, so every reader names
    `self.settings.cumulant_weighting` in its own body -- the form the staleness census
    (`test_every_run_signature_carries_the_settings_its_engine_path_reads`) can see. An
    unrecognised value -- a hand-edited settings.json -- raises naming the setting, the
    landing a hand-edited `cumulant_method` gets at the engine; never a silent fall-back
    to 'sourced'."""
    if policy not in CUMULANT_WEIGHTING_POLICIES:
        raise ValueError(
            f"Settings cumulant_weighting must be one of "
            f"{sorted(CUMULANT_WEIGHTING_POLICIES)}, got {policy!r}. Choose the residual "
            f"weighting in Settings.")
    return CUMULANT_WEIGHTING_POLICIES[policy]


#: The Settings distribution residual-weighting POLICY tokens and the engine request each
#: maps to: 'sourced' is `weighting=None`, the distribution arms' own sourced default;
#: 'g1-squared' is the older constant-variance form on all three arms. A policy, never an
#: engine token, so a later weighting joins here without migrating a persisted setting --
#: which this control is built to take (the whitened form is specified and parked).
DISTRIBUTION_WEIGHTING_POLICIES = {'sourced': None, 'g1-squared': 'g1-squared'}


def distribution_weighting_request(policy: str) -> Optional[str]:
    """The distribution fits' weighting request for a Settings policy token.

    PUBLIC, unlike its cumulant sibling: the DLS tab's per-run weighting control maps its
    choice through this function, so the widget and the Settings seed cannot disagree about
    what a token means. Takes the TOKEN rather than reading the settings. Its one reader
    of the Setting is the seed home `_seed_distribution_kwargs`, a helper the staleness
    census cannot see into (its walk is one level deep), so what keeps a shown result
    honest is each tab's own `_run_signature`: the Distribution tab and the Mode Map each
    ride their own per-run control there (`DistributionFitFields.fit_kwargs`).
    An unrecognised value -- a hand-edited settings.json -- raises naming the setting, the
    landing a hand-edited `cumulant_method` gets at the engine; never a silent fall-back
    to 'sourced'."""
    if policy not in DISTRIBUTION_WEIGHTING_POLICIES:
        raise ValueError(
            f"Settings distribution_weighting must be one of "
            f"{sorted(DISTRIBUTION_WEIGHTING_POLICIES)}, got {policy!r}. Choose the "
            f"residual weighting in Settings.")
    return DISTRIBUTION_WEIGHTING_POLICIES[policy]


# "no argument given", distinct from an explicit None (which is a legal typed-blank
# value meaning "no solvent subtraction"). Module-level so the default is one object.
_UNSET = object()


def _typed_blank_problem(i90: Optional[float], dark_count_rate: float) -> str:
    """Why a typed 90 deg solvent intensity cannot be used, or '' when it can.

    ONE predicate, shared by the commit-time courtesy check (`validate_typed_blank`)
    and by `_typed_blank_for`, where the blank is actually constructed -- so the two
    boundaries cannot be sized differently (rule 12). It judges the RAW inputs; the
    finiteness of the constructed array is checked separately, on the array itself,
    because saturation is a property of the result and only the cross-product of I90
    with every angle in the series can reach it.

    Legal: empty (None) and exactly zero, both meaning "no solvent subtraction"; and
    any value at or above the dark count. Refused: anything non-finite, and anything
    strictly between zero and the dark count -- there the net solvent term
    `(I90 - d)/sin(theta)` is negative, i.e. a solvent that scatters less than the
    detector's own dark floor, which no measurement produces.
    """
    if i90 is None:
        return ''
    try:
        v = float(i90)
    except (TypeError, ValueError):
        return ('The solvent blank intensity at 90 degrees must be a number, or blank '
                'for no solvent subtraction.')
    if not math.isfinite(v):
        return ('The solvent blank intensity at 90 degrees must be finite, or blank '
                'for no solvent subtraction.')
    if v == 0.0:
        return ''
    d = float(dark_count_rate)
    if v < 0.0:
        return (f'The solvent blank intensity at 90 degrees cannot be negative '
                f'(got {v:g}). Leave it blank, or enter zero, to run with no solvent '
                f'subtraction.')
    if v < d:
        return (f'The solvent blank intensity at 90 degrees ({v:g}) is below the '
                f'committed dark count rate ({d:g}), so the net solvent intensity '
                f'would be negative. Raise the blank intensity, lower the dark count '
                f'rate on the SLS tab, or clear the blank intensity to run with no '
                f'solvent subtraction.')
    return ''


def _rh_value_promotable(rh) -> bool:
    """The value half of `_dls_rh_promotable`: a finite Rh inside the technique's size
    range, read with the engine's one size check so the two never drift."""
    if rh is None or not math.isfinite(rh):
        return False
    return bool(_dls_size_range_check(float(rh), dls_engine.CUMULANT_MAX_RH_NM,
                                      dls_engine.CUMULANT_MIN_RH_NM)[0])


def _dls_rh_promotable(result, rh_field: str = 'rh_nm') -> bool:
    """Whether a DLS Rh may be the PASSIVE default (and a replicate average written):
    a flag WARNS, it does not withhold -- the owner's ruling (2026-08-27) under the
    flag-not-refuse rule, normative home CLAUDE.md invariant 4. A precondition the fit
    names (a PDI past the cumulant's validity, an undecayed lag window) travels on the
    label through `_flag_suffix` and the value promotes; only a precondition that leaves
    the number UNUSABLE blocks it -- a non-finite Rh (no viscosity, a failed fit) or one
    outside the technique's own size range (the ISO ceiling, the artifact floor), read
    with the engine's one size check so the two never drift."""
    return _rh_value_promotable(getattr(result, rh_field, None))


def _flag_suffix(result, flag: str = 'rh_reliable') -> str:
    """The label suffix a flagged candidate carries into the picker, `SampleResult.rh_label`
    / `rg_label` and the export's source column: the fit's own `reliability_note`,
    verbatim, so the reason is the one the DLS / SLS tab shows (one wording, one home --
    the CONTIN sweep-edge candidate set the shape). `flag` names the result's own flag
    for the quantity offered: `rh_reliable` (the DLS tiers) or `rg_reliable` (the linear
    SLS family -- Debye / Guinier / Zimm / Berry each carry one). Empty for
    an unflagged result, and for a result that carries no such flag."""
    if getattr(result, flag, True):
        return ''
    return f" — unreliable: {getattr(result, 'reliability_note', '') or 'flagged'}"


def _reliability_note_suffix(result, flag: str) -> str:
    """The engine's OWN reliability sentence as a label suffix — `_flag_suffix` without its
    reason-less `'flagged'` fallback, and the shared body behind `_guinier_suffix` and the
    single-angle ladder candidates.

    The fallback is dropped deliberately: a False `<flag>` whose cause is one of the SCALE
    gates (uncalibrated, no dn/dc) carries no engine sentence, and those causes already travel
    on the label through `_scale_caveat` — so printing "unreliable: flagged" beside that caveat
    states a second, reason-less verdict about the same fact. No sentence, no suffix.
    (`_guinier_suffix`'s own comments record where that was learned.)"""
    if getattr(result, flag, True):
        return ''
    note = getattr(result, 'reliability_note', '') or ''
    return f" — unreliable: {note}" if note else ''


def _guinier_suffix(result) -> str:
    """The Guinier sibling of `_flag_suffix`, read on the Mw candidate: a Guinier fit
    whose largest MEASURED q*Rg exceeds the validity limit is offered with the engine's
    own regime sentence on the label and kept out of the passive default -- the flag used
    to stop at the SLS panel while the value went on into rho = Rg/Rh. A fit with no
    extractable Rg is NOT such a fit: its intercept Mw is reliable and defaults normally
    (owner-ruled 2026-08-27), which is why this reads `mw_reliable` and not the regime
    flag.
    Empty for a fit inside the regime, and for a result that carries no such flag
    (Debye). The wording is `GuinierResult.reliability_note`, which names the limit the
    verdict actually read (`qrg_max_valid`, echoed on the result), so no surface can
    print a bound the predicate did not use."""
    # Reads `mw_reliable`, the Mw verdict itself -- not `guinier_valid`, which is also
    # False when no Rg could be extracted, a state that does NOT flag the intercept
    # (owner-ruled 2026-08-27; analysis/sls.py's guinier block names the reasoning).
    if not hasattr(result, 'guinier_valid'):
        return ''                       # Debye / single-angle: no regime verdict at all
    # A False `mw_reliable` has causes the ENGINE writes no sentence for -- uncalibrated,
    # no dn/dc -- and those already travel on the label through `_scale_caveat`. Without
    # a note this printed a reason-less "unreliable: flagged" beside that caveat, and on a
    # no-extractable-Rg fit it printed a sentence that DENIES the verdict it carries. The
    # suffix exists to surface the engine's own regime sentence; no sentence, no suffix --
    # the rule now lives in `_reliability_note_suffix`, shared with the ladder candidates.
    return _reliability_note_suffix(result, 'mw_reliable')


# ===========================================================================
# Calibration state (committed / working, like a measurement's parameters)
# ===========================================================================

def _library_standard_index(wavelength_nm, temperature_C) -> Optional[float]:
    """Toluene's refractive index at (wavelength, temperature) from the solvent library,
    or None where the library has no value (outside its validity box, or a field that
    is not a number yet). The standard is toluene throughout (`compute_k_c` always calls
    `rayleigh_ratio_toluene`), so the name is not a parameter."""
    try:
        n = float(solvents_lib.refractive_index_solvent(
            'toluene', float(wavelength_nm), float(temperature_C)))
    except (TypeError, ValueError):
        return None
    # Rounded at the source, not at the display: the panel's line edit shows str(value)
    # and re-parses it on every focus-out, so a value that does not round-trip through
    # its own text would stop equalling the library's and read as user-typed. Five
    # decimals sit two orders inside the library's own band (+/- 1.2e-3 for toluene).
    return round(n, 5) if math.isfinite(n) else None


# The shipped prefill for the standard's refractive index: the library's toluene value at
# the shipped standard wavelength / temperature, never a typed literal (invariant 3 -- an
# offered physical value is science-based and provenance-tagged; the tooltip names the
# library). It was a hard-coded 1.496 -- toluene's sodium-D value, 2.9 sigma from the
# library's 1.4996 at 532 nm -- biasing every calibrated Mw by +0.48 %.
_DEFAULT_STANDARD_WAVELENGTH_NM: float = 532.0
_DEFAULT_STANDARD_TEMPERATURE_C: float = 25.0
_DEFAULT_STANDARD_REFRACTIVE_INDEX: Optional[float] = _library_standard_index(
    _DEFAULT_STANDARD_WAVELENGTH_NM, _DEFAULT_STANDARD_TEMPERATURE_C)


@dataclass
class CalibrationState:
    """The session-level SLS calibration.

    The user supplies one calibrant point (intensity, angle, standard liquid +
    geometry); the controller computes k_c. A None k_c means uncalibrated -- the
    analysis still runs, flagged. Mirrors the unified model in sls.py.

    The standard's refractive index is PROPOSED from the solvent library at the
    standard's own wavelength and temperature and re-proposed when either changes
    (`Controller.set_calibration_field`), until the user types a value of their own;
    a blank leaves the (n_solvent/n_standard)^2 step unapplied and the run flagged
    uncalibrated (`analysis.sls.compute_excess_rayleigh_ratio`). WHICH of the two it
    is rides the `standard_refractive_index_source` sidecar -- 'library' (proposed,
    tracks lambda / T, shown with the provenance dot) or 'user' (typed, never
    overwritten) -- the `autofill_solvent_props` pattern the Data tab uses for the
    sample's own n (owner-ruled 2026-08-27). A stored tag, never an
    inference from the value: inferring "still the library's" from value equality
    re-proposed away a value the user had typed because it happened to equal the
    library's, and froze tracking for good once the library had no value at a
    condition. Clearing the field returns it to the library (the only way back).
    Every note a calibration owes the user -- temperature or wavelength
    extrapolation of the toluene table, a blank or off-library standard index -- is
    composed by `Controller.preview_calibration_note` from the `*_note` methods here.
    """
    calibrant_intensity: Optional[float] = None
    calibrant_angle_deg: float = 90.0
    standard_name: str = 'toluene'
    standard_geometry: str = 'VU'
    standard_wavelength_nm: float = _DEFAULT_STANDARD_WAVELENGTH_NM
    standard_temperature_C: float = _DEFAULT_STANDARD_TEMPERATURE_C
    standard_refractive_index: Optional[float] = _DEFAULT_STANDARD_REFRACTIVE_INDEX
    standard_refractive_index_source: str = 'library'    # 'library' | 'user'
    dark_count_rate: float = 0.0
    depolarization_ratio_v: Optional[float] = None
    k_c: Optional[float] = None                          # computed; editable

    def library_standard_index(self) -> Optional[float]:
        """The library's toluene n at THIS standard's wavelength / temperature, or None."""
        return _library_standard_index(self.standard_wavelength_nm,
                                       self.standard_temperature_C)

    def compute_k_c(self) -> Optional[float]:
        """Compute k_c from the calibrant point, or None if not enough info."""
        if self.calibrant_intensity is None:
            return None
        r_std = phys.rayleigh_ratio_toluene(
            self.standard_wavelength_nm, self.standard_temperature_C,
            geometry=self.standard_geometry,
            depolarization_ratio_v=self.depolarization_ratio_v)
        return sls_engine.compute_calibration_constant(
            self.calibrant_intensity, self.calibrant_angle_deg, r_std,
            dark_count_rate=self.dark_count_rate)

    def temperature_extrapolation_note(self) -> Optional[str]:
        """The toluene Rayleigh-ratio extrapolation note for this standard temperature,
        or None if it is inside the S&K 10-50 C table. Keys purely off the standard
        temperature (the standard is toluene throughout, and compute_k_c always calls
        rayleigh_ratio_toluene), so it surfaces even before a calibrant intensity is
        entered. The GUI renders it next to k_c."""
        return phys.toluene_temperature_extrapolation_reason(self.standard_temperature_C)

    def wavelength_extrapolation_note(self) -> Optional[str]:
        """The wavelength sibling of `temperature_extrapolation_note` (rule 10): the
        toluene table's nearest entry was served unscaled for a wavelength within the
        match tolerance but not on the table. None on a tabulated wavelength, and None
        beyond the tolerance (there `compute_k_c` raises, and the k_c line says so)."""
        try:
            return phys.toluene_wavelength_extrapolation_reason(
                float(self.standard_wavelength_nm))
        except (TypeError, ValueError):
            return None

    def standard_index_note(self) -> Optional[str]:
        """The disclosure the standard's refractive index owes, or None.

        Two cases, both invariant-3 shaped (the library PROPOSES, the user commits, and
        the program says what it will do with what was committed):
          * BLANK -- the mandatory (n_solvent/n_standard)^2 step cannot run, so the
            results will be flagged uncalibrated; the note says so before the run,
            beside k_c, because the k_c line alone would show a healthy constant.
          * OFF-LIBRARY -- the committed value sits outside the library's own
            uncertainty band for toluene at this wavelength / temperature; a user's
            measured value is theirs to keep, and this only names the difference and
            its size on Mw (the correction goes as n_standard^-2, so ~2 dn/n).
        None when the library has no value here, or the value agrees with it.
        """
        lam, t_c = self.standard_wavelength_nm, self.standard_temperature_C
        lib = self.library_standard_index()
        if self.standard_refractive_index is None:
            hint = f" (solvent library: toluene {lib:.4f})" if lib is not None else ""
            return (
                f"Standard refractive index is blank: the (n_solvent/n_standard)^2 "
                f"correction cannot be applied, so the results will be flagged "
                f"UNCALIBRATED -- Mw and absolute A2 unreliable; Rg and the "
                f"calibration-free 2 A2 Mw unaffected. Enter the standard's index at "
                f"{lam:g} nm / {t_c:g} C{hint}.")
        if lib is None:
            return None
        try:
            sigma = float(solvents_lib.solvent_uncertainty_n('toluene', float(lam), float(t_c)))
            n_std = float(self.standard_refractive_index)
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(n_std) and math.isfinite(sigma)):
            return None
        if abs(n_std - lib) <= sigma:
            return None
        return (
            f"Standard refractive index {n_std:g} differs from the solvent library's "
            f"toluene value {lib:.4f} at {lam:g} nm / {t_c:g} C (library band "
            f"+/- {sigma:.1e}); the (n_solvent/n_standard)^2 correction, and with it "
            f"Mw and absolute A2, moves by about {200.0 * abs(n_std - lib) / lib:.2f} %.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            'calibrant_intensity': self.calibrant_intensity,
            'calibrant_angle_deg': self.calibrant_angle_deg,
            'standard_name': self.standard_name,
            'standard_geometry': self.standard_geometry,
            'standard_wavelength_nm': self.standard_wavelength_nm,
            'standard_temperature_C': self.standard_temperature_C,
            'standard_refractive_index': self.standard_refractive_index,
            'standard_refractive_index_source': self.standard_refractive_index_source,
            'dark_count_rate': self.dark_count_rate,
            'depolarization_ratio_v': self.depolarization_ratio_v,
            'k_c': self.k_c,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CalibrationState':
        # Forward-compatible: drop unknown keys so a session written by a newer build
        # (extra calibration field) still loads in an older one.
        if not d:
            return cls()
        known = {f.name for f in fields(cls)}
        state = cls(**{k: v for k, v in d.items() if k in known})
        if 'standard_refractive_index_source' not in d:
            # A session written before the sidecar existed: tag it ONCE, at load, by
            # the value-equality inference the live edit path used to make on every
            # edit -- the library's value at the stored condition reads as proposed,
            # anything else (a typed value, a blank) as the user's. Conservative in
            # the direction that matters: a value this cannot place is never
            # re-proposed away.
            lib = state.library_standard_index()
            n_std = state.standard_refractive_index
            try:
                # Guarded like `_library_standard_index`'s own float() calls: a
                # hand-edited or corrupted session can hold a non-numeric value here,
                # and before the tag existed `from_dict` never converted it. Raising
                # would surface through `load_session` as the version-mismatch message,
                # which names the wrong cause. A value that is not a number is not the
                # library's, so it degrades to 'user' -- the branch that never
                # overwrites it.
                proposed = (lib is not None and n_std is not None
                            and math.isclose(float(n_std), lib, rel_tol=1e-9))
            except (TypeError, ValueError):
                proposed = False
            state.standard_refractive_index_source = 'library' if proposed else 'user'
        return state


# ===========================================================================
# Per-sample SLS data mask
# ===========================================================================

def _close(a: float, b: float, rel: float = 1e-6, abs_tol: float = 1e-9) -> bool:
    """Tolerant float match for angles/concentrations (they originate from the
    same parsed values, but compare with a tolerance to be safe)."""
    return abs(a - b) <= max(abs_tol, rel * max(abs(a), abs(b)))


@dataclass
class SLSMask:
    """A non-destructive SLS data mask for one sample: hidden whole angles, whole
    concentrations, and individual (concentration, angle) points. Applied as
    exclusions before analysis; the loaded data is never altered. Persisted in the
    session so masking decisions survive a save/reload.
    """
    masked_angles: set = field(default_factory=set)
    masked_concentrations: set = field(default_factory=set)
    masked_points: set = field(default_factory=set)   # {(concentration, angle)}

    def is_empty(self) -> bool:
        return not (self.masked_angles or self.masked_concentrations
                    or self.masked_points)

    def is_concentration_masked(self, c: float) -> bool:
        return any(_close(c, mc) for mc in self.masked_concentrations)

    def is_angle_masked(self, angle: float) -> bool:
        return any(_close(angle, ma) for ma in self.masked_angles)

    def is_point_masked(self, c: float, angle: float) -> bool:
        return any(_close(c, pc) and _close(angle, pa)
                   for (pc, pa) in self.masked_points)

    def is_masked(self, c: float, angle: float) -> bool:
        """True if this (concentration, angle) datum is excluded for any reason."""
        return (self.is_angle_masked(angle) or self.is_concentration_masked(c)
                or self.is_point_masked(c, angle))

    def to_dict(self) -> Dict[str, Any]:
        return {
            'masked_angles': sorted(self.masked_angles),
            'masked_concentrations': sorted(self.masked_concentrations),
            'masked_points': [list(p) for p in sorted(self.masked_points)],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SLSMask':
        return cls(
            masked_angles=set(d.get('masked_angles', [])),
            masked_concentrations=set(d.get('masked_concentrations', [])),
            masked_points={tuple(p) for p in d.get('masked_points', [])},
        )


@dataclass
class ModeFamilyPicks:
    """One user-picked mode family on a sample's mode map.

    `picks` is VALUE-keyed -- {(angle_deg, gamma_s_inv)} -- never a grid or
    point index (the snapshot ruling, owner 2026-08-14): membership tests go
    through `_close` on BOTH coordinates, so a re-extraction that moves a
    mode's decay rate ORPHANS the pick (it matches nothing, which the family
    fit notes) rather than silently re-aiming it at a different mode.

    `polarization_state` is the state this family is bound to ('VV' | 'VH' |
    None = the one unpolarized class), fixed by its first pick: the mode map
    facets by polarization (doctrine rule 1), so a family can never span
    states. Because None is a REAL member of that closed set, binding is
    carried by the explicit `bound` flag, never inferred from the state field
    -- `x is not None` was unarmed for exactly the member spelled None.
    `role` designates the family for the
    fast/slow amplitude report ('' | 'fast' | 'slow') -- a display
    designation, not data.
    """
    label: str
    picks: set = field(default_factory=set)     # {(angle_deg, gamma_s_inv)}
    polarization_state: Optional[str] = None
    role: str = ''                              # '' | 'fast' | 'slow'
    bound: bool = False                         # True once the first pick fixed the state

    def is_empty(self) -> bool:
        return not self.picks

    def has_pick(self, angle_deg: float, gamma_s_inv: float) -> bool:
        return any(_close(angle_deg, pa) and _close(gamma_s_inv, pg)
                   for (pa, pg) in self.picks)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'label': self.label,
            'picks': [list(p) for p in sorted(self.picks)],
            'polarization_state': self.polarization_state,
            'role': self.role,
            'bound': self.bound,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ModeFamilyPicks':
        picks = {tuple(p) for p in d.get('picks', [])}
        return cls(
            label=d.get('label', ''),
            picks=picks,
            polarization_state=d.get('polarization_state'),
            role=d.get('role', ''),
            # Absent-tolerant legacy inference: a pre-`bound` session's family
            # with picks WAS bound (binding happens at first pick), so infer it
            # from the picks rather than defaulting a loaded family to unbound
            # -- which would make it match nothing forever.
            bound=bool(d.get('bound', bool(picks))),
        )


@dataclass
class ModeMapPicks:
    """Every mode family the user has picked on one (sample, fraction) map.

    `families` is ordered (creation order = the stable colour/marker index the
    GUI assigns). `source_method` records which extraction source the picks
    were made against -- provenance, not a filter.

    `manual_bounds` is per-run user state for the DISTRIBUTION sources:
    item_id -> [(lo_rh_nm, hi_rh_nm), ...] integration intervals replacing the
    automatic lobe split for that run (validated at set time; recorded on the
    extracted points as 'manual' provenance). Ignored by the parametric
    'double' source, whose amplitudes come from the model.
    """
    families: list = field(default_factory=list)    # List[ModeFamilyPicks]
    source_method: str = 'double'
    manual_bounds: dict = field(default_factory=dict)   # item_id -> [(lo, hi), ...]

    def is_empty(self) -> bool:
        return not (self.families or self.manual_bounds)

    def family(self, label: str) -> Optional[ModeFamilyPicks]:
        for f in self.families:
            if f.label == label:
                return f
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'families': [f.to_dict() for f in self.families],
            'source_method': self.source_method,
            'manual_bounds': {iid: [list(b) for b in bs]
                              for iid, bs in self.manual_bounds.items()},
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ModeMapPicks':
        return cls(
            families=[ModeFamilyPicks.from_dict(f)
                      for f in d.get('families', [])],
            source_method=d.get('source_method', 'double'),
            manual_bounds={iid: [tuple(b) for b in bs]
                           for iid, bs in d.get('manual_bounds', {}).items()},
        )


def _nan_to_none(x: Optional[float]) -> Optional[float]:
    """JSON-safe float: the session writer sanitizes non-finite floats to null
    anyway (allow_nan=False), so serialize NaN as None deliberately rather than
    relying on the sanitizer, and read None back as NaN."""
    return None if (x is None or not math.isfinite(x)) else float(x)


def _none_to_nan(x: Optional[float]) -> float:
    return float('nan') if x is None else float(x)


# The unconditional per-mode Mw/A2 disclosures (owner-ruled 2026-08-14: the
# program cannot classify which mode is chain vs domain -- don't police,
# disclose). Appended on EVERY per-mode Debye/Guinier/Zimm/Berry result;
# they travel the ordinary notes -> _analysis_notes -> export Comments path.
_MODE_RESOLVED_MW_NOTE = (
    'Mode-resolved result: computed from one mode\'s share of the excess '
    'intensity. Its Kc/dR ordinate uses the TOTAL concentration and the '
    'chain dn/dc, while the mode\'s own mass concentration and contrast are '
    'unknown -- for a mode that is not the chains themselves (solvent '
    'structure, domains), that contrast is an unverifiable assumption and an '
    'Mw or A2 computed with it is unanchored, not just noisy. Treat Mw/A2 as '
    'apparent-class; see the Theory-and-Equations-Guide.')
_MODE_RESOLVED_CONDITIONS_NOTE = (
    'For charged systems, absolute Mw and A2 additionally require '
    'dialysis-equilibrium (Donnan) conditions; near the theta point a '
    'resolvable slow mode itself signals contamination of Kc/dR.')
# Invariant 8: the same exclusion statement the amplitude-Guinier fitter makes
# for its own SE — the Debye/Zimm branches run on the SAME partitioned series,
# so their +- inherit the identical alpha/bounds systematic.
_MODE_RESOLVED_SE_NOTE = (
    'Any +- on this result is statistical only and additionally excludes the '
    'systematics of the inversion that produced the mode amplitudes '
    '(regularization, integration bounds).')


@dataclass(frozen=True)
class RecordedModePoint:
    """One amplitude datum of one recorded mode -- a snapshot, never live state.

    Frozen (the structural half of the snapshot ruling, owner 2026-08-14):
    recording copies the VALUES off the mode map at record time -- amplitude,
    decay rate, the source run's committed concentration, the inversion's
    regularization, size grid and residual weighting, and the integration
    bounds -- so a later
    re-fit, re-pick or parameter edit cannot re-aim what a partition already
    consumed. `item_id`
    is provenance and MAY dangle after the source run is deleted; the snapshot
    survives and its provenance line says so (never cascade-deleted).
    """
    angle_deg: float
    gamma_s_inv: float
    amplitude_fraction: float
    concentration_g_per_mL: Optional[float]   # the run's committed value at record time
    item_id: str
    low_amplitude: bool
    amplitude_floor: float
    # The lag-window verdict, snapshotted with everything else (see ModePoint). A session
    # written before this flag existed reads back `decay_incomplete=False` with a NaN
    # completeness -- unflagged because no verdict was ever reached on it, not because one
    # was reached and passed. NaN is what the criterion itself returns for "unanswerable".
    decay_incomplete: bool = False
    decay_completeness: float = float('nan')
    min_decay_completeness: float = MIN_DECAY_COMPLETENESS
    alpha: Optional[float] = None              # CONTIN regularization (None otherwise)
    alpha_selection_method: str = ''           # 'gcv'|'lcurve'|'ftest'|'user'|''
    # The robust-GCV gamma beside a 'gcv' token ('gcv' names a family, so the pair is the
    # provenance). None when the selector was not GCV, and on a session written before the
    # field existed -- "this record names no gamma", not a claim about which functional ran.
    gcv_gamma: Optional[float] = None
    rh_min_nm: float = float('nan')            # the size grid the inversion ran on
    rh_max_nm: float = float('nan')
    n_grid: int = 0
    # Which residual weight built the inversion these amplitudes came from, and what that
    # weight read (`ModePoint`'s three, snapshotted with everything else). A session
    # written before they existed reads back '' on all three -- "this record names no
    # weighting form", which is the truth about it, not a claim that the older weight ran.
    weighting: str = ''
    sample_time_provenance: str = ''           # 'instrument' | 'axis' | ''
    run_duration_form: str = ''                # 'finite' | 'infinite' | ''
    bounds_lo_rh_nm: float = float('nan')      # integration lobe (distribution sources)
    bounds_hi_rh_nm: float = float('nan')
    bounds_source: str = ''                    # 'auto' | 'manual' | ''

    def to_dict(self) -> Dict[str, Any]:
        return {
            'angle_deg': self.angle_deg,
            'gamma_s_inv': self.gamma_s_inv,
            'amplitude_fraction': self.amplitude_fraction,
            'concentration_g_per_mL': self.concentration_g_per_mL,
            'item_id': self.item_id,
            'low_amplitude': self.low_amplitude,
            'amplitude_floor': self.amplitude_floor,
            'decay_incomplete': self.decay_incomplete,
            'decay_completeness': _nan_to_none(self.decay_completeness),
            'min_decay_completeness': self.min_decay_completeness,
            'alpha': self.alpha,
            'alpha_selection_method': self.alpha_selection_method,
            'gcv_gamma': self.gcv_gamma,
            'rh_min_nm': _nan_to_none(self.rh_min_nm),
            'rh_max_nm': _nan_to_none(self.rh_max_nm),
            'n_grid': self.n_grid,
            'weighting': self.weighting,
            'sample_time_provenance': self.sample_time_provenance,
            'run_duration_form': self.run_duration_form,
            'bounds_lo_rh_nm': _nan_to_none(self.bounds_lo_rh_nm),
            'bounds_hi_rh_nm': _nan_to_none(self.bounds_hi_rh_nm),
            'bounds_source': self.bounds_source,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'RecordedModePoint':
        return cls(
            angle_deg=float(d['angle_deg']),
            gamma_s_inv=float(d['gamma_s_inv']),
            amplitude_fraction=float(d['amplitude_fraction']),
            concentration_g_per_mL=d.get('concentration_g_per_mL'),
            item_id=d.get('item_id', ''),
            low_amplitude=bool(d.get('low_amplitude', False)),
            # The engine's own floor, never a second copy of it: a session written
            # before the field existed reloads at the floor the extraction used.
            amplitude_floor=float(d.get('amplitude_floor', DEFAULT_AMPLITUDE_FLOOR)),
            decay_incomplete=bool(d.get('decay_incomplete', False)),
            decay_completeness=_none_to_nan(d.get('decay_completeness')),
            min_decay_completeness=float(
                d.get('min_decay_completeness', MIN_DECAY_COMPLETENESS)),
            alpha=d.get('alpha'),
            alpha_selection_method=d.get('alpha_selection_method', ''),
            gcv_gamma=d.get('gcv_gamma'),
            rh_min_nm=_none_to_nan(d.get('rh_min_nm')),
            rh_max_nm=_none_to_nan(d.get('rh_max_nm')),
            n_grid=int(d.get('n_grid', 0)),
            weighting=d.get('weighting', ''),
            sample_time_provenance=d.get('sample_time_provenance', ''),
            run_duration_form=d.get('run_duration_form', ''),
            bounds_lo_rh_nm=_none_to_nan(d.get('bounds_lo_rh_nm')),
            bounds_hi_rh_nm=_none_to_nan(d.get('bounds_hi_rh_nm')),
            bounds_source=d.get('bounds_source', ''),
        )


@dataclass(frozen=True)
class RecordedMode:
    """One mode family inside a recorded set: label, role, frozen points."""
    label: str
    role: str                                  # '' | 'fast' | 'slow' (copied at record time)
    points: Tuple[RecordedModePoint, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {'label': self.label, 'role': self.role,
                'points': [p.to_dict() for p in self.points]}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'RecordedMode':
        return cls(label=d.get('label', ''), role=d.get('role', ''),
                   points=tuple(RecordedModePoint.from_dict(p)
                                for p in d.get('points', [])))


@dataclass(frozen=True)
class RecordedModeSet:
    """A recorded mode decomposition -- the snapshot the SLS side consumes
    read-only (selection flow (a), owner-ruled 2026-08-14: the DLS side owns
    mode definition; this object is what crosses the tab boundary).

    It never auto-tracks upstream changes: re-running a fit or editing a mask
    does not touch it, and re-recording is a deliberate user action producing
    a NEW entry. `sample_id`/`fraction` are deliberately NOT inside -- the
    store key is their single home (the SLSMask precedent), so a regroup
    carry cannot leave a stale copy disagreeing with the key.
    """
    set_id: str                                # unique, never reused
    label: str
    source_method: str                         # 'double' | 'contin' | 'nnls'
    polarization_state: Optional[str]          # ONE state (record refuses a mix)
    recorded: str                              # ISO-8601 date string, stamped at record
    modes: Tuple[RecordedMode, ...]
    notes: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'set_id': self.set_id, 'label': self.label,
            'source_method': self.source_method,
            'polarization_state': self.polarization_state,
            'recorded': self.recorded,
            'modes': [m.to_dict() for m in self.modes],
            'notes': list(self.notes),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'RecordedModeSet':
        return cls(
            set_id=d.get('set_id', ''),
            label=d.get('label', ''),
            source_method=d.get('source_method', 'double'),
            polarization_state=d.get('polarization_state'),
            recorded=d.get('recorded', ''),
            modes=tuple(RecordedMode.from_dict(m) for m in d.get('modes', [])),
            notes=tuple(d.get('notes', [])),
        )


@dataclass
class SampleRho:
    """The shape parameter rho = Rg/Rh for one sample, with full provenance.

    Bundles the underlying RhoResult value with the plain-language source labels
    and the apparent/thermodynamic status of each input. `is_apparent` is True if
    EITHER Rg or Rh is apparent -- rho is only a thermodynamic shape parameter
    when both inputs are infinite-dilution values (Rg from a Zimm/Berry
    extrapolation, Rh from a c->0 extrapolation).
    """
    sample_id: str
    rho: float
    rg_nm: float
    rh_nm: float
    rg_label: str
    rh_label: str
    rg_source: str               # 'computed' | 'picked' | 'user'
    rh_source: str               # 'computed' | 'picked' | 'user'
    is_apparent: bool
    interpretation: str
    shape: str = ''              # concise architecture label (e.g. 'random coil')
    rho_se: Optional[float] = None  # statistical SE (only when both Rg and Rh have one)
    se_estimator: Optional[str] = None  # covariance estimator behind rho_se (Rg/Rh
    #                                     regression SEs); None when rho_se is None
    rg_se: Optional[float] = None   # regression SE of Rg (None if Rg has no honest ±),
    rh_se: Optional[float] = None   # and of Rh — carried so the display can round each
    #                                 value to the place its own σ supports (invariant 8)
    rg_model_dependent: bool = False  # True = Rg came from a form-factor P(q) fit (assumes a
    #                                   shape) -> the display flags it (neutral ⓘ); orthogonal to
    #                                   is_apparent. Rh is always model-agnostic (DLS).
    rg_model: str = ''              # the assumed shape when rg_model_dependent (e.g. 'sphere')
    rg_degenerate: bool = False     # True = Rg rests on a user-selected DEGENERATE form-factor
    #                                 solution (the data can't distinguish the shape); always implies
    #                                 rg_model_dependent, disclosed one tier stronger (review-01; rule 10)
    # Passive ⓘ notices from the underlying RhoResult (e.g. Rg/Rh sample keys disagree).
    # Structurally empty in the cross-sample path (Rg and Rh come from ONE SampleResult,
    # so no keys are compared) — carried for a uniform surface + future cross-sample
    # pairing. Same text also emitted via warnings.warn (keep-both).
    notes: tuple = ()


@dataclass
class ScalingData:
    """Points + power-law fit for a cross-sample scaling plot (Rg-Mw or A2-Mw).

    One point per included sample: its effective Mw (a user Mw wins) against Rg or
    A2. The fit is a log-log regression -> the exponent (Rg ~ Mw^nu, A2 ~ Mw^-a).
    Parallel lists (sample_ids/labels/mw/y) are the plotted points, in order.
    """
    quantity: str                    # 'rg' or 'a2'
    fit: ScalingResult
    sample_ids: List[str]
    labels: List[str]
    mw: List[float]
    y: List[float]
    # A plotted VALUE is on an arbitrary scale: the Mw (every plot's X axis) or, on the a2
    # axis, the A2 itself -- the one definition lives in `_result_arbitrary_scale`. Named for
    # the CONDITION, not for Mw: unlike its ArchitectureData namesake below, this flag has two
    # possible sources, so the two dataclasses deliberately STOP mirroring here.
    any_arbitrary_scale: bool
    n_excluded: int                  # samples dropped (no Mw, or non-positive y)
    # Branched-tagged points: plotted (so the contraction is visible) but held OUT of
    # the linear power-law fit (a branched chain would bias the linear scaling law).
    # Parallel lists, same shape as the fitted points above.
    branched_mw: List[float] = field(default_factory=list)
    branched_y: List[float] = field(default_factory=list)
    branched_labels: List[str] = field(default_factory=list)
    # A plotted point drew a MODEL-DEPENDENT source: a form-factor Mw (any quantity) or, for
    # the rg axis, a form-factor Rg. The exponent then rests on a shape assumption -> flagged
    # (neutral) in the scaling note + export, mirroring any_arbitrary_scale (honesty axis).
    any_model_dependent: bool = False
    # A plotted point drew a DEGENERATE form-factor source (a user-selected solution the data can't
    # distinguish) -> the exponent rests on an ambiguity the user resolved, flagged one tier stronger
    # than model_dependent alone (review-01; rule-10 sibling of any_model_dependent).
    any_degenerate: bool = False
    # Per-point provenance, parallel to the lists above: the plain-language label of the Mw
    # and of the plotted y value, as `_scaling_point_source` composes them. Carried so the
    # scaling EXPORT can name where each point came from, the `export_architecture` Rg
    # source / Rh source pattern applied to the axes this file actually plots -- which is
    # what makes an apparent single-angle-ladder Mw or A2 name its angle in the file rather
    # than only on screen.
    sources: List[str] = field(default_factory=list)
    branched_sources: List[str] = field(default_factory=list)


@dataclass
class ArchitectureRow:
    """One branched (sample, Mw) architecture readout: the scattering factor g (from
    Rg, if the sample carries one) and/or the hydrodynamic factor h (from Rh). Either
    may be None when that radius is absent -- g needs Rg, h needs Rh."""
    sample_id: str
    label: str
    mw: float
    g_result: Optional['util.ArchitectureResult'] = None
    h_result: Optional['util.HydrodynamicResult'] = None
    compare_note: str = ''            # interpretive g-vs-h note (util.compare_g_h)


@dataclass
class ArchitectureData:
    """Branching contraction factors g (Rg) and h (Rh) over the tagged samples + the
    linear reference laws.

    `rg_law` / `rh_law` are the linear-only Rg-Mw and Rh-Mw power laws fit to the
    samples tagged Linear (chain_type; route 2), each independent (a sample can feed
    one and not the other). `rows` is one ArchitectureRow per branched (sample, Mw).
    `message` is a non-empty guidance string when nothing is computable yet (too few
    linear points, or no branched sample tagged).
    """
    rg_law: 'ScalingResult'
    rh_law: 'ScalingResult'
    rg_mw_range: Optional[tuple]     # (min, max) Mw of the linear Rg series; None if no fit
    rh_mw_range: Optional[tuple]     # (min, max) Mw of the linear Rh series; None if no fit
    n_linear_rg: int                 # linear Rg points fitted
    n_linear_rh: int                 # linear Rh points fitted
    rows: List['ArchitectureRow']    # per branched (sample, Mw)
    # Mw-specific by nature, and deliberately NOT renamed alongside ScalingData's
    # `any_arbitrary_scale`: the g/h factors and the reference laws have no A2 axis, so Mw is
    # the only scale-bearing input here and naming it is the more precise caveat.
    any_uncalibrated_mw: bool = False  # a contributing Mw came from an uncalibrated run
    # A contributing Mw (or, for the Rg law/g factor, an Rg) came from a form-factor P(q) fit,
    # so a g/h factor or reference law rests on a particle-shape assumption -> flagged (neutral)
    # in the architecture export, mirroring any_uncalibrated_mw (model-dependence axis).
    any_model_dependent: bool = False
    # A contributing Rg/Mw rests on a user-selected DEGENERATE form-factor solution -> the g/h factor
    # or reference law rests on an ambiguity the user resolved (review-01; rule-10 sibling).
    any_degenerate: bool = False
    message: str = ''

    def row_for(self, sample_id: str, mw: Optional[float]) -> Optional['ArchitectureRow']:
        """The ArchitectureRow matching (sample_id, Mw), or None -- lets the GUI join
        g/h onto its per-(sample, fraction) ρ rows. O(1) via a lazily-built index (the
        GUI calls this once per table row, so a linear scan would be quadratic)."""
        if mw is None or not np.isfinite(mw):
            return None
        idx = self.__dict__.get('_row_index')
        if idx is None:
            idx = {(r.sample_id, round(r.mw, 6)): r for r in self.rows}
            self.__dict__['_row_index'] = idx
        return idx.get((sample_id, round(float(mw), 6)))


# The one-key object a non-finite float is written as in a session file, and the three
# tokens it may carry. An OBJECT, not a bare string: no persisted dataclass field holds a
# dict with this key, so the reader can never mistake a user's own text (a polymer
# literally named "NaN") for a number, while strict JSON stays strict.
_NONFINITE_KEY = '__nonfinite__'
_NONFINITE_TOKENS = {'nan': float('nan'), 'inf': float('inf'), '-inf': float('-inf')}


def _json_sanitize(obj):
    """Recursively replace non-finite floats (NaN / ±Inf) with the ``_NONFINITE_KEY``
    sentinel object so a session serializes to strict, portable JSON
    (``allow_nan=False``) rather than the non-standard ``NaN``/``Infinity`` tokens
    strict/external readers reject. ``_json_restore`` is the exact inverse on load.

    Why a sentinel and not ``None``: ``None`` in a result field
    means "never computed", a NaN means "computed, no finite answer" -- ``run_zimm``
    stores a NaN Rg from a non-positive angular slope beside a live ``rg_source`` and
    label. Writing both as ``null`` made the same regroup gesture store a different Rg
    before and after a save/reload, because the merge branches on presence. A stored
    value must come back as what it was. Sessions written before the sentinel hold
    ``null`` there and load as ``None`` (readers treat the two alike -- the merge now
    included). ``np.float64`` is a ``float`` subclass, so the ``float`` check covers it."""
    if isinstance(obj, float):
        if math.isfinite(obj):
            return obj
        return {_NONFINITE_KEY: ('nan' if math.isnan(obj)
                                 else 'inf' if obj > 0 else '-inf')}
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    return obj


def _json_restore(obj):
    """The inverse of ``_json_sanitize``: turn every ``_NONFINITE_KEY`` sentinel object
    back into its float, recursively, before the payload reaches any ``from_dict``.
    Only an object holding exactly that one key with one of the three tokens is
    restored; anything else passes through untouched, so a pre-sentinel session
    (``null`` for NaN) loads exactly as it did."""
    if isinstance(obj, dict):
        if (len(obj) == 1 and _NONFINITE_KEY in obj
                and obj[_NONFINITE_KEY] in _NONFINITE_TOKENS):
            return _NONFINITE_TOKENS[obj[_NONFINITE_KEY]]
        return {k: _json_restore(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_restore(v) for v in obj]
    return obj


# ===========================================================================
# Controller
# ===========================================================================


class Controller:
    """Owns the workspace and orchestrates everything the GUI asks for."""

    def __init__(self) -> None:
        self.workspace = Workspace()
        # Global user preferences (seed defaults + appearance), loaded from the
        # settings.json at the repo root (or defaults if absent). "Seed, never
        # override": GUI controls initialize from these, analysis methods fall back
        # to them when the caller passes None; the per-run value still wins.
        # load_with_report also returns any invalid saved settings that were reverted to
        # defaults, so the GUI can tell the user at startup (the controller stays Qt-free and
        # holds only the plain list). Empty on the ordinary absent/valid-file cases.
        self.settings, self.settings_load_problems = SettingsState.load_with_report()
        # Session-wide calibration (the default for every sample). Its geometry is
        # SEEDED from settings -- i.e. it is the starting value for a fresh session;
        # the user still edits it in the SLS calibration panel per their instrument.
        self.calibration_working = CalibrationState(
            standard_geometry=self.settings.standard_geometry)
        self.calibration_committed = CalibrationState(
            standard_geometry=self.settings.standard_geometry)
        # Optional per-sample calibration overrides, keyed by sample_id. A sample
        # with an entry uses it; otherwise it falls back to the session default.
        self.sample_calibration_working: Dict[str, CalibrationState] = {}
        self.sample_calibration_committed: Dict[str, CalibrationState] = {}
        # Per-sample SLS data masks (live analysis filters; persisted in sessions).
        self.sls_masks: Dict[str, SLSMask] = {}
        # Sample ids whose user confirmed running SLS with NO MEASURED solvent blank.
        # DELIBERATELY TRANSIENT -- it is an acknowledgement of a disclosure, not data,
        # so it is never written to a session and a reloaded session asks again. Empty
        # by default, which is what keeps `_rayleigh_for_sample` refusing for every
        # headless caller and every existing test.
        self._unmeasured_blank_ack: Set[str] = set()
        # Per-(sample, fraction) mode-map family picks (user state; persisted in
        # sessions exactly like the masks). Value-keyed -- see ModeFamilyPicks.
        self.mode_picks: Dict[tuple, ModeMapPicks] = {}
        # Recorded mode sets -- frozen snapshots-with-provenance the SLS side
        # consumes read-only (selection flow (a)). Keyed like the picks;
        # creation-ordered lists; persisted in sessions; carried through
        # regroups; NEVER auto-updated by a re-fit (the snapshot ruling).
        self.mode_sets: Dict[tuple, List[RecordedModeSet]] = {}
        # last analysis results, keyed for the GUI to fetch and plot
        self.results: Dict[str, Any] = {}
        # The sls_run_signature each stored form-factor fit was computed under, keyed exactly
        # like its self.results entry ('form_factor', sid, model, frac). The Cross-Sample
        # candidate builders READ the stored form-factor fit (never re-fit on the GUI thread)
        # and offer it only while FRESH -- its stamped signature still equals the current one.
        # See run_form_factor (stamps it), _form_factor_solutions (gates on it).
        self._form_factor_run_sig: Dict[tuple, tuple] = {}
        # The DLS analogue, for the user's Γ-vs-q² / D-vs-c runs: the `_dls_run_stamp`
        # each stored run was computed under, keyed like its self.results entry
        # (('gamma_q2' | 'conc_extrap'), sid, frac). The Rh candidate builder READS a fresh
        # stored run -- the fit over the user's ticked points, under their Settings, with
        # its flags -- instead of re-deriving one that ignores all three. See
        # run_gamma_q2 (stamps it), _stored_dls_run (gates on it).
        self._dls_run_sig: Dict[tuple, tuple] = {}
        # The same store holds each cumulant fit's stamp under ('cumulants', item_id), so
        # tier 1 promotes the user's own fresh fit (`_stored_cumulant_fit`); the purges
        # reach it by its key. The fit number inside the stamp is never reused, like
        # `_distribution_stamp_counter`, so a re-Run that moves no input still misses the memo.
        self._cumulant_fit_counter = 0
        # Memoized SLS candidate/A2 computations for the Cross-Sample tab, keyed by the
        # full sls_run_signature: refresh() re-runs the same
        # Zimm/Debye/Guinier fits 3-5x per fraction (auto-select + source panel), so this
        # collapses them to one until an input changes. See _sls_cached.
        self._sls_candidate_cache: Dict[tuple, Any] = {}
        # Memoized DLS Rh-candidate computations for the Cross-Sample tab, keyed by the
        # full dls_run_signature (code-review C2): the same auto-select + source-panel
        # refresh that hammers the SLS cache also calls dls_rh_candidates 3-5x per
        # fraction, each re-running a cumulant fit per measurement + a Γ-q² per
        # concentration + a D-vs-c per angle. See _dls_cached / dls_run_signature.
        self._dls_candidate_cache: Dict[tuple, Any] = {}
        # Monotonic version stamps for stored distribution results (CONTIN/NNLS), keyed
        # by their results key ('distribution', item_id, method). dls_run_signature uses
        # the stamp — NOT id(res) — to detect a re-fit: a freed result's id can be
        # reused by a later re-run (CPython ABA), which would let a stale Cross-Sample
        # signature falsely match and serve outdated peak candidates. A strictly
        # increasing counter is never reused, so a re-run always moves the signature.
        self._distribution_stamp_counter = 0
        self._distribution_stamps: Dict[tuple, int] = {}
        # Undo history: a stack of committed-state snapshots, one pushed before each
        # Update that actually changes something. `undo` steps back through these
        # previously-applied parameter sets.
        self._commit_history: List[Dict[str, Any]] = []

    # ----------------------------------------------------------------------
    # Loading measurements
    # ----------------------------------------------------------------------
    def add_loaded(
        self, kind: str, raw: Dict[str, List[float]], params: Dict[str, Any],
        source_path: Optional[str] = None,
    ) -> str:
        """Register a parsed measurement. `params` seeds both working & committed.

        The GUI's confirmation table edits the working set afterwards; nothing is
        analyzed until commit. Returns the new item_id.
        """
        item_id = self.workspace.new_item_id()
        lm = LoadedMeasurement(
            item_id=item_id, kind=kind, raw=raw,
            working_params=dict(params), committed_params=dict(params),
            source_path=source_path)
        self.workspace.add_measurement(lm)
        self._rekey_sample_stores(self.workspace.regroup())
        return item_id

    # ----------------------------------------------------------------------
    # Parameter editing (working set) + change tracking
    # ----------------------------------------------------------------------
    def set_param(self, item_id: str, key: str, value: Any) -> None:
        """Edit one working parameter of a measurement (does not run anything)."""
        lm = self.workspace.measurements[item_id]
        allowed = _DLS_PARAM_KEYS if lm.kind == 'dls' else _SLS_PARAM_KEYS
        if key not in allowed:
            raise KeyError(f"{key!r} is not an editable parameter for a "
                           f"{lm.kind} measurement.")
        lm.working_params[key] = value
        self._tag_user_edit(lm, key)

    def set_shared_param(self, item_id: str, key: str, value: Any) -> None:
        """Edit a sample-shared parameter on EVERY measurement in the sample.

        Sample-identity and shared-optics parameters (polymer, solvent,
        temperature, wavelength, refractive index, viscosity, dn/dc) are common to
        all of a sample's measurements, so they are entered once and propagated --
        the reason a multi-concentration SLS set does not require typing the same
        value per concentration. Per-measurement AXES (concentration, angle) are
        not shared; edit those with `set_param`. A key not valid for a given
        sibling's kind (e.g. dn/dc on a DLS measurement) is skipped for it.
        """
        if key not in _SHARED_PARAM_KEYS:
            raise KeyError(f"{key!r} is not a sample-shared parameter; use "
                           f"set_param for per-measurement axes.")
        sid = self._sample_id_of(item_id)
        targets = self._item_ids_in_sample(sid) if sid is not None else [item_id]
        for iid in targets:
            lm = self.workspace.measurements[iid]
            allowed = _DLS_PARAM_KEYS if lm.kind == 'dls' else _SLS_PARAM_KEYS
            if key in allowed:
                lm.working_params[key] = value

    def apply_value_to_items(self, item_ids, key: str, value: Any) -> None:
        """Set one working parameter on several measurements at once.

        Used when the Data tab commits with multiple measurements highlighted: the
        value the user typed on the focused measurement is copied to every other
        highlighted one. A shared key still propagates sample-wide (via
        set_shared_param); a per-measurement key (concentration, angle, mw_fraction)
        is set on exactly the listed measurements. A key not valid for a given
        measurement's kind is silently skipped for it.
        """
        for iid in item_ids:
            lm = self.workspace.measurements.get(iid)
            if lm is None:
                continue
            allowed = _DLS_PARAM_KEYS if lm.kind == 'dls' else _SLS_PARAM_KEYS
            if key not in allowed:
                continue
            if key in _SHARED_PARAM_KEYS:
                self.set_shared_param(iid, key, value)
            else:
                lm.working_params[key] = value
                self._tag_user_edit(lm, key)

    # Parameters that a hand edit CLAIMS: writing one through an ordinary edit
    # path stamps its sidecar 'user', so nothing the program derives can overwrite
    # it later. Table-driven, and consulted by every edit entry point, so the two
    # paths cannot diverge -- there is no second predicate left to mis-size.
    # (The solvent fields are claimed separately, sample-wide, via
    # mark_solvent_field_user; they are deliberately not in this per-item table.)
    _USER_TAGGED_PARAM_KEYS = {
        'mean_count_rate_cps': 'mean_count_rate_cps_source',
    }

    def _tag_user_edit(self, lm, key: str) -> None:
        """Stamp a hand-edited parameter's provenance sidecar as 'user'."""
        src_key = self._USER_TAGGED_PARAM_KEYS.get(key)
        if src_key is not None:
            lm.working_params[src_key] = 'user'

    # -- solvent-property autofill (library:primary) --------------------------
    # Maps a value key to its sidecar provenance key. Both value keys are ordinary
    # (shared) params; the *_source keys are the excluded _PROVENANCE_KEYS.
    _SOLVENT_SOURCE_KEYS = {
        'solvent_refractive_index': 'solvent_refractive_index_source',
        'viscosity_Pa_s': 'viscosity_Pa_s_source',
    }

    def autofill_solvent_props(self, item_id: str, *, force: bool = False) -> dict:
        """Propose n (and, for DLS, viscosity) from the solvent-property library.

        THE single chokepoint for library autofill (invariant #5: logic lives in
        the controller, never the GUI). At the sample's WORKING solvent / T / lambda
        it computes each property via ``physics.solvents`` and writes the value plus
        a ``'library:primary'`` source tag across every measurement in the sample.

        Rules:
          - Only primary-tier solvents are served here (the estimator tab handles
            the rest); unknown/estimate -> no fill.
          - A field the user has claimed (``'user'``) on ANY measurement in the
            sample is NEVER overwritten (unless ``force``) -- mirrors the mw_source
            guard; the check is sample-wide because the write is sample-wide.
          - Out-of-range / unavailable (ValueError) -> the field is not filled; a
            previously *library*-sourced value (stale, computed at a different
            condition) is CLEARED, a user value is left untouched.
          - SLS measurements take refractive index only (no viscosity field).
          - Each proposed value is ROUNDED at the last digit the library can stand
            behind (``unc.round_to_uncertainty``: the display sigma chooses the
            decimal place -- and travels no further, invariant #8) before the
            write, so the parameter table never shows false precision. Idempotent;
            only these library-sourced writes are rounded, never a user value.
        Requires both temperature and wavelength; otherwise nothing is filled.
        Returns a small status dict for the UI.
        """
        sid = self._sample_id_of(item_id)
        targets = self._item_ids_in_sample(sid) if sid is not None else [item_id]
        wp = self.workspace.measurements[item_id].working_params
        solvent = wp.get('solvent_name')
        temp_K = wp.get('temperature_K')
        wavelength_nm = wp.get('wavelength_nm')
        status = {'filled': [], 'out_of_range': [], 'skipped_user': [],
                  'tier': None, 'reason': None}
        # Each refusal below means the same thing: the library cannot stand behind a value
        # at this condition. So each must ALSO clear a stale library-sourced value, exactly
        # as the out-of-range (ValueError) path does further down -- otherwise a number
        # computed for the PREVIOUS solvent stays in the cell, still wearing its
        # 'library:primary' tag and its teal provenance dot. Not cosmetic: the stale
        # viscosity feeds Stokes-Einstein, so an Rh would be reported from a solvent the
        # user has changed away from. Clearing only ever touches 'library:primary' fields.
        if not solvent:
            status['reason'] = 'no solvent'
            self._clear_stale_library_fields(targets, status)
            return status
        status['tier'] = tier = solvents_lib.solvent_tier(solvent)
        if tier != 'primary':
            status['reason'] = 'not a primary-tier solvent'
            self._clear_stale_library_fields(targets, status)
            return status
        if temp_K is None or wavelength_nm is None:
            status['reason'] = 'need both temperature and wavelength'
            self._clear_stale_library_fields(targets, status)
            return status
        temp_C = float(temp_K) - 273.15
        # Each computer returns (value, display sigma) via the SAME accessors the
        # Solvent Explorer readout uses (solvent_value_*) -- since the Spec-3
        # engine amendment those return the PER-CONDITION sigma at the sample's
        # committed condition, so the rounding place is chosen by exactly the
        # uncertainty the library stands behind THERE, and the two surfaces can
        # never round/display at different sigmas.
        computers = (
            ('solvent_refractive_index',
             lambda: self.solvent_value_n(solvent, float(wavelength_nm), temp_C)),
            ('viscosity_Pa_s',
             lambda: self.solvent_value_eta(solvent, temp_C)),
        )
        for field_name, compute in computers:
            # User-guard is SAMPLE-WIDE, matching the sample-wide write: if the user
            # has claimed this field on ANY measurement in the sample, skip it
            # entirely (n/viscosity are shared params -- filling only the untagged
            # siblings would both overwrite a user value and split the sample).
            if not force and self._any_target_field_user(targets, field_name):
                status['skipped_user'].append(field_name)
                continue
            try:
                value, sigma = compute()
            except ValueError:
                # Clear a stale library value (+ tag); never touch a user value.
                self._write_solvent_field(targets, field_name, None, None,
                                          only_if_library=True)
                status['out_of_range'].append(field_name)
                continue
            value = unc.round_to_uncertainty(value, sigma)
            if self._write_solvent_field(targets, field_name, value, 'library:primary'):
                status['filled'].append(field_name)
        return status

    def _clear_stale_library_fields(self, targets, status: dict) -> None:
        """Drop every ``'library:primary'`` value across ``targets``, recording each in
        ``status['out_of_range']``.

        The shared tail of ``autofill_solvent_props``'s refusal branches. A field the
        user has claimed is untouched twice over: ``_write_solvent_field``'s
        ``only_if_library`` skips any tag that is not ``'library:primary'``, and the
        sample-wide user-guard is re-checked here so a claimed field is not cleared on a
        sibling measurement that happens to carry the library tag."""
        for field_name in self._SOLVENT_SOURCE_KEYS:
            if self._any_target_field_user(targets, field_name):
                status['skipped_user'].append(field_name)
                continue
            if self._write_solvent_field(targets, field_name, None, None,
                                         only_if_library=True):
                status['out_of_range'].append(field_name)

    def _any_target_field_user(self, targets, field_name: str) -> bool:
        """True if any applicable measurement in the sample has hand-claimed
        ``field_name`` ('user'). Used to make the autofill user-guard sample-wide."""
        src_key = self._SOLVENT_SOURCE_KEYS[field_name]
        for iid in targets:
            lm = self.workspace.measurements[iid]
            allowed = _DLS_PARAM_KEYS if lm.kind == 'dls' else _SLS_PARAM_KEYS
            if field_name in allowed and lm.working_params.get(src_key) == 'user':
                return True
        return False

    def _write_solvent_field(self, targets, field, value, source,
                             *, only_if_library: bool = False) -> int:
        """Write ``field`` (+ its source tag) across ``targets`` where applicable.

        ``source=None`` clears the value and drops the tag. ``only_if_library``
        touches only measurements whose current tag is ``'library:primary'`` (used
        when clearing a stale value -- never disturbs a user field). Returns the
        number of measurements written."""
        src_key = self._SOLVENT_SOURCE_KEYS[field]
        wrote = 0
        for iid in targets:
            lm = self.workspace.measurements[iid]
            allowed = _DLS_PARAM_KEYS if lm.kind == 'dls' else _SLS_PARAM_KEYS
            if field not in allowed:          # e.g. viscosity on an SLS measurement
                continue
            if only_if_library and lm.working_params.get(src_key) != 'library:primary':
                continue
            lm.working_params[field] = value
            if source is None:
                lm.working_params.pop(src_key, None)
            else:
                lm.working_params[src_key] = source
            wrote += 1
        return wrote

    def mark_solvent_field_user(self, item_id: str, field: str) -> None:
        """Tag n or viscosity as user-owned (sample-wide) so autofill never
        overwrites it. Call this when the user hand-edits the field."""
        if field not in self._SOLVENT_SOURCE_KEYS:
            raise KeyError(f"{field!r} is not a solvent-autofill field.")
        sid = self._sample_id_of(item_id)
        targets = self._item_ids_in_sample(sid) if sid is not None else [item_id]
        src_key = self._SOLVENT_SOURCE_KEYS[field]
        for iid in targets:
            lm = self.workspace.measurements[iid]
            allowed = _DLS_PARAM_KEYS if lm.kind == 'dls' else _SLS_PARAM_KEYS
            if field in allowed:
                lm.working_params[src_key] = 'user'

    def clear_library_solvent_field(self, item_id: str, field: str) -> None:
        """Drop a solvent field's value + ``'library:primary'`` tag sample-wide when
        the user clears the cell -- but only where the tag IS library (never disturbs a
        ``'user'`` field). Mirrors the stale-value auto-clear in
        ``autofill_solvent_props`` so a cleared library cell can't keep advertising a
        library value (its provenance dot) on an EMPTY safety-critical cell. A later
        solvent/temperature/wavelength change re-derives it -- autofill is blocked only
        by a ``'user'`` tag, not by the absence of one, so dropping the tag is safe."""
        if field not in self._SOLVENT_SOURCE_KEYS:
            raise KeyError(f"{field!r} is not a solvent-autofill field.")
        sid = self._sample_id_of(item_id)
        targets = self._item_ids_in_sample(sid) if sid is not None else [item_id]
        self._write_solvent_field(targets, field, None, None, only_if_library=True)

    def reautofill_after(self, item_id: str, changed_key: str) -> Optional[dict]:
        """Re-derive n/viscosity when the solvent, temperature, or wavelength
        changes. Returns the autofill status, or None if ``changed_key`` is not a
        re-derive trigger."""
        if changed_key in ('temperature_K', 'wavelength_nm', 'solvent_name'):
            return self.autofill_solvent_props(item_id)
        return None

    # -- Solvent Explorer wrappers (display-only calculator; GUI holds no physics) --
    # All forward to physics.solvents and let ValueError surface for the tab to show.
    # The engine's display-only PER-CONDITION uncertainty (σ_n(λ,T) /
    # σ_η,rel(T)) is returned for the readout / confidence band; per invariant #8 it
    # is NEVER propagated into any analysis SE (these values never reach a
    # measurement — the Explorer is a standalone calculator).
    _SOLVENT_CURVE_POINTS = 121

    def solvent_value_n(self, name: str, wavelength_nm: float,
                        temperature_C: float):
        """(n, absolute PER-CONDITION display uncertainty) at one condition (σ_n(λ,T),
        so the readout's ± matches the band at the selected dot). Raises
        ``ValueError`` (unknown solvent / out-of-range λ or T) for the tab to display."""
        n = solvents_lib.refractive_index_solvent(name, wavelength_nm, temperature_C)
        return n, solvents_lib.solvent_uncertainty_n(name, wavelength_nm, temperature_C)

    def solvent_value_eta(self, name: str, temperature_C: float):
        """(η in Pa·s, absolute PER-CONDITION display uncertainty in Pa·s) at one
        temperature; the relative σ_η,rel(T) is turned into an absolute ± here for
        the readout. Raises ``ValueError`` (unknown / no-viscosity / out-of-range T)."""
        eta = solvents_lib.viscosity_solvent(name, temperature_C)
        return eta, solvents_lib.solvent_uncertainty_eta(name, temperature_C) * eta

    def solvent_property_info(self, name: str) -> dict:
        """Library metadata record for ``name`` (validity boxes, tier, box-wide
        uncertainties, has_viscosity). A thin pass-through so the GUI reads solvent
        descriptors through the controller boundary rather than importing
        physics.solvents directly. Raises ``ValueError``/``TypeError``
        for an unknown or invalid name."""
        return solvents_lib.solvent_property_info(name)

    def available_solvents(self, tier: str = 'primary') -> list:
        """The library's solvent names at ``tier`` (for the Data-tab dropdown). Thin
        pass-through keeping physics.solvents behind the controller boundary."""
        return solvents_lib.available_solvents(tier)

    def _solvent_sample_box(self, lo: float, hi: float, compute):
        """Grid ``compute`` over ``[lo, hi]`` (SOLVENT_CURVE_POINTS), keeping only the
        points where it evaluates. The per-point guard drops an exact box-edge point
        nudged out by a float round-trip; callers validate the fixed axis up front so a
        genuine out-of-range condition raises before the grid rather than emptying it."""
        xs, ys = [], []
        for x in np.linspace(lo, hi, self._SOLVENT_CURVE_POINTS):
            try:
                ys.append(compute(float(x)))
                xs.append(float(x))
            except ValueError:
                continue
        return np.asarray(xs), np.asarray(ys)

    def solvent_curve_n_vs_T(self, name: str, wavelength_nm: float):
        """(T_C[], n[], band_half[]) across the record's refractive-index temperature
        box at the selected wavelength. ``band_half`` is the PER-CONDITION absolute
        σ_n(λ, T[i]) — tightest near the reference temperature, growing
        toward the box edges, never exceeding the box-wide ``n_uncertainty``. Raises
        ``ValueError`` if the wavelength is outside the box (so the tab hides the RI
        curve but can still draw η)."""
        info = solvents_lib.solvent_property_info(name)
        t0, t1 = info['n_temp_min_C'], info['n_temp_max_C']
        # Validate the wavelength up front (its error is the meaningful one).
        solvents_lib.refractive_index_solvent(name, wavelength_nm, 0.5 * (t0 + t1))
        temps, ns = self._solvent_sample_box(
            t0, t1, lambda t: solvents_lib.refractive_index_solvent(name, wavelength_nm, t))
        band = np.array([solvents_lib.solvent_uncertainty_n(name, wavelength_nm, t)
                         for t in temps])
        return temps, ns, band

    def solvent_curve_n_vs_lambda(self, name: str, temperature_C: float):
        """(lambda_nm[], n[], band_half[]) across the refractive-index wavelength box
        at the selected temperature. ``band_half`` is the PER-CONDITION absolute
        σ_n(λ[i], T). Raises ``ValueError`` if the temperature is outside
        the box."""
        info = solvents_lib.solvent_property_info(name)
        l0, l1 = info['n_lambda_min_nm'], info['n_lambda_max_nm']
        solvents_lib.refractive_index_solvent(name, 0.5 * (l0 + l1), temperature_C)
        lams, ns = self._solvent_sample_box(
            l0, l1, lambda lam: solvents_lib.refractive_index_solvent(name, lam, temperature_C))
        band = np.array([solvents_lib.solvent_uncertainty_n(name, lam, temperature_C)
                         for lam in lams])
        return lams, ns, band

    def solvent_curve_eta_vs_T(self, name: str):
        """(T_C[], eta[], band_half[]) across the viscosity temperature box (η in
        Pa·s). ``band_half`` is the PER-CONDITION σ_η,rel(T[i]) · η[i]:
        relative, so it widens with η, and the relative width itself varies where
        the source supports a shape (flat for bulk-grade sources). Raises
        ``ValueError`` if the solvent carries no viscosity block."""
        info = solvents_lib.solvent_property_info(name)
        if not info.get('has_viscosity'):
            raise ValueError(f"Viscosity data is not available for {name!r}.")
        temps, etas = self._solvent_sample_box(
            info['eta_temp_min_C'], info['eta_temp_max_C'],
            lambda t: solvents_lib.viscosity_solvent(name, t))
        rel = np.array([solvents_lib.solvent_uncertainty_eta(name, t) for t in temps])
        return temps, etas, rel * etas

    def sample_id_of(self, item_id: str) -> Optional[str]:
        """Public: the sample currently containing this measurement (or None)."""
        return self._sample_id_of(item_id)

    def sample_display_fields(self, sample_id: str) -> Optional[Tuple[str, str, float]]:
        """Verbatim-cased (polymer, solvent, temperature_K) for display use (e.g.
        suggested export filenames), or None for an unknown sample id.

        The Sample dataclass stores the committed names verbatim -- only the
        grouping key casefolds -- so this is the case-preserving counterpart to
        the casefolded sample id."""
        s = self.workspace.samples.get(sample_id)
        if s is None:
            return None
        return (s.polymer_name, s.solvent_name, s.temperature_K)

    def _sample_id_of(self, item_id: str) -> Optional[str]:
        """Which sample currently contains this measurement (by committed grouping)."""
        for sid, s in self.workspace.samples.items():
            if (item_id in s.dls_item_ids or item_id in s.sls_item_ids
                    or item_id == s.solvent_reference_item_id
                    or item_id in s.extra_solvent_reference_item_ids):
                return sid
        return None

    def _item_ids_in_sample(self, sample_id: str) -> List[str]:
        s = self.workspace.samples[sample_id]
        ids = list(s.dls_item_ids) + list(s.sls_item_ids)
        if s.solvent_reference_item_id:
            ids.append(s.solvent_reference_item_id)
        ids += list(s.extra_solvent_reference_item_ids)
        return ids

    def sample_chain_type(self, sample_id: str) -> Optional[str]:
        """The sample's committed chain-type label ('linear'/'branched'/None).

        chain_type is a sample-shared committed parameter (propagated across the
        sample's measurements like polymer_name), so any member carries it; read
        the first non-None. Canonicalized leniently -- an unexpected stored value
        reads as None rather than raising (a read must never break the architecture
        computation). Qt-free."""
        if sample_id not in self.workspace.samples:
            return None
        for iid in self._item_ids_in_sample(sample_id):
            lm = self.workspace.measurements.get(iid)
            if lm is None:
                continue
            raw = lm.committed_params.get('chain_type')
            if raw is None:
                continue
            try:
                code = _normalize_chain_type(raw)
            except ValueError:
                continue
            if code is not None:
                return code
        return None

    def dirty_keys(self, item_id: str) -> List[str]:
        """Which fields of this measurement changed since the last commit."""
        return self.workspace.measurements[item_id].dirty_keys()

    def is_dirty(self) -> bool:
        """True if any measurement OR any calibration has un-committed edits."""
        if self.workspace.any_dirty():
            return True
        if self.calibration_working != self.calibration_committed:
            return True
        if set(self.sample_calibration_working) != set(self.sample_calibration_committed):
            return True
        for sid, cw in self.sample_calibration_working.items():
            if cw != self.sample_calibration_committed.get(sid):
                return True
        return False

    def committed_signature(self, item_ids) -> tuple:
        """A hashable signature of the *committed* analysis parameters of `item_ids`.
        Two signatures compare equal iff none of those measurements' fit-affecting
        committed parameters changed. Provenance sidecar keys (`_PROVENANCE_KEYS`) are
        excluded — they don't affect a fit (mirrors `LoadedMeasurement.is_dirty`).

        The GUI records this when it caches a fit and re-checks it on redisplay, so a
        result shown after the user commits a parameter edit is flagged stale — the
        window the coarse `is_dirty()` note misses, since committing clears `is_dirty`
        while leaving the displayed fit computed from the old values (feedback
        2026-07-06). The signature embeds each item_id, so a tab's `_run_signature`
        can recover the input set from a stored signature without a side table."""
        def _hashable(v):
            return tuple(v) if isinstance(v, list) else v

        out = []
        for iid in sorted(item_ids):
            lm = self.workspace.measurements.get(iid)
            if lm is None:
                out.append((iid, None))
                continue
            params = tuple(sorted(
                (k, _hashable(v)) for k, v in lm.committed_params.items()
                if k not in _PROVENANCE_KEYS))
            out.append((iid, params))
        return tuple(out)

    def sls_run_signature(self, sample_id: str,
                          fraction: Optional[str] = None) -> tuple:
        """A hashable signature of everything an SLS fit for (sample, fraction) depends
        on: the committed params of its SLS measurements + the shared solvent reference,
        the fraction's data mask, the effective (per-sample or session) calibration, the
        settings-seeded fit inputs (SE estimator + Guinier qRg-max validity limit), and
        the sample's stored Mw *with its provenance* (the calibration-free A₂ reads the
        value, and reads the source + calibration flag to decide whether it may divide by
        it at all — see `run_calibration_free_a2`). Two signatures compare
        equal iff a re-run would read identical inputs — so the SLS tab can redisplay a
        cached fit and flag it stale exactly when recomputing would change it.
        `committed_signature` alone would miss the mask, calibration, the two settings
        inputs (`se_estimator`; `guinier_qrg_max`, which sets the Guinier validity
        caution), and the stored Mw — the inputs the SLS fits read beyond the measurement
        params. If a future fit path reads another `self.settings.*` value or workspace
        field, add it here or the redisplay could show a fit against a stale input.

        BASE vs EXTRAS checklist — when you ADD an input here, decide which side of the
        split it belongs to. The narrow base (`_form_factor_run_signature`) is the part a
        FORM-FACTOR fit reads; the extras this method appends (`guinier_qrg_max`, stored Mw)
        are read only by other SLS fits or MUTATED by a consumer (the Cross-Sample auto-
        select writes the stored Mw). If a form-factor fit READS the new input, it must go in
        the base; if not — especially if a consumer mutates it mid-refresh — keep it out of
        the base or the freshness gate will drop fresh form-factor fits (the v1.11.0 bug; see
        `_form_factor_run_signature`)."""
        # The calibration-free A₂ reads the sample's stored Mw when none is passed (the
        # SLS tab always lets it), so a changed Mw must flag a cached 'calfree' fit stale
        # (an SLS run does NOT store Mw, so this stays stable across run→redisplay; only a
        # manual/Cross-Sample Mw edit moves it). Conservative for the other methods, which
        # don't read the stored Mw — the safe direction for a stale guard.
        sample = self.workspace.samples.get(sample_id)
        # `calibrated` joins the value and source because that fit reads the stored Mw's
        # PROVENANCE too, refusing an arbitrary-scale one. It cannot drop a fresh fit:
        # every writer co-writes the value and/or source, both already here, so it never
        # moves this signature alone. EXTRAS, never the base: it is consumer-mutated by the
        # Cross-Sample auto-select, the disqualifier `_form_factor_run_signature` names.
        r = None if sample is None else sample.result_for(fraction)
        mw_sig = ((None, None, None) if r is None else
                  (r.effective_mw(), r.mw_source, r.calibrated))
        return self._form_factor_run_signature(sample_id, fraction) + (
            self.settings.guinier_qrg_max, mw_sig)

    def _form_factor_run_signature(self, sample_id: str,
                                   fraction: Optional[str] = None) -> tuple:
        """The subset of `sls_run_signature` a FORM-FACTOR fit actually reads: committed
        params (incl. the Mie/core-shell contrast RIs) + the shared solvent ref, the mask, the
        effective calibration, and the SE estimator. Deliberately EXCLUDES the stored Mw and
        `guinier_qrg_max` — a form-factor fit reads neither, and the stored Mw is MUTATED by the
        Cross-Sample auto-select during the very refresh that displays the pooled candidate, so
        gating form-factor freshness on the full `sls_run_signature` would spuriously mark a
        just-run fit stale. This is the signature stamped on / gated against a stored form-factor
        fit (see run_form_factor / _form_factor_solutions).

        LOAD-BEARING INVARIANT (the reason this signature is narrower than `sls_run_signature`,
        not equal to it): *a freshness signature must contain EXACTLY what the fit reads — no
        less, and no more that a CONSUMER of the result mutates.* Too little (drop an input the
        fit reads) → a stale fit is offered as fresh. Too much (include an input only a consumer
        touches) → a genuinely fresh fit is dropped. The v1.11.0 bug was the "no
        more" half: the gate keyed on the full `sls_run_signature`, whose stored-Mw component the
        Cross-Sample auto-select writes DURING the reading refresh, so a just-run fit was marked
        stale mid-refresh. Fix: gate on this narrow base, which excludes exactly the consumer-
        mutated Mw (and `guinier_qrg_max`, which a form-factor fit never reads). When editing
        either signature, re-check both halves for every input."""
        series = [lm for lm in self.workspace.sample_measurements(sample_id, 'sls')
                  if lm.committed_params.get('mw_fraction') == fraction]
        ids = [lm.item_id for lm in series]
        # Each series is referenced against ITS OWN blank (solvent_reference_for), so the
        # signature carries every blank the fraction resolves to -- de-duplicated, in
        # load order -- not the sample slot alone: a load-order or pairing change that
        # moves a series onto a different blank must stale the fit.
        ids += list(dict.fromkeys(self._blank_ids_for(series)))
        mask = self.sls_masks.get((sample_id, fraction))
        mask_sig = () if (mask is None or mask.is_empty()) else (
            tuple(sorted(mask.masked_angles)),
            tuple(sorted(mask.masked_concentrations)),
            tuple(sorted(tuple(p) for p in mask.masked_points)))
        cal_sig = tuple(sorted(self._calibration_for_sample(sample_id).to_dict().items()))
        return (self.committed_signature(ids), mask_sig, cal_sig, self.settings.se_estimator)

    def dls_run_signature(self, sample_id: str,
                          fraction: Optional[str] = None) -> tuple:
        """A hashable signature of every input `dls_rh_candidates` reads for
        (sample, fraction). Two signatures compare equal iff a re-run would produce
        the identical candidate list — so `_dls_cached` can serve a memoized list and
        MISS exactly when a recompute would change it. Enumerated to be a *complete*
        invalidation key (an incomplete one would serve a stale Rh — worse than no
        cache), the inputs are:

        - the committed params of the fraction's DLS measurements (feed the tier-1
          cumulant, tier-2 Γ-q², tier-3 D-vs-c fits — all via `lm.build()`);
        - `settings.skip_initial_channels`, `settings.cumulant_method`,
          `settings.cumulant_fit_cutoff`, `settings.cumulant_weighting` and
          `settings.se_estimator`, which the tiers read: tier 1's cumulant fit takes the
          first four, and the Gamma-q2 / D-vs-c fits behind tiers 2 and 3 take all five
          -- the estimator moves the candidate's ±
          (and, through the user's opt-in, the ± promoted onto the sample), so it is a key
          like the SLS one. It used to be left out on the stated ground that tiers 2/3
          were called without it, which was the very omission that let an HC3 ± be
          promoted over the user's OLS choice; `cumulant_fit_cutoff` joined the key with
          the setting itself rather than after the same lesson a second time;
        - EVERY durable `sample_rh_rows` row for this (sample, fraction) that the
          candidate list surfaces but does not compute — replicate averages AND
          mode-family rows, keyed on source_kind so a new kind can never sit
          outside the signature: a
          kind-naming filter here is a KEEP list, and it went stale in the same
          edit that taught the candidates to read 'mode_family'; the gamma_q2/
          conc_extrap rows also enter — over-inclusion, which only costs a miss);
        - the user's stored Gamma-q2 / D-vs-c runs for this (sample, fraction) with their
          freshness stamps (`_dls_run_state`): tiers 2/3 READ a fresh stored run instead
          of re-deriving, so (re)running one on the DLS tab -- which moves no committed
          param -- must MISS here, exactly as the form-factor run-state does for SLS;
        - the version stamp of each distribution result (CONTIN/NNLS) this sample's
          measurements have in `self.results` — the peak-picker candidates read them.
          The stamp (not `id(res)`) is the change key: a freed result's id can be
          reused by a later re-run (CPython ABA), which would let a stale signature
          falsely match; the strictly-increasing `_distribution_stamps` counter is
          never reused, so a re-fit always moves the signature and misses;
        - `settings.cumulant_order`, which every tier re-derives at, and the stamp of
          each of these measurements' stored Correlogram cumulant fits: tier 1 READS a
          fresh one instead of re-deriving, and a re-Run moves no committed param or
          Setting, so the stamp's never-reused fit number is what makes it miss.

        The raw correlogram data is not keyed: it is immutable once loaded, and a
        reload clears every cache."""
        lms = [lm for lm in self.workspace.sample_measurements(sample_id, 'dls')
               if lm.committed_params.get('mw_fraction') == fraction]
        dls_ids = [lm.item_id for lm in lms]
        committed = self.committed_signature(dls_ids)
        # Durable Rh rows this list surfaces (replicate averages, mode families —
        # and any future kind, unconditionally). Capture the fields that reach the
        # candidate so an edited/removed/added row misses; source_kind is in the
        # tuple so two kinds' rows can never alias.
        rep = tuple(sorted(
            (r.source_kind, r.source_set, r.rh_nm, r.rh_se,
             bool(r.is_apparent), r.from_label, r.se_estimator)
            for r in self.workspace.sample_rh_rows.values()
            if r.sample_id == sample_id and r.fraction == fraction))
        # distribution results feeding the peak-picker candidates (only this
        # sample's measurements).
        dls_id_set = set(dls_ids)
        dist = tuple(sorted(
            (k[1], k[2], self._distribution_stamps.get(k, 0))
            for k in self.results
            if isinstance(k, tuple) and len(k) == 3 and k[0] == 'distribution'
            and k[1] in dls_id_set))
        cumulant_fits = tuple(sorted(
            (iid, self._dls_run_sig.get(('cumulants', iid)))
            for iid in dls_ids if ('cumulants', iid) in self.results))
        return (committed, self.settings.skip_initial_channels,
                self.settings.cumulant_method, self.settings.cumulant_fit_cutoff,
                self.settings.cumulant_weighting, self.settings.cumulant_order,
                self.settings.se_estimator, rep, dist,
                self._dls_run_state(sample_id, fraction), cumulant_fits)

    # ----------------------------------------------------------------------
    # Stored Γ-vs-q² / D-vs-c runs: freshness stamps (the form-factor pattern)
    # ----------------------------------------------------------------------
    def _dls_run_stamp(self, item_ids) -> tuple:
        """Everything a stored Γ-vs-q² / D-vs-c run was computed from: the committed
        params of the measurements it FITTED (their item_ids ride inside, as
        `committed_signature` embeds them) plus the six Settings the run seeds -- the
        cumulant order among them, since the points fit at it. Written
        beside the result by the run (`run_gamma_q2` / `run_concentration_extrapolation`)
        and compared by `_stored_dls_run` -- the DLS analogue of `_form_factor_run_sig`.
        The fitted set, not the whole fraction: a new measurement loaded beside a run
        leaves that run exactly what the Summary table still shows. The analysis region a
        checked run read is not here: it is the tab's input, not a Setting, and moving it
        later flags the tab's display stale without withdrawing a promoted value."""
        return (self.committed_signature(item_ids), self.settings.skip_initial_channels,
                self.settings.cumulant_method, self.settings.cumulant_fit_cutoff,
                self.settings.cumulant_weighting, self.settings.cumulant_order,
                self.settings.se_estimator)

    def _stored_dls_run(self, kind: str, sample_id: str, fraction: Optional[str]):
        """The user's stored run of `kind` ('gamma_q2' | 'conc_extrap') for (sample,
        fraction) while it is FRESH -- its stamp still equals what the same inputs give
        now -- else None. A stale run (a committed edit, a Settings change since) is not
        offered; the candidate builder re-derives instead, and the DLS tab flags the run
        stale on its own surface. Unstamped (a result published without its stamp) reads
        as stale, the safe direction. Runtime state only: a reopened session has no
        stored runs (the builder re-derives until the user runs again) while its
        persisted Summary Table 2 row keeps the run's number -- a recorded residual."""
        key = (kind, sample_id, fraction)
        res = self.results.get(key)
        stamp = self._dls_run_sig.get(key)
        if res is None or stamp is None:
            return None
        item_ids = [entry[0] for entry in stamp[0]]
        return res if stamp == self._dls_run_stamp(item_ids) else None

    def _stored_run_params(self, kind: str, sample_id: str, fraction: Optional[str],
                           param: str) -> list:
        """One committed parameter per measurement a stored Γ-q² / D-vs-c run FITTED,
        read off the run's stamp (its item_ids ride inside `committed_signature`) --
        what the candidate builder needs to say whether the run pooled an axis the
        passive default keeps apart. Empty when the run is absent or unstamped."""
        stamp = self._dls_run_sig.get((kind, sample_id, fraction))
        if not stamp:
            return []
        out = []
        for entry in stamp[0]:
            m = self.workspace.measurements.get(entry[0])
            if m is not None:
                out.append(m.committed_params.get(param))
        return out

    def _dls_run_state(self, sample_id: str, fraction: Optional[str]) -> tuple:
        """The stored Γ-q² / D-vs-c runs for (sample, fraction) as sorted (kind, stamp)
        pairs -- folded into `dls_run_signature` so a first run, a re-run or a cleared run
        moves the candidate memo the way `_form_factor_run_state` does for SLS. Iterates a
        snapshot of `self.results` (the run writes it on the worker thread)."""
        pairs = [(key[0], self._dls_run_sig.get(key)) for key in list(self.results)
                 if isinstance(key, tuple) and len(key) == 3
                 and key[0] in ('gamma_q2', 'conc_extrap')
                 and key[1] == sample_id and key[2] == fraction]
        return tuple(sorted(pairs, key=lambda p: p[0]))

    def commit(self) -> None:
        """Adopt all working edits as committed and re-group. Runs no analysis."""
        # Record the about-to-be-replaced applied state as an undo restore point,
        # but only when this Update actually changes something (no empty entries).
        if self.is_dirty():
            self._commit_history.append(self._snapshot_committed())
        for m in self.workspace.measurements.values():
            m.commit()
        self._commit_calibration(self.calibration_working)
        self.calibration_committed = CalibrationState(
            **self.calibration_working.to_dict())
        # Per-sample overrides: rebuild committed from working (drops any disabled).
        new_committed: Dict[str, CalibrationState] = {}
        for sid, cw in self.sample_calibration_working.items():
            self._commit_calibration(cw)
            new_committed[sid] = CalibrationState(**cw.to_dict())
        self.sample_calibration_committed = new_committed
        self._rekey_sample_stores(self.workspace.regroup())

    @staticmethod
    def _commit_calibration(cal: CalibrationState) -> None:
        """Recompute k_c in place (so working and its committed copy agree).

        compute_k_c() can raise ValueError -- an unspecified/invalid geometry
        or an unsupported standard wavelength. (An out-of-range standard
        temperature only WARNS and still computes, so it does not
        reach here.) Treat a raise like compute_k_c's own "not enough info" case and
        commit k_c = None (uncomputable -> the result is flagged uncalibrated), rather
        than letting it crash the Update handler -- the preview surface (preview_k_c)
        has already shown the user the specific reason before they commit. Fail-loud
        on the preview, degrade gracefully on commit.
        """
        try:
            cal.k_c = cal.compute_k_c()
        except ValueError:
            cal.k_c = None

    def undo_to_committed(self) -> None:
        """Discard all un-committed edits (working <- committed)."""
        for m in self.workspace.measurements.values():
            m.revert()
        self.calibration_working = CalibrationState(
            **self.calibration_committed.to_dict())
        self.sample_calibration_working = {
            sid: CalibrationState(**cc.to_dict())
            for sid, cc in self.sample_calibration_committed.items()}

    def _snapshot_committed(self) -> Dict[str, Any]:
        """A deep-enough copy of every committed parameter set + calibration, used
        as an undo restore point (see `undo`)."""
        return {
            'measurements': {iid: dict(m.committed_params)
                             for iid, m in self.workspace.measurements.items()},
            'calibration': self.calibration_committed.to_dict(),
            'sample_calibration': {sid: cc.to_dict() for sid, cc
                                   in self.sample_calibration_committed.items()},
        }

    def snapshot_working_params(self) -> Dict[str, Dict[str, Any]]:
        """Every measurement's WORKING parameters, copied -- a restore point for a
        gesture that must mutate in order to decide whether to mutate.

        The working-params sibling of `_snapshot_committed`, and public because the
        Data tab calls it: its committed counterpart is private only because nothing
        outside this class needs it.

        Two deliberate differences from that sibling. It carries no calibration state,
        because the one caller's pre-decision writes (`apply_value_to_items` and
        `set_shared_param`, plus `_tag_user_edit`'s provenance sidecar) land inside
        `working_params` and nowhere else. And its restore does not regroup, because
        nothing was committed, so the grouping never moved.

        Whole-workspace on purpose. A shared key propagates to every member of the
        measurement's committed sample, so the touched set is wider than any selection
        the caller knows about -- and COMPUTING that set here would be a second
        computation of `set_shared_param`'s propagation boundary, which is exactly the
        shape that has gone wrong before (rule 12). Copying everything is cheap and
        cannot be wrong.
        """
        return {iid: dict(m.working_params)
                for iid, m in self.workspace.measurements.items()}

    def restore_working_params(self, snap: Dict[str, Dict[str, Any]]) -> None:
        """Put a `snapshot_working_params` restore point back.

        Measurements created since the snapshot are left alone and ones that have gone
        are skipped, matching `_restore_committed`'s own skip rule. Committed state is
        untouched, so no regroup is owed.

        The contract this upholds is the shared regroup confirm's: "Cancel ... the
        caller applies nothing on it" (`gui/widgets.py`). What would invalidate it is a
        caller growing a pre-decision write that reaches something OTHER than
        `working_params` -- that write would survive the restore silently.
        """
        for iid, params in snap.items():
            m = self.workspace.measurements.get(iid)
            if m is not None:
                m.working_params = dict(params)

    def _restore_committed(self, snap: Dict[str, Any]) -> None:
        """Restore a snapshot into BOTH committed and working state, then re-group.
        Measurements that no longer exist are skipped; measurements created after the
        snapshot are left untouched (they were not part of that applied state)."""
        for iid, params in snap['measurements'].items():
            m = self.workspace.measurements.get(iid)
            if m is not None:
                m.committed_params = dict(params)
                m.working_params = dict(params)
        self.calibration_committed = CalibrationState(**snap['calibration'])
        self.calibration_working = CalibrationState(**snap['calibration'])
        self.sample_calibration_committed = {
            sid: CalibrationState(**d) for sid, d in snap['sample_calibration'].items()}
        self.sample_calibration_working = {
            sid: CalibrationState(**d) for sid, d in snap['sample_calibration'].items()}
        # Rides regroup's silent tier default: an undo that re-merges two
        # deliberate-tier values is a contrived corner (see Workspace.regroup).
        self._rekey_sample_stores(self.workspace.regroup())

    def can_undo(self) -> bool:
        """True if Undo has anything to do: either un-applied edits to discard, or a
        previously-applied state to step back to."""
        return self.is_dirty() or bool(self._commit_history)

    def undo(self) -> bool:
        """Two-mode Undo, each returning to an APPLIED parameter set (feedback #5):

        - If there are un-applied edits, discard them (working <- committed): this
          returns to the current applied state.
        - Otherwise step back to the previously-applied state by popping the commit
          history into both committed and working.

        Returns True when the committed state (and thus grouping) may have changed,
        so the GUI knows to re-group/refresh the sidebar.
        """
        if self.is_dirty():
            self.undo_to_committed()
            return False
        if self._commit_history:
            self._restore_committed(self._commit_history.pop())
            return True
        return False

    def reset_working_params(self, item_ids) -> None:
        """Blank the entered parameters of the given measurements — a clean slate to
        re-enter. Identity/optics/concentration/temperature/
        viscosity/dn-dc/label/geometry are cleared; the per-measurement scattering
        ANGLE is preserved (it is a structural fact from the instrument file, not a
        typed parameter), as is the raw data. Edits are left in the WORKING set only
        (pending), so the change is visible, undoable, and applies on Update."""
        for iid in item_ids:
            m = self.workspace.measurements.get(iid)
            if m is None:
                continue
            for key in list(m.working_params):
                if key in _RESET_PRESERVE_KEYS:
                    continue
                m.working_params[key] = None

    # ----------------------------------------------------------------------
    # Calibration editing
    # ----------------------------------------------------------------------
    def set_calibration_field(self, key: str, value: Any,
                              sample_id: Optional[str] = None) -> None:
        """Edit a calibration field. With sample_id and a per-sample override
        active, edits that sample's calibration; otherwise the session default."""
        cal = self._working_calibration(sample_id)
        if not hasattr(cal, key):
            raise KeyError(f"{key!r} is not a calibration field.")
        if key == 'standard_refractive_index':
            # A typed value is the user's: tagged so, and never re-proposed away
            # (invariant 3). A CLEARED field goes back to the library -- the one way
            # to resume tracking after typing -- and stays blank, still library-tagged,
            # only where the library has no value at this condition (the k_c note then
            # says the correction cannot be applied).
            if value is None:
                cal.standard_refractive_index = cal.library_standard_index()
                cal.standard_refractive_index_source = 'library'
            else:
                cal.standard_refractive_index = value
                cal.standard_refractive_index_source = 'user'
            return
        if key in ('standard_wavelength_nm', 'standard_temperature_C'):
            # The standard's refractive index is the library's toluene value AT the
            # standard's wavelength and temperature (invariant 3, the Data tab's own
            # rule for the sample's n: re-derived when lambda or T changes, blocked only
            # by a user-typed value). The stored provenance tag decides -- never the
            # value: a library-tagged index follows the condition, a user-tagged one is
            # left exactly as typed.
            setattr(cal, key, value)
            if cal.standard_refractive_index_source == 'library':
                # The library's value at the NEW condition, or -- when the library has
                # none there (outside toluene's n validity box) -- a BLANK, never the old
                # condition's value left standing: `autofill_solvent_props` clears a stale
                # library value on the same grounds, and the blank carries its own note.
                # The tag stays 'library', so the blank re-proposes as soon as the
                # condition comes back inside the box.
                cal.standard_refractive_index = cal.library_standard_index()
            return
        setattr(cal, key, value)

    def preview_k_c(self, sample_id: Optional[str] = None) -> Optional[float]:
        """k_c from the working calibration in scope (sample override or session)."""
        return self._working_calibration(sample_id).compute_k_c()

    def committed_k_c(self, sample_id: Optional[str] = None) -> Optional[float]:
        return self._committed_calibration(sample_id).k_c

    def preview_calibration_note(self, sample_id: Optional[str] = None) -> Optional[str]:
        """Every note the working calibration in scope owes the user, one per line, or
        None. Mirrors preview_k_c; the GUI surfaces it next to k_c.

        Began as the toluene-temperature extrapolation note alone; the
        siblings joined it on the same surface, because each is a precondition of
        the calibration the k_c line alone cannot show: the wavelength extrapolation of
        the same table, a blank or off-library standard index, and a standard
        wavelength that is not the wavelength the SLS data in scope carry.
        Disclosures, never gates: the fields stay the
        user's to set, and every run still computes.
        """
        cal = self._working_calibration(sample_id)
        parts = [
            cal.temperature_extrapolation_note(),
            cal.wavelength_extrapolation_note(),
            cal.standard_index_note(),
            self._standard_wavelength_mismatch_note(cal, sample_id),
        ]
        lines = [p for p in parts if p]
        return '\n'.join(lines) if lines else None

    def _sls_data_wavelengths_nm(self, sample_id: Optional[str]) -> List[float]:
        """The distinct wavelengths (committed) carried by the SLS measurements a
        calibration in scope applies to: the sample's own when one is named (the panel
        the user is looking at), else every SLS measurement in the session."""
        if sample_id is not None and sample_id in self.workspace.samples:
            items = list(self.workspace.sample_measurements(sample_id, 'sls'))
        else:
            items = [lm for lm in self.workspace.measurements.values()
                     if lm.kind == 'sls']
        seen: List[float] = []
        for lm in items:
            lam = lm.committed_params.get('wavelength_nm')
            try:
                lam = float(lam)
            except (TypeError, ValueError):
                continue
            if math.isfinite(lam) and not any(math.isclose(lam, s) for s in seen):
                seen.append(lam)
        return sorted(seen)

    def _standard_wavelength_mismatch_note(self, cal: CalibrationState,
                                          sample_id: Optional[str]) -> Optional[str]:
        """The cross-check the prefills never had: k_c is computed
        from the toluene Rayleigh ratio at the STANDARD's wavelength, so a standard left
        at the shipped 532 nm beside 660 nm data gives a k_c off by R(532)/R(660) = 2.8x
        -- and every Mw and A2 with it, flagged calibrated. The program already holds the
        data's wavelength, so it says when the two disagree. A disclosure the user acts
        on by editing the field; the value is theirs (invariant 3). None with no SLS
        data in scope, or when every measurement matches."""
        try:
            std = float(cal.standard_wavelength_nm)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(std):
            return None
        others = [lam for lam in self._sls_data_wavelengths_nm(sample_id)
                  if not math.isclose(lam, std, abs_tol=1e-6)]
        if not others:
            return None
        listed = ', '.join(f'{lam:g}' for lam in others)
        # Size it from the table where both wavelengths are served (the same lookup
        # compute_k_c uses); a wavelength the table refuses gets the qualitative note.
        factor = ''
        if len(others) == 1:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    r_std = phys.rayleigh_ratio_toluene(
                        std, float(cal.standard_temperature_C),
                        geometry=cal.standard_geometry,
                        depolarization_ratio_v=cal.depolarization_ratio_v)
                    r_data = phys.rayleigh_ratio_toluene(
                        others[0], float(cal.standard_temperature_C),
                        geometry=cal.standard_geometry,
                        depolarization_ratio_v=cal.depolarization_ratio_v)
                if math.isfinite(r_std) and math.isfinite(r_data) and r_data > 0:
                    factor = (f" At {others[0]:g} nm the toluene Rayleigh ratio is "
                              f"{r_std / r_data:.2f}x smaller than at {std:g} nm, and k_c, "
                              f"Mw and absolute A2 scale with it.")
                    if r_std / r_data < 1.0:
                        factor = (f" At {others[0]:g} nm the toluene Rayleigh ratio is "
                                  f"{r_data / r_std:.2f}x larger than at {std:g} nm, and "
                                  f"k_c, Mw and absolute A2 scale with it.")
            except (TypeError, ValueError):
                factor = ''
        return (
            f"The calibration standard's wavelength ({std:g} nm) is not the wavelength "
            f"the SLS data carry ({listed} nm): k_c is computed from the toluene "
            f"Rayleigh ratio at the standard's wavelength, so the two should normally "
            f"match.{factor}")

    # ---- per-sample calibration override (session-wide is the default) ----
    def has_sample_calibration(self, sample_id: str) -> bool:
        """True if this sample carries its own calibration (working) override."""
        return sample_id in self.sample_calibration_working

    def enable_sample_calibration(self, sample_id: str) -> None:
        """Give the sample its own calibration, seeded from the session default."""
        if sample_id not in self.sample_calibration_working:
            self.sample_calibration_working[sample_id] = CalibrationState(
                **self.calibration_working.to_dict())

    def disable_sample_calibration(self, sample_id: str) -> None:
        """Drop the sample's override; it returns to the session calibration."""
        self.sample_calibration_working.pop(sample_id, None)
        self.sample_calibration_committed.pop(sample_id, None)

    def _working_calibration(self, sample_id: Optional[str]) -> CalibrationState:
        if sample_id is not None and sample_id in self.sample_calibration_working:
            return self.sample_calibration_working[sample_id]
        return self.calibration_working

    def _committed_calibration(self, sample_id: Optional[str]) -> CalibrationState:
        if sample_id is not None and sample_id in self.sample_calibration_committed:
            return self.sample_calibration_committed[sample_id]
        return self.calibration_committed

    def _calibration_for_sample(self, sample_id: str) -> CalibrationState:
        """Resolver used by analysis: committed per-sample override, else session."""
        return self.sample_calibration_committed.get(
            sample_id, self.calibration_committed)

    def calibration_fields(self, sample_id: Optional[str] = None) -> Dict[str, Any]:
        """The in-scope working calibration (per-sample override or session) as a
        dict, for the GUI to populate its fields."""
        return self._working_calibration(sample_id).to_dict()

    # ----------------------------------------------------------------------
    # Grouping passthroughs (hybrid model)
    # ----------------------------------------------------------------------
    def assign_to_sample(self, item_id: str, sample_id: str) -> None:
        self._rekey_sample_stores(
            self.workspace.assign_to_sample(item_id, sample_id))

    def clear_override(self, item_id: str) -> None:
        self._rekey_sample_stores(self.workspace.clear_override(item_id))

    def assign_many_to_sample(self, item_ids, sample_id: str) -> None:
        """Batch manual move, applied as ONE regroup so the act applies exactly
        the plan `move_regroup_losses` previewed (per-id application can end
        differently -- see `Workspace.assign_many`)."""
        self._rekey_sample_stores(
            self.workspace.assign_many(item_ids, sample_id))

    def clear_many_overrides(self, item_ids) -> None:
        """Batch return-to-auto, one regroup (see `assign_many_to_sample`)."""
        self._rekey_sample_stores(self.workspace.clear_overrides(item_ids))

    def _rekey_sample_stores(self, plan) -> None:
        """Follow a regroup's carries in every controller-owned per-sample store
        that is USER STATE -- the SLS masks, the mode-map picks and recorded
        sets, and the per-sample calibration overrides -- exactly as the
        workspace re-keys its own rows; then prune EVERY sample-keyed store,
        user state and cache alike, to the live sample set. A colliding MASK is
        merged as a set union (owner-ruled 2026-08-13 round 2: a mask means
        "don't trust this point", and the honest merge of two don't-trust lists
        is both lists; over-masking a coincident point stays visible and
        reversible). A colliding CALIBRATION keeps the target's -- one sample
        holds one calibration, so a winner is structurally forced -- and the
        loser is NAMED in the pre-regroup ask (`_calibration_regroup_losses`).

        The CACHES (`results`, the run stamps, the candidate memos) are pruned,
        never carried: a cached fit is a function of the
        sample's membership, which the regroup just changed, and the SLS tab
        already re-runs after a rename. Before this they were neither carried nor
        pruned, and because auto-derived sample ids are deterministic
        ('poly|solv|tempK') an orphaned entry silently came back to life when an
        identity edit was reverted -- a Zimm result under a re-formed id whose
        committed solvent index had changed in between. Entries whose sample no
        longer exists are pruned, so a stale override or a stale fit can never
        reattach to a later re-formed sample id. The roster is pinned by identity
        in `tests/test_workspace.py` (the store-family test), so a store added
        later cannot be left out of this prune unnoticed."""
        for old_sid, new_sid in plan.carries.items():
            for (sid, frac) in [k for k in self.sls_masks if k[0] == old_sid]:
                mask = self.sls_masks.pop((sid, frac))
                existing = self.sls_masks.get((new_sid, frac))
                if existing is None:
                    self.sls_masks[(new_sid, frac)] = mask
                else:
                    existing.masked_angles |= mask.masked_angles
                    existing.masked_concentrations |= mask.masked_concentrations
                    existing.masked_points |= mask.masked_points
            # Mode-map picks carry like the masks: same-label families UNION
            # their pick sets (the mask's set-union ruling -- the honest merge
            # of two pick lists is both); distinct labels append. A duplicated
            # ROLE keeps the TARGET's designation (a display designation, not
            # data -- one deterministic winner, like the calibration). A
            # same-label pair bound to different polarization states keeps the
            # target's binding; the source's other-state picks then match
            # nothing and the family fit notes them.
            for (sid, frac) in [k for k in self.mode_picks if k[0] == old_sid]:
                mp = self.mode_picks.pop((sid, frac))
                existing_mp = self.mode_picks.get((new_sid, frac))
                if existing_mp is None:
                    self.mode_picks[(new_sid, frac)] = mp
                    continue
                # Manual bounds carry with the merge (this branch iterated only
                # the families and silently dropped them, reverting the next
                # map Run to the automatic split). item_ids are globally
                # unique, so a same-key collision means the same run --
                # target-wins, the store family's standing direction.
                for iid, bs in mp.manual_bounds.items():
                    existing_mp.manual_bounds.setdefault(iid, bs)
                for fam in mp.families:
                    target = existing_mp.family(fam.label)
                    # Single ownership survives the merge: a pick value some
                    # TARGET-side family already claims is dropped from the
                    # incoming family before it lands, whatever the labels --
                    # else two families silently fit the same extracted mode.
                    # The target's claim wins (the same target-wins direction
                    # as the calibration and the role below).
                    fam.picks = {p for p in fam.picks
                                 if not any(o.has_pick(p[0], p[1])
                                            for o in existing_mp.families
                                            if o is not target)}
                    # The (picks, bound, state) invariant survives the merge: an emptied
                    # family arrives UNBOUND, free to bind at its next pick.
                    if not fam.picks:
                        fam.bound = False
                        fam.polarization_state = None
                    if target is None:
                        if fam.role and any(f.role == fam.role
                                            for f in existing_mp.families):
                            fam.role = ''            # target's holder wins
                        existing_mp.families.append(fam)
                    elif (target.bound and fam.bound
                          and target.polarization_state
                          != fam.polarization_state):
                        # Two same-label families bound to DIFFERENT states
                        # are never pooled (a VH pick inside a VV family is
                        # the silent re-aim the snapshot ruling forbids) --
                        # the incoming one keeps its binding under the first
                        # free numbered label, visible in the panel.
                        n = 2
                        while existing_mp.family(f'{fam.label} ({n})'):
                            n += 1
                        fam.label = f'{fam.label} ({n})'
                        if fam.role and any(f.role == fam.role
                                            for f in existing_mp.families):
                            fam.role = ''
                        existing_mp.families.append(fam)
                    else:
                        target.picks |= fam.picks
                        # An unbound target ADOPTS a bound source's binding
                        # with its picks -- else the merged picks are dead
                        # (bound=False matches nothing forever).
                        if fam.bound and not target.bound:
                            target.polarization_state = fam.polarization_state
                            target.bound = True
            # Recorded mode sets carry by LIST-APPEND: set_ids are unique and
            # the entries frozen, so there are no merge semantics to invent --
            # a collision simply shows both sets in the (creation-ordered)
            # list, each with its own provenance.
            for (sid, frac) in [k for k in self.mode_sets if k[0] == old_sid]:
                sets = self.mode_sets.pop((sid, frac))
                self.mode_sets.setdefault((new_sid, frac), []).extend(sets)
            for store in (self.sample_calibration_working,
                          self.sample_calibration_committed):
                cal = store.pop(old_sid, None)
                if cal is not None:
                    store.setdefault(new_sid, cal)
        live = set(self.workspace.samples)
        self.sls_masks = {k: v for k, v in self.sls_masks.items()
                          if k[0] in live}
        self.mode_picks = {k: v for k, v in self.mode_picks.items()
                           if k[0] in live}
        self.mode_sets = {k: v for k, v in self.mode_sets.items()
                          if k[0] in live}
        for store in (self.sample_calibration_working,
                      self.sample_calibration_committed):
            for sid in [s for s in store if s not in live]:
                store.pop(sid)
        # The caches. A results key is ('kind', owner_id, ...) where the owner is a
        # sample_id for per-sample results (SLS, Γ-q², the mode map) or an item_id for
        # per-measurement ones (DLS); a regroup never removes a measurement, so an
        # owner that is neither a live sample nor a measurement is a vanished sample.
        # The stamps and the SLS memo share the results' key shape; the DLS memo is
        # keyed (sample_id, fraction, sig).
        def owner_live(owner) -> bool:
            return owner in live or owner in self.workspace.measurements
        for name in ('results', '_form_factor_run_sig', '_dls_run_sig',
                     '_sls_candidate_cache'):
            store = getattr(self, name)
            setattr(self, name, {k: v for k, v in store.items()
                                 if not (isinstance(k, tuple) and len(k) > 1)
                                 or owner_live(k[1])})
        self._dls_candidate_cache = {
            k: v for k, v in self._dls_candidate_cache.items() if k[0] in live}
        # The blankless-run acknowledgement is DROPPED, never carried (owner-ruled
        # 2026-09-17). Every other store this function follows is work the user did --
        # masks, picks, recorded sets, a hand-entered calibration -- where losing it
        # costs real effort. This one is consent to a disclosure, and re-giving it costs
        # one dialog. It is already non-persisted so a reloaded session re-asks; dropping
        # it here is that same principle at the other trigger. Without this the ids are
        # deterministic enough that renaming a sample away and back resurrected consent
        # the re-formed sample had never been given.
        self._unmeasured_blank_ack &= live

    # ---- pre-checks: what would a regroup destroy? (asked BEFORE the act) ----
    def _calibration_regroup_losses(self, plan) -> List['RegroupLoss']:
        """The per-sample calibration overrides a regroup would drop.

        A merged sample holds ONE calibration, so a collision is resolved
        target-wins by construction; the hand-entered loser is NAMED in the ask
        (owner-ruled 2026-08-13, round 2 -- the deliberate exception to the
        results-only ask, because a discarded override silently rescales the
        absolute Mw). A scattered sample's override is dropped with it and
        named the same way. Mirrors `_rekey_sample_stores`' apply exactly:
        same carries, same first-arrival/target-wins order."""
        store = self.sample_calibration_working

        def _kc(sid):
            cal = store.get(sid)
            try:
                v = cal.compute_k_c() if cal is not None else None
            except ValueError:
                v = None
            return float(v) if v is not None else float('nan')

        losses: List[RegroupLoss] = []
        # target sid -> the sid whose calibration will occupy it after the apply
        sim = {t: t for t in set(plan.carries.values()) if t in store}
        for old_sid, new_sid in plan.carries.items():
            if old_sid not in store:
                continue
            holder = sim.get(new_sid)
            if holder is None:
                sim[new_sid] = old_sid
                continue
            losses.append(RegroupLoss(
                sample_id=old_sid, fraction=None, quantity='calibration',
                value=_kc(old_sid), tier='user',
                label='per-sample calibration', kind='merge',
                kept_sample_id=holder, kept_value=_kc(holder),
                kept_tier='user'))
        new_sids = set(plan.member_sid.values())
        for old_sid, s in self.workspace.samples.items():
            if old_sid in new_sids or old_sid not in store:
                continue
            members = [iid for iid in s.member_item_ids()
                       if iid in plan.member_sid]
            if members and len({plan.member_sid[iid] for iid in members}) > 1:
                losses.append(RegroupLoss(
                    sample_id=old_sid, fraction=None, quantity='calibration',
                    value=_kc(old_sid), tier='user',
                    label='per-sample calibration', kind='scatter'))
        return losses

    def commit_regroup_losses(self) -> List['RegroupLoss']:
        """Deliberate-tier values (and hand-entered calibrations) the next
        commit's regroup would drop.

        Read on the GUI main thread before `commit()` -- the same
        `plan_regroup` computation `regroup()` applies, planned from the
        WORKING params the commit is about to adopt, so the ask and the act
        cannot disagree. Empty for the ordinary commit (no identity change, or
        only 'computed' values move)."""
        plan = self.workspace.plan_regroup(use_working=True)
        return plan.losses + self._calibration_regroup_losses(plan)

    def move_regroup_losses(self, item_ids, sample_id: str) -> List['RegroupLoss']:
        """Deliberate-tier values that moving `item_ids` into `sample_id` would
        drop -- the whole gesture simulated at once, and applied the same way
        (`assign_many_to_sample`, one regroup), so the act performs literally
        the plan this previewed."""
        ov = dict(self.workspace.overrides)
        for iid in item_ids:
            ov[iid] = sample_id
        plan = self.workspace.plan_regroup(overrides=ov)
        return plan.losses + self._calibration_regroup_losses(plan)

    def clear_override_regroup_losses(self, item_ids) -> List['RegroupLoss']:
        """Deliberate-tier values that returning `item_ids` to auto-grouping
        would drop (the counterpart of `move_regroup_losses`; applied by
        `clear_many_overrides`, one regroup)."""
        drop = set(item_ids)
        ov = {k: v for k, v in self.workspace.overrides.items() if k not in drop}
        plan = self.workspace.plan_regroup(overrides=ov)
        return plan.losses + self._calibration_regroup_losses(plan)

    def undo_regroup_losses(self) -> List['RegroupLoss']:
        """What the next Undo's regroup would drop -- read on the GUI main
        thread before `undo()`.

        Dialogs guard the destruction of stored values, and an undo that
        scatters a previously-merged sample destroys its carried results
        exactly like a forward scatter (the snapshot restores parameters,
        never results). Planned from the snapshot Undo would restore. Empty
        when Undo would only revert pending edits (no regroup runs) or there
        is nothing to undo."""
        if self.is_dirty() or not self._commit_history:
            return []
        snap = self._commit_history[-1]
        plan = self.workspace.plan_regroup(
            params_override=snap['measurements'])
        return plan.losses + self._calibration_regroup_losses(plan)

    @staticmethod
    def regroup_losses_message(losses) -> str:
        """Dialog body for the confirm before a regroup drops stored values.

        One spelling for EVERY asking surface -- the GUI sites that pre-check
        a `*_regroup_losses` method before their mutating call (grep those
        callers; a hand-written list here dropped a member the day it was
        written). Names per quantity what is kept and what is dropped: a merge
        keeps the higher provenance and the surviving sample wins a tie
        (owner-ruled 2026-08-13); each parenthetical says whose value it WAS
        ("from ..."), never where it ends up. The GUI shows this with
        Yes/Cancel, Cancel as the default, and Cancel applies nothing.
        """
        titles = {**RESULT_QUANTITY_TITLES, 'calibration': 'Calibration (k_c)'}
        tiers = {'user': 'hand-entered', 'picked': 'picked'}
        lines = []
        for x in losses:
            qty = titles.get(x.quantity, x.quantity)
            frac = f" [fraction {x.fraction!r}]" if x.fraction is not None else ''
            if x.kind == 'merge':
                lines.append(
                    f'  • {qty}{frac}: keeps {x.kept_value:g} '
                    f'({tiers.get(x.kept_tier, x.kept_tier)}, '
                    f'from {x.kept_sample_id}) — drops {x.value:g} '
                    f'({tiers.get(x.tier, x.tier)}, from {x.sample_id})')
            elif x.kind == 'move':
                lines.append(
                    f'  • {qty}{frac}: {x.value:g} '
                    f'({tiers.get(x.tier, x.tier)}, from {x.sample_id}) '
                    f'moves to {x.kept_sample_id}')
            else:
                lines.append(
                    f'  • {qty}{frac}: drops {x.value:g} '
                    f'({tiers.get(x.tier, x.tier)}, from {x.sample_id}) — its '
                    f'measurements split into different samples')
        return ('This change regroups measurements whose samples hold stored '
                'results:\n\n' + '\n'.join(lines) +
                '\n\nCancel applies nothing.')

    def new_sample_id(self) -> str:
        """Mint a fresh, collision-proof sample_id for a manually created sample.

        The GUI uses this for "move to new sample": it mints an id here, then
        assigns the selected measurements into it. Kept on the controller so the
        widget never reaches into the workspace directly (invariant: GUI talks
        only to the controller)."""
        return self.workspace.new_sample_id()

    def samples(self) -> List[Sample]:
        return list(self.workspace.samples.values())

    # ----------------------------------------------------------------------
    # Global settings (seed defaults + appearance)
    # ----------------------------------------------------------------------
    def apply_settings(self, settings: SettingsState, persist: bool = True) -> None:
        """Adopt new global settings and (by default) persist them to settings.json.

        Settings only *seed* future runs/controls -- existing results are untouched.
        A write failure (e.g. a read-only location) must not crash the app."""
        self.settings = settings
        if persist:
            try:
                settings.save()
            except OSError:
                pass

    # ----------------------------------------------------------------------
    # Cumulant-method switch: identify + clear results that depend on it
    # ----------------------------------------------------------------------
    def _cumulant_dependent_cache_keys(self, *, order_only: bool = False) -> List[tuple]:
        """Result-cache keys whose value depends on the chosen cumulant method:
        every per-measurement cumulant fit, every DDLS result (each of its VV/VH
        correlograms is a cumulant fit from the seed home, at the Settings method and
        order), plus Gamma-vs-q^2 / D-vs-c results that derived their per-point Gamma
        from the cumulant (gamma_source).

        `order_only` narrows the set to what reads the Settings cumulant ORDER: the
        Gamma-vs-q^2, D-vs-c and DDLS results, which fit every point at it. A
        per-measurement cumulant fit ran at the Correlogram tab's own order box and is
        kept. Every order-cleared kind is also method-cleared, so a change of the order
        and another input together clears the wider set alone.

        Clearing a DDLS result takes nothing from the DDLS tab: its shape models are
        computed from the fit the tab shows (`ddls_shape`'s `result`), which the tab
        keeps and marks out of date."""
        keys = []
        for k, v in self.results.items():
            if k[0] == 'cumulants':
                if not order_only:
                    keys.append(k)
            elif k[0] == 'ddls':
                keys.append(k)
            elif (k[0] in ('gamma_q2', 'conc_extrap')
                  and getattr(v, 'gamma_source', None) == 'cumulant'):
                keys.append(k)
        return keys

    def cumulant_dependent_result_count(self, *, order_only: bool = False) -> int:
        """How many cumulant-derived results currently exist (cumulant fits,
        cumulant Gamma-vs-q^2 / D-vs-c, DDLS, cumulant replicate averages). Used to
        decide whether switching the cumulant method needs a clear-and-warn.

        `order_only` counts only what reads the Settings cumulant order -- the scope a
        change of that order clears (`_cumulant_dependent_cache_keys`)."""
        n = 0
        if not order_only:
            n += sum(1 for r in self.workspace.dls_result_rows.values()
                     if r.method == 'cumulant')
        n += sum(1 for k in self._cumulant_dependent_cache_keys(order_only=order_only)
                 if k[0] in ('gamma_q2', 'conc_extrap', 'ddls'))
        n += sum(1 for r in self.workspace.sample_rh_rows.values()
                 if r.source_kind == 'replicate_avg'
                 and str(r.source_set).split('|')[0] == 'cumulant')
        return n

    def _cumulant_dependent_picks(self, *, order_only: bool = False) -> List[tuple]:
        """(sample_id, fraction, SampleResult) for every PICKED sample Rh whose
        producing analysis a cumulant-method switch invalidates.

        ONE enumeration behind both the Settings confirm's count and the clear
        below, so the dialog and the deed cannot disagree. Matched on the result
        KEY, never on row membership, cache membership or value identity: a pick
        can outlive its Table-2 row (a declined re-run removes the row and spares
        the pick), and the results cache is ephemeral and is not the pick's
        producer. A keyless pick (pre-key session) is spared, fail-closed --
        `result_key_matches` doctrine, the safe branch is "don't clear".

        Picked Gamma-vs-q^2 / D-vs-c values clear by KIND (owner-ruled 2026-08-12):
        nothing durable records which gamma source fed a pick, and every pick the
        app can mint is cumulant-fed (`gamma_source` is not user-selectable), so
        kind implies dependence. If gamma_source ever becomes user-selectable, the
        minted source_id gains the token and this test sharpens with it.

        `order_only` keeps only the picks whose producer read the Settings cumulant
        order: a cumulant replicate average, every picked Gamma-vs-q^2 / D-vs-c value
        (stored run and re-derivation alike fit at it), and a single-run cumulant Rh the
        candidate builder RE-DERIVED at it. A single-run pick of the user's own
        Correlogram fit, which ran at its order box, is told apart by its source id
        (`_stored_fit_source_id`) and kept."""
        picks = []
        for sid, s in self.workspace.samples.items():
            for frac, r in s.fraction_results.items():
                if r.rh_source != 'picked' or r.rh_nm is None:
                    continue
                replicate = (r.rh_kind == 'dls_replicate_avg'
                             and r.rh_source_id == replicate_rh_key(
                                 f'cumulant|{frac}')[1])
                pooled = r.rh_kind in ('dls_gamma_q2', 'dls_conc_extrap')
                single = r.rh_kind == 'dls_cumulant'
                if order_only:
                    single = single and not _is_stored_fit_source_id(r.rh_source_id)
                dependent = replicate or pooled or single
                if dependent:
                    picks.append((sid, frac, r))
        return picks

    def cumulant_dependent_picked_count(self, *, order_only: bool = False) -> int:
        """How many PICKED sample Rh values a cumulant-method switch would clear.
        Read by the Settings confirm so it can name them before the clear runs --
        and it must gate the dialog alongside `cumulant_dependent_result_count`,
        whose rows/cache can be empty while a dangling pick persists. `order_only`
        is the Settings-order scope, as there."""
        return len(self._cumulant_dependent_picks(order_only=order_only))

    def clear_cumulant_dependent_results(self, *, order_only: bool = False) -> int:
        """Drop every cumulant-derived result -- the cumulant fits, the cumulant
        Gamma-vs-q^2 / D-vs-c runs, DDLS and the cumulant replicate averages -- so the
        workspace stays consistent after the cumulant method changes. NEVER touches
        distributions, SLS, traces, single/double/KWW (cumulant is only their seed), a
        hand-entered (user) Rh, or a picked Rh of an unaffected kind (distribution peak).
        Returns the number of result-cache entries removed.

        THE REPLICATE-AVERAGE HALF IS DELEGATED TO `_clear_replicate_rh`, the exact
        inverse of the write. A cumulant replicate average is identified by its keyed
        Table-2 row and the value promoted from it, never by its display prose: the
        same value carries two spellings -- one written by the promotion, one by a
        passive Cross-Sample refresh -- so a prose test finds only the one it was
        written against.

        A PICKED Rh whose producing analysis is invalidated here is cleared with it
        (step 5): the switch is a deliberate act, and the Settings dialog confirms
        first, with Cancel backing the whole switch out -- the tier rule lives in
        the `SampleResult` docstring.

        `order_only` clears only what reads the Settings cumulant ORDER -- the scope of
        a change to that order: the Gamma-vs-q^2, D-vs-c and DDLS results, the cumulant
        replicate averages, and the picks those and the tier-1 re-derivation produced. It
        never reaches the Correlogram tab's own fits (they ran at its order box). The same
        enumerations, narrowed, so the Settings confirm's count stays the set cleared.
        """
        # 1. result cache (cumulant fits, DDLS, cumulant-sourced gamma_q2 / conc_extrap)
        dead = self._cumulant_dependent_cache_keys(order_only=order_only)
        # remember which sample Rh rows the cleared multi-step results fed
        rh_rows_to_drop = set()
        for k in dead:
            if k[0] == 'gamma_q2':
                rh_rows_to_drop.add(('gamma_q2', k[1], k[2]))
            elif k[0] == 'conc_extrap':
                rh_rows_to_drop.add(('conc_extrap', k[1], k[2]))
            self.results.pop(k, None)
            self._dls_run_sig.pop(k, None)      # keep the run stamps in step
        # 2. per-measurement cumulant Summary rows (fitted at their own order box, so a
        #    Settings-order change leaves them)
        if not order_only:
            self.workspace.dls_result_rows = {
                key: r for key, r in self.workspace.dls_result_rows.items()
                if r.method != 'cumulant'}
        # 3. sample Rh rows fed by the cleared multi-step results (gamma_q2 /
        #    conc_extrap). The cumulant REPLICATE rows are handled in step 4, which
        #    deletes the row and the promotion together.
        self.workspace.sample_rh_rows = {
            key: r for key, r in self.workspace.sample_rh_rows.items()
            if not (r.source_kind in ('gamma_q2', 'conc_extrap')
                    and (r.source_kind, r.sample_id, r.fraction) in rh_rows_to_drop)}
        # 4. cumulant replicate averages: row + promoted Rh, via the inverse of the
        #    write. `source_set` is composed f'{method}|{fraction}', so the split-off
        #    method token plus `row.fraction` recompose the key (`upsert_sample_rh_row`).
        stale_reps = [(r.sample_id, r.fraction, str(r.source_set).split('|')[0])
                      for r in self.workspace.sample_rh_rows.values()
                      if r.source_kind == 'replicate_avg'
                      and str(r.source_set).split('|')[0] == 'cumulant']
        for sid, frac, method in stale_reps:
            if sid in self.workspace.samples:
                # Clears a 'computed' promotion of this row; never a hand-entered
                # ('user') Rh, nor a 'computed' one written by another path that
                # shares the field but not the row -- see `_clear_replicate_rh`.
                # A PICKED promotion is step 5's job, matched on its key.
                self._clear_replicate_rh(sid, frac, method)
            else:
                # An orphaned row (its sample regrouped away before `prune_result_rows`
                # ran) has no SampleResult to clear, and `_clear_replicate_rh` would
                # raise on the lookup. Drop the row, which is all there is to drop.
                self.workspace.remove_sample_rh_row(
                    sid, 'replicate_avg', f'{method}|{frac}')
        # 5. picked promotions whose producing analysis was invalidated go with it
        #    (the deliberate-act arm of the tier rule -- SampleResult docstring; the
        #    Settings dialog confirmed before this ran). Key-matched, so it also
        #    catches a pick whose Table-2 row was already removed by a declined
        #    re-run, which step 4's row-driven loop can never visit.
        for _sid, _frac, r in self._cumulant_dependent_picks(order_only=order_only):
            reset_result_quantity(r, 'rh')
        return len(dead)

    # ----------------------------------------------------------------------
    # Skipped-lag-channels change: identify + clear every DLS-derived result
    # ----------------------------------------------------------------------
    # The result-cache kinds a change of the skipped lag channels invalidates: every DLS
    # fit of a correlogram, and every result built on one -- the live Mode Map run and what
    # is read off its points among them. The scope is owner-ruled 2026-09-23; its home is
    # `clear_skip_dependent_results`' docstring.
    _SKIP_DEPENDENT_KINDS = ('cumulants', 'single_exp', 'double_exp', 'kww', 'distribution',
                             'gamma_q2', 'conc_extrap', 'ddls',
                             'mode_map', 'mode_family', 'mode_amplitude')
    # The Summary Table 2 row kinds the same change clears.
    _SKIP_DEPENDENT_ROW_KINDS = ('gamma_q2', 'conc_extrap', 'mode_family', 'replicate_avg')

    def _skip_dependent_cache_keys(self) -> List[tuple]:
        """Result-cache keys whose value depends on the Settings skipped lag channels:
        every per-measurement DLS fit of every method, the Gamma-vs-q^2 and D-vs-c runs
        whatever their Gamma source, DDLS, and the live Mode Map run with the mode-family
        fits and amplitude report read off its points. A superset of
        `_cumulant_dependent_cache_keys` in either of its scopes."""
        return [k for k in self.results if k[0] in self._SKIP_DEPENDENT_KINDS]

    def _skip_dependent_rh_rows(self) -> List[SampleRhRow]:
        """Every Summary Table 2 row a skip change clears: the Gamma-vs-q^2, D-vs-c and
        mode-family rows, and the replicate averages of every method. ONE enumeration
        behind the Settings confirm's count and the clear."""
        return [r for r in self.workspace.sample_rh_rows.values()
                if r.source_kind in self._SKIP_DEPENDENT_ROW_KINDS]

    def _skip_dependent_pooled_results(self) -> set:
        """(kind, sample_id, fraction) of every Gamma-vs-q^2 and D-vs-c result a skip
        change clears: a stored run, its Summary Table 2 row, or both. A reopened session
        holds the row without the run (runs are runtime state), and the clear removes it
        all the same, so the count reads the union the clear acts on."""
        out = {k for k in self._skip_dependent_cache_keys()
               if k[0] in ('gamma_q2', 'conc_extrap')}
        out |= {(r.source_kind, r.sample_id, r.fraction)
                for r in self._skip_dependent_rh_rows()
                if r.source_kind in ('gamma_q2', 'conc_extrap')}
        return out

    def skip_dependent_result_count(self) -> int:
        """How many DLS-derived results a change of the skipped lag channels clears: every
        Summary row of every method, the Gamma-vs-q^2 and D-vs-c results, the DDLS
        results, the live Mode Map runs and their mode-family rows, and the replicate
        averages of every method -- read off the enumerations the clear acts on. The
        Settings confirm reads it, beside `skip_dependent_picked_count`, before
        `clear_skip_dependent_results` runs. A per-measurement fit is counted by its
        Summary row, as the cumulant arm counts it, and a map run stands for the family
        fits and amplitude report read off it."""
        keys = self._skip_dependent_cache_keys()
        n = len(self.workspace.dls_result_rows)
        n += len(self._skip_dependent_pooled_results())
        n += sum(1 for k in keys if k[0] in ('ddls', 'mode_map'))
        n += sum(1 for r in self._skip_dependent_rh_rows()
                 if r.source_kind in ('mode_family', 'replicate_avg'))
        return n

    def _skip_dependent_picks(self) -> List[tuple]:
        """(sample_id, fraction, SampleResult) for every PICKED sample Rh whose producer a
        change of the skipped lag channels invalidates: every DLS kind. ONE enumeration
        behind the Settings confirm's count and the clear.

        Matched on the key's kind, so a DLS kind minted later is cleared by default --
        over-inclusion is the safe direction. A pick taken from a mode family goes too
        (owner-ruled 2026-09-23): the change clears the row it was taken from, and the
        estimator switch, which moves only its ±, already clears it. A keyless pick and a
        hand-entered Rh are spared as `_cumulant_dependent_picks` spares them."""
        picks = []
        for sid, s in self.workspace.samples.items():
            for frac, r in s.fraction_results.items():
                if r.rh_source != 'picked' or r.rh_nm is None:
                    continue
                if (r.rh_kind or '').startswith('dls_'):
                    picks.append((sid, frac, r))
        return picks

    def skip_dependent_picked_count(self) -> int:
        """How many PICKED sample Rh values a change of the skipped lag channels would
        clear. Gates the Settings confirm alongside `skip_dependent_result_count`, whose
        rows and cache can be empty while a dangling pick persists."""
        return len(self._skip_dependent_picks())

    def clear_skip_dependent_results(self) -> int:
        """Drop every DLS-derived result after the Settings skipped lag channels change
        (owner-ruled 2026-09-23): every DLS fit reads that setting, so every result built
        from a fit of a correlogram is stale. Returns the number of result-cache entries
        removed.

        CLEARED: the per-measurement fits of every method and their Summary rows; the
        Gamma-vs-q^2 and D-vs-c runs whatever their Gamma source, with every Summary
        Table 2 row of theirs; DDLS; the live Mode Map run, which `run_mode_map` extracted
        at the old setting, with what is read off its points -- the mode-family fits and
        their Table 2 rows (`run_mode_family_fits`) and the amplitude report
        (`mode_amplitude_report`); the replicate averages of every method, row and
        promotion together (`_clear_replicate_rh`, as the cumulant arm does); and every
        picked Rh of a DLS kind, a mode family's included (`_skip_dependent_picks`). Until
        the map is re-run, recording a mode set refuses for want of one (`record_mode_set`).

        KEPT, each for a stated reason:
        - The recorded mode sets, and the partition and per-mode static results computed
          from one (`run_mode_partition`, `run_mode_static`): they read the recorded
          snapshot, never the live map, and a recorded set is the user's frozen record.
        - The mode families picked on the map (`mode_picks`): the user's assignments, kept
          as such. They are matched by value, and the new setting moves the modes, so on
          the double-exponential source a re-run map orphans them, which the Families panel
          shows; a distribution source re-matches only where its peaks stay on the grid.
        - SLS, and every hand-entered value.

        The scope contains the cumulant arm's in either scope, so a skip change made with
        another cumulant input clears this set alone, after one confirm."""
        # 1. result cache, with each result's run stamps
        dead = self._skip_dependent_cache_keys()
        for k in dead:
            self.results.pop(k, None)
            self._dls_run_sig.pop(k, None)
            self._distribution_stamps.pop(k, None)
        # 2. the Summary rows, every method
        self.workspace.dls_result_rows = {}
        # 3. the Summary Table 2 rows of a cleared kind: a pooled run's (stored run or not)
        #    and a mode family's outright; a replicate average's row and the Rh promoted
        #    from it together, via the inverse of the write, an orphaned row (its sample
        #    regrouped away) dropped alone -- the cumulant arm's step 4.
        for r in self._skip_dependent_rh_rows():
            if r.source_kind != 'replicate_avg':
                self.workspace.remove_sample_rh_row(r.sample_id, r.source_kind,
                                                    r.source_set)
                continue
            method = str(r.source_set).split('|')[0]
            if r.sample_id in self.workspace.samples:
                self._clear_replicate_rh(r.sample_id, r.fraction, method)
            else:
                self.workspace.remove_sample_rh_row(
                    r.sample_id, 'replicate_avg', f'{method}|{r.fraction}')
        # 4. picked promotions of a DLS kind, key-matched (the cumulant arm's step 5)
        for _sid, _frac, r in self._skip_dependent_picks():
            reset_result_quantity(r, 'rh')
        return len(dead)

    # ----------------------------------------------------------------------
    # SE-estimator switch: identify + clear results whose ± depends on it
    # ----------------------------------------------------------------------
    # Cached analyses that carry a regression standard error routed through the
    # selectable covariance estimator (uncertainty.linear_fit / multilinear_fit).
    # Switching HC3 <-> OLS changes only the ±, never the point estimate.
    _SE_BEARING_KINDS = ('zimm', 'debye', 'guinier', 'cal_free_a2', 'single_angle_ladder',
                         'form_factor',
                         'gamma_q2', 'conc_extrap', 'mode_family', 'mode_static', 'ddls')

    def se_dependent_result_count(self) -> int:
        """How many cached results carry a regression SE whose value depends on the
        chosen covariance estimator (HC3 vs classical OLS). Used to decide whether
        switching the estimator needs a clear-and-warn. Point estimates (Mw, Rg, D,
        Rh) are unaffected — only the ± changes."""
        return sum(1 for k in self.results if k[0] in self._SE_BEARING_KINDS)

    def _se_dependent_picks(self) -> List[tuple]:
        """(sample_id, fraction, quantity, SampleResult) for every PICKED sample
        value whose ± was routed through the selectable covariance estimator.

        ONE enumeration behind both the Settings confirm's count and the clear
        below. The predicate is the estimator tag itself -- `<q>_se_estimator` is
        set exactly when the ± came through the HC3/OLS dispatch -- so no kind map
        can drift out of step with the writers: it includes a picked Zimm/Berry/
        Debye/Guinier/form-factor value and a picked regression-sourced Rh, and it
        spares a replicate-average SEM, a distribution peak, a ±-withheld pick and
        every hand-entered value (their tag is None) by the same test.

        Deliberately BLANKET (owner-ruled): any estimator switch clears every
        estimator-routed pick, without comparing the tag to the incoming
        estimator -- so a pick whose tag already names the target (possible when
        a session tagged on one machine is opened under another machine's
        settings) is cleared too. The simpler always-clears rule was chosen over
        a tag-aware spare; this is broader than invariant 8 strictly requires,
        and the D2 dialog words the count accordingly ("estimator-routed")."""
        picks = []
        for sid, s in self.workspace.samples.items():
            for frac, r in s.fraction_results.items():
                for q in RESULT_QUANTITIES:
                    if (getattr(r, f'{q}_source') == 'picked'
                            and getattr(r, f'{q}_se_estimator') is not None):
                        picks.append((sid, frac, q, r))
        return picks

    def se_dependent_picked_count(self) -> int:
        """How many PICKED sample values an estimator switch would clear. Read by
        the Settings confirm so it can name them before the clear runs -- and it
        must gate the dialog alongside `se_dependent_result_count`, whose cache is
        empty after a session reload while the picks persist."""
        return len(self._se_dependent_picks())

    def clear_se_dependent_results(self) -> int:
        """Drop cached SE-bearing analyses (Zimm/Berry, Debye, Guinier, calibration-
        free A2, Γ-vs-q², D-vs-c, DDLS) so they recompute under the newly-selected
        covariance estimator, and clear every PICKED sample value whose ± the
        de-selected estimator produced. A pick is inert to the passive refresh
        that would re-tag its ±, so keeping it would freeze a ± from a de-selected
        estimator on the workspace indefinitely (invariant 8 Clause A); the switch
        is a deliberate act and the Settings dialog confirms first, Cancel backing
        the whole switch out -- the tier rule lives in the `SampleResult`
        docstring. Hand-entered values and 'computed' point estimates are
        untouched (a 'computed' ± re-tags on the next run or refresh).
        Returns the number of cache entries removed."""
        dead = [k for k in self.results if k[0] in self._SE_BEARING_KINDS]
        for k in dead:
            self.results.pop(k, None)
            self._form_factor_run_sig.pop(k, None)   # keep the form-factor stamps in step
            self._dls_run_sig.pop(k, None)           # ... and the Γ-q² / D-vs-c ones
        for _sid, _frac, q, r in self._se_dependent_picks():
            reset_result_quantity(r, q)
        return len(dead)

    # ----------------------------------------------------------------------
    # DLS analysis orchestration
    # ----------------------------------------------------------------------
    def _seed_cumulant_kwargs(self, kw: dict) -> dict:
        """Seed one `fit_cumulants` call's kwargs from Settings -- the ONE home.

        Every controller path that fits a cumulant seeds from here: the Correlogram tab's
        own fit, the tier-1 Rh candidate, the Gamma-vs-q^2 / D-vs-c points and pooled fits,
        DDLS and replicate averaging. They share no other code, so a Settings value one of
        them read and another missed is how the promoted number and the shown number came
        to be two fits of one correlogram (enforced by
        `test_every_dls_fit_call_site_reads_its_inputs_from_their_one_home`).

        `setdefault` throughout, because Settings SEED and never override: an explicit
        caller value -- the Correlogram tab's order box, an API call, a test -- still wins.
        The ORDER is seeded here too: every path without an order control of its own fits
        at the Settings order, so the Settings order means one thing on all of them. The
        weighting is looked up only when the caller gave none, so an explicit value never
        runs through the policy mapping and its refusal of an unknown token.

        Returns the kwargs under `fit_cumulants`' own names, `order` included."""
        kw.setdefault('order', self.settings.cumulant_order)
        kw.setdefault('skip_initial_channels', self.settings.skip_initial_channels)
        kw.setdefault('method', self.settings.cumulant_method)
        kw.setdefault('fit_cutoff', self.settings.cumulant_fit_cutoff)
        if 'weighting' not in kw:
            kw['weighting'] = _cumulant_weighting_request(self.settings.cumulant_weighting)
        return kw

    @staticmethod
    def _pooled_cumulant_kwargs(seeds: dict) -> dict:
        """`_seed_cumulant_kwargs`' values under the names the Gamma-vs-q^2 and D-vs-c
        engines take them by -- a renaming, never a second seed."""
        return dict(skip_initial_channels=seeds['skip_initial_channels'],
                    cumulant_method=seeds['method'],
                    cumulant_fit_cutoff=seeds['fit_cutoff'],
                    cumulant_weighting=seeds['weighting'],
                    cumulant_order=seeds['order'])

    def analysis_region_kwargs(self, item_id: str, method: str,
                               region: Optional[RegionState]) -> Dict:
        """The engine kwargs one `method` fit of measurement `item_id` reads from the DLS
        analysis region -- the one reading every surface and path shares
        (`app/analysis_region.py` states the rule).

        Reads the measurement's raw correlogram and the Settings the fit itself reads
        (the skipped channels, and the cumulant method, which decides between the two
        baseline inputs), so the channels a span is counted over are the fit's own. A
        measurement no longer in the workspace -- a ticked id can outlive it between a
        removal and the selection's prune -- gets the window alone, so a staleness check
        compares rather than raises. `region` None is the unticked box: nothing."""
        lm = self.workspace.measurements.get(item_id)
        if lm is None:
            return region_fit_kwargs(np.zeros(0), np.zeros(0), method,
                                     None if region is None else RegionState(
                                         region.tau_min_s, region.tau_max_s,
                                         window_touched=region.window_touched))
        return region_fit_kwargs(
            np.asarray(lm.raw['delay_times_s'], dtype=float),
            np.asarray(lm.raw['correlogram'], dtype=float), method, region,
            cumulant_method=self.settings.cumulant_method,
            skip_initial_channels=self.settings.skip_initial_channels)

    @staticmethod
    def region_note(region: Optional[RegionState]) -> str:
        """The sentence a checked run records in its notes (`describe_region`): '' for
        an unticked box, which runs on each run's own defaults -- the silent default."""
        return describe_region(region)

    def _region_inputs(self, lms, method: str,
                       region: Optional[RegionState]) -> Dict[str, Dict]:
        """{item_id: that measurement's region kwargs} for a checked path, {} unticked."""
        if region is None:
            return {}
        return {lm.item_id: self.analysis_region_kwargs(lm.item_id, method, region)
                for lm in lms}

    def _cumulant_fit_stamp(self, item_id: str, seeds: dict) -> tuple:
        """What a stored per-measurement cumulant fit read from OUTSIDE its own tab: the
        measurement's committed params and the Settings seeds the fit took -- the skipped
        channels, method, cutoff and weighting request. Not the order or the region,
        which were the tab's own (its order box, its handles) and are not a Setting that
        could move under it."""
        return (self.committed_signature([item_id]), seeds['skip_initial_channels'],
                seeds['method'], seeds['fit_cutoff'], seeds['weighting'])

    def _stored_cumulant_fit(self, item_id: str):
        """The user's own Correlogram cumulant fit of `item_id` while it is FRESH -- its
        stamp still equals what the same committed params and Settings give now -- else
        None. The `_stored_dls_run` shape, one measurement wide: stale when a commit or a
        Settings change moved an input the fit read, and unstamped reads as stale.
        A Settings value the seed home refuses (a hand-edited weighting) reads as stale
        too, so the builder re-derives and isolates that refusal as before."""
        key = ('cumulants', item_id)
        res = self.results.get(key)
        stamp = self._dls_run_sig.get(key)
        if res is None or stamp is None:
            return None
        try:
            now = self._cumulant_fit_stamp(item_id, self._seed_cumulant_kwargs({}))
        except ValueError:
            return None
        return res if stamp[1] == now else None

    def run_cumulants(self, item_id: str, order: Optional[int] = None, **kw):
        m = self.workspace.measurements[item_id].build()
        if order is not None:                   # an explicit order (the tab's box) wins
            kw['order'] = order
        # Settings-seeded from the one home -- `setdefault`, so an explicit caller value
        # still wins. The committed mean count rate and the parser's per-channel sample
        # times -- the inputs the nonlinear arm's sourced residual weight reads -- ride on
        # `m` itself (`LoadedMeasurement.build()`), so nothing is threaded here for them;
        # `CumulantResult.weighting` and `.sample_time_provenance` disclose what entered.
        self._seed_cumulant_kwargs(kw)
        key = ('cumulants', item_id)
        try:
            res = dls_engine.fit_cumulants(m, **kw)
        except Exception:
            # A refused re-Run leaves no fit behind, as on the Correlogram tab, which drops
            # its own copy: the tier-1 candidate must not go on promoting the fit the user
            # just tried to replace.
            self.results.pop(key, None)
            self._dls_run_sig.pop(key, None)
            raise
        # STAMP-BEFORE-PUBLISH, as run_gamma_q2: the tier-1 candidate reads the fit only
        # while its stamp matches.
        self._cumulant_fit_counter += 1
        self._dls_run_sig[key] = (self._cumulant_fit_counter,
                                  self._cumulant_fit_stamp(item_id, kw))
        self.results[key] = res
        self._snapshot_parametric(item_id, 'cumulant', res)
        return res

    def run_single_exponential(self, item_id: str, **kw):
        m = self.workspace.measurements[item_id].build()
        kw.setdefault('skip_initial_channels', self.settings.skip_initial_channels)
        res = dls_engine.fit_single_exponential(m, **kw)
        self.results[('single_exp', item_id)] = res
        self._snapshot_parametric(item_id, 'single', res)
        return res

    def run_double_exponential(self, item_id: str, **kw):
        m = self.workspace.measurements[item_id].build()
        kw.setdefault('skip_initial_channels', self.settings.skip_initial_channels)
        res = dls_engine.fit_double_exponential(m, **kw)
        self.results[('double_exp', item_id)] = res
        self._snapshot_parametric(item_id, 'double', res)
        return res

    def run_kww(self, item_id: str, **kw):
        m = self.workspace.measurements[item_id].build()
        kw.setdefault('skip_initial_channels', self.settings.skip_initial_channels)
        res = dls_engine.fit_kww(m, **kw)
        self.results[('kww', item_id)] = res
        self._snapshot_parametric(item_id, 'kww', res)
        return res

    def _seed_distribution_kwargs(self, kw: dict, method: str) -> dict:
        """Seed one distribution inversion's kwargs from Settings -- the ONE home.

        Every controller path that runs a CONTIN/NNLS/lognormal inversion seeds
        from here, so a Settings value the inversion reads cannot reach one path
        and silently miss another -- a real hazard rather than a theoretical one,
        because those paths share no other code (enforced by
        `test_every_inversion_call_site_seeds_from_the_one_home`).

        `setdefault` throughout, because Settings SEED and never override: an
        explicit caller kwarg -- the Distribution tab's per-run fields, an API
        call, a test -- still wins over the seed.

        The method split mirrors the fitter dispatch below: the alpha sweep and
        its selection method are CONTIN's alone, and NNLS/lognormal would refuse
        them. The residual weighting is NOT in that split: all three arms take
        `weighting`, and a weight that reached only CONTIN would give the same
        measurement two analyses depending on which arm the user picked.
        """
        kw.setdefault('rh_min_nm', self.settings.rh_grid_min_nm)
        kw.setdefault('rh_max_nm', self.settings.rh_grid_max_nm)
        kw.setdefault('n_grid', self.settings.rh_grid_points)
        kw.setdefault('skip_initial_channels', self.settings.skip_initial_channels)
        # Not `setdefault`: its default is evaluated eagerly, so the mapping's refusal of an
        # unrecognised token fired even when the caller's own choice was about to win.
        if 'weighting' not in kw:
            kw['weighting'] = distribution_weighting_request(
                self.settings.distribution_weighting)
        if method not in ('nnls', 'lognormal'):
            kw.setdefault('alpha_min', self.settings.lcurve_alpha_min)
            kw.setdefault('alpha_max', self.settings.lcurve_alpha_max)
            kw.setdefault('alpha_method', self.settings.contin_alpha_method)
            kw.setdefault('ftest_prob_reject',
                          self.settings.contin_ftest_prob_reject)
            kw.setdefault('gcv_gamma', self.settings.contin_gcv_gamma)
        return kw

    def run_distribution(self, item_id: str, method: str = 'contin', **kw):
        m = self.workspace.measurements[item_id].build()
        self._seed_distribution_kwargs(kw, method)
        if method == 'nnls':
            res = dls_engine.fit_nnls(m, **kw)
        elif method == 'lognormal':
            res = dls_engine.fit_lognormal(m, **kw)
        else:
            res = dls_engine.fit_contin(m, **kw)
        key = ('distribution', item_id, method)
        self.results[key] = res
        # Bump the never-reused version stamp so a re-fit moves dls_run_signature
        # (guards the DLS candidate cache against id() reuse — see __init__).
        self._distribution_stamp_counter += 1
        self._distribution_stamps[key] = self._distribution_stamp_counter
        self._snapshot_distribution(item_id, method, res)
        return res

    def reweight_distribution(self, item_id: str, distribution, weighting: str,
                              model: Optional[str] = None, *,
                              on_invalid: str = 'flag', **cutoffs):
        """Re-express an intensity distribution as volume/surface/number-weighted (Phase 1b).

        `distribution` is an already-computed intensity ``DistributionResult`` (from
        ``run_distribution``); the optics (medium refractive index, wavelength, and the
        hand-entered solute refractive index) come from the measurement's COMMITTED params,
        so the reweighting uses exactly the numbers the distribution was fit with. Pure
        pass-through to ``analysis.optical_model`` (controller-mediated; the engine stays
        Qt-free). Report-only -- the result carries no ± (invariant 8), so it is
        deliberately NOT cached in ``self.results`` alongside the SE-bearing fits and does
        not enter ``_SE_BEARING_KINDS``; the GUI holds it and passes it to export.

        Raises ``OpticalModelError`` for a non-spherical model, RDG/Mie without a solute RI,
        or ``on_invalid='refuse'`` with out-of-range bins.
        """
        p = self.workspace.measurements[item_id].committed_params
        return optical.reweight_distribution(
            distribution, weighting, model,
            n_medium=p['solvent_refractive_index'],
            wavelength_nm=p['wavelength_nm'],
            n_solute=p.get('solute_refractive_index'),
            on_invalid=on_invalid, **cutoffs)

    def dls_sample_rows(self, sample_id: str, kind: str,
                        fraction: Optional[str] = None):
        """Metadata-only enumeration for the Γ vs q² / D vs c tables — **no fitting**.
        One dict per DLS measurement of the sample, so the tabs
        can populate cheaply on load/commit without silently computing an analysis. The
        Γ/D values themselves are produced only by an explicit Run (see `run_gamma_q2` /
        `run_concentration_extrapolation`, whose results carry per-point values).

        `kind` is 'gamma_q2' (rows sorted by angle) or 'conc_extrap' (sorted by
        concentration). Each dict: item_id, angle_deg, concentration_g_per_mL,
        mw_fraction, reason, ok. `reason` explains eligibility so the GUI can gray a
        row and say *why*:
          'ok'             — buildable (parameters confirmed); tickable.
          'unbuilt'        — parameters not confirmed yet (`build()` fails).
          'other_fraction' — a different Mw fraction than the focused one (a single fit
                             must not mix fractions), shown grayed rather than hidden.
        `ok == (reason == 'ok')`. Non-ok rows sort after the ok rows.
        """
        lms = self.workspace.sample_measurements(sample_id, 'dls')
        rows = []
        for lm in lms:
            p = lm.committed_params
            lm_frac = p.get('mw_fraction')
            # Combination doctrine rule 1 (partition), core/data_models.py:
            # fraction=None is the literal untagged series, never a wildcard.
            if lm_frac != fraction:
                reason = 'other_fraction'
            else:
                try:
                    lm.build()
                    reason = 'ok'
                except Exception:  # noqa: BLE001 — unbuilt is the disclosed reason
                    reason = 'unbuilt'
            rows.append({
                'item_id': lm.item_id,
                'angle_deg': p.get('angle_deg'),
                'concentration_g_per_mL': p.get('concentration_g_per_mL'),
                'mw_fraction': lm_frac,
                'reason': reason,
                'ok': reason == 'ok',
            })

        def _sort_key(r):
            ok_rank = 0 if r['ok'] else 1
            if kind == 'gamma_q2':
                v = r['angle_deg']
            else:
                v = r['concentration_g_per_mL']
            return (ok_rank, v if v is not None else float('inf'))
        rows.sort(key=_sort_key)
        return rows

    def dls_pending_temperature_siblings(self, sample_id: str) -> int:
        """Count DLS measurements of the same polymer/solvent as `sample_id` whose
        temperature is **not yet confirmed** — they group into a separate NaN-temperature
        sample (see `sample_group_key`) and so are invisible from this sample's tabs.
        The GUI surfaces this as a one-line note so the user knows why some siblings are
        missing. Returns 0 if the sample is unknown."""
        s = self.workspace.samples.get(sample_id)
        if s is None:
            return 0
        # Answers "same polymer/solvent?", so it uses THE identity-comparison function
        # rather than a hand-rolled .strip().lower(). This is the same question
        # make_sample_key and sample_group_key answer, and another normalization here would
        # agree only while every name in play is ASCII (.lower() and .casefold() part
        # company on ss/SS).
        poly = casefold_identity_name(s.polymer_name or '')
        solv = casefold_identity_name(s.solvent_name or '')
        members = set(s.dls_item_ids)
        n = 0
        for iid, lm in self.workspace.measurements.items():
            if lm.kind != 'dls' or iid in members:
                continue
            p = lm.committed_params
            if p.get('temperature_K') is not None:      # temperature confirmed -> grouped
                continue
            if (casefold_identity_name(p.get('polymer_name', '') or '') == poly
                    and casefold_identity_name(p.get('solvent_name', '') or '') == solv):
                n += 1
        return n

    def _dls_run_points(self, sample_id: str, fraction: Optional[str],
                        region: Optional[RegionState] = None):
        """Per-measurement cumulant points for one fraction of a sample, computed at
        **Run** time (not on focus). One cumulant fit per buildable measurement at the
        Settings order, on each measurement's own region inputs when a checked run passes
        `region` -- the same fit the line uses, input for input -- yielding Γ, q², D_app,
        and a `quality` tag the GUI shows so the user can vet/untick a bad point in
        place. Failure-tolerant: a fit that raises is tagged 'fit_failed' and its reason
        rides the point (`fit_error`), so a checked run can say why it left it out.

        `quality`: 'ok' | 'fit_failed' (Γ non-finite) | 'nonpositive_gamma' (Γ ≤ 0) |
        'nonpositive_q' (q ≤ 0) | 'high_pdi' (PDI > the cumulant validity limit — fit ran
        but is unreliable). At order 1 the fit reports no PDI, so no point is judged on
        one: it is tagged on its other criteria alone.
        Returns a list of dicts aligned with the fraction's buildable measurements.

        `d_app` comes from `_d_from_gamma`, the same helper the engines use, and NOT from
        an inline `gamma / q2`. This table is where the user decides whether to trust a
        point, so it must agree with the fit it belongs to: the engine drops a
        non-positive-Γ measurement from the D-vs-c extrapolation, and tagging such a point
        `'ok'` here would show it as a healthy ticked row carrying a NEGATIVE D — the
        vetting surface contradicting the fit. `'nonpositive_gamma'` is its own tag rather
        than folded into `'fit_failed'`
        because the fit did converge; what it returned is unphysical (rule 10: the
        surfacing paths get swept with the engine).

        This does NOT make the engine's own drop redundant: the engine still has to
        handle a point it is handed, whoever calls it. Both layers, deliberately: this one
        so the user SEES the bad point before running, that one so the answer is right
        whoever calls it."""
        points = []
        for lm in self.workspace.sample_measurements(sample_id, 'dls'):
            p = lm.committed_params
            # Combination doctrine rule 1 (partition), core/data_models.py:
            # fraction=None is the literal untagged series, never a wildcard.
            if p.get('mw_fraction') != fraction:
                continue
            try:
                m = lm.build()
            except Exception:  # noqa: BLE001 — unconfirmed params drop out
                continue                        # unconfirmed params -> not a point
            fit_error = ''
            try:
                seeds = self._seed_cumulant_kwargs(
                    self.analysis_region_kwargs(lm.item_id, 'cumulant', region))
                cu = dls_engine.fit_cumulants(m, **seeds)
                gamma, pdi = float(cu.gamma_s_inv), float(cu.pdi)
            except Exception as exc:  # noqa: BLE001 — NaN gamma is the disclosed failure
                gamma, pdi = float('nan'), float('nan')
                fit_error = str(exc)
            try:
                q = float(phys.scattering_vector_q_m(
                    m.angle_deg, m.wavelength_nm, m.solvent_refractive_index))
            except Exception:  # noqa: BLE001 — NaN q is the disclosed failure
                q = float('nan')
            if not math.isfinite(gamma):
                quality = 'fit_failed'
            elif gamma <= 0.0:
                quality = 'nonpositive_gamma'
            elif not (math.isfinite(q) and q > 0.0):
                quality = 'nonpositive_q'
            elif not _dls_pdi_check(pdi)[0]:
                # The engine's own polydispersity rule, PDI above the limit -- never the PDI
                # cell's wider `pdi_valid`, which also rejects an order-1 fit's missing PDI
                # and the ~ -1e-15 a clean exponential returns; neither is a PDI past it.
                quality = 'high_pdi'
            else:
                quality = 'ok'
            q2 = q * q if math.isfinite(q) else float('nan')
            # One computation, shared with the engines -- `_d_from_gamma` withholds for a
            # non-positive Gamma, a q outside the representable range, AND a quotient that
            # over/underflows, so the table cannot disagree with the fit about which
            # points have a usable D (rule 12's "remove the gap, don't police it").
            d_app = (dls_engine._d_from_gamma(gamma, q)
                     if quality in FIT_ELIGIBLE_QUALITIES else float('nan'))
            points.append({
                'item_id': lm.item_id,
                'angle_deg': p.get('angle_deg'),
                'concentration_g_per_mL': p.get('concentration_g_per_mL'),
                'gamma_s_inv': gamma if math.isfinite(gamma) else float('nan'),
                'q2_m2': q2,
                'd_app_m2_s': d_app,
                'quality': quality,
                'fit_error': fit_error,
            })
        return points

    @staticmethod
    def _too_few_usable_message(points, include_ids, kind):
        """A clear error when the fit is left with < 2 usable points *because* ticked
        measurements were dropped for a fit-quality reason (failed cumulant / q ≤ 0) —
        the run-enable gate counts buildable rows, but the run drops unfittable ones, so
        without this the engine would raise a bare "needs at least two" that looks wrong
        to a user who ticked enough. Returns None when nothing was
        dropped, so the engine's own message applies (genuinely too few ticked)."""
        considered = [pt for pt in points
                      if include_ids is None or pt['item_id'] in include_ids]
        dropped = [pt for pt in considered
                   if pt['quality'] not in FIT_ELIGIBLE_QUALITIES]
        if not dropped:
            return None
        word = 'angles' if kind == 'gamma_q2' else 'concentrations'
        # The wording below ENUMERATES the tags that are not in FIT_ELIGIBLE_QUALITIES
        # ('fit_failed', 'nonpositive_q'). If a tag is ever moved out of that tuple, this
        # sentence starts naming a cause that did not occur — the defect class this whole
        # change exists to remove. Reword here in the same edit.
        return (f'Only {len(considered) - len(dropped)} of the ticked measurements '
                f'could be fit — {len(dropped)} failed their cumulant fit or had a '
                f'non-physical scattering vector, leaving fewer than two usable {word}. '
                f'Untick or fix the flagged rows (hover them for the reason).')

    @staticmethod
    def _mixed_polarization_message(meas):
        """A clear refusal when one fit would pool more than one analyzer geometry.

        Combination doctrine rule 1's backstop (core/data_models.py), and the refusal
        is KEPT by decision (owner-ruled 2026-08-13): a pooled VV/VH line is a category
        error -- the number would estimate nothing -- and the user's real decision
        surface is the tick boxes (untick one geometry, or run DDLS; the message names
        both). An ask-and-proceed dialog was considered and rejected: its "proceed"
        branch computes a number with no use case, and consent cannot make a
        meaningless number meaningful.

        `mw_fraction` and `analyzer_geometry` are deliberately NOT in the sample key, so
        one sample legitimately holds a DPLS set's VV and VH runs — and the angular /
        concentration fits filter on fraction but never on geometry, so both entered one
        line. Gamma_VH = Gamma_VV + 6 D_r is a PHYSICAL difference (invariant 7:
        polarized and depolarized are never conflated), so the pooled line has the wrong
        slope and a spurious intercept, and its Rh is exported as an ordinary apparent
        one. Refusing beats filtering to a chosen geometry: the choice would be the
        program's rather than the user's, and their other runs would vanish unremarked.

        The predicate compares POLARIZATION STATE, not the label (`_polarization_state`,
        the one shared mapping). Only VV and VH select a polarization; 'VU' means no
        analyzer in the beam and an absent tag means the user has not said, and on an
        instrument with no analyzer those are the same physical run -- so counting raw
        labels refused a half-tagged VU sample, costing it both multi-angle analyses
        with a Gamma_VH message that was untrue of it. Mapping every non-VV/VH label to
        one value keeps every real refusal ({VV, VH}, and {VV, ...} where the other
        run's polarization is simply unproven). Returns None when there is nothing to
        refuse, like `_too_few_usable_message` above.
        """
        geoms = {_polarization_state(m.analyzer_geometry) for m in meas}
        if len(geoms) < 2:
            return None
        named = ', '.join(sorted(g if g else 'untagged' for g in geoms))
        return (f'The ticked runs mix polarizations ({named}), which are never pooled '
                f'into one fit: Γ_VH = Γ_VV + 6·D_r is a physical difference, not '
                f'scatter, so one line through both fits neither. Tick a single '
                f'geometry, or use the DDLS analysis to get D_r from the VV/VH pair.')

    def _pooled_run_inputs(self, sample_id: str, fraction: Optional[str], include_ids,
                           region: Optional[RegionState], kind: str, kw: dict):
        """The shared front half of `run_gamma_q2` / `run_concentration_extrapolation`:
        the per-point table, the fitted measurements, their per-measurement region
        inputs and the seeded engine kwargs -- ONE computation, so the table the user
        vets and the line it belongs to are one fit, input for input (at the Settings
        order, and on the region's inputs when the box is checked).

        Returns (points, fitted LoadedMeasurements, built measurements, engine kwargs,
        notes). A ticked run a CHECKED region leaves unfittable -- the window holds too
        few of its channels, say -- is left out of the line and named in a note, as the
        Correlogram tab records such a run rather than failing the whole set."""
        points = self._dls_run_points(sample_id, fraction, region)
        usable = {pt['item_id'] for pt in points
                  if pt['quality'] in FIT_ELIGIBLE_QUALITIES}
        # Combination doctrine rule 1 (partition), core/data_models.py: fraction=None
        # is the literal untagged series, never a wildcard, so a mixed-fraction fit
        # is unconstructible here (the refusal lives in the structure, not a dialog).
        fitted = [lm for lm in self.workspace.sample_measurements(sample_id, 'dls')
                  if lm.committed_params.get('mw_fraction') == fraction
                  and (include_ids is None or lm.item_id in include_ids)
                  and lm.item_id in usable]
        meas = [lm.build() for lm in fitted]
        if len(meas) < 2:
            msg = self._too_few_usable_message(points, include_ids, kind)
            if msg:
                raise ValueError(msg)
        msg = self._mixed_polarization_message(meas)
        if msg:
            raise ValueError(msg)
        seeds = self._seed_cumulant_kwargs({})
        for name, value in self._pooled_cumulant_kwargs(seeds).items():
            kw.setdefault(name, value)
        kw.setdefault('estimator', self.settings.se_estimator)
        notes: List[str] = []
        if region is not None:
            inputs = self._region_inputs(fitted, 'cumulant', region)
            kw['fit_kwargs'] = [inputs[lm.item_id] for lm in fitted]
            axis = 'angle_deg' if kind == 'gamma_q2' else 'concentration_g_per_mL'
            for pt in points:
                if ((include_ids is None or pt['item_id'] in include_ids)
                        and pt['quality'] == 'fit_failed' and pt.get('fit_error')):
                    where = pt.get(axis)
                    at = ('' if where is None else
                          f' at {where:g} deg' if kind == 'gamma_q2' else
                          f' at {where:.4g} g/mL')
                    notes.append(f"Left out of the fit{at}: its cumulant fit refused the "
                                 f"inputs it was given ({pt['fit_error']})")
            notes.append(self.region_note(region))
        return points, fitted, meas, kw, notes

    @staticmethod
    def _with_notes(res, notes: List[str]):
        """`res` with `notes` appended to its own, in order."""
        if notes:
            res.notes = tuple(res.notes) + tuple(n for n in notes if n)
        return res

    def run_gamma_q2(self, sample_id: str, fraction: Optional[str] = None,
                     include_ids=None, region: Optional[RegionState] = None, **kw):
        """Gamma vs q^2 across the DLS angles of a sample (optionally one fraction).

        `include_ids` (a set of measurement item_ids) restricts the fit to the ticked
        measurements, so the user can choose a subset / drop outliers and recompute.
        `None` = every eligible measurement.

        `region` is the DLS analysis region when the tab's "use the Correlogram tab's
        delay window and baseline" box is ticked, None when it is not: each point is then
        fitted on that measurement's own reading of it (`analysis_region_kwargs`), and the
        result's notes and Summary row say so.

        The result also carries `all_points` — every eligible measurement's per-point
        Γ/q²/D_app + quality, so the tab can fill its table and gray
        the excluded points from the run itself rather than a separate on-focus fit. Only
        fit-usable ticked points (quality 'ok'/'high_pdi', finite) enter the line fit."""
        points, fitted, meas, kw, notes = self._pooled_run_inputs(
            sample_id, fraction, include_ids, region, 'gamma_q2', kw)
        res = self._with_notes(dls_engine.analyze_gamma_q2(meas, **kw), notes)
        res.all_points = points
        key = ('gamma_q2', sample_id, fraction)
        # STAMP-BEFORE-PUBLISH, as run_form_factor: the Cross-Sample candidate builder
        # reads the result only while its stamp matches, so the stamp must be visible
        # before the result is (worker thread writes, main thread reads).
        self._dls_run_sig[key] = self._dls_run_stamp([lm.item_id for lm in fitted])
        self.results[key] = res
        # Gamma -> q->0 is APPARENT (q-extrapolated only; single c folded out). The
        # angle count is the engine's FITTED count, not the ticked one: the engine drops
        # an unusable decay rate and says so, and the label must not count it.
        self.workspace.upsert_sample_rh_row(SampleRhRow(
            sample_id=sample_id, source_kind='gamma_q2', source_set=f'{fraction}',
            rh_nm=getattr(res, 'rh_nm', None), rh_se=getattr(res, 'rh_se', None),
            is_apparent=True, rh_type_label='apparent',
            from_label=(f'Γ vs q² ({int(np.isfinite(np.asarray(res.gamma_s_inv, dtype=float)).sum())} '
                        f'angles{", " + REGION_LABEL if region is not None else ""})'),
            # ^ the FITTED angles: a dropped one rides as NaN
            fraction=fraction,
            se_estimator=getattr(res, 'se_estimator', None)))
        return res

    def run_concentration_extrapolation(self, sample_id: str,
                                        fraction: Optional[str] = None,
                                        include_ids=None,
                                        region: Optional[RegionState] = None, **kw):
        """D vs c -> c->0 across the DLS concentrations of a sample (apparent D per
        measurement is angle-independent, so a multi-angle set still extrapolates).

        `include_ids` (a set of measurement item_ids) restricts the fit to the ticked
        measurements, so the user can choose a subset / drop outliers and recompute.
        `None` = every eligible measurement. `region` as in `run_gamma_q2`.

        As with `run_gamma_q2`, the result carries `all_points` (every eligible
        measurement's per-point D_app + quality); only fit-usable
        ticked points enter the extrapolation."""
        # The same front half as run_gamma_q2 -- same partition rule, same refusal of a
        # mixed-polarization set, same seeds -- so the two cannot drift apart.
        points, fitted, meas, kw, notes = self._pooled_run_inputs(
            sample_id, fraction, include_ids, region, 'conc_extrap', kw)
        res = self._with_notes(
            dls_engine.extrapolate_diffusion_vs_concentration(meas, **kw), notes)
        res.all_points = points
        key = ('conc_extrap', sample_id, fraction)
        # STAMP-BEFORE-PUBLISH -- see run_gamma_q2.
        self._dls_run_sig[key] = self._dls_run_stamp([lm.item_id for lm in fitted])
        self.results[key] = res
        # D vs c -> c->0 is THERMODYNAMIC (invariant 7).
        self.workspace.upsert_sample_rh_row(SampleRhRow(
            sample_id=sample_id, source_kind='conc_extrap', source_set=f'{fraction}',
            rh_nm=getattr(res, 'rh0_nm', None), rh_se=getattr(res, 'rh0_se', None),
            is_apparent=False, rh_type_label='thermodynamic',
            # `res.n_concentrations`, not `len(meas)`: the field now counts concentrations
            # FITTED, so a supplied-count here would label the same run 5 in Cross-Sample
            # and 4 in the DLS tab after a drop (rule 11 -- the sites that rot are the ones
            # that never referenced the field).
            from_label=(f'D vs c ({res.n_concentrations} concentrations'
                        f'{", " + REGION_LABEL if region is not None else ""})'),
            fraction=fraction,
            se_estimator=getattr(res, 'se_estimator', None)))
        return res

    # ----------------------------------------------------------------------
    # Mode-resolved dynamics (the mode map)
    # ----------------------------------------------------------------------
    def run_mode_map(self, sample_id: str, fraction: Optional[str] = None,
                     source: str = 'double', include_ids=None,
                     region: Optional[RegionState] = None, **kw):
        """Extract every mode of every eligible DLS run onto one (q², Γ) map.

        `region` is the DLS analysis region when the tab's "use the Correlogram tab's
        delay window and baseline" box is ticked: each run is then extracted on its own
        reading of it -- the double exponential takes the window and a subtracted
        baseline, CONTIN and NNLS the window and a normalizing baseline -- and the map's
        notes say so. None (unticked) extracts every run on its own defaults.

        Heavy (one double-exponential or CONTIN/NNLS fit per measurement) --
        the GUI dispatches it on the worker. Failure-tolerant per run: an
        unfittable measurement contributes no points and its reason rides the
        map's notes (the engine's extract_modes contract). Provenance
        (item_id, polarization state via the shared `_polarization_state`
        predicate) is attached here, the layer that knows it.

        An empty map (no eligible runs, or none fittable) is a RESULT, not an
        error -- the tab shows its empty state and the notes say why.
        Re-running never touches the user's picks (value-keyed; a moved mode
        orphans its pick, disclosed by the family fit). Ends by re-resolving
        the family fits so a fresh map immediately re-aims nothing silently.
        """
        # A bad `source` is a CALLER bug and raises loudly HERE, before the
        # failure-tolerant per-run loop -- the loop's except would otherwise
        # downgrade it to per-angle notes and return an empty "successful"
        # map a user reads as "no resolvable modes".
        if source not in dls_engine.MODE_SOURCES:
            raise ValueError(
                f'source must be one of {dls_engine.MODE_SOURCES}, '
                f'got {source!r}.')
        # Settings seeds. skip_initial_channels applies to every source; the Rh
        # grid and the alpha sweep are meaningless to the parametric 'double'
        # source, which is why they are gated here rather than passed and
        # ignored -- extract_modes would silently DROP them (its 'double' branch
        # never forwards **dist_kw), so an ungated seed would travel unnoticed
        # instead of failing. Shared home with run_distribution, so the map
        # cannot invert on library defaults while the Distribution tab honours
        # the user's settings for the same measurement.
        kw.setdefault('skip_initial_channels', self.settings.skip_initial_channels)
        if source != 'double':
            self._seed_distribution_kwargs(kw, source)
        points: list = []
        # note -> the angles of the runs that carried it, in first-seen order. A note that is
        # word for word the same on several runs is reported ONCE with every angle named in
        # front (the replicate report's "every kind, once"): an N-angle map on a run with no
        # committed count rate would otherwise print that one sentence N times on the panel
        # and in the export's Angle cell, burying the per-angle caveats under it.
        noted: Dict[str, List[str]] = {}
        for lm in self.workspace.sample_measurements(sample_id, 'dls'):
            p = lm.committed_params
            # Combination doctrine rule 1 (partition): fraction=None is the
            # literal untagged series, never a wildcard.
            if p.get('mw_fraction') != fraction:
                continue
            if include_ids is not None and lm.item_id not in include_ids:
                continue
            try:
                m = lm.build()
            except Exception:  # noqa: BLE001 — disclosed below, not silent
                # Disclosed, not silent: the map must not render complete while missing a loaded
                # run. Ordinary GUI runs pass include_ids of buildable rows,
                # so this note fires mainly for direct API calls.
                angle = p.get('angle_deg')
                noted.setdefault(
                    'measurement could not be built (parameters not '
                    'confirmed or invalid); not on the map.', []).append(
                        '' if angle is None else f'{angle:g}')
                continue
            # Stored manual integration bounds apply to DISTRIBUTION sources
            # only ('double' is parametric and would refuse them; they stay
            # stored, dormant, until a distribution run). Validated at set
            # time (set_mode_bounds), so a raise here means state from an
            # older session -- disclosed as a per-run note, never a dead map
            # (the extraction's one-weird-run contract).
            store = self.mode_picks.get((sample_id, fraction))
            mb = (store.manual_bounds.get(lm.item_id)
                  if store is not None and source != 'double' else None)
            angle = p.get('angle_deg')
            # The run's own reading of a checked region, laid over the shared kwargs; the
            # double exponential reads it as the parametric fit it is, the inversions as
            # an inversion (`analysis_region_kwargs` names the method).
            run_kw = {**kw, **self.analysis_region_kwargs(lm.item_id, source, region)}
            try:
                ext = dls_engine.extract_modes(
                    m, source, manual_bounds_rh_nm=mb, **run_kw)
            except ValueError as exc:
                noted.setdefault(f'not on the map: {exc}', []).append(
                    '' if angle is None else f'{angle:g}')
                continue
            pol = _polarization_state(p.get('analyzer_geometry'))
            for pt in ext.modes:
                points.append(replace(
                    pt, item_id=lm.item_id, polarization_state=pol))
            for n in ext.notes:
                noted.setdefault(n, []).append('' if angle is None else f'{angle:g}')
        notes = []
        if source == 'contin' and kw.get('alpha') is None:
            # Each point records its own alpha, selector, gamma and grid; the sweep's bounds
            # and the F-test level are not on the point, so the map's notes -- the export's
            # Angle comments and the tab's notes line -- name them. `.15g`, so the record is
            # the value that ran. A caller's fixed `alpha` runs no sweep (its points record
            # the selector 'user'), so no sweep is named then.
            note = (f"CONTIN alpha swept from {kw['alpha_min']:.15g} "
                    f"to {kw['alpha_max']:.15g}")
            if kw.get('alpha_method') == 'ftest':
                note += f", F-test probability to reject {kw['ftest_prob_reject']:.15g}"
            notes.append(note + '.')
        for text, angles in noted.items():
            named = [a for a in dict.fromkeys(angles) if a]
            notes.append(f"{', '.join(named)} deg: {text}" if named else text)
        if region is not None:
            notes.append(self.region_note(region))
        res = dls_engine.ModeMapRun(points=tuple(points), source_method=source,
                                    notes=tuple(notes))
        self.results[('mode_map', sample_id, fraction)] = res
        # Record the source the map (and any picks made on it) was built from.
        self.mode_picks_for(sample_id, fraction).source_method = source
        self.run_mode_family_fits(sample_id, fraction)
        return res

    def _matched_mode_points(self, fam: ModeFamilyPicks, map_res):
        """The map points a family's value-keyed picks select, state-filtered.

        A family is BOUND to its polarization state (set at first pick), so a
        point of another state can never match -- the structural half of the
        partition rule; the refusal in run_mode_family_fits is the backstop.
        Equality alone does the filtering (None == None is the unpolarized
        class matching itself); an unbound family matches nothing, which is
        also what its empty pick set implies.
        """
        if not fam.bound:
            return []
        return [p for p in map_res.points
                if p.polarization_state == fam.polarization_state
                and fam.has_pick(p.angle_deg, p.gamma_s_inv)]

    def run_mode_family_fits(self, sample_id: str, fraction: Optional[str] = None):
        """Fit Γ vs q² for every picked mode family against the cached map.

        Synchronous by design (linear algebra on at most dozens of points --
        the fast-ops rule). Returns {label: ModeFamilyFitResult | None}; a
        family with fewer than two matched distinct-angle points gets None
        (its pick/match counts are the GUI's to show via
        `mode_family_status`). Unmatched picks are disclosed, never silently
        dropped: PARTIAL orphaning is noted on the family's fit, and TOTAL
        orphaning -- where no fit exists to carry a note -- is the families
        panel's to show ("matches nothing", from the same status counts).

        Durable rows: one SampleRhRow per FITTED family (apparent,
        mode-resolved), except VH-bound families -- Γ_VH = q²D_t + 6D_r
        yields no hydrodynamic radius on its own, so a durable Table-2 "Rh"
        from it would be the conflation invariant 7 forbids (the same physics
        behind dls_rh_candidates' VH exclusion). The tab still shows the
        VH family's fit, labeled. A family that stops producing a fit has its
        row removed, so Table 2 never shows a value no run stands behind.
        """
        map_res = self.results.get(('mode_map', sample_id, fraction))
        store = self.mode_picks.get((sample_id, fraction))
        fits: Dict[str, Any] = {}
        for fam in (store.families if store is not None else []):
            fit = None
            fitted_states: set = set()
            if map_res is not None and not fam.is_empty():
                matched = self._matched_mode_points(fam, map_res)
                # Backstop refusal behind the GUI facet + the state binding
                # (defense in depth; reachable only through hand-built state).
                states = {p.polarization_state for p in matched}
                if len(states) > 1:
                    named = ', '.join(sorted(s if s else 'untagged'
                                             for s in states))
                    raise ValueError(
                        f'The picks of mode family {fam.label!r} match modes '
                        f'of mixed polarization ({named}), which are never '
                        f'pooled into one fit: Γ_VH = Γ_VV + 6·D_r is a '
                        f'physical difference, not scatter.')
                unmatched = sum(
                    1 for (pa, pg) in fam.picks
                    if not any(_close(pa, p.angle_deg)
                               and _close(pg, p.gamma_s_inv) for p in matched))
                angles = {p.angle_deg for p in matched}
                if len(angles) >= 2:
                    first = self.workspace.measurements[matched[0].item_id]
                    cp = first.committed_params
                    fit = dls_engine.fit_mode_family(
                        matched,
                        temperature_K=cp.get('temperature_K'),
                        viscosity_Pa_s=cp.get('viscosity_Pa_s'),
                        estimator=self.settings.se_estimator)
                    fitted_states = states
                    # Stokes-Einstein used ONE run's committed T/eta; if the
                    # family's runs disagree, say so and name what was used
                    # (the analyze_gamma_q2 identity-mix disclosure shape; never a
                    # silently order-dependent Rh).
                    conds = {(self.workspace.measurements[p.item_id]
                              .committed_params.get('temperature_K'),
                              self.workspace.measurements[p.item_id]
                              .committed_params.get('viscosity_Pa_s'))
                             for p in matched
                             if p.item_id in self.workspace.measurements}
                    if len(conds) > 1:
                        fit.notes = fit.notes + (
                            f'The picked runs\' committed temperature/'
                            f'viscosity differ across the family; '
                            f'Stokes-Einstein used T = '
                            f'{cp.get("temperature_K")} K, eta = '
                            f'{cp.get("viscosity_Pa_s")} Pa*s (run '
                            f'{measurement_display_name(first)}). Align the '
                            f'committed values for a well-defined Rh.',)
                    if unmatched:
                        fit.notes = fit.notes + (
                            f'{unmatched} pick(s) matched no extracted mode -- '
                            f'the extraction changed since they were picked '
                            f'(re-pick, or re-run with the original source).',)
            fits[fam.label] = fit
            row_set = f'{fam.label}|{fraction}'
            # The VH exclusion keys on the states of the points ACTUALLY
            # fitted, never on the binding snapshot from pick time -- a
            # re-tagged run moves the data under the snapshot (rule 12). One computation: the same
            # `states` set the mixed-state refusal above examined.
            if fit is not None and 'VH' not in fitted_states:
                # A map run with the region box ticked says so in its row too, read off
                # the map's own region note (record_mode_set's predicate).
                region = (f', {REGION_LABEL}'
                          if any(n.startswith(REGION_NOTE_PREFIX) for n in map_res.notes)
                          else '')
                self.workspace.upsert_sample_rh_row(SampleRhRow(
                    sample_id=sample_id, source_kind='mode_family',
                    source_set=row_set,
                    rh_nm=fit.rh_nm, rh_se=fit.rh_se,
                    is_apparent=True, rh_type_label='apparent, mode-resolved',
                    from_label=(f'Mode map: {fam.label} '
                                f'({fit.angles_deg.size} modes{region})'),
                    fraction=fraction,
                    se_estimator=fit.se_estimator))
            elif map_res is not None:
                # Retract ONLY when a map exists and the family genuinely
                # failed to re-resolve against it. With NO cached map (a
                # freshly loaded session), the persisted rows stand -- their
                # provenance is the session, and "not re-derived yet" is not
                # "stopped producing" (the rename follow-up must not eat them).
                self.workspace.remove_sample_rh_row(
                    sample_id, 'mode_family', row_set)
        self.results[('mode_family', sample_id, fraction)] = fits
        return fits

    def mode_map_states(self, sample_id: str,
                        fraction: Optional[str] = None) -> list:
        """The polarization states the (sample, fraction) map facets over,
        ordered VV, VH, unpolarized -- from the cached map when present, else
        the committed params through the one shared predicate. The GUI's facet
        combo reads THIS, never re-deriving the classification itself
        (invariant 5)."""
        res = self.results.get(('mode_map', sample_id, fraction))
        if res is not None:
            states = {p.polarization_state for p in res.points}
        else:
            states = set()
            for lm in self.workspace.sample_measurements(sample_id, 'dls'):
                if lm.committed_params.get('mw_fraction') != fraction:
                    continue
                states.add(_polarization_state(
                    lm.committed_params.get('analyzer_geometry')))
        return [s for s in ('VV', 'VH', None) if s in states]

    def mode_family_status(self, sample_id: str, fraction: Optional[str] = None):
        """{label: (n_picks, n_matched)} for the families panel -- how many of
        each family's picks currently land on a map point (metadata only, no
        fitting)."""
        map_res = self.results.get(('mode_map', sample_id, fraction))
        store = self.mode_picks.get((sample_id, fraction))
        out: Dict[str, tuple] = {}
        for fam in (store.families if store is not None else []):
            n_matched = (len(self._matched_mode_points(fam, map_res))
                         if map_res is not None else 0)
            out[fam.label] = (len(fam.picks), n_matched)
        return out

    def mode_amplitude_report(self, sample_id: str,
                              fraction: Optional[str] = None, **kw):
        """The fast/slow amplitude-vs-q report for the role-designated families.

        Needs exactly one family with role 'fast' and one with role 'slow'
        (set in the families panel); raises a clear message otherwise -- the
        report is DEFINED on that pair (Sedlák & Amis 1992a Fig. 4), so there
        is no default the program could honestly pick.
        """
        map_res = self.results.get(('mode_map', sample_id, fraction))
        store = self.mode_picks.get((sample_id, fraction))
        if map_res is None or store is None:
            raise ValueError('Run the mode map first, then pick and '
                             'role-designate a fast and a slow family.')
        fast = [f for f in store.families if f.role == 'fast']
        slow = [f for f in store.families if f.role == 'slow']
        if len(fast) != 1 or len(slow) != 1:
            raise ValueError(
                'The amplitude report needs exactly one family designated '
                '"fast" and one designated "slow" (set the roles in the '
                'Families panel).')
        res = dls_engine.amplitude_vs_q(
            self._matched_mode_points(fast[0], map_res),
            self._matched_mode_points(slow[0], map_res), **kw)
        self.results[('mode_amplitude', sample_id, fraction)] = res
        return res

    # ---- recorded mode sets + the Sedlak partition (mode-resolved SLS) ----
    # The DLS side OWNS mode definition; the SLS side consumes the recorded
    # snapshots read-only, never a live view (owner-ruled 2026-08-14 --
    # normative home: RecordedModeSet's docstring). Per-mode static results
    # run LIVE on partitioned Rayleigh clones, never materialized as
    # workspace measurements (owner-ruled 2026-08-19 -- normative home:
    # run_mode_static's docstring: a per-mode "raw" measurement would be
    # fabricated pseudo-raw and contaminate the parent sample's fits).

    def set_mode_bounds(self, sample_id: str, fraction: Optional[str],
                        item_id: str, bounds_nm) -> None:
        """Store manual integration bounds for one run's distribution modes.

        Validated HERE, at user-input time, through the engine's own validator
        (one predicate home -- rule 12), so `run_mode_map` can trust the store
        and a bad interval raises where the user can fix it.
        """
        pairs = dls_engine.validate_integration_bounds(bounds_nm)
        if not pairs:
            # An empty interval list means "no manual bounds": store nothing
            # rather than an empty list, which would otherwise steer the
            # extraction onto the manual path with zero intervals — no modes,
            # plus a manual-bounds caveat for a run that produced none.
            self.clear_mode_bounds(sample_id, fraction, item_id)
            return
        self.mode_picks_for(sample_id, fraction).manual_bounds[item_id] = pairs

    def clear_mode_bounds(self, sample_id: str, fraction: Optional[str],
                          item_id: str) -> None:
        store = self.mode_picks.get((sample_id, fraction))
        if store is not None:
            store.manual_bounds.pop(item_id, None)

    @staticmethod
    def validate_mode_bounds(bounds_nm) -> list:
        """Validate one row's intervals WITHOUT storing anything — the bounds
        dialog's phase-1 check, so a later row's error cannot leave earlier
        rows silently committed.
        Same single predicate home as `set_mode_bounds` (rule 12)."""
        return dls_engine.validate_integration_bounds(bounds_nm)

    def mode_bounds_for(self, sample_id: str,
                        fraction: Optional[str] = None) -> Dict[str, list]:
        """The stored manual bounds, item_id -> [(lo, hi), ...] (a copy)."""
        store = self.mode_picks.get((sample_id, fraction))
        if store is None:
            return {}
        return {iid: [tuple(b) for b in bs]
                for iid, bs in store.manual_bounds.items()}

    def mode_set_readiness(self, sample_id: str,
                           fraction: Optional[str] = None) -> Dict[str, Any]:
        """Whether 'Record mode set' can act, with the reason CODE when not
        ('no_map' | 'no_families' | 'no_matches'); prose is the GUI's."""
        map_res = self.results.get(('mode_map', sample_id, fraction))
        if map_res is None:
            return {'ok': False, 'reason_code': 'no_map'}
        store = self.mode_picks.get((sample_id, fraction))
        fams = [f for f in (store.families if store is not None else [])
                if not f.is_empty()]
        if not fams:
            return {'ok': False, 'reason_code': 'no_families'}
        if not any(self._matched_mode_points(f, map_res) for f in fams):
            return {'ok': False, 'reason_code': 'no_matches'}
        return {'ok': True, 'reason_code': None}

    def mode_set_bounds_defaults(self, sample_id: str,
                                 fraction: Optional[str] = None) -> Dict[str, list]:
        """item_id -> the cached map's automatic lobe bounds for that run
        (fast->slow order), the record dialog's numeric-bounds pre-fill.
        Runs whose points carry no bounds (parametric source) are omitted."""
        map_res = self.results.get(('mode_map', sample_id, fraction))
        out: Dict[str, list] = {}
        for p in (map_res.points if map_res is not None else ()):
            # 'auto' EXACTLY, never truthiness: a run mapped with manual
            # bounds would otherwise echo the user's own choice back
            # mislabeled as the automatic split.
            if p.bounds_source == 'auto' and math.isfinite(p.bounds_lo_rh_nm):
                out.setdefault(p.item_id, []).append(
                    (p.bounds_lo_rh_nm, p.bounds_hi_rh_nm))
        return out

    def record_mode_set(self, sample_id: str, fraction: Optional[str] = None,
                        *, include_labels=None,
                        label: Optional[str] = None) -> RecordedModeSet:
        """Snapshot the current picked mode families into a RecordedModeSet.

        Copies VALUES off the cached map (amplitudes, decay rates, alpha,
        the residual weighting and what it read, bounds) plus each source run's
        committed concentration -- so the set is
        self-contained and a later re-fit cannot re-aim it. Refusals (rule-1
        backstop, mirroring the family-fit ones): no cached map; no non-empty
        picked family with matched points; included families bound to
        DIFFERENT polarization states (one decomposition = one scattering
        channel). Always appends a NEW entry -- re-recording never replaces
        (the snapshot ruling); duplicate labels get a ' (2)' suffix.
        """
        map_res = self.results.get(('mode_map', sample_id, fraction))
        if map_res is None:
            raise ValueError('Run the mode map first -- a mode set records '
                             'what the map currently shows.')
        store = self.mode_picks.get((sample_id, fraction))
        fams = list(store.families) if store is not None else []
        if include_labels is not None:
            unknown = [x for x in include_labels
                       if not any(f.label == x for f in fams)]
            if unknown:
                raise ValueError(f'No mode family named {unknown[0]!r}.')
            fams = [f for f in fams if f.label in include_labels]
        notes: List[str] = []
        modes: List[RecordedMode] = []
        states: set = set()
        n_no_conc = 0
        for fam in fams:
            matched = self._matched_mode_points(fam, map_res)
            if not matched:
                if not fam.is_empty():
                    notes.append(f'Family {fam.label!r} matched no map point '
                                 f'and was not recorded.')
                continue
            states |= {p.polarization_state for p in matched}
            pts = []
            for p in sorted(matched, key=lambda p: (p.angle_deg, -p.gamma_s_inv)):
                lm = self.workspace.measurements.get(p.item_id)
                conc = (lm.committed_params.get('concentration_g_per_mL')
                        if lm is not None else None)
                if conc is None:
                    n_no_conc += 1
                pts.append(RecordedModePoint(
                    angle_deg=p.angle_deg, gamma_s_inv=p.gamma_s_inv,
                    amplitude_fraction=p.amplitude_fraction,
                    concentration_g_per_mL=conc, item_id=p.item_id,
                    low_amplitude=p.low_amplitude,
                    amplitude_floor=p.amplitude_floor,
                    decay_incomplete=p.decay_incomplete,
                    decay_completeness=p.decay_completeness,
                    min_decay_completeness=p.min_decay_completeness,
                    alpha=p.alpha,
                    alpha_selection_method=p.alpha_selection_method,
                    gcv_gamma=p.gcv_gamma,
                    rh_min_nm=p.rh_min_nm, rh_max_nm=p.rh_max_nm,
                    n_grid=p.n_grid,
                    weighting=p.weighting,
                    sample_time_provenance=p.sample_time_provenance,
                    run_duration_form=p.run_duration_form,
                    bounds_lo_rh_nm=p.bounds_lo_rh_nm,
                    bounds_hi_rh_nm=p.bounds_hi_rh_nm,
                    bounds_source=p.bounds_source))
            modes.append(RecordedMode(label=fam.label, role=fam.role,
                                      points=tuple(pts)))
        if not modes:
            raise ValueError('No picked family currently matches a map point; '
                             'there is nothing to record.')
        if len(states) > 1:
            named = ', '.join(sorted(s if s else 'untagged' for s in states))
            raise ValueError(
                f'The picked families span mixed polarization states '
                f'({named}); one mode set records ONE decomposition, and a '
                f'decomposition belongs to one scattering channel.')
        if n_no_conc:
            notes.append(
                f'{n_no_conc} recorded point(s) come from runs without a '
                f'committed concentration; they cannot join a '
                f'per-concentration partition until one is set and the set '
                f're-recorded.')
        # A map extracted on the Correlogram tab's region says so in its notes; the set
        # records that sentence too, so a partition built from it can still tell which
        # window and baseline its amplitudes came from. Found by the one prefix
        # `describe_region` writes, never by rewording it.
        notes.extend(n for n in map_res.notes if n.startswith(REGION_NOTE_PREFIX))
        existing = self.mode_sets.setdefault((sample_id, fraction), [])
        base = label or f'Mode set {len(existing) + 1}'
        final = base
        n = 2
        while any(s.label == final for s in existing):
            final = f'{base} ({n})'
            n += 1
        rset = RecordedModeSet(
            set_id=self.workspace.new_item_id('ms'), label=final,
            source_method=map_res.source_method,
            polarization_state=next(iter(states)),
            # Stamped HERE, the side-effect layer -- analysis code stays
            # clock-free (tests construct RecordedModeSet with a fixed string).
            recorded=datetime.now().astimezone().isoformat(timespec='seconds'),
            modes=tuple(modes), notes=tuple(notes))
        existing.append(rset)
        return rset

    def mode_sets_for(self, sample_id: str,
                      fraction: Optional[str] = None) -> List[RecordedModeSet]:
        """The recorded mode sets, read-only (a copy of the list; the entries
        are frozen)."""
        return list(self.mode_sets.get((sample_id, fraction), []))

    def _mode_set(self, sample_id: str, fraction: Optional[str],
                  set_id: str) -> RecordedModeSet:
        for s in self.mode_sets.get((sample_id, fraction), []):
            if s.set_id == set_id:
                return s
        raise ValueError(f'No recorded mode set {set_id!r} on this sample.')

    def remove_mode_set(self, sample_id: str, fraction: Optional[str],
                        set_id: str) -> None:
        """Remove one recorded set (an explicit user act, confirmed by the
        GUI) and its cached partition/static results."""
        sets = self.mode_sets.get((sample_id, fraction), [])
        self.mode_sets[(sample_id, fraction)] = [
            s for s in sets if s.set_id != set_id]
        if not self.mode_sets[(sample_id, fraction)]:
            del self.mode_sets[(sample_id, fraction)]
        self.results = {k: v for k, v in self.results.items()
                        if not (isinstance(k, tuple) and set_id in k)}

    def rename_mode_set(self, sample_id: str, fraction: Optional[str],
                        set_id: str, new_label: str) -> None:
        """Relabel a recorded set (frozen -> replace the entry). Cached
        partition results are dropped: their provenance strings carry the
        label, and a stale label inside a Comments cell is a wrong record."""
        sets = self.mode_sets.get((sample_id, fraction), [])
        for i, s in enumerate(sets):
            if s.set_id == set_id:
                sets[i] = replace(s, label=new_label)
                self.results = {k: v for k, v in self.results.items()
                                if not (isinstance(k, tuple) and set_id in k)}
                return
        raise ValueError(f'No recorded mode set {set_id!r} on this sample.')

    def mode_set_alpha_summary(self, sample_id: str, fraction: Optional[str],
                               set_id: str) -> str:
        """The one-line alpha summary of a recorded set ('' for non-CONTIN
        sources) -- THE one computation behind both the sets-table Method cell
        and the exported provenance, so the two surfaces cannot drift (the
        GUI's own copy already omitted the selection method).

        'gcv' names a family, so a GCV point's gamma is named beside the token
        (`alpha = 0.17, gcv, gamma=1 (classical)`), in the distribution
        export's one spelling (`exporter.gcv_gamma_tag`); a record written
        before gamma existed says `gcv` alone -- it names no gamma."""
        s = self._mode_set(sample_id, fraction, set_id)
        alphas = {(p.alpha, p.alpha_selection_method, p.gcv_gamma)
                  for m in s.modes for p in m.points if p.alpha is not None}
        if not alphas:
            return ''

        def selector(meth: str, gamma) -> str:
            if meth == 'gcv' and gamma is not None:
                return f'{meth}, {exporter.gcv_gamma_tag(gamma)}'
            return meth

        if len(alphas) == 1:
            a, meth, gamma = next(iter(alphas))
            return (f'alpha = {a:g}, {selector(meth, gamma)}' if meth
                    else f'alpha = {a:g}')
        meths = {selector(meth, gamma) for _a, meth, gamma in alphas if meth}
        return (f'alpha per run, {"/".join(sorted(meths))}'
                if meths else 'alpha per run')

    def mode_set_grid_summary(self, sample_id: str, fraction: Optional[str],
                              set_id: str) -> str:
        """The one-line size-grid summary of a recorded set ('' when no point
        carries one, i.e. the parametric source).

        The grid is a user setting, so two recordings of one sample can rest on
        different inversions; this names which. A COMPONENT of
        `mode_set_inversion_summary`, not a surface's own source -- consumers
        take the joined text from there, so a new component reaches every
        surface at once.
        """
        s = self._mode_set(sample_id, fraction, set_id)
        grids = {(p.rh_min_nm, p.rh_max_nm, p.n_grid)
                 for m in s.modes for p in m.points if p.n_grid}
        if not grids:
            return ''
        if len(grids) == 1:
            lo, hi, n = next(iter(grids))
            return f'Rh grid {lo:g}-{hi:g} nm, {n} pts'
        return 'Rh grid per run'

    def mode_set_weighting_summary(self, sample_id: str, fraction: Optional[str],
                                   set_id: str) -> str:
        """The one-line residual-weighting summary of a recorded set ('' when no
        point carries one, i.e. the parametric source).

        The weight is a user choice that moves the recovered distribution, the
        selected alpha and therefore every amplitude in the set -- the
        `se_estimator` obligation of invariant 8 clause A in another material --
        so the durable record says which form produced it. A COMPONENT of
        `mode_set_inversion_summary`, like the alpha and grid summaries: a
        consumer takes the joined text from there.

        The AXIS-DERIVED sample time is named here too, because the partition file takes
        a recorded set's provenance from this composed line
        (`tests/test_export.py::test_the_partition_export_names_the_recorded_sets_residual_weighting`
        drives it end to end): an amplitude weighted on an inferred quantity never
        travels unlabelled, the obligation `export._sample_time_note` discharges for a
        per-fit file. Silent for the instrument's own sample times and for a form that
        read none.
        """
        s = self._mode_set(sample_id, fraction, set_id)
        points = [p for m in s.modes for p in m.points]
        forms = {p.weighting for p in points if p.weighting}
        if not forms:
            return ''
        text = (f'residual weighting: {next(iter(forms))}' if len(forms) == 1
                else f'residual weighting per run: {"/".join(sorted(forms))}')
        if any(p.sample_time_provenance == 'axis' for p in points):
            text += ' (sample time derived from the delay axis)'
        return text

    def mode_set_inversion_summary(self, sample_id: str,
                                   fraction: Optional[str], set_id: str) -> str:
        """How the recorded set's amplitudes were obtained, in one line --
        regularization, size grid and residual weighting, '' for the parametric
        source.

        THE one computation behind every surface that shows it: the sets-table
        Method cell and the exported provenance both read this rather than
        joining the components themselves. Joining at each surface is what let
        the grid reach the export and not the table, which is the drift the
        components' single home was supposed to prevent in the first place.
        """
        parts = (self.mode_set_alpha_summary(sample_id, fraction, set_id),
                 self.mode_set_grid_summary(sample_id, fraction, set_id),
                 self.mode_set_weighting_summary(sample_id, fraction, set_id))
        return '; '.join(t for t in parts if t)

    def _mode_set_state(self, sample_id: str, fraction: Optional[str]) -> tuple:
        """The recorded-set component of the SLS candidate memo key: identity +
        label per set (contents are frozen, so nothing else can move)."""
        return tuple((s.set_id, s.label)
                     for s in self.mode_sets.get((sample_id, fraction), []))

    def mode_set_provenance(self, sample_id: str, fraction: Optional[str],
                            set_id: str) -> str:
        """The one-line provenance for a recorded set -- ONE computation
        serving the GUI list and the export Comments (the
        `derived_intensity_provenance` pattern). Dangling sources are counted
        live against the current workspace, so a deletion shows up without
        mutating the snapshot."""
        s = self._mode_set(sample_id, fraction, set_id)
        ids = {p.item_id for m in s.modes for p in m.points if p.item_id}
        n_removed = sum(1 for i in ids
                        if i not in self.workspace.measurements)
        inner = self.mode_set_inversion_summary(sample_id, fraction, set_id)
        alpha_txt = f' ({inner})' if inner else ''
        srcs = {p.bounds_source for m in s.modes for p in m.points
                if p.bounds_source}
        bounds_txt = ('' if not srcs else
                      f'; bounds: {"+".join(sorted(srcs))}')
        parts = [
            f"mode set '{s.label}': {s.source_method}{alpha_txt}{bounds_txt}",
            f'{len(ids)} source run(s)'
            + (f', {n_removed} removed' if n_removed else ''),
            f'recorded {s.recorded[:10]}' if s.recorded else '',
        ]
        return '; '.join(p for p in parts if p)

    def _partition_inputs(self, rset: RecordedModeSet, ckey) -> list:
        """A recorded set's modes at ONE concentration, as engine primitives."""
        modes = []
        for m in rset.modes:
            pts = [p for p in m.points
                   if p.concentration_g_per_mL is not None
                   and round_concentration_for_identity(
                       p.concentration_g_per_mL) == ckey]
            if not pts:
                continue
            modes.append(mode_partition.PartitionModeInput(
                label=m.label, role=m.role,
                angles_deg=tuple(p.angle_deg for p in pts),
                amplitude_fractions=tuple(p.amplitude_fraction for p in pts),
                gamma_s_inv=tuple(p.gamma_s_inv for p in pts),
                item_ids=tuple(p.item_id for p in pts),
                low_amplitude=tuple(p.low_amplitude for p in pts),
                decay_incomplete=tuple(p.decay_incomplete for p in pts),
                bounds_sources=tuple(p.bounds_source for p in pts)))
        return modes

    def _mode_point_ids(self, rset: RecordedModeSet, mode_label: str, ckey,
                        angles_deg) -> list:
        """Per-angle source run ids aligned with an SLS series' angles (''
        where no single recorded point matches) -- what licenses or withholds
        the amplitude-Guinier regression SE."""
        buckets: Dict[Any, list] = {}
        for m in rset.modes:
            if m.label != mode_label:
                continue
            for p in m.points:
                if (p.concentration_g_per_mL is not None
                        and round_concentration_for_identity(
                            p.concentration_g_per_mL) == ckey):
                    buckets.setdefault(
                        round_angle_for_identity(p.angle_deg),
                        []).append(p.item_id)
        return [(buckets[k][0]
                 if len(buckets.get(k, [])) == 1 else '')
                for k in (round_angle_for_identity(float(a))
                          for a in angles_deg)]

    def run_mode_partition(self, sample_id: str, set_id: str,
                           fraction: Optional[str] = None) -> Dict[float, Any]:
        """Partition every masked Rayleigh series that shares a concentration
        with the recorded set: {concentration: ModePartitionResult}.

        Synchronous by design (arithmetic over at most dozens of points -- the
        fast-ops rule) and recomputed on every call: the inputs are the live
        masked Rayleigh pipeline plus a frozen snapshot, so recomputing is
        cheaper than policing staleness. The latest result is stored for the
        GUI to fetch and plot, exactly like every other results entry.
        """
        rset = self._mode_set(sample_id, fraction, set_id)
        rr = self.masked_rayleigh(sample_id, fraction)
        prov = self.mode_set_provenance(sample_id, fraction, set_id)
        out: Dict[float, Any] = {}
        # ONE partition per identity-form concentration, first-wins with a
        # disclosed note — never keyed on the raw float: _rayleigh_for_sample returns one
        # result per MEASUREMENT, so a repeat acquisition at one concentration
        # would last-wins-collapse silently while the Analysis sub-tab's
        # sibling _rayleigh_at_concentration keeps the FIRST — first-wins here
        # keeps the two surfaces computing from the same series).
        first_by_key: Dict[float, float] = {}
        for r in rr:
            if r.concentration_g_per_mL == 0:
                continue
            ckey = round_concentration_for_identity(r.concentration_g_per_mL)
            modes = self._partition_inputs(rset, ckey)
            if not modes:
                continue
            if ckey in first_by_key:
                res0 = out[first_by_key[ckey]]
                res0.notes = res0.notes + (
                    'A repeat SLS series at this concentration was not '
                    'partitioned — the partition uses the first-loaded series '
                    'at each concentration, the same series the Analysis '
                    'sub-tab fits. Mask or remove the repeat to partition it '
                    'instead.',)
                continue
            first_by_key[ckey] = float(r.concentration_g_per_mL)
            out[float(r.concentration_g_per_mL)] = \
                mode_partition.partition_rayleigh(
                    r, modes, provenance=prov,
                    closure_tol=mode_partition.DEFAULT_CLOSURE_TOL,
                    separation_ratio_min=(
                        mode_partition.DEFAULT_SEPARATION_RATIO_MIN))
        if not out:
            raise ValueError(
                "No SLS series shares a concentration with the recorded mode "
                "set -- the partition needs an intensity series (loaded or "
                "trace-derived) at a concentration the set's runs carry.")
        self.results[('mode_partition', sample_id, fraction, set_id)] = out
        return out

    def run_mode_static(self, sample_id: str, set_id: str, mode_label: str,
                        method: str, fraction: Optional[str] = None,
                        concentration_g_per_mL: Optional[float] = None):
        """A static analysis of ONE mode's partitioned series.

        `method` is 'amplitude_guinier' (Sedlak Eq. 7 -- the sourced per-mode
        size), 'guinier' / 'debye' (single concentration), or 'zimm' / 'berry'
        (across >= 2 concentrations). Ordinary machinery on any user-selected
        mode, no opt-in gate (owner-ruled 2026-08-14; the doctrine's fenced
        rule-1 counter-example) -- the honesty load is the appended
        mode-resolved disclosure notes, the partition provenance in
        `derived_provenance`, and the fitters' own flags.

        Per-mode series are NEVER materialized as workspace measurements
        (owner-ruled 2026-08-19): the partition happens after solvent
        subtraction, so a per-mode "raw" measurement would store fabricated
        pseudo-raw intensities that re-subtract the blank and rescale wrongly
        on recalibration -- and a same-identity member would contaminate the
        parent sample's fits. Deliberately writes NOTHING into SampleResult
        either: a mode's Mw is not the sample's.
        """
        rset = self._mode_set(sample_id, fraction, set_id)
        parts = self.run_mode_partition(sample_id, set_id, fraction=fraction)
        per_c: Dict[float, Any] = {}
        for c, part in parts.items():
            for pm in part.per_mode:
                if pm.label == mode_label:
                    per_c[c] = pm.rayleigh
        if not per_c:
            raise ValueError(
                f'The recorded set has no mode named {mode_label!r} with a '
                f'matching SLS series.')
        est = self.settings.se_estimator

        def _one_concentration():
            if concentration_g_per_mL is not None:
                ckey = round_concentration_for_identity(concentration_g_per_mL)
                for c, clone in per_c.items():
                    if round_concentration_for_identity(c) == ckey:
                        return c, clone
                raise ValueError(
                    f'No partitioned series at '
                    f'{concentration_g_per_mL:g} g/mL for this mode.')
            if len(per_c) == 1:
                return next(iter(per_c.items()))
            cs = ', '.join(f'{c:g}' for c in sorted(per_c))
            raise ValueError(
                f'This mode has partitioned series at several concentrations '
                f'({cs} g/mL); pass concentration_g_per_mL to pick one.')

        def _partition_caveats(concs) -> tuple:
            """The partition-level validity notes of the concentration(s) this
            fit consumed — separation, closure, manual bounds, exclusions.
            They qualify the per-mode result itself, so they travel ON it
            (screen + the affected export columns), never only on the
            partition view."""
            seen: List[str] = []
            for cc in concs:
                for n in parts[cc].notes:
                    if n not in seen:
                        seen.append(n)
            return tuple(seen)

        if method == 'amplitude_guinier':
            c, clone = _one_concentration()
            ids = self._mode_point_ids(
                rset, mode_label, round_concentration_for_identity(c),
                clone.angles_deg)
            res = mode_partition.amplitude_guinier(
                clone.q2_nm2, clone.excess_rayleigh_cm_inv,
                source_item_ids=ids, estimator=est,
                qrg_max_valid=self.settings.guinier_qrg_max,
                derived_provenance=clone.derived_provenance,
                # `amplitude_guinier` takes bare arrays, so the blank fact cannot ride
                # in with the data -- both callers fill it from the Rayleigh result the
                # amplitudes were partitioned out of. The default is True, so a MISSED
                # caller reports a typed-blank Rg as reliable and nothing goes red.
                blank_measured=bool(getattr(clone, 'blank_measured', True)))
            res.notes = res.notes + _partition_caveats([c])
        elif method in ('guinier', 'debye'):
            c, clone = _one_concentration()
            if method == 'guinier':
                res = sls_engine.guinier_analysis(
                    clone, qrg_max_valid=self.settings.guinier_qrg_max,
                    estimator=est)
            else:
                res = sls_engine.debye_analysis(clone, estimator=est)
            res.notes = res.notes + (_MODE_RESOLVED_MW_NOTE,
                                     _MODE_RESOLVED_CONDITIONS_NOTE,
                                     _MODE_RESOLVED_SE_NOTE)
            res.notes = res.notes + _partition_caveats([c])
        elif method in ('zimm', 'berry'):
            clones = [per_c[c] for c in sorted(per_c)]
            n_conc = len({round_concentration_for_identity(c)
                          for c in per_c})
            if n_conc < 2:
                raise ValueError(
                    f'A per-mode {method.capitalize()} extrapolation needs '
                    f'partitioned series at >= 2 concentrations; this mode '
                    f'has {n_conc}. Use the amplitude Guinier or Debye/'
                    f'Guinier at one concentration.')
            res = sls_engine.zimm_analysis(clones, method=method,
                                           estimator=est)
            res.notes = res.notes + (_MODE_RESOLVED_MW_NOTE,
                                     _MODE_RESOLVED_CONDITIONS_NOTE,
                                     _MODE_RESOLVED_SE_NOTE)
            res.notes = res.notes + _partition_caveats(sorted(per_c))
        else:
            raise ValueError(
                f"method must be 'amplitude_guinier', 'guinier', 'debye', "
                f"'zimm' or 'berry', got {method!r}.")
        self.results[('mode_static', sample_id, fraction, set_id,
                      mode_label, method)] = res
        return res

    def mode_sets_index(self, sample_id: str) -> List[tuple]:
        """(fraction, RecordedModeSet) for every recorded set of this sample,
        across fractions -- what the SLS-side list shows (read-only; the GUI
        never walks the store itself, invariant 5)."""
        out: List[tuple] = []
        for (sid, frac), sets in self.mode_sets.items():
            if sid == sample_id:
                out.extend((frac, s) for s in sets)
        return out

    def mode_set_dangling_sources(self, sample_id: str,
                                  fraction: Optional[str],
                                  set_id: str) -> List[str]:
        """The recorded source runs that no longer exist (the set survives
        them -- the snapshot ruling; the list view shows the warning)."""
        s = self._mode_set(sample_id, fraction, set_id)
        ids = {p.item_id for m in s.modes for p in m.points if p.item_id}
        return sorted(i for i in ids if i not in self.workspace.measurements)

    def export_mode_partition_csv(self, sample_id: str,
                                  fraction: Optional[str], set_id: str,
                                  file_path: str, *,
                                  concentration_g_per_mL: Optional[float] = None,
                                  ) -> str:
        """Export one concentration's partition, with an amplitude-Guinier fit
        for every mode that yields one -- RECOMPUTED at export time so nothing
        cached can disagree with the file (the export_mode_map discipline)."""
        parts = self.run_mode_partition(sample_id, set_id, fraction=fraction)
        if concentration_g_per_mL is not None:
            ckey = round_concentration_for_identity(concentration_g_per_mL)
            match = [c for c in parts
                     if round_concentration_for_identity(c) == ckey]
            if not match:
                raise ValueError(
                    f'No partitioned series at '
                    f'{concentration_g_per_mL:g} g/mL.')
            c = match[0]
        elif len(parts) == 1:
            c = next(iter(parts))
        else:
            cs = ', '.join(f'{x:g}' for x in sorted(parts))
            raise ValueError(
                f'Several concentrations are partitioned ({cs} g/mL); pass '
                f'concentration_g_per_mL to pick one for the file.')
        part = parts[c]
        fits: Dict[str, Any] = {}
        for pm in part.per_mode:
            try:
                fits[pm.label] = self.run_mode_static(
                    sample_id, set_id, pm.label, 'amplitude_guinier',
                    fraction=fraction, concentration_g_per_mL=c)
            except ValueError:
                pass                       # no usable fit for this mode
        return exporter.export_mode_partition(part, fits, file_path)

    # ---- DLS Summary snapshot writers (durable display scalars) ----
    def _snapshot_parametric(self, item_id: str, method: str, res) -> None:
        """Write one MeasurementResultRow for a parametric fit (Summary Table 1).
        Per-measurement DLS is always apparent (single q, single c)."""
        self.workspace.replace_dls_rows_for(item_id, method)
        row = MeasurementResultRow(item_id=item_id, method=method, is_apparent=True)
        # The reliability flag travels WITH the number into the durable row, so it reaches
        # the Summary table and the export instead of stopping at the DLS panel. NOT on
        # `row.label`, which is short provenance feeding the export's `From` column.
        # Finite-gated: a NaN Rh is a blank cell with no shown number to qualify.
        #
        # EVERY arm, not just the cumulant. While an exponential's only unreliability cause
        # NaN'd its Rh the finite gate silently excused the other three; the undecayed-mode
        # criterion leaves it FINITE, so a flagged value reached this row bare.

        # Finiteness no longer gates the note. It did while a NaN Rh meant "no fit ran";
        # now a flagged no-decay correlogram returns a NaN Rh WITH the sentence explaining
        # it, and dropping the note left the Summary row blank and unexplained while the
        # DLS tab showed the reason. `success is False` joins the test so a
        # converged-but-withheld result and a failed one read alike.
        def _note_for(source, value) -> str:
            unreliable = (not getattr(source, 'rh_reliable', True)
                          or getattr(source, 'success', True) is False)
            return getattr(source, 'reliability_note', '') if unreliable else ''

        if method == 'cumulant':
            row.rh_nm, row.pdi = res.rh_nm, res.pdi
            row.reliability_note = _note_for(res, res.rh_nm)
        elif method == 'single':
            row.rh_nm = res.mode.rh_nm
            row.reliability_note = _note_for(res.mode, res.mode.rh_nm)
        elif method == 'double':
            # Two numbers, one note field: label whichever is flagged, and collapse a cause
            # they share -- the same convention the DLS status line uses, so a reader meets
            # one wording on both surfaces.
            row.rh_fast_nm, row.rh_slow_nm = res.mode1.rh_nm, res.mode2.rh_nm
            fast_note = _note_for(res.mode1, res.mode1.rh_nm)
            slow_note = _note_for(res.mode2, res.mode2.rh_nm)
            if fast_note and fast_note == slow_note:
                row.reliability_note = f'both modes: {fast_note}'
            else:
                row.reliability_note = ' '.join(
                    f'{label} mode: {note}'
                    for label, note in (('fast', fast_note), ('slow', slow_note)) if note)
        elif method == 'kww':
            # KWW's verdict is result-level by design: one decay, two ways of reporting it.
            row.rh_nm = res.rh_from_tau_c_nm
            row.reliability_note = _note_for(res, res.rh_from_tau_c_nm)
        # The fit's remaining quantities and its passive notices, driven from the one
        # detail spec so the saved row and the Summary columns cannot disagree about what
        # a method reports. Both are what makes a RELOADED session a full record of the
        # fit rather than the headline the tab's glance tables show.
        row.fit_details = _collect_fit_details(method, res)
        row.passive_notes = _collect_passive_notes(res)
        self.workspace.upsert_dls_result_row(row)

    def _snapshot_distribution(self, item_id: str, method: str, res) -> None:
        """Write one MeasurementResultRow per resolved peak (Summary Table 1)."""
        self.workspace.replace_dls_rows_for(item_id, method)
        dist = getattr(res, 'distribution', res)     # CONTIN wraps its DistributionResult
        # A seed-shaped (unconverged) result writes no rows: find_distribution_peaks
        # itself refuses it (the one home for that rule), so `peaks` comes back empty.
        try:
            peaks = dls_engine.find_distribution_peaks(dist)
        except Exception:  # noqa: BLE001 — unresolvable fit writes no rows
            peaks = []
        # The CONTIN sweep-edge flags travel on the rows exactly as the parametric
        # rows above carry their reliability_note -- the Summary table and its rh_note
        # export must not show a flagged peak as clean.
        edge_note = getattr(res, 'reliability_note', '')   # '' for NNLS/lognormal
        for j, pk in enumerate(peaks):
            self.workspace.upsert_dls_result_row(MeasurementResultRow(
                item_id=item_id, method=method, peak_index=j,
                rh_nm=float(pk.rh_nm), int_fraction=float(pk.weight_fraction),
                is_apparent=True, label=f'{method.upper()} peak {j + 1}',
                reliability_note=edge_note,
                # The PASSIVE tier travels too, on the same row as the flag tier beside
                # it. All three distribution fitters emit the dead-channel note, and
                # without this it stopped at the Distribution tab: the Summary showed a
                # cumulant cell marked and the CONTIN cell beside it silent about the
                # same correlogram, and a reload lost it entirely. These notes are
                # fit-level ('' target), so they land on the method's headline cell.
                passive_notes=_collect_passive_notes(res)))

    def distribution_peaks(self, result):
        """Resolved peaks of a distribution result as (rh_nm, gamma_s_inv,
        weight_fraction) tuples, Rh-ascending. A thin wrapper so the GUI can label
        peaks without importing the analysis layer (invariant 5)."""
        dist = getattr(result, 'distribution', result)
        try:
            peaks = dls_engine.find_distribution_peaks(dist)
        except Exception:  # noqa: BLE001 — label helper, no peaks
            return []
        return [(float(p.rh_nm), float(p.gamma_s_inv), float(p.weight_fraction))
                for p in peaks]

    def weighted_distribution_peaks(self, weighted_result):
        """Resolved peaks of a reweighted distribution as (rh_nm, weight_fraction) tuples
        (Rh-ascending, fraction in the chosen weighting). Thin wrapper so the GUI labels
        the reweighted view without importing the analysis layer (invariant 5)."""
        try:
            peaks = optical.find_weighted_peaks(weighted_result)
        except Exception:  # noqa: BLE001 — label helper, no peaks
            return []
        return [(float(p.rh_nm), float(p.weight_fraction)) for p in peaks]

    # ---- DLS Summary tab: read-back accessors over the snapshot store ----
    def _measurement_label(self, item_id: str) -> str:
        """A DLS measurement's display name, from `core.naming` -- the one builder every
        surface shares (axes, then the name: '90° · 3'). Qt-free.

        No sample prefix: every consumer here carries the sample in its own column.
        An id this workspace does not hold IS the subject of the answer, so it is
        returned as itself.
        """
        lm = self.workspace.measurements.get(item_id)
        if lm is None:
            return item_id
        return measurement_display_name(lm)

    def dls_summary_measurement_rows(
            self, ticked_only_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Table-1 data: one dict per measurement that has any snapshot rows,
        folding all methods/peaks into display fields. Restricted to
        `ticked_only_ids` when given. Sorted by sample then measurement label."""
        allow = set(ticked_only_ids) if ticked_only_ids is not None else None
        by_item: Dict[str, Dict[str, Any]] = {}
        for row in self.workspace.dls_result_rows.values():
            if allow is not None and row.item_id not in allow:
                continue
            sid = self._sample_id_of(row.item_id)
            rec = by_item.get(row.item_id)
            if rec is None:
                rec = {
                    'item_id': row.item_id,
                    'sample_id': sid,
                    'sample_label': self._sample_label(sid) if sid else '',
                    'measurement_label': self._measurement_label(row.item_id),
                    'cumulant_rh': None, 'cumulant_rh_note': '',
                    'pdi': None, 'single_rh': None, 'single_rh_note': '',
                    'kww_rh': None, 'kww_rh_note': '',
                    'double_fast': None, 'double_slow': None, 'double_note': '',
                    'peaks': {},     # method -> [(rh_nm, int_fraction), ...]
                    'peak_notes': {},  # method -> reliability_note ('' = unflagged)
                    # The durable fit details and passive notices, per parametric
                    # method, exactly as the saved row holds them. Carried whether or
                    # not the table is currently showing details: the controller answers
                    # what was computed, and what to DISPLAY is the GUI's decision
                    # (invariant 5 -- no display state reaches back into this layer).
                    'details': {},   # method -> {display name: value}
                    'notes': {},     # method -> {note target: [sentence, ...]}
                }
                by_item[row.item_id] = rec
            # `getattr` defaults, not direct access: a row loaded from a session written
            # before these fields existed has neither, and reads as "no details".
            #
            # NOT gated on the detail-spec roster. The details are parametric-only because
            # only those arms have a fixed parameter set, but the NOTES are every method's
            # -- a distribution fit emits the same dead-channel notice, and gating both
            # halves on one roster silently dropped it.
            if getattr(row, 'fit_details', None):
                rec['details'][row.method] = dict(row.fit_details)
            if getattr(row, 'passive_notes', None):
                rec['notes'][row.method] = {k: list(v)
                                            for k, v in row.passive_notes.items()}
            if row.method == 'cumulant':
                rec['cumulant_rh'], rec['pdi'] = row.rh_nm, row.pdi
                rec['cumulant_rh_note'] = getattr(row, 'reliability_note', '')
            elif row.method == 'single':
                rec['single_rh'] = row.rh_nm
                rec['single_rh_note'] = getattr(row, 'reliability_note', '')
            elif row.method == 'kww':
                rec['kww_rh'] = row.rh_nm
                rec['kww_rh_note'] = getattr(row, 'reliability_note', '')
            elif row.method == 'double':
                rec['double_fast'], rec['double_slow'] = row.rh_fast_nm, row.rh_slow_nm
                rec['double_note'] = getattr(row, 'reliability_note', '')
            else:                    # distribution method -> a peak
                rec['peaks'].setdefault(row.method, []).append(
                    (row.peak_index, row.rh_nm, row.int_fraction))
                # One note per method (every peak of one fit shares it), so the Summary
                # cell can flag the way the parametric cells beside it do.
                note = getattr(row, 'reliability_note', '')
                if note and not rec['peak_notes'].get(row.method):
                    rec['peak_notes'][row.method] = note
        # Sort each method's peaks by peak_index, drop the index after sorting.
        for rec in by_item.values():
            for method, lst in rec['peaks'].items():
                lst.sort(key=lambda t: t[0])
                rec['peaks'][method] = [(rh, frac) for _i, rh, frac in lst]
        return sorted(by_item.values(),
                      key=lambda r: (r['sample_label'], r['measurement_label']))

    def dls_summary_sample_rows(self) -> List[Dict[str, Any]]:
        """Table-2 data: every sample-level Rh row as a display dict, sorted by
        sample then source."""
        out = []
        for row in self.workspace.sample_rh_rows.values():
            out.append({
                'sample_id': row.sample_id,
                'sample_label': self._sample_label(row.sample_id, row.fraction)
                if row.sample_id in self.workspace.samples else row.sample_id,
                'source_kind': row.source_kind,
                'rh_nm': row.rh_nm, 'rh_se': row.rh_se,
                'rh_type_label': row.rh_type_label, 'from_label': row.from_label,
                # The estimator behind rh_se travels with it onto the on-screen table,
                # as `export_dls_summary` already sends it into the file: an OLS ± must
                # never show without saying so (invariant 8 clause A).
                'se_estimator': row.se_estimator,
            })
        return sorted(out, key=lambda r: (r['sample_label'], r['source_kind']))

    # ----------------------------------------------------------------------
    # DLS replicate averaging (user-initiated; never automatic)
    # ----------------------------------------------------------------------
    def average_dls_correlograms(self, item_ids: List[str]) -> str:
        """Average true replicate correlograms into a new, tagged DLS measurement.

        Channel-by-channel mean of g2 - 1 over the selected replicates (which must
        share an identical lag grid and matching identity/optics -- including
        concentration and Mw fraction, see `check_replicate_identity`). The
        result is registered
        as an ordinary DLS measurement -- it flows through the sidebar, the DLS tab,
        and session JSON like any other -- but tagged `derived_kind='replicate_average'`
        with `derived_from` listing its sources. Its parameters are copied from the
        first replicate, so it auto-groups into the same sample and is analysable at
        once. Returns the new item_id.

        Note this produces the cleaner POINT-ESTIMATE curve; the defensible Rh ±
        comes from `average_dls_results` (the spread across per-replicate fits).
        """
        lms = self._require_dls_items(item_ids)
        _refuse_an_average_as_a_repeat(lms)
        meas = [lm.build() for lm in lms]
        avg = dls_engine.average_replicate_correlograms(meas)   # raises on mismatch

        new_id = self.workspace.new_item_id()
        # The averaging above enforced agreement on the identity fields
        # `check_replicate_identity` compares, so those are every replicate's values.
        # Everything else is inherited from the FIRST replicate, including the physical
        # `viscosity_Pa_s` and the two refractive indices -- deliberately not compared (the
        # reasoning is in `check_replicate_identity`'s docstring, and the parameters are
        # the user's: invariant 3). The one exception is the LABEL, which names ONE
        # replicate: inherited, the average reads and exports as the name of one of its
        # own inputs. It is named for what it is -- 'avg of N' -- until the user labels it.
        params = dict(lms[0].committed_params)
        params.pop('label', None)
        raw = {
            'delay_times_s': [float(x) for x in avg.delay_times_s],
            'correlogram': [float(x) for x in avg.mean_g2m1],
            # per-channel SD across replicates, kept for reference (ignored by build())
            'g2m1_sd': [float(x) for x in avg.sd_g2m1],
        }
        # The replicates share one lag grid (`check_replicate_identity`), so a per-channel
        # sample time they ALL carry, identically, is the average's too. Any replicate
        # without one, or a disagreement, leaves the average without: the fit then
        # derives a stand-in and says so, rather than inheriting one run's layout.
        stated = [lm.raw.get('sample_times_s') for lm in lms]
        if all(s is not None for s in stated) and all(
                np.allclose(s, stated[0], rtol=1e-9, atol=0.0) for s in stated[1:]):
            raw['sample_times_s'] = [float(x) for x in stated[0]]
        # The run duration follows the same rule: each replicate is one run of that length
        # and averaging divides the variance by their number, leaving Eq. (9)'s relative
        # shape and its T as they were -- so a duration every replicate states, identically,
        # is the average's; any other case leaves it off and the fit says so.
        durations = [lm.raw.get('duration_s') for lm in lms]
        if all(d is not None for d in durations) and len(set(durations)) == 1:
            raw['duration_s'] = float(durations[0])
        lm = LoadedMeasurement(
            item_id=new_id, kind='dls', raw=raw,
            working_params=dict(params), committed_params=dict(params),
            source_path=None,
            derived_from=list(item_ids),
            derived_kind='replicate_average',
        )
        self.workspace.add_measurement(lm)
        self._rekey_sample_stores(self.workspace.regroup())
        return new_id

    def dls_average_writes_rh(self, method: str) -> bool:
        """Does 'Average derived results' with this method write a sample Rh?

        Exactly the parametric methods with one unambiguous Rh mode; double-exp
        and the distribution methods are report-only (`_DLS_PARAM_SPECS` /
        `_average_distribution_results`). The GUI's ask-first confirm keys on
        this, so it can never fire for a run that writes nothing."""
        spec = _DLS_PARAM_SPECS.get(method)
        return bool(spec and spec[1])

    def average_inputs_text(self, method: str, *, cumulant_order: Optional[int] = None,
                            distribution_kwargs: Optional[dict] = None) -> str:
        """The fit settings a replicate average of `method` runs on, in one sentence: the
        Settings values when nothing was copied (None), else the copy the averaging
        prompt took from the Correlogram tab (the order) or the Distribution tab (the
        inversion's fields). '' for a method with no such settings (single, double, KWW).

        ONE computation behind the prompt's "the Settings values" line, the summary's
        note and -- through `_average_detail` -- the written row's label, so what the
        prompt showed, what ran and what was recorded cannot disagree."""
        if method == 'cumulant':
            if cumulant_order is None:
                return f'Fitted at the Settings cumulant order ({self.settings.cumulant_order}).'
            return f"Fitted at the Correlogram tab's cumulant order ({int(cumulant_order)})."
        if method not in _DLS_DISTRIBUTION_METHODS:
            return ''
        if distribution_kwargs is None:
            kw, whose = self._seed_distribution_kwargs({}, method), 'the Settings values'
        else:
            kw, whose = dict(distribution_kwargs), "the Distribution tab's values"
        weighting = kw.get('weighting')
        parts = [f"Rh grid {kw['rh_min_nm']:g}–{kw['rh_max_nm']:g} nm "
                 f"({int(kw['n_grid'])} points)",
                 f"residual weighting {weighting or 'sourced (default)'}"]
        if method == 'contin':
            selector = {'gcv': f"GCV (γ = {kw.get('gcv_gamma'):g})",
                        'lcurve': 'L-curve corner',
                        'ftest': f"F-test (probability to reject "
                                 f"{kw.get('ftest_prob_reject'):g})"}.get(
                kw.get('alpha_method'), str(kw.get('alpha_method')))
            parts.append(f"α swept from {kw['alpha_min']:g} to {kw['alpha_max']:g}, "
                         f"selected by {selector}")
        return f"Inverted on {whose}: {'; '.join(parts)}."

    def _average_detail(self, n_runs: int, method: str, cumulant_order: Optional[int],
                        region: Optional[RegionState]) -> str:
        """The parenthetical a written replicate-average Rh carries -- its run count and
        method, the cumulant ORDER it ran at (named whether it came from Settings or the
        Correlogram tab, since an order-1 average has no PDI to caveat it), and the
        region when the box was ticked."""
        parts = [f'{n_runs} runs', method]
        if method == 'cumulant':
            order = (self.settings.cumulant_order if cumulant_order is None
                     else int(cumulant_order))
            parts.append(f'order {order}')
        if region is not None:
            parts.append(REGION_LABEL)
        return ', '.join(parts)

    def average_dls_results(self, item_ids: List[str], method: str,
                            tau_min_s: Optional[float] = None,
                            tau_max_s: Optional[float] = None,
                            write_to_sample: bool = True, *,
                            region: Optional[RegionState] = None,
                            cumulant_order: Optional[int] = None,
                            distribution_kwargs: Optional[dict] = None) -> Dict[str, Any]:
        """Fit each replicate independently and report the mean +/- SD/sqrt(N).

        This is the ISO 22412 repeat-measurement path and the ONLY place the
        platform reports a DLS dynamic +/-: the uncertainty is the spread across
        genuine repeats, never a covariance from one correlogram (whose lag
        channels are correlated, Schaetzel 1990). Each replicate is fit on its OWN
        grid with the shared tau window, so -- unlike correlogram averaging -- the
        grids need not be identical. The IDENTITY fields must still match, and are
        checked by the same `check_replicate_identity` the correlogram path uses
        (polymer, solvent, temperature, angle, wavelength, concentration, Mw
        fraction, analyzer geometry): "repeats of one condition" is what makes the
        spread an uncertainty rather than a systematic difference.

        `method` is one of 'cumulant', 'single', 'double', 'kww' (parametric) or
        'nnls', 'contin', 'lognormal' (distribution). For the parametric methods
        with a single unambiguous Rh (cumulant, single, kww) the averaged Rh +/- SE
        is written into the sample's SampleResult (provenance-labeled, never
        overwriting a hand-entered Rh). Double-exponential (two modes) and the
        distribution methods (variable peak count) are reported in the returned
        summary only. The distribution methods average their peaks positionally
        (Rh-ascending) and warn when replicates disagree on the peak count (see
        `_average_distribution_results`). Returns a summary dict the GUI renders
        (and the validation asserts on).

        THE AVERAGING PROMPT'S TWO CHOICES arrive here. `region` is the DLS analysis
        region when its "use the Correlogram tab's delay window and baseline" box was
        ticked (None when not): each replicate is then fitted on its own reading of it.
        `cumulant_order` / `distribution_kwargs` are the one-time copy of the Correlogram
        tab's order or the Distribution tab's fields, taken when the prompt's copy box was
        ticked; None runs on the Settings values. Either way the summary's `notes` say
        which settings and which region the fits read, and a written Summary row names
        the order it ran at.
        """
        lms = self._require_dls_items(item_ids)
        _refuse_an_average_as_a_repeat(lms)
        # All replicates must belong to one sample (we write one Rh to one sample).
        sids = {self._sample_id_of(iid) for iid in item_ids}
        if len(sids) > 1:
            raise ValueError(
                "Averaging derived results needs replicates from a single sample; "
                "the selection spans more than one. Move them into one sample "
                "first, or select within one sample.")
        sample_id = sids.pop()
        # THE identity check, shared with the correlogram path. It is needed here because
        # `sample_group_key` deliberately drops concentration (a Zimm ladder is ONE
        # sample), so the single-sample test above cannot see a ladder -- and pooling
        # per-replicate fits across concentrations is not an ISO 22412 replicate statistic.
        # Built once and reused by the fit loop below; the distribution branch is the
        # exception, rebuilding inside `_average_distribution_results` rather than taking a
        # changed signature for one cheap dataclass.
        meas = [lm.build() for lm in lms]
        dls_engine.check_replicate_identity(meas)      # raises on mismatch
        fraction = lms[0].committed_params.get('mw_fraction')
        region_kw = self._region_inputs(lms, method, region)
        notes = [n for n in (self.average_inputs_text(
            method, cumulant_order=cumulant_order,
            distribution_kwargs=distribution_kwargs), self.region_note(region)) if n]

        if method in _DLS_DISTRIBUTION_METHODS:
            summary = self._average_distribution_results(
                lms, method, sample_id, fraction,
                replicate_kwargs={lm.item_id: {**(distribution_kwargs or {}),
                                               **region_kw.get(lm.item_id, {})}
                                  for lm in lms})
            summary['notes'] = [*notes, *summary['notes']]   # the fits' relayed notes last
            return summary

        # Per-method extraction: name -> (unit, value), plus which name is the Rh
        # to write back. `primary_rh` is None for multi-mode methods.
        specs, primary_rh = _DLS_PARAM_SPECS[method]
        order_run = (self.settings.cumulant_order if cumulant_order is None
                     else int(cumulant_order))
        if method == 'cumulant' and order_run < 2:
            # Order 1 fits a single decay rate and reports no PDI (NaN by construction).
            # The all-or-nothing drop below would discard every replicate for a quantity
            # the order does not produce, so at order 1 the PDI row is not averaged --
            # the rule the Gamma-vs-q^2 table applies to an order-1 point's PDI tag.
            specs = [(name, getter) for name, getter in specs if name != 'PDI']
        window = {'tau_min_s': tau_min_s, 'tau_max_s': tau_max_s}
        if method == 'cumulant' and cumulant_order is not None:
            window['order'] = int(cumulant_order)

        per_param: Dict[str, List[float]] = {name: [] for name, _ in specs}
        relayed: List[tuple] = []   # (run name, its passive notes) of each contributing fit
        n_ok = 0
        n_flagged = 0            # contributing fits whose Rh is UNUSABLE (outside the
                                 # technique's size range -- `_dls_rh_promotable`); a
                                 # merely flagged Rh (PDI, lag window) is counted in
                                 # `warn_note` and written WITH its flag (owner ruling
                                 # 2026-08-27, CLAUDE.md invariant 4: flag, not refuse)
        warn_note = ''           # the first flagged-but-usable fit's note, for the label
        flagged_note = ''        # the first such fit's own note, whatever its cause. Only
                                 # finite-Rh fits reach this count (a NaN Rh drops the fit
                                 # in the vals loop below), which USED to mean only the
                                 # _size_range_check wording could appear -- the
                                 # missing-viscosity note always rode a NaN. The
                                 # undecayed-mode criterion is a third
                                 # producer and leaves the Rh FINITE, so the cause is no
                                 # longer knowable here and the message below must not
                                 # name one.
        for lm, m in zip(lms, meas, strict=True):
            try:
                # The shared window, with this replicate's own reading of a checked
                # region laid over it (its window and its baseline).
                res = self._fit_one_replicate(
                    m, method, {**window, **region_kw.get(lm.item_id, {})})
            except Exception as exc:  # noqa: BLE001 — replicate isolation
                if region is not None:
                    # A replicate the checked region left unfittable is named, as the
                    # pooled fits and DDLS name theirs; unchecked, the count says it.
                    notes.append(f"Left out of the average: "
                                 f"{measurement_display_name(lm)}'s fit refused the "
                                 f"inputs it was given ({exc})")
                continue                       # a single failed fit just drops out
            # Skip a converged-but-flagged-failed fit (single/double/kww carry it).
            if getattr(res, 'success', True) is False:
                continue
            vals = {}
            ok = True
            for name, getter in specs:
                val = getter(res)
                if val is None or not math.isfinite(float(val)):
                    ok = False
                    break
                vals[name] = float(val)
            if not ok:
                continue
            for name, val in vals.items():
                per_param[name].append(val)
            n_ok += 1
            relayed.append((measurement_display_name(lm), [
                note for notes in _collect_passive_notes(res).values() for note in notes]))
            # Counted AFTER the all-or-nothing drop above, so a fit that never
            # contributed to the mean is never counted against it. Per-fit, not on the
            # averaged MEAN: four fits at 30 nm plus one at 2e4 nm average to 4152 nm,
            # below the ceiling, and would be written with a ±. The cause is read off
            # the note rather than assumed. Each method names its own primary-Rh field
            # (`_DLS_PARAM_SPECS`), so the promotability test reads the VALUE the ladder
            # will average, not `rh_nm`.
            if primary_rh is not None and not _rh_value_promotable(vals.get(primary_rh)):
                n_flagged += 1
                flagged_note = flagged_note or getattr(res, 'reliability_note', '')
            elif not getattr(res, 'rh_reliable', True):
                warn_note = warn_note or getattr(res, 'reliability_note', '')
        # The average is the only surface showing its number, so each contributing fit's
        # passive notes reach it -- notes, not verdicts: no value or flag moves.
        notes.extend(_relayed_run_notes(relayed))

        parameters = []
        rh_stats = None
        for name, _getter in specs:
            stats = unc.replicate_mean_se(per_param[name])
            parameters.append({
                'name': name, 'unit': _DLS_PARAM_UNITS.get(name, ''),
                'mean': stats.mean, 'sd': stats.sd, 'sem': stats.sem, 'n': stats.n,
            })
            if name == primary_rh:
                rh_stats = stats

        # The flag is DERIVED from the reason, never tracked beside it. Tracked
        # independently, a chain covering only some causes lets the pair travel as
        # (False, None), which the dialog renders as SILENCE about the sample Rh. One
        # computation, one representation, so a future arm cannot set one without the
        # other. The chain is exhaustive by construction; `n` is tested BEFORE finiteness
        # because `replicate_mean_se([])` returns mean=nan with n=0, and "no usable fits"
        # is a different message from a saturated mean.

        # WHICH ARMS ALSO CLEAR A PREVIOUS WRITE. The split is carried by WHICH CALL BUILDS
        # THE REASON, not by a second `clear_previous` flag -- that would reintroduce the
        # two-representation drift the paragraph above exists to prevent.
        #   * `_decline_and_clear(...)` -- A COMPUTATION FINISHED AND ITS ANSWER IS
        #     UNUSABLE: a non-finite mean, an out-of-range size. The fits ran, so a PREVIOUS
        #     average of the same (method, fraction) goes with it.
        #   * a plain assignment -- NOTHING FINISHED THAT COULD SUPERSEDE ANYTHING:
        #     report-only mode, a multi-mode method, the defensive arm, and n == 0 / n < 2.

        # n == 0 / n < 2 sit in the second group deliberately. Every fit failing is as
        # often a FIT-CONFIGURATION failure -- a tau window that suits no replicate -- as
        # it is unfittable data, and that says nothing about the previous average. Clearing
        # there would destroy a valid result on a mistake the user fixes in seconds, with
        # no undo, so it clears only on a COMPLETED fit whose answer is unusable.

        # Clearing at all is the consistent behaviour rather than the destructive one
        # because `upsert_sample_rh_row` keys on f'{method}|{fraction}', NOT on which runs
        # were averaged -- so a SUCCESSFUL re-average already replaces its predecessor in
        # silence, and without this a failed attempt would protect state a successful one
        # destroys. Keying the row by the run selection, so two averages coexist as
        # pickable candidates, is a roadmap feature.
        rh_skip_reason: Optional[str]
        rh_replaced_pick: Optional[str] = None   # label of a pick the write replaced
        if not write_to_sample:
            rh_skip_reason = ("averaging was run in report-only mode; the sample Rh "
                              "was not touched.")
        elif primary_rh is None:
            rh_skip_reason = (f"{method} has more than one Rh mode; not written to "
                              f"the sample (shown above only).")
        elif rh_stats is None:                      # pragma: no cover - defensive
            # Unreachable today: `primary_rh` is one of the method's own spec names
            # for all four parametric methods, so the loop above always sets this.
            # Kept so a future _DLS_PARAM_SPECS entry that names a field it does not
            # emit degrades to a disclosure rather than an AttributeError below.
            rh_skip_reason = (f"{method} reported no {primary_rh} statistics; not "
                              f"written to the sample.")
        # Neither count message may name `primary_rh` as the thing that failed. The
        # drop loop above is ALL-OR-NOTHING across the whole spec list -- one non-finite
        # sibling (cumulant's PDI, any of kww's three) drops a replicate whose primary
        # Rh was perfectly good -- so "produced a usable {primary_rh}" would be a false
        # statement inside the very disclosure this method exists to make.
        elif rh_stats.n == 0:
            # No count here: `rh_stats.n` and `n_ok` are equal by construction, so a
            # parenthetical would always read "0 of N" -- which the summary's second
            # line already prints. A viscosity-less sample is named as the cause when
            # it is one: every fit then converges yet NaNs its Rh, and the generic
            # sentence would misread as a fitting failure.
            if all(m.viscosity_Pa_s is None for m in meas):
                rh_skip_reason = ("this sample has no viscosity, so no fit can report "
                                  "an Rh (Stokes-Einstein needs it) and the per-method "
                                  "averages are withheld with it. Set a viscosity and "
                                  "re-run; each fit's decay-rate outputs are unaffected.")
            else:
                rh_skip_reason = ("no replicate produced a complete parameter set; every "
                                  "fit either failed or returned a non-finite value.")
        elif rh_stats.n < 2:
            rh_skip_reason = (f"only {rh_stats.n} replicate produced a complete "
                              f"parameter set (at least two are needed for a ±); "
                              f"check the excluded fits.")
        elif not math.isfinite(rh_stats.mean):
            rh_skip_reason = self._decline_and_clear(
                sample_id, fraction, method,
                f"the averaged {primary_rh} is not a finite number "
                f"(the replicate values overflowed); not written to "
                f"the sample.")
        # Last, because it presumes a finite mean over at least two fits. An out-of-range
        # size reaching SampleResult.rh_nm travels on into rho = Rg/Rh, the Cross-Sample
        # tables and the scaling plots CARRYING A ±, dressing a number outside what DLS can
        # establish as a precise one (invariant 8). Declining the write stops that at one
        # point rather than teaching every downstream consumer the same flag.

        # Nothing is lost but the durable promotion: the mean ± SEM and n are printed in the
        # averaging summary above, every per-replicate Rh stays in the Correlogram table,
        # Summary Table 1 and the per-fit export, and the value is still offered as an
        # explicit Cross-Sample pick -- which is the escape, since raising the threshold is
        # not something a user of the shipped app can do. Precedence: `_write_replicate_rh`'s
        # own "already hand-entered or picked" guard lives inside the `else` below, so a
        # sample with BOTH reports the size reason. Both end in "not written".
        elif n_flagged:
            rh_skip_reason = self._decline_and_clear(
                sample_id, fraction, method,
                f"{n_flagged} of {rh_stats.n} averaged fits returned an Rh flagged as "
                f"unreliable, so the mean is reported above but not written to the sample; "
                f"set the sample Rh yourself if you mean to use it. One of them: "
                f"{flagged_note}")
        else:
            # Snapshot the pick BEFORE the write: the promotion now replaces a
            # picked Rh (a Run is a deliberate act), and the summary must say so
            # -- the disclosure is unconditional, per the tier rule's confirm/
            # disclose split (SampleResult docstring).
            replaced = self.picks_a_run_would_replace(sample_id, fraction, ('rh',))
            rh_skip_reason = self._write_replicate_rh(
                sample_id, fraction, rh_stats, method, warn_note=warn_note,
                detail=self._average_detail(rh_stats.n, method, cumulant_order, region))
            if rh_skip_reason is None and replaced:
                rh_replaced_pick = replaced[0][1]
        rh_written = rh_skip_reason is None

        return {
            'method': method,
            'sample_id': sample_id,
            'fraction': fraction,
            'n_replicates': len(lms),
            'n_fit_ok': n_ok,
            'parameters': parameters,
            'peaks': None,                 # parametric methods have no peak list
            'peak_count_warning': None,
            'rh_written': rh_written,
            'rh_skip_reason': rh_skip_reason,
            'rh_replaced_pick': rh_replaced_pick,
            'notes': notes,
        }

    def _average_distribution_results(self, lms, method: str, sample_id: str,
                                      fraction: Optional[str], *,
                                      replicate_kwargs: Optional[Dict[str, dict]] = None
                                      ) -> Dict[str, Any]:
        """Average a distribution method's PEAKS across replicates, positionally.

        Each replicate is fitted (NNLS / CONTIN / lognormal) and its peaks resolved
        (`find_distribution_peaks`, already Rh-ascending). Peaks are aligned by
        **order of appearance**: position k averages the k-th-smallest-Rh peak over
        every replicate that resolved at least k peaks. Per position we report
        Rh = mean ± SD/√N (Theory-and-Equations-Guide §6.4, replicate mean/SEM),
        the mean intensity weight, and how many runs
        resolved it. When runs disagree on the peak count, a warning is set --
        positional alignment can misalign if a run misses a peak, so the user is
        told to inspect the distribution overlay.

        Distribution peaks are **report-only**: nothing is written to the sample
        (the peak positions of an ill-posed inversion are regularization-dependent;
        the cross-run SD is an honest spread but should not silently feed ρ). To get
        an averaged Rh that feeds ρ, average the correlograms and fit that one curve.
        """
        per_replicate_peaks = []           # list of (Rh-ascending) peak lists
        n_ok = 0
        edge_notes: List[str] = []
        weight_forms: List[str] = []       # the recorded form of each contributing fit
        relayed: List[tuple] = []          # (run name, its passive notes), the same fits
        for lm in lms:
            m = lm.build()
            try:
                # `replicate_kwargs` are the prompt's copied fields and this replicate's
                # reading of a checked region; the seed home fills the rest from Settings.
                res = self._fit_distribution_replicate(
                    m, method, **(replicate_kwargs or {}).get(lm.item_id, {}))
                dist = getattr(res, 'distribution', res)   # ContinResult wraps it
                peaks = dls_engine.find_distribution_peaks(dist)
            except Exception:  # noqa: BLE001 — replicate isolation
                continue                   # a single failed fit just drops out
            if not peaks:
                continue                   # incl. seed-shaped fits: the scanner refuses them
            per_replicate_peaks.append(peaks)
            n_ok += 1
            # Counted AFTER the peak guard, so the caveat's numerator counts only fits
            # that contribute to the average (its denominator).
            note = getattr(res, 'reliability_note', '')
            if note:
                edge_notes.append(note)
            weight_forms.append(getattr(dist, 'weighting', '') or '')
            # The passive notes off the result as the engine returned it -- the object
            # `_snapshot_distribution` records a single run's notes from.
            relayed.append((measurement_display_name(lm), [
                n for notes in _collect_passive_notes(res).values() for n in notes]))

        counts = [len(p) for p in per_replicate_peaks]
        max_k = max(counts) if counts else 0
        peaks_out = []
        for k in range(max_k):
            rhs = [p[k].rh_nm for p in per_replicate_peaks if len(p) > k]
            wts = [p[k].weight_fraction for p in per_replicate_peaks if len(p) > k]
            stats = unc.replicate_mean_se(rhs)
            peaks_out.append({
                'position': k + 1,
                'rh_mean': stats.mean, 'rh_sd': stats.sd, 'rh_sem': stats.sem,
                'rh_n': stats.n,
                'weight_mean': float(np.mean(wts)) if wts else float('nan'),
                'n_resolved': len(rhs), 'n_total': n_ok,
            })

        warning = None
        if counts and len(set(counts)) > 1:
            warning = (f"runs resolved {min(counts)}–{max(counts)} peaks; positional "
                       f"alignment may be unreliable (a run that misses a peak shifts "
                       f"the rest) — inspect the distribution overlay before trusting "
                       f"per-peak averages.")
        if edge_notes:
            # The single-run surfaces carry the CONTIN sweep-edge caveat; the replicate
            # report must not present edge-pinned fits as a clean statistic.
            distinct = '; '.join(dict.fromkeys(edge_notes))    # every edge kind, once
            edge = (f"{len(edge_notes)} of {n_ok} replicate fits pinned at a sweep edge "
                    f"({distinct}); the per-peak averages inherit that caveat.")
            warning = edge if warning is None else warning + ' ' + edge
        if len(set(weight_forms)) > 1:
            # Same reason, same place: the inversion result is dropped above, and a fit
            # whose sourced weight could not be formed ran under the older weight -- a
            # different precision map from its siblings' -- before entering the mean.
            tally = ', '.join(f'{weight_forms.count(form)} × {form}'
                              for form in dict.fromkeys(weight_forms))
            mixed = (f"the replicate fits ran under different residual weights ({tally}); "
                     f"the per-peak averages mix them.")
            warning = mixed if warning is None else warning + ' ' + mixed

        return {
            'method': method,
            'sample_id': sample_id,
            'fraction': fraction,
            'n_replicates': len(lms),
            'n_fit_ok': n_ok,
            'parameters': [],
            'peaks': peaks_out,
            'peak_count_warning': warning,
            'rh_written': False,
            'rh_skip_reason': (
                f"{method.upper()} peaks are reported only, not written to the "
                f"sample. For an averaged Rh that feeds ρ, use 'Average correlation "
                f"functions' and fit that curve."),
            'rh_replaced_pick': None,      # report-only: nothing written or replaced
            'notes': _relayed_run_notes(relayed),
        }

    def _fit_distribution_replicate(self, m, method: str, **kw):
        """Fit one replicate with a distribution method and return the engine's
        result as-is (CONTIN's ContinResult keeps its sweep-edge flags; the caller
        unwraps `.distribution`). Takes `run_distribution`'s Settings seeds from their
        one home -- a replicate is fitted the way a single run is -- but does not
        touch the results cache. An explicit kwarg (a copied field, a region's window or
        baseline) wins over its seed."""
        kw = self._seed_distribution_kwargs(kw, method)
        if method == 'nnls':
            return dls_engine.fit_nnls(m, **kw)
        if method == 'lognormal':
            return dls_engine.fit_lognormal(m, **kw)
        if method == 'contin':
            return dls_engine.fit_contin(m, **kw)
        raise ValueError(f"Unknown distribution method {method!r}.")

    def _require_dls_items(self, item_ids: List[str]) -> List[LoadedMeasurement]:
        """Fetch the loaded measurements for averaging, or raise a clear error."""
        if len(item_ids) < 2:
            raise ValueError(
                f"Replicate averaging needs at least two measurements, got "
                f"{len(item_ids)}.")
        lms = []
        for iid in item_ids:
            lm = self.workspace.measurements.get(iid)
            if lm is None:
                raise ValueError(f"No measurement {iid!r}.")
            if lm.kind != 'dls':
                # Named, not id'd: unlike the branch above, this measurement exists.
                raise ValueError(
                    f"Replicate averaging is DLS-only; "
                    f"{measurement_display_name(lm)!r} is {lm.kind!r}.")
            lms.append(lm)
        return lms

    def _fit_one_replicate(self, m, method: str, window: Dict[str, Any]):
        """Fit one replicate with the chosen dynamic method and the shared window.

        The replicate's committed count rate and sample times ride on `m` itself
        (`LoadedMeasurement.build()`), so every cumulant fit in the program is the same
        fit (owner-ruled 2026-09-08; `analysis/dls/angular.py::_gamma_for_measurement`)."""
        window = {**window}   # don't mutate the caller's dict
        if method == 'cumulant':
            # From the one seed home, the order among them -- unless the averaging
            # prompt's copy of the Correlogram tab's order rides in `window`.
            return dls_engine.fit_cumulants(m, **self._seed_cumulant_kwargs(window))
        window.pop('order', None)
        window.setdefault('skip_initial_channels', self.settings.skip_initial_channels)
        if method == 'single':
            return dls_engine.fit_single_exponential(m, **window)
        if method == 'double':
            return dls_engine.fit_double_exponential(m, **window)
        if method == 'kww':
            return dls_engine.fit_kww(m, **window)
        raise ValueError(f"Unknown averaging method {method!r}.")

    def _decline_and_clear(self, sample_id: str, fraction: Optional[str],
                           method: str, reason: str) -> str:
        """Build a skip reason for an arm that ALSO supersedes a previous write.

        One call produces both halves, so "did this arm clear?" and "what does the
        user get told?" cannot drift apart -- the same discipline `_write_replicate_rh`
        follows by returning its reason rather than setting a flag beside it. The
        group split, and why only these arms clear, is documented at the ladder.
        """
        cleared = self._clear_replicate_rh(sample_id, fraction, method)
        return reason if cleared is None else f'{reason} {cleared}'

    def _clear_replicate_rh(self, sample_id: str, fraction: Optional[str],
                            method: str) -> Optional[str]:
        """Remove a previous replicate average for this (method, fraction).

        Returns a sentence naming what was cleared, or None when there was nothing
        to clear -- so the caller can disclose it instead of dropping a value in
        silence.

        The inverse of `_write_replicate_rh`, half for half: BOTH functions touch the
        Table-2 row unconditionally, and BOTH gate the `SampleResult` half. The two
        gates are not the same predicate, deliberately -- the write declines a
        promotion over a 'user'/'picked' Rh, while the clear additionally requires the
        stored value to BE the row it is deleting, because that field is shared with the
        gamma-q2 and concentration-extrapolation writers and the row is not.

        WHAT IT IS ALLOWED TO TOUCH. The durable Table-2 row is keyed, so deleting it
        is unambiguous. The `SampleResult` slot is NOT keyed -- it is one field that
        several paths write -- so the `SampleResult` half is cleared only when the
        stored value IS the row being deleted: `rh_source == 'computed'` AND
        `rh_nm == row.rh_nm`. Both were assigned from the same `float(rh_stats.mean)`,
        so exact equality holds by construction. That ties the clear to the artifact
        instead of sniffing `rh_label` for a prefix, and it protects a 'computed' Rh
        written by the gamma-q2 or concentration-extrapolation paths, which share the
        field but not the row. A 'user' Rh is never touched -- the same value
        `_write_replicate_rh` declines to overwrite -- and a 'picked' Rh is spared
        HERE too: this clear also runs on the run-decline path, a failure, and
        destroying a pick is reserved for the deliberate acts that invalidate it
        (a successful Run replaces it; a cumulant switch clears it, key-matched,
        after a confirm). Note the write guard protects only the PROMOTION: its
        row write, like the deletion here, is unconditional.

        THE SENTENCE IS BUILT FROM WHICH HALVES ACTUALLY RAN (round-1 finding 1). The
        two deletions sit under different predicates, so one unconditional "has been
        cleared" is a FALSE statement in exactly the arm the guard exists to protect:
        a user who picked this same average in Cross-Sample would be told it was
        cleared while it stayed on the sample and kept feeding rho with its +/-. Each
        return below describes the state it is returning from, so the disclosure and
        the stored state cannot disagree -- the same discipline the skip-reason ladder
        follows for `rh_written`.
        """
        row = self.workspace.remove_sample_rh_row(
            sample_id, 'replicate_avg', f'{method}|{fraction}')
        if row is None:
            return None
        r = self.workspace.samples[sample_id].result_for(fraction)
        gone = f'The previous replicate average for this method ({row.rh_nm:.4g} nm)'
        if r.rh_source == 'computed' and r.rh_nm == row.rh_nm:
            # Back to SampleResult's own defaults, field for field -- NOT to a
            # hand-picked "empty" state. One definition of "clear a stored
            # quantity": `reset_result_quantity` (which also nulls the key).
            reset_result_quantity(r, 'rh')
            return f'{gone} has been cleared, since it was superseded by this one.'
        if r.rh_nm is None:
            return f'{gone} has been removed from Summary Table 2.'
        # Reached when the sample Rh came from somewhere else: an explicit pick, a
        # hand-entered value, or another 'computed' writer (gamma-q2, concentration
        # extrapolation) that shares the field but not the row. Naming the value and its
        # provenance keeps this from reading as "nothing happened". It deliberately does
        # NOT claim the retained value came from a DIFFERENT source: that is false where the
        # sample holds an earlier run of this same average (pick it in Cross-Sample,
        # re-average, decline), and deciding it here would mean sniffing `rh_label` against
        # a spelling it has two of. Printing `rh_label` says more and is always true.
        return (f'{gone} has been removed from Summary Table 2. The sample Rh is '
                f'unchanged: it still holds {r.rh_nm:.4g} nm'
                f'{f" ({r.rh_label})" if r.rh_label else ""}.')

    def picks_a_run_would_replace(self, sample_id: str, fraction: Optional[str],
                                  quantities) -> List[tuple]:
        """(quantity, label) for each explicitly picked value an explicit Run over
        (sample, fraction) would replace.

        The write-back guards protect only 'user' -- an explicit Run is a
        deliberate act, so it replaces a pick (SampleResult docstring). ONE
        computation behind the ask-first confirm AND the disclose-after line, so
        the dialog, the write and the disclosure cannot disagree. Read on the GUI
        main thread before the worker dispatch; the passive `auto_select_*`
        refreshers still spare every deliberate tier and never consult this."""
        r = self.workspace.samples[sample_id].result_for(fraction)
        out = []
        for q in quantities:
            if (getattr(r, f'{q}_source') == 'picked'
                    and getattr(r, RESULT_VALUE_FIELDS[q]) is not None):
                out.append((q, getattr(r, f'{q}_label') or q))
        return out

    def _write_replicate_rh(self, sample_id: str, fraction: Optional[str],
                            rh_stats, method: str, warn_note: str = '',
                            detail: Optional[str] = None) -> Optional[str]:
        """Write the replicate-averaged Rh +/- SE into the sample, respecting a
        user override. `warn_note` is the first contributing fit's reliability note
        when a fit was flagged but usable (a PDI past the validity limit, an undecayed
        window): the value promotes and the note rides its label, the `_flag_suffix`
        shape -- a flag warns, it does not withhold (CLAUDE.md invariant 4). `detail` is
        the label's parenthetical (`_average_detail`: runs, method, the cumulant order,
        the region), written into both the promotion and the Summary row; None is the
        bare run count and method.

        Returns the skip reason, or None when the write happened -- reason-only, so
        "was it written" has exactly one representation and cannot drift from its
        explanation.

        TWO WRITES, ONE OF THEM CONDITIONAL. The Table-2 row is a RECORD of what this
        method's replicate average currently is; the `SampleResult` slot is a PROMOTION
        of that value to "the sample's Rh". The override guard was written to protect
        the promotion, and protecting the record with it was collateral: a sample with a
        picked Rh kept a `replicate_avg` row frozen at whatever it held before the pick,
        so Summary Table 2 showed a stale average indefinitely, while a run that FAILED
        deleted that same row. Success and failure disagreed about the record in
        opposite directions (raised in batch A round 1, confirmed by the fan-out,
        owner-ruled 2026-08-07).

        So the row is written either way and only the promotion is declined. The
        promotion guard protects 'user' ALONE: pressing Run is a deliberate act, so
        it replaces an explicit pick -- and the run's summary discloses the
        replacement (the tier rule lives in the `SampleResult` docstring). The
        passive path stays inert: `auto_select_rh` returns early on 'user'/'picked'
        before it looks at any candidate, so a fresher row cannot climb back into
        the sample behind a pick between runs.
        """
        r = self.workspace.samples[sample_id].result_for(fraction)
        declined = r.rh_source == 'user' and r.rh_nm is not None
        # The "± = SD/√N" clause is only appended when there IS a ±: a zero spread
        # withholds the SEM, and a label naming the method beside an empty ± cell would
        # describe an uncertainty the row does not carry.
        detail = detail or f'{rh_stats.n} runs, {method}'
        label = f"DLS replicate average ({detail}); "
        label += ("± = SD/√N across runs (ISO 22412)" if rh_stats.sem is not None else
                  "no ± — the runs agreed exactly, so the spread is not a measurement "
                  "of precision")
        if warn_note:
            label += f" — unreliable: {warn_note}"
        if not declined:
            r.rh_nm = float(rh_stats.mean)
            r.rh_source = 'computed'
            # A single-angle dynamic Rh is apparent (not extrapolated to q->0, c->0).
            r.rh_apparent = True
            r.rh_label = label
            r.rh_se = unc.se_or_none(rh_stats.sem)
            r.rh_se_estimator = None      # a replicate SEM is not a regression SE -> estimator-independent
            # The one promotion with no candidate behind it, so the key comes from the
            # shared spelling the candidate builder also uses -- see replicate_rh_key.
            set_result_key(r, 'rh', *replicate_rh_key(f'{method}|{fraction}'))
        # The durable Summary Table-2 row, written whether or not the promotion was --
        # see the docstring. It is what `_clear_replicate_rh` is the inverse of.
        self.workspace.upsert_sample_rh_row(SampleRhRow(
            sample_id=sample_id, source_kind='replicate_avg',
            source_set=f'{method}|{fraction}', rh_nm=float(rh_stats.mean),
            rh_se=unc.se_or_none(rh_stats.sem), is_apparent=True,
            rh_type_label='apparent',
            from_label=f'replicate avg ({detail})',
            fraction=fraction))
        if declined:
            return ("the sample already has a hand-entered Rh, which is never "
                    "overwritten; the averaged value is recorded in Summary Table 2 "
                    "and offered in the Cross-Sample picker.")
        return None

    # ----------------------------------------------------------------------
    # SLS analysis orchestration (uses committed calibration)
    # ----------------------------------------------------------------------
    def _blank_ids_for(self, series) -> List[str]:
        """The item_id of the blank each series resolves to, in series order
        (a series with no blank contributes nothing)."""
        out: List[str] = []
        for lm in series:
            b, _ = self.workspace.solvent_reference_for(lm.item_id)
            if b is not None:
                out.append(b.item_id)
        return out

    def solvent_blank_status(self, item_id: str, *,
                             use_working: bool = False) -> Dict[str, Any]:
        """Which solvent blank an SLS series is referenced against, and why.

        One computation behind the Data tab's dropdown tooltip and the per-series
        `blank_provenance` record the exporters print (the `mean_count_rate_status`
        pattern); the dropdown's entry labels come from `blank_label`, the same
        helper this reads. `use_working=True` resolves the series' WORKING choice
        (an uncommitted pick), so a tooltip can describe what is selected rather
        than what the last fit used. Keys: `item_id` (the blank, or None), `source`
        ('user' / 'auto:same-file' / 'auto:sample' -- the same family:method shape as
        `mean_count_rate_cps_source`, computed here and never stored), `label`
        (`blank_label` of the blank: 'folder/file.csv', else its label), `differs`
        (the blank is not the sample's reference slot), `n_blanks` (how many blanks
        the sample holds -- the export names the blank whenever this exceeds one, so
        a one-file-per-sample format keeps its export strings byte-identical while a
        pooled sample discloses every series' blank, owner-ruled), and `note` (a
        chosen `solvent_blank_item_id` that no longer resolves, else '').
        """
        blank, source = self.workspace.solvent_reference_for(
            item_id, use_working=use_working)
        lm = self.workspace.measurements.get(item_id)
        sid = self.workspace.sample_id_of(item_id)
        slot = self.workspace.solvent_reference(sid) if sid else None
        n_blanks = len(self.workspace.blank_candidates(sid)) if sid else 0
        params = (lm.working_params if use_working else lm.committed_params) if lm else {}
        wanted = params.get('solvent_blank_item_id')
        note = ''
        if wanted and source != 'user':
            note = ('the chosen solvent blank is no longer in this sample; '
                    'paired automatically instead')
        return {'item_id': blank.item_id if blank else None, 'source': source,
                'label': blank_label(blank) if blank is not None else '',
                'differs': blank is not None and blank is not slot,
                'n_blanks': n_blanks, 'note': note}

    def _rayleigh_for_sample(self, sample_id: str, fraction: Optional[str] = None):
        """Excess Rayleigh ratios for one molecular-weight fraction of a sample.

        `fraction` selects the concentration series whose `mw_fraction` label equals
        it. For an unfractioned sample every measurement carries None, so the default
        (None) returns the whole series (backward-compatible). Each series is
        subtracted against ITS OWN c = 0 blank, resolved by
        `Workspace.solvent_reference_for` (the user's per-series choice, else the blank
        from the same source file, else the sample's reference slot); the resolver's
        docstring is the normative home of that order. A pure READ of the workspace,
        like every fit path the worker thread runs: how a blank was chosen is
        recomputed on demand by `solvent_blank_status`, never stored. When the blank is
        not the sample reference -- or the sample holds more than one blank -- it is
        named in the result's own `blank_provenance` record.

        NO MEASURED BLANK -- refuse by default, compute on an explicit acknowledgement.
        A sample whose resolver finds no blank at all still raises, exactly as it always
        has, so a headless caller and every existing test are unchanged. Only after
        `acknowledge_unmeasured_blank(sample_id)` -- transient, never written to a
        session -- does the series compute, against a blank BUILT from the committed
        `solvent_blank_intensity_90deg` (`_typed_blank_for`). Every result then carries
        `blank_measured = False`, which gates every reliability verdict downstream.
        """
        typed_blank_needed = self.workspace.solvent_reference(sample_id) is None
        if typed_blank_needed and sample_id not in self._unmeasured_blank_ack:
            raise ValueError(
                f"Sample {sample_id!r} has no solvent reference (c = 0); SLS "
                f"calibration needs one.")
        cal = self._calibration_for_sample(sample_id)   # per-sample override or session
        k_c = cal.k_c
        n_std = cal.standard_refractive_index
        dark = cal.dark_count_rate
        built: Dict[str, Any] = {}      # blank item_id -> (SLSMeasurement, provenance)
        rr = []
        for lm in self.workspace.sample_measurements(sample_id, 'sls'):
            if lm.committed_params.get('mw_fraction') != fraction:
                continue
            sample = lm.build()
            status = self.solvent_blank_status(lm.item_id)
            if status['item_id'] is None:
                # No measured blank anywhere in this sample. Build one at THIS series'
                # own angles from the typed 90 deg value. Deliberately not cached in
                # `built`: a typed blank has no item_id to key on, and the angle grid
                # differs per series, so it is built per series.
                solvent, which = self._typed_blank_for(sample_id, sample, dark)
                blank_provenance = ''
                blank_measured = False
            else:
                ref_lm = self.workspace.measurements[status['item_id']]
                if ref_lm.item_id not in built:
                    # The BLANK can be trace-derived too -- a c = 0 derived point lands
                    # in a blank slot through the ordinary regroup -- so its origin
                    # belongs on the same disclosure as the series'. Built once per
                    # distinct blank.
                    built[ref_lm.item_id] = (
                        ref_lm.build(), self.derived_intensity_provenance(ref_lm.item_id))
                solvent, blank_provenance = built[ref_lm.item_id]
                blank_measured = True
                which = ''
                if status['differs'] or status['n_blanks'] > 1:
                    which = 'solvent blank: {0}'.format(status['label'])
                    if status['source'] == 'user':
                        which += ' (user-selected)'
            rr.append(sls_engine.compute_excess_rayleigh_ratio(
                sample, solvent, calibration_constant=k_c,
                standard_refractive_index=n_std, dark_count_rate=dark,
                # Carried so a derived point's export says where its intensities
                # came from; '' for an ordinary loaded measurement.
                derived_provenance=self._join_derived_provenance_parts(
                    self.derived_intensity_provenance(lm.item_id),
                    blank_provenance),
                # WHICH blank, in its own record: the derivation reducers must never
                # read it.
                blank_provenance=which,
                blank_measured=blank_measured))
        return rr

    # ---- the typed 90 deg solvent blank ----

    def committed_typed_blank(self, sample_id: Optional[str]) -> Optional[float]:
        """The sample's COMMITTED `solvent_blank_intensity_90deg`, or None.

        A sample-shared parameter, so every SLS member carries the same value and the
        first one answers. Reads COMMITTED params like every analysis path: an
        un-applied edit never reaches a fit.
        """
        if sample_id is None or sample_id not in self.workspace.samples:
            return None
        for lm in self.workspace.sample_measurements(sample_id, 'sls'):
            v = lm.committed_params.get('solvent_blank_intensity_90deg')
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return None
        return None

    def validate_typed_blank(self, sample_id: str,
                             i90: Optional[float] = _UNSET) -> str:
        """'' if this sample's typed blank is usable, else why not -- the commit-time
        courtesy check. The SAME predicate runs again in `_typed_blank_for`, which is
        where the blank is built and therefore where the guard has to bind: the dark
        count is a calibration field the user can edit after committing this value.

        Pass `i90` to judge a value the user has typed but not yet committed; omit it
        to judge what is committed now.
        """
        value = self.committed_typed_blank(sample_id) if i90 is _UNSET else i90
        dark = self._calibration_for_sample(sample_id).dark_count_rate
        return _typed_blank_problem(value, dark)

    def sample_has_measured_blank(self, sample_id: Optional[str]) -> bool:
        """Does this sample hold ANY measured c = 0 blank? The question the Data tab's
        greying and the SLS tab's confirmation both ask -- one predicate, so the field
        cannot be editable while the run is asking, or the reverse.

        A sample the workspace does not hold answers False: it holds nothing, which is
        `committed_typed_blank`'s shape one screen up. `Workspace.solvent_reference`
        subscripts `samples` and so RAISES on an unknown id; a tab caches its
        `sample_id` across workspace edits, so the guard binds here rather than at each
        of the four call sites.
        """
        if sample_id is None or sample_id not in self.workspace.samples:
            return False
        return self.workspace.solvent_reference(sample_id) is not None

    def acknowledge_unmeasured_blank(self, sample_id: str) -> None:
        """Record that the user confirmed running this sample without a measured blank.

        TRANSIENT and never persisted: it lives only on this controller instance, so a
        reloaded session asks again (`load_session` clears it). The controller stays
        Qt-free -- the asking is the GUI's job; this is only where the answer lands.
        """
        self._unmeasured_blank_ack.add(sample_id)

    def clear_unmeasured_blank_acknowledgement(
            self, sample_id: Optional[str] = None) -> None:
        """Forget one sample's acknowledgement, or all of them (`sample_id=None`)."""
        if sample_id is None:
            self._unmeasured_blank_ack.clear()
        else:
            self._unmeasured_blank_ack.discard(sample_id)

    def has_unmeasured_blank_acknowledgement(self, sample_id: str) -> bool:
        """Has this sample's blankless run already been confirmed in this session?"""
        return sample_id in self._unmeasured_blank_ack

    # The sentences a computed or absent blank writes into `blank_provenance`, and from
    # there into the affected export columns' Comments cells. Each NAMES THE FAILED
    # PRECONDITION rather than a proxy (invariant 4): what was not measured, and what
    # stood in for it.
    _TYPED_BLANK_NOTE = (
        'solvent blank not measured: computed from a typed 90° intensity under an '
        'angle-independent-solvent assumption')
    _NO_BLANK_NOTE = 'solvent blank not measured: no solvent subtraction applied'

    def _typed_blank_for(self, sample_id: str, sample, dark_count_rate: float):
        """The stand-in solvent blank for a series with no measured c = 0 point.

        Returns `(SLSMeasurement, provenance_sentence)`. The blank is an ORDINARY
        measurement at the series' own angles, so `compute_excess_rayleigh_ratio`
        gains no branch (invariant 1) and every later step -- volume correction,
        refractive-index correction, calibration -- is byte-identical to the
        measured-blank path.

        The model. With `d` the committed dark count rate and `I90` the typed 90 deg
        intensity, the constructed blank is

            I_solvent(theta) = d + (I90 - d) / sin(theta)

        so that after the pipeline's own dark subtraction the NET solvent term is
        `(I90 - d) / sin(theta)`. The dark count is additive and is not volume-scaled,
        which is why `d` sits outside the 1/sin(theta): a measured blank reads
        `d + I_true/sin(theta)`.

        Empty or zero `I90` both mean NO SOLVENT SUBTRACTION, and they are the same
        case -- zero is not a special code. The blank is then `d` at every angle, so
        the engine still removes the dark count from the sample and subtracts nothing
        else. Neither of the tempting alternatives is right: an array of zeros would
        CANCEL the sample's own dark subtraction, and `d - d/sin(theta)` has net solvent
        term `-d/sin(theta)`, which ADDS intensity at every angle -- including 90 deg,
        where it adds exactly `d`. The three forms coincide only when `d = 0`.

        The guard lives HERE, where the blank is constructed, not at commit alone
        (rule 12). The dark count is a calibration field the user may edit on the SLS tab
        after committing the typed value, so a commit-time check is invalidated by an
        ordinary edit; `validate_typed_blank` runs the same predicate at commit as a
        courtesy. The finiteness check is on the CONSTRUCTED ARRAY, so guard and operation
        are one computation and no gap sits between `I90 / sin(theta)` and
        `d + (I90 - d)/sin(theta)`.

        WHICH TAIL EACH HALF CLOSES, stated rather than implied. The array check closes
        the OVERFLOW tail only: `I90 - d >= 0` is already guaranteed by the scalar
        predicate, and `sin(theta)` is confined to (0, 1] because the data model refuses
        0 and 180 deg, so dividing by it only increases magnitude -- the result can
        saturate upward but can never underflow below `d`. The lower bound is therefore
        closed by `_typed_blank_problem`, not here. The `blank < d` term is kept as a
        cheap structural assertion of that reasoning, not because an input can reach it;
        if the angle bounds or the scalar predicate ever change, it is what fires.
        """
        angles = np.asarray(sample.angles_deg, dtype=float)
        d = float(dark_count_rate)
        i90 = self.committed_typed_blank(sample_id)
        if i90 is None or i90 == 0.0:
            blank = np.full(angles.shape, d, dtype=float)
            note = self._NO_BLANK_NOTE
        else:
            problem = _typed_blank_problem(i90, d)
            if problem:
                raise ValueError(problem)
            blank = d + (i90 - d) / np.sin(np.radians(angles))
            note = self._TYPED_BLANK_NOTE
        # Two-sided finiteness on the RESULT, not on the inputs: a huge I90 overflows
        # at the smallest angle, a tiny one underflows, and only the constructed array
        # sees the cross-product of both with every angle in the series.
        if not np.all(np.isfinite(blank)) or np.any(blank < d):
            smallest = float(np.min(angles)) if angles.size else float('nan')
            raise ValueError(
                f"The typed solvent blank intensity at 90 deg ({i90!r}) is too large to "
                f"use with the committed dark count rate ({d!r}): scaled by 1/sin(theta) "
                f"it overflows at this series' smallest angle ({smallest:g} deg). Enter a "
                f"smaller value, or clear it to run with no solvent subtraction.")
        return replace(sample, intensities=blank,
                       concentration_g_per_mL=0.0), note

    def run_zimm(self, sample_id: str, method: str = 'zimm',
                 fraction: Optional[str] = None):
        rr = self.masked_rayleigh(sample_id, fraction)
        n_conc = len({round_concentration_for_identity(r.concentration_g_per_mL)
                      for r in rr if r.concentration_g_per_mL != 0})
        if n_conc < 2:
            raise ValueError(
                f"Only {n_conc} concentration(s) remain after masking; a "
                f"{method.capitalize()} extrapolation needs at least two. Use "
                f"Debye or Guinier for single-concentration analysis.")
        res = sls_engine.zimm_analysis(rr, method=method, estimator=self.settings.se_estimator)
        self.results[('zimm', sample_id, method, fraction)] = res
        if not getattr(res, 'blank_measured', True):
            # Owner-ruled: a blankless fit is not offered to Cross-Sample, and what that
            # BUYS is an outcome -- the sample result and a saved session stay
            # byte-identical to a run that never made it. Withholding this write is the
            # only way to deliver it: `compute_scaling` reads these stored values and
            # never the candidate rosters, and filtering later would need a field on
            # `SampleResult` the same ruling leaves untouched. The fit is unaffected
            # (returned, cached, shown with its flags, exported) and the non-write is
            # disclosed in the pre-run confirmation and on the status line after.
            return res
        # attach Mw/Rg/A2 to this fraction's sample result, marking provenance.
        # Each bundle guard protects 'user' ALONE: an explicit Run is a deliberate
        # act, so it replaces a pick, and the GUI reports what it replaced
        # (SampleResult docstring; the passive auto_select_* refreshers still
        # spare every deliberate tier).
        s = self.workspace.samples[sample_id].result_for(fraction)
        if s.mw_source != 'user':             # never clobber a manual Mw
            s.mw_g_per_mol = res.mw_g_per_mol
            s.mw_source = 'computed'
            s.calibrated = res.calibrated
            s.dn_dc_supplied = res.dn_dc_supplied   # second scale flag (rule 10)
            s.mw_apparent = res.is_apparent      # False: thermodynamic
            # The whole bundle, mirroring the a2 arm below: an SE and its estimator
            # tag must never be left stale beside a new value (invariant 8) -- the
            # replaced value's ± could be another fit's entirely, now that a Run
            # replaces a pick.
            s.mw_se = unc.se_or_none(getattr(res, 'mw_se', None))
            s.mw_se_estimator = (getattr(res, 'se_estimator', None)
                                 if s.mw_se is not None else None)
            s.mw_model_dependent = False         # Zimm is model-AGNOSTIC -> clear any stale
            s.mw_model = ''                      # form-factor flag from a prior pick (rule 10)
            s.mw_degenerate = False              # ... and any degenerate flag (rule 10)
            cal_note = _scale_caveat(res)
            s.mw_label = (
                f"{method.capitalize()} extrapolation, {n_conc} concentrations "
                f"(thermodynamic{cal_note})")
            set_result_key(s, 'mw', *extrapolation_result_key(method))
        if s.rg_source != 'user':             # nor a manual Rg
            s.rg_nm = res.rg_nm
            s.rg_source = 'computed'
            s.rg_apparent = res.is_apparent      # False: Zimm/Berry are thermodynamic
            s.rg_se = unc.se_or_none(getattr(res, 'rg_se', None))   # full bundle, as mw above
            s.rg_se_estimator = (getattr(res, 'se_estimator', None)
                                 if s.rg_se is not None else None)
            s.rg_model_dependent = False         # Zimm Rg is a slope (model-agnostic) -> clear
            s.rg_model = ''                      # any stale form-factor flag (rule 10)
            s.rg_degenerate = False              # ... and any degenerate flag (rule 10)
            # An explicit Run writes the value even when its flag is down (flag, never
            # withhold); the label carries the reason, as the DLS tiers' `rh_label` does.
            s.rg_label = (
                f"{method.capitalize()} extrapolation, {n_conc} concentrations "
                f"(thermodynamic, R² = {_fmt_r2(float(res.r_squared))})"
                f"{_flag_suffix(res, 'rg_reliable')}")
            set_result_key(s, 'rg', *extrapolation_result_key(method))
        if s.a2_source != 'user':             # nor a hand-set (reserved) A2
            # Write the whole A2 bundle atomically, mirroring set_sample_a2 — an SE,
            # calibration flag and label must never be left stale beside a new value
            # (invariant 8: a uncertainty never travels with the wrong estimate).
            s.a2_mol_mL_per_g2 = res.a2_mol_mL_per_g2
            s.a2_source = 'computed'
            s.a2_se = unc.se_or_none(getattr(res, 'a2_se', None))
            # Copy the estimator only alongside a real SE -- a source fit sets se_estimator
            # unconditionally, so guard the orphan case (Clause A; see ResultCandidate.__post_init__).
            s.a2_se_estimator = getattr(res, 'se_estimator', None) if s.a2_se is not None else None
            s.a2_calibrated = res.calibrated
            s.a2_dn_dc_supplied = res.dn_dc_supplied
            a2_cal_note = _scale_caveat(res)
            s.a2_label = (
                f"{method.capitalize()} extrapolation, {n_conc} concentrations "
                f"(thermodynamic{a2_cal_note}, R² = {_fmt_r2(float(res.r_squared))})")
            set_result_key(s, 'a2', *extrapolation_result_key(method))
        return res

    def sls_solute_refractive_index(self, sample_id: str,
                                    fraction: Optional[str] = None) -> Optional[float]:
        """The hand-entered solute refractive index for a sample's SLS series (or None).

        Read from the committed SLS params (like dn/dc); it feeds the full-Mie form-factor
        fit's contrast m = n_solute/n_solvent. Blank for every non-Mie fit."""
        for lm in self.workspace.sample_measurements(sample_id, 'sls'):
            if lm.committed_params.get('mw_fraction') == fraction:
                n = lm.committed_params.get('solute_refractive_index')
                if n is not None:
                    return float(n)
        return None

    def _sls_mie_contrast(self, sample_id: str,
                          fraction: Optional[str] = None) -> Optional[float]:
        """The Mie contrast m = n_solute/n_solvent for a sample's SLS series, formed from a
        SINGLE measurement's committed params (both indices from the same row, so a masked or
        other-concentration row can never supply a mismatched half). None when no measurement
        carries a solute RI -> the engine raises the clear 'solute RI unset' error."""
        for lm in self.workspace.sample_measurements(sample_id, 'sls'):
            if lm.committed_params.get('mw_fraction') != fraction:
                continue
            n_sol = lm.committed_params.get('solute_refractive_index')
            n_med = lm.committed_params.get('solvent_refractive_index')
            if n_sol is not None and n_med:
                return float(n_sol) / float(n_med)
        return None

    def sls_shell_refractive_index(self, sample_id: str,
                                   fraction: Optional[str] = None) -> Optional[float]:
        """The hand-entered SHELL refractive index for a sample's SLS series (or None).

        Read from the committed SLS params (like the core solute RI); it feeds the core-shell
        form-factor fit's shell contrast m2 = n_shell/n_solvent. Blank for every non-core-shell
        fit."""
        for lm in self.workspace.sample_measurements(sample_id, 'sls'):
            if lm.committed_params.get('mw_fraction') == fraction:
                n = lm.committed_params.get('shell_refractive_index')
                if n is not None:
                    return float(n)
        return None

    def _sls_core_shell_contrasts(self, sample_id: str, fraction: Optional[str] = None):
        """The core-shell contrasts (m1, m2) = (n_core/n_solvent, n_shell/n_solvent) for a
        sample's SLS series, both read from a SINGLE measurement's committed params (never core
        RI from one row and solvent RI from another -- the single-row rule; solute_refractive_index
        is reused as the core index). m1 and m2 are formed INDEPENDENTLY, so a missing core RI
        yields (None, m2) and a missing shell RI yields (m1, None) -- the engine then raises for
        the ACTUALLY-missing index (not always the core), so the Run-error names the right one.
        Reads from the first row carrying either RI (skipping the bare c=0 solvent reference)."""
        for lm in self.workspace.sample_measurements(sample_id, 'sls'):
            if lm.committed_params.get('mw_fraction') != fraction:
                continue
            n_core = lm.committed_params.get('solute_refractive_index')
            n_shell = lm.committed_params.get('shell_refractive_index')
            n_med = lm.committed_params.get('solvent_refractive_index')
            if n_med and (n_core is not None or n_shell is not None):
                m1 = float(n_core) / float(n_med) if n_core is not None else None
                m2 = float(n_shell) / float(n_med) if n_shell is not None else None
                return m1, m2
        return None, None

    def run_form_factor(self, sample_id: str, model: str = 'sphere',
                        fraction: Optional[str] = None):
        """Full-angular form-factor P(q) fit (sphere / Gaussian coil / ellipsoid / Mie).

        Fits the whole masked SLS set: one non-zero concentration -> apparent (1/Mw_app,
        Rg); two or more -> thermodynamic (1/Mw, Rg, A2). Extends the Zimm/Guinier
        expansion to larger qRg using the full form factor (analysis/form_factor.py).
        Like the other SLS fits it honours the (sample, fraction) mask and reads the
        settings-seeded SE estimator; the analysis stays pure (controller-mediated).

        Returns a single ``FormFactorResult`` for the sphere / Gaussian-coil / Mie models,
        but a LIST of ``EllipsoidFormFactorResult`` for the ellipsoid and a LIST of
        ``CoreShellFormFactorResult`` for the core-shell sphere -- both are multi-valued fits
        (prolate/oblate ellipsoids, or several radius pairs, reproduce the same P(q)), so every
        fitting solution is returned and flagged ``degenerate``. Callers branch on the model.

        The Mie model needs the contrast m = n_solute/n_solvent, formed from the hand-entered
        solute refractive index (Data tab); the core-shell model needs BOTH the core
        (solute_refractive_index) and shell (shell_refractive_index) indices -> (m1, m2). If a
        required index is unset the pure engine raises a clear ValueError, which the GUI worker
        surfaces as a flag ("enter the refractive index"). The contrasts are fixed user inputs,
        never fitted (invariant 3).
        """
        res = self._form_factor_result(sample_id, model, fraction)
        key = ('form_factor', sample_id, model, fraction)
        # STAMP-BEFORE-PUBLISH INVARIANT: write the freshness stamp into
        # `_form_factor_run_sig` BEFORE publishing the result into self.results. This runs on
        # the worker thread while a Cross-Sample candidate read runs on the main thread; the
        # two dict writes are not atomic together, so the ordering is what guarantees
        # 'result visible ⟹ stamp visible' under the GIL. Reversed, a reader could see the
        # result present but UNSTAMPED and `_form_factor_solutions` would exclude a
        # fully-published fit as stale. Never reorder these two lines.

        # KNOWN RESIDUAL, the opposite direction and only on a RE-RUN over an existing stored
        # fit: between the two writes the stamp is already the NEW signature while
        # self.results[key] still holds the PREVIOUS result, so a main-thread reader whose
        # `current` has moved on can briefly match the new stamp and read the old result,
        # offering a stale form-factor Rg as fresh until the next refresh. The window is one
        # bytecode wide and self-corrects on the next read; the airtight fix is to publish
        # (result, stamp) as a SINGLE atomic dict value. Kept as parallel dicts by decision
        # unless the race is shown to bite.
        self._form_factor_run_sig[key] = self._form_factor_run_signature(sample_id, fraction)
        self.results[key] = res
        return res

    def _form_factor_result(self, sample_id: str, model: str,
                            fraction: Optional[str] = None):
        """Run one form-factor P(q) fit against the CURRENT masked data and return it,
        WITHOUT storing (the pure dispatch shared by `run_form_factor`, which stores, and
        by the Cross-Sample candidate builders, which re-run it fresh so a pooled Rg/Mw
        always saw the same masked data as the Zimm candidates beside it).

        Returns a single ``FormFactorResult`` (sphere / coil / Mie) or a LIST of solutions
        (ellipsoid / core-shell). Reads the settings-seeded SE estimator and, for Mie /
        core-shell, the committed contrast RI(s) -- an unset RI raises a clear ValueError,
        exactly as `run_form_factor` documents. The contrasts are fixed user inputs, never
        fitted (invariant 3); the analysis stays pure (controller-mediated)."""
        rr = [r for r in self.masked_rayleigh(sample_id, fraction)
              if r.concentration_g_per_mL != 0]
        if model == ff.ELLIPSOID:
            return sls_engine.ellipsoid_form_factor_analysis(
                rr, estimator=self.settings.se_estimator)
        if model == ff.CORE_SHELL:
            # m1 (core), m2 (shell) = n_core/n_solvent, n_shell/n_solvent, ALL read from the
            # same measurement's committed params (never mixed across rows). (None, None) ->
            # the engine raises a clear "core-shell RI unset" ValueError.
            m1, m2 = self._sls_core_shell_contrasts(sample_id, fraction)
            return sls_engine.core_shell_form_factor_analysis(
                rr, m1, m2, estimator=self.settings.se_estimator)
        if model == ff.MIE:
            # m = n_solute/n_solvent, BOTH read from the same measurement's committed params
            # (never n_solute from one row and n_solvent from another). None -> the engine
            # raises a clear "solute RI unset" ValueError.
            m = self._sls_mie_contrast(sample_id, fraction)
            return sls_engine.form_factor_analysis(
                rr, model=model, estimator=self.settings.se_estimator, m=m)
        return sls_engine.form_factor_analysis(
            rr, model=model, estimator=self.settings.se_estimator)

    def set_manual_mw(self, sample_id: str, mw_g_per_mol: float,
                      fraction: Optional[str] = None) -> None:
        """Record a hand-entered Mw for a sample (e.g. characterized in water).

        Marks the provenance 'user' so later runs do not overwrite it and scaling
        plots can prefer it -- the correct path for PVP in a deep eutectic solvent,
        where the in-situ Mw is biased by co-solvent adsorption. A hand-entered Mw is
        treated as a trusted (thermodynamic, calibrated) value.
        """
        # Guarded like its set_manual_rh/set_manual_rg siblings -- this one had no numeric
        # check at all, and a hand-typed Mw is the value the scaling plots PREFER over a
        # computed one, so a non-finite entry would travel furthest of the three
        # (2026-07-31 spec-01 re-review).
        if not (math.isfinite(mw_g_per_mol) and mw_g_per_mol > 0):
            raise ValueError(
                f"mw_g_per_mol must be a positive finite number, got {mw_g_per_mol!r}.")
        r = self.workspace.samples[sample_id].result_for(fraction)
        r.set_mw(mw_g_per_mol, source='user')
        r.mw_apparent = False
        r.calibrated = True               # trusted value; also clears any stale flag from a prior run
        r.dn_dc_supplied = True           # ... and the optical-scale flag beside it (rule 10):
        #                                   a hand-entered Mw needs no dn/dc to be on a real scale
        r.mw_label = 'user-entered'
        r.mw_se = None                    # a hand-entered value carries no statistical SE
        r.mw_se_estimator = None          # ... and no estimator provenance; clear any stale tag
        r.mw_model_dependent = False      # a hand-entered Mw assumes no shape; clear any stale flag
        r.mw_model = ''
        r.mw_degenerate = False           # ... and any degenerate flag (rule 10)
        set_result_key(r, 'mw', 'user', None)

    # --- depolarized light scattering (static, Phase 1) ---
    def compute_depolarization(self, *, i_vv: Optional[float] = None,
                               i_vh: Optional[float] = None,
                               rho_v: Optional[float] = None,
                               dark_count: float = 0.0,
                               i_vv_se: Optional[float] = None,
                               i_vh_se: Optional[float] = None):
        """Static depolarization analysis (assumes vertically polarized incident light).

        Provide EITHER the VV and VH intensities (in the SLS file's own units; only
        their ratio matters), OR a depolarization ratio rho_v directly. Returns a
        DepolarizationResult (rho_v, rho_u, delta^2, Cabannes factor, validity).
        Pure pass-through to analysis.depolarization -- no state is changed.
        """
        if rho_v is not None:
            # A direct ratio: analyze with unit VV so I_VH / I_VV = rho_v exactly.
            return depol_engine.analyze_depolarization(1.0, float(rho_v))
        if i_vv is None or i_vh is None:
            raise ValueError(
                "Provide both VV and VH intensities, or a depolarization ratio rho_v.")
        return depol_engine.analyze_depolarization(
            float(i_vv), float(i_vh), dark_count=float(dark_count),
            i_vv_se=i_vv_se, i_vh_se=i_vh_se)

    # --- depolarized light scattering (dynamic, Phase 2) ---
    def ddls_correlogram_summary(self, sample_id: str) -> List[Dict[str, Any]]:
        """The sample's DLS correlograms with their polarization tag, for the UI.

        Returns one dict per correlogram: item_id, angle_deg, geometry ('VV'/'VH'/
        'VU'/None), and `paired` (True if its angle has both a VV and a VH). Sorted
        by angle then geometry. The DDLS sub-tab renders this and shows which angles
        will actually pair.

        Combination doctrine rule 2 (disclosure), core/data_models.py: pairing
        eligibility is disclosed on the row, mirroring what run_ddls will use.
        """
        rows: List[Dict[str, Any]] = []
        built = []
        for lm in self.workspace.sample_measurements(sample_id, 'dls'):
            # Read angle/geometry straight from committed params -- do NOT build()
            # the full measurement, which would demand wavelength/polymer/solvent
            # the user has not supplied yet (a freshly-loaded correlogram, e.g. a
            # Zetasizer file, carries none of those). A measurement with no angle
            # cannot pair for DDLS, so skip it from the summary until it is set.
            p = lm.committed_params
            angle = p.get('angle_deg')
            if angle is None:
                continue
            geom = p.get('analyzer_geometry')
            built.append((lm.item_id, float(angle), geom))
        paired = _ddls_paired_angles((a, g) for _, a, g in built)
        for item_id, angle, geom in built:
            rows.append({
                'item_id': item_id, 'angle_deg': angle, 'geometry': geom,
                'paired': round_angle_for_identity(angle) in paired,
            })
        rows.sort(key=lambda r: (r['angle_deg'], r['geometry'] or ''))
        return rows

    def run_ddls(self, sample_id: str, rod_length_nm: Optional[float] = None,
                 cumulant_order: Optional[int] = None, exclude_angles=(),
                 region: Optional[RegionState] = None):
        """Pair a sample's VV/VH correlograms and extract rotational diffusion D_r.

        Each correlogram is fitted from the cumulant seed home -- at the Settings order
        unless `cumulant_order` is given -- and, when the tab's "use the Correlogram
        tab's delay window and baseline" box is ticked, on its own reading of `region`
        (the result's notes then say so).

        Gathers the sample's DLS correlograms, groups them by polarization tag
        (analyzer_geometry), pairs a VV with a VH at each shared angle, fits each
        with the cumulant engine to its field decay rate Gamma, and runs
        analyze_ddls (D_r = (Gamma_VH - Gamma_VV)/6; D_t from the VV channel).

        Multiple VV (or VH) correlograms at the SAME angle are treated as replicates
        and averaged: each replicate is fitted, and the per-replicate field rates are
        averaged to one Gamma per (angle, geometry) -- rather than silently keeping
        one. (Averaging the fitted rate is grid-agnostic; to average the correlograms
        themselves -- better for weak VH -- pre-average via the sidebar "Average
        correlation functions" action and tag the result.)

        Returns (DDLSResult, info) where info records the angles paired, any
        VV-only / VH-only / untagged correlograms skipped, and the replicate counts
        per angle. Raises ValueError if no angle has both a VV and a VH (nothing to
        pair).

        The cross-bucket identity check below REFUSES a mixed set (e.g. two
        concentrations split across angles) where `analyze_gamma_q2` merely warns
        on the same mix. That asymmetry is deliberate (owner-ruled 2026-08-13): by
        the combination doctrine's own criterion both are degraded-not-meaningless,
        but warning would matter only for a workflow that exists solely in the
        synthetic generator (DPLS is hardware-gated; no real instrument here
        produces a multi-concentration DDLS set), so converting a fresh guard to a
        warn for taxonomy purity is churn. One real wrongly-refused set reopens it.
        """
        vv: Dict[float, list] = {}
        vh: Dict[float, list] = {}
        untagged = []
        angle_geom = []
        # Each correlogram's own fit inputs: the seeds, plus its reading of a checked
        # region. Keyed by the built measurement's identity, which the buckets hold.
        seeds_for: Dict[int, dict] = {}
        base_seeds = {} if cumulant_order is None else {'order': int(cumulant_order)}
        for lm in self.workspace.sample_measurements(sample_id, 'dls'):
            m = lm.build()
            seeds_for[id(m)] = self._seed_cumulant_kwargs(
                {**base_seeds,
                 **self.analysis_region_kwargs(lm.item_id, 'cumulant', region)})
            a = round_angle_for_identity(m.angle_deg)
            state = _polarization_state(m.analyzer_geometry)
            # The helper re-applies both forms; they are idempotent, so feeding
            # the classified pair keeps this the ONE classification pass (the
            # vv/vh dicts and the pairing can never disagree about a run).
            angle_geom.append((a, state))
            if state == 'VV':
                vv.setdefault(a, []).append(m)
            elif state == 'VH':
                vh.setdefault(a, []).append(m)
            else:                                   # VU or None -> not pairable
                untagged.append((lm.item_id, m.analyzer_geometry))

        # THE pairing rule, shared with ddls_correlogram_summary (the UI preview).
        paired_angles = sorted(_ddls_paired_angles(angle_geom))
        if not paired_angles:
            raise ValueError(
                "No angle has both a VV and a VH correlogram. Tag each correlogram's "
                "polarization (VV/VH) in the Data tab; DDLS pairs them by angle.")
        # Drop user-excluded outlier angles.
        if exclude_angles:
            paired_angles = [a for a in paired_angles
                             if not _value_in(a, exclude_angles)]
            if not paired_angles:
                raise ValueError('All paired angles are excluded — nothing to fit.')

        paired = [m for a in paired_angles for m in (vv[a] + vh[a])]

        # ONE identity question over every run entering the fit -- the same
        # `{m.sample_key for m in ...}` comparison analyze_gamma_q2 makes of its angle set,
        # so both multi-angle analyses answer "is this one sample?" the same way. It spans
        # the buckets rather than sitting in one because every bucket's Gamma feeds a
        # single D_t/D_r fit: a per-bucket check says nothing about that fit, is vacuous
        # for the common bucket of one, and never compares VV against VH. Angle is absent
        # by construction (it MUST differ here); mw_fraction rides alongside sample_key
        # because fractions are analysed independently.
        keys = {(m.sample_key, m.mw_fraction) for m in paired}
        if len(keys) > 1:
            raise ValueError(
                "The paired VV/VH correlograms are not all from one sample and "
                "condition, so DDLS cannot combine them: their sample identity "
                "(polymer, solvent, concentration, temperature) or Mw fraction differs. "
                "The sidebar groups a concentration series as ONE sample, so a second "
                "concentration reaches this analysis looking like a replicate. Re-tag or "
                "remove the odd run in the Data tab.")

        left_out: List[str] = []

        def _run_name(m) -> str:
            return f'{m.angle_deg:g} deg {m.analyzer_geometry or "untagged"}'

        def _avg_gamma(measurements: list) -> tuple:
            """Mean field decay rate over replicate correlograms (one fit each), with the
            (run, fit) pairs that entered it and the refusals of the runs that did not.

            Identity is settled above, across the whole fit; this only averages.
            Under a checked region a replicate whose fit refuses the inputs the region
            gives it (a delay window holding too few of its channels) is left out and
            named in a note, as the Correlogram tab records such a run, rather than
            failing the whole set; unchecked, a refusal raises as it always has.
            """
            fits, refused = [], []
            for m in measurements:
                try:
                    fits.append((m, dls_engine.fit_cumulants(m, **seeds_for[id(m)])))
                except ValueError as exc:
                    if region is None:
                        raise
                    refused.append(f'{_run_name(m)}: {exc}')
            # DROP the failed replicates and average the rest, which is what the
            # replicate-averaging path already does. A bare `np.mean` was safe only while
            # an unfittable correlogram RAISED; now that it returns a flagged NaN, one bad
            # replicate NaN'd the whole angle, the point was dropped, and a downstream
            # raise blamed the VV/VH pairing -- naming a cause that was not the one.
            # NaN when NOTHING survives: the caller leaves out an angle with no VV rate.
            # A rate at or below zero is no decay rate: `analyze_ddls` refuses one, and
            # the Gamma-vs-q^2 fits drop one, so it enters no mean here either.
            entered = [(m, f) for m, f in fits
                       if f.success and np.isfinite(f.gamma_s_inv) and f.gamma_s_inv > 0]
            # A fit that returned but did not enter is named with the refusals, so the notes
            # account for every replicate supplied, as the Gamma-vs-q^2 fits name a dropped Gamma.
            kept = {id(f) for _m, f in entered}
            refused.extend(f'{_run_name(m)}: its fit returned no usable decay rate'
                           for m, f in fits if id(f) not in kept)
            gamma = (float(np.mean([float(f.gamma_s_inv) for _m, f in entered]))
                     if entered else float('nan'))
            return gamma, entered, refused

        points = []
        replicates: Dict[float, tuple] = {}
        relayed: List[tuple] = []     # (run name, its passive notes), each fit that entered
        entered_runs: list = []       # the runs whose fit entered a fitted angle's mean
        for a in paired_angles:
            g_vv, vv_in, vv_refused = _avg_gamma(vv[a])
            g_vh, vh_in, vh_refused = _avg_gamma(vh[a])
            if not math.isfinite(g_vv):
                # D_t is a through-origin fit that drops no point, so an angle with no VV
                # rate is left out here and named rather than failing the whole set with a
                # cause that is not the one. A missing VH rate stays: `analyze_ddls` keeps
                # the angle's VV rate in D_t, drops it from D_r, and flags D_r for it.
                cause = ('no VV run could be fitted' if math.isfinite(g_vh)
                         else 'no VV run and no VH run could be fitted')
                refused = vv_refused + vh_refused
                why = f' ({"; ".join(refused)})' if refused else ''
                left_out.append(f'{a:g} deg — {cause}{why}')
                continue
            left_out.extend(vv_refused + vh_refused)
            relayed.extend((_run_name(m), [n for notes in _collect_passive_notes(f).values()
                                           for n in notes])
                           for m, f in vv_in + vh_in)
            entered_runs.extend(m for m, _f in vv_in + vh_in)
            replicates[a] = (len(vv_in), len(vh_in))     # the runs averaged, not supplied
            # q from the optics of a run whose rate entered, as T and eta are read below.
            q = phys.scattering_vector_q_m(
                a, vv_in[0][0].wavelength_nm, vv_in[0][0].solvent_refractive_index)
            points.append(depol_engine.DDLSRatePoint(
                angle_deg=a, q_m_inv=q, gamma_vv_s_inv=g_vv, gamma_vh_s_inv=g_vh))
        if not points:
            raise ValueError(
                'No paired angle has a VV run that could be fitted, so DDLS has nothing '
                'to fit. ' + ' '.join(f'Left out of the fit: {entry}.' for entry in left_out))
        fitted_angles = list(replicates)

        # Temperature and viscosity come from the runs that actually enter the fit, not
        # from whichever correlogram the workspace happened to list first: a VU or
        # untagged run is skipped as unpairable, and a left-out angle's runs and a refused or
        # unfitted replicate never enter, so their eta has no business setting the
        # Stokes-Einstein scale for D_t / D_r. (Viscosity is deliberately outside the
        # identity check -- see check_replicate_identity -- so it can genuinely differ.)
        t_k = entered_runs[0].temperature_K
        eta = next((m.viscosity_Pa_s for m in entered_runs
                    if m.viscosity_Pa_s is not None), None)

        res = depol_engine.analyze_ddls(
            points, temperature_K=t_k, viscosity_Pa_s=eta, rod_length_nm=rod_length_nm,
            estimator=self.settings.se_estimator)
        extra = [f'Left out of the fit: {entry}.' for entry in left_out]
        extra.append(self.region_note(region))
        # Each entered fit's passive notes, relayed as replicate averaging relays them:
        # this result is the only surface showing D_t and D_r. Notes, not verdicts.
        extra.extend(_relayed_run_notes(relayed))
        res.notes = ' '.join(n for n in [res.notes, *extra] if n)
        info = {
            'paired_angles': fitted_angles,
            'vv_only': sorted(set(vv) - set(vh)),
            'vh_only': sorted(set(vh) - set(vv)),
            'untagged': untagged,
            'replicates': replicates,
            'n_replicate_angles': sum(1 for (nv, nh) in replicates.values()
                                      if nv > 1 or nh > 1),
        }
        self.results[('ddls', sample_id)] = res
        return res, info

    def ddls_shape(self, sample_id: str, model: str = 'both',
                   result=None) -> Dict[str, Any]:
        """Particle dimensions from a DDLS D_t, D_r (model-dependent).

        Runs the shape inverse on `result` -- the fit the DDLS tab shows, which the tab
        passes -- or, when none is given, on the sample's stored DDLS result: the
        **sphere** model (R from D_r, with a sphericity check vs Rh), the **rod** model
        (L, d from the Tirado inversion), and the **ellipsoid** model (a, b from the Perrin
        inversion). `model` selects 'sphere', 'rod', 'ellipsoid', or 'both'/'all'
        (default 'both' = sphere+rod+ellipsoid). Returns a dict with the requested result
        objects. NB: `out['ellipsoid']` is a **list** (a given D_t/D_r admits up to three
        ellipsoids -- 1 prolate + 2 oblate), whereas 'sphere'/'rod' are single objects;
        the two display/export call sites special-case it. These are dimensions of an
        ASSUMED shape, not measurements -- the result objects carry that caveat.

        It never runs DDLS itself: a re-run here would fit on the defaults -- no region,
        every angle back in -- and put shapes from that fit beside the D_t/D_r the tab
        shows. With neither a `result` nor a stored fit it raises, naming the Run. It
        also raises if D_t/D_r are non-positive or the solvent viscosity is missing (the
        shape models need it; the decay rates do not).
        """
        res = result if result is not None else self.results.get(('ddls', sample_id))
        if res is None:
            raise ValueError(
                'No DDLS fit for this sample yet — press Run on the DDLS tab; the shape '
                'models are computed from that fit.')
        # isfinite as well as > 0: a bare `> 0` admits +inf, which would sail past this
        # purpose-written message into the physics layer and surface as its generic
        # refusal instead (2026-07-31 spec-01; the same gate shape hardened across
        # analysis/dls/ and analysis/depolarization.py in that change).
        if not (math.isfinite(res.d_t_m2_s) and res.d_t_m2_s > 0
                and math.isfinite(res.d_r_rad2_s) and res.d_r_rad2_s > 0):
            raise ValueError(
                "Shape models need positive, finite D_t and D_r; the DDLS result has a "
                "non-physical D_r (Gamma_VH <= Gamma_VV) -- check the VV/VH pairing.")
        # THE (T, eta) that produced this result, read off the result itself rather than
        # re-scanned here. The old scan walked every DLS measurement of the sample and took
        # the first viscosity it found -- including a VU or untagged run that run_ddls had
        # skipped as unpairable -- so the shape radii and the DDLS Rh_t could be computed
        # at different viscosities and disagree, reporting a sphere as non-spherical.
        t_k, eta = res.temperature_K, res.viscosity_Pa_s
        if eta is None:
            raise ValueError(
                "The shape models need the solvent viscosity; set it in the Data tab.")
        out: Dict[str, Any] = {}
        # The estimator tag crosses the boundary WITH the SEs it produced: the shape ±
        # derive from `d_t_se`, so the DDLS result's `se_estimator` is theirs too, and
        # the export/GUI label an OLS ± only if the tag arrives (invariant 8 clause A).
        est = getattr(res, 'se_estimator', 'hc3')
        if model in ('sphere', 'both', 'all'):
            out['sphere'] = depol_engine.sphere_dimensions_from_diffusion(
                res.d_t_m2_s, res.d_r_rad2_s, temperature_K=t_k, viscosity_Pa_s=eta,
                d_t_se=res.d_t_se, d_r_se=res.d_r_se, se_estimator=est)
        if model in ('rod', 'both', 'all'):
            out['rod'] = depol_engine.rod_dimensions_from_diffusion(
                res.d_t_m2_s, res.d_r_rad2_s, temperature_K=t_k, viscosity_Pa_s=eta,
                d_t_se=res.d_t_se, d_r_se=res.d_r_se, se_estimator=est)
        if model in ('ellipsoid', 'both', 'all'):
            # A LIST (0-3 ellipsoids fit a given D_t/D_r); callers iterate it. Isolated
            # in its own guard so a numeric edge in the (newer, heavier) ellipsoid
            # inverse can't discard the rod/sphere results that computed fine.
            try:
                out['ellipsoid'] = depol_engine.ellipsoid_dimensions_from_diffusion(
                    res.d_t_m2_s, res.d_r_rad2_s, temperature_K=t_k, viscosity_Pa_s=eta,
                    d_t_se=res.d_t_se, d_r_se=res.d_r_se, se_estimator=est)
            except Exception:                        # noqa: BLE001 — degrade, don't lose rod/sphere
                out['ellipsoid'] = []
        return out

    def cabannes_corrected_mw(self, sample_id: str, fraction: Optional[str],
                              cabannes_factor: float):
        """Isotropic-corrected Mw = f * Mw for this sample's current Mw (display only).

        Returns (mw_apparent, mw_corrected, mw_source) or None if the sample has no
        Mw yet. Phase 1 shows the correction; it does NOT write it back to the
        SampleResult (that, with provenance, is a later step). The factor f comes
        from a DepolarizationResult (= 1 - 4/3 rho_v).
        """
        s = self.workspace.samples[sample_id].result_for(fraction)
        mw = s.mw_g_per_mol
        if mw is None or not math.isfinite(mw):
            return None
        return (mw, mw * float(cabannes_factor), s.mw_source)

    # --- additional SLS analyses (all use the committed calibration) ---
    def run_rayleigh(self, sample_id: str, fraction: Optional[str] = None):
        """Excess Rayleigh ratio per concentration (one RayleighRatioResult each)."""
        rr = self._rayleigh_for_sample(sample_id, fraction)
        self.results[('rayleigh', sample_id, fraction)] = rr
        return rr

    def run_debye(self, sample_id: str, concentration_g_per_mL: float,
                  fraction: Optional[str] = None):
        """Single-concentration apparent Debye analysis (Kc/dR vs q^2)."""
        r = self._rayleigh_at_concentration(sample_id, concentration_g_per_mL, fraction)
        res = sls_engine.debye_analysis(r, estimator=self.settings.se_estimator)
        self.results[('debye', sample_id, concentration_g_per_mL, fraction)] = res
        return res

    def run_guinier(self, sample_id: str, concentration_g_per_mL: float,
                    qrg_max_valid: Optional[float] = None,
                    fraction: Optional[str] = None):
        """Single-concentration apparent Guinier analysis (ln dR vs q^2).
        The qRg validity limit seeds from settings when not given."""
        r = self._rayleigh_at_concentration(sample_id, concentration_g_per_mL, fraction)
        if qrg_max_valid is None:
            qrg_max_valid = self.settings.guinier_qrg_max
        res = sls_engine.guinier_analysis(r, qrg_max_valid=qrg_max_valid,
                                          estimator=self.settings.se_estimator)
        self.results[('guinier', sample_id, concentration_g_per_mL, fraction)] = res
        return res

    def run_single_angle(self, sample_id: str, concentration_g_per_mL: float,
                         angle_deg: float, fraction: Optional[str] = None):
        """Apparent Mw from one angle of one concentration."""
        r = self._rayleigh_at_concentration(sample_id, concentration_g_per_mL, fraction)
        res = sls_engine.single_angle_mw(r, angle_deg)
        self.results[('single_angle', sample_id, concentration_g_per_mL, angle_deg, fraction)] = res
        return res

    def run_single_angle_ladder(self, sample_id: str, angle_deg: float,
                                fraction: Optional[str] = None):
        """Apparent Mw and A2 from the concentration series at ONE angle -- the
        fixed-angle Debye plot, the analysis a single-angle ladder (a derived
        90-degree series, a single-angle instrument) can reach where Zimm/Berry
        refuse for want of a second angle. Everything it returns is apparent
        (invariant 7); the masked Rayleigh results feed it like every sibling."""
        rr = [r for r in self.masked_rayleigh(sample_id, fraction)
              if r.concentration_g_per_mL != 0]
        res = sls_engine.single_angle_ladder(rr, angle_deg,
                                             estimator=self.settings.se_estimator)
        self.results[('single_angle_ladder', sample_id, angle_deg, fraction)] = res
        return res

    def run_calibration_free_a2(self, sample_id: str, angle_deg: float,
                               mw_g_per_mol: Optional[float] = None,
                               fraction: Optional[str] = None):
        """2 A2 Mw (and A2 if a TRUSTWORTHY Mw is known) from intensity ratios.

        The PRODUCT 2*A2*Mw is genuinely calibration-free: it is slope/intercept of a
        ratio of excess intensities, so the unknown scale cancels. The ABSOLUTE
        A2 = (2*A2*Mw)/(2*Mw) is NOT -- it inherits the scale of whatever Mw divides it.

        So the sample's STORED Mw is passed through only when it is on a real absolute
        scale: hand-entered, or from a calibrated run (`mw_is_arbitrary_scale`, the one
        predicate). Otherwise None goes to the engine, which returns a2 = a2_se = None --
        the honest withhold: an absolute A2 built on an arbitrary-scale Mw is not an A2,
        and a SE on it is a SE on a quantity that has no scale (invariant 8). An explicitly
        supplied `mw_g_per_mol` is the caller's own value and is trusted as given.
        """
        rr = [r for r in self.masked_rayleigh(sample_id, fraction)
              if r.concentration_g_per_mL != 0]
        mw = mw_g_per_mol
        if mw is None:
            r = self.workspace.samples[sample_id].result_for(fraction)
            stored = r.effective_mw()
            # TWO questions, both about the value actually being passed (rule 12: size the
            # guard to the operation, and check both tails). `is_arbitrary_scale` answers
            # only "is the SCALE unknown?" and deliberately says False for a non-finite
            # value -- there is no number there to be on a scale -- so on its own it leaves
            # a gap: measured, a stored Mw of inf reached the engine and printed
            # A2 = 0.0 +/- 0.0, which reads as a clean theta-solvent result, and nan
            # printed nan +/- nan. Zero is already refused downstream, and refused here too
            # rather than relying on that.
            if (stored is not None and math.isfinite(stored) and stored > 0
                    and not r.mw_is_arbitrary_scale()):
                mw = stored
        res = sls_engine.calibration_free_a2(rr, angle_deg, mw_g_per_mol=mw,
                                             estimator=self.settings.se_estimator)
        self.results[('cal_free_a2', sample_id, angle_deg, fraction)] = res
        return res

    def _rayleigh_at_concentration(self, sample_id: str,
                                  concentration_g_per_mL: float,
                                  fraction: Optional[str] = None):
        # Use the masked results so Debye/Guinier honor hidden angles/points and
        # refuse a concentration that has been hidden entirely.
        for r in self.masked_rayleigh(sample_id, fraction):
            if r.concentration_g_per_mL == concentration_g_per_mL:
                return r
        raise ValueError(
            f"No analysable data at concentration {concentration_g_per_mL} g/mL "
            f"(is it hidden by the mask?).")

    # ----------------------------------------------------------------------
    # SLS data mask (per sample; a live analysis filter, persisted in sessions)
    # ----------------------------------------------------------------------
    def sls_mask(self, sample_id: str, fraction: Optional[str] = None) -> SLSMask:
        """The (sample, fraction) mask (created empty on first access). Masks are
        per-fraction so masking a point in one Mw fraction does not hide an
        overlapping concentration in another."""
        return self.sls_masks.setdefault((sample_id, fraction), SLSMask())

    def mask_angle(self, sample_id: str, angle_deg: float,
                   fraction: Optional[str] = None) -> None:
        self.sls_mask(sample_id, fraction).masked_angles.add(float(angle_deg))

    def unmask_angle(self, sample_id: str, angle_deg: float,
                     fraction: Optional[str] = None) -> None:
        m = self.sls_mask(sample_id, fraction)
        m.masked_angles = {a for a in m.masked_angles if not _close(a, angle_deg)}

    def mask_concentration(self, sample_id: str, c_g_per_mL: float,
                           fraction: Optional[str] = None) -> None:
        self.sls_mask(sample_id, fraction).masked_concentrations.add(float(c_g_per_mL))

    def unmask_concentration(self, sample_id: str, c_g_per_mL: float,
                             fraction: Optional[str] = None) -> None:
        m = self.sls_mask(sample_id, fraction)
        m.masked_concentrations = {x for x in m.masked_concentrations
                                   if not _close(x, c_g_per_mL)}

    def mask_point(self, sample_id: str, c_g_per_mL: float, angle_deg: float,
                   fraction: Optional[str] = None) -> None:
        self.sls_mask(sample_id, fraction).masked_points.add(
            (float(c_g_per_mL), float(angle_deg)))

    def unmask_point(self, sample_id: str, c_g_per_mL: float, angle_deg: float,
                     fraction: Optional[str] = None) -> None:
        m = self.sls_mask(sample_id, fraction)
        m.masked_points = {p for p in m.masked_points
                           if not (_close(p[0], c_g_per_mL) and _close(p[1], angle_deg))}

    def clear_sls_mask(self, sample_id: str, fraction: Optional[str] = None) -> None:
        self.sls_masks.pop((sample_id, fraction), None)

    # ---- mode-map family picks (the SLSMask pattern, value-keyed) ----
    def mode_picks_for(self, sample_id: str,
                       fraction: Optional[str] = None) -> ModeMapPicks:
        """The (sample, fraction) mode-map picks, created empty on first access."""
        return self.mode_picks.setdefault((sample_id, fraction), ModeMapPicks())

    def add_mode_family(self, sample_id: str, fraction: Optional[str] = None,
                        label: Optional[str] = None) -> ModeFamilyPicks:
        """Create a new (empty) mode family; auto-labels 'family N' when unnamed.
        Labels must be unique per map -- they key the durable Table-2 rows and
        the regroup merge."""
        store = self.mode_picks_for(sample_id, fraction)
        if label is None:
            n = 1
            while store.family(f'family {n}') is not None:
                n += 1
            label = f'family {n}'
        if store.family(label) is not None:
            raise ValueError(f'A mode family named {label!r} already exists '
                             f'on this map.')
        fam = ModeFamilyPicks(label=label)
        store.families.append(fam)
        return fam

    def remove_mode_family(self, sample_id: str, fraction: Optional[str],
                           label: str) -> None:
        """Delete a family, its cached fit, and its durable Table-2 row."""
        store = self.mode_picks_for(sample_id, fraction)
        store.families = [f for f in store.families if f.label != label]
        fits = self.results.get(('mode_family', sample_id, fraction))
        if isinstance(fits, dict):
            fits.pop(label, None)
        self.results.pop(('mode_amplitude', sample_id, fraction), None)
        self.workspace.remove_sample_rh_row(
            sample_id, 'mode_family', f'{label}|{fraction}')

    def rename_mode_family(self, sample_id: str, fraction: Optional[str],
                           old: str, new: str) -> None:
        """Rename a family; its durable row is re-keyed by the next fit run."""
        store = self.mode_picks_for(sample_id, fraction)
        fam = store.family(old)
        if fam is None:
            raise ValueError(f'No mode family named {old!r}.')
        if new == old:
            return
        if store.family(new) is not None:
            raise ValueError(f'A mode family named {new!r} already exists '
                             f'on this map.')
        fam.label = new
        fits = self.results.get(('mode_family', sample_id, fraction))
        if isinstance(fits, dict) and old in fits:
            fits[new] = fits.pop(old)
        self.results.pop(('mode_amplitude', sample_id, fraction), None)
        # RE-KEY the durable row, never just remove it: after a session load
        # the rows exist while the map cache does not, so "the next fit run
        # re-writes it" is false and the persisted Rh (and its candidate)
        # would silently vanish.
        row = self.workspace.remove_sample_rh_row(
            sample_id, 'mode_family', f'{old}|{fraction}')
        if row is not None:
            row.source_set = f'{new}|{fraction}'
            row.from_label = row.from_label.replace(
                f'Mode map: {old} ', f'Mode map: {new} ', 1)
            self.workspace.upsert_sample_rh_row(row)

    def set_mode_family_role(self, sample_id: str, fraction: Optional[str],
                             label: str, role: str) -> None:
        """Designate a family 'fast' or 'slow' for the amplitude report ('' clears).
        At most one family holds each role: assigning it clears the previous
        holder (a display designation, not data -- deterministic, visible in
        the panel)."""
        if role not in ('', 'fast', 'slow'):
            raise ValueError(f"role must be '', 'fast' or 'slow', got {role!r}.")
        store = self.mode_picks_for(sample_id, fraction)
        fam = store.family(label)
        if fam is None:
            raise ValueError(f'No mode family named {label!r}.')
        if role:
            for other in store.families:
                if other is not fam and other.role == role:
                    other.role = ''
        fam.role = role
        self.results.pop(('mode_amplitude', sample_id, fraction), None)

    def toggle_mode_pick(self, sample_id: str, fraction: Optional[str],
                         label: str, angle_deg: float,
                         gamma_s_inv: float,
                         polarization_state: Optional[str] = None) -> str:
        """Toggle a map point's membership in the named family, by VALUE.

        Returns what happened ('added' | 'removed' | 'moved') so the GUI can
        disclose a cross-family move in its status line. A family is bound to
        one polarization state at its first pick (the map facets by state, so
        a cross-state pick is unreachable from the GUI; this is the structural
        backstop -- a clear refusal, the partition rule).
        """
        store = self.mode_picks_for(sample_id, fraction)
        fam = store.family(label)
        if fam is None:
            raise ValueError(f'No mode family named {label!r}.')
        key = (float(angle_deg), float(gamma_s_inv))
        if fam.has_pick(*key):
            fam.picks = {p for p in fam.picks
                         if not (_close(key[0], p[0]) and _close(key[1], p[1]))}
            if fam.is_empty():
                fam.polarization_state = None      # unbound once empty
                fam.bound = False
            return 'removed'
        # Armed on the explicit `bound` flag, not on `is not None`: None is a
        # real member (the unpolarized class), and the None-bound family must
        # refuse a VV/VH pick exactly as a VV-bound one refuses VH.
        if fam.bound and polarization_state != fam.polarization_state:
            held = fam.polarization_state or 'unpolarized'
            offered = polarization_state or 'unpolarized'
            raise ValueError(
                f'Mode family {fam.label!r} holds {held} modes; a '
                f'{offered} mode cannot '
                f'join it -- polarizations are never pooled into one fit.')
        moved = False
        for other in store.families:
            if other is not fam and other.has_pick(*key):
                other.picks = {p for p in other.picks
                               if not (_close(key[0], p[0])
                                       and _close(key[1], p[1]))}
                if other.is_empty():
                    other.polarization_state = None
                    other.bound = False
                moved = True
        fam.picks.add(key)
        if not fam.bound:
            fam.polarization_state = polarization_state
            fam.bound = True
        # A pick mutation stales any cached amplitude report (finding 4's
        # class); the export recomputes, this keeps the cache honest too.
        self.results.pop(('mode_amplitude', sample_id, fraction), None)
        return 'moved' if moved else 'added'

    def clear_mode_picks(self, sample_id: str,
                         fraction: Optional[str] = None) -> None:
        """Drop every family on the (sample, fraction) map, incl. durable rows."""
        store = self.mode_picks.pop((sample_id, fraction), None)
        for fam in (store.families if store is not None else []):
            self.workspace.remove_sample_rh_row(
                sample_id, 'mode_family', f'{fam.label}|{fraction}')
        self.results.pop(('mode_family', sample_id, fraction), None)
        self.results.pop(('mode_amplitude', sample_id, fraction), None)

    def masked_rayleigh(self, sample_id: str, fraction: Optional[str] = None):
        """Per-concentration RayleighRatioResults with the (sample, fraction) mask
        applied: masked points -> NaN (dropped by the engine's finite filters); a
        wholly masked concentration is omitted. Used for analysis and the Zimm grid.
        The unmasked results (for graying hidden points on a plot) come from
        run_rayleigh."""
        results = self._rayleigh_for_sample(sample_id, fraction)
        mask = self.sls_masks.get((sample_id, fraction))
        if mask is None or mask.is_empty():
            return results
        out = []
        for r in results:
            c = r.concentration_g_per_mL
            if mask.is_concentration_masked(c):
                continue
            kc = np.asarray(r.kc_over_dR_mol_per_g, dtype=float).copy()
            dR = np.asarray(r.excess_rayleigh_cm_inv, dtype=float).copy()
            for i, ang in enumerate(r.angles_deg):
                if mask.is_masked(c, float(ang)):
                    kc[i] = np.nan
                    dR[i] = np.nan
            out.append(replace(r, kc_over_dR_mol_per_g=kc,
                               excess_rayleigh_cm_inv=dR))
        return out

    #: Above this ratio two c = 0 blanks are treated as materially different rather
    #: than as the same blank measured twice. A general light-scattering bar, not a
    #: system-specific one (invariant 4): a blank reproducible to 2 % is well inside
    #: what a stable cell and alignment deliver, so anything above it is worth the
    #: user's attention rather than the program's silence. A parameter of
    #: `solvent_blank_verdict`, overridable per call.
    BLANK_DISAGREEMENT_NOTABLE = 1.02

    def solvent_blank_verdict(self, sample_ids, *,
                              notable_ratio: Optional[float] = None) -> tuple:
        """(worst ratio, verdict) for the c = 0 blanks pooled onto `sample_ids`.

        The VERDICT that goes with `solvent_blank_disagreement`'s measurement -- whether
        these blanks are interchangeable -- lives here rather than in the shell, because
        it is a reliability statement about Mw, A2 and Rg and the bar it is judged
        against is a light-scattering criterion, not a widget's constant (invariants 4
        and 5). Returns ``(None, '')`` when no comparison was possible: no figure is not
        the same as agreement, and a caller must never narrate one it did not measure.
        The verdict is ``'differ'`` or ``'agree'``; the shell writes the sentences.
        """
        limit = (self.BLANK_DISAGREEMENT_NOTABLE if notable_ratio is None
                 else float(notable_ratio))
        ratios = [r for r in (self.solvent_blank_disagreement(sid)
                              for sid in sample_ids) if r is not None]
        worst = max(ratios, default=None)
        if worst is None:
            return None, ''
        return worst, ('differ' if worst >= limit else 'agree')

    def solvent_blank_disagreement(self, sample_id: str) -> Optional[float]:
        """How far the UNUSED c = 0 blanks sit from the one actually used, as a ratio.

        Returns the largest ``max(a/b, b/a)`` between the active solvent reference and
        any extra blank parked on this sample -- ``1.0`` when they agree, ``None`` when
        no comparison was possible (no extras, or no extra with usable intensities).
        ``None`` means "no figure", never "they agree": a caller must not narrate an
        agreement this did not measure.

        **Why this exists.** A sample holds ONE solvent reference, but Brookhaven's
        angular-scan layout can carry a blank per FILE, so N such files pooled into one
        sample leave N - 1 blanks parked and every series referenced against the
        first-loaded one. `compute_excess_rayleigh_ratio` subtracts that blank from all of
        them, so if the blanks genuinely differ the excess ratio is wrong by roughly this
        factor -- and with it Mw, A2, and the angular shape feeding Rg.

        Measured on real data (review finding): two acquisitions of one sample
        differed 2.2x at the first angle, which is a 1.6x error in the excess ratio. That
        is why the load-time notice reports this NUMBER and not merely the fact that
        extras exist: "kept but unused" reads as bookkeeping, "2.2x" does not.

        Since then each series is paired with its own blank (`Workspace.solvent_reference_for`:
        same source file, or the user's per-series choice in the Data tab), so this
        figure is informational: it says the pooled blanks differ, and therefore that
        the pairing matters -- a series that fell back to the sample slot because its
        file carried no blank is the one to check.

        Compared per angle wherever the two blanks share an angle grid, since per angle is
        what the subtraction actually does; falls back to mean intensity when they do not
        overlap at all, which still catches a level shift.
        """
        s = self.workspace.samples.get(sample_id)
        ref = self.workspace.solvent_reference(sample_id)
        if s is None or ref is None or not s.extra_solvent_reference_item_ids:
            return None

        def series(lm):
            """Finite, strictly-positive (angle, intensity) pairs, or (None, None).

            Positive because the comparison is a RATIO: a zero or negative intensity --
            a dark-count-subtracted blank can be either -- would divide by zero or flip
            the sign, and neither is a meaningful disagreement figure.
            """
            try:
                a = np.asarray(lm.raw['angles_deg'], dtype=float)
                i = np.asarray(lm.raw['intensities'], dtype=float)
            except (KeyError, TypeError, ValueError):
                return None, None
            if a.shape != i.shape:
                return None, None
            ok = np.isfinite(a) & np.isfinite(i) & (i > 0.0)
            return (a[ok], i[ok]) if ok.any() else (None, None)

        ref_a, ref_i = series(ref)
        if ref_i is None:
            return None

        # `worst` starts unset, NOT at 1.0. If no extra blank turns out to be comparable,
        # the honest answer is "no figure" -- returning 1.0 would report agreement we did
        # not measure, and the caller would tell the user the blanks match when the truth
        # is that one of them has no usable intensities.
        worst: Optional[float] = None
        for item_id in s.extra_solvent_reference_item_ids:
            lm = self.workspace.measurements.get(item_id)
            if lm is None:
                continue
            ex_a, ex_i = series(lm)
            if ex_i is None:
                continue
            # Match on the angle grid where possible, rounded to 0.1 deg: two acquisitions
            # of the same scan write the same NOMINAL angles, and an exact float compare
            # would drop every pair over a formatting difference.
            ref_r, ex_r = np.round(ref_a, 1), np.round(ex_a, 1)
            shared = np.intersect1d(ref_r, ex_r)
            if shared.size:
                r = np.array([ref_i[ref_r == v][0] for v in shared])
                e = np.array([ex_i[ex_r == v][0] for v in shared])
            else:
                r, e = np.array([ref_i.mean()]), np.array([ex_i.mean()])
            ratio = float(np.nanmax(np.maximum(r / e, e / r)))
            worst = ratio if worst is None else max(worst, ratio)
        return worst

    # --- accessors for the SLS GUI (axes available in a sample) ---
    def sample_fractions(self, sample_id: str,
                         kind: str = 'sls') -> List[Optional[str]]:
        """Distinct molecular-weight fraction labels among a sample's measurements
        (of the given kind). Returns [None] when nothing is labeled (the common
        single-fraction case). Sorted with None (unlabeled) first."""
        fracs = {lm.committed_params.get('mw_fraction')
                 for lm in self.workspace.sample_measurements(sample_id, kind)}
        if not fracs:
            return [None]
        return sorted(fracs, key=lambda f: (f is not None, f or ''))

    def sample_concentrations(self, sample_id: str,
                             include_zero: bool = False,
                             fraction: Optional[str] = None) -> List[float]:
        cs = [lm.committed_params.get('concentration_g_per_mL')
              for lm in self.workspace.sample_measurements(sample_id, 'sls')
              if lm.committed_params.get('mw_fraction') == fraction]
        cs = [c for c in cs if c is not None]
        if include_zero:
            ref = self.workspace.solvent_reference(sample_id)
            if ref is not None and ref.committed_params.get('concentration_g_per_mL') is not None:
                cs.append(ref.committed_params['concentration_g_per_mL'])
        return sorted(set(cs))

    def sample_angles(self, sample_id: str,
                      fraction: Optional[str] = None) -> List[float]:
        """Measured angles for the sample/fraction (a Zimm set shares one angle set)."""
        for lm in self.workspace.sample_measurements(sample_id, 'sls'):
            if lm.committed_params.get('mw_fraction') == fraction:
                return sorted(float(a) for a in lm.raw.get('angles_deg', []))
        ref = self.workspace.solvent_reference(sample_id)
        if ref is not None:
            return sorted(float(a) for a in ref.raw.get('angles_deg', []))
        return []

    def sls_pooling_hint(self, sample_id: str,
                         fraction: Optional[str] = None) -> Optional[str]:
        """A passive note when an SLS fit that consumes a fraction's concentration series
        (Zimm/Berry/Debye/Guinier/single-angle/calibration-free A2) may SILENTLY MIX more
        than one molar-mass series.

        Different molar masses of one polymer/solvent/temperature share a SampleKey, so
        without distinct **Mw-fraction** tags (Data tab) their concentration ladders live in
        one fraction. A ladder-regression fit (Zimm/Berry/calfree) then pools them into a
        single fit whose intercept (1/Mw) is a meaningless mix; a single-concentration fit
        (Debye/Guinier/single-angle) instead makes an ambiguous pick when two series share a
        concentration. Nothing in the pure fit can detect it (Mw is an output, not an
        input). This surfaces the two observable signatures of merged acquisitions:
        DUPLICATE concentration values in the fitted set (two overlapping ladders -- or
        harmless replicates of one Mw) and/or MORE THAN ONE c = 0 solvent blank. Returns
        None when neither is present.

        Honest, non-blocking, and two-sided (it names the benign replicate reading too) --
        it never gates the run; the user decides whether to tag fractions. Pure Python.

        Combination doctrine rule 2 (disclosure), core/data_models.py: the passive
        hint is the sanctioned mode for a hazard the program cannot verify.
        """
        s = self.workspace.samples.get(sample_id)
        if s is None:
            return None
        concs = [lm.committed_params.get('concentration_g_per_mL')
                 for lm in self.workspace.sample_measurements(sample_id, 'sls')
                 if lm.committed_params.get('mw_fraction') == fraction]
        concs = [c for c in concs if c is not None]
        rounded = [round_concentration_for_identity(c) for c in concs]
        signals = []
        if len(rounded) != len(set(rounded)):
            signals.append('some concentrations repeat')
        # The >1-blank signal points at pooling only for an UNTAGGED sample (one None
        # fraction). With distinct Mw-fraction tags an extra blank is instead the
        # load-order blank-SELECTION ambiguity (a separate, deferred issue), not pooling --
        # so it must not nag an already-tagged user to tag fractions (review finding).
        if s.extra_solvent_reference_item_ids and self.sample_fractions(sample_id) == [None]:
            signals.append('more than one solvent blank was loaded')
        if not signals:
            return None
        return ('Possible mixed molar-mass series — ' + ' and '.join(signals) + '. If this '
                'sample holds more than one molar mass, tag each series with an Mw fraction '
                'on the Data tab so they are fit separately; otherwise a fit can silently '
                'mix them. Ignore if these are replicates of a single Mw.')

    # ----------------------------------------------------------------------
    # Cross-sample: rho = Rg/Rh (Rh source selection)
    # ----------------------------------------------------------------------
    # Rg flows into the SampleResult from the SLS Zimm/Berry run (thermodynamic).
    # Rh has several possible DLS sources, so it is chosen from candidates: the
    # default is a labeled, deterministic pick (most-extrapolated first), and the
    # GUI can override it or accept a hand-entered value. See ResultCandidate /
    # select_default_candidate in analysis.utilities for the selection contract.

    def dls_rh_candidates(self, sample_id: str,
                          fraction: Optional[str] = None) -> List[ResultCandidate]:
        """Available DLS-derived Rh values for a sample/fraction, each with provenance
        (memoized by `_dls_cached` under `dls_run_signature` — see
        `_dls_rh_candidates_uncached` for the computation)."""
        # Return a shallow copy so a caller sorting/filtering in place can't poison the
        # memo — the load-bearing "cache never changes a result" rule (mirrors
        # sls_rg_candidates).
        return list(self._dls_cached(
            sample_id, fraction,
            lambda: self._dls_rh_candidates_uncached(sample_id, fraction)))

    def _dls_rh_candidates_uncached(self, sample_id: str,
                                    fraction: Optional[str] = None) -> List[ResultCandidate]:
        """Available DLS-derived Rh values for a sample/fraction, each with provenance.

        Three kinds, in increasing physical preference (higher tier = preferred):
          tier 1  per-measurement cumulant Rh -- apparent (single q, single c);
          tier 2  Gamma vs q^2 -> q->0        -- apparent (q-extrapolated), needs
                                                 >= 2 angles at one concentration;
          tier 3  D vs c -> c->0              -- thermodynamic, needs >= 2 distinct
                                                 concentrations.
        A fit that fails or is non-finite is simply skipped (it does not appear as
        a candidate).

        A FLAGGED value is offered but never the passive default (`auto_selectable`
        off, the reason on the label): every tier reads the fit's own `rh_reliable`,
        so an undecayed window, an out-of-range size, a PDI past the cumulant's
        validity or a flagged per-angle fit inside a pooled line stops at the picker
        rather than travelling into rho = Rg/Rh unflagged. Same treatment a degenerate
        form-factor solution and a CONTIN sweep-edge peak already get.

        Tiers 2 and 3 offer THE USER'S OWN RUN when one is stored and fresh
        (`_stored_dls_run`) -- the fit over the points they ticked on the DLS tab,
        under their Settings estimator, carrying its flags -- so the promoted number
        is the one Summary Table 2 shows. Only without a fresh run do they re-derive
        live, and then over the same fit-eligible points the run path uses
        (`_dls_run_points` + `FIT_ELIGIBLE_QUALITIES`) with the same estimator, so one
        unfittable angle cannot poison the whole tier away. The form-factor candidates
        set this read-the-stored-run-gated-on-freshness shape.

        Distribution-model (CONTIN/NNLS) peaks are also offered, but only from
        results the user has ALREADY computed in the DLS tab (a CONTIN L-curve is
        too expensive to run on every refresh). Each resolved population is one
        candidate; the dominant peak competes as a default (tier 1, like the
        cumulant), secondary peaks are selectable but never the default (tier 0) so
        a minor population is never silently chosen -- this is the peak picker for a
        multi-population sample.

        Replicate-averaged Rh (parametric replicate averaging, which carries the
        only honest single-sample dynamic SE -- invariant 8) is offered from the
        durable `sample_rh_rows` store: one candidate per averaged (method, fraction),
        apparent, and PREFERRED over a single cumulant (a finite positive quality beats
        the single-fit `-rms` within tier 1) so an auto-refresh's default pick never
        clobbers a value the user deliberately averaged. The Gamma-q^2 / D-c store ROWS
        are not read here: tiers 2/3 read the stored RESULT behind them (which carries
        the flags and R^2 the row does not) or re-derive live, never both, so nothing is
        double-counted.
        """
        # Combination doctrine rule 1 (partition), core/data_models.py, enforced
        # structurally: a passive path never writes a durable value rule 1 would
        # refuse, so partition tags (fraction, VH) filter the input set up front.
        # VH is excluded from EVERY tier because Gamma_VH = q^2 D_t + 6 D_r yields no
        # hydrodynamic radius on its own -- the physics behind the interactive fits'
        # refusal (_mixed_polarization_message). Exclusion rather than refusal because
        # this path has no user to tell and feeds auto_select_rh, so offering nothing
        # is the only honest answer. VV, VU and untagged carry Gamma = q^2 D_t; pooled.
        lms = [lm for lm in self.workspace.sample_measurements(sample_id, 'dls')
               if lm.committed_params.get('mw_fraction') == fraction
               and _polarization_state(
                   lm.committed_params.get('analyzer_geometry')) != 'VH']
        cands: List[ResultCandidate] = []

        # tier 1 -- per-measurement cumulant Rh (apparent, single point). THE USER'S OWN
        # Correlogram fit while it is fresh (`_stored_cumulant_fit`), labelled as theirs
        # and promoted as it ran -- its own order box and its own region -- so the Rh the
        # panel shows is the Rh that travels. Only without one does the builder
        # re-derive, from the one seed home at the Settings order and each run's own
        # defaults (no region: a passive refresh has no box to tick).
        for lm in lms:
            stored = self._stored_cumulant_fit(lm.item_id)
            if stored is not None:
                res, source_id, how = (stored, _stored_fit_source_id(lm.item_id),
                                       'as fitted in the DLS tab, ')
            else:
                try:
                    res = dls_engine.fit_cumulants(
                        lm.build(), **self._seed_cumulant_kwargs({}))
                except Exception:  # noqa: BLE001 — candidate isolation
                    continue
                source_id, how = lm.item_id, ''
            if not np.isfinite(res.rh_nm):
                continue
            p = lm.committed_params
            ang = p.get('angle_deg')
            conc = p.get('concentration_g_per_mL')
            # A flagged Rh carries its flag onto the label (`_flag_suffix`) and still
            # promotes -- a flag warns, it does not withhold (owner ruling, CLAUDE.md
            # invariant 4); only an UNUSABLE number, outside the technique's size range,
            # is offered but never silently chosen (`_dls_rh_promotable`). Before, the
            # flag stopped at the DLS panel: the same number was auto-selected into
            # SampleResult.rh_nm with nothing anywhere saying it was flagged.
            cands.append(ResultCandidate(
                value=float(res.rh_nm),
                label=(f"Cumulant Rh at {_fmt_deg(ang)}, {_fmt_conc(conc)} "
                       f"({how}apparent, single angle){_flag_suffix(res)}"),
                kind='dls_cumulant', is_apparent=True, tier=1,
                quality=(-float(res.rms_error)
                         if np.isfinite(res.rms_error) else None),
                quality_kind='neg_rms', source_id=source_id,
                auto_selectable=_dls_rh_promotable(res)))

        # replicate-averaged Rh from the durable Summary store (apparent, but the only
        # single-sample dynamic Rh with an honest SE). Preferred over a single cumulant
        # (quality 1.0 > the single-fit -rms in tier 1) so an auto-refresh never
        # overwrites a value the user deliberately averaged. Only 'replicate_avg' rows
        # are read here; the gamma_q2/conc_extrap rows are re-derived live below.
        for row in self.workspace.sample_rh_rows.values():
            if (row.sample_id != sample_id or row.fraction != fraction
                    or row.source_kind != 'replicate_avg'):
                continue
            if row.rh_nm is None or not np.isfinite(row.rh_nm):
                continue
            rep_kind, rep_source_id = replicate_rh_key(row.source_set)
            cands.append(ResultCandidate(
                value=float(row.rh_nm),
                label=f'{row.from_label} — apparent (± SD/√N)',
                kind=rep_kind, is_apparent=bool(row.is_apparent), tier=1,
                quality=1.0, quality_kind='replicate_average',
                source_id=rep_source_id,
                value_se=unc.se_or_none(row.rh_se)))

        # Mode-family Rh from the durable store: apparent AND mode-resolved --
        # one relaxation mode's size, not the sample's -- so it is OFFERED,
        # never auto-selected (auto_selectable=False: flags travel with the
        # value; the pick is the user's). Read from the store like the
        # replicate rows (re-deriving needs picks + a cached map a passive
        # refresh may not have). VH families never wrote a row, so the VH
        # exclusion above holds by construction. The rho-coupling hook: a
        # per-mode Rh reaches the tier flagged, so rho_mode is a picker later.
        for row in self.workspace.sample_rh_rows.values():
            if (row.sample_id != sample_id or row.fraction != fraction
                    or row.source_kind != 'mode_family'):
                continue
            if row.rh_nm is None or not np.isfinite(row.rh_nm):
                continue
            cands.append(ResultCandidate(
                value=float(row.rh_nm),
                label=f'{row.from_label} — apparent, mode-resolved',
                kind='dls_mode_family', is_apparent=True, tier=1,
                source_id=row.source_set,
                value_se=unc.se_or_none(row.rh_se),
                value_se_estimator=row.se_estimator,
                auto_selectable=False))

        # distribution-model peaks (CONTIN/NNLS) from results already computed in
        # the DLS tab -- one candidate per resolved population (the peak picker).
        sample_dls_ids = {lm.item_id for lm in lms}
        for key, res in self.results.items():
            if not (isinstance(key, tuple) and len(key) == 3
                    and key[0] == 'distribution'):
                continue
            item_id, method = key[1], key[2]
            if item_id not in sample_dls_ids:
                continue
            dist = getattr(res, 'distribution', res)   # ContinResult wraps it; NNLS is bare
            try:
                # A failed (seed-shaped) fit yields no peaks -- the scanner refuses it,
                # so it can never become a rho source here.
                peaks = dls_engine.find_distribution_peaks(dist)
            except Exception:  # noqa: BLE001 — candidate isolation
                continue
            # A CONTIN fit whose alpha selection carries a reliability note -- pinned to a
            # sweep edge, no corner, or an F-test precondition that failed -- is OFFERED
            # but never silently chosen (same treatment as the out-of-range cumulant Rh
            # above): the flag would otherwise stop at the DLS panel while the peak went
            # on into rho = Rg/Rh. The roll-up is read, not the two edge flags, because a
            # failed precondition is a flagged result exactly as an edge pick is.
            flag_note = getattr(res, 'reliability_note', '')   # '' for NNLS/lognormal
            suffix = f' — {flag_note}' if flag_note else ''
            p = self.workspace.measurements[item_id].committed_params
            ang = p.get('angle_deg')
            conc = p.get('concentration_g_per_mL')
            rms = float(dist.rms_error)
            for j, pk in enumerate(peaks):
                if not np.isfinite(pk.rh_nm):
                    continue
                # The technique's size range (the ISO ceiling, the artifact floor) is a
                # limit of DLS, not of the cumulant method: a peak past it is OFFERED but
                # never silently chosen, as the flagged cumulant Rh beside it is not.
                size_ok, size_note = _dls_size_range_check(
                    float(pk.rh_nm), dls_engine.CUMULANT_MAX_RH_NM,
                    dls_engine.CUMULANT_MIN_RH_NM)
                peak_suffix = suffix if size_ok else f' — unreliable: {size_note}'
                cands.append(ResultCandidate(
                    value=float(pk.rh_nm),
                    label=(f"{method.upper()} peak {j + 1}: Rh = {pk.rh_nm:.3g} nm "
                           f"({pk.weight_fraction * 100:.0f}% of intensity) at "
                           f"{_fmt_deg(ang)}, {_fmt_conc(conc)} (apparent){peak_suffix}"),
                    kind='dls_distribution_peak', is_apparent=True,
                    tier=(1 if pk.is_dominant else 0),
                    quality=(-rms if np.isfinite(rms) else None),
                    quality_kind='neg_rms',
                    source_id=f'{method}@{item_id}#{j}',
                    auto_selectable=bool(not flag_note and size_ok)))

        # The pooled tiers see only the points the RUN path would fit: the same per-point
        # quality ladder, the same eligibility tuple. Fitting the raw set let one dead
        # channel poison the whole tier to NaN, and the default Rh then fell silently
        # from the q->0 extrapolation to a single-angle apparent cumulant.
        usable = {pt['item_id'] for pt in self._dls_run_points(sample_id, fraction)
                  if pt['quality'] in FIT_ELIGIBLE_QUALITIES}
        fitted_lms = [lm for lm in lms if lm.item_id in usable]
        meas = [lm.build() for lm in fitted_lms]

        # tier 2 -- Gamma vs q^2 -> q->0 (apparent). The user's stored run first.
        gq = self._stored_dls_run('gamma_q2', sample_id, fraction)
        if gq is not None and np.isfinite(gq.rh_nm):
            # The run is the user's: offered as run, including a set that pooled several
            # concentrations into one q^2 line (the fit does not police the ticks). But
            # the PASSIVE default keeps the partition the re-derivation below enforces --
            # a concentration-mixed q->0 Rh is offered, labelled, never silently chosen.
            run_concs = {
                (round_concentration_for_identity(c) if c is not None else None)
                for c in self._stored_run_params(
                    'gamma_q2', sample_id, fraction, 'concentration_g_per_mL')}
            mixed = len(run_concs) > 1
            n_fit = int(np.isfinite(np.asarray(gq.gamma_s_inv, dtype=float)).sum())
            r2 = float(gq.r_squared)
            cands.append(ResultCandidate(
                value=float(gq.rh_nm),
                label=(f"Γ vs q² → q→0 over {n_fit} angles"
                       + (f" at {len(run_concs)} concentrations (mixed — offered, "
                          f"not defaulted)" if mixed else '')
                       + f", as run in the DLS tab "
                       f"(apparent, R² = {_fmt_r2(r2)}){_flag_suffix(gq)}"),
                kind='dls_gamma_q2', is_apparent=True, tier=2,
                quality=(r2 if np.isfinite(r2) else None), quality_kind='r_squared',
                source_id='gamma_q2@run',
                value_se=unc.se_or_none(getattr(gq, 'rh_se', None)),
                value_se_estimator=getattr(gq, 'se_estimator', None),
                auto_selectable=_dls_rh_promotable(gq) and not mixed))
        else:
            # Re-derived live, PER CONCENTRATION: mixing concentrations would fold the
            # concentration dependence into the q^2 fit. Needs >= 2 angles at that c.
            # Concentration is OPTIONAL for DLS: measurements with none group under
            # the `None` key -- a multi-angle set at an unknown concentration is still
            # a valid q->0 Rh (analyze_gamma_q2 is q-based, never reads concentration;
            # the label reads 'c = ?'). The sort key keeps that group last without
            # comparing None to a float.
            by_conc: Dict[Optional[float], List[Any]] = {}
            for m in meas:
                c = m.concentration_g_per_mL
                by_conc.setdefault(
                    round_concentration_for_identity(c) if c is not None else None,
                    []).append(m)
            for c, ms in sorted(by_conc.items(), key=lambda kv: (kv[0] is None, kv[0])):
                if len({round_angle_for_identity(m.angle_deg) for m in ms}) < 2:
                    continue
                try:
                    gq = dls_engine.analyze_gamma_q2(
                        ms, estimator=self.settings.se_estimator,
                        **self._pooled_cumulant_kwargs(self._seed_cumulant_kwargs({})))
                except Exception:  # noqa: BLE001 — candidate isolation
                    continue
                if not np.isfinite(gq.rh_nm):
                    continue
                r2 = float(gq.r_squared)
                cands.append(ResultCandidate(
                    value=float(gq.rh_nm),
                    label=(f"Γ vs q² → q→0 at {_fmt_conc(c)} over "
                           f"{int(np.isfinite(np.asarray(gq.gamma_s_inv, dtype=float)).sum())} angles "
                           f"(apparent, R² = {_fmt_r2(r2)}){_flag_suffix(gq)}"),
                    kind='dls_gamma_q2', is_apparent=True, tier=2,
                    quality=(r2 if np.isfinite(r2) else None), quality_kind='r_squared',
                    source_id=f'gamma_q2@{c}',
                    value_se=unc.se_or_none(getattr(gq, 'rh_se', None)),
                    value_se_estimator=getattr(gq, 'se_estimator', None),
                    auto_selectable=_dls_rh_promotable(gq)))

        # tier 3 -- D vs c -> c->0 (thermodynamic). The user's stored run first.
        ce = self._stored_dls_run('conc_extrap', sample_id, fraction)
        if ce is not None and np.isfinite(ce.rh0_nm):
            # Same rule as tier 2: a run pooling several angles into one D-vs-c line is
            # offered as run but never the passive default (q-dependence would fold in).
            run_angles = {
                round_angle_for_identity(a)
                for a in self._stored_run_params(
                    'conc_extrap', sample_id, fraction, 'angle_deg') if a is not None}
            mixed = len(run_angles) > 1
            r2 = float(ce.r_squared)
            cands.append(ResultCandidate(
                value=float(ce.rh0_nm),
                label=(f"D vs c → c→0 over {ce.n_concentrations} concentrations"
                       + (f" at {len(run_angles)} angles (mixed — offered, not "
                          f"defaulted)" if mixed else '')
                       + f", as run in the DLS tab "
                       f"(thermodynamic, R² = {_fmt_r2(r2)}){_flag_suffix(ce)}"),
                kind='dls_conc_extrap', is_apparent=False, tier=3,
                quality=(r2 if np.isfinite(r2) else None), quality_kind='r_squared',
                source_id='conc_extrap@run',
                value_se=unc.se_or_none(getattr(ce, 'rh0_se', None)),
                value_se_estimator=getattr(ce, 'se_estimator', None),
                auto_selectable=_dls_rh_promotable(ce, 'rh0_nm') and not mixed))
        else:
            # Re-derived live, PER ANGLE, so any q-dependence is not folded into the
            # concentration fit. Needs >= 2 distinct concentrations at that angle.
            by_angle: Dict[float, List[Any]] = {}
            for m in meas:
                by_angle.setdefault(round_angle_for_identity(m.angle_deg), []).append(m)
            for a, ms in sorted(by_angle.items()):
                # Concentration is the x-axis here, so a measurement without one can't
                # be a point on the series -- drop it before counting/fitting (unlike
                # tier 2).
                ms_c = [m for m in ms if m.concentration_g_per_mL is not None]
                if len({round_concentration_for_identity(m.concentration_g_per_mL)
                        for m in ms_c}) < 2:
                    continue
                try:
                    ce = dls_engine.extrapolate_diffusion_vs_concentration(
                        ms_c, estimator=self.settings.se_estimator,
                        **self._pooled_cumulant_kwargs(self._seed_cumulant_kwargs({})))
                except Exception:  # noqa: BLE001 — candidate isolation
                    continue
                if not np.isfinite(ce.rh0_nm):
                    continue
                r2 = float(ce.r_squared)
                cands.append(ResultCandidate(
                    value=float(ce.rh0_nm),
                    label=(f"D vs c → c→0 at {_fmt_deg(a)} over "
                           f"{ce.n_concentrations} concentrations "
                           f"(thermodynamic, R² = {_fmt_r2(r2)}){_flag_suffix(ce)}"),
                    kind='dls_conc_extrap', is_apparent=False, tier=3,
                    quality=(r2 if np.isfinite(r2) else None), quality_kind='r_squared',
                    source_id=f'conc_extrap@{a}',
                    value_se=unc.se_or_none(getattr(ce, 'rh0_se', None)),
                    value_se_estimator=getattr(ce, 'se_estimator', None),
                    auto_selectable=_dls_rh_promotable(ce, 'rh0_nm')))

        return cands

    def set_sample_rh(self, sample_id: str, candidate: ResultCandidate,
                      fraction: Optional[str] = None, *, explicit: bool = False) -> None:
        """Store a chosen DLS-derived Rh candidate as the sample/fraction's Rh.

        An explicit selection (the GUI picked this candidate), so it overrides any
        previously stored value, including a hand-entered one. Records the
        provenance label and apparent/thermodynamic status for rho. `explicit=True`
        marks the source 'picked' (a deliberate user choice, which a later passive
        auto-select refresh must NOT clobber back to the default); the default fill
        (`auto_select_*`) leaves it 'computed'.
        """
        r = self.workspace.samples[sample_id].result_for(fraction)
        r.rh_nm = float(candidate.value)
        r.rh_source = 'picked' if explicit else 'computed'
        r.rh_apparent = bool(candidate.is_apparent)
        r.rh_label = candidate.label
        r.rh_se = candidate.value_se
        r.rh_se_estimator = candidate.value_se_estimator  # copied provenance (invariant-8 Clause A)
        set_result_key(r, 'rh', candidate.kind, candidate.source_id)

    def auto_select_rh(self, sample_id: str,
                       fraction: Optional[str] = None) -> Optional[ResultCandidate]:
        """Pick and store the default Rh for a sample (most-extrapolated first).

        Never overwrites a hand-entered Rh ('user') or an explicit pick ('picked');
        returns the stored-or-chosen candidate, or None if no DLS Rh is available. The
        default is labeled -- the GUI shows where the value came from; never silent.
        """
        r = self.workspace.samples[sample_id].result_for(fraction)
        cands = self.dls_rh_candidates(sample_id, fraction)
        if r.rh_source in DELIBERATE_TIERS and r.rh_nm is not None:
            return None                      # respect a manual override or explicit pick
        chosen = select_default_candidate(cands)
        if chosen is not None:
            self.set_sample_rh(sample_id, chosen, fraction)
        return chosen

    def set_manual_rh(self, sample_id: str, rh_nm: float,
                      is_apparent: bool = False,
                      fraction: Optional[str] = None) -> None:
        """Record a hand-entered Rh (e.g. from external characterization).

        Marked provenance 'user' so a re-selection does not silently replace it.
        `is_apparent` lets the user say whether their value is an infinite-dilution
        (thermodynamic) Rh or a single-condition (apparent) one; default
        thermodynamic, since a hand-entered Rh is usually the best available value.
        """
        # isfinite too: this is a hand-typed field and float(text) accepts 'inf'
        # (2026-07-31 spec-01 re-review).
        if not (math.isfinite(rh_nm) and rh_nm > 0):
            raise ValueError(f"rh_nm must be a positive finite number, got {rh_nm!r}.")
        r = self.workspace.samples[sample_id].result_for(fraction)
        r.rh_nm = float(rh_nm)
        r.rh_source = 'user'
        r.rh_apparent = bool(is_apparent)
        r.rh_label = 'user-entered'
        r.rh_se = None                    # a hand-entered value carries no statistical SE
        r.rh_se_estimator = None          # ... and no estimator provenance; clear any stale tag
        # No record produced it, so there is no source_id to name -- 'user' is the whole key.
        set_result_key(r, 'rh', 'user', None)

    def _form_factor_run_state(self, sample_id: str,
                               fraction: Optional[str] = None) -> tuple:
        """The stored form-factor fits for this (sample, fraction) as sorted
        ``(model, stamped-signature)`` pairs — folded into the candidate cache key
        (`_sls_cached`) so a (re)stored fit invalidates the memo. Two cases the live
        `sls_run_signature` alone would miss: (1) first-running a NEW model (inputs unchanged,
        so the live signature is unchanged) adds a pair -> key changes -> MISS; (2) re-running a
        model to REFRESH a stale fit re-stamps its signature old->new (the live signature is
        also unchanged since the intervening mask edit) -> its pair changes -> MISS -> the
        now-fresh candidate reappears. Pairs the FRESHNESS the builder gates on into the key."""
        # self.results holds MIXED-arity keys (e.g. ('zimm', sid, frac) is a 3-tuple), so match
        # the 4-tuple ('form_factor', sid, model, frac) shape by length. Iterate a SNAPSHOT
        # (`list(...)`): run_form_factor writes self.results on the worker thread while this
        # reads on the main thread, so a live iteration could raise "dictionary changed size".
        pairs = [(key[2], self._form_factor_run_sig.get(key))
                 for key in list(self.results)
                 if len(key) == 4 and key[0] == 'form_factor'
                 and key[1] == sample_id and key[3] == fraction]
        return tuple(sorted(pairs, key=lambda p: p[0]))   # sort by model (never compare sigs)

    def _sls_cached(self, kind: str, sample_id: str, fraction: Optional[str], compute):
        """Memoize an SLS candidate/A2 computation for (sample, fraction) under the full
        `sls_run_signature` plus the form-factor run-state. The Cross-Sample refresh calls the
        candidate + A2 functions several times per fraction and each re-runs the same
        Zimm/Debye/Guinier fits (form-factor fits are READ, not re-run — see
        `_form_factor_solutions`); this returns the first result until an input changes. Keyed on
        the run signature, so a mask/commit/calibration/estimator/Mw change simply MISSES — no
        explicit invalidation needed. The form-factor run-state (models + their stamped
        signatures) is in the key too, so (re)running a form-factor fit on the SLS tab — which
        does not move the live signature — also MISSES and surfaces/refreshes/expires its pooled
        candidate. `sls_run_signature` itself is left untouched (it drives the SLS tab's own
        staleness hint — adding form-factor state there would spuriously flag a sibling model's
        cached SLS payload stale). The estimator is IN the key, so this cache is deliberately NOT
        swept by `clear_se_dependent_results` (redundant; keying on the signature is the single
        invalidation mechanism). The recorded-mode-set state is in the key for the same
        reason as the form-factor run-state and via the same reasoning as its exclusion
        from `sls_run_signature`: the Rg candidates READ the recorded sets (the per-mode
        amplitude-Guinier branch), while no ordinary SLS fit does -- keying the live
        signature on them would spuriously flag every cached Zimm/Debye payload stale the
        moment a set is recorded. The single-angle ladder ROSTER is in the key for the third
        time on the same reasoning (running a ladder moves no committed input), carrying only
        the angles because the ladder's value is re-derived rather than read. Bounded: only
        the current signature is kept per (kind, sample, fraction)."""
        sig = self.sls_run_signature(sample_id, fraction)
        key = (kind, sample_id, fraction, sig,
               self._form_factor_run_state(sample_id, fraction),
               self._mode_set_state(sample_id, fraction),
               # The ladder ROSTER (`_ladder_angles`), for the form-factor run-state's reason:
               # running a ladder on the SLS tab moves no committed input, so the live
               # signature is unchanged and a memoized list would keep hiding the new
               # candidate. Only the angles, not a freshness stamp -- the ladder's value is
               # re-derived, so there is nothing stored here to go stale.
               tuple(self._ladder_angles(sample_id, fraction)))
        cache = self._sls_candidate_cache
        if key in cache:
            return cache[key]
        val = compute()
        for stale in [k for k in cache if k[:3] == (kind, sample_id, fraction)]:
            del cache[stale]                 # evict the same slot under an older signature
        cache[key] = val
        return val

    def _dls_cached(self, sample_id: str, fraction: Optional[str], compute):
        """Memoize the DLS Rh-candidate computation for (sample, fraction) under the
        full `dls_run_signature` — the DLS analog of `_sls_cached`. The Cross-Sample
        refresh calls `dls_rh_candidates` several times per fraction (auto-select +
        the source panel) and each re-runs a cumulant fit per measurement, a Γ-q² per
        concentration, and a D-vs-c per angle; this returns the first result until an
        input changes. Keyed on the run signature, so a commit / param edit / seed
        change / new distribution fit / replicate-row change simply MISSES — no
        explicit invalidation needed (measurement removal purges the slot in
        `remove_measurements`, only to keep the cache bounded). Bounded: only the
        current signature is kept per (sample, fraction)."""
        sig = self.dls_run_signature(sample_id, fraction)
        key = (sample_id, fraction, sig)
        cache = self._dls_candidate_cache
        if key in cache:
            return cache[key]
        val = compute()
        for stale in [k for k in cache if k[:2] == (sample_id, fraction)]:
            del cache[stale]                 # evict the same slot under an older signature
        cache[key] = val
        return val

    @staticmethod
    def _offers_cross_sample_candidates(rayleigh_results) -> bool:
        """May a sample's SLS fits be offered to the Cross-Sample tab? (owner-ruled)

        ONE predicate, keyed on the record and applied where the candidates are
        GATHERED -- not a check repeated at each of the twenty `ResultCandidate(` mint
        sites, where the twenty-first would be the one that forgets it.

        A fit computed without a measured solvent blank is not offered. It stays on the
        SLS tab, flagged and exportable, and a user who wants its value in a scaling
        plot enters it by hand -- the existing route for a number the program does not
        vouch for. The reason is the feature's own purpose: it exists for a quick look
        BEFORE a real run, and `plot_scaling` carries no scale disclosure at all (flags
        are GUI-owned overlays, never drawn into a figure), so a badge on one input is
        easy to lose by the time it is a point on a scaling line. A sample's
        measured-blank fits are offered exactly as before.
        """
        return all(bool(getattr(r, 'blank_measured', True)) for r in rayleigh_results)

    def sls_rg_candidates(self, sample_id: str,
                          fraction: Optional[str] = None) -> List[ResultCandidate]:
        """Available SLS-derived Rg values for a sample, each with provenance (memoized by
        `_sls_cached` under the run signature — see `_sls_rg_candidates_uncached`)."""
        # Return a shallow copy so a caller can't mutate (sort/filter in place) the cached
        # list and poison the memo — the load-bearing "cache never changes a result" rule.
        return list(self._sls_cached(
            'rg', sample_id, fraction,
            lambda: self._sls_rg_candidates_uncached(sample_id, fraction)))

    def _sls_rg_candidates_uncached(self, sample_id: str,
                                    fraction: Optional[str] = None) -> List[ResultCandidate]:
        """Available SLS-derived Rg values for a sample, each with provenance.

        Two kinds (higher tier = preferred):
          tier 3  Zimm / Berry double extrapolation -- thermodynamic, needs >= 2
                  concentrations;
          tier 2  Debye or Guinier at one concentration -- apparent, needs >= 2
                  angles at that concentration.
        Rg is scale-independent, so these are valid even when the run is
        uncalibrated. A fit that fails or is non-finite is simply skipped. The
        sample must have a solvent reference (c = 0) for the Rayleigh ratios; with
        none, no SLS candidate is produced.
        """
        cands: List[ResultCandidate] = []
        try:
            rr = self.masked_rayleigh(sample_id, fraction)
        except Exception:  # noqa: BLE001 — no reference, no candidates
            return cands
        if not self._offers_cross_sample_candidates(rr):
            return cands
        nonzero = [r for r in rr if r.concentration_g_per_mL != 0]
        n_conc = len({round_concentration_for_identity(r.concentration_g_per_mL)
                      for r in nonzero})

        # tier 3 -- thermodynamic Zimm / Berry (need >= 2 concentrations)
        if n_conc >= 2:
            for method in ('zimm', 'berry'):
                try:
                    res = sls_engine.zimm_analysis(rr, method=method,
                                                   estimator=self.settings.se_estimator)
                except Exception:  # noqa: BLE001 — candidate isolation
                    continue
                if not np.isfinite(res.rg_nm):
                    continue
                r2 = float(res.r_squared)
                extrap_kind, extrap_source_id = extrapolation_result_key(method)
                # The size's own verdict travels with it (`rg_reliable`: a degenerate
                # design condemns the Rg it just condemned Mw/A2 for) -- offered,
                # labelled, never the passive default. Same shape as the DLS tiers.
                cands.append(ResultCandidate(
                    value=float(res.rg_nm),
                    label=(f"{method.capitalize()} extrapolation, "
                           f"{res.n_concentrations} concentrations "
                           f"(thermodynamic, R² = {_fmt_r2(r2)})"
                           f"{_flag_suffix(res, 'rg_reliable')}"),
                    kind=extrap_kind, is_apparent=False, tier=3,
                    quality=(r2 if np.isfinite(r2) else None),
                    quality_kind='r_squared', source_id=extrap_source_id,
                    value_se=unc.se_or_none(getattr(res, 'rg_se', None)),
                    value_se_estimator=getattr(res, 'se_estimator', None),
                    auto_selectable=bool(getattr(res, 'rg_reliable', True))))

        # tier 2 -- apparent Debye / Guinier, one per SLS MEASUREMENT (a repeat run at
        # one concentration yields two), so the identity is keyed on position -- see
        # `_per_series_source_id`.
        for i_series, r in enumerate(nonzero):
            c = r.concentration_g_per_mL
            try:
                d = sls_engine.debye_analysis(r, estimator=self.settings.se_estimator)
                if np.isfinite(d.rg_apparent_nm):
                    r2 = float(d.r_squared)
                    cands.append(ResultCandidate(
                        value=float(d.rg_apparent_nm),
                        label=(f"Debye at {_fmt_conc(c)} over {d.n_angles} angles "
                               f"(apparent, R² = {_fmt_r2(r2)})"
                               f"{_flag_suffix(d, 'rg_reliable')}"),
                        kind='sls_debye', is_apparent=True, tier=2,
                        quality=(r2 if np.isfinite(r2) else None),
                        quality_kind='r_squared',
                        source_id=_per_series_source_id('debye', i_series, c),
                        value_se=unc.se_or_none(getattr(d, 'rg_apparent_se', None)),
                        value_se_estimator=getattr(d, 'se_estimator', None),
                        auto_selectable=bool(getattr(d, 'rg_reliable', True))))
            except Exception:  # noqa: BLE001 — candidate isolation
                pass
            try:
                # The Settings qRg limit is seeded here as on every other Guinier call
                # site (run_guinier, run_mode_static). The size's verdict `rg_reliable`
                # folds the regime verdict in (an out-of-regime fit fails a precondition
                # of the method) beside the design's conditioning, so one flag and one
                # wording travel with the value: offered, never the silent default.
                g = sls_engine.guinier_analysis(
                    r, estimator=self.settings.se_estimator,
                    qrg_max_valid=self.settings.guinier_qrg_max)
                if np.isfinite(g.rg_nm):
                    r2 = float(g.r_squared)
                    cands.append(ResultCandidate(
                        value=float(g.rg_nm),
                        label=(f"Guinier at {_fmt_conc(c)} over {g.n_angles} angles "
                               f"(apparent, R² = {_fmt_r2(r2)})"
                               f"{_flag_suffix(g, 'rg_reliable')}"),
                        kind='sls_guinier', is_apparent=True, tier=2,
                        quality=(r2 if np.isfinite(r2) else None),
                        quality_kind='r_squared',
                        source_id=_per_series_source_id('guinier', i_series, c),
                        value_se=unc.se_or_none(getattr(g, 'rg_se', None)),
                        value_se_estimator=getattr(g, 'se_estimator', None),
                        auto_selectable=bool(getattr(g, 'rg_reliable', True))))
            except Exception:  # noqa: BLE001 — candidate isolation
                pass

        # tier 2 -- per-mode amplitude-Guinier from every recorded mode set
        # (apparent, MODE-RESOLVED; Sedlak 1996 Eq. 7). Live-computed off the
        # same masked Rayleigh series (no durable row store: the snapshot is
        # the durable object, and this whole function is memoized under a key
        # that includes _mode_set_state). auto_selectable=False: offered in
        # the picker, never the passive default -- with the tier-1 per-mode Rh
        # rows already shipped, a later rho_mode is a picker, not plumbing
        # (the spec's coupling clause).
        for rset in self.mode_sets.get((sample_id, fraction), []):
            # First-wins per identity concentration, mirroring
            # run_mode_partition: `nonzero` holds one entry per SLS
            # MEASUREMENT, so a repeat acquisition would otherwise mint a
            # second candidate with an IDENTICAL source_id (ckey-keyed) --
            # fitted from the very series the partition refuses (the
            # _per_series_source_id precedent one block up).
            seen_ckeys: set = set()
            for r in nonzero:
                c = r.concentration_g_per_mL
                ckey = round_concentration_for_identity(c)
                if ckey in seen_ckeys:
                    continue
                inputs = self._partition_inputs(rset, ckey)
                if not inputs:
                    continue
                seen_ckeys.add(ckey)
                try:
                    part = mode_partition.partition_rayleigh(r, inputs)
                except Exception:  # noqa: BLE001 — candidate isolation
                    continue
                for pm in part.per_mode:
                    try:
                        ag = mode_partition.amplitude_guinier(
                            pm.rayleigh.q2_nm2,
                            pm.rayleigh.excess_rayleigh_cm_inv,
                            source_item_ids=self._mode_point_ids(
                                rset, pm.label, ckey, r.angles_deg),
                            estimator=self.settings.se_estimator,
                            qrg_max_valid=self.settings.guinier_qrg_max,
                            # See the run arm above: the second of the two callers that
                            # must fill this, and the reason the sweep drives both.
                            blank_measured=bool(
                                getattr(pm.rayleigh, 'blank_measured', True)))
                    except Exception:  # noqa: BLE001 — candidate isolation
                        continue
                    if not np.isfinite(ag.rg_app_nm):
                        continue
                    r2 = float(ag.r_squared)
                    cands.append(ResultCandidate(
                        value=float(ag.rg_app_nm),
                        label=(f"Amplitude Guinier, mode '{pm.label}' of "
                               f"'{rset.label}' at {_fmt_conc(c)} (apparent, "
                               f"mode-resolved, R² = {_fmt_r2(r2)})"),
                        kind='sls_mode_guinier', is_apparent=True, tier=2,
                        quality=(r2 if np.isfinite(r2) else None),
                        quality_kind='r_squared',
                        source_id=(f'mode_guinier|{rset.set_id}|{pm.label}'
                                   f'|{ckey}'),
                        value_se=unc.se_or_none(ag.rg_se),
                        value_se_estimator=(ag.se_estimator
                                            if ag.rg_se is not None else None),
                        auto_selectable=False))

        # model-DEPENDENT form-factor Rg (sphere/coil/ellipsoid/Mie/core-shell) -- SAME tier as
        # the agnostic sibling (thermodynamic across c -> tier 3 like Zimm; apparent single c ->
        # tier 2 like Guinier), but flagged model_dependent so it never auto-wins a tie against a
        # slope-based Rg (select_default_candidate). Non-converged / contrast-matched solutions are
        # excluded by `_ff_rg_offered`. A DEGENERATE fit is no longer excluded (review-01): each
        # solution is offered as a non-auto-selectable pick (G3), grouped per model.
        offered_by_model: Dict[str, List[tuple]] = {}
        for model, sol in self._form_factor_solutions(sample_id, fraction):
            if not _ff_rg_offered(sol):
                continue
            offered_by_model.setdefault(model, []).append(sol)
        for model, sols in offered_by_model.items():
            # Solution cap (owner-set): a well-formed degenerate set is 3-4; if a fit returns more,
            # offer the top-N by fit quality with a "N of M shown" note rather than flooding the picker.
            capped, n_total, was_capped = _cap_solutions(sols)
            for sol in capped:
                cand = self._form_factor_rg_candidate(model, sol)
                if was_capped and cand.degenerate:
                    cand.label += (f" [showing {_FF_MAX_SOLUTIONS_OFFERED} of {n_total} "
                                   "by fit quality]")
                cands.append(cand)
            # Part C -- agree-on-Rg consensus: when a degenerate fit's solutions agree on Rg
            # (size determined even though the shape/radii are not), emit ONE extra auto-selectable
            # mean-Rg candidate so the common size-determined case is rescued without the user
            # picking a shape. model_dependent (a form factor was assumed) but degenerate=False (the
            # SIZE is not ambiguous); value_se=None (a mean of degenerate solutions has no honest ±).
            # Judge agreement + take the mean over the CAPPED (shown) set, not the uncapped one, so a
            # dropped outlier the user never sees can't suppress the rescue for the offered solutions.
            degen = [s for s in capped if getattr(s, 'degenerate', False)]
            rgs = [float(s.rg_nm) for s in degen]
            if len(degen) > 1 and _rg_values_agree(rgs):
                name = _FORM_FACTOR_LABELS.get(model, model)
                apparent = bool(degen[0].is_apparent)
                regime = 'apparent' if apparent else 'thermodynamic'
                cands.append(ResultCandidate(
                    value=float(sum(rgs) / len(rgs)),
                    label=(f"{name.capitalize()} P(q): size determined, shape ambiguous "
                           f"(mean of {len(degen)} solutions, {regime}) — model-dependent "
                           f"({name} P(q))"),
                    kind=f'sls_ff_{model}_consensus',
                    is_apparent=apparent, tier=(2 if apparent else 3),
                    quality=None, quality_kind='r_squared',
                    source_id=f'ff_{model}_consensus',
                    value_se=None, value_se_estimator=None,
                    model_dependent=True, model=name,
                    degenerate=False, auto_selectable=True))

        return cands

    @staticmethod
    def _form_factor_rg_candidate(model: str, sol) -> ResultCandidate:
        """Build the model-dependent Rg `ResultCandidate` for one offered form-factor solution
        (caller has already gated it with `_ff_rg_offered`).

        A DEGENERATE solution (ellipsoid/core-shell the data can't resolve to one shape) is flagged
        `degenerate=True`, made non-auto-selectable (G3 -- offered but never the silent default), and
        labelled by its discriminating shape parameters + 'user-selected among N degenerate solutions'
        (G2, neutral). A single resolved fit keeps the prior model-dependent-only label.

        Uncertainty (invariant 8): a resolved fit copies the fit's rg_se/estimator as a passthrough.
        A degenerate solution's per-branch regression SE reflects only the WITHIN-branch fit and
        under-reports the dominant size uncertainty (which branch), so it is WITHHELD (value_se=None)
        -- ill-posed in the shape dimension, exactly like NNLS/CONTIN and the agree-on-Rg consensus;
        the honest ± comes from selecting a shape externally, and the guided-manual-entry pick then
        mirrors a hand-entered Rg (no statistical SE). The degenerate flag carries the ambiguity."""
        name = _FORM_FACTOR_LABELS.get(model, model)
        r2 = float(getattr(sol, 'r_squared', float('nan')))
        degenerate = bool(getattr(sol, 'degenerate', False))
        descriptor = _ff_solution_descriptor(model, sol)
        return ResultCandidate(
            value=float(sol.rg_nm),
            label=_ff_candidate_label(name, sol, degenerate, descriptor, r2),
            kind=f'sls_ff_{model}', is_apparent=bool(sol.is_apparent),
            tier=(2 if sol.is_apparent else 3),
            quality=(r2 if np.isfinite(r2) else None),
            quality_kind='r_squared', source_id=_ff_source_id(model, descriptor),
            value_se=(None if degenerate else unc.se_or_none(getattr(sol, 'rg_se', None))),
            value_se_estimator=(None if degenerate else getattr(sol, 'se_estimator', None)),
            model_dependent=True, model=name,
            degenerate=degenerate, auto_selectable=not degenerate,
            solution_descriptor=descriptor)

    @staticmethod
    def _form_factor_mw_candidate(model: str, sol) -> ResultCandidate:
        """Build the model-dependent Mw `ResultCandidate` for one offered form-factor solution
        (caller has already gated it with `_ff_mw_offered`: calibrated + reliable). A resolved fit
        copies the fit's mw_se/estimator as a passthrough (invariant-8). Degenerate solutions are
        flagged + made non-auto-selectable and labelled by shape parameters, mirroring the Rg
        candidate (G2/G3), and their per-branch mw_se is WITHHELD for the same reason (the shape
        ambiguity dominates the scale uncertainty; see `_form_factor_rg_candidate`). Unlike Rg there
        is no Mw consensus candidate (Part C is agree-on-Rg / size-determined)."""
        name = _FORM_FACTOR_LABELS.get(model, model)
        r2 = float(getattr(sol, 'r_squared', float('nan')))
        degenerate = bool(getattr(sol, 'degenerate', False))
        descriptor = _ff_solution_descriptor(model, sol)
        return ResultCandidate(
            value=float(sol.mw_g_per_mol),
            label=_ff_candidate_label(name, sol, degenerate, descriptor, r2),
            kind=f'sls_ff_{model}', is_apparent=bool(sol.is_apparent),
            tier=(2 if sol.is_apparent else 3),
            quality=(r2 if np.isfinite(r2) else None),
            quality_kind='r_squared', source_id=_ff_source_id(model, descriptor),
            value_se=(None if degenerate else unc.se_or_none(getattr(sol, 'mw_se', None))),
            value_se_estimator=(None if degenerate else getattr(sol, 'se_estimator', None)),
            calibrated=bool(sol.calibrated),
            dn_dc_supplied=bool(sol.dn_dc_supplied),
            model_dependent=True, model=name,
            degenerate=degenerate, auto_selectable=not degenerate,
            solution_descriptor=descriptor)

    @staticmethod
    def _ladder_mw_candidate(res) -> ResultCandidate:
        """Build the apparent-Mw `ResultCandidate` for one single-angle ladder fit.

        Mirrors `_form_factor_mw_candidate` — one builder per offered fit — with
        three deliberate differences, all from the owner's 2026-09-03 ruling:

        * `auto_selectable=False`, ALWAYS. The ladder is "made selectable, never chosen for
          you": the angle is the user's choice, and nothing here may pick one for them. It is
          therefore never the passive default (`select_default_candidate` filters it out), even
          when it is the only candidate the sample has.
        * `is_apparent=True` / tier 2, always (invariant 7 — one angle, no q → 0 extrapolation),
          so a thermodynamic Zimm/Berry Mw always outranks it.
        * an uncalibrated or dn/dc-less ladder is still OFFERED (the Debye/Guinier rule), not
          excluded the way `_ff_mw_offered` excludes an uncalibrated form-factor Mw: the two
          scale flags travel as real fields and the label carries `_scale_caveat`, so the
          picker badges it and the user can still pick it deliberately. `model_dependent` stays
          False — the ladder is a slope/intercept, assuming no shape.
        """
        r2 = float(getattr(res, 'r_squared', float('nan')))
        return ResultCandidate(
            value=float(res.mw_apparent_g_per_mol),
            label=(_ladder_candidate_label(res)
                   + _reliability_note_suffix(res, 'mw_reliable')),
            kind=_LADDER_KIND, is_apparent=True, tier=2,
            quality=(r2 if np.isfinite(r2) else None),
            quality_kind='r_squared',
            source_id=_ladder_source_id(res.angle_deg),
            value_se=unc.se_or_none(getattr(res, 'mw_apparent_se', None)),
            value_se_estimator=getattr(res, 'se_estimator', None),
            calibrated=bool(res.calibrated),
            dn_dc_supplied=bool(res.dn_dc_supplied),
            auto_selectable=False)

    @staticmethod
    def _ladder_a2_candidate(res) -> ResultCandidate:
        """The A2 sibling of `_ladder_mw_candidate` — same fit, same label, same rules.

        A2 = slope/2 of the very line whose intercept gives Mw, so it inherits the angle
        contingency in full: it is 2·A2 only where P(q) ≈ 1, which is why it is apparent
        (tier 2) beside the thermodynamic Zimm/Berry A2 (tier 3). Its ± is the linear
        pass-through `slope_se/2` the engine already computed, copied as provenance
        (invariant 8 clause A), and its reliability sentence reads `a2_reliable` — the flag
        the engine writes for A2 — not Mw's.

        The calibration-free 2·A2·Mw product the same fit carries is NOT enrolled: it is a
        different quantity from A2 (a product, not a second virial coefficient), the picker
        has no axis for it, and offering it beside A2 would put two incomparable numbers under
        one heading."""
        r2 = float(getattr(res, 'r_squared', float('nan')))
        return ResultCandidate(
            value=float(res.a2_apparent_mol_mL_per_g2),
            label=(_ladder_candidate_label(res)
                   + _reliability_note_suffix(res, 'a2_reliable')),
            kind=_LADDER_KIND, is_apparent=True, tier=2,
            quality=(r2 if np.isfinite(r2) else None),
            quality_kind='r_squared',
            source_id=_ladder_source_id(res.angle_deg),
            value_se=unc.se_or_none(getattr(res, 'a2_apparent_se', None)),
            value_se_estimator=getattr(res, 'se_estimator', None),
            calibrated=bool(res.calibrated),
            dn_dc_supplied=bool(res.dn_dc_supplied),
            auto_selectable=False)

    def _ladder_angles(self, sample_id: str,
                       fraction: Optional[str] = None) -> List[float]:
        """The angles this (sample, fraction) has a STORED single-angle ladder for, sorted —
        the picker's ladder roster.

        The roster is the stored results (`results[('single_angle_ladder', sid, angle,
        fraction)]`, written by `run_single_angle_ladder`), because the ANGLE is the user's
        choice and the picker must not make it: one candidate per ladder the user actually ran,
        several angles coexisting as separate explicit picks (owner-ruled 2026-09-03).

        The VALUE, however, is re-derived from the current masked series by the candidate
        builders, exactly as the Zimm / Debye / Guinier candidates beside it are — the
        divergence from `_form_factor_solutions`, which reads its stored fit and gates it on a
        freshness stamp. The reason is cost, and it is the reason that stamp exists: a
        form-factor fit is a heavy multi-start solve dispatched to the worker thread and must
        never be re-run on a Cross-Sample refresh, while a ladder is one two-column least
        squares over a handful of points, cheaper than the Debye fit already recomputed in the
        same loop. Re-deriving removes the staleness question outright — there is no stored
        value here that could be offered against inputs it never saw — at the price of the
        picker's number tracking the current mask while the SLS tab still shows the fit as run
        (which that tab flags stale on its own surface).

        Iterates a SNAPSHOT (`list(...)`): `run_single_angle_ladder` writes `self.results` on
        the worker thread while this reads on the main thread, so a live iteration could raise
        "dictionary changed size". `self.results` holds MIXED-arity keys, so the 4-tuple shape
        is matched by length AND by `key[0]` (`('guinier', sid, conc, frac)` and
        `('form_factor', sid, model, frac)` are 4-tuples too)."""
        angles = {float(key[2]) for key in list(self.results)
                  if len(key) == 4 and key[0] == 'single_angle_ladder'
                  and key[1] == sample_id and key[3] == fraction}
        return sorted(angles)

    def _ladder_fits(self, sample_id: str, fraction: Optional[str],
                     nonzero_rayleigh) -> List:
        """One freshly-derived `SingleAngleLadderResult` per angle on the roster
        (`_ladder_angles`), in angle order; angles whose ladder no longer fits are dropped.

        The engine raises where the current data cannot support the fit — fewer than two
        non-zero concentrations carry the angle, or dropping the non-positive Kc/dR points
        leaves fewer than two — which is exactly the "no honest candidate" case, so it is
        swallowed per angle the way the Debye/Guinier loop swallows its own. Its
        `RuntimeWarning` for a dropped non-positive point is left to fire, exactly as the
        Debye and Guinier fits in the same builder leave theirs (`debye_analysis` warns on the
        same drop and nothing suppresses it): `warnings.catch_warnings` mutates a PROCESS-GLOBAL
        filter list, so suppressing here on the main thread would also swallow a warning a
        worker-thread fit raised in the same window -- a cost the sibling fits do not pay and
        this one has no reason to."""
        out: List = []
        for angle in self._ladder_angles(sample_id, fraction):
            try:
                out.append(sls_engine.single_angle_ladder(
                    nonzero_rayleigh, angle,
                    estimator=self.settings.se_estimator))
            except Exception:  # noqa: BLE001 — ladder-angle isolation
                continue
        return out

    def _form_factor_solutions(self, sample_id: str,
                               fraction: Optional[str] = None) -> List[tuple]:
        """The ``(model, solution)`` pairs from the form-factor fits the user has run for this
        (sample, fraction) — READ from the stored result the SLS-tab fit already computed on the
        worker thread, NOT re-fit here (so a Cross-Sample refresh never runs a heavy multi-start
        fit on the GUI thread — the read-stored design).

        A stored fit is offered ONLY while FRESH: its stamped `sls_run_signature` still equals the
        current one. A stale fit (the user edited the mask / calibration / SE-estimator after
        running it) is silently excluded until re-run on the SLS tab — which is also where it is
        flagged stale. This makes the honesty guarantee exact: a pooled form-factor size saw the
        SAME inputs as the Zimm candidates beside it (identical signature), rather than being
        recomputed each refresh. Ellipsoid / core-shell store a LIST -> one tuple per solution.

        A plain O(models) dict read (no memo needed): iterates a `list(self.results)` snapshot
        (run_form_factor writes it on the worker thread; this reads on the main thread)."""
        current = self._form_factor_run_signature(sample_id, fraction)
        out: List[tuple] = []
        for key in list(self.results):
            if not (len(key) == 4 and key[0] == 'form_factor'
                    and key[1] == sample_id and key[3] == fraction):
                continue
            if self._form_factor_run_sig.get(key) != current:
                continue                          # stale (or unstamped) -> not offered
            res = self.results.get(key)
            if res is None:
                continue
            model = key[2]
            for sol in (res if isinstance(res, list) else [res]):
                out.append((model, sol))
        return out

    def set_sample_rg(self, sample_id: str, candidate: ResultCandidate,
                      fraction: Optional[str] = None, *, explicit: bool = False) -> None:
        """Store a chosen SLS-derived Rg candidate as the sample/fraction's Rg.

        An explicit selection -- overrides any previously stored value, including a
        hand-entered one. Records provenance and apparent/thermodynamic status.
        `explicit=True` marks the source 'picked' (a deliberate user choice a later
        passive auto-select must not clobber); the default fill leaves it 'computed'.
        """
        r = self.workspace.samples[sample_id].result_for(fraction)
        r.rg_nm = float(candidate.value)
        r.rg_source = 'picked' if explicit else 'computed'
        r.rg_apparent = bool(candidate.is_apparent)
        r.rg_label = candidate.label
        r.rg_se = candidate.value_se
        r.rg_se_estimator = candidate.value_se_estimator  # copied provenance (invariant-8 Clause A)
        r.rg_model_dependent = bool(candidate.model_dependent)  # orthogonal axis (not rg_apparent)
        r.rg_model = candidate.model
        r.rg_degenerate = bool(candidate.degenerate)  # review-01: stronger qualifier (rule 10)
        set_result_key(r, 'rg', candidate.kind, candidate.source_id)

    def auto_select_rg(self, sample_id: str,
                       fraction: Optional[str] = None) -> Optional[ResultCandidate]:
        """Pick and store the default Rg for a sample (most-extrapolated first).

        Never overwrites a hand-entered Rg ('user') or an explicit pick ('picked');
        returns the chosen candidate, or None if no SLS Rg is available. Labeled,
        never silent.
        """
        r = self.workspace.samples[sample_id].result_for(fraction)
        if r.rg_source in DELIBERATE_TIERS and r.rg_nm is not None:
            return None
        chosen = select_default_candidate(self.sls_rg_candidates(sample_id, fraction))
        if chosen is not None:
            self.set_sample_rg(sample_id, chosen, fraction)
        return chosen

    def set_manual_rg(self, sample_id: str, rg_nm: float,
                      is_apparent: bool = False,
                      fraction: Optional[str] = None) -> None:
        """Record a hand-entered Rg (e.g. from external characterization).

        Marked provenance 'user' so a re-selection or a later Zimm run does not
        silently replace it. `is_apparent` defaults to thermodynamic (a hand-entered
        Rg is usually the best available value).
        """
        # isfinite too: hand-typed field, as for set_manual_rh.
        if not (math.isfinite(rg_nm) and rg_nm > 0):
            raise ValueError(f"rg_nm must be a positive finite number, got {rg_nm!r}.")
        r = self.workspace.samples[sample_id].result_for(fraction)
        r.rg_nm = float(rg_nm)
        r.rg_source = 'user'
        r.rg_apparent = bool(is_apparent)
        r.rg_label = 'user-entered'
        r.rg_se = None                    # a hand-entered value carries no statistical SE
        r.rg_se_estimator = None          # ... and no estimator provenance; clear any stale tag
        r.rg_model_dependent = False      # a hand-entered Rg assumes no shape; clear any stale flag
        r.rg_model = ''
        r.rg_degenerate = False           # ... and any degenerate flag (rule 10)
        set_result_key(r, 'rg', 'user', None)

    def sls_mw_candidates(self, sample_id: str,
                          fraction: Optional[str] = None) -> List[ResultCandidate]:
        """Available SLS-derived Mw values for a sample, each with provenance (memoized by
        `_sls_cached` under the run signature — see `_sls_mw_candidates_uncached`)."""
        # Shallow copy — see sls_rg_candidates (don't let a caller poison the memo).
        return list(self._sls_cached(
            'mw', sample_id, fraction,
            lambda: self._sls_mw_candidates_uncached(sample_id, fraction)))

    def _sls_mw_candidates_uncached(self, sample_id: str,
                                    fraction: Optional[str] = None) -> List[ResultCandidate]:
        """Available SLS-derived Mw values for a sample, each with provenance.

        tier 3  Zimm / Berry (thermodynamic, needs >= 2 concentrations);
        tier 2  Debye or Guinier per concentration (apparent).
        Unlike Rg, **Mw is calibration-dependent** -- an uncalibrated run gives an
        arbitrary-scale Mw, flagged in the label. For scaling plots the trustworthy
        Mw is usually the hand-entered one (e.g. characterized in water); that is
        offered separately as the 'user' value and preferred by the defaults.
        """
        cands: List[ResultCandidate] = []
        try:
            rr = self.masked_rayleigh(sample_id, fraction)
        except Exception:  # noqa: BLE001 — no reference, no candidates
            return cands
        if not self._offers_cross_sample_candidates(rr):
            return cands
        nonzero = [r for r in rr if r.concentration_g_per_mL != 0]
        n_conc = len({round_concentration_for_identity(r.concentration_g_per_mL)
                      for r in nonzero})

        if n_conc >= 2:
            for method in ('zimm', 'berry'):
                try:
                    res = sls_engine.zimm_analysis(rr, method=method,
                                                   estimator=self.settings.se_estimator)
                except Exception:  # noqa: BLE001 — candidate isolation
                    continue
                if not np.isfinite(res.mw_g_per_mol):
                    continue
                r2 = float(res.r_squared)
                extrap_kind, extrap_source_id = extrapolation_result_key(method)
                cands.append(ResultCandidate(
                    value=float(res.mw_g_per_mol),
                    label=(f"{method.capitalize()} extrapolation, "
                           f"{res.n_concentrations} concentrations (thermodynamic"
                           f"{_scale_caveat(res)}, R² = {_fmt_r2(r2)})"),
                    kind=extrap_kind, is_apparent=False, tier=3,
                    quality=(r2 if np.isfinite(r2) else None),
                    quality_kind='r_squared', source_id=extrap_source_id,
                    value_se=unc.se_or_none(getattr(res, 'mw_se', None)),
                    value_se_estimator=getattr(res, 'se_estimator', None),
                    calibrated=bool(res.calibrated),
                    dn_dc_supplied=bool(res.dn_dc_supplied)))

        for i_series, r in enumerate(nonzero):
            c = r.concentration_g_per_mL
            # The Guinier call seeds the Settings qRg limit like every other Guinier
            # site; the Guinier Mw verdict gates its own passive default (`_guinier_suffix`);
            # Debye carries no such flag, so its kwargs are the plain ones.
            for kind, fn, mw_attr, extra in (
                    ('sls_debye', sls_engine.debye_analysis, 'mw_apparent_g_per_mol', {}),
                    ('sls_guinier', sls_engine.guinier_analysis, 'mw_apparent_g_per_mol',
                     {'qrg_max_valid': self.settings.guinier_qrg_max})):
                try:
                    res = fn(r, estimator=self.settings.se_estimator, **extra)
                except Exception:  # noqa: BLE001 — candidate isolation
                    continue
                mw = float(getattr(res, mw_attr))
                if not np.isfinite(mw):
                    continue
                r2 = float(res.r_squared)
                name = 'Debye' if kind == 'sls_debye' else 'Guinier'
                cands.append(ResultCandidate(
                    value=mw,
                    label=(f"{name} at {_fmt_conc(c)} over {res.n_angles} angles "
                           f"(apparent{_scale_caveat(res)}, R² = {_fmt_r2(r2)})"
                           f"{_guinier_suffix(res)}"),
                    kind=kind, is_apparent=True, tier=2,
                    quality=(r2 if np.isfinite(r2) else None),
                    quality_kind='r_squared',
                    source_id=_per_series_source_id(kind, i_series, c),
                    value_se=unc.se_or_none(getattr(res, 'mw_apparent_se', None)),
                    value_se_estimator=getattr(res, 'se_estimator', None),
                    calibrated=bool(res.calibrated),
                    dn_dc_supplied=bool(res.dn_dc_supplied),
                    # Guinier results only. This loop builds the Debye candidate too,
                    # and `DebyeResult` HAS `mw_reliable` -- a conjunction including
                    # calibration and dn/dc -- so reading it here (rather than the regime
                    # verdict Debye does not carry) took an uncalibrated run's Debye Mw
                    # out of the passive default as well, leaving the sample with no Mw
                    # at all. `hasattr` is the same scope `_guinier_suffix` uses.
                    auto_selectable=(
                        bool(res.mw_reliable) if hasattr(res, 'guinier_valid')
                        else True)))

        # The single-angle ladders the user has RUN, one candidate each (owner-ruled
        # 2026-09-03): apparent, angle-labeled, and never the passive default. Placed after
        # the per-concentration loop so the picker lists it below Debye/Guinier at the same
        # tier, and offered whatever its scale flags say -- the Debye/Guinier rule, not
        # `_ff_mw_offered`'s exclusion (see `_ladder_mw_candidate`).
        for lad in self._ladder_fits(sample_id, fraction, nonzero):
            if np.isfinite(lad.mw_apparent_g_per_mol):
                cands.append(self._ladder_mw_candidate(lad))

        # model-DEPENDENT form-factor Mw (only when calibrated + reliable -- Mw is scale-
        # dependent, exactly like Zimm's). Same tier as the agnostic sibling; flagged
        # model_dependent so it never auto-wins a tie. Uncalibrated / non-converged solutions are
        # excluded by `_ff_mw_offered`. A DEGENERATE fit is no longer excluded (review-01): each
        # solution is offered as a non-auto-selectable pick (G3), grouped + capped per model.
        offered_by_model: Dict[str, List[tuple]] = {}
        for model, sol in self._form_factor_solutions(sample_id, fraction):
            if not _ff_mw_offered(sol):
                continue
            offered_by_model.setdefault(model, []).append(sol)
        for model, sols in offered_by_model.items():
            capped, n_total, was_capped = _cap_solutions(sols)
            for sol in capped:
                cand = self._form_factor_mw_candidate(model, sol)
                if was_capped and cand.degenerate:
                    cand.label += (f" [showing {_FF_MAX_SOLUTIONS_OFFERED} of {n_total} "
                                   "by fit quality]")
                cands.append(cand)
        return cands

    def set_sample_mw(self, sample_id: str, candidate: ResultCandidate,
                      fraction: Optional[str] = None, *, explicit: bool = False) -> None:
        """Store a chosen SLS-derived Mw candidate as the sample/fraction's Mw
        (explicit pick; overrides any previous value, including a hand-entered one).
        `explicit=True` marks the source 'picked' (a deliberate user choice a later
        passive auto-select must not clobber); the default fill leaves it 'computed'."""
        r = self.workspace.samples[sample_id].result_for(fraction)
        r.mw_g_per_mol = float(candidate.value)
        r.mw_source = 'picked' if explicit else 'computed'
        r.mw_apparent = bool(candidate.is_apparent)
        r.mw_label = candidate.label
        r.mw_se = candidate.value_se
        r.mw_se_estimator = candidate.value_se_estimator  # copied provenance (invariant-8 Clause A)
        r.mw_model_dependent = bool(candidate.model_dependent)  # orthogonal axis (not mw_apparent)
        r.mw_model = candidate.model
        r.mw_degenerate = bool(candidate.degenerate)  # review-01: stronger qualifier (rule 10)
        # Carry the calibration flag as a real field off the candidate -- never
        # parse it back out of the human-readable label string.
        r.calibrated = candidate.calibrated
        r.dn_dc_supplied = candidate.dn_dc_supplied
        set_result_key(r, 'mw', candidate.kind, candidate.source_id)

    def auto_select_mw(self, sample_id: str,
                       fraction: Optional[str] = None) -> Optional[ResultCandidate]:
        """Pick and store the default Mw (most-extrapolated first); never overwrites
        a hand-entered Mw ('user') or an explicit pick ('picked'). Returns the chosen
        candidate, or None."""
        r = self.workspace.samples[sample_id].result_for(fraction)
        if r.mw_source in DELIBERATE_TIERS and r.mw_g_per_mol is not None:
            return None
        chosen = select_default_candidate(self.sls_mw_candidates(sample_id, fraction))
        if chosen is not None:
            self.set_sample_mw(sample_id, chosen, fraction)
        return chosen

    def sls_a2_candidates(self, sample_id: str,
                          fraction: Optional[str] = None) -> List[ResultCandidate]:
        """Available SLS-derived A2 values for a sample, each with provenance (memoized by
        `_sls_cached` under the run signature — see `_sls_a2_candidates_uncached`)."""
        # Shallow copy — see sls_rg_candidates (don't let a caller poison the memo).
        return list(self._sls_cached(
            'a2', sample_id, fraction,
            lambda: self._sls_a2_candidates_uncached(sample_id, fraction)))

    def _sls_a2_candidates_uncached(self, sample_id: str,
                                    fraction: Optional[str] = None) -> List[ResultCandidate]:
        """Available SLS-derived A2 values for a sample, each with provenance.

        A2 is a second-virial (two-concentration) coefficient, so only the
        thermodynamic Zimm / Berry double extrapolation yields a THERMODYNAMIC one --
        there is no single-concentration A2 (Debye/Guinier give none). A single-angle
        LADDER does give one, and since 2026-09-03 it is offered: apparent
        (angle-contingent, tier 2), labeled with its angle and never the passive
        default -- see `_ladder_a2_candidate`. Like Mw, **A2 is
        calibration-dependent**: an uncalibrated run puts A2 on an arbitrary scale, and
        so does a missing dn/dc (K = 1). BOTH are recorded as real flags on each
        candidate (never parsed back out of the label).
        Needs >= 2 concentrations and a solvent reference; empty otherwise.
        """
        cands: List[ResultCandidate] = []
        try:
            rr = self.masked_rayleigh(sample_id, fraction)
        except Exception:  # noqa: BLE001 — no reference, no candidates
            return cands
        if not self._offers_cross_sample_candidates(rr):
            return cands
        nonzero = [r for r in rr if r.concentration_g_per_mL != 0]
        n_conc = len({round_concentration_for_identity(r.concentration_g_per_mL)
                      for r in nonzero})
        if n_conc < 2:
            return cands
        for method in ('zimm', 'berry'):
            try:
                res = sls_engine.zimm_analysis(rr, method=method,
                                               estimator=self.settings.se_estimator)
            except Exception:  # noqa: BLE001 — candidate isolation
                continue
            if not np.isfinite(res.a2_mol_mL_per_g2):
                continue
            r2 = float(res.r_squared)
            cal = bool(res.calibrated)
            dn_dc = bool(res.dn_dc_supplied)
            cal_note = _scale_caveat(res)
            extrap_kind, extrap_source_id = extrapolation_result_key(method)
            cands.append(ResultCandidate(
                value=float(res.a2_mol_mL_per_g2),
                label=(f"{method.capitalize()} extrapolation, "
                       f"{res.n_concentrations} concentrations (thermodynamic"
                       f"{cal_note}, R² = {_fmt_r2(r2)})"),
                kind=extrap_kind, is_apparent=False, tier=3,
                quality=(r2 if np.isfinite(r2) else None),
                quality_kind='r_squared', source_id=extrap_source_id,
                value_se=unc.se_or_none(getattr(res, 'a2_se', None)),
                value_se_estimator=getattr(res, 'se_estimator', None),
                calibrated=cal, dn_dc_supplied=dn_dc))

        # The apparent ladder A2, one per stored ladder -- the Mw builder's sibling, offered
        # under the same rules (owner-ruled 2026-09-03). The `n_conc < 2` return above is not
        # in its way: a ladder needs two concentrations too, so a sample that fails that test
        # has no ladder to offer either.
        for lad in self._ladder_fits(sample_id, fraction, nonzero):
            if np.isfinite(lad.a2_apparent_mol_mL_per_g2):
                cands.append(self._ladder_a2_candidate(lad))
        return cands

    def set_sample_a2(self, sample_id: str, candidate: ResultCandidate,
                      fraction: Optional[str] = None, *, explicit: bool = False) -> None:
        """Store a chosen SLS-derived A2 candidate as the sample/fraction's A2
        (explicit pick; overrides any previous value). Carries the calibration flag
        as a real field off the candidate -- never parsed from the label. A2 has no
        hand-entered path (unlike Mw): it is a solvent/T-specific interaction
        coefficient with no external 'standard' and is the y-axis the scaling plot
        fits, so only fit-derived candidates are offered. `explicit=True` marks the
        source 'picked' (a deliberate user choice a later passive auto-select must not
        clobber); the default fill leaves it 'computed'."""
        r = self.workspace.samples[sample_id].result_for(fraction)
        r.a2_mol_mL_per_g2 = float(candidate.value)
        r.a2_source = 'picked' if explicit else 'computed'
        r.a2_label = candidate.label
        r.a2_se = candidate.value_se
        r.a2_se_estimator = candidate.value_se_estimator  # copied provenance (invariant-8 Clause A)
        r.a2_calibrated = candidate.calibrated
        r.a2_dn_dc_supplied = candidate.dn_dc_supplied
        set_result_key(r, 'a2', candidate.kind, candidate.source_id)

    def auto_select_a2(self, sample_id: str,
                       fraction: Optional[str] = None) -> Optional[float]:
        """Pick and store the default A2 (best-tier Zimm/Berry fit) for the A2-Mw
        scaling plot; never overwrites a hand-set A2 ('user', reserved for a future
        external-value path) or an explicit pick ('picked'). Returns the stored A2, or
        None if none could be computed. Labeled via `set_sample_a2`, never silent."""
        r = self.workspace.samples[sample_id].result_for(fraction)
        if r.a2_source in DELIBERATE_TIERS:
            return r.a2_mol_mL_per_g2
        chosen = select_default_candidate(self.sls_a2_candidates(sample_id, fraction))
        if chosen is not None:
            self.set_sample_a2(sample_id, chosen, fraction)
            return r.a2_mol_mL_per_g2
        return None

    def compute_sample_rho(self, sample_id: str,
                           fraction: Optional[str] = None) -> SampleRho:
        """rho = Rg/Rh for a sample/fraction, from its stored Rg (SLS) and Rh (DLS).

        Requires both to be present (run a Zimm/Berry Rg in the SLS tab and select
        an Rh source first). rho is flagged apparent if EITHER input is apparent.

        No sample keys are passed to `compute_rho`, deliberately (owner-ruled
        2026-08-13): both inputs come out of ONE `result_for(fraction)` slot, so
        the Rg/Rh identity match is guaranteed by construction and re-checking it
        would re-derive what the construction already proves. The engine keeps its
        optional key check as an honest API for library callers assembling the two
        values from separate sources.
        """
        r = self.workspace.samples[sample_id].result_for(fraction)
        if r.rg_nm is None or not np.isfinite(r.rg_nm):
            raise ValueError(
                "No Rg for this sample. Select an SLS Rg source (Zimm/Berry for a "
                "thermodynamic Rg, or Debye/Guinier for an apparent one), or enter "
                "one.")
        if r.rh_nm is None or not np.isfinite(r.rh_nm):
            raise ValueError(
                "No Rh for this sample. Select a DLS Rh source (or enter one) "
                "first.")
        rr = compute_rho(rg_nm=float(r.rg_nm), rh_nm=float(r.rh_nm),
                         rg_se=r.rg_se, rh_se=r.rh_se)
        is_apparent = bool(r.rg_apparent) or bool(r.rh_apparent)
        # rho_se is a ratio of the Rg and Rh REGRESSION SEs, so it inherits their
        # covariance estimator. Read that provenance from the STORED per-SE tags on
        # the SampleResult -- NOT from the current global setting, which may have been
        # switched since these SEs were computed (the durable bundle is not recomputed
        # on a switch). Disclose the potentially-under-reporting one: 'ols' if either
        # input SE is OLS, else 'hc3'; None when rho_se is absent or neither input SE is
        # a regression SE. Invariant 8 clause A: an OLS-derived rho ± is never shown
        # unlabeled (see also gui/cross_module.py, which renders this tag).
        if rr.rho_se is None:
            rho_estimator = None
        else:
            ests = {e for e in (r.rg_se_estimator, r.rh_se_estimator) if e}
            rho_estimator = 'ols' if 'ols' in ests else ('hc3' if ests else None)
        return SampleRho(
            sample_id=sample_id, rho=rr.rho,
            rg_nm=float(r.rg_nm), rh_nm=float(r.rh_nm),
            rg_label=(r.rg_label or 'Rg'), rh_label=(r.rh_label or 'Rh'),
            rg_source=r.rg_source, rh_source=r.rh_source,
            is_apparent=is_apparent, interpretation=rr.interpretation,
            shape=rr.shape, rho_se=rr.rho_se, se_estimator=rho_estimator,
            rg_se=unc.se_or_none(r.rg_se), rh_se=unc.se_or_none(r.rh_se),
            rg_model_dependent=bool(getattr(r, 'rg_model_dependent', False)),
            rg_model=getattr(r, 'rg_model', ''),
            rg_degenerate=bool(getattr(r, 'rg_degenerate', False)),
            notes=rr.notes)

    def samples_with_sls(self) -> List[str]:
        """Sample ids with SLS data -- the Cross-Sample universe.

        Both cross-sample analyses need SLS: rho needs Rg (and DLS for Rh), the
        scaling plots need Mw and Rg. A DLS-only sample cannot contribute to either,
        so the tab's include/exclude list is the SLS samples; each view (rho table,
        scaling plots) then filters this set by its own eligibility.
        """
        return [sid for sid, s in self.workspace.samples.items() if s.has_sls]

    # ----------------------------------------------------------------------
    # Cross-sample scaling plots (Rg-Mw, A2-Mw)
    # ----------------------------------------------------------------------
    @staticmethod
    def _scaling_point(res, quantity: str):
        """The (Mw, y) pair a fraction contributes to a `quantity` scaling plot, or
        None if it contributes no point. y is Rg / Rh / A2 by quantity. A log-log
        scaling fit needs a positive finite Mw and a positive finite y -- A2 can be
        negative in a poor solvent (no point), and a non-positive Mw is meaningless --
        so both must be positive.

        This is the single source of truth for "does this fraction sit on the scaling
        plot?", shared by samples_for_scaling (eligibility) and compute_scaling (the
        points it emits), so the offered set and the plotted set can never diverge.
        """
        m = res.effective_mw()
        yv = (res.rg_nm if quantity == 'rg'
              else res.rh_nm if quantity == 'rh'
              else res.a2_mol_mL_per_g2)
        if (m is not None and np.isfinite(m) and m > 0
                and yv is not None and np.isfinite(yv) and yv > 0):
            return float(m), float(yv)
        return None

    def samples_for_scaling(self, quantity: str = 'rg') -> List[str]:
        """Sample ids with at least one fraction yielding a usable scaling point
        (a positive Mw and a positive y: Rg, Rh, or A2 for the A2 plot).

        Consults the SAME per-fraction enumeration compute_scaling emits points from
        (via the shared _scaling_point helper): a Mw series stored entirely in named
        fractions ("250k", "1M", ...) with an empty None fraction is still offered --
        it is eligible if ANY fraction contributes a point, not only the None fraction.
        """
        out = []
        for sid in self.workspace.samples:
            s = self.workspace.samples[sid]
            if any(self._scaling_point(s.result_for(frac), quantity) is not None
                   for frac in self.sample_fractions_any(sid)):
                out.append(sid)
        return out

    def compute_scaling(self, sample_ids: List[str],
                        quantity: str = 'rg') -> ScalingData:
        """Build the points + power-law fit for a scaling plot over `sample_ids`.

        quantity 'rg' -> Rg vs Mw (exponent nu); 'rh' -> Rh vs Mw (also a size
        exponent nu, Rh from DLS); 'a2' -> A2 vs Mw (exponent -a).
        Uses each sample's effective Mw (a hand-entered Mw wins). Samples without an
        Mw, or with a non-positive y, are excluded (and counted). This is a
        homologous-series analysis -- the caller curates which samples to include;
        mixing polymers/solvents on one fit is the user's responsibility.
        """
        if quantity not in ('rg', 'rh', 'a2'):
            raise ValueError(
                f"quantity must be 'rg', 'rh', or 'a2', got {quantity!r}.")
        labels, mw, y, ids = [], [], [], []
        b_mw, b_y, b_labels = [], [], []       # branched points: plotted, NOT fitted
        # Per-point provenance, parallel to each block (see `_scaling_point_source`).
        sources, b_sources = [], []
        arbitrary_scale = False
        model_dependent = False
        degenerate = False
        excluded = 0
        # One point per (sample, molecular-weight fraction): a Mw series of one
        # polymer stored as several fractions of one sample becomes several points.
        for sid in sample_ids:
            s = self.workspace.samples[sid]
            is_branched = self.sample_chain_type(sid) == 'branched'
            for frac in self.sample_fractions_any(sid):      # the table's unit list
                res = s.result_for(frac)
                point = self._scaling_point(res, quantity)   # shared eligibility test
                if point is None:
                    excluded += 1
                    continue
                m, yv = point
                # All three disclosure flags are PLOT-scoped and symmetric: set for ANY plotted point
                # (fitted or branched-hollow), BEFORE the branched `continue`. A plotted value on
                # an arbitrary scale, or one that assumes a particle shape, is worth surfacing even
                # when it is a hollow branched point shown only for visual comparison. (The caveat
                # wording is about a plotted VALUE, not a claim that the exponent itself is
                # shape-contingent — a branched point is excluded from the fit.)
                if _result_arbitrary_scale(res, quantity):
                    arbitrary_scale = True
                if _result_model_dependent(res, quantity):
                    model_dependent = True
                if _result_degenerate(res, quantity):
                    degenerate = True
                if is_branched:
                    # A branched chain contracts, so it must not bias the LINEAR
                    # scaling law -- keep it out of the fit but still plot it (drawn
                    # hollow). Untagged samples are treated as linear (the default).
                    b_mw.append(m); b_y.append(yv)
                    b_labels.append(self._sample_label(sid, frac))
                    b_sources.append(_scaling_point_source(res, quantity))
                    continue
                ids.append(sid)
                labels.append(self._sample_label(sid, frac))
                mw.append(m)
                y.append(yv)
                sources.append(_scaling_point_source(res, quantity))
        fit = fit_power_law(mw, y, estimator=self.settings.se_estimator)
        return ScalingData(quantity=quantity, fit=fit, sample_ids=ids, labels=labels,
                           mw=mw, y=y, any_arbitrary_scale=arbitrary_scale,
                           n_excluded=excluded, branched_mw=b_mw, branched_y=b_y,
                           branched_labels=b_labels, any_model_dependent=model_dependent,
                           any_degenerate=degenerate,
                           sources=sources, branched_sources=b_sources)

    def sample_fractions_any(self, sample_id: str) -> List[Optional[str]]:
        """Distinct Mw-fraction labels across BOTH kinds (SLS for Rg, DLS for Rh), so a
        DLS-only branched sample (h needs only Rh) still enumerates its fractions.

        THE unit enumeration for every Cross-Sample surface -- the table rows, the
        auto-picks, the Fraction combo, the scaling points and both exports read this one
        list. A surface reading the SLS list alone shows (or drops) units the others do
        not: a row the Source panel could never be pointed at, a point the plot omits.

        Only the kinds the sample HAS contribute: `sample_fractions` answers [None] for a
        kind with no measurements, which is "the default fraction" for an unlabeled series
        but a phantom unit for an absent kind (an SLS-only sample stored in named
        fractions would grow an empty "(unlabeled)" row and an excluded point)."""
        fr: set = set()
        for kind in ('sls', 'dls'):
            if self.workspace.sample_measurements(sample_id, kind):
                fr |= set(self.sample_fractions(sample_id, kind))
        # A unit that HOLDS a value -- a hand-entered Mw / Rg / Rh on a key no
        # measurement carries any more (entered before the series was tagged) -- stays
        # enumerated: a user's value never leaves the table, the reference law or the
        # exports silently (owner ruling 2026-08-27; CLAUDE.md's Mw-provenance rule).
        # Only an EMPTY unit is the phantom the kind filter above removes.
        sample = self.workspace.samples.get(sample_id)
        if sample is not None:
            for frac, r in sample.fraction_results.items():
                if any(not _absent(getattr(r, f, None))
                       for f in ('mw_g_per_mol', 'rg_nm', 'rh_nm')):
                    fr.add(frac)
        return sorted(fr, key=lambda f: (f is not None, f or '')) or [None]

    def compute_architecture(
            self, included_sample_ids: Optional[List[str]] = None,
            *, branching_model: str = 'regular_star') -> ArchitectureData:
        """Branching contraction factors g (from Rg) and h (from Rh) for the samples
        tagged Branched, against linear-only Rg-Mw and Rh-Mw laws fit to the samples
        tagged Linear (route 2).

        Chain-type tags are read from the samples themselves (`sample_chain_type`, a
        committed shared parameter set on the Data tab), restricted to
        `included_sample_ids` (all samples when None). The two laws are independent:
        `g` needs the Rg-Mw law + a branched Rg; `h` needs the Rh-Mw law + a branched
        Rh, so a sample can feed one and not the other. Each law needs >= 2 linear
        points at DISTINCT Mw and EXCLUDES branched points (non-circular). A branched
        sample's fractions sharing one Mw are treated as REPLICATES -> a ± = the
        replicate SEM; otherwise a point value (invariant 8: no single-shot analytic
        SE).

        `branching_model` selects how each `g` is READ ('regular_star' -> arm number;
        'random_tri' / 'random_tetra' -> a branching index n, Zimm & Stockmayer Eqs.
        47a/47b). It changes only the g reading, never the g value; `h` is unaffected
        (a scattering-g concept in this phase). Pure Python (no Qt).

        This is a homologous-series analysis, same contract as `compute_scaling`
        (owner-ruled 2026-08-13): the caller curates which samples to include, and
        mixing polymers/solvents in one reference law is the user's responsibility
        (combination doctrine rule 2, core/data_models.py -- the include/exclude
        list is the disclosure surface). The reference law is the more sensitive
        of the two, since a foreign linear point recalibrates every branched g.
        """
        samples = self.workspace.samples
        included = (list(samples) if included_sample_ids is None
                    else [s for s in included_sample_ids if s in samples])
        uncalibrated = False           # any contributing Mw came from an uncalibrated run
        model_dependent = False        # any contributing Mw (or, for an Rg point, Rg) is a
        #                                shape-assuming form-factor value
        degenerate = False             # ... and rests on a user-selected degenerate solution (rule 10)

        def _mw_uncalibrated(res) -> bool:
            return res.mw_is_arbitrary_scale()

        # Model-dependence uses the shared `_result_model_dependent` (single-sourced with the
        # scaling path, rule 10). Here every contributing point feeds an output (a linear point
        # feeds its law; a branched point feeds its g/h), so all are 'contributing'.

        # --- linear reference points (Rg and Rh independently) -------------------
        lin_rg_mw: List[float] = []; lin_rg: List[float] = []
        lin_rh_mw: List[float] = []; lin_rh: List[float] = []
        branched: List[str] = []
        for sid in included:
            role = self.sample_chain_type(sid)
            if role == 'branched':
                branched.append(sid)
                continue
            if role != 'linear':
                continue
            s = samples[sid]
            for frac in self.sample_fractions_any(sid):
                res = s.result_for(frac)
                pg = self._scaling_point(res, 'rg')
                if pg is not None:
                    lin_rg_mw.append(pg[0]); lin_rg.append(pg[1])
                    uncalibrated |= _mw_uncalibrated(res)
                    model_dependent |= _result_model_dependent(res, 'rg')
                    degenerate |= _result_degenerate(res, 'rg')
                ph = self._scaling_point(res, 'rh')
                if ph is not None:
                    lin_rh_mw.append(ph[0]); lin_rh.append(ph[1])
                    uncalibrated |= _mw_uncalibrated(res)   # Rh-only points count too
                    model_dependent |= _result_model_dependent(res, 'rh')
                    degenerate |= _result_degenerate(res, 'rh')

        est = self.settings.se_estimator
        rg_law = fit_power_law(lin_rg_mw, lin_rg, estimator=est)
        rh_law = fit_power_law(lin_rh_mw, lin_rh, estimator=est)
        # A power law needs >= 2 points AT DISTINCT Mw; two same-Mw linear samples make
        # fit_power_law report a fabricated fit_valid=True law (a min-norm lstsq slope),
        # which would then classify a branched sample from noise -- guard each law out.
        # A law is usable only if it can be EVALUATED at a branched Mw below: a prefactor
        # that left float range (NaN, see fit_power_law) has no value at any Mw.
        rg_ok = (rg_law.fit_valid and len({round(m, 6) for m in lin_rg_mw}) >= 2
                 and math.isfinite(rg_law.prefactor))
        rh_ok = (rh_law.fit_valid and len({round(m, 6) for m in lin_rh_mw}) >= 2
                 and math.isfinite(rh_law.prefactor))
        rg_range = (min(lin_rg_mw), max(lin_rg_mw)) if rg_ok else None
        rh_range = (min(lin_rh_mw), max(lin_rh_mw)) if rh_ok else None

        def _pack(msg: str) -> ArchitectureData:
            return ArchitectureData(
                rg_law=rg_law, rh_law=rh_law, rg_mw_range=rg_range, rh_mw_range=rh_range,
                n_linear_rg=len(lin_rg_mw), n_linear_rh=len(lin_rh_mw), rows=[],
                any_uncalibrated_mw=uncalibrated, any_model_dependent=model_dependent,
                any_degenerate=degenerate, message=msg)

        if not (rg_ok or rh_ok):
            # Name the precondition that actually failed: a law that fitted but whose
            # prefactor left float range cannot be read at any Mw, and telling the user
            # to tag more samples would name a step they have already taken.
            unrepresentable = [
                f'{lbl} (ν = {law.exponent:.3g})' for lbl, law in
                (('Rg', rg_law), ('Rh', rh_law))
                if law.fit_valid and not math.isfinite(law.prefactor)]
            if unrepresentable:
                return _pack('The fitted linear reference law for '
                             + ' and '.join(unrepresentable)
                             + ' has a prefactor outside floating-point range, so it cannot '
                             'be evaluated at any Mw — the linear points give a slope too '
                             'steep for a physical scaling law. Check the tagged Linear '
                             "samples' Mw and sizes.")
            return _pack('Tag at least two Linear samples at DIFFERENT Mw (each with an '
                         'Rg and/or Rh and an Mw) to fit the linear reference law(s).')
        if not branched:
            return _pack('Tag at least one Branched sample to compute g / h.')

        rg_note = (f'fitted Rg-Mw law · {len(lin_rg_mw)} linear points · '
                   f'Mw {rg_range[0]:.3g}-{rg_range[1]:.3g}') if rg_ok else ''
        rh_note = (f'fitted Rh-Mw law · {len(lin_rh_mw)} linear points · '
                   f'Mw {rh_range[0]:.3g}-{rh_range[1]:.3g}') if rh_ok else ''
        rows: List[ArchitectureRow] = []
        for sid in branched:
            s = samples[sid]
            rg_by_mw: Dict[float, List[float]] = {}
            rh_by_mw: Dict[float, List[float]] = {}
            for frac in self.sample_fractions_any(sid):
                res = s.result_for(frac)
                pg = self._scaling_point(res, 'rg')
                if pg is not None:
                    rg_by_mw.setdefault(round(pg[0], 6), []).append(pg[1])
                    uncalibrated |= _mw_uncalibrated(res)
                    model_dependent |= _result_model_dependent(res, 'rg')
                    degenerate |= _result_degenerate(res, 'rg')
                ph = self._scaling_point(res, 'rh')
                if ph is not None:
                    rh_by_mw.setdefault(round(ph[0], 6), []).append(ph[1])
                    uncalibrated |= _mw_uncalibrated(res)   # Rh-only points count too
                    model_dependent |= _result_model_dependent(res, 'rh')
                    degenerate |= _result_degenerate(res, 'rh')
            for mw in sorted(set(rg_by_mw) | set(rh_by_mw)):
                g_res = h_res = None
                if rg_ok and mw in rg_by_mw:
                    rgs = rg_by_mw[mw]
                    # np.power saturates where a Python-float `**` would raise (rule 12).
                    rg_lin = rg_law.prefactor * float(np.power(mw, rg_law.exponent))
                    g_res = util.branching_contraction_factor(
                        rgs if len(rgs) > 1 else rgs[0], rg_lin, mw_branched=mw,
                        mw_linear_range=rg_range, linear_ref_note=rg_note,
                        model=branching_model)
                if rh_ok and mw in rh_by_mw:
                    rhs = rh_by_mw[mw]
                    rh_lin = rh_law.prefactor * float(np.power(mw, rh_law.exponent))
                    h_res = util.hydrodynamic_contraction_factor(
                        rhs if len(rhs) > 1 else rhs[0], rh_lin, mw_branched=mw,
                        mw_linear_range=rh_range, linear_ref_note=rh_note)
                if g_res is None and h_res is None:
                    continue
                note = util.compare_g_h(g_res.g if g_res is not None else None,
                                        h_res.h if h_res is not None else None)
                rows.append(ArchitectureRow(
                    sample_id=sid, label=self._sample_label(sid), mw=mw,
                    g_result=g_res, h_result=h_res, compare_note=note))
        if not rows:
            # Branched samples are tagged, but none yielded a g or h: each needs the
            # matching linear reference law fitted AND its own Rg / Rh at a Mw. Without
            # this a Branched sample whose only radius lacks a valid law (e.g. Rg-only
            # while just the Rh law is fittable) would vanish with no explanation.
            return _pack('No g / h computed for the Branched sample(s): each needs the '
                         'matching linear reference law (≥2 Linear samples at different '
                         'Mw carrying that radius) AND its own Rg / Rh at a molar mass.')
        return ArchitectureData(
            rg_law=rg_law, rh_law=rh_law, rg_mw_range=rg_range, rh_mw_range=rh_range,
            n_linear_rg=len(lin_rg_mw), n_linear_rh=len(lin_rh_mw), rows=rows,
            any_uncalibrated_mw=uncalibrated, any_model_dependent=model_dependent,
            any_degenerate=degenerate, message='')

    def _sample_label(self, sample_id: str,
                      fraction: Optional[str] = None) -> str:
        """A sample's display name, from `core.naming` -- the one builder the GUI
        surfaces share. Qt-free, so the exports and the controller read the same body.

        An unconfirmed sample reads '(unconfirmed sample)', never the raw
        `polymer|solvent|temperature` key: that key used to reach the DLS Summary table
        and the exported CSV (owner-ruled 2026-09-17). An id the workspace does not
        hold still raises, as it did -- a failed lookup's subject IS the id.
        """
        return sample_display_name(self.workspace.samples[sample_id], fraction)

    # ----------------------------------------------------------------------
    # Utilities: I*sin(theta) instrument check + synthetic generator
    # ----------------------------------------------------------------------
    def run_i_sin_theta(self, sample_id: str, mode: str = 'absolute',
                        include_reference: bool = True):
        """I*sin(theta) vs angle, SOLVENT-ONLY (c = 0) measurements for a sample.

        I*sin(theta) is an optical-quality / alignment check: for an ideal
        isotropic, dust-free scattering volume it is flat across angle. That
        interpretation only holds for the solvent reference (c = 0); a polymer
        solution at finite c carries the chain form factor P(q), which is itself
        angle-dependent, so including finite-c curves muddies the diagnostic.
        We therefore plot ONLY the c = 0 measurements (the solvent reference and
        any other c = 0 SLS series). `mode` is 'absolute' or 'normalized'.

        `include_reference` is retained for API compatibility; the c = 0 filter
        already selects the solvent reference, so it no longer changes the set.
        """
        candidates = list(self.workspace.sample_measurements(sample_id, 'sls'))
        # EVERY blank in the sample, not the reference slot alone: an extra blank is
        # the subtrahend for the series from its own file, so it is exactly the c = 0
        # curve to vet here.
        candidates += self.workspace.blank_candidates(sample_id)
        blanks = [lm for lm in candidates
                  if lm.committed_params.get('concentration_g_per_mL') == 0]
        if not blanks:
            raise ValueError(
                "I·sin θ shows only solvent-reference (c = 0) curves, and this "
                "sample has none. Load or assign a c = 0 SLS measurement.")
        meas = [lm.build() for lm in blanks]
        # The engine formats a name it is GIVEN (invariant 1). Every curve here is a
        # c = 0 blank of ONE sample, so the sample is the plot's subject and the name
        # carries only what tells the blanks apart.
        res = util.i_sin_theta(
            meas, mode=mode,
            display_names=[measurement_display_name(lm) for lm in blanks])
        self.results[('i_sin_theta', sample_id, mode)] = res
        return res

    def export_synthetic(self, result, file_path: str, delay_unit: str = 's') -> str:
        """Write a generated correlogram as a generic-parser CSV (no headers,
        two columns)."""
        return util.export_synthetic_correlogram_csv(
            result, file_path, delay_unit=delay_unit)

    # ----------------------------------------------------------------------
    # Result export (Origin-compatible CSV). The GUI passes the result object it
    # is displaying; the controller stays the only caller of `exporting/`. These
    # are thin wrappers so the widgets never import the export layer directly.
    # ----------------------------------------------------------------------
    def _export_source_record(self, item_id: Optional[str]) -> Tuple[str, str]:
        """`(Source file cell, Comments note)` for a one-measurement export.

        The workspace is the only place a measurement's provenance lives -- the
        engine object the exporters receive is built from the committed parameters
        alone -- so this is where it is read, once, for every such export.

        A loaded measurement gives its bare file name and no note. A measurement
        with no file of its own gives an empty cell: a REPLICATE AVERAGE then names
        its contributors in the note (it is a many-source export by the column's
        rule, and the note is the export's only record of what went into it, so the
        list is complete rather than capped), and anything else says so plainly. An
        id the workspace no longer holds yields no cell rather than an error: an
        export already in flight must not fail over a name.
        """
        lm = self.workspace.measurements.get(item_id) if item_id else None
        if lm is None:
            return '', ''
        name = source_file_name(lm)
        if name:
            return name, ''
        sources = list(getattr(lm, 'derived_from', None) or [])
        if getattr(lm, 'derived_kind', None) == 'replicate_average' and sources:
            named = ', '.join(
                measurement_name(self.workspace.measurements[s])
                if s in self.workspace.measurements else s for s in sources)
            return '', f'averaged from {len(sources)} measurements: {named}'
        return '', ''

    def export_correlogram_fit(self, item_id: str, result, file_path: str) -> str:
        """Export a parametric DLS fit (cumulant / single / double / KWW)."""
        m = self.workspace.measurements[item_id].build()
        source_file, source_note = self._export_source_record(item_id)
        return exporter.export_correlogram_fit(
            m, result, file_path, source_file=source_file, source_note=source_note)

    def export_distribution(self, result, file_path: str, axis: str = 'rh', *,
                            item_id: Optional[str] = None) -> str:
        """Export an NNLS / CONTIN / lognormal distribution. Accepts a ContinResult
        (the exporter reads how α was chosen and the sweep-edge note off it) or a
        bare DistributionResult; the result is passed through unflattened."""
        source_file, source_note = self._export_source_record(item_id)
        return exporter.export_distribution(
            result, file_path, axis=axis,
            source_file=source_file, source_note=source_note)

    def export_lcurve(self, result, file_path: str, *,
                      item_id: Optional[str] = None) -> str:
        """Export a CONTIN alpha-selection sweep. Accepts a ContinResult or a bare
        LCurveResult and hands the RESULT through: the exporter unwraps the sweep itself
        and reads the gamma off the ContinResult, which the bare sweep does not carry."""
        source_file, source_note = self._export_source_record(item_id)
        return exporter.export_lcurve(result, file_path, source_file=source_file,
                                      source_note=source_note)

    def export_gamma_q2(self, result, file_path: str) -> str:
        return exporter.export_gamma_q2(result, file_path)

    def export_mode_map(self, sample_id: str, fraction: Optional[str],
                        file_path: str) -> str:
        """Export the cached mode map + per-family fits + amplitude report.

        Assembled from the controller's own stores (the exporter is pure):
        per-point family labels come from the picks by the same value match
        the fits use, so the Family column can never disagree with the fit."""
        map_res = self.results.get(('mode_map', sample_id, fraction))
        if map_res is None:
            raise ValueError('Run the mode map before exporting it.')
        fits = self.results.get(('mode_family', sample_id, fraction)) or {}
        fits = {label: f for label, f in fits.items() if f is not None}
        # The amplitude block is RECOMPUTED from the current picks/roles at
        # export time, never read from the view's cache -- no mutation
        # invalidated that cache, so it could disagree with the other blocks
        # in the same file. No
        # role pair -> no block, exactly as the view behaves.
        try:
            amplitude = self.mode_amplitude_report(sample_id, fraction)
        except ValueError:
            amplitude = None
        store = self.mode_picks.get((sample_id, fraction))
        fams = []
        for p in map_res.points:
            label = ''
            for fam in (store.families if store is not None else []):
                # The same state-EQUALITY match the fits use (rule 10 -- the
                # `is None` wildcard clause was finding 11's defect shape).
                if (fam.bound
                        and p.polarization_state == fam.polarization_state
                        and fam.has_pick(p.angle_deg, p.gamma_s_inv)):
                    label = fam.label
                    break
            fams.append(label)
        family_states = {f.label: f.polarization_state
                         for f in (store.families if store is not None else [])}
        return exporter.export_mode_map(map_res, fits, amplitude, file_path,
                                        point_families=fams,
                                        family_states=family_states)

    def _trace_provenance(self, lt) -> Tuple[str, str]:
        """Where a stored trace came from, and what was committed alongside it.

        Shared by every trace exporter so the two cannot describe the same trace
        differently -- the raw (t, cps) export and the windowed-Hampel
        decomposition carry identical provenance because they read it here.
        """
        bits = []
        if lt.sample_label:
            bits.append(str(lt.sample_label))
        if lt.source_path:
            bits.append(f'from {os.path.basename(lt.source_path)}')
        if lt.measurement_id and lt.measurement_id in self.workspace.measurements:
            label = self.workspace.measurements[lt.measurement_id].committed_params.get(
                'label') or lt.measurement_id
            bits.append(f'linked to {label}')
        else:
            bits.append('standalone')
        disclosure = ''
        if lt.measurement_id:
            status = self.mean_count_rate_status(lt.measurement_id)
            if status['value'] is not None:
                origin = status['source'] or 'unrecorded'
                # A 'trace:*' tag on a measurement with no trace linked NOW is
                # history, not a live provenance -- say which, or the comment reads
                # as a claim about a link the session no longer has.
                if status['trace_id'] is None and origin.startswith('trace'):
                    origin = f'{origin}, from a trace no longer linked'
                disclosure = (f"committed mean count rate "
                              f"{status['value']:.6g} cps (source: {origin})")
                if status['note']:
                    disclosure = f'{disclosure}; {status["note"]}'
        return '; '.join(bits), disclosure

    def export_trace(self, trace_id: str, file_path: str) -> str:
        """Export one stored count-rate trace as Origin-style (t, cps) columns."""
        lt = self.workspace.traces[trace_id]
        provenance, disclosure = self._trace_provenance(lt)
        return exporter.export_intensity_trace(
            lt.build(), file_path,
            provenance=provenance, disclosure=disclosure)

    def export_trace_windowed_hampel(self, trace_id: str, file_path: str, *,
                                     k: Optional[float] = None,
                                     window_pts: Optional[int] = None) -> str:
        """Export the windowed-Hampel decomposition of one stored trace.

        Both envelopes are computed so the file records what each said: the
        local decomposition as columns, the global band's parameters beside
        them. They share k, so a reader comparing the two is comparing windows
        and not settings.
        """
        lt = self.workspace.traces[trace_id]
        provenance, disclosure = self._trace_provenance(lt)
        trace = lt.build()
        if k is None:
            k = _TRACE_OUTLIER_K
        result = trace_engine.flag_outliers_windowed(trace, k=k,
                                                     window_pts=window_pts)
        return exporter.export_windowed_hampel(
            result, trace, file_path, provenance=provenance,
            disclosure=disclosure,
            global_flags=trace_engine.flag_outliers(trace, k=k))

    def export_trace_diagnostic(self, trace_id: str, view: str, file_path: str,
                                **kw) -> str:
        """Export one trace diagnostic's own data, Origin-style.

        Dispatched by view name so the GUI's "export what is on screen" button
        has exactly one call site, and so a view added to the tab without an
        export raises here instead of silently writing the wrong file. Recomputes
        from the stored trace rather than taking the GUI's memoized result,
        matching `export_trace_windowed_hampel`.
        """
        lt = self.workspace.traces[trace_id]
        provenance, disclosure = self._trace_provenance(lt)
        trace = lt.build()
        if view == 'histogram':
            result = trace_engine.fit_count_rate_histogram(trace, **kw)
            writer = exporter.export_count_rate_histogram
        elif view == 'block_variance':
            kw.setdefault('correlation_threshold', _TRACE_BLOCKVAR_THRESHOLD)
            result = trace_engine.block_variance(trace, **kw)
            writer = exporter.export_block_variance
        elif view == 'allan':
            kw.setdefault('correlation_threshold', _TRACE_BLOCKVAR_THRESHOLD)
            result = trace_engine.block_variance(trace, **kw)
            writer = exporter.export_allan_deviation
        elif view == 'psd':
            result = trace_engine.welch_psd(trace, **kw)
            writer = exporter.export_welch_psd
        elif view == 'periodogram':
            kw.setdefault(
                'correlations_detected',
                trace_engine.block_variance(
                    trace,
                    correlation_threshold=_TRACE_BLOCKVAR_THRESHOLD,
                ).correlations_detected)
            result = trace_engine.periodogram_false_alarm(trace, **kw)
            writer = exporter.export_periodogram
        elif view == 'running_average':
            # `running_average` demands an explicit width, so "auto" is resolved
            # here -- through the engine's rule, not a second copy of it.
            if kw.get('window_s') is None and kw.get('window_points') is None:
                kw['window_points'] = trace_engine.resolve_running_average_window(
                    len(trace.count_rates_cps))
            result = trace_engine.running_average(trace, **kw)
            writer = exporter.export_running_average
        else:
            raise ValueError(
                f"Unknown trace diagnostic {view!r}; expected one of "
                f"{sorted(TRACE_DIAGNOSTIC_VIEWS)}."
            )
        return writer(result, file_path, provenance=provenance,
                      disclosure=disclosure)

    def export_dls_summary(self, file_path: str,
                           ticked_only_ids: Optional[List[str]] = None) -> str:
        """Export the DLS Summary store as ONE long/tidy CSV: one row per result
        or distribution peak (NOT the wide on-screen shape), with separate numeric
        columns so Origin/pandas can plot them. The per-measurement rows honor the
        ticked filter; sample-level rows are always included."""
        allow = set(ticked_only_ids) if ticked_only_ids is not None else None
        records: List[Dict[str, Any]] = []
        # Table 1 -- per-measurement results (one record per peak/parametric row).
        meas_rows = sorted(
            self.workspace.dls_result_rows.values(),
            key=lambda r: (r.item_id, r.method, r.peak_index))
        for row in meas_rows:
            if allow is not None and row.item_id not in allow:
                continue
            sid = self._sample_id_of(row.item_id)
            lm = self.workspace.measurements.get(row.item_id)
            records.append({
                'sample': self._sample_label(sid) if sid else '',
                'measurement': self._measurement_label(row.item_id),
                # Which file this row was loaded from -- empty for a measurement
                # with no file of its own (a replicate average), whose contributors
                # the `From` column already names.
                'source_file': source_file_name(lm) if lm is not None else '',
                'method': row.method,
                'rh': row.rh_nm, 'rh_se': row.rh_se, 'pdi': row.pdi,
                'int_pct': None if row.int_fraction is None
                else row.int_fraction * 100.0,
                'rh_fast': row.rh_fast_nm, 'rh_slow': row.rh_slow_nm,
                'rh_type': 'apparent', 'is_average': False,
                'from': row.label, 'se_estimator': None,   # per-measurement fits
                # The size flag travels WITH the value on the way out. Kept key-identical
                # with the Table-2 dict below so the two record shapes cannot drift.
                'rh_note': getattr(row, 'reliability_note', ''),
            })                                             # carry no regression SE
        # Table 2 -- sample-level Rh (averages, Gamma-q^2, D-c).
        sample_rows = sorted(
            self.workspace.sample_rh_rows.values(),
            key=lambda r: (r.sample_id, r.source_kind, r.source_set))
        for row in sample_rows:
            records.append({
                'sample': self._sample_label(row.sample_id, row.fraction)
                if row.sample_id in self.workspace.samples else row.sample_id,
                'measurement': '', 'method': row.source_kind,
                # A sample-level row is built from several measurements: no single
                # source file, and `From` carries its contributors.
                'source_file': '',
                'rh': row.rh_nm, 'rh_se': row.rh_se, 'pdi': None,
                'int_pct': None, 'rh_fast': None, 'rh_slow': None,
                'rh_type': row.rh_type_label,
                'is_average': row.source_kind == 'replicate_avg',
                'from': row.from_label, 'se_estimator': row.se_estimator,
                # Sample-level rows carry no note: a flagged replicate average is never
                # written in the first place (the averaging declines it), and a flagged
                # Γ-q² / D-vs-c run -- which DOES write its row -- has no note channel on
                # `SampleRhRow` to carry the verdict here. Its own exporters
                # (`export_gamma_q2`, `export_concentration_extrapolation`) carry it; a
                # row-level note is a recorded follow-up.
                'rh_note': '',
            })
        return exporter.export_dls_summary(records, file_path)

    def export_ddls(self, result, file_path: str, shapes=None) -> str:
        return exporter.export_ddls(result, file_path, shapes=shapes)

    def export_concentration_extrapolation(self, result, file_path: str) -> str:
        return exporter.export_concentration_extrapolation(result, file_path)

    def export_zimm(self, rayleigh_results, zimm_result, file_path: str) -> str:
        return exporter.export_zimm(rayleigh_results, zimm_result, file_path)

    def export_debye(self, result, file_path: str) -> str:
        return exporter.export_debye(result, file_path)

    def export_guinier(self, result, file_path: str) -> str:
        return exporter.export_guinier(result, file_path)

    def export_form_factor(self, result, file_path: str) -> str:
        """Export a form-factor P(q) fit. Dispatches on the result shape/model: the sphere /
        Gaussian-coil / Mie fit is one FormFactorResult; the ellipsoid and core-shell fits are
        LISTs of solutions (multi-valued), routed by model to their own multi-block CSV."""
        if isinstance(result, (list, tuple)):
            results = list(result)
            if results and results[0].model == ff.CORE_SHELL:
                return exporter.export_core_shell_form_factor(results, file_path)
            return exporter.export_ellipsoid_form_factor(results, file_path)
        return exporter.export_form_factor(result, file_path)

    def export_weighted_distribution(self, result, file_path: str, *,
                                     item_id: Optional[str] = None) -> str:
        source_file, source_note = self._export_source_record(item_id)
        return exporter.export_weighted_distribution(
            result, file_path, source_file=source_file, source_note=source_note)

    def export_single_angle(self, result, file_path: str) -> str:
        return exporter.export_single_angle(result, file_path)

    def export_calibration_free_a2(self, result, file_path: str) -> str:
        return exporter.export_calibration_free_a2(result, file_path)

    def export_single_angle_ladder(self, result, file_path: str) -> str:
        return exporter.export_single_angle_ladder(result, file_path)

    def export_rayleigh_series(self, results, file_path: str) -> str:
        return exporter.export_rayleigh_series(results, file_path)

    def export_scaling(self, scaling_data: 'ScalingData', file_path: str) -> str:
        """Export a cross-sample scaling plot (Rg/Rh/A2 vs Mw) + its power-law fit."""
        sd = scaling_data
        # Every plotted point travels: the branched (hollow, unfitted) points and the
        # excluded count the tab prints, not just the fitted subset.
        return exporter.export_scaling(sd.quantity, sd.labels, sd.mw, sd.y,
                                       sd.fit, file_path,
                                       model_dependent=sd.any_model_dependent,
                                       arbitrary_scale=sd.any_arbitrary_scale,
                                       degenerate=sd.any_degenerate,
                                       branched_labels=sd.branched_labels,
                                       branched_mw=sd.branched_mw,
                                       branched_y=sd.branched_y,
                                       n_excluded=sd.n_excluded,
                                       sources=sd.sources,
                                       branched_sources=sd.branched_sources)

    def export_architecture(self, included_sample_ids: List[str],
                            file_path: str, *,
                            branching_model: str = 'regular_star') -> str:
        """Export the Architecture tab (the ρ / Chain-type table + the branching g/h
        results) as one Origin-format CSV. Chain-type tags come from the samples
        (committed shared param). `branching_model` matches the selected g reading, so a
        random model also exports the branching density n (Eqs. 47a/47b). Pure Python
        (no Qt)."""
        role_disp = {'linear': 'Linear', 'branched': 'Branched'}
        rho_rows = []
        for sid in included_sample_ids:
            s = self.workspace.samples.get(sid)
            if s is None or not s.can_pair_rho:
                continue
            chain = role_disp.get(self.sample_chain_type(sid), '')
            for frac in self.sample_fractions_any(sid):      # the table's unit list
                try:
                    r = self.compute_sample_rho(sid, frac)
                except Exception:  # noqa: BLE001 — row isolation
                    continue
                # The ± and its estimator travel with rho, as the g/h block's SEs do
                # (invariant 8 clause A): the screen shows them, so the file carries them.
                rho_rows.append((self._sample_label(sid, frac), r.rg_nm, r.rh_nm,
                                 r.rho, r.shape, chain, r.rg_label, r.rh_label,
                                 r.rho_se, r.se_estimator))
        data = self.compute_architecture(included_sample_ids,
                                         branching_model=branching_model)

        def _rng(res) -> str:
            return ('in range' if res.mw_in_range
                    else 'out of range (extrapolated)' if res.mw_in_range is False else '')

        gh_rows = []
        for row in data.rows:
            g, h = row.g_result, row.h_result
            rng = _rng(g) if g is not None else (_rng(h) if h is not None else '')
            gh_rows.append((
                row.label, row.mw,
                g.g if g is not None else float('nan'), g.g_se if g is not None else None,
                g.effective_arms if g is not None else None,
                h.h if h is not None else float('nan'), h.h_se if h is not None else None,
                h.effective_arms if h is not None else None,
                rng,
                g.branching_index if g is not None else None))

        def _law_tuple(law, rng):
            if not (law.fit_valid and rng is not None):
                return None
            return (law.exponent, law.prefactor, law.r_squared, law.n_points, rng[0], rng[1])

        return exporter.export_architecture(
            rho_rows, gh_rows,
            _law_tuple(data.rg_law, data.rg_mw_range),
            _law_tuple(data.rh_law, data.rh_mw_range),
            file_path, uncalibrated_mw=data.any_uncalibrated_mw,
            model_dependent=data.any_model_dependent,
            degenerate=data.any_degenerate,
            branching_model=branching_model)

    # ----------------------------------------------------------------------
    # Utilities: unified synthetic dataset generator
    # ----------------------------------------------------------------------
    # The Utilities ▸ Synthetic data tab generates several artifact kinds from one
    # set of inputs. Each artifact is BUILT once into a plain data object (these
    # methods), which the GUI can then preview (via the plotting layer), SAVE to a
    # loadable instrument file, and/or INJECT into the workspace as a sample. All the
    # physics lives in analysis.synthetic_dataset; the controller only orchestrates.

    @staticmethod
    def _pops(population_specs: List[Dict[str, float]]):
        return [synth.SyntheticPopulation(
                    rh_nm=float(p['rh_nm']), weight=float(p['weight']),
                    spread_cv=float(p.get('spread_cv', 0.0)))
                for p in population_specs]

    def synth_correlogram(self, population_specs: List[Dict[str, float]], **kw):
        """A single synthetic correlogram (g2-1) from size populations."""
        return synth.build_correlogram(self._pops(population_specs), **kw)

    def synth_trace(self, **kw) -> 'synth.SyntheticTrace':
        """A synthetic count-rate trace (shot noise + optional drift/spikes)."""
        return synth.build_count_rate_trace(**kw)

    def synth_multi_angle_dls(self, population_specs: List[Dict[str, float]], **kw):
        """Correlograms (+ count-rate trace) at every angle, one ALV-file's worth."""
        return _in_axis_terms(
            lambda: synth.build_multi_angle_dls(self._pops(population_specs), **kw), kw)

    def synth_two_mode_dls(self, **kw):
        """A fast+slow two-mode multi-angle set with per-angle ground truth
        (defaults to the 82° crossing configuration — see build_two_mode_dls)."""
        return _in_axis_terms(lambda: synth.build_two_mode_dls(**kw), kw)

    def synth_mode_resolved_tuple(self, **kw):
        """The linked mode-resolved set: two-mode correlograms + matched traces
        + per-angle static intensities from one partition ground truth (see
        build_mode_resolved_tuple)."""
        return synth.build_mode_resolved_tuple(**kw)

    def save_synth_mode_resolved(self, tup, out_dir: str, *,
                                 viscosity_Pa_s: Optional[float] = None) -> List[str]:
        """Write a mode-resolved tuple as a loadable folder (two .ASC with
        traces, calibrant trace CSV, parameters.txt; no pre-derived SLS)."""
        kw = {} if viscosity_Pa_s is None else {'viscosity_Pa_s': viscosity_Pa_s}
        return synth.save_mode_resolved_dataset(tup, out_dir, **kw)

    def inject_mode_resolved_tuple(self, tup, *, polymer_name: str,
                                   solvent_name: str) -> List[str]:
        """Add a mode-resolved tuple to the workspace: one DLS run (with its
        trace) per angle and concentration -- the c = 0 blank included, landing
        in the same sample through the ordinary grouping -- plus the calibrant
        trace as a standalone trace."""
        ids: List[str] = []
        for conc in sorted(tup.dls):
            ids += self.inject_multi_angle_dls(
                tup.dls[conc], polymer_name=polymer_name,
                solvent_name=solvent_name)
        if tup.calibrant_trace is not None:
            self.inject_trace(tup.calibrant_trace)
        return ids

    def synth_composition_to_intensity_specs(
            self, population_specs: List[Dict[str, float]], weighting: str, *,
            angle_deg: float, wavelength_nm: float, n_solvent: float,
            model: str = 'rayleigh', m: Optional[float] = None) -> List[Dict[str, float]]:
        """Forward-model population specs whose ``weight`` is a NUMBER/VOLUME/SURFACE
        composition into intensity-weighted specs, so a generated DLS set round-trips
        through the Phase-1b/Phase-3 reweighting from a known composition (the in-app
        generator-coverage path). Rayleigh's R^6 law is angle-independent; RDG's and Mie's
        carry the angular form factor (reweight at the SAME angle). ``model='mie'`` needs the
        relative index ``m`` (= n_solute/n_solvent). Spread CV is carried through unchanged.
        Returns new spec dicts (the caller feeds them to ``synth_multi_angle_dls`` /
        ``synth_correlogram``)."""
        pops = synth.intensity_populations_from_composition(
            [(s['rh_nm'], s['weight']) for s in population_specs], weighting,
            angle_deg=angle_deg, wavelength_nm=wavelength_nm, n_solvent=n_solvent,
            model=model, m=m)
        return [{'rh_nm': p.rh_nm, 'weight': p.weight,
                 'spread_cv': float(s.get('spread_cv', 0.0))}
                for p, s in zip(pops, population_specs, strict=True)]

    def synth_sls_set(self, *, mw: float, rg_nm: float, a2_mol_mL_per_g2: float,
                      angles_deg, concentrations_g_per_mL, wavelength_nm: float,
                      temperature_K: float, n_solvent: float, dn_dc: float,
                      calibrated: bool, geometry: Optional[str] = None,
                      solvent_intensity_90: float = synth.DEFAULT_SOLVENT_INTENSITY_90,
                      noise_level: float = 0.0, seed=None, polymer_name: str = '',
                      solvent_name: str = '', label: str = '', kind: str = 'zimm',
                      form_factor_model: str = 'guinier',
                      ellipsoid_axial_ratio: float = 1.0, m: Optional[float] = None,
                      r_core_nm: Optional[float] = None, r_outer_nm: Optional[float] = None,
                      m1: Optional[float] = None, m2: Optional[float] = None,
                      dn_dc_in_files: bool = True):
        """An SLS intensity set (full Zimm grid or single-angle/-concentration slice).

        `calibrated` only controls whether a SAVED Brookhaven file carries the
        calibration metadata (and whether a workspace injection prefills the session
        calibration); the intensities are generated the same way either way.
        `dn_dc_in_files` is its dn/dc twin: off, the saved file carries no dn/dc row
        and an injected measurement carries none, so the analysis takes the K = 1
        arbitrary-scale path (owner-ruled 2026-08-26; CLAUDE.md invariant 3 --
        the program still never proposes or defaults a dn/dc).
        `form_factor_model` picks the angular model: 'guinier' (small-qRg Zimm expansion)
        or the full form factor 'sphere' / 'debye' / 'ellipsoid' / 'mie' / 'core_shell'
        (large-qRg, or for Mie/core-shell large-x/high-contrast, curvature the Form factor P(q)
        fit recovers); `ellipsoid_axial_ratio` (eps = polar/equatorial) sets the ellipsoid shape,
        `m` (= n_solute/n_solvent) the Mie contrast, and the core/outer radii `r_core_nm`/
        `r_outer_nm` + contrasts `m1`/`m2` the core-shell geometry.
        """
        if geometry is None:
            geometry = self.settings.standard_geometry
        cal = synth.CalibrationSpec(
            wavelength_nm=wavelength_nm, temperature_C=temperature_K - 273.15,
            geometry=geometry, include_in_files=bool(calibrated))
        return synth.build_sls_set(
            mw=mw, rg_nm=rg_nm, a2_mol_mL_per_g2=a2_mol_mL_per_g2, angles_deg=angles_deg,
            concentrations_g_per_mL=concentrations_g_per_mL, wavelength_nm=wavelength_nm,
            temperature_K=temperature_K, n_solvent=n_solvent, dn_dc=dn_dc, cal=cal,
            solvent_intensity_90=solvent_intensity_90, noise_level=noise_level, seed=seed,
            polymer_name=polymer_name, solvent_name=solvent_name, label=label, kind=kind,
            form_factor_model=form_factor_model,
            ellipsoid_axial_ratio=ellipsoid_axial_ratio, m=m,
            r_core_nm=r_core_nm, r_outer_nm=r_outer_nm, m1=m1, m2=m2,
            dn_dc_in_files=dn_dc_in_files)

    def synth_depolarized(self, *, rho_v: float, shape: str, dims_nm, angles_deg,
                          wavelength_nm: float, n_solvent: float, temperature_K: float,
                          viscosity_Pa_s: float, beta: float = 0.9,
                          vh_noise_level: float = 0.02, n_points: int = 200,
                          seed=None, label: str = 'DDLS',
                          concentration_g_per_mL: float = 0.0, baseline: float = 0.0):
        """The 'Depolarized (VV/VH)' artifact: a static intensity pair (drives the SLS
        Cabannes readout) + a dynamic VV/VH multi-angle set (drives DDLS D_r + shape).
        `baseline` is the generator's flat offset, planted on both channels.

        `shape` is 'rod' (dims = (L, d), Tirado), 'ellipsoid' (dims = (a, b), Perrin) or
        'sphere' (dims = (R, R), Stokes-Einstein-Debye -- the one-unknown model whose
        R from D_r and R from D_t must agree, which is what `ddls_shape(model='sphere')`
        checks). `rho_v` is the static depolarization ratio in [0, 3/4). Returns a
        DepolarizedArtifact (`.pair`, `.ddls`).
        """
        return synth.build_depolarized(
            rho_v=rho_v, shape=shape, dims_nm=dims_nm, angles_deg=angles_deg,
            wavelength_nm=wavelength_nm, n_solvent=n_solvent, temperature_K=temperature_K,
            viscosity_Pa_s=viscosity_Pa_s, beta=beta, vh_noise_level=vh_noise_level,
            n_points=n_points, seed=seed, label=label,
            concentration_g_per_mL=concentration_g_per_mL, baseline=baseline)

    def synth_architecture_series(self) -> Dict[str, dict]:
        """The branching-architecture ground-truth spec (linear ladder + stars + random
        tri/tetra fractions) shared with the developer generator -- one source of truth."""
        return synth.build_architecture_series()

    def synth_form_factor_series(self) -> Dict[str, dict]:
        """The form-factor SET ground-truth spec (shape-verdict panel + degenerate cases +
        sphere scaling ladder + apparent/uncalibrated variants), shared with the developer
        generator -- one source of truth."""
        return synth.build_form_factor_series()

    # ---- save a built artifact to its loadable instrument file ----
    def save_synth_trace(self, trace, path: str, count_rate_unit: str = 'kcps') -> str:
        return synth.write_trace_csv(path, trace, count_rate_unit=count_rate_unit)

    def save_synth_multi_angle_dls(self, dls, path: str,
                                   viscosity_cp: Optional[float] = None) -> str:
        return synth.write_alv_asc(path, dls, viscosity_cp=viscosity_cp)

    def save_synth_sls_set(self, sls, path: str) -> str:
        return synth.write_brookhaven_sls(path, sls)

    def save_synth_depolarized(self, artifact, out_dir: str,
                               viscosity_cp: Optional[float] = None) -> List[str]:
        """Write a DepolarizedArtifact's dynamic set as two loadable ALV files -- the VV
        and VH multi-angle correlograms. The static intensity pair has no instrument-file
        form (it is a readout the user types into the SLS anisotropy panel), so it is not
        written. Returns the paths written."""
        import os
        os.makedirs(out_dir, exist_ok=True)
        vv_path = os.path.join(out_dir, 'DDLS VV.ASC')
        vh_path = os.path.join(out_dir, 'DDLS VH.ASC')
        synth.write_alv_asc(vv_path, artifact.ddls.vv, viscosity_cp=viscosity_cp)
        synth.write_alv_asc(vh_path, artifact.ddls.vh, viscosity_cp=viscosity_cp)
        return [vv_path, vh_path]

    def generate_full_test_set(self, out_dir: str, profile_name: str) -> List[str]:
        """Regenerate a whole test-data/Synthetic * folder (data + parameters.txt)."""
        return synth.generate_full_dataset(out_dir, profile_name)

    def save_synth_architecture(self, out_dir: str) -> List[str]:
        """Regenerate the branching-architecture test set (SLS + DLS per sample +
        parameters.txt) into `out_dir`. Returns the paths written."""
        return synth.generate_architecture_dataset(out_dir)

    def save_synth_form_factor(self, out_dir: str) -> List[str]:
        """Regenerate the form-factor SET test data (SLS form-factor P(q) + paired DLS per
        sample + parameters.txt) into `out_dir`. Returns the paths written."""
        return synth.generate_form_factor_dataset(out_dir)

    # ---- inject a built artifact into the workspace as analysable data ----
    def inject_correlogram(self, result, *, polymer_name: str, solvent_name: str,
                           concentration_g_per_mL: float, temperature_K: float,
                           viscosity_Pa_s: Optional[float] = None,
                           trace=None, trace_label: Optional[str] = None) -> str:
        """Add a generated correlogram to the workspace as a DLS measurement.

        Assumes the generated signal is g2-1 (the analysable form). Returns the
        new item_id; the GUI should refresh the navigator + Data tab afterwards.

        `trace` (a SyntheticTrace from build_count_rate_trace), when given, is
        stored alongside and LINKED to the new measurement -- the single-measurement
        mirror of inject_multi_angle_dls's add_traces, so an attached-trace workflow
        is reachable from synthetic ground truth end to end. Find it afterwards with
        traces_for_measurement(item_id).
        """
        raw = {'delay_times_s': [float(x) for x in result.delay_times_s],
               'correlogram': [float(x) for x in result.signal]}
        params = dict(
            polymer_name=polymer_name, solvent_name=solvent_name,
            concentration_g_per_mL=float(concentration_g_per_mL),
            temperature_K=float(temperature_K), angle_deg=float(result.angle_deg),
            wavelength_nm=float(result.wavelength_nm),
            solvent_refractive_index=float(result.solvent_refractive_index),
            viscosity_Pa_s=float(viscosity_Pa_s if viscosity_Pa_s is not None
                                 else result.viscosity_Pa_s))
        self._seed_generator_inputs(raw, params, sample_times_s=result.sample_times_s,
                                    grid=result.grid, rate_cps=result.mean_count_rate_cps,
                                    duration_s=result.duration_s)
        # After the axes are in `params`: the label only needs a suffix where they
        # cannot tell this run from one already loaded.
        params['label'] = self._generated_label(
            result, polymer_name, solvent_name, params)
        item_id = self.add_loaded('dls', raw, params)
        if trace is not None:
            self.add_trace(
                [float(x) for x in trace.times_s],
                [float(x) for x in trace.count_rates_cps],
                sample_label=(trace_label or trace.label or 'synthetic trace'),
                measurement_id=item_id)
        return item_id

    def _generated_label(self, artifact, polymer_name: str, solvent_name: str,
                         params: Dict[str, Any]) -> str:
        """The label a GENERATED measurement carries, in the generator's own idiom --
        made DISTINCT where the within-sample axes cannot do it.

        A generated measurement has no source file, so without a label it would be
        named by its internal id, which a saved figure or an exported file carries to
        a reader who has no workspace to look it up in. The producers already build a
        name (`'{poly} in {solv}'`, `'... DDLS VH'`) and the injectors dropped it; this
        hands it to the measurement, falling back to the same idiom when an artifact
        carries none.

        The suffix is decided on the prospective DISPLAY NAME rather than on the label,
        because the axes already tell most generated runs apart: a multi-angle set gets
        no suffix at all. Replicates at ONE angle and concentration are the case the
        sample-level idiom cannot distinguish -- and the case this sheet exists to fix,
        where six runs otherwise read as one string and the engine's
        count-rate-disagreement note names three runs identically in an exported cell
        (owner-ruled 2026-09-18).
        """
        label = str(getattr(artifact, 'label', '') or '').strip()
        if not label:
            poly, solv = (polymer_name or '').strip(), (solvent_name or '').strip()
            label = f'{poly} in {solv}' if poly and solv else (poly or solv or 'generated')
        # The ONE builder decides the collision, on a stand-in carrying the parameters
        # this measurement is about to be created with -- never a second copy of the
        # display format (invariant: `core.naming` is the only place a name is built).
        taken = {measurement_display_name(lm)
                 for lm in self.workspace.measurements.values()}
        probe = SimpleNamespace(committed_params=dict(params), item_id='',
                                derived_kind=None, derived_from=None, source_path=None)
        base, k = label, 1
        while True:
            probe.committed_params['label'] = label
            if measurement_display_name(probe) not in taken:
                return label
            k += 1
            label = f'{base} #{k}'

    @staticmethod
    def _seed_generator_inputs(raw: Dict[str, Any], params: Dict[str, Any], *,
                               sample_times_s, grid, rate_cps, duration_s) -> None:
        """Hand a generated draw's own instrument inputs to the measurement being injected.

        The multi-tau builder's EXACT per-channel sample times go into `raw` beside the
        lags (the same place a parser's go), and the mean count rate the draw's
        shot-noise terms consumed (`rate_cps`) is seeded as the committed count rate,
        tagged 'generator' -- so the cumulant fit's sourced weight is evaluated on the
        very inputs the noise was drawn with, and synthetic ground truth exercises the
        instrument-input path end to end. Only the multi-tau grid's sample times are the
        layout's own: on a plain log or linear grid the generator's array is the same
        axis-derived stand-in the engine would produce, so seeding it would let the fit
        record 'instrument' for a sample time nothing stated. A draw on a plain grid, or
        one that consumed no rate, seeds nothing for that input. The run duration
        (`duration_s`) goes into `raw` too, an instrument fact like the sample times,
        when the producer states one. The inputs are named by the caller, never read off
        a producer by attribute: a producer missing one fails at the call site instead of
        silently seeding a default. Mutates `raw` and `params`.
        """
        if sample_times_s is not None and grid == 'multitau':
            raw['sample_times_s'] = [float(x) for x in sample_times_s]
        rate = usable_rate(rate_cps)
        if rate is not None:
            params['mean_count_rate_cps'] = rate
            params['mean_count_rate_cps_source'] = 'generator'
        if duration_s is not None:
            raw['duration_s'] = float(duration_s)

    def inject_multi_angle_dls(self, dls, *, polymer_name: str, solvent_name: str,
                               add_traces: bool = True,
                               analyzer_geometry: Optional[str] = None,
                               chain_type: Optional[str] = None,
                               mw_fraction: Optional[str] = None) -> List[str]:
        """Add a multi-angle DLS set as one DLS measurement per angle (and, by
        default, each angle's count-rate trace, linked back to its measurement).

        `analyzer_geometry` ('VV'/'VH'/'VU'), when given, tags every injected
        correlogram's polarization -- exactly what the Data-tab dropdown sets -- so a
        VV set and a VH set injected at the same angles pair for DDLS out of the box.
        `chain_type` ('linear'/'branched'), when given, tags the sample's architecture
        (the Data-tab Chain-type dropdown), so the Cross-Sample Architecture tab reads it.
        `mw_fraction`, when given, tags every injected correlogram's fraction label. It
        lives here rather than on the generated set because it is a label, not physics --
        the same reason `analyzer_geometry` and `chain_type` do.

        Each angle is committed the count rate ITS OWN draw consumed (`noise_rate_cps`,
        the set's brightness model at that angle -- the number its trace is built
        around), never one set-wide scalar: the committed rate and the linked trace then
        describe one quantity, and a static series derived from the set carries the
        set's angular shape rather than a flat line against its own traces.
        """
        delay = [float(x) for x in dls.delay_times_s]
        rates = dls.noise_rate_cps or {}
        ids: List[str] = []
        for a in dls.angles_deg:
            raw = {'delay_times_s': delay,
                   'correlogram': [float(x) for x in dls.signals[a]]}
            params = dict(
                polymer_name=polymer_name, solvent_name=solvent_name,
                concentration_g_per_mL=float(dls.concentration_g_per_mL),
                temperature_K=float(dls.temperature_K), angle_deg=float(a),
                wavelength_nm=float(dls.wavelength_nm),
                solvent_refractive_index=float(dls.n_solvent),
                viscosity_Pa_s=float(dls.viscosity_Pa_s))
            if analyzer_geometry is not None:
                params['analyzer_geometry'] = analyzer_geometry
            if chain_type is not None:
                params['chain_type'] = chain_type
            if mw_fraction is not None:
                params['mw_fraction'] = mw_fraction
            # The set's run duration goes to every angle whatever its noise model: it is
            # what the set's ALV file states and its parser reads, so the injected set and
            # the file written from it hand the engine one T.
            self._seed_generator_inputs(raw, params, sample_times_s=dls.sample_times_s,
                                        grid=dls.grid, rate_cps=rates.get(a),
                                        duration_s=dls.run_duration_s)
            params['label'] = self._generated_label(
                dls, polymer_name, solvent_name, params)
            iid = self.add_loaded('dls', raw, params)
            ids.append(iid)
            if add_traces:
                self.add_trace(
                    [float(x) for x in dls.trace_times_s],
                    [float(x) for x in dls.trace_cps[a]],
                    sample_label=f'{dls.label} @ {a:g}°', measurement_id=iid)
        return ids

    def inject_sls_set(self, sls, *, polymer_name: str, solvent_name: str,
                       prefill_calibration: bool = False,
                       chain_type: Optional[str] = None) -> List[str]:
        """Add an SLS set as one SLS measurement per concentration (the c = 0 entry
        groups as the sample's solvent reference). With `prefill_calibration`, and if
        the set is calibrated, also set the session calibration to the generator's
        standard so an immediate Zimm run gives an absolute Mw. `chain_type`
        ('linear'/'branched'), when given, tags the sample's architecture."""
        angles = [float(a) for a in sls.angles_deg]
        ids: List[str] = []
        for c in sls.concentrations_g_per_mL:
            raw = {'angles_deg': angles,
                   'intensities': [float(x) for x in sls.intensities[c]]}
            params = dict(
                polymer_name=polymer_name, solvent_name=solvent_name,
                concentration_g_per_mL=float(c), temperature_K=float(sls.temperature_K),
                wavelength_nm=float(sls.wavelength_nm),
                solvent_refractive_index=float(sls.n_solvent),
                # A dn/dc-less set injects with NO increment, as its written file
                # would load: the K = 1 arbitrary-scale path, never a guessed value.
                dn_dc_mL_per_g=(float(sls.dn_dc) if getattr(sls, 'dn_dc_in_files', True)
                                else None))
            # A Mie set carries the hand-entered solute RI, and a core-shell set the CORE
            # (solute) + SHELL indices, so the injected artifact is fittable end-to-end; None
            # for every other model, left unset. (c = 0 solvent reference carries no RI.)
            if sls.solute_refractive_index is not None and float(c) != 0.0:
                params['solute_refractive_index'] = float(sls.solute_refractive_index)
            if sls.shell_refractive_index is not None and float(c) != 0.0:
                params['shell_refractive_index'] = float(sls.shell_refractive_index)
            if chain_type is not None:
                params['chain_type'] = chain_type
            params['label'] = self._generated_label(
                sls, polymer_name, solvent_name, params)
            ids.append(self.add_loaded('sls', raw, params))
        if prefill_calibration and sls.calibrated:
            self._prefill_calibration_from_spec(sls.cal)
        return ids

    def inject_trace(self, trace, *, label: Optional[str] = None) -> str:
        """Add a generated count-rate trace to the workspace trace store."""
        return self.add_trace(
            [float(x) for x in trace.times_s],
            [float(x) for x in trace.count_rates_cps],
            sample_label=(label or trace.label or 'synthetic trace'))

    def inject_depolarized(self, artifact, *, polymer_name: str, solvent_name: str,
                           mw_fraction: Optional[str] = None) -> List[str]:
        """Add a DepolarizedArtifact's dynamic set as VV + VH correlograms for one
        sample, each tagged with its polarization (`analyzer_geometry`), so the sample's
        DDLS sub-tab pairs them by angle out of the box. `mw_fraction`, when given, tags
        both channels. The static intensity pair is a readout only (no measurement type),
        so it is not injected. Returns the item_ids."""
        ids = self.inject_multi_angle_dls(
            artifact.ddls.vv, polymer_name=polymer_name, solvent_name=solvent_name,
            analyzer_geometry='VV', mw_fraction=mw_fraction)
        ids += self.inject_multi_angle_dls(
            artifact.ddls.vh, polymer_name=polymer_name, solvent_name=solvent_name,
            analyzer_geometry='VH', mw_fraction=mw_fraction)
        return ids

    def inject_architecture_series(self, *, solvent_name: str = 'water',
                                   seed: Optional[int] = 17) -> Dict[str, List[str]]:
        """Add the whole branching-architecture series to the workspace: each sample's
        SLS Zimm set + multi-angle DLS, with its Chain-type tag (Linear/Branched) set so
        the Cross-Sample Architecture tab reads g / f_eff / n immediately. Also prefills
        the session calibration (the sets are calibrated), so an immediate Zimm run gives
        an absolute Mw. Returns {sample_name: [item_ids]}."""
        series = synth.build_architecture_series()
        rng = np.random.default_rng(seed)
        out: Dict[str, List[str]] = {}
        prefilled = False
        for name, p in series.items():
            chain = 'linear' if p['role'] == 'linear' else 'branched'
            sls = self.synth_sls_set(
                mw=p['mw'], rg_nm=p['rg'], a2_mol_mL_per_g2=p['a2'],
                angles_deg=synth.DEFAULT_SLS_ANGLES,
                concentrations_g_per_mL=[c * 1e-3 for c in synth.DEFAULT_SLS_CONCS_MG],
                wavelength_nm=synth.DEFAULT_WAVELENGTH_NM,
                temperature_K=synth.DEFAULT_TEMPERATURE_K, n_solvent=synth.DEFAULT_N_SOLVENT,
                dn_dc=synth.DEFAULT_DN_DC, calibrated=True,
                noise_level=0.005, seed=int(rng.integers(1 << 31)),
                polymer_name=name, solvent_name=solvent_name, label=f'{name} in {solvent_name}')
            dls = self.synth_multi_angle_dls(
                [{'rh_nm': p['rh'], 'weight': 1.0}], angles_deg=synth.DEFAULT_DLS_ANGLES,
                wavelength_nm=synth.DEFAULT_WAVELENGTH_NM,
                solvent_refractive_index=synth.DEFAULT_N_SOLVENT,
                temperature_K=synth.DEFAULT_TEMPERATURE_K,
                viscosity_Pa_s=synth.DEFAULT_VISCOSITY_PA_S, noise_level=0.005,
                label=f'{name} in {solvent_name}',
                concentration_g_per_mL=synth.DEFAULT_DLS_SINGLE_MG * 1e-3,
                seed=int(rng.integers(1 << 31)))   # per sample, like the SLS build above
            ids = self.inject_sls_set(
                sls, polymer_name=name, solvent_name=solvent_name,
                prefill_calibration=not prefilled, chain_type=chain)
            prefilled = True
            ids += self.inject_multi_angle_dls(
                dls, polymer_name=name, solvent_name=solvent_name, chain_type=chain)
            out[name] = ids
        return out

    def inject_form_factor_series(self, *, solvent_name: str = 'water',
                                  seed: Optional[int] = 23) -> Dict[str, List[str]]:
        """Add the form-factor SET panel to the workspace: each sample's SLS form-factor set +
        paired multi-angle DLS, with the hard-sphere scaling ladder tagged Linear. Prefills the
        session calibration from the first calibrated sample (so a form-factor/Zimm run gives an
        absolute Mw). The UNCALIBRATED sample is left OUT — calibration is session-global, so it
        can only show its uncalibrated behaviour loaded in its own session (the generated files +
        parameters.txt cover that). Returns {sample_name: [item_ids]}."""
        series = synth.build_form_factor_series()
        rng = np.random.default_rng(seed)
        out: Dict[str, List[str]] = {}
        prefilled = False
        for name, p in series.items():
            if not p['calibrated']:
                continue                          # uncalibrated variant: files only (see docstring)
            chain = 'linear' if p['role'] == 'linear' else None
            concs_mg = ([synth.DEFAULT_DLS_SINGLE_MG] if p['single_conc']
                        else [c for c in synth.DEFAULT_SLS_CONCS_MG if c > 0])
            sls = self.synth_sls_set(
                mw=p['mw'], rg_nm=p['rg'], a2_mol_mL_per_g2=p['a2'],
                angles_deg=synth.DEFAULT_SLS_ANGLES,
                concentrations_g_per_mL=[c * 1e-3 for c in concs_mg],
                wavelength_nm=synth.DEFAULT_WAVELENGTH_NM,
                temperature_K=synth.DEFAULT_TEMPERATURE_K, n_solvent=synth.DEFAULT_N_SOLVENT,
                dn_dc=synth.DEFAULT_DN_DC, calibrated=True,
                noise_level=0.005, seed=int(rng.integers(1 << 31)),
                polymer_name=name, solvent_name=solvent_name, label=f'{name} in {solvent_name}',
                kind=('single_concentration' if p['single_conc'] else 'zimm'),
                form_factor_model=p['model'], **p['params'])
            dls = self.synth_multi_angle_dls(
                [{'rh_nm': p['rh'], 'weight': 1.0}], angles_deg=synth.DEFAULT_DLS_ANGLES,
                wavelength_nm=synth.DEFAULT_WAVELENGTH_NM,
                solvent_refractive_index=synth.DEFAULT_N_SOLVENT,
                temperature_K=synth.DEFAULT_TEMPERATURE_K,
                viscosity_Pa_s=synth.DEFAULT_VISCOSITY_PA_S, noise_level=0.005,
                label=f'{name} in {solvent_name}',
                concentration_g_per_mL=synth.DEFAULT_DLS_SINGLE_MG * 1e-3,
                seed=int(rng.integers(1 << 31)))   # per sample, like the SLS build above
            ids = self.inject_sls_set(
                sls, polymer_name=name, solvent_name=solvent_name,
                prefill_calibration=not prefilled, chain_type=chain)
            prefilled = True
            ids += self.inject_multi_angle_dls(
                dls, polymer_name=name, solvent_name=solvent_name, chain_type=chain)
            out[name] = ids
        return out

    def _prefill_calibration_from_spec(self, cal) -> None:
        """Set the session calibration (working + committed) to a generator's
        calibrant point, so injected calibrated SLS data analyses on an absolute
        scale without the user re-entering the standard."""
        for state in (self.calibration_working, self.calibration_committed):
            state.calibrant_intensity = cal.calibrant_intensity
            state.calibrant_angle_deg = 90.0
            state.standard_geometry = cal.geometry
            state.standard_wavelength_nm = cal.wavelength_nm
            state.standard_temperature_C = cal.temperature_C
            state.standard_refractive_index = cal.n_standard
            # The generator's own record of its standard, not a library proposal at
            # this condition: 'user', so a wavelength edit leaves it as the set says.
            state.standard_refractive_index_source = 'user'
            state.dark_count_rate = 0.0
            state.k_c = cal.k_c()

    # ----------------------------------------------------------------------
    # Utilities: intensity-trace store + diagnostics
    # ----------------------------------------------------------------------
    def add_trace(self, times_s, count_rates_cps, sample_label=None,
                  measurement_id=None, source_path=None) -> str:
        """Register an intensity trace (already in canonical units: s, cps).

        Stored in the workspace's separate trace collection, not among the DLS/SLS
        measurements. `measurement_id` optionally links it to a DLS item. Returns
        the new trace id."""
        tid = self.workspace.new_item_id('t')
        lt = LoadedTrace(
            trace_id=tid,
            times_s=[float(x) for x in times_s],
            count_rates_cps=[float(x) for x in count_rates_cps],
            sample_label=sample_label, measurement_id=measurement_id,
            source_path=source_path)
        self.workspace.add_trace(lt)
        return tid

    def add_trace_from_preview(self, preview, source_path=None, *,
                               measurement_id: Optional[str] = None) -> str:
        """Build an IntensityTrace from a parser TraceFilePreview and store it.

        The preview must be ready (its units confirmed, for the generic parser);
        instrument parsers like the ALV one return ready previews.

        `measurement_id`, when given, overrides the preview's own back-reference.
        This is the confirmation step the trace parsers defer to: a loader mints
        the DLS item id first, then links that file's own trace to it.
        """
        trace = preview.build()
        return self.add_trace(
            trace.times_s, trace.count_rates_cps,
            sample_label=trace.sample_label,
            measurement_id=measurement_id or trace.measurement_id,
            source_path=source_path or getattr(preview, 'source_file', None))

    def traces(self) -> List[LoadedTrace]:
        return list(self.workspace.traces.values())

    def traces_for_measurement(self, item_id: str) -> List[LoadedTrace]:
        """Every stored trace linked to one DLS measurement."""
        return self.workspace.traces_for_measurement(item_id)

    def remove_trace(self, trace_id: str) -> None:
        """Delete a stored trace, invalidating anything derived from it first.

        Deleting a trace and detaching one are the same event seen from two
        directions, so removal routes through the detach cleanup rather than
        repeating it -- otherwise a mean count rate keeps a 'trace:*' tag naming a
        trace that no longer exists.
        """
        self.detach_trace(trace_id)
        self.workspace.remove_trace(trace_id)

    def attach_trace(self, trace_id: str, item_id: str, *,
                     autofill_mean: bool = True) -> Dict[str, Any]:
        """Link a stored trace to a DLS measurement.

        Attachment is bookkeeping the user can override, in the same spirit as
        sample grouping: it never changes a COMMITTED analysis result. The optional
        mean-count-rate autofill writes WORKING params only, and only into a cell
        that is empty or already carries a derived ('trace:*') tag -- so a value the
        user typed, or one the instrument file supplied, is left alone.

        A measurement carries AT MOST ONE trace (owner-ruled 2026-08-14): attaching
        a second detaches the incumbent, which keeps "the trace of this run" single-
        valued at the source instead of leaving every consumer to pick one and risk
        picking differently. The incumbent is detached, never deleted -- it is
        separately-loaded data that becomes standalone.

        Returns {'filled', 'value', 'source'}: whether the autofill wrote, and what.
        """
        if trace_id not in self.workspace.traces:
            raise KeyError(f'No such trace: {trace_id!r}')
        lm = self.workspace.measurements.get(item_id)
        # An unresolvable id IS the subject of its own refusal; a measurement of the
        # wrong kind is nameable, and is named (the `_require_dls_items` shape).
        if lm is None:
            raise ValueError(
                f'{item_id!r} is not a DLS measurement; a count-rate trace can '
                f'only be attached to one.')
        if lm.kind != 'dls':
            raise ValueError(
                f'{measurement_display_name(lm)!r} is not a DLS measurement; a '
                f'count-rate trace can only be attached to one.')
        # Detach through the same path a user-driven detach takes, so a value the
        # displaced trace produced is invalidated rather than left behind -- both
        # for this measurement's incumbent and for wherever this trace came from.
        self.detach_trace(trace_id)
        for other in self.workspace.traces_for_measurement(item_id):
            if other.trace_id != trace_id:
                self.detach_trace(other.trace_id)
        self.workspace.set_trace_link(trace_id, item_id)
        if not autofill_mean:
            return {'filled': False, 'value': None, 'source': None}
        value = self._trace_mean_cps(trace_id, 'trace:mean')
        filled = self._write_mean_count_rate(
            item_id, value, 'trace:mean', only_if_derived=True)
        return {'filled': filled,
                'value': value if filled else None,
                'source': 'trace:mean' if filled else None}

    def detach_trace(self, trace_id: str) -> None:
        """Clear a trace's link to its DLS measurement.

        A mean count rate DERIVED from that link is dropped from the WORKING params
        with it, so the cell stops offering a number whose basis has gone; a
        'vendor' or 'user' value is untouched.

        The COMMITTED value deliberately stays, tag and all (owner-ruled
        2026-08-14). Nothing in this program edits one committed field behind the
        user's back, and the tag is accurate history: that number WAS derived from a
        trace at the moment it was committed. The cleared working cell shows as a
        pending change, and the user's next Update carries the clearing through --
        the same path a cleared solvent cell takes.
        """
        trace = self.workspace.traces.get(trace_id)
        if trace is None:
            return
        item_id = trace.measurement_id
        self.workspace.set_trace_link(trace_id, None)
        if item_id is not None and not self.workspace.traces_for_measurement(item_id):
            self._write_mean_count_rate(item_id, None, None, only_if_derived=True)

    # -- mean count rate: offers, writer, divergence ------------------------
    # Precedence is user > trace > vendor (owner-ruled 2026-08-14): a value the
    # user typed is never overwritten; a linked trace's own mean is preferred to
    # the instrument file's scalar, which stays available as a labelled offer.
    _MEAN_CR_KEY = 'mean_count_rate_cps'
    _MEAN_CR_SOURCE_KEY = 'mean_count_rate_cps_source'

    @staticmethod
    def _mean_cr_family(tag: Optional[str]) -> str:
        """The source family of a mean-count-rate tag ('trace', 'vendor', 'generator',
        'user')."""
        return (tag or '').split(':')[0]

    def _trace_mean_cps(self, trace_id: str,
                        source: str = 'trace:mean') -> Optional[float]:
        """The linked trace's mean count rate by the method the tag names.

        'trace:mean' is the plain arithmetic mean of the recorded count rates;
        'trace:sigma_clipped' is the 3-sigma-clipped mean. Both come off one
        TraceStatistics object, so the two can never be computed differently.
        """
        trace = self.workspace.traces.get(trace_id)
        if trace is None or not trace.count_rates_cps:
            return None
        stats = trace_engine.compute_trace_statistics(trace.build())
        value = (stats.sigma_clipped_mean_cps
                 if source == 'trace:sigma_clipped' else stats.mean_cps)
        return float(value) if np.isfinite(value) else None

    def _write_mean_count_rate(self, item_id: str, value: Optional[float],
                               source: Optional[str], *,
                               only_if_derived: bool = False) -> bool:
        """Write a measurement's WORKING mean count rate and its source tag.

        `source=None` clears both. `only_if_derived` restricts the write to a cell
        that is empty or already carries a derived ('trace:*') tag -- the guard that
        makes a re-derivation incapable of overwriting a 'user' or 'vendor' value.

        A value that is not a usable rate -- non-finite, zero or negative
        (`core.data_models.usable_rate`, the one predicate every reader shares) -- is
        refused: a photon count rate can be none of these, and an all-zero trace's mean
        must not commit a placeholder the fit would then read as "no rate". Returns
        True if it wrote.
        """
        lm = self.workspace.measurements.get(item_id)
        if lm is None or lm.kind != 'dls':
            return False
        if only_if_derived:
            current = lm.working_params.get(self._MEAN_CR_KEY)
            tag = lm.working_params.get(self._MEAN_CR_SOURCE_KEY)
            if current is not None and self._mean_cr_family(tag) != 'trace':
                return False
        if source is None:
            lm.working_params.pop(self._MEAN_CR_KEY, None)
            lm.working_params.pop(self._MEAN_CR_SOURCE_KEY, None)
            return True
        value = usable_rate(value)
        if value is None:
            return False
        lm.working_params[self._MEAN_CR_KEY] = value
        lm.working_params[self._MEAN_CR_SOURCE_KEY] = source
        return True

    def mean_count_rate_offers(self, item_id: str) -> List[Dict[str, Any]]:
        """Every mean count rate the program can OFFER for this measurement.

        Each carries its origin. Nothing is written by this call: the value reaches
        an analysis only through the user's explicit commit (invariant 3). Ordered
        by precedence -- a linked trace's own mean first, then its outlier-clipped
        mean, then whatever the instrument file recorded.
        """
        lm = self.workspace.measurements.get(item_id)
        if lm is None or lm.kind != 'dls':
            return []
        offers: List[Dict[str, Any]] = []
        linked = self.workspace.traces_for_measurement(item_id)
        if linked:
            tid = linked[0].trace_id
            for source, label in (('trace:mean', 'linked trace, mean'),
                                  ('trace:sigma_clipped',
                                   'linked trace, outlier-clipped mean')):
                value = self._trace_mean_cps(tid, source)
                if value is not None:
                    offers.append({'source': source, 'value': value,
                                   'label': label})
        vendor = lm.working_params.get('vendor_mean_count_rate_cps')
        if vendor is not None:
            offers.append({'source': 'vendor', 'value': float(vendor),
                           'label': "the file's own mean count rate"})
        return offers

    def seed_mean_count_rate_at_load(self, item_id: str) -> bool:
        """Seed a freshly-loaded measurement's mean count rate from its own file.

        A load-time SEED, not a later edit: it writes the working AND committed
        tiers, exactly as `add_loaded` already does for every other value the parser
        proposes (angle, temperature, refractive index, viscosity), so a file that
        ships its own count-rate trace does not arrive looking like the user has
        unsaved changes. It refuses once anything is committed for the field, which
        is what keeps it a seed -- nothing else in this program edits one committed
        parameter behind the user's back.

        Safe in the invariant-3 sense, on the same footing as the other values a
        parser proposes: `add_loaded` already seeds the working AND committed tiers
        for angle, temperature, refractive index and viscosity, every one of which
        DOES reach the engine. What makes that safe is not inertness -- it is that
        the value is transparent, labelled with its origin, freely overridable, and
        never overwritten once the user types over it.

        The mean count rate is now the same kind of value, and its safety rests on
        the same footing -- the Data tab's row, where it is shown with its source
        tag, is editable, and once typed over is never overwritten -- not on any
        gesture between the commit and the analysis. It has two consumers: the
        Derived Intensity sub-tab (`preview_derived_static_series`, which reads
        `committed_params` and computes nothing until Derive is pressed) and every
        cumulant fit of the run, which reads the committed rate off the measurement
        `LoadedMeasurement.build()` constructs and records on the result whether
        it entered. Two earlier versions of this paragraph rested the argument on
        the field having no consumer, then on the Derive gesture; each was true only
        until the next consumer arrived, which is why it now rests on the row.

        Where nothing has written the working cell yet, the highest-precedence OFFER
        is adopted first, so a format that records its own mean count rate but ships
        no trace (Brookhaven .dat) arrives seeded exactly as one that ships a trace
        (ALV .ASC) does. `mean_count_rate_offers` is already ordered trace-then-
        vendor, so reading its head is the precedence rule rather than a second copy
        of it -- and an attached trace has written the cell by the time this runs, so
        that branch is untouched.
        """
        lm = self.workspace.measurements.get(item_id)
        if lm is None or lm.kind != 'dls':
            return False
        if lm.committed_params.get(self._MEAN_CR_KEY) is not None:
            return False
        if lm.working_params.get(self._MEAN_CR_KEY) is None:
            offers = self.mean_count_rate_offers(item_id)
            if offers:
                self._write_mean_count_rate(
                    item_id, offers[0]['value'], offers[0]['source'])
        value = lm.working_params.get(self._MEAN_CR_KEY)
        source = lm.working_params.get(self._MEAN_CR_SOURCE_KEY)
        if value is None or source is None:
            return False
        lm.committed_params[self._MEAN_CR_KEY] = value
        lm.committed_params[self._MEAN_CR_SOURCE_KEY] = source
        return True

    def apply_mean_count_rate_offer(self, item_id: str, source: str) -> bool:
        """Adopt one offered mean count rate into the WORKING params.

        A deliberate user action, so it MAY replace a value the user typed -- they
        asked for it. Returns False for an unknown or unavailable source.
        """
        for offer in self.mean_count_rate_offers(item_id):
            if offer['source'] == source:
                return self._write_mean_count_rate(
                    item_id, offer['value'], source)
        return False

    def mean_count_rate_status(self, item_id: str, *, rel_tol: float = 0.05,
                               use_working: bool = False) -> Dict[str, Any]:
        """What mean count rate this measurement carries, where it came from, and
        whether it disagrees with the linked trace.

        Reports on the COMMITTED value by default (what an analysis would use);
        `use_working=True` gives the live reading while the Data table is being
        edited. One computation serves the cell tooltip, the Data-tab note and the
        export comment, so the three can never disagree.

        Divergence is a DISCLOSURE, never a refusal and never a silent
        re-derivation. `rel_tol` is a documented parameter rather than a magic
        number: a vendor scalar and its own trace's mean agree to about 0.5 % on
        real ALV files (measured 2026-08-14 across three instrument families),
        while the failures worth catching are gross -- a kHz/cps unit slip is a
        factor of 1000, and a trace attached to the wrong angle is 30 % or more on
        a multi-angle set.

        The committed value is compared against the mean ITS OWN TAG NAMES, so a
        user who chose the outlier-clipped mean does not earn a permanent note for
        having chosen it.
        """
        out: Dict[str, Any] = {
            'value': None, 'source': None, 'trace_id': None,
            'trace_mean_cps': None, 'rel_diff': None,
            'diverges': False, 'note': None, 'usable': None,
        }
        lm = self.workspace.measurements.get(item_id)
        if lm is None or lm.kind != 'dls':
            return out
        params = lm.working_params if use_working else lm.committed_params
        value = params.get(self._MEAN_CR_KEY)
        source = params.get(self._MEAN_CR_SOURCE_KEY)
        out['value'] = value
        out['source'] = source
        # The same predicate `build()` applies: a committed value it maps to None (a
        # legacy 0.0, a typed 0) is one the fit never sees, so the status says so instead
        # of comparing it against a trace as if the fit used it.
        out['usable'] = (usable_rate(value) is not None) if value is not None else None
        if out['usable'] is False:
            out['note'] = (
                'The committed mean count rate is not a usable rate (a positive, finite '
                'number is needed), so the nonlinear cumulant runs in the classical '
                'limit, as if none were committed. Enter a rate or clear the cell.')
            return out

        linked = self.workspace.traces_for_measurement(item_id)
        if not linked:
            return out
        tid = linked[0].trace_id
        out['trace_id'] = tid
        method = (source if self._mean_cr_family(source) == 'trace'
                  else 'trace:mean')
        trace_mean = self._trace_mean_cps(tid, method)
        out['trace_mean_cps'] = trace_mean
        if (value is None or trace_mean is None
                or not np.isfinite(value) or not np.isfinite(trace_mean)):
            return out
        # Guard the RESULT, not the operands: a zero trace mean makes the relative
        # difference meaningless, so fall back to "any positive value disagrees".
        if trace_mean == 0.0:
            out['diverges'] = value > 0.0
        else:
            rel = (value - trace_mean) / trace_mean
            out['rel_diff'] = float(rel)
            out['diverges'] = abs(rel) > rel_tol
        if out['diverges']:
            out['note'] = (
                'The committed mean count rate differs from the linked '
                'count-rate trace. The value is used exactly as entered.')
        return out

    # -- derived static points: a DLS run's mean count rate as an SLS intensity --
    # The engine half is analysis/derived_sls.py; this half reads the workspace,
    # groups by concentration, and materializes the result. Nothing here recomputes a mean
    # count rate: it reads the value the user committed, so the Data tab stays the
    # one place a source is chosen (invariant 3).
    DERIVED_SLS_KIND = 'trace_derived_sls'

    def derived_intensity_candidates(
            self, sample_id: str,
            trace_params: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Every DLS run in this sample, and every UNATTACHED count-rate trace, with
        whether each can supply a static point.

        Reports rather than filters: an ineligible run is listed with the reason,
        so the surface can show the user what to fix instead of silently offering
        a shorter list (doctrine rule 2 -- disclosure, at the display surface).
        Nothing is written.

        Each row carries `kind`: 'dls' for a run of the sample, 'trace' for a
        standalone trace (`measurement_id` None). A solvent blank is often measured
        WITHOUT a correlogram -- there is nothing to fit, the run exists to measure
        the solvent's intensity -- so such a blank reaches the derived route only
        through the trace it left behind. A trace carries no identity or angle of
        its own, so a trace row takes the sample's identity from the first ticked
        DLS run at derive time (`preview_derived_static_series`), its concentration
        defaults to 0 (a blank), and its angle is SEEDED when the sample's DLS runs
        share one angle -- otherwise it must be entered. `trace_params` maps a
        trace id to the user's own `angle_deg` / `concentration_g_per_mL` for that
        row; it overrides the seeds and is what the surface hands back.
        """
        sample = self.workspace.samples.get(sample_id)
        if sample is None:
            return []
        rows: List[Dict[str, Any]] = []
        dls_angles = set()
        n_template = 0
        for item_id in sample.dls_item_ids:
            lm = self.workspace.measurements.get(item_id)
            if lm is None:
                continue
            n_template += 1
            p = lm.committed_params
            rate = p.get(self._MEAN_CR_KEY)
            conc = p.get('concentration_g_per_mL')
            if p.get('angle_deg') is not None:
                dls_angles.add(round_angle_for_identity(p['angle_deg']))
            # A CODE, not a sentence: the wording is the display surface's, so the
            # controller stays free of user-facing prose and the tooltip text can be
            # resolved statically (tests/test_method_docs_sync.py).
            reason = None
            if lm.derived_kind == 'replicate_average':
                # First: it copies its first source's rate, so it would otherwise pass
                # as eligible (combination doctrine rule 1, core/data_models.py).
                reason = 'replicate_average'
            elif rate is None:
                reason = 'no_count_rate'
            elif usable_rate(rate) is None:
                reason = 'bad_count_rate'
            elif conc is None:
                reason = 'no_concentration'
            rows.append({
                'item_id': item_id,
                'kind': 'dls',
                # The BARE name, not the display format: this row's own Angle and
                # c columns already carry the axes (item 1f), and the hover card
                # carries the rest.
                'label': measurement_name(lm),
                'angle_deg': p.get('angle_deg'),
                'concentration_g_per_mL': conc,
                'mean_count_rate_cps': rate,
                'source': p.get(self._MEAN_CR_SOURCE_KEY),
                'analyzer_geometry': p.get('analyzer_geometry'),
                'eligible': reason is None,
                'reason_code': reason,
            })
        seed_angle = next(iter(dls_angles)) if len(dls_angles) == 1 else None
        overrides = trace_params or {}
        for trace in self.workspace.traces.values():
            if trace.measurement_id is not None:
                continue                      # attached: its run is listed above
            given = overrides.get(trace.trace_id, {})
            angle = given.get('angle_deg', seed_angle)
            conc = given.get('concentration_g_per_mL', 0.0)
            rate = self._trace_mean_cps(trace.trace_id, 'trace:mean')
            reason = None
            if n_template == 0:
                reason = 'no_template_run'
            elif usable_rate(rate) is None:
                reason = 'bad_count_rate'
            elif angle is None:
                reason = 'no_angle'
            # The data model's own range predicates (a scattering angle lies strictly
            # inside (0, 180); a concentration is finite and non-negative), applied on
            # the row so a typed value is refused where it was typed, not at Derive.
            elif not (np.isfinite(angle) and 0.0 < angle < 180.0):
                reason = 'bad_angle'
            elif conc is None:
                reason = 'no_concentration'
            elif not (np.isfinite(conc) and conc >= 0.0):
                reason = 'bad_concentration'
            rows.append({
                'item_id': trace.trace_id,
                'kind': 'trace',
                'label': trace.sample_label or trace.trace_id,
                'angle_deg': angle,
                'concentration_g_per_mL': conc,
                'mean_count_rate_cps': rate,
                'source': 'trace:mean',
                'analyzer_geometry': None,
                'eligible': reason is None,
                'reason_code': reason,
            })
        rows.sort(key=lambda r: (r['concentration_g_per_mL'] is None,
                                 r['concentration_g_per_mL'] or 0.0,
                                 r['angle_deg'] is None, r['angle_deg'] or 0.0))
        return rows

    def preview_derived_static_series(
            self, item_ids: Sequence[str], *,
            trace_params: Optional[Dict[str, Dict[str, Any]]] = None,
            template_sample_id: Optional[str] = None) -> List[Any]:
        """The static series these runs WOULD produce, one per concentration.

        Pure with respect to the workspace -- it creates nothing. The surface uses
        it to show the points and any disclosure before the user commits them.
        Raises ValueError through the engine on a category error (mixed
        polarization, mixed concentration within a group, an unusable rate).

        An id may name an UNATTACHED trace (see `derived_intensity_candidates`).
        Such an entry has no measurement of its own, so it is built as a snapshot
        of a TEMPLATE run's identity -- polymer, solvent, temperature, wavelength,
        refractive index, geometry and fraction -- with the concentration and angle
        the user gave for it (`trace_params`; concentration 0 and the sample's
        one shared angle when not given). The template is the first DLS run in the
        selection, else the first DLS run of `template_sample_id`: the user's tick
        picks the fraction the blank belongs to, exactly as it picks the ladder.
        The engine's own disclosure then compares the trace entry against the runs
        it is collated with like any other member.
        """
        by_conc: Dict[Any, List[derived_sls.DerivedIntensityInput]] = {}
        built: List[Any] = []
        overrides = trace_params or {}
        template = self._derived_template_run(item_ids, template_sample_id)
        for item_id in item_ids:
            trace = self.workspace.traces.get(item_id)
            if trace is not None and trace.measurement_id is None:
                entry = self._trace_derived_entry(
                    trace, template, overrides.get(item_id, {}))
                built.append(entry.measurement)
                by_conc.setdefault(
                    round_concentration_for_identity(
                        entry.measurement.concentration_g_per_mL), []).append(entry)
                continue
            lm = self.workspace.measurements.get(item_id)
            if lm is None:
                # An unresolvable id IS the subject of this answer: a lookup failed on
                # an id the user never chose, so there is nothing to name it by.
                raise ValueError(
                    '{0!r} is not a DLS measurement, so it has no count rate to '
                    'derive a static point from.'.format(item_id))
            name = measurement_display_name(lm)
            if lm.kind != 'dls':
                raise ValueError(
                    '{0!r} is not a DLS measurement, so it has no count rate to '
                    'derive a static point from.'.format(name))
            _refuse_an_average_as_a_repeat([lm])
            p = lm.committed_params
            rate = p.get(self._MEAN_CR_KEY)
            if rate is None:
                raise ValueError(
                    'Measurement {0!r} has no committed mean count rate. Set one on '
                    'the Data tab, or leave it out.'.format(name))
            conc = p.get('concentration_g_per_mL')
            if conc is None:
                raise ValueError(
                    'Measurement {0!r} has no committed concentration. A static '
                    'measurement needs one; set it on the Data tab.'.format(name))
            key = round_concentration_for_identity(conc)
            measurement = lm.build()
            built.append(measurement)
            by_conc.setdefault(key, []).append(
                derived_sls.DerivedIntensityInput(
                    item_id=item_id,
                    measurement=measurement,
                    # The committed value as-is: the engine applies the one usable-rate
                    # predicate and refuses what is not a rate, naming the run.
                    mean_count_rate_cps=rate,
                    source=p.get(self._MEAN_CR_SOURCE_KEY) or 'user',
                    # The engine prints names it is GIVEN (invariant 1): its
                    # count-rate-disagreement note reaches the exported Comments cell,
                    # where an internal id resolves to nothing.
                    display_name=name,
                ))
        # The partition is checked across the WHOLE selection, before grouping.
        # Grouping by concentration makes every group internally uniform, so the
        # per-series check inside the engine can never see a selection mixing VV at
        # one concentration with VH at another -- which would derive into two
        # measurements that then pool into one Zimm. Rule 1's structural half.
        derived_sls.require_one_polarization_state(built)
        return [derived_sls.derive_static_series(by_conc[k])
                for k in sorted(by_conc)]

    def _derived_template_run(self, item_ids: Sequence[str],
                              template_sample_id: Optional[str]):
        """The DLS run an unattached trace takes its identity from, or None.

        The first DLS measurement in the selection wins -- the ticked ladder names
        the fraction and the optics a trace-only blank belongs to. With no DLS run
        ticked, the sample's first DLS run stands in, so a blank can be derived on
        its own for a ladder that already exists as loaded SLS data.
        """
        for item_id in item_ids:
            lm = self.workspace.measurements.get(item_id)
            if lm is not None and lm.kind == 'dls':
                return lm
        sample = (self.workspace.samples.get(template_sample_id)
                  if template_sample_id else None)
        if sample is not None:
            for item_id in sample.dls_item_ids:
                lm = self.workspace.measurements.get(item_id)
                if lm is not None:
                    return lm
        return None

    def _trace_derived_entry(self, trace, template, given: Dict[str, Any]):
        """One `DerivedIntensityInput` for an unattached trace.

        The measurement is a snapshot of the template run's identity with the
        trace's own concentration and angle substituted; its correlogram arrays
        are the template's and are never read by the derived route, which
        collates count rates. Everything the point will carry -- polymer, solvent,
        temperature, optics, geometry, fraction -- is therefore the user's own
        committed parameters on a run they chose (invariant 3), and the source
        tag says the number is the trace's plain mean.
        """
        if template is None:
            raise ValueError(
                'Trace {0!r} cannot supply a static point on its own: an unattached '
                'trace takes its sample identity (polymer, solvent, temperature, '
                'optics) from a DLS run, and this sample has none.'.format(
                    trace.sample_label or trace.trace_id))
        seed = self._trace_seed_angle(template)
        angle = given.get('angle_deg', seed)
        if angle is None:
            raise ValueError(
                'Trace {0!r} has no scattering angle: the sample\'s runs span '
                'several angles, so enter the angle this trace was recorded at.'
                .format(trace.sample_label or trace.trace_id))
        conc = given.get('concentration_g_per_mL', 0.0)
        if conc is None:
            raise ValueError(
                'Trace {0!r} has no concentration; a static measurement needs one '
                '(0 for a solvent blank).'.format(
                    trace.sample_label or trace.trace_id))
        rate = self._trace_mean_cps(trace.trace_id, 'trace:mean')
        if rate is None:
            raise ValueError(
                'Trace {0!r} has no usable mean count rate.'.format(
                    trace.sample_label or trace.trace_id))
        measurement = replace(
            template.build(), concentration_g_per_mL=float(conc),
            angle_deg=float(angle))
        return derived_sls.DerivedIntensityInput(
            item_id=trace.trace_id, measurement=measurement,
            mean_count_rate_cps=float(rate), source='trace:mean',
            # A trace is not a measurement, so it keeps the trace name every other
            # trace surface shows -- the engine's note just stops printing a raw id.
            display_name=trace.sample_label or trace.trace_id)

    def _trace_seed_angle(self, template) -> Optional[float]:
        """The one angle the template's sample shares, else None (must be entered)."""
        sid = self._sample_id_of(template.item_id)
        sample = self.workspace.samples.get(sid) if sid else None
        angles = set()
        for item_id in (sample.dls_item_ids if sample else ()):
            lm = self.workspace.measurements.get(item_id)
            if lm is not None and lm.committed_params.get('angle_deg') is not None:
                angles.add(round_angle_for_identity(lm.committed_params['angle_deg']))
        return next(iter(angles)) if len(angles) == 1 else None

    def derive_static_points(self, item_ids: Sequence[str], *,
                             dn_dc_mL_per_g: Optional[float] = None,
                             trace_params: Optional[Dict[str, Dict[str, Any]]] = None,
                             template_sample_id: Optional[str] = None,
                             ) -> Dict[str, Any]:
        """Materialize one SLS measurement per concentration from these DLS runs.

        Returns {'item_ids', 'series', 'notes'}.

        The derived point is a SNAPSHOT with provenance (owner-ruled 2026-08-14),
        the replicate-average precedent: it records the runs and the mean-count-rate
        sources it was built from and then stops tracking them. Re-running a fit,
        editing a mask or re-committing a source run does NOT touch it, and
        re-deriving is a deliberate user action. A source run that is later deleted
        leaves a dangling id, never a cascade delete.

        `dn_dc_mL_per_g` is optional and never guessed (invariant 3). Absent, the
        point carries no dn/dc and the existing arbitrary-optical-scale disclosure
        travels with every result computed from it. `trace_params` and
        `template_sample_id` are `preview_derived_static_series`'s, for the
        unattached-trace entries.
        """
        series_list = self.preview_derived_static_series(
            item_ids, trace_params=trace_params,
            template_sample_id=template_sample_id)
        # Deriving again REPLACES the previous derived point at each concentration
        # rather than adding a second one: the tab's help promises a refresh, and two
        # points at one concentration would sit in the same Zimm with no way for the
        # (concentration, angle) mask to tell them apart -- the stale intensities
        # included. Only DERIVED points are superseded; a loaded measurement at the
        # same concentration is the user's data and is never touched.
        superseded = self._superseded_derived_points(series_list)
        new_ids: List[str] = []
        notes: List[str] = []
        for series in series_list:
            source_ids = list(series.source_item_ids)
            # Sources named by KIND, the sibling of `derived_intensity_provenance`'s
            # sentence: a trace-only blank has no DLS run behind it.
            n_tr = sum(1 for i in source_ids if i in self.workspace.traces)
            n_run = len(source_ids) - n_tr
            what = ('{0} DLS run(s)'.format(n_run) if n_tr == 0
                    else '{0} count-rate trace(s)'.format(n_tr) if n_run == 0
                    else '{0} DLS run(s) and {1} count-rate trace(s)'.format(n_run, n_tr))
            params: Dict[str, Any] = {
                'label': 'Derived from {0}'.format(what),
                'polymer_name': series.polymer_name,
                'solvent_name': series.solvent_name,
                'concentration_g_per_mL': series.concentration_g_per_mL,
                'temperature_K': series.temperature_K,
                'wavelength_nm': series.wavelength_nm,
                'solvent_refractive_index': series.solvent_refractive_index,
                'dn_dc_mL_per_g': dn_dc_mL_per_g,
                'mw_fraction': series.mw_fraction,
                # A refusal to MIX implies an obligation to record which side won:
                # the engine raises on a mixed polarization state, so the object it
                # produces has to say which state that was.
                'analyzer_geometry': series.analyzer_geometry,
                'derived_intensity_source': ', '.join(series.source_tags),
                'derived_intensity_notes': ' '.join(series.notes),
            }
            raw = {
                'angles_deg': [float(a) for a in series.angles_deg],
                'intensities': [float(i) for i in series.intensities],
            }
            new_id = self.workspace.new_item_id()
            lm = LoadedMeasurement(
                item_id=new_id, kind='sls', raw=raw,
                working_params=dict(params), committed_params=dict(params),
                source_path=None,
                derived_from=source_ids,
                derived_kind=self.DERIVED_SLS_KIND,
            )
            self.workspace.add_measurement(lm)
            new_ids.append(new_id)
            notes.extend(series.notes)
        if superseded:
            # After the new points exist, so the sample never momentarily loses its
            # only member and gets purged out from under its stored results.
            self.remove_measurements(superseded)
            notes.append(
                '{0} earlier derived point(s) at the same concentration were '
                'replaced.'.format(len(superseded)))
        # One regroup after the whole batch: a Zimm ladder derived in one gesture
        # should land as one grouping event, not N.
        self._rekey_sample_stores(self.workspace.regroup())
        return {'item_ids': new_ids, 'series': series_list,
                'notes': notes, 'superseded': superseded}

    @staticmethod
    def _join_derived_provenance_parts(sample_note: Optional[str],
                                       blank_note: Optional[str]) -> str:
        """One provenance string for a series and the blank subtracted from it.

        Either side may be derived independently, so both are stated and neither
        ranks the other -- the same join the export's Comments cells use for two
        facts that can hold at once.
        """
        parts = []
        if sample_note:
            parts.append(str(sample_note))
        if blank_note:
            parts.append('solvent blank: {0}'.format(blank_note))
        return '; '.join(parts)

    def _superseded_derived_points(self, series_list) -> List[str]:
        """Existing DERIVED points these series would duplicate.

        Matched on the same identity a sample is grouped by, PLUS the two axes the
        sample key deliberately leaves out and the fits filter on -- the
        molecular-weight fraction and the concentration -- so re-deriving a ladder
        replaces it entry for entry, and a derived point of another fraction, sample
        or concentration is left alone. The fraction is compared EXACTLY (None and a
        label differ), the same comparison `_rayleigh_for_sample` selects a series
        by and `check_replicate_identity` averages by. Without it every fraction's
        blank collides at c = 0: deriving one fraction's blank deleted another's
        (measured on a five-fraction molar-mass series; three fractions bitten).
        Loaded measurements are never returned: the user's own data is not ours to
        supersede.
        """
        wanted = {(casefold_identity_name(sr.polymer_name),
                   casefold_identity_name(sr.solvent_name),
                   round_temperature_for_identity(sr.temperature_K),
                   sr.mw_fraction,
                   round_concentration_for_identity(sr.concentration_g_per_mL))
                  for sr in series_list
                  if sr.concentration_g_per_mL is not None}
        out: List[str] = []
        for item_id, lm in self.workspace.measurements.items():
            if lm.kind != 'sls' or lm.derived_kind != self.DERIVED_SLS_KIND:
                continue
            p = lm.committed_params
            conc = p.get('concentration_g_per_mL')
            if conc is None:
                continue
            key = (casefold_identity_name(p.get('polymer_name') or ''),
                   casefold_identity_name(p.get('solvent_name') or ''),
                   round_temperature_for_identity(p.get('temperature_K') or 0.0),
                   p.get('mw_fraction'),
                   round_concentration_for_identity(conc))
            if key in wanted:
                out.append(item_id)
        return out

    def derived_intensity_provenance(self, item_id: str) -> Optional[str]:
        """The one-line provenance for a derived static point, or None.

        One computation serves the tab's display and the export Comments cell, so
        the two can never disagree -- the `mean_count_rate_status` pattern.
        """
        lm = self.workspace.measurements.get(item_id)
        if lm is None or lm.derived_kind != self.DERIVED_SLS_KIND:
            return None
        sources = list(lm.derived_from or ())
        n_traces = sum(1 for i in sources if i in self.workspace.traces)
        n_runs = len(sources) - n_traces
        source = lm.committed_params.get('derived_intensity_source') or 'unknown'
        # Sources named by KIND: a trace-only blank has no DLS run behind it, and
        # saying it did misstates what was measured. The run-only wording is
        # unchanged -- the exports pin it.
        if n_traces == 0:
            what = '{0} DLS run(s)'.format(n_runs)
        elif n_runs == 0:
            what = '{0} count-rate trace(s)'.format(n_traces)
        else:
            what = '{0} DLS run(s) and {1} count-rate trace(s)'.format(n_runs, n_traces)
        parts = ['derived from {0}; mean count rate source: {1}'.format(what, source)]
        # A source may be a run OR an unattached trace (a trace-only blank), so
        # both stores are consulted before a source is called removed.
        missing = [i for i in (lm.derived_from or ())
                   if i not in self.workspace.measurements
                   and i not in self.workspace.traces]
        if missing:
            parts.append('{0} source run(s) removed'.format(len(missing)))
        # The engine's rule-2 disclosures ride the SAME path as the provenance, so a
        # first-wins resolution of a disagreeing wavelength or refractive index --
        # both of which enter q, and so move Rg -- cannot be a fact the export omits.
        notes = lm.committed_params.get('derived_intensity_notes') or ''
        if notes.strip():
            parts.append(notes.strip())
        return '; '.join(parts)

    def remove_measurements(self, item_ids) -> None:
        """Remove one or more measurements from the workspace and purge every bit
        of state keyed to them: their cached results, and -- for any sample that
        becomes empty as a result -- that sample's results, SLS mask, and
        per-sample calibration override. Safe to call with item_ids that no longer
        exist.

        A SURVIVING sample's durable AGGREGATE Rh rows -- the replicate average,
        Γ-vs-q² and D-vs-c of the fraction that lost a run -- and the stored runs
        behind them are DROPPED, together with a 'computed' sample Rh promoted from
        one of them. Before this only the mode-map branch noticed a removal: a
        replicate average went on reading "35.6 nm ± 6.1 (5 runs)" after the 60 nm
        outlier it was built from was deleted, kept its ±, and stayed the top
        tier-1 candidate. Dropped rather than re-fitted, unlike the mode map: an
        average or a Γ-q² fit is the USER'S selection of runs (the mode map re-fits
        because its picks -- user state -- would otherwise dangle), and the
        promotion rule reserves a new promotion for a deliberate Run. Every row of
        the (sample, fraction) goes, whether or not the removed run was in its
        selection: the selection is not recorded on the row (a reloaded session has
        no stamps at all), so over-inclusion is the safe direction -- one re-run
        restores a row that did not need it, while a stale mean feeds ρ, the
        scaling plots and the Summary export. The clear is keyed to the artifact
        (`_clear_replicate_rh`'s predicate): a hand-entered or picked Rh is never
        touched, nor a 'computed' one written by another path."""
        ids = {iid for iid in item_ids if iid in self.workspace.measurements}
        if not ids:
            return
        samples_before = set(self.workspace.samples.keys())
        # (sample, fraction) of every DLS run being removed, read BEFORE the removal
        # regroups the workspace -- the aggregate-row sweep below needs them.
        dls_owners = {
            (self._sample_id_of(iid),
             self.workspace.measurements[iid].committed_params.get('mw_fraction'))
            for iid in ids if self.workspace.measurements[iid].kind == 'dls'}
        # The SLS sibling, read at the same moment and for the same reason. Every SLS fit
        # consumes the WHOLE series (and its c = 0 blank), so a sample that loses any SLS
        # member has had the inputs behind its stored Mw/Rg/A2 changed underneath it.
        sls_owners = {
            (self._sample_id_of(iid),
             self.workspace.measurements[iid].committed_params.get('mw_fraction'))
            for iid in ids if self.workspace.measurements[iid].kind == 'sls'}
        for iid in ids:
            self.workspace.remove_measurement(iid)
        vanished = samples_before - set(self.workspace.samples.keys())
        # A results key is ('kind', owner_id, ...); owner_id is an item_id for
        # per-measurement results (DLS) or a sample_id for per-sample results (SLS).
        dead = ids | vanished
        self.results = {k: v for k, v in self.results.items()
                        if not (len(k) > 1 and k[1] in dead)}
        # Keep the form-factor signature stamps in step with self.results (same owner-id filter).
        self._form_factor_run_sig = {k: v for k, v in self._form_factor_run_sig.items()
                                     if not (len(k) > 1 and k[1] in dead)}
        self._dls_run_sig = {k: v for k, v in self._dls_run_sig.items()
                             if not (len(k) > 1 and k[1] in dead)}
        # The SLS candidate memo is keyed (kind, sample_id, fraction, sig); drop any
        # entry for a vanished sample (stale entries would otherwise just miss, but this
        # keeps it bounded).
        self._sls_candidate_cache = {
            k: v for k, v in self._sls_candidate_cache.items() if k[1] not in dead}
        # The DLS candidate memo is keyed (sample_id, fraction, sig); drop any entry
        # for a vanished sample. (A surviving sample that merely lost a measurement is
        # invalidated by the signature — its committed_signature no longer includes
        # the gone id — so it would miss anyway; this is the bounded-ness sweep.)
        self._dls_candidate_cache = {
            k: v for k, v in self._dls_candidate_cache.items() if k[0] not in dead}
        # Drop version stamps for distribution results just pruned above (their keys
        # are ('distribution', item_id, method); item_id in dead) — keeps it bounded.
        self._distribution_stamps = {
            k: v for k, v in self._distribution_stamps.items() if k[1] not in dead}
        if vanished:
            # sls_masks and mode_picks are keyed by (sample_id, fraction); drop
            # every fraction of a vanished sample. Picks on a SURVIVING sample
            # are untouched (value-keyed; the map prune below is what makes
            # them match nothing).
            self.sls_masks = {k: v for k, v in self.sls_masks.items()
                              if k[0] not in vanished}
            self.mode_picks = {k: v for k, v in self.mode_picks.items()
                               if k[0] not in vanished}
            # A vanished sample's recorded mode sets go with it. A SURVIVING
            # sample's sets are untouched even when a source run named in a
            # point's item_id was just deleted -- the snapshot ruling: the set
            # DANGLES its removed sources ('N source run(s) removed' in
            # mode_set_provenance), never cascade-deletes.
            self.mode_sets = {k: v for k, v in self.mode_sets.items()
                              if k[0] not in vanished}
            # ...and the acknowledgement goes with the sample, for the reason above.
            self._unmeasured_blank_ack -= vanished
        # A SURVIVING sample's cached mode map must not keep a removed run's
        # points: the value-keyed picks still match cached points, so a stale
        # map lets a fit read (or KeyError on) deleted measurements. Prune the points, note the drop,
        # and re-resolve the fits so no family silently rests on removed data.
        for key in [k for k in self.results
                    if isinstance(k, tuple) and len(k) == 3
                    and k[0] == 'mode_map']:
            res = self.results[key]
            keep = tuple(p for p in res.points if p.item_id not in ids)
            if len(keep) != len(res.points):
                n_dropped = len(res.points) - len(keep)
                self.results[key] = dls_engine.ModeMapRun(
                    points=keep, source_method=res.source_method,
                    notes=res.notes + (
                        f'{n_dropped} mode(s) from removed measurement(s) '
                        f'were dropped from the map.',))
                self.run_mode_family_fits(key[1], key[2])
        # A SURVIVING sample's aggregate Rh rows of the fraction that lost a run are
        # dropped with their stored runs and an equal 'computed' promotion -- see the
        # docstring for why dropped, not re-fitted, and why every row of the fraction.
        for sid, frac in dls_owners:
            if sid is None or sid not in self.workspace.samples:
                continue
            kinds = ['replicate_avg', 'gamma_q2', 'conc_extrap']
            if ('mode_map', sid, frac) not in self.results:
                # The mode-family row is spared above only because the cached map is
                # re-fitted there; a reloaded session HAS no cached map (results are
                # not persisted, the rows are), so nothing would re-derive it and it
                # is dropped with its siblings rather than left describing a run that
                # is gone.
                kinds.append('mode_family')
            rows = [r for r in self.workspace.sample_rh_rows.values()
                    if r.sample_id == sid and r.fraction == frac
                    and r.source_kind in kinds]
            for row in rows:
                if row.source_kind == 'replicate_avg':
                    self._clear_replicate_rh(sid, frac, str(row.source_set).split('|')[0])
                    continue
                key = (row.source_kind, sid, frac)
                self.results.pop(key, None)
                self._dls_run_sig.pop(key, None)
                self.workspace.remove_sample_rh_row(sid, row.source_kind, row.source_set)
                r = self.workspace.samples[sid].result_for(frac)
                if r.rh_source == 'computed' and r.rh_nm == row.rh_nm:
                    reset_result_quantity(r, 'rh')
        # The SLS triple, mirroring the aggregate-Rh sweep above. Removing an SLS member
        # invalidates every fit built on that series -- most sharply the c = 0 blank,
        # whose departure changes every excess Rayleigh ratio -- yet the stored Mw/Rg/A2
        # survived, so Cross-Sample kept serving a value under a label the loaded data
        # could no longer produce. Over-inclusion is the safe direction, the same call
        # the Rh sweep documents: one re-Run restores a value that did not need clearing,
        # while a stale Mw feeds the scaling plots, rho and the Summary export. Keyed to
        # the tier, so a hand-entered or picked value is never touched.
        for sid, frac in sls_owners:
            if sid is None or sid not in self.workspace.samples:
                continue                      # a vanished sample is handled wholesale above
            r = self.workspace.samples[sid].result_for(frac)
            for q in ('mw', 'rg', 'a2'):
                if getattr(r, f'{q}_source', '') == 'computed':
                    reset_result_quantity(r, q)
        for sid in vanished:
            self.sample_calibration_working.pop(sid, None)
            self.sample_calibration_committed.pop(sid, None)

    def run_trace_statistics(self, trace_id: str,
                             baseline_method: Optional[str] = None,
                             baseline_parameter: Optional[float] = None):
        """Summary statistics (mean, CV, baseline, ...) for a stored trace.
        Baseline method/parameter seed from settings when not given."""
        trace = self.workspace.traces[trace_id].build()
        return trace_engine.compute_trace_statistics(
            trace,
            baseline_method=(baseline_method if baseline_method is not None
                             else _TRACE_BASELINE_METHOD),
            baseline_parameter=(baseline_parameter if baseline_parameter is not None
                                else _TRACE_BASELINE_PARAMETER))

    def build_trace(self, trace_id: str):
        """The IntensityTrace for a stored trace id (for plotting)."""
        return self.workspace.traces[trace_id].build()

    def trace_point_count(self, trace_id: str) -> int:
        """Number of samples in a stored trace (O(1); no array build). The GUI uses
        it to decide whether a trace is long enough to compute diagnostics off the
        UI thread."""
        return len(self.workspace.traces[trace_id].count_rates_cps)

    # --- richer trace diagnostics (thin engine wrappers; seed from settings) ---
    def flag_trace_outliers(self, trace_id: str, k: Optional[float] = None):
        """Points outside median ± k·(robust σ), the robust outlier band."""
        if k is None:
            k = _TRACE_OUTLIER_K
        return trace_engine.flag_outliers(self.build_trace(trace_id), k=k)

    def flag_trace_outliers_windowed(self, trace_id: str, k: Optional[float] = None,
                                     window_pts: Optional[int] = None):
        """The moving-window companion: rolling baseline, impulse flags, abstentions.

        Shares k with `flag_trace_outliers` -- one control drives both bands.
        `window_pts=None` lets the engine resolve the window from the trace
        length.
        """
        if k is None:
            k = _TRACE_OUTLIER_K
        return trace_engine.flag_outliers_windowed(
            self.build_trace(trace_id), k=k, window_pts=window_pts)

    def trace_running_average(self, trace_id: str, window_s=None,
                              window_points=None):
        return trace_engine.running_average(self.build_trace(trace_id),
                                            window_s=window_s,
                                            window_points=window_points)

    def trace_block_variance(self, trace_id: str, **kw):
        kw.setdefault('correlation_threshold', _TRACE_BLOCKVAR_THRESHOLD)
        return trace_engine.block_variance(self.build_trace(trace_id), **kw)

    def trace_histogram(self, trace_id: str, distribution: str = 'both', **kw):
        return trace_engine.fit_count_rate_histogram(
            self.build_trace(trace_id), distribution=distribution, **kw)

    def trace_welch_psd(self, trace_id: str, **kw):
        return trace_engine.welch_psd(self.build_trace(trace_id), **kw)

    def trace_periodogram(self, trace_id: str, **kw):
        """Periodogram + false-alarm test for one stored trace.

        The blocking analysis's correlation verdict is supplied here rather than
        inside the engine, which keeps `periodogram_false_alarm` pure. It is what
        marks the verdict unreliable: Scargle's null is white noise, and a
        correlated trace breaks it in the direction that manufactures
        significant-looking low-frequency peaks.
        """
        trace = self.build_trace(trace_id)
        kw.setdefault(
            'correlations_detected',
            trace_engine.block_variance(
                trace,
                correlation_threshold=_TRACE_BLOCKVAR_THRESHOLD,
            ).correlations_detected)
        return trace_engine.periodogram_false_alarm(trace, **kw)

    def trace_stationarity(self, trace_id: str, significance: Optional[float] = None):
        """ADF stationarity test (needs statsmodels; raises if unavailable)."""
        if significance is None:
            significance = _TRACE_ADF_SIGNIFICANCE
        return trace_engine.test_stationarity_adf(
            self.build_trace(trace_id), significance=significance)

    # ----------------------------------------------------------------------
    # Session save / load (self-contained JSON + source paths)
    # ----------------------------------------------------------------------
    def save_session(self, file_path: str) -> str:
        payload = self.workspace.to_dict()
        payload['calibration'] = self.calibration_committed.to_dict()
        payload['sample_calibrations'] = {
            sid: cc.to_dict()
            for sid, cc in self.sample_calibration_committed.items()}
        # The WORKING copies too: every measurement's
        # working_params is serialized, so a half-typed parameter edit comes back with
        # its pending highlight, while a calibrant intensity typed and not yet Updated
        # -- or a per-sample override just enabled -- was gone with no notice. The
        # calibration is the field that sets the absolute Rayleigh scale behind every
        # Mw; pending state is pending state. Additive keys, absent-tolerant on load.
        payload['calibration_working'] = self.calibration_working.to_dict()
        payload['sample_calibrations_working'] = {
            sid: cw.to_dict()
            for sid, cw in self.sample_calibration_working.items()}
        # Masks are keyed by (sample_id, fraction); serialize as a list since JSON
        # object keys can't be tuples.
        payload['sls_masks'] = [
            {'sample_id': sid, 'fraction': frac, 'mask': m.to_dict()}
            for (sid, frac), m in self.sls_masks.items() if not m.is_empty()]
        # Mode-map picks, same list shape (additive, absent-tolerant on load).
        # `is_empty` = no families at all: a labeled-but-unpicked family is
        # still user state (its label and role survive the reload).
        payload['mode_picks'] = [
            {'sample_id': sid, 'fraction': frac, 'picks': mp.to_dict()}
            for (sid, frac), mp in self.mode_picks.items() if not mp.is_empty()]
        # Recorded mode sets: one list entry per set (additive, absent-tolerant
        # on load) -- the snapshot IS the persistent object; the partitions and
        # per-mode fits computed from it recompute on demand.
        payload['mode_sets'] = [
            {'sample_id': sid, 'fraction': frac, 'set': s.to_dict()}
            for (sid, frac), sets in self.mode_sets.items() for s in sets]
        # ATOMIC: serialize to a sibling temp file and os.replace() it into position.
        # `open(file_path, 'w')` truncates BEFORE json.dump streams, so any failure during
        # the dump -- a full disk, a value the sanitizer does not cover, the process being
        # killed -- would leave a TRUNCATED file where the session was. The session file is
        # the program's only persistence, and the two layers above are both fail-silent, so
        # a partial write is a silent total loss of the user's work. os.replace is atomic on
        # Windows and POSIX, and a failed dump leaves the PREVIOUS session intact. Same
        # directory on purpose: os.replace is only atomic within one filesystem.
        directory = os.path.dirname(os.path.abspath(file_path)) or '.'
        fd, tmp_path = tempfile.mkstemp(suffix='.tmp', prefix='.session-',
                                        dir=directory)
        try:
            # No encoding= here, nor on load_session's open(), so both use the locale
            # encoding (cp1252 on Windows). Safe ONLY because json.dump below defaults to
            # ensure_ascii=True and escapes every non-ASCII character before it reaches the
            # file -- labels really do carry em dashes, +/- and roots. Passing
            # ensure_ascii=False to shrink the file would emit raw non-ASCII bytes and
            # corrupt sessions on any non-UTF-8 locale.
            with os.fdopen(fd, 'w') as fh:
                # Non-finite floats go out as the `_NONFINITE_KEY` sentinel, the tokens
                # forbidden (allow_nan=False): strict JSON that round-trips a NaN.
                # Write compact (no indent): the payload embeds raw correlogram/trace
                # arrays of thousands of floats, and indent=2 puts each on its own line,
                # inflating the file ~1.5x. A session is a self-contained data artifact,
                # not meant for hand-editing; pretty-print on demand with
                # `python -m json.tool <session>.lsjson`. load_session reads via
                # json.load, so compact and indented files parse identically.
                json.dump(_json_sanitize(payload), fh, separators=(',', ':'),
                          allow_nan=False)
            os.replace(tmp_path, file_path)
        except BaseException:
            # BaseException, not Exception: a KeyboardInterrupt mid-dump must not leave
            # the temp file behind either. The original file is already safe -- it has
            # not been touched at this point.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        return file_path

    def load_session(self, file_path: str) -> None:
        with open(file_path) as fh:
            payload = _json_restore(json.load(fh))
        # Build the whole new state in LOCALS first, and only assign it to self after
        # every parse step succeeds — so a malformed or schema-mismatched session can't
        # leave a HALF-loaded controller (a new workspace beside stale/default calibration
        # + masks). The GUI reopen-last-session path relies on this "all or nothing":
        # on any parse failure self is untouched, so its fail-soft catch truly starts empty.
        workspace = Workspace.from_dict(payload)
        calibration_committed = CalibrationState.from_dict(
            payload.get('calibration', {}))
        sample_calibration_committed = {
            sid: CalibrationState.from_dict(d)
            for sid, d in payload.get('sample_calibrations', {}).items()}
        raw_masks = payload.get('sls_masks', [])
        if isinstance(raw_masks, list):     # current (sample, fraction) format
            sls_masks = {
                (e['sample_id'], e.get('fraction')): SLSMask.from_dict(e['mask'])
                for e in raw_masks}
        else:                               # back-compat: {sample_id: mask}
            sls_masks = {
                (sid, None): SLSMask.from_dict(d)
                for sid, d in raw_masks.items()}
        # Mode-map picks: additive, absent-tolerant (a pre-pick session loads
        # with none), parsed into the local before the atomic assign.
        mode_picks = {
            (e['sample_id'], e.get('fraction')):
                ModeMapPicks.from_dict(e['picks'])
            for e in payload.get('mode_picks', [])}
        mode_sets: Dict[tuple, List[RecordedModeSet]] = {}
        for e in payload.get('mode_sets', []):
            mode_sets.setdefault(
                (e['sample_id'], e.get('fraction')), []).append(
                RecordedModeSet.from_dict(e['set']))
        # The working copies: read when the file carries them, else rebuilt from the
        # committed ones (a session written before they were persisted has no pending
        # calibration state to restore).
        if 'calibration_working' in payload:
            calibration_working = CalibrationState.from_dict(
                payload['calibration_working'])
        else:
            calibration_working = CalibrationState(**calibration_committed.to_dict())
        if 'sample_calibrations_working' in payload:
            sample_calibration_working = {
                sid: CalibrationState.from_dict(d)
                for sid, d in payload['sample_calibrations_working'].items()}
        else:
            sample_calibration_working = {
                sid: CalibrationState(**cc.to_dict())
                for sid, cc in sample_calibration_committed.items()}
        # All parsed cleanly — commit the new state atomically.
        self.workspace = workspace
        self.calibration_committed = calibration_committed
        self.calibration_working = calibration_working
        self.sample_calibration_committed = sample_calibration_committed
        self.sample_calibration_working = sample_calibration_working
        self.sls_masks = sls_masks
        self.mode_picks = mode_picks
        self.mode_sets = mode_sets
        self.results.clear()
        self._form_factor_run_sig.clear()   # stamps belong to the old workspace's fits
        self._dls_run_sig.clear()           # (the Γ-q² / D-vs-c stamps likewise)
        self._sls_candidate_cache.clear()   # the memo belongs to the old workspace
        self._dls_candidate_cache.clear()   # (same — keyed to the old measurements)
        self._distribution_stamps.clear()   # stamps belong to the old results
        # A blankless-run acknowledgement is never persisted, so a reloaded session must
        # ask again -- and these sample ids belong to the OLD workspace anyway.
        self._unmeasured_blank_ack.clear()
        # The undo history belongs to the previous session's measurements; drop it so
        # Undo can't step back into a no-longer-loaded state.
        self._commit_history.clear()

    def source_paths(self) -> Dict[str, Optional[str]]:
        """item_id -> original file path, for an optional reload-from-source.

        Intentional scaffolding for the Planned reload-from-source feature (see
        CLAUDE.md "Planned"): the per-measurement ``source_path`` field is already
        populated and serialized, and this aggregator is the accessor that feature
        will consume. Kept deliberately though currently uncalled -- not dead code.
        """
        return {i: m.source_path for i, m in self.workspace.measurements.items()}
