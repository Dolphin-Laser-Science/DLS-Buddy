"""
app/sample_state.py
===================

Per-sample user state the controller keeps beside the workspace: the SLS data mask
(`SLSMask`) and the Mode Map's picked families and recorded mode sets
(`ModeFamilyPicks` through `RecordedModeSet`), with the session conversions they carry;
and `BlankSwitch`, the plan a removal's confirm is built from for a series whose solvent
blank the removal would change. The controller imports them. Qt-free (invariant 5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from core.workspace import notes_from_json, notes_to_json
from core.notes import Note
from analysis.dls import DEFAULT_AMPLITUDE_FLOOR, MIN_DECAY_COMPLETENESS


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
class BlankSwitch:
    """One surviving SLS series whose solvent blank a removal would change.

    Planned by `Controller.blank_switches_on_removal` before the act and composed into
    the removal confirm by `Controller.blank_switch_message`. The labels are the
    roster-unique `blank_labels` entries, both named against the sample's blanks before
    the removal, the roster on screen while the confirm is read. `to_item_id` and
    `to_label` are None when no blank remains; `to_source` is
    the resolver's tag after the removal ('user' / 'auto:same-file' / 'auto:sample').
    """
    item_id: str
    series_name: str
    from_item_id: str
    from_label: str
    to_item_id: Optional[str]
    to_label: Optional[str]
    to_source: str


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
    # The grid-end verdict (`ModePoint.grid_end`), snapshotted with everything else. A
    # session written before it existed reads back False: no verdict was reached on it.
    grid_end: bool = False
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
    # The mode's POSITION (`ModePoint`'s two, snapshotted with everything else): a
    # distribution lobe's geometric mean, the Gamma every derived quantity reads; the
    # node `gamma_s_inv` above stays the identity. A session written before they existed
    # reads back NaN on both, and `plotted_gamma_s_inv` then reads the node -- which is
    # what that record held.
    mean_gamma_s_inv: float = float('nan')
    mean_rh_nm: float = float('nan')

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
            'grid_end': self.grid_end,
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
            'mean_gamma_s_inv': _nan_to_none(self.mean_gamma_s_inv),
            'mean_rh_nm': _nan_to_none(self.mean_rh_nm),
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
            grid_end=bool(d.get('grid_end', False)),
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
            mean_gamma_s_inv=_none_to_nan(d.get('mean_gamma_s_inv')),
            mean_rh_nm=_none_to_nan(d.get('mean_rh_nm')),
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
    # The record's notes (`core.notes.Note`): a family that matched no point, points
    # without a concentration, the map's region. A version-2 session's sentences load as
    # `legacy` notes (`core.workspace.note_from_json`).
    notes: Tuple[Note, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'set_id': self.set_id, 'label': self.label,
            'source_method': self.source_method,
            'polarization_state': self.polarization_state,
            'recorded': self.recorded,
            'modes': [m.to_dict() for m in self.modes],
            'notes': notes_to_json(self.notes),
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
            notes=notes_from_json(d.get('notes', [])),
        )
