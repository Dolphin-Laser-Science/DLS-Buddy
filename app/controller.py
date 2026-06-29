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
  - is_dirty / dirty_items            : has anything changed since the last commit?
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
Not enabled, but designed for: every expensive call is a single controller method
with no widget interaction, so it can later be dispatched to a worker thread
without touching the analysis or the widgets.

Change history
--------------
2026-06-13  controller.py v1: workspace management, commit/working parameter
            state, calibration state, DLS + SLS analysis orchestration, JSON
            session save/load (self-contained + reload-from-source).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

import numpy as np

from core.workspace import (
    Workspace, LoadedMeasurement, LoadedTrace, Sample, SampleResult,
    MeasurementResultRow, SampleRhRow,
    _DLS_PARAM_KEYS, _SLS_PARAM_KEYS,
)
from app.settings import SettingsState
from analysis import dls as dls_engine
from analysis import sls as sls_engine
from analysis import depolarization as depol_engine
from analysis import utilities as util
from analysis import trace_analysis as trace_engine
from analysis import synthetic_dataset as synth
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
_TRACE_OUTLIER_K = 3.0
_TRACE_BLOCKVAR_THRESHOLD = 1.5
_TRACE_ADF_SIGNIFICANCE = 0.05


def _value_in(value: Optional[float], excluded) -> bool:
    """True if `value` matches any entry in `excluded` (float-tolerant). Used to drop
    user-excluded outlier angles/concentrations from a multi-point fit (#9)."""
    if value is None or not excluded:
        return False
    return any(np.isclose(float(value), float(e)) for e in excluded)


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


# Parameters shared by every measurement in a sample (entered once, propagated).
# Concentration and angle are deliberately excluded -- they are per-measurement
# axes within a sample, not shared identity.
_SHARED_PARAM_KEYS = (
    'polymer_name', 'solvent_name', 'temperature_K', 'wavelength_nm',
    'solvent_refractive_index', 'viscosity_Pa_s', 'dn_dc_mL_per_g',
)

# Parameter keys the Reset button preserves (feedback 2026-06-29 #6): the scattering
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

# Distribution methods yield a VARIABLE number of size-distribution peaks (not a
# fixed parameter set), so replicate averaging aligns their peaks positionally
# (Rh-ascending) instead of by name. They are report-only (no single Rh written).
_DLS_DISTRIBUTION_METHODS = ('nnls', 'contin', 'lognormal')


# ===========================================================================
# Calibration state (committed / working, like a measurement's parameters)
# ===========================================================================

@dataclass
class CalibrationState:
    """The session-level SLS calibration.

    The user supplies one calibrant point (intensity, angle, standard liquid +
    geometry); the controller computes k_c. A None k_c means uncalibrated -- the
    analysis still runs, flagged. Mirrors the unified model in sls.py.
    """
    calibrant_intensity: Optional[float] = None
    calibrant_angle_deg: float = 90.0
    standard_name: str = 'toluene'
    standard_geometry: str = 'VU'
    standard_wavelength_nm: float = 532.0
    standard_temperature_C: float = 25.0
    standard_refractive_index: Optional[float] = 1.496   # toluene ~1.496 at 532 nm
    dark_count_rate: float = 0.0
    depolarization_ratio_v: Optional[float] = None
    k_c: Optional[float] = None                          # computed; editable

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

    def to_dict(self) -> Dict[str, Any]:
        return {
            'calibrant_intensity': self.calibrant_intensity,
            'calibrant_angle_deg': self.calibrant_angle_deg,
            'standard_name': self.standard_name,
            'standard_geometry': self.standard_geometry,
            'standard_wavelength_nm': self.standard_wavelength_nm,
            'standard_temperature_C': self.standard_temperature_C,
            'standard_refractive_index': self.standard_refractive_index,
            'dark_count_rate': self.dark_count_rate,
            'depolarization_ratio_v': self.depolarization_ratio_v,
            'k_c': self.k_c,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CalibrationState':
        return cls(**d) if d else cls()


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
    rg_source: str               # 'computed' or 'user'
    rh_source: str               # 'computed' or 'user'
    is_apparent: bool
    interpretation: str
    shape: str = ''              # concise architecture label (e.g. 'random coil')
    rho_se: Optional[float] = None  # statistical SE (only when both Rg and Rh have one)


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
    any_uncalibrated_mw: bool        # a plotted Mw came from an uncalibrated run
    n_excluded: int                  # samples dropped (no Mw, or non-positive y)


# ===========================================================================
# Controller
# ===========================================================================

class Controller:
    """Owns the workspace and orchestrates everything the GUI asks for."""

    def __init__(self) -> None:
        self.workspace = Workspace()
        # Global user preferences (seed defaults + appearance), loaded from the
        # settings.json at the repo root (or defaults if absent). "Seed, never
        # override": GUI controls initialise from these, analysis methods fall back
        # to them when the caller passes None; the per-run value still wins.
        self.settings = SettingsState.load()
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
        # last analysis results, keyed for the GUI to fetch and plot
        self.results: Dict[str, Any] = {}
        # Undo history: a stack of committed-state snapshots, one pushed before each
        # Update that actually changes something. `undo` steps back through these
        # previously-applied parameter sets (feedback 2026-06-29 #5).
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
        analysed until commit. Returns the new item_id.
        """
        item_id = self.workspace.new_item_id()
        lm = LoadedMeasurement(
            item_id=item_id, kind=kind, raw=raw,
            working_params=dict(params), committed_params=dict(params),
            source_path=source_path)
        self.workspace.add_measurement(lm)
        self.workspace.regroup()
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
        """Set one working parameter on several measurements at once (feedback A2).

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

    def sample_id_of(self, item_id: str) -> Optional[str]:
        """Public: the sample currently containing this measurement (or None)."""
        return self._sample_id_of(item_id)

    def _sample_id_of(self, item_id: str) -> Optional[str]:
        """Which sample currently contains this measurement (by committed grouping)."""
        for sid, s in self.workspace.samples.items():
            if (item_id in s.dls_item_ids or item_id in s.sls_item_ids
                    or item_id == s.solvent_reference_item_id):
                return sid
        return None

    def _item_ids_in_sample(self, sample_id: str) -> List[str]:
        s = self.workspace.samples[sample_id]
        ids = list(s.dls_item_ids) + list(s.sls_item_ids)
        if s.solvent_reference_item_id:
            ids.append(s.solvent_reference_item_id)
        return ids

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

    def dirty_items(self) -> List[str]:
        """item_ids with un-committed edits (for the pending-update indicator)."""
        return [i for i, m in self.workspace.measurements.items() if m.is_dirty()]

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
        self.workspace.regroup()

    @staticmethod
    def _commit_calibration(cal: CalibrationState) -> None:
        """Recompute k_c in place (so working and its committed copy agree)."""
        cal.k_c = cal.compute_k_c()

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
        self.workspace.regroup()

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
        re-enter (feedback 2026-06-29 #6). Identity/optics/concentration/temperature/
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
        setattr(cal, key, value)

    def preview_k_c(self, sample_id: Optional[str] = None) -> Optional[float]:
        """k_c from the working calibration in scope (sample override or session)."""
        return self._working_calibration(sample_id).compute_k_c()

    def committed_k_c(self, sample_id: Optional[str] = None) -> Optional[float]:
        return self._committed_calibration(sample_id).k_c

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
        self.workspace.assign_to_sample(item_id, sample_id)

    def clear_override(self, item_id: str) -> None:
        self.workspace.clear_override(item_id)

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
    def _cumulant_dependent_cache_keys(self) -> List[tuple]:
        """Result-cache keys whose value depends on the chosen cumulant method:
        every per-measurement cumulant fit, plus Gamma-vs-q^2 / D-vs-c results
        that derived their per-point Gamma from the cumulant (gamma_source)."""
        keys = []
        for k, v in self.results.items():
            if k[0] == 'cumulants':
                keys.append(k)
            elif (k[0] in ('gamma_q2', 'conc_extrap')
                  and getattr(v, 'gamma_source', None) == 'cumulant'):
                keys.append(k)
        return keys

    @staticmethod
    def _is_cumulant_replicate_label(label: str) -> bool:
        s = (label or '').lower()
        return 'replicate average' in s and 'cumulant' in s

    def cumulant_dependent_result_count(self) -> int:
        """How many cumulant-derived results currently exist (cumulant fits,
        cumulant Gamma-vs-q^2 / D-vs-c, cumulant replicate averages). Used to
        decide whether switching the cumulant method needs a clear-and-warn."""
        n = sum(1 for r in self.workspace.dls_result_rows.values()
                if r.method == 'cumulant')
        n += sum(1 for k in self._cumulant_dependent_cache_keys()
                 if k[0] in ('gamma_q2', 'conc_extrap'))
        n += sum(1 for r in self.workspace.sample_rh_rows.values()
                 if r.source_kind == 'replicate_avg'
                 and str(r.source_set).split('|')[0] == 'cumulant')
        return n

    def clear_cumulant_dependent_results(self) -> int:
        """Drop every cumulant-derived result so the workspace stays consistent
        after the cumulant method changes. NEVER touches distributions, SLS,
        traces, single/double/KWW (cumulant is only their seed), or a hand-entered
        (user) Rh. Returns the number of result-cache entries removed."""
        # 1. result cache (cumulant fits + cumulant-sourced gamma_q2 / conc_extrap)
        dead = self._cumulant_dependent_cache_keys()
        # remember which sample Rh rows the cleared multi-step results fed
        rh_rows_to_drop = set()
        for k in dead:
            if k[0] == 'gamma_q2':
                rh_rows_to_drop.add(('gamma_q2', k[1], k[2]))
            elif k[0] == 'conc_extrap':
                rh_rows_to_drop.add(('conc_extrap', k[1], k[2]))
            self.results.pop(k, None)
        # 2. per-measurement cumulant Summary rows
        self.workspace.dls_result_rows = {
            key: r for key, r in self.workspace.dls_result_rows.items()
            if r.method != 'cumulant'}
        # 3. sample Rh rows: cumulant-sourced gamma_q2 / conc_extrap + cumulant
        #    replicate averages
        self.workspace.sample_rh_rows = {
            key: r for key, r in self.workspace.sample_rh_rows.items()
            if not (
                (r.source_kind in ('gamma_q2', 'conc_extrap')
                 and (r.source_kind, r.sample_id, r.fraction) in rh_rows_to_drop)
                or (r.source_kind == 'replicate_avg'
                    and str(r.source_set).split('|')[0] == 'cumulant'))}
        # 4. reset a SampleResult Rh that came from a cumulant replicate average,
        #    never overwriting a hand-entered (user) value.
        for sample in self.workspace.samples.values():
            for r in sample.fraction_results.values():
                if (r.rh_source != 'user'
                        and r.rh_nm is not None
                        and self._is_cumulant_replicate_label(r.rh_label)):
                    r.rh_nm = None
                    r.rh_se = None
                    r.rh_label = ''
                    r.rh_apparent = None
                    r.rh_source = 'computed'
        return len(dead)

    # ----------------------------------------------------------------------
    # DLS analysis orchestration
    # ----------------------------------------------------------------------
    def run_cumulants(self, item_id: str, order: Optional[int] = None, **kw):
        m = self.workspace.measurements[item_id].build()
        if order is None:                       # seed from settings when not given
            order = self.settings.cumulant_order
        kw.setdefault('skip_initial_channels', self.settings.skip_initial_channels)
        kw.setdefault('method', self.settings.cumulant_method)
        res = dls_engine.fit_cumulants(m, order=order, **kw)
        self.results[('cumulants', item_id)] = res
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

    def run_distribution(self, item_id: str, method: str = 'contin', **kw):
        m = self.workspace.measurements[item_id].build()
        # Seed the Rh grid from settings unless the caller specified it.
        kw.setdefault('rh_min_nm', self.settings.rh_grid_min_nm)
        kw.setdefault('rh_max_nm', self.settings.rh_grid_max_nm)
        kw.setdefault('n_grid', self.settings.rh_grid_points)
        kw.setdefault('skip_initial_channels', self.settings.skip_initial_channels)
        if method == 'nnls':
            res = dls_engine.fit_nnls(m, **kw)
        elif method == 'lognormal':
            res = dls_engine.fit_lognormal(m, **kw)
        else:
            # CONTIN: seed the L-curve alpha sweep range from settings too.
            kw.setdefault('alpha_min', self.settings.lcurve_alpha_min)
            kw.setdefault('alpha_max', self.settings.lcurve_alpha_max)
            res = dls_engine.fit_contin(m, **kw)
        self.results[('distribution', item_id, method)] = res
        self._snapshot_distribution(item_id, method, res)
        return res

    def run_gamma_q2(self, sample_id: str, fraction: Optional[str] = None,
                     exclude_angles=(), **kw):
        """Gamma vs q^2 across the DLS angles of a sample (optionally one fraction).

        `exclude_angles` drops the named angles (degrees) from the fit so the user
        can remove outlier points and recompute (feedback 2026-06-26 #9)."""
        meas = [lm.build() for lm in self.workspace.sample_measurements(sample_id, 'dls')
                if (fraction is None or lm.committed_params.get('mw_fraction') == fraction)
                and not _value_in(lm.committed_params.get('angle_deg'), exclude_angles)]
        kw.setdefault('skip_initial_channels', self.settings.skip_initial_channels)
        kw.setdefault('cumulant_method', self.settings.cumulant_method)
        res = dls_engine.analyze_gamma_q2(meas, **kw)
        self.results[('gamma_q2', sample_id, fraction)] = res
        # Gamma -> q->0 is APPARENT (q-extrapolated only; single c folded out).
        self.workspace.upsert_sample_rh_row(SampleRhRow(
            sample_id=sample_id, source_kind='gamma_q2', source_set=f'{fraction}',
            rh_nm=getattr(res, 'rh_nm', None), rh_se=getattr(res, 'rh_se', None),
            is_apparent=True, rh_type_label='apparent',
            from_label=f'Γ vs q² ({len(meas)} angles)', fraction=fraction))
        return res

    def run_concentration_extrapolation(self, sample_id: str,
                                        fraction: Optional[str] = None,
                                        exclude_concentrations=(), **kw):
        """D vs c -> c->0 across the DLS concentrations of a sample (apparent D per
        measurement is angle-independent, so a multi-angle set still extrapolates).

        `exclude_concentrations` drops the named concentrations (g/mL) from the fit
        so the user can remove outlier points and recompute (feedback 2026-06-26 #9)."""
        meas = [lm.build() for lm in self.workspace.sample_measurements(sample_id, 'dls')
                if (fraction is None or lm.committed_params.get('mw_fraction') == fraction)
                and not _value_in(lm.committed_params.get('concentration_g_per_mL'),
                                  exclude_concentrations)]
        kw.setdefault('skip_initial_channels', self.settings.skip_initial_channels)
        kw.setdefault('cumulant_method', self.settings.cumulant_method)
        res = dls_engine.extrapolate_diffusion_vs_concentration(meas, **kw)
        self.results[('conc_extrap', sample_id, fraction)] = res
        # D vs c -> c->0 is THERMODYNAMIC (invariant 7).
        self.workspace.upsert_sample_rh_row(SampleRhRow(
            sample_id=sample_id, source_kind='conc_extrap', source_set=f'{fraction}',
            rh_nm=getattr(res, 'rh0_nm', None), rh_se=getattr(res, 'rh0_se', None),
            is_apparent=False, rh_type_label='thermodynamic',
            from_label=f'D vs c ({len(meas)} concentrations)', fraction=fraction))
        return res

    # ---- DLS Summary snapshot writers (durable display scalars) ----
    def _snapshot_parametric(self, item_id: str, method: str, res) -> None:
        """Write one MeasurementResultRow for a parametric fit (Summary Table 1).
        Per-measurement DLS is always apparent (single q, single c)."""
        self.workspace.replace_dls_rows_for(item_id, method)
        row = MeasurementResultRow(item_id=item_id, method=method, is_apparent=True)
        if method == 'cumulant':
            row.rh_nm, row.pdi = res.rh_nm, res.pdi
        elif method == 'single':
            row.rh_nm = res.mode.rh_nm
        elif method == 'double':
            row.rh_fast_nm, row.rh_slow_nm = res.mode1.rh_nm, res.mode2.rh_nm
        elif method == 'kww':
            row.rh_nm = res.rh_from_tau_c_nm
        self.workspace.upsert_dls_result_row(row)

    def _snapshot_distribution(self, item_id: str, method: str, res) -> None:
        """Write one MeasurementResultRow per resolved peak (Summary Table 1)."""
        self.workspace.replace_dls_rows_for(item_id, method)
        dist = getattr(res, 'distribution', res)     # CONTIN wraps its DistributionResult
        try:
            peaks = dls_engine.find_distribution_peaks(dist)
        except Exception:
            peaks = []
        for j, pk in enumerate(peaks):
            self.workspace.upsert_dls_result_row(MeasurementResultRow(
                item_id=item_id, method=method, peak_index=j,
                rh_nm=float(pk.rh_nm), int_fraction=float(pk.weight_fraction),
                is_apparent=True, label=f'{method.upper()} peak {j + 1}'))

    def distribution_peaks(self, result):
        """Resolved peaks of a distribution result as (rh_nm, gamma_s_inv,
        weight_fraction) tuples, Rh-ascending. A thin wrapper so the GUI can label
        peaks without importing the analysis layer (invariant 5)."""
        dist = getattr(result, 'distribution', result)
        try:
            peaks = dls_engine.find_distribution_peaks(dist)
        except Exception:
            return []
        return [(float(p.rh_nm), float(p.gamma_s_inv), float(p.weight_fraction))
                for p in peaks]

    # ---- DLS Summary tab: read-back accessors over the snapshot store ----
    def _measurement_label(self, item_id: str) -> str:
        """A short human label for a DLS measurement (angle / concentration /
        fraction), built from committed params. Qt-free."""
        lm = self.workspace.measurements.get(item_id)
        if lm is None:
            return item_id
        p = lm.committed_params
        bits = []
        ang = p.get('angle_deg')
        if ang is not None and np.isfinite(ang):
            bits.append(f'{ang:g}°')
        conc = p.get('concentration_g_per_mL')
        if conc is not None and np.isfinite(conc):
            bits.append(f'{conc:g} g/mL')
        frac = p.get('mw_fraction')
        if frac:
            bits.append(f'[{frac}]')
        if lm.derived_kind == 'replicate_average':
            bits.append('(avg correlogram)')
        return ' '.join(bits) if bits else item_id

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
                    'cumulant_rh': None, 'pdi': None, 'single_rh': None,
                    'kww_rh': None, 'double_fast': None, 'double_slow': None,
                    'peaks': {},     # method -> [(rh_nm, int_fraction), ...]
                }
                by_item[row.item_id] = rec
            if row.method == 'cumulant':
                rec['cumulant_rh'], rec['pdi'] = row.rh_nm, row.pdi
            elif row.method == 'single':
                rec['single_rh'] = row.rh_nm
            elif row.method == 'kww':
                rec['kww_rh'] = row.rh_nm
            elif row.method == 'double':
                rec['double_fast'], rec['double_slow'] = row.rh_fast_nm, row.rh_slow_nm
            else:                    # distribution method -> a peak
                rec['peaks'].setdefault(row.method, []).append(
                    (row.peak_index, row.rh_nm, row.int_fraction))
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
            })
        return sorted(out, key=lambda r: (r['sample_label'], r['source_kind']))

    # ----------------------------------------------------------------------
    # DLS replicate averaging (user-initiated; never automatic)
    # ----------------------------------------------------------------------
    def average_dls_correlograms(self, item_ids: List[str]) -> str:
        """Average true replicate correlograms into a new, tagged DLS measurement.

        Channel-by-channel mean of g2 - 1 over the selected replicates (which must
        share an identical lag grid and matching optics). The result is registered
        as an ordinary DLS measurement -- it flows through the sidebar, the DLS tab,
        and session JSON like any other -- but tagged `derived_kind='replicate_average'`
        with `derived_from` listing its sources. Its parameters are copied from the
        first replicate, so it auto-groups into the same sample and is analysable at
        once. Returns the new item_id.

        Note this produces the cleaner POINT-ESTIMATE curve; the defensible Rh ±
        comes from `average_dls_results` (the spread across per-replicate fits).
        """
        lms = self._require_dls_items(item_ids)
        meas = [lm.build() for lm in lms]
        avg = dls_engine.average_replicate_correlograms(meas)   # raises on mismatch

        new_id = self.workspace.new_item_id()
        params = dict(lms[0].committed_params)         # replicates share identity
        raw = {
            'delay_times_s': [float(x) for x in avg.delay_times_s],
            'correlogram': [float(x) for x in avg.mean_g2m1],
            # per-channel SD across replicates, kept for reference (ignored by build())
            'g2m1_sd': [float(x) for x in avg.sd_g2m1],
        }
        lm = LoadedMeasurement(
            item_id=new_id, kind='dls', raw=raw,
            working_params=dict(params), committed_params=dict(params),
            source_path=None,
            derived_from=list(item_ids),
            derived_kind='replicate_average',
        )
        self.workspace.add_measurement(lm)
        self.workspace.regroup()
        return new_id

    def average_dls_results(self, item_ids: List[str], method: str,
                            tau_min_s: Optional[float] = None,
                            tau_max_s: Optional[float] = None,
                            write_to_sample: bool = True) -> Dict[str, Any]:
        """Fit each replicate independently and report the mean +/- SD/sqrt(N).

        This is the ISO 22412 repeat-measurement path and the ONLY place the
        platform reports a DLS dynamic +/-: the uncertainty is the spread across
        genuine repeats, never a covariance from one correlogram (whose lag
        channels are correlated, Schaetzel 1990). Each replicate is fit on its OWN
        grid with the shared tau window, so -- unlike correlogram averaging -- the
        grids need not be identical.

        `method` is one of 'cumulant', 'single', 'double', 'kww' (parametric) or
        'nnls', 'contin', 'lognormal' (distribution). For the parametric methods
        with a single unambiguous Rh (cumulant, single, kww) the averaged Rh +/- SE
        is written into the sample's SampleResult (provenance-labelled, never
        overwriting a hand-entered Rh). Double-exponential (two modes) and the
        distribution methods (variable peak count) are reported in the returned
        summary only. The distribution methods average their peaks positionally
        (Rh-ascending) and warn when replicates disagree on the peak count (see
        `_average_distribution_results`). Returns a summary dict the GUI renders
        (and the validation asserts on).
        """
        lms = self._require_dls_items(item_ids)
        # All replicates must belong to one sample (we write one Rh to one sample).
        sids = {self._sample_id_of(iid) for iid in item_ids}
        if len(sids) > 1:
            raise ValueError(
                "Averaging derived results needs replicates from a single sample; "
                "the selection spans more than one. Move them into one sample "
                "first, or select within one sample.")
        sample_id = sids.pop()
        fraction = lms[0].committed_params.get('mw_fraction')

        if method in _DLS_DISTRIBUTION_METHODS:
            return self._average_distribution_results(
                lms, method, sample_id, fraction)

        # Per-method extraction: name -> (unit, value), plus which name is the Rh
        # to write back. `primary_rh` is None for multi-mode methods.
        specs, primary_rh = _DLS_PARAM_SPECS[method]
        window = {'tau_min_s': tau_min_s, 'tau_max_s': tau_max_s}

        per_param: Dict[str, List[float]] = {name: [] for name, _ in specs}
        n_ok = 0
        for lm in lms:
            m = lm.build()
            try:
                res = self._fit_one_replicate(m, method, window)
            except Exception:
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

        rh_written = False
        rh_skip_reason = None
        if write_to_sample and primary_rh is not None and rh_stats is not None \
                and math.isfinite(rh_stats.mean) and rh_stats.n >= 2:
            rh_written, rh_skip_reason = self._write_replicate_rh(
                sample_id, fraction, rh_stats, method)
        elif primary_rh is None:
            rh_skip_reason = (f"{method} has more than one Rh mode; not written to "
                              f"the sample (shown above only).")

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
        }

    def _average_distribution_results(self, lms, method: str, sample_id: str,
                                      fraction: Optional[str]) -> Dict[str, Any]:
        """Average a distribution method's PEAKS across replicates, positionally.

        Each replicate is fitted (NNLS / CONTIN / lognormal) and its peaks resolved
        (`find_distribution_peaks`, already Rh-ascending). Peaks are aligned by
        **order of appearance**: position k averages the k-th-smallest-Rh peak over
        every replicate that resolved at least k peaks. Per position we report
        Rh = mean ± SD/√n (Eq. 37), the mean intensity weight, and how many runs
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
        for lm in lms:
            m = lm.build()
            try:
                dist = self._fit_distribution_replicate(m, method)
                peaks = dls_engine.find_distribution_peaks(dist)
            except Exception:
                continue                   # a single failed fit just drops out
            if not peaks:
                continue
            per_replicate_peaks.append(peaks)
            n_ok += 1

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
        }

    def _fit_distribution_replicate(self, m, method: str):
        """Fit one replicate with a distribution method and return its
        DistributionResult (CONTIN's is unwrapped from its ContinResult). Mirrors
        `run_distribution`'s Settings seeds but does not touch the results cache."""
        kw = dict(rh_min_nm=self.settings.rh_grid_min_nm,
                  rh_max_nm=self.settings.rh_grid_max_nm,
                  n_grid=self.settings.rh_grid_points)
        if method == 'nnls':
            return dls_engine.fit_nnls(m, **kw)
        if method == 'lognormal':
            return dls_engine.fit_lognormal(m, **kw)
        if method == 'contin':
            kw['alpha_min'] = self.settings.lcurve_alpha_min
            kw['alpha_max'] = self.settings.lcurve_alpha_max
            res = dls_engine.fit_contin(m, **kw)
            return getattr(res, 'distribution', res)   # ContinResult wraps it
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
                raise ValueError(
                    f"Replicate averaging is DLS-only; {iid!r} is {lm.kind!r}.")
            lms.append(lm)
        return lms

    def _fit_one_replicate(self, m, method: str, window: Dict[str, Any]):
        """Fit one replicate with the chosen dynamic method and the shared window."""
        window = {**window}   # don't mutate the caller's dict
        window.setdefault('skip_initial_channels', self.settings.skip_initial_channels)
        if method == 'cumulant':
            return dls_engine.fit_cumulants(
                m, order=self.settings.cumulant_order,
                method=self.settings.cumulant_method, **window)
        if method == 'single':
            return dls_engine.fit_single_exponential(m, **window)
        if method == 'double':
            return dls_engine.fit_double_exponential(m, **window)
        if method == 'kww':
            return dls_engine.fit_kww(m, **window)
        raise ValueError(f"Unknown averaging method {method!r}.")

    def _write_replicate_rh(self, sample_id: str, fraction: Optional[str],
                            rh_stats, method: str):
        """Write the replicate-averaged Rh +/- SE into the sample, respecting a
        user override. Returns (written, skip_reason)."""
        r = self.workspace.samples[sample_id].result_for(fraction)
        if r.rh_source == 'user' and r.rh_nm is not None:
            return False, ("the sample already has a hand-entered Rh, which is not "
                           "overwritten.")
        label = (f"DLS replicate average ({rh_stats.n} runs, {method}); "
                 f"± = SD/√N across runs (ISO 22412)")
        r.rh_nm = float(rh_stats.mean)
        r.rh_source = 'computed'
        # A single-angle dynamic Rh is apparent (not extrapolated to q->0, c->0).
        r.rh_apparent = True
        r.rh_label = label
        r.rh_se = unc.se_or_none(rh_stats.sem)
        # Also snapshot it as a durable Summary Table-2 row (alongside the
        # SampleResult write above), so it persists and reads back in the table.
        self.workspace.upsert_sample_rh_row(SampleRhRow(
            sample_id=sample_id, source_kind='replicate_avg',
            source_set=f'{method}|{fraction}', rh_nm=float(rh_stats.mean),
            rh_se=unc.se_or_none(rh_stats.sem), is_apparent=True,
            rh_type_label='apparent',
            from_label=f'replicate avg ({rh_stats.n} runs, {method})',
            fraction=fraction))
        return True, None

    # ----------------------------------------------------------------------
    # SLS analysis orchestration (uses committed calibration)
    # ----------------------------------------------------------------------
    def _rayleigh_for_sample(self, sample_id: str, fraction: Optional[str] = None):
        """Excess Rayleigh ratios for one molecular-weight fraction of a sample.

        `fraction` selects the concentration series whose `mw_fraction` label equals
        it. For an unfractioned sample every measurement carries None, so the default
        (None) returns the whole series (backward-compatible). The c = 0 solvent
        reference is fetched independently and is therefore SHARED across fractions.
        """
        ref_lm = self.workspace.solvent_reference(sample_id)
        if ref_lm is None:
            raise ValueError(
                f"Sample {sample_id!r} has no solvent reference (c = 0); SLS "
                f"calibration needs one.")
        solvent = ref_lm.build()
        cal = self._calibration_for_sample(sample_id)   # per-sample override or session
        k_c = cal.k_c
        n_std = cal.standard_refractive_index
        dark = cal.dark_count_rate
        rr = []
        for lm in self.workspace.sample_measurements(sample_id, 'sls'):
            if lm.committed_params.get('mw_fraction') != fraction:
                continue
            sample = lm.build()
            rr.append(sls_engine.compute_excess_rayleigh_ratio(
                sample, solvent, calibration_constant=k_c,
                standard_refractive_index=n_std, dark_count_rate=dark))
        return rr

    def run_zimm(self, sample_id: str, method: str = 'zimm',
                 fraction: Optional[str] = None):
        rr = self.masked_rayleigh(sample_id, fraction)
        n_conc = len({round(r.concentration_g_per_mL, 12) for r in rr
                      if r.concentration_g_per_mL != 0})
        if n_conc < 2:
            raise ValueError(
                f"Only {n_conc} concentration(s) remain after masking; a "
                f"{method.capitalize()} extrapolation needs at least two. Use "
                f"Debye or Guinier for single-concentration analysis.")
        res = sls_engine.zimm_analysis(rr, method=method)
        self.results[('zimm', sample_id, method, fraction)] = res
        # attach Mw/Rg/A2 to this fraction's sample result, marking provenance.
        s = self.workspace.samples[sample_id].result_for(fraction)
        if s.mw_source != 'user':       # never clobber a manual Mw
            s.mw_g_per_mol = res.mw_g_per_mol
            s.mw_source = 'computed'
            s.calibrated = res.calibrated
            s.mw_apparent = res.is_apparent      # False: thermodynamic
            cal_note = '' if res.calibrated else '; UNCALIBRATED — arbitrary scale'
            s.mw_label = (
                f"{method.capitalize()} extrapolation, {n_conc} concentrations "
                f"(thermodynamic{cal_note})")
        if s.rg_source != 'user':       # nor a manual Rg
            s.rg_nm = res.rg_nm
            s.rg_source = 'computed'
            s.rg_apparent = res.is_apparent      # False: Zimm/Berry are thermodynamic
            s.rg_label = (
                f"{method.capitalize()} extrapolation, {n_conc} concentrations "
                f"(thermodynamic, R² = {_fmt_r2(float(res.r_squared))})")
        s.a2_mol_mL_per_g2 = res.a2_mol_mL_per_g2
        s.a2_source = 'computed'
        return res

    def set_manual_mw(self, sample_id: str, mw_g_per_mol: float,
                      fraction: Optional[str] = None) -> None:
        """Record a hand-entered Mw for a sample (e.g. characterised in water).

        Marks the provenance 'user' so later runs do not overwrite it and scaling
        plots can prefer it -- the correct path for PVP in a deep eutectic solvent,
        where the in-situ Mw is biased by co-solvent adsorption. A hand-entered Mw is
        treated as a trusted (thermodynamic, calibrated) value.
        """
        r = self.workspace.samples[sample_id].result_for(fraction)
        r.set_mw(mw_g_per_mol, source='user')
        r.mw_apparent = False
        r.mw_label = 'user-entered'
        r.mw_se = None                    # a hand-entered value carries no statistical SE

    # --- depolarized light scattering (static, Phase 1) ---
    def compute_depolarization(self, *, i_vv: Optional[float] = None,
                               i_vh: Optional[float] = None,
                               rho_v: Optional[float] = None,
                               dark_count: float = 0.0,
                               i_vv_se: Optional[float] = None,
                               i_vh_se: Optional[float] = None):
        """Static depolarization analysis (assumes vertically polarised incident light).

        Provide EITHER the VV and VH intensities (in the SLS file's own units; only
        their ratio matters), OR a depolarisation ratio rho_v directly. Returns a
        DepolarizationResult (rho_v, rho_u, delta^2, Cabannes factor, validity).
        Pure pass-through to analysis.depolarization -- no state is changed.
        """
        if rho_v is not None:
            # A direct ratio: analyse with unit VV so I_VH / I_VV = rho_v exactly.
            return depol_engine.analyze_depolarization(1.0, float(rho_v))
        if i_vv is None or i_vh is None:
            raise ValueError(
                "Provide both VV and VH intensities, or a depolarisation ratio rho_v.")
        return depol_engine.analyze_depolarization(
            float(i_vv), float(i_vh), dark_count=float(dark_count),
            i_vv_se=i_vv_se, i_vh_se=i_vh_se)

    # --- depolarized light scattering (dynamic, Phase 2) ---
    def ddls_correlogram_summary(self, sample_id: str) -> List[Dict[str, Any]]:
        """The sample's DLS correlograms with their polarisation tag, for the UI.

        Returns one dict per correlogram: item_id, angle_deg, geometry ('VV'/'VH'/
        'VU'/None), and `paired` (True if its angle has both a VV and a VH). Sorted
        by angle then geometry. The DDLS sub-tab renders this and shows which angles
        will actually pair.
        """
        rows: List[Dict[str, Any]] = []
        by_angle_geom: Dict[float, set] = {}
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
            by_angle_geom.setdefault(round(float(angle), 3), set()).add(geom)
        for item_id, angle, geom in built:
            pair = by_angle_geom.get(round(angle, 3), set())
            rows.append({
                'item_id': item_id, 'angle_deg': angle, 'geometry': geom,
                'paired': ('VV' in pair and 'VH' in pair),
            })
        rows.sort(key=lambda r: (r['angle_deg'], r['geometry'] or ''))
        return rows

    def run_ddls(self, sample_id: str, rod_length_nm: Optional[float] = None,
                 cumulant_order: int = 2, exclude_angles=()):
        """Pair a sample's VV/VH correlograms and extract rotational diffusion D_r.

        Gathers the sample's DLS correlograms, groups them by polarisation tag
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
        """
        vv: Dict[float, list] = {}
        vh: Dict[float, list] = {}
        untagged, t_k, eta = [], None, None
        for lm in self.workspace.sample_measurements(sample_id, 'dls'):
            m = lm.build()
            a = round(float(m.angle_deg), 3)
            if m.analyzer_geometry == 'VV':
                vv.setdefault(a, []).append(m)
            elif m.analyzer_geometry == 'VH':
                vh.setdefault(a, []).append(m)
            else:                                   # VU or None -> not pairable
                untagged.append((lm.item_id, m.analyzer_geometry))
            if t_k is None:
                t_k = m.temperature_K
            if eta is None and m.viscosity_Pa_s is not None:
                eta = m.viscosity_Pa_s

        paired_angles = sorted(set(vv) & set(vh))
        if not paired_angles:
            raise ValueError(
                "No angle has both a VV and a VH correlogram. Tag each correlogram's "
                "polarisation (VV/VH) in the Data tab; DDLS pairs them by angle.")
        # Drop user-excluded outlier angles (feedback 2026-06-26 #9).
        if exclude_angles:
            paired_angles = [a for a in paired_angles
                             if not _value_in(a, exclude_angles)]
            if not paired_angles:
                raise ValueError('All paired angles are excluded — nothing to fit.')

        def _avg_gamma(measurements: list) -> float:
            """Mean field decay rate over replicate correlograms (one fit each)."""
            gammas = [float(dls_engine.fit_cumulants(
                          m, order=cumulant_order,
                          skip_initial_channels=self.settings.skip_initial_channels,
                          method=self.settings.cumulant_method).gamma_s_inv)
                      for m in measurements]
            return float(np.mean(gammas))

        points = []
        replicates: Dict[float, tuple] = {}
        for a in paired_angles:
            g_vv = _avg_gamma(vv[a])
            g_vh = _avg_gamma(vh[a])
            replicates[a] = (len(vv[a]), len(vh[a]))
            q = phys.scattering_vector_q_m(
                a, vv[a][0].wavelength_nm, vv[a][0].solvent_refractive_index)
            points.append(depol_engine.DDLSRatePoint(
                angle_deg=a, q_m_inv=q, gamma_vv_s_inv=g_vv, gamma_vh_s_inv=g_vh))

        res = depol_engine.analyze_ddls(
            points, temperature_K=t_k, viscosity_Pa_s=eta, rod_length_nm=rod_length_nm)
        info = {
            'paired_angles': paired_angles,
            'vv_only': sorted(set(vv) - set(vh)),
            'vh_only': sorted(set(vh) - set(vv)),
            'untagged': untagged,
            'replicates': replicates,
            'n_replicate_angles': sum(1 for (nv, nh) in replicates.values()
                                      if nv > 1 or nh > 1),
        }
        self.results[('ddls', sample_id)] = res
        return res, info

    def ddls_shape(self, sample_id: str, model: str = 'both') -> Dict[str, Any]:
        """Particle dimensions from the sample's DDLS D_t, D_r (model-dependent).

        Runs the shape inverse on the cached DDLS result (computing it first if
        needed): the **sphere** model (R from D_r, with a sphericity check vs Rh) and
        the **rod** model (L, d from the Tirado inversion). `model` selects 'sphere',
        'rod', or 'both' (default). Returns a dict with the requested result objects.
        These are dimensions of an ASSUMED shape, not measurements -- the result
        objects carry that caveat. Raises if D_t/D_r are non-positive or the solvent
        viscosity is missing (the shape models need it; the decay rates do not).
        """
        res = self.results.get(('ddls', sample_id))
        if res is None:
            res, _ = self.run_ddls(sample_id)
        if not (res.d_t_m2_s > 0 and res.d_r_rad2_s > 0):
            raise ValueError(
                "Shape models need positive D_t and D_r; the DDLS result has a "
                "non-physical D_r (Gamma_VH <= Gamma_VV) -- check the VV/VH pairing.")
        t_k, eta = None, None
        for lm in self.workspace.sample_measurements(sample_id, 'dls'):
            m = lm.build()
            if t_k is None:
                t_k = m.temperature_K
            if eta is None and m.viscosity_Pa_s is not None:
                eta = m.viscosity_Pa_s
        if eta is None:
            raise ValueError(
                "The shape models need the solvent viscosity; set it in the Data tab.")
        out: Dict[str, Any] = {}
        if model in ('sphere', 'both'):
            out['sphere'] = depol_engine.sphere_dimensions_from_diffusion(
                res.d_t_m2_s, res.d_r_rad2_s, temperature_K=t_k, viscosity_Pa_s=eta,
                d_r_se=res.d_r_se)
        if model in ('rod', 'both'):
            out['rod'] = depol_engine.rod_dimensions_from_diffusion(
                res.d_t_m2_s, res.d_r_rad2_s, temperature_K=t_k, viscosity_Pa_s=eta,
                d_t_se=res.d_t_se, d_r_se=res.d_r_se)
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
        res = sls_engine.debye_analysis(r)
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
        res = sls_engine.guinier_analysis(r, qrg_max_valid=qrg_max_valid)
        self.results[('guinier', sample_id, concentration_g_per_mL, fraction)] = res
        return res

    def run_single_angle(self, sample_id: str, concentration_g_per_mL: float,
                         angle_deg: float, fraction: Optional[str] = None):
        """Apparent Mw from one angle of one concentration."""
        r = self._rayleigh_at_concentration(sample_id, concentration_g_per_mL, fraction)
        res = sls_engine.single_angle_mw(r, angle_deg)
        self.results[('single_angle', sample_id, concentration_g_per_mL, angle_deg, fraction)] = res
        return res

    def run_calibration_free_a2(self, sample_id: str, angle_deg: float,
                               mw_g_per_mol: Optional[float] = None,
                               fraction: Optional[str] = None):
        """2 A2 Mw (and A2 if Mw known) from intensity ratios -- no calibration."""
        rr = [r for r in self.masked_rayleigh(sample_id, fraction)
              if r.concentration_g_per_mL != 0]
        mw = mw_g_per_mol
        if mw is None:
            mw = self.workspace.samples[sample_id].result_for(fraction).effective_mw()
        res = sls_engine.calibration_free_a2(rr, angle_deg, mw_g_per_mol=mw)
        self.results[('cal_free_a2', sample_id, angle_deg, fraction)] = res
        return res

    def _rayleigh_at_concentration(self, sample_id: str,
                                  concentration_g_per_mL: float,
                                  fraction: Optional[str] = None):
        # Use the masked results so Debye/Guinier honour hidden angles/points and
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

    def masked_rayleigh(self, sample_id: str, fraction: Optional[str] = None):
        """Per-concentration RayleighRatioResults with the (sample, fraction) mask
        applied: masked points -> NaN (dropped by the engine's finite filters); a
        wholly masked concentration is omitted. Used for analysis and the Zimm grid.
        The unmasked results (for greying hidden points on a plot) come from
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

    # --- accessors for the SLS GUI (axes available in a sample) ---
    def sample_fractions(self, sample_id: str,
                         kind: str = 'sls') -> List[Optional[str]]:
        """Distinct molecular-weight fraction labels among a sample's measurements
        (of the given kind). Returns [None] when nothing is labelled (the common
        single-fraction case). Sorted with None (unlabelled) first."""
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

    # ----------------------------------------------------------------------
    # Cross-sample: rho = Rg/Rh (Rh source selection)
    # ----------------------------------------------------------------------
    # Rg flows into the SampleResult from the SLS Zimm/Berry run (thermodynamic).
    # Rh has several possible DLS sources, so it is chosen from candidates: the
    # default is a labelled, deterministic pick (most-extrapolated first), and the
    # GUI can override it or accept a hand-entered value. See ResultCandidate /
    # select_default_candidate in analysis.utilities for the selection contract.

    def dls_rh_candidates(self, sample_id: str,
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

        Distribution-model (CONTIN/NNLS) peaks are also offered, but only from
        results the user has ALREADY computed in the DLS tab (a CONTIN L-curve is
        too expensive to run on every refresh). Each resolved population is one
        candidate; the dominant peak competes as a default (tier 1, like the
        cumulant), secondary peaks are selectable but never the default (tier 0) so
        a minor population is never silently chosen -- this is the peak picker for a
        multi-population sample.

        Future: the DLS Summary store's `sample_rh_rows` already carry
        rh_nm/rh_se/is_apparent/source_kind, so the replicate-average / Gamma-q^2 /
        D-c rows could be mapped straight to ResultCandidate tiers here (apparent
        for replicate_avg & gamma_q2, thermodynamic for conc_extrap) to wire
        averaged Rh into rho. Not wired yet -- left as a deliberate next step.
        """
        lms = [lm for lm in self.workspace.sample_measurements(sample_id, 'dls')
               if lm.committed_params.get('mw_fraction') == fraction]
        cands: List[ResultCandidate] = []

        # tier 1 -- per-measurement cumulant Rh (apparent, single point)
        for lm in lms:
            try:
                res = dls_engine.fit_cumulants(
                    lm.build(), order=2,
                    skip_initial_channels=self.settings.skip_initial_channels)
            except Exception:
                continue
            if not np.isfinite(res.rh_nm):
                continue
            p = lm.committed_params
            ang = p.get('angle_deg')
            conc = p.get('concentration_g_per_mL')
            cands.append(ResultCandidate(
                value=float(res.rh_nm),
                label=(f"Cumulant Rh at {_fmt_deg(ang)}, {_fmt_conc(conc)} "
                       f"(apparent, single angle)"),
                kind='dls_cumulant', is_apparent=True, tier=1,
                quality=(-float(res.rms_error)
                         if np.isfinite(res.rms_error) else None),
                quality_kind='neg_rms', source_id=lm.item_id))

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
                peaks = dls_engine.find_distribution_peaks(dist)
            except Exception:
                continue
            p = self.workspace.measurements[item_id].committed_params
            ang = p.get('angle_deg')
            conc = p.get('concentration_g_per_mL')
            rms = float(dist.rms_error)
            for j, pk in enumerate(peaks):
                if not np.isfinite(pk.rh_nm):
                    continue
                cands.append(ResultCandidate(
                    value=float(pk.rh_nm),
                    label=(f"{method.upper()} peak {j + 1}: Rh = {pk.rh_nm:.3g} nm "
                           f"({pk.weight_fraction * 100:.0f}% of intensity) at "
                           f"{_fmt_deg(ang)}, {_fmt_conc(conc)} (apparent)"),
                    kind='dls_distribution_peak', is_apparent=True,
                    tier=(1 if pk.is_dominant else 0),
                    quality=(-rms if np.isfinite(rms) else None),
                    quality_kind='neg_rms',
                    source_id=f'{method}@{item_id}#{j}'))

        meas = [lm.build() for lm in lms]

        # tier 2 -- Gamma vs q^2 -> q->0, PER CONCENTRATION (apparent). Done one
        # concentration at a time: mixing concentrations would fold the
        # concentration dependence into the q^2 fit. Needs >= 2 angles at that c.
        by_conc: Dict[float, List[Any]] = {}
        for m in meas:
            by_conc.setdefault(round(float(m.concentration_g_per_mL), 12), []).append(m)
        for c, ms in sorted(by_conc.items()):
            if len({round(float(m.angle_deg), 6) for m in ms}) < 2:
                continue
            try:
                gq = dls_engine.analyze_gamma_q2(ms)
            except Exception:
                continue
            if not np.isfinite(gq.rh_nm):
                continue
            r2 = float(gq.r_squared)
            cands.append(ResultCandidate(
                value=float(gq.rh_nm),
                label=(f"Γ vs q² → q→0 at {_fmt_conc(c)} over "
                       f"{gq.angles_deg.size} angles "
                       f"(apparent, R² = {_fmt_r2(r2)})"),
                kind='dls_gamma_q2', is_apparent=True, tier=2,
                quality=(r2 if np.isfinite(r2) else None), quality_kind='r_squared',
                source_id=f'gamma_q2@{c}',
                value_se=unc.se_or_none(getattr(gq, 'rh_se', None))))

        # tier 3 -- D vs c -> c->0, PER ANGLE (thermodynamic). Done one angle at a
        # time so any q-dependence is not folded into the concentration fit. Needs
        # >= 2 distinct concentrations at that angle.
        by_angle: Dict[float, List[Any]] = {}
        for m in meas:
            by_angle.setdefault(round(float(m.angle_deg), 6), []).append(m)
        for a, ms in sorted(by_angle.items()):
            if len({round(float(m.concentration_g_per_mL), 12) for m in ms}) < 2:
                continue
            try:
                ce = dls_engine.extrapolate_diffusion_vs_concentration(ms)
            except Exception:
                continue
            if not np.isfinite(ce.rh0_nm):
                continue
            r2 = float(ce.r_squared)
            cands.append(ResultCandidate(
                value=float(ce.rh0_nm),
                label=(f"D vs c → c→0 at {_fmt_deg(a)} over "
                       f"{ce.n_concentrations} concentrations "
                       f"(thermodynamic, R² = {_fmt_r2(r2)})"),
                kind='dls_conc_extrap', is_apparent=False, tier=3,
                quality=(r2 if np.isfinite(r2) else None), quality_kind='r_squared',
                source_id=f'conc_extrap@{a}',
                value_se=unc.se_or_none(getattr(ce, 'rh0_se', None))))

        return cands

    def set_sample_rh(self, sample_id: str, candidate: ResultCandidate,
                      fraction: Optional[str] = None) -> None:
        """Store a chosen DLS-derived Rh candidate as the sample/fraction's Rh.

        An explicit selection (the GUI picked this candidate), so it overrides any
        previously stored value, including a hand-entered one. Records the
        provenance label and apparent/thermodynamic status for rho.
        """
        r = self.workspace.samples[sample_id].result_for(fraction)
        r.rh_nm = float(candidate.value)
        r.rh_source = 'computed'
        r.rh_apparent = bool(candidate.is_apparent)
        r.rh_label = candidate.label
        r.rh_se = candidate.value_se

    def auto_select_rh(self, sample_id: str,
                       fraction: Optional[str] = None) -> Optional[ResultCandidate]:
        """Pick and store the default Rh for a sample (most-extrapolated first).

        Never overwrites a hand-entered Rh (rh_source == 'user'); returns the
        stored-or-chosen candidate, or None if no DLS Rh is available. The default
        is labelled -- the GUI shows where the value came from; it is never silent.
        """
        r = self.workspace.samples[sample_id].result_for(fraction)
        cands = self.dls_rh_candidates(sample_id, fraction)
        if r.rh_source == 'user' and r.rh_nm is not None:
            return None                      # respect the manual override
        chosen = select_default_candidate(cands)
        if chosen is not None:
            self.set_sample_rh(sample_id, chosen, fraction)
        return chosen

    def set_manual_rh(self, sample_id: str, rh_nm: float,
                      is_apparent: bool = False,
                      fraction: Optional[str] = None) -> None:
        """Record a hand-entered Rh (e.g. from external characterisation).

        Marked provenance 'user' so a re-selection does not silently replace it.
        `is_apparent` lets the user say whether their value is an infinite-dilution
        (thermodynamic) Rh or a single-condition (apparent) one; default
        thermodynamic, since a hand-entered Rh is usually the best available value.
        """
        if not (rh_nm > 0):
            raise ValueError(f"rh_nm must be positive, got {rh_nm!r}.")
        r = self.workspace.samples[sample_id].result_for(fraction)
        r.rh_nm = float(rh_nm)
        r.rh_source = 'user'
        r.rh_apparent = bool(is_apparent)
        r.rh_label = 'user-entered'
        r.rh_se = None                    # a hand-entered value carries no statistical SE

    def sls_rg_candidates(self, sample_id: str,
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
        except Exception:
            return cands
        nonzero = [r for r in rr if r.concentration_g_per_mL != 0]
        n_conc = len({round(r.concentration_g_per_mL, 12) for r in nonzero})

        # tier 3 -- thermodynamic Zimm / Berry (need >= 2 concentrations)
        if n_conc >= 2:
            for method in ('zimm', 'berry'):
                try:
                    res = sls_engine.zimm_analysis(rr, method=method)
                except Exception:
                    continue
                if not np.isfinite(res.rg_nm):
                    continue
                r2 = float(res.r_squared)
                cands.append(ResultCandidate(
                    value=float(res.rg_nm),
                    label=(f"{method.capitalize()} extrapolation, "
                           f"{res.n_concentrations} concentrations "
                           f"(thermodynamic, R² = {_fmt_r2(r2)})"),
                    kind=f'sls_{method}', is_apparent=False, tier=3,
                    quality=(r2 if np.isfinite(r2) else None),
                    quality_kind='r_squared', source_id=method,
                    value_se=unc.se_or_none(getattr(res, 'rg_se', None))))

        # tier 2 -- apparent Debye / Guinier, one per concentration
        for r in nonzero:
            c = r.concentration_g_per_mL
            try:
                d = sls_engine.debye_analysis(r)
                if np.isfinite(d.rg_apparent_nm):
                    r2 = float(d.r_squared)
                    cands.append(ResultCandidate(
                        value=float(d.rg_apparent_nm),
                        label=(f"Debye at {_fmt_conc(c)} over {d.n_angles} angles "
                               f"(apparent, R² = {_fmt_r2(r2)})"),
                        kind='sls_debye', is_apparent=True, tier=2,
                        quality=(r2 if np.isfinite(r2) else None),
                        quality_kind='r_squared', source_id=f'debye@{c}',
                        value_se=unc.se_or_none(getattr(d, 'rg_apparent_se', None))))
            except Exception:
                pass
            try:
                g = sls_engine.guinier_analysis(r)
                if np.isfinite(g.rg_nm):
                    r2 = float(g.r_squared)
                    cands.append(ResultCandidate(
                        value=float(g.rg_nm),
                        label=(f"Guinier at {_fmt_conc(c)} over {g.n_angles} angles "
                               f"(apparent, R² = {_fmt_r2(r2)})"),
                        kind='sls_guinier', is_apparent=True, tier=2,
                        quality=(r2 if np.isfinite(r2) else None),
                        quality_kind='r_squared', source_id=f'guinier@{c}',
                        value_se=unc.se_or_none(getattr(g, 'rg_se', None))))
            except Exception:
                pass

        return cands

    def set_sample_rg(self, sample_id: str, candidate: ResultCandidate,
                      fraction: Optional[str] = None) -> None:
        """Store a chosen SLS-derived Rg candidate as the sample/fraction's Rg.

        An explicit selection -- overrides any previously stored value, including a
        hand-entered one. Records provenance and apparent/thermodynamic status.
        """
        r = self.workspace.samples[sample_id].result_for(fraction)
        r.rg_nm = float(candidate.value)
        r.rg_source = 'computed'
        r.rg_apparent = bool(candidate.is_apparent)
        r.rg_label = candidate.label
        r.rg_se = candidate.value_se

    def auto_select_rg(self, sample_id: str,
                       fraction: Optional[str] = None) -> Optional[ResultCandidate]:
        """Pick and store the default Rg for a sample (most-extrapolated first).

        Never overwrites a hand-entered Rg (rg_source == 'user'); returns the
        chosen candidate, or None if no SLS Rg is available. Labelled, never silent.
        """
        r = self.workspace.samples[sample_id].result_for(fraction)
        if r.rg_source == 'user' and r.rg_nm is not None:
            return None
        chosen = select_default_candidate(self.sls_rg_candidates(sample_id, fraction))
        if chosen is not None:
            self.set_sample_rg(sample_id, chosen, fraction)
        return chosen

    def set_manual_rg(self, sample_id: str, rg_nm: float,
                      is_apparent: bool = False,
                      fraction: Optional[str] = None) -> None:
        """Record a hand-entered Rg (e.g. from external characterisation).

        Marked provenance 'user' so a re-selection or a later Zimm run does not
        silently replace it. `is_apparent` defaults to thermodynamic (a hand-entered
        Rg is usually the best available value).
        """
        if not (rg_nm > 0):
            raise ValueError(f"rg_nm must be positive, got {rg_nm!r}.")
        r = self.workspace.samples[sample_id].result_for(fraction)
        r.rg_nm = float(rg_nm)
        r.rg_source = 'user'
        r.rg_apparent = bool(is_apparent)
        r.rg_label = 'user-entered'
        r.rg_se = None                    # a hand-entered value carries no statistical SE

    def sls_mw_candidates(self, sample_id: str,
                          fraction: Optional[str] = None) -> List[ResultCandidate]:
        """Available SLS-derived Mw values for a sample, each with provenance.

        tier 3  Zimm / Berry (thermodynamic, needs >= 2 concentrations);
        tier 2  Debye or Guinier per concentration (apparent).
        Unlike Rg, **Mw is calibration-dependent** -- an uncalibrated run gives an
        arbitrary-scale Mw, flagged in the label. For scaling plots the trustworthy
        Mw is usually the hand-entered one (e.g. characterised in water); that is
        offered separately as the 'user' value and preferred by the defaults.
        """
        cands: List[ResultCandidate] = []
        try:
            rr = self.masked_rayleigh(sample_id, fraction)
        except Exception:
            return cands
        nonzero = [r for r in rr if r.concentration_g_per_mL != 0]
        n_conc = len({round(r.concentration_g_per_mL, 12) for r in nonzero})

        def _cal_note(calibrated: bool) -> str:
            return '' if calibrated else '; UNCALIBRATED — arbitrary scale'

        if n_conc >= 2:
            for method in ('zimm', 'berry'):
                try:
                    res = sls_engine.zimm_analysis(rr, method=method)
                except Exception:
                    continue
                if not np.isfinite(res.mw_g_per_mol):
                    continue
                r2 = float(res.r_squared)
                cands.append(ResultCandidate(
                    value=float(res.mw_g_per_mol),
                    label=(f"{method.capitalize()} extrapolation, "
                           f"{res.n_concentrations} concentrations (thermodynamic"
                           f"{_cal_note(res.calibrated)}, R² = {_fmt_r2(r2)})"),
                    kind=f'sls_{method}', is_apparent=False, tier=3,
                    quality=(r2 if np.isfinite(r2) else None),
                    quality_kind='r_squared', source_id=method,
                    value_se=unc.se_or_none(getattr(res, 'mw_se', None))))

        for r in nonzero:
            c = r.concentration_g_per_mL
            for kind, fn, mw_attr in (('sls_debye', sls_engine.debye_analysis, 'mw_apparent_g_per_mol'),
                                      ('sls_guinier', sls_engine.guinier_analysis, 'mw_apparent_g_per_mol')):
                try:
                    res = fn(r)
                except Exception:
                    continue
                mw = float(getattr(res, mw_attr))
                if not np.isfinite(mw):
                    continue
                r2 = float(res.r_squared)
                name = 'Debye' if kind == 'sls_debye' else 'Guinier'
                cands.append(ResultCandidate(
                    value=mw,
                    label=(f"{name} at {_fmt_conc(c)} over {res.n_angles} angles "
                           f"(apparent{_cal_note(res.calibrated)}, R² = {_fmt_r2(r2)})"),
                    kind=kind, is_apparent=True, tier=2,
                    quality=(r2 if np.isfinite(r2) else None),
                    quality_kind='r_squared', source_id=f'{kind}@{c}',
                    value_se=unc.se_or_none(getattr(res, 'mw_apparent_se', None))))
        return cands

    def set_sample_mw(self, sample_id: str, candidate: ResultCandidate,
                      fraction: Optional[str] = None) -> None:
        """Store a chosen SLS-derived Mw candidate as the sample/fraction's Mw
        (explicit pick; overrides any previous value, including a hand-entered one)."""
        r = self.workspace.samples[sample_id].result_for(fraction)
        r.mw_g_per_mol = float(candidate.value)
        r.mw_source = 'computed'
        r.mw_apparent = bool(candidate.is_apparent)
        r.mw_label = candidate.label
        r.mw_se = candidate.value_se
        r.calibrated = 'UNCALIBRATED' not in candidate.label

    def auto_select_mw(self, sample_id: str,
                       fraction: Optional[str] = None) -> Optional[ResultCandidate]:
        """Pick and store the default Mw (most-extrapolated first); never overwrites
        a hand-entered Mw. Returns the chosen candidate, or None."""
        r = self.workspace.samples[sample_id].result_for(fraction)
        if r.mw_source == 'user' and r.mw_g_per_mol is not None:
            return None
        chosen = select_default_candidate(self.sls_mw_candidates(sample_id, fraction))
        if chosen is not None:
            self.set_sample_mw(sample_id, chosen, fraction)
        return chosen

    def auto_select_a2(self, sample_id: str,
                       fraction: Optional[str] = None) -> Optional[float]:
        """Populate A2 from the thermodynamic Zimm fit (for the A2-Mw scaling plot),
        unless a user A2 is set. A2 is calibration-dependent -- an uncalibrated A2 is
        on an arbitrary scale (the run's `calibrated` flag records this). Returns the
        stored A2, or None if none could be computed. There is no A2 source picker
        (yet): the scaling plot uses the Zimm/Berry A2."""
        r = self.workspace.samples[sample_id].result_for(fraction)
        if r.a2_source == 'user':
            return r.a2_mol_mL_per_g2
        try:
            res = sls_engine.zimm_analysis(
                self.masked_rayleigh(sample_id, fraction), method='zimm')
        except Exception:
            return None
        if not np.isfinite(res.a2_mol_mL_per_g2):
            return None
        r.a2_mol_mL_per_g2 = float(res.a2_mol_mL_per_g2)
        r.a2_source = 'computed'
        r.a2_se = unc.se_or_none(getattr(res, 'a2_se', None))
        return r.a2_mol_mL_per_g2

    def compute_sample_rho(self, sample_id: str,
                           fraction: Optional[str] = None) -> SampleRho:
        """rho = Rg/Rh for a sample/fraction, from its stored Rg (SLS) and Rh (DLS).

        Requires both to be present (run a Zimm/Berry Rg in the SLS tab and select
        an Rh source first). rho is flagged apparent if EITHER input is apparent.
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
        return SampleRho(
            sample_id=sample_id, rho=rr.rho,
            rg_nm=float(r.rg_nm), rh_nm=float(r.rh_nm),
            rg_label=(r.rg_label or 'Rg'), rh_label=(r.rh_label or 'Rh'),
            rg_source=r.rg_source, rh_source=r.rh_source,
            is_apparent=is_apparent, interpretation=rr.interpretation,
            shape=rr.shape, rho_se=rr.rho_se)

    def samples_pairable_rho(self) -> List[str]:
        """Sample ids that have both DLS and SLS (rho = Rg/Rh is possible)."""
        return [sid for sid, s in self.workspace.samples.items()
                if s.can_pair_rho]

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
    def samples_for_scaling(self, quantity: str = 'rg') -> List[str]:
        """Sample ids with an Mw and a positive y (Rg, or A2 for the A2 plot).

        A2 can be negative (poor solvent); only positive A2 can sit on a log-log
        scaling plot, so such samples are not eligible for the A2 plot.
        """
        out = []
        for sid, s in self.workspace.samples.items():
            mw = s.result.effective_mw()
            y = s.result.rg_nm if quantity == 'rg' else s.result.a2_mol_mL_per_g2
            if (mw is not None and np.isfinite(mw) and mw > 0
                    and y is not None and np.isfinite(y) and y > 0):
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
        uncalibrated = False
        excluded = 0
        # One point per (sample, molecular-weight fraction): a Mw series of one
        # polymer stored as several fractions of one sample becomes several points.
        for sid in sample_ids:
            s = self.workspace.samples[sid]
            for frac in self.sample_fractions(sid, 'sls'):
                res = s.result_for(frac)
                m = res.effective_mw()
                yv = (res.rg_nm if quantity == 'rg'
                      else res.rh_nm if quantity == 'rh'
                      else res.a2_mol_mL_per_g2)
                if not (m is not None and np.isfinite(m) and m > 0
                        and yv is not None and np.isfinite(yv) and yv > 0):
                    excluded += 1
                    continue
                ids.append(sid)
                labels.append(self._sample_label(sid, frac))
                mw.append(float(m))
                y.append(float(yv))
                if res.mw_source != 'user' and res.calibrated is False:
                    uncalibrated = True
        fit = fit_power_law(mw, y)
        return ScalingData(quantity=quantity, fit=fit, sample_ids=ids, labels=labels,
                           mw=mw, y=y, any_uncalibrated_mw=uncalibrated,
                           n_excluded=excluded)

    def _sample_label(self, sample_id: str,
                      fraction: Optional[str] = None) -> str:
        s = self.workspace.samples[sample_id]
        if s.polymer_name and s.solvent_name and s.temperature_K is not None \
                and np.isfinite(s.temperature_K):
            base = f'{s.polymer_name} / {s.solvent_name} @ {s.temperature_K:g} K'
        else:
            base = sample_id
        return f'{base} [{fraction}]' if fraction is not None else base

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
        ref = self.workspace.solvent_reference(sample_id)
        if ref is not None:
            candidates.append(ref)
        meas = [lm.build() for lm in candidates
                if lm.committed_params.get('concentration_g_per_mL') == 0]
        if not meas:
            raise ValueError(
                "I·sin θ shows only solvent-reference (c = 0) curves, and this "
                "sample has none. Load or assign a c = 0 SLS measurement.")
        res = util.i_sin_theta(meas, mode=mode)
        self.results[('i_sin_theta', sample_id, mode)] = res
        return res

    def generate_synthetic(self, population_specs: List[Dict[str, float]], **kwargs):
        """Generate a synthetic correlogram from a list of population specs
        ({'rh_nm', 'weight', 'spread_cv'}) plus generate_synthetic_correlogram
        keyword args (angle_deg, wavelength_nm, ... beta, noise_level, n_points,
        output_form, seed). Returns a SyntheticCorrelogramResult."""
        pops = [
            util.SyntheticPopulation(
                rh_nm=float(p['rh_nm']), weight=float(p['weight']),
                spread_cv=float(p.get('spread_cv', 0.0)))
            for p in population_specs
        ]
        res = util.generate_synthetic_correlogram(pops, **kwargs)
        self.results[('synthetic',)] = res
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
    def export_correlogram_fit(self, item_id: str, result, file_path: str) -> str:
        """Export a parametric DLS fit (cumulant / single / double / KWW)."""
        m = self.workspace.measurements[item_id].build()
        return exporter.export_correlogram_fit(m, result, file_path)

    def export_distribution(self, result, file_path: str, axis: str = 'rh') -> str:
        """Export an NNLS / CONTIN / lognormal distribution."""
        d = getattr(result, 'distribution', result)
        return exporter.export_distribution(d, file_path, axis=axis)

    def export_gamma_q2(self, result, file_path: str) -> str:
        return exporter.export_gamma_q2(result, file_path)

    def export_dls_summary(self, file_path: str,
                           ticked_only_ids: Optional[List[str]] = None) -> str:
        """Export the DLS Summary store as ONE long/tidy CSV: one row per result
        or distribution peak (NOT the wide on-screen shape), with separate numeric
        columns so Origin/pandas can plot them. The per-measurement rows honour the
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
            records.append({
                'sample': self._sample_label(sid) if sid else '',
                'measurement': self._measurement_label(row.item_id),
                'method': row.method,
                'rh': row.rh_nm, 'rh_se': row.rh_se, 'pdi': row.pdi,
                'int_pct': None if row.int_fraction is None
                else row.int_fraction * 100.0,
                'rh_fast': row.rh_fast_nm, 'rh_slow': row.rh_slow_nm,
                'rh_type': 'apparent', 'is_average': False,
                'from': row.label,
            })
        # Table 2 -- sample-level Rh (averages, Gamma-q^2, D-c).
        sample_rows = sorted(
            self.workspace.sample_rh_rows.values(),
            key=lambda r: (r.sample_id, r.source_kind, r.source_set))
        for row in sample_rows:
            records.append({
                'sample': self._sample_label(row.sample_id, row.fraction)
                if row.sample_id in self.workspace.samples else row.sample_id,
                'measurement': '', 'method': row.source_kind,
                'rh': row.rh_nm, 'rh_se': row.rh_se, 'pdi': None,
                'int_pct': None, 'rh_fast': None, 'rh_slow': None,
                'rh_type': row.rh_type_label,
                'is_average': row.source_kind == 'replicate_avg',
                'from': row.from_label,
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

    def export_single_angle(self, result, file_path: str) -> str:
        return exporter.export_single_angle(result, file_path)

    def export_calibration_free_a2(self, result, file_path: str) -> str:
        return exporter.export_calibration_free_a2(result, file_path)

    def export_rayleigh_series(self, results, file_path: str) -> str:
        return exporter.export_rayleigh_series(results, file_path)

    def export_scaling(self, scaling_data: 'ScalingData', file_path: str) -> str:
        """Export a cross-sample scaling plot (Rg/Rh/A2 vs Mw) + its power-law fit."""
        sd = scaling_data
        return exporter.export_scaling(sd.quantity, sd.labels, sd.mw, sd.y,
                                       sd.fit, file_path)

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
        return synth.build_multi_angle_dls(self._pops(population_specs), **kw)

    def synth_sls_set(self, *, mw: float, rg_nm: float, a2_mol_mL_per_g2: float,
                      angles_deg, concentrations_g_per_mL, wavelength_nm: float,
                      temperature_K: float, n_solvent: float, dn_dc: float,
                      calibrated: bool, geometry: Optional[str] = None,
                      solvent_intensity_90: float = synth.DEFAULT_SOLVENT_INTENSITY_90,
                      noise_level: float = 0.0, seed=None, polymer_name: str = '',
                      solvent_name: str = '', label: str = '', kind: str = 'zimm'):
        """An SLS intensity set (full Zimm grid or single-angle/-concentration slice).

        `calibrated` only controls whether a SAVED Brookhaven file carries the
        calibration metadata (and whether a workspace injection prefills the session
        calibration); the intensities are generated the same way either way.
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
            polymer_name=polymer_name, solvent_name=solvent_name, label=label, kind=kind)

    # ---- save a built artifact to its loadable instrument file ----
    def save_synth_trace(self, trace, path: str, count_rate_unit: str = 'kcps') -> str:
        return synth.write_trace_csv(path, trace, count_rate_unit=count_rate_unit)

    def save_synth_multi_angle_dls(self, dls, path: str,
                                   viscosity_cp: Optional[float] = None) -> str:
        return synth.write_alv_asc(path, dls, viscosity_cp=viscosity_cp)

    def save_synth_sls_set(self, sls, path: str) -> str:
        return synth.write_brookhaven_sls(path, sls)

    def generate_full_test_set(self, out_dir: str, profile_name: str) -> List[str]:
        """Regenerate a whole test-data/Synthetic * folder (data + parameters.txt)."""
        return synth.generate_full_dataset(out_dir, profile_name)

    # ---- inject a built artifact into the workspace as analysable data ----
    def inject_correlogram(self, result, *, polymer_name: str, solvent_name: str,
                           concentration_g_per_mL: float, temperature_K: float,
                           viscosity_Pa_s: Optional[float] = None) -> str:
        """Add a generated correlogram to the workspace as a DLS measurement.

        Assumes the generated signal is g2-1 (the analysable form). Returns the
        new item_id; the GUI should refresh the navigator + Data tab afterwards.
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
        return self.add_loaded('dls', raw, params)

    def inject_multi_angle_dls(self, dls, *, polymer_name: str, solvent_name: str,
                               add_traces: bool = True) -> List[str]:
        """Add a multi-angle DLS set as one DLS measurement per angle (and, by
        default, each angle's count-rate trace, linked back to its measurement)."""
        delay = [float(x) for x in dls.delay_times_s]
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
            iid = self.add_loaded('dls', raw, params)
            ids.append(iid)
            if add_traces:
                self.add_trace(
                    [float(x) for x in dls.trace_times_s],
                    [float(x) for x in dls.trace_cps[a]],
                    sample_label=f'{dls.label} @ {a:g}°', measurement_id=iid)
        return ids

    def inject_sls_set(self, sls, *, polymer_name: str, solvent_name: str,
                       prefill_calibration: bool = False) -> List[str]:
        """Add an SLS set as one SLS measurement per concentration (the c = 0 entry
        groups as the sample's solvent reference). With `prefill_calibration`, and if
        the set is calibrated, also set the session calibration to the generator's
        standard so an immediate Zimm run gives an absolute Mw."""
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
                dn_dc_mL_per_g=float(sls.dn_dc))
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

    def add_trace_from_preview(self, preview, source_path=None) -> str:
        """Build an IntensityTrace from a parser TraceFilePreview and store it.

        The preview must be ready (its units confirmed, for the generic parser);
        instrument parsers like the ALV one return ready previews."""
        trace = preview.build()
        return self.add_trace(
            trace.times_s, trace.count_rates_cps,
            sample_label=trace.sample_label,
            measurement_id=trace.measurement_id,
            source_path=source_path or getattr(preview, 'source_file', None))

    def traces(self) -> List[LoadedTrace]:
        return list(self.workspace.traces.values())

    def remove_trace(self, trace_id: str) -> None:
        self.workspace.traces.pop(trace_id, None)

    def remove_measurements(self, item_ids) -> None:
        """Remove one or more measurements from the workspace and purge every bit
        of state keyed to them: their cached results, and -- for any sample that
        becomes empty as a result -- that sample's results, SLS mask, and
        per-sample calibration override. Safe to call with item_ids that no longer
        exist."""
        ids = {iid for iid in item_ids if iid in self.workspace.measurements}
        if not ids:
            return
        samples_before = set(self.workspace.samples.keys())
        for iid in ids:
            self.workspace.remove_measurement(iid)
        vanished = samples_before - set(self.workspace.samples.keys())
        # A results key is ('kind', owner_id, ...); owner_id is an item_id for
        # per-measurement results (DLS) or a sample_id for per-sample results (SLS).
        dead = ids | vanished
        self.results = {k: v for k, v in self.results.items()
                        if not (len(k) > 1 and k[1] in dead)}
        if vanished:
            # sls_masks is keyed by (sample_id, fraction); drop every fraction of a
            # vanished sample.
            self.sls_masks = {k: v for k, v in self.sls_masks.items()
                              if k[0] not in vanished}
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

    # --- richer trace diagnostics (thin engine wrappers; seed from settings) ---
    def flag_trace_outliers(self, trace_id: str, k: Optional[float] = None):
        """Points outside mean ± k·√mean (Poisson shot-noise band)."""
        if k is None:
            k = _TRACE_OUTLIER_K
        return trace_engine.flag_outliers(self.build_trace(trace_id), k=k)

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
        # Masks are keyed by (sample_id, fraction); serialise as a list since JSON
        # object keys can't be tuples.
        payload['sls_masks'] = [
            {'sample_id': sid, 'fraction': frac, 'mask': m.to_dict()}
            for (sid, frac), m in self.sls_masks.items() if not m.is_empty()]
        with open(file_path, 'w') as fh:
            json.dump(payload, fh, indent=2, allow_nan=True)
        return file_path

    def load_session(self, file_path: str) -> None:
        with open(file_path) as fh:
            payload = json.load(fh)
        self.workspace = Workspace.from_dict(payload)
        self.calibration_committed = CalibrationState.from_dict(
            payload.get('calibration', {}))
        self.calibration_working = CalibrationState(
            **self.calibration_committed.to_dict())
        self.sample_calibration_committed = {
            sid: CalibrationState.from_dict(d)
            for sid, d in payload.get('sample_calibrations', {}).items()}
        self.sample_calibration_working = {
            sid: CalibrationState(**cc.to_dict())
            for sid, cc in self.sample_calibration_committed.items()}
        raw_masks = payload.get('sls_masks', [])
        if isinstance(raw_masks, list):     # current (sample, fraction) format
            self.sls_masks = {
                (e['sample_id'], e.get('fraction')): SLSMask.from_dict(e['mask'])
                for e in raw_masks}
        else:                               # back-compat: {sample_id: mask}
            self.sls_masks = {
                (sid, None): SLSMask.from_dict(d)
                for sid, d in raw_masks.items()}
        self.results.clear()
        # The undo history belongs to the previous session's measurements; drop it so
        # Undo can't step back into a no-longer-loaded state.
        self._commit_history.clear()

    def source_paths(self) -> Dict[str, Optional[str]]:
        """item_id -> original file path, for an optional reload-from-source."""
        return {i: m.source_path for i, m in self.workspace.measurements.items()}
