"""
core/workspace.py
=================

The workspace data model: the framework-agnostic state that the controller
manages and the GUI displays. Contains NO GUI code and NO matplotlib; it is plain
Python so it can be unit-tested headless and serialized to a session file.

A loaded data file becomes a `LoadedMeasurement` -- the raw arrays plus an
editable parameter set (working) and the parameter set last used for analysis
(committed). Measurements are grouped into `Sample`s by their sample identity
(polymer, solvent, rounded temperature); concentration and angle are axes WITHIN
a sample, not identity. Each sample carries labeled DLS / SLS / solvent-reference
slots and a `SampleResult` holding Mw, Rg, A2, Rh -- which may be computed by the
engine OR entered by hand (e.g. a PVP molecular weight characterized in water, to
avoid the co-solvent adsorption bias of a reline measurement).

Grouping is the hybrid model: the sample key PROPOSES a grouping, and a manual
override map lets the user reassign a measurement to a different sample.

Serialization is self-contained (the numerical data is embedded) and also records
the original source path, so a session can either stand alone or offer
reload-from-source. JSON is used rather than pickle: inspectable, portable, and
not a security or version-fragility risk.

Change history
--------------
2026-06-13  workspace.py v1: LoadedMeasurement, SampleResult, Sample, Workspace;
            hybrid grouping; JSON session serialize/deserialize.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field, fields, asdict, MISSING
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.data_models import (
    DLSMeasurement, IntensityTrace, SLSMeasurement,
    casefold_identity_name, round_temperature_for_identity, usable_rate)


def _known_only(cls, d: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Keep only keys that are fields of dataclass `cls`. Makes a `from_dict` forward-
    compatible: a session written by a NEWER build (with an extra field) loads in
    an older one — the unknown key is ignored rather than raising `TypeError`. Mirrors
    `SettingsState.from_dict` / `Workspace.from_dict`."""
    known = {f.name for f in fields(cls)}
    return {k: v for k, v in (d or {}).items() if k in known}


# ===========================================================================
# Loaded measurement: raw data + editable parameters (working / committed)
# ===========================================================================

# The physical parameters a user confirms or edits for a measurement. Kept as a
# plain dict so change-tracking is a dict comparison and serialization is trivial.
_DLS_PARAM_KEYS = (
    'label',
    'polymer_name', 'solvent_name', 'concentration_g_per_mL', 'temperature_K',
    'angle_deg', 'wavelength_nm', 'solvent_refractive_index', 'viscosity_Pa_s',
    # Mean photon count rate for this run, in cps. PER-MEASUREMENT (each angle has
    # its own count rate), so it is deliberately absent from the controller's
    # shared-parameter set. The program may OFFER a value -- from a linked
    # count-rate trace or from the instrument file's own scalar -- but it reaches
    # an analysis only through the user's explicit commit, and a hand-entered value
    # is never overwritten (invariant 3; the source tag below records which). It
    # reaches the engine on the measurement object -- `LoadedMeasurement.build()`
    # reads the COMMITTED value by name, like viscosity -- and nowhere else.
    'mean_count_rate_cps',
    # Optional, hand-entered (like dn/dc on the SLS side): consumed only by the
    # out-of-Rayleigh (RDG) distribution reweighting; blank for pure DLS runs.
    'solute_refractive_index',
    'mw_fraction', 'analyzer_geometry', 'chain_type',
)
_SLS_PARAM_KEYS = (
    'label',
    'polymer_name', 'solvent_name', 'concentration_g_per_mL', 'temperature_K',
    'wavelength_nm', 'solvent_refractive_index', 'dn_dc_mL_per_g',
    # Optional, hand-entered (like dn/dc): consumed only by the full-Mie form-factor P(q)
    # fit (model='mie'), which forms m = n_solute/n_solvent; blank for every other SLS fit.
    # For the core-shell fit, solute_refractive_index is the CORE index and
    # shell_refractive_index the SHELL index (m1/m2); both blank for every non-core-shell fit.
    'solute_refractive_index', 'shell_refractive_index',
    'mw_fraction', 'chain_type',
    # Which c = 0 solvent blank this series is referenced against: the item_id of a
    # blank in the SAME sample, or None for the automatic pairing (the blank from
    # this series' own source file, else the sample's reference slot -- see
    # Workspace.solvent_reference_for). PER-SERIES: an acquisition defines the pair,
    # so a format that ships one blank per file, or a blank as a separate file, needs
    # a per-series link; it is an item_id, never a filename, so a rename cannot
    # break it. Stored on the CONSUMER because one blank serves many series.
    'solvent_blank_item_id',
    # OPTIONAL, hand-entered (like dn/dc -- never proposed, defaulted or estimated,
    # invariant 3): the solvent's intensity AT 90 degrees, in the loaded intensity
    # column's own units before dark-count subtraction -- the SLS calibration panel's
    # dark-count convention, which it is compared against, NOT the Data tab's
    # unit-converted `intensity` quantity. Consulted ONLY where
    # `solvent_reference_for` returns no blank; a measured blank always wins. Empty and
    # zero both mean no solvent subtraction. A run using it is flagged on every output
    # (sls.RayleighRatioResult.blank_measured).
    'solvent_blank_intensity_90deg',
)

# Sidecar provenance tags for solvent-property autofill: which SOURCE produced the
# refractive index / viscosity ('user' vs 'library:primary'). Deliberately NOT in
# the *_PARAM_KEYS tuples above -- so they never become Data-table rows, never
# reach LoadedMeasurement.build() (which reads named keys, not **params -- the
# invariant-3 linchpin), and are rejected by set_param/set_shared_param. They ride
# in the param dicts only via the dedicated controller writers, and are excluded
# from dirty-tracking below so a source-only flip never raises a phantom pending.
_PROVENANCE_KEYS = frozenset({
    'solvent_refractive_index_source', 'viscosity_Pa_s_source',
    # Which baseline normalized a raw-counts correlogram into g2-1, and what the
    # alternative was. Written at load time by the Brookhaven .dat path (the only
    # format that ships raw counts plus two baselines); absent for every other
    # format, whose files already store g2-1. It is here rather than in
    # _DLS_PARAM_KEYS because it is a record of what happened, not a knob: editing
    # the string would not re-derive anything, since the workspace keeps the
    # correlogram and not the raw counts it came from.
    'correlogram_baseline_source',
    # Which source produced the mean count rate above:
    #   'trace:mean'           -- the plain mean of the linked count-rate trace
    #   'trace:sigma_clipped'  -- that trace's 3-sigma-clipped mean
    #   'vendor'               -- the instrument file's own scalar
    #   'generator'            -- a synthetic draw's own rate (its noise model's input)
    #   'user'                 -- typed by hand; never silently overwritten
    # The method rides inside the tag rather than in a second sidecar key, so the
    # two can never drift out of sync; the family is the part before the colon.
    'mean_count_rate_cps_source',
    # What the instrument FILE said, kept verbatim as a record even when the user
    # commits a different value: the vendor's own mean count rate (cps) and the
    # nominal run length (s). Records, not knobs -- the same argument as
    # correlogram_baseline_source: editing them would not re-derive anything.
    # The nominal duration is NOT a substitute for the trace's own time span (an
    # ALV file reports 10 s for a trace section spanning 9.88 s).
    'vendor_mean_count_rate_cps',
    'vendor_duration_s',
    # How a DERIVED static (SLS) point was made: which mean-count-rate sources its
    # contributing DLS runs carried, and any disclosure raised while collating them.
    # Records, not knobs -- the contributing run ids live in LoadedMeasurement's
    # `derived_from`, and the point is a SNAPSHOT (owner-ruled 2026-08-14): editing
    # a source run never re-derives it, so nothing here would recompute if changed.
    'derived_intensity_source',
    'derived_intensity_notes',
})
# HOW a series' solvent blank was chosen ('auto:same-file' / 'auto:sample' / 'user')
# is deliberately NOT a stored sidecar: it is a pure function of the committed
# `solvent_blank_item_id` and the source paths, recomputed by
# Workspace.solvent_reference_for whenever it is needed, so no fit path has to write
# into a measurement's params from the worker thread.

# The vendor RECORD fields a parser preview can carry into a measurement's params.
# A roster, so a loader copies the family rather than a hand-listed pair (rule 11).
_PREVIEW_RECORD_KEYS = ('vendor_mean_count_rate_cps', 'vendor_duration_s')


def seed_vendor_records(params: Dict[str, Any], preview: Any) -> Dict[str, Any]:
    """Copy a parser preview's vendor RECORD fields into a params dict.

    The instrument file's own mean count rate and run length are treated exactly
    like the angle and temperature a vendor header proposes: the parser offers
    them, they are labelled with their origin, and they reach an analysis only
    through the user's explicit commit. A field the preview does not carry is left
    absent rather than written as None, so "the file did not say" and "the file
    said nothing useful" stay distinguishable.

    Mutates and returns `params`.
    """
    for key in _PREVIEW_RECORD_KEYS:
        value = getattr(preview, key, None)
        if value is not None:
            params[key] = value
    return params


@dataclass
class LoadedMeasurement:
    """One loaded file: raw arrays plus working and committed parameter sets.

    `working_params` is what the user is currently editing (the table contents);
    `committed_params` is what the last analysis actually used. A measurement is
    "dirty" when they differ -- the controller uses this to highlight changed
    fields and to show that an update is pending.
    """
    item_id: str
    kind: str                          # 'dls' or 'sls'
    # The raw arrays (lists, for serialization) and the one scalar instrument fact beside
    # them, the run duration `duration_s` (seconds) when the file states it.
    raw: Dict[str, Any]
    working_params: Dict[str, Any]
    committed_params: Dict[str, Any]
    source_path: Optional[str] = None
    # ---- provenance for DERIVED measurements (not loaded from a file) ----
    # A measurement the program synthesized from others (e.g. the channel-by-channel
    # mean of a replicate set) records what it came from. `derived_kind` tags HOW it
    # was made ('replicate_average'); `derived_from` lists the source item_ids.
    # Both None for an ordinary loaded measurement.
    derived_from: Optional[List[str]] = None
    derived_kind: Optional[str] = None

    # ---- change tracking ----
    def is_dirty(self) -> bool:
        """True if any non-provenance working parameter differs from committed.

        Provenance source tags (_PROVENANCE_KEYS) are ignored: a re-derive that
        only flips a source tag must not raise a phantom "changes pending".
        """
        keys = (set(self.working_params) | set(self.committed_params)) - _PROVENANCE_KEYS
        return any(self.working_params.get(k) != self.committed_params.get(k)
                   for k in keys)

    def dirty_keys(self) -> List[str]:
        """The parameter names whose working value differs from committed.

        Excludes provenance source tags (_PROVENANCE_KEYS)."""
        keys = (set(self.working_params) | set(self.committed_params)) - _PROVENANCE_KEYS
        return sorted(k for k in keys
                      if self.working_params.get(k) != self.committed_params.get(k))

    def commit(self) -> None:
        """Adopt the working parameters as committed."""
        self.committed_params = dict(self.working_params)

    def revert(self) -> None:
        """Discard edits: restore working parameters from committed."""
        self.working_params = dict(self.committed_params)

    # ---- build engine objects from the committed parameters ----
    def build(self) -> Any:
        """Construct the DLS/SLS measurement object from the committed params.

        Uses committed (not working) parameters: analysis runs on what has been
        confirmed, never on un-applied edits.

        INVARIANT-3 LINCHPIN: this reads *named* keys off the committed dict (never
        `**params`). That is what lets the param dict carry sidecar provenance keys
        (e.g. solvent_refractive_index_source) that never reach the pure engine. Do
        NOT switch this to `**p` / dict-splat construction — it would silently feed
        every stray key into the measurement and break the provenance separation.
        """
        p = self.committed_params
        if self.kind == 'dls':
            return DLSMeasurement(
                delay_times_s=np.asarray(self.raw['delay_times_s'], dtype=float),
                correlogram=np.asarray(self.raw['correlogram'], dtype=float),
                polymer_name=p['polymer_name'], solvent_name=p['solvent_name'],
                concentration_g_per_mL=p.get('concentration_g_per_mL'),
                temperature_K=p['temperature_K'], angle_deg=p['angle_deg'],
                wavelength_nm=p['wavelength_nm'],
                solvent_refractive_index=p['solvent_refractive_index'],
                viscosity_Pa_s=p.get('viscosity_Pa_s'),
                # Optional solute RI (RDG reweighting); .get keeps None the default.
                solute_refractive_index=p.get('solute_refractive_index'),
                mw_fraction=p.get('mw_fraction'),
                # Forward-ready (depolarized dynamic analysis); .get keeps it from
                # being dropped once the VV/VH tagging is wired through the params.
                analyzer_geometry=p.get('analyzer_geometry'),
                # The fit's instrument inputs. The per-channel sample time is a
                # parser-written instrument fact kept in `raw` beside the lags (absent
                # when the file states none: the engine derives a stand-in and says so).
                # The count rate is the user's COMMITTED, provenance-tagged parameter;
                # anything that is not a usable rate -- nothing committed, a legacy 0.0
                # -- goes as None, the classical limit the fit result discloses, never
                # as a value the data model would refuse.
                sample_times_s=(np.asarray(self.raw['sample_times_s'], dtype=float)
                                if self.raw.get('sample_times_s') is not None else None),
                mean_count_rate_cps=usable_rate(p.get('mean_count_rate_cps')),
                # The third instrument input, a parser-written fact in `raw` like the
                # sample times: absent when the file states no run duration.
                duration_s=self.raw.get('duration_s'),
            )
        elif self.kind == 'sls':
            return SLSMeasurement(
                angles_deg=np.asarray(self.raw['angles_deg'], dtype=float),
                intensities=np.asarray(self.raw['intensities'], dtype=float),
                polymer_name=p['polymer_name'], solvent_name=p['solvent_name'],
                concentration_g_per_mL=p['concentration_g_per_mL'],
                temperature_K=p['temperature_K'], wavelength_nm=p['wavelength_nm'],
                solvent_refractive_index=p['solvent_refractive_index'],
                dn_dc_mL_per_g=p.get('dn_dc_mL_per_g'),
                # Optional solute RI (full-Mie / core-shell core index); .get keeps None default.
                solute_refractive_index=p.get('solute_refractive_index'),
                # Optional shell RI (core-shell shell index); .get keeps None the default.
                shell_refractive_index=p.get('shell_refractive_index'),
                mw_fraction=p.get('mw_fraction'),
                # Forward-ready: None until the depolarization wiring (GUI VV/VH/VU
                # picker + param capture) lands; .get keeps it from being dropped once set.
                analyzer_geometry=p.get('analyzer_geometry'),
            )
        raise ValueError(f"Unknown measurement kind {self.kind!r}.")

    # ---- serialization ----
    def to_dict(self) -> Dict[str, Any]:
        return {
            'item_id': self.item_id, 'kind': self.kind, 'raw': self.raw,
            'working_params': self.working_params,
            'committed_params': self.committed_params,
            'source_path': self.source_path,
            'derived_from': self.derived_from,
            'derived_kind': self.derived_kind,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'LoadedMeasurement':
        return cls(
            item_id=d['item_id'], kind=d['kind'], raw=d['raw'],
            working_params=dict(d['working_params']),
            committed_params=dict(d['committed_params']),
            source_path=d.get('source_path'),
            # .get keeps sessions written before these fields existed loadable.
            derived_from=d.get('derived_from'),
            derived_kind=d.get('derived_kind'),
        )


@dataclass
class LoadedTrace:
    """One loaded intensity (count-rate) trace, stored in canonical units (s, cps).

    Traces live in their own workspace collection, NOT among the DLS/SLS
    measurements: a trace carries no polymer/solvent/temperature, so it takes no
    part in sample grouping. `measurement_id` optionally back-references the DLS
    item whose count-rate history this is (e.g. an ALV angle's trace beside its
    correlogram); it may be None for a standalone trace.
    """
    trace_id: str
    times_s: List[float]
    count_rates_cps: List[float]
    sample_label: Optional[str] = None
    measurement_id: Optional[str] = None
    source_path: Optional[str] = None

    def build(self) -> IntensityTrace:
        return IntensityTrace(
            times_s=np.asarray(self.times_s, dtype=float),
            count_rates_cps=np.asarray(self.count_rates_cps, dtype=float),
            sample_label=self.sample_label,
            measurement_id=self.measurement_id,
            source_file=self.source_path,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'trace_id': self.trace_id,
            'times_s': list(self.times_s),
            'count_rates_cps': list(self.count_rates_cps),
            'sample_label': self.sample_label,
            'measurement_id': self.measurement_id,
            'source_path': self.source_path,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'LoadedTrace':
        return cls(
            trace_id=d['trace_id'],
            times_s=list(d['times_s']),
            count_rates_cps=list(d['count_rates_cps']),
            sample_label=d.get('sample_label'),
            measurement_id=d.get('measurement_id'),
            source_path=d.get('source_path'),
        )


# ===========================================================================
# Sample result: Mw / Rg / A2 / Rh, computed or user-supplied
# ===========================================================================

def blank_label(lm: 'LoadedMeasurement') -> str:
    """How a solvent blank is named to the user: the last TWO components of its
    source path ('folder/file.csv'), else its label, else its item_id.

    A basename alone cannot discriminate in the layout this exists for -- one
    angular-scan export per sample folder, every one named the same -- and two
    components are short enough for a Comments cell. Used by the Data tab's dropdown
    and by the export's per-series blank record, so the two always agree.
    """
    if lm.source_path:
        norm = str(lm.source_path).replace('\\', '/').rstrip('/')
        parts = [p for p in norm.split('/') if p]
        if parts:
            return '/'.join(parts[-2:])
    return lm.committed_params.get('label') or lm.item_id


def is_arbitrary_scale(value: Optional[float], source: str,
                       calibrated: Optional[bool],
                       dn_dc_supplied: Optional[bool] = None) -> bool:
    """Is a stored scale-dependent quantity on an arbitrary scale?

    THE one body of this predicate, for every "may I print, or divide by, this
    value?" decision in the program. It applies to any quantity whose scale rides on
    the Rayleigh calibration or on the optical constant K -- Mw and the absolute A2
    today -- so it takes the (value, source, calibrated, dn_dc_supplied) tuple rather
    than reading field names: `SampleResult` spells the flags asymmetrically
    (`calibrated`/`dn_dc_supplied` for Mw, `a2_calibrated`/`a2_dn_dc_supplied` for A2),
    and a field-name-derived helper could not serve both.

    TWO independent ways the scale can be unknown, and either is enough:

    * `calibrated is False` -- no calibration constant, so dR is on an arbitrary
      Rayleigh scale.
    * `dn_dc_supplied is False` -- no refractive-index increment, so K was set to 1
      and Kc/dR is on an arbitrary OPTICAL scale.

    They are not the same absence and neither subsumes the other: a fully calibrated
    run with no dn/dc still yields an Mw wrong by orders of magnitude. Callers that
    render a REASON must therefore read the two flags, not this verdict -- this
    answers "is the scale unknown?", never "why".

    Three load-bearing edges, all deliberate:

    * `source == 'user'` is NEVER arbitrary. A hand-entered value is a trusted
      external number (e.g. a PVP Mw characterized in water), independent of whether
      this session's SLS run happened to be calibrated.
    * `calibrated is None` / `dn_dc_supplied is None` -- never run, or unknown -- is
      NOT arbitrary. Only a run that KNOWS the scale was unavailable sets a flag False;
      absence of evidence is not a scale verdict, and treating it as one would blank
      freshly-loaded sessions. A session that records no verdict for one of
      these flags therefore reads as "scale known", which is also what keeps older
      session files loading unchanged.
    * A missing or non-finite value is not arbitrary either: there is no number there
      to be on a scale. Callers that must also refuse a non-finite value check that
      separately -- this predicate answers "is the scale unknown?", not "is it usable?".

    `tests/test_arbitrary_scale_predicate_family.py` gates the single-body property by
    scanning product source for the predicate's SHAPE, so a hand-written copy fails there
    rather than shipping and drifting.
    """
    return (value is not None and math.isfinite(value) and source != 'user'
            and (calibrated is False or dn_dc_supplied is False))


# The deliberately-chosen provenance tiers -- the set a passive refresh must spare.
# THE one spelling of the tier set; guards and display sites import this rather than
# re-listing the tiers, so the set has a single definition (see the SampleResult
# docstring for what each tier means and which events may replace it).
DELIBERATE_TIERS = ('user', 'picked')


@dataclass
class SampleResult:
    """Analysis outputs attached to a sample.

    Each quantity records its provenance TIER: 'computed' (an auto-selected
    analysis result), 'picked' (a result the user explicitly chose in
    Cross-Sample), or 'user' (entered by hand). This docstring is THE normative
    statement of the tier rule; every other statement in the tree points here
    rather than re-deriving it.

    * A 'user' value is entered from outside the program and is never removed or
      overwritten except by the user.
    * A 'picked' value is a choice of SOURCE among results the program computed:
      it is inert against a passive refresh, and is replaced or cleared by a
      deliberate act -- an explicit Run, or one of the three clear-and-warn
      Settings switches taking the source it names: the cumulant inputs (method,
      fit cutoff, residual weighting; the order clears the narrower set fitted at
      it), the skipped lag channels (every DLS kind), and the SE estimator.
      Exactly those three switches clear picks, and this sentence names them
      rather than a class of settings: a wider "any settings change that
      invalidates" form would promise clears the program does not perform
      (owner-ruled). Where a deliberate act destroys a pick, the user is asked
      first and can back out.

    The three event classes, so a new call site can be classified:

        event            | 'computed' | 'picked'                    | 'user'
        passive refresh  | replaced   | spared                      | spared
          (auto_select_* on a Cross-Sample refresh)
        deliberate act   | replaced   | replaced or cleared; asked  | spared
          (an explicit Run; the cumulant-input,    first where a clear
           skipped-channels or SE-estimator        destroys the pick
           switch)
        user action      | replaced   | replaced                    | replaced
          (a fresh pick; a manual entry)

    The tier set a passive refresh spares is spelled once as DELIBERATE_TIERS
    above. The 'user' override matters for systems where the measured Mw is
    biased -- e.g. PVP in a deep eutectic solvent, where co-solvent adsorption
    skews Mw, so the trustworthy value is the one characterized in water and
    entered manually. Cross-sample scaling plots (Rg-Mw, A2-Mw) should prefer
    the user value when present.
    """
    mw_g_per_mol: Optional[float] = None
    mw_source: str = 'computed'        # 'computed' (auto-default) | 'picked' (explicit) | 'user' (manual)
    mw_apparent: Optional[bool] = None  # True = single-condition (Debye/Guinier/single-angle); False = Zimm/Berry
    mw_label: str = ''                  # plain-language provenance of mw_g_per_mol
    mw_se: Optional[float] = None       # statistical SE (None for user/no-SE sources)
    mw_se_estimator: Optional[str] = None  # covariance estimator behind mw_se (copied from the
    #                                        source SLS fit: 'hc3'/'ols'; None for a user/no-SE Mw)
    mw_model_dependent: bool = False    # True = Mw came from a form-factor P(q) fit (assumes a shape);
    #                                     orthogonal to mw_apparent (model-dependence axis)
    mw_model: str = ''                  # the assumed shape when mw_model_dependent (e.g. 'sphere')
    mw_degenerate: bool = False         # True = Mw came from a DEGENERATE form-factor fit the data can't
    #                                     resolve to one shape (a user-selected solution); always implies
    #                                     mw_model_dependent, flagged one tier stronger (review-01; rule 10)
    rg_nm: Optional[float] = None
    rg_source: str = 'computed'
    rg_apparent: Optional[bool] = None  # True = single-conc (Debye/Guinier); False = Zimm/Berry (thermodynamic)
    rg_label: str = ''                  # plain-language provenance of rg_nm (shown in Cross-Sample)
    rg_se: Optional[float] = None
    rg_se_estimator: Optional[str] = None  # covariance estimator behind rg_se (copied from source; None if no SE)
    rg_model_dependent: bool = False    # True = Rg came from a form-factor P(q) fit (assumes a shape),
    #                                     right only if that shape is right; orthogonal to rg_apparent
    #                                     (model-dependence axis). False = slope-derived (Zimm/Guinier).
    rg_model: str = ''                  # the assumed shape when rg_model_dependent (e.g. 'sphere')
    rg_degenerate: bool = False         # True = Rg came from a DEGENERATE form-factor fit the data can't
    #                                     resolve to one shape (a user-selected solution); always implies
    #                                     rg_model_dependent, flagged one tier stronger (review-01; rule 10)
    a2_mol_mL_per_g2: Optional[float] = None
    a2_source: str = 'computed'
    a2_label: str = ''                  # plain-language provenance of a2_mol_mL_per_g2 (shown in Cross-Sample)
    a2_se: Optional[float] = None
    a2_se_estimator: Optional[str] = None  # covariance estimator behind a2_se (copied from source; None if no SE)
    a2_calibrated: Optional[bool] = None  # whether the SLS run that filled A2 was calibrated (A2 is scale-dependent)
    a2_dn_dc_supplied: Optional[bool] = None  # whether that run had a dn/dc (K=1 -> arbitrary optical scale)
    rh_nm: Optional[float] = None       # from DLS (infinite-dilution if available)
    rh_source: str = 'computed'
    rh_apparent: Optional[bool] = None  # True = apparent (single q or single c); False = thermodynamic (c->0)
    rh_label: str = ''                  # plain-language provenance of rh_nm (shown in Cross-Sample)
    rh_se: Optional[float] = None
    rh_se_estimator: Optional[str] = None  # covariance estimator behind rh_se for a REGRESSION source
    #                                        (gamma_q2/conc_extrap); None for the replicate-average SEM
    #                                        (not a regression SE) or a hand-entered Rh
    calibrated: Optional[bool] = None   # whether the SLS run that filled Mw was calibrated
    # The second, INDEPENDENT scale flag on Mw: whether that run had a refractive-index
    # increment. Without one K is 1, so Mw is on an arbitrary OPTICAL scale even when the
    # run was fully calibrated. Spelled asymmetrically from its A2 sibling for the same
    # reason `calibrated`/`a2_calibrated` are (see is_arbitrary_scale).
    dn_dc_supplied: Optional[bool] = None
    notes: str = ''
    # ---- the provenance KEY: which result is this value? ----
    # Machine-readable identity beside each `<q>_label`. `<q>_kind` names the producing
    # analysis ('dls_cumulant', 'sls_zimm', 'user'); `<q>_source_id` names the record within
    # it (an item_id, 'debye@0.001'). NOT the `<q>_source` fields above, which answer a
    # different question -- the provenance TIER ('computed'/'picked'/'user'), how deliberate
    # the choice was rather than what produced it (the tier rule: the class docstring).
    # Written only via `set_result_key`, read only via `result_key_matches`, which fails
    # closed. A label is prose for the user, not an identity: one value, two spellings.
    mw_kind: Optional[str] = None
    mw_source_id: Optional[str] = None
    rg_kind: Optional[str] = None
    rg_source_id: Optional[str] = None
    a2_kind: Optional[str] = None
    a2_source_id: Optional[str] = None
    rh_kind: Optional[str] = None
    rh_source_id: Optional[str] = None

    def set_mw(self, value: float, source: str = 'user') -> None:
        self.mw_g_per_mol = value
        self.mw_source = source

    def effective_mw(self) -> Optional[float]:
        """The Mw to use downstream (user value wins if present).

        Returns the RAW stored value: it carries no scale verdict, so a consumer that
        prints it, divides by it, or plots it must ask `mw_is_arbitrary_scale()` first.
        Deliberately NOT gated here -- the existing consumers disagree about the right
        response (the scaling plots and the architecture table FLAG an arbitrary-scale
        Mw and still use it; the calibration-free A2 WITHHOLDS), and folding the gate
        into the accessor would silently blank the first two."""
        return self.mw_g_per_mol

    def mw_is_arbitrary_scale(self) -> bool:
        """Is the stored Mw on an arbitrary (uncalibrated) scale? See
        `is_arbitrary_scale` for the semantics -- this only binds it to the Mw fields."""
        return is_arbitrary_scale(self.mw_g_per_mol, self.mw_source, self.calibrated,
                                  self.dn_dc_supplied)

    def a2_is_arbitrary_scale(self) -> bool:
        """Is the stored absolute A2 on an arbitrary (uncalibrated) scale? See
        `is_arbitrary_scale` for the semantics -- this only binds it to the A2 fields."""
        return is_arbitrary_scale(self.a2_mol_mL_per_g2, self.a2_source,
                                  self.a2_calibrated, self.a2_dn_dc_supplied)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SampleResult':
        return cls(**_known_only(cls, d))


# ---------------------------------------------------------------------------
# The provenance key: the ONE way to write and to test a result's identity
# ---------------------------------------------------------------------------
# The four quantities SampleResult carries a key for.
RESULT_QUANTITIES = ('mw', 'rg', 'a2', 'rh')

# Each quantity's VALUE field, spelled once ('mw' predates the uniform <q>_nm
# naming, so the field name cannot be derived from the quantity token).
RESULT_VALUE_FIELDS = {'mw': 'mw_g_per_mol', 'rg': 'rg_nm',
                       'a2': 'a2_mol_mL_per_g2', 'rh': 'rh_nm'}

# Mw's two scale flags predate the uniform `<q>_` prefix convention, so they are bare
# field names that reflection over `mw_*` cannot find. Spelled ONCE here because two
# separate places must special-case them -- clearing a bundle and merging one across a
# regroup -- and a flag left behind by either is exactly the stale-flag-beside-a-new-
# value defect invariant 8 forbids. Their A2 siblings need no entry: `a2_calibrated`
# and `a2_dn_dc_supplied` already carry the prefix.
MW_BARE_SCALE_FLAGS = ('calibrated', 'dn_dc_supplied')

# Each quantity's display name, spelled once for every dialog/label that names
# one (the SLS tab's confirms and the regroup ask both render these; two
# hand-maintained maps had already drifted to 'A2' vs 'A₂'). Lives here rather
# than in gui/ because the Qt-free controller composes dialog text too.
RESULT_QUANTITY_TITLES = {'mw': 'Mw', 'rg': 'Rg', 'a2': 'A₂', 'rh': 'Rh'}


def set_result_key(result: 'SampleResult', quantity: str,
                   kind: Optional[str], source_id: Optional[str]) -> None:
    """Write one quantity's provenance key. Pass (None, None) to clear it.

    The single writer, so the key cannot acquire a second spelling -- which is exactly how
    the label it replaces became unusable as an identity (the same value carried two
    sentences, and every sniff was written against one of them)."""
    if quantity not in RESULT_QUANTITIES:
        raise ValueError(f"quantity must be one of {RESULT_QUANTITIES}, got {quantity!r}.")
    setattr(result, f'{quantity}_kind', kind)
    setattr(result, f'{quantity}_source_id', source_id)


def reset_result_quantity(result: 'SampleResult', quantity: str) -> None:
    """Reset one quantity's whole bundle to `SampleResult`'s own field defaults.

    THE one definition of "clear a stored quantity" -- value, source tier, label,
    SE + estimator tag, model/apparent flags, and the provenance key together, so
    no clearing path can leave an SE, a calibration flag or a label stale beside
    an emptied value (invariant 8). Field-for-field to the dataclass defaults, NOT
    to a hand-picked "empty" state: `<q>_source` is a plain str defaulting to
    'computed', so writing None here would leave the sample in a shape a fresh one
    never has. The bundle is every `<q>_`-prefixed field, plus MW_BARE_SCALE_FLAGS for
    Mw (its two scale flags predate the prefix convention)."""
    if quantity not in RESULT_QUANTITIES:
        raise ValueError(f"quantity must be one of {RESULT_QUANTITIES}, got {quantity!r}.")
    prefix = f'{quantity}_'
    for f in fields(result):
        if (f.name.startswith(prefix)
                or (quantity == 'mw' and f.name in MW_BARE_SCALE_FLAGS)):
            # Every bundle field carries a plain default today; the factory arm keeps
            # this from writing the MISSING sentinel if one ever gains a mutable default.
            setattr(result, f.name,
                    f.default if f.default is not MISSING else f.default_factory())


def result_key_matches(result: 'SampleResult', quantity: str,
                       kind: Optional[str], source_id: Optional[str]) -> bool:
    """Is this stored value the one produced by (kind, source_id)?

    FAILS CLOSED on an unknown key. A session written before the key existed loads both
    fields as None, and an empty source_id means the producer never identified its record;
    neither is evidence of a match, so both answer False rather than colliding with every
    other keyless candidate. The caller's fallback for "unknown" must therefore be the safe
    branch -- don't preselect, don't clear."""
    stored_kind = getattr(result, f'{quantity}_kind', None)
    stored_id = getattr(result, f'{quantity}_source_id', None)
    if not stored_kind or not stored_id or not kind or not source_id:
        return False
    return stored_kind == kind and stored_id == source_id


def extrapolation_result_key(method: str) -> tuple:
    """The (kind, source_id) of a whole-ladder Zimm/Berry result.

    Shared for the same reason as `replicate_rh_key`: `run_zimm` writes Mw, Rg and A2
    straight onto the SampleResult without going through a candidate, while the three
    candidate builders mint the same identity separately -- six sites, one spelling."""
    return (f'sls_{method}', method)


def replicate_rh_key(source_set: str) -> tuple:
    """The (kind, source_id) of a replicate-average Rh.

    ONE computation behind both spellings: the Table-2 candidate built by
    `dls_rh_candidates` and the direct promotion written by `_write_replicate_rh`, which is
    the only promotion that does not come from a candidate. Two spellings here would
    reproduce the label defect one field over, which is the whole reason the key exists."""
    return ('dls_replicate_avg', f'replicate_avg@{source_set}')


# ===========================================================================
# DLS Summary-tab snapshot rows
# ===========================================================================

# The DURABLE display cache behind the DLS Summary tab. These hold ONLY the scalars each
# table row shows -- never correlograms, grids, or fit curves, which stay in the
# controller's ephemeral `results` cache. Being small and self-contained they round-trip
# cleanly through the session JSON, so the Summary table survives save/reload (unlike the
# per-measurement fits, which are recomputed on demand).

@dataclass
class MeasurementResultRow:
    """One per-measurement DLS scalar result (Summary Table 1).

    Keyed by (item_id, method, peak_index). Distribution methods write one row
    per resolved peak (peak_index 0, 1, ...); parametric methods write a single
    row (peak_index 0). Per-measurement DLS is always APPARENT -- a single q and
    a single concentration, neither extrapolated (invariant 7).

    NOT CARRIED HERE, by design: the analysis choices behind a cumulant row -- the
    amplitude cutoff and the residual weighting (`CumulantResult.fit_cutoff`,
    `CumulantResult.weighting`). They are Settings, not session state, so a row computed
    under one value and reloaded under another does not say which produced it; the
    correlogram-fit export is where a per-fit choice is named in the affected columns'
    Comments cells (`exporting.export`). Persisting them here would change the session
    format for a value the summary table never displays.

    That exclusion is about the SETTINGS specifically, and it still stands. It is not a
    refusal to carry fit quantities: `fit_details` below persists the fitted values the
    summary table does display, which is the distinction the clause above turns on.

    `fit_details` holds a parametric fit's remaining quantities -- Gamma, the moments, the
    order, the baseline, beta, the amplitude fractions, the stretch -- keyed by the display
    name in the detail spec beside `_DLS_PARAM_SPECS` (app/controller.py), which is the one
    place they are named. A MAPPING rather than a dozen method-specific optional fields: a
    method contributes only its own keys, the dataclass stays flat as quantities are added,
    and an older session simply arrives without the key (`_known_only` reads it as absent,
    which is why no session version bump is owed). Values are the fit's OWN -- the raw
    method string, the raw success bool -- never a rendered cell.

    `passive_notes` holds the fit's passive notices (`notes` on a DLS result: dead channels,
    a baseline or a moment resting on its bound), keyed by the result attribute each is
    about, '' for one about the whole fit. A mapping here rather than the engine's
    index-aligned `notes`/`note_targets` pair, because a persisted record has no
    shared-contract reason to carry parallel arrays and a mapping cannot fall out of step.
    It differs from `reliability_note` in TIER, not merely in content: that one says a
    precondition failed (a red flag), these say how the fit was obtained (a neutral
    qualifier), and a surface must never paint one at the other's level (ui_style_guide
    section 5)."""
    item_id: str
    method: str                          # cumulant/single/double/kww/nnls/contin/lognormal
    peak_index: int = 0
    rh_nm: Optional[float] = None
    rh_se: Optional[float] = None         # ~always None: no SE from one correlogram (invariant 8)
    pdi: Optional[float] = None
    int_fraction: Optional[float] = None  # distribution peak intensity weight (0..1)
    rh_fast_nm: Optional[float] = None    # double-exponential fast mode
    rh_slow_nm: Optional[float] = None    # double-exponential slow mode
    is_apparent: bool = True
    label: str = ''                       # short provenance, e.g. 'NNLS peak 2'
    reliability_note: str = ''            # why this row's value is flagged; '' = unflagged
    # NOT carried on `label`. That field is short provenance and feeds the DLS-summary
    # export's `From` column, declared "contributing measurements / method", where a
    # consumer expects a short categorical string rather than a multi-sentence physics
    # warning. The name matches CumulantResult / ZimmBerryResult / _reliability_comment, so
    # one token is
    # greppable from analysis/ through core/ to exporting/, and the flag is DERIVED from
    # the note (flagged iff non-empty) rather than tracked beside it.

    # The parametric fit's remaining quantities, keyed by display name; and its passive
    # notices, keyed by the result attribute each is about ('' = about the whole fit).
    # See the class docstring for both shapes and why each is a mapping.
    fit_details: Dict[str, Any] = field(default_factory=dict)
    passive_notes: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'MeasurementResultRow':
        return cls(**_known_only(cls, d))


@dataclass
class SampleRhRow:
    """One sample-level Rh result (Summary Table 2).

    Keyed by (sample_id, source_kind, source_set). Sources: 'replicate_avg'
    (apparent), 'gamma_q2' (apparent; q->0 only), 'conc_extrap' (thermodynamic;
    c->0), 'mode_family' (apparent AND mode-resolved -- one relaxation mode's
    Stokes-Einstein size from the mode map's per-family fit; source_set is
    'label|fraction'). The apparent/thermodynamic distinction is carried
    explicitly so it is never conflated downstream (invariant 7), and these
    rows are shaped to feed rho = Rg/Rh later (rh_nm/rh_se/is_apparent map
    onto ResultCandidate)."""
    sample_id: str
    source_kind: str                      # 'replicate_avg' | 'gamma_q2' | 'conc_extrap' | 'mode_family'
    source_set: str                       # disambiguator: method/fraction/etc.
    rh_nm: Optional[float] = None
    rh_se: Optional[float] = None
    is_apparent: bool = True
    rh_type_label: str = ''               # 'apparent' / 'thermodynamic' (display convenience)
    from_label: str = ''                  # the 'From' column: contributing measurements/method
    fraction: Optional[str] = None
    se_estimator: Optional[str] = None    # covariance estimator behind rh_se for a
    #                                       regression source (gamma_q2/conc_extrap);
    #                                       None for the replicate-average SEM (not a
    #                                       regression SE, so estimator-independent).

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SampleRhRow':
        return cls(**_known_only(cls, d))


# ===========================================================================
# Sample: a group of measurements + its result
# ===========================================================================

@dataclass
class Sample:
    """A group of measurements sharing (polymer, solvent, rounded temperature).

    Concentration and angle vary WITHIN a sample. The labeled slots make the
    eventual DLS+SLS pairing (for rho = Rg/Rh) a property of the sample -- "this
    sample has both" -- rather than a separate matching pass.
    """
    sample_id: str
    polymer_name: str
    solvent_name: str
    temperature_K: float
    dls_item_ids: List[str] = field(default_factory=list)
    sls_item_ids: List[str] = field(default_factory=list)       # non-zero concentration
    solvent_reference_item_id: Optional[str] = None             # the c = 0 SLS reference
    # Extra c = 0 (solvent-blank) SLS series beyond the single reference slot. The
    # data model keeps ONE reference by design, so a second blank (e.g. a re-measured
    # blank) would otherwise overwrite the slot and orphan the first. regroup keeps the
    # FIRST-loaded blank in solvent_reference_item_id and parks the rest here, so they
    # stay retrievable and can be surfaced -- never silently dropped. NOT pushed into
    # sls_item_ids (Zimm/Debye assume c > 0 and would choke on a c = 0 series).
    extra_solvent_reference_item_ids: List[str] = field(default_factory=list)
    # One SampleResult per molecular-weight fraction label. The key None is the
    # unfractioned default (the common single-sample case). A Mw series stores one
    # result per fraction ("250k", "1M", ...) so each fraction is an independent
    # scaling point / rho row. Fraction-agnostic code uses the `result` property
    # (-> the None fraction); fraction-aware code uses result_for(fraction).
    fraction_results: Dict[Optional[str], SampleResult] = field(
        default_factory=lambda: {None: SampleResult()})

    def result_for(self, fraction: Optional[str]) -> SampleResult:
        """The SampleResult for one fraction label, created on first access."""
        r = self.fraction_results.get(fraction)
        if r is None:
            r = SampleResult()
            self.fraction_results[fraction] = r
        return r

    @property
    def result(self) -> SampleResult:
        """The unfractioned (default) result. Back-compat for all code that does
        not distinguish Mw fractions."""
        return self.result_for(None)

    @property
    def has_dls(self) -> bool:
        return len(self.dls_item_ids) > 0

    @property
    def has_sls(self) -> bool:
        return len(self.sls_item_ids) > 0

    @property
    def can_pair_rho(self) -> bool:
        """Both DLS (for Rh) and SLS (for Rg) present -> rho = Rg/Rh is possible."""
        return self.has_dls and self.has_sls

    def member_item_ids(self) -> List[str]:
        """Every measurement id in this sample: DLS + SLS + the solvent-reference
        slot(s). The membership a regroup plan follows to find a vanished
        sample's successor."""
        ids = list(self.dls_item_ids) + list(self.sls_item_ids)
        if self.solvent_reference_item_id:
            ids.append(self.solvent_reference_item_id)
        ids.extend(self.extra_solvent_reference_item_ids)
        return ids

    def to_dict(self) -> Dict[str, Any]:
        return {
            'sample_id': self.sample_id, 'polymer_name': self.polymer_name,
            'solvent_name': self.solvent_name, 'temperature_K': self.temperature_K,
            'dls_item_ids': list(self.dls_item_ids),
            'sls_item_ids': list(self.sls_item_ids),
            'solvent_reference_item_id': self.solvent_reference_item_id,
            'extra_solvent_reference_item_ids': list(
                self.extra_solvent_reference_item_ids),
            # Serialized as a list (JSON object keys can't be null) of
            # {fraction: <label|null>, result: {...}}.
            'fraction_results': [
                {'fraction': frac, 'result': res.to_dict()}
                for frac, res in self.fraction_results.items()],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Sample':
        if 'fraction_results' in d:
            fr = {entry['fraction']: SampleResult.from_dict(entry['result'])
                  for entry in d['fraction_results']}
        elif 'result' in d:        # back-compat: pre-fraction single-result session
            fr = {None: SampleResult.from_dict(d['result'])}
        else:
            fr = {}
        fr.setdefault(None, SampleResult())   # always have the default slot
        return cls(
            sample_id=d['sample_id'], polymer_name=d['polymer_name'],
            solvent_name=d['solvent_name'], temperature_K=d['temperature_K'],
            dls_item_ids=list(d.get('dls_item_ids', [])),
            sls_item_ids=list(d.get('sls_item_ids', [])),
            solvent_reference_item_id=d.get('solvent_reference_item_id'),
            extra_solvent_reference_item_ids=list(
                d.get('extra_solvent_reference_item_ids', [])),
            fraction_results=fr,
        )


def sample_group_key(polymer_name: str, solvent_name: str,
                     temperature_K: float) -> Tuple[str, str, float]:
    """The identity that groups measurements into one sample.

    Combination doctrine (core/data_models.py): this key builds the workspace's
    bookkeeping PROPOSAL. Membership never by itself licenses combining
    measurements in one fit -- that is the doctrine's two rules, at the consumers.

    Concentration and angle are deliberately excluded -- they are the axes within
    a sample. Temperature is bucketed so floating-point noise (and thermometer
    noise) does not split a sample, while genuinely different set points
    (35 vs 50 C) still separate.

    FIELD SET vs core.data_models.make_sample_key: this key drops concentration
    on purpose (a Zimm ladder is ONE sample), while the SampleKey keeps it
    because it identifies a measurement. That divergence is intentional.
    NORMALIZATION, by contrast, must be identical -- both name fields go through
    casefold_identity_name and the temperature through
    round_temperature_for_identity, the same two functions make_sample_key calls.
    They used to disagree (this side lowercased the polymer name the SampleKey
    preserved, and bucketed temperature at 0.01 K against the SampleKey's six
    significant figures), so the workspace would group measurements that
    analysis/dls/angular.py then rejected as "not the same sample identity".
    Do not reintroduce a normalization here that make_sample_key does not share.

    A freshly loaded measurement may not have its temperature confirmed yet (the
    parsers leave user-supplied fields blank until the confirmation step), so a
    None temperature is tolerated: it groups under a NaN bucket and re-groups to
    its real value once the user fills it in and commits.
    """
    poly = casefold_identity_name(polymer_name or '')
    solv = casefold_identity_name(solvent_name or '')
    temp = (float('nan') if temperature_K is None
            else round_temperature_for_identity(temperature_K))
    return (poly, solv, temp)


# ===========================================================================
# Regroup planning: results follow the sample, not its id string
# ===========================================================================

# Provenance-tier precedence when a regroup merges two results into one slot
# (higher rank wins its quantity; an unknown tier ranks lowest, fail-safe).
_TIER_RANK = {'computed': 0, 'picked': 1, 'user': 2}


@dataclass
class RegroupLoss:
    """One stored deliberate-tier value ('user'/'picked') a regroup would drop
    or move -- an ask-worthy custody event.

    Computed by `Workspace.plan_regroup` so an interactive caller can ask BEFORE
    the mutating call -- dialogs guard the custody of stored values (the
    combination doctrine, core/data_models.py) -- and the apply then performs
    exactly the resolution previewed: one computation behind the ask and the
    act. `kind` is 'merge' (two samples' results land in one slot; the
    tier-resolution loser is dropped), 'scatter' (a vanished sample's members
    split across several new samples, so its results have no unique successor
    and are dropped whole), or 'move' (the value survives but lands on a
    PRE-EXISTING sample -- nothing is dropped, and the ask names the
    acquisition because Undo restores grouping, not attribution).
    """
    sample_id: str                        # whose value is dropped / moves
    fraction: Optional[str]
    quantity: str                         # 'mw' | 'rg' | 'a2' | 'rh'
    value: float
    tier: str                             # 'user' | 'picked'
    label: str
    kind: str                             # 'merge' | 'scatter' | 'move'
    kept_sample_id: Optional[str] = None  # merge: whose value survives;
    #                                       move: the sample taking the value
    kept_value: Optional[float] = None
    kept_tier: Optional[str] = None


@dataclass
class RegroupPlan:
    """What a regroup would do with the stored per-sample state.

    `member_sid` maps every measurement to the sample a regroup places it in;
    `carries` maps each VANISHED sample id to the one new sample holding all
    its surviving members (its `fraction_results` and durable rows follow --
    results follow the sample, not its id string, owner-ruled 2026-08-13);
    `losses` lists every deliberate-tier value the apply would drop, plus
    every one that would MOVE onto a pre-existing sample (kind 'move' -- the
    ask names the acquisition; 'computed' values are recomputable and never
    listed).
    """
    member_sid: Dict[str, str] = field(default_factory=dict)
    carries: Dict[str, str] = field(default_factory=dict)
    losses: List[RegroupLoss] = field(default_factory=list)


def _is_default_result(r: SampleResult) -> bool:
    """True when a SampleResult holds nothing worth preserving (all defaults)."""
    return asdict(r) == asdict(SampleResult())


def _absent(value) -> bool:
    """True when a stored result value carries no number: ``None`` OR a non-finite
    float. The value fields are NaN-capable (``run_zimm`` stores ``res.rg_nm``
    unconditionally, and a non-positive angular slope makes it NaN), so a guard
    spelled ``is None`` is sized to a narrower object than the field it protects
    (rule 12): a NaN read as PRESENT tied at tier 'computed' and the target won,
    silently dropping the vanishing sample's real Rg. One
    predicate for the merge and the two loss enumerations, so the ask and the act
    agree on what a value is."""
    if value is None:
        return True
    return isinstance(value, float) and not math.isfinite(value)


def _quantity_fields(quantity: str) -> List[str]:
    """Every SampleResult field belonging to one quantity: its value field, the
    whole `<q>_*` family, and Mw's bare scale flags. By reflection, so a
    field added later cannot be left behind by a merge (rule 10)."""
    mine = [f.name for f in fields(SampleResult)
            if f.name == RESULT_VALUE_FIELDS[quantity]
            or f.name.startswith(quantity + '_')]
    if quantity == 'mw':
        mine.extend(MW_BARE_SCALE_FLAGS)   # the two unprefixed scale flags
    return mine


def _merge_fraction_results(dst: Dict[Optional[str], SampleResult],
                            src: Dict[Optional[str], SampleResult],
                            dst_sid: str, src_sid: str,
                            losses: Optional[List[RegroupLoss]] = None) -> None:
    """Merge a vanished sample's results INTO a target dict, per quantity, by
    provenance tier: user > picked > computed, and the TARGET wins a tie (the
    surviving sample's value is kept -- owner-ruled 2026-08-13). Every
    deliberate-tier drop is appended to `losses` when given, so the planning
    pass and the applying pass describe the same outcome by construction."""
    for frac, s_res in src.items():
        d_res = dst.get(frac)
        if d_res is None or _is_default_result(d_res):
            dst[frac] = s_res
            continue
        if s_res is d_res or _is_default_result(s_res):
            continue
        for q in RESULT_QUANTITIES:
            s_val = getattr(s_res, RESULT_VALUE_FIELDS[q])
            if _absent(s_val):
                continue
            d_val = getattr(d_res, RESULT_VALUE_FIELDS[q])
            s_tier = getattr(s_res, q + '_source')
            d_tier = getattr(d_res, q + '_source')
            src_wins = _absent(d_val) or (_TIER_RANK.get(s_tier, 0)
                                          > _TIER_RANK.get(d_tier, 0))
            if losses is not None and not _absent(d_val):
                loser_res, loser_sid = ((d_res, dst_sid) if src_wins
                                        else (s_res, src_sid))
                winner_res, winner_sid = ((s_res, src_sid) if src_wins
                                          else (d_res, dst_sid))
                if getattr(loser_res, q + '_source') in DELIBERATE_TIERS:
                    losses.append(RegroupLoss(
                        sample_id=loser_sid, fraction=frac, quantity=q,
                        value=getattr(loser_res, RESULT_VALUE_FIELDS[q]),
                        tier=getattr(loser_res, q + '_source'),
                        label=getattr(loser_res, q + '_label'),
                        kind='merge', kept_sample_id=winner_sid,
                        kept_value=getattr(winner_res, RESULT_VALUE_FIELDS[q]),
                        kept_tier=getattr(winner_res, q + '_source')))
            if src_wins:
                for f in _quantity_fields(q):
                    setattr(d_res, f, getattr(s_res, f))


# ===========================================================================
# Workspace: the whole session state
# ===========================================================================

SESSION_FORMAT = 'ls_session'
# Bumped 1 -> 2 when the solvent vocabulary was deleted and temperature grouping widened to
# 0.1 K. Both change what sample_group_key returns, and sample_id values derived from it are
# written into the session file. A version-1 session is therefore REFUSED, not migrated:
# its ids encode key semantics this build no longer computes, and the WIDER 0.1 K bucket
# can merge two version-1 samples -- a lossy migration whose result merges (which stored
# value survives?) would be decided by code instead of the user who owns the values.
# Deliberately no migration or repair path (owner-ruled; the regroup result-carry moves
# state between ids -- it cannot re-decide what the old key semantics meant).
SESSION_VERSION = 2


@dataclass
class Workspace:
    """Holds every loaded measurement, the samples grouped from them, and the
    user's manual grouping overrides.

    `overrides` maps an item_id to a sample_id the user has forced it into,
    surviving re-grouping (the hybrid model: auto-proposal + manual correction).
    """
    measurements: Dict[str, LoadedMeasurement] = field(default_factory=dict)
    samples: Dict[str, Sample] = field(default_factory=dict)
    overrides: Dict[str, str] = field(default_factory=dict)
    traces: Dict[str, LoadedTrace] = field(default_factory=dict)
    # DLS Summary-tab snapshot store (durable display scalars; see the row
    # dataclasses above). Added as additive, absent-tolerant fields under session
    # version 1 -- they round-trip through the session JSON but never block an old
    # session that predates them from loading.
    dls_result_rows: Dict[str, 'MeasurementResultRow'] = field(default_factory=dict)
    sample_rh_rows: Dict[str, 'SampleRhRow'] = field(default_factory=dict)
    _counter: int = 0

    # ---- adding measurements ----
    def add_measurement(self, loaded: LoadedMeasurement) -> str:
        self.measurements[loaded.item_id] = loaded
        return loaded.item_id

    def add_trace(self, trace: LoadedTrace) -> str:
        """Register an intensity trace (kept separate from the measurements)."""
        self.traces[trace.trace_id] = trace
        return trace.trace_id

    def remove_trace(self, trace_id: str) -> Optional[LoadedTrace]:
        """Drop one trace, returning it (or None if it was not there)."""
        return self.traces.pop(trace_id, None)

    def traces_for_measurement(self, item_id: str) -> List[LoadedTrace]:
        """Every trace whose measurement_id back-references `item_id`.

        A linear scan rather than a maintained reverse index: the trace store holds
        a handful of items, and a second copy of the link would be one more thing
        that can disagree with the first.
        """
        return [t for t in self.traces.values() if t.measurement_id == item_id]

    def set_trace_link(self, trace_id: str, item_id: Optional[str]) -> None:
        """Point a trace at a DLS measurement, or detach it (item_id=None).

        Raises KeyError for an unknown trace_id. The CALLER validates `item_id` --
        the controller does, since only it knows which items are DLS measurements.
        """
        self.traces[trace_id].measurement_id = item_id

    def remove_traces_of(self, item_id: str) -> List[str]:
        """Delete every trace linked to `item_id`, returning the ids removed.

        Owner-ruled 2026-08-14: removing a measurement CASCADES to the count-rate
        traces attached to it, rather than leaving them behind as standalone. The
        user-facing confirm names the count before this runs.
        """
        removed = [t.trace_id for t in self.traces_for_measurement(item_id)]
        for tid in removed:
            self.traces.pop(tid, None)
        return removed

    def remove_measurement(self, item_id: str) -> None:
        """Drop a measurement, any manual grouping override it carried, and any
        count-rate trace linked to it, then re-group. Samples left empty simply do
        not get rebuilt by regroup()."""
        self.measurements.pop(item_id, None)
        self.overrides.pop(item_id, None)
        self.remove_traces_of(item_id)
        self.regroup()

    def new_item_id(self, prefix: str = 'm') -> str:
        self._counter += 1
        return f'{prefix}{self._counter:04d}'

    def new_sample_id(self) -> str:
        """Mint a fresh sample_id for a manually created sample.

        Auto-grouped ids always look like 'poly|solv|tempK' (they contain '|', see
        _sid_for_key), so the 'override-####' form here can never collide with an
        auto key. The shared monotonic counter keeps it unique and stable across a
        save -> reload (the counter is serialized with the workspace)."""
        self._counter += 1
        return f'override-{self._counter:04d}'

    # ---- grouping (hybrid: auto-propose, honor overrides) ----
    def plan_regroup(self, *, use_working: bool = False,
                     overrides: Optional[Dict[str, str]] = None,
                     params_override: Optional[Dict[str, Dict[str, Any]]] = None,
                     ) -> RegroupPlan:
        """What `regroup()` would do -- computed WITHOUT mutating anything.

        THE one computation behind both the interactive pre-checks (the
        controller reads this on the GUI main thread, the GUI asks about
        `losses`, and Cancel means the mutating call never happens) and
        `regroup()` itself, which applies exactly this plan -- so the ask and
        the act cannot disagree (rule 12: one computation).

        `use_working=True` plans from the WORKING parameter sets (what a commit
        is about to adopt); `overrides` substitutes a hypothetical override map
        (what a manual move is about to install); `params_override` substitutes
        stored parameter dicts per item_id (what an Undo is about to restore --
        a measurement absent from it keeps its committed params, matching the
        restore's own skip rule). Defaults plan the status quo.
        """
        ov = self.overrides if overrides is None else overrides
        member_sid: Dict[str, str] = {}
        for item_id, lm in self.measurements.items():
            if params_override is not None:
                p = params_override.get(item_id, lm.committed_params)
            else:
                p = lm.working_params if use_working else lm.committed_params
            key = sample_group_key(p.get('polymer_name', ''),
                                   p.get('solvent_name', ''),
                                   p.get('temperature_K', float('nan')))
            member_sid[item_id] = ov.get(item_id, self._sid_for_key(key))
        new_sids = set(member_sid.values())

        carries: Dict[str, str] = {}
        losses: List[RegroupLoss] = []
        for old_sid, s in self.samples.items():
            if old_sid in new_sids:
                continue                    # sid stable -> keeps its own state
            members = [iid for iid in s.member_item_ids() if iid in member_sid]
            if not members:
                continue                    # sample fully removed with its data
            dests = {member_sid[iid] for iid in members}
            if len(dests) == 1:
                # Unique successor: everything follows (results, durable rows,
                # the controller's per-sample stores) -- even when the results
                # are still default, so the rows/masks never orphan.
                carries[old_sid] = next(iter(dests))
                continue
            # Scatter: no unique successor -- the results are dropped whole,
            # and every deliberate-tier value in them is a reportable loss.
            for frac, r in s.fraction_results.items():
                for q in RESULT_QUANTITIES:
                    val = getattr(r, RESULT_VALUE_FIELDS[q])
                    tier = getattr(r, q + '_source')
                    if not _absent(val) and tier in DELIBERATE_TIERS:
                        losses.append(RegroupLoss(
                            sample_id=old_sid, fraction=frac, quantity=q,
                            value=val, tier=tier,
                            label=getattr(r, q + '_label'), kind='scatter'))

        # Merge losses: rehearse each carry against a deep copy of what the
        # receiving sample will hold, in the same order the apply merges.
        staging: Dict[str, Dict[Optional[str], SampleResult]] = {}
        for old_sid, new_sid in carries.items():
            if new_sid not in staging:
                base = self.samples.get(new_sid)
                staging[new_sid] = (copy.deepcopy(base.fraction_results)
                                    if base is not None else {})
            n_before = len(losses)
            _merge_fraction_results(
                staging[new_sid],
                copy.deepcopy(self.samples[old_sid].fraction_results),
                new_sid, old_sid, losses)
            if new_sid not in self.samples:
                continue        # a fresh id (pure rename / new sample): silent
            # A deliberate-tier value MOVING into a pre-existing sample is
            # ask-worthy even when nothing collides (owner-ruled 2026-08-13):
            # the acquisition is otherwise silent and Undo restores grouping,
            # not attribution. Slots a merge entry already names are skipped --
            # one dialog line per quantity, whichever way it resolved.
            covered = {(x.fraction, x.quantity) for x in losses[n_before:]}
            for frac, r in self.samples[old_sid].fraction_results.items():
                for q in RESULT_QUANTITIES:
                    val = getattr(r, RESULT_VALUE_FIELDS[q])
                    tier = getattr(r, q + '_source')
                    if (not _absent(val) and tier in DELIBERATE_TIERS
                            and (frac, q) not in covered):
                        losses.append(RegroupLoss(
                            sample_id=old_sid, fraction=frac, quantity=q,
                            value=val, tier=tier,
                            label=getattr(r, q + '_label'), kind='move',
                            kept_sample_id=new_sid))
        return RegroupPlan(member_sid=member_sid, carries=carries, losses=losses)

    def regroup(self) -> RegroupPlan:
        """Rebuild `samples` from the committed parameters of all measurements.

        Auto-buckets by sample_group_key, then applies manual overrides. Stored
        results follow the sample, not its id string (owner-ruled 2026-08-13):
        a sample whose id is stable keeps its own `fraction_results`; a sample
        whose id VANISHED (an identity edit, a manual move) carries them to its
        unique successor, merging per quantity by provenance tier where the
        successor already holds results (user > picked > computed; the
        surviving sample wins a tie -- `_merge_fraction_results`). Interactive
        callers pre-check `plan_regroup().losses` and ask before any
        deliberate-tier value is dropped OR moved onto a pre-existing sample;
        this method applies the same plan silently. Undo restores parameters
        and grouping, never result attribution: a value a merge moved stays
        with the sample the user merged it into (which is why the move itself
        is asked about up front), and re-entering it is the recovery.
        Returns the applied plan so the controller can re-key its own
        per-sample stores from `carries`.
        """
        plan = self.plan_regroup()
        previous_results = {sid: s.fraction_results
                            for sid, s in self.samples.items()}
        new_samples: Dict[str, Sample] = {}

        def ensure_sample(sid, poly, solv, temp):
            if sid not in new_samples:
                new_samples[sid] = Sample(
                    sample_id=sid, polymer_name=poly, solvent_name=solv,
                    temperature_K=temp,
                    fraction_results=previous_results.get(
                        sid, {None: SampleResult()}))
            return new_samples[sid]

        for item_id, lm in self.measurements.items():
            p = lm.committed_params
            sid = plan.member_sid[item_id]
            # If overridden into an existing/other sample, take that sample's
            # identity from its first member; else use this measurement's params.
            samp = ensure_sample(sid, p.get('polymer_name', ''),
                                 p.get('solvent_name', ''),
                                 p.get('temperature_K', float('nan')))
            if lm.kind == 'dls':
                samp.dls_item_ids.append(item_id)
            else:  # sls
                conc = p.get('concentration_g_per_mL', None)
                if conc is not None and conc == 0:
                    # First-loaded blank wins the single reference slot; any further
                    # c = 0 series is parked as an extra so it can't silently vanish.
                    # Measurements iterate in load order, so "first" is stable.
                    if samp.solvent_reference_item_id is None:
                        samp.solvent_reference_item_id = item_id
                    else:
                        samp.extra_solvent_reference_item_ids.append(item_id)
                else:
                    samp.sls_item_ids.append(item_id)

        # Results follow the sample: carry each vanished sample's results to its
        # unique successor (wholesale into a fresh sample; a per-quantity tier
        # merge where the successor already holds results), and re-key its
        # durable Summary rows the same way (the target's row wins a collision).
        for old_sid, new_sid in plan.carries.items():
            samp = new_samples.get(new_sid)
            if samp is None:
                continue
            if (new_sid not in previous_results
                    and all(_is_default_result(r)
                            for r in samp.fraction_results.values())):
                samp.fraction_results = previous_results[old_sid]
            else:
                _merge_fraction_results(samp.fraction_results,
                                        previous_results[old_sid],
                                        new_sid, old_sid)
            for k in list(self.sample_rh_rows):
                row = self.sample_rh_rows[k]
                if row.sample_id != old_sid:
                    continue
                del self.sample_rh_rows[k]
                nk = self._sample_row_key(new_sid, row.source_kind,
                                          row.source_set)
                # A COLLIDING Table-2 row is dropped without an ask, by decision
                # (owner-ruled 2026-08-13): rows carry no provenance tier and one
                # Run recomputes them, so the ask names stored RESULT values only.
                if nk not in self.sample_rh_rows:
                    row.sample_id = new_sid
                    self.sample_rh_rows[nk] = row

        self.samples = new_samples
        # Drop Summary snapshot rows for measurements/samples that no longer exist
        # (a removed measurement, or a scattered sample whose results found no
        # unique successor). Mirrors _OverlaySelection.prune.
        self.prune_result_rows(set(self.measurements))
        return plan

    @staticmethod
    def _sid_for_key(key: Tuple[str, str, float]) -> str:
        poly, solv, temp = key
        return f'{poly or "?"}|{solv or "?"}|{temp:g}K'

    def assign_many(self, item_ids, sample_id: str) -> RegroupPlan:
        """Manually force measurements into a given sample, as ONE regroup.

        The whole gesture is applied as one override-map change followed by a
        single regroup, so the act applies literally the batch plan the ask
        previewed -- applying k overrides one regroup at a time can END
        DIFFERENTLY (an intermediate step can hand a vanished sample a unique
        successor the batch plan said it would not have)."""
        for iid in item_ids:
            if iid not in self.measurements:
                raise KeyError(f"No measurement {iid!r}.")
        for iid in item_ids:
            self.overrides[iid] = sample_id
        return self.regroup()

    def clear_overrides(self, item_ids) -> RegroupPlan:
        """Remove manual overrides, returning the measurements to auto-grouping
        as ONE regroup (the batch counterpart of `assign_many`, same reason)."""
        for iid in item_ids:
            self.overrides.pop(iid, None)
        return self.regroup()

    def assign_to_sample(self, item_id: str, sample_id: str) -> RegroupPlan:
        """Manually force a measurement into a given sample (an override)."""
        return self.assign_many([item_id], sample_id)

    def clear_override(self, item_id: str) -> RegroupPlan:
        """Remove a manual override, returning the measurement to auto-grouping."""
        return self.clear_overrides([item_id])

    # ---- convenience accessors ----
    def sample_measurements(self, sample_id: str, kind: str) -> List[LoadedMeasurement]:
        """The loaded measurements of one kind ('dls'/'sls') in a sample."""
        s = self.samples[sample_id]
        ids = s.dls_item_ids if kind == 'dls' else s.sls_item_ids
        return [self.measurements[i] for i in ids]

    def solvent_reference(self, sample_id: str) -> Optional[LoadedMeasurement]:
        ref_id = self.samples[sample_id].solvent_reference_item_id
        return self.measurements.get(ref_id) if ref_id else None

    def blank_candidates(self, sample_id: str) -> List[LoadedMeasurement]:
        """Every c = 0 solvent blank in a sample, reference slot first then the
        extras, in load order -- the roster a per-series blank may be chosen from."""
        s = self.samples[sample_id]
        ids = ([s.solvent_reference_item_id] if s.solvent_reference_item_id else [])
        ids += list(s.extra_solvent_reference_item_ids)
        return [self.measurements[i] for i in ids if i in self.measurements]

    def sample_id_of(self, item_id: str) -> Optional[str]:
        """The sample currently holding this measurement (any slot), or None."""
        for sid, s in self.samples.items():
            if item_id in s.member_item_ids():
                return sid
        return None

    def solvent_reference_for(self, item_id: str, *, use_working: bool = False
                              ) -> Tuple[Optional[LoadedMeasurement], str]:
        """The solvent blank THIS series is referenced against, and how it was chosen.

        Returns `(blank, source)` with `source` one of 'user' / 'auto:same-file' /
        'auto:sample' (the family:method shape of `mean_count_rate_cps_source`,
        computed on demand and never stored); `(None, 'auto:sample')` when the sample
        has no blank at all. `use_working=True` reads the series' working
        `solvent_blank_item_id` (an uncommitted pick) instead of the committed one;
        analysis always resolves the committed choice. Resolution order, the normative rule for
        every analysis that subtracts a blank from a series:

        1. The series' committed `solvent_blank_item_id`, when it names a c = 0 blank
           in the same sample. A stale id (its blank removed, or moved to another
           sample) is ignored and the automatic rules apply -- the number is never
           withheld for a dangling link, only re-paired and disclosed.
        2. The blank sharing this series' `source_path` (first in load order if the
           file carries several). An acquisition that records its blank beside the
           data defines the pair; a session or a derived point with no source path
           skips this rule.
        3. The sample's single reference slot -- the pre-existing behaviour, and the
           only rule a one-file-per-sample format ever reaches, so for such data this
           resolver is identical to `solvent_reference(sample_id)`.

        The sample-level slot stays the answer for sample-scoped questions (axis grids,
        collision reporting); only per-series subtraction resolves through here.
        """
        lm = self.measurements.get(item_id)
        sid = self.sample_id_of(item_id)
        if lm is None or sid is None:
            return None, 'auto:sample'
        candidates = self.blank_candidates(sid)
        by_id = {b.item_id: b for b in candidates}
        params = lm.working_params if use_working else lm.committed_params
        wanted = params.get('solvent_blank_item_id')
        if wanted and wanted in by_id and wanted != item_id:
            return by_id[wanted], 'user'
        if lm.source_path is not None:
            for b in candidates:
                if b.item_id != item_id and b.source_path == lm.source_path:
                    return b, 'auto:same-file'
        return self.solvent_reference(sid), 'auto:sample'

    def solvent_reference_collisions(self) -> Dict[str, List[str]]:
        """sample_id -> the extra c = 0 solvent-blank series that did NOT get the
        single reference slot, for every sample that loaded more than one blank.

        Empty when no sample has a collision (the common case). The GUI reads this
        to warn the user that a second blank isn't the active reference -- the model
        keeps the first-loaded one, and the extras stay here so they are never
        silently dropped."""
        return {sid: list(s.extra_solvent_reference_item_ids)
                for sid, s in self.samples.items()
                if s.extra_solvent_reference_item_ids}

    def any_dirty(self) -> bool:
        return any(m.is_dirty() for m in self.measurements.values())

    # ---- DLS Summary snapshot store ----
    @staticmethod
    def _meas_row_key(item_id: str, method: str, peak_index: int) -> str:
        return f'{item_id}|{method}|{peak_index}'

    @staticmethod
    def _sample_row_key(sample_id: str, source_kind: str, source_set: str) -> str:
        return f'{sample_id}|{source_kind}|{source_set}'

    def upsert_dls_result_row(self, row: 'MeasurementResultRow') -> None:
        self.dls_result_rows[self._meas_row_key(
            row.item_id, row.method, row.peak_index)] = row

    def replace_dls_rows_for(self, item_id: str, method: str) -> None:
        """Drop every prior row for (item_id, method) before writing fresh ones,
        so a re-run that resolves fewer peaks leaves no stale higher-index rows."""
        self.dls_result_rows = {
            k: v for k, v in self.dls_result_rows.items()
            if not (v.item_id == item_id and v.method == method)}

    def upsert_sample_rh_row(self, row: 'SampleRhRow') -> None:
        self.sample_rh_rows[self._sample_row_key(
            row.sample_id, row.source_kind, row.source_set)] = row

    def remove_sample_rh_row(self, sample_id: str, source_kind: str,
                             source_set: str) -> Optional['SampleRhRow']:
        """Drop one keyed sample-Rh row, returning it (or None if there was none).

        The exact inverse of `upsert_sample_rh_row`, and the reason it exists rather
        than the caller composing the key itself: the key format is this class's
        business, and a second place that spells it is a second place it can drift.
        """
        return self.sample_rh_rows.pop(
            self._sample_row_key(sample_id, source_kind, source_set), None)

    def prune_result_rows(self, live_ids: set) -> None:
        """Drop measurement rows whose item_id is gone and sample rows whose
        sample_id no longer exists."""
        self.dls_result_rows = {k: v for k, v in self.dls_result_rows.items()
                                if v.item_id in live_ids}
        live_samples = set(self.samples)
        self.sample_rh_rows = {k: v for k, v in self.sample_rh_rows.items()
                               if v.sample_id in live_samples}

    # ---- serialization (self-contained JSON-ready dict) ----
    def to_dict(self) -> Dict[str, Any]:
        return {
            'format': SESSION_FORMAT, 'version': SESSION_VERSION,
            'measurements': [m.to_dict() for m in self.measurements.values()],
            'samples': [s.to_dict() for s in self.samples.values()],
            'overrides': dict(self.overrides),
            'traces': [t.to_dict() for t in self.traces.values()],
            'dls_result_rows': [r.to_dict() for r in self.dls_result_rows.values()],
            'sample_rh_rows': [r.to_dict() for r in self.sample_rh_rows.values()],
            'counter': self._counter,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Workspace':
        if d.get('format') != SESSION_FORMAT:
            raise ValueError(
                f"Not an {SESSION_FORMAT} file (format={d.get('format')!r})."
            )
        if d.get('version') != SESSION_VERSION:
            # forward-compatible read could go here; for now, be explicit.
            raise ValueError(
                f"Unsupported session version {d.get('version')!r}; this build "
                f"reads version {SESSION_VERSION}."
            )
        ws = cls()
        for md in d.get('measurements', []):
            lm = LoadedMeasurement.from_dict(md)
            ws.measurements[lm.item_id] = lm
        ws.overrides = dict(d.get('overrides', {}))
        for td in d.get('traces', []):
            t = LoadedTrace.from_dict(td)
            ws.traces[t.trace_id] = t
        # Summary snapshot rows -- absent in sessions written before this feature,
        # so .get([]) keeps those loading unchanged.
        for rd in d.get('dls_result_rows', []):
            row = MeasurementResultRow.from_dict(rd)
            ws.dls_result_rows[ws._meas_row_key(
                row.item_id, row.method, row.peak_index)] = row
        for rd in d.get('sample_rh_rows', []):
            row = SampleRhRow.from_dict(rd)
            ws.sample_rh_rows[ws._sample_row_key(
                row.sample_id, row.source_kind, row.source_set)] = row
        ws._counter = int(d.get('counter', len(ws.measurements)))
        # Rebuild samples from the stored sample dicts (keeps results + slots),
        # falling back to a regroup if none were stored.
        stored = d.get('samples', [])
        if stored:
            for sd in stored:
                s = Sample.from_dict(sd)
                ws.samples[s.sample_id] = s
        else:
            ws.regroup()
        return ws
