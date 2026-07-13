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

from dataclasses import dataclass, field, fields, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.data_models import DLSMeasurement, IntensityTrace, SLSMeasurement


# Temperature rounding for the sample-grouping key: two samples whose
# temperatures differ only in the 5th decimal are the same sample. 35.0 and
# 50.0 C are different samples; 298.15 and 298.150001 K are not.
_TEMPERATURE_GROUP_DECIMALS = 2   # round Kelvin to 0.01 K for grouping


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
    'mw_fraction', 'analyzer_geometry',
)
_SLS_PARAM_KEYS = (
    'label',
    'polymer_name', 'solvent_name', 'concentration_g_per_mL', 'temperature_K',
    'wavelength_nm', 'solvent_refractive_index', 'dn_dc_mL_per_g',
    'mw_fraction',
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
})


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
    raw: Dict[str, List[float]]        # the raw arrays (lists, for serialization)
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
                mw_fraction=p.get('mw_fraction'),
                # Forward-ready (depolarized dynamic analysis); .get keeps it from
                # being dropped once the VV/VH tagging is wired through the params.
                analyzer_geometry=p.get('analyzer_geometry'),
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

@dataclass
class SampleResult:
    """Analysis outputs attached to a sample.

    Each quantity records its provenance: 'computed' (an auto-selected analysis
    result), 'picked' (a result the user explicitly chose in Cross-Sample), or
    'user' (entered by hand). Both 'picked' and 'user' are deliberate choices and
    are never overwritten by a later analysis run (only a fresh explicit pick or
    manual entry replaces them). The user override matters for systems where the
    measured Mw is biased -- e.g. PVP in a deep eutectic solvent, where co-solvent
    adsorption skews Mw, so the trustworthy value is the one characterized in
    water and entered manually. Cross-sample scaling plots (Rg-Mw, A2-Mw) should
    prefer the user value when present.
    """
    mw_g_per_mol: Optional[float] = None
    mw_source: str = 'computed'        # 'computed' (auto-default) | 'picked' (explicit) | 'user' (manual)
    mw_apparent: Optional[bool] = None  # True = single-condition (Debye/Guinier/single-angle); False = Zimm/Berry
    mw_label: str = ''                  # plain-language provenance of mw_g_per_mol
    mw_se: Optional[float] = None       # statistical SE (None for user/no-SE sources)
    rg_nm: Optional[float] = None
    rg_source: str = 'computed'
    rg_apparent: Optional[bool] = None  # True = single-conc (Debye/Guinier); False = Zimm/Berry (thermodynamic)
    rg_label: str = ''                  # plain-language provenance of rg_nm (shown in Cross-Sample)
    rg_se: Optional[float] = None
    a2_mol_mL_per_g2: Optional[float] = None
    a2_source: str = 'computed'
    a2_label: str = ''                  # plain-language provenance of a2_mol_mL_per_g2 (shown in Cross-Sample)
    a2_se: Optional[float] = None
    a2_calibrated: Optional[bool] = None  # whether the SLS run that filled A2 was calibrated (A2 is scale-dependent)
    rh_nm: Optional[float] = None       # from DLS (infinite-dilution if available)
    rh_source: str = 'computed'
    rh_apparent: Optional[bool] = None  # True = apparent (single q or single c); False = thermodynamic (c->0)
    rh_label: str = ''                  # plain-language provenance of rh_nm (shown in Cross-Sample)
    rh_se: Optional[float] = None
    calibrated: Optional[bool] = None   # whether the SLS run that filled Mw was calibrated
    notes: str = ''

    def set_mw(self, value: float, source: str = 'user') -> None:
        self.mw_g_per_mol = value
        self.mw_source = source

    def effective_mw(self) -> Optional[float]:
        """The Mw to use downstream (user value wins if present)."""
        return self.mw_g_per_mol

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SampleResult':
        return cls(**_known_only(cls, d))


# ===========================================================================
# DLS Summary-tab snapshot rows
# ===========================================================================
#
# These two dataclasses are the DURABLE display cache behind the DLS Summary
# tab. They hold ONLY the scalars each table row shows -- never correlograms,
# grids, or fit curves (those stay in the controller's ephemeral `results`
# cache). Because they are small and self-contained they round-trip cleanly
# through the session JSON, so the Summary table survives save/reload (unlike
# the per-measurement fits, which are recomputed on demand).

@dataclass
class MeasurementResultRow:
    """One per-measurement DLS scalar result (Summary Table 1).

    Keyed by (item_id, method, peak_index). Distribution methods write one row
    per resolved peak (peak_index 0, 1, ...); parametric methods write a single
    row (peak_index 0). Per-measurement DLS is always APPARENT -- a single q and
    a single concentration, neither extrapolated (invariant 7)."""
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
    c->0). The apparent/thermodynamic distinction is carried explicitly so it is
    never conflated downstream (invariant 7), and these rows are shaped to feed
    rho = Rg/Rh later (rh_nm/rh_se/is_apparent map onto ResultCandidate)."""
    sample_id: str
    source_kind: str                      # 'replicate_avg' | 'gamma_q2' | 'conc_extrap'
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

    Concentration and angle are deliberately excluded -- they are the axes within
    a sample. Temperature is rounded so floating-point noise does not split a
    sample, while genuinely different set points (35 vs 50 C) still separate.

    A freshly loaded measurement may not have its temperature confirmed yet (the
    parsers leave user-supplied fields blank until the confirmation step), so a
    None temperature is tolerated: it groups under a NaN bucket and re-groups to
    its real value once the user fills it in and commits.
    """
    poly = (polymer_name or '').strip().lower()
    solv = (solvent_name or '').strip().lower()
    temp = (float('nan') if temperature_K is None
            else round(float(temperature_K), _TEMPERATURE_GROUP_DECIMALS))
    return (poly, solv, temp)


# ===========================================================================
# Workspace: the whole session state
# ===========================================================================

SESSION_FORMAT = 'ls_session'
SESSION_VERSION = 1


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

    def remove_measurement(self, item_id: str) -> None:
        """Drop a measurement and any manual grouping override it carried, then
        re-group. Samples left empty simply do not get rebuilt by regroup()."""
        self.measurements.pop(item_id, None)
        self.overrides.pop(item_id, None)
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
    def regroup(self) -> None:
        """Rebuild `samples` from the committed parameters of all measurements.

        Auto-buckets by sample_group_key, then applies manual overrides. Existing
        SampleResults are preserved across a regroup where the sample_id is stable.
        """
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
            key = sample_group_key(p.get('polymer_name', ''),
                                   p.get('solvent_name', ''),
                                   p.get('temperature_K', float('nan')))
            auto_sid = self._sid_for_key(key)
            sid = self.overrides.get(item_id, auto_sid)
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

        self.samples = new_samples
        # Drop Summary snapshot rows for measurements/samples that no longer exist
        # (a removed measurement, or a sample whose id changed when its grouping
        # key did -- e.g. a temperature edit). Mirrors _OverlaySelection.prune.
        self.prune_result_rows(set(self.measurements))

    @staticmethod
    def _sid_for_key(key: Tuple[str, str, float]) -> str:
        poly, solv, temp = key
        return f'{poly or "?"}|{solv or "?"}|{temp:g}K'

    def assign_to_sample(self, item_id: str, sample_id: str) -> None:
        """Manually force a measurement into a given sample (an override)."""
        if item_id not in self.measurements:
            raise KeyError(f"No measurement {item_id!r}.")
        self.overrides[item_id] = sample_id
        self.regroup()

    def clear_override(self, item_id: str) -> None:
        """Remove a manual override, returning the measurement to auto-grouping."""
        self.overrides.pop(item_id, None)
        self.regroup()

    # ---- convenience accessors ----
    def sample_measurements(self, sample_id: str, kind: str) -> List[LoadedMeasurement]:
        """The loaded measurements of one kind ('dls'/'sls') in a sample."""
        s = self.samples[sample_id]
        ids = s.dls_item_ids if kind == 'dls' else s.sls_item_ids
        return [self.measurements[i] for i in ids]

    def solvent_reference(self, sample_id: str) -> Optional[LoadedMeasurement]:
        ref_id = self.samples[sample_id].solvent_reference_item_id
        return self.measurements.get(ref_id) if ref_id else None

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
        # so .get([]) keeps those (still version 1) loading unchanged.
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
