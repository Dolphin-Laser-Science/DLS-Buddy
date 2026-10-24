"""
app/fit_requirements.py
=======================

What each fit on the SLS and DLS tabs needs before it can run: the requirement rows,
the predicates they test, and the two tables `SLS_FIT_REQUIREMENTS` and
`DLS_FIT_REQUIREMENTS` with the universal rows each starts from, beside the two helpers
the controller's runs share with them (`_value_in`, `_ddls_paired_angles`). Data and
predicates only: the controller, the DLS tab and the manual's requirements-table
generator import it. Qt-free (invariant 5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Optional, Sequence, Set, Tuple, Union

import numpy as np

from core.workspace import LoadedMeasurement
from core.data_models import (require_non_empty_string, require_non_negative,
                              require_positive, require_scattering_angle,
                              polarization_state as _polarization_state,
                              round_angle_for_identity,
                              round_concentration_for_identity)
from analysis import form_factor as ff
from analysis import optical_model as optical


def _value_in(value: Optional[float], excluded) -> bool:
    """True if `value` matches any entry in `excluded` (float-tolerant). Used to drop
    user-excluded outlier angles/concentrations from a multi-point fit (#9)."""
    if value is None or not excluded:
        return False
    return any(np.isclose(float(value), float(e)) for e in excluded)


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


# ===========================================================================
# SLS fit requirements: what each fit on the SLS tab needs before it can run
# ===========================================================================

@dataclass(frozen=True)
class MissingInput:
    """One input a fit cannot run without, as `Controller.sls_fit_availability` and
    `Controller.dls_fit_availability` report it: the requirement's `code` and `phrase`,
    the Data-tab parameter key it is entered at (`param`; None for a count, which the
    loaded data, the mask or the ticks decide), for a count what the selection holds
    (`have`) against the fit's minimum (`need`), and a short clause read off the state
    when the requirement writes one (`detail`, such as the polarizations a ticked set
    mixes; '' otherwise). The GUI names the Data-tab row from `param`, so the controller
    carries no display name."""
    code: str
    phrase: str
    param: Optional[str] = None
    have: Optional[int] = None
    need: Optional[int] = None
    detail: str = ''


@dataclass(frozen=True)
class FitAvailability:
    """Whether one fit can run on the committed state it was judged on (`ok`), and every
    input it lacks, in roster order (`missing`; empty when `ok`)."""
    ok: bool
    missing: Tuple[MissingInput, ...] = ()


@dataclass(frozen=True, eq=False)
class _SLSSeries:
    """One concentration series as the fit requirements read it. `angles` are the
    loaded angles sorted, the order the engine's arrays carry; `hidden` flags each one
    the mask hides (`SLSMask.is_masked`, the predicate `masked_rayleigh` applies), and is
    None when the committed concentration is unusable, since the mask is keyed by it.
    `raw_angles` are the angles as loaded and `blank` the measured blank the series
    resolves to (None without one), for the blank's own comparison."""
    params: Dict[str, Any]
    angles: np.ndarray
    raw_angles: np.ndarray
    hidden: Optional[np.ndarray]
    blank: Optional[LoadedMeasurement]

    @property
    def concentration(self) -> Optional[float]:
        """The committed concentration when it is one a fit can use, else None."""
        c = self.params.get('concentration_g_per_mL')
        return float(c) if _accepts(require_non_negative, c, 'concentration_g_per_mL') else None


@dataclass(frozen=True, eq=False)
class _SLSFitState:
    """The committed sample state the requirements are judged on, built once per
    `Controller.sls_fit_availability` call: the fraction's series, whether a measured
    blank exists and the typed blank's problem, the selection (the chosen
    concentration's shown angles, the chosen angle), and the contrasts the Mie and
    core-shell fits are handed. Committed parameters and the mask only, never a value
    computed from the intensities."""
    series: Tuple[_SLSSeries, ...]
    has_measured_blank: bool
    typed_blank_problem: str
    chosen_angles: Tuple[float, ...]
    angle: Optional[float]
    mie_contrast: Optional[float]
    core_shell_contrasts: Tuple[Optional[float], Optional[float]]


@dataclass(frozen=True)
class FitRequirement:
    """One input a fit needs to run: a stable `name`, the `code` and `phrase` a missing
    one is reported with, the Data-tab key it is entered at (`param`), and its check over
    the roster's state (`_SLSFitState` for an SLS row, `_DLSFitState` for a DLS one),
    either `holds` (a condition) or `count` with its minimum `need`; a condition may also
    write a `detail` from the state for its missing line. The phrase says what is needed
    in plain words, the one wording every surface that names the requirement reads, the
    User-Manual's tables included."""
    name: str
    code: str
    phrase: str
    param: Optional[str] = None
    holds: Optional[Callable[[Any], bool]] = None
    count: Optional[Callable[[Any], int]] = None
    need: Optional[int] = None
    detail: Optional[Callable[[Any], str]] = None

    def judge(self, state) -> Optional[MissingInput]:
        """None when the requirement holds on `state`, else what is missing."""
        if self.count is not None:
            have = int(self.count(state))
            if have >= self.need:
                return None
            return MissingInput(self.code, self.phrase, self.param, have, self.need)
        if self.holds(state):
            return None
        return MissingInput(self.code, self.phrase, self.param,
                            detail=self.detail(state) if self.detail is not None else '')


def _availability(requirements: Sequence[FitRequirement], state) -> FitAvailability:
    """One roster entry judged on `state`: every requirement it lacks, in roster order."""
    missing = tuple(m for m in (r.judge(state) for r in requirements) if m is not None)
    return FitAvailability(ok=not missing, missing=missing)


def _accepts(validator, value, key: str) -> bool:
    """Does the data model's own field check accept this committed value? The check
    `SLSMeasurement.__post_init__` applies, asked rather than restated, so the roster
    and the refusal cannot be sized differently."""
    try:
        validator(value, key)
    except (TypeError, ValueError):
        return False
    return True


def _every_built_measurement_accepts(key: str, validator) -> Callable[[_SLSFitState], bool]:
    """A check that `key` passes `validator` on every measurement `_rayleigh_for_sample`
    builds for the fraction: each series, and the measured blank each resolves to."""
    def holds(state: _SLSFitState) -> bool:
        for s in state.series:
            for params in (s.params, s.blank.committed_params if s.blank else None):
                if params is not None and not _accepts(validator, params.get(key), key):
                    return False
        return True
    return holds


def _if_entered(validator):
    """`validator` applied only to a value that was entered, a None passing: the shape
    `SLSMeasurement.__post_init__` gives the optional solute and shell indices."""
    def check(value, key: str) -> None:
        if value is not None:
            validator(value, key)
    return check


def _typed_blank_holds(state: _SLSFitState) -> bool:
    """The typed 90° blank passes `_typed_blank_problem`, the predicate the blank is
    built under, whenever a series will be built from it (no measured blank anywhere)."""
    return state.has_measured_blank or not state.series or not state.typed_blank_problem


def _blank_angles_hold(state: _SLSFitState) -> bool:
    """Every series' measured blank shares its angles: `compute_excess_rayleigh_ratio`'s
    own comparison (equal shape, then `np.allclose` on the sorted angles)."""
    for s in state.series:
        if s.blank is None:
            continue
        b = np.asarray(s.blank.raw.get('angles_deg', []), dtype=float)
        if (s.raw_angles.shape != b.shape
                or not np.allclose(np.sort(s.raw_angles), np.sort(b))):
            return False
    return True


def _shown_concentration_count(series: Sequence[_SLSSeries]) -> int:
    """Distinct concentrations above zero with at least one point the mask leaves, the
    count `zimm_analysis` and the form-factor fits read off their design (and
    `Controller.run_zimm`'s own pre-check, so the two are one computation). Distinct as
    `round_concentration_for_identity` judges it, the house's one comparison form: one
    concentration entered by two routes (0.00045 g/mL parsed, 0.45 mg/mL typed) differs
    in its last bit, and counting it twice would offer Zimm on one real concentration."""
    return len({round_concentration_for_identity(s.concentration) for s in series
                if s.concentration and s.hidden is not None and not s.hidden.all()})


def _shown_angle_count(state: _SLSFitState) -> int:
    """Distinct angles with at least one point the mask leaves, over the series above
    zero concentration: the angle count of the same designs."""
    return len({float(a) for s in state.series
                if s.concentration and s.hidden is not None
                for a, h in zip(s.angles, s.hidden, strict=True) if not h})


def _shown_at_angle_count(state: _SLSFitState) -> int:
    """Series above zero concentration whose point at the chosen angle the mask leaves,
    counted as `single_angle_ladder` and `calibration_free_a2` count them: one per
    series (a repeated concentration counts twice), the point found by `np.isclose`."""
    if state.angle is None:
        return 0
    n = 0
    for s in state.series:
        if not s.concentration or s.hidden is None:
            continue
        idx = np.where(np.isclose(s.angles, state.angle))[0]
        if idx.size and not s.hidden[int(idx[0])]:
            n += 1
    return n


def _usable_contrast(m: Optional[float]) -> bool:
    """The form-factor fits' own test of a relative refractive index m."""
    return m is not None and math.isfinite(m) and m > 0


# The contrasts are n/n_solvent, so a solvent index the universal check already names
# leaves them unformed, and an entered solute or shell index the universal checks
# refuse is named there; the Mie, core and shell rows hold then, so no entered index
# is named twice.
_SOLVENT_INDEX_HOLDS = _every_built_measurement_accepts(
    'solvent_refractive_index', require_positive)
_SOLUTE_INDEX_HOLDS = _every_built_measurement_accepts(
    'solute_refractive_index', _if_entered(require_positive))
_SHELL_INDEX_HOLDS = _every_built_measurement_accepts(
    'shell_refractive_index', _if_entered(require_positive))

_REQ_SOLUTE_NAME = FitRequirement(
    'solute_name', 'missing_param', 'a solute name', param='polymer_name',
    holds=_every_built_measurement_accepts('polymer_name', require_non_empty_string))
_REQ_SOLVENT_NAME = FitRequirement(
    'solvent_name', 'missing_param', 'a solvent name', param='solvent_name',
    holds=_every_built_measurement_accepts('solvent_name', require_non_empty_string))
_REQ_CONCENTRATION = FitRequirement(
    'concentration', 'missing_param', 'a concentration, zero or above, on every series',
    param='concentration_g_per_mL',
    holds=_every_built_measurement_accepts('concentration_g_per_mL', require_non_negative))
_REQ_TEMPERATURE = FitRequirement(
    'temperature', 'missing_param', 'a temperature above absolute zero',
    param='temperature_K',
    holds=_every_built_measurement_accepts('temperature_K', require_positive))
_REQ_WAVELENGTH = FitRequirement(
    'wavelength', 'missing_param', 'a wavelength above zero', param='wavelength_nm',
    holds=_every_built_measurement_accepts('wavelength_nm', require_positive))
_REQ_SOLVENT_INDEX = FitRequirement(
    'solvent_index', 'missing_param', 'a solvent refractive index above zero',
    param='solvent_refractive_index', holds=_SOLVENT_INDEX_HOLDS)
_REQ_ENTERED_SOLUTE_INDEX = FitRequirement(
    'entered_solute_index', 'missing_param',
    'a solute refractive index above zero, or none',
    param='solute_refractive_index', holds=_SOLUTE_INDEX_HOLDS)
_REQ_ENTERED_SHELL_INDEX = FitRequirement(
    'entered_shell_index', 'missing_param',
    'a shell refractive index above zero, or none',
    param='shell_refractive_index', holds=_SHELL_INDEX_HOLDS)
_REQ_TYPED_BLANK = FitRequirement(
    'typed_blank', 'missing_param',
    'when no solvent blank was measured, a solvent blank intensity at 90° at or above '
    'the dark count rate, or none',
    param='solvent_blank_intensity_90deg', holds=_typed_blank_holds)
_REQ_BLANK_ANGLES = FitRequirement(
    'blank_angles', 'blank_angles_differ',
    'a solvent blank measured at the same angles as the series it is paired with',
    param='solvent_blank_item_id', holds=_blank_angles_hold)

_REQ_TWO_CONCENTRATIONS = FitRequirement(
    'two_concentrations', 'too_few_concentrations',
    'at least two concentrations above zero that are not hidden',
    count=lambda state: _shown_concentration_count(state.series), need=2)
_REQ_ONE_CONCENTRATION = FitRequirement(
    'one_concentration', 'too_few_concentrations',
    'at least one concentration above zero that is not hidden',
    count=lambda state: _shown_concentration_count(state.series), need=1)
_REQ_TWO_ANGLES = FitRequirement(
    'two_angles', 'too_few_angles', 'at least two angles that are not hidden',
    count=_shown_angle_count, need=2)
_REQ_THREE_ANGLES = FitRequirement(
    'three_angles', 'too_few_angles', 'at least three angles that are not hidden',
    count=_shown_angle_count, need=3)
_REQ_ANGLES_AT_CONCENTRATION = FitRequirement(
    'angles_at_concentration', 'too_few_angles',
    'at least two angles not hidden at the chosen concentration',
    count=lambda state: len(state.chosen_angles), need=2)
_REQ_CONCENTRATIONS_AT_ANGLE = FitRequirement(
    'concentrations_at_angle', 'too_few_concentrations',
    'at least two concentrations above zero not hidden at the chosen angle',
    count=_shown_at_angle_count, need=2)
_REQ_MIE_SOLUTE_INDEX = FitRequirement(
    'mie_solute_index', 'missing_param', 'a solute refractive index above zero',
    param='solute_refractive_index',
    holds=lambda state: (not _SOLVENT_INDEX_HOLDS(state) or not _SOLUTE_INDEX_HOLDS(state)
                         or _usable_contrast(state.mie_contrast)))
_REQ_CORE_INDEX = FitRequirement(
    'core_index', 'missing_param', 'a solute refractive index above zero, for the core',
    param='solute_refractive_index',
    holds=lambda state: (not _SOLVENT_INDEX_HOLDS(state) or not _SOLUTE_INDEX_HOLDS(state)
                         or _usable_contrast(state.core_shell_contrasts[0])))
_REQ_SHELL_INDEX = FitRequirement(
    'shell_index', 'missing_param', 'a shell refractive index above zero',
    param='shell_refractive_index',
    holds=lambda state: (not _SOLVENT_INDEX_HOLDS(state) or not _SHELL_INDEX_HOLDS(state)
                         or _usable_contrast(state.core_shell_contrasts[1])))

#: What every SLS fit needs: the identity and optics each measurement is built from,
#: an entered solute or shell index among them, and the solvent blank it is subtracted
#: against. A missing calibrant reading, standard refractive index or dn/dc is not
#: here: the fit runs and flags its scale.
SLS_UNIVERSAL_REQUIREMENTS: Tuple[FitRequirement, ...] = (
    _REQ_SOLUTE_NAME, _REQ_SOLVENT_NAME, _REQ_CONCENTRATION, _REQ_TEMPERATURE,
    _REQ_WAVELENGTH, _REQ_SOLVENT_INDEX, _REQ_ENTERED_SOLUTE_INDEX,
    _REQ_ENTERED_SHELL_INDEX, _REQ_TYPED_BLANK, _REQ_BLANK_ANGLES,
)

_FORM_FACTOR_FLOOR = SLS_UNIVERSAL_REQUIREMENTS + (_REQ_ONE_CONCENTRATION, _REQ_TWO_ANGLES)

#: The one declaration of what each SLS fit needs to run, keyed by the SLS tab's Method
#: keys and by ('form_factor', model) for each P(q) model; every tuple opens with the
#: universal requirements. The plain 'form_factor' key holds what every model needs,
#: so the Method entry follows it and the P(q) entries and Run follow the model's key.
#: Only committed inputs and counts appear here; a condition on the measured values
#: (an excess at or below zero, say) is the fit's own refusal when it runs.
SLS_FIT_REQUIREMENTS: Dict[Union[str, Tuple[str, str]], Tuple[FitRequirement, ...]] = {
    'zimm': SLS_UNIVERSAL_REQUIREMENTS + (_REQ_TWO_CONCENTRATIONS, _REQ_TWO_ANGLES),
    'berry': SLS_UNIVERSAL_REQUIREMENTS + (_REQ_TWO_CONCENTRATIONS, _REQ_TWO_ANGLES),
    'form_factor': _FORM_FACTOR_FLOOR,
    ('form_factor', ff.SPHERE): _FORM_FACTOR_FLOOR,
    ('form_factor', ff.DEBYE): _FORM_FACTOR_FLOOR,
    ('form_factor', ff.ELLIPSOID): SLS_UNIVERSAL_REQUIREMENTS + (
        _REQ_ONE_CONCENTRATION, _REQ_THREE_ANGLES),
    ('form_factor', ff.MIE): _FORM_FACTOR_FLOOR + (_REQ_MIE_SOLUTE_INDEX,),
    ('form_factor', ff.CORE_SHELL): SLS_UNIVERSAL_REQUIREMENTS + (
        _REQ_ONE_CONCENTRATION, _REQ_THREE_ANGLES, _REQ_CORE_INDEX, _REQ_SHELL_INDEX),
    'debye': SLS_UNIVERSAL_REQUIREMENTS + (_REQ_ANGLES_AT_CONCENTRATION,),
    'guinier': SLS_UNIVERSAL_REQUIREMENTS + (_REQ_ANGLES_AT_CONCENTRATION,),
    'ladder': SLS_UNIVERSAL_REQUIREMENTS + (_REQ_CONCENTRATIONS_AT_ANGLE,),
    'calfree': SLS_UNIVERSAL_REQUIREMENTS + (_REQ_CONCENTRATIONS_AT_ANGLE,),
    'rayleigh': SLS_UNIVERSAL_REQUIREMENTS,
}


# ---- what each DLS analysis needs before it can run (DLS_FIT_REQUIREMENTS) ----

@dataclass(frozen=True, eq=False)
class _DLSFitState:
    """The committed state the DLS requirements are judged on, built once per
    `Controller.dls_fit_availability` call: the committed parameters of the runs named
    (`runs`, in the order named), their display names (`names`, for a missing line that
    must say which runs lack an input) and the angles a DDLS fit is told to leave out
    (`exclude_angles`). Committed parameters and ticks only, never a value computed from
    a correlogram."""
    runs: Tuple[Dict[str, Any], ...]
    names: Tuple[str, ...] = ()
    exclude_angles: Tuple[float, ...] = ()


def _each_run(check: Callable[[Dict[str, Any]], bool]) -> Dict[str, Callable]:
    """The `holds` and `detail` of a requirement every run judged must meet: that each
    run's committed parameters pass `check`, and, when more than one run is judged, the
    names of those that do not (one run needs no naming: the tab asking names it)."""
    def holds(state: _DLSFitState) -> bool:
        return all(check(p) for p in state.runs)

    def detail(state: _DLSFitState) -> str:
        if len(state.runs) < 2:
            return ''
        return ', '.join(n for n, p in zip(state.names, state.runs, strict=True)
                         if not check(p))
    return {'holds': holds, 'detail': detail}


def _every_run_accepts(key: str, validator) -> Dict[str, Callable]:
    """`_each_run` for `key` passing `validator`, the data model's own field check:
    `_every_built_measurement_accepts`'s question asked of DLS runs."""
    return _each_run(lambda p: _accepts(validator, p.get(key), key))


def _distinct_count(key: str, validator, form) -> Callable[[_DLSFitState], int]:
    """Distinct values of `key` among the runs judged, each in its identity `form`, a
    value the data model would refuse not counted: the angle or concentration count the
    pooled fits need two of."""
    def count(state: _DLSFitState) -> int:
        return len({form(p.get(key)) for p in state.runs
                    if _accepts(validator, p.get(key), key)})
    return count


def _run_polarization_states(state: _DLSFitState) -> Set[Optional[str]]:
    """The polarization states of the runs judged, through `polarization_state`, the one
    mapping (VU and an absent tag are one unpolarized state)."""
    return {_polarization_state(p.get('analyzer_geometry')) for p in state.runs}


def _named_polarization_states(states) -> str:
    """The states a set mixes, named as the pooled fits' refusal names them."""
    return ', '.join(sorted(g if g else 'untagged' for g in states))


def _usable_solute_index(p: Dict[str, Any]) -> bool:
    """The run carries a solute refractive index `reweight_distribution` can form a
    contrast from: present, finite and above zero."""
    try:
        n = float(p.get('solute_refractive_index'))
    except (TypeError, ValueError):
        return False
    return math.isfinite(n) and n > 0


def _ddls_pairs(state: _DLSFitState) -> set:
    """The angles holding both a VV and a VH run among the runs judged, by
    `_ddls_paired_angles`, the pairing rule `run_ddls` applies."""
    return _ddls_paired_angles(
        (p['angle_deg'], p.get('analyzer_geometry')) for p in state.runs
        if _accepts(require_scattering_angle, p.get('angle_deg'), 'angle_deg'))


# The universal rows, one per Data-tab parameter `DLSMeasurement.__post_init__` refuses,
# worded as the SLS row for the same parameter is where one exists, so both tabs name a
# missing input one way.
_DLS_REQ_SOLUTE_NAME = replace(
    _REQ_SOLUTE_NAME, **_every_run_accepts('polymer_name', require_non_empty_string))
_DLS_REQ_SOLVENT_NAME = replace(
    _REQ_SOLVENT_NAME, **_every_run_accepts('solvent_name', require_non_empty_string))
_DLS_REQ_TEMPERATURE = replace(
    _REQ_TEMPERATURE, **_every_run_accepts('temperature_K', require_positive))
_DLS_REQ_ANGLE = FitRequirement(
    'angle', 'missing_param', 'a scattering angle between 0° and 180°', param='angle_deg',
    **_every_run_accepts('angle_deg', require_scattering_angle))
_DLS_REQ_WAVELENGTH = replace(
    _REQ_WAVELENGTH, **_every_run_accepts('wavelength_nm', require_positive))
_DLS_REQ_SOLVENT_INDEX = replace(
    _REQ_SOLVENT_INDEX,
    **_every_run_accepts('solvent_refractive_index', require_positive))
_DLS_REQ_ENTERED_CONCENTRATION = FitRequirement(
    'entered_concentration', 'missing_param', 'a concentration, zero or above, or none',
    param='concentration_g_per_mL',
    **_every_run_accepts('concentration_g_per_mL', _if_entered(require_non_negative)))
_DLS_REQ_ENTERED_VISCOSITY = FitRequirement(
    'entered_viscosity', 'missing_param', 'a viscosity above zero, or none',
    param='viscosity_Pa_s',
    **_every_run_accepts('viscosity_Pa_s', _if_entered(require_positive)))
_DLS_REQ_ENTERED_SOLUTE_INDEX = replace(
    _REQ_ENTERED_SOLUTE_INDEX,
    **_every_run_accepts('solute_refractive_index', _if_entered(require_positive)))

# A viscosity the data model refuses is the universal row's, so this one asks only that
# one was entered, the condition `_require_viscosity` refuses.
_DLS_REQ_VISCOSITY = FitRequirement(
    'viscosity', 'missing_param', 'a viscosity', param='viscosity_Pa_s',
    **_each_run(lambda p: p.get('viscosity_Pa_s') is not None))
_DLS_REQ_VIEW_SOLUTE_INDEX = replace(
    _REQ_MIE_SOLUTE_INDEX, name='view_solute_index', **_each_run(_usable_solute_index))
_DLS_REQ_TWO_ANGLES = FitRequirement(
    'two_angles', 'too_few_angles', 'at least two distinct angles',
    count=_distinct_count('angle_deg', require_scattering_angle, round_angle_for_identity),
    need=2)
_DLS_REQ_TWO_CONCENTRATIONS = FitRequirement(
    'two_concentrations', 'too_few_concentrations', 'at least two distinct concentrations',
    count=_distinct_count('concentration_g_per_mL', require_non_negative,
                          round_concentration_for_identity),
    need=2)
_DLS_REQ_CONCENTRATION_EACH = FitRequirement(
    'concentration_each', 'missing_param', 'a concentration on every ticked measurement',
    param='concentration_g_per_mL',
    **_each_run(lambda p: p.get('concentration_g_per_mL') is not None))
_DLS_REQ_ONE_POLARIZATION = FitRequirement(
    'one_polarization', 'mixed_polarization', 'measurements of one polarization',
    param='analyzer_geometry',
    holds=lambda state: len(_run_polarization_states(state)) < 2,
    detail=lambda state: (
        f'{_named_polarization_states(_run_polarization_states(state))} ticked'))
_DLS_REQ_VV_VH_PAIR = FitRequirement(
    'vv_vh_pair', 'no_vv_vh_pair', 'an angle with both a VV and a VH measurement',
    param='analyzer_geometry', holds=lambda state: bool(_ddls_pairs(state)))
# With no pair at all the row above names the cause, so this one holds then.
_DLS_REQ_TICKED_PAIR = FitRequirement(
    'ticked_pair', 'too_few_angles', 'a paired angle ticked',
    holds=lambda state: (not _ddls_pairs(state) or any(
        not _value_in(a, state.exclude_angles) for a in _ddls_pairs(state))))

#: What every DLS fit of a run needs: the identity and optics each run is built from,
#: an entered concentration, viscosity or solute index among them. A missing viscosity
#: is not here: the decay-rate fits run and flag the Rh they cannot give.
DLS_UNIVERSAL_REQUIREMENTS: Tuple[FitRequirement, ...] = (
    _DLS_REQ_SOLUTE_NAME, _DLS_REQ_SOLVENT_NAME, _DLS_REQ_TEMPERATURE, _DLS_REQ_ANGLE,
    _DLS_REQ_WAVELENGTH, _DLS_REQ_SOLVENT_INDEX, _DLS_REQ_ENTERED_CONCENTRATION,
    _DLS_REQ_ENTERED_VISCOSITY, _DLS_REQ_ENTERED_SOLUTE_INDEX,
)

_DLS_INVERSION = DLS_UNIVERSAL_REQUIREMENTS + (_DLS_REQ_VISCOSITY,)

#: The one declaration of what each DLS analysis needs to run, keyed by the DLS tab's own
#: keys (a sub-tab's method, view model or source; 'gamma_q2', 'conc_extrap' and 'ddls' for
#: the fits that pool runs), each judged over the runs `Controller.dls_fit_availability` is
#: handed: once per run where a tab fits runs one at a time, once for the ticked set (whose
#: rows build) where it pools, and for every run the DDLS fit builds. A view re-presents a
#: finished fit, so it asks only for its own input. Only committed inputs and ticks appear
#: here; a condition on the measured values is the fit's own refusal when it runs.
DLS_FIT_REQUIREMENTS: Dict[Union[str, Tuple[str, str]], Tuple[FitRequirement, ...]] = {
    ('correlogram', 'cumulant'): DLS_UNIVERSAL_REQUIREMENTS,
    ('correlogram', 'single'): DLS_UNIVERSAL_REQUIREMENTS,
    ('correlogram', 'double'): DLS_UNIVERSAL_REQUIREMENTS,
    ('correlogram', 'kww'): DLS_UNIVERSAL_REQUIREMENTS,
    ('distribution', 'lognormal'): _DLS_INVERSION,
    ('distribution', 'nnls'): _DLS_INVERSION,
    ('distribution', 'contin'): _DLS_INVERSION,
    ('sphere_model', optical.RAYLEIGH): (),
    ('sphere_model', optical.RDG): (_DLS_REQ_VIEW_SOLUTE_INDEX,),
    ('sphere_model', optical.MIE): (_DLS_REQ_VIEW_SOLUTE_INDEX,),
    'gamma_q2': (_DLS_REQ_TWO_ANGLES, _DLS_REQ_ONE_POLARIZATION),
    'conc_extrap': (_DLS_REQ_CONCENTRATION_EACH, _DLS_REQ_TWO_CONCENTRATIONS,
                    _DLS_REQ_ONE_POLARIZATION),
    ('mode_map', 'double'): DLS_UNIVERSAL_REQUIREMENTS,
    ('mode_map', 'contin'): _DLS_INVERSION,
    ('mode_map', 'nnls'): _DLS_INVERSION,
    'ddls': DLS_UNIVERSAL_REQUIREMENTS + (_DLS_REQ_VV_VH_PAIR, _DLS_REQ_TICKED_PAIR),
}
