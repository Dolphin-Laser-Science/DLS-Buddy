"""
core/data_models.py
===================

The shared internal data structures for the light scattering analysis platform.

This is the most important file in the project. Every parser produces these
objects, and every analysis function consumes them. The structures defined here
are the "common internal format" that decouples the parser layer from everything
downstream: a parser's only job is to turn a file on disk into one of these
objects, and an analysis function's only job is to turn these objects into
results. Neither side needs to know anything about the other.

Canonical internal units (decided deliberately; conversions happen at load time
inside the parsers, never here and never in analysis code):

    delay time      -> seconds            (s)
    temperature     -> kelvin             (K)
    concentration   -> grams per mL       (g/mL)
    wavelength      -> nanometres         (nm)   [q formula stays in nm/nm]
    angle           -> degrees            (deg)  [converted to radians in physics]
    viscosity       -> pascal-seconds     (Pa.s)
    count rate      -> counts per second  (cps)

Four primary objects live here:

    SampleKey      - the identity tuple used to associate DLS and SLS data
                     measured on the same sample.
    IntensityTrace - a count-rate-vs-time record. A first-class object:
                     loadable and analysable on its own, with or without an
                     associated correlogram.
    DLSMeasurement - one DLS correlogram measured at one angle.
    SLSMeasurement - one SLS angular series measured at one concentration.

Implementation note:
    These are @dataclasses (Optional[X] fields default to None where a
    measurement can exist without them). We pass eq=False because the objects
    hold NumPy arrays: the auto-generated __eq__ would compare arrays with ==,
    which returns an array rather than a bool and raises. Identity and
    association are handled by SampleKey instead.

Change history
--------------
2026-06-12  Initial implementation. (data_models v1)
            Defines SampleKey, IntensityTrace, DLSMeasurement, SLSMeasurement.
            Canonical units: s, K, g/mL, nm, deg, Pa.s, cps.
            DLSMeasurement renamed from ScatteringMeasurement for symmetry
            with SLSMeasurement.
            IntensityTrace field renamed count_rates_kcps -> count_rates_cps;
            Brookhaven parser will multiply kcps values by 1000 on load.
            Viscosity stored as Pa.s; user-facing mPa.s option handled at
            the parser / confirmation layer (multiply by 1e-3 on load).
2026-06-12  Added solvent_name. (data_models v2)
            solvent_name is now a required field on DLSMeasurement and
            SLSMeasurement and a fourth field in SampleKey, so the same
            polymer in different solvents no longer collides. Solvent names
            are normalized against a controlled vocabulary
            (normalize_solvent_name); unrecognized names are used as-is with
            a one-time warning. IntensityTrace does not carry a solvent.
2026-06-19  Added analyzer_geometry to SLSMeasurement. (data_models v3)
            Optional polarization geometry ('VV'/'VH'/'VU' or None), the bridge
            for static depolarized light scattering: a VV and a VH series sharing
            sample identity + angle pair into the Cabannes correction and the
            depolarization ratio (analysis/depolarization.py). Optional with a None
            default -> non-breaking; all existing polarized workflows are unchanged.
            DLSMeasurement gains the same field when the dynamic DPLS phase begins.
2026-06-19  Added analyzer_geometry to DLSMeasurement. (data_models v4)
            The same optional 'VV'/'VH'/'VU' field on the correlogram object, for
            the DYNAMIC depolarized analysis: a VV and a VH correlogram at one angle
            pair into the rotational diffusion coefficient D_r = (Gamma_VH -
            Gamma_VV)/6 (analysis/depolarization.analyze_ddls). Non-breaking (None
            default); workspace.build() forwards it.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import NamedTuple, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Sample identity / join key
# ---------------------------------------------------------------------------

# How many significant figures to keep when building a SampleKey from floating
# point concentration and temperature values. See make_sample_key() below for
# why this exists. Six sig figs is generous enough never to merge two genuinely
# different samples, while absorbing the tiny rounding noise that unit
# conversions introduce (e.g. converting mg/mL to g/mL).
SAMPLE_KEY_SIG_FIGS = 6


class SampleKey(NamedTuple):
    """
    Identifies a physical sample for the purpose of associating datasets.

    This is the join key that will eventually let the program pair an Rh value
    from a DLS measurement with an Rg value from an SLS measurement on the same
    sample, to compute rho = Rg / Rh -- without any future refactor.

    It is a NamedTuple, which means two SampleKeys are equal (and hash equal,
    so they work as dictionary keys) when all four fields match. That is
    exactly the behavior a join key needs.

    Fields: (polymer, solvent, concentration, temperature).
    Solvent is part of sample identity because the same polymer at the same
    concentration and temperature in two different solvents is two physically
    different samples (different chain conformation, different solvent quality).
    Including solvent lets you compare, e.g., PVP in water vs PVP in reline
    without their keys colliding. Solvent names are normalized against a small
    controlled vocabulary (see normalize_solvent_name) so that "water", "Water",
    and "H2O" all map to the same key.

    This granularity is deliberately more than the rho = Rg/Rh use case strictly
    needs (which pairs extrapolated c->0 results), because more granularity is
    always recoverable -- you can match on a subset of fields -- whereas lost
    granularity cannot be reconstructed.

    IMPORTANT: build these with make_sample_key(), not by calling SampleKey(...)
    directly. The factory rounds the floating-point fields so that values which
    are "the same number" in physical terms compare as equal, and normalizes the
    solvent name against the controlled vocabulary.
    """
    polymer_name: str
    solvent_name: str
    concentration_g_per_mL: Optional[float]   # None for a DLS-only measurement
    temperature_K: float


# ---------------------------------------------------------------------------
# Solvent name controlled vocabulary
# ---------------------------------------------------------------------------
#
# Solvent names are normalized against this table before being placed in a
# SampleKey, so that common spelling/casing variants map to one canonical form
# and do not break joins. The table maps a lowercased input string to its
# canonical name. To add a solvent, add its variants here. Unrecognized names
# are used as-is (lowercased + stripped) with a one-time warning -- the program
# stays usable, the user just gets nudged toward a canonical name.

_SOLVENT_ALIASES = {
    # water
    'water': 'water',
    'h2o': 'water',
    'di water': 'water',
    'deionized water': 'water',
    'distilled water': 'water',
    'milliq': 'water',
    'milli-q': 'water',
    # toluene
    'toluene': 'toluene',
    # reline (choline chloride : urea 1:2 deep eutectic solvent)
    'reline': 'reline',
    # glycerol
    'glycerol': 'glycerol',
    'glycerine': 'glycerol',
    'glycerin': 'glycerol',
    # tetrahydrofuran
    'thf': 'thf',
    'tetrahydrofuran': 'thf',
    # dimethylformamide
    'dmf': 'dmf',
    'dimethylformamide': 'dmf',
    'n,n-dimethylformamide': 'dmf',
    # dimethyl sulfoxide
    'dmso': 'dmso',
    'dimethyl sulfoxide': 'dmso',
    'dimethylsulfoxide': 'dmso',
    # chloroform
    'chloroform': 'chloroform',
    'chcl3': 'chloroform',
    # dichloromethane
    'dcm': 'dcm',
    'dichloromethane': 'dcm',
    'methylene chloride': 'dcm',
    # acetone
    'acetone': 'acetone',
    # methanol
    'methanol': 'methanol',
    'meoh': 'methanol',
    # ethanol
    'ethanol': 'ethanol',
    'etoh': 'ethanol',
    # acetonitrile
    'acetonitrile': 'acetonitrile',
    'mecn': 'acetonitrile',
    'acn': 'acetonitrile',
    # benzene
    'benzene': 'benzene',
    # cyclohexane
    'cyclohexane': 'cyclohexane',
    # hexane
    'hexane': 'hexane',
    'n-hexane': 'hexane',
    # ethyl acetate
    'ethyl acetate': 'ethyl acetate',
    'etoac': 'ethyl acetate',
    # n-methyl-2-pyrrolidone
    'nmp': 'nmp',
    'n-methyl-2-pyrrolidone': 'nmp',
    # ethylene glycol
    'ethylene glycol': 'ethylene glycol',
    # carbon tetrachloride
    'carbon tetrachloride': 'ccl4',
    'ccl4': 'ccl4',
    'tetrachloromethane': 'ccl4',
    # carbon disulfide
    'carbon disulfide': 'cs2',
    'carbon disulphide': 'cs2',
    'cs2': 'cs2',
}

# Tracks which unrecognized solvent names have already triggered a warning,
# so the user is warned once per unique name rather than once per measurement.
_warned_unrecognized_solvents: set = set()


def normalize_solvent_name(raw_name: str) -> str:
    """Return the canonical form of a solvent name (silent; no warning).

    Strips whitespace, lowercases, and looks up the controlled vocabulary.
    If the name is not recognized, returns the cleaned (stripped + lowercased)
    string unchanged. This function never warns; it is safe to call repeatedly
    (e.g. on every sample_key access). Use is_recognized_solvent() to decide
    whether to warn.
    """
    cleaned = raw_name.strip().lower()
    return _SOLVENT_ALIASES.get(cleaned, cleaned)


def is_recognized_solvent(raw_name: str) -> bool:
    """True if raw_name (after strip + lowercase) is in the controlled vocabulary."""
    return raw_name.strip().lower() in _SOLVENT_ALIASES


def _warn_if_unrecognized_solvent(raw_name: str) -> None:
    """Emit a one-time UserWarning if raw_name is not in the vocabulary.

    Called from the measurement classes' __post_init__ so the warning fires
    once when a measurement is created, not every time its sample_key is
    accessed. Deduplicated per unique canonical name across the session.
    """
    if is_recognized_solvent(raw_name):
        return
    canonical = normalize_solvent_name(raw_name)
    if canonical in _warned_unrecognized_solvents:
        return
    _warned_unrecognized_solvents.add(canonical)
    warnings.warn(
        f"Solvent name {raw_name!r} is not in the recognized solvent "
        f"vocabulary; using {canonical!r} as-is for the sample key. If this "
        f"is a real solvent you use often, add it to _SOLVENT_ALIASES in "
        f"core/data_models.py to silence this warning and ensure consistent "
        f"matching.",
        UserWarning,
        stacklevel=3,
    )


def _round_to_sig_figs(value: float, sig_figs: int) -> float:
    """Round a float to a fixed number of significant figures (not decimals).

    We use significant figures rather than decimal places because canonical
    units span very different magnitudes: concentration is a small number like
    0.0003638 g/mL, while temperature is a few hundred K. A fixed number of
    decimal places that suited one would ruin the other.
    """
    if value == 0.0:
        return 0.0
    # math.floor(log10(|value|)) gives the position of the leading digit.
    decimals = sig_figs - 1 - math.floor(math.log10(abs(value)))
    return round(value, decimals)


def make_sample_key(
    polymer_name: str,
    solvent_name: str,
    concentration_g_per_mL: Optional[float],
    temperature_K: float,
) -> SampleKey:
    """Canonical constructor for SampleKey. Always use this, never SampleKey().

    Normalization applied:
      - polymer_name: surrounding whitespace stripped; case preserved.
      - solvent_name: normalized against the controlled vocabulary
        (stripped, lowercased, mapped to canonical form). This function does
        NOT warn on unrecognized solvents; the measurement classes warn once
        at construction via _warn_if_unrecognized_solvent.
      - concentration and temperature: rounded to SAMPLE_KEY_SIG_FIGS
        significant figures so floating-point noise cannot break a join.
    """
    name = polymer_name.strip()
    solvent = normalize_solvent_name(solvent_name)
    # Concentration may be absent for a DLS-only measurement (size/diffusion
    # analysis does not use it). Keep None in the key so such measurements still
    # join by (polymer, solvent, temperature) without colliding with a real c.
    conc = (None if concentration_g_per_mL is None
            else _round_to_sig_figs(float(concentration_g_per_mL),
                                    SAMPLE_KEY_SIG_FIGS))
    temp = _round_to_sig_figs(float(temperature_K), SAMPLE_KEY_SIG_FIGS)
    return SampleKey(name, solvent, conc, temp)


# ---------------------------------------------------------------------------
# Small validation helpers (private to this module)
# ---------------------------------------------------------------------------
#
# Centralizing these means each dataclass validates itself the same way, with
# clear error messages, instead of repeating checks inline. Failing loudly at
# construction time -- the moment bad data enters the system -- is far easier
# to debug than a confusing error deep inside an analysis function later.

def _as_1d_float_array(value, field_name: str) -> np.ndarray:
    """Coerce input to a 1-D float NumPy array, or raise a clear error."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1:
        raise ValueError(
            f"{field_name} must be one-dimensional, but has shape {arr.shape}."
        )
    if arr.size == 0:
        raise ValueError(f"{field_name} must not be empty.")
    return arr


def _require_same_length(a, b, name_a: str, name_b: str) -> None:
    if len(a) != len(b):
        raise ValueError(
            f"{name_a} and {name_b} must have the same length "
            f"({len(a)} != {len(b)})."
        )


def _require_positive(value, field_name: str) -> None:
    if value is None or not (value > 0):
        raise ValueError(f"{field_name} must be a positive number, got {value!r}.")


def _require_non_negative(value, field_name: str) -> None:
    if value is None or value < 0:
        raise ValueError(f"{field_name} must be zero or positive, got {value!r}.")


def _require_scattering_angle(value, field_name: str) -> None:
    if value is None or not (0 < value < 180):
        raise ValueError(
            f"{field_name} must be strictly between 0 and 180 degrees, "
            f"got {value!r}."
        )


def _require_non_empty_string(value, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{field_name} must be a non-empty string, got {value!r}."
        )


# Polarization/analyzer geometry of a measurement, as (incident, analyzer):
#   VV  vertical incident, vertical analyzer     -- ordinary polarized intensity
#   VH  vertical incident, horizontal analyzer   -- depolarized intensity (DPLS)
#   VU  vertical incident, no analyzer           -- e.g. the BI-200SM
# None means the geometry was not recorded (legacy / unspecified). This vocabulary
# matches the Rayleigh-geometry codes in physics/constants.py; it is duplicated here
# (not imported) to keep core/ free of any physics/ dependency. The depolarization
# analysis pairs a VV measurement with a VH measurement by sample identity + angle.
ANALYZER_GEOMETRIES = ('VV', 'VH', 'VU')


def _normalize_analyzer_geometry(value, field_name: str):
    """Return the canonical upper-case geometry code, or None. Raise if invalid."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(
            f"{field_name} must be a string (one of {ANALYZER_GEOMETRIES}) or None, "
            f"got {value!r}."
        )
    code = value.strip().upper()
    if code not in ANALYZER_GEOMETRIES:
        raise ValueError(
            f"{field_name} must be one of {ANALYZER_GEOMETRIES} or None, got {value!r}."
        )
    return code


# ---------------------------------------------------------------------------
# IntensityTrace  (count rate vs time)
# ---------------------------------------------------------------------------

@dataclass(eq=False, repr=False)
class IntensityTrace:
    """A count-rate-vs-time record from a scattering experiment.

    This is a first-class, independent object. It can be loaded from a count
    rate file and analyzed on its own (drift, stationarity, dust spikes,
    basic statistics) with no correlogram present at all. When it does belong
    to a correlogram, the association is optional and made via measurement_id.

    Fields
    ------
    times_s
        Elapsed time of each sample, in seconds from the start of the run.
        Parsers convert native timestamp formats (e.g. HH:MM:SS.sss) into
        elapsed seconds.
    count_rates_cps
        Photon count rate at each time point, in counts per second (cps).
        The Brookhaven export is in kcps; the parser multiplies by 1000
        before constructing this object.
    measurement_id
        Optional back-reference to an associated DLSMeasurement. None if
        this trace stands alone.
    sample_label
        Human-readable label carried from the source file, if present.
    source_file
        Path to the file this trace was loaded from (provenance).
    """

    times_s: np.ndarray
    count_rates_cps: np.ndarray

    measurement_id: Optional[str] = None
    sample_label: Optional[str] = None
    source_file: Optional[str] = None

    def __post_init__(self) -> None:
        # Clean and validate inputs (runs right after the dataclass __init__).
        self.times_s = _as_1d_float_array(self.times_s, "times_s")
        self.count_rates_cps = _as_1d_float_array(
            self.count_rates_cps, "count_rates_cps"
        )
        _require_same_length(
            self.times_s, self.count_rates_cps, "times_s", "count_rates_cps"
        )

    def __repr__(self) -> str:
        # A compact repr keeps the console and debugger readable
        # instead of dumping the full arrays.
        label = self.sample_label if self.sample_label is not None else "unlabeled"
        return (
            f"IntensityTrace(label={label!r}, "
            f"n_points={self.times_s.size}, "
            f"duration_s={self.times_s[-1]:.1f})"
        )


# ---------------------------------------------------------------------------
# DLSMeasurement  (one correlogram at one angle)
# ---------------------------------------------------------------------------

@dataclass(eq=False, repr=False)
class DLSMeasurement:
    """One dynamic light scattering correlogram, measured at a single angle.

    The stored correlogram is always g2(tau) - 1, the baseline-subtracted
    intensity autocorrelation function. Parsers are responsible for converting
    whatever a file contains (raw g1, un-subtracted g2, etc.) into this single
    canonical form before constructing the object, so that all analysis code
    can assume one representation.

    Required data
    -------------
    delay_times_s, correlogram
        The measured curve. Equal-length 1-D arrays. delay_times_s is in
        seconds; correlogram is g2(tau) - 1 (dimensionless).

    Required identity  (these four together define the SampleKey)
    -------------------------------------------------------------
    polymer_name, solvent_name, concentration_g_per_mL, temperature_K

    Required optics / geometry  (needed to compute the scattering vector q)
    -----------------------------------------------------------------------
    angle_deg, wavelength_nm, solvent_refractive_index

    Optional
    --------
    viscosity_Pa_s
        Solvent viscosity at the measurement temperature, in Pa.s.
        Required only for the Stokes-Einstein step (D -> Rh). Cumulant
        fitting and Gamma/q^2 analysis do not need it, so a measurement
        may exist without it. The Stokes-Einstein function raises a clear
        error if it is called without this value.
        User-facing entry in mPa.s is supported at the parser / parameter
        confirmation layer; the value is multiplied by 1e-3 before storage.
    sample_label, instrument_name, source_file
        Provenance carried from the source file where available.
    trace
        Optional associated IntensityTrace. May be attached after
        construction: measurement.trace = some_trace.
    """

    # --- measured data ---
    delay_times_s: np.ndarray
    correlogram: np.ndarray          # g2(tau) - 1

    # --- sample identity ---
    polymer_name: str
    solvent_name: str
    # Concentration is OPTIONAL for DLS: cumulant/single/double/KWW/NNLS/CONTIN
    # and Gamma/q^2 size analysis never use it. It is still required for the
    # multi-concentration diffusion extrapolation, which raises a clear error if
    # asked to run without it. None is allowed; a supplied value must be >= 0.
    concentration_g_per_mL: Optional[float]
    temperature_K: float

    # --- optics / geometry ---
    angle_deg: float
    wavelength_nm: float
    solvent_refractive_index: float

    # --- optional ---
    viscosity_Pa_s: Optional[float] = None
    # Polarization/analyzer geometry of this correlogram, as (incident, analyzer):
    # 'VV' (polarized), 'VH' (depolarized), 'VU' (no analyzer), or None if not
    # recorded. The depolarized dynamic analysis pairs a VV with a VH correlogram
    # (same sample identity + angle) to extract the rotational diffusion coefficient
    # D_r = (Gamma_VH - Gamma_VV)/6. None leaves all existing (polarized) DLS
    # workflows unchanged.
    analyzer_geometry: Optional[str] = None
    # Free-text molecular-weight fraction label (e.g. "250k", "1M"). A WITHIN-sample
    # axis used to partition a Mw series into independent analyses; NOT part of the
    # SampleKey. None = unfractioned (the common single-sample case).
    mw_fraction: Optional[str] = None
    sample_label: Optional[str] = None
    instrument_name: Optional[str] = None
    source_file: Optional[str] = None
    trace: Optional[IntensityTrace] = None

    def __post_init__(self) -> None:
        self.delay_times_s = _as_1d_float_array(self.delay_times_s, "delay_times_s")
        self.correlogram = _as_1d_float_array(self.correlogram, "correlogram")
        _require_same_length(
            self.delay_times_s, self.correlogram, "delay_times_s", "correlogram"
        )
        _require_non_empty_string(self.polymer_name, "polymer_name")
        _require_non_empty_string(self.solvent_name, "solvent_name")
        _require_positive(self.temperature_K, "temperature_K")
        _require_positive(self.wavelength_nm, "wavelength_nm")
        _require_positive(self.solvent_refractive_index, "solvent_refractive_index")
        _require_scattering_angle(self.angle_deg, "angle_deg")
        if self.concentration_g_per_mL is not None:
            _require_non_negative(
                self.concentration_g_per_mL, "concentration_g_per_mL")
        if self.viscosity_Pa_s is not None:
            _require_positive(self.viscosity_Pa_s, "viscosity_Pa_s")
        self.analyzer_geometry = _normalize_analyzer_geometry(
            self.analyzer_geometry, "analyzer_geometry")
        _warn_if_unrecognized_solvent(self.solvent_name)

    @property
    def sample_key(self) -> SampleKey:
        """The identity of this measurement, derived from its identity fields.

        Computed on demand rather than stored, so it can never drift out of
        sync with polymer_name / solvent_name / concentration / temperature.
        """
        return make_sample_key(
            self.polymer_name, self.solvent_name,
            self.concentration_g_per_mL, self.temperature_K,
        )

    def __repr__(self) -> str:
        c = self.concentration_g_per_mL
        c_str = 'n/a' if c is None else f'{c:g} g/mL'
        return (
            f"DLSMeasurement(polymer={self.polymer_name!r}, "
            f"solvent={self.solvent_name!r}, "
            f"c={c_str}, "
            f"T={self.temperature_K:g} K, "
            f"angle={self.angle_deg:g} deg, "
            f"n_points={self.delay_times_s.size}, "
            f"trace={'yes' if self.trace is not None else 'no'})"
        )


# ---------------------------------------------------------------------------
# SLSMeasurement  (one angular series at one concentration)
# ---------------------------------------------------------------------------

@dataclass(eq=False, repr=False)
class SLSMeasurement:
    """One static light scattering angular series at a single concentration.

    One object = one concentration, measured across one or more angles. A full
    Zimm-plot dataset is a list of these (one per concentration); the pure
    solvent reference is the object whose concentration is 0. A single-angle
    measurement is just an angular series of length one -- no special case
    anywhere in the analysis code.

    Required data
    -------------
    angles_deg, intensities
        Equal-length 1-D arrays: scattering angles and measured intensity at
        each. Intensities are stored as raw instrument values; conversion to
        excess Rayleigh ratios happens in the SLS analysis module, which needs
        the solvent reference and calibration standard to do so correctly.

    Required identity
    -----------------
    polymer_name, solvent_name, concentration_g_per_mL, temperature_K

    Required optics
    ---------------
    wavelength_nm, solvent_refractive_index, dn_dc_mL_per_g

    Optional calibration reference  (informational; NOT used in analysis)
    ----------------------------------------------------------------------
    The Rayleigh ratio used in analysis is always the program-computed,
    temperature-corrected toluene value from physics/constants.py (Takahashi
    et al. 2019 at 532 nm; Sivokhin & Kazantsev 2021 at 660 nm), not the value
    embedded in the instrument file. These fields record what the file reported
    for traceability and comparison.
    """

    # --- measured data ---
    angles_deg: np.ndarray
    intensities: np.ndarray

    # --- sample identity ---
    polymer_name: str
    solvent_name: str
    concentration_g_per_mL: float
    temperature_K: float

    # --- optics ---
    wavelength_nm: float
    solvent_refractive_index: float
    dn_dc_mL_per_g: float

    # --- optional calibration reference (informational only) ---
    calibration_constant: Optional[float] = None
    standard_name: Optional[str] = None
    standard_rayleigh_ratio_file: Optional[float] = None
    standard_refractive_index: Optional[float] = None

    # --- optional polarization geometry ---
    # Analyzer/incident polarization of this intensity series, as (incident,
    # analyzer): 'VV' (polarized), 'VH' (depolarized), 'VU' (no analyzer), or None
    # if not recorded. Used by the depolarization analysis to pair a VV series with
    # a VH series (same sample identity + angle) for the Cabannes correction and the
    # depolarization ratio. None leaves all existing (polarized) workflows unchanged.
    analyzer_geometry: Optional[str] = None

    # --- optional provenance ---
    # Free-text molecular-weight fraction label (e.g. "250k", "1M"). A WITHIN-sample
    # axis that partitions a Mw series into independent Zimm fits; NOT part of the
    # SampleKey. The c = 0 solvent reference is shared across fractions.
    mw_fraction: Optional[str] = None
    sample_label: Optional[str] = None
    instrument_name: Optional[str] = None
    source_file: Optional[str] = None

    def __post_init__(self) -> None:
        self.angles_deg = _as_1d_float_array(self.angles_deg, "angles_deg")
        self.intensities = _as_1d_float_array(self.intensities, "intensities")
        _require_same_length(
            self.angles_deg, self.intensities, "angles_deg", "intensities"
        )
        for angle in self.angles_deg:
            _require_scattering_angle(angle, "each value in angles_deg")
        _require_non_empty_string(self.polymer_name, "polymer_name")
        _require_non_empty_string(self.solvent_name, "solvent_name")
        _require_positive(self.temperature_K, "temperature_K")
        _require_positive(self.wavelength_nm, "wavelength_nm")
        _require_positive(self.solvent_refractive_index, "solvent_refractive_index")
        _require_non_negative(self.concentration_g_per_mL, "concentration_g_per_mL")
        # dn/dc is deliberately never defaulted (invariant 3 -- the central
        # low-contrast vulnerability), so the common "not yet entered" path
        # arrives here as None. Guard it explicitly, like the DLS build guards
        # viscosity, so the user sees the clear ValueError below rather than a
        # TypeError from math.isfinite(None) deep in the build.
        if self.dn_dc_mL_per_g is None or not math.isfinite(self.dn_dc_mL_per_g):
            raise ValueError(
                f"dn_dc_mL_per_g must be a finite number, "
                f"got {self.dn_dc_mL_per_g!r}."
            )
        self.analyzer_geometry = _normalize_analyzer_geometry(
            self.analyzer_geometry, "analyzer_geometry"
        )
        _warn_if_unrecognized_solvent(self.solvent_name)

    @property
    def sample_key(self) -> SampleKey:
        return make_sample_key(
            self.polymer_name, self.solvent_name,
            self.concentration_g_per_mL, self.temperature_K,
        )

    def __repr__(self) -> str:
        return (
            f"SLSMeasurement(polymer={self.polymer_name!r}, "
            f"solvent={self.solvent_name!r}, "
            f"c={self.concentration_g_per_mL:g} g/mL, "
            f"T={self.temperature_K:g} K, "
            f"n_angles={self.angles_deg.size})"
        )
