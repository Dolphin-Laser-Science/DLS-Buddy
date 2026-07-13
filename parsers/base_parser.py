"""
parsers/base_parser.py
======================

Abstract base class for all instrument parsers, and the ParsedFilePreview
dataclass that carries extracted parameters from a file to the user-confirmation
step.

Architecture recap
------------------
The parser layer is a two-layer system:

    Layer 1 — format parsing:
        Instrument-specific parsers (brookhaven_dls.py, generic_dls.py, etc.)
        each know how to read one file format off disk. Their only job is to
        extract whatever they can find and return a ParsedFilePreview.

    Layer 2 — user confirmation:
        The calling code (eventually the GUI, currently a utility function)
        presents the preview to the user, fills in missing fields, lets the
        user correct pre-filled values, and then calls preview.build() to
        produce the final DLSMeasurement / SLSMeasurement / IntensityTrace.

This separation means:
  - Adding a new instrument format requires writing one new parser subclass.
    No analysis, export, or GUI code changes.
  - Instrument-specific quirks (encoding, column layout, sentinel rows) are
    fully contained within the relevant parser.
  - User confirmation is uniform across all parsers. A Brookhaven file and a
    plain-text file go through the same review step.

How to add a new instrument parser
-----------------------------------
1. Create a new file in parsers/ (e.g. parsers/malvern_dls.py).
2. Define a class that inherits from BaseDLSParser or BaseSLSParser.
3. Implement the parse() method: read the file, populate a ParsedFilePreview,
   return List[ParsedFilePreview].
4. That is all. Analysis, export, and plotting code never change.

Contents
--------
Unit conversion helpers
    convert_delay_times     convert delay times from any user-facing unit to s
    convert_temperature     convert temperature from C or K to K
    convert_concentration   convert concentration from any unit to g/mL
    convert_viscosity       convert viscosity from mPa.s or Pa.s to Pa.s
    convert_count_rate      convert a count rate from kcps/Mcps/Hz to cps
    convert_trace_times     convert trace elapsed times from ms/min/h to s

Preview dataclasses
    DLSFilePreview          one DLS measurement's extracted + pending fields
    SLSFilePreview          one SLS measurement's extracted + pending fields
    TraceFilePreview        one intensity trace's extracted + pending fields

Abstract base classes
    BaseDLSParser           must implement parse() -> List[DLSFilePreview]
    BaseSLSParser           must implement parse() -> List[SLSFilePreview]
    BaseTraceParser         must implement parse() -> List[TraceFilePreview]

Change history
--------------
2026-06-12  Initial implementation. (base_parser.py v1)
            Defines unit conversion helpers, DLSFilePreview, SLSFilePreview,
            TraceFilePreview, and abstract base classes for all three parser
            types.
2026-06-12  Added solvent_name. (base_parser.py v2)
            DLSFilePreview and SLSFilePreview gained a required solvent_name
            field (now 9 required fields each). TraceFilePreview unchanged
            (intensity traces carry no solvent). build() passes solvent_name
            through to the measurement constructors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

# We import the final data model classes so that preview.build() can
# construct them. The parsers themselves never import analysis modules.
from core.data_models import DLSMeasurement, IntensityTrace, SLSMeasurement


# ---------------------------------------------------------------------------
# Unit conversion helpers
# ---------------------------------------------------------------------------
#
# These are plain functions, not methods, so they can be called anywhere
# (parsers, tests, future GUI code) without instantiating anything.
# Each function is a single, testable conversion with a clear contract.

# Supported user-facing units and their conversion factors to the canonical
# internal unit. Adding a new unit means adding one entry to the dict.

_DELAY_TIME_FACTORS = {
    's':  1.0,
    'ms': 1.0e-3,
    'us': 1.0e-6,   # also written µs; parsers normalize the string first
    'µs': 1.0e-6,
    'ns': 1.0e-9,
}

_CONCENTRATION_FACTORS = {
    'g/ml':  1.0,
    'g/mL':  1.0,
    'mg/ml': 1.0e-3,
    'mg/mL': 1.0e-3,
}

_VISCOSITY_FACTORS = {
    'pa.s':  1.0,
    'pa·s':  1.0,
    'mpa.s': 1.0e-3,
    'mpa·s': 1.0e-3,
    'mpas':  1.0e-3,
    'cp':    1.0e-3,   # centipoise; 1 cP = 1 mPa.s exactly
}

_COUNT_RATE_FACTORS = {
    'cps':      1.0,
    'counts/s': 1.0,
    'hz':       1.0,    # photon counts per second
    'kcps':     1.0e3,
    'khz':      1.0e3,
    'mcps':     1.0e6,
    'mhz':      1.0e6,
}

# Intensity-trace *elapsed* time: longer scales than the correlogram lag, so this
# is a separate table from _DELAY_TIME_FACTORS (which has us/ns but no min/h).
_TRACE_TIME_FACTORS = {
    's':       1.0,
    'sec':     1.0,
    'second':  1.0,
    'seconds': 1.0,
    'ms':      1.0e-3,
    'min':     60.0,
    'minute':  60.0,
    'minutes': 60.0,
    'h':       3600.0,
    'hr':      3600.0,
    'hour':    3600.0,
    'hours':   3600.0,
}


def convert_delay_times(
    values: np.ndarray,
    from_unit: str,
) -> np.ndarray:
    """Convert delay time array to seconds (canonical internal unit).

    Parameters
    ----------
    values : np.ndarray
        Raw delay time values as read from the file.
    from_unit : str
        Unit string as supplied by the user or detected from the file.
        Case-insensitive. Accepted values: 's', 'ms', 'us', 'µs', 'ns'.

    Returns
    -------
    np.ndarray
        Delay times in seconds.

    Raises
    ------
    ValueError
        If from_unit is not recognized.
    """
    key = from_unit.strip().lower()
    # Normalize common unicode variant
    key = key.replace('μ', 'µ')
    if key not in _DELAY_TIME_FACTORS:
        raise ValueError(
            f"Unrecognized delay time unit {from_unit!r}. "
            f"Accepted: {sorted(_DELAY_TIME_FACTORS)}."
        )
    return np.asarray(values, dtype=float) * _DELAY_TIME_FACTORS[key]


def convert_temperature(
    value: float,
    from_unit: str,
) -> float:
    """Convert temperature to kelvin (canonical internal unit).

    Parameters
    ----------
    value : float
        Temperature value as supplied by the user.
    from_unit : str
        'C' or 'celsius' for degrees Celsius; 'K' or 'kelvin' for kelvin.
        Case-insensitive.

    Returns
    -------
    float
        Temperature in kelvin.

    Raises
    ------
    ValueError
        If from_unit is not recognized, or if the resulting kelvin value
        is non-positive (physically impossible).
    """
    unit = from_unit.strip().lower()
    if unit in ('c', 'celsius', '°c'):
        temp_K = value + 273.15
    elif unit in ('k', 'kelvin'):
        temp_K = float(value)
    else:
        raise ValueError(
            f"Unrecognized temperature unit {from_unit!r}. "
            f"Use 'C' for Celsius or 'K' for kelvin."
        )
    if not (temp_K > 0):
        raise ValueError(
            f"Converted temperature is {temp_K} K, which is non-positive. "
            f"Check the input value ({value!r} {from_unit})."
        )
    return temp_K


def convert_concentration(
    value: float,
    from_unit: str,
) -> float:
    """Convert concentration to g/mL (canonical internal unit).

    Parameters
    ----------
    value : float
        Concentration value as supplied by the user.
    from_unit : str
        Unit string. Accepted: 'g/mL', 'g/ml', 'mg/mL', 'mg/ml'.
        Case-insensitive.

    Returns
    -------
    float
        Concentration in g/mL.

    Raises
    ------
    ValueError
        If from_unit is not recognized, or if the value is negative.
    """
    key = from_unit.strip()
    # Try case-insensitive match
    matched = None
    for k in _CONCENTRATION_FACTORS:
        if k.lower() == key.lower():
            matched = k
            break
    if matched is None:
        raise ValueError(
            f"Unrecognized concentration unit {from_unit!r}. "
            f"Accepted: {list(_CONCENTRATION_FACTORS)}."
        )
    result = float(value) * _CONCENTRATION_FACTORS[matched]
    if result < 0:
        raise ValueError(
            f"Concentration must be zero or positive, got {value!r} {from_unit}."
        )
    return result


def convert_viscosity(
    value: float,
    from_unit: str,
) -> float:
    """Convert viscosity to Pa.s (canonical internal unit).

    Parameters
    ----------
    value : float
        Viscosity value as supplied by the user.
    from_unit : str
        Unit string. Accepted: 'Pa.s', 'Pa·s', 'mPa.s', 'mPa·s', 'mPas',
        'cP' (centipoise). Case-insensitive.

    Returns
    -------
    float
        Viscosity in Pa.s.

    Raises
    ------
    ValueError
        If from_unit is not recognized or value is non-positive.
    """
    key = from_unit.strip().lower()
    if key not in _VISCOSITY_FACTORS:
        raise ValueError(
            f"Unrecognized viscosity unit {from_unit!r}. "
            f"Accepted: {sorted(_VISCOSITY_FACTORS)}."
        )
    result = float(value) * _VISCOSITY_FACTORS[key]
    if not (result > 0):
        raise ValueError(
            f"Viscosity must be positive, got {value!r} {from_unit}."
        )
    return result


def convert_count_rate(
    values: np.ndarray,
    from_unit: str,
) -> np.ndarray:
    """Convert a count-rate array to cps (counts per second, canonical unit).

    Parameters
    ----------
    values : np.ndarray
        Raw count-rate values as read from the file.
    from_unit : str
        Case-insensitive. Accepted: 'cps' / 'counts/s' / 'Hz', 'kcps' / 'kHz',
        'Mcps' / 'MHz'. (A photon count rate in Hz is counts per second.)

    Returns
    -------
    np.ndarray
        Count rates in cps.

    Raises
    ------
    ValueError
        If from_unit is not recognized.
    """
    key = from_unit.strip().lower()
    if key not in _COUNT_RATE_FACTORS:
        raise ValueError(
            f"Unrecognized count-rate unit {from_unit!r}. "
            f"Accepted: {sorted(_COUNT_RATE_FACTORS)}."
        )
    return np.asarray(values, dtype=float) * _COUNT_RATE_FACTORS[key]


def convert_trace_times(
    values: np.ndarray,
    from_unit: str,
) -> np.ndarray:
    """Convert an intensity-trace elapsed-time array to seconds (canonical unit).

    Parameters
    ----------
    values : np.ndarray
        Raw trace time values as read from the file.
    from_unit : str
        Case-insensitive. Accepted: 's'/'sec'/'seconds', 'ms', 'min'/'minutes',
        'h'/'hr'/'hours'. (This differs from delay-time units, which run to
        us/ns; a count-rate trace spans seconds to hours.)

    Returns
    -------
    np.ndarray
        Times in seconds.

    Raises
    ------
    ValueError
        If from_unit is not recognized.
    """
    key = from_unit.strip().lower()
    if key not in _TRACE_TIME_FACTORS:
        raise ValueError(
            f"Unrecognized trace-time unit {from_unit!r}. "
            f"Accepted: {sorted(_TRACE_TIME_FACTORS)}."
        )
    return np.asarray(values, dtype=float) * _TRACE_TIME_FACTORS[key]


# ---------------------------------------------------------------------------
# ParsedFilePreview dataclasses
# ---------------------------------------------------------------------------
#
# A ParsedFilePreview is what a parser produces. Every field is Optional:
# the parser fills in what it can find in the file; the rest are left as None
# to be filled in by the user at the confirmation step.
#
# The .build() method on each preview validates that all required fields
# are present, applies any remaining unit conversions, and constructs the
# final data model object. Calling .build() before all required fields are
# filled in raises a ParseError with a list of which fields are still missing.
#
# Design note: ParsedFilePreview objects are transient. They live only during
# the load-and-confirm workflow and are discarded after .build() succeeds.
# They are not stored, exported, or passed to analysis functions.


class ParseError(ValueError):
    """Raised when a file cannot be parsed, or when .build() is called with
    required fields still missing.

    Carries a human-readable message and an optional list of missing field
    names so the UI layer can highlight them specifically.
    """
    def __init__(self, message: str, missing_fields: Optional[List[str]] = None):
        super().__init__(message)
        self.missing_fields: List[str] = missing_fields or []


@dataclass
class DLSFilePreview:
    """Intermediate state for one DLS measurement, between parsing and confirmation.

    Fields mirror DLSMeasurement but are all Optional. The parser populates
    what it finds; the user fills in or overrides the rest.

    The _unit_ fields record what units were used for the pre-filled values,
    so the confirmation layer can display them correctly and convert on build.

    Attributes marked '# REQUIRED' must be non-None before .build() is called.
    All others are optional in the final DLSMeasurement too.
    """

    # --- source ---
    source_file: Optional[str] = None
    instrument_name: Optional[str] = None

    # --- raw data (always populated by the parser; no unit fields needed
    #     because the parser converts to canonical units before storing) ---
    delay_times_s: Optional[np.ndarray] = None          # REQUIRED
    correlogram: Optional[np.ndarray] = None            # REQUIRED
    # (correlogram is always g2(tau)-1 by the time it reaches the preview;
    #  any g1 / un-subtracted g2 conversion happens inside the parser.)

    # --- sample identity ---
    sample_label: Optional[str] = None
    polymer_name: Optional[str] = None                  # REQUIRED
    solvent_name: Optional[str] = None                  # REQUIRED
    concentration_g_per_mL: Optional[float] = None      # REQUIRED
    temperature_K: Optional[float] = None               # REQUIRED

    # --- optics / geometry ---
    angle_deg: Optional[float] = None                   # REQUIRED
    wavelength_nm: Optional[float] = None               # REQUIRED
    solvent_refractive_index: Optional[float] = None    # REQUIRED

    # --- optional physical parameters ---
    viscosity_Pa_s: Optional[float] = None              # optional in DLSMeasurement

    def missing_required_fields(self) -> List[str]:
        """Return a list of required field names that are still None."""
        required = {
            'delay_times_s': self.delay_times_s,
            'correlogram': self.correlogram,
            'polymer_name': self.polymer_name,
            'solvent_name': self.solvent_name,
            'concentration_g_per_mL': self.concentration_g_per_mL,
            'temperature_K': self.temperature_K,
            'angle_deg': self.angle_deg,
            'wavelength_nm': self.wavelength_nm,
            'solvent_refractive_index': self.solvent_refractive_index,
        }
        return [name for name, value in required.items() if value is None]

    def is_ready(self) -> bool:
        """True if all required fields are present and .build() can succeed."""
        return len(self.missing_required_fields()) == 0

    def build(self) -> DLSMeasurement:
        """Construct and return a validated DLSMeasurement.

        Raises
        ------
        ParseError
            If any required field is still None, with a list of the
            missing field names.

        Notes
        -----
        DLSMeasurement.__post_init__ performs its own validation, so this
        method does not repeat those checks. A ParseError from here means
        fields are missing; a ValueError from DLSMeasurement means a
        field has a physically invalid value.
        """
        missing = self.missing_required_fields()
        if missing:
            raise ParseError(
                f"Cannot build DLSMeasurement: the following required fields "
                f"have not been supplied: {missing}. "
                f"These must be filled in at the parameter confirmation step.",
                missing_fields=missing,
            )
        return DLSMeasurement(
            delay_times_s=self.delay_times_s,
            correlogram=self.correlogram,
            polymer_name=self.polymer_name,
            solvent_name=self.solvent_name,
            concentration_g_per_mL=self.concentration_g_per_mL,
            temperature_K=self.temperature_K,
            angle_deg=self.angle_deg,
            wavelength_nm=self.wavelength_nm,
            solvent_refractive_index=self.solvent_refractive_index,
            viscosity_Pa_s=self.viscosity_Pa_s,
            sample_label=self.sample_label,
            instrument_name=self.instrument_name,
            source_file=self.source_file,
        )

    def __repr__(self) -> str:
        filled = sum(
            1 for v in [
                self.delay_times_s, self.correlogram, self.polymer_name,
                self.solvent_name, self.concentration_g_per_mL, self.temperature_K,
                self.angle_deg, self.wavelength_nm, self.solvent_refractive_index,
            ] if v is not None
        )
        return (
            f"DLSFilePreview(source={self.source_file!r}, "
            f"required_fields_filled={filled}/9, "
            f"ready={self.is_ready()})"
        )


@dataclass
class TraceFilePreview:
    """Intermediate state for one intensity trace, between parsing and confirmation.

    Simpler than DLSFilePreview because IntensityTrace has fewer required
    fields and no physical parameters beyond time and count rate.
    """

    # --- source ---
    source_file: Optional[str] = None
    instrument_name: Optional[str] = None

    # --- raw data ---
    times_s: Optional[np.ndarray] = None                # REQUIRED
    count_rates_cps: Optional[np.ndarray] = None        # REQUIRED

    # --- identity ---
    sample_label: Optional[str] = None
    measurement_id: Optional[str] = None                # optional back-reference

    def missing_required_fields(self) -> List[str]:
        required = {
            'times_s': self.times_s,
            'count_rates_cps': self.count_rates_cps,
        }
        return [name for name, value in required.items() if value is None]

    def is_ready(self) -> bool:
        return len(self.missing_required_fields()) == 0

    def build(self) -> IntensityTrace:
        """Construct and return a validated IntensityTrace."""
        missing = self.missing_required_fields()
        if missing:
            raise ParseError(
                f"Cannot build IntensityTrace: required fields missing: {missing}.",
                missing_fields=missing,
            )
        return IntensityTrace(
            times_s=self.times_s,
            count_rates_cps=self.count_rates_cps,
            sample_label=self.sample_label,
            measurement_id=self.measurement_id,
            source_file=self.source_file,
        )

    def __repr__(self) -> str:
        return (
            f"TraceFilePreview(source={self.source_file!r}, "
            f"ready={self.is_ready()})"
        )


@dataclass
class SLSFilePreview:
    """Intermediate state for one SLS angular series, between parsing and confirmation.

    One preview = one concentration. A multi-concentration SLS file (Zimm plot)
    produces a List[SLSFilePreview], one per concentration.
    """

    # --- source ---
    source_file: Optional[str] = None
    instrument_name: Optional[str] = None

    # --- raw data ---
    angles_deg: Optional[np.ndarray] = None             # REQUIRED
    intensities: Optional[np.ndarray] = None            # REQUIRED

    # --- sample identity ---
    sample_label: Optional[str] = None
    polymer_name: Optional[str] = None                  # REQUIRED
    solvent_name: Optional[str] = None                  # REQUIRED
    concentration_g_per_mL: Optional[float] = None      # REQUIRED
    temperature_K: Optional[float] = None               # REQUIRED

    # --- optics ---
    wavelength_nm: Optional[float] = None               # REQUIRED
    solvent_refractive_index: Optional[float] = None    # REQUIRED
    dn_dc_mL_per_g: Optional[float] = None              # REQUIRED

    # --- calibration reference (informational; parser fills from file) ---
    calibration_constant: Optional[float] = None
    standard_name: Optional[str] = None
    standard_rayleigh_ratio_file: Optional[float] = None
    standard_refractive_index: Optional[float] = None

    # --- passive load-time notices ---
    # Glyph-less, human-readable notes the parser attaches (e.g. negative intensities
    # after background subtraction). The load flow surfaces these as a passive ⓘ
    # message; the parser also emits the same text via warnings.warn (keep-both).
    notes: tuple = ()

    def missing_required_fields(self) -> List[str]:
        required = {
            'angles_deg': self.angles_deg,
            'intensities': self.intensities,
            'polymer_name': self.polymer_name,
            'solvent_name': self.solvent_name,
            'concentration_g_per_mL': self.concentration_g_per_mL,
            'temperature_K': self.temperature_K,
            'wavelength_nm': self.wavelength_nm,
            'solvent_refractive_index': self.solvent_refractive_index,
            'dn_dc_mL_per_g': self.dn_dc_mL_per_g,
        }
        return [name for name, value in required.items() if value is None]

    def is_ready(self) -> bool:
        return len(self.missing_required_fields()) == 0

    def build(self) -> SLSMeasurement:
        """Construct and return a validated SLSMeasurement."""
        missing = self.missing_required_fields()
        if missing:
            raise ParseError(
                f"Cannot build SLSMeasurement: required fields missing: {missing}.",
                missing_fields=missing,
            )
        return SLSMeasurement(
            angles_deg=self.angles_deg,
            intensities=self.intensities,
            polymer_name=self.polymer_name,
            solvent_name=self.solvent_name,
            concentration_g_per_mL=self.concentration_g_per_mL,
            temperature_K=self.temperature_K,
            wavelength_nm=self.wavelength_nm,
            solvent_refractive_index=self.solvent_refractive_index,
            dn_dc_mL_per_g=self.dn_dc_mL_per_g,
            calibration_constant=self.calibration_constant,
            standard_name=self.standard_name,
            standard_rayleigh_ratio_file=self.standard_rayleigh_ratio_file,
            standard_refractive_index=self.standard_refractive_index,
            sample_label=self.sample_label,
            instrument_name=self.instrument_name,
            source_file=self.source_file,
        )

    def __repr__(self) -> str:
        filled = sum(
            1 for v in [
                self.angles_deg, self.intensities, self.polymer_name,
                self.solvent_name, self.concentration_g_per_mL, self.temperature_K,
                self.wavelength_nm, self.solvent_refractive_index,
                self.dn_dc_mL_per_g,
            ] if v is not None
        )
        return (
            f"SLSFilePreview(source={self.source_file!r}, "
            f"c={self.concentration_g_per_mL!r} g/mL, "
            f"required_fields_filled={filled}/9, "
            f"ready={self.is_ready()})"
        )


# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------
#
# Each base class defines one abstract method: parse(). Subclasses must
# implement it. Attempting to instantiate a subclass that hasn't implemented
# parse() raises a TypeError immediately, which is much better than a
# mysterious AttributeError later.
#
# The base classes are intentionally thin. They do not enforce file extension
# checks, encoding, or any format-specific logic -- that is the subclass's
# responsibility.

class BaseDLSParser(ABC):
    """Abstract base class for all DLS file parsers.

    Subclasses must implement parse() to translate a native file format
    into a list of DLSFilePreview objects. One preview per measurement.
    The base class provides the unit conversion helpers as static methods
    so subclasses can call them without imports.

    Usage example (in a subclass)::

        class BrookhavenDLSParser(BaseDLSParser):
            def parse(self, file_path: str) -> List[DLSFilePreview]:
                preview = DLSFilePreview(source_file=file_path)
                # ... read file, populate preview fields ...
                return [preview]
    """

    @abstractmethod
    def parse(self, file_path: str) -> List[DLSFilePreview]:
        """Parse a DLS data file and return a list of previews.

        Parameters
        ----------
        file_path : str
            Absolute or relative path to the file on disk.

        Returns
        -------
        List[DLSFilePreview]
            One preview per measurement found in the file. For instruments
            that produce one measurement per file, this is always a
            one-element list. Never returns an empty list -- raise a
            ParseError instead if the file contains no usable data.

        Raises
        ------
        ParseError
            If the file cannot be read, does not match the expected format,
            or contains no usable data.
        FileNotFoundError
            If file_path does not exist on disk.
        """
        ...   # pragma: no cover

    # Expose unit converters as static methods so subclasses can call
    # self.convert_delay_times(...) without a separate import.
    convert_delay_times = staticmethod(convert_delay_times)
    convert_temperature = staticmethod(convert_temperature)
    convert_concentration = staticmethod(convert_concentration)
    convert_viscosity = staticmethod(convert_viscosity)


class BaseTraceParser(ABC):
    """Abstract base class for all intensity trace file parsers."""

    @abstractmethod
    def parse(self, file_path: str) -> List[TraceFilePreview]:
        """Parse a count rate history file and return a list of previews.

        Parameters
        ----------
        file_path : str
            Path to the file on disk.

        Returns
        -------
        List[TraceFilePreview]
            One preview per trace found in the file.

        Raises
        ------
        ParseError
            If the file cannot be read or contains no usable data.
        FileNotFoundError
            If file_path does not exist on disk.
        """
        ...   # pragma: no cover

    convert_delay_times = staticmethod(convert_delay_times)
    convert_temperature = staticmethod(convert_temperature)
    convert_concentration = staticmethod(convert_concentration)
    convert_viscosity = staticmethod(convert_viscosity)
    convert_count_rate = staticmethod(convert_count_rate)
    convert_trace_times = staticmethod(convert_trace_times)


class BaseSLSParser(ABC):
    """Abstract base class for all SLS file parsers."""

    @abstractmethod
    def parse(self, file_path: str) -> List[SLSFilePreview]:
        """Parse an SLS data file and return a list of previews.

        Parameters
        ----------
        file_path : str
            Path to the file on disk.

        Returns
        -------
        List[SLSFilePreview]
            One preview per concentration found in the file. A Zimm-plot
            file containing seven concentrations returns seven previews.

        Raises
        ------
        ParseError
            If the file cannot be read or contains no usable data.
        FileNotFoundError
            If file_path does not exist on disk.
        """
        ...   # pragma: no cover

    convert_delay_times = staticmethod(convert_delay_times)
    convert_temperature = staticmethod(convert_temperature)
    convert_concentration = staticmethod(convert_concentration)
    convert_viscosity = staticmethod(convert_viscosity)
