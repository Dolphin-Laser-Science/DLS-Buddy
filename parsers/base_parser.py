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
    KCPS_TO_CPS             counts per second in one kcps (a kHz photon rate is the same)

Shared text-file read
    read_text_lines         a text file's lines, line endings stripped, trying the
                            caller's encodings in turn

Shared plain-table helpers (the generic parsers)
    detect_delimiter        ',' or tab, from a table's first data line
    MAX_BAD_ROW_FRACTION    the malformed-row share above which a table is refused

Correlogram form conversion
    apply_data_form         convert a raw second column (g1 / un-normalized g2 /
                            already-subtracted g2m1) to the canonical g2(tau)-1

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
2026-08-03  apply_data_form moved here from generic_dls.py, where it was the
            module-private _apply_data_form. Two parsers now need the 'g2'
            branch -- the generic plain-text parser and the Brookhaven
            TurboCorr .dat parser, whose channel contents are raw correlator
            counts. Keeping one copy keeps the positive-finite baseline guard
            in one place instead of two. generic_dls.py keeps a private alias
            so its call sites and tests are unchanged.
"""

from __future__ import annotations

import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

# We import the final data model classes so that preview.build() can
# construct them. The parsers themselves never import analysis modules.
from core.data_models import (
    KELVIN_AT_ZERO_CELSIUS,
    DLSMeasurement,
    IntensityTrace,
    SLSMeasurement,
)
# A load-time notice is a record note (core/notes.py), its words the registry's.
from core.notes import Note


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
    'mg/ml': 1.0e-3,
}

_VISCOSITY_FACTORS = {
    'pa.s':  1.0,
    'pa·s':  1.0,
    'mpa.s': 1.0e-3,
    'mpa·s': 1.0e-3,
    'mpas':  1.0e-3,
    'cp':    1.0e-3,   # centipoise; 1 cP = 1 mPa.s exactly
}

# Counts per second in one kcps; a kHz photon count rate is the same quantity.
KCPS_TO_CPS = 1.0e3

_COUNT_RATE_FACTORS = {
    'cps':      1.0,
    'counts/s': 1.0,
    'hz':       1.0,    # photon counts per second
    'kcps':     KCPS_TO_CPS,
    'khz':      KCPS_TO_CPS,
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
    # The generic previews hold their raw columns and convert on build(), so this is the
    # only point at which their delay axis passes through shared code.
    require_finite_data(values, 'delay times')
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
        temp_K = value + KELVIN_AT_ZERO_CELSIUS
    elif unit in ('k', 'kelvin'):
        temp_K = float(value)
    else:
        raise ValueError(
            f"Unrecognized temperature unit {from_unit!r}. "
            f"Use 'C' for Celsius or 'K' for kelvin."
        )
    if not (math.isfinite(temp_K) and temp_K > 0):
        raise ValueError(
            f"Converted temperature is {temp_K} K, which is not a positive finite "
            f"number. Check the input value ({value!r} {from_unit})."
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
    key = from_unit.strip().lower()
    if key not in _CONCENTRATION_FACTORS:
        raise ValueError(
            f"Unrecognized concentration unit {from_unit!r}. "
            f"Accepted: {list(_CONCENTRATION_FACTORS)}."
        )
    result = float(value) * _CONCENTRATION_FACTORS[key]
    if not (math.isfinite(result) and result >= 0):
        raise ValueError(
            f"Concentration must be a finite number, zero or positive, "
            f"got {value!r} {from_unit}."
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
    if not (math.isfinite(result) and result > 0):
        raise ValueError(
            f"Viscosity must be a positive finite number, got {value!r} {from_unit}."
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
    require_finite_data(values, 'count rates')
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
    require_finite_data(values, 'trace times')
    return np.asarray(values, dtype=float) * _TRACE_TIME_FACTORS[key]


# ---------------------------------------------------------------------------
# ParsedFilePreview dataclasses
# ---------------------------------------------------------------------------

# A ParsedFilePreview is what a parser produces. Every field is Optional: the parser fills
# in what it can find and the rest stay None, for the user to supply at the confirmation
# step. `.build()` validates that all required fields are present, applies any remaining
# unit conversions, and constructs the final data model object; called too early it raises a
# ParseError listing what is missing. Previews are transient -- they live only during the
# load-and-confirm workflow and are never stored, exported, or passed to analysis.


class ParseError(ValueError):
    """Raised when a file cannot be parsed, or when .build() is called with
    required fields still missing.

    Carries a human-readable message and an optional list of missing field
    names so the UI layer can highlight them specifically.
    """
    def __init__(self, message: str, missing_fields: Optional[List[str]] = None):
        super().__init__(message)
        self.missing_fields: List[str] = missing_fields or []


class MalformedDataError(ParseError):
    """This IS my format, and its data is unusable -- as opposed to "not my format".

    Both are ParseErrors, because both mean the parser produced nothing. The distinction
    is who the message is for. A plain ParseError from a sniff is the parser declining a
    file that belongs to someone else, and the loader is right to swallow it and try the
    next parser. A MalformedDataError is a diagnosis of a file the parser understood
    perfectly well -- naming the row, the channel and the reason -- and swallowing that
    leaves the user with a bare "could not read 1 file(s)", which is a WORSE outcome than
    before the diagnosis existed.

    A parser raises this for any failure AFTER its format-commit point: the line at which
    every earlier statement has established that no other parser can claim the file (e.g.
    Brookhaven's `C(t)` header row -- once that is seen, a bad data row is malformed data,
    not "not mine"). Each parser marks that line with a `--- format commit point ---`
    comment. `tests/test_parser_error_taxonomy.py` holds the classification.

    Subclassing rather than a flag on ParseError so that `except ParseError` at the
    boundaries that genuinely want both -- `.build()`, the Settings-driven baseline
    re-normalization -- keeps working untouched. The cost is that arm ORDER matters: a
    handler for this must precede `except ParseError`, or it is dead code. That is not
    hypothetical; it is exactly how the reverted round's version of this failed.
    """


# ---------------------------------------------------------------------------
# Shared text-file read
# ---------------------------------------------------------------------------

def read_text_lines(file_path: str, encodings: Tuple[str, ...]) -> List[str]:
    """Read a text file into a list of lines, each with its line ending stripped.

    The file is opened with an empty ``newline`` argument, so line endings reach this
    function untranslated, and each line has its trailing carriage returns and line feeds
    removed. The encodings are tried in the order given, and a ``UnicodeDecodeError``
    moves on to the next one. There is no default order: which encodings a format may
    arrive in is that format's fact, so every caller names its own.

    Raises
    ------
    FileNotFoundError
        If ``file_path`` is not an existing file.
    ParseError
        If the OS refuses the read (a file another program still holds open, a share the
        user can list but not read), or if no encoding in ``encodings`` decodes the file.
        A plain ParseError, never the malformed-data subclass: no format decision has been
        made yet, and a plain ParseError is the class ``autodetect`` skips past.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path!r}")
    last_exc: Optional[UnicodeDecodeError] = None
    for encoding in encodings:
        try:
            with open(file_path, encoding=encoding, newline='') as fh:
                return [line.rstrip('\r\n') for line in fh]
        except UnicodeDecodeError as exc:
            last_exc = exc
            continue
        except OSError as exc:
            raise ParseError(f"Could not read {file_path!r}: {exc}") from exc
    raise ParseError(
        f"Could not decode {file_path!r} with any of {encodings}: {last_exc}"
    ) from last_exc


# ---------------------------------------------------------------------------
# Shared plain-table helpers (the generic parsers)
# ---------------------------------------------------------------------------

# Maximum fraction of rows allowed to be malformed before a parser raises rather than
# silently discarding them. Above it the file almost certainly has headers or is the
# wrong format.
MAX_BAD_ROW_FRACTION = 0.05


def detect_delimiter(first_line: str) -> str:
    """Return ',' or '\\t' based on the first data line, or raise ParseError."""
    if '\t' in first_line and ',' not in first_line:
        return '\t'
    if ',' in first_line and '\t' not in first_line:
        return ','
    if ',' in first_line and '\t' in first_line:
        # Both present — comma-separated files sometimes contain tabs in
        # labels. Count occurrences and pick the majority.
        if first_line.count(',') >= first_line.count('\t'):
            return ','
        return '\t'
    raise ParseError(
        f"Cannot detect delimiter in first data line: {first_line!r}. "
        f"The file must be comma- or tab-delimited."
    )


# ---------------------------------------------------------------------------
# Shared data-column validation
# ---------------------------------------------------------------------------

def require_finite_data(
    values: Optional[np.ndarray],
    field_name: str,
    file_path: Optional[str] = None,
) -> None:
    """Reject a data column containing nan or +/-inf, naming where it came from.

    The single finiteness guard for every parser. `float()` accepts the literal tokens
    'nan', 'inf' and '1e400', and a blank cell filled with np.nan outright would let a
    non-finite value reach the workspace intact.
    `DLSMeasurement` does reject it -- but at *Run*, with a bare ValueError naming
    neither the file nor the channel, long after the file that caused it is out of view.
    Rejecting here converts that into a ParseError at load, which is what the parser
    boundary is for.

    Deliberately shared rather than repeated per parser: with nine parsers, N hand-added
    predicates is the shape that lets the tenth silently opt out (rule 10). Callers are
    the preview dataclasses' ``__post_init__`` and the shared converters, which together
    cover every route a numeric column takes into a preview.

    ``None`` and empty arrays pass: absence is the required-field layer's business, not
    this one's.
    """
    if values is None:
        return
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return
    bad = ~np.isfinite(arr)
    if not bad.any():
        return
    where = np.flatnonzero(bad)
    source = f'{file_path!r}: ' if file_path else ''
    shown = ', '.join(f'[{i}]={arr[i]!r}' for i in where[:3])
    if where.size > 3:
        shown += f', ... ({where.size} in total)'
    raise MalformedDataError(
        f"{source}{field_name} must contain only finite numbers, but "
        f"{where.size} of {arr.size} value(s) are nan or inf: {shown}. "
        f"The file is malformed at those positions -- a blank or non-numeric cell, "
        f"or a value too large to represent."
    )


# ---------------------------------------------------------------------------
# Correlogram form conversion
# ---------------------------------------------------------------------------

VALID_DATA_FORMS = ('g2m1', 'g2', 'g1')

# How far the g2 baseline B may sit from the raw data's own scale before it is rejected, as
# a factor either side.

# SIZED FROM THE RELATION THE DATA OBEYS, not picked as a round number. Here the raw column
# IS un-subtracted g2, so raw = B(1 + beta|g1|^2) with beta|g1|^2 in [0, 1]. That puts raw/B
# in [1, 2] and B/median(|raw|) in [0.5, 1] by construction, so two decades either side is
# already ~100x more slack than the quantity can take. A wider band leaves the silent
# failure reachable: B at 1e5 x scale returns a uniform -0.99999 correlogram, finite, smooth
# and shaped like data. Scale-RELATIVE because the raw column may be counts, kHz or
# already-normalized values, and no absolute bound serves all three.
BASELINE_SCALE_TOLERANCE = 1.0e2

# The band a g1 column's PEAK must lie in.
#
# SIZED FROM THE RELATION THE DATA OBEYS, like the baseline band above. Here the column IS
# the normalized field autocorrelation, so |g1(tau)| <= 1 with |g1| -> 1 as tau -> 0; a
# legitimate column peaks just under 1 for every input, fixing both edges of ONE band on
# max|raw|. The PEAK, not the median: the relation pins |g1| at tau -> 0, while a median sits
# at an arbitrary point on the decay, obeying nothing. A quantile would be more robust to one
# outlying channel -- and would hide the single out-of-range channel this exists to catch.

# CEILING: the |g1| <= 1 ceiling plus an allowance for an instrument that normalizes
# against a measured rather than a computed baseline and can report a first channel a per
# cent or two above 1.
G1_PEAK_CEILING = 1.05
# FLOOR: two decades below full correlation. There is no PHYSICAL floor -- a column whose
# decay ran out before the first lag peaks low -- but such a column carries no measurable
# correlation, and this is the SILENT tail: 1e-200 squares to exactly 0.0 and returns an
# all-zero correlogram that is finite, flat and shaped like a dead channel (measured).
# Sizing the floor to the ARITHMETIC instead (|x|^2 underflows only below ~2.2e-162) is the
# mistake BASELINE_SCALE_TOLERANCE's own comment names: it rejects the absurd tail and
# leaves a peak of 1e-100 -- an equally dead channel -- accepted.
G1_PEAK_FLOOR = 1.0e-2
# WHAT THIS BAND DOES NOT DECIDE. A raw g2 column mis-declared as g1 peaks at 1 + beta. For
# any working instrument (beta > 0.05) that is above the ceiling and rejected; as beta -> 0
# it converges on 1 and becomes indistinguishable from a legitimate g1 by ANY magnitude
# test. So the message below states the measured peak and offers 'g2' as one possibility in
# a list, never as a diagnosis: an earlier attempt asserted the diagnosis while its bound
# let through the exact low-coherence case the assertion was wrong about. The remedy for the
# undecidable case is PREVENTION -- the tooltip on the load dialog's "Column 2 is" control.


def carried_run_duration(vendor_duration_s: Optional[float],
                         delay_s: np.ndarray,
                         live: np.ndarray) -> Tuple[Optional[float], Optional[Note]]:
    """`(duration_s, note)` -- the run duration T to carry onto a measurement, or the
    `parser.duration_withheld` record of why not (None when T is carried or absent).

    ONE rule for every format that states a run length, because there is one reason it
    can be used: Biganzoli & Ferri Eq. (9)'s `1/(M - k) = Delta_t / (T - tau)` needs a T
    longer than every lag the fit will reach, and the engine refuses the weight outright
    when it is not. A file whose header disagrees with its own data is not a guess this
    layer is entitled to resolve, so T is withheld, the weight takes its documented
    `T -> infinity` form, and a record note says what was seen.

    The test is on the LIVE lags, not the declared delay range. A layout whose channels
    run past the end of the acquisition is ordinary: those channels collect nothing,
    arrive as `g2 - 1 = -1` exactly, and every DLS fitter drops them before it fits
    (`analysis.dls._common.live_channels`), so no lag past T ever reaches the weight;
    that property is pinned over the committed corpus by
    `tests/test_parsers.py::test_every_committed_brookhaven_dat_fills_its_run_duration_or_says_why`
    rather than restated as a count here.

    `live` is the caller's liveness mask, computed in whichever layer holds the data --
    a raw channel count above zero, or `g2 - 1 > -1` where a format ships that directly.
    The two spellings are the same predicate (`g2 - 1 = (raw - B)/B` with `B > 0`), and
    their agreement is asserted on the committed corpus in `tests/test_parsers.py` rather
    than argued here.

    The predicate is the quantifier "every live lag lies inside T", so a correlogram with
    no live channel at all satisfies it vacuously and T is carried. Such a file has no fit
    either way -- the live-channel mask refuses it first -- so the vacuous branch decides
    nothing a user sees.
    """
    if vendor_duration_s is None:
        return None, None
    live = np.asarray(live, dtype=bool)
    if not live.any():
        return float(vendor_duration_s), None
    last_live_s = float(np.asarray(delay_s, dtype=float)[live].max())
    if last_live_s < vendor_duration_s:
        return float(vendor_duration_s), None
    return None, Note('parser.duration_withheld', ('duration_s',),
                      {'duration': float(vendor_duration_s), 'last_lag': last_live_s})


def apply_data_form(
    raw_col2: np.ndarray,
    data_form: str,
    baseline_B: Optional[float],
    beta: Optional[float],
    file_path: str,
) -> np.ndarray:
    """Convert raw column-2 values to g²(τ)-1 according to data_form.

    Shared by every parser whose file does not already store the canonical
    g²(τ)-1: the generic plain-text parser (where the user declares the form)
    and the Brookhaven TurboCorr .dat parser (whose channel contents are raw
    correlator counts, i.e. data_form 'g2' with the baseline read from the
    file header).

    Parameters
    ----------
    raw_col2 : np.ndarray
        Values as read from the file.
    data_form : str
        One of 'g2m1', 'g2', 'g1'.
    baseline_B : float or None
        Required if data_form == 'g2'. The baseline value B such that
        g²(τ)-1 = (G(τ) - B) / B.
    beta : float or None
        Required if data_form == 'g1'. The coherence factor β such that
        g²(τ)-1 = β · |g¹(τ)|².
    file_path : str
        Used only in error messages.

    Returns
    -------
    np.ndarray
        g²(τ)-1 values.

    Raises
    ------
    ParseError
        If a required parameter (B or β) is missing or physically invalid.
    """
    # The generic and .dat paths assign the correlogram AFTER their preview is built, so
    # the preview's own __post_init__ guard never sees it. This is the funnel both go
    # through, so the same check belongs here. Note this is NOT the magnitude guard below
    # -- a finite-input check cannot substitute for it,
    # since B = 1e+300 is perfectly finite and still ruins the quotient.
    require_finite_data(raw_col2, 'the raw data column', file_path)

    if data_form == 'g2m1':
        # No magnitude band here, deliberately: this branch performs no arithmetic, so
        # there is no operation to size a guard to. A |g2-1| plausibility ceiling would be
        # a statement about the CANONICAL
        # correlogram, which every parser produces directly -- guarding it here would cover
        # one of six producers, which is rule 10's asymmetry in reverse. Its home is
        # DLSFilePreview/DLSMeasurement, and it opens its own thread at N = 0.
        return raw_col2.copy()

    if data_form == 'g2':
        if baseline_B is None:
            raise ParseError(
                f"data_form='g2' requires baseline_B to be supplied, "
                f"but it is None. Set preview.baseline_B before calling "
                f"preview.build() for {file_path!r}."
            )
        if not (math.isfinite(baseline_B) and baseline_B > 0):
            raise ParseError(
                f"baseline_B must be a positive finite number, got {baseline_B!r}."
            )
        # The check above bounds B's SIGN; `(raw - B)/B` is sensitive to its MAGNITUDE.
        # Closing that gap needs a bound on both tails, because they fail differently and
        # only one announces itself: B = 5e-324 overflows the quotient to inf, while
        # B = 1e+300 returns a uniform -1.0 that is finite, smooth and shaped like data. A
        # guard on non-finite OUTPUT would catch only the loud tail, so the bound is on the
        # input, relative to the data's own scale.
        scale = float(np.median(np.abs(raw_col2)))
        if not (math.isfinite(scale) and scale > 0.0):
            raise MalformedDataError(
                f"{file_path!r}: cannot judge baseline_B={baseline_B!r} because the raw "
                f"data column has no usable scale (its median magnitude is {scale!r}). "
                f"More than half the column is zero or non-finite, which is itself a "
                f"dead correlogram: every such channel converts to exactly -1.0."
            )
        # ONE computation, so there is no second predicate that could be mis-sized
        # against the operation it protects (rule 12's "prefer removing the gap").
        # Both saturation tails land OUTSIDE the band, which is what makes the single
        # ratio sufficient: B=1e300 over scale=1e-300 saturates to inf, and B=1e-300
        # over scale=1e300 underflows to 0.0 -- each rejected by the bound it exceeds.
        ratio = baseline_B / scale
        if not (1.0 / BASELINE_SCALE_TOLERANCE <= ratio <= BASELINE_SCALE_TOLERANCE):
            raise ParseError(
                f"{file_path!r}: baseline_B={baseline_B!r} is {ratio:.3g} times the raw "
                f"data's own scale (median magnitude {scale:.6g}), beyond the "
                f"{BASELINE_SCALE_TOLERANCE:.0e}x tolerance either side. B is the "
                f"long-time limit of that column, so a baseline this far from it is "
                f"wrong by construction -- check the value and the column's units."
            )
        return (raw_col2 - baseline_B) / baseline_B

    if data_form == 'g1':
        if beta is None:
            raise ParseError(
                f"data_form='g1' requires beta (coherence factor) to be "
                f"supplied, but it is None. Set preview.beta before calling "
                f"preview.build() for {file_path!r}."
            )
        if not (0 < beta <= 1.0):
            raise ParseError(
                f"beta must be in the range (0, 1], got {beta!r}. "
                f"The coherence factor is an instrument property, typically "
                f"between 0.01 and 1.0."
            )
        # The two guards above bound BETA; the operation below is equally sensitive to the
        # COLUMN's magnitude, whose two tails fail differently and only one of which
        # announces itself. 1e+200 squares to inf, escaping to *Run* as an unnamed
        # "correlogram must contain only finite numbers"; 1e-200 squares to exactly 0.0, and
        # an all-zero correlogram is finite, flat and shaped like a dead channel, accepted
        # in silence. Hence ONE band on ONE quantity with both edges anchored to the same
        # relation. `initial=0.0` keeps an empty column inside the same predicate (peak
        # 0.0 -> rejected) instead of numpy's bare zero-size-reduction ValueError.
        peak = float(np.max(np.abs(raw_col2), initial=0.0))
        # A PLAIN ParseError, i.e. the DIALOG's channel rather than the batched
        # malformed-file box. This band fires because the COLUMN does not match the
        # DECLARED form, a setting the user picked in that dialog -- so it belongs with
        # every other way of getting the g1 choice wrong (a missing or unparseable beta),
        # all of which come back inline through `_try_generic_dls`'s ParseError arm. In the
        # batched box it would tell the user to change a setting inside a box telling them
        # to fix the file. The load is NOT thereby retryable: the user re-loads to try
        # another data form. Only `generic_dls` reaches here; `brookhaven_dat` passes 'g2'.
        if not (G1_PEAK_FLOOR <= peak <= G1_PEAK_CEILING):
            raise ParseError(
                f"{file_path!r}: data_form='g1' but the raw data column peaks at "
                f"{peak:.6g}, outside the [{G1_PEAK_FLOOR:g}, {G1_PEAK_CEILING:g}] range a "
                f"normalized g1 can take. g1 is the NORMALIZED field autocorrelation, so "
                f"|g1| <= 1 and its first channel sits just under 1. Too large usually "
                f"means raw correlator counts or kHz, an un-normalized or un-subtracted g2 "
                f"(choose the 'g2(t)' data form and give its baseline B), or a column in "
                f"per cent; too small means the column carries no measurable correlation, "
                f"or the two columns are the wrong way round. Check the 'Column 2 is' "
                f"setting."
            )
        # Siegert relation: g²(τ)-1 = β · |g¹(τ)|²
        return beta * np.abs(raw_col2) ** 2

    raise ParseError(
        f"Unknown data_form {data_form!r}. "
        f"Accepted values: {VALID_DATA_FORMS}."
    )


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

    # The numeric columns require_finite_data() guards at construction. Declared as a
    # roster rather than checked inline so a generic scan can assert every preview
    # publishes one (rule 11) -- a preview that grows a third column and forgets to list
    # it here turns the drift guard red instead of quietly opting out.
    _DATA_ARRAY_FIELDS = ('delay_times_s', 'correlogram', 'sample_times_s')

    def __post_init__(self) -> None:
        for name in self._DATA_ARRAY_FIELDS:
            require_finite_data(getattr(self, name), name, self.source_file)

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

    # --- what the instrument file itself recorded (records, not knobs) ---
    # The vendor's own mean count rate for this measurement (cps) and the nominal
    # run length (s), where the format carries them, kept verbatim. Offered to the user
    # like any other header value the parser proposes; the records themselves never reach
    # DLSMeasurement (the run duration does, separately, as `duration_s` below).
    vendor_mean_count_rate_cps: Optional[float] = None
    vendor_duration_s: Optional[float] = None
    # Per-channel sample time (s) when the FILE states its sampling structure -- the
    # Brookhaven .dat's correlator sections. None for every format that states none;
    # the engine then derives a stand-in from the delay axis and records which it used.
    # Carried onto DLSMeasurement: an instrument fact like the lags, not a knob.
    sample_times_s: Optional[np.ndarray] = None
    # The run duration T (s) of THIS correlogram when the file states it unambiguously
    # (the ALV `Duration` of a one-run acquisition; the Brookhaven header's run length
    # where it covers every live channel -- see `carried_run_duration`); None otherwise,
    # and the weight then
    # takes its T -> infinity form. Carried onto DLSMeasurement like the sample times.
    duration_s: Optional[float] = None
    # The count-rate trace this file recorded BESIDE this correlogram, for the
    # formats that ship both in one file (currently ALV .ASC). The loader stores it
    # in the workspace's trace collection with a link back to the measurement it
    # creates. Deliberately not carried onto DLSMeasurement: traces live in one
    # place, so that a standalone solvent or calibrant trace is representable too.
    trace_preview: Optional['TraceFilePreview'] = None

    # --- load-time notices ---
    # Record notes the parser attaches (`parser.<name>` codes, e.g. a record column that
    # had a header but no data and was not loaded). The load flow renders them into a
    # passive ⓘ message (`collect_parse_notes`); the parser also emits each note's export
    # clause via warnings.warn (keep-both).
    notes: Tuple[Note, ...] = ()

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
            sample_times_s=self.sample_times_s,
            duration_s=self.duration_s,
            # Deliberately none: the vendor's mean count rate is an OFFER
            # (`vendor_mean_count_rate_cps`) that reaches the engine only through the
            # user's commit, in `LoadedMeasurement.build()`. Named so the omission is
            # visible at the one other place a DLSMeasurement is constructed.
            mean_count_rate_cps=None,
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

    # --- load-time notices ---
    # See DLSFilePreview.notes. Carried here for the same reason the finiteness roster is:
    # the three preview types are one family with three loaders, and wiring two of three is
    # the shape that leaves the third silently opted out (rule 10) -- which is exactly what
    # the reverted round did with the autodetect loops one file over.
    notes: Tuple[Note, ...] = ()

    # See DLSFilePreview._DATA_ARRAY_FIELDS for why this roster is declared, not inline.
    _DATA_ARRAY_FIELDS = ('times_s', 'count_rates_cps')

    def __post_init__(self) -> None:
        for name in self._DATA_ARRAY_FIELDS:
            require_finite_data(getattr(self, name), name, self.source_file)

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
    # NOT required: dn/dc enters only the optical constant K, which cancels in every
    # coefficient ratio, so the scale-free analyses (every Rg, the form-factor sizes,
    # the calibration-free 2*A2*Mw) run without it. Absent -> the run is flagged and
    # noted, exactly as an absent calibration constant is. See SLSMeasurement's
    # "Optional optics". Still never proposed or defaulted (invariant 3).
    dn_dc_mL_per_g: Optional[float] = None

    # --- calibration reference (informational; parser fills from file) ---
    calibration_constant: Optional[float] = None
    standard_name: Optional[str] = None
    standard_rayleigh_ratio_file: Optional[float] = None
    standard_refractive_index: Optional[float] = None

    # --- load-time notices ---
    # Record notes the parser attaches (e.g. negative intensities after background
    # subtraction); see DLSFilePreview.notes.
    notes: Tuple[Note, ...] = ()

    # See DLSFilePreview._DATA_ARRAY_FIELDS for why this roster is declared, not inline.
    _DATA_ARRAY_FIELDS = ('angles_deg', 'intensities')

    def __post_init__(self) -> None:
        for name in self._DATA_ARRAY_FIELDS:
            require_finite_data(getattr(self, name), name, self.source_file)

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
            ] if v is not None
        )
        return (
            f"SLSFilePreview(source={self.source_file!r}, "
            f"c={self.concentration_g_per_mL!r} g/mL, "
            f"required_fields_filled={filled}/8, "
            f"ready={self.is_ready()})"
        )


def collect_parse_notes(previews, file_path: str, into: list) -> None:
    """Append each preview's notes to `into` as `(file name, note)` pairs, ONCE each.

    The load flow shows `into` in a single passive ⓘ box (`gui.widgets.show_parse_notes`,
    which renders each note's clause and its remedy), so this is the point where the three
    loaders (DLS, SLS, trace) share one behaviour instead of three copies -- the same
    rule-10 reasoning as `require_finite_data`. The note itself travels, never its words:
    the clause is the registry's and the remedy the GUI's.

    De-duplication is not cosmetic. A note describes the FILE, and a multi-record parser
    hands the same note to every preview it returns (a preview must not disagree with its
    own file's parse), so a twenty-record clipboard paste would otherwise show the same
    note nineteen times. A pair is one file and one note (equal code, attributes and
    values).

    `getattr(p, 'notes', ())` rather than `p.notes`: a hand-built preview in a test, or a
    future preview type that has not been enrolled, degrades to "no notes" instead of
    raising inside a loader.
    """
    base = os.path.basename(file_path)
    for p in previews:
        for note in getattr(p, 'notes', ()):
            pair = (base, note)
            if pair not in into:
                into.append(pair)


# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------
#
# Each base class defines one abstract method, parse(); a subclass that has not implemented
# it raises TypeError on instantiation rather than a mysterious AttributeError later. The
# base classes are intentionally thin: they do not enforce file extension checks, encoding,
# or any format-specific logic -- that is the subclass's responsibility.

class BaseDLSParser(ABC):
    """Abstract base class for all DLS file parsers.

    Subclasses must implement parse() to translate a native file format
    into a list of DLSFilePreview objects. One preview per measurement.
    The base class provides the unit conversion helpers as static methods
    so subclasses can call them without imports.

    Usage example (in a subclass)::

        class BrookhavenDLSParser(BaseDLSParser):
            def parse(self, file_path: str) -> Sequence[DLSFilePreview]:
                preview = DLSFilePreview(source_file=file_path)
                # ... read file, populate preview fields ...
                return [preview]
    """

    @abstractmethod
    def parse(self, file_path: str) -> Sequence[DLSFilePreview]:
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
    def parse(self, file_path: str) -> Sequence[TraceFilePreview]:
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
    def parse(self, file_path: str) -> Sequence[SLSFilePreview]:
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
