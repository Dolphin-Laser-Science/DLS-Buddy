"""
parsers/brookhaven_dat.py
=========================

Parser for the Brookhaven raw ASCII DLS export:

    BrookhavenDATParser     reads the exported .dat correlation function

This is a different file format from the one in ``brookhaven_dls.py``, not a
variant of it. Particle Explorer can write either, and the two share nothing
but the vendor: the ``.csv`` has a text header and pre-normalized ``C(t)`` with
no instrument metadata, while the ``.dat`` has a positional numeric header, raw
correlator counts, and the angle, wavelength, temperature, viscosity, and
refractive index that the ``.csv`` omits. Same instrument, two exports, two
parsers -- as with ``zetasizer_clipboard.py`` / ``zetasizer_export.py``.

File format
-----------
Authoritative source: Brookhaven TurboCorr / BI-DLSW manual, Appendix A "Raw
Data File Formats", pp. A-1 to A-2 (PDF pp. 89-90), which enumerates every
header parameter by position. Verified against two real exports.

    Encoding    : latin-1. Line endings CRLF.
    Lines 1-37  : one scalar per line, written '%.9e' (item 19 is a bare int).
                  The positional meanings are the _ITEM_* constants below.
    Then        : `n_channels` rows of 'delay_us, channel_contents', where the
                  contents are RAW correlator counts, not g2-1.
    Then        : a four-line trailer -- Sample ID, Operator ID, Date, Time.

Header length is read from item 17, which the manual defines as "one less than
the number of parameters preceding the x and y data" (currently 36, so 37
parameters). Reading it rather than hard-coding 37 is what the field is for:
if Brookhaven ever appends a parameter, the layout stays self-describing.

Normalization
-------------
The channel contents are raw counts, so the correlogram is

    C(tau) = contents(tau) / B - 1

with B a baseline. The file supplies both of the baselines Particle Explorer
knows about:

    calculated (item 22)  = total_counts_A * total_counts_B / n_samples
                            -- a whole-run statistic.
    measured (item 23)    = the mean of the channels item 35 .. item 36.

Both identities were confirmed numerically on both real exports (to 8 and 9
significant figures respectively); we read the header values rather than
re-deriving them, so the identities serve only as a cross-check.

How much the choice matters depends entirely on the fit it feeds, because
changing B is an AFFINE transform of the correlogram rather than a plain offset:

    C_meas = (B_calc/B_meas) * C_calc + (B_calc/B_meas - 1)

-- a slight amplitude rescale plus a constant shift. Any fit that leaves both a
baseline and an amplitude free absorbs both terms exactly, so the recovered size
does not move at all. Measured on the two real files: nonlinear cumulants
(Frisken 2001, the app default, which floats baseline and beta) and
NNLS / CONTIN / lognormal (which estimate the baseline from the long-delay tail)
all change by under 0.01 %. The LINEAR cumulant (Koppel), zero-baseline by
construction, has no such freedom and moves by 3-7 %; so does any distribution
fit whose baseline the user pins by hand. So the choice is close to free under
the program's defaults and matters where that freedom is removed. Pinned by
``test_brookhaven_dat_baseline_choice_is_absorbed_by_a_floating_baseline_fit``.

Item 8 records what the acquiring software intended
(0 = ask the user, 1 = calculated, 2 = measured). When it names one, this
parser honours it. When it says "ask" -- as both real files do -- the default
is the CALCULATED baseline, because it is a whole-run statistic that does not
depend on where the software happened to place its baseline channels. In one
of the two real files those channels sit at 105-140 us, mid-lag-range, so on a
slower sample they could land inside the decay and bias the baseline high,
which would truncate the amplitude and bias the size low. The calculated
baseline has no such failure mode. Both values and their percent difference
travel on the preview either way.

``set_baseline_choice()`` re-derives the correlogram, so the choice is
reversible **on the preview object, before ``build()``** -- which is what the
loader uses to apply the app setting. It is NOT reversible after loading: the
workspace persists the correlogram, not the raw counts, so the user-facing
route today is to reload the file with the setting flipped. A genuine post-load
toggle needs the raw counts carried through the data model.

What is deliberately NOT propagated
-----------------------------------
Item 15 (refractive index of the particle, real part) is read and kept on the
preview -- for the record, and for a surface that does not exist yet; nothing
displays it today -- but is NOT written into the built ``DLSMeasurement``.
That is not a judgement made here: ``core/data_models.py`` declares
``solute_refractive_index`` a field that is never proposed, defaulted, or
guessed (invariant 3), in the same class as dn/dc, because it is
material-specific and there is no particle library. Note the contrast with the
five parameters this parser *does* pre-fill -- angle, wavelength, temperature,
viscosity, solvent RI -- which carry no such contract and are pre-filled
legitimately, exactly as ``alv_asc.py`` and ``zetasizer_export.py`` pre-fill
theirs. Being in an instrument file is not by itself the disqualifier; the
data-model contract on this particular field is. Item 16 (imaginary part) has
no consumer in this program and is not read at all.

Change history
--------------
2026-08-03  Initial implementation. (brookhaven_dat.py v1)
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from parsers.base_parser import (
    BaseDLSParser,
    DLSFilePreview,
    MalformedDataError,
    ParseError,
    apply_data_form,
    convert_delay_times,
    convert_viscosity,
)

# ---------------------------------------------------------------------------
# Module-level constants for this format
# ---------------------------------------------------------------------------

_ENCODING = 'latin-1'
_DELIMITER = ','

_INSTRUMENT_NAME = 'Brookhaven Particle Explorer'

# Header items we read, as 1-based positions in the manual's Appendix A list.
# Named rather than inlined so the code reads against the manual's numbering.
_ITEM_TOTAL_COUNTS_A = 2
_ITEM_TOTAL_COUNTS_B = 3
_ITEM_N_SAMPLES = 4
_ITEM_FIRST_DELAY_US = 6
_ITEM_N_CHANNELS = 7
_ITEM_BASELINE_MODE = 8
_ITEM_ANGLE_DEG = 9
_ITEM_WAVELENGTH_NM = 10
_ITEM_TEMPERATURE_K = 11
_ITEM_VISCOSITY_CP = 12
_ITEM_N_LIQUID = 14
_ITEM_N_PARTICLE_RE = 15
# Item 16 (particle RI, imaginary part) has no consumer here and is not read.
_ITEM_N_PARAMS_LESS_ONE = 17
_ITEM_FIRST_CHANNEL_USED = 18
_ITEM_DELAY_MODE = 19
_ITEM_ANALYSIS_MODE = 20
_ITEM_BASELINE_CALCULATED = 22
_ITEM_BASELINE_MEASURED = 23
_ITEM_LAST_DELAY_US = 24
_ITEM_SAMPLING_TIME_US = 25
_ITEM_BASELINE_FIRST_CHANNEL = 35
_ITEM_BASELINE_LAST_CHANNEL = 36

# The sniff needs a fixed prefix to read before it knows the true header
# length; item 17 lives inside it, so this many lines are always scalars.
_SNIFF_PREFIX_LINES = _ITEM_N_PARAMS_LESS_ONE

# Item 17's value at the date of the manual: the parameter count (37) minus one.
# It is also the LOWEST value we can accept, and not by coincidence -- the highest
# item this parser reads is 36 (last measured-baseline channel), so a header of
# 36 + 1 parameters is exactly the shortest one in which every field we need
# exists. A larger value is a later Brookhaven revision appending parameters,
# which the format is self-describing precisely so that we can follow; pinning
# the sniff to 36 would have made that claim false.
_MIN_N_PARAMS_LESS_ONE = 36

# Item 19, time-delay mode: -2 constant ratio, -1 from delay file, 1 linear.
_VALID_DELAY_MODES = (-2, -1, 1)
# Item 20, analysis mode: 2 autocorrelation, 3 cross correlation, 4 test.
_VALID_ANALYSIS_MODES = (2, 3, 4)

# Item 8, which baseline the acquiring software intended to be used.
_BASELINE_MODE_PROMPT = 0
_BASELINE_MODE_CALCULATED = 1
_BASELINE_MODE_MEASURED = 2

BASELINE_CALCULATED = 'calculated'
BASELINE_MEASURED = 'measured'
VALID_BASELINE_CHOICES = (BASELINE_CALCULATED, BASELINE_MEASURED)

# Used when item 8 says "ask the user". See the module docstring for why the
# calculated baseline is the safer default.
_DEFAULT_BASELINE_CHOICE = BASELINE_CALCULATED

# The trailer after the data block: Sample ID, Operator ID, Date, Time.
_TRAILER_LINES = 4
_TRAILER_SAMPLE_ID = 0
_TRAILER_OPERATOR = 1
_TRAILER_DATE = 2
_TRAILER_TIME = 3

# Delay times are in microseconds throughout the file (items 6 and 24, and
# column 1 of every data row).
_DELAY_TIME_UNIT = 'us'
# Item 12 is in centipoise.
_VISCOSITY_UNIT = 'cp'


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_lines(file_path: str) -> List[str]:
    """Read all lines from a .dat file, stripping line endings.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ParseError
        If the file cannot be decoded.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path!r}")
    try:
        with open(file_path, encoding=_ENCODING, newline='') as fh:
            return [line.rstrip('\r\n') for line in fh]
    except (UnicodeDecodeError, OSError) as exc:
        raise ParseError(
            f"Could not read {file_path!r} with encoding {_ENCODING!r}: {exc}"
        ) from exc


def _scalar(lines: List[str], item: int) -> float:
    """Return header item `item` (1-based, per the manual) as a float.

    Raises ValueError if the line is not a single scalar -- callers in the
    sniff path catch this to reject the file.
    """
    return float(lines[item - 1].strip())


def _int_item(header: Dict[int, float], item: int, file_path: str) -> int:
    """Return header item `item` as an int, or raise ParseError.

    Every whole-number header field goes through here. `float('nan')` and
    `float('inf')` parse cleanly out of the file -- they are valid floats -- and
    only blow up at the `int()` cast, which raises ValueError / OverflowError.
    Those are NOT ParseError (ParseError subclasses ValueError, not the reverse),
    so they would escape both this layer's contract and the loader's
    `except (ParseError, FileNotFoundError)`, crashing the load instead of
    rejecting the file. One helper rather than a predicate at each of the five
    call sites: there is then no second boundary to mis-size.
    """
    value = header[item]
    # `is_integer()` before the cast, not just a guarded cast: int() TRUNCATES, so
    # a fractional 1.9 would silently become 1 and be acted on as a valid code --
    # for item 8 that means "the file specified a baseline", discarding the user's
    # setting with no warning. These fields are counts and enumerations; a
    # fractional one means the file is not what it claims, not that it wants
    # rounding. (nan/inf return False here, so the same check covers them.)
    try:
        if not float(value).is_integer():
            raise ValueError('not integral')
        return int(value)
    except (ValueError, OverflowError) as exc:
        raise MalformedDataError(
            f"{file_path!r}: header item {item} is {value!r}, which is not a "
            f"whole number."
        ) from exc


def _vendor_mean_count_rate_cps(header: Dict[int, float]) -> Optional[float]:
    """The whole-run mean count rate the header implies, in cps -- or None.

    Items 2 and 4 are the total counts and the number of correlator samples, and
    item 25 is the sampling time those samples were accumulated over, so their
    quotient is the mean photon rate over the acquisition:

        rate = total_counts_A / (n_samples * sampling_time)

    Every term is read from the file; nothing here is an instrument constant. The
    manual's Appendix A-2 names item 25 "Sampling time used to generate number of
    samples" -- the same enumeration this parser's header map comes from.

    Input A is used rather than B because the exports we hold are autocorrelation
    (item 20 = 2), where the two inputs are one detector and agree to seven figures;
    in cross correlation they are different detectors and neither is "the" rate, so
    A is reported and the caller may compare it against the trace.

    Returns None rather than raising when any term is missing, non-finite or
    non-positive: this is a convenience record the user may override, not a value
    an analysis depends on, so an odd header costs the offer and nothing else. The
    guard is two-sided -- a saturating product and an underflowing one both leave
    the quotient unusable, so the result itself is what gets checked.
    """
    total = header.get(_ITEM_TOTAL_COUNTS_A)
    n_samples = header.get(_ITEM_N_SAMPLES)
    sampling_us = header.get(_ITEM_SAMPLING_TIME_US)
    if total is None or n_samples is None or sampling_us is None:
        return None
    values = (float(total), float(n_samples), float(sampling_us))
    if not all(np.isfinite(v) for v in values):
        return None
    if values[1] <= 0.0 or values[2] <= 0.0 or values[0] < 0.0:
        return None
    # Spelled as a product then a divide, and checked on the RESULT: `*` saturates
    # to inf where `**` would raise, and a denominator that overflows to inf or
    # underflows to 0.0 is only visible in the quotient.
    duration_s = values[1] * values[2] * 1e-6
    if not np.isfinite(duration_s) or duration_s <= 0.0:
        return None
    rate = values[0] / duration_s
    if not np.isfinite(rate):
        return None
    return float(rate)


def _sniff(lines: List[str]) -> None:
    """Reject anything that is not a Brookhaven raw ASCII .dat export.

    The format has no text header to match on, so the check is structural:
    the fixed scalar prefix must parse, item 17 must carry its documented
    self-describing value, the two mode fields must hold documented codes,
    and the first data row must be a two-column numeric pair. Measured
    against the committed corpus, this accepts the .dat files and rejects
    all 182 other files in test-data/, so autodetect stays unambiguous.

    Raises
    ------
    ParseError
        If any structural check fails.
    """
    def reject(why: str) -> None:
        raise ParseError(
            f"Not a Brookhaven raw ASCII .dat correlogram: {why}."
        )

    # Shortest possible valid file: header + at least one data row + trailer.
    minimum = _SNIFF_PREFIX_LINES + 1 + _TRAILER_LINES
    if len(lines) < minimum:
        reject(f"only {len(lines)} lines, need at least {minimum}")

    try:
        prefix = [_scalar(lines, i) for i in range(1, _SNIFF_PREFIX_LINES + 1)]
    except ValueError:
        reject(f"the first {_SNIFF_PREFIX_LINES} lines are not all single numbers")

    n_params_less_one = prefix[_ITEM_N_PARAMS_LESS_ONE - 1]
    if not float(n_params_less_one).is_integer() or n_params_less_one < _MIN_N_PARAMS_LESS_ONE:
        reject(
            f"header item {_ITEM_N_PARAMS_LESS_ONE} (parameter count less one) "
            f"is {n_params_less_one!r}, expected a whole number of at least "
            f"{_MIN_N_PARAMS_LESS_ONE}"
        )

    header_len = int(n_params_less_one) + 1
    if len(lines) < header_len + 1 + _TRAILER_LINES:
        reject(f"only {len(lines)} lines for a {header_len}-parameter header")

    try:
        delay_mode = _scalar(lines, _ITEM_DELAY_MODE)
        analysis_mode = _scalar(lines, _ITEM_ANALYSIS_MODE)
    except ValueError:
        reject(f"header items {_ITEM_DELAY_MODE} and {_ITEM_ANALYSIS_MODE} are not numbers")

    if delay_mode not in _VALID_DELAY_MODES:
        reject(
            f"header item {_ITEM_DELAY_MODE} (time-delay mode) is {delay_mode!r}, "
            f"not one of {_VALID_DELAY_MODES}"
        )
    if analysis_mode not in _VALID_ANALYSIS_MODES:
        reject(
            f"header item {_ITEM_ANALYSIS_MODE} (analysis mode) is {analysis_mode!r}, "
            f"not one of {_VALID_ANALYSIS_MODES}"
        )

    first_data = lines[header_len].split(_DELIMITER)
    if len(first_data) < 2:
        reject(f"the first row after the header is not {_DELIMITER!r}-separated")
    try:
        float(first_data[0].strip())
        float(first_data[1].strip())
    except ValueError:
        reject("the first row after the header is not a numeric (delay, contents) pair")


def _read_header(lines: List[str], file_path: str) -> Tuple[Dict[int, float], int]:
    """Parse the scalar header block. Returns ({item_number: value}, header_len)."""
    header_len = int(_scalar(lines, _ITEM_N_PARAMS_LESS_ONE)) + 1
    header: Dict[int, float] = {}
    for item in range(1, header_len + 1):
        try:
            header[item] = _scalar(lines, item)
        except ValueError as exc:
            raise MalformedDataError(
                f"{file_path!r}: header item {item} is not a number "
                f"({lines[item - 1]!r})."
            ) from exc
    return header, header_len


def _read_data_block(
    lines: List[str], header_len: int, n_channels: int, file_path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Read `n_channels` 'delay, contents' rows starting after the header."""
    delay_us: List[float] = []
    contents: List[float] = []
    for offset in range(n_channels):
        row_index = header_len + offset
        raw = lines[row_index]
        parts = raw.split(_DELIMITER)
        if len(parts) < 2:
            raise MalformedDataError(
                f"{file_path!r}: data row {offset + 1} of {n_channels} "
                f"(line {row_index + 1}) is not a '{_DELIMITER}'-separated pair: {raw!r}."
            )
        try:
            tau, count = float(parts[0].strip()), float(parts[1].strip())
        except ValueError as exc:
            raise MalformedDataError(
                f"{file_path!r}: data row {offset + 1} of {n_channels} "
                f"(line {row_index + 1}) is not numeric: {raw!r}."
            ) from exc
        # 'nan' and 'inf' are numeric to float(), so the check above lets them
        # through. Unguarded they reach DLSMeasurement, which rejects non-finite
        # arrays with a bare ValueError -- at analysis time, past the loader's
        # except (ParseError, FileNotFoundError), and long after the file that
        # caused it is out of view. Reject here, where the row can be named.
        if not (math.isfinite(tau) and math.isfinite(count)):
            raise MalformedDataError(
                f"{file_path!r}: data row {offset + 1} of {n_channels} "
                f"(line {row_index + 1}) is not finite: {raw!r}."
            )
        delay_us.append(tau)
        contents.append(count)
    return np.array(delay_us, dtype=float), np.array(contents, dtype=float)


# ---------------------------------------------------------------------------
# BrookhavenDATPreview
# ---------------------------------------------------------------------------

class BrookhavenDATPreview(DLSFilePreview):
    """DLSFilePreview extended with the .dat baseline choice.

    The file stores raw correlator counts plus both baselines, so unlike every
    other instrument preview this one can legitimately be re-normalized after
    parsing. ``parse()`` applies a default so the preview is immediately usable;
    ``set_baseline_choice()`` re-derives the correlogram if the user picks the
    other baseline at the confirmation step.

    Extra attributes
    ----------------
    raw_counts : np.ndarray
        Channel contents exactly as read -- raw correlator counts, before any
        baseline normalization. Kept so the choice stays reversible.
    baseline_calculated, baseline_measured : float
        Header items 22 and 23.
    baseline_choice : str
        Which of the two is currently applied: 'calculated' or 'measured'.
    baseline_mode_from_file : int
        Header item 8 as found in the file: 0 ask, 1 calculated, 2 measured.
    baseline_channel_range : tuple[int, int]
        Header items 35 and 36 -- the 1-based inclusive channel range the
        measured baseline was averaged over.
    solute_refractive_index_from_file : float or None
        Header item 15. Displayed, never propagated -- see the module docstring.
    total_counts_a, total_counts_b, n_samples : float
        Header items 2, 3 and 4.
    first_channel_used : int
        Header item 18 -- the first channel the acquiring software used for its
        own calculations. Recorded for the user's information; this parser
        applies no masking of its own.
    operator, acquired_date, acquired_time : str or None
        The trailer fields.
    """

    def __init__(self, **kwargs):
        self.raw_counts: Optional[np.ndarray] = kwargs.pop('raw_counts', None)
        self.baseline_calculated: Optional[float] = kwargs.pop('baseline_calculated', None)
        self.baseline_measured: Optional[float] = kwargs.pop('baseline_measured', None)
        self.baseline_choice: Optional[str] = kwargs.pop('baseline_choice', None)
        self.baseline_mode_from_file: Optional[int] = kwargs.pop('baseline_mode_from_file', None)
        self.baseline_channel_range: Optional[Tuple[int, int]] = kwargs.pop(
            'baseline_channel_range', None)
        self.solute_refractive_index_from_file: Optional[float] = kwargs.pop(
            'solute_refractive_index_from_file', None)
        self.total_counts_a: Optional[float] = kwargs.pop('total_counts_a', None)
        self.total_counts_b: Optional[float] = kwargs.pop('total_counts_b', None)
        self.n_samples: Optional[float] = kwargs.pop('n_samples', None)
        self.first_channel_used: Optional[int] = kwargs.pop('first_channel_used', None)
        self.operator: Optional[str] = kwargs.pop('operator', None)
        self.acquired_date: Optional[str] = kwargs.pop('acquired_date', None)
        self.acquired_time: Optional[str] = kwargs.pop('acquired_time', None)
        super().__init__(**kwargs)

    @property
    def baseline_is_file_specified(self) -> bool:
        """True if the file's header item 8 named a baseline itself.

        False means the file said "ask the user", so the caller's own default
        (the app setting) is the one to apply.
        """
        return self.baseline_mode_from_file in (
            _BASELINE_MODE_CALCULATED, _BASELINE_MODE_MEASURED)

    @property
    def baseline_percent_difference(self) -> Optional[float]:
        """100 * (measured - calculated) / calculated, or None if unavailable.

        Particle Explorer reports the same quantity. A large value means the
        long-time correlation sits above the whole-run product, the classic
        signature of dust or a slow drifting mode.
        """
        # `is None` for absence; a separate zero check for the division, so a
        # degenerate 0.0 baseline is not silently reported as "not available".
        if self.baseline_calculated is None or self.baseline_measured is None:
            return None
        if self.baseline_calculated == 0.0:
            return None
        return 100.0 * (self.baseline_measured - self.baseline_calculated) / self.baseline_calculated

    @property
    def baseline_provenance(self) -> str:
        """One line naming the baseline that produced this correlogram.

        Under the program's default fits the choice is absorbed and changes
        nothing, but it is NOT free for a zero-baseline linear cumulant or a
        hand-pinned distribution baseline, so a result has to be able to say
        which one it came from -- and the workspace keeps only the correlogram,
        so if this is not recorded at load time the information is gone. Names
        the applied value, the alternative, and where the decision came from
        (the file's own header item 8, or the app default).
        """
        applied = (self.baseline_calculated if self.baseline_choice == BASELINE_CALCULATED
                   else self.baseline_measured)
        other_name = (BASELINE_MEASURED if self.baseline_choice == BASELINE_CALCULATED
                      else BASELINE_CALCULATED)
        other = (self.baseline_measured if self.baseline_choice == BASELINE_CALCULATED
                 else self.baseline_calculated)
        decided_by = ('file header' if self.baseline_is_file_specified
                      else 'application default')
        pct = self.baseline_percent_difference
        spread = '' if pct is None else f", {pct:+.3f} % apart"
        return (f"Brookhaven .dat: {self.baseline_choice} baseline "
                f"{applied:.6e} (chosen by {decided_by}; {other_name} "
                f"{other:.6e}{spread})")

    def set_baseline_choice(self, choice: str) -> None:
        """Re-normalize the correlogram against the other baseline.

        Parameters
        ----------
        choice : str
            'calculated' or 'measured'.

        Raises
        ------
        ParseError
            If `choice` is not one of the two, if the raw counts are missing,
            or if the requested baseline is absent or not positive-finite.
        """
        if choice not in VALID_BASELINE_CHOICES:
            raise ParseError(
                f"baseline_choice must be one of {VALID_BASELINE_CHOICES}, got {choice!r}."
            )
        if self.raw_counts is None:
            raise ParseError(
                "Cannot re-normalize: the preview carries no raw channel counts."
            )
        baseline = (self.baseline_calculated if choice == BASELINE_CALCULATED
                    else self.baseline_measured)
        # apply_data_form's 'g2' branch is (raw - B)/B and owns the
        # positive-finite guard on B, so there is no second predicate here.
        self.correlogram = apply_data_form(
            self.raw_counts, 'g2', baseline, None,
            self.source_file or '<unknown>',
        )
        self.baseline_choice = choice

    def __repr__(self) -> str:
        n = len(self.raw_counts) if self.raw_counts is not None else 0
        return (
            f"BrookhavenDATPreview(source={self.source_file!r}, "
            f"n_points={n}, "
            f"baseline_choice={self.baseline_choice!r}, "
            f"ready={self.is_ready()})"
        )


# ---------------------------------------------------------------------------
# BrookhavenDATParser
# ---------------------------------------------------------------------------

class BrookhavenDATParser(BaseDLSParser):
    """Parser for Brookhaven raw ASCII ``.dat`` DLS exports.

    Returns a one-element list containing a :class:`BrookhavenDATPreview` with
    the correlogram, the sample label, and the five physical parameters the
    file records -- angle, wavelength, temperature, viscosity, and solvent
    refractive index. Sample identity (polymer, solvent, concentration) is not
    in the file and is left for the confirmation step.

    Usage
    -----
    ::

        parser = BrookhavenDATParser()
        preview = parser.parse('path/to/export.dat')[0]

        # Optics and temperature arrive pre-filled from the file; only
        # identity is left to supply:
        preview.polymer_name = 'PVP'
        preview.solvent_name = 'reline'
        preview.concentration_g_per_mL = parser.convert_concentration(1.0, 'mg/mL')

        # Optional: normalize against the measured baseline instead.
        preview.set_baseline_choice('measured')

        measurement = preview.build()
    """

    def parse(self, file_path: str) -> List[BrookhavenDATPreview]:
        """Parse a Brookhaven raw ASCII .dat correlogram.

        Parameters
        ----------
        file_path : str
            Path to the .dat file.

        Returns
        -------
        List[BrookhavenDATPreview]
            A one-element list.

        Raises
        ------
        FileNotFoundError
            If file_path does not exist.
        ParseError
            If the file is not this format, if the header and the data block
            disagree, or if content remains after the trailer (which would
            mean a multi-record file -- see the note in the code).
        """
        lines = _read_lines(file_path)
        _sniff(lines)

        # --- format commit point ---------------------------------------------------
        # Above this line a failure means "not my format" and stays a plain ParseError,
        # so autodetect can hand the file to the next parser. Below it the file has
        # proved it IS this format, so a failure is malformed DATA and raises
        # MalformedDataError, whose message is meant for the user.

        header, header_len = _read_header(lines, file_path)

        n_channels = _int_item(header, _ITEM_N_CHANNELS, file_path)
        if n_channels <= 0:
            raise MalformedDataError(
                f"{file_path!r}: header item {_ITEM_N_CHANNELS} gives "
                f"{n_channels} data channels."
            )
        expected_lines = header_len + n_channels + _TRAILER_LINES
        if len(lines) < expected_lines:
            raise MalformedDataError(
                f"{file_path!r}: truncated. Header item {_ITEM_N_CHANNELS} declares "
                f"{n_channels} channels, so the file should have {expected_lines} "
                f"lines (header {header_len} + data {n_channels} + trailer "
                f"{_TRAILER_LINES}), but it has {len(lines)}."
            )

        delay_us, raw_counts = _read_data_block(
            lines, header_len, n_channels, file_path)

        trailer = lines[header_len + n_channels:expected_lines]

        # Anything past the trailer would mean a second record appended to the
        # file. The manual documents one record per export and we have never
        # seen otherwise, so rather than guess at the concatenation rule -- and
        # silently drop the extra data -- refuse and say what was found.
        leftover = [ln for ln in lines[expected_lines:] if ln.strip()]
        if leftover:
            raise MalformedDataError(
                f"{file_path!r}: {len(leftover)} non-blank line(s) follow the "
                f"trailer of record 1. This looks like a multi-record .dat, "
                f"which is not supported; the documented format holds one "
                f"measurement per file. First unread line: {leftover[0]!r}."
            )

        # --- consistency checks against the header's own delay bookends ---
        # The header states the first and last delay independently of the data
        # rows, so a disagreement means the file is malformed or mis-sliced.
        for item, expected, actual, which in (
            (_ITEM_FIRST_DELAY_US, header[_ITEM_FIRST_DELAY_US], delay_us[0], 'first'),
            (_ITEM_LAST_DELAY_US, header[_ITEM_LAST_DELAY_US], delay_us[-1], 'last'),
        ):
            if not np.isclose(expected, actual, rtol=1e-6, atol=0.0):
                raise MalformedDataError(
                    f"{file_path!r}: header item {item} gives the {which} delay as "
                    f"{expected!r} us, but the {which} data row has {actual!r} us."
                )

        # --- baseline ---
        baseline_calculated = header[_ITEM_BASELINE_CALCULATED]
        baseline_measured = header[_ITEM_BASELINE_MEASURED]
        baseline_mode = _int_item(header, _ITEM_BASELINE_MODE, file_path)
        if baseline_mode == _BASELINE_MODE_CALCULATED:
            choice = BASELINE_CALCULATED
        elif baseline_mode == _BASELINE_MODE_MEASURED:
            choice = BASELINE_MEASURED
        else:
            # _BASELINE_MODE_PROMPT, or an unrecognized code: fall back to the
            # documented default rather than inventing a third behaviour.
            choice = _DEFAULT_BASELINE_CHOICE

        # --- viscosity ---
        # `is not None` rather than truthiness, mirroring alv_asc.py: a header
        # viscosity of 0.0 is present-but-invalid, not absent, so it must reach
        # convert_viscosity to be rejected, and that ValueError must surface as
        # a ParseError -- this layer's contract.
        visc_cP = header.get(_ITEM_VISCOSITY_CP)
        try:
            viscosity_Pa_s = (convert_viscosity(visc_cP, _VISCOSITY_UNIT)
                              if visc_cP is not None else None)
        except ValueError as exc:
            raise MalformedDataError(
                f"{file_path!r}: invalid viscosity {visc_cP!r} cP in header item "
                f"{_ITEM_VISCOSITY_CP} ({exc})."
            ) from exc

        # --- trailer ---
        def _trailer(index: int) -> Optional[str]:
            value = trailer[index].strip() if index < len(trailer) else ''
            return value or None

        particle_ri = header.get(_ITEM_N_PARTICLE_RE)

        preview = BrookhavenDATPreview(
            source_file=os.path.abspath(file_path),
            instrument_name=_INSTRUMENT_NAME,
            delay_times_s=convert_delay_times(delay_us, _DELAY_TIME_UNIT),
            sample_label=_trailer(_TRAILER_SAMPLE_ID),
            angle_deg=header[_ITEM_ANGLE_DEG],
            wavelength_nm=header[_ITEM_WAVELENGTH_NM],
            temperature_K=header[_ITEM_TEMPERATURE_K],
            solvent_refractive_index=header[_ITEM_N_LIQUID],
            viscosity_Pa_s=viscosity_Pa_s,
            raw_counts=raw_counts,
            baseline_calculated=baseline_calculated,
            baseline_measured=baseline_measured,
            baseline_mode_from_file=baseline_mode,
            baseline_channel_range=(
                _int_item(header, _ITEM_BASELINE_FIRST_CHANNEL, file_path),
                _int_item(header, _ITEM_BASELINE_LAST_CHANNEL, file_path)),
            # `is None`, not truthiness: a header 0.0 is a present-but-wrong
            # reading, not an absent one, and must not be laundered into None.
            solute_refractive_index_from_file=particle_ri,
            total_counts_a=header[_ITEM_TOTAL_COUNTS_A],
            total_counts_b=header[_ITEM_TOTAL_COUNTS_B],
            n_samples=header[_ITEM_N_SAMPLES],
            vendor_mean_count_rate_cps=_vendor_mean_count_rate_cps(header),
            first_channel_used=_int_item(header, _ITEM_FIRST_CHANNEL_USED, file_path),
            operator=_trailer(_TRAILER_OPERATOR),
            acquired_date=_trailer(_TRAILER_DATE),
            acquired_time=_trailer(_TRAILER_TIME),
        )
        # Populates preview.correlogram from raw_counts. Raises ParseError if
        # the chosen baseline is missing or not positive-finite.
        preview.set_baseline_choice(choice)

        return [preview]
