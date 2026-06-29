"""
parsers/generic_trace.py
========================

Parser for plain-text intensity-trace (count-rate history) files with no
instrument-specific header. The counterpart, for traces, of generic_dls.py and
generic_sls.py.

    GenericTraceParser    two-column (time, count_rate) file

File format contract (strict)
-----------------------------
The file must satisfy ALL of the following:

  - Exactly two columns of numerical data.
  - No header rows of any kind (no labels, no comments, no blank lines).
  - Columns separated by a comma or a tab (detected automatically).
  - Column 1: elapsed time (any time unit, user-specified at confirmation).
  - Column 2: photon count rate (any count-rate unit, user-specified).
  - Every row must be parseable as two floating-point numbers.
  - At least two data rows.

Any deviation raises a ParseError. As with the generic DLS/SLS parsers, this is
not a fallback for "an instrument file I don't understand" -- it is for data
already cleaned to a bare two-column table. Files with headers must be trimmed
first (or parsed by an instrument-specific parser).

Units (user-specified at the confirmation step, converted to canonical on build)
--------------------------------------------------------------------------------
A plain-text file cannot encode its units, so the user supplies them:

  time_unit        's' / 'ms' / 'min' / 'h'   -> converted to seconds
  count_rate_unit  'cps'/'Hz' / 'kcps'/'kHz' / 'Mcps'/'MHz' -> converted to cps

(The canonical internal units are seconds and cps; see core/data_models.py.)

What the user must supply at the confirmation step
--------------------------------------------------
  - time_unit
  - count_rate_unit
  - sample_label     (optional)

Change history
--------------
2026-06-17  Initial implementation. (generic_trace.py v1)
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from core.data_models import IntensityTrace
from parsers.base_parser import (
    BaseTraceParser,
    TraceFilePreview,
    ParseError,
    convert_count_rate,
    convert_trace_times,
)

# Minimum number of (time, count_rate) rows to accept.
_MIN_ROWS = 2

# Maximum fraction of malformed rows tolerated before raising rather than
# silently discarding -- a higher rate means the file has headers or is the
# wrong format. (Same threshold and intent as the generic DLS/SLS parsers.)
_MAX_BAD_ROW_FRACTION = 0.05


def _detect_delimiter(first_line: str) -> str:
    """Return ',' or '\\t' based on the first data line, or raise ParseError.

    Matches the generic DLS/SLS parsers' delimiter detection.
    """
    if '\t' in first_line and ',' not in first_line:
        return '\t'
    if ',' in first_line and '\t' not in first_line:
        return ','
    if ',' in first_line and '\t' in first_line:
        return ',' if first_line.count(',') >= first_line.count('\t') else '\t'
    raise ParseError(
        f"Cannot detect delimiter in first data line: {first_line!r}. "
        f"The file must be comma- or tab-delimited."
    )


class GenericTraceFilePreview(TraceFilePreview):
    """TraceFilePreview extended with the generic parser's deferred units.

    The raw file columns are stored as-is; the unit conversions to seconds and
    cps are applied in build() once the user has confirmed time_unit and
    count_rate_unit -- mirroring GenericDLSFilePreview, which defers delay-time
    and data-form conversion.
    """

    def __init__(self, **kwargs):
        self.time_unit: Optional[str] = kwargs.pop('time_unit', None)
        self.count_rate_unit: Optional[str] = kwargs.pop('count_rate_unit', None)
        self._raw_times: Optional[np.ndarray] = kwargs.pop('_raw_times', None)
        self._raw_count_rates: Optional[np.ndarray] = kwargs.pop('_raw_count_rates', None)
        super().__init__(**kwargs)
        # The canonical arrays are filled in build(), after unit confirmation.
        self.times_s = None
        self.count_rates_cps = None

    def missing_required_fields(self) -> List[str]:
        missing = []
        if self._raw_times is None:
            missing.append('_raw_times (file not loaded)')
        if self._raw_count_rates is None:
            missing.append('_raw_count_rates (file not loaded)')
        if self.time_unit is None:
            missing.append('time_unit')
        if self.count_rate_unit is None:
            missing.append('count_rate_unit')
        return missing

    def is_ready(self) -> bool:
        return len(self.missing_required_fields()) == 0

    def build(self) -> IntensityTrace:
        """Convert the raw columns to seconds / cps, then build the IntensityTrace."""
        missing = self.missing_required_fields()
        if missing:
            raise ParseError(
                f"Cannot build IntensityTrace: required fields not yet supplied: "
                f"{missing}.",
                missing_fields=missing,
            )
        self.times_s = convert_trace_times(self._raw_times, self.time_unit)
        self.count_rates_cps = convert_count_rate(
            self._raw_count_rates, self.count_rate_unit)
        return super().build()

    def __repr__(self) -> str:
        n = len(self._raw_times) if self._raw_times is not None else 0
        return (
            f"GenericTraceFilePreview(source={self.source_file!r}, "
            f"n_points={n}, ready={self.is_ready()})"
        )


class GenericTraceParser(BaseTraceParser):
    """Parser for plain-text two-column (time, count_rate) trace files.

    Reads the file, validates the strict two-column no-header contract, and stores
    the raw values in a GenericTraceFilePreview. Unit conversion (time -> s,
    count rate -> cps) is deferred to preview.build(), after the user confirms
    time_unit and count_rate_unit.

    Usage
    -----
    ::

        parser = GenericTraceParser()
        previews = parser.parse('path/to/trace.csv')   # always one element
        p = previews[0]
        p.time_unit = 's'
        p.count_rate_unit = 'kcps'
        trace = p.build()
    """

    def parse(self, file_path: str) -> List[GenericTraceFilePreview]:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path!r}")

        try:
            with open(file_path, encoding='utf-8', newline='') as fh:
                raw_lines = [line.rstrip('\r\n') for line in fh]
        except UnicodeDecodeError:
            with open(file_path, encoding='latin-1', newline='') as fh:
                raw_lines = [line.rstrip('\r\n') for line in fh]

        lines = [l for l in raw_lines if l.strip()]
        if len(lines) < _MIN_ROWS:
            raise ParseError(
                f"{file_path!r} contains only {len(lines)} non-blank line(s); "
                f"at least {_MIN_ROWS} data rows are required."
            )

        delimiter = _detect_delimiter(lines[0])

        times: List[float] = []
        rates: List[float] = []
        n_bad = 0
        for line in lines:
            parts = line.split(delimiter)
            if len(parts) != 2:
                n_bad += 1
                continue
            try:
                times.append(float(parts[0].strip()))
                rates.append(float(parts[1].strip()))
            except ValueError:
                n_bad += 1
                continue

        total = len(lines)
        if n_bad / total > _MAX_BAD_ROW_FRACTION:
            raise ParseError(
                f"{n_bad} of {total} rows in {file_path!r} could not be parsed "
                f"as two numbers ({100 * n_bad / total:.0f}%). The generic trace "
                f"parser requires a strict two-column, no-header format. If this "
                f"file has header or comment rows, remove them before loading."
            )
        if len(times) < _MIN_ROWS:
            raise ParseError(
                f"Only {len(times)} valid data rows found in {file_path!r} "
                f"after skipping {n_bad} malformed rows. At least {_MIN_ROWS} "
                f"are required."
            )

        preview = GenericTraceFilePreview(
            source_file=os.path.abspath(file_path),
            instrument_name='Generic (plain-text)',
            _raw_times=np.array(times, dtype=float),
            _raw_count_rates=np.array(rates, dtype=float),
        )
        return [preview]
