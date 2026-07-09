"""
parsers/brookhaven_dls.py
=========================

Parsers for Brookhaven Particle Explorer DLS exports:

    BrookhavenDLSParser     reads the correlation function .CSV file
    BrookhavenTraceParser   reads the count rate history .CSV file

Both inherit from the appropriate abstract base class in base_parser.py and
produce preview objects that the user confirms before final data model objects
are constructed.

File format notes (Brookhaven Particle Explorer, confirmed from real exports)
-----------------------------------------------------------------------------

Correlation function .CSV
    Encoding    : ISO-8859-1 (latin-1). The column header contains a µ
                  character (byte 0xB5) that is not valid UTF-8.
    Line endings: CRLF (\r\n)
    Row 1       : Sample label string, followed by two empty comma fields.
                  e.g. 'PnVP (40k) in Water - 6/1/2026 4:46:59 PM,,'
    Row 2       : Column headers: 't(µs),C(t),Fitted (CONTIN)'
    Rows 3+     : Data rows: delay_time_us, g2m1, contin_fit (three columns)
    Padding rows: Unused correlator channels are filled with sentinel rows:
                  t=0, C(t)=-1, contin_value. These appear after all real
                  data. Sentinel condition: t == 0.0 AND C(t) == -1.0.
                  NOTE: real data near the baseline can have small negative
                  C(t) values (noise around zero), so C(t) < 0 alone is NOT
                  a reliable sentinel. Both conditions must be true together.
    Column 3    : The Brookhaven CONTIN fit. Silently ignored -- this program
                  performs its own CONTIN analysis.
    Metadata    : The file contains NO instrument metadata (no angle, no
                  wavelength, no temperature, no concentration). All physical
                  parameters must be supplied by the user at confirmation.

Count rate history .CSV
    Encoding    : ASCII (no non-ASCII characters)
    Line endings: CRLF (\r\n)
    Row 1       : Sample label string followed by one empty comma field.
                  e.g. 'Pn VP (40k) in Water - 6/1/2026 4:46:59 PM,'
                  NOTE: The label in this file may differ slightly from the
                  label in the correlogram file (spacing differences have been
                  observed). Pairing of the two files must be user-confirmed.
    Row 2       : Column headers: 'Time (seconds),Count Rate (kcps)'
    Rows 3+     : HH:MM:SS.sssssss timestamp, count_rate_kcps
    Timestamps  : Elapsed time from start of run (not wall-clock absolute
                  time). The first timestamp is when the first sample
                  completed, not 00:00:00.
    Count rates : In kilocounts per second (kcps). Multiplied by 1000 on
                  load to convert to the canonical internal unit (cps).
    No padding  : Count rate files have no sentinel rows.

Change history
--------------
2026-06-12  Initial implementation. (brookhaven_dls.py v1)
2026-06-12  solvent_name added to the data model. The DLS file contains no
            solvent metadata, so BrookhavenDLSParser leaves solvent_name as
            None in the preview; the user supplies it at the confirmation
            step (docstring example updated). No parsing-logic change.
            (Note: the sample label often contains the solvent, e.g.
            "PnVP (40k) in Water", but auto-extracting it is too fragile to
            rely on, so it is left to the user.)
"""

from __future__ import annotations

import os
from typing import List

import numpy as np

from parsers.base_parser import (
    BaseDLSParser,
    BaseTraceParser,
    DLSFilePreview,
    ParseError,
    TraceFilePreview,
)

# ---------------------------------------------------------------------------
# Module-level constants for this format
# ---------------------------------------------------------------------------

_ENCODING = 'latin-1'
_DELIMITER = ','

# Row indices (0-based) in the correlogram file
_CORR_LABEL_ROW = 0
_CORR_HEADER_ROW = 1
_CORR_DATA_START_ROW = 2

# Column indices in the correlogram data rows
_COL_DELAY_TIME = 0
_COL_G2M1 = 1
# Column 2 (CONTIN fit) is deliberately not assigned -- we never read it.

# Padding sentinel: a row is padding if AND ONLY IF both conditions are true.
# C(t) alone is not reliable because baseline noise produces small negative
# values in real data.
_PADDING_DELAY_TIME = 0.0
_PADDING_G2M1 = -1.0

# Row indices in the count rate history file
_TRACE_LABEL_ROW = 0
_TRACE_HEADER_ROW = 1
_TRACE_DATA_START_ROW = 2

# Conversion factors applied by these parsers before storing
_MICROSECONDS_TO_SECONDS = 1.0e-6   # delay times: µs -> s
_KCPS_TO_CPS = 1000.0               # count rates: kcps -> cps


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_elapsed_seconds(timestamp_str: str) -> float:
    """Convert a Brookhaven HH:MM:SS.sssssss timestamp to elapsed seconds.

    The timestamp represents time elapsed since the start of the run, not
    an absolute wall-clock time. The first sample is typically around 1 s,
    not 00:00:00.

    Parameters
    ----------
    timestamp_str : str
        Timestamp in the format 'HH:MM:SS.sssssss' as written by
        Brookhaven Particle Explorer.

    Returns
    -------
    float
        Elapsed time in seconds.

    Raises
    ------
    ValueError
        If the string does not match the expected format.
    """
    try:
        parts = timestamp_str.strip().split(':')
        if len(parts) != 3:
            raise ValueError("Expected HH:MM:SS.sss format.")
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600.0 + minutes * 60.0 + seconds
    except (ValueError, IndexError) as exc:
        raise ValueError(
            f"Cannot parse timestamp {timestamp_str!r} as HH:MM:SS.sss: {exc}"
        ) from exc


def _read_lines(file_path: str, encoding: str) -> List[str]:
    """Read all lines from a file, stripping line endings.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ParseError
        If the file cannot be decoded with the specified encoding.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path!r}")
    try:
        with open(file_path, encoding=encoding, newline='') as fh:
            return [line.rstrip('\r\n') for line in fh]
    except (UnicodeDecodeError, OSError) as exc:
        raise ParseError(
            f"Could not read {file_path!r} with encoding {encoding!r}: {exc}"
        ) from exc


def _extract_label(raw_line: str) -> str:
    """Extract the sample label from the first row of a Brookhaven CSV.

    Brookhaven appends one or more trailing comma fields to the label row
    (e.g. 'Sample Name,,'). Strip those out and return the label text only.

    Returns an empty string if the line is blank or only commas.
    """
    # Split on comma, take the first non-empty segment
    parts = [p.strip() for p in raw_line.split(_DELIMITER)]
    non_empty = [p for p in parts if p]
    return non_empty[0] if non_empty else ''


# ---------------------------------------------------------------------------
# BrookhavenDLSParser
# ---------------------------------------------------------------------------

class BrookhavenDLSParser(BaseDLSParser):
    """Parser for Brookhaven Particle Explorer correlation function .CSV files.

    Reads one correlogram file and returns a one-element list containing a
    DLSFilePreview. The preview has the correlogram data and sample label
    populated; all physical parameters (angle, wavelength, n, T, viscosity,
    concentration, polymer name) are left as None for the user to supply at
    the confirmation step.

    Usage
    -----
    ::

        parser = BrookhavenDLSParser()
        previews = parser.parse('path/to/file.csv')
        preview = previews[0]

        # At the confirmation step, fill in physical parameters:
        preview.polymer_name = 'PVP'
        preview.solvent_name = 'water'
        preview.concentration_g_per_mL = parser.convert_concentration(0.5, 'mg/mL')
        preview.temperature_K = parser.convert_temperature(25.0, 'C')
        preview.angle_deg = 90.0
        preview.wavelength_nm = 532.0
        preview.solvent_refractive_index = 1.33
        preview.viscosity_Pa_s = parser.convert_viscosity(0.891, 'mPa.s')

        measurement = preview.build()
    """

    def parse(self, file_path: str) -> List[DLSFilePreview]:
        """Parse a Brookhaven DLS correlation function .CSV file.

        Parameters
        ----------
        file_path : str
            Path to the .CSV file. The file must be a Brookhaven Particle
            Explorer DLS export with the standard two-row header structure.

        Returns
        -------
        List[DLSFilePreview]
            A one-element list. The preview contains:
              - delay_times_s  : converted from µs to seconds
              - correlogram    : g²(τ)-1 values (column 2 only;
                                 the CONTIN fit column is ignored)
              - sample_label   : extracted from row 1
              - source_file    : the file path
              - instrument_name: 'Brookhaven Particle Explorer'
            All physical parameters are None (user must supply them).

        Raises
        ------
        FileNotFoundError
            If file_path does not exist.
        ParseError
            If the file format is not recognized, or fewer than one
            valid data row is found after stripping padding.
        """
        lines = _read_lines(file_path, _ENCODING)

        if len(lines) < _CORR_DATA_START_ROW + 1:
            raise ParseError(
                f"{file_path!r} has only {len(lines)} lines; "
                f"expected at least {_CORR_DATA_START_ROW + 1} "
                f"(label row + header row + at least one data row)."
            )

        # --- strict format sniff: the header row (row 2) must be the Brookhaven
        #     correlation header 't(µs),C(t),Fitted (CONTIN)'. Without it this is
        #     not a Brookhaven DLS file -- reject so the loader can try another
        #     parser (otherwise a plain two-column table would be silently and
        #     wrongly accepted, with its first two rows eaten as label/header).
        if 'c(t)' not in lines[_CORR_DATA_START_ROW - 1].lower():
            raise ParseError(
                f"{file_path!r} is not a Brookhaven DLS correlogram: its second "
                f"row is not the expected 't(...),C(t),...' column header.")

        # --- row 1: sample label ---
        sample_label = _extract_label(lines[_CORR_LABEL_ROW])

        # --- rows 3+: data ---
        delay_times_us: List[float] = []
        g2m1_values: List[float] = []
        n_padding = 0
        n_bad = 0

        for _row_idx, line in enumerate(lines[_CORR_DATA_START_ROW:], start=_CORR_DATA_START_ROW + 1):
            line = line.strip()
            if not line:
                continue   # skip blank lines

            parts = line.split(_DELIMITER)
            if len(parts) < 2:
                n_bad += 1
                continue   # skip malformed rows silently; count them

            try:
                t = float(parts[_COL_DELAY_TIME].strip())
                ct = float(parts[_COL_G2M1].strip())
            except ValueError:
                n_bad += 1
                continue   # non-numeric row (e.g. a stray text line)

            # Padding sentinel: BOTH t == 0 and C(t) == -1 must be true.
            # C(t) alone is unreliable because baseline noise produces small
            # negative values in real data near the correlogram plateau end.
            if t == _PADDING_DELAY_TIME and ct == _PADDING_G2M1:
                n_padding += 1
                continue

            delay_times_us.append(t)
            g2m1_values.append(ct)

        if len(delay_times_us) == 0:
            raise ParseError(
                f"No valid data rows found in {file_path!r} after stripping "
                f"padding rows (found {n_padding} padding rows, "
                f"{n_bad} malformed rows)."
            )

        # Convert delay times from µs to seconds (canonical internal unit)
        delay_times_s = np.array(delay_times_us, dtype=float) * _MICROSECONDS_TO_SECONDS
        correlogram = np.array(g2m1_values, dtype=float)

        # Build the preview. Physical parameters are left as None.
        preview = DLSFilePreview(
            source_file=os.path.abspath(file_path),
            instrument_name='Brookhaven Particle Explorer',
            delay_times_s=delay_times_s,
            correlogram=correlogram,
            sample_label=sample_label if sample_label else None,
        )

        return [preview]


# ---------------------------------------------------------------------------
# BrookhavenTraceParser
# ---------------------------------------------------------------------------

class BrookhavenTraceParser(BaseTraceParser):
    """Parser for Brookhaven Particle Explorer count rate history .CSV files.

    Reads one count rate file and returns a one-element list containing a
    TraceFilePreview. The preview has times and count rates populated; the
    sample label is extracted from the file.

    Count rates are converted from kcps (the file's native unit) to cps
    (the canonical internal unit) by multiplying by 1000.

    Timestamps are parsed from HH:MM:SS.sssssss format to elapsed seconds.
    The first timestamp is when the first sample completed (typically ~1 s),
    not zero; this is preserved as-is because it reflects the true sampling
    schedule.

    Pairing with a correlogram
    --------------------------
    This parser makes no attempt to automatically match the trace to a
    DLSMeasurement. The sample labels in the two files may differ slightly
    (a known Brookhaven quirk -- spacing differences have been observed).
    Pairing must be done explicitly by the user at the confirmation step,
    by setting preview.measurement_id to the ID of the associated measurement.

    Usage
    -----
    ::

        parser = BrookhavenTraceParser()
        previews = parser.parse('path/to/count_rate.csv')
        trace = previews[0].build()
    """

    def parse(self, file_path: str) -> List[TraceFilePreview]:
        """Parse a Brookhaven count rate history .CSV file.

        Parameters
        ----------
        file_path : str
            Path to the count rate history .CSV file.

        Returns
        -------
        List[TraceFilePreview]
            A one-element list. The preview contains:
              - times_s         : elapsed seconds from start of run
              - count_rates_cps : converted from kcps to cps (× 1000)
              - sample_label    : extracted from row 1
              - source_file     : the file path
              - measurement_id  : None (set by user at confirmation)

        Raises
        ------
        FileNotFoundError
            If file_path does not exist.
        ParseError
            If the file format is not recognized, or no valid data rows
            are found.
        """
        # Count rate files are plain ASCII; no encoding surprises expected,
        # but we use latin-1 defensively in case a label contains a non-ASCII
        # character on some systems.
        lines = _read_lines(file_path, _ENCODING)

        if len(lines) < _TRACE_DATA_START_ROW + 1:
            raise ParseError(
                f"{file_path!r} has only {len(lines)} lines; "
                f"expected at least {_TRACE_DATA_START_ROW + 1}."
            )

        # --- row 1: sample label ---
        sample_label = _extract_label(lines[_TRACE_LABEL_ROW])

        # --- rows 3+: data ---
        times_s: List[float] = []
        count_rates_cps: List[float] = []
        n_bad = 0

        for _row_idx, line in enumerate(lines[_TRACE_DATA_START_ROW:], start=_TRACE_DATA_START_ROW + 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split(_DELIMITER)
            if len(parts) < 2:
                n_bad += 1
                continue

            try:
                t_s = _parse_elapsed_seconds(parts[0])
                rate_kcps = float(parts[1].strip())
            except ValueError:
                n_bad += 1
                continue

            times_s.append(t_s)
            count_rates_cps.append(rate_kcps * _KCPS_TO_CPS)   # kcps -> cps

        if len(times_s) == 0:
            raise ParseError(
                f"No valid data rows found in {file_path!r} "
                f"({n_bad} malformed rows encountered)."
            )

        preview = TraceFilePreview(
            source_file=os.path.abspath(file_path),
            instrument_name='Brookhaven Particle Explorer',
            times_s=np.array(times_s, dtype=float),
            count_rates_cps=np.array(count_rates_cps, dtype=float),
            sample_label=sample_label if sample_label else None,
        )

        return [preview]
