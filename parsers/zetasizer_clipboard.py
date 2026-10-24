"""
parsers/zetasizer_clipboard.py
==============================

Parser for Malvern Zetasizer **clipboard** correlation-function data -- the
tab-separated text produced by copying the correlation data out of the Zetasizer
software. One file = one **or more** records, each record = one DLS measurement.

File format (confirmed from real exports)
-----------------------------------------
Tab-separated. Column 0 is the shared lag-time axis; every column after it is one
record's correlation coefficient (a trailing empty column from the final tab is
ignored). A single-record copy is just the one-column case:

    Row 1   : 'X Lag Time' <tab> 'Record M: <label>' <tab> 'Record M+1: <label>' ...
    Rows 2+ : lag time      <tab> coefficient          <tab> coefficient          ...

- **Lag time** is in microseconds; converted to seconds (x 1e-6) on load. This is
  a format fact, not a physical assumption.
- **Each record column is g2(tau) - 1** (the Zetasizer "correlation coefficient":
  its intercept is the coherence factor beta and it decays to 0). It is stored
  as-is; no g1 / Siegert conversion is applied. (g1 would start at 1.0; this starts
  at beta ~ 0.3-0.9.)
- Long-lag channels reported as exactly 0 are kept as baseline points. Trim them
  with the analysis delay-time (tau) window if they are unwanted; the cumulant
  amplitude cutoff already ignores them.

Instrument-agnostic, like every parser here: NO physical parameters are read from
the file. Angle, wavelength, refractive index, temperature, viscosity,
concentration, polymer and solvent are all supplied by the user at the
confirmation step. The Zetasizer clipboard format carries none of them.

A second Zetasizer format -- the software's structured 'export' -- formats data
differently (comma-separated, one measurement per row, and it *does* carry sample
parameters); it is handled by `parsers/zetasizer_export.py`.

Change history
--------------
2026-06-16  Initial implementation. (zetasizer_clipboard.py v1)
2026-06-21  Multi-record support: one preview per record column, so a copy of many
            records in one file loads as many measurements. (v2)
"""

from __future__ import annotations

import os
import re
import warnings
from typing import List

import numpy as np

from core.notes import Note, render
from parsers.base_parser import (
    MAX_BAD_ROW_FRACTION,
    BaseDLSParser,
    DLSFilePreview,
    MalformedDataError,
    ParseError,
    convert_delay_times,
    read_text_lines,
)


# ---------------------------------------------------------------------------
# Module-level constants for this format
# ---------------------------------------------------------------------------

# Try UTF-8 (with optional BOM) first; fall back to latin-1, which decodes any
# byte stream, so a stray non-UTF-8 character never crashes the read.
_ENCODINGS = ('utf-8-sig', 'latin-1')
_DELIMITER = '\t'

_HEADER_ROW = 0          # 'X Lag Time' / 'Record N: ...'
_DATA_START_ROW = 1

_COL_LAG_TIME = 0
_FIRST_RECORD_COL = 1    # every column from here on is one record's correlogram

# Header column-1 text used to recognize the format (case-insensitive). The full
# phrase 'lag time' (not a bare 'lag' substring) so an unrelated tab file that
# merely contains the letters "lag" somewhere in its first cell isn't mistaken for
# Zetasizer clipboard data and read as wholesale NaN. Real clipboard copies write
# 'X Lag Time'.
_LAG_HEADER_TOKEN = 'lag time'
# Reject the file if more than this fraction of its data rows have a non-numeric
# lag cell — the generic parsers' threshold, so a mostly-non-numeric file that
# slips past the header sniff fails loudly instead of loading as all-NaN.
_MAX_BAD_ROW_FRACTION = MAX_BAD_ROW_FRACTION
# 'Record N: <label>' -> capture <label>.
_RECORD_LABEL_RE = re.compile(r'^\s*Record\s+\d+\s*:\s*(.*)$')


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_lines(file_path: str) -> List[str]:
    """Read all lines, trying each supported encoding in turn.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ParseError
        If the file cannot be read at all.
    """
    return read_text_lines(file_path, _ENCODINGS)


def _extract_label(raw_cell: str) -> str:
    """Pull the sample label from one header cell ('Record N: <label>').

    Returns the text after 'Record N:' if that pattern is present, otherwise the
    whole cell (stripped).
    """
    raw = raw_cell.strip()
    match = _RECORD_LABEL_RE.match(raw)
    return (match.group(1).strip() if match else raw)


# ---------------------------------------------------------------------------
# ZetasizerClipboardParser
# ---------------------------------------------------------------------------

class ZetasizerClipboardParser(BaseDLSParser):
    """Parser for Malvern Zetasizer clipboard correlation-function text.

    Reads every record column in the file and returns one DLSFilePreview per
    record, each with its correlogram (g2-1) and sample label. A single-record
    copy yields a one-element list. All physical parameters are None -- the user
    supplies them at the confirmation step.

    Example
    -------
        parser = ZetasizerClipboardParser()
        previews = parser.parse('Correlation Function - PEG (8k) in Water.txt')
        preview = previews[0]
        preview.polymer_name = 'PEG (8k)'
        preview.solvent_name = 'water'
        preview.concentration_g_per_mL = 0.001
        preview.temperature_K = 298.15
        preview.angle_deg = 173.0          # Zetasizer Nano backscatter (user-set)
        preview.wavelength_nm = 633.0
        preview.solvent_refractive_index = 1.330
        preview.viscosity_Pa_s = 0.00089
        measurement = preview.build()
    """

    def parse(self, file_path: str) -> List[DLSFilePreview]:
        """Parse a Zetasizer clipboard correlation-function file.

        Returns
        -------
        List[DLSFilePreview]
            One preview per record column. Each preview contains:
              - delay_times_s  : lag times converted µs -> seconds (shared axis)
              - correlogram    : that record's g2(tau)-1 (stored as-is)
              - sample_label   : from the record's 'Record N: ...' header cell
              - source_file    : the file path
              - instrument_name: 'Malvern Zetasizer'
            All physical parameters are None (user must supply them).

        Raises
        ------
        FileNotFoundError
            If file_path does not exist.
        ParseError
            If the file does not look like a Zetasizer clipboard export, or no
            valid data rows / record columns are found.
        """
        lines = _read_lines(file_path)

        if len(lines) < _DATA_START_ROW + 1:
            raise ParseError(
                f"{file_path!r} has only {len(lines)} line(s); expected a header "
                f"row plus at least one data row."
            )

        # --- recognize the format from the first column of the header row ---
        header_parts = lines[_HEADER_ROW].split(_DELIMITER)
        header_col0 = header_parts[0].strip().lower() if header_parts else ''
        if _LAG_HEADER_TOKEN not in header_col0:
            raise ParseError(
                f"{file_path!r} does not look like a Zetasizer clipboard export: "
                f"expected a first-column header containing 'Lag Time', got "
                f"{header_parts[0]!r}. (Is this tab-separated clipboard data with "
                f"its header row intact?)"
            )

        # --- format commit point ---------------------------------------------------
        # Above this line a failure means "not my format" and stays a plain ParseError,
        # so autodetect can hand the file to the next parser. Below it the file has
        # proved it IS this format, so a failure is malformed DATA and raises
        # MalformedDataError, whose message is meant for the user. TWO pre-commit checks
        # still live BELOW this line, both deliberate: the bad-lag-row fraction and the
        # foreign-file record sniff. Both say "this is not Zetasizer clipboard data" about a
        # file that happens to have a Lag Time header, so both must read as a format verdict.

        # --- record columns: every non-empty header cell after the lag column ---
        # (the trailing empty cell from the final tab is skipped.)
        record_cols = [
            i for i in range(_FIRST_RECORD_COL, len(header_parts))
            if header_parts[i].strip()
        ]
        if not record_cols:
            raise ParseError(
                f"{file_path!r} has a lag-time header but no record columns. "
                f"(Expected at least one 'Record N: ...' column after the lag "
                f"column.)"
            )
        labels = {i: _extract_label(header_parts[i]) for i in record_cols}

        # --- data rows: shared lag axis + one value list per record column ---
        data_lines = [ln for ln in lines[_DATA_START_ROW:] if ln.strip()]
        lag_us: List[float] = []
        values: dict = {i: [] for i in record_cols}
        n_bad = 0
        for line in data_lines:
            parts = line.split(_DELIMITER)
            try:
                t = float(parts[_COL_LAG_TIME].strip())
            except (ValueError, IndexError):
                n_bad += 1
                continue
            lag_us.append(t)
            # A missing/blank/non-numeric cell is recorded as None for that record only,
            # keeping every record on the shared lag axis (ragged rows tolerated). Which
            # of those Nones is padding and which is malformed cannot be decided here --
            # it depends on whether anything follows in the same record -- so the
            # decision is deferred to the per-record pass below.
            for i in record_cols:
                cell = parts[i].strip() if i < len(parts) else ''
                try:
                    values[i].append(float(cell))
                except ValueError:
                    values[i].append(None)

        total = len(data_lines)
        if total and n_bad / total > _MAX_BAD_ROW_FRACTION:
            raise ParseError(
                f"{n_bad} of {total} data rows in {file_path!r} have a non-numeric "
                f"lag-time cell ({100 * n_bad / total:.0f}%). This does not look "
                f"like Zetasizer clipboard data (tab-separated, first column the lag "
                f"axis). If it has header/comment lines, remove them before loading."
            )

        if len(lag_us) == 0:
            raise MalformedDataError(
                f"No valid data rows found in {file_path!r} "
                f"({n_bad} malformed rows)."
            )

        # The n_bad guard above only inspects the LAG cell, so a foreign file with a
        # numeric first column but non-numeric record columns would slip through and store
        # a meaningless all-NaN correlogram. Reject the FOREIGN-file case FIRST: a numeric
        # lag axis with NO record column carrying any numeric value is not Zetasizer
        # clipboard data with its record columns intact. Order matters -- run after the
        # per-record strictness below, a foreign file would be reported as a malformed
        # Zetasizer row rather than as the wrong format. It ALSO guarantees the per-record
        # drop below can never empty the file: at least one column is numeric here.
        if not any(any(v is not None for v in values[i]) for i in record_cols):
            raise ParseError(
                f"{file_path!r} has a numeric lag axis but no record column with any "
                f"numeric value (every 'Record N' column is non-numeric). This does not "
                f"look like Zetasizer clipboard data with its record columns intact."
            )

        # Per-record pass. A blank cell is PADDING only where it is part of the trailing
        # run -- a clipboard paste can genuinely hold records of different lengths on one
        # lag axis. Anywhere else it is a hole in the middle of a series, i.e. a malformed
        # file, and it is named HERE rather than deferred to DLSMeasurement's "correlogram
        # must contain only finite numbers" at *Run*, which names neither the file, the
        # record, nor the row.
        corr = {}
        dropped = []
        for i in record_cols:
            vals = values[i]
            n = len(vals)
            while n > 0 and vals[n - 1] is None:
                n -= 1                              # strip the trailing padding run
            holes = [r for r in range(n) if vals[r] is None]
            # A 'Record N:' header with no sample name is named as the file numbers it.
            label = labels[i] or header_parts[i].strip().rstrip(':')
            if n == 0:
                # A header with no numeric value ANYWHERE carries no data, so dropping it
                # loses nothing, where raising would cost the paste's other good records.
                # It is never dropped silently: see the note built below, which reaches the
                # GUI. The asymmetry with `holes` just below is deliberate -- an empty
                # column has nothing to lose, while a HOLED column carries real pasted
                # values, so naming the row and lag beats discarding them.
                dropped.append(label)
                continue
            if holes:
                first = holes[0]
                raise MalformedDataError(
                    f"{file_path!r}: record {label!r} is blank or non-numeric at data "
                    f"row {first + 1} (lag {lag_us[first]!r} us), with {n - first - 1} "
                    f"further value(s) after it. A gap inside a record is a malformed "
                    f"paste; only blanks at the END of a record are padding."
                )
            corr[i] = np.array(vals[:n], dtype=float)

        # ONE note string, both channels (the keep-both convention `generic_sls.py` set):
        # `notes` reaches the GUI as a passive info box, and the UserWarning keeps the
        # scripted/headless caller informed. The GUI half is the load-bearing one --
        # `tests/test_notes_bridge.py` exists to establish that stderr is invisible in the
        # packaged app, which is exactly why a previous stderr-only attempt at this drop
        # was rejected.
        found_notes = []
        # A data row whose lag cell is not a number is skipped for EVERY record (the
        # 5 % ceiling above tolerates a few); that is a lag channel the user never
        # sees, so it is disclosed like the empty-column drop below.
        if n_bad:
            found_notes.append(Note('parser.rows_skipped', ('delay_times_s', 'correlogram'),
                                    {'n_skipped': n_bad, 'n': total,
                                     'n_loaded': len(lag_us)}))
        if dropped:
            # Every dropped column, by name: a note's values hold names and numbers, never
            # display text (ui_style_guide §5, the `values` rule).
            names = tuple(dropped)
            found_notes.append(Note('parser.empty_columns', ('correlogram',),
                                    {'n_empty': len(dropped), 'n': len(record_cols),
                                     'columns': names}))
        notes = tuple(found_notes)
        for note in notes:
            warnings.warn(f"{file_path!r}: {render(note, surface='export')}", UserWarning,
                          stacklevel=2)

        # µs -> s through the shared table: the lag axis feeds Gamma, D and Rh, so
        # its factor lives in one place rather than in a copy per format.
        delay_times_s = convert_delay_times(np.array(lag_us, dtype=float), 'us')
        abs_path = os.path.abspath(file_path)

        # A record whose trailing padding was stripped is shorter than the shared lag
        # axis, so each preview takes the leading slice matching its own length. For the
        # rectangular case -- what the instrument actually writes -- this is the whole
        # axis, unchanged.
        #
        # Every surviving preview carries the same note: it describes the FILE, and a
        # preview must not disagree with its own file's parse. The load flow de-duplicates
        # (see base_parser.collect_parse_notes), so a 20-record paste still renders one line.
        previews = [
            DLSFilePreview(
                source_file=abs_path,
                instrument_name='Malvern Zetasizer',
                delay_times_s=delay_times_s[:corr[i].size].copy(),
                correlogram=corr[i],
                sample_label=labels[i] if labels[i] else None,
                notes=notes,
            )
            for i in record_cols if i in corr
        ]
        return previews
