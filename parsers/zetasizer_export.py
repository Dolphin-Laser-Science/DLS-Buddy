"""
parsers/zetasizer_export.py
===========================

Parser for Malvern Zetasizer **export** correlation-function data -- the
comma-separated table produced by the Zetasizer software's structured *export*
function (as opposed to the clipboard copy handled by
`parsers/zetasizer_clipboard.py`). One file = one **or more** measurements, one
per data row.

Unlike the clipboard format, the export *does* carry some sample parameters, so
this parser pre-fills them (the user still confirms/edits them at the Data tab).

File format (confirmed from real exports)
-----------------------------------------
Comma-separated, with a header row. One measurement per data row. Columns:

    Sample Name, Material Name, Dispersant Name, Dispersant RI,
    Temperature (°C), Viscosity (cP),
    Correlation Delay Times[1] (µs) ... Correlation Delay Times[K] (µs),
    Correlation Data[1] ... Correlation Data[K]

- Columns are located by **header name** (case-insensitive, ignoring the
  parenthetical unit), NOT by fixed position, because the Zetasizer exporter is
  configurable. The two correlation blocks (Delay Times + Data) are **required**;
  the named parameter columns are each optional (whatever is present is pre-filled).
- **Units are read from the header.** Each located column's ``(unit)`` suffix is
  honoured through the shared converters -- ``Temperature (K)`` is kelvin,
  ``Correlation Delay Times[k] (ms)`` is milliseconds -- and a cell with no suffix
  takes the documented template's unit: lag time in µs (-> seconds), temperature
  in °C (-> kelvin), viscosity in cP (-> Pa·s). A lag unit the converter does not
  know is malformed data (the lag axis cannot be loaded on a guess); an unknown
  temperature or viscosity unit leaves that field blank for the user to enter,
  exactly like an unreadable cell.
- **Correlation Data is g2(tau) - 1** (the Zetasizer "correlation coefficient":
  intercept ~ beta, decaying to 0), stored as-is -- no g1 / Siegert conversion.
  Trailing zero channels are kept as baseline points (trim with the analysis tau
  window if unwanted), matching the clipboard parser.

Column -> field mapping
-----------------------
    Sample Name       -> sample_label
    Material Name     -> polymer_name
    Dispersant Name   -> solvent_name        (normalized against the vocabulary)
    Dispersant RI     -> solvent_refractive_index
    Temperature (°C)  -> temperature_K        (by the declared unit; °C -> +273.15)
    Viscosity (cP)    -> viscosity_Pa_s       (by the declared unit; cP -> x 1e-3)

Angle, wavelength and concentration are NOT in the export and are left for the
user to supply at the confirmation step (the Zetasizer file carries none of them).

Zetasizer export template (document for the user)
-------------------------------------------------
The export is configurable; this parser targets a common export template:

    Settings : include header row; use commas as separators.
    Parameters (in order): Sample Name, Material Name, Dispersant Name,
        Dispersant RI, Temperature, Viscosity, Correlation Delay Times,
        Correlation Data.

Change history
--------------
2026-06-21  Initial implementation. (zetasizer_export.py v1)
"""

from __future__ import annotations

import csv
import os
import re
import warnings
from typing import List, Optional

import numpy as np

from parsers.base_parser import (
    convert_delay_times,
    BaseDLSParser,
    DLSFilePreview,
    MalformedDataError,
    ParseError,
    convert_temperature,
    convert_viscosity,
)


# ---------------------------------------------------------------------------
# Module-level constants for this format
# ---------------------------------------------------------------------------

# Try UTF-8 (with optional BOM) first; fall back to latin-1, which decodes any
# byte stream -- the Zetasizer writes ° and µ as cp1252/latin-1 bytes.
_ENCODINGS = ('utf-8-sig', 'latin-1')
_DELIMITER = ','

# The template's units, applied to a header cell that declares none.
_TEMPLATE_LAG_UNIT = 'µs'
_TEMPLATE_TEMPERATURE_UNIT = '°C'
_TEMPLATE_VISCOSITY_UNIT = 'cP'

# Header prefixes that identify each column. Matched case-insensitively against
# the header cell with its parenthetical unit stripped (e.g. 'Temperature (°C)').
_PREFIX_SAMPLE = 'sample name'
_PREFIX_MATERIAL = 'material name'
_PREFIX_DISPERSANT = 'dispersant name'
_PREFIX_RI = 'dispersant ri'
_PREFIX_TEMPERATURE = 'temperature'
_PREFIX_VISCOSITY = 'viscosity'
_PREFIX_DELAY = 'correlation delay times'
_PREFIX_DATA = 'correlation data'

# Strip a trailing '(unit)' and any '[k]' index from a header cell for matching.
_PAREN_RE = re.compile(r'\([^)]*\)')
_INDEX_RE = re.compile(r'\[\s*\d+\s*\]')


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_lines(file_path: str) -> List[str]:
    """Read all lines, trying each supported encoding in turn."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path!r}")
    last_exc = None
    for encoding in _ENCODINGS:
        try:
            with open(file_path, encoding=encoding, newline='') as fh:
                return [line.rstrip('\r\n') for line in fh]
        except UnicodeDecodeError as exc:
            last_exc = exc
            continue
        except OSError as exc:
            raise ParseError(f"Could not read {file_path!r}: {exc}") from exc
    raise ParseError(
        f"Could not decode {file_path!r} with any of {_ENCODINGS}: {last_exc}"
    ) from last_exc


def _norm_header(cell: str) -> str:
    """Lowercase a header cell with its '(unit)' and '[index]' stripped."""
    text = _PAREN_RE.sub('', cell)
    text = _INDEX_RE.sub('', text)
    return text.strip().lower()


def _parse_float_or_none(cell: str) -> Optional[float]:
    """float(cell), or None if blank/non-numeric (never raises)."""
    try:
        return float(cell.strip())
    except (ValueError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# ZetasizerExportParser
# ---------------------------------------------------------------------------

class ZetasizerExportParser(BaseDLSParser):
    """Parser for Malvern Zetasizer structured-export correlation data.

    Returns one DLSFilePreview per data row. Each preview carries the correlogram
    (g2-1) plus the parameters present in the file (sample label, polymer/material,
    solvent/dispersant, refractive index, temperature, viscosity). Angle,
    wavelength and concentration are None -- the user supplies them.
    """

    def parse(self, file_path: str) -> List[DLSFilePreview]:
        """Parse a Zetasizer export correlation file.

        Returns
        -------
        List[DLSFilePreview]
            One preview per data row.

        Raises
        ------
        FileNotFoundError
            If file_path does not exist.
        ParseError
            If the file does not look like a Zetasizer export (no correlation
            delay-time / data columns), the two blocks disagree in length, or no
            valid data rows are found.
        """
        lines = _read_lines(file_path)
        if len(lines) < 2:
            raise ParseError(
                f"{file_path!r} has only {len(lines)} line(s); expected a header "
                f"row plus at least one data row."
            )

        # Parse with csv.reader (not str.split) so a comma inside a quoted text
        # field — a Sample/Material Name like "PS, 900k" — stays one cell instead of
        # shifting every downstream column for that row (silent misalignment →
        # wrong correlogram/RI/temperature). Rows are then width-checked below.
        table = list(csv.reader(lines, delimiter=_DELIMITER))
        header = table[0]
        norm = [_norm_header(cell) for cell in header]
        n_cols = len(header)

        # --- locate the two required correlation blocks (in header order) ---
        delay_cols = [i for i, h in enumerate(norm) if h.startswith(_PREFIX_DELAY)]
        data_cols = [i for i, h in enumerate(norm) if h.startswith(_PREFIX_DATA)]
        if not delay_cols or not data_cols:
            raise ParseError(
                f"{file_path!r} does not look like a Zetasizer export: expected "
                f"'Correlation Delay Times[...]' and 'Correlation Data[...]' "
                f"columns in the header row. (Is this the comma-separated export "
                f"with its header row included?)"
            )
        if len(delay_cols) != len(data_cols):
            raise ParseError(
                f"{file_path!r}: {len(delay_cols)} delay-time column(s) but "
                f"{len(data_cols)} correlation-data column(s); they must match."
            )

        # --- format commit point ---------------------------------------------------
        # Above this line a failure means "not my format" and stays a plain ParseError,
        # so autodetect can hand the file to the next parser. Below it the file has
        # proved it IS this format, so a failure is malformed DATA and raises
        # MalformedDataError, whose message is meant for the user.
        # The delay/data COUNT mismatch just above is still pre-commit: an equal number
        # of the two column families is part of what identifies this export.

        # --- locate the optional parameter columns (first match wins) ---
        def _find(prefix: str) -> Optional[int]:
            for i, h in enumerate(norm):
                if h.startswith(prefix):
                    return i
            return None

        # 'dispersant ri' must be checked before 'dispersant name' would shadow it;
        # they have distinct prefixes, so independent lookups are unambiguous.
        col_sample = _find(_PREFIX_SAMPLE)
        col_material = _find(_PREFIX_MATERIAL)
        col_dispersant = _find(_PREFIX_DISPERSANT)
        col_ri = _find(_PREFIX_RI)
        col_temp = _find(_PREFIX_TEMPERATURE)
        col_visc = _find(_PREFIX_VISCOSITY)

        # The unit each located column DECLARES: the LAST '(...)' group of the cell,
        # read back here instead of thrown away. `_norm_header` strips every group for
        # matching, so a cell may carry another parenthetical ahead of its unit; the
        # trailing one is the unit suffix. It used to be discarded and the template's
        # µs / °C / cP applied regardless, so a `(K)` column read 298.15 as 571 K and
        # an `(ms)` lag axis came in 1000x short. A cell with no suffix keeps the
        # template's unit.
        def _declared_unit(col: Optional[int], template: str) -> str:
            groups = _PAREN_RE.findall(header[col]) if col is not None else []
            return (groups[-1][1:-1].strip() or template) if groups else template

        lag_unit = _declared_unit(delay_cols[0], _TEMPLATE_LAG_UNIT)
        temp_unit = _declared_unit(col_temp, _TEMPLATE_TEMPERATURE_UNIT)
        visc_unit = _declared_unit(col_visc, _TEMPLATE_VISCOSITY_UNIT)

        abs_path = os.path.abspath(file_path)
        previews: List[DLSFilePreview] = []
        n_bad = 0

        for row_no, row in enumerate(table[1:], start=2):
            if not any(cell.strip() for cell in row):
                continue
            # A row whose field count differs from the header is misaligned (an
            # unquoted delimiter in a text field, or a truncated/corrupt row).
            # Reject it loudly rather than silently shift its columns.
            if len(row) != n_cols:
                raise MalformedDataError(
                    f"{file_path!r}: data row {row_no} has {len(row)} field(s) but "
                    f"the header has {n_cols}; the row is misaligned (an unquoted "
                    f"comma in a text field, or a corrupt row). Re-export with "
                    f"quoted text fields."
                )

            preview = self._row_to_preview(
                row, abs_path, row_no=row_no,
                delay_cols=delay_cols, data_cols=data_cols,
                col_sample=col_sample, col_material=col_material,
                col_dispersant=col_dispersant, col_ri=col_ri,
                col_temp=col_temp, col_visc=col_visc,
                lag_unit=lag_unit, temp_unit=temp_unit, visc_unit=visc_unit,
            )
            if preview is None:
                n_bad += 1
                continue
            previews.append(preview)

        if not previews:
            raise MalformedDataError(
                f"No valid data rows found in {file_path!r} "
                f"({n_bad} unreadable row(s))."
            )
        # A row with no readable channel is a measurement the user expected to see and
        # will not; every surviving preview carries the file-level note (de-duplicated
        # by the load flow), and a headless caller hears it as a UserWarning.
        if n_bad:
            note = (
                f"{n_bad} data row(s) carried no readable correlation channel and were "
                f"not loaded; {len(previews)} measurement(s) loaded. An empty row usually "
                f"means the export was cut short -- re-export if those rows were meant "
                f"to be included."
            )
            notes = (note,)
            for preview in previews:
                preview.notes = notes
            warnings.warn(f"{file_path!r}: {note}", UserWarning, stacklevel=2)
        return previews

    @staticmethod
    def _row_to_preview(
        row: List[str],
        abs_path: str,
        *,
        row_no: int,
        delay_cols: List[int],
        data_cols: List[int],
        col_sample: Optional[int],
        col_material: Optional[int],
        col_dispersant: Optional[int],
        col_ri: Optional[int],
        col_temp: Optional[int],
        col_visc: Optional[int],
        lag_unit: str = _TEMPLATE_LAG_UNIT,
        temp_unit: str = _TEMPLATE_TEMPERATURE_UNIT,
        visc_unit: str = _TEMPLATE_VISCOSITY_UNIT,
    ) -> Optional[DLSFilePreview]:
        """Build one preview from a data row, or None if it has no usable data."""

        def cell(idx: Optional[int]) -> str:
            if idx is None or idx >= len(row):
                return ''
            return row[idx].strip()

        # --- correlogram: pair each (delay, data) channel; keep finite-lag ones ---
        lag_us: List[float] = []
        corr: List[float] = []
        for chan, (di, ci) in enumerate(zip(delay_cols, data_cols, strict=True), start=1):
            t = _parse_float_or_none(row[di]) if di < len(row) else None
            if t is None:
                continue   # padding / missing channel -> drop it
            raw_c = row[ci] if ci < len(row) else ''
            c = _parse_float_or_none(raw_c)
            if c is None:
                # A delay with no paired value is a MALFORMED row, not padding: the
                # channel exists but has no reading. This used to append np.nan and
                # report the row as a valid preview, so the user first heard about it at
                # *Run*, as "correlogram must contain only finite numbers" -- naming
                # neither this file nor this row. Padding is a blank DELAY cell, handled by
                # the `t is None` branch above and deliberately left alone.
                raise MalformedDataError(
                    f"{abs_path!r}: data row {row_no}, Correlation Data[{chan}] is "
                    f"blank or non-numeric ({raw_c.strip()!r}) while its paired "
                    f"Correlation Delay Times[{chan}] is {t!r}. A channel with a delay "
                    f"but no value is a malformed row; padding would leave the DELAY "
                    f"cell blank instead."
                )
            lag_us.append(t)
            corr.append(c)
        if not lag_us:
            return None

        # Through the shared converter, by the unit the header declared. An unknown
        # lag unit is malformed data: the axis cannot be loaded on a guess, and the
        # user is never asked for it (unlike temperature and viscosity below).
        try:
            delay_times_s = convert_delay_times(np.array(lag_us, dtype=float), lag_unit)
        except MalformedDataError:
            raise                        # a non-finite lag cell; the loader's report
            #                              names the file (autodetect's Detection.reason)
        except ValueError as exc:
            raise MalformedDataError(
                f"{abs_path!r}: the 'Correlation Delay Times' header declares the unit "
                f"({lag_unit}), which this program cannot convert to seconds ({exc}). "
                f"Re-export with the delay times in µs, ms or s."
            ) from exc
        correlogram = np.array(corr, dtype=float)

        # --- parameters (each optional; a bad value degrades to None, no crash) ---
        sample_label = cell(col_sample) or None
        polymer_name = cell(col_material) or None

        # Stored VERBATIM, exactly like polymer_name above. The dispersant cell is
        # free text typed by the operator on the instrument ("My Phosphate Buffer" is
        # an ordinary value for it), so mapping it onto our own solvent names would be
        # guessing at data we have no claim on. cell() has already stripped it.
        solvent_name = cell(col_dispersant) or None

        solvent_ri = _parse_float_or_none(cell(col_ri)) if col_ri is not None else None

        temperature_K: Optional[float] = None
        temp_raw = _parse_float_or_none(cell(col_temp)) if col_temp is not None else None
        if temp_raw is not None:
            try:
                temperature_K = convert_temperature(temp_raw, temp_unit)
            except ValueError:
                temperature_K = None     # unusable value OR unknown unit: the user enters it

        viscosity_Pa_s: Optional[float] = None
        visc_raw = _parse_float_or_none(cell(col_visc)) if col_visc is not None else None
        if visc_raw is not None:
            try:
                viscosity_Pa_s = convert_viscosity(visc_raw, visc_unit)
            except ValueError:
                viscosity_Pa_s = None    # unusable value OR unknown unit: the user enters it

        return DLSFilePreview(
            source_file=abs_path,
            instrument_name='Malvern Zetasizer',
            delay_times_s=delay_times_s,
            correlogram=correlogram,
            sample_label=sample_label,
            polymer_name=polymer_name,
            solvent_name=solvent_name,
            solvent_refractive_index=solvent_ri,
            temperature_K=temperature_K,
            viscosity_Pa_s=viscosity_Pa_s,
        )
