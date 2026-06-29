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
- **Units are format facts** of the documented export template: lag time in µs
  (-> seconds, x 1e-6), temperature in °C (-> kelvin), viscosity in cP (-> Pa·s).
- **Correlation Data is g2(tau) - 1** (the Zetasizer "correlation coefficient":
  intercept ~ beta, decaying to 0), stored as-is -- no g1 / Siegert conversion.
  Trailing zero channels are kept as baseline points (trim with the analysis tau
  window if unwanted), matching the clipboard parser.

Column -> field mapping
-----------------------
    Sample Name       -> sample_label
    Material Name     -> polymer_name
    Dispersant Name   -> solvent_name        (normalised against the vocabulary)
    Dispersant RI     -> solvent_refractive_index
    Temperature (°C)  -> temperature_K        (+273.15)
    Viscosity (cP)    -> viscosity_Pa_s       (x 1e-3)

Angle, wavelength and concentration are NOT in the export and are left for the
user to supply at the confirmation step (the Zetasizer file carries none of them).

Zetasizer export template (document for the user)
-------------------------------------------------
The export is configurable; this parser targets the template the owner uses:

    Settings : include header row; use commas as separators.
    Parameters (in order): Sample Name, Material Name, Dispersant Name,
        Dispersant RI, Temperature, Viscosity, Correlation Delay Times,
        Correlation Data.

Change history
--------------
2026-06-21  Initial implementation. (zetasizer_export.py v1)
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

import numpy as np

from core.data_models import normalize_solvent_name
from parsers.base_parser import (
    BaseDLSParser,
    DLSFilePreview,
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

_MICROSECONDS_TO_SECONDS = 1.0e-6   # lag times: µs -> s

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

        header = lines[0].split(_DELIMITER)
        norm = [_norm_header(cell) for cell in header]

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

        abs_path = os.path.abspath(file_path)
        previews: List[DLSFilePreview] = []
        n_bad = 0

        for line in lines[1:]:
            if not line.strip():
                continue
            row = line.split(_DELIMITER)

            preview = self._row_to_preview(
                row, abs_path,
                delay_cols=delay_cols, data_cols=data_cols,
                col_sample=col_sample, col_material=col_material,
                col_dispersant=col_dispersant, col_ri=col_ri,
                col_temp=col_temp, col_visc=col_visc,
            )
            if preview is None:
                n_bad += 1
                continue
            previews.append(preview)

        if not previews:
            raise ParseError(
                f"No valid data rows found in {file_path!r} "
                f"({n_bad} unreadable row(s))."
            )
        return previews

    @staticmethod
    def _row_to_preview(
        row: List[str],
        abs_path: str,
        *,
        delay_cols: List[int],
        data_cols: List[int],
        col_sample: Optional[int],
        col_material: Optional[int],
        col_dispersant: Optional[int],
        col_ri: Optional[int],
        col_temp: Optional[int],
        col_visc: Optional[int],
    ) -> Optional[DLSFilePreview]:
        """Build one preview from a data row, or None if it has no usable data."""

        def cell(idx: Optional[int]) -> str:
            if idx is None or idx >= len(row):
                return ''
            return row[idx].strip()

        # --- correlogram: pair each (delay, data) channel; keep finite-lag ones ---
        lag_us: List[float] = []
        corr: List[float] = []
        for di, ci in zip(delay_cols, data_cols):
            t = _parse_float_or_none(row[di]) if di < len(row) else None
            if t is None:
                continue   # padding / missing channel -> drop it
            c = _parse_float_or_none(row[ci]) if ci < len(row) else None
            lag_us.append(t)
            corr.append(c if c is not None else np.nan)
        if not lag_us:
            return None

        delay_times_s = np.array(lag_us, dtype=float) * _MICROSECONDS_TO_SECONDS
        correlogram = np.array(corr, dtype=float)

        # --- parameters (each optional; a bad value degrades to None, no crash) ---
        sample_label = cell(col_sample) or None
        polymer_name = cell(col_material) or None

        dispersant = cell(col_dispersant)
        solvent_name = normalize_solvent_name(dispersant) if dispersant else None

        solvent_ri = _parse_float_or_none(cell(col_ri)) if col_ri is not None else None

        temperature_K: Optional[float] = None
        temp_raw = _parse_float_or_none(cell(col_temp)) if col_temp is not None else None
        if temp_raw is not None:
            try:
                temperature_K = convert_temperature(temp_raw, 'C')
            except ValueError:
                temperature_K = None

        viscosity_Pa_s: Optional[float] = None
        visc_raw = _parse_float_or_none(cell(col_visc)) if col_visc is not None else None
        if visc_raw is not None:
            try:
                viscosity_Pa_s = convert_viscosity(visc_raw, 'cP')
            except ValueError:
                viscosity_Pa_s = None

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
