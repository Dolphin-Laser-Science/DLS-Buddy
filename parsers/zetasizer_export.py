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
Comma-separated, with a header row. One measurement per data row, and two column
families carrying the correlogram:

    Correlation Delay Times[1] (µs) ... Correlation Delay Times[K] (µs),
    Correlation Data[1] ... Correlation Data[K]

plus whichever named parameter columns the operator put in their template.

- The export is a **vocabulary, not a layout**: every named column is optional,
  their order carries no meaning, the two families may appear in either order
  (both committed templates are Delay-first; data-first is accepted and no
  committed template exercises it), K is whatever the operator acquired, and a
  header this parser does not name is ignored. Columns are
  located by **header name**, matched against `_COLUMNS` -- the single place the
  vocabulary lives.
- Matching is **exact** on the normalised header (lower-cased, with the ``(unit)``
  suffix and ``[k]`` index stripped), never a prefix. The vendor namespace carries
  near-miss labels -- one real template emits both ``Concentration`` and
  ``Concentrations``, and both ``Duration`` and ``Duration Used`` -- and a prefix
  lets the shorter key claim the longer column silently, by header order, with a
  wrong value. Exact matching turns that into a blank field the user fills in.
- **Units are read from the header.** Each located column's ``(unit)`` suffix is
  honoured -- ``Temperature (K)`` is kelvin, ``Correlation Delay Times[k] (ms)`` is
  milliseconds -- and a cell with no suffix takes its `_COLUMNS` template unit. A
  lag unit the converter does not know is malformed data (the lag axis cannot be
  loaded on a guess); for every other column an unknown unit, an unusable value or
  an unreadable cell all leave the field blank for the user to enter.
- **Correlation Data is g2(tau) - 1** (the Zetasizer "correlation coefficient":
  intercept ~ beta, decaying to 0), stored as-is -- no g1 / Siegert conversion.
  Note the software's own reported ``Intercept`` is on the **g1** convention, i.e.
  sqrt(beta), so it is not comparable with this curve's amplitude directly.
  Trailing zero channels are kept as baseline points (trim with the analysis tau
  window if unwanted), matching the clipboard parser.

Column -> field mapping
-----------------------
`_COLUMNS` is the mapping, in code; it is not restated here, so the two cannot
drift. It also names the columns this parser recognises and deliberately does NOT
read, each with its reason.

**Concentration** is not among them: the Zetasizer computes that field rather than
echoing what the operator typed (it varies across repeat runs of one sample), so it
is left for the user to supply at the confirmation step like any other.

Zetasizer export template (document for the user)
-------------------------------------------------
The export is configurable, so this is a suggestion, not a requirement -- the
parser reads whatever subset is present, in any order:

    Settings : include header row; use commas as separators.
    Parameters : Sample Name, Material Name, Dispersant Name, Dispersant RI,
        Temperature, Viscosity, Wavelength, Scattering Angle, Mean Count Rate,
        Correlation Delay Times, Correlation Data.

Change history
--------------
2026-06-21  Initial implementation. (zetasizer_export.py v1)
2026-09-20  Wider column set and exact-match lookup. (zetasizer_export.py v2)
            The six prefix constants and their hand-written lookups became the
            `_COLUMNS` table; Wavelength, Scattering Angle and Mean Count Rate are
            now pre-filled, and Material RI / Material Absorption / Derived Count
            Rate are recognised without being read.
"""

from __future__ import annotations

import csv
import math
import os
import re
import warnings
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from parsers.base_parser import (
    convert_delay_times,
    BaseDLSParser,
    DLSFilePreview,
    MalformedDataError,
    ParseError,
    convert_count_rate,
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

# The template's lag unit, applied to a header cell that declares none.
_TEMPLATE_LAG_UNIT = 'µs'

# The two correlation column families. Their normalised keys, like every key in
# `_COLUMNS` below, are matched EXACTLY -- see `_find`.
_KEY_DELAY = 'correlation delay times'
_KEY_DATA = 'correlation data'

# Strip a trailing '(unit)' and any '[k]' index from a header cell for matching.
_PAREN_RE = re.compile(r'\([^)]*\)')
_INDEX_RE = re.compile(r'\[\s*\d+\s*\]')

# Units this parser accepts for the two quantities `base_parser` has no factor table
# for. It has none because no other format declares a unit for either; this one can,
# so the spelling is checked rather than assumed.
_WAVELENGTH_UNITS = frozenset({'nm'})
_ANGLE_UNITS = frozenset({'°', 'deg', 'degree', 'degrees'})


def _checked(value: float, unit: str, accepted: frozenset, quantity: str) -> float:
    """`value` unchanged, once `unit` is one of `accepted` -- a unit check, not a
    conversion, for a quantity with no factor table.

    Raises ValueError (never MalformedDataError) on an unknown unit or a non-finite
    value, so the caller degrades the field to blank exactly as an unrecognised
    temperature or viscosity unit already does. A record column must never condemn
    the whole file.
    """
    if unit.strip().lower() not in accepted:
        raise ValueError(
            f"unrecognized {quantity} unit {unit!r}. Accepted: {sorted(accepted)}."
        )
    if not math.isfinite(value):
        raise ValueError(f"{quantity} must be finite, got {value!r}.")
    return float(value)


def _as_nm(value: float, unit: str) -> float:
    """Wavelength in nm. Positive, as `DLSMeasurement` requires it."""
    nm = _checked(value, unit, _WAVELENGTH_UNITS, 'wavelength')
    if nm <= 0:
        raise ValueError(f"wavelength must be positive, got {value!r}.")
    return nm


def _as_deg(value: float, unit: str) -> float:
    """Scattering angle in degrees, on `_require_scattering_angle`'s range.

    Bounds-checked here so an out-of-range cell arrives blank for the user to enter,
    rather than reaching `DLSMeasurement` and raising at Update.
    """
    deg = _checked(value, unit, _ANGLE_UNITS, 'scattering angle')
    if not (0 < deg < 180):
        raise ValueError(
            f"scattering angle must be strictly between 0 and 180 degrees, "
            f"got {value!r}."
        )
    return deg


def _as_cps(value: float, unit: str) -> float:
    """Mean count rate in cps, through the shared factor table.

    The guard is on the RESULT, not the input: the conversion is a multiply, so
    saturation is a property of the product and no input is to blame for it (rule 12).
    A cell of 1e306 kcps passes any check on `value` and returns `inf`.

    Wrapped at all because `convert_count_rate` is written for arrays and this is the
    tree's only scalar caller: its own `require_finite_data` raises IndexError, not
    MalformedDataError, on a 0-d array, and that escapes `autodetect` and fails the
    load. The finite-and-positive check here runs first and keeps it unreachable.
    """
    if not math.isfinite(value):
        raise ValueError(f"mean count rate must be finite, got {value!r}.")
    rate = float(convert_count_rate(value, unit))
    if not (math.isfinite(rate) and rate > 0):
        raise ValueError(
            f"mean count rate must convert to a finite positive rate, got {value!r} "
            f"{unit} -> {rate!r} cps."
        )
    return rate


class _Column(NamedTuple):
    """One column the export template may carry.

    `key` is the EXACT normalised header (`_norm_header`) that identifies it. `field`
    is the `DLSFilePreview` attribute it fills, or **None** for a column we recognise
    and deliberately do not read -- those rows are documentary, and they are here so
    the vendor vocabulary lives in one place and the reason sits beside the column it
    governs. `unit` is the template's unit, used when the header cell declares none,
    and `convert` takes `(value, unit)` to canonical units; a numeric column with no
    converter is stored as read. `text` marks a column stored verbatim as a string.
    """

    key: str
    field: Optional[str]
    unit: Optional[str] = None
    convert: Optional[Callable[[float, str], float]] = None
    text: bool = False


# The export template is user-configurable, so this is a vocabulary, not a layout:
# every column is optional, order carries no meaning, and a header the table does not
# name is ignored. Adding a column the vendor starts emitting is one row.
_COLUMNS = (
    _Column('sample name', 'sample_label', text=True),
    _Column('material name', 'polymer_name', text=True),
    # Stored VERBATIM, exactly like polymer_name above. The dispersant cell is free
    # text typed by the operator on the instrument ("My Phosphate Buffer" is an
    # ordinary value for it), so mapping it onto our own solvent names would be
    # guessing at data we have no claim on.
    _Column('dispersant name', 'solvent_name', text=True),
    _Column('dispersant ri', 'solvent_refractive_index'),
    _Column('temperature', 'temperature_K', '°C', convert_temperature),
    _Column('viscosity', 'viscosity_Pa_s', 'cP', convert_viscosity),
    _Column('wavelength', 'wavelength_nm', 'nm', _as_nm),
    _Column('scattering angle', 'angle_deg', '°', _as_deg),
    # A RECORD: user-owned and provenance-tagged, carried no differently from a
    # Brookhaven .dat's. `core.workspace.seed_vendor_records` offers it and the load flow
    # seeds working AND committed at once, as it does for angle and temperature -- what
    # makes that invariant-3-safe is the editable, labelled Data-tab row, not a gesture
    # between load and analysis.
    _Column('mean count rate', 'vendor_mean_count_rate_cps', 'kcps', _as_cps),
    # --- recognised, deliberately not read ---
    # Material RI would fill `solute_refractive_index`, which core.data_models holds in
    # the dn/dc class: never proposed, defaulted or guessed (invariant 3). Owner ruled
    # 2026-09-20 that the parser recognise it and not fill it while that policy is
    # revisited; the roadmap carries the revisit. The sibling half-answer the revisit
    # should weigh: `parsers/brookhaven_dat.py` keeps its file's particle RI as a RECORD
    # (`solute_refractive_index_from_file`), never propagated -- dropping it here and
    # keeping it there both satisfy the ruling, and they differ.
    _Column('material ri', None),
    # The imaginary part of the same index. Malvern needs both only for its volume
    # distribution; nothing here consumes one yet.
    _Column('material absorption', None),
    # The mean count rate rescaled to no attenuation, so NOT the detected photon rate
    # the residual weight's shot-noise terms need. `mean count rate` is that.
    _Column('derived count rate', None),
)


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
    (g2-1) plus whichever parameters `_COLUMNS` maps and the file carries; anything
    absent stays None for the user to supply at the confirmation step. Concentration
    is always among those -- see the module docstring on why the vendor's own
    Concentration column is not read.
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
        delay_cols = [i for i, h in enumerate(norm) if h == _KEY_DELAY]
        data_cols = [i for i, h in enumerate(norm) if h == _KEY_DATA]
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

        # --- locate the optional parameter columns (first exact match wins) ---
        # Exactly, never by prefix. The vendor vocabulary carries near-miss labels --
        # a real template emits 'Concentration' beside 'Concentrations', and 'Duration'
        # beside 'Duration Used' -- and under a prefix the shorter key takes the longer
        # column silently, by header order, with a wrong VALUE. Exact matching makes
        # that a blank field the user fills instead, and keeps a
        # 'Correlation Data Quality[k]' family out of `data_cols`, where it would
        # unbalance the two blocks and lose the file to the generic fallbacks.
        def _find(key: str) -> Optional[int]:
            for i, h in enumerate(norm):
                if h == key:
                    return i
            return None

        # The unit each located column DECLARES: the LAST '(...)' group of the cell,
        # read back here instead of thrown away. `_norm_header` strips every group for
        # matching, so a cell may carry another parenthetical ahead of its unit; the
        # trailing one is the unit suffix. It used to be discarded and the template's
        # µs / °C / cP applied regardless, so a `(K)` column read 298.15 as 571 K and
        # an `(ms)` lag axis came in 1000x short. A cell with no suffix keeps the
        # template's unit.
        def _declared_unit(col: Optional[int], template: Optional[str]) -> str:
            groups = _PAREN_RE.findall(header[col]) if col is not None else []
            template = template or ''
            return (groups[-1][1:-1].strip() or template) if groups else template

        lag_unit = _declared_unit(delay_cols[0], _TEMPLATE_LAG_UNIT)

        # Each mapped column the file actually carries, with the unit it declared.
        # A column the table recognises but does not read (`field is None`) resolves
        # to nothing: it is named so the vocabulary is complete, not to be extracted.
        resolved: List[Tuple[_Column, int, str]] = [
            (spec, idx, _declared_unit(idx, spec.unit))
            for spec, idx in ((s, _find(s.key)) for s in _COLUMNS)
            if idx is not None and spec.field is not None
        ]

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
                resolved=resolved, lag_unit=lag_unit,
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
        resolved: List[Tuple[_Column, int, str]] = (),
        lag_unit: str = _TEMPLATE_LAG_UNIT,
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

        # --- parameters (each optional; a bad value degrades to absent, no crash) ---
        # One loop over the resolved columns, so every mapped field follows the same
        # rule and a new row in `_COLUMNS` needs no code here. A field left out of
        # `values` takes DLSFilePreview's own None, which is what "the file did not
        # say" and "the file said something unusable" both mean to the user: a blank
        # cell they fill in at the confirmation step.
        values: Dict[str, Any] = {}
        for spec, idx, unit in resolved:
            raw = cell(idx)
            if spec.text:
                if raw:
                    values[spec.field] = raw
                continue
            number = _parse_float_or_none(raw)
            if number is None:
                continue
            if spec.convert is None:
                values[spec.field] = number
                continue
            try:
                values[spec.field] = spec.convert(number, unit)
            except ValueError:
                pass         # unusable value OR unknown unit: the user enters it

        # Spelled out, not `**values`. Every physical parameter a parser writes has to
        # stay VISIBLE as a keyword here, because the provenance audit for invariant 3
        # reads this call to prove each is None or file-derived and never a literal the
        # file did not state; splatting the dict passed the same values and made this
        # parser vanish from it. `_COLUMNS` still decides what is read and how, and
        # `test_zetasizer_export_every_mapped_column_reaches_the_preview` walks the
        # table by reflection, so a row added without a keyword below is caught.
        return DLSFilePreview(
            source_file=abs_path,
            instrument_name='Malvern Zetasizer',
            delay_times_s=delay_times_s,
            correlogram=correlogram,
            sample_label=values.get('sample_label'),
            polymer_name=values.get('polymer_name'),
            solvent_name=values.get('solvent_name'),
            solvent_refractive_index=values.get('solvent_refractive_index'),
            temperature_K=values.get('temperature_K'),
            viscosity_Pa_s=values.get('viscosity_Pa_s'),
            angle_deg=values.get('angle_deg'),
            wavelength_nm=values.get('wavelength_nm'),
            vendor_mean_count_rate_cps=values.get('vendor_mean_count_rate_cps'),
        )
