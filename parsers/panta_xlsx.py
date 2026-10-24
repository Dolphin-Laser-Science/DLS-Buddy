"""
parsers/panta_xlsx.py
=====================

Parser for the NanoTemper Prometheus Panta DLS autocorrelation .xlsx export:

    PantaXlsxParser     reads the 'ACF' worksheet; one file = one measurement

File format notes (Prometheus Panta, confirmed from a real export)
------------------------------------------------------------------
Container   : .xlsx -- a zip of XML parts. THE FIRST NON-TEXT FORMAT IN THIS
              TREE; every other parser opens a text file with an explicit
              encoding. The sniff is therefore structural (zip -> spreadsheet
              package -> worksheet -> header cells) rather than a first-line
              token match, and the reader is openpyxl rather than a hand-rolled
              zipfile + ElementTree pass: Excel OMITS the <c> element for a
              blank cell, so naive positional column reading silently mis-pairs
              tau with g2 the moment one cell is empty -- the wrong place to
              save a dependency on a path carrying measured numbers.
Worksheets  : 'Chart' (empty) and 'ACF' (the data). 'ACF' is NOT the first
              sheet, so it is looked up by name.
Header row  : A1 = 'Tau [µs]', B1 = 'Autocorrelation'. The delay-time unit is
              read from the bracketed token rather than assumed.
Data rows   : two columns, float32 values widened to float64 by the
              spreadsheet -- the first lag reads 0.20000000298023224 µs, not
              0.2. Any test written against the "nice" value fails.
Extra cols  : ignored (max_col=2). Deliberate; a future export gaining a third
              column (fit residual, per-channel sigma) loads as if it were not
              there.
Metadata    : NONE. See "What this parser cannot supply" below.

Three properties of the file decide the design
----------------------------------------------
1.  THE LAG AXIS IS MULTI-TAU. Sample time doubles every block, spanning
    0.2 µs to ~3.1 s in a few hundred channels instead of the ~6e7 a linear
    correlator would need.

2.  THE BLOCK BOUNDARIES REPEAT. The Panta emits each boundary lag from BOTH
    adjacent blocks (an inclusive-endpoint convention), so a lag value appears
    twice with two slightly different g2 estimates. THE LATER MEMBER IS
    DISCARDED and the finer block's estimate kept.

    THIS IS THE CANONICAL CONSTRUCTION, NOT A CONVENIENCE. Magatti & Ferri
    (2001) §3 prescribe exactly this for multi-tau: adjacent blocks overlap in
    their first P/m lags, and "the first P/m points of each linear correlator
    can be discarded", keeping the finer block's estimates (Eq. (11) and Fig. 3,
    printed p. 4014). That the overlap is half the channels in the commonest
    hardware -- "most of the hardware correlators work with values of P = 16 and
    m = 2; thus each linear correlator uses eight different lag times, and half
    of the points are discarded" -- is on the NEXT page, printed p. 4015; two
    claims, two loci. An export that ships the overlap has simply not applied
    the step the scheme prescribes.

    Averaging the pair instead was measured at +0.021 % on cumulant Rh_z and
    0.000 % on the CONTIN peak, against 0.5-17 % for ordinary analysis choices
    (fit window, skipped channels, cumulant order, 1 °C of temperature) on the
    same file -- and a correct merge would need the symmetric-normalisation
    weights of Schätzel, Drewel & Stimac (1988). So WHICH member is kept is a
    determinism decision far below the noise floor, and is recorded as such.

    `core.data_models._require_increasing_lag_axis` is strict and stays strict;
    this parser hands it a clean axis.

3.  THE FILE STORES NORMALISED g2 -- baseline at 1.0000, intercept near 1.12 --
    whereas DLSMeasurement.correlogram is g2(tau) - 1. The unit baseline is a
    FORMAT FACT of the documented export, in the same category as the µs -> s
    factor, so it is asserted here rather than read; apply_data_form's
    scale-relative band then doubles as a check that column B really is
    normalised.

What this parser CANNOT supply
------------------------------
Nothing at all. The export carries no scattering angle, wavelength, refractive
index, temperature, viscosity, concentration, sample name, duration or count
rate. angle_deg and wavelength_nm are left None: this parser does not fabricate
a physical constant it cannot read (architecture invariant 3), exactly as
parsers/zetasizer_export.py leaves them for the user to supply at the
confirmation step. The scattering angle in particular is load-bearing -- it
enters the recovered size through q squared.

A consequence worth recording so it is not rediscovered: because the file
carries neither a duration nor a count rate, the Panta is the one instrument
whose data can never drive a variance model derived from those quantities.
"""

from __future__ import annotations

import os
import re
import warnings
import zipfile
from typing import List, NoReturn, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

import numpy as np
from openpyxl import load_workbook
from openpyxl.utils.exceptions import InvalidFileException

from core.notes import Note, render
from parsers.base_parser import (
    BaseDLSParser,
    DLSFilePreview,
    MalformedDataError,
    ParseError,
    apply_data_form,
    convert_delay_times,
)

# --------------------------------------------------------------------------- #
# Format constants                                                             #
# --------------------------------------------------------------------------- #

_INSTRUMENT_NAME = 'NanoTemper Prometheus Panta'

_WORKBOOK_PART = 'xl/workbook.xml'          # every .xlsx has one; .docx/.odt do not
_SHEET_NAME = 'ACF'
_HEADER_TAU_RE = re.compile(r'^\s*tau\s*\[\s*(?P<unit>[^\]]+?)\s*\]\s*$', re.IGNORECASE)
_HEADER_ACF = 'autocorrelation'
_N_COLUMNS = 2
_MIN_CHANNELS = 2                           # np.diff and the median scale both need >= 2

# The export stores g2 normalised to a UNIT baseline. This is a format fact of
# the documented export -- not a user parameter and not a fitted quantity.
_NORMALISED_BASELINE = 1.0

# Every prior parser streams text line by line, so an oversized input costs
# nothing. An eager spreadsheet reader does not, so the one guard the tree has
# never needed before is bounded here, from the central directory, before
# anything is decompressed. Sized to the FORMAT, not to what a machine can
# survive: the real export is ~15 kB for 348 channels, so 8 MB is some 500x
# headroom and still bounds openpyxl, whose object graph runs several times the
# byte count. NB this reads the DECLARED sizes -- a crafted archive can
# understate them, which is why the cheap gates above it do the real work.
_MAX_UNCOMPRESSED_BYTES = 8 * 1024 * 1024

# openpyxl surfaces a malformed package as any of these. NB zipfile.BadZipFile
# and LargeZipFile derive from Exception, NOT from OSError, so all three must be
# named; SyntaxError covers both ElementTree's and lxml's XML parse errors.
_WORKBOOK_OPEN_ERRORS = (
    InvalidFileException,
    zipfile.BadZipFile,
    zipfile.LargeZipFile,
    KeyError,
    ValueError,
    SyntaxError,
    TypeError,
    OSError,
)


# --------------------------------------------------------------------------- #
# Internal helpers                                                             #
# --------------------------------------------------------------------------- #

def _sheet_names(workbook_xml: bytes) -> Optional[List[str]]:
    """Worksheet names read straight from `xl/workbook.xml`; None if it is not XML.

    The sniff needs the sheet roster BEFORE openpyxl builds the workbook object.
    An eager load of an unrelated spreadsheet is not cheap -- a 6 MB foreign
    workbook measured 38 s and ~400 MB -- and it would run on the GUI thread, with
    no cancel, now that `.xlsx` is offered in the DLS open dialog. The roster lives
    in this one small part, so reading it here leaves the `_SHEET_NAME` gate
    structurally identical while making it cost a few kB instead of the file.

    Tag matching ignores the namespace so a writer that omits it still resolves.
    """
    try:
        root = ET.fromstring(workbook_xml)
    except ET.ParseError:
        return None
    return [element.get('name') or '' for element in root.iter()
            if element.tag.rsplit('}', 1)[-1] == 'sheet' and element.get('name')]


def _reject(why: str) -> NoReturn:
    """Raise the one plain ParseError this module owns: 'this file is not mine'.

    Every structural sniff branch routes through here, so autodetect can hand the
    file to the next parser (parsers/autodetect.py catches ParseError and
    continues). Anything below the format commit point raises MalformedDataError
    instead.
    """
    raise ParseError(f"Not a NanoTemper Prometheus Panta .xlsx DLS export: {why}.")


def _cell_float(value: object, *, row: int, column: str, file_path: str) -> float:
    """Coerce one worksheet cell to a float, or raise naming where it sits.

    POST-COMMIT. `bool` is rejected explicitly because it is an `int` subclass and
    a TRUE/FALSE cell is never a lag or a correlation value. A string is accepted
    and float()-ed, so a column the spreadsheet stored as text still loads.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            pass
    raise MalformedDataError(
        f"{file_path!r}: worksheet {_SHEET_NAME!r} cell {column}{row} is {value!r}, "
        f"which is not a number. Both columns must be numeric throughout. (A cell "
        f"holding a formula the spreadsheet has never calculated also reads as None "
        f"here.)"
    )


def _read_acf_columns(ws, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read the two data columns of the ACF sheet as float64 arrays.

    POST-COMMIT. A row whose BOTH cells are empty is padding and is skipped --
    Excel's <dimension> routinely overshoots the last written row. A row with
    exactly one empty cell is a malformed row, not padding, and is rejected with
    its sheet row number.
    """
    tau: List[float] = []
    g2: List[float] = []
    for offset, values in enumerate(
        ws.iter_rows(min_row=2, max_col=_N_COLUMNS, values_only=True)
    ):
        row = offset + 2                       # 1-based sheet row, header is row 1
        padded = (tuple(values) + (None, None))[:_N_COLUMNS]
        if all(v is None for v in padded):
            continue
        tau.append(_cell_float(padded[0], row=row, column='A', file_path=file_path))
        g2.append(_cell_float(padded[1], row=row, column='B', file_path=file_path))
    return np.asarray(tau, dtype=float), np.asarray(g2, dtype=float)


def _discard_overlap_channels(
    tau: np.ndarray, g2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep the FIRST member of each run of equal lag times; drop the rest.

    Returns ``(tau_kept, g2_kept, dropped_indices)``.

    Generic over run LENGTH rather than written for pairs: each element is
    compared with its immediate predecessor, so a run of k equal lags drops k-1
    points and a correlator with a binning ratio above 2 needs no second code
    path.

    EXACT equality, no tolerance. The two members of a boundary pair are the same
    correlator channel reported twice and are bit-identical in tau. A tolerance
    would be a free parameter deciding which points reach every fit, and the axis
    gives it nothing to be sized from -- adjacent DISTINCT lags differ by 25 % at
    the fine end and by a factor 2**k at the coarse end.

    Runs are found among ADJACENT entries only. A non-adjacent repeat survives
    here and is caught by the strictly-increasing post-condition in parse(),
    loudly. Note the same post-condition fires if a future firmware ever writes
    the two members of a boundary pair with different last-bit rounding: nothing
    is dropped, and the file is refused with "still not strictly increasing after
    discarding 0 channel(s)". That is the intended failure, not a parser bug.
    """
    keep = np.empty(tau.size, dtype=bool)
    keep[:1] = True                            # not keep[0]: size 0 must not IndexError
    np.not_equal(tau[1:], tau[:-1], out=keep[1:])
    return tau[keep], g2[keep], np.flatnonzero(~keep)


# --------------------------------------------------------------------------- #
# PantaXlsxParser                                                              #
# --------------------------------------------------------------------------- #

class PantaXlsxParser(BaseDLSParser):
    """Parser for the NanoTemper Prometheus Panta .xlsx autocorrelation export.

    Returns exactly one DLSFilePreview carrying the correlogram (g2-1) and its
    lag axis in seconds. Every physical-parameter field is None: the file has
    none, and this parser invents none.

    Usage example::

        previews = PantaXlsxParser().parse('acf_export.xlsx')
        p = previews[0]
        p.polymer_name = 'lysozyme'
        p.solvent_name = 'water'
        p.concentration_g_per_mL = 1.0e-3
        p.temperature_K = 298.15
        p.angle_deg = 147.0              # from the instrument, not the file
        p.wavelength_nm = 405.0          # from the instrument, not the file
        p.solvent_refractive_index = 1.332
        measurement = p.build()
    """

    def parse(self, file_path: str) -> Sequence[DLSFilePreview]:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"No such file: {file_path!r}")

        # --- sniff: cheapest structural check first ---------------------------
        if not zipfile.is_zipfile(file_path):
            _reject('the file is not a zip container, and an .xlsx is a zip of XML '
                    'parts (a password-protected workbook is an OLE2 compound file '
                    'and lands here too)')

        try:
            with zipfile.ZipFile(file_path) as archive:
                infos = archive.infolist()
                if _WORKBOOK_PART not in {info.filename for info in infos}:
                    _reject(f'the zip has no {_WORKBOOK_PART!r} part, so it is not '
                            f'a spreadsheet package')
                workbook_xml = archive.read(_WORKBOOK_PART)
        except (zipfile.BadZipFile, zipfile.LargeZipFile, OSError) as exc:
            _reject(f'the zip container could not be read '
                    f'({type(exc).__name__}: {exc})')

        # The ACF gate runs HERE, on the roster in xl/workbook.xml, and not after
        # load_workbook: an unrelated spreadsheet must be rejected before anything
        # decompresses it. See _sheet_names.
        sheet_names = _sheet_names(workbook_xml)
        if sheet_names is None:
            _reject(f'{_WORKBOOK_PART!r} is present but is not parseable XML, so '
                    f'the package is not a readable spreadsheet')
        if _SHEET_NAME not in sheet_names:
            _reject(f'the workbook has no {_SHEET_NAME!r} worksheet '
                    f'(its sheets are {sheet_names})')

        total_bytes = sum(info.file_size for info in infos)
        if total_bytes > _MAX_UNCOMPRESSED_BYTES:
            _reject(f'the package expands to {total_bytes / 1e6:.0f} MB, above the '
                    f'{_MAX_UNCOMPRESSED_BYTES / 1e6:.0f} MB this reader accepts; a '
                    f'Panta ACF export is a two-column table of a few hundred rows')

        try:
            # A file handle, not the path: openpyxl checks the FILENAME EXTENSION
            # against its supported list on the path branch, which would make the
            # sniff depend on what the file is called rather than what it is.
            # read_only=False keeps the read eager, so the handle closes with the
            # `with` and no lazy worksheet outlives it. data_only=True yields the
            # cached value if a user has round-tripped the export through Excel
            # and left a formula in a column.
            with open(file_path, 'rb') as handle:
                workbook = load_workbook(handle, read_only=False, data_only=True)
        except _WORKBOOK_OPEN_ERRORS as exc:
            _reject(f'openpyxl could not open it as a workbook '
                    f'({type(exc).__name__}: {exc})')

        worksheet = workbook[_SHEET_NAME]
        header = next(
            worksheet.iter_rows(min_row=1, max_row=1, max_col=_N_COLUMNS,
                                values_only=True),
            None,
        )
        if header is None:
            _reject(f'worksheet {_SHEET_NAME!r} is empty')
        cell_a1, cell_b1 = (tuple(header) + (None, None))[:_N_COLUMNS]
        unit_match = _HEADER_TAU_RE.match(str(cell_a1 or ''))
        if unit_match is None or str(cell_b1 or '').strip().lower() != _HEADER_ACF:
            _reject(f'worksheet {_SHEET_NAME!r} has header cells {cell_a1!r}, '
                    f'{cell_b1!r}; expected \'Tau [<unit>]\' and \'Autocorrelation\'')

        # --- format commit point ----------------------------------------------
        # Above: "not my format", a plain ParseError, so autodetect tries the next
        # parser. Below: the file has proved it IS this format (a spreadsheet
        # package with an 'ACF' worksheet whose two header cells match), so every
        # failure is malformed DATA and raises MalformedDataError, whose message
        # reaches the user. The header's UNIT SHAPE is part of the signature and is
        # checked above; whether the bracketed token names a CONVERTIBLE unit is a
        # statement about a file already identified, so it sits below.

        tau_raw, g2_raw = _read_acf_columns(worksheet, file_path)

        tau_kept, g2_kept, dropped = _discard_overlap_channels(tau_raw, g2_raw)

        # Sized to what LEAVES this function, not to what was read: the discard above
        # shrinks the axis, so guarding tau_raw would let an all-duplicate sheet emit a
        # 1-channel correlogram -- which _require_increasing_lag_axis also passes,
        # vacuously, on a length-0 np.diff. This must also precede the post-condition
        # below, which indexes tau_kept[0].
        if tau_kept.size < _MIN_CHANNELS:
            raise MalformedDataError(
                f"{file_path!r}: worksheet {_SHEET_NAME!r} yields {tau_kept.size} "
                f"channel(s) -- {tau_raw.size} data row(s) below its header, "
                f"{dropped.size} of them discarded as repeated delay times. A "
                f"correlogram needs at least {_MIN_CHANNELS}."
            )

        # Post-condition on the discard, checked here rather than deferred to
        # DLSMeasurement: the GUI does not build a measurement at load time, so a
        # deferred failure would surface only after the user has typed every
        # physical parameter, as a bare ValueError naming neither file nor row.
        bad = np.flatnonzero(np.diff(tau_kept) <= 0.0)
        if tau_kept[0] < 0.0 or bad.size:
            if tau_kept[0] < 0.0:
                where = f'the first lag {tau_kept[0]!r} is negative'
            else:
                where = (f'lag {tau_kept[bad[0] + 1]!r} does not exceed the preceding '
                         f'{tau_kept[bad[0]]!r}')
            raise MalformedDataError(
                f"{file_path!r}: after discarding {dropped.size} repeated "
                f"block-boundary channel(s), the delay axis is still not strictly "
                f"increasing from a non-negative start -- {where} (values as written, "
                f"in the header's units). A correlator lag axis increases by "
                f"construction, so this export does not have the multi-tau block "
                f"layout whose overlap this parser removes; loading it would silently "
                f"reorder or double-weight channels in every fit."
            )

        unit = unit_match.group('unit')
        try:
            delay_times_s = convert_delay_times(tau_kept, unit)
        except MalformedDataError:
            # convert_delay_times ALSO raises this, via require_finite_data, when the
            # lag column holds a non-finite value -- and MalformedDataError is a
            # ValueError, so without this arm the ValueError arm below would relabel
            # bad DATA as a bad UNIT. Must precede it, or it is dead code.
            raise
        except ValueError as exc:
            # An unrecognised unit token: convert_delay_times raises a BARE ValueError
            # there, which autodetect does not catch -- it would escape the load loop
            # as a traceback.
            raise MalformedDataError(
                f"{file_path!r}: worksheet {_SHEET_NAME!r} header cell A1 is "
                f"{cell_a1!r}, whose delay-time unit {unit!r} cannot be converted to "
                f"seconds. The sheet and both header cells match, so this is a "
                f"Prometheus Panta export, but its lag axis cannot be read. ({exc})"
            ) from exc

        try:
            correlogram = apply_data_form(
                g2_kept, 'g2', _NORMALISED_BASELINE, None, file_path)
        except MalformedDataError:
            raise                       # already the right class and message
        except ParseError as exc:
            # apply_data_form's baseline-vs-scale band raises a plain ParseError,
            # and we are past the commit point. With B pinned to the format's unit
            # baseline that band is really a check that column B is NORMALISED, so
            # say so. This arm MUST follow the MalformedDataError arm above --
            # MalformedDataError subclasses ParseError, so the reverse order would
            # make the first arm dead code.
            raise MalformedDataError(
                f"{file_path!r}: worksheet {_SHEET_NAME!r} column B is headed "
                f"'Autocorrelation' but does not look like a NORMALISED "
                f"autocorrelation (the Panta writes g2 with its baseline at 1.0000 "
                f"and an intercept near 1.12). {exc}"
            ) from exc

        notes = []
        if dropped.size:
            # The record names the count and that the first copy of each repeated lag
            # was kept (the module docstring, point 2, gives why).
            overlap_note = Note('parser.repeated_lags_dropped',
                                ('delay_times_s', 'correlogram'),
                                {'n_dropped': int(dropped.size), 'n': int(tau_raw.size),
                                 'n_kept': int(tau_kept.size)})
            notes.append(overlap_note)
            # Only the overlap gets a warnings.warn: it is the one event where
            # this parser DISCARDS measured data, which a headless caller with no
            # access to `notes` must still hear. The no-metadata note is a
            # standing property of the format rather than something that happened
            # to this file, so warning on every healthy load would be noise -- it
            # travels on `notes`, which is the channel the GUI actually surfaces.
            warnings.warn(f"{file_path!r}: {render(overlap_note, surface='export')}",
                          UserWarning, stacklevel=2)
        notes.append(Note('parser.no_metadata'))

        preview = DLSFilePreview(
            source_file=os.path.abspath(file_path),
            instrument_name=_INSTRUMENT_NAME,
            delay_times_s=delay_times_s,
            correlogram=correlogram,
            notes=tuple(notes),
        )
        return [preview]
