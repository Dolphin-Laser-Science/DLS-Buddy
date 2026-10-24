"""
parsers/brookhaven_sls.py
=========================

Parsers for Brookhaven Particle Explorer SLS intensity exports. The instrument emits
TWO .CSV layouts, one class each:

    BrookhavenSLSParser      layout A -- the concentration series (Zimm / Debye plot)
    BrookhavenSLSScanParser  layout B -- the single-run angular scan (Guinier)

Both inherit from BaseSLSParser and both are flat "key,value" files sharing a metadata
header; they are told apart by the data block alone. Layout A yields one
SLSFilePreview per concentration (seven concentrations -> seven previews). Layout B
yields the sample, preceded by the solvent blank when the run recorded one -- one
preview or two.

The two sniffs are mutually exclusive by construction, not by registration order --
layout A is the one declaring "Number of Concentrations Measured", layout B the one
with "Sample Intensity - Angle N" rows, and each parser tests both facts. See
`BrookhavenSLSScanParser` for layout B; the rest of this docstring describes layout A.

Layout A format notes (confirmed from real export)
---------------------------------------------------

The file is a flat list of "key,value" rows. There are three regions:

  1. Metadata header — one key-value pair per row, e.g.:
        Sample ID,PS (900k) in Toluene | Post Re-realignment Test
        Sample Liquid,Toluene
        Refractive Index of Sample Liquid,1.502
        Refractive Index Inc. (dn/dc) (mL/g),0.11
        Calibration Constant,3.224E-10
        Calibration Liquid,Toluene
        Refractive Index of Calibration Liquid,1.502
        Rayleigh Ratio of Calibration Liquid,2.803E-05
        Wavelength (nm),532
        Number of Angles Measured,13
        Number of Concentrations Measured,7

  2. Angle and concentration tables — numbered rows:
        Angle 1 (degrees),35
        ...
        Angle 13 (degrees),145
        Concentration 1 (mg/mL),0
        ...
        Concentration 7 (mg/mL),1.364

  3. Intensity data — flat key-value pairs, one per (concentration, angle):
        Intensity - Concentration 1 - Angle 1,153006
        Intensity - Concentration 1 - Angle 2,123982
        ...
     The N x M grid is serialized concentration-major (all angles of
     concentration 1, then all angles of concentration 2, ...).

What the parser extracts vs. what the user must supply
------------------------------------------------------
EXTRACTED from the file (populated in each preview):
    - angles_deg              from the Angle table
    - intensities             from the Intensity grid (per concentration)
    - concentration_g_per_mL  from the Concentration table (mg/mL -> g/mL)
    - solvent_name            from "Sample Liquid"
    - solvent_refractive_index from "Refractive Index of Sample Liquid"
    - dn_dc_mL_per_g          from "Refractive Index Inc. (dn/dc)" -- OPTIONAL: a
                              blank or unparseable cell leaves it absent rather than
                              failing the load (the scale-free analyses do not use it)
    - wavelength_nm           from "Wavelength (nm)"
    - calibration_constant    from "Calibration Constant"
    - standard_name           from "Calibration Liquid"
    - standard_rayleigh_ratio_file  from "Rayleigh Ratio of Calibration Liquid"
                              (informational only; analysis uses the program-
                               computed toluene value instead — Takahashi 2019
                               at 532 nm / Sivokhin & Kazantsev 2021 at 660 nm)
    - standard_refractive_index  from "Refractive Index of Calibration Liquid"
    - sample_label            from "Sample ID"

USER must supply at the confirmation step (not in the file):
    - polymer_name            (the Sample ID contains it but not cleanly,
                               e.g. "PS (900k) in Toluene | ..."; too fragile
                               to auto-extract)
    - temperature_K           (Brookhaven SLS export has no temperature field)

Concentration units
-------------------
The file labels concentrations in mg/mL. The parser converts to g/mL (the
canonical internal unit) by multiplying by 1e-3.

Change history
--------------
2026-06-12  Initial implementation. (brookhaven_sls.py v1)
2026-08-07  Added BrookhavenSLSScanParser for layout B (the angular-scan export), and
            factored the table check into `_table_mismatch_message` so a pre- and a
            post-commit caller can share the arithmetic.
"""

from __future__ import annotations

import math
import os
import re
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

from parsers.base_parser import (
    BaseSLSParser,
    MalformedDataError,
    ParseError,
    SLSFilePreview,
    read_text_lines,
)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_ENCODING = 'latin-1'
_DELIMITER = ','

_INSTRUMENT_NAME = 'Brookhaven Particle Explorer'

# Exact metadata keys as they appear in the file. Centralizing them here means
# that if Brookhaven changes a label in a future software version, there is one
# obvious place to update.
_KEY_SAMPLE_ID = 'Sample ID'
_KEY_SAMPLE_LIQUID = 'Sample Liquid'
_KEY_N_SOLVENT = 'Refractive Index of Sample Liquid'
_KEY_DN_DC = 'Refractive Index Inc. (dn/dc) (mL/g)'
_KEY_CAL_CONSTANT = 'Calibration Constant'
_KEY_CAL_LIQUID = 'Calibration Liquid'
_KEY_N_STANDARD = 'Refractive Index of Calibration Liquid'
_KEY_RAYLEIGH_STANDARD = 'Rayleigh Ratio of Calibration Liquid'
_KEY_WAVELENGTH = 'Wavelength (nm)'
_KEY_N_ANGLES = 'Number of Angles Measured'
_KEY_N_CONCENTRATIONS = 'Number of Concentrations Measured'

# Regular expressions for the numbered rows.
# Angle row:          "Angle 3 (degrees)" -> captures 3
# Concentration row:  "Concentration 2 (mg/mL)" -> captures 2
# Intensity row:      "Intensity - Concentration 2 - Angle 7" -> captures 2, 7
_ANGLE_KEY_RE = re.compile(r'^Angle\s+(\d+)\s*\(degrees\)\s*$', re.IGNORECASE)
_CONC_KEY_RE = re.compile(r'^Concentration\s+(\d+)\s*\(mg/mL\)\s*$', re.IGNORECASE)
_INTENSITY_KEY_RE = re.compile(
    r'^Intensity\s*-\s*Concentration\s+(\d+)\s*-\s*Angle\s+(\d+)\s*$',
    re.IGNORECASE,
)

# Layout B (angular scan) intensity rows -- see the class docstring of
# `BrookhavenSLSScanParser`. BOTH ends are anchored: `^Intensity` above cannot match
# "Sample Intensity - Angle 1", and `$` here is what stops "Sample Intensity - Angle 1"
# ever being read out of an "Intensity - Concentration 1 - Angle 1" row. The mutual
# exclusion of the two layouts rests on these anchors, so do not relax them.
_SAMPLE_INTENSITY_KEY_RE = re.compile(
    r'^Sample\s+Intensity\s*-\s*Angle\s+(\d+)\s*$', re.IGNORECASE)
_LIQUID_INTENSITY_KEY_RE = re.compile(
    r'^Liquid\s+Intensity\s*-\s*Angle\s+(\d+)\s*$', re.IGNORECASE)

_MG_PER_ML_TO_G_PER_ML = 1.0e-3

# Ceiling on a declared Angle/Concentration count -- see `_parse_count`.
_MAX_DECLARED_COUNT = 10_000

# The suffix distinguishing the c = 0 series from its sample: both previews come from one
# file and would otherwise take its single "Sample ID" verbatim.
#
# A PREVIEW-level label, not what the sidebar shows: the GUI names its SLS rows from the
# source filename plus a 'solvent ref' marker on whichever series holds the reference slot,
# which already tells the two apart. The suffix is still worth setting -- it is what a
# caller using the parser directly (or `SLSFilePreview.build()`) gets on the measurement.
_BLANK_LABEL_SUFFIX = ' (solvent blank)'


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_rows(file_path: str) -> List[Tuple[str, str]]:
    """Read the file into a list of (key, value) string pairs.

    Each row is split on the FIRST comma only, because some values (notably
    the Sample ID) contain commas and other punctuation. Rows that do not
    contain a comma are skipped.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ParseError
        If the file cannot be read (``read_text_lines``).
    """
    raw_lines = read_text_lines(file_path, (_ENCODING,))

    rows: List[Tuple[str, str]] = []
    for line in raw_lines:
        if _DELIMITER not in line:
            continue
        key, value = line.split(_DELIMITER, 1)   # split on first comma only
        rows.append((key.strip(), value.strip()))
    return rows


def _missing_key_message(key: str, file_path: str) -> str:
    return (f"Required metadata key {key!r} not found in {file_path!r}. "
            f"This file may not be a Brookhaven SLS intensity export, or the "
            f"export format has changed.")


def _table_mismatch_message(
    table: Mapping[int, object], expected_n: int, label: str, file_path: str
) -> Optional[str]:
    """The complaint about a numbered table, or None when it is consistent.

    Builds the message; never raises. That split is what lets the SAME set arithmetic
    serve a PRE-commit caller (which owes a plain `ParseError`) and a POST-commit one
    (which owes `MalformedDataError`) without an `exc_cls=` parameter -- the indirection
    the comment block below forbids for a gate reason, not a style one.
    """
    expected = set(range(1, expected_n + 1))
    found = set(table.keys())
    if found == expected:
        return None
    missing = sorted(expected - found)
    extra = sorted(found - expected)
    return (f"{label} table in {file_path!r} is inconsistent with the "
            f"declared count ({expected_n}). "
            f"Missing indices: {missing or 'none'}; "
            f"unexpected indices: {extra or 'none'}.")


def _bad_float_message(value: str, key: str, file_path: str) -> str:
    return (f"Could not parse value {value!r} for key {key!r} in {file_path!r} "
            f"as a number.")


# TWO THIN PAIRS, not one helper with an `exc_cls=` keyword, for a GATE reason rather than a
# style one: `tests/test_parser_error_taxonomy.py` enumerates every `raise ParseError(...)`
# in `parsers/` by AST, matching on the class NAME, so a `raise exc_cls(...)` is invisible
# to it -- and a call site that forgot to pass the subclass would not turn it red.

# WHICH ONE TO CALL. Everything ABOVE `parse`'s `--- format commit point ---` marker uses
# the plain pair: reading `Number of Angles` IS the sniff, and a plain two-column SLS CSV
# lacks it, so raising the subclass there would stop the loader before the generic fallback,
# breaking generic SLS loading outright. Everything BELOW uses the `_committed` pair -- the
# file has proved what it is, so a problem is malformed data and its message is for the user.

def _require_key(metadata: Dict[str, str], key: str, file_path: str) -> str:
    """Pre-commit: a missing key means "this may not be my format"."""
    if key not in metadata:
        raise ParseError(_missing_key_message(key, file_path))
    return metadata[key]


def _require_key_committed(metadata: Dict[str, str], key: str, file_path: str) -> str:
    """Post-commit: the file IS this format, so a missing key is malformed data."""
    if key not in metadata:
        raise MalformedDataError(_missing_key_message(key, file_path))
    return metadata[key]


def _parse_float(value: str, key: str, file_path: str) -> float:
    """Pre-commit: see `_require_key`."""
    try:
        return float(value)
    except ValueError as exc:
        raise ParseError(_bad_float_message(value, key, file_path)) from exc


def _parse_float_committed(value: str, key: str, file_path: str) -> float:
    """Post-commit: see `_require_key_committed`."""
    try:
        return float(value)
    except ValueError as exc:
        raise MalformedDataError(_bad_float_message(value, key, file_path)) from exc


def _check_table_complete_committed(
    table: Mapping[int, object], expected_n: int, label: str, file_path: str
) -> None:
    """Post-commit table check: a gap is malformed DATA, and the message says where.

    BOTH layouts commit before they get here -- each puts its format-commit marker above
    its own table checks -- so there is no pre-commit twin left: one table check, one
    class. "Angle table is
    inconsistent with the declared count" is a diagnosis the user can act on rather
    than a sniff verdict that gets swallowed into "unrecognised format".
    """
    msg = _table_mismatch_message(table, expected_n, label, file_path)
    if msg is not None:
        raise MalformedDataError(msg)


def _parse_count(value: str, key: str, file_path: str) -> int:
    """Read a declared table size: finite, non-negative, and plausibly small.

    PRE-commit for both layouts -- reading a declared count IS part of each sniff -- so a
    bad one is a plain `ParseError`, the "not my format" verdict autodetect swallows.

    THREE tails, and they fail differently, which is why one helper owns all of them
    instead of a predicate at each call site (rule 12):

      * non-numeric  -> `_parse_float`'s ParseError, as before.
      * non-finite   -> `int(float('nan'))` raises ValueError and `int(float('inf'))`
                        raises OverflowError. Neither is a ParseError, so neither is
                        caught by `parsers/autodetect.py`'s handler -- they escape the
                        detection loop and take the whole LOAD down, for a file this
                        parser was merely sniffing and may not even own.
      * absurdly big -> nothing raises at all. The count flows into
                        `_table_mismatch_message`, which evaluates
                        `set(range(1, n + 1))`; at n = 1e30 that allocates without bound
                        and the app stops responding. A hang is worse than a crash --
                        no message, nothing to report -- so a finiteness check alone is
                        not enough. The count needs a CEILING.

    The bound is deliberately far above any real instrument (a goniometer scans tens of
    angles, a Zimm set tens of concentrations) and far below anything that costs memory:
    it exists to separate "a plausible table" from "a corrupt header", not to police the
    hardware. Sized to the operation it protects -- `set(range(1, n + 1))` at the ceiling
    is ten thousand integers.
    """
    n = _parse_float(value, key, file_path)
    if not math.isfinite(n) or n < 0 or n > _MAX_DECLARED_COUNT:
        raise ParseError(
            f"Declared count {value!r} for key {key!r} in {file_path!r} is not a "
            f"plausible table size (expected a whole number between 0 and "
            f"{_MAX_DECLARED_COUNT}). This file may not be a Brookhaven SLS intensity "
            f"export, or its header is corrupt."
        )
    return int(n)


def _optional_float(value: Optional[str]) -> Optional[float]:
    """Parse an optional metadata value to float, returning None if absent or
    unparseable. Used for informational calibration fields that should not abort the
    parse if malformed -- both layouts treat them the same way, so it is module-level
    rather than a staticmethod on either class."""
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# BrookhavenSLSParser
# ---------------------------------------------------------------------------

class BrookhavenSLSParser(BaseSLSParser):
    """Parser for Brookhaven Particle Explorer SLS intensity .CSV files.

    Returns one SLSFilePreview per concentration. Each preview shares the same
    optics and calibration metadata (extracted once from the header) but has
    its own concentration value and its own intensity array across all angles.

    Usage
    -----
    ::

        parser = BrookhavenSLSParser()
        previews = parser.parse('path/to/zimm_intensities.csv')

        # previews is a list, one per concentration (including c = 0 solvent).
        # At the confirmation step, supply the two fields the file lacks --
        # applied to every preview, since they share polymer and temperature:
        for preview in previews:
            preview.polymer_name = 'PS'
            preview.temperature_K = parser.convert_temperature(25.0, 'C')

        measurements = [p.build() for p in previews]
    """

    def parse(self, file_path: str) -> List[SLSFilePreview]:
        """Parse a Brookhaven SLS intensity export.

        Parameters
        ----------
        file_path : str
            Path to the SLS intensity .CSV file.

        Returns
        -------
        List[SLSFilePreview]
            One preview per concentration, in concentration order. The
            solvent (concentration 0) is included as a normal preview.
            polymer_name and temperature_K are None (user supplies them).

        Raises
        ------
        FileNotFoundError
            If file_path does not exist.
        ParseError
            If required metadata keys are missing, the angle/concentration
            counts do not match the data found, or the intensity grid is
            incomplete.
        """
        rows = _read_rows(file_path)
        if not rows:
            raise ParseError(f"{file_path!r} contains no comma-separated rows.")

        # --- separate scalar metadata from the numbered/structured rows ---
        # We build a plain dict for the scalar metadata. Numbered rows (angles,
        # concentrations, intensities) are matched by regex and collected
        # separately so a stray duplicate scalar key cannot shadow them.
        metadata: Dict[str, str] = {}
        angle_table: Dict[int, float] = {}
        conc_table: Dict[int, float] = {}
        intensity_grid: Dict[Tuple[int, int], float] = {}

        for key, value in rows:
            angle_match = _ANGLE_KEY_RE.match(key)
            if angle_match:
                idx = int(angle_match.group(1))
                angle_table[idx] = _parse_float(value, key, file_path)
                continue

            conc_match = _CONC_KEY_RE.match(key)
            if conc_match:
                idx = int(conc_match.group(1))
                conc_table[idx] = _parse_float(value, key, file_path)
                continue

            intensity_match = _INTENSITY_KEY_RE.match(key)
            if intensity_match:
                c_idx = int(intensity_match.group(1))
                a_idx = int(intensity_match.group(2))
                intensity_grid[(c_idx, a_idx)] = _parse_float(value, key, file_path)
                continue

            # Otherwise it is a scalar metadata row. First occurrence wins
            # (the header appears before any numbered rows).
            if key not in metadata:
                metadata[key] = value

        # --- extract and validate the declared counts ---
        n_angles = _parse_count(
            _require_key(metadata, _KEY_N_ANGLES, file_path),
            _KEY_N_ANGLES, file_path,
        )
        n_conc = _parse_count(
            _require_key(metadata, _KEY_N_CONCENTRATIONS, file_path),
            _KEY_N_CONCENTRATIONS, file_path,
        )

        # THE STRUCTURAL DISCRIMINANT, and the second half of the sniff. Declaring the two
        # counts is not enough on its own: `GenericSLSParser` accepts a file with up to 5 %
        # unparseable rows, so a plain two-column table of >= 40 numeric rows can carry both
        # count key strings in its header and still be a file the fallback would have
        # loaded. So layout A requires DELIVERED intensity data too, as layout B requires
        # delivered `Sample Intensity - Angle N` rows. Cost: a layout-A export truncated to
        # zero intensity rows reports "not my format" rather than a table mismatch -- the
        # same verdict layout B gives its own empty-rows case.
        if not intensity_grid:
            raise ParseError(
                f"{file_path!r} is not a Brookhaven SLS concentration series: it "
                f"declares {_KEY_N_ANGLES!r} and {_KEY_N_CONCENTRATIONS!r} but carries "
                f"no 'Intensity - Concentration C - Angle A' rows."
            )

        # --- format commit point ---------------------------------------------------
        # The file has now DECLARED both counts AND DELIVERED intensity rows, which
        # together identify this layout: a plain two-column SLS CSV has none of them, so
        # `_require_key`'s plain-ParseError default above still routes it to the generic
        # fallback, and layout B declares no concentration count, so it too exits pre-commit
        # here.

        # The marker sits ABOVE the table checks because the DECLARATION is the
        # discriminant, not the delivery: a file declaring both counts is this format
        # whether or not its tables are complete, and "incomplete" is then malformed DATA
        # rather than a verdict about the format -- which is how the user gets a message
        # naming the gap. NOT closed by that: a blank or non-numeric INTENSITY CELL is
        # float-parsed by the row loop far above, which cannot move below the counts (it
        # builds the tables they describe), so that case still reports "not my format".
        # --- the tables must deliver what the counts declared -----------------------
        _check_table_complete_committed(angle_table, n_angles, 'Angle', file_path)
        _check_table_complete_committed(conc_table, n_conc, 'Concentration', file_path)

        # --- build the shared optics / calibration values (once) ---
        wavelength_nm = _parse_float_committed(
            _require_key_committed(metadata, _KEY_WAVELENGTH, file_path),
            _KEY_WAVELENGTH, file_path,
        )
        solvent_n = _parse_float_committed(
            _require_key_committed(metadata, _KEY_N_SOLVENT, file_path),
            _KEY_N_SOLVENT, file_path,
        )
        # dn/dc is OPTIONAL, unlike the geometry and solvent keys above -- it feeds only
        # the optical constant K, which cancels in every coefficient ratio (see
        # SLSMeasurement's "Optional optics"), so requiring it aborted the whole load for
        # exactly the samples the absent-dn/dc path serves. `_optional_float` maps a blank
        # or unparseable cell to None rather than raising, the right reading for a field
        # the export always emits but the operator may leave empty. Never defaulted
        # (invariant 3): absent stays absent, and the scan layout already agreed.
        dn_dc = _optional_float(metadata.get(_KEY_DN_DC))
        solvent_name = _require_key_committed(metadata, _KEY_SAMPLE_LIQUID, file_path)
        sample_label = metadata.get(_KEY_SAMPLE_ID)

        # Calibration reference values (informational only; never used in
        # analysis, which uses the program-computed toluene value — Takahashi
        # 2019 at 532 nm / Sivokhin & Kazantsev 2021 at 660 nm).
        cal_constant = _optional_float(metadata.get(_KEY_CAL_CONSTANT))
        standard_name = metadata.get(_KEY_CAL_LIQUID)
        standard_rayleigh = _optional_float(metadata.get(_KEY_RAYLEIGH_STANDARD))
        standard_n = _optional_float(metadata.get(_KEY_N_STANDARD))

        # --- assemble angle array (ordered by angle index 1..n_angles) ---
        angles_deg = np.array(
            [angle_table[i] for i in range(1, n_angles + 1)], dtype=float
        )

        # --- build one preview per concentration ---
        previews: List[SLSFilePreview] = []
        abs_path = os.path.abspath(file_path)

        for c_idx in range(1, n_conc + 1):
            # Collect this concentration's intensities across all angles, in
            # angle-index order. Every (c_idx, a_idx) cell must be present.
            intensities = np.empty(n_angles, dtype=float)
            for a_idx in range(1, n_angles + 1):
                cell = (c_idx, a_idx)
                if cell not in intensity_grid:
                    raise MalformedDataError(
                        f"Missing intensity for Concentration {c_idx}, "
                        f"Angle {a_idx} in {file_path!r}. The intensity grid "
                        f"is incomplete (expected {n_conc} x {n_angles} = "
                        f"{n_conc * n_angles} values)."
                    )
                intensities[a_idx - 1] = intensity_grid[cell]

            conc_g_per_mL = conc_table[c_idx] * _MG_PER_ML_TO_G_PER_ML

            preview = SLSFilePreview(
                source_file=abs_path,
                instrument_name=_INSTRUMENT_NAME,
                angles_deg=angles_deg.copy(),
                intensities=intensities,
                sample_label=sample_label if sample_label else None,
                solvent_name=solvent_name if solvent_name else None,
                concentration_g_per_mL=conc_g_per_mL,
                wavelength_nm=wavelength_nm,
                solvent_refractive_index=solvent_n,
                dn_dc_mL_per_g=dn_dc,
                calibration_constant=cal_constant,
                standard_name=standard_name if standard_name else None,
                standard_rayleigh_ratio_file=standard_rayleigh,
                standard_refractive_index=standard_n,
                # polymer_name and temperature_K intentionally left None.
            )
            previews.append(preview)

        return previews

    # NB no `_check_table_complete` staticmethod here: both layouts commit BEFORE their
    # table checks and share the module-level `_check_table_complete_committed`, so there is
    # one table check in this module rather than a pre-commit and a post-commit spelling of
    # the same arithmetic.


# ---------------------------------------------------------------------------
# BrookhavenSLSScanParser  (layout B -- the single-run angular scan)
# ---------------------------------------------------------------------------

class BrookhavenSLSScanParser(BaseSLSParser):
    """Parser for the Brookhaven SLS **angular scan** .CSV -- the OTHER layout.

    One acquisition at one concentration, plus the solvent blank when one was measured
    in the same run. This is the export the instrument produces for a single-
    concentration experiment (a Guinier scan); a concentration series arrives in the
    layout `BrookhavenSLSParser` reads.

    The two share a header. They are told apart by the data block:

        layout A     Number of Concentrations Measured, 7
                     Concentration 1 (mg/mL), 0
                     Intensity - Concentration 1 - Angle 1, 153006
        layout B     Sample Intensity - Angle 1, 1460639
                     Liquid Intensity - Angle 1, 773009.1      (only with an in-run blank)

    Returns ONE or TWO previews. The `Liquid Intensity` block is present only when a
    blank was measured in the same run: the instrument's plain export -- a sample scan,
    a solvent scan, the toluene calibrant -- carries `Sample Intensity` rows alone. When
    the block is present the blank comes first (`concentration_g_per_mL = 0.0`, label
    suffixed `(solvent blank)`), then the sample; `Workspace.regroup` routes a c = 0 SLS
    series into the sample's solvent-reference slot, so one such file is enough to run
    an excess-Rayleigh analysis once the user has confirmed the parameters. Without it
    the single preview is the sample, referenced against whichever blank the sample
    holds (a separately loaded solvent scan, or the one picked in the Data tab).

    What the file does NOT carry, and the user therefore supplies at the confirmation
    step -- two more than layout A:

        polymer_name             (the Sample ID contains it, but not cleanly)
        temperature_K            (no temperature field in either layout)
        concentration_g_per_mL   (the sample's; layout B records no concentration)
        dn_dc_mL_per_g           (no dn/dc row -- and it is never defaulted anyway)

    There is no calibration block either, so Mw and absolute A2 need a calibrant point
    entered separately, exactly as for an uncalibrated layout-A file. The optional
    calibration fields are still read if a future software version emits them.
    """

    def parse(self, file_path: str) -> List[SLSFilePreview]:
        """Parse a Brookhaven SLS angular-scan export.

        Returns
        -------
        List[SLSFilePreview]
            The sample (concentration None), preceded by the solvent blank (c = 0)
            when the file carries a `Liquid Intensity` block. polymer_name,
            temperature_K, dn_dc_mL_per_g and the sample's concentration are left for
            the user.

        Raises
        ------
        FileNotFoundError
            If file_path does not exist.
        ParseError
            If the file is not this layout (the sniff -- see the commit-point marker).
        MalformedDataError
            If it IS this layout but the angle or intensity tables are incomplete or
            carry a value that is not a number.
        """
        rows = _read_rows(file_path)
        if not rows:
            raise ParseError(f"{file_path!r} contains no comma-separated rows.")

        # Values stay STRINGS through the bucketing pass and are converted after the
        # commit point. Converting here would spend the pre-commit `_parse_float`, which
        # raises plain ParseError -- so a layout-B file with one blank data cell would
        # report as "not a supported format" instead of naming the cell. Layout A still
        # converts inline; this is the worked template for moving it.
        metadata: Dict[str, str] = {}
        angle_table: Dict[int, str] = {}
        sample_rows: Dict[int, str] = {}
        liquid_rows: Dict[int, str] = {}

        for key, value in rows:
            angle_match = _ANGLE_KEY_RE.match(key)
            if angle_match:
                angle_table[int(angle_match.group(1))] = value
                continue

            sample_match = _SAMPLE_INTENSITY_KEY_RE.match(key)
            if sample_match:
                sample_rows[int(sample_match.group(1))] = value
                continue

            liquid_match = _LIQUID_INTENSITY_KEY_RE.match(key)
            if liquid_match:
                liquid_rows[int(liquid_match.group(1))] = value
                continue

            if key not in metadata:            # first occurrence wins, as in layout A
                metadata[key] = value

        n_angles = _parse_count(
            _require_key(metadata, _KEY_N_ANGLES, file_path),
            _KEY_N_ANGLES, file_path,
        )

        # THE SNIFF -- one raise site, deliberately. Two independent discriminants, so
        # the two Brookhaven SLS parsers exclude each other by construction rather than
        # by registration order: layout B is the one with `Sample Intensity - Angle N`
        # rows, and layout A is the one that declares a concentration count. A foreign
        # key-value CSV that happens to carry `Number of Angles Measured` fails the
        # first test and is handed on to the generic fallback.
        if not sample_rows or _KEY_N_CONCENTRATIONS in metadata:
            raise ParseError(
                f"{file_path!r} is not a Brookhaven SLS angular scan: expected "
                f"'Sample Intensity - Angle N' rows and no {_KEY_N_CONCENTRATIONS!r} "
                f"row (found {len(sample_rows)} sample-intensity row(s); "
                f"concentration count "
                f"{'present' if _KEY_N_CONCENTRATIONS in metadata else 'absent'})."
            )

        # --- format commit point ---------------------------------------------------
        # The file has now declared an angle count AND delivered `Sample Intensity -
        # Angle N` rows, which nothing else in the corpus does. Everything above is the
        # sniff and raises plain ParseError so autodetect can hand a non-Brookhaven file
        # to the generic SLS fallback; everything below describes a file we have
        # committed to, so a problem is malformed DATA and the message is for the user.

        # The marker sits ABOVE the table checks on purpose: "Angle table is inconsistent
        # with the declared count" is a real, actionable diagnosis of a genuinely Brookhaven
        # file, and below the line it reaches the user instead of being flattened into
        # "unrecognised format". Committing here costs nothing on the load-breaking axis --
        # GenericSLSParser rejects any file where more than 5 % of rows fail a two-float
        # parse, and a Brookhaven key-value CSV fails ~100 % of them, so no file the
        # fallback could have loaded is skipped by an early `malformed`.
        _check_table_complete_committed(angle_table, n_angles, 'Angle', file_path)
        _check_table_complete_committed(
            sample_rows, n_angles, 'Sample Intensity', file_path)
        # The blank table is OPTIONAL: the instrument writes it only when a blank was
        # measured in the same run, and its plain export (every `raw/` file in the
        # committed corpus, the calibrant scan included) carries none. Requiring it
        # refused every unmodified export. A PARTIAL table is still a gap in a file we
        # have committed to, so once any row is present the whole table is checked.
        if liquid_rows:
            _check_table_complete_committed(
                liquid_rows, n_angles, 'Liquid Intensity', file_path)

        wavelength_nm = _parse_float_committed(
            _require_key_committed(metadata, _KEY_WAVELENGTH, file_path),
            _KEY_WAVELENGTH, file_path,
        )
        solvent_n = _parse_float_committed(
            _require_key_committed(metadata, _KEY_N_SOLVENT, file_path),
            _KEY_N_SOLVENT, file_path,
        )
        solvent_name = _require_key_committed(metadata, _KEY_SAMPLE_LIQUID, file_path)
        sample_label = metadata.get(_KEY_SAMPLE_ID)

        # No dn/dc row in this layout, and dn/dc is never proposed or defaulted
        # (architecture invariant 3) -- it stays None for the user to enter.

        # Calibration reference values: absent from every layout-B export seen so far,
        # read optionally in case a future software version emits them. Informational
        # either way; analysis uses the program-computed toluene value.
        cal_constant = _optional_float(metadata.get(_KEY_CAL_CONSTANT))
        standard_name = metadata.get(_KEY_CAL_LIQUID)
        standard_rayleigh = _optional_float(metadata.get(_KEY_RAYLEIGH_STANDARD))
        standard_n = _optional_float(metadata.get(_KEY_N_STANDARD))

        angles_deg = np.array(
            [_parse_float_committed(
                angle_table[i], f'Angle {i} (degrees)', file_path)
             for i in range(1, n_angles + 1)],
            dtype=float,
        )

        abs_path = os.path.abspath(file_path)

        def _build(table: Dict[int, str], row_label: str,
                   label: Optional[str], conc: Optional[float]) -> SLSFilePreview:
            intensities = np.array(
                [_parse_float_committed(
                    table[i], f'{row_label} - Angle {i}', file_path)
                 for i in range(1, n_angles + 1)],
                dtype=float,
            )
            return SLSFilePreview(
                source_file=abs_path,
                instrument_name=_INSTRUMENT_NAME,
                angles_deg=angles_deg.copy(),
                intensities=intensities,
                sample_label=label,
                solvent_name=solvent_name if solvent_name else None,
                concentration_g_per_mL=conc,
                wavelength_nm=wavelength_nm,
                solvent_refractive_index=solvent_n,
                dn_dc_mL_per_g=None,
                calibration_constant=cal_constant,
                standard_name=standard_name if standard_name else None,
                standard_rayleigh_ratio_file=standard_rayleigh,
                standard_refractive_index=standard_n,
                # polymer_name and temperature_K intentionally left None.
            )

        # Blank FIRST when there is one, matching layout A's concentration-ordered output
        # (whose concentration 1 is the c = 0 reference) so both layouts hand the loader
        # the same shape.
        previews: List[SLSFilePreview] = []
        if liquid_rows:
            blank_label = (sample_label + _BLANK_LABEL_SUFFIX) if sample_label else None
            previews.append(_build(liquid_rows, 'Liquid Intensity', blank_label, 0.0))
        previews.append(_build(sample_rows, 'Sample Intensity', sample_label or None, None))
        return previews
