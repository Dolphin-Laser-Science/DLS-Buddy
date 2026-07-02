"""
parsers/brookhaven_sls.py
=========================

Parser for Brookhaven Particle Explorer SLS intensity exports:

    BrookhavenSLSParser     reads the intensity-vs-angle .CSV file

Inherits from BaseSLSParser and produces one SLSFilePreview per concentration.
A Zimm-plot file with seven concentrations yields seven previews; a single-
concentration angular scan (Guinier) yields one.

File format notes (Brookhaven Particle Explorer SLS, confirmed from real export)
--------------------------------------------------------------------------------

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
     The N x M grid is serialised concentration-major (all angles of
     concentration 1, then all angles of concentration 2, ...).

What the parser extracts vs. what the user must supply
------------------------------------------------------
EXTRACTED from the file (populated in each preview):
    - angles_deg              from the Angle table
    - intensities             from the Intensity grid (per concentration)
    - concentration_g_per_mL  from the Concentration table (mg/mL -> g/mL)
    - solvent_name            from "Sample Liquid"
    - solvent_refractive_index from "Refractive Index of Sample Liquid"
    - dn_dc_mL_per_g          from "Refractive Index Inc. (dn/dc)"
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
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Tuple

import numpy as np

from parsers.base_parser import (
    BaseSLSParser,
    ParseError,
    SLSFilePreview,
)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_ENCODING = 'latin-1'
_DELIMITER = ','

_INSTRUMENT_NAME = 'Brookhaven Particle Explorer'

# Exact metadata keys as they appear in the file. Centralising them here means
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

_MG_PER_ML_TO_G_PER_ML = 1.0e-3


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
        If the file cannot be read with the expected encoding.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path!r}")
    try:
        with open(file_path, encoding=_ENCODING, newline='') as fh:
            raw_lines = [line.rstrip('\r\n') for line in fh]
    except (UnicodeDecodeError, OSError) as exc:
        raise ParseError(
            f"Could not read {file_path!r} with encoding {_ENCODING!r}: {exc}"
        ) from exc

    rows: List[Tuple[str, str]] = []
    for line in raw_lines:
        if _DELIMITER not in line:
            continue
        key, value = line.split(_DELIMITER, 1)   # split on first comma only
        rows.append((key.strip(), value.strip()))
    return rows


def _require_key(metadata: Dict[str, str], key: str, file_path: str) -> str:
    """Return metadata[key], or raise a clear ParseError if it is missing."""
    if key not in metadata:
        raise ParseError(
            f"Required metadata key {key!r} not found in {file_path!r}. "
            f"This file may not be a Brookhaven SLS intensity export, or the "
            f"export format has changed."
        )
    return metadata[key]


def _parse_float(value: str, key: str, file_path: str) -> float:
    """Parse a string to float, or raise a ParseError naming the offending key."""
    try:
        return float(value)
    except ValueError as exc:
        raise ParseError(
            f"Could not parse value {value!r} for key {key!r} in {file_path!r} "
            f"as a number."
        ) from exc


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
        n_angles = int(_parse_float(
            _require_key(metadata, _KEY_N_ANGLES, file_path),
            _KEY_N_ANGLES, file_path,
        ))
        n_conc = int(_parse_float(
            _require_key(metadata, _KEY_N_CONCENTRATIONS, file_path),
            _KEY_N_CONCENTRATIONS, file_path,
        ))

        # The angle and concentration tables must contain exactly the declared
        # number of entries, indexed 1..N. A mismatch means the file is
        # truncated or malformed.
        self._check_table_complete(angle_table, n_angles, 'Angle', file_path)
        self._check_table_complete(conc_table, n_conc, 'Concentration', file_path)

        # --- build the shared optics / calibration values (once) ---
        wavelength_nm = _parse_float(
            _require_key(metadata, _KEY_WAVELENGTH, file_path),
            _KEY_WAVELENGTH, file_path,
        )
        solvent_n = _parse_float(
            _require_key(metadata, _KEY_N_SOLVENT, file_path),
            _KEY_N_SOLVENT, file_path,
        )
        dn_dc = _parse_float(
            _require_key(metadata, _KEY_DN_DC, file_path),
            _KEY_DN_DC, file_path,
        )
        solvent_name = _require_key(metadata, _KEY_SAMPLE_LIQUID, file_path)
        sample_label = metadata.get(_KEY_SAMPLE_ID)

        # Calibration reference values (informational only; never used in
        # analysis, which uses the program-computed toluene value — Takahashi
        # 2019 at 532 nm / Sivokhin & Kazantsev 2021 at 660 nm).
        cal_constant = self._optional_float(metadata.get(_KEY_CAL_CONSTANT))
        standard_name = metadata.get(_KEY_CAL_LIQUID)
        standard_rayleigh = self._optional_float(metadata.get(_KEY_RAYLEIGH_STANDARD))
        standard_n = self._optional_float(metadata.get(_KEY_N_STANDARD))

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
                    raise ParseError(
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

    # --- small helpers used only by this parser ---

    @staticmethod
    def _check_table_complete(
        table: Dict[int, float], expected_n: int, label: str, file_path: str
    ) -> None:
        """Verify a numbered table has exactly entries 1..expected_n."""
        expected = set(range(1, expected_n + 1))
        found = set(table.keys())
        if found != expected:
            missing = sorted(expected - found)
            extra = sorted(found - expected)
            raise ParseError(
                f"{label} table in {file_path!r} is inconsistent with the "
                f"declared count ({expected_n}). "
                f"Missing indices: {missing or 'none'}; "
                f"unexpected indices: {extra or 'none'}."
            )

    @staticmethod
    def _optional_float(value):
        """Parse an optional metadata value to float, returning None if absent
        or unparseable. Used for informational calibration fields that should
        not abort the parse if malformed."""
        if value is None:
            return None
        try:
            return float(value)
        except ValueError:
            return None
