"""
parsers/generic_sls.py
======================

Parser for plain-text SLS intensity files with no instrument-specific
header. Handles output from any instrument or simulation that can produce
a simple angle-vs-intensity table.

    GenericSLSParser    two-column (angle_deg, intensity) file

File format contract (strict)
-----------------------------
The file must satisfy ALL of the following:

  - Exactly two columns of numerical data.
  - No header rows of any kind (no labels, no comments, no blank lines).
  - Columns separated by a comma or a tab (detected automatically).
  - Column 1: scattering angles in degrees (each strictly between 0 and 180).
  - Column 2: raw intensity values (positive numbers).
  - Every row must be parseable as two floating-point numbers.
  - At least two data rows.

Any deviation raises a ParseError with a clear message.

Relationship to the Brookhaven SLS parser
------------------------------------------
The Brookhaven SLS parser handles multi-concentration files natively.
This generic parser handles ONE concentration per file. To build a
Zimm-plot dataset from generic files, the user loads one file per
concentration (including the solvent reference at c=0) and the program
assembles the collection. This is consistent with the generic DLS parser's
contract: one set of conditions per file.

What the user must supply at the confirmation step
--------------------------------------------------
Everything not in the file:
  - polymer_name
  - solvent_name
  - concentration_g_per_mL  (and its unit: 'g/mL' or 'mg/mL')
  - temperature_K            (and its unit: 'C' or 'K')
  - wavelength_nm
  - solvent_refractive_index
  - dn_dc_mL_per_g

Change history
--------------
2026-06-12  Initial implementation. (generic_sls.py v1)
"""

from __future__ import annotations

import os
from typing import List

import numpy as np

from parsers.base_parser import (
    BaseSLSParser,
    ParseError,
    SLSFilePreview,
)

# Minimum number of angle/intensity data rows to accept.
# One point is physically useless for any SLS analysis.
_MIN_ROWS = 2

# Maximum fraction of rows allowed to be malformed before raising.
_MAX_BAD_ROW_FRACTION = 0.05


def _detect_delimiter(first_line: str) -> str:
    """Return ',' or '\t' from the first data line, or raise ParseError."""
    if '\t' in first_line and ',' not in first_line:
        return '\t'
    if ',' in first_line and '\t' not in first_line:
        return ','
    if ',' in first_line and '\t' in first_line:
        if first_line.count(',') >= first_line.count('\t'):
            return ','
        return '\t'
    raise ParseError(
        f"Cannot detect delimiter in first data line: {first_line!r}. "
        f"The file must be comma- or tab-delimited."
    )


class GenericSLSParser(BaseSLSParser):
    """Parser for plain-text two-column SLS intensity files.

    Each file represents one concentration at multiple angles. The parser
    reads angle (degrees) and intensity values and returns a single
    SLSFilePreview with all physical parameters left for the user to
    supply at the confirmation step.

    Usage
    -----
    ::

        parser = GenericSLSParser()

        # Load solvent reference (c = 0)
        solvent_previews = parser.parse('solvent.txt')
        p = solvent_previews[0]
        p.polymer_name = 'PS'
        p.solvent_name = 'toluene'
        p.concentration_g_per_mL = 0.0
        p.temperature_K = parser.convert_temperature(25.0, 'C')
        p.wavelength_nm = 532.0
        p.solvent_refractive_index = 1.502
        p.dn_dc_mL_per_g = 0.11
        solvent = p.build()

        # Load a polymer solution (c = 0.3638 mg/mL)
        soln_previews = parser.parse('solution_c1.txt')
        p2 = soln_previews[0]
        p2.polymer_name = 'PS'
        p2.solvent_name = 'toluene'
        p2.concentration_g_per_mL = parser.convert_concentration(0.3638, 'mg/mL')
        p2.temperature_K = parser.convert_temperature(25.0, 'C')
        p2.wavelength_nm = 532.0
        p2.solvent_refractive_index = 1.502
        p2.dn_dc_mL_per_g = 0.11
        solution = p2.build()

        # The two SLSMeasurement objects can then be combined into a
        # Zimm/Berry/Debye dataset in the SLS analysis module.
    """

    def parse(self, file_path: str) -> List[SLSFilePreview]:
        """Parse a plain-text two-column SLS angle/intensity file.

        Parameters
        ----------
        file_path : str
            Path to the file.

        Returns
        -------
        List[SLSFilePreview]
            A one-element list. angles_deg and intensities are populated;
            all physical parameters are None for user confirmation.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ParseError
            If the file violates the two-column no-header contract, if
            more than 5% of rows are malformed, if any angle is outside
            (0, 180), or if any intensity is negative.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path!r}")

        # Try UTF-8 first; fall back to latin-1 for instruments that write
        # non-ASCII degree symbols or other characters in their output.
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

        angles: List[float] = []
        intensities: List[float] = []
        n_bad = 0

        for line in lines:
            parts = line.split(delimiter)
            if len(parts) != 2:
                n_bad += 1
                continue
            try:
                angle = float(parts[0].strip())
                intensity = float(parts[1].strip())
            except ValueError:
                n_bad += 1
                continue
            angles.append(angle)
            intensities.append(intensity)

        total = len(lines)
        if n_bad / total > _MAX_BAD_ROW_FRACTION:
            raise ParseError(
                f"{n_bad} of {total} rows in {file_path!r} could not be "
                f"parsed as two numbers ({100*n_bad/total:.0f}%). "
                f"The generic SLS parser requires a strict two-column, "
                f"no-header format. If this file has header rows or comment "
                f"lines, remove them before loading."
            )

        if len(angles) < _MIN_ROWS:
            raise ParseError(
                f"Only {len(angles)} valid data rows found in {file_path!r} "
                f"after skipping {n_bad} malformed rows. "
                f"At least {_MIN_ROWS} are required."
            )

        # Validate angle values before storing. Bad angles here mean the
        # columns are probably swapped or the file is the wrong type.
        bad_angles = [a for a in angles if not (0 < a < 180)]
        if bad_angles:
            raise ParseError(
                f"{len(bad_angles)} angle value(s) in {file_path!r} are "
                f"outside the valid range (0, 180) degrees: "
                f"{bad_angles[:5]}{'...' if len(bad_angles) > 5 else ''}. "
                f"Check that column 1 contains scattering angles and column 2 "
                f"contains intensities (not the other way around)."
            )

        # Warn on negative intensities rather than raising, because some
        # instruments report small negative values after background subtraction.
        # Surface the same message both ways (keep-both): a passive note carried on
        # the preview (-> a load-time ⓘ in the GUI) and a UserWarning (stderr/headless).
        notes: tuple = ()
        negative_intensities = [v for v in intensities if v < 0]
        if negative_intensities:
            import warnings
            note = (
                f"{len(negative_intensities)} negative intensity value(s) found "
                "(can occur after background subtraction). Stored as-is; verify the "
                "data is physically reasonable before proceeding with analysis."
            )
            notes = (note,)
            warnings.warn(f"{file_path!r}: {note}", UserWarning, stacklevel=2)

        preview = SLSFilePreview(
            source_file=os.path.abspath(file_path),
            instrument_name='Generic (plain-text)',
            angles_deg=np.array(angles, dtype=float),
            intensities=np.array(intensities, dtype=float),
            notes=notes,
            # All physical parameters intentionally left as None.
        )
        return [preview]
