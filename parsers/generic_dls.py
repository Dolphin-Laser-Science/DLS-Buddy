"""
parsers/generic_dls.py
======================

Parser for plain-text DLS correlogram files with no instrument-specific
header. Handles data from any correlator or simulation output that can be
expressed as a two-column numerical table.

    GenericDLSParser    two-column (delay_time, correlation_value) file

File format contract (strict)
-----------------------------
The file must satisfy ALL of the following:

  - Exactly two columns of numerical data.
  - No header rows of any kind (no labels, no comments, no blank lines).
  - Columns separated by a comma or a tab (detected automatically).
  - Every row must be parseable as two floating-point numbers.
  - At least two data rows.

Any deviation from these rules raises a ParseError with a clear message.
This strictness is intentional: the generic parser is not a fallback for
"I have some instrument file I don't understand." It is for data that has
already been cleaned and prepared. If a file has headers, the user must
remove them before loading.

Data form options
-----------------
The column-2 interpretation is user-specified at the confirmation step,
because a plain-text file cannot encode this information:

  'g2m1'   g²(τ) - 1, already baseline-subtracted. Stored as-is.
  'g2'     g²(τ), baseline NOT subtracted. User supplies baseline B.
            Stored as (G(τ) - B) / B.
  'g1'     Field autocorrelation g¹(τ). User supplies coherence factor β.
            Converted via Siegert: g²(τ)-1 = β · |g¹(τ)|².
            Stored as g²(τ)-1.

In all cases the stored correlogram is g²(τ)-1. Analysis code never sees
any other form.

Delay time units
----------------
User-specified. Accepted: 's', 'ms', 'us' / 'µs', 'ns'. Converted to
seconds (canonical internal unit) on load.

What the user must supply at the confirmation step
--------------------------------------------------
Everything, because the file contains no metadata:
  - polymer_name
  - solvent_name
  - concentration_g_per_mL  (and its unit)
  - temperature_K            (and its unit, C or K)
  - angle_deg
  - wavelength_nm
  - solvent_refractive_index
  - viscosity_Pa_s           (optional; needed only for Stokes-Einstein)
  - delay_time_unit          ('s', 'ms', 'us', 'ns')
  - data_form                ('g2m1', 'g2', 'g1')
  - baseline_B               (required only if data_form == 'g2')
  - beta                     (required only if data_form == 'g1')

Change history
--------------
2026-06-12  Initial implementation. (generic_dls.py v1)
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from core.data_models import DLSMeasurement
from parsers.base_parser import (
    BaseDLSParser,
    DLSFilePreview,
    ParseError,
    convert_delay_times,
)

# Accepted data form strings
_VALID_DATA_FORMS = ('g2m1', 'g2', 'g1')

# Minimum number of data rows to accept
_MIN_ROWS = 2

# Maximum fraction of rows allowed to be malformed before the parser raises
# rather than silently discarding. If more than this fraction fails to parse,
# the file almost certainly has headers or is the wrong format.
_MAX_BAD_ROW_FRACTION = 0.05


def _detect_delimiter(first_line: str) -> str:
    """Return ',' or '\t' based on the first data line, or raise ParseError."""
    if '\t' in first_line and ',' not in first_line:
        return '\t'
    if ',' in first_line and '\t' not in first_line:
        return ','
    if ',' in first_line and '\t' in first_line:
        # Both present — comma-separated files sometimes contain tabs in
        # labels. Count occurrences and pick the majority.
        if first_line.count(',') >= first_line.count('\t'):
            return ','
        return '\t'
    raise ParseError(
        f"Cannot detect delimiter in first data line: {first_line!r}. "
        f"The file must be comma- or tab-delimited."
    )


def _apply_data_form(
    raw_col2: np.ndarray,
    data_form: str,
    baseline_B: Optional[float],
    beta: Optional[float],
    file_path: str,
) -> np.ndarray:
    """Convert raw column-2 values to g²(τ)-1 according to data_form.

    Parameters
    ----------
    raw_col2 : np.ndarray
        Values as read from the file.
    data_form : str
        One of 'g2m1', 'g2', 'g1'.
    baseline_B : float or None
        Required if data_form == 'g2'. The baseline value B such that
        g²(τ)-1 = (G(τ) - B) / B.
    beta : float or None
        Required if data_form == 'g1'. The coherence factor β such that
        g²(τ)-1 = β · |g¹(τ)|².
    file_path : str
        Used only in error messages.

    Returns
    -------
    np.ndarray
        g²(τ)-1 values.

    Raises
    ------
    ParseError
        If a required parameter (B or β) is missing or physically invalid.
    """
    if data_form == 'g2m1':
        return raw_col2.copy()

    if data_form == 'g2':
        if baseline_B is None:
            raise ParseError(
                f"data_form='g2' requires baseline_B to be supplied, "
                f"but it is None. Set preview.baseline_B before calling "
                f"preview.build() for {file_path!r}."
            )
        if not (baseline_B > 0):
            raise ParseError(
                f"baseline_B must be a positive number, got {baseline_B!r}."
            )
        return (raw_col2 - baseline_B) / baseline_B

    if data_form == 'g1':
        if beta is None:
            raise ParseError(
                f"data_form='g1' requires beta (coherence factor) to be "
                f"supplied, but it is None. Set preview.beta before calling "
                f"preview.build() for {file_path!r}."
            )
        if not (0 < beta <= 1.0):
            raise ParseError(
                f"beta must be in the range (0, 1], got {beta!r}. "
                f"The coherence factor is an instrument property, typically "
                f"between 0.01 and 1.0."
            )
        # Siegert relation: g²(τ)-1 = β · |g¹(τ)|²
        return beta * np.abs(raw_col2) ** 2

    raise ParseError(
        f"Unknown data_form {data_form!r}. "
        f"Accepted values: {_VALID_DATA_FORMS}."
    )


class GenericDLSFilePreview(DLSFilePreview):
    """DLSFilePreview extended with generic-parser-specific parameters.

    Adds three fields that the generic parser needs at build() time but
    that are meaningless for instrument-specific parsers (which handle all
    data-form conversion internally before constructing the preview).

    Extra fields
    ------------
    delay_time_unit : str or None
        The unit of the delay times as read from the file.
        One of 's', 'ms', 'us', 'µs', 'ns'.
        Set by the parser from the file (if detectable) or by the user.
    data_form : str or None
        How to interpret column 2. One of 'g2m1', 'g2', 'g1'.
        Must be set by the user at the confirmation step.
    baseline_B : float or None
        Baseline value B for data_form == 'g2'. Ignored otherwise.
    beta : float or None
        Coherence factor β for data_form == 'g1'. Ignored otherwise.

    These fields are also reflected in missing_required_fields() and
    is_ready(), which override the parent class versions.
    """

    def __init__(self, **kwargs):
        # Pull out generic-parser fields before passing the rest to the parent.
        self.delay_time_unit: Optional[str] = kwargs.pop('delay_time_unit', None)
        self.data_form: Optional[str] = kwargs.pop('data_form', None)
        self.baseline_B: Optional[float] = kwargs.pop('baseline_B', None)
        self.beta: Optional[float] = kwargs.pop('beta', None)
        # _raw_col2 holds the file data before conversion; the parent's
        # delay_times_s and correlogram are set in build() after the user
        # has confirmed the unit and data form.
        self._raw_delay_times: Optional[np.ndarray] = kwargs.pop('_raw_delay_times', None)
        self._raw_col2: Optional[np.ndarray] = kwargs.pop('_raw_col2', None)
        # Initialize the parent dataclass (uses object.__setattr__ because
        # dataclasses with eq=False can be tricky with __init__ overrides).
        super().__init__(**kwargs)
        # Overwrite the parent's data arrays with None; they will be filled
        # in build() after conversion.
        self.delay_times_s = None
        self.correlogram = None

    def missing_required_fields(self) -> List[str]:
        missing = []
        # Raw data (internal)
        if self._raw_delay_times is None:
            missing.append('_raw_delay_times (file not loaded)')
        if self._raw_col2 is None:
            missing.append('_raw_col2 (file not loaded)')
        # Generic parameters (user-supplied at confirmation)
        if self.delay_time_unit is None:
            missing.append('delay_time_unit')
        if self.data_form is None:
            missing.append('data_form')
        if self.data_form == 'g2' and self.baseline_B is None:
            missing.append('baseline_B')
        if self.data_form == 'g1' and self.beta is None:
            missing.append('beta')
        # Identity and optics (same as parent)
        for field, value in [
            ('polymer_name', self.polymer_name),
            ('solvent_name', self.solvent_name),
            ('concentration_g_per_mL', self.concentration_g_per_mL),
            ('temperature_K', self.temperature_K),
            ('angle_deg', self.angle_deg),
            ('wavelength_nm', self.wavelength_nm),
            ('solvent_refractive_index', self.solvent_refractive_index),
        ]:
            if value is None:
                missing.append(field)
        return missing

    def is_ready(self) -> bool:
        return len(self.missing_required_fields()) == 0

    def apply_data_conversion(self) -> None:
        """Populate delay_times_s + correlogram from the raw columns using the
        confirmed delay-time unit and data form, WITHOUT requiring the identity
        fields (polymer, solvent, ...). This lets a loader convert the arrays at
        load time and then hand the measurement to the standard confirmation flow,
        where the identity is filled in. Raises if the unit / data form is unset."""
        from parsers.base_parser import ParseError as _PE
        if self.delay_time_unit is None or self.data_form is None:
            raise _PE('delay_time_unit and data_form must be set before conversion.')
        if self.data_form == 'g2' and self.baseline_B is None:
            raise _PE("data_form 'g2' requires baseline_B.")
        if self.data_form == 'g1' and self.beta is None:
            raise _PE("data_form 'g1' requires beta.")
        self.delay_times_s = convert_delay_times(self._raw_delay_times, self.delay_time_unit)
        self.correlogram = _apply_data_form(self._raw_col2, self.data_form,
                                            self.baseline_B, self.beta,
                                            self.source_file or '<unknown>')

    def build(self) -> DLSMeasurement:
        """Apply unit conversion and data-form conversion, then build.

        This override converts the raw file data (delay times and column-2
        values) using the user-confirmed unit and data_form, populates
        the parent's delay_times_s and correlogram fields, and then calls
        the standard DLSMeasurement constructor via build().
        """
        missing = self.missing_required_fields()
        if missing:
            from parsers.base_parser import ParseError as _PE
            raise _PE(
                f"Cannot build DLSMeasurement: required fields not yet "
                f"supplied: {missing}.",
                missing_fields=missing,
            )

        # Apply delay time unit conversion
        self.delay_times_s = convert_delay_times(
            self._raw_delay_times, self.delay_time_unit
        )

        # Apply data-form conversion to get g²(τ)-1
        self.correlogram = _apply_data_form(
            self._raw_col2,
            self.data_form,
            self.baseline_B,
            self.beta,
            self.source_file or '<unknown>',
        )

        # Delegate to the standard DLSMeasurement constructor
        return super().build()

    def __repr__(self) -> str:
        n_raw = len(self._raw_delay_times) if self._raw_delay_times is not None else 0
        return (
            f"GenericDLSFilePreview(source={self.source_file!r}, "
            f"n_points={n_raw}, "
            f"data_form={self.data_form!r}, "
            f"ready={self.is_ready()})"
        )


class GenericDLSParser(BaseDLSParser):
    """Parser for plain-text two-column DLS correlogram files.

    Reads the file, validates the two-column no-header contract, and stores
    the raw values in a GenericDLSFilePreview. Unit conversion and data-form
    conversion (g1/g2/g2m1) are deferred to preview.build(), after the user
    has confirmed the relevant parameters at the confirmation step.

    Usage
    -----
    ::

        parser = GenericDLSParser()
        previews = parser.parse('path/to/correlogram.txt')
        preview = previews[0]   # always one element

        # At the confirmation step:
        preview.delay_time_unit = 'us'
        preview.data_form = 'g2m1'
        preview.polymer_name = 'PVP'
        preview.solvent_name = 'water'
        preview.concentration_g_per_mL = parser.convert_concentration(0.5, 'mg/mL')
        preview.temperature_K = parser.convert_temperature(25.0, 'C')
        preview.angle_deg = 90.0
        preview.wavelength_nm = 532.0
        preview.solvent_refractive_index = 1.33
        preview.viscosity_Pa_s = parser.convert_viscosity(0.891, 'mPa.s')

        measurement = preview.build()

    For g1 data, also set preview.beta before calling build().
    For unsubtracted g2 data, also set preview.baseline_B.
    """

    def parse(self, file_path: str) -> List[GenericDLSFilePreview]:
        """Parse a plain-text two-column DLS file.

        Parameters
        ----------
        file_path : str
            Path to the file.

        Returns
        -------
        List[GenericDLSFilePreview]
            A one-element list. The preview has raw delay times and
            column-2 values stored internally; all other fields are None.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ParseError
            If the file violates the two-column no-header contract, or
            more than 5% of rows are malformed.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path!r}")

        try:
            with open(file_path, encoding='utf-8', newline='') as fh:
                raw_lines = [line.rstrip('\r\n') for line in fh]
        except UnicodeDecodeError:
            # Fall back to latin-1 for files with non-UTF-8 characters
            with open(file_path, encoding='latin-1', newline='') as fh:
                raw_lines = [line.rstrip('\r\n') for line in fh]

        # Remove blank lines
        lines = [l for l in raw_lines if l.strip()]

        if len(lines) < _MIN_ROWS:
            raise ParseError(
                f"{file_path!r} contains only {len(lines)} non-blank line(s); "
                f"at least {_MIN_ROWS} data rows are required."
            )

        delimiter = _detect_delimiter(lines[0])

        col1: List[float] = []
        col2: List[float] = []
        n_bad = 0

        for line in lines:
            parts = line.split(delimiter)
            if len(parts) != 2:
                n_bad += 1
                continue
            try:
                col1.append(float(parts[0].strip()))
                col2.append(float(parts[1].strip()))
            except ValueError:
                n_bad += 1
                continue

        total = len(lines)
        if n_bad / total > _MAX_BAD_ROW_FRACTION:
            raise ParseError(
                f"{n_bad} of {total} rows in {file_path!r} could not be "
                f"parsed as two numbers ({100*n_bad/total:.0f}%). "
                f"The generic parser requires a strict two-column, no-header "
                f"format. If this file has header rows or comment lines, "
                f"remove them before loading."
            )

        if len(col1) < _MIN_ROWS:
            raise ParseError(
                f"Only {len(col1)} valid data rows found in {file_path!r} "
                f"after skipping {n_bad} malformed rows. "
                f"At least {_MIN_ROWS} are required."
            )

        preview = GenericDLSFilePreview(
            source_file=os.path.abspath(file_path),
            instrument_name='Generic (plain-text)',
            _raw_delay_times=np.array(col1, dtype=float),
            _raw_col2=np.array(col2, dtype=float),
        )
        return [preview]
