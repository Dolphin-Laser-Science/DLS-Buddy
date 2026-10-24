"""
app/calibration.py
==================

The session-level SLS calibration, `CalibrationState`: the one calibrant point, its
standard and geometry, and the calibration constant k_c computed from them, with the
solvent-library lookup that seeds the standard's refractive index and the shipped prefill
it seeds. The controller imports it and holds a committed and a working copy for the
session, like a measurement's parameters, and one pair per sample that overrides the
session's. Qt-free (invariant 5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Dict, Optional

from core.notes import Note, render
from physics import solvents as solvents_lib
from analysis import sls as sls_engine
from physics import constants as phys


# ===========================================================================
# Calibration state (committed / working, like a measurement's parameters)
# ===========================================================================

def _library_standard_index(wavelength_nm, temperature_C) -> Optional[float]:
    """Toluene's refractive index at (wavelength, temperature) from the solvent library,
    or None where the library has no value (outside its validity box, or a field that
    is not a number yet). The standard is toluene throughout (`compute_k_c` always calls
    `rayleigh_ratio_toluene`), so the name is not a parameter."""
    try:
        n = float(solvents_lib.refractive_index_solvent(
            'toluene', float(wavelength_nm), float(temperature_C)))
    except (TypeError, ValueError):
        return None
    # Rounded at the source, not at the display: the panel's line edit shows str(value)
    # and re-parses it on every focus-out, so a value that does not round-trip through
    # its own text would stop equalling the library's and read as user-typed. Five
    # decimals sit two orders inside the library's own band (+/- 1.2e-3 for toluene).
    return round(n, 5) if math.isfinite(n) else None


# The shipped prefill for the standard's refractive index: the library's toluene value at
# the shipped standard wavelength / temperature, never a typed literal (invariant 3 -- an
# offered physical value is science-based and provenance-tagged; the tooltip names the
# library). It was a hard-coded 1.496 -- toluene's sodium-D value, 2.9 sigma from the
# library's 1.4996 at 532 nm -- biasing every calibrated Mw by +0.48 %.
_DEFAULT_STANDARD_WAVELENGTH_NM: float = 532.0
_DEFAULT_STANDARD_TEMPERATURE_C: float = 25.0
_DEFAULT_STANDARD_REFRACTIVE_INDEX: Optional[float] = _library_standard_index(
    _DEFAULT_STANDARD_WAVELENGTH_NM, _DEFAULT_STANDARD_TEMPERATURE_C)


@dataclass
class CalibrationState:
    """The session-level SLS calibration.

    The user supplies one calibrant point (intensity, angle, standard liquid +
    geometry); the controller computes k_c. A None k_c means uncalibrated -- the
    analysis still runs, flagged. Mirrors the unified model in sls.py.

    The standard's refractive index is PROPOSED from the solvent library at the
    standard's own wavelength and temperature and re-proposed when either changes
    (`Controller.set_calibration_field`), until the user types a value of their own;
    a blank leaves the (n_solvent/n_standard)^2 step unapplied and the run's `calibrated`
    down, its note `no_standard_index` once a k_c is applied
    (`analysis.sls.compute_excess_rayleigh_ratio`). WHICH of the two it
    is rides the `standard_refractive_index_source` sidecar -- 'library' (proposed,
    tracks lambda / T, shown with the provenance dot) or 'user' (typed, never
    overwritten) -- the `autofill_solvent_props` pattern the Data tab uses for the
    sample's own n (owner-ruled 2026-08-27). A stored tag, never an
    inference from the value: inferring "still the library's" from value equality
    re-proposed away a value the user had typed because it happened to equal the
    library's, and froze tracking for good once the library had no value at a
    condition. Clearing the field returns it to the library (the only way back).
    Every note a calibration owes the user -- temperature or wavelength
    extrapolation of the toluene table, a blank or off-library standard index -- is
    composed by `Controller.preview_calibration_note` from the `*_note` methods here.
    """
    calibrant_intensity: Optional[float] = None
    calibrant_angle_deg: float = 90.0
    standard_name: str = 'toluene'
    standard_geometry: str = 'VU'
    standard_wavelength_nm: float = _DEFAULT_STANDARD_WAVELENGTH_NM
    standard_temperature_C: float = _DEFAULT_STANDARD_TEMPERATURE_C
    standard_refractive_index: Optional[float] = _DEFAULT_STANDARD_REFRACTIVE_INDEX
    standard_refractive_index_source: str = 'library'    # 'library' | 'user'
    dark_count_rate: float = 0.0
    depolarization_ratio_v: Optional[float] = None
    k_c: Optional[float] = None                          # computed; editable

    def library_standard_index(self) -> Optional[float]:
        """The library's toluene n at THIS standard's wavelength / temperature, or None."""
        return _library_standard_index(self.standard_wavelength_nm,
                                       self.standard_temperature_C)

    def compute_k_c(self) -> Optional[float]:
        """Compute k_c from the calibrant point, or None if not enough info."""
        if self.calibrant_intensity is None:
            return None
        r_std = phys.rayleigh_ratio_toluene(
            self.standard_wavelength_nm, self.standard_temperature_C,
            geometry=self.standard_geometry,
            depolarization_ratio_v=self.depolarization_ratio_v)
        return sls_engine.compute_calibration_constant(
            self.calibrant_intensity, self.calibrant_angle_deg, r_std,
            dark_count_rate=self.dark_count_rate)

    def temperature_extrapolation_note(self) -> Optional[str]:
        """The toluene Rayleigh-ratio extrapolation note for this standard temperature,
        or None if it is inside the S&K 10-50 C table. Keys purely off the standard
        temperature (the standard is toluene throughout, and compute_k_c always calls
        rayleigh_ratio_toluene), so it surfaces even before a calibrant intensity is
        entered. The GUI renders it next to k_c."""
        return phys.toluene_temperature_extrapolation_reason(self.standard_temperature_C)

    def wavelength_extrapolation_note(self) -> Optional[str]:
        """The wavelength sibling of `temperature_extrapolation_note` (rule 10): the
        toluene table's nearest entry was served unscaled for a wavelength within the
        match tolerance but not on the table. None on a tabulated wavelength, and None
        beyond the tolerance (there `compute_k_c` raises, and the k_c line says so)."""
        try:
            return phys.toluene_wavelength_extrapolation_reason(
                float(self.standard_wavelength_nm))
        except (TypeError, ValueError):
            return None

    def standard_index_note(self) -> Optional[str]:
        """The disclosure the standard's refractive index owes, or None.

        Two cases, both invariant-3 shaped (the library PROPOSES, the user commits, and
        the program says what it will do with what was committed):
          * BLANK -- the mandatory (n_solvent/n_standard)^2 step cannot run; the note
            says so before the run, beside k_c, because the k_c line alone would show a
            healthy constant. It names the cause the fit will (`analysis.sls.scale_notes`),
            in the registry's words: `no_standard_index` when this calibration yields a
            k_c, `uncalibrated` when it yields none, since the index corrects a k_c.
          * OFF-LIBRARY -- the committed value sits outside the library's own
            uncertainty band for toluene at this wavelength / temperature; a user's
            measured value is theirs to keep, and this only names the difference and
            its size on Mw (the correction goes as n_standard^-2, so ~2 dn/n).
        None when the library has no value here, or the value agrees with it.
        """
        lam, t_c = self.standard_wavelength_nm, self.standard_temperature_C
        lib = self.library_standard_index()
        if self.standard_refractive_index is None:
            hint = f" (solvent library: toluene {lib:.4f})" if lib is not None else ""
            # Whether the run will receive a k_c is the commit's rule
            # (`Controller._commit_calibration`): `compute_k_c`, a raise reading as none.
            try:
                k_c = self.compute_k_c()
            except (TypeError, ValueError):
                k_c = None
            if k_c is None:
                lead = ("Standard refractive index is blank, and there is no k_c for it to "
                        "correct, so")
                cause = Note('uncalibrated')
            else:
                lead = ("Standard refractive index is blank: the (n_solvent/n_standard)^2 "
                        "correction cannot be applied, so")
                cause = Note('no_standard_index')
            return (
                f"{lead} the results will be flagged "
                f"({render(cause, surface='export')}) -- Mw and absolute A2 "
                f"unreliable; Rg and the calibration-free 2 A2 Mw unaffected. Enter the "
                f"standard's index at {lam:g} nm / {t_c:g} C{hint}.")
        if lib is None:
            return None
        try:
            sigma = float(solvents_lib.solvent_uncertainty_n('toluene', float(lam), float(t_c)))
            n_std = float(self.standard_refractive_index)
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(n_std) and math.isfinite(sigma)):
            return None
        if abs(n_std - lib) <= sigma:
            return None
        return (
            f"Standard refractive index {n_std:g} differs from the solvent library's "
            f"toluene value {lib:.4f} at {lam:g} nm / {t_c:g} C (library band "
            f"+/- {sigma:.1e}); the (n_solvent/n_standard)^2 correction, and with it "
            f"Mw and absolute A2, moves by about {200.0 * abs(n_std - lib) / lib:.2f} %.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            'calibrant_intensity': self.calibrant_intensity,
            'calibrant_angle_deg': self.calibrant_angle_deg,
            'standard_name': self.standard_name,
            'standard_geometry': self.standard_geometry,
            'standard_wavelength_nm': self.standard_wavelength_nm,
            'standard_temperature_C': self.standard_temperature_C,
            'standard_refractive_index': self.standard_refractive_index,
            'standard_refractive_index_source': self.standard_refractive_index_source,
            'dark_count_rate': self.dark_count_rate,
            'depolarization_ratio_v': self.depolarization_ratio_v,
            'k_c': self.k_c,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CalibrationState':
        # Forward-compatible: drop unknown keys so a session written by a newer build
        # (extra calibration field) still loads in an older one.
        if not d:
            return cls()
        known = {f.name for f in fields(cls)}
        state = cls(**{k: v for k, v in d.items() if k in known})
        if 'standard_refractive_index_source' not in d:
            # A session written before the sidecar existed: tag it ONCE, at load, by
            # the value-equality inference the live edit path used to make on every
            # edit -- the library's value at the stored condition reads as proposed,
            # anything else (a typed value, a blank) as the user's. Conservative in
            # the direction that matters: a value this cannot place is never
            # re-proposed away.
            lib = state.library_standard_index()
            n_std = state.standard_refractive_index
            try:
                # Guarded like `_library_standard_index`'s own float() calls: a
                # hand-edited or corrupted session can hold a non-numeric value here,
                # and before the tag existed `from_dict` never converted it. Raising
                # would surface through `load_session` as the version-mismatch message,
                # which names the wrong cause. A value that is not a number is not the
                # library's, so it degrades to 'user' -- the branch that never
                # overwrites it.
                proposed = (lib is not None and n_std is not None
                            and math.isclose(float(n_std), lib, rel_tol=1e-9))
            except (TypeError, ValueError):
                proposed = False
            state.standard_refractive_index_source = 'library' if proposed else 'user'
        return state
