"""
app/units.py
============

Unit conversions for the GUI input/display boundary.

The data model and analysis engine ALWAYS work in canonical/SI units
(concentration g/mL, temperature K, viscosity Pa·s, delay time s, count rate
cps, ...). These helpers convert only at the widget edge, so a user can type or
read a quantity in a convenient human-scale unit while nothing below the widgets
ever sees anything but the canonical value. Every factor lives here once, so a
conversion is defined (and tested) in a single place — no scale constant is ever
duplicated in a widget.

Usage::

    canonical = to_canonical('viscosity', 0.89, 'mPa·s')   # -> 8.9e-4 (Pa·s)
    shown     = from_canonical('viscosity', 8.9e-4, 'mPa·s')  # -> 0.89

`unit_options(q)[0]` is the human-scale DEFAULT for quantity ``q``.
"""

from __future__ import annotations

from typing import Dict, List

# Linear quantities: unit label -> multiplicative factor to the canonical unit.
# The FIRST entry of each dict is the human-scale default the GUI starts on.
_LINEAR: Dict[str, Dict[str, float]] = {
    # canonical g/mL
    'concentration': {'mg/mL': 1e-3, 'g/L': 1e-3, 'µg/mL': 1e-6, 'g/mL': 1.0},
    # canonical Pa·s   (cP was dropped per feedback 2026-06-26 #15 — it is identical
    # to mPa·s, so offering both only invited confusion)
    'viscosity': {'mPa·s': 1e-3, 'Pa·s': 1.0},
    # canonical s
    'time': {'µs': 1e-6, 'ns': 1e-9, 'ms': 1e-3, 's': 1.0},
    # canonical cps (counts per second)
    'intensity': {'kcps': 1e3, 'cps': 1.0, 'Mcps': 1e6},
    # --- plot-axis display quantities (feedback 2026-06-26 #8). The first entry of
    # each is the human-scale default the plots start on. ---
    'scattering_q2': {'nm⁻²': 1e18, 'm⁻²': 1.0},          # canonical m^-2
    'diffusion': {'µm²/s': 1e-12, 'cm²/s': 1e-4, 'm²/s': 1.0},   # canonical m^2/s
    'decay_rate': {'s⁻¹': 1.0, 'ms⁻¹': 1e3},              # canonical s^-1
    'radius': {'nm': 1.0, 'µm': 1e3},                     # canonical nm (Rh/Rg)
    'molar_mass': {'g/mol': 1.0, 'kg/mol': 1e3},          # canonical g/mol
}

# Temperature is affine (offset), not a single factor, so it is handled apart.
_TEMPERATURE_UNITS: List[str] = ['°C', 'K', '°F']     # default first

_CANONICAL_UNIT: Dict[str, str] = {
    'concentration': 'g/mL', 'viscosity': 'Pa·s', 'time': 's',
    'intensity': 'cps', 'temperature': 'K',
    'scattering_q2': 'm⁻²', 'diffusion': 'm²/s', 'decay_rate': 's⁻¹',
    'radius': 'nm', 'molar_mass': 'g/mol',
}


def quantities() -> List[str]:
    """All quantity names this module knows how to convert."""
    return list(_CANONICAL_UNIT.keys())


def unit_options(quantity: str) -> List[str]:
    """Ordered unit labels for a quantity; the first is the human-scale default."""
    if quantity == 'temperature':
        return list(_TEMPERATURE_UNITS)
    try:
        return list(_LINEAR[quantity].keys())
    except KeyError:
        raise ValueError(f"Unknown quantity {quantity!r}.") from None


def default_unit(quantity: str) -> str:
    """The human-scale unit the GUI should start on for this quantity."""
    return unit_options(quantity)[0]


def canonical_unit(quantity: str) -> str:
    """The internal/SI unit the value is stored in."""
    try:
        return _CANONICAL_UNIT[quantity]
    except KeyError:
        raise ValueError(f"Unknown quantity {quantity!r}.") from None


def to_canonical(quantity: str, value: float, unit: str) -> float:
    """Convert a user-entered `value` in `unit` to the canonical unit."""
    if quantity == 'temperature':
        if unit == 'K':
            return value
        if unit == '°C':
            return value + 273.15
        if unit == '°F':
            return (value - 32.0) * 5.0 / 9.0 + 273.15
        raise ValueError(f"Unknown temperature unit {unit!r}.")
    try:
        return value * _LINEAR[quantity][unit]
    except KeyError:
        raise ValueError(f"Unknown unit {unit!r} for quantity {quantity!r}.") from None


def from_canonical(quantity: str, value: float, unit: str) -> float:
    """Convert a canonical `value` to `unit` for display/editing."""
    if quantity == 'temperature':
        if unit == 'K':
            return value
        if unit == '°C':
            return value - 273.15
        if unit == '°F':
            return (value - 273.15) * 9.0 / 5.0 + 32.0
        raise ValueError(f"Unknown temperature unit {unit!r}.")
    try:
        return value / _LINEAR[quantity][unit]
    except KeyError:
        raise ValueError(f"Unknown unit {unit!r} for quantity {quantity!r}.") from None
