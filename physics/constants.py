"""
physics/constants.py
====================

All physical constants and formulas used by the light scattering analysis
platform. Every number or equation that has physical meaning lives here and
nowhere else. Analysis modules call these functions; they do not re-implement
the physics themselves.

The functions in this module are deliberately simple: they take explicit
numerical arguments and return explicit numerical results. There are no
default values for physical parameters (wavelength, temperature, viscosity,
etc.) because this platform never assumes anything about the experiment.

Units contract
--------------
Functions in this module expect and return values in the canonical internal
units defined in core/data_models.py. The calling code (parsers, analysis
functions) is responsible for ensuring inputs have been converted before
calling these functions. Units are documented explicitly on every function.

Contents
--------
Physical constants
    BOLTZMANN_K         Boltzmann constant (J/K)
    AVOGADRO_NA         Avogadro's number (mol^-1)

Scattering geometry
    scattering_vector_q         q in nm^-1 from angle, wavelength, n
    scattering_vector_q_m       q in m^-1 (for Stokes-Einstein chain)

Dynamic light scattering
    stokes_einstein_rh          Rh from D, T, eta
    stokes_einstein_diffusion_coefficient   D from Rh, T, eta (inverse)

Static light scattering
    rayleigh_ratio_toluene      Toluene R_theta (Takahashi 2019 @532; S&K 2021 @660)
    refractive_index_correction Correction factor when solvent != standard
    optical_constant_K          K = 4pi^2 n^2 (dn/dc)^2 / (Na lambda^4)

Depolarized light scattering (static)
    depolarization_ratio_unpolarized   rho_u from rho_v (and inverse)
    depolarization_ratio_vertical      rho_v from rho_u
    cabannes_isotropic_factor          V-incident: 1 - (4/3) rho_v (Chu 1991)
    cabannes_isotropic_factor_natural  natural light: (6-7 rho_u)/(6+6 rho_u)
    optical_anisotropy_squared         delta^2 from rho_v (Chu 1991)

Change history
--------------
2026-06-12  Initial implementation. (constants.py v1)
            Implements: q (nm^-1 and m^-1), Stokes-Einstein, toluene
            Rayleigh ratio (Sivokhin & Kazantsev 2021), refractive index
            mismatch correction, optical constant K.
2026-06-19  Depolarized light scattering (static) helpers. (constants.py v2)
            Depolarization-ratio conversions (rho_u <-> rho_v), the Cabannes
            isotropic factor for vertically polarized incident light
            (R_iso = R_VV (1 - 4/3 rho_v); Chu 1991 Sec. 8.4.1.A) and for
            natural light ((6-7 rho_u)/(6+6 rho_u); Coumou et al. 1964), and the
            optical-anisotropy parameter delta^2 = 5 rho_v / (3 - 4 rho_v).
"""

import math
import warnings
from typing import Optional


# ---------------------------------------------------------------------------
# Fundamental physical constants
# ---------------------------------------------------------------------------

BOLTZMANN_K: float = 1.380649e-23
"""Boltzmann constant in J/K (exact; 2019 SI redefinition, CODATA 2022 / NIST SP 961)."""

AVOGADRO_NA: float = 6.02214076e23
"""Avogadro's number in mol^-1 (exact; 2019 SI redefinition, CODATA 2022 / NIST SP 961)."""


# ---------------------------------------------------------------------------
# Scattering geometry — the scattering vector q
# ---------------------------------------------------------------------------

def scattering_vector_q(
    angle_deg: float,
    wavelength_nm: float,
    solvent_refractive_index: float,
) -> float:
    """Magnitude of the scattering vector q, in nm^-1.

    Definition:
        q = (4 * pi * n / lambda) * sin(theta / 2)

    where lambda is the vacuum wavelength and n is the solvent refractive
    index at that wavelength and temperature. Both are always user-supplied;
    nothing is assumed.

    This form of q is the standard definition for solution scattering
    (Chu 1991). The factor of n converts the vacuum wavelength to the
    in-medium wavelength.

    Parameters
    ----------
    angle_deg : float
        Scattering angle in degrees. Must be strictly between 0 and 180.
    wavelength_nm : float
        Vacuum wavelength of the laser in nanometres (e.g. 532.0 for a
        532 nm Nd:YAG laser).
    solvent_refractive_index : float
        Refractive index of the solvent at the measurement wavelength and
        temperature.

    Returns
    -------
    float
        q in nm^-1.

    Notes
    -----
    Wavelength is kept in nm here (not converted to m) because nm^-1 is
    the natural unit for q in polymer solution scattering -- Rg values
    are typically tens of nm, so qRg is conveniently of order 1.
    The Stokes-Einstein function uses q in m^-1 (see scattering_vector_q_m).
    """
    if not (0.0 < angle_deg < 180.0):
        raise ValueError(
            f"angle_deg must be strictly between 0 and 180, "
            f"got {angle_deg!r}."
        )
    angle_rad = math.radians(angle_deg)
    return (4.0 * math.pi * solvent_refractive_index / wavelength_nm) * math.sin(angle_rad / 2.0)


def scattering_vector_q_m(
    angle_deg: float,
    wavelength_nm: float,
    solvent_refractive_index: float,
) -> float:
    """Magnitude of the scattering vector q, in m^-1.

    Identical to scattering_vector_q() but returns SI units (m^-1) for
    use in the Stokes-Einstein equation, where D is in m^2/s.

    Parameters
    ----------
    angle_deg : float
        Scattering angle in degrees. Must be strictly between 0 and 180.
    wavelength_nm : float
        Vacuum wavelength in nanometres.
    solvent_refractive_index : float
        Solvent refractive index at measurement wavelength and temperature.

    Returns
    -------
    float
        q in m^-1.
    """
    # 1 nm^-1 = 1e9 m^-1
    return scattering_vector_q(angle_deg, wavelength_nm, solvent_refractive_index) * 1.0e9


def optical_wavevector_k(
    wavelength_nm: float,
    solvent_refractive_index: float,
) -> float:
    """In-medium optical wavevector k = 2 * pi * n / lambda, in nm^-1.

    The scattering vector q = 2k * sin(theta/2) reaches its back-scatter
    maximum at q(180 deg) = 2k, so 2k bounds the physically reachable q range.
    Used to cap drawn P(q) model grids: the Mie and coated-sphere form factors
    are undefined past back-scatter (they clamp there, and a curve drawn past
    2k would flatline). Both parameters are user-supplied; nothing is assumed.

    Parameters
    ----------
    wavelength_nm : float
        Vacuum wavelength of the laser in nanometres.
    solvent_refractive_index : float
        Refractive index of the medium at the measurement wavelength and
        temperature.

    Returns
    -------
    float
        k in nm^-1.
    """
    return 2.0 * math.pi * solvent_refractive_index / wavelength_nm


# ---------------------------------------------------------------------------
# Dynamic light scattering — Stokes-Einstein equation
# ---------------------------------------------------------------------------

def _is_positive_finite(value) -> bool:
    """True when a physical scalar is strictly positive AND a real number.

    The bare ``value > 0`` idiom this replaces rejects NaN (every comparison with NaN is
    False) but ADMITS ``+inf``, and the two failure modes look identical at a glance. That
    asymmetry is not academic here: these functions divide by their arguments, so an
    infinite D turns Rh = kT/(6 pi eta D) into a finite, plausible-looking **0.0** rather
    than an error -- a wrong size reported with no ± to warn anyone.

    One predicate for the whole module so the guards cannot drift apart per function: the
    defect was found in `stokes_einstein_rh` and was equally present in every rate<->size
    inverse beside it (2026-07-31 spec-01; CLAUDE.md rule 10/12).
    """
    return math.isfinite(value) and value > 0


def stokes_einstein_rh(
    diffusion_coefficient_m2_per_s: float,
    temperature_K: float,
    viscosity_Pa_s: float,
) -> float:
    """Hydrodynamic radius Rh from the Stokes-Einstein equation, in metres.

    Definition:
        Rh = k_B * T / (6 * pi * eta * D)

    Standard result (the Stokes-Einstein relation); see Berne & Pecora (2000).
    (Chu 1991 does not print this relation -- it converts D->size by an empirical
    D=k_D M^-alpha scaling -- so it is not the source of record for it.)

    Parameters
    ----------
    diffusion_coefficient_m2_per_s : float
        Translational diffusion coefficient D in m^2/s, as extracted from
        a cumulant fit or CONTIN analysis: D = Gamma / q^2.
    temperature_K : float
        Absolute temperature in kelvin. Always user-supplied.
    viscosity_Pa_s : float
        Solvent dynamic viscosity in Pa.s at the measurement temperature.
        Always user-supplied. (1 mPa.s = 1e-3 Pa.s; conversion happens
        at the parser / confirmation layer before this function is called.)

    Returns
    -------
    float
        Hydrodynamic radius Rh in metres.
        To convert to nanometres, multiply by 1e9.

    Notes
    -----
    This function returns the apparent Rh at a single angle and
    concentration. Extrapolation to q->0 and c->0 to obtain the true
    thermodynamic Rh is the responsibility of the calling analysis code.

    Raises
    ------
    ValueError
        If any argument is non-positive or non-finite (all three must be
        strictly positive, finite physical quantities).
    """
    # Refuse at the OPERATION rather than rely on each caller's gate being sized right,
    # which is how +inf got through to a reported Rh = 0.0 nm (2026-07-31 spec-01).
    # All three arguments, not only D: eta = inf gives Rh = 0.0 and T = inf gives Rh = inf.
    if not _is_positive_finite(diffusion_coefficient_m2_per_s):
        raise ValueError(
            f"diffusion_coefficient_m2_per_s must be a positive finite number, "
            f"got {diffusion_coefficient_m2_per_s!r}."
        )
    if not _is_positive_finite(temperature_K):
        raise ValueError(
            f"temperature_K must be a positive finite number, got {temperature_K!r}."
        )
    if not _is_positive_finite(viscosity_Pa_s):
        raise ValueError(
            f"viscosity_Pa_s must be a positive finite number, got {viscosity_Pa_s!r}."
        )
    return BOLTZMANN_K * temperature_K / (6.0 * math.pi * viscosity_Pa_s * diffusion_coefficient_m2_per_s)


def stokes_einstein_diffusion_coefficient(
    rh_m: float,
    temperature_K: float,
    viscosity_Pa_s: float,
) -> float:
    """Translational diffusion coefficient D from Rh (inverse Stokes-Einstein).

    Definition:
        D = k_B * T / (6 * pi * eta * Rh)

    This is the inverse of stokes_einstein_rh(): given a hydrodynamic radius,
    return the diffusion coefficient. Used, for example, when generating a
    synthetic correlogram from a specified particle size.

    Parameters
    ----------
    rh_m : float
        Hydrodynamic radius in metres.
    temperature_K : float
        Absolute temperature in kelvin.
    viscosity_Pa_s : float
        Solvent dynamic viscosity in Pa.s.

    Returns
    -------
    float
        Diffusion coefficient D in m^2/s.

    Raises
    ------
    ValueError
        If any argument is non-positive or non-finite.
    """
    if not _is_positive_finite(rh_m):
        raise ValueError(f"rh_m must be a positive finite number, got {rh_m!r}.")
    if not _is_positive_finite(temperature_K):
        raise ValueError(
            f"temperature_K must be a positive finite number, got {temperature_K!r}.")
    if not _is_positive_finite(viscosity_Pa_s):
        raise ValueError(
            f"viscosity_Pa_s must be a positive finite number, got {viscosity_Pa_s!r}.")
    return BOLTZMANN_K * temperature_K / (6.0 * math.pi * viscosity_Pa_s * rh_m)


# ---------------------------------------------------------------------------
# Static light scattering — toluene Rayleigh ratio (geometry-aware)
# ---------------------------------------------------------------------------
#
# The Rayleigh ratio of toluene depends on scattering GEOMETRY (the polarization
# of the incident and detected light) because toluene is optically anisotropic.
# For a vertically polarized laser the three relevant geometries are:
#   VV  vertical incident, vertical analyzer           R_VV
#   VU  vertical incident, NO analyzer                 R_VU = R_VV (1 + rho_v)
#   VH  vertical incident, horizontal analyzer         R_VH = R_VV rho_v
# where rho_v = I_VH/I_VV is the (vertical) depolarization ratio. These follow
# from R_VU = V_v + H_v with V_v = R_VV and H_v = R_VH = rho_v R_VV
# (Brookhaven BI-200SM manual Sec. VIII; the R_VU = R_VV(1 + rho_v) identity
# follows from rho_v = I_VH/I_VV and polarization-intensity additivity -- it is
# NOT stated by Wu 2010). The geometry MUST match the
# geometry in which the calibrant intensity was measured, or the absolute scale
# is wrong by a factor of (1 + rho_v) ~ 1.35 for toluene.
#
# Base values are the POLARIZED ratio R_VV at 25 °C:
#   532 nm: 2.34e-5 cm^-1  -- Takahashi, Takano, Kinugasa & Sakurai,
#           Anal. Sci. 2019, 35, 1045 (metrology redetermination via certified
#           reference polymers; the authoritative 532 nm value).
#   660 nm: 8.456e-6 cm^-1 -- Sivokhin & Kazantsev, ChemistrySelect 2021, 6,
#           9499 (Table 2). Also the source of the depolarization and temperature
#           data below.
# Other wavelengths can be added with a properly sourced R_VV; do not interpolate
# (the wavelength dependence is steep, ~lambda^-4.17 for toluene; Wu, Chem. Phys.
# 2010, Eq. 4 -- benzene's raw power is 4.38 there, ~4.2 after its refractive-index
# correction; neither is toluene's).
_TOLUENE_RVV_25C: dict = {
    532.0: 2.34e-5,    # cm^-1, Takahashi et al. 2019
    660.0: 8.456e-6,   # cm^-1, Sivokhin & Kazantsev 2021, Table 2
}

# Temperature coefficient of R_VV: fractional change per °C. POSITIVE -- the
# Rayleigh ratio rises with temperature. Value = Sivokhin & Kazantsev 2021 Eq. (9)
# fitted coefficient (0.00421 /°C, referenced to 25 °C); consistent with their Table 2
# (Rvv 8.00e-6 @10 °C -> 9.41e-6 @50 °C, ~+0.42 %/°C) and with benzene's +0.368 %/°C
# (Wu 2010 Eq. 3). S&K assume the same fractional coefficient at all wavelengths.
# (Was 0.0043, a rough endpoint estimate; corrected to S&K's own fitted 0.00421,
# owner-approved 2026-07-22.)
_TOLUENE_RVV_TEMP_COEFFICIENT: float = 0.00421   # per °C  (relative, positive; S&K Eq. 9)

# Vertical depolarization ratio of toluene, rho_v = I_VH / I_VV.
# Sivokhin & Kazantsev 2021, Table 1: 0.346 at 25 °C (measured at 660 nm),
# decreasing ~linearly with temperature (0.364 at 10 °C to 0.310 at 50 °C).
# rho_v is only weakly wavelength-dependent; this 660 nm value is used at other
# wavelengths unless the user supplies one. It is the key uncertain parameter for
# converting between geometries.
_TOLUENE_DEPOL_V_25C: float = 0.346
_TOLUENE_DEPOL_V_TEMP_COEFFICIENT: float = -0.00135   # per °C  (absolute)

_TOLUENE_RAYLEIGH_REF_TEMP_C: float = 25.0

# How far a declared laser wavelength may sit from a tabulated one (532 / 660 nm) and
# still be served that table entry UNSCALED. A house convenience tolerance with no
# literature source -- a flagged convenience simplification, not a canonical criterion:
# it exists so a nominal "532" laser typed as 531.9 or 532.5 is not refused, and the
# canonical alternative is a separately sourced R_VV at the actual wavelength (the
# do-not-interpolate rule above). Its price is bounded by the wavelength exponent
# below: at 2 nm from 532 nm the served value is ~1.6 % off (4.17 x 2/532), which is
# why the substitution is DISCLOSED (`toluene_wavelength_extrapolation_reason`) rather
# than silent -- a 534 nm laser was served R_VV(532) with nothing said, and since
# Mw ~ R_std the whole bias landed on Mw and A2.
_WAVELENGTH_MATCH_TOLERANCE_NM: float = 2.0

# Toluene's Rayleigh-ratio wavelength exponent, R ~ lambda^-4.17 -- Wu, Chem. Phys.
# 2010, Eq. (4), p. 45: the correlation is for the UNPOLARIZED ratio R_u over 420-700 nm
# (the depolarization ratio is ~wavelength-independent there, so the same exponent
# sizes R_VV's disclosure; benzene's raw power is 4.38, not this one). Used ONLY to
# size the disclosure of a within-tolerance
# substitution; the value served is never scaled by it (the correlation's own scatter is
# up to 6 % at 532 nm, so a scaled value would not be a sourced one).
_TOLUENE_RVV_WAVELENGTH_EXPONENT: float = -4.17

# Temperature validity range for the R_VV(T), rho_v(T) corrections: the span of the
# Sivokhin & Kazantsev 2021 tables (Rvv and rho_v tabulated 10-50 C). Outside this
# the linear coefficients above are unvalidated extrapolation; we WARN (not raise --
# owner 2026-07-11) so elevated-temperature calibrations still run, but the
# extrapolation is never silent.
_TOLUENE_RAYLEIGH_TEMP_MIN_C: float = 10.0
_TOLUENE_RAYLEIGH_TEMP_MAX_C: float = 50.0

_RAYLEIGH_GEOMETRIES = ('VV', 'VU', 'VH')


def toluene_temperature_extrapolation_reason(temperature_C: float) -> Optional[str]:
    """The extrapolation note for a toluene-standard temperature, or None if in range.

    Single source of truth for the wording: `rayleigh_ratio_toluene` emits it via
    `warnings.warn` (headless/stderr + tests), and the GUI surfaces the same string as
    a neutral note next to the calibration k_c. Glyph-less and
    self-contained so either surface can lead it with its own preamble/marker.

    Returns None when `temperature_C` is inside the S&K 10-50 C table; otherwise a
    message stating the value is extrapolated (the value is used, not blocked -- so
    elevated-temperature calibrations still run).
    """
    if _TOLUENE_RAYLEIGH_TEMP_MIN_C <= temperature_C <= _TOLUENE_RAYLEIGH_TEMP_MAX_C:
        return None
    # No author-year citation here: this string is also rendered on-screen (the GUI
    # calibration note), and house convention keeps citations out of UI prose (in the
    # Theory guide instead -- doc_style_guide R6.1). The source (Sivokhin & Kazantsev
    # 2021) is cited in this function's docstring and the Theory-and-Equations-Guide.
    return (
        f"Temperature {temperature_C:g} C is outside the "
        f"{_TOLUENE_RAYLEIGH_TEMP_MIN_C:g}-{_TOLUENE_RAYLEIGH_TEMP_MAX_C:g} C table of "
        f"the toluene Rayleigh-ratio temperature correction; the value is EXTRAPOLATED "
        f"and its accuracy is not validated there."
    )


def toluene_wavelength_extrapolation_reason(wavelength_nm: float) -> Optional[str]:
    """The extrapolation note for a toluene-standard wavelength, or None.

    The wavelength sibling of `toluene_temperature_extrapolation_reason` (rule 10: the
    two axes of the same table are disclosed the same way). None when `wavelength_nm`
    IS a tabulated wavelength, and also None when it is too far from every tabulated
    one to be served at all -- `rayleigh_ratio_toluene` raises there, and the raise is
    that case's surface. In between (within `_WAVELENGTH_MATCH_TOLERANCE_NM` of a table
    entry but not on it) the nearest entry is served UNSCALED, and this names the
    substitution and its approximate size on toluene's ~lambda^-4.17 dependence.
    Single source of the wording: `rayleigh_ratio_toluene` emits it via `warnings.warn`
    and the GUI shows the same string next to k_c.
    """
    if not math.isfinite(wavelength_nm):
        return None
    nearest = min(_TOLUENE_RVV_25C, key=lambda w: abs(wavelength_nm - w))
    d = abs(wavelength_nm - nearest)
    if math.isclose(wavelength_nm, nearest, abs_tol=1e-9) or d > _WAVELENGTH_MATCH_TOLERANCE_NM:
        return None
    # Within tolerance implies 530 <= lambda <= 662 nm, so the ratio below is in
    # (0.99, 1.01) -- no tail of the power can under- or overflow here.
    pct = abs((wavelength_nm / nearest) ** _TOLUENE_RVV_WAVELENGTH_EXPONENT - 1.0) * 100.0
    supported = ', '.join(f'{w:g}' for w in sorted(_TOLUENE_RVV_25C))
    # No author-year citation here for the same reason as the temperature note (this
    # string renders on-screen); the exponent's source is on the module constant.
    return (
        f"Wavelength {wavelength_nm:g} nm is not in the toluene Rayleigh-ratio table "
        f"({supported} nm); the {nearest:g} nm value was used unscaled, which is "
        f"EXTRAPOLATED: on toluene's ~lambda^-4.17 wavelength dependence a {d:g} nm "
        f"difference moves the Rayleigh ratio, and with it k_c, Mw and A2, by about "
        f"{pct:.1f} %."
    )


def rayleigh_ratio_toluene(
    wavelength_nm: float,
    temperature_C: float,
    geometry: Optional[str] = None,
    depolarization_ratio_v: Optional[float] = None,
) -> float:
    """Rayleigh ratio of toluene in cm^-1, for a chosen scattering geometry.

    Computed from the polarized value R_VV (Takahashi et al. 2019 at 532 nm;
    Sivokhin & Kazantsev 2021 at 660 nm), temperature-corrected, then converted
    to the requested geometry using toluene's depolarization ratio. This is the
    value to use for SLS calibration; the value embedded in instrument software
    must NOT be used in its place.

    Geometry conversions (vertically polarized incident light):
        VV : R_VV                       (vertical analyzer)
        VU : R_VV * (1 + rho_v)         (NO analyzer)
        VH : R_VV * rho_v               (horizontal analyzer)
    The geometry MUST match the geometry in which the calibrant intensity was
    measured; otherwise the absolute scale is wrong by ~(1 + rho_v) ~ 1.35.

    Parameters
    ----------
    wavelength_nm : float
        Laser vacuum wavelength in nm. Supported: 532 and 660 nm (+/- 2 nm).
    temperature_C : float
        Measurement temperature in degrees Celsius.
    geometry : str
        'VV', 'VU', or 'VH' -- REQUIRED (no default). Use 'VU' for an instrument
        with a vertically polarized laser and no polarization analyzer.
        Passing None (or omitting it) raises: the geometry must be
        explicit, since VV vs VU differ by (1 + rho_v) ~ 1.35 on the absolute
        scale and a wrong default would silently corrupt every calibrated Mw/A2.
    depolarization_ratio_v : float, optional
        rho_v = I_VH / I_VV for toluene. If omitted, the temperature-dependent
        value from Sivokhin & Kazantsev 2021 is used (0.346 at 25 C). Supply your
        own if you have measured it, especially at wavelengths far from 660 nm.
        Must be in [0, 0.75], the physical range for vertically polarized
        incident light (see _RHO_V_DEPOLARIZATION_LIMIT).

    Returns
    -------
    float
        Rayleigh ratio of toluene in cm^-1 for the requested geometry.

    Raises
    ------
    ValueError
        If the geometry is unspecified (None) or not VV/VU/VH, if the wavelength
        is unsupported, or if a user-supplied depolarization_ratio_v is outside
        [0, 0.75].

    Warns
    -----
    UserWarning
        If the temperature is outside the 10-50 C table of the S&K correction (the
        value is extrapolated, not blocked -- so elevated-temperature calibrations
        still work, but the extrapolation is flagged). Likewise if the wavelength is
        within the +/- 2 nm match tolerance of a tabulated wavelength but not ON it:
        the tabulated value is served unscaled and the substitution is named, with its
        approximate size (`toluene_wavelength_extrapolation_reason`).

    Notes
    -----
    R_VV(T) = R_VV(25) * (1 + alpha (T - 25)), alpha = +0.00421 /C (S&K 2021 Eq. 9;
    the coefficient is POSITIVE -- the Rayleigh ratio rises with temperature).
    rho_v(T) = 0.346 - 0.00135 (T - 25) (S&K Table 1) unless overridden.
    Implements S&K 2021 Eq. (9) (p. 9501) verbatim, anchored at their Table 2
    value at 25 C; it reproduces Table 2 within 1 % over 10-50 C (the table is
    slightly convex, so the linear form sits 0.9 % low at 10 C and 0.7 % low at
    50 C -- the residual of S&K's own linear fit).

    For wavelengths not in the table, add a properly sourced R_VV to
    _TOLUENE_RVV_25C; do not interpolate (R ~ lambda^-4.17 for toluene, Wu 2010 Eq. 4).
    """
    if geometry is None:
        raise ValueError(
            "geometry must be specified: one of 'VV', 'VU', or 'VH' (VU for a "
            "vertically polarized laser with no analyzer). "
            "There is no default -- VV vs VU differ by (1 + rho_v) ~ 1.35 on the "
            "absolute scale, so a silent default would corrupt calibrated Mw/A2."
        )
    geometry = geometry.upper()
    if geometry not in _RAYLEIGH_GEOMETRIES:
        raise ValueError(
            f"geometry must be one of {_RAYLEIGH_GEOMETRIES}, got {geometry!r}."
        )

    # The S&K temperature correction is tabulated only over 10-50 C; outside it the
    # linear R_VV(T)/rho_v(T) coefficients are EXTRAPOLATED. WARN (do not raise) so
    # elevated-temperature calibrations still work, but the extrapolation is never
    # silent (owner 2026-07-11 -- warn, not raise, since reline SLS runs
    # hot). Keep-both: the same reason string is surfaced in the GUI as a neutral note
    # next to the calibration k_c via the controller; this warning is
    # the headless/stderr + pytest.warns channel.
    reason = toluene_temperature_extrapolation_reason(temperature_C)
    if reason is not None:
        warnings.warn(reason, stacklevel=2)

    # Nearest supported wavelength within tolerance.
    best_match, best_distance = None, float('inf')
    for supported_wl in _TOLUENE_RVV_25C:
        d = abs(wavelength_nm - supported_wl)
        if d < best_distance:
            best_distance, best_match = d, supported_wl
    if best_distance > _WAVELENGTH_MATCH_TOLERANCE_NM:
        supported = ', '.join(f'{w} nm' for w in sorted(_TOLUENE_RVV_25C))
        raise ValueError(
            f"Wavelength {wavelength_nm} nm is not within "
            f"{_WAVELENGTH_MATCH_TOLERANCE_NM} nm of a supported value. "
            f"Supported: {supported}. Add a sourced R_VV to _TOLUENE_RVV_25C "
            f"to support a new wavelength."
        )
    # Within tolerance but not on the table: served unscaled, never silently. The
    # wavelength sibling of the temperature warning above (same keep-both shape: this
    # warning is the headless/pytest channel, the GUI renders the same string).
    wl_reason = toluene_wavelength_extrapolation_reason(wavelength_nm)
    if wl_reason is not None:
        warnings.warn(wl_reason, stacklevel=2)

    # Temperature-corrected polarized value R_VV(T).
    dT = temperature_C - _TOLUENE_RAYLEIGH_REF_TEMP_C
    r_vv = _TOLUENE_RVV_25C[best_match] * (1.0 + _TOLUENE_RVV_TEMP_COEFFICIENT * dT)

    if geometry == 'VV':
        return r_vv

    # Depolarization ratio (temperature-dependent default, or user-supplied).
    if depolarization_ratio_v is None:
        rho_v = _TOLUENE_DEPOL_V_25C + _TOLUENE_DEPOL_V_TEMP_COEFFICIENT * dT
        origin = (f"the tabulated rho_v(T) evaluated at {temperature_C!r} C -- its "
                  f"linear form turns negative near 281 C, far past the 10-50 C table")
    else:
        rho_v = float(depolarization_ratio_v)
        origin = 'the supplied value'
    # Same physical-range guard as every other rho_v consumer in this module
    # (depolarization_ratio_unpolarized, cabannes_isotropic_factor,
    # optical_anisotropy_squared), on BOTH arms: a negative Rayleigh ratio is not an
    # extrapolation the warn-not-raise rule for the table covers. The
    # `not (0.0 <= x <= limit)` form also rejects NaN and inf.
    if not (0.0 <= rho_v <= _RHO_V_DEPOLARIZATION_LIMIT):
        raise ValueError(
            f"rho_v must be in [0, {_RHO_V_DEPOLARIZATION_LIMIT}] for the toluene "
            f"calibrant's depolarization ratio, got {rho_v!r} ({origin})."
        )

    if geometry == 'VU':
        return r_vv * (1.0 + rho_v)
    else:  # 'VH'
        return r_vv * rho_v


# ---------------------------------------------------------------------------
# Static light scattering — refractive index mismatch correction
# ---------------------------------------------------------------------------

def refractive_index_correction(
    solvent_refractive_index: float,
    standard_refractive_index: float,
) -> float:
    """Correction factor for excess Rayleigh ratio when solvent != standard.

    When the scattering standard (toluene) and the sample solvent are
    different liquids, their refractive indices differ, and the scattering
    volume seen by the detector is refracted differently at the cell wall.
    This factor corrects for that geometric effect.

    Definition:
        f = (n_solvent / n_standard)^2

    The corrected excess Rayleigh ratio is:
        Delta_R_theta = R_theta_measured * f

    Parameters
    ----------
    solvent_refractive_index : float
        Refractive index of the sample solvent at the measurement
        wavelength and temperature.
    standard_refractive_index : float
        Refractive index of the calibration standard (typically toluene)
        at the measurement wavelength and temperature.

    Returns
    -------
    float
        Dimensionless correction factor f = (n_solvent / n_standard)^2.

    Notes
    -----
    When solvent == standard (e.g. toluene in toluene), f = 1 exactly and
    no correction is needed. The SLS analysis module applies this factor
    whenever solvent_refractive_index != standard_refractive_index.

    The fixed (n/n_std)^2 form is derived from cell-wall refraction of the
    observed scattering volume and solid angle (Helmholtz-Lagrange), and
    Coumou proves it holds regardless of cell shape -- rectangular and
    cylindrical alike [@coumou1960, Eq. [2a], p. 413]. Chu gives a
    generalized detection-optics form with an exponent between 1 and 2
    (Eq. 8.4.19); this platform implements the fixed refraction-derived
    exponent-2 form.
    """
    if not _is_positive_finite(standard_refractive_index):
        raise ValueError(
            f"standard_refractive_index must be a positive finite number, "
            f"got {standard_refractive_index!r}."
        )
    if not _is_positive_finite(solvent_refractive_index):
        raise ValueError(
            f"solvent_refractive_index must be a positive finite number, "
            f"got {solvent_refractive_index!r}."
        )
    # `r * r`, not `r ** 2`: both indices are user-supplied and guarded only for
    # positive-finiteness, so their RATIO can reach the edge where a Python float `**`
    # raises OverflowError while `*` saturates to inf. The sibling square in
    # `optical_constant_K` is spelled the same way and for the same reason; this one is
    # reached from the same call in `compute_excess_rayleigh_ratio`, on the same
    # user-supplied refractive index, so leaving it as `**` would just move the crash
    # one function over.
    ratio = solvent_refractive_index / standard_refractive_index
    return ratio * ratio


# ---------------------------------------------------------------------------
# Static light scattering — optical constant K
# ---------------------------------------------------------------------------

def optical_constant_K(
    solvent_refractive_index: float,
    dn_dc_mL_per_g: float,
    wavelength_nm: float,
) -> float:
    """Optical constant K for SLS in units of mol*cm^2 / g^2.

    Definition:
        K = 4 * pi^2 * n^2 * (dn/dc)^2 / (Na * lambda^4)

    This is the prefactor in the Zimm / Debye / Berry equations:
        Kc / R_theta = 1/Mw * P(q)^-1 + 2*A2*c + ...

    Parameters
    ----------
    solvent_refractive_index : float
        Refractive index of the solvent at the measurement wavelength
        and temperature.
    dn_dc_mL_per_g : float
        Refractive index increment in mL/g, at the measurement wavelength.
        Always user-supplied.
    wavelength_nm : float
        Vacuum wavelength of the laser in nanometres.

    Returns
    -------
    float
        Optical constant K in mol*cm^2 / g^2.

    Notes
    -----
    Unit derivation (using 1 mL = 1 cm^3, so dn/dc in mL/g = cm^3/g):
        lambda is converted from nm to cm internally (1 nm = 1e-7 cm).
        (dn/dc)^2 is in (cm^3/g)^2 = cm^6 / g^2.
        Na is in mol^-1.
        Result: [cm^6 / g^2] / ([mol^-1] * [cm^4])
               = mol * cm^2 / g^2.
        This is consistent with the Zimm equation Kc/R_theta = 1/Mw + ...:
        K * c / R_theta = [mol*cm^2/g^2] * [g/cm^3] / [cm^-1] = mol/g = 1/Mw,
        with c in g/mL (= g/cm^3) and R_theta in cm^-1.

    The factor of 4 pi^2 (not 2 pi^2) is the standard form for VV
    (vertically polarized incident and detected) geometry, which is the
    default for most modern light scattering instruments. If your
    instrument uses a different polarization geometry, K must be modified.
    """
    # Every square and fourth power is spelled as repeated MULTIPLICATION, never `**`.
    # On a Python float `**` RAISES OverflowError where `*` saturates to inf
    # (`1e300 ** 2` raises; `1e300 * 1e300` is inf), and a raise happens before any
    # caller's guard can run -- so `**` here would make an out-of-range input a crash
    # rather than a flaggable value. All three arguments are user-supplied and only
    # range-checked for sign/finiteness, so all three can reach the edge: measured,
    # `dn/dc = 1e200`, `n = 1e200` and `lambda = 1e200` each raised out of this
    # function before this change. Callers guard the RESULT (see
    # analysis.sls.compute_excess_rayleigh_ratio), which is the one quantity that
    # actually has to be usable.
    wavelength_cm = wavelength_nm * 1.0e-7   # nm -> cm
    n2 = solvent_refractive_index * solvent_refractive_index
    dn_dc2 = dn_dc_mL_per_g * dn_dc_mL_per_g
    lambda2 = wavelength_cm * wavelength_cm
    numerator = 4.0 * math.pi * math.pi * n2 * dn_dc2
    denominator = AVOGADRO_NA * lambda2 * lambda2
    if denominator == 0.0:
        # lambda^4 underflowed to zero, so the quotient has no representable value.
        # Saturate rather than raise, for the same reason the squares above use `*`:
        # a ZeroDivisionError here kills the analysis before the caller's K guard can
        # flag the run. Sign-free -- the numerator is a sum of squares, so it is never
        # negative -- and NaN when both ends collapse, which the caller's
        # `isfinite(K) and K > 0` test rejects exactly as it rejects inf.
        return float('inf') if numerator > 0.0 else float('nan')
    return numerator / denominator


# ---------------------------------------------------------------------------
# Depolarized light scattering (static) -- depolarization ratios, the Cabannes
# factor, and the optical-anisotropy parameter
# ---------------------------------------------------------------------------
#
# An optically anisotropic scatterer (a molecule or particle whose polarisability
# is not the same in every direction) rotates the polarization of some of the
# light it scatters. With VERTICALLY polarized incident light -- the default for
# essentially every modern instrument -- this shows up as a horizontally polarized
# ("depolarized") component in the scattered light. Two quantities describe it:
#
#   rho_v = I_VH / I_VV   depolarization ratio, vertically polarized incident light
#                         (I_VH = horizontal analyzer, I_VV = vertical analyzer).
#                         This is the quantity modern instruments measure and the
#                         one physics/constants.py already carries for toluene.
#   rho_u = I_h  / I_v    depolarization ratio, UNPOLARIZED (natural) incident
#                         light. The older light-scattering literature (Coumou,
#                         Cabannes) is written in terms of rho_u.
#
# They describe the same physics in two geometries and convert exactly (no
# approximation) via  rho_u = 2 rho_v / (1 + rho_v)  -- verified to reproduce the
# Sivokhin & Kazantsev (2021) toluene table (rho_v = 0.346 -> rho_u = 0.514 at
# 25 C). See depolarization_ratio_unpolarized below.
#
# Why this matters for SLS: the anisotropic component inflates the measured VV
# Rayleigh ratio, so an uncorrected Mw comes out too high (Mw_app = Mw (1 + 4/5
# delta^2); Chu 1991 Eq. 8.4.8). The Cabannes factor removes that inflation,
# recovering the ISOTROPIC Rayleigh ratio that the Zimm/Debye analysis actually
# wants. For vertically polarized incident light the factor is 1 - (4/3) rho_v
# (derived below); for natural light it is the classic (6 - 7 rho_u)/(6 + 6 rho_u).
#
# Physical range: for vertically polarized incident light rho_v lies in [0, 3/4].
# rho_v = 0 is an optically isotropic scatterer (no depolarization, no
# correction). The upper limit rho_v = 3/4 is the fully-anisotropic limit, where
# the isotropic part vanishes (delta^2 -> infinity, Cabannes factor -> 0). Small
# flexible polymer coils sit very near 0; small anisotropic molecules like toluene
# are ~0.3-0.35; rigid rods and many particles fall in between.

# Largest physically meaningful rho_v for vertically polarized incident light.
# At this value the isotropic scattering vanishes (see module notes above).
_RHO_V_DEPOLARIZATION_LIMIT: float = 0.75


def depolarization_ratio_unpolarized(rho_v: float) -> float:
    """Unpolarized depolarization ratio rho_u from the vertical ratio rho_v.

    Definition:
        rho_u = 2 rho_v / (1 + rho_v)

    Converts a measured vertical-incident depolarization ratio (what modern
    instruments report) to the natural/unpolarized-incident depolarization ratio
    used throughout the classical light-scattering literature (Coumou, Cabannes).
    The relation is exact, not an approximation.

    Parameters
    ----------
    rho_v : float
        Depolarization ratio for vertically polarized incident light,
        rho_v = I_VH / I_VV. Must satisfy 0 <= rho_v <= 3/4 (the physical range;
        see module notes).

    Returns
    -------
    float
        Depolarization ratio rho_u for unpolarized incident light.

    Raises
    ------
    ValueError
        If rho_v is outside [0, 3/4].

    Notes
    -----
    Verified against Sivokhin & Kazantsev (2021) Table 1 (toluene, 660 nm):
    rho_v = 0.364/0.346/0.310 at 10/25/50 C give rho_u = 0.534/0.514/0.473.
    """
    if not (0.0 <= rho_v <= _RHO_V_DEPOLARIZATION_LIMIT):
        raise ValueError(
            f"rho_v must be in [0, {_RHO_V_DEPOLARIZATION_LIMIT}] for vertically "
            f"polarized incident light, got {rho_v!r}."
        )
    return 2.0 * rho_v / (1.0 + rho_v)


def depolarization_ratio_vertical(rho_u: float) -> float:
    """Vertical depolarization ratio rho_v from the unpolarized ratio rho_u.

    Inverse of depolarization_ratio_unpolarized():
        rho_v = rho_u / (2 - rho_u)

    Use this to read a classical-literature rho_u (natural-light) value into the
    vertical-incident convention the instrument and the rest of this code use.

    Parameters
    ----------
    rho_u : float
        Depolarization ratio for unpolarized incident light. Must satisfy
        0 <= rho_u <= 6/7 (the image of [0, 3/4] under the forward relation).

    Returns
    -------
    float
        Depolarization ratio rho_v for vertically polarized incident light.

    Raises
    ------
    ValueError
        If rho_u is outside [0, 6/7].
    """
    rho_u_limit = 2.0 * _RHO_V_DEPOLARIZATION_LIMIT / (1.0 + _RHO_V_DEPOLARIZATION_LIMIT)  # 6/7
    if not (0.0 <= rho_u <= rho_u_limit + 1e-12):
        raise ValueError(
            f"rho_u must be in [0, {rho_u_limit:.6g}] (= 6/7), got {rho_u!r}."
        )
    return rho_u / (2.0 - rho_u)


def cabannes_isotropic_factor(rho_v: float) -> float:
    """Cabannes isotropic factor for VERTICALLY polarized incident light.

    Definition:
        f = 1 - (4/3) rho_v

    Multiply a measured VV excess Rayleigh ratio by this factor to recover the
    ISOTROPIC excess Rayleigh ratio (the part that carries Mw / Rg / A2):
        R_iso = R_VV * (1 - (4/3) rho_v).

    Derivation (Chu 1991, Sec. 8.4.1.A). With vertically polarized incident
    light the apparent molecular weight and the depolarized Rayleigh ratio are
        Mw_app    = Mw (1 + (4/5) delta^2)            (Chu Eq. 8.4.8)
        R_HV/(Hc) = (3/5) delta^2 Mw     (c,K -> 0)   (Chu Eq. 8.4.10)
    so rho_v = R_HV / R_VV = (3/5) delta^2 / (1 + (4/5) delta^2). Eliminating
    delta^2 gives R_VV (1 - (4/3) rho_v) = R_iso exactly. Equivalently, the
    per-molecule optics I_VV ~ abar^2 + (4/45) gamma^2, I_VH ~ (3/45) gamma^2
    give the same 4/3 = (4/45)/(3/45) (Kerker 1969, Ch. 10 Sec. 10.2; Berne &
    Pecora 2000) -- the two routes agree.

    Parameters
    ----------
    rho_v : float
        Vertical-incident depolarization ratio, 0 <= rho_v <= 3/4.

    Returns
    -------
    float
        Isotropic factor f in [0, 1]. f = 1 when rho_v = 0 (no anisotropy, no
        correction); f -> 0 at the depolarization limit rho_v = 3/4.

    Raises
    ------
    ValueError
        If rho_v is outside [0, 3/4].

    Notes
    -----
    This is the vertically-polarized-incident analog of the classical natural-
    light Cabannes factor (6 - 7 rho_u)/(6 + 6 rho_u); see
    cabannes_isotropic_factor_natural(). The two are NOT interchangeable, even
    at small anisotropy: with rho_u = 2 rho_v/(1 + rho_v) the natural-light
    factor removes (13/3) rho_v to first order against this function's
    (4/3) rho_v (a ratio of 3.25), because the unpolarized total it corrects
    already contains the horizontal component. Sources for the two forms:
    Kerker 1969 Ch. 9 Sec. 9.1c (p. 498) states the natural-light factor; Chu
    1991 Sec. 8.4.1.A the vertical-incident relations above. The anisotropic
    FRACTION removed is 1 - f = (4/3) rho_v.
    """
    if not (0.0 <= rho_v <= _RHO_V_DEPOLARIZATION_LIMIT):
        raise ValueError(
            f"rho_v must be in [0, {_RHO_V_DEPOLARIZATION_LIMIT}], got {rho_v!r}."
        )
    return 1.0 - (4.0 / 3.0) * rho_v


def cabannes_isotropic_factor_natural(rho_u: float) -> float:
    """Cabannes isotropic factor for UNPOLARIZED (natural) incident light.

    Definition (the classical Cabannes factor):
        f = (6 - 7 rho_u) / (6 + 6 rho_u)

    Multiply a total Rayleigh ratio measured with unpolarized incident light at
    90 degrees by this factor to recover the isotropic part:
        R_iso = R_total * (6 - 7 rho_u) / (6 + 6 rho_u).

    Provided for completeness and for reading older (natural-light) datasets;
    modern vertically-polarized instruments use cabannes_isotropic_factor().

    Parameters
    ----------
    rho_u : float
        Unpolarized-incident depolarization ratio, 0 <= rho_u <= 6/7.

    Returns
    -------
    float
        Isotropic factor f in [0, 1].

    Raises
    ------
    ValueError
        If rho_u is outside [0, 6/7].

    Notes
    -----
    The (6 - 7 rho_u)/(6 + 6 rho_u) form is Kerker 1969, Ch. 9 Sec. 9.1c (p. 498).
    Coumou, Mackor & Hijmans (1964); verified to reproduce Coumou's Table 3
    (benzene rho_u = 0.42 -> total/isotropic = 1/f = 2.78).
    """
    rho_u_limit = 6.0 / 7.0
    if not (0.0 <= rho_u <= rho_u_limit + 1e-12):
        raise ValueError(
            f"rho_u must be in [0, {rho_u_limit:.6g}] (= 6/7), got {rho_u!r}."
        )
    return (6.0 - 7.0 * rho_u) / (6.0 + 6.0 * rho_u)


def optical_anisotropy_squared(rho_v: float) -> float:
    """Optical-anisotropy parameter delta^2 from the vertical depolarization ratio.

    Definition (inverting Chu 1991 rho_v = (3/5) delta^2 / (1 + (4/5) delta^2)):
        delta^2 = 5 rho_v / (3 - 4 rho_v)

    delta^2 is the mean-square optical anisotropy in Chu's parametrization: 0 for
    an optically isotropic scatterer, growing without bound as rho_v -> 3/4.

    Parameters
    ----------
    rho_v : float
        Vertical-incident depolarization ratio, 0 <= rho_v < 3/4. (rho_v = 3/4 is
        excluded: delta^2 diverges there.)

    Returns
    -------
    float
        delta^2 (dimensionless), >= 0.

    Raises
    ------
    ValueError
        If rho_v is outside [0, 3/4).

    Notes
    -----
    This is Chu's delta^2; other texts use a differently normalized anisotropy
    (e.g. gamma^2 with the per-molecule I_VV ~ abar^2 + (4/45) gamma^2 form).
    Convert via the depolarization ratio, which is convention-independent, not by
    equating delta^2 and gamma^2 directly.
    """
    if not (0.0 <= rho_v < _RHO_V_DEPOLARIZATION_LIMIT):
        raise ValueError(
            f"rho_v must be in [0, {_RHO_V_DEPOLARIZATION_LIMIT}) for a finite "
            f"delta^2, got {rho_v!r}."
        )
    return 5.0 * rho_v / (3.0 - 4.0 * rho_v)


# ---------------------------------------------------------------------------
# Shape models -- diffusion coefficients of a rigid rod and of a sphere
# ---------------------------------------------------------------------------
#
# These map a particle's GEOMETRY to its diffusion coefficients (the forward
# direction). The depolarized dynamic analysis runs them BACKWARDS -- given the
# measured D_t and D_r, solve for the dimensions -- which is a model-dependent
# inverse problem and lives in analysis/depolarization.py. These forward formulas
# (and their algebraic inverses) are the only physics; the solver only calls them.
#
# Rigid circular cylinder (length L, diameter d, axial ratio p = L/d):
#   Tirado, Lopez Martinez & Garcia de la Torre, J. Chem. Phys. 81, 2047 (1984),
#   Eqs. (3),(5),(9),(10). The end-effect correction polynomials are fitted for
#   2 < p < 30; outside that range the formulas are unsupported extrapolations.
#
# Sphere (radius R):
#   translational D_t = kT/(6 pi eta R)         (Stokes-Einstein; stokes_einstein_rh)
#   rotational    D_r = kT/(8 pi eta R^3)        (Stokes-Einstein-Debye; e.g. Balog
#                                                 et al., Nanoscale 7, 5991 (2015))

# Aspect-ratio (p = L/d) range over which the Tirado (1984) rod corrections are
# fitted. A documented validity bound, not a hard domain limit (the formulas
# evaluate for any p > 1); callers flag results outside it.
ROD_ASPECT_RATIO_MIN: float = 2.0
ROD_ASPECT_RATIO_MAX: float = 30.0


def rod_end_corrections(aspect_ratio_p: float) -> tuple:
    """Tirado (1984) end-effect corrections (nu_translational, delta_perp_rotational).

        nu      = 0.312 + 0.565/p - 0.100/p^2      (Eq. 9, translational)
        delta_perp = -0.662 + 0.917/p - 0.050/p^2  (Eq. 10, end-over-end rotational)

    Defined for p > 0; physically meaningful (and fitted) for 2 < p < 30. Returned
    as a pair so the rod diffusion functions and the inverse solver share one source.
    """
    p = aspect_ratio_p
    if not _is_positive_finite(p):
        raise ValueError(
            f"aspect ratio p must be a positive finite number, got {p!r}.")
    nu = 0.312 + 0.565 / p - 0.100 / p ** 2
    delta_perp = -0.662 + 0.917 / p - 0.050 / p ** 2
    return nu, delta_perp


def rod_translational_diffusion(length_m: float, diameter_m: float,
                                temperature_K: float, viscosity_Pa_s: float) -> float:
    """Translational diffusion coefficient D_t of a rigid cylinder, in m^2/s.

        D_t = kT / (3 pi eta L) * (ln p + nu),   p = L/d   (Tirado 1984 Eqs. 3, 9)

    Valid for 2 < p < 30 (rod_aspect_ratio_valid); evaluates outside but the
    correction is then an extrapolation.
    """
    _require_positive_lengths(length_m, diameter_m, temperature_K, viscosity_Pa_s)
    p = length_m / diameter_m
    nu, _ = rod_end_corrections(p)
    return (BOLTZMANN_K * temperature_K
            / (3.0 * math.pi * viscosity_Pa_s * length_m)) * (math.log(p) + nu)


def rod_rotational_diffusion(length_m: float, diameter_m: float,
                             temperature_K: float, viscosity_Pa_s: float) -> float:
    """End-over-end rotational diffusion coefficient D_r of a rigid cylinder, rad^2/s.

        D_r = 3 kT / (pi eta L^3) * (ln p + delta_perp),  p = L/d
        (Tirado 1984 Eqs. 5, 10)

    This is the rotation DDLS observes (the symmetry-axis spin is not seen).
    """
    _require_positive_lengths(length_m, diameter_m, temperature_K, viscosity_Pa_s)
    p = length_m / diameter_m
    _, delta_perp = rod_end_corrections(p)
    return (3.0 * BOLTZMANN_K * temperature_K
            / (math.pi * viscosity_Pa_s * length_m ** 3)) * (math.log(p) + delta_perp)


def rod_length_from_translational_diffusion(diffusion_coefficient_m2_per_s: float,
                                            aspect_ratio_p: float,
                                            temperature_K: float,
                                            viscosity_Pa_s: float) -> float:
    """Cylinder length L (m) from D_t at a KNOWN aspect ratio p (inverse of D_t).

        L = kT (ln p + nu(p)) / (3 pi eta D_t)

    The algebraic inverse of rod_translational_diffusion for fixed p. The depolarized
    rod inversion uses it to reduce the 2-D (L, d) solve to a 1-D root find in p.
    """
    if not _is_positive_finite(diffusion_coefficient_m2_per_s):
        raise ValueError(
            f"D_t must be a positive finite number, "
            f"got {diffusion_coefficient_m2_per_s!r}.")
    if not (_is_positive_finite(temperature_K) and _is_positive_finite(viscosity_Pa_s)):
        raise ValueError("temperature_K and viscosity_Pa_s must be positive finite "
                         "numbers.")
    nu, _ = rod_end_corrections(aspect_ratio_p)
    return (BOLTZMANN_K * temperature_K * (math.log(aspect_ratio_p) + nu)
            / (3.0 * math.pi * viscosity_Pa_s * diffusion_coefficient_m2_per_s))


def rod_aspect_ratio_valid(aspect_ratio_p: float) -> bool:
    """True if p = L/d is in the Tirado (1984) fitted range 2 < p < 30."""
    return ROD_ASPECT_RATIO_MIN < aspect_ratio_p < ROD_ASPECT_RATIO_MAX


def sphere_rotational_diffusion(radius_m: float, temperature_K: float,
                                viscosity_Pa_s: float) -> float:
    """Rotational diffusion coefficient of a sphere, in rad^2/s (Stokes-Einstein-Debye).

        D_r = kT / (8 pi eta R^3)

    The rotational analog of Stokes-Einstein. Used by the depolarized analysis as
    the simple (single-unknown) shape model for an anisotropic NEAR-spherical
    particle (Balog et al. 2015).
    """
    if not _is_positive_finite(radius_m):
        raise ValueError(f"radius_m must be a positive finite number, got {radius_m!r}.")
    if not (_is_positive_finite(temperature_K) and _is_positive_finite(viscosity_Pa_s)):
        raise ValueError("temperature_K and viscosity_Pa_s must be positive finite "
                         "numbers.")
    return BOLTZMANN_K * temperature_K / (8.0 * math.pi * viscosity_Pa_s * radius_m ** 3)


def sphere_radius_from_rotational_diffusion(diffusion_coefficient_rad2_per_s: float,
                                            temperature_K: float,
                                            viscosity_Pa_s: float) -> float:
    """Sphere radius (m) from the rotational diffusion coefficient.

        R = ( kT / (8 pi eta D_r) )^(1/3)

    Inverse of sphere_rotational_diffusion. For a true sphere this equals the
    Stokes (translational) radius; comparing the two is a sphericity check.
    """
    if not _is_positive_finite(diffusion_coefficient_rad2_per_s):
        raise ValueError(
            f"D_r must be a positive finite number, "
            f"got {diffusion_coefficient_rad2_per_s!r}.")
    if not (_is_positive_finite(temperature_K) and _is_positive_finite(viscosity_Pa_s)):
        raise ValueError("temperature_K and viscosity_Pa_s must be positive finite "
                         "numbers.")
    return (BOLTZMANN_K * temperature_K
            / (8.0 * math.pi * viscosity_Pa_s
               * diffusion_coefficient_rad2_per_s)) ** (1.0 / 3.0)


def _require_positive_lengths(length_m, diameter_m, temperature_K, viscosity_Pa_s):
    if not (_is_positive_finite(length_m) and _is_positive_finite(diameter_m)):
        raise ValueError(
            f"length_m and diameter_m must be positive finite numbers, got "
            f"{length_m!r}, {diameter_m!r}.")
    if not (_is_positive_finite(temperature_K) and _is_positive_finite(viscosity_Pa_s)):
        raise ValueError("temperature_K and viscosity_Pa_s must be positive finite "
                         "numbers.")


# ---------------------------------------------------------------------------
# Ellipsoid of revolution (prolate/oblate) -- Perrin friction factors.
#
# Semi-axes: a = polar (symmetry-axis) semi-axis, b = equatorial radius. Axial
# ratio p = a/b (prolate p > 1, oblate p < 1, sphere p = 1). The Perrin
# expressions are naturally written in rho = b/a = 1/p; the equal-volume sphere
# radius is R0 = (a b^2)^(1/3). Callers work in p = a/b (parity with the rod);
# this module converts to rho internally.
#
# Perrin's dimensionless elliptic factor (Koenig 1975 Eqs. 6/7 with S0 = 2/a,
# aS = a*S; ln[(1+sqrt(1-rho^2))/rho] = artanh(sqrt(1-rho^2))):
#     prolate (rho<1):  aS = 2 artanh(sqrt(1-rho^2)) / sqrt(1-rho^2)
#     oblate  (rho>1):  aS = 2 atan (sqrt(rho^2-1)) / sqrt(rho^2-1)
#   aS -> 2 as rho -> 1 (sphere). Shared by BOTH friction factors below.
#
# Translational (Tanford 1961, Physical Chemistry of Macromolecules, Eqs.
#   19-14/19-15; Perrin 1936): f0 = 6 pi eta R0, f/f0 = 2/(rho^(2/3) aS), so
#     D_t = kT/f = kT / (6 pi eta R0 . 2/(rho^(2/3) aS)).
#   Reduces to Stokes-Einstein kT/(6 pi eta R) at a = b.
#
# Rotational -- end-over-end tumbling of the symmetry axis, the coefficient DDLS
#   observes (the spin about the symmetry axis is invisible). Koenig 1975
#   (Biopolymers 14, 2421, correcting Perrin 1934), friction about the equatorial
#   axis (his Eq. 9):
#     C2 = 4 C0 (1-rho^4) / {3 rho^2 [aS (2-rho^2) - 2]},  C0 = 8 pi eta a b^2
#     D_perp = kT/C2.
#   Koenig's sign correction is to the *equatorial-axis relaxation* tau_2, NOT to
#   C2/tau_1 (the tumbling), which Perrin already had right; C2 is used here as
#   Koenig restates it. Reduces to kT/(8 pi eta R^3) (Stokes-Einstein-Debye) at a=b.
#
# Validity band: Koenig tabulates/graphs 0.1 <= rho <= 10 (symmetric), i.e.
# 0.1 <= p <= 10; callers flag results outside it (like the rod's 2 < p < 30).

# Aspect-ratio (p = a/b) range over which Koenig (1975) tabulates the ellipsoid
# friction factors. A documented validity bound, not a hard domain limit.
ELLIPSOID_ASPECT_RATIO_MIN: float = 0.1
ELLIPSOID_ASPECT_RATIO_MAX: float = 10.0


def ellipsoid_aspect_ratio_valid(aspect_ratio_p: float) -> bool:
    """True if p = a/b is in Koenig's (1975) tabulated range 0.1 < p < 10."""
    return ELLIPSOID_ASPECT_RATIO_MIN < aspect_ratio_p < ELLIPSOID_ASPECT_RATIO_MAX


def _perrin_elliptic_aS(rho: float) -> float:
    """Perrin's dimensionless elliptic factor aS(rho) (= a*S; Koenig 1975 Eqs. 6/7).

        prolate (rho<1):  aS = 2 artanh(sqrt(1-rho^2)) / sqrt(1-rho^2)
        oblate  (rho>1):  aS = 2 atan (sqrt(rho^2-1)) / sqrt(rho^2-1)

    rho = b/a (equatorial/polar). Shared by the ellipsoid translational and
    rotational friction. Tends to 2 as rho -> 1 (sphere); the exact sphere is
    guarded (the closed forms are 0/0 there).
    """
    if not _is_positive_finite(rho):
        raise ValueError(
            f"axial ratio rho = b/a must be a positive finite number, got {rho!r}.")
    if abs(rho - 1.0) < 1e-9:
        return 2.0
    if rho < 1.0:                                   # prolate
        s = math.sqrt(1.0 - rho * rho)
        return 2.0 * math.atanh(s) / s
    s = math.sqrt(rho * rho - 1.0)                  # oblate
    return 2.0 * math.atan(s) / s


def _ellipsoid_f_ratio(rho: float) -> float:
    """Perrin translational friction ratio f/f0 = 2/(rho^(2/3) aS) of an ellipsoid of
    revolution (rho = b/a); 1 at the sphere (rho = 1). Shared by the forward D_t
    (ellipsoid_translational_diffusion) and its size inverse
    (ellipsoid_semi_axes_from_translational_diffusion) so the two stay EXACT inverses --
    a single edit to the friction form can never desynchronize them.
    """
    if abs(rho - 1.0) < 1e-9:
        return 1.0                                  # sphere: f = f0
    return 2.0 / (rho ** (2.0 / 3.0) * _perrin_elliptic_aS(rho))


def ellipsoid_translational_diffusion(polar_semi_axis_m: float,
                                      equatorial_radius_m: float,
                                      temperature_K: float,
                                      viscosity_Pa_s: float) -> float:
    """Translational diffusion coefficient D_t of an ellipsoid of revolution, m^2/s.

        D_t = kT / f,   f = 6 pi eta R0 . (f/f0),   R0 = (a b^2)^(1/3)
        f/f0 = 2 / (rho^(2/3) aS),   rho = b/a       (Tanford 1961 Eqs. 19-14/19-15)

    a = polar semi-axis, b = equatorial radius. Reduces to Stokes-Einstein
    kT/(6 pi eta R) at a = b (sphere).
    """
    a, b = polar_semi_axis_m, equatorial_radius_m
    _require_positive_lengths(a, b, temperature_K, viscosity_Pa_s)
    rho = b / a
    r0_m = (a * b * b) ** (1.0 / 3.0)
    f0 = 6.0 * math.pi * viscosity_Pa_s * r0_m
    return BOLTZMANN_K * temperature_K / (f0 * _ellipsoid_f_ratio(rho))


def ellipsoid_rotational_diffusion(polar_semi_axis_m: float,
                                   equatorial_radius_m: float,
                                   temperature_K: float,
                                   viscosity_Pa_s: float) -> float:
    """End-over-end rotational diffusion coefficient D_perp of an ellipsoid, rad^2/s.

        D_perp = kT / C2,   C0 = 8 pi eta a b^2,   rho = b/a
        C2 = 4 C0 (1 - rho^4) / {3 rho^2 [aS (2 - rho^2) - 2]}
        (Koenig 1975 Eq. 9, correcting Perrin 1934)

    This is the tumbling of the symmetry axis -- the rotation DDLS observes (the
    spin about the symmetry axis is not seen). Reduces to the sphere value
    kT/(8 pi eta R^3) at a = b. Near rho = 1 the closed form is 0/0 (both the
    numerator 1-rho^4 and the bracket vanish); the sphere value is used there.
    """
    a, b = polar_semi_axis_m, equatorial_radius_m
    _require_positive_lengths(a, b, temperature_K, viscosity_Pa_s)
    rho = b / a
    c0 = 8.0 * math.pi * viscosity_Pa_s * a * b * b
    if abs(rho - 1.0) < 1e-4:
        c2 = c0                                     # sphere limit (avoids the 0/0)
    else:
        a_s = _perrin_elliptic_aS(rho)
        c2 = (4.0 * c0 * (1.0 - rho ** 4)
              / (3.0 * rho * rho * (a_s * (2.0 - rho * rho) - 2.0)))
    return BOLTZMANN_K * temperature_K / c2


def ellipsoid_semi_axes_from_translational_diffusion(
        diffusion_coefficient_m2_per_s: float, aspect_ratio_p: float,
        temperature_K: float, viscosity_Pa_s: float) -> tuple:
    """Semi-axes (a, b) in m from D_t at a KNOWN axial ratio p = a/b.

    Inverse of ellipsoid_translational_diffusion for fixed p: D_t fixes the total
    friction f = kT/D_t, hence the equal-volume-sphere radius R0 = f/(6 pi eta .
    f/f0); then a = R0 p^(2/3), b = R0 p^(-1/3) (so a/b = p, a b^2 = R0^3). The
    depolarized ellipsoid inversion uses it to reduce the 2-D (a, b) solve to a
    1-D root find in p.
    """
    if not _is_positive_finite(diffusion_coefficient_m2_per_s):
        raise ValueError(
            f"D_t must be a positive finite number, "
            f"got {diffusion_coefficient_m2_per_s!r}.")
    if not _is_positive_finite(aspect_ratio_p):
        raise ValueError(
            f"aspect ratio p must be a positive finite number, got {aspect_ratio_p!r}.")
    if not (_is_positive_finite(temperature_K) and _is_positive_finite(viscosity_Pa_s)):
        raise ValueError("temperature_K and viscosity_Pa_s must be positive finite "
                         "numbers.")
    rho = 1.0 / aspect_ratio_p
    f = BOLTZMANN_K * temperature_K / diffusion_coefficient_m2_per_s
    r0_m = f / (6.0 * math.pi * viscosity_Pa_s * _ellipsoid_f_ratio(rho))
    return (r0_m * aspect_ratio_p ** (2.0 / 3.0),
            r0_m * aspect_ratio_p ** (-1.0 / 3.0))
