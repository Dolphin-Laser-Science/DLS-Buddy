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
            isotropic factor for vertically polarised incident light
            (R_iso = R_VV (1 - 4/3 rho_v); Chu 1991 Sec. 8.4.1.A) and for
            natural light ((6-7 rho_u)/(6+6 rho_u); Coumou et al. 1964), and the
            optical-anisotropy parameter delta^2 = 5 rho_v / (3 - 4 rho_v).
"""

import math
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


# ---------------------------------------------------------------------------
# Dynamic light scattering — Stokes-Einstein equation
# ---------------------------------------------------------------------------

def stokes_einstein_rh(
    diffusion_coefficient_m2_per_s: float,
    temperature_K: float,
    viscosity_Pa_s: float,
) -> float:
    """Hydrodynamic radius Rh from the Stokes-Einstein equation, in metres.

    Definition:
        Rh = k_B * T / (6 * pi * eta * D)

    Standard result (the Stokes-Einstein relation); see Chu (1991).

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
        If any argument is non-positive (all three must be strictly
        positive physical quantities).
    """
    if not (diffusion_coefficient_m2_per_s > 0):
        raise ValueError(
            f"diffusion_coefficient_m2_per_s must be positive, "
            f"got {diffusion_coefficient_m2_per_s!r}."
        )
    if not (temperature_K > 0):
        raise ValueError(
            f"temperature_K must be positive, got {temperature_K!r}."
        )
    if not (viscosity_Pa_s > 0):
        raise ValueError(
            f"viscosity_Pa_s must be positive, got {viscosity_Pa_s!r}."
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
        If any argument is non-positive.
    """
    if not (rh_m > 0):
        raise ValueError(f"rh_m must be positive, got {rh_m!r}.")
    if not (temperature_K > 0):
        raise ValueError(f"temperature_K must be positive, got {temperature_K!r}.")
    if not (viscosity_Pa_s > 0):
        raise ValueError(f"viscosity_Pa_s must be positive, got {viscosity_Pa_s!r}.")
    return BOLTZMANN_K * temperature_K / (6.0 * math.pi * viscosity_Pa_s * rh_m)


# ---------------------------------------------------------------------------
# Static light scattering — toluene Rayleigh ratio (geometry-aware)
# ---------------------------------------------------------------------------
#
# The Rayleigh ratio of toluene depends on scattering GEOMETRY (the polarisation
# of the incident and detected light) because toluene is optically anisotropic.
# For a vertically polarised laser the three relevant geometries are:
#   VV  vertical incident, vertical analyser           R_VV
#   VU  vertical incident, NO analyser (e.g. BI-200SM) R_VU = R_VV (1 + rho_v)
#   VH  vertical incident, horizontal analyser         R_VH = R_VV rho_v
# where rho_v = I_VH/I_VV is the (vertical) depolarisation ratio. These follow
# from R_VU = V_v + H_v with V_v = R_VV and H_v = R_VH = rho_v R_VV
# (Brookhaven BI-200SM manual Sec. VIII; Wu 2010). The geometry MUST match the
# geometry in which the calibrant intensity was measured, or the absolute scale
# is wrong by a factor of (1 + rho_v) ~ 1.35 for toluene.
#
# Base values are the POLARISED ratio R_VV at 25 °C:
#   532 nm: 2.34e-5 cm^-1  -- Takahashi, Takano, Kinugasa & Sakurai,
#           Anal. Sci. 2019, 35, 1045 (metrology redetermination via certified
#           reference polymers; the authoritative 532 nm value).
#   660 nm: 8.456e-6 cm^-1 -- Sivokhin & Kazantsev, ChemistrySelect 2021, 6,
#           9499 (Table 2). Also the source of the depolarisation and temperature
#           data below.
# Other wavelengths can be added with a properly sourced R_VV; do not interpolate
# (the wavelength dependence is steep, ~lambda^-4.2; Wu, Chem. Phys. 2010).
_TOLUENE_RVV_25C: dict = {
    532.0: 2.34e-5,    # cm^-1, Takahashi et al. 2019
    660.0: 8.456e-6,   # cm^-1, Sivokhin & Kazantsev 2021, Table 2
}

# Temperature coefficient of R_VV: fractional change per °C. POSITIVE -- the
# Rayleigh ratio rises with temperature (Sivokhin & Kazantsev 2021, Table 2:
# Rvv from 8.00e-6 at 10 °C to 9.41e-6 at 50 °C, ~+0.43 %/°C; consistent with
# benzene's well-known +0.368 %/°C, Wu 2010 Eq. 3). S&K assume the same
# fractional coefficient at all wavelengths.
_TOLUENE_RVV_TEMP_COEFFICIENT: float = 0.0043   # per °C  (relative, positive)

# Vertical depolarisation ratio of toluene, rho_v = I_VH / I_VV.
# Sivokhin & Kazantsev 2021, Table 1: 0.346 at 25 °C (measured at 660 nm),
# decreasing ~linearly with temperature (0.364 at 10 °C to 0.310 at 50 °C).
# rho_v is only weakly wavelength-dependent; this 660 nm value is used at other
# wavelengths unless the user supplies one. It is the key uncertain parameter for
# converting between geometries.
_TOLUENE_DEPOL_V_25C: float = 0.346
_TOLUENE_DEPOL_V_TEMP_COEFFICIENT: float = -0.00135   # per °C  (absolute)

_TOLUENE_RAYLEIGH_REF_TEMP_C: float = 25.0
_WAVELENGTH_MATCH_TOLERANCE_NM: float = 2.0

_RAYLEIGH_GEOMETRIES = ('VV', 'VU', 'VH')


def rayleigh_ratio_toluene(
    wavelength_nm: float,
    temperature_C: float,
    geometry: str = 'VV',
    depolarization_ratio_v: Optional[float] = None,
) -> float:
    """Rayleigh ratio of toluene in cm^-1, for a chosen scattering geometry.

    Computed from the polarised value R_VV (Takahashi et al. 2019 at 532 nm;
    Sivokhin & Kazantsev 2021 at 660 nm), temperature-corrected, then converted
    to the requested geometry using toluene's depolarisation ratio. This is the
    value to use for SLS calibration; the value embedded in instrument software
    must NOT be used in its place.

    Geometry conversions (vertically polarised incident light):
        VV : R_VV                       (vertical analyser)
        VU : R_VV * (1 + rho_v)         (NO analyser; e.g. the BI-200SM)
        VH : R_VV * rho_v               (horizontal analyser)
    The geometry MUST match the geometry in which the calibrant intensity was
    measured; otherwise the absolute scale is wrong by ~(1 + rho_v) ~ 1.35.

    Parameters
    ----------
    wavelength_nm : float
        Laser vacuum wavelength in nm. Supported: 532 and 660 nm (+/- 2 nm).
    temperature_C : float
        Measurement temperature in degrees Celsius.
    geometry : str
        'VV' (default), 'VU', or 'VH'. Use 'VU' for an instrument with a
        vertically polarised laser and no polarisation analyser.
    depolarization_ratio_v : float, optional
        rho_v = I_VH / I_VV for toluene. If omitted, the temperature-dependent
        value from Sivokhin & Kazantsev 2021 is used (0.346 at 25 C). Supply your
        own if you have measured it, especially at wavelengths far from 660 nm.

    Returns
    -------
    float
        Rayleigh ratio of toluene in cm^-1 for the requested geometry.

    Raises
    ------
    ValueError
        If the wavelength is unsupported or the geometry is not VV/VU/VH.

    Notes
    -----
    R_VV(T) = R_VV(25) * (1 + alpha (T - 25)), alpha = +0.0043 /C (S&K Table 2;
    the coefficient is POSITIVE -- the Rayleigh ratio rises with temperature).
    rho_v(T) = 0.346 - 0.00135 (T - 25) (S&K Table 1) unless overridden.
    Verified to reproduce the S&K Rv (VU) table at 10/25/50 C.

    For wavelengths not in the table, add a properly sourced R_VV to
    _TOLUENE_RVV_25C; do not interpolate (R ~ lambda^-4.2).
    """
    geometry = geometry.upper()
    if geometry not in _RAYLEIGH_GEOMETRIES:
        raise ValueError(
            f"geometry must be one of {_RAYLEIGH_GEOMETRIES}, got {geometry!r}."
        )

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

    # Temperature-corrected polarised value R_VV(T).
    dT = temperature_C - _TOLUENE_RAYLEIGH_REF_TEMP_C
    r_vv = _TOLUENE_RVV_25C[best_match] * (1.0 + _TOLUENE_RVV_TEMP_COEFFICIENT * dT)

    if geometry == 'VV':
        return r_vv

    # Depolarisation ratio (temperature-dependent default, or user-supplied).
    if depolarization_ratio_v is None:
        rho_v = _TOLUENE_DEPOL_V_25C + _TOLUENE_DEPOL_V_TEMP_COEFFICIENT * dT
    else:
        rho_v = float(depolarization_ratio_v)

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

    The (n/n_std)^2 form is the standard correction for a cylindrical
    scattering cell. Some instruments use a more elaborate form that also
    accounts for the refractive index of the cell walls; this simpler
    form is appropriate for most goniometer setups including the
    Brookhaven instruments.
    """
    if not (standard_refractive_index > 0):
        raise ValueError(
            f"standard_refractive_index must be positive, "
            f"got {standard_refractive_index!r}."
        )
    if not (solvent_refractive_index > 0):
        raise ValueError(
            f"solvent_refractive_index must be positive, "
            f"got {solvent_refractive_index!r}."
        )
    return (solvent_refractive_index / standard_refractive_index) ** 2


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
    (vertically polarised incident and detected) geometry, which is the
    default for most modern light scattering instruments. If your
    instrument uses a different polarisation geometry, K must be modified.
    """
    wavelength_cm = wavelength_nm * 1.0e-7   # nm -> cm
    numerator = 4.0 * math.pi ** 2 * solvent_refractive_index ** 2 * dn_dc_mL_per_g ** 2
    denominator = AVOGADRO_NA * wavelength_cm ** 4
    return numerator / denominator


# ---------------------------------------------------------------------------
# Depolarized light scattering (static) -- depolarization ratios, the Cabannes
# factor, and the optical-anisotropy parameter
# ---------------------------------------------------------------------------
#
# An optically anisotropic scatterer (a molecule or particle whose polarisability
# is not the same in every direction) rotates the polarisation of some of the
# light it scatters. With VERTICALLY polarised incident light -- the default for
# essentially every modern instrument -- this shows up as a horizontally polarised
# ("depolarised") component in the scattered light. Two quantities describe it:
#
#   rho_v = I_VH / I_VV   depolarisation ratio, vertically polarised incident light
#                         (I_VH = horizontal analyser, I_VV = vertical analyser).
#                         This is the quantity modern instruments measure and the
#                         one physics/constants.py already carries for toluene.
#   rho_u = I_h  / I_v    depolarisation ratio, UNPOLARISED (natural) incident
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
# wants. For vertically polarised incident light the factor is 1 - (4/3) rho_v
# (derived below); for natural light it is the classic (6 - 7 rho_u)/(6 + 6 rho_u).
#
# Physical range: for vertically polarised incident light rho_v lies in [0, 3/4].
# rho_v = 0 is an optically isotropic scatterer (no depolarisation, no
# correction). The upper limit rho_v = 3/4 is the fully-anisotropic limit, where
# the isotropic part vanishes (delta^2 -> infinity, Cabannes factor -> 0). Small
# flexible polymer coils sit very near 0; small anisotropic molecules like toluene
# are ~0.3-0.35; rigid rods and many particles fall in between.

# Largest physically meaningful rho_v for vertically polarised incident light.
# At this value the isotropic scattering vanishes (see module notes above).
_RHO_V_DEPOLARIZATION_LIMIT: float = 0.75


def depolarization_ratio_unpolarized(rho_v: float) -> float:
    """Unpolarised depolarisation ratio rho_u from the vertical ratio rho_v.

    Definition:
        rho_u = 2 rho_v / (1 + rho_v)

    Converts a measured vertical-incident depolarisation ratio (what modern
    instruments report) to the natural/unpolarised-incident depolarisation ratio
    used throughout the classical light-scattering literature (Coumou, Cabannes).
    The relation is exact, not an approximation.

    Parameters
    ----------
    rho_v : float
        Depolarisation ratio for vertically polarised incident light,
        rho_v = I_VH / I_VV. Must satisfy 0 <= rho_v <= 3/4 (the physical range;
        see module notes).

    Returns
    -------
    float
        Depolarisation ratio rho_u for unpolarised incident light.

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
            f"polarised incident light, got {rho_v!r}."
        )
    return 2.0 * rho_v / (1.0 + rho_v)


def depolarization_ratio_vertical(rho_u: float) -> float:
    """Vertical depolarisation ratio rho_v from the unpolarised ratio rho_u.

    Inverse of depolarization_ratio_unpolarized():
        rho_v = rho_u / (2 - rho_u)

    Use this to read a classical-literature rho_u (natural-light) value into the
    vertical-incident convention the instrument and the rest of this code use.

    Parameters
    ----------
    rho_u : float
        Depolarisation ratio for unpolarised incident light. Must satisfy
        0 <= rho_u <= 6/7 (the image of [0, 3/4] under the forward relation).

    Returns
    -------
    float
        Depolarisation ratio rho_v for vertically polarised incident light.

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
    """Cabannes isotropic factor for VERTICALLY polarised incident light.

    Definition:
        f = 1 - (4/3) rho_v

    Multiply a measured VV excess Rayleigh ratio by this factor to recover the
    ISOTROPIC excess Rayleigh ratio (the part that carries Mw / Rg / A2):
        R_iso = R_VV * (1 - (4/3) rho_v).

    Derivation (Chu 1991, Sec. 8.4.1.A). With vertically polarised incident
    light the apparent molecular weight and the depolarised Rayleigh ratio are
        Mw_app    = Mw (1 + (4/5) delta^2)            (Chu Eq. 8.4.8)
        R_HV/(Hc) = (3/5) delta^2 Mw     (c,K -> 0)   (Chu Eq. 8.4.10)
    so rho_v = R_HV / R_VV = (3/5) delta^2 / (1 + (4/5) delta^2). Eliminating
    delta^2 gives R_VV (1 - (4/3) rho_v) = R_iso exactly. Equivalently, the
    per-molecule optics I_VV ~ abar^2 + (4/45) gamma^2, I_VH ~ (3/45) gamma^2
    give the same 4/3 = (4/45)/(3/45) (Kerker; Berne & Pecora 1976) -- the two
    routes agree.

    Parameters
    ----------
    rho_v : float
        Vertical-incident depolarisation ratio, 0 <= rho_v <= 3/4.

    Returns
    -------
    float
        Isotropic factor f in [0, 1]. f = 1 when rho_v = 0 (no anisotropy, no
        correction); f -> 0 at the depolarisation limit rho_v = 3/4.

    Raises
    ------
    ValueError
        If rho_v is outside [0, 3/4].

    Notes
    -----
    This is the vertically-polarised-incident analogue of the classical natural-
    light Cabannes factor (6 - 7 rho_u)/(6 + 6 rho_u); see
    cabannes_isotropic_factor_natural(). For small rho_v the two agree to first
    order. The anisotropic FRACTION removed is 1 - f = (4/3) rho_v.
    """
    if not (0.0 <= rho_v <= _RHO_V_DEPOLARIZATION_LIMIT):
        raise ValueError(
            f"rho_v must be in [0, {_RHO_V_DEPOLARIZATION_LIMIT}], got {rho_v!r}."
        )
    return 1.0 - (4.0 / 3.0) * rho_v


def cabannes_isotropic_factor_natural(rho_u: float) -> float:
    """Cabannes isotropic factor for UNPOLARISED (natural) incident light.

    Definition (the classical Cabannes factor):
        f = (6 - 7 rho_u) / (6 + 6 rho_u)

    Multiply a total Rayleigh ratio measured with unpolarised incident light at
    90 degrees by this factor to recover the isotropic part:
        R_iso = R_total * (6 - 7 rho_u) / (6 + 6 rho_u).

    Provided for completeness and for reading older (natural-light) datasets;
    modern vertically-polarised instruments use cabannes_isotropic_factor().

    Parameters
    ----------
    rho_u : float
        Unpolarised-incident depolarisation ratio, 0 <= rho_u <= 6/7.

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
    """Optical-anisotropy parameter delta^2 from the vertical depolarisation ratio.

    Definition (inverting Chu 1991 rho_v = (3/5) delta^2 / (1 + (4/5) delta^2)):
        delta^2 = 5 rho_v / (3 - 4 rho_v)

    delta^2 is the mean-square optical anisotropy in Chu's parametrisation: 0 for
    an optically isotropic scatterer, growing without bound as rho_v -> 3/4.

    Parameters
    ----------
    rho_v : float
        Vertical-incident depolarisation ratio, 0 <= rho_v < 3/4. (rho_v = 3/4 is
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
    This is Chu's delta^2; other texts use a differently normalised anisotropy
    (e.g. gamma^2 with the per-molecule I_VV ~ abar^2 + (4/45) gamma^2 form).
    Convert via the depolarisation ratio, which is convention-independent, not by
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
# direction). The depolarised dynamic analysis runs them BACKWARDS -- given the
# measured D_t and D_r, solve for the dimensions -- which is a model-dependent
# inverse problem and lives in analysis/depolarization.py. These forward formulas
# (and their algebraic inverses) are the only physics; the solver only calls them.
#
# Rigid circular cylinder (length L, diameter d, axial ratio p = L/d):
#   Tirado, Lopez Martinez & Garcia de la Torre, J. Chem. Phys. 81, 2047 (1984),
#   Eqs. (1),(5),(9),(10). The end-effect correction polynomials are fitted for
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
    if not (p > 0):
        raise ValueError(f"aspect ratio p must be positive, got {p!r}.")
    nu = 0.312 + 0.565 / p - 0.100 / p ** 2
    delta_perp = -0.662 + 0.917 / p - 0.050 / p ** 2
    return nu, delta_perp


def rod_translational_diffusion(length_m: float, diameter_m: float,
                                temperature_K: float, viscosity_Pa_s: float) -> float:
    """Translational diffusion coefficient D_t of a rigid cylinder, in m^2/s.

        D_t = kT / (3 pi eta L) * (ln p + nu),   p = L/d   (Tirado 1984 Eqs. 1, 9)

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

    The algebraic inverse of rod_translational_diffusion for fixed p. The depolarised
    rod inversion uses it to reduce the 2-D (L, d) solve to a 1-D root find in p.
    """
    if not (diffusion_coefficient_m2_per_s > 0):
        raise ValueError(
            f"D_t must be positive, got {diffusion_coefficient_m2_per_s!r}.")
    if not (temperature_K > 0 and viscosity_Pa_s > 0):
        raise ValueError("temperature_K and viscosity_Pa_s must be positive.")
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

    The rotational analogue of Stokes-Einstein. Used by the depolarised analysis as
    the simple (single-unknown) shape model for an anisotropic NEAR-spherical
    particle (Balog et al. 2015).
    """
    if not (radius_m > 0):
        raise ValueError(f"radius_m must be positive, got {radius_m!r}.")
    if not (temperature_K > 0 and viscosity_Pa_s > 0):
        raise ValueError("temperature_K and viscosity_Pa_s must be positive.")
    return BOLTZMANN_K * temperature_K / (8.0 * math.pi * viscosity_Pa_s * radius_m ** 3)


def sphere_radius_from_rotational_diffusion(diffusion_coefficient_rad2_per_s: float,
                                            temperature_K: float,
                                            viscosity_Pa_s: float) -> float:
    """Sphere radius (m) from the rotational diffusion coefficient.

        R = ( kT / (8 pi eta D_r) )^(1/3)

    Inverse of sphere_rotational_diffusion. For a true sphere this equals the
    Stokes (translational) radius; comparing the two is a sphericity check.
    """
    if not (diffusion_coefficient_rad2_per_s > 0):
        raise ValueError(
            f"D_r must be positive, got {diffusion_coefficient_rad2_per_s!r}.")
    if not (temperature_K > 0 and viscosity_Pa_s > 0):
        raise ValueError("temperature_K and viscosity_Pa_s must be positive.")
    return (BOLTZMANN_K * temperature_K
            / (8.0 * math.pi * viscosity_Pa_s
               * diffusion_coefficient_rad2_per_s)) ** (1.0 / 3.0)


def _require_positive_lengths(length_m, diameter_m, temperature_K, viscosity_Pa_s):
    if not (length_m > 0 and diameter_m > 0):
        raise ValueError(
            f"length_m and diameter_m must be positive, got {length_m!r}, "
            f"{diameter_m!r}.")
    if not (temperature_K > 0 and viscosity_Pa_s > 0):
        raise ValueError("temperature_K and viscosity_Pa_s must be positive.")
