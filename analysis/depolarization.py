"""
analysis/depolarization.py
==========================

Static depolarized light scattering (DPLS) analysis: the depolarization ratio,
the Cabannes isotropic/anisotropic split, and the optical-anisotropy parameter,
from a paired VV (polarized) and VH (depolarized) intensity measurement.

This is Phase 1 of the depolarized-light-scattering module (CLAUDE.md -> Planned
-> DPLS). It is the STATIC side: it works from time-averaged intensities, not
correlograms. The DYNAMIC side (rotational diffusion D_r from the VV/VH
correlation-function decay rates) is a later phase and lives elsewhere.

What it computes
----------------
Given the polarized (VV) and depolarized (VH) scattered intensities measured with
VERTICALLY polarised incident light -- the geometry of essentially every modern
instrument:

  rho_v   = I_VH / I_VV                     depolarization ratio (the primary obs.)
  rho_u   = 2 rho_v / (1 + rho_v)           natural-light equivalent (for the
                                            classical literature)
  delta^2 = 5 rho_v / (3 - 4 rho_v)         optical-anisotropy parameter (Chu 1991)
  f       = 1 - (4/3) rho_v                 Cabannes isotropic factor; R_iso =
                                            R_VV * f removes the anisotropy-inflated
                                            part of the Rayleigh ratio so Mw/Rg are
                                            not overestimated.

All four physical relations live in physics/constants.py (every equation with
physical meaning lives there); this module is the analysis layer that consumes
intensities, applies validity flags and provenance, and returns a result object.

Convention (pinned here, used everywhere)
-----------------------------------------
Two-letter subscripts are (incident, analyser): VV = vertical incident + vertical
analyser; VH = vertical incident + horizontal analyser. This matches Sivokhin &
Kazantsev (2021) and physics/constants.py's Rayleigh-geometry codes. NOTE: Chu
(1991) writes the depolarised ratio as R_HV with the OPPOSITE letter order
(scattered, incident) -- it is the same physical quantity as our I_VH. We use the
(incident, analyser) order throughout.

Uncertainty (per CLAUDE.md invariant 8)
---------------------------------------
A single VV/VH intensity pair gives NO uncertainty on rho_v -- consistent with the
project rule that a single measurement does not carry a statistical SE. When
replicate intensities are available (so an SE on I_VV and I_VH can be formed), the
SE on rho_v is propagated as a ratio of independent quantities (uncertainty.ratio_se,
delta method). delta^2 and the Cabannes factor then propagate from rho_v by the
delta method. No SE is invented from one shot.

References (in project knowledge)
---------------------------------
  Chu 1991, Laser Light Scattering, 2nd ed., Sec. 8.4.1.A (Eqs. 8.4.7-8.4.10):
      vertically-polarised-incident depolarization correction; delta^2.
  Coumou, Mackor & Hijmans 1964 (natural-light Cabannes factor; via constants.py).
  Sivokhin & Kazantsev 2021 (toluene rho_v; rho_u <-> rho_v; via constants.py).

Design contract
---------------
Every function is PURE: it takes intensities (and parameters) and returns a result
object or a number. No plotting, no file I/O, no mutation of inputs.

Change history
--------------
2026-06-19  Phase 1: static depolarization. (depolarization.py v1)
            depolarization_ratio, analyze_depolarization (DepolarizationResult),
            isotropic_rayleigh_ratio. Validated headless against Coumou Table 3
            and Sivokhin & Kazantsev Table 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from scipy import optimize

from analysis import uncertainty as unc
from physics import constants as phys
from physics.constants import (
    depolarization_ratio_unpolarized,
    cabannes_isotropic_factor,
    optical_anisotropy_squared,
    stokes_einstein_rh,
    _RHO_V_DEPOLARIZATION_LIMIT,
)


# ===========================================================================
# Result object
# ===========================================================================

@dataclass
class DepolarizationResult:
    """Static depolarization analysis from one VV/VH intensity pair.

    All quantities assume VERTICALLY polarised incident light (the modern default).
    `physically_valid` is False when rho_v falls outside [0, 3/4]; in that case the
    derived quantities are clamped/NaN and `note` explains why, rather than the
    analysis raising -- a slightly-out-of-range rho_v is usually instrumental
    (stray depolarised light from polariser/analyser leakage, reflections, or an
    over-subtracted dark count), which is a data-quality flag, not a crash.
    """
    # primary observable
    rho_v: float                       # I_VH / I_VV (vertical incident)
    rho_u: float                       # 2 rho_v / (1 + rho_v) (natural-light equiv.)
    # derived
    optical_anisotropy_sq: float       # delta^2 (Chu 1991); nan if rho_v >= 3/4
    cabannes_isotropic_factor: float   # f = 1 - (4/3) rho_v; R_iso = R_VV * f
    anisotropic_fraction: float        # 1 - f = (4/3) rho_v
    # inputs / provenance
    i_vv: float                        # VV intensity used (after dark-count subtraction)
    i_vh: float                        # VH intensity used (after dark-count subtraction)
    dark_count: float                  # dark count subtracted from each
    # quality
    physically_valid: bool             # 0 <= rho_v <= 3/4
    note: str = ''
    # uncertainty (None unless replicate SEs were supplied)
    rho_v_se: Optional[float] = None


# ===========================================================================
# Core analysis
# ===========================================================================

def depolarization_ratio(i_vv: float, i_vh: float, dark_count: float = 0.0) -> float:
    """Depolarization ratio rho_v = I_VH / I_VV for vertically polarised incident light.

    Parameters
    ----------
    i_vv : float
        Polarized (vertical analyser) scattered intensity, vertical incident.
    i_vh : float
        Depolarized (horizontal analyser) scattered intensity, vertical incident.
    dark_count : float, optional
        Detector dark count / baseline subtracted from BOTH intensities before the
        ratio (default 0). The depolarised channel is weak, so an un-subtracted
        dark count biases rho_v high; subtract it when known.

    Returns
    -------
    float
        rho_v = (I_VH - dark) / (I_VV - dark).

    Raises
    ------
    ValueError
        If the dark-subtracted I_VV is not strictly positive (the ratio would be
        undefined or negative).
    """
    vv = i_vv - dark_count
    vh = i_vh - dark_count
    if not (vv > 0):
        raise ValueError(
            f"dark-subtracted I_VV must be positive, got {vv!r} "
            f"(I_VV={i_vv!r}, dark_count={dark_count!r})."
        )
    return vh / vv


def analyze_depolarization(
    i_vv: float,
    i_vh: float,
    *,
    dark_count: float = 0.0,
    i_vv_se: Optional[float] = None,
    i_vh_se: Optional[float] = None,
) -> DepolarizationResult:
    """Full static depolarization analysis from one VV/VH intensity pair.

    Computes rho_v and, from it, the natural-light ratio rho_u, the optical
    anisotropy delta^2, and the Cabannes isotropic factor f (R_iso = R_VV * f).
    Assumes vertically polarised incident light.

    Parameters
    ----------
    i_vv, i_vh : float
        Polarized (VV) and depolarized (VH) scattered intensities.
    dark_count : float, optional
        Dark count subtracted from both intensities (default 0).
    i_vv_se, i_vh_se : float, optional
        Standard errors on I_VV and I_VH (e.g. from replicate measurements). If
        BOTH are given, an SE on rho_v is propagated (ratio of independent
        quantities, uncertainty.ratio_se). If either is None, rho_v_se is None --
        a single shot carries no statistical uncertainty (CLAUDE.md invariant 8).

    Returns
    -------
    DepolarizationResult

    Notes
    -----
    rho_v outside [0, 3/4] does not raise: it sets physically_valid=False, records
    why in `note`, and leaves delta^2 = nan (and the Cabannes factor still computed
    from the linear form, which stays finite). rho_v <= 0 means no resolvable
    depolarisation; rho_v > 3/4 is unphysical for an anisotropic scatterer and
    signals instrumental stray depolarised light.
    """
    rho_v = depolarization_ratio(i_vv, i_vh, dark_count=dark_count)
    vv = i_vv - dark_count
    vh = i_vh - dark_count

    # Validity against the physical range [0, 3/4] for vertical incident light.
    physically_valid = (0.0 <= rho_v <= _RHO_V_DEPOLARIZATION_LIMIT)
    notes = []

    # Natural-light equivalent and optical anisotropy. Guard the closed forms,
    # which are only defined on the physical range, so an out-of-range rho_v
    # degrades gracefully to a flag instead of a raised exception.
    if 0.0 <= rho_v <= _RHO_V_DEPOLARIZATION_LIMIT:
        rho_u = depolarization_ratio_unpolarized(rho_v)
    else:
        rho_u = float('nan')

    if 0.0 <= rho_v < _RHO_V_DEPOLARIZATION_LIMIT:
        delta_sq = optical_anisotropy_squared(rho_v)
    else:
        delta_sq = float('nan')
        if rho_v >= _RHO_V_DEPOLARIZATION_LIMIT:
            notes.append(
                f"rho_v = {rho_v:.4g} is at/above the depolarisation limit 3/4; the "
                f"isotropic scattering vanishes and delta^2 diverges -- check for "
                f"stray depolarised light (polariser/analyser leakage, reflections)."
            )
    if rho_v < 0.0:
        notes.append(
            f"rho_v = {rho_v:.4g} is negative -- I_VH below the dark count or an "
            f"over-subtracted baseline; no resolvable depolarisation."
        )

    # Cabannes factor uses the linear form 1 - (4/3) rho_v, finite for any rho_v;
    # clamp the call to the valid range so it does not raise, but report the raw
    # (possibly >1 or <0) factor for transparency when invalid.
    if 0.0 <= rho_v <= _RHO_V_DEPOLARIZATION_LIMIT:
        f_iso = cabannes_isotropic_factor(rho_v)
    else:
        f_iso = 1.0 - (4.0 / 3.0) * rho_v  # raw linear value, outside guaranteed [0,1]
    anisotropic_fraction = 1.0 - f_iso

    # Uncertainty: only when both intensity SEs are supplied (replicates). The
    # dark count is treated as exact here; rho_v = vh/vv with independent vh, vv.
    rho_v_se = unc.ratio_se(vh, i_vh_se, vv, i_vv_se)

    if not physically_valid and not notes:
        notes.append(f"rho_v = {rho_v:.4g} outside the physical range [0, 3/4].")

    return DepolarizationResult(
        rho_v=rho_v,
        rho_u=rho_u,
        optical_anisotropy_sq=delta_sq,
        cabannes_isotropic_factor=f_iso,
        anisotropic_fraction=anisotropic_fraction,
        i_vv=vv,
        i_vh=vh,
        dark_count=dark_count,
        physically_valid=physically_valid,
        note=' '.join(notes),
        rho_v_se=rho_v_se,
    )


def isotropic_rayleigh_ratio(r_vv: float, rho_v: float) -> float:
    """Isotropic excess Rayleigh ratio from the measured VV ratio (Cabannes correction).

    R_iso = R_VV * (1 - (4/3) rho_v)    [vertically polarised incident light]

    This is the value the Zimm / Debye / Berry Mw-Rg-A2 analysis should use when the
    scatterer is optically anisotropic: it strips the depolarised contribution that
    otherwise inflates the apparent molecular weight (Mw_app = Mw (1 + (4/5) delta^2);
    Chu 1991 Eq. 8.4.8). For an isotropic scatterer rho_v = 0 and R_iso = R_VV.

    Parameters
    ----------
    r_vv : float
        Measured VV excess Rayleigh ratio (any units; the factor is dimensionless).
    rho_v : float
        Vertical-incident depolarisation ratio, 0 <= rho_v <= 3/4.

    Returns
    -------
    float
        Isotropic excess Rayleigh ratio R_iso, same units as r_vv.

    Raises
    ------
    ValueError
        If rho_v is outside [0, 3/4] (cabannes_isotropic_factor enforces this).
    """
    return r_vv * cabannes_isotropic_factor(rho_v)


# ===========================================================================
# Dynamic DDLS -- rotational diffusion from the VV/VH correlogram decay rates
# ===========================================================================
#
# For an optically anisotropic scatterer in the small-qL regime the FIELD (g1)
# correlation decay rates are (Zero & Pecora 1982, Eqs. III.1/III.2; the
# rotational 6*Theta term from Pecora 1964):
#
#     Gamma_VV = q^2 D_t                 (polarised; translation only)
#     Gamma_VH = q^2 D_t + 6 D_r         (depolarised; + rotational diffusion)
#
# so the rotational diffusion coefficient is D_r = (Gamma_VH - Gamma_VV) / 6, and
# a multi-angle plot of Gamma_VV vs q^2 gives D_t (slope, through the origin).
#
# IMPORTANT: Gamma here is the FIELD decay rate (the g1 rate). A homodyne intensity
# correlogram g2(tau)-1 = beta exp(-2 Gamma tau) decays at 2*Gamma; the cumulant
# fitter (analysis.dls.fit_cumulants) already returns the field rate Gamma, so feed
# THAT in -- do not pass an intensity (2*Gamma) rate, or D_r and D_t come out 2x.
#
# Recovery strategy (see test-data/Synthetic DPLS/parameters.txt): take D_t from
# the VV channel (Gamma_VV = q^2 D_t is the whole VV signal, cleanly measured), and
# D_r from the per-angle difference (Gamma_VH - Gamma_VV)/6. Fitting D_t as the
# SLOPE of Gamma_VH vs q^2 is ill-conditioned when rotation dominates (the slope is
# tiny next to the 6 D_r intercept), so it is avoided here.
#
# Uncertainty (CLAUDE.md invariant 8): a SINGLE angle (one VV + one VH correlogram)
# gives NO standard error -- each Gamma is from a single correlogram (Schaetzel
# 1990). Across MULTIPLE angles the points are independent measurements, so the
# through-origin regression SE on D_t and the spread (SD/sqrt(n)) of the per-angle
# D_r are legitimate ensemble uncertainties (the same basis as the multi-angle
# Zimm SE and the ISO 22412 replicate SE), and are reported with that label.

# qL above which the single-exponential VH form starts to break (higher rotational
# modes / intramolecular interference appear). Zero & Pecora 1982: clean below ~3,
# a <10% correction to ~5. A function-parameter default, not a system constant.
_DDLS_QL_SINGLE_EXP_LIMIT: float = 3.0


@dataclass
class DDLSRatePoint:
    """One angle's fitted FIELD decay rates for the polarised/depolarised pair.

    Built by the controller from a VV and a VH correlogram measured at the same
    angle (each fitted with the cumulant engine to its field rate Gamma). The pure
    analysis below consumes these; it never touches a correlogram or a fitter, so
    it stays decoupled from analysis.dls and is trivially testable.
    """
    angle_deg: float
    q_m_inv: float                 # scattering vector magnitude, m^-1
    gamma_vv_s_inv: float          # field decay rate Gamma_VV (g1), s^-1
    gamma_vh_s_inv: float          # field decay rate Gamma_VH (g1), s^-1


@dataclass
class DDLSResult:
    """Rotational (D_r) and translational (D_t) diffusion from a VV/VH angle set.

    `single_exponential_valid` is None when no rod length was supplied (qL cannot
    be evaluated), True when every angle has qL < the limit, False otherwise (the
    single-exponential VH model is then suspect and D_r should be treated with
    caution). SEs are None for a single angle (no ensemble), per invariant 8.
    """
    # per-angle (arrays, angle-ascending)
    angles_deg: np.ndarray
    q2_m2: np.ndarray
    gamma_vv_s_inv: np.ndarray
    gamma_vh_s_inv: np.ndarray
    d_r_per_angle: np.ndarray          # (Gamma_VH - Gamma_VV)/6 at each angle
    qL: Optional[np.ndarray]           # q*L per angle, or None if no rod length
    # combined
    d_t_m2_s: float                    # slope of Gamma_VV vs q^2 (through origin)
    d_t_se: Optional[float]
    d_r_rad2_s: float                  # mean of d_r_per_angle
    d_r_se: Optional[float]            # SD/sqrt(n) across angles; None for n < 2
    rh_t_nm: float                     # Stokes-Einstein radius from D_t
    rotational_time_s: float           # 1 / (6 D_r), the VH relaxation time
    # bookkeeping
    n_angles: int
    method: str                        # 'multi-angle' | 'single-angle'
    single_exponential_valid: Optional[bool]
    notes: str = ''
    se_estimator: str = 'hc3'          # covariance estimator behind d_t_se (the only
    #                                    regression SE here; d_r_se is a replicate SEM)


def rotational_diffusion_from_rates(gamma_vv_s_inv: float,
                                    gamma_vh_s_inv: float) -> float:
    """Rotational diffusion coefficient D_r from one VV/VH field-rate pair.

        D_r = (Gamma_VH - Gamma_VV) / 6

    Both inputs are FIELD (g1) decay rates in s^-1 (Pecora 1964; Zero & Pecora
    1982). The result is in rad^2/s. A non-positive result (Gamma_VH <= Gamma_VV)
    is unphysical for an anisotropic scatterer and signals noise, a mis-paired
    VV/VH, or an isotropic particle; it is returned as-is (possibly <= 0) so the
    caller can flag it rather than having it silently clamped.
    """
    return (gamma_vh_s_inv - gamma_vv_s_inv) / 6.0


def analyze_ddls(points: Sequence[DDLSRatePoint], *,
                 temperature_K: Optional[float] = None,
                 viscosity_Pa_s: Optional[float] = None,
                 rod_length_nm: Optional[float] = None,
                 qL_limit: float = _DDLS_QL_SINGLE_EXP_LIMIT,
                 estimator: str = 'hc3') -> DDLSResult:
    """Combine a VV/VH decay-rate set into D_r, D_t, Rh_t and the rotational time.

    Parameters
    ----------
    points : sequence of DDLSRatePoint
        One per angle (>= 1). Each carries q and the two field decay rates.
    temperature_K, viscosity_Pa_s : float
        For the Stokes-Einstein radius Rh_t from D_t. Always user-supplied.
    rod_length_nm : float, optional
        Rod length L, used only to evaluate qL for the single-exponential guard.
        Omit if unknown (then single_exponential_valid is None).
    qL_limit : float
        qL above which the single-exponential VH form is flagged (default 3.0;
        Zero & Pecora 1982). A documented parameter, not a hidden threshold.

    Returns
    -------
    DDLSResult

    Notes
    -----
    D_t is the slope of Gamma_VV vs q^2 through the origin (robust; the VV signal
    is clean). D_r is the mean of the per-angle (Gamma_VH - Gamma_VV)/6 (each angle
    an independent estimate of the q-independent rotational rate). See the module
    notes for the uncertainty rationale.
    """
    pts = sorted(points, key=lambda p: p.angle_deg)
    if len(pts) == 0:
        raise ValueError("analyze_ddls needs at least one DDLSRatePoint.")
    # T and eta are needed ONLY for the Stokes-Einstein radius Rh_t; D_t, D_r and
    # the rotational time do not use them, so they are optional (Rh_t -> nan if
    # either is missing or non-positive).
    have_se_inputs = (temperature_K is not None and viscosity_Pa_s is not None
                      and temperature_K > 0 and viscosity_Pa_s > 0)

    angles = np.array([p.angle_deg for p in pts], dtype=float)
    q = np.array([p.q_m_inv for p in pts], dtype=float)
    g_vv = np.array([p.gamma_vv_s_inv for p in pts], dtype=float)
    g_vh = np.array([p.gamma_vh_s_inv for p in pts], dtype=float)
    if np.any(q <= 0):
        raise ValueError("every q_m_inv must be positive.")
    if np.any(g_vv <= 0) or np.any(g_vh <= 0):
        raise ValueError("every decay rate must be positive.")
    q2 = q * q
    d_r_per_angle = (g_vh - g_vv) / 6.0
    n = len(pts)
    notes: List[str] = []

    # D_t from the VV channel: Gamma_VV = q^2 D_t, slope through the origin.
    if n >= 2:
        d_t, d_t_se = unc.linear_fit_through_origin(q2, g_vv, estimator)
        d_t_se = unc.se_or_none(d_t_se)
        method = 'multi-angle'
    else:
        d_t = float(g_vv[0] / q2[0])
        d_t_se = None                       # one correlogram -> no SE (Schaetzel)
        method = 'single-angle'

    # D_r from the per-angle differences (independent estimates of one rate).
    d_r_stats = unc.replicate_mean_se(list(d_r_per_angle))
    d_r = d_r_stats.mean
    d_r_se = d_r_stats.sem                   # None for n < 2

    if np.any(d_r_per_angle <= 0):
        notes.append(
            "at least one angle has Gamma_VH <= Gamma_VV (D_r <= 0), which is "
            "unphysical for an anisotropic scatterer -- check the VV/VH pairing, "
            "the signal-to-noise, or whether the particle is optically isotropic.")

    # Stokes-Einstein radius from D_t (translational), and the VH relaxation time.
    rh_t_nm = (stokes_einstein_rh(d_t, temperature_K, viscosity_Pa_s) * 1e9
               if (have_se_inputs and d_t > 0) else float('nan'))
    rotational_time_s = (1.0 / (6.0 * d_r)) if d_r > 0 else float('nan')

    # qL single-exponential guard (only if a rod length is known).
    qL = None
    single_exp_valid: Optional[bool] = None
    if rod_length_nm is not None and rod_length_nm > 0:
        qL = q * (rod_length_nm * 1e-9)     # q[m^-1] * L[m] -> dimensionless
        single_exp_valid = bool(np.all(qL < qL_limit))
        if not single_exp_valid:
            notes.append(
                f"qL reaches {float(np.max(qL)):.2f} (>= {qL_limit}); above ~3 the "
                f"VH correlogram is no longer a single exponential (higher "
                f"rotational modes / intramolecular interference) and D_r from the "
                f"simple difference is biased -- drop the high-angle points or model "
                f"the extra modes (Zero & Pecora 1982).")

    if method == 'single-angle':
        notes.append(
            "single angle: no uncertainty is reported (one correlogram per channel; "
            "a ± needs repeats or multiple angles, per ISO 22412 / Schaetzel 1990).")

    return DDLSResult(
        angles_deg=angles, q2_m2=q2, gamma_vv_s_inv=g_vv, gamma_vh_s_inv=g_vh,
        d_r_per_angle=d_r_per_angle, qL=qL,
        d_t_m2_s=d_t, d_t_se=d_t_se,
        d_r_rad2_s=d_r, d_r_se=d_r_se,
        rh_t_nm=rh_t_nm, rotational_time_s=rotational_time_s,
        n_angles=n, method=method, single_exponential_valid=single_exp_valid,
        notes=' '.join(notes), se_estimator=estimator)


# ===========================================================================
# Shape models (DPLS Phase 3) -- particle dimensions from D_t and D_r
# ===========================================================================
#
# This is the model-DEPENDENT inverse: D_t and D_r are shape-free observables, but
# turning them into dimensions assumes a geometry. Two models are offered:
#
#   * sphere (1 unknown): R from D_r via Stokes-Einstein-Debye; for a TRUE sphere
#     it equals the translational (Stokes) radius Rh, so their ratio is a sphericity
#     check. Appropriate for near-spherical anisotropic particles (Balog et al. 2015).
#   * rigid rod (2 unknowns): length L and diameter d from (D_t, D_r) by inverting
#     the Tirado (1984) cylinder relations. Exactly determined; solved numerically.
#
# Outputs are dimensions of an ASSUMED shape, NOT measurements: the result objects
# carry that caveat and a goodness/validity flag so the GUI never presents an L as
# if it were observed. The two models read together diagnose the shape -- a sphere
# consistency far from 1 says "not a sphere; use the rod", and vice versa.
#
# Uncertainty: SEs on D_t / D_r propagate to the dimensions. The sphere radius is a
# closed-form power law (delta method). The rod inverse is nonlinear with no closed
# form, so its SEs are propagated by MONTE CARLO (sample D_t, D_r from their SEs,
# invert each, take the spread) -- which is the sampling SD directly, so there is no
# linearisation to validate separately (CLAUDE.md invariant 8).


@dataclass
class SphereShapeResult:
    """Sphere-model dimensions from D_t and D_r (model-dependent)."""
    radius_rot_nm: float            # R from D_r (Stokes-Einstein-Debye)
    radius_rot_se: Optional[float]
    radius_trans_nm: float          # R from D_t (= Stokes/hydrodynamic radius Rh)
    sphericity_ratio: float         # radius_rot / radius_trans (1 for a true sphere)
    is_consistent: bool             # |ratio - 1| within tolerance -> sphere plausible
    note: str = ''


@dataclass
class RodShapeResult:
    """Rigid-cylinder dimensions from D_t and D_r (model-dependent; Tirado 1984)."""
    length_nm: float
    length_se: Optional[float]
    diameter_nm: float
    diameter_se: Optional[float]
    aspect_ratio_p: float           # L/d
    aspect_ratio_se: Optional[float]
    in_valid_range: bool            # 2 < p < 30 (Tirado fitted range)
    converged: bool                 # a rod reproduces both D_t and D_r
    note: str = ''


def sphere_dimensions_from_diffusion(
        d_t_m2_s: float, d_r_rad2_s: float, *,
        temperature_K: float, viscosity_Pa_s: float,
        d_r_se: Optional[float] = None,
        sphericity_tol: float = 0.15) -> SphereShapeResult:
    """Sphere-model radii from D_t and D_r, with a sphericity consistency check.

    R_rot comes from D_r (Stokes-Einstein-Debye, R = (kT/8 pi eta D_r)^1/3); R_trans
    is the ordinary Stokes radius from D_t (= Rh). For a true sphere they are equal,
    so `sphericity_ratio` = R_rot/R_trans should be ~1; a ratio outside
    1 +/- `sphericity_tol` flags that the particle is not spherical (e.g. a rod,
    where R_rot > R_trans). R_rot's SE propagates from D_r by the delta method
    (R ∝ D_r^(-1/3) -> SE(R)/R = (1/3) SE(D_r)/D_r).
    """
    r_rot_nm = phys.sphere_radius_from_rotational_diffusion(
        d_r_rad2_s, temperature_K, viscosity_Pa_s) * 1e9
    r_trans_nm = stokes_einstein_rh(d_t_m2_s, temperature_K, viscosity_Pa_s) * 1e9
    ratio = r_rot_nm / r_trans_nm if r_trans_nm > 0 else float('nan')
    r_rot_se = (r_rot_nm * (1.0 / 3.0) * (d_r_se / d_r_rad2_s)
                if (d_r_se is not None and d_r_rad2_s > 0) else None)
    consistent = math.isfinite(ratio) and abs(ratio - 1.0) <= sphericity_tol
    note = ('R from D_r and R from D_t agree -> a sphere is a plausible model.'
            if consistent else
            f'R(D_r) = {r_rot_nm:.2f} nm and R(D_t) = {r_trans_nm:.2f} nm disagree '
            f'(ratio {ratio:.2f}); the particle is not spherical -- prefer the rod '
            f'model (ratio > 1 is the rod signature).')
    return SphereShapeResult(
        radius_rot_nm=r_rot_nm, radius_rot_se=unc.se_or_none(r_rot_se),
        radius_trans_nm=r_trans_nm, sphericity_ratio=ratio,
        is_consistent=consistent, note=note)


def _solve_rod_pld(d_t_m2_s: float, d_r_rad2_s: float,
                   temperature_K: float, viscosity_Pa_s: float):
    """Solve (L, d, p) for a rigid cylinder reproducing D_t and D_r, or None.

    Reduces the 2-D (L, d) system to a 1-D root find in the aspect ratio p: at any
    p, D_t fixes L (rod_length_from_translational_diffusion), then the predicted D_r
    is monotonic decreasing in p, so D_r(p) - D_r_measured has at most one root. Try
    the Tirado-valid window first, then a wider range (flagged out-of-range).
    Returns (L_nm, d_nm, p, in_valid_range) or None if no rod fits.
    """
    def dr_residual(p):
        length_m = phys.rod_length_from_translational_diffusion(
            d_t_m2_s, p, temperature_K, viscosity_Pa_s)
        diameter_m = length_m / p
        return phys.rod_rotational_diffusion(
            length_m, diameter_m, temperature_K, viscosity_Pa_s) - d_r_rad2_s

    def solve_in(lo, hi):
        try:
            f_lo, f_hi = dr_residual(lo), dr_residual(hi)
        except (ValueError, ZeroDivisionError):
            return None
        if not (math.isfinite(f_lo) and math.isfinite(f_hi)) or f_lo * f_hi > 0:
            return None
        return optimize.brentq(dr_residual, lo, hi, xtol=1e-9, rtol=1e-12)

    p = solve_in(phys.ROD_ASPECT_RATIO_MIN, phys.ROD_ASPECT_RATIO_MAX)
    in_range = p is not None
    if p is None:                                   # try a wider, flagged range
        p = solve_in(1.05, 200.0)
    if p is None:
        return None
    length_m = phys.rod_length_from_translational_diffusion(
        d_t_m2_s, p, temperature_K, viscosity_Pa_s)
    return (length_m * 1e9, length_m / p * 1e9, p, in_range)


def rod_dimensions_from_diffusion(
        d_t_m2_s: float, d_r_rad2_s: float, *,
        temperature_K: float, viscosity_Pa_s: float,
        d_t_se: Optional[float] = None, d_r_se: Optional[float] = None,
        n_mc: int = 2000, seed: int = 12345) -> RodShapeResult:
    """Rigid-cylinder length and diameter from D_t and D_r (Tirado 1984 inverse).

    Solves the 2-D problem via the 1-D reduction in `_solve_rod_pld`. When both SEs
    are given, the dimension SEs are propagated by Monte Carlo: sample D_t, D_r from
    independent normals, invert each, and take the spread (the sampling SD -- no
    linearisation). `in_valid_range` is False when the recovered p falls outside the
    Tirado-fitted 2 < p < 30 (the dimensions are then extrapolated); `converged` is
    False when no rod reproduces both coefficients (the rod model does not fit).
    """
    solved = _solve_rod_pld(d_t_m2_s, d_r_rad2_s, temperature_K, viscosity_Pa_s)
    if solved is None:
        return RodShapeResult(
            length_nm=float('nan'), length_se=None, diameter_nm=float('nan'),
            diameter_se=None, aspect_ratio_p=float('nan'), aspect_ratio_se=None,
            in_valid_range=False, converged=False,
            note='No rigid rod reproduces both D_t and D_r -- the rod model does not '
                 'fit (the particle may be flexible, branched, or non-rod-like).')
    length_nm, diameter_nm, p, in_range = solved

    length_se = diameter_se = p_se = None
    if d_t_se is not None and d_r_se is not None and d_t_se > 0 and d_r_se > 0:
        rng = np.random.default_rng(seed)
        Ls, ds, ps = [], [], []
        for _ in range(int(n_mc)):
            dt = float(rng.normal(d_t_m2_s, d_t_se))
            dr = float(rng.normal(d_r_rad2_s, d_r_se))
            if dt <= 0 or dr <= 0:
                continue
            s = _solve_rod_pld(dt, dr, temperature_K, viscosity_Pa_s)
            if s is not None:
                Ls.append(s[0]); ds.append(s[1]); ps.append(s[2])
        if len(Ls) >= 2:
            length_se = unc.se_or_none(float(np.std(Ls, ddof=1)))
            diameter_se = unc.se_or_none(float(np.std(ds, ddof=1)))
            p_se = unc.se_or_none(float(np.std(ps, ddof=1)))

    note = ''
    if not in_range:
        note = (f'aspect ratio p = {p:.2f} is outside the Tirado (1984) fitted range '
                f'2 < p < 30; L and d are extrapolated and should be treated with '
                f'caution.')
    return RodShapeResult(
        length_nm=length_nm, length_se=length_se,
        diameter_nm=diameter_nm, diameter_se=diameter_se,
        aspect_ratio_p=p, aspect_ratio_se=p_se,
        in_valid_range=in_range, converged=True, note=note)
