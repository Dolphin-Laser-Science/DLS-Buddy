"""
analysis/sls.py
===============

Static light scattering (SLS) analysis: molecular weight Mw, radius of gyration
Rg, and second virial coefficient A2 from angular/concentration intensity data.

Built in passes:

  Pass 1 (this file so far): calibration pipeline
    - compute_excess_rayleigh_ratio   raw intensities -> excess Rayleigh ratio
    - zimm_coordinate                  -> Kc / dR (the Zimm/Debye/Berry ordinate)

  Pass 2: single-concentration / apparent analyses (Debye plot, single-angle Mw)

  Pass 3: full Zimm and Berry double extrapolation (Mw, Rg, A2), and the
          calibration-free A2 method.

The basic SLS equation (Zimm; Brookhaven BIZPW manual Eq. 3; Takahashi et al.
2019 Eq. 4):

    K c / dR(theta, c) = (1 / Mw) [1 + q^2 Rg^2 / 3] + 2 A2 c + ...

with the optical constant (VV / vertical polarisation, the factor-4 form)

    K = 4 pi^2 n^2 (dn/dc)^2 / (Na lambda^4)

and the scattering vector  q = (4 pi n / lambda) sin(theta / 2).

Calibration to an absolute excess Rayleigh ratio is platform-independent and
offered by two routes (see compute_excess_rayleigh_ratio):

  1. Standard-reference (general, preferred): divide the solvent-subtracted
     sample intensity by a calibration-liquid (toluene) measurement at each
     angle, then scale by the standard's Rayleigh ratio --
         dR(theta) = [I_sample - I_solvent]/I_standard * R_std * (n_s/n_std)^2.
     Dividing by the standard per angle corrects the angular response, so no
     sin(theta) factor is needed.

  2. Calibration-constant: for instruments that report a constant k_c (Rayleigh
     ratio per unit intensity, from a standard at ~90 deg) --
         dR(theta) = k_c * sin(theta) * [I_sample - I_solvent] * (n_s/n_std)^2,
     with the sin(theta) volume correction (Seery et al. 1989), exact for an
     isotropic standard since I_std(theta) sin(theta) = I_std(90).

The two agree exactly for an ideal isotropic standard; comparing them is a
useful data-quality cross-check.

IMPORTANT calibration conventions (enforced here)
-------------------------------------------------
- The standard Rayleigh ratio defaults to the Sivokhin & Kazantsev (2021)
  temperature-corrected toluene value from physics/constants.py. For a
  calibration constant carrying an instrument's own (often outdated) toluene
  value, that value can be supplied so k_c is rescaled to the authoritative one.
  No instrument's stored Rayleigh ratio is trusted as the absolute reference.
- The (n_solvent / n_standard)^2 refractive-index correction is applied whenever
  the solvent and standard refractive indices differ. It is mandatory.
- Dark-count subtraction is applied to every intensity; the sin(theta) volume
  correction applies to the calibration-constant route by default.

References (in project knowledge)
---------------------------------
  Brookhaven BI-200SM manual, Section VIII (Rayleigh ratios / calibration)
  Brookhaven BIZPW (Zimm Plot) and Particle Explorer manuals (Zimm equation)
  Takahashi et al. 2019, Anal. Sci. 35, 1045 (Rayleigh ratio, Zimm/Berry)
  Seery et al. 1989 (sin(theta) volume correction)
  Sivokhin & Kazantsev 2021 (toluene Rayleigh ratio; via constants.py)

Design contract
---------------
Every function is PURE: it takes measurement objects (and parameters) and returns
a result object. No plotting, no file I/O, no mutation of inputs. Apparent
(single-angle or single-concentration) results are always clearly distinguished
from thermodynamic (fully extrapolated) ones.

Change history
--------------
2026-06-13  Pass 1: calibration pipeline. (sls.py v1)
            compute_excess_rayleigh_ratio, zimm_coordinate, result objects.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from core.data_models import SLSMeasurement
from analysis import uncertainty as unc
from physics.constants import (
    scattering_vector_q,
    optical_constant_K,
    refractive_index_correction,
)


# ===========================================================================
# Result objects
# ===========================================================================

@dataclass
class RayleighRatioResult:
    """Excess Rayleigh ratio for one concentration across angles.

    Carries the excess Rayleigh ratio dR(theta), the scattering vectors, the
    optical constant K, and the Zimm ordinate Kc/dR. The `calibrated` flag is the
    important one: when False, no calibration constant was available, so dR is on
    an ARBITRARY scale. Quantities that depend on that scale (Mw, absolute A2) are
    then unreliable, while scale-independent ones (Rg, and the calibration-free
    2 A2 Mw product) remain valid -- the downstream results carry the
    corresponding flags.
    """
    # identity / context
    concentration_g_per_mL: float
    temperature_K: float
    wavelength_nm: float
    solvent_refractive_index: float
    dn_dc_mL_per_g: float
    # per-angle arrays (sorted by angle ascending)
    angles_deg: np.ndarray
    q_nm_inv: np.ndarray
    q2_nm2: np.ndarray                 # q^2 in nm^-2 (Zimm abscissa, q-space)
    excess_rayleigh_cm_inv: np.ndarray # dR(theta) in cm^-1 (arbitrary scale if not calibrated)
    kc_over_dR_mol_per_g: np.ndarray   # Kc / dR  (the Zimm ordinate)
    # optical constant and calibration bookkeeping
    optical_constant_K: float
    calibrated: bool                   # False -> dR on an arbitrary scale
    k_c_used: float                    # calibration constant applied (1.0 if uncalibrated)
    ri_correction: float               # (n_solvent / n_standard)^2
    dark_count_subtracted: float
    sin_theta_applied: bool
    calibration_note: str = ''


# ===========================================================================
# Calibration: one unified model (compute our own k_c; never trust vendor values)
# ===========================================================================

def compute_calibration_constant(
    calibrant_intensity: float,
    calibrant_angle_deg: float,
    standard_rayleigh_ratio: float,
    dark_count_rate: float = 0.0,
) -> float:
    """Compute the calibration constant k_c from a single calibrant measurement.

    The instrument is calibrated by measuring a standard liquid (usually toluene)
    of known Rayleigh ratio, typically once at 90 deg. The constant that converts
    measured intensity to absolute Rayleigh ratio is

        k_c = R_standard / (I_calibrant_net * sin(theta_cal))

    where I_calibrant_net is dark-subtracted. (At 90 deg, sin = 1.) This single
    value is then used for every sample, regardless of which instrument produced
    the data -- one calibration model for all sources. Vendor-stored calibration
    constants are never used; compute your own here.

    The standard's Rayleigh ratio must be supplied in the SAME scattering geometry
    in which the calibrant was measured (e.g. R_VU for a no-analyser instrument).
    Get it from physics.constants.rayleigh_ratio_toluene(wavelength, T, geometry).

    Parameters
    ----------
    calibrant_intensity : float
        Measured scattered intensity of the pure standard liquid.
    calibrant_angle_deg : float
        Angle at which the calibrant was measured (usually 90).
    standard_rayleigh_ratio : float
        Rayleigh ratio of the standard in cm^-1, in the measurement geometry.
    dark_count_rate : float
        Dark count subtracted from the calibrant intensity. Default 0.

    Returns
    -------
    float
        The calibration constant k_c (cm^-1 per unit intensity).

    Raises
    ------
    ValueError
        If the net intensity is non-positive, or the calibrant angle is not
        strictly between 0 and 180 deg (where the sin-theta scattering-volume
        factor degenerates to zero).
    """
    I_net = calibrant_intensity - dark_count_rate
    if I_net <= 0:
        raise ValueError(
            f"Net calibrant intensity must be positive, got {I_net} "
            f"(intensity {calibrant_intensity} - dark {dark_count_rate})."
        )
    # Guard on the angle directly, not on sin(theta): at exactly 180 deg
    # math.sin(math.radians(180)) is ~1.2e-16 (float round-off), not 0, so a
    # `sin <= 0` test would let the degenerate back-scatter geometry through.
    if not (0.0 < calibrant_angle_deg < 180.0):
        raise ValueError(
            f"calibrant_angle must be strictly between 0 and 180 deg (the "
            f"sin-theta factor degenerates to 0 at 0 and 180 deg), got "
            f"{calibrant_angle_deg} deg."
        )
    s = math.sin(math.radians(calibrant_angle_deg))
    return standard_rayleigh_ratio / (I_net * s)


def compute_excess_rayleigh_ratio(
    sample: SLSMeasurement,
    solvent_reference: SLSMeasurement,
    calibration_constant: Optional[float] = None,
    standard_refractive_index: Optional[float] = None,
    dark_count_rate: float = 0.0,
    apply_sin_theta: bool = True,
    transmittance: Optional[Sequence[float]] = None,
) -> RayleighRatioResult:
    """Convert raw intensities to an excess Rayleigh ratio dR(theta).

    One unified pipeline for all data sources:
      1. dark-count subtraction:   I_net = I_raw - dark
      2. solvent subtraction:      I_excess = I_net,sample - I_net,solvent
      3. volume correction:        x sin(theta)            (if apply_sin_theta)
      4. absorbance correction:    / transmittance         (if transmittance given)
      5. calibration:              x k_c x (n_s/n_cal)^2
    with dR(theta) = k_c * sin(theta) * I_excess * (n_s/n_cal)^2.

    The calibration constant k_c is the single unifying quantity. Compute it from
    a calibrant measurement with `compute_calibration_constant`, or pass a value
    you determined independently. A vendor-stored constant is NEVER used here.

    If `calibration_constant` is None, the analysis still runs but is flagged
    UNCALIBRATED: dR is placed on an arbitrary scale (k_c = 1). Scale-independent
    results (Rg; the calibration-free 2 A2 Mw) remain valid; Mw and absolute A2
    do not, and the downstream results say so.

    Parameters
    ----------
    sample : SLSMeasurement
        Solution at one concentration, across angles.
    solvent_reference : SLSMeasurement
        Pure-solvent measurement (concentration 0) at the SAME angles.
    calibration_constant : float, optional
        k_c (cm^-1 per unit intensity). None -> uncalibrated (flagged).
    standard_refractive_index : float, optional
        Refractive index of the calibration standard, for the (n_s/n_cal)^2
        correction. If omitted, the correction is skipped (factor 1) and a note
        is recorded; supply it whenever the solvent differs from the standard.
    dark_count_rate : float
        Subtracted from sample and solvent intensities. Default 0.
    apply_sin_theta : bool
        Apply the sin(theta) volume correction (default True).
    transmittance : sequence of float, optional
        Per-angle internal transmittance for absorbing samples; dR is divided by
        it. Omit for non-absorbing samples.

    Returns
    -------
    RayleighRatioResult

    Raises
    ------
    ValueError
        If sample and solvent angles differ, or transmittance has the wrong length.
    """
    if (sample.angles_deg.shape != solvent_reference.angles_deg.shape
            or not np.allclose(np.sort(sample.angles_deg),
                               np.sort(solvent_reference.angles_deg))):
        raise ValueError(
            "Sample and solvent_reference must be measured at the same angles."
        )

    s_order = np.argsort(sample.angles_deg)
    r_order = np.argsort(solvent_reference.angles_deg)
    angles = sample.angles_deg[s_order]
    I_sample = sample.intensities[s_order] - dark_count_rate
    I_solvent = solvent_reference.intensities[r_order] - dark_count_rate
    I_excess = I_sample - I_solvent

    sin_applied = False
    if apply_sin_theta:
        I_excess = I_excess * np.sin(np.radians(angles))
        sin_applied = True

    if transmittance is not None:
        t = np.asarray(transmittance, dtype=float)
        if t.shape != angles.shape:
            raise ValueError(
                f"transmittance must have one value per angle ({angles.size}), "
                f"got {t.size}."
            )
        I_excess = I_excess / t

    # Calibration constant (computed elsewhere) or uncalibrated fallback.
    note = ''
    if calibration_constant is None:
        k_c = 1.0
        calibrated = False
        note = ('UNCALIBRATED: no calibration constant supplied; dR is on an '
                'arbitrary scale. Mw and absolute A2 are unreliable; Rg and the '
                'calibration-free 2 A2 Mw remain valid.')
    else:
        k_c = float(calibration_constant)
        calibrated = True

    # (n_s/n_cal)^2 correction when the standard's refractive index is known.
    if standard_refractive_index is not None:
        ri_corr = refractive_index_correction(sample.solvent_refractive_index,
                                              standard_refractive_index)
    else:
        ri_corr = 1.0
        # No standard refractive index supplied -> the (n_s/n_cal)^2 correction was
        # genuinely skipped, so flag it. We do NOT gate this note on the solvent's
        # actual n (e.g. "is it toluene?"): without the standard's n we cannot know
        # whether the solvent matches the calibration standard, and a hard-coded n
        # would be a solvent-specific magic number (invariants 3 & 4).
        extra = (' No standard refractive index given, so the (n_s/n_cal)^2 '
                 'correction was skipped; supply standard_refractive_index '
                 'if the solvent differs from the calibration standard.')
        note = (note + extra) if note else extra.strip()

    dR = k_c * I_excess * ri_corr   # cm^-1 (arbitrary scale if uncalibrated)

    K = optical_constant_K(sample.solvent_refractive_index,
                           sample.dn_dc_mL_per_g, sample.wavelength_nm)
    c = sample.concentration_g_per_mL
    with np.errstate(divide='ignore', invalid='ignore'):
        kc_over_dR = np.where(dR != 0, K * c / dR, np.nan)

    q = np.array([
        scattering_vector_q(float(a), sample.wavelength_nm,
                            sample.solvent_refractive_index)
        for a in angles
    ], dtype=float)

    return RayleighRatioResult(
        concentration_g_per_mL=c,
        temperature_K=sample.temperature_K,
        wavelength_nm=sample.wavelength_nm,
        solvent_refractive_index=sample.solvent_refractive_index,
        dn_dc_mL_per_g=sample.dn_dc_mL_per_g,
        angles_deg=angles,
        q_nm_inv=q,
        q2_nm2=q ** 2,
        excess_rayleigh_cm_inv=dR,
        kc_over_dR_mol_per_g=kc_over_dR,
        optical_constant_K=K,
        calibrated=calibrated,
        k_c_used=k_c,
        ri_correction=ri_corr,
        dark_count_subtracted=float(dark_count_rate),
        sin_theta_applied=sin_applied,
        calibration_note=note,
    )


def zimm_coordinate(rayleigh_result: RayleighRatioResult) -> Tuple[np.ndarray, np.ndarray]:
    """Return (q^2 in nm^-2, Kc/dR in mol/g) from a RayleighRatioResult.

    Convenience accessor for the Zimm/Debye ordinate and abscissa. The same Kc/dR
    is used by the Zimm, Debye, and (after square-rooting) Berry analyses.
    """
    return rayleigh_result.q2_nm2, rayleigh_result.kc_over_dR_mol_per_g


# ===========================================================================
# Pass 2: single-concentration and single-angle (apparent) analyses
# ===========================================================================
#
# These operate on one concentration. They yield APPARENT quantities -- not
# extrapolated to infinite dilution -- and every result says so explicitly, so an
# apparent Mw is never mistaken for the thermodynamic Mw.


@dataclass
class DebyeResult:
    """Single-concentration Debye/Guinier analysis: Kc/dR vs q^2.

    All quantities are APPARENT (at this finite concentration): the intercept is
    1/Mw + 2 A2 c, not 1/Mw. Rg is the apparent radius of gyration. A full Zimm or
    Berry analysis over several concentrations is needed for thermodynamic values.
    """
    concentration_g_per_mL: float
    q2_nm2: np.ndarray
    kc_over_dR: np.ndarray
    # linear fit  Kc/dR = intercept + slope * q^2
    intercept_mol_per_g: float        # = 1/Mw_app + 2 A2 c
    slope: float                      # = (1/Mw)(Rg^2/3) at this c
    mw_apparent_g_per_mol: float      # 1 / intercept
    rg_apparent_nm: float             # sqrt(3 * slope / intercept)
    r_squared: float
    is_apparent: bool                 # always True here
    n_angles: int
    calibrated: bool                  # was the Rayleigh ratio calibrated?
    mw_reliable: bool                 # False if uncalibrated (Rg stays reliable)
    mw_apparent_se: Optional[float] = None   # statistical (regression) SEs --
    rg_apparent_se: Optional[float] = None   # exclude calibration/dn-dc systematics
    se_estimator: str = 'hc3'                 # covariance estimator behind the SEs


@dataclass
class SingleAngleResult:
    """Single-angle, single-concentration apparent molecular weight.

    Mw_app = dR / (K c) at one angle and concentration -- no angular or
    concentration extrapolation. With no angular information there is no Rg, and
    the value still contains the form factor P(q) and the 2 A2 c term. Strictly an
    order-of-magnitude / apparent quantity.
    """
    concentration_g_per_mL: float
    angle_deg: float
    q2_nm2: float
    mw_apparent_g_per_mol: float
    is_apparent: bool                 # always True here
    calibrated: bool = True           # was the Rayleigh ratio calibrated?
    mw_reliable: bool = True          # False if uncalibrated (Mw_app on an arbitrary scale)


@dataclass
class GuinierResult:
    """Single-concentration Guinier analysis: ln(dR) vs q^2.

    The Guinier approximation dR(q) = dR(0) exp(-q^2 Rg^2 / 3) linearises as
    ln(dR) = ln(dR(0)) - (Rg^2/3) q^2, so a straight-line fit gives Rg from the
    SLOPE (Rg = sqrt(-3*slope)) and an apparent Mw from the INTERCEPT
    (dR(0) = K c Mw). Like Debye, Rg is scale-independent -- it comes from the
    slope of a log-intensity plot, so it is reliable even when the run is
    uncalibrated; only the intercept-derived Mw needs calibration.

    Valid only in the Guinier regime, q*Rg <~ 1.3; `guinier_valid` flags this from
    the largest q used. Single concentration -> the values are apparent.
    """
    concentration_g_per_mL: float
    q2_nm2: np.ndarray
    ln_excess_rayleigh: np.ndarray    # ln(dR) at the fitted angles
    slope: float                      # d ln(dR) / d q^2  (= -Rg^2/3), nm^2
    intercept: float                  # ln(dR(0))
    rg_nm: float                      # sqrt(-3*slope); nan if slope >= 0
    mw_apparent_g_per_mol: float      # exp(intercept) / (K c); nan if K or c <= 0
    qrg_max: float                    # q*Rg at the largest fitted angle
    guinier_valid: bool               # qrg_max <= qrg_max_valid
    r_squared: float
    n_angles: int
    is_apparent: bool                 # always True here
    calibrated: bool                  # was the Rayleigh ratio calibrated?
    mw_reliable: bool                 # False if uncalibrated (Rg stays reliable)
    rg_se: Optional[float] = None            # statistical (regression) SEs --
    mw_apparent_se: Optional[float] = None   # exclude calibration/dn-dc systematics
    se_estimator: str = 'hc3'                 # covariance estimator behind the SEs


def debye_analysis(rayleigh_result: RayleighRatioResult,
                   estimator: str = 'hc3') -> DebyeResult:
    """Single-concentration Debye plot: linear fit of Kc/dR vs q^2.

    Gives the apparent molecular weight (from the intercept) and apparent radius
    of gyration (from the slope-to-intercept ratio) at one concentration. Results
    are flagged apparent: the intercept is 1/Mw + 2 A2 c, not 1/Mw.

    Parameters
    ----------
    rayleigh_result : RayleighRatioResult
        One concentration's calibrated data (>= 2 angles).

    Returns
    -------
    DebyeResult

    Raises
    ------
    ValueError
        If fewer than two finite angles are available.
    """
    q2 = rayleigh_result.q2_nm2
    y = rayleigh_result.kc_over_dR_mol_per_g
    good = np.isfinite(q2) & np.isfinite(y)
    q2, y = q2[good], y[good]
    if q2.size < 2:
        raise ValueError(
            "Debye analysis needs at least two angles with finite Kc/dR."
        )

    fit = unc.linear_fit(q2, y, estimator)  # cov order [intercept, slope]
    slope, intercept = fit.slope, fit.intercept
    r2 = fit.r_squared

    mw_app = 1.0 / intercept if intercept != 0 else float('nan')
    # Rg^2 = 3 * slope / intercept  (apparent)
    ratio = slope / intercept if intercept != 0 else float('nan')
    rg_app = math.sqrt(3.0 * ratio) if (math.isfinite(ratio) and ratio > 0) else float('nan')

    # Statistical SEs by first-order propagation through the (intercept, slope) cov.
    mw_se = rg_se = None
    if intercept != 0 and math.isfinite(fit.intercept_se):
        mw_se = unc.se_or_none(unc.propagate([-1.0 / intercept ** 2, 0.0], fit.cov))
        if math.isfinite(rg_app) and rg_app > 0 and slope != 0:
            jac = [-rg_app / (2.0 * intercept), rg_app / (2.0 * slope)]
            rg_se = unc.se_or_none(unc.propagate(jac, fit.cov))

    return DebyeResult(
        concentration_g_per_mL=rayleigh_result.concentration_g_per_mL,
        q2_nm2=q2, kc_over_dR=y,
        intercept_mol_per_g=float(intercept), slope=float(slope),
        mw_apparent_g_per_mol=float(mw_app), rg_apparent_nm=float(rg_app),
        r_squared=float(r2), is_apparent=True, n_angles=int(q2.size),
        calibrated=rayleigh_result.calibrated,
        mw_reliable=rayleigh_result.calibrated,
        mw_apparent_se=mw_se, rg_apparent_se=rg_se,
        se_estimator=estimator,
    )


def single_angle_mw(rayleigh_result: RayleighRatioResult, angle_deg: float) -> SingleAngleResult:
    """Apparent Mw from one angle of one concentration: Mw_app = dR/(Kc).

    No extrapolation in either angle or concentration. Use only as a rough check;
    the value contains both the form factor and the virial term.

    Parameters
    ----------
    rayleigh_result : RayleighRatioResult
    angle_deg : float
        Which measured angle to use (must match one in the result).

    Returns
    -------
    SingleAngleResult

    Raises
    ------
    ValueError
        If the angle is not present, or concentration is zero.
    """
    idx = np.where(np.isclose(rayleigh_result.angles_deg, angle_deg))[0]
    if idx.size == 0:
        raise ValueError(
            f"angle {angle_deg} deg is not among the measured angles "
            f"{rayleigh_result.angles_deg.tolist()}."
        )
    i = int(idx[0])
    c = rayleigh_result.concentration_g_per_mL
    if c == 0:
        raise ValueError("Cannot compute Mw at zero concentration.")
    y = rayleigh_result.kc_over_dR_mol_per_g[i]   # = K c / dR
    mw_app = 1.0 / y if y != 0 else float('nan')
    return SingleAngleResult(
        concentration_g_per_mL=c, angle_deg=float(rayleigh_result.angles_deg[i]),
        q2_nm2=float(rayleigh_result.q2_nm2[i]),
        mw_apparent_g_per_mol=float(mw_app), is_apparent=True,
        calibrated=rayleigh_result.calibrated,
        mw_reliable=rayleigh_result.calibrated,
    )


def guinier_analysis(rayleigh_result: RayleighRatioResult,
                     qrg_max_valid: float = 1.3,
                     estimator: str = 'hc3') -> GuinierResult:
    """Single-concentration Guinier plot: linear fit of ln(dR) vs q^2.

    Fits ln(dR) = ln(dR(0)) - (Rg^2/3) q^2. Rg comes from the slope
    (Rg = sqrt(-3*slope)) and is scale-independent (reliable even uncalibrated);
    the apparent Mw comes from the intercept (dR(0) = K c Mw) and needs
    calibration. Only angles with a finite, POSITIVE excess Rayleigh ratio are
    used (ln requires dR > 0).

    The Guinier approximation is due to Guinier (1939); see Chu (1991) and
    Russo et al. (2021, Ch. 13) for accessible derivations of the plot.

    Parameters
    ----------
    rayleigh_result : RayleighRatioResult
        One concentration's data (>= 2 angles with dR > 0).
    qrg_max_valid : float
        Upper q*Rg for the Guinier approximation (default 1.3, a general
        criterion, not system-specific). `guinier_valid` reflects this.

    Returns
    -------
    GuinierResult

    Raises
    ------
    ValueError
        If fewer than two angles have a finite, positive excess Rayleigh ratio.
    """
    q2 = rayleigh_result.q2_nm2
    q = rayleigh_result.q_nm_inv
    dR = rayleigh_result.excess_rayleigh_cm_inv
    good = np.isfinite(q2) & np.isfinite(dR) & (dR > 0)
    q2g, qg, dRg = q2[good], q[good], dR[good]
    if q2g.size < 2:
        raise ValueError(
            "Guinier analysis needs at least two angles with a finite, positive "
            "excess Rayleigh ratio (ln(dR) requires dR > 0)."
        )

    y = np.log(dRg)
    fit = unc.linear_fit(q2g, y, estimator)  # cov order [intercept, slope]
    slope, intercept = fit.slope, fit.intercept
    r2 = fit.r_squared

    # Rg from the slope: slope = -Rg^2/3  (q^2 in nm^-2 -> Rg in nm).
    rg = math.sqrt(-3.0 * slope) if slope < 0 else float('nan')

    # Apparent Mw from the intercept: dR(0) = exp(intercept) = K c Mw.
    K = rayleigh_result.optical_constant_K
    c = rayleigh_result.concentration_g_per_mL
    dR0 = math.exp(intercept)
    mw_app = dR0 / (K * c) if (K > 0 and c > 0) else float('nan')

    # Statistical SEs: Rg from the slope alone; Mw from the intercept (dMw/dint = Mw).
    rg_se = mw_se = None
    if math.isfinite(fit.slope_se):
        if math.isfinite(rg) and rg > 0:
            rg_se = unc.se_or_none((3.0 / (2.0 * rg)) * fit.slope_se)
        if math.isfinite(mw_app):
            mw_se = unc.se_or_none(abs(mw_app) * fit.intercept_se)

    qrg_max = float(qg.max() * rg) if math.isfinite(rg) else float('nan')
    valid = math.isfinite(qrg_max) and qrg_max <= qrg_max_valid

    return GuinierResult(
        concentration_g_per_mL=c, q2_nm2=q2g, ln_excess_rayleigh=y,
        slope=float(slope), intercept=float(intercept), rg_nm=float(rg),
        mw_apparent_g_per_mol=float(mw_app), qrg_max=qrg_max,
        guinier_valid=bool(valid), r_squared=float(r2), n_angles=int(q2g.size),
        is_apparent=True, calibrated=rayleigh_result.calibrated,
        mw_reliable=rayleigh_result.calibrated,
        rg_se=rg_se, mw_apparent_se=mw_se,
        se_estimator=estimator,
    )


# ===========================================================================
# Pass 3: full Zimm / Berry double extrapolation + calibration-free A2
# ===========================================================================
#
# The first-order Zimm equation
#     Kc/dR = (1/Mw)(1 + q^2 Rg^2/3) + 2 A2 c
# is LINEAR in the parameters (a, b, d) of the model  a + b q^2 + d c, with
#     a = 1/Mw,  b = (1/Mw)(Rg^2/3),  d = 2 A2.
# So a single multilinear regression of Kc/dR on (1, q^2, c) over all points
# gives all three quantities at once -- this is the robust "global" Zimm fit.
# Mw = 1/a,  Rg = sqrt(3 b / a),  A2 = d / 2.
#
# Berry uses sqrt(Kc/dR) = a' + b' q^2 + d' c with
#     a' = 1/sqrt(Mw),  b' = a' Rg^2/6,  d' = A2 sqrt(Mw),
# so Mw = 1/a'^2,  Rg = sqrt(6 b'/a'),  A2 = d' a'. Berry linearises the high-qRg
# regime (Mw above ~1 MDa) better than Zimm.
#
# The per-concentration and per-angle line fits used by the classic Zimm grid
# plot are also returned, for visualisation and as a cross-check on the global fit.


@dataclass
class ZimmBerryResult:
    """Thermodynamic Mw, Rg, and A2 from a Zimm or Berry double extrapolation."""
    method: str                       # 'zimm' or 'berry'
    mw_g_per_mol: float
    rg_nm: float                      # nan if the angular term is non-physical
    a2_mol_mL_per_g2: float
    # global multilinear fit  ord = a + b q^2 + d c  (ord = Kc/dR or its sqrt)
    coef_intercept: float             # a (or a')
    coef_q2: float                    # b (or b')
    coef_c: float                     # d (or d')
    r_squared: float
    n_points: int
    n_concentrations: int
    n_angles: int
    is_apparent: bool                 # False: thermodynamic (fully extrapolated)
    calibrated: bool                  # was the Rayleigh ratio calibrated?
    mw_reliable: bool                 # False if uncalibrated
    a2_reliable: bool                 # False if uncalibrated (absolute A2 needs Mw)
    # per-concentration q->0 intercepts (for the Zimm grid plot / cross-check)
    concentrations_g_per_mL: np.ndarray
    intercept_per_concentration: np.ndarray   # ordinate at q^2 -> 0, each c
    # per-angle c->0 intercepts
    q2_nm2: np.ndarray
    intercept_per_angle: np.ndarray           # ordinate at c -> 0, each angle
    # statistical (regression) standard errors -- exclude calibration/dn-dc
    # systematics. None when too few points to estimate them.
    mw_se: Optional[float] = None
    rg_se: Optional[float] = None
    a2_se: Optional[float] = None
    # consistency cross-check: Mw from the two single-route extrapolations (the
    # q->0 intercepts extrapolated to c->0, and the c->0 intercepts to q->0) should
    # agree; extrapolation_agreement_rel is their relative difference.
    mw_from_c0_g_per_mol: Optional[float] = None
    mw_from_q0_g_per_mol: Optional[float] = None
    extrapolation_agreement_rel: Optional[float] = None
    se_estimator: str = 'hc3'                 # covariance estimator behind the SEs


@dataclass
class CalibrationFreeA2Result:
    """A2 (times Mw) from intensity ratios alone -- no absolute calibration.

    Uses Y(c) = [I_excess(c_ref)/c_ref] / [I_excess(c)/c] at a fixed angle, which
    is linear in c with slope/intercept = 2 A2 Mw. Because only intensity RATIOS
    enter, the optical constant and Rayleigh-ratio calibration cancel -- valuable
    when absolute calibration is uncertain (e.g. low-dn/dc systems). If Mw is
    supplied, A2 itself is returned.
    """
    angle_deg: float
    concentrations_g_per_mL: np.ndarray
    Y: np.ndarray
    slope: float
    intercept: float
    two_a2_mw: float                  # slope / intercept = 2 A2 Mw  (mol*mL/g ... * g/mol)
    a2_mol_mL_per_g2: Optional[float] # if mw provided
    r_squared: float
    # These SEs are free of calibration/dn-dc systematics *by construction*:
    # 2*A2*Mw = slope/intercept is calibration- and dn/dc-independent (the point of
    # this estimator), so unlike the Zimm/Debye SEs there is no such systematic to exclude.
    two_a2_mw_se: Optional[float] = None   # statistical SE (slope/intercept covariance)
    a2_se: Optional[float] = None          # if mw provided (mw treated as exact)
    se_estimator: str = 'hc3'              # covariance estimator behind the SEs


def _collect_zimm_points(rayleigh_results: Sequence[RayleighRatioResult]):
    """Flatten a set of per-concentration results into (q2, c, Kc/dR) arrays."""
    q2_all, c_all, y_all = [], [], []
    for r in rayleigh_results:
        if r.concentration_g_per_mL == 0:
            continue   # skip the solvent reference
        good = np.isfinite(r.q2_nm2) & np.isfinite(r.kc_over_dR_mol_per_g)
        q2_all.append(r.q2_nm2[good])
        c_all.append(np.full(int(good.sum()), r.concentration_g_per_mL))
        y_all.append(r.kc_over_dR_mol_per_g[good])
    if not q2_all:
        raise ValueError("No non-zero-concentration data to analyse.")
    return (np.concatenate(q2_all), np.concatenate(c_all), np.concatenate(y_all))


def zimm_analysis(
    rayleigh_results: Sequence[RayleighRatioResult],
    method: str = 'zimm',
    estimator: str = 'hc3',
) -> ZimmBerryResult:
    """Full Zimm or Berry double extrapolation for Mw, Rg, and A2.

    Performs the global first-order fit (multilinear regression of the ordinate
    on 1, q^2, and c) over all angles and concentrations, and also returns the
    per-concentration and per-angle intercepts used to draw the classic Zimm grid.

    Parameters
    ----------
    rayleigh_results : sequence of RayleighRatioResult
        One per concentration (the solvent reference, c = 0, is ignored). Need at
        least two concentrations and two angles to resolve all three quantities.
    method : str
        'zimm' (ordinate = Kc/dR; default) or 'berry' (ordinate = sqrt(Kc/dR),
        preferred for Mw above ~1 MDa or qRg approaching 1).

    Returns
    -------
    ZimmBerryResult

    Raises
    ------
    ValueError
        If method is invalid or there are too few concentrations/angles.
    """
    if method not in ('zimm', 'berry'):
        raise ValueError(f"method must be 'zimm' or 'berry', got {method!r}.")

    q2, c, kc_over_dR = _collect_zimm_points(rayleigh_results)
    all_calibrated = all(r.calibrated for r in rayleigh_results
                         if r.concentration_g_per_mL != 0)
    n_conc = np.unique(c).size
    n_ang = np.unique(np.round(q2, 12)).size
    if n_conc < 2:
        raise ValueError(
            f"Zimm/Berry analysis needs at least two concentrations, got {n_conc}."
        )
    if n_ang < 2:
        raise ValueError(
            f"Zimm/Berry analysis needs at least two angles, got {n_ang}."
        )

    if method == 'berry':
        # Berry uses sqrt(Kc/dR); a non-positive Kc/dR (noise or an
        # over-subtracted solvent reference, especially at high angle) has no
        # real square root. Drop those points -- silently feeding them to sqrt
        # produces NaNs that poison the global fit (and a RuntimeWarning).
        nonpos = ~(kc_over_dR > 0)
        if np.any(nonpos):
            n_drop = int(nonpos.sum())
            warnings.warn(
                f"Berry analysis dropped {n_drop} point(s) with a non-positive "
                "Kc/dR (noise or over-subtracted solvent reference) before the "
                "square root. If this removes too many points, use Zimm instead.",
                RuntimeWarning, stacklevel=2,
            )
            q2, c, kc_over_dR = q2[~nonpos], c[~nonpos], kc_over_dR[~nonpos]
            n_conc = np.unique(c).size
            n_ang = np.unique(np.round(q2, 12)).size
            if n_conc < 2 or n_ang < 2:
                raise ValueError(
                    "Berry analysis: after dropping non-positive Kc/dR points, "
                    f"only {n_conc} concentration(s) and {n_ang} angle(s) remain "
                    "(need at least two of each). Try the Zimm method instead."
                )
        ordinate = np.sqrt(kc_over_dR)
    else:
        ordinate = kc_over_dR

    # Global multilinear fit: ordinate = a + b q^2 + d c, with its 3x3 covariance.
    A = np.column_stack([np.ones_like(q2), q2, c])
    mf = unc.multilinear_fit(A, ordinate, estimator)
    a, b, d = float(mf.coeffs[0]), float(mf.coeffs[1]), float(mf.coeffs[2])
    cov = mf.cov                          # order (a, b, d)
    r2 = mf.r_squared

    if method == 'zimm':
        mw = 1.0 / a if a != 0 else float('nan')
        rg = math.sqrt(3.0 * b / a) if (a != 0 and b / a > 0) else float('nan')
        a2 = d / 2.0
    else:  # berry
        mw = 1.0 / a ** 2 if a != 0 else float('nan')
        rg = math.sqrt(6.0 * b / a) if (a != 0 and b / a > 0) else float('nan')
        a2 = d * a   # A2 = d' / sqrt(Mw) = d' * a'

    # Statistical SEs by first-order propagation through Cov(a,b,d). Rg uses the
    # same Jacobian form for Zimm (Rg^2=3b/a) and Berry (Rg^2=6b/a): dRg/da=-Rg/2a,
    # dRg/db=+Rg/2b. Mw and A2 differ between the two constructions.
    mw_se = rg_se = a2_se = None
    if np.all(np.isfinite(cov)) and a != 0:
        if method == 'zimm':
            mw_se = unc.se_or_none(unc.propagate([-1.0 / a ** 2, 0.0, 0.0], cov))
            a2_se = unc.se_or_none(unc.propagate([0.0, 0.0, 0.5], cov))
        else:  # berry: Mw=1/a^2, A2=d*a
            mw_se = unc.se_or_none(unc.propagate([-2.0 / a ** 3, 0.0, 0.0], cov))
            a2_se = unc.se_or_none(unc.propagate([d, 0.0, a], cov))
        if math.isfinite(rg) and rg > 0 and b != 0:
            rg_se = unc.se_or_none(
                unc.propagate([-rg / (2.0 * a), rg / (2.0 * b), 0.0], cov))

    # Per-concentration q->0 intercepts and per-angle c->0 intercepts (for the grid).
    conc_levels = np.unique(c)
    inter_per_c = []
    for cl in conc_levels:
        m = c == cl
        if m.sum() >= 2:
            s_i, i_i = np.polyfit(q2[m], ordinate[m], 1)
        else:
            i_i = ordinate[m][0]
        inter_per_c.append(i_i)
    q2_levels = np.unique(np.round(q2, 12))
    inter_per_q = []
    for ql in q2_levels:
        m = np.isclose(q2, ql)
        if np.unique(c[m]).size >= 2:
            s_i, i_i = np.polyfit(c[m], ordinate[m], 1)
        else:
            i_i = ordinate[m][0]
        inter_per_q.append(i_i)

    # Consistency cross-check: the two single-route extrapolations of the ordinate
    # intercept to (q->0, c->0) should agree. Route 1: q->0 intercepts (per c)
    # extrapolated to c->0. Route 2: c->0 intercepts (per angle) extrapolated to q->0.
    def _mw_from_intercept(a0: float) -> Optional[float]:
        if not math.isfinite(a0) or a0 == 0:
            return None
        return 1.0 / a0 if method == 'zimm' else 1.0 / a0 ** 2

    mw_q0 = mw_c0 = agree = None
    if conc_levels.size >= 2:
        mw_q0 = _mw_from_intercept(
            unc.linear_fit(conc_levels, np.array(inter_per_c)).intercept)
    if q2_levels.size >= 2:
        mw_c0 = _mw_from_intercept(
            unc.linear_fit(q2_levels, np.array(inter_per_q)).intercept)
    if mw_q0 and mw_c0 and (mw_q0 + mw_c0) != 0:
        agree = abs(mw_q0 - mw_c0) / (0.5 * abs(mw_q0 + mw_c0))

    return ZimmBerryResult(
        method=method, mw_g_per_mol=float(mw), rg_nm=float(rg),
        a2_mol_mL_per_g2=float(a2),
        coef_intercept=a, coef_q2=b, coef_c=d, r_squared=float(r2),
        n_points=int(q2.size), n_concentrations=int(n_conc), n_angles=int(n_ang),
        is_apparent=False,
        calibrated=all_calibrated,
        mw_reliable=all_calibrated,
        a2_reliable=all_calibrated,
        concentrations_g_per_mL=conc_levels,
        intercept_per_concentration=np.array(inter_per_c),
        q2_nm2=q2_levels, intercept_per_angle=np.array(inter_per_q),
        mw_se=mw_se, rg_se=rg_se, a2_se=a2_se,
        mw_from_c0_g_per_mol=mw_c0, mw_from_q0_g_per_mol=mw_q0,
        extrapolation_agreement_rel=agree,
        se_estimator=estimator,
    )


def calibration_free_a2(
    sample_results,
    angle_deg: float,
    reference_index: int = 0,
    mw_g_per_mol: Optional[float] = None,
    use_excess_intensity=None,
    estimator: str = 'hc3',
) -> CalibrationFreeA2Result:
    """A2 from intensity ratios at a fixed angle, without absolute calibration.

    For a fixed angle, Kc/dR = 1/Mw + 2 A2 c (in the low-q limit) implies the
    excess intensity I_excess proportional to c Mw /(1 + 2 A2 Mw c). Forming

        Y(c) = [I_excess(c_ref)/c_ref] / [I_excess(c)/c]

    gives a line in c whose slope/intercept = 2 A2 Mw, independent of the optical
    constant and Rayleigh-ratio calibration (they cancel in the ratio). This is
    useful when absolute calibration is uncertain, e.g. low-dn/dc systems.

    Parameters
    ----------
    sample_results : sequence
        Either RayleighRatioResult objects or (concentration, angles_deg,
        excess_intensity_array) -- anything from which an excess intensity at the
        chosen angle and the concentration can be read. RayleighRatioResult uses
        its excess_rayleigh_cm_inv at the chosen angle as the excess-intensity
        proxy (the calibration constants cancel in Y).
    angle_deg : float
        Angle at which to form the ratios (ideally the lowest available, closest
        to q -> 0).
    reference_index : int
        Which concentration (after sorting ascending, excluding c = 0) is the
        reference c_ref. Default 0 (lowest non-zero concentration).
    mw_g_per_mol : float, optional
        If provided, A2 is returned (= (slope/intercept) / (2 Mw)).

    Returns
    -------
    CalibrationFreeA2Result

    Raises
    ------
    ValueError
        If fewer than two non-zero concentrations have the requested angle.
    """
    # Build (c, excess) at the requested angle from each result.
    pts = []
    for r in sample_results:
        c = r.concentration_g_per_mL
        if c == 0:
            continue
        idx = np.where(np.isclose(r.angles_deg, angle_deg))[0]
        if idx.size == 0:
            continue
        excess = float(r.excess_rayleigh_cm_inv[int(idx[0])])
        pts.append((c, excess))
    if len(pts) < 2:
        raise ValueError(
            f"Need at least two non-zero concentrations measured at "
            f"{angle_deg} deg; found {len(pts)}."
        )
    pts.sort(key=lambda t: t[0])
    conc = np.array([p[0] for p in pts])
    excess = np.array([p[1] for p in pts])

    if not (0 <= reference_index < conc.size):
        raise ValueError(
            f"reference_index {reference_index} out of range for {conc.size} "
            f"concentrations."
        )
    c_ref = conc[reference_index]
    e_ref = excess[reference_index]

    # Y(c) = (e_ref / c_ref) / (excess / c)
    Y = (e_ref / c_ref) / (excess / conc)
    fit = unc.linear_fit(conc, Y, estimator)  # cov order [intercept, slope]
    slope, intercept = fit.slope, fit.intercept
    r2 = fit.r_squared

    two_a2_mw = float(slope / intercept) if intercept != 0 else float('nan')
    a2 = (two_a2_mw / (2.0 * mw_g_per_mol)) if mw_g_per_mol else None

    # SE of 2A2Mw = slope/intercept: slope & intercept are correlated (same fit),
    # so propagate through the 2x2 covariance, not as an independent ratio. On a short
    # concentration ladder this HC3 SE is CONSERVATIVE (meets-or-exceeds the sampling
    # spread, ~1.3x at n=5; never under-reports) because HC3 up-weights the high-leverage
    # endpoints; it tightens toward exact as the ladder lengthens. See advanced_guide
    # 15.1/11.4 and tests/test_sls.test_calibration_free_a2_se_conservative (Session 97).
    two_a2_mw_se = a2_se = None
    if intercept != 0 and math.isfinite(fit.intercept_se):
        jac = [-slope / intercept ** 2, 1.0 / intercept]   # order [intercept, slope]
        two_a2_mw_se = unc.se_or_none(unc.propagate(jac, fit.cov))
        if two_a2_mw_se is not None and mw_g_per_mol:
            a2_se = two_a2_mw_se / (2.0 * mw_g_per_mol)     # Mw treated as exact

    return CalibrationFreeA2Result(
        angle_deg=float(angle_deg), concentrations_g_per_mL=conc, Y=Y,
        slope=float(slope), intercept=float(intercept), two_a2_mw=two_a2_mw,
        a2_mol_mL_per_g2=a2, r_squared=float(r2),
        two_a2_mw_se=two_a2_mw_se, a2_se=a2_se,
        se_estimator=estimator,
    )
