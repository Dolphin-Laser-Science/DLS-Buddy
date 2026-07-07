"""
analysis/dls/angular.py
=======================

Shared multi-angle / multi-concentration front-end. These functions combine the
single-measurement fits (cumulant / single-exponential) across angle (to test
whether motion is diffusive and extract D) and across concentration (to extrapolate
to infinite dilution). They also provide the public Rh <-> Gamma converters used by
the visualisation layer.

Uncertainty (invariant 8): each angle / each concentration is an independent
measurement, so the regression standard errors here ARE defensible (statistical
only, excluding calibration systematics) -- unlike a single correlogram. They are
computed via analysis/uncertainty.py.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from core.data_models import DLSMeasurement
from analysis import uncertainty as unc
from physics.constants import stokes_einstein_rh
from analysis.dls._common import _decay_rate_to_rh_nm, _measurement_q_m
from analysis.dls.cumulants import fit_cumulants
from analysis.dls.exponentials import fit_single_exponential


@dataclass
class GammaQ2Result:
    """Multi-angle Gamma vs q^2 analysis for one sample (one concentration).

    For purely translational diffusion, Gamma = D q^2: a straight line through
    the origin, with the apparent diffusion coefficient D_app = Gamma/q^2
    independent of angle. Deviations are diagnostic:
      - a nonzero intercept (unconstrained fit) indicates a non-diffusive
        contribution or a slow relaxation mode;
      - an upward trend in D_app with q^2 indicates internal/segmental modes
        (q Rg > 1);
      - low R^2 indicates the data are simply not linear in q^2.
    """
    angles_deg: np.ndarray            # sorted by q^2 ascending
    q_m_inv: np.ndarray
    q2_m2: np.ndarray                 # q^2 for each angle (m^-2)
    gamma_s_inv: np.ndarray           # decay rate at each angle
    d_app_m2_s: np.ndarray            # Gamma/q^2 at each angle
    # through-origin fit (the translational D)
    d_m2_s: float                     # slope of Gamma = D q^2 through origin
    rh_nm: float                      # Rh from D
    # linearity / deviation diagnostics
    r_squared: float                  # R^2 of the unconstrained linear fit
    intercept_s_inv: float            # intercept of the unconstrained fit
    intercept_relative: float         # intercept / mean(Gamma)
    d_app_trend_rel: float            # (slope of D_app vs q^2) * q2_range / mean(D_app)
    is_diffusive: bool                # passes the general linearity criteria
    # context
    gamma_source: str
    temperature_K: float
    r2_threshold: float
    intercept_rel_threshold: float
    # statistical (regression) standard errors (each angle is an independent point)
    d_se: Optional[float] = None
    rh_se: Optional[float] = None
    se_estimator: str = 'hc3'             # covariance estimator behind the SEs
    # every eligible measurement's per-point Γ/q²/D_app + quality tag, attached by the
    # controller run so the GUI table/greying come from the run (feedback 2026-07-06).
    all_points: Optional[list] = None


@dataclass
class ConcentrationExtrapolationResult:
    """Extrapolation of the apparent diffusion coefficient to infinite dilution.

    D(c) = D0 (1 + kD c + ...).  D0 gives the infinite-dilution Rh; kD is the
    diffusion interaction parameter (positive in good solvents, negative in poor).
    """
    concentrations_g_per_mL: np.ndarray   # sorted ascending
    d_values_m2_s: np.ndarray
    d0_m2_s: float                    # intercept: D at c -> 0
    rh0_nm: float                     # Rh from D0
    kd_mL_per_g: float                # slope / D0
    slope: float                      # dD/dc  (m^2/s per g/mL)
    r_squared: float
    temperature_K: float
    n_concentrations: int
    # statistical (regression) standard errors (each concentration independent)
    d0_se: Optional[float] = None
    rh0_se: Optional[float] = None
    kd_se: Optional[float] = None
    se_estimator: str = 'hc3'             # covariance estimator behind the SEs
    # every eligible measurement's per-point D_app + quality tag, attached by the
    # controller run so the GUI table/greying come from the run (feedback 2026-07-06).
    all_points: Optional[list] = None


# ---------------------------------------------------------------------------
# Rh <-> Gamma converters (for the visualisation toggle and general use)
# ---------------------------------------------------------------------------

def gamma_to_rh_nm(gamma_s_inv, q_m_inv, temperature_K, viscosity_Pa_s):
    """Public converter: g1 decay rate Gamma (s^-1) -> hydrodynamic radius (nm)."""
    return _decay_rate_to_rh_nm(gamma_s_inv, q_m_inv, temperature_K, viscosity_Pa_s)


def rh_nm_to_gamma(rh_nm, q_m_inv, temperature_K, viscosity_Pa_s):
    """Public converter: hydrodynamic radius (nm) -> g1 decay rate Gamma (s^-1).

    Gamma = D q^2 with D = kB T / (6 pi eta Rh).
    """
    from physics.constants import stokes_einstein_diffusion_coefficient
    if rh_nm <= 0:
        return float('nan')
    d = stokes_einstein_diffusion_coefficient(rh_nm * 1e-9, temperature_K, viscosity_Pa_s)
    return d * q_m_inv ** 2


# ---------------------------------------------------------------------------
# Gamma vs q^2 (multi-angle)
# ---------------------------------------------------------------------------

def _gamma_for_measurement(measurement, gamma_source, tau_min_s, tau_max_s,
                           skip_initial_channels=0, cumulant_method='linear'):
    """Extract a single decay rate Gamma from one measurement."""
    if gamma_source == 'cumulant':
        return fit_cumulants(measurement, order=2,
                             tau_min_s=tau_min_s, tau_max_s=tau_max_s,
                             skip_initial_channels=skip_initial_channels,
                             method=cumulant_method).gamma_s_inv
    elif gamma_source == 'single':
        return fit_single_exponential(
            measurement, tau_min_s=tau_min_s, tau_max_s=tau_max_s,
            skip_initial_channels=skip_initial_channels).mode.gamma_s_inv
    else:
        raise ValueError(
            f"gamma_source must be 'cumulant' or 'single', got {gamma_source!r}."
        )


def gamma_per_measurement(measurements, gamma_source='cumulant', tau_min_s=None,
                          tau_max_s=None, skip_initial_channels=0,
                          cumulant_method='linear'):
    """Per-measurement decay rate Γ (s⁻¹) and scattering vector q (m⁻¹), in INPUT
    order (no sorting, no ≥2 validation), for populating the GUI's per-measurement
    table (feedback 2026-06-30 #11/#13). It reuses the exact per-measurement path the
    multi-angle/-concentration fits use (`_gamma_for_measurement`), so the tabulated Γ
    match the fitted points. Failure-tolerant: a measurement whose fit or q can't be
    computed yields NaN for that entry rather than raising.

    Returns (gamma, q) as float ndarrays aligned with `measurements`.
    """
    gammas, qs = [], []
    for m in measurements:
        try:
            g = float(_gamma_for_measurement(
                m, gamma_source, tau_min_s, tau_max_s,
                skip_initial_channels, cumulant_method))
        except Exception:
            g = float('nan')
        try:
            q = float(_measurement_q_m(m))
        except Exception:
            q = float('nan')
        gammas.append(g)
        qs.append(q)
    return np.asarray(gammas, dtype=float), np.asarray(qs, dtype=float)


def analyze_gamma_q2(
    measurements: List[DLSMeasurement],
    gamma_source: str = 'cumulant',
    tau_min_s: Optional[float] = None,
    tau_max_s: Optional[float] = None,
    r2_threshold: float = 0.98,
    intercept_rel_threshold: float = 0.1,
    skip_initial_channels: int = 0,
    cumulant_method: str = 'linear',
    estimator: str = 'hc3',
) -> GammaQ2Result:
    """Analyse Gamma vs q^2 across angles for one sample.

    Fits a decay rate at each angle, then examines its dependence on q^2.
    Translational diffusion gives Gamma = D q^2 (a line through the origin); the
    through-origin slope is the diffusion coefficient. Linearity diagnostics
    flag non-diffusive behaviour.

    Parameters
    ----------
    measurements : list of DLSMeasurement
        Two or more measurements of the SAME sample (same polymer, solvent,
        concentration, temperature) at DIFFERENT angles.
    gamma_source : str
        How to get Gamma at each angle: 'cumulant' (first cumulant, default) or
        'single' (single-exponential fit).
    tau_min_s, tau_max_s : float, optional
        Delay-time window passed through to the per-angle fits.
    r2_threshold : float
        Minimum R^2 for the is_diffusive flag (default 0.98). A general
        statistical criterion, not a system-specific tuning value.
    intercept_rel_threshold : float
        Maximum |intercept|/mean(Gamma) for the is_diffusive flag (default 0.1).

    Returns
    -------
    GammaQ2Result

    Raises
    ------
    ValueError
        If fewer than two measurements, or they are not at distinct angles.
    """
    if len(measurements) < 2:
        raise ValueError(
            f"Gamma vs q^2 needs at least two angles, got {len(measurements)}."
        )

    # Check they are the same sample (sample_key ignores angle, so all should match).
    keys = {m.sample_key for m in measurements}
    if len(keys) > 1:
        warnings.warn(
            "Measurements passed to analyze_gamma_q2 do not all share the same "
            "sample identity (polymer/solvent/concentration/temperature). Gamma "
            "vs q^2 assumes one sample across angles.",
            UserWarning, stacklevel=2,
        )

    angles = np.array([m.angle_deg for m in measurements], dtype=float)
    if np.unique(angles).size < 2:
        raise ValueError(
            "Gamma vs q^2 needs measurements at distinct angles; the angles "
            "supplied are not all different."
        )

    q = np.array([_measurement_q_m(m) for m in measurements])
    q2 = q ** 2
    gamma = np.array([
        _gamma_for_measurement(m, gamma_source, tau_min_s, tau_max_s,
                               skip_initial_channels, cumulant_method)
        for m in measurements
    ])

    # sort by q^2 ascending for clean output
    order = np.argsort(q2)
    angles, q, q2, gamma = angles[order], q[order], q2[order], gamma[order]
    d_app = gamma / q2

    # Through-origin slope D (Gamma = D q^2) with its standard error (each angle is
    # an independent measurement, so the regression SE is defensible).
    d, d_se = unc.linear_fit_through_origin(q2, gamma, estimator)
    d = float(d)

    # Unconstrained linear fit for the intercept and R^2.
    slope_u, intercept_u = np.polyfit(q2, gamma, 1)
    gamma_fit = slope_u * q2 + intercept_u
    ss_res = float(np.sum((gamma - gamma_fit) ** 2))
    ss_tot = float(np.sum((gamma - gamma.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    intercept_rel = intercept_u / gamma.mean() if gamma.mean() != 0 else float('nan')

    # q-dependence of D_app (internal-mode indicator), expressed relatively.
    if q2.max() > q2.min():
        dapp_slope = np.polyfit(q2, d_app, 1)[0]
        dapp_trend_rel = float(dapp_slope * (q2.max() - q2.min()) / d_app.mean())
    else:
        dapp_trend_rel = float('nan')

    temperature_K = measurements[0].temperature_K
    eta = measurements[0].viscosity_Pa_s
    if eta is not None and d > 0:
        rh = stokes_einstein_rh(d, temperature_K, eta) * 1e9
    else:
        rh = float('nan')

    # Rh ~ 1/D, so the fractional SE carries over (sigma_Rh/Rh = sigma_D/D).
    rh_se = unc.power_law_se(rh, d, d_se, -1) if math.isfinite(rh) else None

    is_diffusive = bool(
        math.isfinite(r2) and r2 >= r2_threshold
        and abs(intercept_rel) <= intercept_rel_threshold
    )

    return GammaQ2Result(
        angles_deg=angles, q_m_inv=q, q2_m2=q2, gamma_s_inv=gamma,
        d_app_m2_s=d_app, d_m2_s=d, rh_nm=rh,
        r_squared=r2, intercept_s_inv=float(intercept_u),
        intercept_relative=float(intercept_rel),
        d_app_trend_rel=dapp_trend_rel, is_diffusive=is_diffusive,
        gamma_source=gamma_source, temperature_K=temperature_K,
        r2_threshold=r2_threshold, intercept_rel_threshold=intercept_rel_threshold,
        d_se=unc.se_or_none(d_se), rh_se=unc.se_or_none(rh_se),
        se_estimator=estimator,
    )


# ---------------------------------------------------------------------------
# Concentration extrapolation
# ---------------------------------------------------------------------------

def extrapolate_diffusion_vs_concentration(
    measurements: List[DLSMeasurement],
    gamma_source: str = 'cumulant',
    tau_min_s: Optional[float] = None,
    tau_max_s: Optional[float] = None,
    skip_initial_channels: int = 0,
    cumulant_method: str = 'linear',
    estimator: str = 'hc3',
) -> ConcentrationExtrapolationResult:
    """Extrapolate the apparent diffusion coefficient to infinite dilution.

    For each measurement, D_app = Gamma / q^2 is computed (Gamma from the chosen
    source). D_app is then fit linearly against concentration and extrapolated to
    c -> 0:

        D(c) = D0 (1 + kD c)

    giving the infinite-dilution diffusion coefficient D0 (and thus Rh0) and the
    diffusion interaction parameter kD = slope / D0.

    The measurements should be the same polymer/solvent/temperature at different
    concentrations. They may be at a single common angle (the usual case) or each
    reduced via Gamma/q^2; since D_app is angle-independent for pure diffusion,
    either works, but mixing angles will fold any q-dependence into the result.

    Parameters
    ----------
    measurements : list of DLSMeasurement
        Two or more measurements at different concentrations.
    gamma_source : str
        'cumulant' (default) or 'single'.
    tau_min_s, tau_max_s : float, optional
        Delay-time window passed to the per-measurement fits.

    Returns
    -------
    ConcentrationExtrapolationResult

    Raises
    ------
    ValueError
        If fewer than two distinct concentrations are supplied.
    """
    if len(measurements) < 2:
        raise ValueError(
            f"Concentration extrapolation needs at least two measurements, got "
            f"{len(measurements)}."
        )

    # Warn if polymer/solvent/temperature are not consistent.
    ids = {(m.polymer_name, m.solvent_name, m.temperature_K) for m in measurements}
    if len(ids) > 1:
        warnings.warn(
            "Measurements passed to extrapolate_diffusion_vs_concentration do not "
            "all share polymer/solvent/temperature. Concentration extrapolation "
            "assumes one system.",
            UserWarning, stacklevel=2,
        )

    conc = np.array([m.concentration_g_per_mL for m in measurements], dtype=float)
    if np.unique(conc).size < 2:
        raise ValueError(
            "Concentration extrapolation needs at least two distinct "
            "concentrations; the values supplied are not all different."
        )

    d_app = np.array([
        _gamma_for_measurement(m, gamma_source, tau_min_s, tau_max_s,
                               skip_initial_channels, cumulant_method)
        / _measurement_q_m(m) ** 2
        for m in measurements
    ])

    order = np.argsort(conc)
    conc, d_app = conc[order], d_app[order]

    lf = unc.linear_fit(conc, d_app, estimator)  # D(c) = D0 + slope*c; cov [intercept, slope]
    slope, intercept = lf.slope, lf.intercept
    d0 = float(intercept)
    r2 = lf.r_squared
    kd = float(slope / d0) if d0 != 0 else float('nan')

    # Statistical SEs: D0 from the intercept; kD = slope/D0 through the 2x2 cov.
    d0_se = unc.se_or_none(lf.intercept_se)
    kd_se = None
    if d0 != 0 and math.isfinite(lf.intercept_se):
        kd_se = unc.se_or_none(unc.propagate(
            [-slope / d0 ** 2, 1.0 / d0], lf.cov))    # order [intercept, slope]

    temperature_K = measurements[0].temperature_K
    eta = measurements[0].viscosity_Pa_s
    if eta is not None and d0 > 0:
        rh0 = stokes_einstein_rh(d0, temperature_K, eta) * 1e9
    else:
        rh0 = float('nan')
    rh0_se = unc.power_law_se(rh0, d0, d0_se, -1) if math.isfinite(rh0) else None

    return ConcentrationExtrapolationResult(
        concentrations_g_per_mL=conc, d_values_m2_s=d_app,
        d0_m2_s=d0, rh0_nm=rh0, kd_mL_per_g=kd, slope=float(slope),
        r_squared=r2, temperature_K=temperature_K,
        n_concentrations=int(np.unique(conc).size),
        d0_se=d0_se, rh0_se=unc.se_or_none(rh0_se), kd_se=kd_se,
        se_estimator=estimator,
    )
