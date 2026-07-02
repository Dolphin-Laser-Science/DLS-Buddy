"""Synthetic DLS forward-model builders — one unified copy.

Previously each of the retired DLS-module, channel-skip, and nonlinear-cumulant
validators carried its own slightly-drifted copy of these helpers (two returned
``(meas, tau)``, one returned just ``meas``; one added baseline/noise options).
This is the single source used by the whole suite.

The model is the program's OWN forward model — a closed Stokes-Einstein round
trip — so a recovered Rh should equal the input Rh to within the fit's numerical
accuracy. ``monomodal`` returns a :class:`~core.data_models.DLSMeasurement`; the
delay grid is always available as ``meas.delay_times_s`` (no separate ``tau``
return value needed).
"""
from __future__ import annotations

import math

import numpy as np

from core.data_models import DLSMeasurement

# Boltzmann constant (J/K) — matches physics.constants.BOLTZMANN_K; kept local so
# these builders are self-contained forward models independent of the code under test.
KB = 1.380649e-23

# Default optical geometry for the synthetic builders (633 nm HeNe, water).
_LAM_NM = 633.0
_N = 1.33
_T = 298.15
_ETA = 8.9e-4


def q_m(angle_deg=90.0, lam_nm=_LAM_NM, n=_N):
    """Scattering vector q in m^-1 for the synthetic geometry."""
    return (4 * np.pi * n / lam_nm) * math.sin(math.radians(angle_deg) / 2) * 1e9


def gamma_for_rh(rh_nm, angle_deg=90.0, lam_nm=_LAM_NM, n=_N, T=_T, eta=_ETA):
    """Decay rate Gamma (s^-1) for a monodisperse Rh via Stokes-Einstein."""
    D = KB * T / (6 * np.pi * eta * rh_nm * 1e-9)          # m^2/s
    return D * q_m(angle_deg, lam_nm, n) ** 2              # s^-1


def rh_from_D(D, T=_T, eta=_ETA):
    """Inverse Stokes-Einstein: Rh (nm) from a diffusion coefficient (m^2/s)."""
    return KB * T / (6 * np.pi * eta * D) * 1e9


def tau_grid(n_pts=200, tau0_s=2.5e-8, tau_max_s=1.0):
    """Log-spaced delay grid, 25 ns .. 1 s (typical DLS correlator span)."""
    return np.geomspace(tau0_s, tau_max_s, n_pts)


def make_measurement(g2m1, tau, angle_deg=90.0, lam_nm=_LAM_NM, n=_N, T=_T,
                     eta=_ETA, conc=None):
    """Wrap a (g2-1, tau) pair in a DLSMeasurement with the synthetic parameters."""
    return DLSMeasurement(
        delay_times_s=tau, correlogram=g2m1, polymer_name="synthetic",
        solvent_name="water", concentration_g_per_mL=conc, temperature_K=T,
        angle_deg=angle_deg, wavelength_nm=lam_nm, solvent_refractive_index=n,
        viscosity_Pa_s=eta)


def monomodal(rh_nm, beta=0.9, conc=None, baseline=0.0, noise=0.0, seed=0, **opt):
    """Monodisperse correlogram g2-1 = B + beta * g1(tau)^2 for one Rh.

    ``baseline`` adds a flat offset B (to exercise the floating-baseline fit);
    ``noise`` adds Gaussian noise with a fixed ``seed`` (deterministic).
    ``opt`` forwards geometry (angle_deg, lam_nm, n, T, eta) to both the decay
    rate and the measurement so they stay consistent.
    """
    tau = tau_grid()
    g = baseline + beta * np.exp(-gamma_for_rh(rh_nm, **opt) * tau) ** 2
    if noise:
        g = g + np.random.RandomState(seed).normal(0.0, noise, g.size)
    return make_measurement(g, tau, conc=conc, **opt)


def bimodal(rh_small, rh_large, f_small=0.5, beta=0.9, conc=None, **opt):
    """Two-population correlogram: field amplitudes f_small / (1 - f_small)."""
    tau = tau_grid()
    g1 = (f_small * np.exp(-gamma_for_rh(rh_small, **opt) * tau)
          + (1 - f_small) * np.exp(-gamma_for_rh(rh_large, **opt) * tau))
    return make_measurement(beta * g1 ** 2, tau, conc=conc, **opt)


# --------------------------------------------------------------- real-data loaders

def load_alv(path):
    """Load every angle of an ALV .ASC file into a list of DLSMeasurement.

    Returns ``None`` if the file is missing (so a test can skip gracefully).
    """
    from pathlib import Path

    from parsers.alv_asc import ALVCorrelatorParser

    p = Path(path)
    if not p.exists():
        return None
    out = []
    for pv in ALVCorrelatorParser().parse(str(p)):
        eta = getattr(pv, "viscosity_Pa_s", None) or _ETA
        out.append(DLSMeasurement(
            delay_times_s=pv.delay_times_s, correlogram=pv.correlogram,
            polymer_name="PEG", solvent_name="water", concentration_g_per_mL=None,
            temperature_K=pv.temperature_K, angle_deg=pv.angle_deg,
            wavelength_nm=pv.wavelength_nm,
            solvent_refractive_index=pv.solvent_refractive_index, viscosity_Pa_s=eta))
    return out


def nearest_angle(meas_list, target=90.0):
    """The measurement whose angle is closest to ``target`` degrees."""
    return min(meas_list, key=lambda m: abs(m.angle_deg - target))


def load_alv_as(path, polymer, solvent, concentration_g_per_mL=None,
                temperature_K=None):
    """Parse an ALV .ASC (single- or multi-angle) into DLSMeasurement(s) with an
    explicit sample identity.

    ``temperature_K`` overrides each file's logged temperature — useful for true
    replicates whose recorded T jitters at the sub-mK level (real instrument
    noise), which ``average_replicate_correlograms``' strict equality check would
    otherwise reject. Returns a list (length 1 for single-angle files).
    """
    from parsers.alv_asc import ALVCorrelatorParser

    out = []
    for pv in ALVCorrelatorParser().parse(str(path)):
        out.append(DLSMeasurement(
            delay_times_s=pv.delay_times_s, correlogram=pv.correlogram,
            polymer_name=polymer, solvent_name=solvent,
            concentration_g_per_mL=concentration_g_per_mL,
            temperature_K=(temperature_K if temperature_K is not None
                           else pv.temperature_K),
            angle_deg=pv.angle_deg, wavelength_nm=pv.wavelength_nm,
            solvent_refractive_index=pv.solvent_refractive_index,
            viscosity_Pa_s=pv.viscosity_Pa_s))
    return out


def smals_replicate_averaged():
    """Per-angle replicate-averaged SMALS correlograms from ``test-data/ALV/``.

    Loads the 10 committed Noisy050Latex0004 .ASC replicates, averages the
    correlograms per angle (analysis.dls.average_replicate_correlograms), and
    returns ``{angle_deg: DLSMeasurement}``. Returns ``None`` if the committed
    files are absent.
    """
    from analysis import dls as E
    from parsers.alv_asc import ALVCorrelatorParser

    from fixtures.data_paths import SMALS_FILES

    if not SMALS_FILES[0].exists():
        return None
    runs = [{round(p.angle_deg, 3): p for p in ALVCorrelatorParser().parse(str(f))}
            for f in SMALS_FILES]
    angles = sorted(runs[0])
    meta = runs[0][angles[0]]
    out = {}
    for ang in angles:
        ms = [DLSMeasurement(
            delay_times_s=r[ang].delay_times_s, correlogram=r[ang].correlogram,
            polymer_name="Latex", solvent_name="water", concentration_g_per_mL=None,
            temperature_K=meta.temperature_K, angle_deg=ang,
            wavelength_nm=meta.wavelength_nm,
            solvent_refractive_index=meta.solvent_refractive_index,
            viscosity_Pa_s=meta.viscosity_Pa_s) for r in runs]
        avg = E.average_replicate_correlograms(ms)
        out[ang] = DLSMeasurement(
            delay_times_s=avg.delay_times_s, correlogram=avg.mean_g2m1,
            polymer_name="Latex", solvent_name="water", concentration_g_per_mL=None,
            temperature_K=meta.temperature_K, angle_deg=ang,
            wavelength_nm=meta.wavelength_nm,
            solvent_refractive_index=meta.solvent_refractive_index,
            viscosity_Pa_s=meta.viscosity_Pa_s)
    return out
