"""
analysis/dls/_common.py
=======================

Shared helpers used by two or more DLS method families (cumulants, exponentials,
distributions, angular, replicate). Kept private to the package; the public API is
re-exported from ``analysis/dls/__init__.py``.

These are the low-level primitives every fit shares: the scattering vector for a
measurement, the viscosity guard, the decay-rate -> Rh conversion, the RMS-error
helper, the delay-time window (skip + tau bounds), and the distribution-method
baseline/beta/grid builders. They take data + parameters and return plain values
or arrays; none plot, write files, or mutate inputs.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from core.data_models import DLSMeasurement
from physics.constants import scattering_vector_q_m, stokes_einstein_rh


def _measurement_q_m(measurement: DLSMeasurement) -> float:
    """Scattering vector q (m^-1) for a measurement, from its angle/wavelength/n."""
    return scattering_vector_q_m(
        measurement.angle_deg,
        measurement.wavelength_nm,
        measurement.solvent_refractive_index,
    )


def _require_viscosity(measurement: DLSMeasurement) -> float:
    """Return the measurement viscosity (Pa.s), or raise if it is missing.

    Viscosity is needed only to convert a decay rate to a hydrodynamic radius
    (Stokes-Einstein). Decay rates and diffusion coefficients do not need it.
    """
    if measurement.viscosity_Pa_s is None:
        raise ValueError(
            "This measurement has no viscosity, which is required to compute a "
            "hydrodynamic radius (Rh). Set viscosity_Pa_s on the measurement, or "
            "use the decay rate / diffusion coefficient outputs, which do not "
            "need viscosity."
        )
    return measurement.viscosity_Pa_s


def _decay_rate_to_rh_nm(
    gamma_s_inv: float,
    q_m_inv: float,
    temperature_K: float,
    viscosity_Pa_s: float,
) -> float:
    """Convert a g1 decay rate Gamma (s^-1) to a hydrodynamic radius (nm).

    D = Gamma / q^2 ;  Rh = kB T / (6 pi eta D).
    """
    if gamma_s_inv <= 0:
        return float('nan')
    d_m2_s = gamma_s_inv / (q_m_inv ** 2)
    rh_m = stokes_einstein_rh(d_m2_s, temperature_K, viscosity_Pa_s)
    return rh_m * 1e9


def _rms_error(residuals: np.ndarray) -> float:
    """Root-mean-square of an array of residuals."""
    if residuals.size == 0:
        return float('nan')
    return float(np.sqrt(np.mean(residuals ** 2)))


def _apply_tau_window(
    tau: np.ndarray,
    g2m1: np.ndarray,
    tau_min_s: Optional[float],
    tau_max_s: Optional[float],
    min_points: int,
    skip_initial_channels: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Restrict (tau, g2-1) to the analysis window used by every fit.

    Two complementary restrictions, applied in this order:

      1. `skip_initial_channels` -- drop the first N correlator channels by INDEX
         (the shortest, finest-spaced lags). These leading channels are where
         detector afterpulsing and correlator dead-time artefacts live, regardless
         of their absolute lag time, so this is an instrumental cleanup, not a
         physical window. Default 0 leaves the correlogram untouched.
      2. `[tau_min_s, tau_max_s]` -- the user's physical delay-time window
         (inclusive). Either bound may be None to leave that side open.

    The two compose as an intersection: the effective first fitted point is the
    later of "channel N" and "first channel with tau >= tau_min_s". The skip is
    applied FIRST so that a spuriously inflated first channel can never distort a
    downstream intercept/amplitude-cutoff or beta estimate. This whole step runs
    before any method-specific selection (e.g. the cumulant amplitude cutoff).

    Parameters
    ----------
    tau, g2m1 : np.ndarray
        Delay times (s, ascending) and correlogram (g2 - 1).
    tau_min_s, tau_max_s : float or None
        Inclusive lower/upper delay bounds. None leaves that side open.
    min_points : int
        Minimum number of surviving points the calling model needs.
    skip_initial_channels : int
        Number of leading channels to drop by index (default 0). Must be a
        non-negative integer smaller than the number of channels.

    Returns
    -------
    (tau_windowed, g2m1_windowed)

    Raises
    ------
    ValueError
        If skip_initial_channels is invalid, tau_min_s >= tau_max_s, or fewer
        than min_points survive.
    """
    # 1. Leading-channel skip (by index), validated.
    if skip_initial_channels is None:
        skip_initial_channels = 0
    if (isinstance(skip_initial_channels, bool)
            or not isinstance(skip_initial_channels, (int, np.integer))
            or skip_initial_channels < 0):
        raise ValueError(
            f"skip_initial_channels must be a non-negative integer, "
            f"got {skip_initial_channels!r}."
        )
    skip = int(skip_initial_channels)
    if skip >= tau.size:
        raise ValueError(
            f"skip_initial_channels ({skip}) removes the whole correlogram "
            f"({tau.size} channels). Reduce it."
        )
    if skip:
        tau = tau[skip:]
        g2m1 = g2m1[skip:]

    # 2. Physical delay-time window (inclusive).
    if (tau_min_s is not None and tau_max_s is not None
            and tau_min_s >= tau_max_s):
        raise ValueError(
            f"tau_min_s ({tau_min_s}) must be < tau_max_s ({tau_max_s})."
        )
    mask = np.ones(tau.shape, dtype=bool)
    if tau_min_s is not None:
        mask &= tau >= tau_min_s
    if tau_max_s is not None:
        mask &= tau <= tau_max_s
    t = tau[mask]
    y = g2m1[mask]
    if t.size < min_points:
        raise ValueError(
            f"Only {t.size} points remain after skipping the first {skip} "
            f"channel(s) and applying the delay window [{tau_min_s}, {tau_max_s}] s; "
            f"this fit needs at least {min_points}. Reduce skip_initial_channels "
            f"or widen the window."
        )
    return t, y


# ---------------------------------------------------------------------------
# beta / baseline estimation (distribution methods)
# ---------------------------------------------------------------------------

def _estimate_baseline(tau: np.ndarray, g2m1: np.ndarray,
                       tail_fraction: float = 0.25) -> float:
    """Estimate a residual baseline offset from the long-delay tail.

    The correlogram is nominally g2 - 1 (baseline already removed, so ~0 at long
    tau), but real data drift leaves a small offset. We estimate it as the mean
    of the last `tail_fraction` of the (time-ordered) points, where the signal
    has fully decayed.
    """
    n = g2m1.size
    k = max(1, int(round(tail_fraction * n)))
    return float(np.mean(g2m1[-k:]))


def _estimate_beta(tau: np.ndarray, g2m1: np.ndarray,
                   baseline: float) -> float:
    """Estimate the coherence factor beta (intercept) for normalisation.

    Uses a short second-order cumulant-style log-polynomial extrapolation of the
    baseline-subtracted signal to tau = 0. Robust and cheap; the user can always
    override beta explicitly. Falls back to the max if the extrapolation fails.
    """
    y = g2m1 - baseline
    intercept0 = float(y.max())
    mask = (y > 0.1 * intercept0) & (y > 0)
    if mask.sum() >= 3:
        t = tau[mask]
        try:
            coeffs = np.polyfit(t, np.log(y[mask]), 2, w=y[mask])
            return float(math.exp(np.polyval(coeffs, 0.0)))
        except (ValueError, np.linalg.LinAlgError):
            pass
    return max(intercept0, 1e-9)


def _build_rh_gamma_grid(
    rh_min_nm: float,
    rh_max_nm: float,
    n_grid: int,
    q_m_inv: float,
    temperature_K: float,
    viscosity_Pa_s: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build paired log-spaced Rh (nm) and Gamma (s^-1) grids.

    Rh is log-spaced from rh_min_nm to rh_max_nm. Each Rh maps to a decay rate
    via Gamma = D q^2 with D = kB T / (6 pi eta Rh). Because Gamma is inversely
    proportional to Rh, the gamma grid is descending where rh ascends; both are
    returned in Rh-ascending order.
    """
    from physics.constants import stokes_einstein_diffusion_coefficient
    rh_grid_nm = np.geomspace(rh_min_nm, rh_max_nm, n_grid)
    gamma_grid = np.array([
        stokes_einstein_diffusion_coefficient(rh * 1e-9, temperature_K, viscosity_Pa_s) * q_m_inv ** 2
        for rh in rh_grid_nm
    ])
    return rh_grid_nm, gamma_grid
