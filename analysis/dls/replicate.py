"""
analysis/dls/replicate.py
=========================

Replicate averaging: combine several TRUE repeat correlograms (same sample,
re-measured back-to-back at the same settings, so identical lag grids) into one
denoised correlogram by a plain channel-by-channel arithmetic mean of g2(tau) - 1.

The per-channel spread is reported for completeness, but note (ISO 22412;
Schaetzel 1990) that a defensible size UNCERTAINTY does NOT come from one averaged
curve's lag channels (they are correlated): it comes from the spread of the
per-replicate FITTED results, handled separately by the controller's derived-results
averaging (analysis/uncertainty.replicate_mean_se).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from core.data_models import DLSMeasurement
from analysis.dls._common import _measurement_q_m


@dataclass
class AveragedCorrelogramResult:
    """The channel-by-channel mean of several replicate correlograms.

    A "true replicate set" is the same sample re-measured back-to-back on the
    same instrument at the same settings (e.g. the ALV 10-run set), so every run
    lands on the IDENTICAL lag-time grid. Averaging then is a plain element-wise
    mean of g2(tau) - 1, which denoises the curve without distorting it.

    The per-channel spread is reported for completeness, but note (ISO 22412;
    Schaetzel 1990) that a defensible size UNCERTAINTY does NOT come from one
    averaged curve's lag channels (they are correlated): it comes from the spread
    of the per-replicate FITTED results, handled separately by the controller's
    derived-results averaging (analysis/uncertainty.replicate_mean_se).
    """
    delay_times_s: np.ndarray          # the shared lag grid (identical across runs)
    mean_g2m1: np.ndarray              # channel-by-channel mean of g2 - 1
    sd_g2m1: np.ndarray                # per-channel sample SD across replicates (ddof=1)
    sem_g2m1: np.ndarray               # per-channel SD / sqrt(n)
    n_replicates: int
    # carried identity (all replicates share these; taken from the first)
    q_m_inv: float


def average_replicate_correlograms(
    measurements: List[DLSMeasurement],
    rtol: float = 1e-6,
) -> AveragedCorrelogramResult:
    """Average a set of TRUE replicate correlograms onto their shared lag grid.

    All measurements must be genuine repeats of one sample: identical lag grids
    and matching optics/identity (polymer, solvent, temperature, angle,
    wavelength, refractive index). The averaging is a plain channel-by-channel
    arithmetic mean of g2(tau) - 1 -- valid because true replicates share the
    coherence factor beta, and any small residual amplitude difference is absorbed
    by the floating beta in the subsequent fit.

    Identical lag grids are REQUIRED, not interpolated to: resampling onto a
    common grid would blend neighboring lag channels (which are already
    correlated, Schaetzel 1990) and would silently let non-replicates be averaged,
    defeating the "true replicates only" guard.

    Parameters
    ----------
    measurements : list of DLSMeasurement
        Two or more replicate correlograms.
    rtol : float
        Relative tolerance for the identical-grid and matching-optics checks.

    Returns
    -------
    AveragedCorrelogramResult

    Raises
    ------
    ValueError
        If fewer than two measurements are given, the lag grids are not
        identical (same length and values within rtol), or the identity/optics
        fields do not match across the set.
    """
    n = len(measurements)
    if n < 2:
        raise ValueError(
            f"Replicate averaging needs at least two correlograms, got {n}."
        )

    first = measurements[0]
    grid = first.delay_times_s

    # 1) Identical lag grids (same length, same values). No interpolation.
    for i, m in enumerate(measurements[1:], start=1):
        if m.delay_times_s.shape != grid.shape:
            raise ValueError(
                f"Replicate {i} has {m.delay_times_s.size} lag channels but the "
                f"first has {grid.size}. Correlogram averaging requires identical "
                f"lag grids (true replicates from one instrument/setting). Use "
                f"'average derived results' instead to average fits across runs "
                f"with differing grids."
            )
        # A tiny grid-scaled atol (not 0): a pure-relative compare demands an EXACT
        # match on a leading tau=0 channel (rtol*0 == 0), so sub-ns float drift there
        # would wrongly reject genuine replicates. atol stays far below the first real
        # lag channel, so distinct grids still differ.
        grid_atol = 1e-9 * float(np.max(np.abs(grid))) if grid.size else 0.0
        if not np.allclose(m.delay_times_s, grid, rtol=rtol, atol=grid_atol):
            raise ValueError(
                f"Replicate {i}'s lag-time grid differs from the first. "
                f"Correlogram averaging requires identical lag grids; these are "
                f"not the same measurement settings. Use 'average derived "
                f"results' to average fits across runs with differing grids."
            )

    # 2) Matching identity / optics. Strings exact (case/space-insensitive),
    #    floats within rtol. q depends on angle/wavelength/n, so a mismatch there
    #    would mean the correlograms describe different physics.
    def _norm(s: Optional[str]) -> str:
        return (s or '').strip().lower()

    for i, m in enumerate(measurements[1:], start=1):
        if _norm(m.polymer_name) != _norm(first.polymer_name):
            raise ValueError(
                f"Replicate {i} polymer ({m.polymer_name!r}) does not match the "
                f"first ({first.polymer_name!r})."
            )
        if _norm(m.solvent_name) != _norm(first.solvent_name):
            raise ValueError(
                f"Replicate {i} solvent ({m.solvent_name!r}) does not match the "
                f"first ({first.solvent_name!r})."
            )
        for attr, label in (('temperature_K', 'temperature'),
                            ('angle_deg', 'angle'),
                            ('wavelength_nm', 'wavelength'),
                            ('solvent_refractive_index', 'refractive index')):
            a, b = getattr(m, attr), getattr(first, attr)
            if not math.isclose(a, b, rel_tol=rtol, abs_tol=0.0):
                raise ValueError(
                    f"Replicate {i} {label} ({a:g}) does not match the first "
                    f"({b:g}). Correlogram averaging is only valid across true "
                    f"replicates measured at the same conditions."
                )

    stack = np.vstack([m.correlogram for m in measurements])   # (n, n_channels)
    mean_g2m1 = stack.mean(axis=0)
    sd_g2m1 = stack.std(axis=0, ddof=1)
    sem_g2m1 = sd_g2m1 / math.sqrt(n)

    return AveragedCorrelogramResult(
        delay_times_s=np.array(grid, dtype=float),
        mean_g2m1=mean_g2m1, sd_g2m1=sd_g2m1, sem_g2m1=sem_g2m1,
        n_replicates=n, q_m_inv=_measurement_q_m(first),
    )
