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

from core.data_models import (
    DLSMeasurement,
    casefold_identity_name,
    round_temperature_for_identity,
)
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
    # carried identity (all replicates share these -- enforced by
    # check_replicate_identity, not merely assumed; taken from the first)
    q_m_inv: float


def check_replicate_identity(
    measurements: List[DLSMeasurement],
    rtol: float = 1e-6,
) -> None:
    """Raise ValueError unless every measurement is a true replicate of the first.

    THE one answer to "are these the same measurement, repeated?", shared by both
    averaging paths: correlogram averaging (below) and the controller's
    derived-results averaging. (The DDLS fit settles its own identity question
    once, across every bucket, with the same shared sample_key comparison -- see
    run_ddls.) Two copies of an identity check is how the next field goes missing
    from one of them -- concentration and mw_fraction were missing from the only
    copy for the whole of the platform's life.

    The REFUSAL is kept by decision (combination doctrine rule 1's backstop,
    core/data_models.py; owner-ruled 2026-08-13, confirming 2026-08-08): a mean
    over non-replicates describes no real condition, and the replicate +/- is the
    platform's only honest single-sample dynamic SE precisely BECAUSE the inputs
    are true repeats (invariant 8 / ISO 22412). The known false-refusal surface
    was engineered away (the three uncompared fields below), and the "user knows
    better" cases have data-side remedies -- correct the metadata, which the user
    owns -- that beat an override. Flip trigger, recorded: ONE real selection
    wrongly refused in ordinary use reopens this as a warn-and-mark design; if
    friction appears, prefer auto-partitioning (average each condition separately
    in one gesture) before any override.

    EIGHT fields must agree: polymer, solvent, temperature, angle, wavelength,
    concentration, mw_fraction, analyzer_geometry. Each is a name, a category, an
    instrument setting, or a quantity the user enters once per sample -- none has a
    per-run drift mechanism, so the check has no false-refusal surface to size.

    THREE physical fields are deliberately NOT compared: ``viscosity_Pa_s``,
    ``solvent_refractive_index`` and ``solute_refractive_index``. The reason is the
    same for all three and it is worth stating plainly, because two earlier attempts
    got it wrong in opposite directions (rtol=1e-6 refused real replicate sets; a
    1e-2 tolerance was mis-sized against water and would have refused reline and
    glycerol repeats):

        The program cannot validate that a solvent property matches the solvent it
        is labelled with. A single mislabelled file yields a wrong size whether or
        not two files are averaged, and two consistently-mislabelled files yield a
        wrong size with every check green. Comparing these fields does not detect
        bad input -- it detects DISAGREEMENT between inputs, which is a coincidence
        (one good file, one bad) rather than the failure mode. Owner decision,
        2026-08-08: a check that only fires on that coincidence does not earn a
        tolerance constant.

    So an averaged measurement inherits the FIRST replicate's viscosity and both
    refractive indices. Consequences, recorded rather than implied: viscosity and
    the solvent index set the size through Rh = kT / 6 pi eta D and q = 4 pi n /
    lambda sin(theta/2); the solute index sets the out-of-Rayleigh (RDG) and Mie
    distribution reweighting, where at n_medium = 1.333 a solute index of 1.45
    yields usable volume weights while 1.59 blanks the whole reweighted
    distribution to NaN. Garbage in, garbage out -- the parameters are the user's,
    as invariant 3 requires.

    Of DLSMeasurement's remaining fields, four describe the FILE rather than the
    physics -- sample_label, instrument_name, source_file, trace -- and differ
    between genuine repeats by construction.

    * polymer and solvent -- compared through THE identity-comparison function
      (strip + casefold; core.data_models.casefold_identity_name), the same one
      make_sample_key and sample_group_key use, so "are these the same sample?"
      has no private answer here.
    * temperature -- through THE sample-identity bucket
      (core.data_models.round_temperature_for_identity, 0.1 C), for the same
      reason: one computation, so the averager cannot be stricter about "same
      temperature" than the grouping that put these runs in one sample. An
      earlier rtol=1e-6 compare here was ~300x tighter than that bucket and
      refused real per-run thermometer drift (owner ruling, round 1).
    * angle and wavelength -- within rtol. Both are instrument settings written as
      fixed text, so they have no drift mechanism; q depends on both, so a mismatch
      means the correlograms describe different physics.
    * concentration -- within rtol when both are known; both-None passes (unknown
      is the ordinary DLS state), one-None mismatches, since unknown-vs-known is
      not proven identity. D_app = D0(1 + k_D c + ...), so a mean over two
      concentrations is the correlogram of neither and a pooled Rh +/- is not an
      ISO 22412 replicate statistic (repeats of ONE condition).
    * mw_fraction and analyzer_geometry -- EXACT string equality (both-None
      passes). Not casefolded, unlike the names above: Sample.result_for keys
      SampleResults by the raw mw_fraction, so 'A' and 'a' already address
      different result buckets, and analyzer_geometry is already normalized to a
      canonical token by DLSMeasurement.__post_init__. Geometry is identity, not
      an optic: VV and VH runs of one sample agree on every other field here, and
      Gamma_VH = Gamma_VV + 6 D_r is a physical difference (invariant 7 --
      polarized and depolarized are never conflated).

    Note this deliberately does NOT check the lag grid: derived-results averaging
    fits each replicate on its own grid and needs no shared one.
    """
    if not measurements:
        return
    first = measurements[0]

    def _norm(s: Optional[str]) -> str:
        return casefold_identity_name(s or '')

    # start=2: measurements[1] is the SECOND run the user selected, so a 1-based
    # label for it is 2. start=1 named every mismatch one position low.
    for i, m in enumerate(measurements[1:], start=2):
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
        # Temperature through the SAME bucket the sample grouping uses, so the two
        # identity questions are one computation and the averager can never be
        # stricter than the grouping that put these runs in one sample.
        ta, tb = (round_temperature_for_identity(m.temperature_K),
                  round_temperature_for_identity(first.temperature_K))
        if ta != tb:
            raise ValueError(
                f"Replicate {i} temperature ({m.temperature_K:g} K) does not match "
                f"the first ({first.temperature_K:g} K). Replicate averaging is "
                f"only valid across true replicates measured at the same "
                f"conditions."
            )

        for attr, label in (('angle_deg', 'angle'),
                            ('wavelength_nm', 'wavelength')):
            a, b = getattr(m, attr), getattr(first, attr)
            if not math.isclose(a, b, rel_tol=rtol, abs_tol=0.0):
                raise ValueError(
                    f"Replicate {i} {label} ({a:g}) does not match the first "
                    f"({b:g}). Replicate averaging is only valid across true "
                    f"replicates measured at the same conditions."
                )

        # Concentration takes its own arm rather than joining the loop above:
        # math.isclose RAISES TypeError on None, and None (concentration unknown or
        # not needed) is the ordinary case for DLS.
        for attr, label, unit, tol, why in (
            ('concentration_g_per_mL', 'concentration', ' g/mL', rtol,
             "The decay rate is concentration-dependent "
             "(D_app = D0(1 + k_D c + ...)), so this is a concentration series, "
             "not a replicate set."),
        ):
            a, b = getattr(m, attr), getattr(first, attr)
            if (a is None) != (b is None):
                raise ValueError(
                    f"Replicate {i} {label} ({a!r}{unit}) does not match the "
                    f"first ({b!r}{unit}): one is unknown, which is not proven "
                    f"identity. Replicate averaging is only valid across true "
                    f"replicates measured at the same conditions."
                )
            if a is not None and b is not None and not math.isclose(
                    a, b, rel_tol=tol, abs_tol=0.0):
                raise ValueError(
                    f"Replicate {i} {label} ({a:g}{unit}) does not match the "
                    f"first ({b:g}{unit}). {why}"
                )

        for attr, label, why in (
            ('mw_fraction', 'Mw fraction',
             "Fractions are analysed independently, so they are not replicates "
             "of each other."),
            ('analyzer_geometry', 'analyzer geometry',
             "Polarized and depolarized runs are never conflated: "
             "Gamma_VH = Gamma_VV + 6 D_r is a physical difference, not a "
             "repeat."),
        ):
            a, b = getattr(m, attr), getattr(first, attr)
            if a != b:
                raise ValueError(
                    f"Replicate {i} {label} ({a!r}) does not match the first "
                    f"({b!r}). {why}"
                )


def average_replicate_correlograms(
    measurements: List[DLSMeasurement],
    rtol: float = 1e-6,
) -> AveragedCorrelogramResult:
    """Average a set of TRUE replicate correlograms onto their shared lag grid.

    All measurements must be genuine repeats of one sample: identical lag grids
    plus the eight identity fields `check_replicate_identity` enforces (polymer,
    solvent, temperature, angle, wavelength, concentration, Mw fraction, analyzer
    geometry). The averaging is a plain channel-by-channel
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
    # start=2 for the same reason check_replicate_identity uses it: measurements[1] is
    # the SECOND run the user selected. Both guards must name one run by one number.
    for i, m in enumerate(measurements[1:], start=2):
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

    # 2) Matching identity / optics / concentration / fraction.
    check_replicate_identity(measurements, rtol=rtol)

    stack = np.vstack([m.correlogram for m in measurements])   # (n, n_channels)
    mean_g2m1 = stack.mean(axis=0)
    sd_g2m1 = stack.std(axis=0, ddof=1)
    sem_g2m1 = sd_g2m1 / math.sqrt(n)

    return AveragedCorrelogramResult(
        delay_times_s=np.array(grid, dtype=float),
        mean_g2m1=mean_g2m1, sd_g2m1=sd_g2m1, sem_g2m1=sem_g2m1,
        n_replicates=n, q_m_inv=_measurement_q_m(first),
    )
