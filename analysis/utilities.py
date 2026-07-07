"""
analysis/utilities.py
=====================

General-purpose analysis utilities that do not belong specifically to the DLS
or SLS modules. Three broad categories live here:

  1. Data-quality diagnostics  -- the SLS I*sin(theta) optical-quality check.
     (The intensity-trace signal-analysis diagnostics -- statistics, baseline,
     outlier flagging, running average, block variance, count-rate histogram,
     ADF stationarity -- were promoted to analysis/trace_analysis.py in
     Session 59.)

  2. Cross-analysis            -- quantities that link the DLS and SLS pipelines,
     specifically the shape parameter rho = Rg / Rh, and the Rg/Mw/A2 scaling
     power-law plus the provenance-aware result-candidate picker.

  3. Synthetic data            -- generation of artificial DLS correlograms with
     known ground truth, for validating the cumulant / CONTIN analysis routines
     or for testing other software.

Design contract (consistent with the rest of the platform)
----------------------------------------------------------
Every analysis function here is a PURE function: it takes data objects and
parameters and returns a result object. No analysis function draws a plot,
writes a file, or mutates its inputs.

The one exception is export_synthetic_correlogram_csv(), whose explicit purpose
is to write a file. It is clearly separated from the pure generator
(generate_synthetic_correlogram, which only computes and returns a result), so
the generation logic remains pure and testable while the file write is a thin,
obvious helper.

Units
-----
All inputs are assumed to be in the canonical internal units defined in
core/data_models.py (count rate in cps, time in seconds, etc.). The
I*sin(theta) function operates on SLSMeasurement objects.

Change history
--------------
2026-06-13  Initial implementation. (utilities.py v1)
            Trace: compute_trace_statistics, identify_baseline, flag_outliers,
            normalize_trace, running_average, block_variance,
            fit_count_rate_histogram, test_stationarity_adf.
            SLS: i_sin_theta.
            Cross-analysis: compute_rho.
2026-06-13  Added synthetic correlogram generator. (utilities.py v2)
            SyntheticPopulation, SyntheticCorrelogramResult,
            generate_synthetic_correlogram (pure), and
            export_synthetic_correlogram_csv (writes a generic-parser CSV).
            Uses physics.constants for q and Stokes-Einstein so the generated
            data uses the same conventions as the analysis code.
2026-06-26  Promoted the intensity-trace signal-analysis cluster to
            analysis/trace_analysis.py (no behaviour change). Pruned the now
            unused imports (scipy optimize/stats, IntensityTrace). (v3)
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from core.data_models import SLSMeasurement, SampleKey
from analysis import uncertainty as unc
from physics.constants import (
    scattering_vector_q_m,
    stokes_einstein_rh,
    stokes_einstein_diffusion_coefficient,
)


# ===========================================================================
# Result objects
# ===========================================================================
#
# These are defined here, local to the utilities module, rather than in
# core/result_models.py. They are specific to this module's outputs. If later
# modules (dls.py, sls.py) produce result types that need to be shared, those
# shared types will go in result_models.py at that point. Moving these here
# later would be a mechanical change; keeping them local now keeps the module
# self-contained and easy to read.


@dataclass
class ISinThetaCurve:
    """One I*sin(theta) curve for a single SLS measurement."""
    label: str
    angles_deg: np.ndarray
    i_sin_theta: np.ndarray        # absolute or normalised, per `mode`


@dataclass
class ISinThetaResult:
    """A set of I*sin(theta) curves for overlaying on one plot."""
    curves: List[ISinThetaCurve]
    mode: str                      # 'absolute' or 'normalized'


@dataclass
class RhoResult:
    """Shape parameter rho = Rg / Rh and its interpretation."""
    rho: float
    rg_nm: float
    rh_nm: float
    interpretation: str                   # full sentence (guidance, not a boundary)
    shape: str = ''                       # concise architecture label (e.g. 'random coil')
    sample_key: Optional[SampleKey] = None
    keys_matched: Optional[bool] = None   # None if keys not supplied
    rho_se: Optional[float] = None        # propagated from Rg, Rh SEs (both required)


# ===========================================================================
# SLS diagnostics
# ===========================================================================

def i_sin_theta(
    measurements: Sequence[SLSMeasurement],
    mode: str = 'absolute',
) -> ISinThetaResult:
    """Compute I*sin(theta) curves for one or more SLS measurements.

    I*sin(theta) is a classic optical-quality diagnostic: for an ideal,
    isotropic, dust-free scattering volume it is flat across angle. Curvature,
    asymmetry about 90 degrees, or upturns at low/high angle reveal alignment
    problems, stray light, or dust. Comparing a good solvent (e.g. toluene) to
    a harder one (e.g. water) on the same axes is a useful instrument check.

    Parameters
    ----------
    measurements : sequence of SLSMeasurement
        One or more measurements. A single-element sequence is fine.
    mode : str
        'absolute' (default): raw I*sin(theta).
        'normalized': each curve divided by its own mean, so curves from
            different samples or scales can be overlaid for shape comparison.

    Returns
    -------
    ISinThetaResult

    Raises
    ------
    ValueError
        If mode is unrecognised or the sequence is empty.
    """
    if mode not in ('absolute', 'normalized'):
        raise ValueError(
            f"mode must be 'absolute' or 'normalized', got {mode!r}."
        )
    if len(measurements) == 0:
        raise ValueError("measurements is empty; supply at least one.")

    curves: List[ISinThetaCurve] = []
    for m in measurements:
        sin_theta = np.sin(np.radians(m.angles_deg))
        values = m.intensities * sin_theta
        if mode == 'normalized':
            mean_val = values.mean()
            if mean_val == 0:
                raise ValueError(
                    f"Cannot normalise: mean I*sin(theta) is zero for a "
                    f"measurement (label={m.sample_label!r})."
                )
            values = values / mean_val
        # Build a human-readable label
        if m.sample_label:
            label = m.sample_label
        else:
            label = (f"{m.polymer_name} in {m.solvent_name}, "
                     f"c={m.concentration_g_per_mL:g} g/mL")
        curves.append(ISinThetaCurve(
            label=label,
            angles_deg=m.angles_deg.copy(),
            i_sin_theta=values,
        ))

    return ISinThetaResult(curves=curves, mode=mode)


# ===========================================================================
# Cross-analysis:  rho = Rg / Rh
# ===========================================================================

def _interpret_rho(rho: float) -> str:
    """Return a short textual interpretation of a rho value.

    These are well-known reference values for common architectures (Chu 1991).
    They are guides, not hard boundaries -- real systems span ranges and depend
    on solvent quality and polydispersity.
    """
    if not math.isfinite(rho) or rho <= 0:
        return "rho is not a positive finite number; check the inputs."
    if rho < 0.6:
        return ("rho < ~0.6: unusually compact; below the hard-sphere value. "
                "Check inputs or consider a core-shell / dense structure.")
    if rho < 0.9:
        return ("rho ~ 0.775 is the hard-sphere limit; values near here "
                "suggest a compact, near-spherical particle.")
    if rho < 1.2:
        return ("rho ~ 1.0 is typical of hyperbranched polymers, microgels, "
                "or vesicles.")
    if rho < 1.4:
        return ("rho ~ 1.2-1.3 is intermediate, e.g. branched or moderately "
                "swollen structures.")
    if rho < 2.0:
        return ("rho ~ 1.5-1.8 is the classic linear random-coil range "
                "(theta solvent ~1.5, good solvent ~1.78).")
    return ("rho > ~2: extended or rigid-rod-like conformation, or a very "
            "polydisperse / aggregated system.")


def rho_shape_label(rho: float) -> str:
    """A concise architecture label for a rho value (the headline of
    `_interpret_rho`), for a compact 'Shape' column. Guidance, not a hard
    boundary."""
    if not math.isfinite(rho) or rho <= 0:
        return '—'
    if rho < 0.6:
        return 'compact / core-shell'
    if rho < 0.9:
        return 'hard sphere (~0.78)'
    if rho < 1.2:
        return 'microgel / vesicle (~1.0)'
    if rho < 1.4:
        return 'branched / swollen'
    if rho < 2.0:
        return 'random (Gaussian) coil (~1.5)'
    return 'rod / extended / aggregated'


def compute_rho(
    rg_nm: float,
    rh_nm: float,
    rg_sample_key: Optional[SampleKey] = None,
    rh_sample_key: Optional[SampleKey] = None,
    require_match: bool = False,
    rg_se: Optional[float] = None,
    rh_se: Optional[float] = None,
) -> RhoResult:
    """Compute the shape parameter rho = Rg / Rh.

    Rg comes from SLS (radius of gyration, from a Zimm/Berry/Debye
    extrapolation) and Rh from DLS (hydrodynamic radius, from Stokes-Einstein,
    extrapolated to q->0 and c->0). Both should be the infinite-dilution values
    for a physically meaningful rho.

    Parameters
    ----------
    rg_nm : float
        Radius of gyration in nm (from SLS).
    rh_nm : float
        Hydrodynamic radius in nm (from DLS).
    rg_sample_key, rh_sample_key : SampleKey, optional
        If both are supplied, the polymer, solvent, and temperature fields are
        checked for consistency (concentration is ignored because both values
        are extrapolated to c->0). A mismatch raises if require_match is True,
        otherwise warns.
    require_match : bool
        If True and the keys disagree on polymer/solvent/temperature, raise a
        ValueError instead of warning. Default False.

    Returns
    -------
    RhoResult

    Raises
    ------
    ValueError
        If rg_nm or rh_nm is non-positive, or (with require_match=True) the
        sample keys disagree.
    """
    if not (rg_nm > 0):
        raise ValueError(f"rg_nm must be positive, got {rg_nm!r}.")
    if not (rh_nm > 0):
        raise ValueError(f"rh_nm must be positive, got {rh_nm!r}.")

    keys_matched: Optional[bool] = None
    matched_key: Optional[SampleKey] = None
    if rg_sample_key is not None and rh_sample_key is not None:
        keys_matched = (
            rg_sample_key.polymer_name == rh_sample_key.polymer_name
            and rg_sample_key.solvent_name == rh_sample_key.solvent_name
            and rg_sample_key.temperature_K == rh_sample_key.temperature_K
        )
        if keys_matched:
            matched_key = rg_sample_key
        else:
            msg = (
                f"Rg and Rh sample keys disagree on polymer/solvent/"
                f"temperature: Rg from {rg_sample_key}, Rh from {rh_sample_key}. "
                f"rho is only meaningful for the same sample."
            )
            if require_match:
                raise ValueError(msg)
            warnings.warn(msg, UserWarning, stacklevel=2)

    rho = rg_nm / rh_nm
    # Rg (SLS) and Rh (DLS) are independent measurements, so rho's SE is the
    # standard ratio propagation -- only when both inputs carry an SE.
    rho_se = unc.ratio_se(rg_nm, rg_se, rh_nm, rh_se)
    return RhoResult(
        rho=rho,
        rg_nm=rg_nm,
        rh_nm=rh_nm,
        interpretation=_interpret_rho(rho),
        shape=rho_shape_label(rho),
        sample_key=matched_key,
        keys_matched=keys_matched,
        rho_se=rho_se,
    )


# ===========================================================================
# Cross-sample scaling: power-law fits (Rg-Mw, A2-Mw)
# ===========================================================================

@dataclass
class ScalingResult:
    """A power-law fit y = prefactor * x^exponent, from a log-log linear regression.

    For polymer scaling laws across a homologous series: Rg = k Mw^nu (nu ~ 0.33
    sphere, 0.5 theta-coil, ~0.588 good-solvent coil) and A2 = k Mw^-a (a ~ 0.2-0.3
    for flexible coils). The exponent is the slope in log-log; it is the physically
    meaningful quantity and is independent of the (calibration-dependent) prefactor.
    """
    x: np.ndarray                     # the finite, positive (x, y) actually fitted
    y: np.ndarray
    exponent: float                   # slope of log10(y) vs log10(x)
    prefactor: float                  # 10**intercept (y at x = 1)
    r_squared: float
    n_points: int
    fit_valid: bool                   # at least two positive (x, y) points
    exponent_se: Optional[float] = None   # regression SE of the exponent (None if < 2 dof)
    se_estimator: str = 'hc3'             # covariance estimator behind exponent_se


def fit_power_law(x: Sequence[float], y: Sequence[float],
                  estimator: str = 'hc3') -> 'ScalingResult':
    """Fit y = prefactor * x^exponent by least squares in log-log space.

    Only finite, strictly positive (x, y) pairs are used (a log needs them). With
    fewer than two such points the fit is marked invalid and exponent/prefactor are
    NaN -- the caller decides how to present a not-yet-fittable scaling plot.
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya) & (xa > 0) & (ya > 0)
    xv, yv = xa[mask], ya[mask]
    n = int(xv.size)
    if n < 2:
        return ScalingResult(xv, yv, float('nan'), float('nan'), float('nan'),
                             n, False)
    lx, ly = np.log10(xv), np.log10(yv)
    lf = unc.linear_fit(lx, ly, estimator)  # exponent = slope of log10(y) vs log10(x)
    return ScalingResult(xv, yv, float(lf.slope), float(10.0 ** lf.intercept),
                         float(lf.r_squared), n, True,
                         exponent_se=unc.se_or_none(lf.slope_se),
                         se_estimator=estimator)


def interpret_scaling_exponent(exponent: float, quantity: str) -> str:
    """A short textual interpretation of a power-law scaling exponent, mirroring the
    rho interpretation on the cross-sample tab (feedback B8).

    For a SIZE exponent nu (Rg ~ Mw^nu or Rh ~ Mw^nu) the Flory values are the
    references: nu ~ 1/3 compact sphere / collapsed globule (poor solvent),
    nu ~ 1/2 ideal random coil (theta solvent), nu ~ 0.588 (~3/5) swollen coil with
    excluded volume (good solvent), nu ~ 1 rigid rod. For A2 ~ Mw^slope the slope is
    mildly negative for flexible coils and ~0 near the theta point. These are guides,
    not hard boundaries -- they depend on chemistry, the Mw range, and polydispersity,
    and a homologous series (one polymer, one solvent) is assumed.
    """
    if exponent is None or not math.isfinite(exponent):
        return ("exponent not determined — need >=2 samples with both Mw and the "
                "plotted quantity.")
    if quantity == 'a2':
        if exponent > 0.05:
            return (f"slope = {exponent:+.2f}: A2 rising with Mw is unusual for a "
                    "homologous series — check for mixed systems or scatter.")
        if exponent > -0.05:
            return (f"slope ~ 0 ({exponent:+.2f}): A2 nearly Mw-independent — "
                    "near-theta behaviour (A2 -> 0 at the theta point).")
        if exponent >= -0.40:
            return (f"slope = {exponent:.2f}: A2 ~ Mw^{exponent:.2f}, the mild "
                    "negative scaling typical of flexible coils in a good solvent "
                    "(a ~ 0.2-0.3).")
        return (f"slope = {exponent:.2f}: steep A2 decline — check the Mw range / "
                "polydispersity.")
    # size exponent nu (rg or rh)
    nu = exponent
    if nu < 0.25:
        return (f"nu = {nu:.2f} < 1/3: more compact than a solid sphere — check "
                "inputs or a dense / aggregated structure.")
    if nu < 0.42:
        return (f"nu ~ 1/3 ({nu:.2f}): compact sphere or collapsed globule "
                "(poor solvent).")
    if nu < 0.54:
        return (f"nu ~ 1/2 ({nu:.2f}): ideal random coil — theta solvent.")
    if nu < 0.65:
        return (f"nu ~ 3/5 ({nu:.2f}): swollen coil with excluded volume — good "
                "solvent (Flory ~0.588).")
    if nu < 0.90:
        return (f"nu = {nu:.2f}: between coil and rod — semiflexible / extended chain.")
    return (f"nu ~ 1 ({nu:.2f}): rigid-rod-like, fully extended chain.")


# ===========================================================================
# Result-source selection (generic provenance-aware picker)
# ===========================================================================
#
# Wherever one calculation consumes the output of a previous analysis -- Rh and
# Rg for rho = Rg/Rh, Mw for the scaling plots -- the input is chosen from a list
# of candidate sources. Each candidate carries its value AND its provenance, so
# the GUI can show in plain language where a number came from, default to a
# sensible choice, and let the user override (a hand-entered value always wins).
#
# Design principles enforced here:
#   * NEVER a silent choice. The default is a *labelled* default; the GUI shows
#     `label` verbatim next to the value it used.
#   * Apparent vs thermodynamic is tracked per candidate (`is_apparent`) and is
#     never conflated -- a derived quantity (e.g. rho) is apparent if ANY of its
#     inputs is apparent.
#   * The default is deterministic: rank by tier (a physical hierarchy), then by
#     fit quality within a tier, then by original list order. Ties among true
#     replicates resolve to the first in the list (never random); explicit
#     averaging is a separate, user-initiated action, not done here.

@dataclass
class ResultCandidate:
    """One candidate value for a derived quantity, with its provenance.

    Used by the GUI's source pickers (Rh and Rg for rho; Mw for scaling plots).
    `value` is the number; everything else describes where it came from and how
    good it is, so the choice can be made transparently.
    """
    value: float
    label: str                       # plain-language provenance, shown verbatim
    kind: str                        # machine tag, e.g. 'dls_cumulant', 'manual'
    is_apparent: bool                # True = apparent (single q or single c)
    tier: int = 0                    # selection priority; higher = preferred
    quality: Optional[float] = None  # comparison score, higher = better (see below)
    quality_kind: str = ''           # what `quality` means, for display ('r_squared', ...)
    source_id: str = ''              # optional identity for re-selecting (item id / conc)
    value_se: Optional[float] = None # statistical SE of `value` (None if undefined)
    calibrated: Optional[bool] = None  # None = calibration N/A (Rg/Rh); real flag for scale-dependent SLS Mw/A2


def select_default_candidate(
    candidates: Sequence['ResultCandidate'],
) -> Optional['ResultCandidate']:
    """Pick the default candidate by the documented deterministic rule.

    Ranking (best first):
      1. highest `tier`            -- a physical hierarchy (e.g. thermodynamic
                                      c->0 extrapolation beats an apparent single
                                      point); the caller assigns tiers.
      2. highest `quality`         -- fit quality within a tier (R^2, etc.).
                                      A None quality sorts below any real number.
      3. earliest in the input     -- a stable, deterministic tiebreak. For true
                                      replicates this is "the first in the list";
                                      it is never random.

    Candidates whose value is non-finite are ignored. Returns None if no finite
    candidate exists.
    """
    finite = [c for c in candidates if c.value is not None and math.isfinite(c.value)]
    if not finite:
        return None
    # sorted() is stable, so equal (tier, quality) keys keep their original order
    # -> the earliest finite candidate wins the final tiebreak. A missing or
    # non-finite quality (e.g. a degenerate fit) sorts below any real quality.
    neg_inf = float('-inf')

    def _q(c: 'ResultCandidate') -> float:
        return c.quality if (c.quality is not None and math.isfinite(c.quality)) else neg_inf

    return sorted(finite, key=lambda c: (c.tier, _q(c)), reverse=True)[0]


# ===========================================================================
# Synthetic DLS correlogram generation
# ===========================================================================
#
# Generates an artificial g2(tau)-1 correlogram (or g1 / unsubtracted g2) from
# a user-specified set of size populations, with known ground-truth cumulants.
# Useful for validating the cumulant / CONTIN routines (feed in a known
# distribution, confirm the analysis recovers it) and for producing test files
# for other software. The output can be written as a two-column CSV that the
# generic DLS parser reads back.

@dataclass
class SyntheticPopulation:
    """One size population for a synthetic correlogram.

    Attributes
    ----------
    rh_nm : float
        Median hydrodynamic radius of the population, in nm.
    weight : float
        Relative scattering amplitude of this population -- its contribution to
        the field autocorrelation g1 (equivalently, to the scattered intensity).
        This is NOT a number or mass fraction. Larger real particles scatter far
        more strongly than small ones; specify the weights to reflect the
        intensity contribution you want to see in the correlogram.
    spread_cv : float
        Coefficient of variation (sigma/median) of the size distribution within
        this population, modelled as log-normal. 0.0 (default) means a perfectly
        monodisperse population (a single exponential). 0.1 means a 10% spread.
    """
    rh_nm: float
    weight: float
    spread_cv: float = 0.0


@dataclass
class SyntheticCorrelogramResult:
    """A generated correlogram plus the ground truth used to make it.

    The ground-truth cumulant fields (gamma_bar, mu2, pdi) are what an ideal
    cumulant analysis of this correlogram should recover, which makes this
    object directly useful for validating the DLS analysis routines.
    """
    # the generated curve
    delay_times_s: np.ndarray
    signal: np.ndarray                 # the correlation column, in `output_form`
    output_form: str                   # 'g2m1' | 'g2' | 'g1'
    grid: str                          # 'log' | 'linear'

    # physical context used to generate it
    angle_deg: float
    wavelength_nm: float
    solvent_refractive_index: float
    temperature_K: float
    viscosity_Pa_s: float
    q_m_inv: float
    beta: float
    noise_level: float
    seed: Optional[int]
    populations: List[SyntheticPopulation]

    # ground truth (intensity/amplitude-weighted, as a cumulant fit sees it)
    gamma_bar_s_inv: float             # first cumulant: mean decay rate
    mu2_s_inv2: float                  # second cumulant: variance of decay rate
    pdi: float                         # mu2 / gamma_bar^2
    d_eff_m2_s: float                  # effective diffusion coefficient
    rh_eff_nm: float                   # effective (apparent) hydrodynamic radius


def generate_synthetic_correlogram(
    populations: Sequence[SyntheticPopulation],
    angle_deg: float,
    wavelength_nm: float,
    solvent_refractive_index: float,
    temperature_K: float,
    viscosity_Pa_s: float,
    beta: float = 0.8,
    noise_level: float = 0.0,
    delay_min_s: float = 1.0e-7,
    delay_max_s: float = 1.0,
    n_points: int = 200,
    grid: str = 'log',
    output_form: str = 'g2m1',
    n_grid_per_population: int = 80,
    seed: Optional[int] = None,
) -> SyntheticCorrelogramResult:
    """Generate an artificial DLS correlogram from specified size populations.

    The physics mirrors a real measurement exactly, using the same q and
    Stokes-Einstein functions as the analysis code:

        each size  ->  D = k_B T / (6 pi eta Rh)     (Stokes-Einstein)
                   ->  Gamma = D q^2                  (decay rate)
        g1(tau)    =  sum_i A_i exp(-Gamma_i tau)     (field autocorrelation)
        g2(tau)-1  =  beta * |g1(tau)|^2              (Siegert relation)

    Each population is spread into a log-normal distribution of sizes (unless
    spread_cv is 0, giving a single exponential). Amplitudes A_i are the
    scattering-weighted contributions and are normalised so g1(0) = 1.

    Parameters
    ----------
    populations : sequence of SyntheticPopulation
        One or more size populations. Use several for multimodal distributions.
    angle_deg, wavelength_nm, solvent_refractive_index
        Define the scattering vector q (same convention as the analysis code).
    temperature_K, viscosity_Pa_s
        Define the size-to-decay-rate mapping via Stokes-Einstein.
    beta : float
        Coherence factor (intercept of g2-1 at tau=0). In (0, 1]. Default 0.8.
    noise_level : float
        Standard deviation of Gaussian noise added to the output `signal`
        column. 0.0 (default) gives a clean correlogram. A realistic value for
        g2m1 data is around 0.002-0.01. The noise is applied on the scale of
        the chosen output_form.
    delay_min_s, delay_max_s : float
        Range of delay times, in seconds. Default 1e-7 s to 1 s spans the
        typical DLS window.
    n_points : int
        Number of delay-time points (default 200, matching a Brookhaven export).
    grid : str
        'log' (default; quasi-logarithmic, like a real correlator) or 'linear'.
    output_form : str
        'g2m1' (default): the column is g2(tau)-1.
        'g2'  : the column is g2(tau) = 1 + g2(tau)-1 (baseline 1, not subtracted).
        'g1'  : the column is the field autocorrelation g1(tau).
        These map onto the three data forms the generic DLS parser accepts, so
        you can exercise every parser path.
    n_grid_per_population : int
        Number of discrete sizes used to represent each polydisperse population
        (ignored for monodisperse populations). Default 80.
    seed : int, optional
        Seed for the noise RNG, for reproducibility.

    Returns
    -------
    SyntheticCorrelogramResult

    Raises
    ------
    ValueError
        On invalid inputs (empty populations, non-positive sizes/weights,
        beta outside (0, 1], bad grid/output_form, delay_min >= delay_max,
        n_points < 2).
    """
    pops = list(populations)
    if len(pops) == 0:
        raise ValueError("populations is empty; supply at least one.")
    for p in pops:
        if not (p.rh_nm > 0):
            raise ValueError(f"population rh_nm must be positive, got {p.rh_nm!r}.")
        if not (p.weight > 0):
            raise ValueError(f"population weight must be positive, got {p.weight!r}.")
        if p.spread_cv < 0:
            raise ValueError(f"population spread_cv must be >= 0, got {p.spread_cv!r}.")
    if not (0 < beta <= 1.0):
        raise ValueError(f"beta must be in (0, 1], got {beta!r}.")
    if noise_level < 0:
        raise ValueError(f"noise_level must be >= 0, got {noise_level!r}.")
    if not (delay_min_s > 0 and delay_max_s > delay_min_s):
        raise ValueError(
            f"require 0 < delay_min_s < delay_max_s, got "
            f"{delay_min_s!r} and {delay_max_s!r}."
        )
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points!r}.")
    if grid not in ('log', 'linear'):
        raise ValueError(f"grid must be 'log' or 'linear', got {grid!r}.")
    if output_form not in ('g2m1', 'g2', 'g1'):
        raise ValueError(
            f"output_form must be 'g2m1', 'g2', or 'g1', got {output_form!r}."
        )

    # Scattering vector (m^-1), using the same physics as the analysis code.
    q = scattering_vector_q_m(angle_deg, wavelength_nm, solvent_refractive_index)

    # Delay-time grid.
    if grid == 'log':
        tau = np.geomspace(delay_min_s, delay_max_s, n_points)
    else:
        tau = np.linspace(delay_min_s, delay_max_s, n_points)

    # Build the discrete decay-rate spectrum: accumulate (Gamma, amplitude).
    gammas: List[float] = []
    amps: List[float] = []
    for p in pops:
        if p.spread_cv <= 0:
            radii = np.array([p.rh_nm], dtype=float)
            a = np.array([p.weight], dtype=float)
        else:
            # Lognormal population: convert the coefficient of variation to the
            # log-space width (sigma_ln; exact for a lognormal), treat rh_nm as the
            # median so mu_ln = ln(rh_nm), and sample the lognormal pdf on a
            # geometric grid spanning +/-3 sigma_ln (covers ~99.7% of the mass).
            sigma_ln = math.sqrt(math.log(1.0 + p.spread_cv ** 2))
            mu_ln = math.log(p.rh_nm)
            lo = p.rh_nm * math.exp(-3.0 * sigma_ln)
            hi = p.rh_nm * math.exp(3.0 * sigma_ln)
            radii = np.geomspace(lo, hi, n_grid_per_population)
            pdf = (1.0 / (radii * sigma_ln * math.sqrt(2.0 * math.pi))) * \
                  np.exp(-(np.log(radii) - mu_ln) ** 2 / (2.0 * sigma_ln ** 2))
            a = pdf / pdf.sum() * p.weight   # normalise to the population weight
        for r, amp in zip(radii, a, strict=True):
            d = stokes_einstein_diffusion_coefficient(r * 1e-9, temperature_K, viscosity_Pa_s)
            gammas.append(d * q ** 2)
            amps.append(float(amp))

    gammas = np.array(gammas, dtype=float)
    amps = np.array(amps, dtype=float)
    amps = amps / amps.sum()           # normalise so g1(0) = 1

    # Field autocorrelation and Siegert relation.
    # exp_matrix[i, j] = exp(-gamma_j * tau_i)
    exp_matrix = np.exp(-np.outer(tau, gammas))
    g1 = exp_matrix @ amps
    g2m1 = beta * g1 ** 2

    if output_form == 'g2m1':
        signal = g2m1.copy()
    elif output_form == 'g2':
        signal = 1.0 + g2m1
    else:  # 'g1'
        signal = g1.copy()

    if noise_level > 0:
        rng = np.random.default_rng(seed)
        signal = signal + rng.normal(0.0, noise_level, size=signal.shape)

    # Ground-truth cumulants (amplitude-weighted moments of the decay rate).
    gamma_bar = float(np.sum(amps * gammas))
    mu2 = float(np.sum(amps * (gammas - gamma_bar) ** 2))
    pdi = mu2 / gamma_bar ** 2 if gamma_bar != 0 else float('nan')
    d_eff = gamma_bar / q ** 2
    rh_eff_nm = stokes_einstein_rh(d_eff, temperature_K, viscosity_Pa_s) * 1e9

    return SyntheticCorrelogramResult(
        delay_times_s=tau,
        signal=signal,
        output_form=output_form,
        grid=grid,
        angle_deg=angle_deg,
        wavelength_nm=wavelength_nm,
        solvent_refractive_index=solvent_refractive_index,
        temperature_K=temperature_K,
        viscosity_Pa_s=viscosity_Pa_s,
        q_m_inv=q,
        beta=beta,
        noise_level=noise_level,
        seed=seed,
        populations=pops,
        gamma_bar_s_inv=gamma_bar,
        mu2_s_inv2=mu2,
        pdi=pdi,
        d_eff_m2_s=d_eff,
        rh_eff_nm=rh_eff_nm,
    )


# Conversion factors: seconds -> output unit (multiply by these)
_DELAY_OUTPUT_FACTORS = {'s': 1.0, 'ms': 1.0e3, 'us': 1.0e6, 'ns': 1.0e9}


def export_synthetic_correlogram_csv(
    result: SyntheticCorrelogramResult,
    file_path: str,
    delay_unit: str = 's',
    delimiter: str = ',',
) -> str:
    """Write a synthetic correlogram as a two-column CSV for the generic parser.

    The output has no header rows and two columns -- delay time and the
    correlation value -- exactly matching the generic DLS parser's contract.
    Load it back with GenericDLSParser, setting delay_time_unit to match
    `delay_unit` and data_form to match the result's output_form (and beta or
    baseline_B if the form is 'g1' or 'g2').

    This is the one function in utilities.py that performs file I/O.

    Parameters
    ----------
    result : SyntheticCorrelogramResult
        The output of generate_synthetic_correlogram().
    file_path : str
        Destination path for the CSV.
    delay_unit : str
        Unit to write the delay times in: 's' (default), 'ms', 'us', or 'ns'.
        When loading, set the generic parser's delay_time_unit to the same value.
    delimiter : str
        Column delimiter, ',' (default) or '\\t'.

    Returns
    -------
    str
        The file path written.

    Raises
    ------
    ValueError
        If delay_unit is unrecognised.
    """
    if delay_unit not in _DELAY_OUTPUT_FACTORS:
        raise ValueError(
            f"delay_unit must be one of {sorted(_DELAY_OUTPUT_FACTORS)}, "
            f"got {delay_unit!r}."
        )
    factor = _DELAY_OUTPUT_FACTORS[delay_unit]
    delays_out = result.delay_times_s * factor

    with open(file_path, 'w', newline='') as fh:
        for t, v in zip(delays_out, result.signal, strict=True):
            fh.write(f"{t:.10e}{delimiter}{v:.10e}\n")

    return file_path
