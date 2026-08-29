"""
analysis/utilities.py
=====================

General-purpose analysis utilities that do not belong specifically to the DLS
or SLS modules. Three broad categories live here:

  1. Data-quality diagnostics  -- the SLS I*sin(theta) optical-quality check.
     (The intensity-trace signal-analysis diagnostics -- statistics, baseline,
     outlier flagging, running average, block variance, count-rate histogram,
     ADF stationarity -- were promoted to analysis/trace_analysis.py.)

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
            analysis/trace_analysis.py (no behavior change). Pruned the now
            unused imports (scipy optimize/stats, IntensityTrace). (v3)
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import brentq

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
# Local to this module rather than in core/result_models.py: they are specific to its
# outputs, and keeping them here keeps the module self-contained. A result type that later
# needs sharing moves to result_models.py at that point -- a mechanical change.


@dataclass
class ISinThetaCurve:
    """One I*sin(theta) curve for a single SLS measurement."""
    label: str
    angles_deg: np.ndarray
    i_sin_theta: np.ndarray        # absolute or normalized, per `mode`


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
    # passive, glyph-less notices for the GUI ⓘ tier (e.g. Rg/Rh keys disagree) --
    # the same text the module also emits via warnings.warn (keep-both).
    notes: tuple = ()


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
        If mode is unrecognized or the sequence is empty.
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
                    f"Cannot normalize: mean I*sin(theta) is zero for a "
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

#: Appended ONLY to bands whose reference value is a Kirkwood-Riseman result.
#: Burchard 1999's 10-14% figure is about KR's treatment of CHAIN hydrodynamic
#: interaction, so it does not describe every row of Table 4: the hard sphere is the
#: exact geometric ratio (Rg = sqrt(3/5)*R, Rh = R -- no HI approximation, and the one
#: row with no reference cited) and the microgel range is empirical, so applying it
#: there would double-count a bias on an already-measured number and contradict our
#: own generator's sphere ground truth. Which bands carry it is decided in `_rho_band`,
#: beside the text, so there is no second ladder to fall out of step.
_RHO_MEASURED_BIAS_NOTE = (
    " ⓘ These are TABULATED (Kirkwood-Riseman) reference values; measured ρ typically "
    "runs ~10-14% below them (Burchard 1999), so a real sample can read one band low. "
    "A narrow θ-solvent coil, tabulated at 1.50, often measures 1.29-1.35 -- the "
    "branched/star range.")


def _rho_band(rho: float) -> Tuple[str, bool]:
    """`(band verdict, is the band's reference value a Kirkwood-Riseman result?)`.

    For a rho already known positive and finite. Returns BOTH in one pass on purpose:
    the note's scope is a property of each band, and a separate predicate over the same
    cuts is the second boundary rule 12 exists to warn about. A band added here must
    state its own flag -- it cannot inherit the wrong one by omission."""
    if rho < 0.3:
        return ("ρ < ~0.3: more compact than a microgel -- check the inputs, "
                "or a dense core-shell structure."), False       # below every reference
    if rho < 0.7:
        return ("ρ ~ 0.3-0.6 is the microgel / internally cross-linked sphere "
                "range; the hard-sphere value 0.78 sits just above."), False  # empirical
    if rho < 0.9:                                    # sqrt(3/5), exact geometric ratio
        # Its OWN reference is geometric, so the KR note would be false here -- but the
        # band ABOVE is dendrimer, whose KR value 0.977 measures 0.84-0.88 and lands in
        # this band. The alternative is named inline instead, which is true of both.
        return ("ρ ~ 0.78 is the hard-sphere value: a compact, near-spherical "
                "particle. Dendrimers / soft spheres sit just above (~0.98) -- and the same "
                "10-14% offset puts a dendrimer near 0.84-0.88, so one can land in "
                "this band."), False
    if rho < 1.05:
        return ("ρ ~ 1.0: a dendrimer or soft-sphere / compact object "
                "(dendrimers are near ~0.98)."), True
    if rho < 1.42:
        return ("ρ ~ 1.1-1.4 points to a branched or star architecture -- "
                "regular stars run ~1.08 (many-arm) to 1.33 (4-arm), "
                "hyperbranched ~1.23, cyclic ~1.25 -- all below the linear coil; "
                "ρ falls as the arm number / branching rises."), True
    if rho < 1.9:
        return ("ρ ~ 1.5-1.8 is the linear random-coil range (θ solvent "
                "~1.50, good solvent ~1.78); polydispersity and random branching "
                "push it toward ~1.7."), True
    if rho < 2.2:
        return ("ρ ~ 1.9-2.1: a broad or swollen linear coil (a polydisperse "
                "good-solvent coil reaches ~2.05) -- the upper coil range, "
                "bordering rod-like."), True
    return ("ρ > ~2.2: extended or rigid-rod-like (ρ grows as ln N), "
            "or a very polydisperse / aggregated system."), False   # open-ended, no
    # tabulated reference to be biased away from


def _rho_band_text(rho: float) -> str:
    """The band verdict alone, without the measured-bias note."""
    return _rho_band(rho)[0]


def _interpret_rho(rho: float) -> str:
    """Return a short textual interpretation of a rho = Rg/Rh value.

    Reference values are architecture-resolved from Burchard (1999, Table 4;
    theta-solvent unless noted), with the classical coil/sphere limits (Chu
    1991). rho is a *qualitative* architecture fingerprint, not a hard
    classifier: branched and star objects sit below the linear-coil value, but
    their ranges overlap the sphere-to-coil span, so rho narrows the
    possibilities rather than naming one. It also shifts with solvent quality
    and polydispersity (a broad distribution *raises* rho, e.g. a theta coil
    1.50 -> 1.73), and the Kirkwood-Riseman theory behind these numbers
    underestimates the hydrodynamic interaction, so measured rho tends to run
    ~10-14% below the tabulated values (Burchard 1999) -- which is why that last
    caveat is APPENDED (`_RHO_MEASURED_BIAS_NOTE`) to the bands whose reference IS
    a KR result, rather than left here where only a maintainer would read it. The
    hard-sphere and microgel bands do not carry it; see `_rho_band`.
    """
    if not math.isfinite(rho) or rho <= 0:
        return "ρ is not a positive finite number; check the inputs."
    text, kr_derived = _rho_band(rho)
    return (text + _RHO_MEASURED_BIAS_NOTE) if kr_derived else text


def rho_shape_label(rho: float) -> str:
    """A concise architecture label for a rho value (the headline of
    `_interpret_rho`), for a compact 'Shape' column. Qualitative guidance, not a
    hard boundary -- see `_interpret_rho` for the reference values and caveats.

    Deliberately does NOT carry `_RHO_MEASURED_BIAS_NOTE`: this is a one-line table
    cell and a headline, and the caveat belongs with the detail sentence the two are
    always shown together with (`gui/cross_module` renders the label as the headline
    ABOVE the interpretation). Adding it here would put a paragraph in a grid column."""
    if not math.isfinite(rho) or rho <= 0:
        return '—'
    if rho < 0.3:
        return 'compact / core-shell'
    if rho < 0.7:
        return 'microgel (~0.3-0.6)'
    if rho < 0.9:
        return 'hard sphere (~0.78)'
    if rho < 1.05:
        return 'dendrimer / soft sphere (~1.0)'
    if rho < 1.42:
        return 'branched / star (<1.4)'
    if rho < 1.9:
        return 'random (Gaussian) coil (~1.5-1.8)'
    if rho < 2.2:
        return 'swollen / polydisperse coil'
    return 'rod / extended / aggregated'


# Temperature match tolerance for rho's sample-key check: guards float round-off
# between the DLS and SLS paths without merging distinct set points (>= 0.1 K apart
# under the grouping key -- round_temperature_for_identity's bucket). Redundant but
# harmless on already-bucketed SampleKey fields; kept for library callers passing
# hand-built keys.
_RHO_TEMPERATURE_TOL_K = 1.0e-6


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
        otherwise warns (combination doctrine rule 2, core/data_models.py).
        The app's own caller passes neither: it reads both radii from one
        stored result slot, so the match holds by construction (see
        `compute_sample_rho`). The check stays for library callers assembling
        Rg and Rh from separate sources.
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
    # isfinite as well as > 0. Both values can be HAND-ENTERED (Cross-Sample's manual
    # Rg/Rh fields parse with float(text), which accepts 'inf'/'nan'/'1e400'), and a bare
    # `> 0` admits +inf -- which made rho = Rg/inf report a clean 0.0 with no note and no
    # flag (2026-07-31 spec-01 re-review).
    if not (math.isfinite(rg_nm) and rg_nm > 0):
        raise ValueError(f"rg_nm must be a positive finite number, got {rg_nm!r}.")
    if not (math.isfinite(rh_nm) and rh_nm > 0):
        raise ValueError(f"rh_nm must be a positive finite number, got {rh_nm!r}.")

    keys_matched: Optional[bool] = None
    matched_key: Optional[SampleKey] = None
    notes: list = []       # passive ⓘ notices surfaced on the result (keep-both)
    if rg_sample_key is not None and rh_sample_key is not None:
        keys_matched = (
            rg_sample_key.polymer_name == rh_sample_key.polymer_name
            and rg_sample_key.solvent_name == rh_sample_key.solvent_name
            # Tolerance, not exact ==, on temperature: the DLS and SLS paths can
            # carry the same set-point with sub-µK float drift, and an exact compare
            # would spuriously refuse a genuine match. Well below any real set-point
            # separation, so 35 vs 50 C still differ.
            and math.isclose(rg_sample_key.temperature_K, rh_sample_key.temperature_K,
                             rel_tol=0.0, abs_tol=_RHO_TEMPERATURE_TOL_K)
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
            notes.append(msg)   # single source -> both the note and the warning
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
        notes=tuple(notes),
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
    prefactor: float                  # 10**intercept (y at x = 1); NaN when it leaves float
    #                                   range (the fit stays valid -- see fit_power_law)
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

    The prefactor alone can leave float range while the fit is perfectly good: the
    intercept is log10(y) - slope*log10(x), so two points close in x and far apart in
    y (a near-equal-Mw pair with a 10x A2 or Rh spread) push it past +-308. It is then
    NaN -- withheld, never raised, never a subnormal -- and `fit_valid`, the exponent
    (the physically meaningful quantity), R^2 and n_points are unaffected.
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
    # Two-tailed (CLAUDE.md rule 12): a Python-float `**` RAISES past +308 where NumPy
    # saturates to inf, and past -308 it underflows to a subnormal that would plot as a
    # fitted line. So saturate, then apply the engine's usable-number predicate --
    # `power_or_nan` at n = 1 withholds inf, 0.0 and the subnormal window as NaN.
    with np.errstate(over='ignore'):
        prefactor = unc.power_or_nan(float(np.power(10.0, lf.intercept)), 1)
    return ScalingResult(xv, yv, float(lf.slope), prefactor,
                         float(lf.r_squared), n, True,
                         exponent_se=unc.se_or_none(lf.slope_se),
                         se_estimator=estimator)


def interpret_scaling_exponent(exponent: float, quantity: str) -> str:
    """A short textual interpretation of a power-law scaling exponent, mirroring the
    rho interpretation on the cross-sample tab.

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
            return (f"slope = {exponent:+.2f}: A₂ rising with Mw is unusual for a "
                    "homologous series — check for mixed systems or scatter.")
        if exponent > -0.05:
            return (f"slope ~ 0 ({exponent:+.2f}): A₂ nearly Mw-independent — "
                    "near-θ behaviour (A₂ → 0 at the θ point).")
        if exponent >= -0.40:
            nu_eq = (exponent + 2.0) / 3.0
            return (f"slope = {exponent:.2f}: A₂ ~ Mw^{exponent:.2f}, the mild "
                    "negative scaling of flexible coils in a good solvent "
                    f"(a ~ 0.2-0.3). If the good-solvent relation A₂ ~ Mw^(3ν−2) "
                    f"holds, ν ≈ {nu_eq:.2f}.")
        return (f"slope = {exponent:.2f}: steep A₂ decline — check the Mw range / "
                "polydispersity.")
    # size exponent ν (rg or rh)
    nu = exponent
    if nu < 0.25:
        return (f"ν = {nu:.2f} < 1/3: more compact than a solid sphere — check "
                "inputs or a dense / aggregated structure.")
    if nu < 0.42:
        return (f"ν ~ 1/3 ({nu:.2f}): compact sphere or collapsed globule "
                "(poor solvent).")
    if nu < 0.54:
        return (f"ν ~ 1/2 ({nu:.2f}): ideal random coil — θ solvent.")
    if nu < 0.65:
        return (f"ν ~ 3/5 ({nu:.2f}): swollen coil with excluded volume — good "
                "solvent (~0.588).")
    if nu < 0.90:
        return (f"ν = {nu:.2f}: between coil and rod — semiflexible / extended chain.")
    return (f"ν ~ 1 ({nu:.2f}): rigid-rod-like, fully extended chain.")


# ===========================================================================
# Branching contraction factor  g = <Rg^2>_branched / <Rg^2>_linear  (architecture)
# ===========================================================================

#
# For flexible chains of the SAME chemistry and molar mass, a branched molecule is more
# compact than its linear analog, so g = <Rg^2>_br / <Rg^2>_lin < 1 (Zimm & Stockmayer
# 1949). For a REGULAR f-arm star, g = (3f - 2)/f^2, so a measured g inverts to an
# effective arm number. INTERPRETIVE guidance (like the rho labels), never a hard flag: g
# itself is model-free, a ratio of radii; only the arm number assumes a uniform-arm star.


#: The dead band both contraction factors are read against, expressed ONCE and in the
#: unit it is a tolerance on: a RADIUS excess of 5 %. It exists to keep ordinary
#: measurement scatter from being read as architecture, so its natural unit is the
#: measured radius, not whatever power of it a given factor happens to be.
#: Its deficit twin is the INVERSE, not `2 - tol`: a 5 % excess is x1.05 and a 5 %
#: deficit is /1.05 = 0.9524. The two agree to 0.25 % at 1.05 and diverge linearly
#: (at tol = 1.20, `2 - tol` = 0.80 against the correct 0.833), so the additive form
#: would quietly stop being one tolerance the moment the band was retuned.
_CONTRACTION_RADIUS_TOL = 1.05
_CONTRACTION_RADIUS_TOL_LOW = 1.0 / _CONTRACTION_RADIUS_TOL


def _radius_excess(factor: float, squared: bool) -> float:
    """`factor` re-expressed as the radius ratio it implies.

    g = (Rg_br/Rg_lin)**2 is a SQUARED radius ratio and h = Rh_br/Rh_lin a plain one,
    so a single numeric cut applied to both is not one tolerance -- it is 5 % on a
    radius for h and 2.5 % for g, which is backwards besides (Rg is the noisier
    measurement). Taking the root here means both are compared in radius units against
    `_CONTRACTION_RADIUS_TOL`, so there is one threshold to reason about rather than a
    coincidence between two.

    EVERY comparison that mixes the two factors goes through this, including the g-vs-h
    ordering. An earlier version left the ordering on the raw pair, arguing that each
    factor is monotonic in its radius ratio so the verdict could not change: that is
    false for a CROSS-comparison, and the disagreement region is exactly h**2 < g < h.
    At g = 0.85, h = 0.90 the raw test says g < h ("as expected for a star") while
    sqrt(g) = 0.9220 > h -- the hydrodynamic contraction is the stronger one, and every
    pair in that band is NOT star-like (a real regular star at h = 0.90 has g = 0.645).
    Nor is there a source convention pulling the other way: Burchard 1980 compares each
    factor to unity, which is invariant under the root, and states no g-vs-h ordering."""
    return math.sqrt(factor) if squared else factor


@dataclass
class HydrodynamicResult:
    """Hydrodynamic branching contraction factor h and its regular-star reading.

    h = Rh_branched / Rh_linear at matched molar mass (Burchard, Schmidt &
    Stockmayer 1980). `effective_arms` inverts the regular f-arm-star relation
    h = sqrt(f)/[(2-f)+sqrt(2)(f-1)] -- a "regular-star equivalent", NOT g's
    (3f-2)/f^2. Uncertainty is honest: a ± appears ONLY from replicate branched
    measurements (their spread, via `replicate_mean_se`); a single measurement
    carries no ± (invariant 8). That ± is the branched replicate spread ONLY -- it
    treats the fitted linear reference law as exact and does NOT propagate the
    reference-law fit uncertainty. h is a ratio of radii -> scale-independent
    (survives an uncalibrated run when the Mw scale is consistent across the
    tagged samples). The replicate SEM of h is a LINEAR pass-through of the
    already-validated Rh replicate SEM (h_i = Rh_i / Rh_lin, Rh_lin constant), so
    it inherits that validation -- no separate Monte-Carlo gate.
    """
    h: float
    effective_arms: float                 # f_eff from the star h(f); nan if h not in (0,1]
    interpretation: str
    h_se: Optional[float] = None          # replicate SEM of h; None for a single measurement
    effective_arms_se: Optional[float] = None  # f_eff inherits h's uncertainty (shown on h)
    se_source: Optional[str] = None       # 'replicate' when a ± is reported, else None
    n_replicates: int = 1
    mw_branched: Optional[float] = None
    mw_linear_range: Optional[tuple] = None    # (min, max) Mw of the linear reference series
    mw_in_range: Optional[bool] = None    # branched Mw within the linear span? None if unknown
    linear_ref_note: str = ''             # provenance of the linear reference (shown verbatim)


def _star_arms_from_h(h: float) -> float:
    """Invert the regular f-arm-star hydrodynamic relation
    h = sqrt(f)/[(2 - f) + sqrt(2)(f - 1)] = sqrt(f)/[(sqrt2 - 1)(f + sqrt2)] for f.

    With a = h*(sqrt2 - 1) and u = sqrt(f), the relation is a*u^2 - u + a*sqrt2 = 0,
    whose physical (f >= 2) root is u = [1 + sqrt(1 - 4*sqrt2*a^2)] / (2a), f = u^2.
    Real and physical only for 0 < h <= 1 (a branched Rh >= its linear analog is not
    a star); returns NaN otherwise. At h = 1 this gives f = 2 (a linear/two-arm
    chain). Note h(f) > g(f) for f > 2 -- the hydrodynamic contraction is weaker."""
    if not (math.isfinite(h) and 0.0 < h <= 1.0):
        return float('nan')
    a = h * (math.sqrt(2.0) - 1.0)
    disc = 1.0 - 4.0 * math.sqrt(2.0) * a * a
    if disc < 0.0:                        # guard rounding at the h = 1 boundary
        disc = 0.0
    u = (1.0 + math.sqrt(disc)) / (2.0 * a)
    return u * u


@dataclass
class ArchitectureResult:
    """Scattering branching contraction factor g and its regular-star reading.

    g = Rg_branched^2 / Rg_linear^2 at matched molar mass (Zimm & Stockmayer
    1949). `effective_arms` inverts the regular f-arm-star relation g=(3f-2)/f^2
    -- a "regular-star equivalent", not a literal count -- and is reported for
    every model (the star reading is always available). When a RANDOM-branching
    model is selected, `branching_index` (n = branch units/molecule) additionally
    inverts the Zimm & Stockmayer FRACTION relation for that functionality (Eqs.
    47a/47b = Burchard 1999 Eqs. 65/66); `branching_model` names the assumption.
    The same g gives a DIFFERENT n under a different model -- model-dependence is
    the headline caveat, surfaced in `interpretation`. Uncertainty is honest:
    a ± appears ONLY from replicate branched measurements (their spread, via
    `replicate_mean_se`); a single measurement carries no ± (invariant 8). That ±
    is the branched replicate spread ONLY -- it treats the fitted linear reference
    law as exact and does NOT propagate the reference-law fit uncertainty. Both
    `effective_arms` and `branching_index` are deterministic transforms of the
    REPORTED g, so their uncertainty IS g's (shown on g); neither carries a
    separate ± (`branching_index_se` is always None). g is a ratio of radii ->
    scale-independent (survives an uncalibrated run when the Mw scale is
    consistent across the tagged samples).
    """
    g: float
    effective_arms: float                 # f_eff from g=(3f-2)/f^2; nan if g not in (0,1]
    interpretation: str
    g_se: Optional[float] = None          # replicate SEM of g; None for a single measurement
    effective_arms_se: Optional[float] = None  # replicate SEM of f_eff; None if single/invalid
    se_source: Optional[str] = None       # 'replicate' when a ± is reported, else None
    n_replicates: int = 1
    mw_branched: Optional[float] = None
    mw_linear_range: Optional[tuple] = None    # (min, max) Mw of the linear reference series
    mw_in_range: Optional[bool] = None    # branched Mw within the linear span? None if unknown
    linear_ref_note: str = ''             # provenance of the linear reference (shown verbatim)
    branching_model: str = 'regular_star'      # 'regular_star' | 'random_tri' | 'random_tetra'
    branching_index: Optional[float] = None    # n (branch units/molecule) for a random model; None for the regular star OR g out of (0,1]
    branching_index_se: Optional[float] = None  # ALWAYS None: n is a deterministic transform of the reported g (its ± is shown on g)


def _star_arms_from_g(g: float) -> float:
    """Invert the regular f-arm-star relation g = (3f - 2)/f^2 for f:
    f = [3 + sqrt(9 - 8g)] / (2g). Real and physical (f >= 2) only for
    0 < g <= 1; returns NaN otherwise (a branched Rg >= its linear analog is not
    a star). At g = 1 this gives f = 2, i.e. a linear (two-arm) chain."""
    if not (math.isfinite(g) and 0.0 < g <= 1.0):
        return float('nan')
    return (3.0 + math.sqrt(9.0 - 8.0 * g)) / (2.0 * g)


# Random-branching models beyond the regular star: label -> branch-unit functionality f.
# Only f = 3 and f = 4 have Zimm & Stockmayer's tabulated closed-form fraction g(n).
_RANDOM_BRANCHING_FUNCTIONALITY = {'random_tri': 3, 'random_tetra': 4}
_BRANCHING_MODELS = ('regular_star',) + tuple(_RANDOM_BRANCHING_FUNCTIONALITY)


def _g_fraction(n: float, functionality: int) -> float:
    """Forward scattering contraction factor g for a randomly-branched FRACTION (one
    molar mass, branch units distributed at random) carrying n f-functional branch
    units per molecule -- Zimm & Stockmayer 1949 Eqs. (47a)/(47b), identical to
    Burchard 1999 Eqs. (65)/(66):

        f = 3:  g = [ (1 + n/7)^(1/2) + 4n/(9 pi) ]^(-1/2)
        f = 4:  g = [ (1 + n/6)^(1/2) + 4n/(3 pi) ]^(-1/2)

    Monotonically decreasing from g(0) = 1 toward 0 as n grows. Defined only for
    f = 3 or 4 (the tabulated cases). See Theory-and-Equations-Guide §2.4.1."""
    if functionality == 3:
        return ((1.0 + n / 7.0) ** 0.5 + 4.0 * n / (9.0 * math.pi)) ** -0.5
    if functionality == 4:
        return ((1.0 + n / 6.0) ** 0.5 + 4.0 * n / (3.0 * math.pi)) ** -0.5
    raise ValueError("random-branching fraction g is defined only for functionality "
                     f"3 or 4, got {functionality!r}.")


def random_branching_index(g: float, *, functionality: int) -> float:
    """Invert the randomly-branched FRACTION relation g -> n, the average number of
    f-functional branch units per molecule (Zimm & Stockmayer 1949 Eqs. 47a/47b;
    f = 3 or 4).

    g(n) is monotone-decreasing from g(0) = 1 toward 0, so a unique n >= 0 exists for
    every 0 < g <= 1; it is solved by a bracketing root-find (SciPy `brentq`, the
    canonical/authoritative bracketing method -- Brent 1973). Returns 0.0 at g = 1 (no
    contraction -> no branching) and NaN for g <= 0, g > 1, or non-finite g, mirroring
    `_star_arms_from_g`'s domain guard (a branched Rg >= its linear analog is not a
    contracted fraction). Model-dependent: the SAME g gives a DIFFERENT n for a
    different functionality -- the caller must state the assumed model.

    Parameters
    ----------
    g : float
        The measured scattering contraction factor (Rg_br^2 / Rg_lin^2 at matched Mw).
    functionality : int
        The branch-unit functionality: 3 (trifunctional) or 4 (tetrafunctional).

    Returns
    -------
    float
        n >= 0 (branch units per molecule), or NaN when g is outside (0, 1].

    Raises
    ------
    ValueError
        If `functionality` is not 3 or 4.
    """
    if functionality not in _RANDOM_BRANCHING_FUNCTIONALITY.values():
        raise ValueError("random_branching_index supports functionality 3 or 4, "
                         f"got {functionality!r}.")
    if not (math.isfinite(g) and 0.0 < g <= 1.0):
        return float('nan')
    if g >= 1.0:                                # g == 1 exactly: no contraction, n = 0
        return 0.0
    # Bracket the root: g(0) = 1 > g, so expand the upper end until g(n_hi) < g.
    n_hi = 1.0
    for _ in range(64):
        if _g_fraction(n_hi, functionality) < g:
            break
        n_hi *= 2.0
    else:                                       # defensive: unreachable for g in (0, 1)
        return float('nan')
    return float(brentq(lambda n: _g_fraction(n, functionality) - g, 0.0, n_hi))


def _interpret_g(g: float, f_eff: float, mw_in_range: Optional[bool],
                 *, model: str = 'regular_star',
                 branching_index: Optional[float] = None) -> str:
    """A short, guidance-tier reading of a branching contraction factor, phrased for
    the selected `model` (regular star -> f_eff; random tri/tetra -> the branching
    index n). The reading is model-consistent in EVERY g band -- it never calls a
    chain "linear" while quoting a nonzero n, nor omits the model it was asked for --
    and the model-dependence of the random reading is stated explicitly."""
    if not math.isfinite(g) or g <= 0:
        return "g is not a positive finite number; check the Rg inputs."
    range_note = ''
    if mw_in_range is False:
        range_note = (" ⓘ the branched Mw is out of the linear-reference range, "
                      "so g relies on extrapolating the Rg-Mw law.")
    is_random = model in _RANDOM_BRANCHING_FUNCTIONALITY
    fdesc = 'tri-functional' if model == 'random_tri' else 'tetra-functional'
    n_finite = branching_index is not None and math.isfinite(branching_index)
    n_txt = f"{branching_index:.1f}" if n_finite else "n/a"
    g_r = _radius_excess(g, squared=True)
    if g_r > _CONTRACTION_RADIUS_TOL:
        tail = ("Not branched (no branching index)." if is_random
                else "Not a star (no arm number).")
        return (f"g = {g:.3f} > 1: the branched Rg exceeds its linear reference at "
                "this Mw. This can reflect an input problem (check the linear "
                "reference / Mw, or measurement noise) or the expected signature of "
                f"a polydisperse branched architecture. {tail}" + range_note)
    if g_r >= _CONTRACTION_RADIUS_TOL_LOW:
        # Near-unity: essentially linear. Under a random model n is tiny (< ~1 branch
        # unit) when finite, or undefined for g just above 1 -- name it so the text
        # never contradicts a stored n.
        extra = (f" Assuming random {fdesc} branching, n ≈ {n_txt} branch units per "
                 "molecule (essentially linear)." if (is_random and n_finite) else '')
        return (f"g = {g:.3f} ~ 1: little or no contraction -- consistent with a "
                f"linear (unbranched) chain.{extra}" + range_note)
    arms = f"{f_eff:.1f}" if math.isfinite(f_eff) else "n/a"
    if is_random:
        return (f"g = {g:.3f} (< 1: more compact than a linear chain at the same Mw). "
                f"Assuming random {fdesc} branching: n ≈ {n_txt} branch units per "
                "molecule -- model-dependent (a different model gives a different n; "
                f"as a regular star, f_eff ≈ {arms})." + range_note)
    return (f"g = {g:.3f} (< 1: more compact than a linear chain at the same Mw). "
            f"Regular-star equivalent f_eff ~ {arms} arms -- guidance, not a literal "
            "count (assumes uniform arms; biased for polydisperse arms and high f)."
            + range_note)


def branching_contraction_factor(
    rg_branched,
    rg_linear: float,
    *,
    mw_branched: Optional[float] = None,
    mw_linear_range: Optional[Sequence[float]] = None,
    linear_ref_note: str = '',
    model: str = 'regular_star',
) -> ArchitectureResult:
    """Scattering branching contraction factor g = Rg_br^2 / Rg_lin^2 at matched Mw.

    (Theory-and-Equations-Guide, architecture section; Zimm & Stockmayer 1949)

    Parameters
    ----------
    rg_branched : float or sequence of float
        The branched sample's radius of gyration, in the same unit as
        `rg_linear`. A sequence is treated as REPLICATE measurements of one
        sample: g and the effective arm number are then reported as the replicate
        mean +/- SEM (`replicate_mean_se`); a single value carries no ± (deferred
        to replicate-averaging, invariant 8). Non-positive / non-finite values are
        dropped.
    rg_linear : float
        The linear-reference Rg at the branched sample's Mw -- from the fitted
        linear-only Rg-Mw power law (route 2). Same unit as `rg_branched`.
    mw_branched : float, optional
        The branched sample's Mw, for the `mw_in_range` extrapolation check and
        provenance (does not enter g).
    mw_linear_range : (min, max), optional
        The Mw span of the linear reference series. With `mw_branched`, sets
        `mw_in_range` (False -> g relies on extrapolating the law).
    linear_ref_note : str
        Human-readable provenance of the linear reference (shown verbatim).
    model : {'regular_star', 'random_tri', 'random_tetra'}
        Which architecture the reading assumes. The regular-star `effective_arms`
        is computed for EVERY model (always available); a random model ADDITIONALLY
        inverts g -> `branching_index` (n, branch units/molecule) via the Zimm &
        Stockmayer fraction relation for that functionality (Eqs. 47a/47b). g itself
        is model-free -- only its interpretation changes.

    Returns
    -------
    ArchitectureResult

    Raises
    ------
    ValueError
        If `rg_linear` is not positive, no positive `rg_branched` remains, or
        `model` is not a recognised branching model.
    """
    if model not in _BRANCHING_MODELS:
        raise ValueError(f"model must be one of {_BRANCHING_MODELS}, got {model!r}.")
    if not (math.isfinite(rg_linear) and rg_linear > 0):
        raise ValueError(f"rg_linear must be positive, got {rg_linear!r}.")
    vals = [rg_branched] if np.ndim(rg_branched) == 0 else list(rg_branched)
    rg = np.asarray([v for v in vals if v is not None], dtype=float)
    rg = rg[np.isfinite(rg) & (rg > 0)]
    if rg.size == 0:
        raise ValueError("branching_contraction_factor needs at least one "
                         "positive rg_branched value.")

    g_vals = (rg / rg_linear) ** 2
    g_stats = unc.replicate_mean_se(g_vals)
    g, g_se = g_stats.mean, g_stats.sem            # sem is None for n < 2 or a zero spread

    # f_eff is the deterministic regular-star inverse of the REPORTED g, so it can
    # never contradict the displayed g. (Averaging per-replicate inversions over a
    # different, silently-shrunk subset -- some replicates' g>1 drop out of the arms
    # calc but not the g calc -- gave self-contradictory readings: a "linear" mean g
    # beside a many-arm f_eff.) Its uncertainty IS g's (reported on g), so f_eff
    # carries no separate ±; it is NaN when g is outside the star domain (g > 1).
    f_eff = _star_arms_from_g(g)

    # A random model ADDITIONALLY inverts g -> n (branch units/molecule) for its
    # functionality; same deterministic-transform-of-reported-g logic as f_eff, so n
    # carries no separate ± either (invariant 8; branching_index_se stays None).
    branching_index: Optional[float] = None
    if model in _RANDOM_BRANCHING_FUNCTIONALITY:
        branching_index = random_branching_index(
            g, functionality=_RANDOM_BRANCHING_FUNCTIONALITY[model])

    mw_in_range: Optional[bool] = None
    mw_range_out: Optional[tuple] = None
    if mw_linear_range is not None:
        lo, hi = float(min(mw_linear_range)), float(max(mw_linear_range))
        mw_range_out = (lo, hi)
        if mw_branched is not None and math.isfinite(mw_branched):
            mw_in_range = bool(lo <= mw_branched <= hi)

    return ArchitectureResult(
        g=float(g),
        effective_arms=float(f_eff),
        interpretation=_interpret_g(g, f_eff, mw_in_range, model=model,
                                    branching_index=branching_index),
        g_se=g_se,
        effective_arms_se=None,        # f_eff inherits g's uncertainty (shown on g)
        se_source='replicate' if g_se is not None else None,
        n_replicates=int(g_stats.n),
        mw_branched=mw_branched,
        mw_linear_range=mw_range_out,
        mw_in_range=mw_in_range,
        linear_ref_note=linear_ref_note,
        branching_model=model,
        # None when no random model was selected OR the inversion is undefined (g out of
        # (0,1]): a non-finite n must NOT leak as a stored value (consumers guard `is None`).
        branching_index=(float(branching_index)
                         if (branching_index is not None
                             and math.isfinite(branching_index))
                         else None),
        branching_index_se=None,       # n inherits g's uncertainty (shown on g); invariant 8
    )


# ===========================================================================
# Hydrodynamic contraction factor  h = Rh_branched / Rh_linear  (architecture)
# ===========================================================================

# The hydrodynamic SIBLING of g (Burchard, Schmidt & Stockmayer 1980). Where g compares
# mean-square radii of gyration (SLS), h compares hydrodynamic radii (DLS):
# h = D_linear/D_branched = <Rh^-1>_lin / <Rh^-1>_br, i.e. Rh_br/Rh_lin at matched Mw. A
# REGULAR f-arm star follows its OWN relation, NOT g's (3f-2)/f^2 -- the Kirkwood/Zimm
# pre-averaged, non-draining, theta closed form
#     h = sqrt(f) / [ (2 - f) + sqrt(2) (f - 1) ]   (Burchard 1980, Table II)
# -- weaker than the geometric contraction (h > g typically), so comparing g and h probes
# chain topology / draining. Interpretive guidance, never a hard flag.


def _interpret_h(h: float, f_eff: float, mw_in_range: Optional[bool]) -> str:
    """A short, guidance-tier reading of a hydrodynamic contraction factor."""
    if not math.isfinite(h) or h <= 0:
        return "h is not a positive finite number; check the Rh inputs."
    range_note = ''
    if mw_in_range is False:
        range_note = (" ⓘ the branched Mw is out of the linear-reference range, "
                      "so h relies on extrapolating the Rh-Mw law.")
    if h > _CONTRACTION_RADIUS_TOL:
        return (f"h = {h:.3f} > 1: the branched Rh exceeds its linear reference at "
                "this Mw. This can reflect an input problem (check the linear "
                "reference / Mw, or measurement noise) or the expected signature of "
                "a polydisperse branched architecture. Not a star (no arm number)."
                + range_note)
    if h >= _CONTRACTION_RADIUS_TOL_LOW:
        return (f"h = {h:.3f} ~ 1: little or no hydrodynamic contraction -- "
                "consistent with a linear (unbranched) chain." + range_note)
    arms = f"{f_eff:.1f}" if math.isfinite(f_eff) else "n/a"
    return (f"h = {h:.3f} (< 1: smaller Rh than a linear chain at the same Mw). "
            f"Regular-star equivalent f_eff ~ {arms} arms (hydrodynamic relation, "
            "NOT g's) -- guidance, not a literal count (assumes uniform arms; "
            "pre-averaged, non-draining, theta)." + range_note)


def hydrodynamic_contraction_factor(
    rh_branched,
    rh_linear: float,
    *,
    mw_branched: Optional[float] = None,
    mw_linear_range: Optional[Sequence[float]] = None,
    linear_ref_note: str = '',
) -> HydrodynamicResult:
    """Hydrodynamic branching contraction factor h = Rh_br / Rh_lin at matched Mw.

    (Theory-and-Equations-Guide, architecture section; Burchard, Schmidt &
    Stockmayer 1980)

    The hydrodynamic sibling of `branching_contraction_factor` (g). Parameters
    mirror it exactly, with Rh in place of Rg. A sequence `rh_branched` is treated
    as REPLICATE measurements of one sample -> h and its effective arm number are
    the replicate mean +/- SEM (`replicate_mean_se`); a single value carries no ±
    (deferred to replicate-averaging, invariant 8). Non-positive / non-finite
    values are dropped.

    Returns
    -------
    HydrodynamicResult

    Raises
    ------
    ValueError
        If `rh_linear` is not positive, or no positive `rh_branched` remains.
    """
    if not (math.isfinite(rh_linear) and rh_linear > 0):
        raise ValueError(f"rh_linear must be positive, got {rh_linear!r}.")
    vals = [rh_branched] if np.ndim(rh_branched) == 0 else list(rh_branched)
    rh = np.asarray([v for v in vals if v is not None], dtype=float)
    rh = rh[np.isfinite(rh) & (rh > 0)]
    if rh.size == 0:
        raise ValueError("hydrodynamic_contraction_factor needs at least one "
                         "positive rh_branched value.")

    h_vals = rh / rh_linear
    h_stats = unc.replicate_mean_se(h_vals)
    h, h_se = h_stats.mean, h_stats.sem            # sem is None for n < 2 or a zero spread

    # f_eff is the deterministic regular-star inverse of the REPORTED h, so it can
    # never contradict the displayed h (mirrors the g path; its uncertainty IS h's,
    # reported on h). NaN when h is outside the star domain (h > 1).
    f_eff = _star_arms_from_h(h)

    mw_in_range: Optional[bool] = None
    mw_range_out: Optional[tuple] = None
    if mw_linear_range is not None:
        lo, hi = float(min(mw_linear_range)), float(max(mw_linear_range))
        mw_range_out = (lo, hi)
        if mw_branched is not None and math.isfinite(mw_branched):
            mw_in_range = bool(lo <= mw_branched <= hi)

    return HydrodynamicResult(
        h=float(h),
        effective_arms=float(f_eff),
        interpretation=_interpret_h(h, f_eff, mw_in_range),
        h_se=h_se,
        effective_arms_se=None,        # f_eff inherits h's uncertainty (shown on h)
        se_source='replicate' if h_se is not None else None,
        n_replicates=int(h_stats.n),
        mw_branched=mw_branched,
        mw_linear_range=mw_range_out,
        mw_in_range=mw_in_range,
        linear_ref_note=linear_ref_note,
    )


def compare_g_h(g: Optional[float], h: Optional[float]) -> str:
    """A short interpretive note comparing the geometric (g) and hydrodynamic (h)
    contraction factors for one branched sample (Burchard 1980). Guidance only --
    no threshold gates a decision. Empty string when a clean comparison isn't
    possible (a missing or non-star value).

    Every comparison below -- the "exceeds 1" tests AND the g/h ordering -- runs on
    each factor's RADIUS ratio, not on the factor itself (see `_radius_excess`).
    Ordering on the raw pair is NOT equivalent: per-variable monotonicity says nothing
    about a cross-comparison, and g = 0.85, h = 0.90 flips between the two orderings
    (measured in review)."""
    if g is None or h is None or not (math.isfinite(g) and math.isfinite(h)):
        return ''
    if g <= 0 or h <= 0:
        return ''
    g_r = _radius_excess(g, squared=True)     # sqrt(g): Rg_br / Rg_lin
    h_r = _radius_excess(h, squared=False)    # h already IS Rh_br / Rh_lin
    if g_r > _CONTRACTION_RADIUS_TOL and h_r > _CONTRACTION_RADIUS_TOL:
        return ("Both g and h exceed 1 -- not necessarily an input problem: this is "
                "the expected signature of a polydisperse or randomly-branched "
                "architecture (an A_f polycondensate has g ≈ 2 and h ≈ 1.2-1.4; "
                "Burchard 1980, Table II). If a contracted single architecture was "
                "expected, check the linear reference / Mw.")
    if h_r > _CONTRACTION_RADIUS_TOL or g_r > _CONTRACTION_RADIUS_TOL:
        return ("g and h disagree on whether the chain is contracted -- treat with "
                "care (linear reference, Mw match, or noise).")
    if math.isclose(g_r, h_r, rel_tol=1e-9):
        return ("g ≈ h: the geometric and hydrodynamic contractions are equal -- "
                "little architectural distinction to draw at this Mw.")
    if h_r > g_r:
        return ("g < h: the hydrodynamic contraction is weaker than the geometric "
                "one, as expected for a star -- consistent with a non-draining "
                "branched chain. A larger g/h gap leans toward more draining.")
    return ("h < g: the hydrodynamic contraction exceeds the geometric one -- "
            "unusual for a simple star; check the inputs or consider a different "
            "architecture (guidance, not a hard rule).")


# ===========================================================================
# Result-source selection (generic provenance-aware picker)
# ===========================================================================

#
# Wherever one calculation consumes the output of a previous analysis -- Rh and Rg for
# rho = Rg/Rh, Mw for the scaling plots -- the input is chosen from a list of candidate
# sources. Each candidate carries its value AND its provenance, so the GUI can show in
# plain language where a number came from, default to a sensible choice, and let the user
# override (a hand-entered value always wins).

# Design principles enforced here:
#   * NEVER a silent choice. The default is a *labeled* default; the GUI shows `label`
#     verbatim next to the value it used.
#   * Apparent vs thermodynamic is tracked per candidate (`is_apparent`) and is never
#     conflated -- a derived quantity is apparent if ANY of its inputs is apparent.
#   * The default is deterministic: rank by tier (a physical hierarchy), then by fit
#     quality within a tier, then by original list order. Ties among true replicates
#     resolve to the first (never random); explicit averaging is a separate action.

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
    value_se_estimator: Optional[str] = None  # covariance estimator behind value_se, copied from the
    #                                  source fit's se_estimator ('hc3'/'ols'); None for a non-regression
    #                                  SE or no SE. Named value_se_estimator (not se_estimator) so it pairs
    #                                  with value_se and stays a passthrough, not a primary-fit tag.
    calibrated: Optional[bool] = None  # None = calibration N/A (Rg/Rh); real flag for scale-dependent SLS Mw/A2
    # The second, INDEPENDENT scale flag, carried for the same reason `calibrated` is: a
    # picked value must arrive with its own scale verdict, never one re-derived later from
    # a label string. None = N/A (Rg/Rh) or unknown; False = the source run had no dn/dc,
    # so K was 1 and the value is on an arbitrary optical scale.
    dn_dc_supplied: Optional[bool] = None
    model_dependent: bool = False    # True = the value assumes a particle SHAPE (a form-factor P(q) fit:
    #                                  sphere/coil/ellipsoid/Mie/core-shell), so it is right only if that
    #                                  shape is right. False = model-AGNOSTIC (a slope: Zimm/Berry/Debye/
    #                                  Guinier/DLS), right regardless of shape. An orthogonal honesty axis
    #                                  to `is_apparent` -- a model-agnostic candidate outranks a contingent
    #                                  one at equal tier (see select_default_candidate).
    model: str = ''                  # the assumed shape when model_dependent (e.g. 'sphere'), for the label
    degenerate: bool = False         # True = this value comes from a DEGENERATE form-factor fit (an
    #                                  ellipsoid/core-shell whose scattering data several solutions
    #                                  reproduce comparably). Always implies model_dependent. A stronger
    #                                  qualifier than model_dependent alone: the shape assumption is not
    #                                  merely required, it is one the data cannot pin down -> the pick is
    #                                  guided manual entry among physically-admissible solutions. Rendered
    #                                  neutral ⓘ (ui_style_guide §5), and propagated like model_dependent
    #                                  (rule 10). See select_default_candidate + auto_selectable (G1/G3).
    auto_selectable: bool = True     # False = never chosen by the passive default (select_default_candidate
    #                                  filters it out) -- it is offered in the picker but is only ever an
    #                                  EXPLICIT user pick. Each degenerate per-solution candidate sets this
    #                                  False (G3: a degenerate solution is never the tab's silent default);
    #                                  the agree-on-Rg consensus candidate keeps it True (size is determined).
    solution_descriptor: str = ''    # the discriminating shape params of a degenerate solution, e.g.
    #                                  'prolate (ε≈2.1, Rg 129 nm)' / 'core 12 nm / outer 37 nm' -- lets
    #                                  the user map an offered solution to their external picture (G3);
    #                                  '' for a non-degenerate candidate.

    def __post_init__(self):
        # An estimator label is meaningless without an SE for it to govern. A source fit sets
        # se_estimator UNCONDITIONALLY but None-guards its *_se when the point estimate is
        # non-physical (e.g. a Debye fit with a non-positive q->0 intercept), so a caller may
        # copy value_se=None beside value_se_estimator='ols'. Drop the orphan tag here so the
        # "None if no SE" contract holds for every consumer (invariant-8 Clause A).
        if self.value_se is None:
            self.value_se_estimator = None


def select_default_candidate(
    candidates: Sequence['ResultCandidate'],
) -> Optional['ResultCandidate']:
    """Pick the default candidate by the documented deterministic rule.

    Ranking (best first):
      1. highest `tier`            -- a physical hierarchy (e.g. thermodynamic
                                      c->0 extrapolation beats an apparent single
                                      point); the caller assigns tiers.
      2. model-AGNOSTIC over       -- at equal tier, a slope-derived value
         model-CONTINGENT             (Zimm/Berry/Debye/Guinier; model_dependent
                                      =False) always outranks a shape-assuming
                                      form-factor value (model_dependent=True),
                                      regardless of quality. A form-factor Rg is
                                      therefore never the silent default over a
                                      slope-based one *at the same tier* -- it wins
                                      only as the sole candidate at the top tier, or
                                      when it sits a strictly higher tier than every
                                      agnostic one (tier is consulted first).
      3. highest `quality`         -- fit quality within a (tier, model-dependence)
                                      group (R^2, etc.). A None quality sorts below
                                      any real number.
      4. earliest in the input     -- a stable, deterministic tiebreak. For true
                                      replicates this is "the first in the list";
                                      it is never random.

    Candidates whose value is non-finite are ignored, as are candidates flagged
    `auto_selectable=False` (a degenerate form-factor solution -- offered in the
    picker but never the passive default; G3). Returns None if no eligible finite
    candidate exists.
    """
    finite = [c for c in candidates
              if c.value is not None and math.isfinite(c.value) and c.auto_selectable]
    if not finite:
        return None
    # sorted() is stable, so equal (tier, agnostic, quality) keys keep their original
    # order -> the earliest finite candidate wins the final tiebreak. A missing or
    # non-finite quality (e.g. a degenerate fit) sorts below any real quality.
    neg_inf = float('-inf')

    def _q(c: 'ResultCandidate') -> float:
        return c.quality if (c.quality is not None and math.isfinite(c.quality)) else neg_inf

    # `not model_dependent` is True (1) for agnostic, False (0) for contingent -> under
    # reverse=True, agnostic sorts ABOVE contingent at equal tier before quality is
    # consulted (the model-dependence axis, orthogonal to the apparent/thermo tier).
    return sorted(finite,
                  key=lambda c: (c.tier, (not c.model_dependent), _q(c)),
                  reverse=True)[0]


# ===========================================================================
# Synthetic DLS correlogram generation
# ===========================================================================

#
# Generates an artificial g2(tau)-1 correlogram (or g1 / unsubtracted g2) from a
# user-specified set of size populations, with known ground-truth cumulants -- for
# validating the cumulant / CONTIN routines against a known distribution, and for producing
# test files for other software. The output can be written as a two-column CSV that the
# generic DLS parser reads back.

@dataclass
class MultiTauAxis:
    """A quasi-logarithmic (multi-tau) lag axis plus its per-channel sample time.

    Attributes
    ----------
    delay_times_s : np.ndarray
        The lag axis, in seconds.
    sample_times_s : np.ndarray
        The sample time Delta_t_k of the block each channel belongs to, in seconds.
        This is what the per-channel accumulation count M_k = T / Delta_t_k is built
        from; it is NOT recoverable from the axis alone once a duplicated boundary
        lag is present.
    dt0_s : float
        The finest sample time (block 0).
    channels_per_octave : int
        L -- how many channels the sample time stays constant for before doubling.
    n_blocks : int
        Number of blocks (octaves), counting block 0.
    duplicate_indices : np.ndarray
        Indices of channels that repeat the preceding lag exactly. Empty unless
        ``duplicate_boundary_lag=True``.
    """
    delay_times_s: np.ndarray
    sample_times_s: np.ndarray
    dt0_s: float
    channels_per_octave: int
    n_blocks: int
    duplicate_indices: np.ndarray


#: Ceiling on a generated multi-tau axis's channel count. Not physics -- a resource
#: bound, so a mistyped field cannot exhaust memory on the uncancellable worker
#: thread. ~600x the largest axis any supported instrument emits (Panta, 348).
_MULTITAU_MAX_CHANNELS = 200_000


def multi_tau_lag_axis(dt0_s: float, channels_per_octave: int = 8,
                       n_blocks: int = 22, *,
                       duplicate_boundary_lag: bool = False) -> MultiTauAxis:
    """Build a real correlator's multi-tau lag axis.

    Every instrument this program supports emits a quasi-logarithmic staircase, not
    the uniform log grid `np.geomspace` produces: the sample time doubles every L
    channels, so a few hundred channels span nine decades of lag.

    The layout is Schaetzel (1990) Sec. 3.2, printed p. 300 -- "subsequent doubling of
    sample time and lag time increments every L channels". Block 0 uses 2L channels at
    Delta_t_0 covering lags 0 ... (2L-1)*Delta_t_0; every later block j uses L channels
    at Delta_t_j = 2^j * Delta_t_0 covering lags L*Delta_t_j ... (2L-1)*Delta_t_j. That
    is what makes the density uniform in log(tau).

    ONE DOCUMENTED DEPARTURE: the tau = 0 channel is omitted, so block 0 runs
    k = 1 ... 2L-1 (2L-1 channels). No real export ships a zero lag, and
    `core.data_models` requires a strictly increasing axis. The blocks stay exactly
    contiguous either way -- block 0 ends at (2L-1)*Delta_t_0 and block 1 begins at
    L*Delta_t_1 = 2L*Delta_t_0, one fine step later.

    Parameters
    ----------
    dt0_s : float
        The finest sample time Delta_t_0, in seconds (e.g. 25e-9 for an ALV).
    channels_per_octave : int
        L. Schaetzel's own conclusion (Sec. 3.2, printed p. 300, in the same passage
        that prescribes this layout; derivation in Sec. 3.1, Eqs. 53-54) is that
        L = 8 already reduces triangular-averaging distortion below 1e-3. Real
        hardware uses 8 or 16.
    n_blocks : int
        Number of blocks including block 0, i.e. the number of octaves spanned.
    duplicate_boundary_lag : bool
        If True, every block runs to an INCLUSIVE endpoint (k = L ... 2L instead of
        L ... 2L-1), which is the Prometheus Panta's convention. Block j-1's closing
        lag is then 2L*Delta_t_(j-1) = L*Delta_t_j, block j's opening lag -- the same
        lag reported twice, once per block, with two slightly different g2 estimates.
        `parsers.panta_xlsx` discards the later member at load (Magatti and Ferri
        2001, Eq. 11, p. 4014). The emitted axis is non-monotonic on purpose, so it is
        NOT directly loadable; it exists to exercise that discard from synthetic
        ground truth.

    Returns
    -------
    MultiTauAxis

    Raises
    ------
    ValueError
        On a non-positive or non-finite dt0_s, channels_per_octave < 2, n_blocks < 1,
        or an n_blocks whose coarsest sample time would overflow from this dt0_s.
    """
    if not (math.isfinite(dt0_s) and dt0_s > 0):
        raise ValueError(f"dt0_s must be a positive finite number, got {dt0_s!r}.")
    if channels_per_octave < 2:
        raise ValueError(
            f"channels_per_octave must be >= 2, got {channels_per_octave!r}.")
    if n_blocks < 1:
        raise ValueError(f"n_blocks must be >= 1, got {n_blocks!r}.")

    ell = int(channels_per_octave)
    # BOTH guards below are sized to what LEAVES this function, not to an input and not
    # to an intermediate. An earlier version guarded the block sample time dt_j and let
    # an axis of `inf` LAGS through, because the emitted value is k*dt_j with k up to
    # 2L -- the guard and the operation named different quantities, twice over. There is
    # one emitted array; these are its two failure modes, magnitude and size.
    k_last = 2 * ell if duplicate_boundary_lag else 2 * ell - 1
    # MAGNITUDE. The largest emitted lag is k_last * dt0_s * 2**(n_blocks-1). Bounded in
    # EXPONENT space, which is exact and cannot itself overflow: with dt0_s = m*2**e
    # (0.5 <= m < 1) and k_last = mk*2**ek, that product is strictly below
    # 2**(e + ek + n_blocks - 1). Conservative by at most one exponent, which refuses a
    # hair early at the boundary and never lets a non-finite lag out.
    if (math.frexp(dt0_s)[1] + math.frexp(float(k_last))[1]
            + (int(n_blocks) - 1)) > 1024:
        raise ValueError(
            f"the axis overflows: the longest lag would be about "
            f"{k_last} * dt0_s * 2**{int(n_blocks) - 1}, which is not representable "
            f"from dt0_s = {dt0_s!r}. Reduce n_blocks, channels_per_octave, or dt0_s."
        )
    # SIZE. Total channels is (k_last) + (n_blocks-1)*L, accumulated in Python lists, so
    # an unbounded channels_per_octave or n_blocks exhausts memory on a worker thread the
    # GUI cannot cancel -- a freeze rather than an error. Bounded on the PRODUCT for the
    # same reason as above: either input alone is innocent. The ceiling is ~600x the
    # largest axis any supported instrument emits (the Panta's 348).
    n_channels = k_last + (int(n_blocks) - 1) * ell
    if n_channels > _MULTITAU_MAX_CHANNELS:
        raise ValueError(
            f"the axis would have {n_channels} channels, above the "
            f"{_MULTITAU_MAX_CHANNELS} ceiling; real correlators emit a few hundred. "
            f"Reduce channels_per_octave ({ell}) or n_blocks ({int(n_blocks)})."
        )
    # Inclusive endpoint (k runs up to 2L) is what MAKES the duplicate: block j-1 then
    # closes on 2L*Delta_t_(j-1), which is exactly L*Delta_t_j, block j's opening lag.
    # Both products are an integer times dt0 scaled by an exact power of two, so they
    # round identically -- the bit-for-bit equality panta_xlsx compares on (it uses
    # exact equality, no tolerance).
    taus: List[float] = []
    dts: List[float] = []
    dupes: List[int] = []
    for j in range(int(n_blocks)):
        # ldexp(dt0, j), NOT dt0 * 2.0**j: the latter forms 2**j as an intermediate,
        # which OverflowErrors past j = 1023 even where the product dt0*2**j is
        # perfectly representable (dt0 = 25e-9, n_blocks = 1025 is exactly that case).
        # Guarding that intermediate would have needed a second predicate naming a
        # different quantity from the one the guard above states -- rule 12's gap. ldexp
        # scales the exponent directly, so there is no intermediate left to overflow and
        # the frexp guard above is sized to the only operation that remains. Exact for
        # powers of two, so the duplicate-boundary bit-identity is unaffected.
        dt_j = math.ldexp(dt0_s, j)
        # Block 0 spans k = 1 ... k_last (the tau = 0 channel is dropped -- see the
        # docstring); every later block spans k = L ... k_last, which begins exactly
        # one fine step past the previous block's end (or ON it, when inclusive).
        for k in range(1 if j == 0 else ell, k_last + 1):
            tau = k * dt_j
            if taus and tau == taus[-1]:
                dupes.append(len(taus))
            taus.append(tau)
            dts.append(dt_j)

    return MultiTauAxis(
        delay_times_s=np.array(taus, dtype=float),
        sample_times_s=np.array(dts, dtype=float),
        dt0_s=float(dt0_s),
        channels_per_octave=ell,
        n_blocks=int(n_blocks),
        duplicate_indices=np.array(dupes, dtype=int),
    )


def _schatzel_classical_sd(tau: np.ndarray, sample_times: np.ndarray,
                           gamma_bar: float, beta: float) -> np.ndarray:
    """Relative per-channel SD of g2-1 from Schaetzel (1990) Eq. (49), CLASSICAL LIMIT.

    Eq. (49) (printed p. 296) gives the
    variance of a photon-correlation estimate for a single exponential in six term
    groups. The last two carry `n_bar^-1` and `n_bar^-2` -- the cross and shot-noise
    contributions -- and vanish as the mean count rate goes to infinity. What is left
    is the CLASSICAL group, which depends only on

        beta          the intercept,
        Gamma_k       = gamma_bar * Delta_t_k, the decay per sample time,
        k             = tau_k / Delta_t_k, the channel index within its block,
        M_k           = T / Delta_t_k, the number of products averaged,

    all of which the generator already knows. Peters (1993, p. 152) measured that this
    classical intensity noise dominates in almost every channel except roughly the
    first 32, so the limit is a good description of most of a real axis; it degrades at
    short lag and at low count rate.

    THE RUN DURATION T CANCELS, so this function does not take one. M_k enters only as
    1/M_k = Delta_t_k / T, in which T is the same for every channel; the caller uses
    the RELATIVE shape and sets the absolute scale itself, and a common factor divides
    out of a ratio. (This was carried as an unverified assumption for a while -- it
    holds, both algebraically here and numerically in
    tests/test_utilities.py::test_schatzel_shape_is_independent_of_run_duration.) The
    per-channel Delta_t_k survives, which is what makes the coarse blocks the noisy
    ones.

    TWO STATED APPROXIMATIONS. (1) Eq. (49) is derived for a SINGLE exponential; a
    multimodal spectrum is fed in through its amplitude-weighted mean decay rate
    gamma_bar, which is the generator's own ground truth rather than a fitted guess,
    but it is still an effective-Gamma substitution. (2) Only the SHAPE is used --
    the caller sets the absolute scale via noise_level (see
    `generate_synthetic_correlogram`), because an absolute prediction would need a real
    photon count rate and the acquisition's normalisation convention, neither of which
    a generator can honestly supply. The return is
    therefore in ARBITRARY UNITS -- only ratios between its entries mean anything.

    (3) Eq. (49) is a CLOSED FORM derived for Gamma = Delta_t / tau_c << 1 (Biganzoli
    and Ferri 2018, p. 29378). A multi-tau axis violates that in its coarsest blocks by
    construction, which is where this shape is least trustworthy. Their Eqs. (8)-(9)
    are the exact-in-Gamma generalisation and are the canonical replacement; they are
    not used here because they must be re-read from the source (the printed forms carry
    prefactor inconsistencies) and because an approximate SHAPE for generated test data
    is a far weaker claim than a variance used to weight a real fit. Anything that
    weights real data should start from the exact forms, not from this.
    """
    dt = sample_times
    gamma_dt = gamma_bar * dt
    # Guarded on the COMPUTED product, not on gamma_bar or dt separately: coth(Gamma)
    # is 1/tanh, so a Gamma that underflows to 0.0 divides by zero, and neither input
    # alone can be blamed -- a small gamma_bar with a fine sample time is what gets
    # there. The other tail is safe: a huge Gamma sends tanh to 1 and every
    # exponential to 0.
    if not (np.all(np.isfinite(gamma_dt)) and np.min(gamma_dt) > 0):
        raise ValueError(
            f"the Schaetzel noise model needs a positive finite decay-per-sample-time "
            f"gamma_bar * Delta_t on every channel; the smallest is "
            f"{float(np.min(gamma_dt))!r} for gamma_bar = {gamma_bar!r} s^-1. Check the "
            f"population sizes and multitau_dt0_s."
        )
    k = tau / dt
    coth_2g = 1.0 / np.tanh(2.0 * gamma_dt)
    coth_g = 1.0 / np.tanh(gamma_dt)
    e4 = np.exp(-4.0 * gamma_dt * k)
    e2 = np.exp(-2.0 * gamma_dt * k)
    # 1/M_k = Delta_t_k / T, with the common 1/T dropped -- see the docstring.
    var = (beta * beta * dt) * (
        coth_2g
        + e4 * (2.0 * k + coth_2g)
        + 2.0 * beta * e4 * (-2.0 * k + coth_2g - 2.0 * coth_g)
        + 4.0 * beta * e2 * coth_g
    )
    # The third group is negative-going, so floating cancellation can leave a variance
    # a few ulp below zero where the true value is ~0; clamping keeps that from
    # becoming a nan SD. It is a rounding floor, not a physical one.
    return np.sqrt(np.maximum(var, 0.0))


def _channel_sample_times(tau: np.ndarray) -> np.ndarray:
    """Per-channel sample time for an axis that does not carry one explicitly.

    On a multi-tau axis Delta_t_k is a property of the block and comes from
    `MultiTauAxis`; on a plain log or linear grid the only sensible stand-in is the
    local channel spacing, with the first channel borrowing the second's (there is no
    preceding channel to difference against). Used by the 'schatzel_classical' noise
    model so it stays orthogonal to the choice of axis.
    """
    if tau.size < 2:
        return np.full(tau.shape, 1.0)
    step = np.diff(tau)
    return np.concatenate(([step[0]], step))


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
        this population, modeled as log-normal. 0.0 (default) means a perfectly
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
    grid: str                          # 'log' | 'linear' | 'multitau'

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

    # Multi-tau / per-channel-noise context. Defaulted so every existing caller and
    # every existing constructed instance is unaffected; the first two are what a
    # variance-weighting study reads back.
    noise_model: str = 'constant'      # 'constant' | 'schatzel_classical'
    sample_times_s: Optional[np.ndarray] = None    # per-channel Delta_t_k
    noise_sd: Optional[np.ndarray] = None          # per-channel sigma actually drawn
    duplicate_lag_indices: Optional[np.ndarray] = None


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
    noise_model: str = 'constant',
    multitau_dt0_s: float = 2.5e-8,
    multitau_channels_per_octave: int = 8,
    multitau_n_blocks: int = 22,
    multitau_duplicate_boundary_lag: bool = False,
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
    scattering-weighted contributions and are normalized so g1(0) = 1.

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
        typical DLS window. IGNORED when grid='multitau', whose axis is set by the
        multitau_* parameters instead.
    n_points : int
        Number of delay-time points (default 200, matching a Brookhaven export).
        IGNORED when grid='multitau' (see above).
    grid : str
        'log' (default; UNIFORMLY log-spaced), 'linear', or 'multitau' -- the
        quasi-logarithmic staircase a real correlator actually emits, built by
        `multi_tau_lag_axis` from the multitau_* parameters below. Every instrument
        this program supports emits a multi-tau axis, so 'multitau' is the realistic
        choice; 'log' remains the default so existing callers are untouched.
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
    noise_model : str
        'constant' (default): one SD across every channel, the historical behaviour --
        and a premise no correlator satisfies.
        'schatzel_classical': the SD varies channel by channel with the SHAPE of
        Schaetzel (1990) Eq. (49) in its classical (infinite-count-rate) limit, scaled
        so that `noise_level` is the SD of the FIRST channel. See
        `_schatzel_classical_sd` for the formula, its two stated approximations, and
        why only the shape is taken.

        Why the shape matters: it is not monotonic. The per-channel noise falls
        through the correlogram's decay, then TURNS UP in the coarsest blocks where
        M_k = T / Delta_t_k leaves few products to average -- the upturn Biganzoli and
        Ferri (2018, p. 29381) report in the last stage. Over a 22-octave ALV-like axis
        it spans a factor of ~20. That span is RELATIVE -- the absolute scale is the
        caller's noise_level -- so Peters' measured per-channel range is not a check
        on it; what the sources corroborate is the shape, not the magnitude.
        Weighting the accumulation count alone would instead ramp
        monotonically by ~1400x and bury the longest lags in noise larger than the
        signal.

        The run duration does not appear: it is common to every channel and cancels
        out of the relative shape (see `_schatzel_classical_sd`).

        Orthogonal to `grid`: on a 'log' or 'linear' axis Delta_t_k is the local
        channel spacing.
    multitau_dt0_s : float
        Finest sample time Delta_t_0, in seconds (default 25 ns, an ALV's). grid='multitau' only.
    multitau_channels_per_octave : int
        L, channels before the sample time doubles (default 8). grid='multitau' only.
    multitau_n_blocks : int
        Number of octaves (default 22, reaching ~0.79 s from a 25 ns dt0 in 183
        channels). grid='multitau' only.
    multitau_duplicate_boundary_lag : bool
        Emit each block-boundary lag twice, the Prometheus Panta's convention (default
        False). The resulting axis is NOT strictly increasing and so is not loadable
        until a parser discards the repeats; see `multi_tau_lag_axis`.

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
        if not (math.isfinite(p.rh_nm) and p.rh_nm > 0):
            raise ValueError(
                f"population rh_nm must be a positive finite number, got {p.rh_nm!r}.")
        if not (math.isfinite(p.weight) and p.weight > 0):
            raise ValueError(
                f"population weight must be a positive finite number, got {p.weight!r}.")
        if not (math.isfinite(p.spread_cv) and p.spread_cv >= 0):
            raise ValueError(
                f"population spread_cv must be a finite number >= 0, got {p.spread_cv!r}.")
    if not (0 < beta <= 1.0):
        raise ValueError(f"beta must be in (0, 1], got {beta!r}.")
    if not (math.isfinite(noise_level) and noise_level >= 0):
        raise ValueError(f"noise_level must be a finite number >= 0, got {noise_level!r}.")
    if not (math.isfinite(delay_min_s) and delay_min_s > 0
            and math.isfinite(delay_max_s) and delay_max_s > delay_min_s):
        raise ValueError(
            f"require 0 < delay_min_s < delay_max_s, got "
            f"{delay_min_s!r} and {delay_max_s!r}."
        )
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points!r}.")
    if grid not in ('log', 'linear', 'multitau'):
        raise ValueError(
            f"grid must be 'log', 'linear', or 'multitau', got {grid!r}.")
    if noise_model not in ('constant', 'schatzel_classical'):
        raise ValueError(
            f"noise_model must be 'constant' or 'schatzel_classical', got "
            f"{noise_model!r}.")
    if output_form not in ('g2m1', 'g2', 'g1'):
        raise ValueError(
            f"output_form must be 'g2m1', 'g2', or 'g1', got {output_form!r}."
        )

    # Scattering vector (m^-1), using the same physics as the analysis code.
    q = scattering_vector_q_m(angle_deg, wavelength_nm, solvent_refractive_index)
    # q**2, not q: the generator divides by the SQUARE below (d_eff = gamma_bar / q**2),
    # and an extreme-but-finite wavelength or refractive index makes q underflow so that
    # q**2 reaches exactly 0.0 while q itself is still non-zero -- a bare ZeroDivisionError
    # out of a data generator. Refused up front rather than guarded at the division:
    # nothing downstream of an unusable q is worth generating.
    _q2 = q * q          # `*` saturates to inf; `**` RAISES OverflowError on a Python
    #                      float, i.e. inside the guard, before it can refuse. See
    #                      analysis/dls/_common.py::_d_from_gamma for the full note.
    if not (math.isfinite(_q2) and _q2 > 0):
        raise ValueError(
            f"the scattering vector is unusable for these optics (q = {q!r} m^-1, so "
            f"q^2 underflows or is not finite) -- check angle_deg={angle_deg!r}, "
            f"wavelength_nm={wavelength_nm!r}, "
            f"solvent_refractive_index={solvent_refractive_index!r}."
        )

    # Delay-time grid. 'multitau' ignores delay_min_s/delay_max_s/n_points -- its span
    # and length follow from dt0 and the block count (documented in the docstring).
    axis: Optional[MultiTauAxis] = None
    if grid == 'multitau':
        axis = multi_tau_lag_axis(
            multitau_dt0_s, multitau_channels_per_octave, multitau_n_blocks,
            duplicate_boundary_lag=multitau_duplicate_boundary_lag)
        tau = axis.delay_times_s
        sample_times = axis.sample_times_s
    elif grid == 'log':
        tau = np.geomspace(delay_min_s, delay_max_s, n_points)
        sample_times = _channel_sample_times(tau)
    else:
        tau = np.linspace(delay_min_s, delay_max_s, n_points)
        sample_times = _channel_sample_times(tau)

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
            # Discretize the continuous pdf onto the GEOMETRIC grid: the per-bin
            # mass is pdf(r)*dr, and on a log-spaced grid dr = r*d(ln r) with
            # d(ln r) constant, so the discrete weight is proportional to pdf*r.
            # Omitting the r Jacobian centres the emitted spectrum's geometric
            # mean at rh_nm/(1+CV^2) instead of the intended median rh_nm
            # (-8% at CV=0.3, -19.5% at CV=0.5).
            a = pdf * radii
            a = a / a.sum() * p.weight   # normalize to the population weight
        for r, amp in zip(radii, a, strict=True):
            d = stokes_einstein_diffusion_coefficient(r * 1e-9, temperature_K, viscosity_Pa_s)
            gammas.append(d * q ** 2)
            amps.append(float(amp))

    gammas = np.array(gammas, dtype=float)
    amps = np.array(amps, dtype=float)
    amps = amps / amps.sum()           # normalize so g1(0) = 1

    # First cumulant. Computed here rather than with the other ground-truth moments
    # below because the Schaetzel noise model needs it BEFORE the noise is drawn.
    gamma_bar = float(np.sum(amps * gammas))

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

    # Per-channel noise SD. 'constant' broadcasts one value (the historical
    # behaviour); 'schatzel_classical' takes the per-channel SHAPE of Eq. (49)'s
    # classical limit and rescales it so noise_level remains the first channel's SD --
    # see the docstring for what that models and, just as importantly, what it does not.
    if noise_model == 'schatzel_classical':
        shape = _schatzel_classical_sd(tau, sample_times, gamma_bar, beta)
        # Normalised on the FIRST channel, whose shape value is the largest of the
        # short-lag plateau, so noise_level keeps one meaning across both models.
        # Guarded because the reference is a computed quantity: an underflowed or
        # non-finite shape[0] would silently scale the whole field to 0 or nan.
        if not (math.isfinite(shape[0]) and shape[0] > 0):
            raise ValueError(
                f"the Schaetzel noise model produced a non-positive reference SD "
                f"({shape[0]!r}) on the first channel, so noise_level cannot be "
                f"anchored to it; check beta and the population sizes."
            )
        sigma = noise_level * (shape / shape[0])
    else:
        sigma = np.full(tau.shape, float(noise_level))
    if noise_level > 0:
        rng = np.random.default_rng(seed)
        # rng.normal broadcasts scale elementwise, so one draw covers both models.
        signal = signal + rng.normal(0.0, sigma, size=signal.shape)

    # Ground-truth cumulants (amplitude-weighted moments of the decay rate).
    # gamma_bar is computed above, before the noise draw.
    mu2 = float(np.sum(amps * (gammas - gamma_bar) ** 2))
    # The guard is on the SQUARE, not on gamma_bar: the division squares it, so an
    # extreme-but-finite decay rate underflows the denominator to 0.0 while
    # `gamma_bar != 0` still passes. Same mis-sizing as the q^2 guard above.
    # (Written as `gamma_bar * gamma_bar`, never `** 2` -- a Python-float `**` RAISES
    # OverflowError on the other tail, inside the guard; see _d_from_gamma's note.)
    _gb2 = gamma_bar * gamma_bar        # `*` not `**`: see _d_from_gamma's note
    pdi = mu2 / _gb2 if (math.isfinite(_gb2) and _gb2 > 0) else float('nan')
    d_eff = gamma_bar / _q2
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
        noise_model=noise_model,
        sample_times_s=sample_times,
        noise_sd=sigma,
        duplicate_lag_indices=(axis.duplicate_indices if axis is not None
                               else np.array([], dtype=int)),
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
        If delay_unit is unrecognized.
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
