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
writes a file, or mutates its inputs. The CSV writer for a generated correlogram,
export_synthetic_correlogram_csv(), lives in exporting/instrument_files.py.

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
from core.notes import Note, render
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
    # record notes (the Rg and Rh sample keys disagree); the module also emits each as a
    # UserWarning, its export clause the message (keep-both).
    notes: Tuple[Note, ...] = ()


# ===========================================================================
# SLS diagnostics
# ===========================================================================

def i_sin_theta(
    measurements: Sequence[SLSMeasurement],
    mode: str = 'absolute',
    display_names: Optional[Sequence[str]] = None,
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
    display_names : sequence of str, optional
        One name per measurement, in order -- what each curve is CALLED on the
        plot's legend. Supplied by the caller because a pure analysis function may
        not look anything up (invariant 1), and because the name a user recognises
        is the one the rest of the program shows. Omitted (the default) the label
        falls back to the measurement's own `sample_label`, then to
        "{polymer} in {solvent}, c=...".

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
    if display_names is not None and len(display_names) != len(measurements):
        raise ValueError(
            f"display_names has {len(display_names)} entries for "
            f"{len(measurements)} measurements; they are positional."
        )

    curves: List[ISinThetaCurve] = []
    for i, m in enumerate(measurements):
        # The name this curve is CALLED: the caller's, else the measurement's own
        # sample label, else its identity. Resolved before it is needed, so the
        # legend and the refusal below name the same curve the same way.
        given = display_names[i] if display_names is not None else ''
        if given:
            label = given
        elif m.sample_label:
            label = m.sample_label
        else:
            label = (f"{m.polymer_name} in {m.solvent_name}, "
                     f"c={m.concentration_g_per_mL:g} g/mL")
        sin_theta = np.sin(np.radians(m.angles_deg))
        values = m.intensities * sin_theta
        if mode == 'normalized':
            mean_val = values.mean()
            if mean_val == 0:
                raise ValueError(
                    f"Cannot normalize: mean I*sin(theta) is zero for a "
                    f"measurement ({label})."
                )
            values = values / mean_val
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
    "runs ~10-14% below them (Theory-and-Equations-Guide §5.1), so a real sample can "
    "read one band low. "
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


def _interpret_rho(rho: float) -> str:
    """Return a short textual interpretation of a rho = Rg/Rh value.

    Reference values are architecture-resolved from Burchard (1999, Table 4;
    theta-solvent unless noted); the hard sphere's geometric half, Rg^2 = (3/5) R^2
    for a uniform solid sphere, is Chu (1991) Eq. 2.A.7, p. 51, and Chu prints no
    Rg/Rh ratio. rho is a *qualitative* architecture fingerprint, not a hard
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
    notes: List[Note] = []
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
            if require_match:
                raise ValueError(
                    f"Rg and Rh sample keys disagree on polymer/solvent/"
                    f"temperature: Rg from {rg_sample_key}, Rh from {rh_sample_key}. "
                    f"rho is only meaningful for the same sample.")
            # The three fields the check compares; concentration is extrapolated away.
            note = Note('rho_sample_mismatch', (), {
                'rg_key': (rg_sample_key.polymer_name, rg_sample_key.solvent_name,
                           float(rg_sample_key.temperature_K)),
                'rh_key': (rh_sample_key.polymer_name, rh_sample_key.solvent_name,
                           float(rh_sample_key.temperature_K))})
            notes.append(note)   # one source -> both the note and the warning
            warnings.warn(render(note, surface='export'), UserWarning, stacklevel=2)

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
    # record notes: the linear reference law h was read against (`linear_reference_law`)
    notes: Tuple[Note, ...] = ()


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
    effective_arms_se: Optional[float] = None  # f_eff inherits g's uncertainty (shown on g)
    se_source: Optional[str] = None       # 'replicate' when a ± is reported, else None
    n_replicates: int = 1
    mw_branched: Optional[float] = None
    mw_linear_range: Optional[tuple] = None    # (min, max) Mw of the linear reference series
    mw_in_range: Optional[bool] = None    # branched Mw within the linear span? None if unknown
    # record notes: the linear reference law g was read against (`linear_reference_law`)
    notes: Tuple[Note, ...] = ()
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
    f = 3 or 4 (the tabulated cases). See the Theory-and-Equations-Guide, "Random branching: the branching index m_b"."""
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


def _linear_reference_notes(quantity: str, about: str, n_linear_points: Optional[int],
                            mw_range: Optional[tuple]) -> Tuple[Note, ...]:
    """The record of the linear reference law a contraction factor was read against: its
    radius, how many linear samples fitted it and their Mw span; none when the caller
    did not say both."""
    if n_linear_points is None or mw_range is None:
        return ()
    return (Note('linear_reference_law', (about,),
                 {'quantity': quantity, 'n': int(n_linear_points),
                  'lo': float(mw_range[0]), 'hi': float(mw_range[1])}),)


def _mw_range_verdict(mw_linear_range: Optional[Sequence[float]],
                      mw_branched: Optional[float]
                      ) -> Tuple[Optional[bool], Optional[tuple]]:
    """`(mw_in_range, mw_range_out)` for a contraction factor: whether the branched Mw lies
    inside the linear reference series' Mw span, and that span as `(lo, hi)`. The first is
    None without a span or a finite branched Mw, the second None without a span."""
    mw_in_range: Optional[bool] = None
    mw_range_out: Optional[tuple] = None
    if mw_linear_range is not None:
        lo, hi = float(min(mw_linear_range)), float(max(mw_linear_range))
        mw_range_out = (lo, hi)
        if mw_branched is not None and math.isfinite(mw_branched):
            mw_in_range = bool(lo <= mw_branched <= hi)
    return mw_in_range, mw_range_out


def branching_contraction_factor(
    rg_branched,
    rg_linear: float,
    *,
    mw_branched: Optional[float] = None,
    mw_linear_range: Optional[Sequence[float]] = None,
    n_linear_points: Optional[int] = None,
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
    n_linear_points : int, optional
        How many linear samples the reference law was fitted to. With
        `mw_linear_range`, recorded as the `linear_reference_law` note.
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

    mw_in_range, mw_range_out = _mw_range_verdict(mw_linear_range, mw_branched)

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
        notes=_linear_reference_notes('Rg', 'g', n_linear_points, mw_range_out),
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
    n_linear_points: Optional[int] = None,
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

    mw_in_range, mw_range_out = _mw_range_verdict(mw_linear_range, mw_branched)

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
        notes=_linear_reference_notes('Rh', 'h', n_linear_points, mw_range_out),
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
                "Theory-and-Equations-Guide §5.4.1). If a contracted single "
                "architecture was "
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
    standard_index_known: Optional[bool] = None  # the source fit's own: False = it applied k_c with the
    #                                  standard's refractive index blank, the cause of calibrated=False;
    #                                  None = N/A (Rg/Rh). A store copies it beside `calibrated`.
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
    notes: Tuple[Note, ...] = ()     # the failed preconditions behind the label's clause, each
    #                                  about `value`: what a store copies onto the sample, re-aimed
    #                                  at its value field, so the Cross-Sample row marks the value
    #                                  (core.workspace.set_value_notes). () when it failed none.
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


def _schatzel_sd(tau: np.ndarray, sample_times: np.ndarray, gamma_bar: float,
                 beta: float,
                 mean_count_rate_cps: Optional[float] = None) -> np.ndarray:
    """Relative per-channel SD of g2-1 from Schaetzel (1990) Eq. (49), ALL SIX groups.

    Eq. (49) (printed p. 296) writes the variance of a photon-correlation estimate for
    a single exponential as a prefactor `M_k^-1 beta^2` times six term groups. Four of
    them depend only on the intercept beta, the decay per sample time
    Gamma_k = gamma_bar * Delta_t_k, and the channel index k = tau_k / Delta_t_k. The
    remaining two carry the mean counts per sample time n_bar -- the cross and
    shot-noise contributions -- and sit inside the same braces:

        + 2 beta^-1 n_bar^-1 [1 + e^(-4 Gamma_k k) + 2 beta e^(-2 Gamma_k k)]
        + beta^-2 n_bar^-2 [1 + beta e^(-2 Gamma_k k)]

    `mean_count_rate_cps=None` (the default) drops those two, which is the
    infinite-count-rate CLASSICAL LIMIT that `_schatzel_classical_sd` exposes as its own
    entry point. That function's docstring is the home for the shared approximations,
    the citations, and how far the limit is from real data; this one is the computation
    both go through, so the two forms cannot drift apart.

    n_bar_k = mean_count_rate_cps * Delta_t_k -- mean counts per sample time, evaluated
    per block, the multi-tau substitution of Biganzoli and Ferri (2018, p. 29381). The
    run duration T still cancels with the shot groups present: it enters only through
    the common 1/M_k = Delta_t_k / T prefactor, and neither new group carries an M of
    its own. The return is therefore a RELATIVE shape in arbitrary units, exactly as in
    the classical case -- only ratios between its entries mean anything.

    The Theory-and-Equations-Guide, "The Noise on a Correlogram" carries the equation
    for a reader who does not have this file.
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
    braces = (
        coth_2g
        + e4 * (2.0 * k + coth_2g)
        + 2.0 * beta * e4 * (-2.0 * k + coth_2g - 2.0 * coth_g)
        + 4.0 * beta * e2 * coth_g
    )
    # 1/M_k = Delta_t_k / T, with the common 1/T dropped -- see the docstring.
    if mean_count_rate_cps is None:
        var = (beta * beta * dt) * braces
    else:
        # n_bar is COMPUTED from two inputs, so the guard is on it and on the variance
        # it produces rather than on the count rate alone, and it is two-sided: a huge
        # rate is harmless (the shot groups vanish, recovering the classical limit)
        # while a tiny one sends 1/n_bar^2 to inf long before n_bar itself underflows
        # to 0. Computing first and testing the result is what keeps the predicate and
        # the operation the same expression. `nbar * nbar`, never `nbar ** 2`: on a
        # Python float `**` RAISES OverflowError where `*` saturates to inf, which
        # would beat the guard to the answer.
        nbar = mean_count_rate_cps * dt
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            braces = braces + (
                (2.0 / (beta * nbar)) * (1.0 + e4 + 2.0 * beta * e2)
                + (1.0 / (beta * beta * nbar * nbar)) * (1.0 + beta * e2)
            )
            var = (beta * beta * dt) * braces
        if not (np.all(np.isfinite(nbar)) and np.min(nbar) > 0
                and np.all(np.isfinite(var))):
            raise ValueError(
                f"mean_count_rate_cps = {mean_count_rate_cps!r} gives mean counts per "
                f"sample time that the Schätzel variance's shot-noise terms cannot use "
                f"(Theory-and-Equations-Guide §2.3): the smallest "
                f"n_bar over the channels is {float(np.min(nbar))!r}, and the terms in "
                f"1/n_bar and 1/n_bar^2 must stay finite on every channel. Use a "
                f"realistic photon count rate (a real file's is tens to hundreds of kHz)."
            )
    # The third group is negative-going, so floating cancellation can leave a variance
    # a few ulp below zero where the true value is ~0; clamping keeps that from
    # becoming a nan SD. It is a rounding floor, not a physical one.
    return np.sqrt(np.maximum(var, 0.0))


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

    all of which the generator already knows. Peters (1993, p. 152) reports, for one
    latex sample (0.25 ms decay time, 300 kHz mean count rate, ALV-5000), that this
    classical intensity noise "dominates in almost every correlation channel except the
    first 32 channels, where photon noise still contributes significantly". On that run
    the limit describes most of the axis; the channel count belongs to that sample and
    count rate, not to axes in general. That the limit degrades at low count rate is this
    program's own reading of the two dropped groups (see (4) below), not Peters'.

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
    construction, which is where this shape is least trustworthy. Their Eq. (9) is the
    exact-in-Gamma generalisation and the canonical replacement, implemented beside this
    function as `_biganzoli_sd`; the generator offers that exact form as
    noise_model='biganzoli_classical' / 'biganzoli_full' and keeps this closed form as
    its 'schatzel_*' models. This closed form weights no real data: the nonlinear
    cumulant's sourced weight evaluates the exact form, of which this is the Gamma << 1
    limit -- kept as the generator's 'schatzel_*' models and as the limit oracle the
    exact form is tested against, never as a user-selectable weighting.

    (4) HOW WRONG THE CLASSICAL LIMIT IS, measured 2026-09-06 rather than left
    qualitative. Dropping the two n_bar terms is not a small approximation over the
    channels that carry fit weight: across the retained range this shape spans 1.9x
    where real replicate groups span 2.5-22.6x (ALV) and 34-145x (Brookhaven). The
    retained range IS the short-lag region, which is exactly where Peters (1993, p. 152)
    says shot noise still matters, so the limit is at its worst precisely where the fit
    looks. Scored as log10-RMS against the measured per-channel SD on the SIX scorable
    ALV replicate groups, this shape reaches a median 0.180 where the full six-term
    Eq. (49) reaches 0.101 -- and the sample-SD estimator's own noise floor at n = 10 is
    0.107, so the full form is at the floor and this one is not. (The five Brookhaven
    groups are n = 3, where the floor is 0.277 and the arms are 0.335 against 0.282:
    same ordering, far less resolving power.) Those shot terms now EXIST, opt-in: they
    take one new input, a photon count rate, which the ALV and Brookhaven headers
    already carry, and `generate_synthetic_correlogram`'s `noise_model='schatzel_full'`
    turns them on.
    The Theory-and-Equations-Guide, "The Noise on a Correlogram" carries this limit
    and its figures for the reader who does not have this file.

    This function is the `mean_count_rate_cps=None` branch of `_schatzel_sd`, which
    carries all six term groups. Both forms go through that one computation, so the
    limit cannot drift from the equation it is a limit of.
    """
    return _schatzel_sd(tau, sample_times, gamma_bar, beta,
                        mean_count_rate_cps=None)


#: Above this decay per sample time Gamma_k, `_biganzoli_sd` evaluates Eq. (9) in its
#: large-Gamma asymptotic form rather than directly: sinh^6(Gamma) leaves float near
#: Gamma ~ 120, and past 30 every factor the asymptote drops is 1 to below float
#: resolution (6 e^(-60) ~ 5e-26). Measured 2026-09-11 on k = 1..30, beta in {1, 0.8,
#: 0.5}, classical and n_bar in {1, 0.1}, with and without a run duration: the two forms
#: agree on the variance to 1.0e-15 just below the switch and 6.7e-16 just above it, and
#: each to 8.9e-16 of Eq. (9) in 250-digit arithmetic (never above 7e-15 on 15..100).
_BIGANZOLI_ASYMPTOTE_GAMMA = 30.0

#: Below this Gamma, Eq. (10)'s chi0^2 is summed from its Taylor series rather than
#: formed as written, whose cancellation costs ~eps / Gamma. Measured 2026-09-11 on
#: [0.05, 0.2] against 60-digit arithmetic: the series to 2.2e-16, the closed form to
#: 3.6e-15, the two within 3.4e-15 of each other on either side of the switch.
_BIGANZOLI_CHI0_SERIES_GAMMA = 0.1
#: chi0^2 = 2 * sum_{n >= 0} (-2 Gamma)^n / (n + 2)!, coefficients of Gamma^n.
_BIGANZOLI_CHI0_SERIES = tuple(2.0 * (-2.0) ** n / math.factorial(n + 2)
                               for n in range(14))

#: How far from an integer, relative to it, a lag index k = tau / Delta_t is still that
#: integer in `_biganzoli_sd`: the tolerance the Brookhaven parser matches a delay written
#: as '%.9e' to (`parsers.brookhaven_dat._HS_DELAY_RTOL`). Why it is needed: near k = 1
#: Eq. (9)'s S5 group grows as (k - 1) e^Gamma, so k = 1 + ulp moves the SD by -15 % at
#: Gamma = 48 and k = 1 - ulp inflates it x153 at Gamma = 60.
_BIGANZOLI_LAG_INDEX_RTOL = 1e-6


class _Eq9Refused(ValueError):
    """`_biganzoli_sd` refused its inputs, and on which: `cause` is the refusal's key
    (`decay_per_sample_time`, `lag_index`, `run_duration`, `count_rate`, `channel_sd`),
    the one a weight that falls back names (`cumulants._WEIGHT_CAUSES`), and the message
    names the input to change. A `ValueError`, so a caller that catches one still does."""

    def __init__(self, cause: str, message: str) -> None:
        super().__init__(message)
        self.cause = cause


def _biganzoli_sd(tau: np.ndarray, sample_times: np.ndarray, gamma_bar: float,
                  beta: float, mean_count_rate_cps: Optional[float] = None,
                  duration_s: Optional[float] = None) -> np.ndarray:
    """Relative per-channel SD of g2-1 from Biganzoli and Ferri (2018) Eq. (9), exact in Gamma.

    The synthetic generator's two 'biganzoli_*' noise models draw through this same
    function, so a planted correlogram's SD and a weight built from it are one
    computation.

    Eq. (9) with Eq. (10) (printed p. 29378) is the variance of a photon-correlation
    estimate for a single exponential with triangular averaging carried to all orders,
    so it holds for any decay per sample time Gamma = Delta_t / tau_c, not only
    Gamma << 1. With k the lag index, M = T / Delta_t the number of sample times in the
    run, n_bar the mean counts per sample time, and the shorthands
    S_n = sinh^n(Gamma) / Gamma^n, E2 = e^(-2 Gamma k), E4 = e^(-4 Gamma k):

        sigma^2(k) = beta^2 / (M - k) * {
              2 S2 chi0^2 E4 + 4 beta S2 chi0^2 E2 + 8 beta S3 chi0 E4
            + S4 [coth(2 Gamma) - 1 + 4 beta E2 (coth(Gamma) - 1)
                  + E4 (a coth(2 Gamma) + c)]
            + 4 beta S6 E4 (coth(Gamma) - 1) - 8 beta S5 E4 (k - 2 + coth(Gamma))
            + chi0^4
            + 2 beta^-1 n_bar^-1 [chi0^2 + S2 E2 (E2 + 2 beta chi0)
                                  + 2 beta S3 E4 - 2 beta S4 E4]
            + beta^-2 n_bar^-2 [1 + beta S2 E2] }

    with, from Eq. (10), chi0^2 = (2 Gamma - 1 + e^(-2 Gamma)) / (2 Gamma^2),
    a = 1 + 2 beta and c = 2k + 4 beta k - 2 - 8 beta + 4 beta chi0^2 - 16 beta chi0.
    The same page states that Eq. (9) tends to Schaetzel (1990) Eq. (49), printed p. 296
    (`_schatzel_sd`), as Gamma << 1; the sum converges at O(Gamma^2), though its term
    groups do not map one to one onto Eq. (49)'s. The paper's own restatement of
    Eq. (49), its printed Eq. (7), carries a misprint (-2 coth(2 Gamma) where Schaetzel
    has -2 coth(Gamma)); the paper's Fig. 2 was computed with Schaetzel's form, so
    Eq. (49) is the reference and Eq. (7) is not coded from.

    THE RUN DURATION NO LONGER CANCELS. Eq. (9) replaces Eq. (49)'s 1/M by 1/(M - k),
    the effective number of products at lag k (p. 29378), and on a channel of sample
    time Delta_t_k that is Delta_t_k / (T - tau_k). The tau_k in it differs per channel,
    so T drops out of the RELATIVE shape only when T >> tau_max. Given `duration_s`, the
    prefactor is Delta_t_k / (T - tau_k); without one it is Delta_t_k, the T -> infinity
    form with the common 1/T dropped, exactly the convention `_schatzel_sd` uses for its
    1/M. Either way the return is a relative shape in arbitrary units. A lag at or past
    the end of the run has no products and is refused, never returned as an infinite or
    negative variance.

    Per channel, Gamma_k = gamma_bar * Delta_t_k, k = tau_k / Delta_t_k and
    n_bar_k = mean_count_rate_cps * Delta_t_k, with beta the same on every stage -- the
    multi-tau substitution of Biganzoli and Ferri (2018, p. 29381). k enters Eq. (9) as
    a whole lag index of at least 1 (tau_k = k Delta_t_k, their Eq. (2), p. 29376), which
    a multi-tau axis satisfies, so a k within `_BIGANZOLI_LAG_INDEX_RTOL` of an integer --
    the rounding a written delay carries -- is evaluated AT that integer. Any other k on a
    channel with tau_k > 0 is REFUSED before anything is evaluated, naming the channel's
    lag, sample time and index: off the integers the expression is not a variance (near
    k = 1 its S5 group grows as (k - 1) e^Gamma, so a k below 1 can give an SD inflated
    without limit, and one just above 1 a collapsed one, both finite and positive). A
    sample time that does not divide its lag a whole number of times contradicts it. The
    axis stand-in (`_channel_sample_times`) always gives a whole index, so only a stated
    sample time can reach the refusal. A tau = 0 channel (k = 0 exactly) is evaluated as
    it is. The estimator it describes
    uses the symmetric normalization of their Eqs. (2)-(3) (p. 29376). That premise is
    inherited from Eq. (49) and disclosed here, not chosen: no export records which
    normalization the correlator applied.

    CANONICAL-METHOD ANCHOR: the canonical form. Eq. (9) is the exact-in-Gamma variance
    and supersedes Eq. (49) outside Gamma << 1. Three named practicality divergences,
    each a limit or a rewriting of Eq. (9), none a different model:
    (1) the CLASSICAL LIMIT when `mean_count_rate_cps` is None -- the two n_bar brackets
        are dropped, as `_schatzel_sd` drops Eq. (49)'s;
    (2) the T -> INFINITY FORM when `duration_s` is None -- 1/(M - k) taken as 1/M, the
        common 1/T dropped;
    (3) the LARGE-GAMMA ASYMPTOTIC BRANCH above `_BIGANZOLI_ASYMPTOTE_GAMMA`. Direct
        evaluation leaves float range near Gamma ~ 120, where sinh^6(Gamma) overflows,
        and a correlator's coarsest blocks reach past that on slow samples. Above the
        switch every S_n is taken as e^(n Gamma) / (2 Gamma)^n and every coth as 1 plus
        its leading exponential; the factors dropped are 1 to below float resolution
        there. Each surviving exponential is merged with the E2 or E4 it multiplies into
        one exp of an integer multiple of Gamma, and the classical braces are carried
        scaled by Gamma^2, since they fall as chi0^4 ~ 1/Gamma^2 and the variance would
        otherwise underflow past Gamma ~ 1e154 while the SD stays representable. The
        switch sits at Gamma = 30; measured 2026-09-11, the two forms agree there with
        each other and with Eq. (9) in 250-digit arithmetic to about 1e-15 relative on
        the variance, on both sides (the constant's comment carries the figures).

    Below `_BIGANZOLI_CHI0_SERIES_GAMMA`, chi0^2 is summed from its Taylor series,
    1 - 2 Gamma / 3 + Gamma^2 / 3 - ..., because the closed form cancels to a relative
    error ~ eps / Gamma and reaches zero or below near Gamma ~ 1e-16. Every coth - 1 is
    formed as 2 / expm1(2 x), and k - 2 + coth(Gamma) as (k - 1) + (coth(Gamma) - 1), so
    the k = 1 channel does not cancel.

    Raises `_Eq9Refused`, a ValueError carrying the refusal's cause key, naming the input
    to change, unless Gamma_k, n_bar_k (when a rate is given) and T - tau_k (when a
    duration is given) are finite and positive on every channel, every channel with
    tau_k > 0 has a whole lag index of at least 1, and the SD returned is finite and
    positive on every channel.
    """
    tau, dt = np.broadcast_arrays(np.asarray(tau, dtype=float),
                                  np.asarray(sample_times, dtype=float))
    gamma_dt = gamma_bar * dt
    # Guarded on the COMPUTED product, as in `_schatzel_sd`: an underflowed Gamma divides
    # by zero in every coth, and no single input is to blame for the product.
    if not (np.all(np.isfinite(gamma_dt)) and np.min(gamma_dt) > 0):
        raise _Eq9Refused(
            'decay_per_sample_time',
            f"the Biganzoli-Ferri variance needs a positive finite decay-per-sample-time "
            f"gamma_bar * Delta_t on every channel; the smallest is "
            f"{float(np.min(gamma_dt))!r} for gamma_bar = {gamma_bar!r} s^-1."
        )
    # Eq. (9) is a function of an INTEGER lag index, and k = tau / Delta_t comes from two
    # floats a file or a layout wrote: a k one rounding from an integer IS that integer.
    # Snapped here, where k is formed, so no second predicate stands beside it (owner-
    # ruled 2026-09-11, this comment its normative home).
    with np.errstate(over='ignore', invalid='ignore'):
        k = tau / dt
        k_int, whole = _whole_lag_index(k)
    k = np.where(whole, k_int, k)
    # What the snap leaves off the integers is refused before anything is evaluated: Eq. (9)
    # is written on tau_k = k Delta_t with k a whole number of at least 1, and off that
    # premise the expression is not a variance. A tau = 0 channel (k = 0 exactly) is exempt
    # and evaluated as it always was: whether Eq. (9) should weight a zero lag at all is a
    # separate question this refusal does not settle.
    off_index = (tau > 0) & ~whole
    if np.any(off_index):
        bad = int(np.argmax(off_index))
        raise _Eq9Refused(
            'lag_index',
            f"The Biganzoli–Ferri variance is written on a whole lag index k = tau / "
            f"Delta_t of at least 1 (Theory-and-Equations-Guide §3.2.9), and "
            f"the channel at lag tau = {float(tau[bad])!r} s has sample time Delta_t = "
            f"{float(dt[bad])!r} s, so k = {float(k[bad])!r}: the sample time contradicts "
            f"its lag. Give each channel the sample time its correlator accumulated it over, "
            f"which divides its lag a whole number of times."
        )
    if duration_s is None:
        inv_products = dt
    else:
        # The guard names the prefactor the variance multiplies, not only T - tau_k: a
        # positive but subnormal remainder still overflows dt / (T - tau_k).
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            remaining = duration_s - tau
            inv_products = dt / remaining
        usable = (np.isfinite(remaining) & (remaining > 0)
                  & np.isfinite(inv_products))
        if not (math.isfinite(duration_s) and duration_s > 0 and np.all(usable)):
            bad = int(np.argmin(usable))
            raise _Eq9Refused(
                'run_duration',
                f"duration_s = {duration_s!r} s leaves no products at lag tau = "
                f"{float(tau[bad])!r} s: the Biganzoli–Ferri variance's 1/(M - k) = "
                f"Delta_t / (T - tau) needs a finite run duration longer than every lag "
                f"(Theory-and-Equations-Guide §3.2.9). Give the measurement's run "
                f"duration, or None for the T -> infinity form."
            )
    nbar = None if mean_count_rate_cps is None else mean_count_rate_cps * dt

    asymptotic = gamma_dt > _BIGANZOLI_ASYMPTOTE_GAMMA
    direct = ~asymptotic
    root_braces = np.empty(gamma_dt.shape)
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        if np.any(direct):
            root_braces[direct] = np.sqrt(_biganzoli_direct_braces(
                k[direct], gamma_dt[direct], beta,
                None if nbar is None else nbar[direct]))
        if np.any(asymptotic):
            root_braces[asymptotic] = _biganzoli_asymptotic_root_braces(
                k[asymptotic], gamma_dt[asymptotic], beta,
                None if nbar is None else nbar[asymptotic])
        # The square roots are taken apart so neither product can underflow on the way.
        sd = beta * np.sqrt(inv_products) * root_braces

    if nbar is not None:
        # Two-sided, as in `_schatzel_sd`: a huge rate is harmless (the shot brackets
        # vanish), a tiny one sends 1/n_bar^2 to inf long before n_bar underflows. The
        # predicate names the shot terms themselves, which is what the message blames: an
        # SD left non-finite by anything else (a negative beta) is the guard below's.
        with np.errstate(over='ignore', divide='ignore'):
            inv_nbar2 = 1.0 / (nbar * nbar)
        if not (np.all(np.isfinite(nbar)) and np.min(nbar) > 0
                and np.all(np.isfinite(inv_nbar2))):
            raise _Eq9Refused(
                'count_rate',
                f"mean_count_rate_cps = {mean_count_rate_cps!r} gives mean counts per "
                f"sample time that the Biganzoli–Ferri variance's shot-noise terms cannot "
                f"use (Theory-and-Equations-Guide §3.2.9): the smallest "
                f"n_bar over the channels is {float(np.min(nbar))!r}, and the terms in "
                f"1/n_bar and 1/n_bar^2 must stay finite on every channel. Use a "
                f"realistic photon count rate (a real file's is tens to hundreds of kHz)."
            )
    if not (np.all(np.isfinite(sd)) and np.min(sd) > 0):
        bad = int(np.argmin(np.isfinite(sd) & (sd > 0)))
        raise _Eq9Refused(
            'channel_sd',
            f"The Biganzoli–Ferri variance (Theory-and-Equations-Guide §3.2.8) gives no "
            f"finite positive SD at lag tau = {float(tau[bad])!r} s "
            f"(Gamma_k = {float(gamma_dt[bad])!r}, k = {float(k[bad])!r}) for "
            f"gamma_bar = {gamma_bar!r} s^-1 and beta = {beta!r}. Check beta, the fitted "
            f"intercept (0 < beta <= 1); that gamma_bar * Delta_t is not so small that "
            f"1/Gamma_k overflows; and, with a count rate, that it is not so low that the "
            f"1/n_bar^2 term overflows."
        )
    return sd


def _whole_lag_index(k: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The nearest whole lag index to k, at least 1, and where k already is that index.

    "Already is" means within `_BIGANZOLI_LAG_INDEX_RTOL` of it, relative -- the rounding a
    written delay carries. One computation for the two sites that ask the question:
    `_biganzoli_sd` snaps such a k onto the integer and refuses the rest, and
    `_channel_sample_times` keeps the forward spacing wherever it already gives such a k.
    k = 0 is not whole (its nearest index is 1), and a non-finite k never is.
    """
    k_int = np.maximum(np.round(k), 1.0)
    return k_int, np.abs(k - k_int) <= _BIGANZOLI_LAG_INDEX_RTOL * k_int


def _biganzoli_direct_braces(k: np.ndarray, gamma: np.ndarray, beta: float,
                             nbar: Optional[np.ndarray]) -> np.ndarray:
    """The braces of Eq. (9), evaluated as printed, for `_biganzoli_sd` below its switch.

    `nbar=None` drops the two shot-noise brackets. The closed forms are rearranged only
    where float would cancel; the docstring of `_biganzoli_sd` says which and why.
    """
    series = gamma < _BIGANZOLI_CHI0_SERIES_GAMMA
    g_closed = np.maximum(gamma, _BIGANZOLI_CHI0_SERIES_GAMMA)
    chi2 = np.where(
        series,
        np.polynomial.polynomial.polyval(gamma, _BIGANZOLI_CHI0_SERIES),
        (2.0 * g_closed + np.expm1(-2.0 * g_closed)) / (2.0 * g_closed * g_closed))
    chi = np.sqrt(chi2)
    s1 = np.sinh(gamma) / gamma
    s2 = s1 * s1
    s3 = s2 * s1
    s4 = s2 * s2
    s5 = s4 * s1
    s6 = s4 * s2
    e2 = np.exp(-2.0 * gamma * k)
    e4 = np.exp(-4.0 * gamma * k)
    coth_g_m1 = 2.0 / np.expm1(2.0 * gamma)           # coth(Gamma) - 1
    coth_2g_m1 = 2.0 / np.expm1(4.0 * gamma)          # coth(2 Gamma) - 1
    a = 1.0 + 2.0 * beta
    c = (2.0 * k + 4.0 * beta * k - 2.0 - 8.0 * beta
         + 4.0 * beta * chi2 - 16.0 * beta * chi)
    braces = (
        2.0 * s2 * chi2 * e4
        + 4.0 * beta * s2 * chi2 * e2
        + 8.0 * beta * s3 * chi * e4
        + s4 * (coth_2g_m1 + 4.0 * beta * e2 * coth_g_m1
                + e4 * (a * (1.0 + coth_2g_m1) + c))
        + 4.0 * beta * s6 * e4 * coth_g_m1
        - 8.0 * beta * s5 * e4 * ((k - 1.0) + coth_g_m1)
        + chi2 * chi2
    )
    if nbar is not None:
        braces = braces + (
            (2.0 / (beta * nbar)) * (chi2 + s2 * e2 * (e2 + 2.0 * beta * chi)
                                     + 2.0 * beta * s3 * e4 - 2.0 * beta * s4 * e4)
            + (1.0 / (beta * beta * nbar * nbar)) * (1.0 + beta * s2 * e2)
        )
    return braces


def _biganzoli_asymptotic_root_braces(k: np.ndarray, gamma: np.ndarray, beta: float,
                                      nbar: Optional[np.ndarray]) -> np.ndarray:
    """Square root of Eq. (9)'s braces in the large-Gamma form, for `_biganzoli_sd`.

    With u = 1/Gamma: S_n -> e^(n Gamma) u^n / 2^n, coth(Gamma) - 1 -> 2 e^(-2 Gamma),
    coth(2 Gamma) - 1 -> 2 e^(-4 Gamma), coth(2 Gamma) -> 1 inside a coth(2 Gamma) + c,
    and Gamma chi0^2 -> 1 - u/2. Every exponential is merged with the E2 or E4 it
    multiplies. The classical braces are formed times Gamma^2 and square-rooted before
    the 1/Gamma is put back; the shot-noise brackets, which do not fall with Gamma, are
    added in quadrature unscaled. `nbar=None` drops them.
    """
    u = 1.0 / gamma
    u2 = u * u
    q = 1.0 - 0.5 * u                                  # Gamma * chi0^2
    chi = np.sqrt(u * q)
    x24 = np.exp((2.0 - 4.0 * k) * gamma)              # S2 E4 without its u^2 / 4
    x22 = np.exp((2.0 - 2.0 * k) * gamma)              # S2 E2, S4 E2 (coth(Gamma) - 1)
    x34 = np.exp((3.0 - 4.0 * k) * gamma)              # S3 E4, S5 E4 (coth(Gamma) - 1)
    x44 = np.exp((4.0 - 4.0 * k) * gamma)              # S4 E4, S6 E4 (coth(Gamma) - 1)
    km1 = k - 1.0
    # (k - 1) S5 E4: zero on k = 1 exactly, where exp(Gamma) alone would overflow.
    x54 = np.where(km1 == 0.0, 0.0, km1 * np.exp((5.0 - 4.0 * k) * gamma))
    a = 1.0 + 2.0 * beta
    c = (2.0 * k + 4.0 * beta * k - 2.0 - 8.0 * beta
         + 4.0 * beta * u * q - 16.0 * beta * chi)
    classical_scaled = (
        0.5 * u * q * x24
        + beta * u * q * x22
        + beta * u * chi * x34
        + 0.125 * u2
        + 0.5 * beta * u2 * x22
        + 0.0625 * u2 * (a + c) * x44
        + 0.125 * beta * u2 * u2 * x44
        - 0.25 * beta * u2 * u * (x54 + 2.0 * x34)
        + q * q
    )
    root = np.sqrt(classical_scaled) * u
    if nbar is None:
        return root
    shot = (
        (2.0 / (beta * nbar)) * (u * q + 0.25 * u2 * (x24 + 2.0 * beta * chi * x22)
                                 + 0.25 * beta * u2 * u * x34
                                 - 0.125 * beta * u2 * u2 * x44)
        + (1.0 / (beta * beta * nbar * nbar)) * (1.0 + 0.25 * beta * u2 * x22)
    )
    return np.hypot(root, np.sqrt(shot))


#: How far from a half-integer, relative to it, a lag-to-spacing ratio still counts as that
#: half-integer in `_channel_sample_times`, so a tie rounds up on every channel. The written
#: precision measured on the committed delay axes (2026-09-14: ALV rounding to 7.7e-5,
#: Brookhaven CSV ties to 9.2e-5, the Panta's to 1.5e-6). Its own constant, not
#: `_BIGANZOLI_LAG_INDEX_RTOL`, which is about evaluating Eq. (9) AT an integer (owner-ruled
#: 2026-09-14). Wider than needed is harmless: a ratio that near a tie has no preferred
#: direction, and above k = 1 / (2 * this), where it covers every ratio, rounding up moves
#: Delta_t by at most 1 / k.
_SAMPLE_TIME_HALF_SNAP_RTOL = 1e-4


def _channel_sample_times(tau: np.ndarray) -> np.ndarray:
    """Per-channel sample time for an axis that does not carry one explicitly.

    On a multi-tau axis Delta_t_k is a property of the block and comes from
    `MultiTauAxis`; an axis that states none gets a stand-in read off its lags, which
    Eq. (9) (`_biganzoli_sd`) then evaluates at the lag index k = tau_k / Delta_t_k. The
    rule has two halves.

    WHERE THE LAGS ARE WHOLE MULTIPLES, THE FORWARD SPACING. Each channel's step to the one
    after it, the last channel borrowing the one before it, is kept wherever it already
    gives a whole lag index (`_whole_lag_index`). That is the reading under which Schaetzel's printed
    multi-tau layout (Sec. 3.2, p. 300; what `multi_tau_lag_axis` builds) comes back
    exactly on every channel: block j spans the lags L*Delta_t_j ... (2L-1)*Delta_t_j, so
    block j begins one fine step after block j-1 ends, at L*Delta_t_j = 2L*Delta_t_(j-1)
    -- no lag is shared -- and that channel's forward step is its own block's sample time.
    The backward spacing reads half of it there (owner-ruled 2026-09-08: follow the paper;
    the engine's `analysis.dls._common.channel_sample_times` is the ruling's home).

    ELSEWHERE, THE NEAREST SAMPLE TIME WITH A WHOLE LAG INDEX. On a plain log or linear
    grid, or a real staircase whose written lags carry rounding or sit offset from their
    block's multiples (most channels of every committed real axis), the forward spacing s_k
    gives a fractional k, on which Eq. (9) is not a variance. There Delta_t_k =
    tau_k / n_k, with n_k = tau_k / s_k rounded half up and at least 1: a lag shorter than
    its spacing takes Delta_t_k = tau_k. A ratio within `_SAMPLE_TIME_HALF_SNAP_RTOL` of a
    half-integer counts as that half-integer first, so a tie rounds up on every channel,
    exact or carrying float noise either side; above k = 5000 that relative window covers
    every ratio and the ratio simply rounds up. This is a stand-in with no source -- no
    source evaluates the variance off the integers or says what sample time to use when
    a file states none -- chosen so Eq. (9) stays on the premise it is written on, not as
    a more accurate Delta_t (owner-ruled 2026-09-14; `channel_sample_times`'s docstring in
    `analysis.dls._common` is the ruling's home).

    A tau = 0 channel keeps its forward spacing, as does a channel whose ratio is not
    finite. Only an infinite lag gets there, and the data model refuses one: on a strictly
    increasing finite axis no step is zero (gradual underflow), and tau / step stays below
    2^53. A one-channel axis returns Delta_t = tau (k = 1).
    Used by the four per-channel noise models on non-staircase axes, and by the engine,
    so a planted SD and the weight built from it are one computation.
    """
    tau = np.asarray(tau, dtype=float)
    if tau.size < 2:
        return tau.copy()
    step = np.diff(tau)
    spacing = np.concatenate((step, [step[-1]]))
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        ratio = tau / spacing
        _nearest, whole = _whole_lag_index(ratio)
        half = np.floor(ratio) + 0.5
        tied = np.abs(ratio - half) <= _SAMPLE_TIME_HALF_SNAP_RTOL * half
        n = np.maximum(np.floor(np.where(tied, half, ratio) + 0.5), 1.0)
        rounded = tau / n
    replace = (tau > 0) & np.isfinite(ratio) & ~whole
    return np.where(replace, rounded, spacing)


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

    The ground-truth moment fields are exact amplitude-weighted moments of the
    DISCRETISED decay-rate spectrum this curve was actually built from -- the
    (gammas, amps) pairs summed into `signal` -- so they hold for any population
    shape, and are never a closed form for one assumed distribution.

    TWO THINGS THAT SENTENCE IS CAREFUL ABOUT, both of which have already bitten:

    1. "the discretised spectrum", NOT the nominal distribution asked for. A
       `spread_cv` population is sampled on a grid truncated at +/-3 sigma_ln (~99.7 %
       of the mass); that truncation is negligible at order 2 and not above it.
       Measured against the analytic lognormal over spread_cv 0.05..0.5: mu2 off by
       2.4..5.0 %, mu3 by 12..22 %, mu4 by 10..36 %, and kappa4 -- a difference of two
       nearly-equal quantities, so the error is amplified -- by 47..410 %, including a
       SIGN FLIP below spread_cv ~ 0.15. None of that makes these fields wrong: an
       ideal cumulant fit of THIS curve recovers THESE numbers, which is exactly what
       an oracle needs. It does mean they must not be checked against a closed form for
       the nominal distribution. For a fixture whose moments must match a named
       distribution to high order, plant it explicitly as zero-width populations --
       `tests/test_utilities.py::_schulz_populations` is the worked example.
    2. Only the first three are recoverable by ANY cumulant fit. The fourth order is
       ARM-SPECIFIC: a linear (`mailer2015` Eq. 12) fit's quartic coefficient gives
       kappa4 and never mu4; a nonlinear (Eq. 13) fit gives mu4 and never kappa4. Both
       are planted so each arm can be checked against the one it actually returns.
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
    # Third and fourth order. The block header says "as a cumulant fit sees it", so the
    # moment-vs-cumulant split has to be spelled out rather than left to the reader:
    # `mailer2015`'s terminology note (printed p. 6) states that the cumulants differ
    # from the central moments only at ORDER 4 AND HIGHER. Hence mu3 is both, while the
    # fourth order needs two fields because the two arms return different quantities --
    # Mailer Eq. 13 (nonlinear) fits mu4, Eq. 12 (linear) gives its quartic coefficient
    # as (mu4 - 3 mu2^2). Planting both is what lets each arm be checked against the
    # quantity it actually produces instead of one standing in for the other.
    mu3_s_inv3: float                  # third central moment == third cumulant
    mu4_s_inv4: float                  # fourth CENTRAL MOMENT (Mailer Eq. 13's parameter)
    kappa4_s_inv4: float               # fourth CUMULANT, mu4 - 3 mu2^2 (Mailer Eq. 12's c4)
    d_eff_m2_s: float                  # effective diffusion coefficient
    rh_eff_nm: float                   # effective (apparent) hydrodynamic radius

    # Multi-tau / per-channel-noise context. Defaulted so every existing caller and
    # every existing constructed instance is unaffected; the first two are what a
    # variance-weighting study reads back.
    noise_model: str = 'constant'      # 'constant' | 'schatzel_classical' | 'schatzel_full'
    #                                    | 'biganzoli_classical' | 'biganzoli_full'
    sample_times_s: Optional[np.ndarray] = None    # per-channel Delta_t_k
    noise_sd: Optional[np.ndarray] = None          # per-channel sigma actually drawn
    duplicate_lag_indices: Optional[np.ndarray] = None
    # The flat g2-1 offset planted in `signal` (0.0 = none). Ground truth for the
    # floating-baseline fits: what they must recover is this number.
    baseline: float = 0.0
    # The opt-in noise knobs, echoed so a generated curve says how its noise was made:
    # the count rate the shot-noise terms consumed (None unless noise_model is one of
    # the full models), the run duration Eq. (9)'s 1/(M - k) consumed (None unless a
    # 'biganzoli_*' model was given one), and the lag-1 channel-to-channel correlation
    # planted in the draw (0.0 = independent channels, the historical behaviour).
    mean_count_rate_cps: Optional[float] = None
    duration_s: Optional[float] = None
    noise_correlation: float = 0.0


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
    baseline: float = 0.0,
    mean_count_rate_cps: Optional[float] = None,
    noise_correlation: float = 0.0,
    duration_s: Optional[float] = None,
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

        TWO WAYS THIS NOISE IS NOT WHAT AN INSTRUMENT PRODUCES, EACH NOW
        OPT-IN -- the defaults still produce neither, so read both before
        drawing a correctness conclusion from generated data. Both were measured
        against the committed real replicate groups.

        (1) The draws are INDEPENDENT PER CHANNEL unless `noise_correlation` is
        set. Real correlogram noise is correlated between neighbouring lag
        channels: measured lag-1 correlation is +0.27 to +0.43, against -0.019
        for an independent draw. The bracket's two ends are two
        different nuisance corrections and neither is a point estimate -- the
        lower end removes per-run amplitude AND per-run decay-rate drift, which
        also removes some real noise, and at n = 3 it over-corrects; the clean
        ALV-only figure is +0.304. Correlated noise is the kind a regularized
        fit absorbs: Peters (1993, ch. 3) measured distorted CONTIN
        distributions on single real measurements against their mean (Figs.
        3.14-3.19, printed pp. 165-168) and found that added synthetic photon
        noise left "almost no visible difference" (p. 166; his Fig. 3.20,
        p. 168). The effect on the smoothing-parameter selection itself is his
        stated expectation, not a measurement: such algorithms "assume
        independent normally distributed noise to ensure correct determination
        of the optimum regularizer strength" (p. 174). At the default
        `noise_correlation=0.0` this generator cannot exercise that failure mode
        at all, and a validation run there runs under the one condition that
        makes the pathology vanish.

        (2) The per-channel SD SPAN is far too narrow over the channels that
        carry fit weight unless `noise_model` is one of the full models: 1.9x for the
        classical limit against 2.5-22.6x (ALV) and 34-145x (Brookhaven)
        measured. The cause is the missing shot-noise terms -- see
        `_schatzel_sd`, and `mean_count_rate_cps` for what turning them on buys.

        The Theory-and-Equations-Guide, "The Noise on a Correlogram" states
        both limits with their figures, for the reader who does not have this
        file.
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
        'schatzel_full': the same shape with Eq. (49)'s two shot-noise term groups
        restored. It REQUIRES `mean_count_rate_cps` -- there is no honest default for a
        photon count rate -- and it is the arm that reproduces the per-channel SD span
        real files show; the models without '_full' refuse that argument. See
        `_schatzel_sd`.
        'biganzoli_classical' / 'biganzoli_full': the per-channel SD with the SHAPE of
        Biganzoli and Ferri (2018) Eq. (9), printed p. 29378, which is exact in
        Gamma = Delta_t / tau_c where Eq. (49) is a closed form for Gamma << 1 -- the
        difference that matters in a multi-tau axis's coarsest blocks. Anchored on the
        first channel exactly as the Schaetzel models are, and evaluated by
        `_biganzoli_sd`, the same function a fit weight built on Eq. (9) evaluates, so
        the planted SD and such a weight are one computation. 'biganzoli_full' REQUIRES
        `mean_count_rate_cps` as 'schatzel_full' does, and 'biganzoli_classical', its
        classical limit, refuses it. Both consume `duration_s`.

        Why the shape matters: it is not monotonic. The per-channel noise falls
        through the correlogram's decay, then TURNS UP, by a different mechanism in
        each family. Under the two Schaetzel models it rises through the coarse
        blocks, where 1/M_k = Delta_t_k / T grows with the block's sample time and
        few products are left to average. Under the two exact models it keeps
        falling and rises only in the last stage, where the lag times become
        comparable with the measuring time and Eq. (9)'s 1/(M - k) factor grows
        (printed p. 29378): the upturn Biganzoli and Ferri (2018, printed p. 29382)
        report, and absent when the run is long against the longest lag. Over a 22-octave ALV-like axis
        it spans a factor of ~20. That span is RELATIVE -- the absolute scale is the
        caller's noise_level -- so Peters' measured per-channel range is not a check
        on it; what the sources corroborate is the shape, not the magnitude.
        Weighting the accumulation count alone would instead ramp
        monotonically by ~1400x and bury the longest lags in noise larger than the
        signal.

        Under the two Schaetzel models the run duration does not appear: it is common
        to every channel and cancels out of the relative shape (see
        `_schatzel_classical_sd`). Under the two exact models it does not cancel; see
        `duration_s`.

        Orthogonal to `grid`: on a 'log' or 'linear' axis Delta_t_k is the axis
        stand-in `_channel_sample_times` gives, the nearest sample time with a whole lag
        index (a lag shorter than its channel spacing takes its own lag), which is also
        what the engine's weight reads off an axis that states no sample times.
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
    baseline : float
        A FLAT offset B added to g2(tau)-1, i.e. the column becomes
        ``B + beta*|g1|^2`` (default 0.0, which changes nothing). This is the residual
        instrument baseline a real correlator leaves behind when its measured baseline
        does not equal its calculated one -- the condition the floating-baseline
        cumulant fit exists to absorb, and the one the KWW data-form decay criterion is
        sensitive to. Named ``baseline`` (never ``offset``) to match the fixture it
        replaces, ``tests/fixtures/synthetic_dls.py::monomodal``, and to stay clear of
        ``build_count_rate_trace``'s ``offset_*`` family, which is a count-rate STEP on
        a different builder.

        Applied on the g2-1 scale BEFORE the output-form mapping, so ``'g2'`` carries it
        too (``1 + B + beta|g1|^2``). ``output_form='g1'`` with a non-zero baseline is
        REFUSED rather than silently ignored: a flat offset on the intensity correlation
        has no field-correlation equivalent, and handing back a clean g1 curve labelled
        as carrying an offset is the kind of quiet no-op a generator must not do.
        The planted value is echoed on the result as ``baseline``, so a fit's recovered
        B has a ground truth to be checked against.
    mean_count_rate_cps : float, optional
        Mean photon count rate in counts per second, the one extra measured input
        Eq. (49)'s shot-noise groups need (per channel they consume
        ``mean_count_rate_cps * Delta_t_k``, the mean counts per sample time). REQUIRED
        by the two full models, ``noise_model='schatzel_full'`` and
        ``'biganzoli_full'``, and REFUSED by the other three rather
        than silently ignored -- the same refusal ``baseline`` carries against
        ``output_form='g1'``, and for the same reason: a knob the generator accepts and
        then does not use is a quiet no-op the caller has no way to notice.

        What it buys: with a real file's vendor count rate the full model reproduces
        that file's measured per-channel SD to within the sample-SD estimator's own
        noise floor on every one of the twelve committed replicate groups -- seven
        ALV angles at n = 10 and five Brookhaven fractions at n = 3, the dust-affected
        ALV 44 degree run excluded by identity -- where the classical limit exceeds
        that floor on seven of them -- four ALV angles and three Brookhaven fractions,
        two of those by less than the floor estimator's own spread (measured
        2026-09-07; the fidelity test in
        `tests/test_utilities.py` is the record). Real files carry tens to hundreds of
        kHz, and the shot terms dominate the span at the low end of that range.
    noise_correlation : float
        Lag-1 correlation coefficient rho of the noise between neighbouring channels,
        strictly inside (-1, 1). 0.0 (default) draws every channel independently,
        exactly as before. A non-zero value draws a stationary AR(1) sequence in CHANNEL
        INDEX with unit variance per channel, so the per-channel SD is untouched and
        only the neighbour correlation moves; the planted value is echoed on the result.

        A FLAGGED CONVENIENCE SIMPLIFICATION, not the canonical model. An AR(1) in
        channel index reproduces the measured lag-1 statistic (+0.27 to +0.43 on real
        data; +0.30 is the clean ALV figure) and NOTHING ELSE about the covariance --
        neither the multi-tau block structure nor the coherence-time correlation that
        reaches beyond one channel. The canonical alternative is the full covariance:
        Schaetzel (1990) Eq. (48), printed p. 296, generalised across multi-tau stages
        by Peters (1993, sec. 3.3.1, Eqs. 3.9 and 3.11), drawn through a Cholesky factor
        of that matrix. It is not done here for two reasons, both stated so the choice
        is visible: those equations must be re-read from the source before they can be
        implemented faithfully, and the noise a generator plants is a far weaker claim
        than a variance used to weight or whiten a real fit. Anything that weights real
        data starts from the full covariance, never from this.
    duration_s : float, optional
        The run duration T, in seconds, consumed only by the two exact models,
        ``noise_model='biganzoli_classical'`` and ``'biganzoli_full'``. Eq. (9)'s
        1/(M - k) = Delta_t_k / (T - tau_k) differs per channel, so T stays in the
        relative shape unless T is much longer than the last lag. None (default) is
        the T -> infinity form. REFUSED with every other model, as
        ``mean_count_rate_cps`` is: Eq. (49)'s relative shape cancels T. A duration
        that the axis's lags reach or pass leaves a channel with no products and is
        refused by `_biganzoli_sd`, naming ``duration_s``.

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
    if noise_model not in ('constant', 'schatzel_classical', 'schatzel_full',
                           'biganzoli_classical', 'biganzoli_full'):
        raise ValueError(
            f"noise_model must be 'constant', 'schatzel_classical', 'schatzel_full', "
            f"'biganzoli_classical' or 'biganzoli_full', got {noise_model!r}.")
    if output_form not in ('g2m1', 'g2', 'g1'):
        raise ValueError(
            f"output_form must be 'g2m1', 'g2', or 'g1', got {output_form!r}."
        )
    if not math.isfinite(baseline):
        raise ValueError(f"baseline must be a finite number, got {baseline!r}.")
    if baseline != 0.0 and output_form == 'g1':
        raise ValueError(
            f"baseline={baseline!r} is defined on the g2-1 scale and has no field-"
            f"correlation equivalent, so it cannot be emitted with output_form='g1'; "
            f"use 'g2m1' or 'g2', or leave baseline at 0.0."
        )
    if (noise_model in ('schatzel_full', 'biganzoli_full')
            and not (mean_count_rate_cps is not None
                     and math.isfinite(mean_count_rate_cps)
                     and mean_count_rate_cps > 0)):
        raise ValueError(
            f"noise_model={noise_model!r} evaluates its equation's shot-noise terms, "
            f"which need a positive finite mean_count_rate_cps (counts per second); got "
            f"{mean_count_rate_cps!r}. There is no honest default for a photon count "
            f"rate -- take it from the file you are imitating, or use noise_model="
            f"{noise_model.replace('_full', '_classical')!r}, its infinite-count-rate "
            f"limit."
        )
    if mean_count_rate_cps is not None and noise_model not in ('schatzel_full',
                                                               'biganzoli_full'):
        raise ValueError(
            f"mean_count_rate_cps={mean_count_rate_cps!r} is consumed only by "
            f"the full models ('schatzel_full', 'biganzoli_full'), so passing it with "
            f"noise_model={noise_model!r} would leave it unused. Refused rather than "
            f"silently ignored, exactly as a baseline is refused with output_form='g1': "
            f"a knob the generator accepts and does not use is a quiet no-op the caller "
            f"has no way to notice."
        )
    if not (math.isfinite(noise_correlation) and -1.0 < noise_correlation < 1.0):
        raise ValueError(
            f"noise_correlation is a lag-1 correlation coefficient and must be finite "
            f"and strictly inside (-1, 1), got {noise_correlation!r}. The endpoints are "
            f"excluded because the AR(1) innovation scale sqrt(1 - rho^2) collapses to "
            f"zero there, leaving one draw repeated across every channel."
        )
    # Only presence is checked here: a duration the exact models cannot use is refused
    # by `_biganzoli_sd`, the one computation that consumes it.
    if duration_s is not None and noise_model not in ('biganzoli_classical',
                                                      'biganzoli_full'):
        raise ValueError(
            f"duration_s={duration_s!r} is consumed only by the exact-in-Gamma models "
            f"('biganzoli_classical', 'biganzoli_full'), so passing it with noise_model="
            f"{noise_model!r} would leave it unused: the Schätzel variance's relative "
            f"shape cancels the run duration (Theory-and-Equations-Guide §2.3). Refused rather than silently ignored, exactly as "
            f"mean_count_rate_cps is refused with a model that does not consume it."
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
    #                      analysis/dls/_common.py::d_from_gamma for the full note.
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
    # The flat instrument baseline rides on the INTENSITY correlation, so it is added
    # here, on the g2-1 scale, before the output form is chosen -- which is why 'g2'
    # carries it and 'g1' (refused above) cannot.
    g2m1 = beta * g1 ** 2 + baseline

    if output_form == 'g2m1':
        signal = g2m1.copy()
    elif output_form == 'g2':
        signal = 1.0 + g2m1
    else:  # 'g1'
        signal = g1.copy()

    # Per-channel noise SD. 'constant' broadcasts one value (the historical
    # behaviour); the two Schaetzel models take the per-channel SHAPE of Eq. (49) and
    # the two Biganzoli models that of Eq. (9) -- each its classical limit, or with the
    # shot-noise terms at the given count rate -- and all four rescale it so noise_level
    # remains the first channel's SD. See the docstring for what that models and, just
    # as importantly, what it does not.
    if noise_model != 'constant':
        if noise_model in ('biganzoli_classical', 'biganzoli_full'):
            shape = _biganzoli_sd(tau, sample_times, gamma_bar, beta,
                                  mean_count_rate_cps=mean_count_rate_cps,
                                  duration_s=duration_s)
        else:
            shape = _schatzel_sd(tau, sample_times, gamma_bar, beta,
                                 mean_count_rate_cps=mean_count_rate_cps)
        # Normalised on the FIRST channel, whose shape value is the largest of the
        # short-lag plateau, so noise_level keeps one meaning across all four models.
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
        if noise_correlation != 0.0:
            # Correlated channels: a stationary AR(1) in CHANNEL INDEX, unit variance
            # per channel by construction (e[0] is a unit draw and the innovation is
            # scaled by sqrt(1 - rho^2)), so sigma still sets the per-channel SD and
            # only the neighbour correlation changes -- corr(e_k, e_k+1) = rho exactly.
            # The recursion runs straight across multi-tau block boundaries: real
            # correlation does not reset at a sample-time doubling, and modelling how it
            # changes there needs the full covariance (see the noise_correlation docs).
            rho = float(noise_correlation)
            z = rng.standard_normal(signal.size)
            e = np.empty_like(z)
            e[0] = z[0]
            innovation = math.sqrt(1.0 - rho * rho)
            for ch in range(1, z.size):
                e[ch] = rho * e[ch - 1] + innovation * z[ch]
            signal = signal + sigma * e
        else:
            # rng.normal broadcasts scale elementwise, so one draw covers every model.
            signal = signal + rng.normal(0.0, sigma, size=signal.shape)

    # Ground-truth cumulants (amplitude-weighted moments of the decay rate).
    # gamma_bar is computed above, before the noise draw.
    mu2 = float(np.sum(amps * (gammas - gamma_bar) ** 2))
    # Same amplitude-weighted central-moment sum, two orders further. Computed from the
    # planted distribution itself rather than from a Schulz closed form (`mailer2015`
    # Eq. 18), because the generator plants ARBITRARY populations and that closed form
    # holds only for a Schulz: an oracle built on it would be approximate exactly where
    # the populations stop being Schulz-shaped. `tests/test_utilities.py` checks these
    # sums AGAINST Eq. 18 on a deliberately Schulz-shaped case, which is the right way
    # round -- the literature validates the computation, it does not define it.
    mu3 = float(np.sum(amps * (gammas - gamma_bar) ** 3))
    mu4 = float(np.sum(amps * (gammas - gamma_bar) ** 4))
    # `mu2 * mu2`, never `mu2 ** 2`: a Python-float `**` RAISES OverflowError on the
    # upper tail where `*` saturates to inf -- the same note as _gb2 just below.
    kappa4 = mu4 - 3.0 * mu2 * mu2
    # The guard is on the SQUARE, not on gamma_bar: the division squares it, so an
    # extreme-but-finite decay rate underflows the denominator to 0.0 while
    # `gamma_bar != 0` still passes. Same mis-sizing as the q^2 guard above.
    # (Written as `gamma_bar * gamma_bar`, never `** 2` -- a Python-float `**` RAISES
    # OverflowError on the other tail, inside the guard; see d_from_gamma's note.)
    _gb2 = gamma_bar * gamma_bar        # `*` not `**`: see d_from_gamma's note
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
        baseline=float(baseline),
        mean_count_rate_cps=(None if mean_count_rate_cps is None
                             else float(mean_count_rate_cps)),
        duration_s=None if duration_s is None else float(duration_s),
        noise_correlation=float(noise_correlation),
        gamma_bar_s_inv=gamma_bar,
        mu2_s_inv2=mu2,
        pdi=pdi,
        mu3_s_inv3=mu3,
        mu4_s_inv4=mu4,
        kappa4_s_inv4=kappa4,
        d_eff_m2_s=d_eff,
        rh_eff_nm=rh_eff_nm,
        noise_model=noise_model,
        sample_times_s=sample_times,
        noise_sd=sigma,
        duplicate_lag_indices=(axis.duplicate_indices if axis is not None
                               else np.array([], dtype=int)),
    )
