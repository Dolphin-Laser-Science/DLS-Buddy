"""
analysis/depolarization.py
==========================

Static depolarized light scattering (DPLS) analysis: the depolarization ratio,
the Cabannes isotropic/anisotropic split, and the optical-anisotropy parameter,
from a paired VV (polarized) and VH (depolarized) intensity measurement.

This is Phase 1 of the depolarized-light-scattering module (CLAUDE.md -> Planned
-> DPLS). It is the STATIC side: it works from time-averaged intensities, not
correlograms. The DYNAMIC side, rotational diffusion D_r from the VV/VH
correlogram decay rates (`assemble_ddls_rate_points`, `analyze_ddls`) and the shape
models built on D_t and D_r, sits further down this module.

What it computes
----------------
Given the polarized (VV) and depolarized (VH) scattered intensities measured with
VERTICALLY polarized incident light -- the geometry of essentially every modern
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
Two-letter subscripts are (incident, analyzer): VV = vertical incident + vertical
analyzer; VH = vertical incident + horizontal analyzer. This matches Sivokhin &
Kazantsev (2021) and physics/constants.py's Rayleigh-geometry codes. NOTE: Chu
(1991) writes the depolarized ratio as R_HV with the OPPOSITE letter order
(scattered, incident) -- it is the same physical quantity as our I_VH. We use the
(incident, analyzer) order throughout.

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
      vertically-polarized-incident depolarization correction; delta^2.
  Coumou, Mackor & Hijmans 1964 (natural-light Cabannes factor; via constants.py).
  Sivokhin & Kazantsev 2021 (toluene rho_v; rho_u <-> rho_v; via constants.py).

Design contract
---------------
Every function is PURE: it takes intensities, decay rates or correlograms (and
parameters) and returns a result object or a number. No plotting, no file I/O, no mutation of inputs.

Change history
--------------
2026-06-19  Phase 1: static depolarization. (depolarization.py v1)
            depolarization_ratio, analyze_depolarization (DepolarizationResult),
            isotropic_rayleigh_ratio. Validated headless against Coumou Table 3
            and Sivokhin & Kazantsev Table 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize

from analysis import dls
from analysis import uncertainty as unc
from core.data_models import DLSMeasurement
from core.note_registry import DECAY_BEARING_CODES
from core.notes import WHERE, WHERE_UNIT, Note, RefusedInputs, find_notes
from physics import constants as phys
from physics.constants import (
    depolarization_ratio_unpolarized,
    cabannes_isotropic_factor,
    optical_anisotropy_squared,
    stokes_einstein_rh,
    RHO_V_DEPOLARIZATION_LIMIT,
)


# ===========================================================================
# Result object
# ===========================================================================

@dataclass
class DepolarizationResult:
    """Static depolarization analysis from one VV/VH intensity pair.

    All quantities assume VERTICALLY polarized incident light (the modern default).
    `physically_valid` is False when rho_v falls outside [0, 3/4); in that case the
    derived quantities are clamped/NaN and `notes` holds the `rho_v_out_of_range`
    precondition, rather than the analysis raising -- a slightly-out-of-range rho_v is
    usually instrumental (stray depolarized light from polarizer/analyzer leakage,
    reflections, or an over-subtracted dark count), which is a data-quality flag, not a
    crash.
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
    physically_valid: bool             # 0 <= rho_v < 3/4
    notes: Tuple[Note, ...] = ()
    # uncertainty (None unless replicate SEs were supplied)
    rho_v_se: Optional[float] = None


# ===========================================================================
# Core analysis
# ===========================================================================

def depolarization_ratio(i_vv: float, i_vh: float, dark_count: float = 0.0) -> float:
    """Depolarization ratio rho_v = I_VH / I_VV for vertically polarized incident light.

    Parameters
    ----------
    i_vv : float
        Polarized (vertical analyzer) scattered intensity, vertical incident.
    i_vh : float
        Depolarized (horizontal analyzer) scattered intensity, vertical incident.
    dark_count : float, optional
        Detector dark count / baseline subtracted from BOTH intensities before the
        ratio (default 0). The depolarized channel is weak, so an un-subtracted
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
    # Finite as well as positive: an infinite I_VV passes a bare `> 0` and gives
    # rho_v = vh/inf = 0.0 -- which reads as a perfectly isotropic scatterer, and drives
    # the Cabannes factor to exactly 1 (no anisotropy correction) with no flag anywhere.
    if not (math.isfinite(vv) and vv > 0):
        raise ValueError(
            f"dark-subtracted I_VV must be a positive finite number, got {vv!r} "
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
    Assumes vertically polarized incident light.

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
    rho_v outside [0, 3/4) does not raise: it sets physically_valid=False, notes it
    (`rho_v_out_of_range`, with rho_v), and leaves delta^2 = nan (and the Cabannes
    factor still computed from the linear form, which stays finite). rho_v <= 0 means
    no resolvable depolarization; rho_v >= 3/4 is unphysical for an anisotropic
    scatterer -- at the limit itself the isotropic Rayleigh contribution vanishes
    (Cabannes factor f = 0) and delta^2 is undefined, and above it it signals
    instrumental stray depolarized light.
    """
    rho_v = depolarization_ratio(i_vv, i_vh, dark_count=dark_count)
    vv = i_vv - dark_count
    vh = i_vh - dark_count

    # Validity against the physical range [0, 3/4) for vertical incident light. The
    # limit itself is excluded: at rho_v = 3/4 the Cabannes factor is exactly 0
    # (R_iso vanishes) and delta^2 is undefined, so the limit is not a value the
    # method can meaningfully produce -- it takes the same branch as rho_v > 3/4.
    physically_valid = (0.0 <= rho_v < RHO_V_DEPOLARIZATION_LIMIT)
    # One precondition for every way out of the range -- at or above the limit, negative,
    # or not a number -- each case readable off rho_v itself.
    notes: Tuple[Note, ...] = () if physically_valid else (
        Note('rho_v_out_of_range', ('rho_v',),
             {'rho_v': rho_v, 'lo': 0.0, 'hi': RHO_V_DEPOLARIZATION_LIMIT}),)

    # Natural-light equivalent and optical anisotropy. Guard the closed forms,
    # which are only defined on the physical range, so an out-of-range rho_v
    # degrades gracefully to a flag instead of a raised exception.
    if 0.0 <= rho_v <= RHO_V_DEPOLARIZATION_LIMIT:
        rho_u = depolarization_ratio_unpolarized(rho_v)
    else:
        rho_u = float('nan')

    if 0.0 <= rho_v < RHO_V_DEPOLARIZATION_LIMIT:
        delta_sq = optical_anisotropy_squared(rho_v)
    else:
        delta_sq = float('nan')

    # Cabannes factor uses the linear form 1 - (4/3) rho_v, finite for any rho_v;
    # clamp the call to the valid range so it does not raise, but report the raw
    # (possibly >1 or <0) factor for transparency when invalid.
    if 0.0 <= rho_v <= RHO_V_DEPOLARIZATION_LIMIT:
        f_iso = cabannes_isotropic_factor(rho_v)
    else:
        f_iso = 1.0 - (4.0 / 3.0) * rho_v  # raw linear value, outside guaranteed [0,1]
    anisotropic_fraction = 1.0 - f_iso

    # Uncertainty: only when both intensity SEs are supplied (replicates). The
    # dark count is treated as exact here; rho_v = vh/vv with independent vh, vv.
    rho_v_se = unc.ratio_se(vh, i_vh_se, vv, i_vv_se)

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
        notes=notes,
        rho_v_se=rho_v_se,
    )


def isotropic_rayleigh_ratio(r_vv: float, rho_v: float) -> float:
    """Isotropic excess Rayleigh ratio from the measured VV ratio (Cabannes correction).

    R_iso = R_VV * (1 - (4/3) rho_v)    [vertically polarized incident light]

    This is the value the Zimm / Debye / Berry Mw-Rg-A2 analysis should use when the
    scatterer is optically anisotropic: it strips the depolarized contribution that
    otherwise inflates the apparent molecular weight (Mw_app = Mw (1 + (4/5) delta^2);
    Chu 1991 Eq. 8.4.8). For an isotropic scatterer rho_v = 0 and R_iso = R_VV.

    Parameters
    ----------
    r_vv : float
        Measured VV excess Rayleigh ratio (any units; the factor is dimensionless).
    rho_v : float
        Vertical-incident depolarization ratio, 0 <= rho_v <= 3/4.

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

# In the small-qL regime the FIELD (g1) decay rates are Gamma_VV = q^2 D_t and
# Gamma_VH = q^2 D_t + 6 D_r, so D_r = (Gamma_VH - Gamma_VV) / 6 and a multi-angle plot of
# Gamma_VV vs q^2 gives D_t. Sources, recovery strategy and uncertainty are on
# `analyze_ddls`. IMPORTANT: Gamma is the FIELD rate; a homodyne intensity correlogram
# decays at 2*Gamma.

# qL above which the single-exponential VH form starts to break (additional
# exponentials arise from intramolecular interference). Zero & Pecora 1982, p. 88: the
# single-exponential pair for qL <= 3, the additional amplitudes under 10% of the total
# up to about qL = 5; they cite Berne & Pecora 1976 for both the pair and the cause.
# Guarding at 3 rather than 5 is this program's conservative choice. A
# function-parameter default, not a system constant.
_DDLS_QL_SINGLE_EXP_LIMIT: float = 3.0

# The preconditions of a VH replicate's fit that bear on its decay rate: the decay-bearing
# roster less its two size-range codes. A VH rate carries rotation (Gamma_VH = q^2 D_t +
# 6 D_r), so the Stokes-Einstein radius a VH fit's size check reads is no size, and a
# check on it says nothing about the rate.
_VH_DECAY_BEARING_CODES: frozenset[str] = DECAY_BEARING_CODES - {'rh_above_ceiling',
                                                                'rh_below_floor'}


@dataclass
class DDLSRatePoint:
    """One angle's fitted FIELD decay rates for the polarized/depolarized pair.

    Built by `assemble_ddls_rate_points` from the VV and VH correlograms measured at
    the same angle (each fitted with the cumulant engine to its field rate Gamma).
    `analyze_ddls` consumes these; it never touches a correlogram or a fitter, so it
    is trivially testable from hand-built points.
    """
    angle_deg: float
    q_m_inv: float                 # scattering vector magnitude, m^-1
    gamma_vv_s_inv: float          # field decay rate Gamma_VV (g1), s^-1
    gamma_vh_s_inv: float          # field decay rate Gamma_VH (g1), s^-1
    # The precondition notes of each VV replicate whose rate entered `gamma_vv_s_inv`, one
    # tuple per replicate, as its fit carried them; () from a caller that fitted none.
    # D_t's and D_r's verdicts read them (`analyze_ddls`); they change no rate.
    vv_preconditions: Tuple[Tuple[Note, ...], ...] = ()
    # The same for each VH replicate whose rate entered `gamma_vh_s_inv`; D_r's verdict
    # reads them.
    vh_preconditions: Tuple[Tuple[Note, ...], ...] = ()


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
    d_r_se: Optional[float]            # SD/sqrt(n) across angles; None for n < 2 or zero spread
    rh_t_nm: float                     # Stokes-Einstein radius from D_t
    # Rh_t ∝ D_t^-1: the fractional error carries over exactly (sigma_Rh/Rh = sigma_D/D),
    # so this is the already-validated reciprocal pass-through of d_t_se (power_law_se,
    # exponent -1) -- the SAME conversion the DLS-angular sibling GammaQ2Result.rh_se uses
    # (rule 10). None whenever d_t_se is (single angle -> no ensemble SE).
    rh_t_se: Optional[float]
    rotational_time_s: float           # 1 / (6 D_r), the VH relaxation time
    # bookkeeping
    n_angles: int
    method: str                        # 'multi-angle' | 'single-angle'
    single_exponential_valid: Optional[bool]
    notes: Tuple[Note, ...] = ()
    # The D_r verdict as a FLAG, its causes as precondition notes in `notes`. Consumers
    # that ACT on the verdict (the export Comments cell, the GUI warning tier) read the
    # flag; only the human-facing surfaces read the notes, so a reword cannot disarm them
    # and no consumer re-derives the condition for itself. D_r = (Gamma_VH - Gamma_VV)/6 is
    # formed from both rates, so beside its own causes it fails through
    # `dr_inputs_flagged_decay`, over the VV and VH replicates whose own fit carries a
    # precondition bearing on its decay rate (`analyze_ddls`).
    d_r_reliable: bool = True
    # The D_t verdict. D_t is the through-origin line on the VV rates, and it reads their
    # decay causes as `analysis.dls.angular.analyze_gamma_q2`'s `rh_reliable` reads its
    # angles': through `inputs_flagged_decay`, over the VV replicates whose own fit carries
    # a precondition in `core.note_registry.DECAY_BEARING_CODES`. Rh_t, Stokes-Einstein on
    # D_t, shares it. It does NOT judge the line's own linearity through the origin, the
    # sibling's `is_diffusive`, which the anisotropic VV term (Berne & Pecora 2000,
    # Eq. 7.3.17) can fail: a VV intercept leaves D_t high and this flag True. It moves
    # no value, and a fallback weight, a record, is never read by it.
    d_t_reliable: bool = True
    se_estimator: str = 'hc3'          # covariance estimator behind d_t_se (the only
    #                                    regression SE here; d_r_se is a replicate SEM)
    # What d_r_se IS: the SEM of the per-angle D_r values, which no regression estimator
    # produced, so `core.notes.qualifiers` names no estimator about it.
    d_r_se_kind: str = 'replicate_sem'
    # The (T, eta) THIS result was computed with. Carried rather than looked up again,
    # because the shape models (below) need the same pair: rh_t_nm here and the models'
    # radii are both kT / 6 pi eta D, so two independent lookups can disagree and report
    # a particle as non-spherical purely because they picked different viscosities.
    # Optional only so an older cached/hand-built result still constructs.
    temperature_K: Optional[float] = None
    viscosity_Pa_s: Optional[float] = None


def rotational_diffusion_from_rates(gamma_vv_s_inv: float | np.ndarray,
                                    gamma_vh_s_inv: float | np.ndarray) -> float | np.ndarray:
    """Rotational diffusion coefficient D_r from one VV/VH field-rate pair.

        D_r = (Gamma_VH - Gamma_VV) / 6

    Both inputs are FIELD (g1) decay rates in s^-1, as floats or as equal-length arrays
    (one pair per angle; the result is then elementwise). The combined VV/VH pair is Berne &
    Pecora 2000, Eqs. 7.3.17-7.3.18 p.122; Zero & Pecora 1982, Eqs. III.1/III.2 is the
    rod-specific restatement. (NB Pecora 1964 gives the translational and rotational
    broadening only as SEPARATE limits, not this combined pair.) The result is in rad^2/s. A
    non-positive result (Gamma_VH <= Gamma_VV)
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
        For the Stokes-Einstein radius Rh_t from D_t. Always user-supplied. Without a
        usable viscosity Rh_t is NaN and the result carries the `no_viscosity` note.
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
    an independent estimate of the q-independent rotational rate). Fitting D_t as the
    SLOPE of Gamma_VH vs q^2 instead is ill-conditioned when rotation dominates -- the
    slope is tiny next to the 6 D_r intercept -- so it is avoided.

    Uncertainty (invariant 8): a SINGLE angle (one VV + one VH correlogram) gives NO
    standard error, each Gamma coming from a single correlogram (Schaetzel 1990). Across
    MULTIPLE angles the points are independent measurements, so the through-origin
    regression SE on D_t and the spread (SD/sqrt(n)) of the per-angle D_r are legitimate
    ensemble uncertainties -- the same basis as the multi-angle Zimm SE and the replicate
    SE (the GUM's standard error of the mean, sec. 4.2.3; ISO 22412 states precision from
    repeats as a standard deviation) -- and are reported with that label.

    Reliability of D_r (invariant 4): `d_r_reliable` is False, with a precondition note
    for each cause independently (`dr_nonpositive`, `vh_rate_missing`), when a per-angle
    rate is unphysical (Gamma_VH <= Gamma_VV) OR when an angle had no VH rate at all (a
    NaN Gamma_VH from a failed per-angle VH fit). The second cause matters because the
    mean tolerates it -- `replicate_mean_se` averages the angles that remain -- so a flag
    that could not see the drop once let an all-NaN VH set ship D_r = NaN as reliable
    with an empty note. The guard is sized to the operation it protects: whatever the
    mean silently drops, the flag names. Flag-not-refuse: D_r stays the mean over the
    surviving angles, and the note counts the angles without a VH rate. D_r also fails,
    as D_t does, when a replicate behind one of its two rates carries a precondition
    bearing on its decay rate: a VV replicate by `core.note_registry.DECAY_BEARING_CODES`,
    a VH replicate by that roster less its two size-range codes (`_VH_DECAY_BEARING_CODES`);
    `dr_inputs_flagged_decay` counts them over both channels, the first one's own reason
    beside it.

    Reliability of D_t: `d_t_reliable` is False when a VV replicate behind a point's rate
    carries a precondition bearing on its decay rate (`DDLSRatePoint.vv_preconditions`,
    `core.note_registry.DECAY_BEARING_CODES`), the decay rule `analyze_gamma_q2` applies to
    its angles: `inputs_flagged_decay` counts them, the first one's own reason beside it.
    The line's linearity through the origin, that sibling's `is_diffusive`, is not judged
    here. D_t and every other value stay as computed.
    """
    pts = sorted(points, key=lambda p: p.angle_deg)
    if len(pts) == 0:
        raise ValueError("analyze_ddls needs at least one DDLSRatePoint.")
    # T and eta are needed ONLY for the Stokes-Einstein radius Rh_t; D_t, D_r and
    # the rotational time do not use them, so they are optional (Rh_t -> nan if
    # either is missing or non-positive).
    # Every argument Stokes-Einstein will receive is gated here, not only D:
    # `stokes_einstein_rh` refuses a non-finite D, T *or* eta, so an infinite T or eta
    # reaching it raises -- the same escape the d_t gate below prevents, one argument over.
    have_t_and_eta = (temperature_K is not None and viscosity_Pa_s is not None
                      and math.isfinite(temperature_K) and temperature_K > 0
                      and math.isfinite(viscosity_Pa_s) and viscosity_Pa_s > 0)

    angles = np.array([p.angle_deg for p in pts], dtype=float)
    q = np.array([p.q_m_inv for p in pts], dtype=float)
    g_vv = np.array([p.gamma_vv_s_inv for p in pts], dtype=float)
    g_vh = np.array([p.gamma_vh_s_inv for p in pts], dtype=float)
    # NaN and +inf are NOT equivalent on the decay rates, and the difference is
    # load-bearing. NaN is TOLERATED: a failed VH cumulant fit at one angle yields a NaN
    # Gamma, and D_r is a `replicate_mean_se` over the per-angle rates, which drops
    # non-finite members -- so the remaining angles still give an honest D_r rather than
    # the whole multi-angle run being refused. +inf must NOT pass: nothing downstream drops
    # it, so it survives the mean, poisons the through-origin D_t fit, and Stokes-Einstein
    # turns it into a plausible R_h = 0.0 nm. (q is different: computed from validated
    # optics, never fitted, so it takes the full finite check.)
    if not np.all(np.isfinite(q) & (q > 0)):
        raise ValueError("every q_m_inv must be a positive finite number.")
    # ...and the tolerance is stated PER CHANNEL, because the survival argument only holds
    # for one of them. D_r is a replicate_mean_se over the per-angle (Gamma_VH - Gamma_VV),
    # which drops non-finite members -- so a NaN VH rate costs that angle and no more.
    # D_t is a through-origin fit of Gamma_VV vs q^2, which drops NOTHING: one NaN VV rate
    # NaNs the whole D_t and R_h,t. Tolerating NaN there would be silent withholding, so
    # VV takes the full finite check and VH does not.
    if not np.all(np.isfinite(g_vv) & (g_vv > 0)):
        raise ValueError(
            "every Gamma_VV must be a positive finite number (the D_t fit drops no "
            "points, so one bad angle would withhold D_t and R_h,t silently).")
    if np.any((g_vh <= 0) | np.isinf(g_vh)):
        raise ValueError(
            "every Gamma_VH must be positive and finite; NaN from a failed per-angle VH "
            "fit is tolerated and that angle is dropped from D_r.")
    q2 = q * q
    d_r_per_angle = rotational_diffusion_from_rates(g_vv, g_vh)
    n = len(pts)
    notes: List[Note] = []

    # D_t from the VV channel: Gamma_VV = q^2 D_t, slope through the origin.
    if n >= 2:
        d_t, d_t_se = unc.linear_fit_through_origin(q2, g_vv, estimator)
        d_t_se = unc.se_or_none(d_t_se)
        method = 'multi-angle'
    else:
        d_t = float(g_vv[0] / q2[0])
        d_t_se = None                       # one correlogram -> no SE (Schaetzel)
        method = 'single-angle'
    # D_t's verdict, the pooled lines' rule (one treatment): how many VV replicates carry a
    # decay-bearing precondition, then the first one's own reason where it was found.
    replicates = [(float(p.angle_deg), pre) for p in pts for pre in p.vv_preconditions]
    decay_flagged = [(a, kept) for a, pre in replicates
                     for kept in (tuple(n for n in pre if n.code in DECAY_BEARING_CODES),)
                     if kept]
    d_t_reliable = not decay_flagged
    if decay_flagged:
        d_t_about = ('d_t_m2_s', 'rh_t_nm')
        notes.append(Note('inputs_flagged_decay', d_t_about, {
            'k': len(decay_flagged), 'n': len(replicates), 'inputs': 'replicates',
            WHERE: tuple(dict.fromkeys(a for a, _pre in decay_flagged)), WHERE_UNIT: 'deg'}))
        first_at, first = decay_flagged[0]
        notes.extend(replace(n, about=d_t_about,
                             values={**n.values, WHERE: (first_at,), WHERE_UNIT: 'deg'})
                     for n in first)

    # D_r from the per-angle differences (independent estimates of one rate).
    d_r_stats = unc.replicate_mean_se(list(d_r_per_angle))
    d_r = d_r_stats.mean
    d_r_se = d_r_stats.sem                   # None for n < 2 or a zero spread

    # Two INDEPENDENT preconditions on D_r, each stated on its own: the per-angle rate
    # must be physical (Gamma_VH > Gamma_VV), and it must exist at all (a NaN is the
    # angle `replicate_mean_se` silently drops -- see the Notes above for why the flag
    # must see it). The NaN-vs-inf split of the guard above is kept: the unphysical
    # note is about Gamma_VH <= Gamma_VV, so a dropped angle must not fire it, and
    # +inf must. Consumers read the FLAG; the controller's shape-model entry guard
    # deliberately re-checks positivity on the AGGREGATE D_r instead (coarser: a flagged
    # mean keeps computing, caveat in Comments -- owner-ruled 2026-08-10).
    n_unphysical = int(np.count_nonzero((d_r_per_angle <= 0) | np.isinf(d_r_per_angle)))
    n_dropped = int(np.count_nonzero(np.isnan(d_r_per_angle)))
    d_r_reliable = n_unphysical == 0 and n_dropped == 0
    if n_unphysical:
        notes.append(Note('dr_nonpositive', ('d_r_rad2_s',),
                          {'n': n_unphysical, 'limit': 0.0}))
    if n_dropped:
        notes.append(Note('vh_rate_missing', ('d_r_rad2_s',),
                          {'n_missing': n_dropped, 'n': n}))
    # D_r's inputs, D_t's rule over both rates (one treatment): how many VV and VH
    # replicates carry a decay-bearing precondition, then the first one's own reason.
    rate_inputs = ([(a, pre, DECAY_BEARING_CODES) for a, pre in replicates]
                   + [(float(p.angle_deg), pre, _VH_DECAY_BEARING_CODES)
                      for p in pts for pre in p.vh_preconditions])
    rate_flagged = [(a, kept) for a, pre, codes in rate_inputs
                    for kept in (tuple(n for n in pre if n.code in codes),) if kept]
    d_r_reliable = d_r_reliable and not rate_flagged
    if rate_flagged:
        d_r_about = ('d_r_rad2_s',)
        notes.append(Note('dr_inputs_flagged_decay', d_r_about, {
            'k': len(rate_flagged), 'n': len(rate_inputs),
            WHERE: tuple(dict.fromkeys(a for a, _pre in rate_flagged)), WHERE_UNIT: 'deg'}))
        first_at, first = rate_flagged[0]
        notes.extend(replace(n, about=d_r_about,
                             values={**n.values, WHERE: (first_at,), WHERE_UNIT: 'deg'})
                     for n in first)

    # Stokes-Einstein radius from D_t (translational), and the VH relaxation time.
    # isfinite as well as > 0: +inf passes a bare `> 0` and Stokes-Einstein would return a
    # finite, plausible Rh = 0.0 nm (2026-07-31 spec-01). The physics function now refuses
    # a non-finite D, so without this gate the refusal would escape as a ValueError.
    rh_t_nm = (stokes_einstein_rh(d_t, temperature_K, viscosity_Pa_s) * 1e9
               if (have_t_and_eta and math.isfinite(d_t) and d_t > 0) else float('nan'))
    if not (viscosity_Pa_s is not None and math.isfinite(viscosity_Pa_s)
            and viscosity_Pa_s > 0):
        # Rh_t is blank for the one cause the DLS arms name the same way: D_t and D_r
        # need no viscosity, Stokes-Einstein does.
        notes.append(Note('no_viscosity', ('rh_t_nm', 'rh_t_se')))
    # Rh_t inherits d_t_se by the exact fractional relation (Rh_t ∝ D_t^-1), the same
    # already-validated reciprocal pass-through used for GammaQ2Result.rh_se (invariant 8;
    # no separate MC gate needed -- see power_law_se's carve-out note). None when d_t_se is.
    rh_t_se = (unc.power_law_se(rh_t_nm, d_t, d_t_se, -1)
               if math.isfinite(rh_t_nm) else None)
    rotational_time_s = (1.0 / (6.0 * d_r) if (math.isfinite(d_r) and d_r > 0)
                         else float('nan'))

    # qL single-exponential guard (only if a rod length is known).
    qL = None
    single_exp_valid: Optional[bool] = None
    if rod_length_nm is not None and rod_length_nm > 0:
        qL = q * (rod_length_nm * 1e-9)     # q[m^-1] * L[m] -> dimensionless
        single_exp_valid = bool(np.all(qL < qL_limit))
        if not single_exp_valid:
            notes.append(Note('ql_above_limit', ('d_r_rad2_s',),
                              {'ql': float(np.max(qL)), 'limit': float(qL_limit)}))

    return DDLSResult(
        angles_deg=angles, q2_m2=q2, gamma_vv_s_inv=g_vv, gamma_vh_s_inv=g_vh,
        d_r_per_angle=d_r_per_angle, qL=qL,
        d_t_m2_s=d_t, d_t_se=d_t_se,
        d_r_rad2_s=d_r, d_r_se=d_r_se,
        rh_t_nm=rh_t_nm, rh_t_se=rh_t_se, rotational_time_s=rotational_time_s,
        n_angles=n, method=method, single_exponential_valid=single_exp_valid,
        notes=tuple(notes), d_r_reliable=d_r_reliable, d_t_reliable=d_t_reliable,
        se_estimator=estimator,
        temperature_K=temperature_K, viscosity_Pa_s=viscosity_Pa_s)


@dataclass(frozen=True)
class DDLSRateAssembly:
    """Each paired angle's VV and VH field decay rates, averaged over its replicate
    correlograms' cumulant fits, with what entered each mean and what did not.

    The six per-angle fields hold one entry per paired angle, in the order the caller
    gave, a left-out angle included; `points` holds only the angles that kept a VV
    rate, and `entered_runs` only their runs. `temperature_K` and `viscosity_Pa_s` are
    read from those runs, None when no angle kept a rate.
    """
    points: Tuple[DDLSRatePoint, ...]
    # The (run, fit) pairs whose rate entered the angle's mean, in the caller's run order.
    vv_entered: Tuple[Tuple[Tuple[DLSMeasurement, dls.CumulantResult], ...], ...]
    vh_entered: Tuple[Tuple[Tuple[DLSMeasurement, dls.CumulantResult], ...], ...]
    # The runs whose fit refused its inputs, each with the refusal's own notes
    # (`core.notes.RefusedInputs.notes`; () for a refusal that names no cause).
    vv_refused: Tuple[Tuple[Tuple[DLSMeasurement, Tuple[Note, ...]], ...], ...]
    vh_refused: Tuple[Tuple[Tuple[DLSMeasurement, Tuple[Note, ...]], ...], ...]
    # The runs whose fit returned but whose rate did not enter: not a success, or a rate
    # that is non-finite or at or below zero.
    vv_unusable: Tuple[Tuple[DLSMeasurement, ...], ...]
    vh_unusable: Tuple[Tuple[DLSMeasurement, ...], ...]
    angles_left_out: Tuple[float, ...]       # the paired angles with no VV rate
    temperature_K: Optional[float]
    viscosity_Pa_s: Optional[float]
    entered_runs: Tuple[DLSMeasurement, ...]


def assemble_ddls_rate_points(
        vv_runs: Mapping[float, Sequence[Tuple[DLSMeasurement, Mapping[str, Any]]]],
        vh_runs: Mapping[float, Sequence[Tuple[DLSMeasurement, Mapping[str, Any]]]],
        paired_angles: Sequence[float], *, tolerate_refusals: bool) -> DDLSRateAssembly:
    """The `DDLSRatePoint`s `analyze_ddls` takes, from each paired angle's VV and VH
    replicate correlograms.

    `vv_runs` and `vh_runs` map each angle of `paired_angles` to its runs, each a
    (measurement, `fit_cumulants` keyword arguments) pair. Every run is fitted once, VV
    then VH at each angle in the order given, and a channel's field decay rate is the
    mean over the replicates whose rate entered: a fit that succeeded with a finite rate
    above zero. The caller settles identity across the whole set; this only averages.

    A fit that refuses its inputs (a `ValueError`) is recorded with its notes and left
    out of the mean when `tolerate_refusals` is True; when False it propagates
    unchanged. An angle with no VV rate is left out of `points` and listed in
    `angles_left_out`. q is read from the optics of the angle's first entered VV run,
    T from the first entered run, and eta from the first entered run carrying one.
    """
    points: List[DDLSRatePoint] = []
    entered_by_channel: Tuple[list, list] = ([], [])
    refused_by_channel: Tuple[list, list] = ([], [])
    unusable_by_channel: Tuple[list, list] = ([], [])
    entered_runs: List[DLSMeasurement] = []
    angles_left_out: List[float] = []
    for a in paired_angles:
        rates = []
        for ch, runs in enumerate((vv_runs[a], vh_runs[a])):
            fits, refused = [], []
            for m, kw in runs:
                try:
                    fits.append((m, dls.fit_cumulants(m, **kw)))
                except ValueError as exc:
                    if not tolerate_refusals:
                        raise
                    refused.append((m, tuple(exc.notes) if isinstance(exc, RefusedInputs)
                                    else ()))
            # DROP the failed replicates and average the rest, which is what the
            # replicate-averaging path already does. A bare `np.mean` was safe only while
            # an unfittable correlogram RAISED; now that it returns a flagged NaN, one bad
            # replicate NaN'd the whole angle, the point was dropped, and a downstream
            # raise blamed the VV/VH pairing -- naming a cause that was not the one.
            # NaN when NOTHING survives: an angle with no VV rate is left out below.
            # A rate at or below zero is no decay rate: `analyze_ddls` refuses one, and
            # the Gamma-vs-q^2 fits drop one, so it enters no mean here either.
            entered = [(m, f) for m, f in fits
                       if f.success and np.isfinite(f.gamma_s_inv) and f.gamma_s_inv > 0]
            # A fit that returned but did not enter is kept with the refusals, so the
            # caller can account for every replicate supplied.
            kept = {id(f) for _m, f in entered}
            unusable = [m for m, f in fits if id(f) not in kept]
            gamma = (float(np.mean([float(f.gamma_s_inv) for _m, f in entered]))
                     if entered else float('nan'))
            rates.append(gamma)
            entered_by_channel[ch].append(tuple(entered))
            refused_by_channel[ch].append(tuple(refused))
            unusable_by_channel[ch].append(tuple(unusable))
        g_vv, g_vh = rates
        vv_in, vh_in = entered_by_channel[0][-1], entered_by_channel[1][-1]
        if not math.isfinite(g_vv):
            # D_t is a through-origin fit that drops no point, so an angle with no VV
            # rate is left out here, for the caller to name, rather than failing the whole
            # set with a cause that is not the one. A missing VH rate stays: `analyze_ddls`
            # keeps the angle's VV rate in D_t, drops it from D_r, and flags D_r for it.
            angles_left_out.append(float(a))
            continue
        entered_runs.extend(m for m, _f in vv_in + vh_in)
        # q from the optics of a run whose rate entered, as T and eta are read below.
        q = phys.scattering_vector_q_m(
            a, vv_in[0][0].wavelength_nm, vv_in[0][0].solvent_refractive_index)
        # Each replicate's own preconditions ride its point for the D_t and D_r
        # verdicts; the rate is the mean above either way.
        points.append(DDLSRatePoint(
            angle_deg=a, q_m_inv=q, gamma_vv_s_inv=g_vv, gamma_vh_s_inv=g_vh,
            vv_preconditions=tuple(find_notes(f, kind='precondition') for _m, f in vv_in),
            vh_preconditions=tuple(find_notes(f, kind='precondition') for _m, f in vh_in)))

    # Temperature and viscosity come from the runs that actually enter the fit, not
    # from whichever correlogram was listed first: a left-out angle's runs and a refused
    # or unfitted replicate never enter, so their eta has no business setting the
    # Stokes-Einstein scale for D_t / D_r. (Viscosity is deliberately outside the
    # identity check -- see check_replicate_identity -- so it can genuinely differ.)
    t_k = entered_runs[0].temperature_K if entered_runs else None
    eta = next((m.viscosity_Pa_s for m in entered_runs
                if m.viscosity_Pa_s is not None), None)
    return DDLSRateAssembly(
        points=tuple(points),
        vv_entered=tuple(entered_by_channel[0]), vh_entered=tuple(entered_by_channel[1]),
        vv_refused=tuple(refused_by_channel[0]), vh_refused=tuple(refused_by_channel[1]),
        vv_unusable=tuple(unusable_by_channel[0]), vh_unusable=tuple(unusable_by_channel[1]),
        angles_left_out=tuple(angles_left_out), temperature_K=t_k, viscosity_Pa_s=eta,
        entered_runs=tuple(entered_runs))


# ===========================================================================
# Shape models (DPLS Phase 3) -- particle dimensions from D_t and D_r
# ===========================================================================
#
# The model-DEPENDENT inverse: D_t and D_r are shape-free observables, but turning them into
# dimensions assumes a geometry. Outputs are dimensions of an ASSUMED shape, NOT measurements
# -- each result object carries that caveat and a validity flag so the GUI never presents an
# L as if it were observed. Read together the models diagnose the shape.


@dataclass
class SphereShapeResult:
    """Sphere-model dimensions from D_t and D_r (model-dependent)."""
    radius_rot_nm: float            # R from D_r (Stokes-Einstein-Debye)
    radius_rot_se: Optional[float]
    radius_trans_nm: float          # R from D_t (= Stokes/hydrodynamic radius Rh)
    radius_trans_se: Optional[float]  # inherited from d_t_se (R_trans ∝ D_t^-1)
    sphericity_ratio: float         # radius_rot / radius_trans (1 for a true sphere)
    is_consistent: bool             # |ratio - 1| within tolerance -> sphere plausible
    interpretation: str = ''        # the reading of is_consistent (guidance, not a note)
    # The shape-model family's notes field. The sphere fit has no precondition of its
    # own to fail -- its verdict is `is_consistent` and its reading `interpretation` --
    # so nothing mints one here.
    notes: Tuple[Note, ...] = ()
    # The covariance estimator behind `radius_trans_se` -- a strict pass-through of the
    # DDLS `d_t_se`, which is HC3 or the classical-OLS opt-in at the user's choice. The
    # tag travels with the ± onto every surface that prints it (invariant 8 clause A);
    # `radius_rot_se` is a replicate SEM and has no estimator.
    se_estimator: str = 'hc3'
    # What radius_rot_se IS: the DDLS replicate SEM of D_r carried through R ∝ D_r^(-1/3),
    # which no regression estimator produced, so `core.notes.qualifiers` names no
    # estimator about it (`DDLSResult.d_r_se_kind`'s rule).
    radius_rot_se_kind: str = 'replicate_sem'


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
    notes: Tuple[Note, ...] = ()
    # The covariance estimator behind the DDLS `d_t_se` the Monte Carlo draws from
    # (invariant 8 clause A: the ± travels with the estimator that produced it).
    se_estimator: str = 'hc3'
    # The Monte-Carlo census behind the three ± -- how many (D_t, D_r) draws were
    # taken, how many a rigid rod actually reproduced, and the seed. The ± is the SD
    # over the draws that solved, so the count is part of what the ± means: a draw
    # fails only when the pair lies outside the rod model (a negative rate, or a D_r
    # faster than any rod with that D_t admits). Zero draws when no SEs were given.
    mc_n_draws: int = 0
    mc_n_used: int = 0
    mc_seed: Optional[int] = None


@dataclass
class EllipsoidShapeResult:
    """Ellipsoid-of-revolution dimensions from D_t and D_r (model-dependent).

    Perrin friction: translation Tanford 1961 (Eqs. 19-14/19-15), rotation Koenig
    1975 (Eq. 9, correcting Perrin 1934). One solution per fitting branch -- a
    given (D_t, D_r) admits up to THREE ellipsoids (1 prolate + 2 oblate; the
    oblate D_perp is non-monotonic), so ellipsoid_dimensions_from_diffusion
    returns a LIST of these and `degenerate` is set when more than one fits;
    `n_solutions` is how many do, which `core.notes.qualifiers` reads beside
    `degenerate` and `is_prolate` to name the degeneracy.
    """
    semi_axis_polar_nm: float           # a (polar / symmetry-axis semi-axis)
    semi_axis_polar_se: Optional[float]
    equatorial_radius_nm: float         # b (equatorial radius)
    equatorial_radius_se: Optional[float]
    axial_ratio_p: float                # p = a/b (>1 prolate, <1 oblate)
    axial_ratio_se: Optional[float]
    is_prolate: bool
    in_valid_range: bool                # 0.1 < p < 10 (Koenig tabulated range)
    converged: bool                     # an ellipsoid reproduces both D_t and D_r
    degenerate: bool                    # >1 ellipsoid fits this (D_t, D_r)
    n_solutions: int = 0                # ellipsoids reproducing this (D_t, D_r), 0 to 3
    notes: Tuple[Note, ...] = ()
    # As on RodShapeResult: the estimator behind the DDLS `d_t_se` the draws come from,
    # and the Monte-Carlo census PER SOLUTION (a draw counts for this branch only when
    # its re-solve has a root nearest this branch's p -- see the matching rule in
    # `ellipsoid_dimensions_from_diffusion`).
    se_estimator: str = 'hc3'
    mc_n_draws: int = 0
    mc_n_used: int = 0
    mc_seed: Optional[int] = None


def sphere_dimensions_from_diffusion(
        d_t_m2_s: float, d_r_rad2_s: float, *,
        temperature_K: float, viscosity_Pa_s: float,
        d_t_se: Optional[float] = None,
        d_r_se: Optional[float] = None,
        sphericity_tol: float = 0.15,
        se_estimator: str = 'hc3') -> SphereShapeResult:
    """Sphere-model radii from D_t and D_r, with a sphericity consistency check.

    R_rot comes from D_r (Stokes-Einstein-Debye, R = (kT/8 pi eta D_r)^1/3); R_trans
    is the ordinary Stokes radius from D_t (= Rh). For a true sphere they are equal,
    so `sphericity_ratio` = R_rot/R_trans should be ~1; a ratio outside
    1 +/- `sphericity_tol` flags that the particle is not spherical (e.g. a rod,
    where R_rot > R_trans). One unknown, and appropriate for near-spherical anisotropic
    particles (Balog et al. 2015). R_rot's SE propagates from D_r by the delta method
    (R ∝ D_r^(-1/3) -> SE(R)/R = (1/3) SE(D_r)/D_r); R_trans's SE propagates from D_t as
    the reciprocal Stokes radius (R_trans ∝ D_t^-1 -> SE(R)/R = SE(D_t)/D_t), the same
    already-validated pass-through as the DLS-angular Rh_se / DDLS rh_t_se (invariant 8).
    `se_estimator` names the covariance estimator behind `d_t_se` (the DDLS result's
    own tag) and is recorded on the result so R_trans's ± never travels unlabelled.
    """
    r_rot_nm = phys.sphere_radius_from_rotational_diffusion(
        d_r_rad2_s, temperature_K, viscosity_Pa_s) * 1e9
    r_trans_nm = stokes_einstein_rh(d_t_m2_s, temperature_K, viscosity_Pa_s) * 1e9
    # isfinite: an infinite R_trans gives sphericity_ratio = 0.0, and the
    # 'outside 1 +/- tol -> not spherical' verdict is then issued on that 0.0.
    ratio = (r_rot_nm / r_trans_nm
             if (math.isfinite(r_trans_nm) and r_trans_nm > 0) else float('nan'))
    # Both ± legs gate on the finiteness of the VALUE they qualify, not on their input
    # rate: gating on the rate admits +inf, and an infinite D_r would give
    # radius_rot = 0.00 nm ± 0.00 -- a ± of zero on a size of zero, from a rate that is not
    # a number. Both go through the SAME shared helper with the same gate, so they are
    # identical bar the exponent (R_rot ∝ D_r^(-1/3), R_trans ∝ D_t^-1), and `power_law_se`
    # carries the x == 0 and non-finite-SE handling a hand-rolled form would not
    # (invariant 8: a ± comes from the shared toolkit).
    r_rot_se = (unc.power_law_se(r_rot_nm, d_r_rad2_s, d_r_se, -1.0 / 3.0)
                if math.isfinite(r_rot_nm) else None)
    r_trans_se = (unc.power_law_se(r_trans_nm, d_t_m2_s, d_t_se, -1)
                  if math.isfinite(r_trans_nm) else None)
    consistent = math.isfinite(ratio) and abs(ratio - 1.0) <= sphericity_tol
    interpretation = (
        'R from D_r and R from D_t agree -> a sphere is a plausible model.'
        if consistent else
        f'R(D_r) = {r_rot_nm:.2f} nm and R(D_t) = {r_trans_nm:.2f} nm disagree '
        f'(ratio {ratio:.2f}); the particle is not spherical -- prefer the rod '
        f'model (ratio > 1 is the rod signature).')
    return SphereShapeResult(
        radius_rot_nm=r_rot_nm, radius_rot_se=unc.se_or_none(r_rot_se),
        radius_trans_nm=r_trans_nm, radius_trans_se=unc.se_or_none(r_trans_se),
        sphericity_ratio=ratio, is_consistent=consistent, interpretation=interpretation,
        se_estimator=se_estimator)


def _same_sign(a: float, b: float) -> bool:
    """Do two finite residuals share a sign, so that NO root is bracketed between them?

    A residual of exactly ZERO means the root is AT that endpoint, so the bracket is
    valid and this must answer False. `math.copysign` alone cannot say that: it reports
    +1 for `0.0` and -1 for `-0.0`, so this function's FIRST version bailed on `(0.0, 5.0)`
    while proceeding on `(0.0, -5.0)` -- the same bracket, opposite answers, decided by the
    sign bit of a zero. `brentq` accepts `f(a) == 0` and returns that endpoint, so bailing
    there throws away an exact solution and reports "the rod model does not fit". A zero is
    also reachable by UNDERFLOW.

    Written as a sign comparison, NOT as ``a * b > 0``. The product form is the sheet's
    own model defect: it UNDERFLOWS to exactly 0.0 for same-signed values around 1e-200
    (at 1e-160 the product is 1e-320 -- subnormal but non-zero, so still ``> 0``; the
    magnitude that reaches bitwise zero is two decades further down, which is what the
    companion test drives), so ``a * b > 0`` reads False, the "no sign change, bail out"
    branch does not fire, and
    `brentq` is called on a bracket it rejects -- raising ``ValueError`` from OUTSIDE the
    ``try`` that wraps the residual evaluations. (It overflows to ``inf`` at the other
    tail, which happens to read True, so only one end was broken -- the usual asymmetry.)

    The question the guard is asking is about SIGNS, and signs are exact at every
    magnitude: expressing the predicate on the raw inputs removes both tails instead of
    guarding one (CLAUDE.md rule 12).
    Shared by the rod and ellipsoid solvers so the two cannot drift apart (rule 10).
    """
    if a == 0.0 or b == 0.0:
        return False                       # the root is at an endpoint: bracket is valid
    return math.copysign(1.0, a) == math.copysign(1.0, b)


# The rod inverse's fallback bracket in p = L/d. p > 1 is the cylinder domain (L > d);
# the upper end is a float-range ceiling, not a physical one -- the Tirado forms are
# monotone in ln p out to any p a float holds, and the Monte-Carlo tail below needs the
# whole domain open (see `_solve_rod_pld`). The same shape as the ellipsoid pair below.
_ROD_P_BRACKET_LO: float = 1.0 + 1e-4
_ROD_P_BRACKET_HI: float = 1e9
# The ellipsoid inverse's outer brackets in p = a/b, prolate (> 1) and oblate (< 1). The
# Perrin factors are defined on all of p > 0; the old [0.01, 100] window clipped the
# same Monte-Carlo tail the rod window did.
_ELLIPSOID_P_PROLATE_HI: float = 1e4
_ELLIPSOID_P_OBLATE_LO: float = 1e-4


def _solve_rod_pld(d_t_m2_s: float, d_r_rad2_s: float,
                   temperature_K: float, viscosity_Pa_s: float):
    """Solve (L, d, p) for a rigid cylinder reproducing D_t and D_r, or None.

    Reduces the 2-D (L, d) system to a 1-D root find in the aspect ratio p: at any
    p, D_t fixes L (rod_length_from_translational_diffusion), then the predicted D_r
    is monotonic decreasing in p, so D_r(p) - D_r_measured has at most one root. Try
    the Tirado-valid window first, then the whole range a cylinder admits (flagged
    out-of-range). Returns (L_nm, d_nm, p, in_valid_range) or None if no rod fits.

    The fallback bracket is the full domain of the Tirado forms (p > 1, where L > d;
    the end corrections are defined for every positive p), not a convenient window.
    It used to be [1.05, 200], and that clip fell exactly on the Monte-Carlo tail of
    `rod_dimensions_from_diffusion`: a (D_t, D_r) draw with a slightly slow D_r has a
    very large p, and dropping those draws made the reported ± the SD of a censored
    sample -- it stopped tracking the input ± and could shrink as the input ± grew.
    With the bracket open, a draw fails only when NO cylinder reproduces it (D_r faster
    than the p -> 1 limit for that D_t) -- the model's own boundary. The remaining end is
    a FLOAT ceiling, not a physical one: a D_r slow enough to need p > 1e9 is still lost,
    which is far outside any rod light scattering sizes.
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
        if not (math.isfinite(f_lo) and math.isfinite(f_hi)) or _same_sign(f_lo, f_hi):
            return None
        return optimize.brentq(dr_residual, lo, hi, xtol=1e-9, rtol=1e-12)

    p = solve_in(phys.ROD_ASPECT_RATIO_MIN, phys.ROD_ASPECT_RATIO_MAX)
    in_range = p is not None
    if p is None:                                   # the whole cylinder domain, flagged
        p = solve_in(_ROD_P_BRACKET_LO, _ROD_P_BRACKET_HI)
    if p is None:
        return None
    length_m = phys.rod_length_from_translational_diffusion(
        d_t_m2_s, p, temperature_K, viscosity_Pa_s)
    return (length_m * 1e9, length_m / p * 1e9, p, in_range)


def rod_dimensions_from_diffusion(
        d_t_m2_s: float, d_r_rad2_s: float, *,
        temperature_K: float, viscosity_Pa_s: float,
        d_t_se: Optional[float] = None, d_r_se: Optional[float] = None,
        n_mc: int = 2000, seed: int = 12345,
        se_estimator: str = 'hc3') -> RodShapeResult:
    """Rigid-cylinder length and diameter from D_t and D_r (Tirado 1984 inverse).

    Solves the 2-D problem via the 1-D reduction in `_solve_rod_pld`. When both SEs
    are given, the dimension SEs are propagated by Monte Carlo: sample D_t, D_r from
    independent normals, invert each, and take the spread (the sampling SD -- no
    linearization) over the draws a rigid rod reproduces. That census is recorded on
    the result (`mc_n_draws`, `mc_n_used`, `mc_seed`, and the `mc_census` record note
    about the three ±) because it is part of what the
    ± means: a draw is lost only where the pair leaves the rod model altogether (a
    negative rate, or a D_r faster than any cylinder with that D_t admits -- the p -> 1
    limit), not to the old aspect-ratio window; the surviving bracket end is a float
    ceiling far outside any physical rod (see `_solve_rod_pld`). The aspect ratio's
    distribution is heavy-tailed at large input SEs (a slow D_r draw means a very long
    thin rod), so `aspect_ratio_se` can be large next to the point estimate -- that is
    the honest spread, not a defect; L and d stay well behaved. `se_estimator` is the
    DDLS result's tag for the `d_t_se` fed in, carried so the ± travels labelled.
    `in_valid_range` is False when the recovered p falls outside the Tirado-fitted
    2 < p < 30 (the dimensions are then extrapolated); `converged` is False when no
    rod reproduces both coefficients (the rod model does not fit).
    """
    solved = _solve_rod_pld(d_t_m2_s, d_r_rad2_s, temperature_K, viscosity_Pa_s)
    if solved is None:
        return RodShapeResult(
            length_nm=float('nan'), length_se=None, diameter_nm=float('nan'),
            diameter_se=None, aspect_ratio_p=float('nan'), aspect_ratio_se=None,
            in_valid_range=False, converged=False,
            notes=(Note('no_shape_solution', (), {'model': 'rigid rod'}),),
            se_estimator=se_estimator)
    length_nm, diameter_nm, p, in_range = solved

    length_se = diameter_se = p_se = None
    n_draws = n_used = 0
    mc_seed: Optional[int] = None
    if d_t_se is not None and d_r_se is not None and d_t_se > 0 and d_r_se > 0:
        rng = np.random.default_rng(seed)
        n_draws, mc_seed = int(n_mc), int(seed)
        Ls, ds, ps = [], [], []
        for _ in range(n_draws):
            dt = float(rng.normal(d_t_m2_s, d_t_se))
            dr = float(rng.normal(d_r_rad2_s, d_r_se))
            if dt <= 0 or dr <= 0:
                continue
            s = _solve_rod_pld(dt, dr, temperature_K, viscosity_Pa_s)
            if s is not None:
                Ls.append(s[0]); ds.append(s[1]); ps.append(s[2])
        n_used = len(Ls)
        if n_used >= 2:
            length_se = unc.se_or_none(float(np.std(Ls, ddof=1)))
            diameter_se = unc.se_or_none(float(np.std(ds, ddof=1)))
            p_se = unc.se_or_none(float(np.std(ps, ddof=1)))

    notes: Tuple[Note, ...] = () if in_range else (
        Note('aspect_ratio_out_of_range', ('aspect_ratio_p',),
             {'p': p, 'lo': phys.ROD_ASPECT_RATIO_MIN, 'hi': phys.ROD_ASPECT_RATIO_MAX,
              'model': 'rod'}),)
    if n_draws > 0:
        notes += (Note('mc_census', ('length_se', 'diameter_se', 'aspect_ratio_se'),
                       {'used': n_used, 'n': n_draws, 'seed': mc_seed}),)
    return RodShapeResult(
        length_nm=length_nm, length_se=length_se,
        diameter_nm=diameter_nm, diameter_se=diameter_se,
        aspect_ratio_p=p, aspect_ratio_se=p_se,
        in_valid_range=in_range, converged=True, notes=notes,
        se_estimator=se_estimator,
        mc_n_draws=n_draws, mc_n_used=n_used, mc_seed=mc_seed)


def _ellipsoid_oblate_peak_p() -> float:
    """p* where the oblate branch's D_perp(p) at fixed D_t peaks (~0.722).

    D_perp|_{D_t} is a pure shape function of the axial ratio (proportional to
    (f/f0)^3/(C2/C0), independent of D_t, T, eta), so p* is a universal constant of
    the Perrin factors. Computed once here (not transcribed) so the oblate-solve split
    is exactly on the peak -- a slightly-off split could fail to bracket a root sitting
    right at the maximum. Any fixed (D_t, T, eta) gives the same p*.
    """
    def neg_dperp(p):
        a_m, b_m = phys.ellipsoid_semi_axes_from_translational_diffusion(
            1e-11, p, 298.15, 1e-3)
        return -phys.ellipsoid_rotational_diffusion(a_m, b_m, 298.15, 1e-3)
    return float(optimize.minimize_scalar(
        neg_dperp, bounds=(0.15, 0.999), method='bounded').x)


# The oblate branch is non-monotonic; it is split at its (D_t-independent) peak into
# two monotonic sub-intervals so each brackets at most one root.
_ELLIPSOID_OBLATE_PEAK_P: float = _ellipsoid_oblate_peak_p()


def _solve_ellipsoid_axes(d_t_m2_s: float, d_r_rad2_s: float,
                          temperature_K: float, viscosity_Pa_s: float):
    """Every ellipsoid of revolution reproducing D_t and D_r (0-3 of them).

    At fixed D_t each axial ratio p fixes the size (ellipsoid_semi_axes_from_
    translational_diffusion); the predicted D_perp is then matched to the measured
    value. The prolate branch (p>1) is monotonic in p (<=1 root); the oblate branch
    (p<1) has an interior maximum at _ELLIPSOID_OBLATE_PEAK_P, so it is split there
    into two monotonic pieces and can yield up to two roots. Returns a list of
    (a_nm, b_nm, p, is_prolate, in_valid_range) sorted by p, near-p duplicates
    (the shared sphere limit) merged.
    """
    def residual(p):
        a_m, b_m = phys.ellipsoid_semi_axes_from_translational_diffusion(
            d_t_m2_s, p, temperature_K, viscosity_Pa_s)
        return phys.ellipsoid_rotational_diffusion(
            a_m, b_m, temperature_K, viscosity_Pa_s) - d_r_rad2_s

    def root_in(lo, hi):
        # Each interval is monotonic in D_perp(p), so an endpoint sign change
        # brackets the single root.
        try:
            f_lo, f_hi = residual(lo), residual(hi)
        except (ValueError, ZeroDivisionError):
            return None
        if not (math.isfinite(f_lo) and math.isfinite(f_hi)) or _same_sign(f_lo, f_hi):
            return None
        return optimize.brentq(residual, lo, hi, xtol=1e-12, rtol=1e-12)

    ps = []
    r = root_in(1.0 + 1e-4, _ELLIPSOID_P_PROLATE_HI)     # prolate (monotonic)
    if r is not None:
        ps.append(r)
    for lo, hi in ((_ELLIPSOID_OBLATE_PEAK_P, 1.0 - 1e-4),   # oblate, near-sphere side
                   (_ELLIPSOID_P_OBLATE_LO, _ELLIPSOID_OBLATE_PEAK_P)):  # oblate, far side
        r = root_in(lo, hi)
        if r is not None:
            ps.append(r)

    out = []
    for p in sorted(ps):
        a_m, b_m = phys.ellipsoid_semi_axes_from_translational_diffusion(
            d_t_m2_s, p, temperature_K, viscosity_Pa_s)
        out.append((a_m * 1e9, b_m * 1e9, p, p > 1.0,
                    phys.ellipsoid_aspect_ratio_valid(p)))
    merged = []
    for sol in out:                                     # drop the shared p~1 duplicate
        if merged and abs(sol[2] - merged[-1][2]) < 1e-3:
            continue
        merged.append(sol)
    return merged


def ellipsoid_dimensions_from_diffusion(
        d_t_m2_s: float, d_r_rad2_s: float, *,
        temperature_K: float, viscosity_Pa_s: float,
        d_t_se: Optional[float] = None, d_r_se: Optional[float] = None,
        n_mc: int = 2000, seed: int = 12345,
        se_estimator: str = 'hc3') -> List[EllipsoidShapeResult]:
    """Ellipsoid-of-revolution semi-axes from D_t and D_r (Perrin inverse).

    Returns a LIST of EllipsoidShapeResult -- one per ellipsoid that reproduces
    both coefficients (Tanford 1961 translation + Koenig 1975 rotation via
    `_solve_ellipsoid_axes`). A given (D_t, D_r) admits up to three (1 prolate +
    2 oblate); every fitting solution is returned, `degenerate` is set when more
    than one does, and none is silently preferred. No ellipsoid fitting -> a single
    `converged=False` NaN sentinel (never an empty list). When both SEs are given,
    each solution's dimension SEs are Monte-Carlo-propagated (sample D_t, D_r from
    independent normals, re-solve, match each draw's roots to the nearest base
    solution by p, take the spread) -- the same MC pattern as the rod, locked per
    solution so the branches do not cross-contaminate. Each solution records its own
    census (`mc_n_draws`, `mc_n_used`, `mc_seed`, and the `mc_census` record note about
    its ±): a draw counts for a branch only
    when its re-solve has a root nearest that branch, so a branch that exists for a
    minority of draws reports a ± over that minority and says so. The brackets are
    the full p > 0 domain of the Perrin factors, so a draw is lost to the model's own
    boundary rather than to a shape window (the same censoring the rod inverse had); the
    surviving bracket ends are float limits far outside any physical particle.
    `se_estimator` is the DDLS result's tag for the `d_t_se` fed in. `in_valid_range`
    is False outside Koenig's tabulated 0.1 < p < 10 (dimensions then extrapolated).
    """
    solved = _solve_ellipsoid_axes(d_t_m2_s, d_r_rad2_s,
                                   temperature_K, viscosity_Pa_s)
    if not solved:
        return [EllipsoidShapeResult(
            semi_axis_polar_nm=float('nan'), semi_axis_polar_se=None,
            equatorial_radius_nm=float('nan'), equatorial_radius_se=None,
            axial_ratio_p=float('nan'), axial_ratio_se=None,
            is_prolate=False, in_valid_range=False, converged=False,
            degenerate=False, n_solutions=0,
            notes=(Note('no_shape_solution', (), {'model': 'ellipsoid of revolution'}),),
            se_estimator=se_estimator)]

    degenerate = len(solved) > 1
    base_ps = [s[2] for s in solved]
    ses = [(None, None, None) for _ in solved]
    n_used = [0 for _ in solved]
    n_draws = 0
    mc_seed: Optional[int] = None
    if d_t_se is not None and d_r_se is not None and d_t_se > 0 and d_r_se > 0:
        rng = np.random.default_rng(seed)
        n_draws, mc_seed = int(n_mc), int(seed)
        samples = [[] for _ in solved]
        log_base = [math.log(bp) for bp in base_ps]
        for _ in range(n_draws):
            dt = float(rng.normal(d_t_m2_s, d_t_se))
            dr = float(rng.normal(d_r_rad2_s, d_r_se))
            if dt <= 0 or dr <= 0:
                continue
            # Assign each draw's roots to their nearest base solution by log(p), but keep
            # only the CLOSEST root per base. Near an existence fold a draw can gain a
            # branch the base set lacks (e.g. an oblate root when the base is a lone
            # prolate); absorbing that far root into an unrelated base would grossly
            # inflate its dimension SE (invariant 8, non-conservative over-report). The
            # closest-per-base rule drops those spurious extra roots.
            best = {}                               # base index -> (log-dist, (a,b,p))
            for a_nm, b_nm, p, _is_pro, _in in _solve_ellipsoid_axes(
                    dt, dr, temperature_K, viscosity_Pa_s):
                lp = math.log(p)
                j = min(range(len(log_base)), key=lambda k: abs(lp - log_base[k]))
                dist = abs(lp - log_base[j])
                if j not in best or dist < best[j][0]:
                    best[j] = (dist, (a_nm, b_nm, p))
            for j, (_dist, sol) in best.items():
                samples[j].append(sol)
        for j, s in enumerate(samples):
            n_used[j] = len(s)
            if len(s) >= 2:
                arr = np.asarray(s)
                ses[j] = (unc.se_or_none(float(np.std(arr[:, 0], ddof=1))),
                          unc.se_or_none(float(np.std(arr[:, 1], ddof=1))),
                          unc.se_or_none(float(np.std(arr[:, 2], ddof=1))))

    # The degeneracy is a qualifier, read off `degenerate`, `n_solutions` and
    # `is_prolate` by `core.notes.qualifiers`; only the range precondition is minted.
    results = []
    for (a_nm, b_nm, p, is_pro, in_range), (a_se, b_se, p_se), used in zip(
            solved, ses, n_used, strict=True):
        notes: Tuple[Note, ...] = () if in_range else (
            Note('aspect_ratio_out_of_range', ('axial_ratio_p',),
                 {'p': p, 'lo': phys.ELLIPSOID_ASPECT_RATIO_MIN,
                  'hi': phys.ELLIPSOID_ASPECT_RATIO_MAX, 'model': 'ellipsoid'}),)
        if n_draws > 0:
            notes += (Note('mc_census', ('semi_axis_polar_se', 'equatorial_radius_se',
                                         'axial_ratio_se'),
                           {'used': used, 'n': n_draws, 'seed': mc_seed}),)
        results.append(EllipsoidShapeResult(
            semi_axis_polar_nm=a_nm, semi_axis_polar_se=a_se,
            equatorial_radius_nm=b_nm, equatorial_radius_se=b_se,
            axial_ratio_p=p, axial_ratio_se=p_se,
            is_prolate=is_pro, in_valid_range=in_range, converged=True,
            degenerate=degenerate, n_solutions=len(solved), notes=notes,
            se_estimator=se_estimator,
            mc_n_draws=n_draws, mc_n_used=used, mc_seed=mc_seed))
    return results
