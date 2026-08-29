"""
analysis/optical_model.py
=========================

Optical scattering model for **re-weighting a DLS size distribution** (Phase 1b of the
weighted-distributions / form-factor spec). It converts an *intensity*-weighted size
distribution (the only thing NNLS / CONTIN / lognormal return) into **volume-, surface-,
and number-weighted** distributions.

Why a model is needed
---------------------
Large particles scatter far more light than small ones, so an intensity distribution
over-represents them. To answer "how MANY particles are there" from "how much light this
size bin scatters" you must divide out the per-particle scattering ``I1(R, q)``:

    intensity weight  w_I(R) proportional to  N(R) * I1(R, q)
    => number         N(R)   proportional to  w_I(R) / I1(R, q)
    => surface        proportional to  N(R) * R^2
    => volume / mass  proportional to  N(R) * R^3

This inversion is well-posed **only for homogeneous spheres**, where the DLS-measured
hydrodynamic radius Rh equals the geometric radius R and ``I1`` is a clean law. So this
module is spheres-only (Rayleigh point / RDG sphere / full Mie sphere) and **refuses a
non-spherical model**. A coil/rod would need an assumed architecture + rho + the Flory
exponent -- a molar-mass-distribution calculation that is explicitly out of scope.

Per-particle sphere scattering ``I1``, by regime (the Rayleigh -> RDG -> Mie ladder)
------------------------------------------------------------------------------------
* **Rayleigh (point) scatterer**, valid ``x = 2*pi*R/lambda <= ~0.1``:
      I1 proportional to R^6              (no refractive-index contrast needed)
* **Rayleigh-Debye-Gans (RDG) sphere**, valid ``|m-1| << 1`` and ``2*x*|m-1| << 1``:
      I1 proportional to (m^2 - 1)^2 * V^2 * P(qR)  proportional to  R^6 * P(qR)
  with ``V = (4/3) pi R^3`` and ``P`` the sphere form factor (Guinier & Fournet 1955;
  Pedersen 1997 Eq. 55). The bulk optical-contrast prefactor -- proportional to (m^2 - 1)^2
  (~ (2(m-1))^2 in the RDG small-contrast limit) -- is the SAME for every size bin
  (one material), so it CANCELS in ``N proportional to w_I / I1`` after renormalisation --
  therefore ``m`` (from the solute refractive index) enters **only the RDG validity gate**
  ``2*x*|m-1|``, never the reweighting arithmetic itself. RDG reduces to the Rayleigh law
  ``P -> 1`` as ``qR -> 0``.
* **Full Mie sphere** (the GENERAL case -- large x, non-negligible contrast):
      I1 = |S1(x, m, theta)|^2, the full Mie VV per-particle intensity (``analysis.mie``,
  Bohren & Huffman 1983). Here ``m`` enters the *arithmetic* itself (not just a gate), and
  the contrast does NOT cancel size-independently, so each bin carries its own ``x, m``
  dependence. This is what makes the classic aqueous-nanoparticle case (polystyrene latex in
  water, ``m ~ 1.19``, ``x`` large -- outside RDG *and* Rayleigh) work instead of being
  refused. Mie reduces to the RDG ``R^6 P(qR)`` law as ``m -> 1`` and to Rayleigh ``R^6`` as
  ``x -> 0`` (asserted in tests), so it composes as the top of the same ladder.

Uncertainty (architecture invariant 8)
--------------------------------------
Reweighted distributions carry **NO +/-**. They derive from the ill-posed NNLS/CONTIN
inversion (invariant 8 already forbids an SE there), and the ``/R^6``-type division
**amplifies** small-R noise into large spurious small-particle populations. They are
**report-only** and carry the model-dependence + amplified-noise caveats. ISO 22412:2017
Sec. 9.1 goes further and *deprecates* number distributions from DLS; this capability is
offered (commercial software provides it) but honestly labelled, never presented as a
measurement.

Validity criteria (documented defaults, overridable -- invariant 4)
------------------------------------------------------------------
From Bohren & Huffman 1983 Ch. 6: RDG requires ``|m-1| << 1`` (Eq. 6.2) and
``k*d*|m-1| << 1`` (Eq. 6.3); for a sphere ``d = 2R`` gives ``2*x*|m-1| << 1``. B&H p.464
notes errors grow as ``|m-1| -> ~0.1`` (citing Kerker 1969 p.428), so ``0.1`` is used as
the documented rendering of "<<1". The Rayleigh point-scatterer limit is ``x <= ~0.1``.
Past the sphere form factor's first minimum (``qR ~ 4.4934``) the RDG per-particle
intensity passes through zero and the ``/I1`` inversion is unstable, so that is a third RDG
gate (the same first-zero physics as ``analysis.sls.SPHERE_FIRST_MINIMUM_QRG``). **Mie has
no upper validity bound** (it is the general case); its only gate is that ``m`` is required
and a convergence/large-x ceiling ``DEFAULT_MIE_X_MAX`` (past which the series is expensive
and DLS sizing is moot). Where the contrast is within the RDG limit an informational note
says the cheaper RDG model agrees -- it is a note, never a refusal.

Design contract: pure NumPy/math, no GUI/IO/mutation (invariant 1); no material-, solvent-,
or instrument-specific constant (invariant 3). The solute refractive index, ``m``, model
choice, and cutoffs are all user-supplied.

References
----------
  Bohren, C. F., & Huffman, D. R. (1983). Absorption and Scattering of Light by Small
    Particles, Ch. 6 (RDG validity criteria, Eqs. 6.2/6.3), Sec. 4.3-4.4 (full Mie).
  Guinier, A., & Fournet, G. (1955); Pedersen, J. S. (1997), Eq. 55 (sphere form factor).
  Wiscombe, W. J. (1980) (Mie numerical algorithm -- via analysis.mie).
  ISO 22412:2017, Sec. 3 (intensity/volume/number-weighted PSD) + Sec. 9.1 (caution).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from analysis import mie
from analysis.form_factor import sphere_P_from_u
from analysis.dls.distributions import DistributionResult, scan_distribution_peaks
from physics import constants as phys


# --- weighting keys (one spelling shared by engine, controller, GUI, export) ---
INTENSITY = 'intensity'
VOLUME = 'volume'
SURFACE = 'surface'
NUMBER = 'number'
WEIGHTINGS: Tuple[str, ...] = (INTENSITY, VOLUME, SURFACE, NUMBER)
WEIGHTING_LABELS = {
    INTENSITY: 'Intensity', VOLUME: 'Volume', SURFACE: 'Surface', NUMBER: 'Number',
}

# --- sphere optical-model keys ---
RAYLEIGH = 'rayleigh'
RDG = 'rdg'
MIE = 'mie'
OPTICAL_MODELS: Tuple[str, ...] = (RAYLEIGH, RDG, MIE)
OPTICAL_MODEL_LABELS = {RAYLEIGH: 'Rayleigh point', RDG: 'RDG sphere', MIE: 'Mie sphere'}

# --- documented default validity cutoffs (overridable function parameters, invariant 4) ---
# Rayleigh point-scatterer size-parameter limit x = 2*pi*R/lambda_medium.
DEFAULT_RAYLEIGH_X_MAX = 0.1
# RDG contrast bounds: |m-1| <= 0.1 (B&H Eq. 6.2) and 2*x*|m-1| <= 0.1 (Eq. 6.3, sphere).
# 0.1 is B&H p.464's "large errors begin" figure -- the documented reading of "<<1".
DEFAULT_RDG_M1_MAX = 0.1
DEFAULT_RDG_2XM1_MAX = 0.1
# Mie is the GENERAL homogeneous-sphere case -- no upper contrast/size *validity* bound
# (it reduces to Rayleigh as x -> 0 and RDG as m -> 1). Its only gate is (i) m is REQUIRED
# (the Mie arithmetic uses it, unlike RDG where m only gates), and (ii) a convergence/large-x
# ceiling: past it the series has ~x + 4 x^(1/3) + 2 terms, DLS sizing is moot (geometric
# optics), and the cost is wasteful. 100 is a generous documented default -- at 532 nm in
# water (n = 1.33), x = 100 is a geometric RADIUS of ~6.4 um (~12.7 um diameter) --
# overridable, invariant 4.
DEFAULT_MIE_X_MAX = 100.0
# Mie minimum-avoidance gate (the general-case analog of the RDG first-minimum gate
# rdg_qr_max): near a Mie form-factor minimum the per-particle intensity |S1(theta)|^2 dips
# toward zero, so N = w_I / I1 spikes on noise -- exactly the instability the RDG branch's
# rdg_qr_max prevents. A bin is flagged when its NORMALIZED angular intensity
# P(theta) = |S1(theta)|^2 / |S1(0)|^2 falls below this floor -- i.e. it is suppressed by the
# ANGULAR form factor (a minimum), NOT merely small (a small particle has P ~ 1 and is the
# legitimate weak-scatterer the reweighting is FOR, so it is NOT flagged). Default 0.01 (a bin
# scattering < 1% of its forward intensity at this angle). Documented, overridable (invariant 4).
DEFAULT_MIE_P_MIN = 0.01
# Purely INFORMATIONAL Mie -> RDG contrast overlap: where |m-1| falls within the cheaper
# RDG limit, a note tells the user the RDG sphere model agrees for small-to-moderate
# particles (NOT a validity refusal -- Mie is correct there too). A parallel size-based
# Mie -> Rayleigh hand-back (small x) was never wired into any note; only this contrast
# half exists.
DEFAULT_MIE_HANDBACK_M1 = 0.1
# First zero of the sphere amplitude 3(sin u - u cos u)/u^3 at u = qR = 4.49340945790906
# (first root of tan u = u = first zero of the spherical Bessel j1). Past it P(qR) passes
# through zero and I1 -> 0, making the /I1 inversion unstable -- the RDG qR gate. Same
# physics as analysis.sls.SPHERE_FIRST_MINIMUM_QRG (there expressed in qRg).
SPHERE_FIRST_MINIMUM_QR = 4.49340945790906


class OpticalModelError(ValueError):
    """Reweighting refused (non-spherical model, missing solute RI, or on_invalid='refuse'
    with out-of-range bins)."""


@dataclass
class WeightedDistributionResult:
    """An intensity distribution re-expressed under a volume/surface/number weighting.

    Deliberately carries NO uncertainty (no ``*_se`` / ``se_estimator``) and NO
    ``*_reliable`` / ``reliability_note`` field: reweighted distributions are report-only
    (invariant 8 -- ill-posed inversion + small-R noise amplification), so it is excluded
    from the SE / reliability result families by construction. It is a *derived,
    model-dependent* view, distinct from the measured intensity ``DistributionResult`` it
    comes from (invariant 7) -- its provenance (``weighting``, ``optical_model``,
    ``assumed_shape``, ``m``) travels into display AND export.
    """
    source_method: str                # the intensity method it derives from ('contin'...)
    weighting: str                    # 'intensity' | 'volume' | 'surface' | 'number'
    optical_model: Optional[str]      # 'rayleigh' | 'rdg' | 'mie' | None (intensity: no model)
    assumed_shape: str                # always 'sphere' (spheres-only reweighting)
    m: Optional[float]                # relative refractive index n_solute/n_medium (RDG + Mie)
    # the grid + reweighted weights (paired with rh_grid_nm; sum = 1)
    rh_grid_nm: np.ndarray
    weights: np.ndarray               # normalised weights under `weighting`
    valid_mask: np.ndarray            # per-bin: True where the model is in validity range
    # summary statistics under this weighting
    mean_rh_nm: float
    peak_rh_nm: float
    # validity summary
    all_valid: bool
    n_invalid_bins: int
    # context echoed for display/export
    n_solute: Optional[float] = None
    n_medium: Optional[float] = None
    wavelength_nm: Optional[float] = None
    q_m_inv: float = float('nan')
    # passive, glyph-less notices for the GUI (model-dependence + validity)
    notes: tuple = ()
    # The SOURCE fit's convergence, copied from DistributionResult.success: the summary
    # values here are re-expressions of that fit's numbers, so the flag must travel with
    # them (a False means the underlying lognormal returned its seed, not a fit).
    source_success: bool = True


# ---------------------------------------------------------------------------
# Per-particle sphere scattering + validity
# ---------------------------------------------------------------------------

def size_parameter(r_nm, wavelength_nm: float, n_medium: float) -> np.ndarray:
    """Size parameter x = 2*pi*R/lambda, with lambda = lambda0/n_medium the IN-MEDIUM
    wavelength (Bohren & Huffman). R and lambda0 in nm -> x dimensionless."""
    r = np.asarray(r_nm, dtype=float)
    return 2.0 * math.pi * r * n_medium / wavelength_nm


def _mie_cos_theta(q_nm_inv: float, k_medium_nm_inv: float) -> float:
    """cos(theta) of the scattering angle from q and the in-medium wavenumber k.

    q = 2 k sin(theta/2)  ->  sin(theta/2) = q/(2k),  cos(theta) = 1 - 2 sin^2(theta/2).
    Clamped to [-1, 1] against float round-off at back-scatter (q -> 2k).
    """
    sin_half = q_nm_inv / (2.0 * k_medium_nm_inv)
    sin_half = min(1.0, max(0.0, sin_half))
    return 1.0 - 2.0 * sin_half * sin_half


def sphere_per_particle_intensity(model: str, r_nm, q_nm_inv: float, *,
                                  wavelength_nm: Optional[float] = None,
                                  n_medium: Optional[float] = None,
                                  m: Optional[float] = None) -> np.ndarray:
    """Relative per-particle scattered intensity ``I1(R, q)`` of a homogeneous sphere.

    Returns I1 in ARBITRARY relative units (only ratios across the size grid matter for the
    reweighting, so every size-independent constant -- including the RDG contrast factor
    ``(m^2 - 1)^2`` -- is dropped). ``r_nm`` is the geometric radius R (= Rh for a sphere),
    ``q_nm_inv`` the scattering vector in nm^-1 (so qR is dimensionless).

      * ``rayleigh``: I1 ∝ R^6 (point scatterer). No optical context needed.
      * ``rdg``:      I1 ∝ R^6 * P(qR), P the sphere form factor (V^2 P, V ∝ R^3). No context.
      * ``mie``:      I1 = |S1(x, m, theta)|^2, the full Mie VV per-particle intensity
                      (``analysis.mie``). REQUIRES ``wavelength_nm``, ``n_medium`` (to form the
                      per-bin size parameter x = 2*pi*n_medium*R/lambda and the angle theta from
                      q) and ``m`` (the Mie arithmetic uses it). Reduces to the RDG law as m -> 1.

    Raises OpticalModelError for a non-spherical / unknown model, or Mie without its context.
    """
    r = np.asarray(r_nm, dtype=float)
    if model == RAYLEIGH:
        return r ** 6
    if model == RDG:
        return r ** 6 * sphere_P_from_u(q_nm_inv * r)
    if model == MIE:
        if not (wavelength_nm and n_medium and m is not None and math.isfinite(m)):
            raise OpticalModelError(
                "Mie reweighting needs wavelength_nm, n_medium, and m (= n_solute/n_medium); "
                "one is missing.")
        k = phys.optical_wavevector_k(wavelength_nm, n_medium)   # nm^-1
        cos_theta = _mie_cos_theta(q_nm_inv, k)
        x = k * r                                          # per-bin size parameter
        # Per-bin Mie |S1|^2. x varies across bins; theta (from q) is common to the whole
        # distribution, so a Python loop over bins is fine (~100 bins, each a short series).
        return np.array([float(mie.mie_phase_vv(float(xi), m, cos_theta))
                         if (math.isfinite(xi) and xi > 0) else 0.0
                         for xi in np.atleast_1d(x)],
                        dtype=float).reshape(r.shape)
    raise OpticalModelError(
        f"reweighting is well-posed only for spheres; got optical model {model!r} "
        f"(expected one of {OPTICAL_MODELS}).")


def sphere_validity(
    model: str,
    r_nm,
    q_nm_inv: float,
    *,
    wavelength_nm: float,
    n_medium: float,
    m: Optional[float] = None,
    rayleigh_x_max: float = DEFAULT_RAYLEIGH_X_MAX,
    rdg_m1_max: float = DEFAULT_RDG_M1_MAX,
    rdg_2xm1_max: float = DEFAULT_RDG_2XM1_MAX,
    rdg_qr_max: float = SPHERE_FIRST_MINIMUM_QR,
    mie_x_max: float = DEFAULT_MIE_X_MAX,
    mie_p_min: float = DEFAULT_MIE_P_MIN,
) -> np.ndarray:
    """Per-bin boolean validity mask for the chosen sphere model.

    Rayleigh: valid where x = 2*pi*R/lambda <= ``rayleigh_x_max`` (point-scatterer limit).
    RDG: valid where ``|m-1| <= rdg_m1_max`` AND ``2*x*|m-1| <= rdg_2xm1_max`` (B&H Ch. 6
    Eqs. 6.2/6.3) AND ``qR <= rdg_qr_max`` (inside the sphere form factor's first minimum,
    past which the /I1 inversion is unstable). ``m`` is REQUIRED for RDG -- it is the only
    thing the solute refractive index feeds (the contrast cancels in the reweighting itself).
    Mie: the GENERAL case -- no upper contrast/size validity bound (it reduces to Rayleigh
    and RDG in their limits); valid where ``x <= mie_x_max`` (the convergence/large-x ceiling)
    AND the normalized angular intensity ``P(theta) = |S1|^2/|S1(0)|^2 >= mie_p_min`` (the
    general-case analog of the RDG first-minimum gate: near a Mie form-factor minimum the
    /I1 reweighting spikes on noise). ``m`` is REQUIRED (the Mie arithmetic uses it). The
    cheaper-limit overlap is surfaced as an informational note, not a refusal.

    Raises OpticalModelError for a non-sphere model, or RDG/Mie without ``m``.
    """
    r = np.asarray(r_nm, dtype=float)
    x = size_parameter(r, wavelength_nm, n_medium)
    if model == RAYLEIGH:
        return x <= rayleigh_x_max
    if model == RDG:
        if m is None or not math.isfinite(m):
            raise OpticalModelError(
                "RDG reweighting needs the solute refractive index (to form m = "
                "n_solute/n_medium and check the |m-1| validity bounds); it is unset.")
        m1 = abs(m - 1.0)
        contrast_ok = m1 <= rdg_m1_max                     # scalar, applies to all bins
        per_bin = (2.0 * x * m1 <= rdg_2xm1_max) & (q_nm_inv * r <= rdg_qr_max)
        return per_bin & bool(contrast_ok)
    if model == MIE:
        if m is None or not math.isfinite(m):
            raise OpticalModelError(
                "Mie reweighting needs the solute refractive index (to form m = "
                "n_solute/n_medium, which enters the Mie arithmetic); it is unset.")
        # x ceiling AND the minimum-avoidance gate: flag bins whose normalized angular
        # intensity P(theta) is below the floor -- i.e. suppressed by the ANGULAR form factor
        # (a Mie minimum, where N = w_I/I1 spikes), NOT merely small (a small particle has
        # P ~ 1 and stays valid -- it is the legitimate weak scatterer the reweighting is for).
        k = phys.optical_wavevector_k(wavelength_nm, n_medium)   # nm^-1
        cos_theta = _mie_cos_theta(q_nm_inv, k)
        p_theta = np.array(
            [float(mie.mie_form_factor(float(xi), m, cos_theta))
             if (math.isfinite(xi) and xi > 0) else 1.0
             for xi in np.atleast_1d(x)], dtype=float).reshape(r.shape)
        return (x <= mie_x_max) & (p_theta >= mie_p_min)
    raise OpticalModelError(
        f"reweighting is well-posed only for spheres; got optical model {model!r} "
        f"(expected one of {OPTICAL_MODELS}).")


# ---------------------------------------------------------------------------
# Reweighting
# ---------------------------------------------------------------------------

def _moment_exponent(weighting: str) -> int:
    """Power of R that multiplies the number distribution for each weighting."""
    return {NUMBER: 0, SURFACE: 2, VOLUME: 3}[weighting]


def reweight_distribution(
    distribution: DistributionResult,
    weighting: str,
    model: Optional[str] = None,
    *,
    n_medium: float,
    wavelength_nm: float,
    n_solute: Optional[float] = None,
    rayleigh_x_max: float = DEFAULT_RAYLEIGH_X_MAX,
    rdg_m1_max: float = DEFAULT_RDG_M1_MAX,
    rdg_2xm1_max: float = DEFAULT_RDG_2XM1_MAX,
    rdg_qr_max: float = SPHERE_FIRST_MINIMUM_QR,
    mie_x_max: float = DEFAULT_MIE_X_MAX,
    mie_p_min: float = DEFAULT_MIE_P_MIN,
    on_invalid: str = 'flag',
) -> WeightedDistributionResult:
    """Re-express an intensity ``DistributionResult`` under a volume/surface/number weighting.

    Parameters
    ----------
    distribution : DistributionResult
        The measured intensity-weighted distribution (NNLS / CONTIN / lognormal); its
        ``rh_grid_nm`` (= geometric R for spheres) and ``weights`` are consumed, unchanged.
    weighting : str
        'intensity' (identity pass-through), 'volume', 'surface', or 'number'.
    model : str, optional
        'rayleigh', 'rdg', or 'mie'. Required for any non-intensity weighting; a non-spherical
        / unknown model is refused (spheres-only). Ignored for 'intensity'.
    n_medium, wavelength_nm : float
        Medium refractive index and vacuum wavelength (nm) -- set the size parameter x and,
        for RDG/Mie, m = n_solute/n_medium.
    n_solute : float, optional
        Solute (particle) refractive index. REQUIRED for RDG and Mie (forms m); unused for
        Rayleigh. For RDG it only gates validity; for Mie it enters the scattering arithmetic.
    rayleigh_x_max, rdg_m1_max, rdg_2xm1_max, rdg_qr_max, mie_x_max : float
        Documented validity cutoffs (invariant 4). See module docstring / `sphere_validity`.
    on_invalid : str
        'flag' (default): always compute; mark out-of-range bins in ``valid_mask`` and note
        them (no silent R^6 -- the flag is the honesty). 'refuse': raise OpticalModelError if
        any bin is out of range. A non-spherical model is refused under BOTH.

    Returns
    -------
    WeightedDistributionResult  (report-only; carries no +/-)

    Raises
    ------
    OpticalModelError
        Non-spherical / unknown model, RDG without a solute RI, or on_invalid='refuse' with
        out-of-range bins.
    ValueError
        Unknown weighting.
    """
    if weighting not in WEIGHTINGS:
        raise ValueError(
            f"unknown weighting {weighting!r}; expected one of {WEIGHTINGS}.")
    if on_invalid not in ('flag', 'refuse'):
        raise ValueError(f"on_invalid must be 'flag' or 'refuse', got {on_invalid!r}.")

    # np.array(..., dtype=float), never np.asarray: the source is already float64, so
    # asarray is a no-op and the result would alias distribution.rh_grid_nm / .weights --
    # two persisted objects sharing one buffer (architecture invariant 1).
    rh = np.array(distribution.rh_grid_nm, dtype=float)
    w_I = np.array(distribution.weights, dtype=float)
    q_m = float(distribution.q_m_inv)
    q_nm = q_m * 1e-9                                   # nm^-1 for qR

    # -- intensity: identity pass-through (uniform result type; no model needed) --
    if weighting == INTENSITY:
        weights = _normalise(w_I)
        return _assemble(
            distribution, weighting, None, None, rh, weights,
            np.ones(rh.shape, dtype=bool), None, n_medium, wavelength_nm, q_m,
            note_lines=[])

    # -- non-intensity: a sphere model is mandatory --
    if model is None:
        raise OpticalModelError(
            f"a sphere model ('rayleigh', 'rdg', or 'mie') is required for {weighting} "
            "weighting.")
    m = None
    if model in (RDG, MIE):
        if n_solute is None or not (math.isfinite(n_solute) and n_solute > 0):
            raise OpticalModelError(
                f"{OPTICAL_MODEL_LABELS.get(model, model)} reweighting needs the solute "
                "refractive index as a positive finite number (hand-entered on the Data "
                "tab, like dn/dc); it is unset or unusable. "
                "Use the Rayleigh model for very small particles, or enter the solute "
                "refractive index.")
        m = float(n_solute) / float(n_medium)

    valid = sphere_validity(
        model, rh, q_nm, wavelength_nm=wavelength_nm, n_medium=n_medium, m=m,
        rayleigh_x_max=rayleigh_x_max, rdg_m1_max=rdg_m1_max,
        rdg_2xm1_max=rdg_2xm1_max, rdg_qr_max=rdg_qr_max, mie_x_max=mie_x_max,
        mie_p_min=mie_p_min)
    all_valid = bool(valid.all())
    if on_invalid == 'refuse' and not all_valid:
        raise OpticalModelError(
            f"{int((~valid).sum())} of {valid.size} size bins fall outside the {model!r} "
            f"model's validity range (on_invalid='refuse').")

    # number distribution N ∝ w_I / I1, then the weighting moment N * R^p. Mie's I1 needs the
    # optical context (wavelength/n_medium form the per-bin x and the angle; m enters the
    # series); Rayleigh/RDG ignore the extra kwargs.
    i1 = sphere_per_particle_intensity(
        model, rh, q_nm, wavelength_nm=wavelength_nm, n_medium=n_medium, m=m)
    with np.errstate(divide='ignore', invalid='ignore'):
        number = np.where(np.isfinite(i1) & (i1 > 0), w_I / i1, 0.0)   # I1=0 at a P(qR) zero
    number = np.where(np.isfinite(number), number, 0.0)
    out = number * rh ** _moment_exponent(weighting)
    # ZERO the out-of-validity bins before normalising (flag mode). Otherwise a bin past
    # the RDG second minimum -- where P(qR) is small-but-positive, so N = w_I/I1 spikes --
    # would survive the i1>0 guard and, once renormalised, DOMINATE the reported weights and
    # the mean/peak summary, corrupting the VALID part of the distribution rather than just
    # marking a flagged tail. "Flag-and-show" means show the valid distribution with the
    # invalid Rh region marked (valid_mask + the GUI overlay), NOT apply the wrong law there
    # (no silent R^6 / divide-by-tiny-P outside validity). on_invalid='refuse' already
    # raised above; a fully-invalid distribution normalises to all-zero (empty + flagged).
    out = np.where(valid, out, 0.0)
    weights = _normalise(out)

    note_lines = _model_note(weighting, model, m, valid)
    return _assemble(
        distribution, weighting, model, m, rh, weights, valid, n_solute,
        n_medium, wavelength_nm, q_m, note_lines=note_lines)


def _normalise(w: np.ndarray) -> np.ndarray:
    """Normalise to sum 1 for display; a degenerate all-zero solution stays zero.

    Copies (np.array, never asarray) so the degenerate return is this function's OWN array,
    never the caller's `w` handed straight back (architecture invariant 1).
    """
    w = np.array(w, dtype=float)
    total = float(w.sum())
    return w / total if (math.isfinite(total) and total > 0) else w


def _model_note(weighting: str, model: Optional[str], m: Optional[float],
                valid: np.ndarray) -> List[str]:
    """The one neutral model-dependence note + a validity line when bins are out of range
    (+ an informational Mie->RDG overlap line when the contrast is within the RDG limit)."""
    label = WEIGHTING_LABELS[weighting]
    model_label = OPTICAL_MODEL_LABELS.get(model, model) if model else 'intensity'
    m_txt = f", m = {m:.4g}" if m is not None else ""
    lines = [
        f"{label} weighting is a derived, model-dependent view ({model_label}"
        f"{m_txt}), not a measurement; it re-weights the same intensity fit and amplifies "
        f"small-size noise."
    ]
    n_bad = int((~valid).sum())
    if n_bad:
        lines.append(
            f"{n_bad} of {valid.size} size bins fall outside the {model_label} model's "
            f"validity range and are flagged; treat them as unreliable.")
    # Informational (not a refusal): Mie is the general case, but where the contrast is low
    # enough the cheaper RDG limit agrees -- tell the user they need not pay for Mie there.
    if model == MIE and m is not None and abs(m - 1.0) <= DEFAULT_MIE_HANDBACK_M1:
        lines.append(
            f"This system's contrast is low (|m-1| = {abs(m - 1.0):.3g} <= "
            f"{DEFAULT_MIE_HANDBACK_M1:g}); the cheaper RDG sphere model agrees with Mie in "
            "this range for small-to-moderate particles.")
    return lines


def _assemble(distribution, weighting, model, m, rh, weights, valid, n_solute,
              n_medium, wavelength_nm, q_m, *, note_lines) -> WeightedDistributionResult:
    """Build a WeightedDistributionResult with its summary statistics."""
    total = float(weights.sum())
    if math.isfinite(total) and total > 0 and rh.size:
        mean_rh = float(np.sum(weights * rh))
        peak_rh = float(rh[int(np.argmax(weights))])
    else:
        mean_rh = peak_rh = float('nan')
    return WeightedDistributionResult(
        source_method=distribution.method, weighting=weighting, optical_model=model,
        assumed_shape='sphere', m=m, rh_grid_nm=rh, weights=weights,
        valid_mask=np.asarray(valid, dtype=bool),
        mean_rh_nm=mean_rh, peak_rh_nm=peak_rh,
        all_valid=bool(np.all(valid)), n_invalid_bins=int(np.sum(~np.asarray(valid, bool))),
        n_solute=n_solute, n_medium=n_medium, wavelength_nm=wavelength_nm, q_m_inv=q_m,
        notes=tuple(note_lines),
        source_success=getattr(distribution, 'success', True))


# ---------------------------------------------------------------------------
# Peaks for the reweighted view (same scan as the intensity distribution)
# ---------------------------------------------------------------------------

@dataclass
class WeightedPeak:
    """One resolved population in a reweighted distribution. ``weight_fraction`` is the
    fraction under that peak's lobe IN THE CHOSEN WEIGHTING (number/volume/surface), not an
    intensity fraction."""
    rh_nm: float
    weight_fraction: float
    is_dominant: bool


def find_weighted_peaks(result: WeightedDistributionResult,
                        min_weight_fraction: float = 0.05) -> List[WeightedPeak]:
    """Resolve the peaks of a reweighted distribution (reuses the intensity distribution's
    peak scan -- one definition for the whole distribution family). A view whose SOURCE
    fit did not converge has no peaks, same refusal as find_distribution_peaks."""
    if getattr(result, 'source_success', True) is False:
        return []
    scanned, rh, _w = scan_distribution_peaks(
        result.rh_grid_nm, result.weights, min_weight_fraction)
    if not scanned:
        return []
    peaks = [WeightedPeak(rh_nm=float(rh[i]), weight_fraction=frac, is_dominant=False)
             for i, frac, _lo, _hi in scanned]
    dominant = max(range(len(peaks)), key=lambda j: peaks[j].weight_fraction)
    peaks[dominant].is_dominant = True
    return peaks
