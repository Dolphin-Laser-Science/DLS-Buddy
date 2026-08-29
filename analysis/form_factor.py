"""
analysis/form_factor.py
=======================

Single-particle form factors P(q) for static light scattering, and the size
conversions that go with them.

These are the closed-form scattering form factors used by the full-angular
``form_factor_analysis`` fit in :mod:`analysis.sls` (Phase 1a of the weighted-
distributions / form-factor spec). The module carries each of the two analytic forms
(sphere, Debye) TWICE, deliberately: a scalar pair, ``sphere_form_factor`` /
``debye_form_factor``, is the regression suite's INDEPENDENT oracle transcription of the
textbook forms (its callers are ``tests/fixtures/synthetic_sls.py``'s forward model and the
crossover-ladder tests in ``tests/test_optical_model.py``), kept apart from the production
engine so the SLS round-trip tests are never checked against the code under test; the
vectorized pair, ``sphere_P_from_u`` / the Debye branch of ``form_factor_P``, is the
engine's own definition (see ``sphere_P_from_u``'s docstring). The two pairs share only
the small-argument switch constants and the series polynomial below them, so a misplaced
switch would be common to both -- which is why the ladder tests check each pair against
an independent closed-form reference rather than one pair against the other.

Four models are provided:

  - **Sphere** (solid homogeneous sphere):
        P(u) = [3 (sin u - u cos u) / u^3]^2,   u = q R,
    with R the geometric radius and Rg^2 = (3/5) R^2 (Rayleigh-Debye-Gans /
    Guinier & Fournet 1955; Pedersen 1997 Eq. 55).
  - **Gaussian coil** (Debye function):
        P(x) = (2 / x^2)(e^-x - 1 + x),   x = (q Rg)^2,
    returning Rg directly (Debye 1947 Eq. 8; Pedersen 1997 Eq. 70).
  - **Ellipsoid of revolution** (semi-axes R, R, epsilon*R; Phase 2):
        P(q,R,eps) = integral_0^{pi/2} F^2[q r(R,eps,a)] sin a da,
        r(R,eps,a) = R (sin^2 a + eps^2 cos^2 a)^(1/2),
    with F the sphere amplitude (Eq. 55) and the average taken over orientation
    (Guinier; Pedersen 1997 Eq. 61). `eps` is the axial ratio (polar/equatorial):
    eps > 1 prolate, eps < 1 oblate, eps = 1 the sphere. The size is reported as the
    radius of gyration Rg^2 = R^2 (2 + eps^2)/5, so at eps = 1 the model collapses
    exactly onto the sphere P(u) above (R = Rg sqrt(5/3)). See `ellipsoid_P`.
  - **Mie sphere** (the general homogeneous sphere -- large size parameter, non-negligible
    contrast, where RDG and Rayleigh fail; e.g. latex in water):
        P(q) = |S1(theta)|^2 / |S1(0)|^2,   S1 the Mie amplitude function (analysis/mie.py),
    parameterized by Rg (R = Rg sqrt(5/3)) and the relative refractive index m =
    n_solute/n_medium. Unlike the analytic models, m enters the scattering arithmetic itself,
    so `mie_P` also needs the optical context (wavelength, n_medium). As m -> 1 it collapses
    onto the sphere P(u) above -- the continuity that validates it. See `mie_P`.

Sphere, Debye and Mie all reduce to the small-qRg Guinier limit P -> 1 - (q Rg)^2/3
(the ellipsoid does too, independent of eps -- which is why eps is unidentifiable
at small qRg), so a full-P(q) fit extends the valid qRg range of the
Guinier/Zimm/Berry expansions rather than replacing them.

To keep the fit's size parameter physically comparable across models (and directly
comparable to the Rg reported by Zimm/Guinier), both models are parameterized by the
**radius of gyration Rg**; for the sphere the geometric radius R = Rg * sqrt(5/3) is
formed internally and reported alongside.

Design contract: pure NumPy/math, no GUI/IO/mutation (architecture invariant 1); no
material-, solvent-, or instrument-specific constant (invariant 3). The models are a
general optical description selected by the user.

References
----------
  Debye, P. (1947). J. Phys. Colloid Chem. 51(1), 18-32 (Gaussian-coil P(x)).
  Guinier, A., & Fournet, G. (1955). Small-Angle Scattering of X-Rays (sphere P(u),
    Guinier limit).
  Pedersen, J. S. (1997). Adv. Colloid Interface Sci. 70, 171-210 (P(q) catalogue).
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from analysis import mie
from physics import constants as phys

# Model keys, one spelling shared by the engine, controller, and GUI selector.
SPHERE = 'sphere'
DEBYE = 'debye'
ELLIPSOID = 'ellipsoid'
MIE = 'mie'
CORE_SHELL = 'core_shell'
# All known P(q) models (validation + the GUI dropdown ordering). Two axes distinguish them.
# EVALUATION: the sphere and Debye coil are Rg-only and go through the 2-arg `form_factor_P`;
# the ellipsoid (`ellipsoid_P`), Mie (`mie_P`, needs the contrast m) and core-shell
# (`core_shell_P`, needs two radii and two contrasts) each raise through it and have their
# own evaluator. FITTING: a model single-valued in its radius -- sphere, Debye, and Mie given
# m -- is fit by `form_factor_analysis`; one that is often under-determined from a single
# angular curve is fit by a LIST-returning, degeneracy-flagged analysis instead
# (`ellipsoid_form_factor_analysis`, `core_shell_form_factor_analysis`).
FORM_FACTOR_MODELS: Tuple[str, ...] = (SPHERE, DEBYE, ELLIPSOID, MIE, CORE_SHELL)
# The subset `form_factor_P` / `form_factor_analysis` handle with a plain 2-arg P(q, Rg).
# (Mie is fit by `form_factor_analysis` too, but through the m-aware `mie_P`, not `form_factor_P`.)
_RG_ONLY_MODELS: Tuple[str, ...] = (SPHERE, DEBYE)

# Human-facing labels (GUI dropdown, exports, guide). Kept here so the engine owns
# the canonical wording and the GUI never re-spells a model name.
MODEL_LABELS = {
    SPHERE: 'Sphere',
    DEBYE: 'Gaussian coil',
    ELLIPSOID: 'Ellipsoid',
    MIE: 'Mie sphere',
    CORE_SHELL: 'Core–shell sphere',
}

# Rg^2 = (3/5) R^2 for a solid sphere -> R = Rg * sqrt(5/3).
_RG_TO_R = math.sqrt(5.0 / 3.0)

# Small-argument switches for the sphere/Debye scalar and vectorized transcriptions below.
# Just ABOVE a too-low switch the closed form is in its worst cancellation regime:
# 3(sin u - u cos u)/u^3 cancels to ~u^3/3 (relative error ~3*eps/u^2), and
# (2/x^2)(e^-x - 1 + x) cancels to ~x^2/2 (~4*eps/x^2) -- both readable as P > 1, which no
# form factor can be. Measured in double precision against a 40-term series, the two
# branches' errors cross near u ~ 3e-2 and x ~ 1.5e-3; the switches sit below those
# crossings, where the one-term-extended series is exact to double precision and the
# closed form's cancellation error is already under 1e-11 -- deliberately conservative.
_SPHERE_SERIES_U_MAX = 6.0e-3
_DEBYE_SERIES_X_MAX = 3.0e-4

# Fixed-node Gauss-Legendre quadrature for the ellipsoid orientation average (Eq. 61).
# The substitution u = cos a maps integral_0^{pi/2} f(a) sin a da -> integral_0^1 f(u) du
# (a smooth, non-singular integrand), so the sin-a weight is absorbed and the nodes below
# are on [0, 1] with weights summing to 1. Fixed nodes keep P(q; Rg, eps) a smooth,
# deterministic function of its parameters -- essential for the least-squares Jacobian.
# 64 nodes integrate the (mildly oscillatory) sphere kernel to ~1e-12 across the useful
# qRg range; validated against scipy.integrate.quad in the test suite.
_ELLIPSOID_N_NODES = 64
_gl_nodes, _gl_weights = np.polynomial.legendre.leggauss(_ELLIPSOID_N_NODES)  # on [-1, 1]
_ELLIPSOID_U = 0.5 * (_gl_nodes + 1.0)          # quadrature nodes u in [0, 1]
_ELLIPSOID_W = 0.5 * _gl_weights                # weights (sum = 1, so eps=1 -> exact sphere)


def sphere_radius_from_rg(rg_nm: float) -> float:
    """Geometric sphere radius R from the radius of gyration Rg (Rg^2 = 3/5 R^2)."""
    return rg_nm * _RG_TO_R


def sphere_form_factor(u: float) -> float:
    """Solid-sphere form factor P as a function of u = q R (R the sphere radius).

    Uses the series limit P -> 1 - u^2/5 + 3u^4/175 below `_SPHERE_SERIES_U_MAX` to avoid
    the 0/0 cancellation in 3(sin u - u cos u)/u^3 (see the constant's own comment for the
    cancellation-vs-truncation balance that sets the switch).
    """
    if abs(u) < _SPHERE_SERIES_U_MAX:
        u2 = u * u
        return 1.0 - u2 / 5.0 + 3.0 * u2 * u2 / 175.0
    return (3.0 * (math.sin(u) - u * math.cos(u)) / u ** 3) ** 2


def debye_form_factor(x: float) -> float:
    """Debye (Gaussian-coil) form factor P as a function of x = (q Rg)^2.

    Uses the series limit P -> 1 - x/3 + x^2/12 below `_DEBYE_SERIES_X_MAX` to avoid 0/0 in
    (2/x^2)(e^-x - 1 + x) (see the constant's own comment for the switch's placement).
    """
    if abs(x) < _DEBYE_SERIES_X_MAX:
        return 1.0 - x / 3.0 + x * x / 12.0
    return (2.0 / x ** 2) * (math.exp(-x) - 1.0 + x)


def sphere_P_from_u(u: np.ndarray) -> np.ndarray:
    """Vectorized solid-sphere form factor P as a function of u = q R.

    P(u) = [3 (sin u - u cos u) / u^3]^2, with the series limit P -> 1 - u^2/5 + 3u^4/175
    below `_SPHERE_SERIES_U_MAX` to avoid the 0/0 cancellation. The ONE vectorized
    definition of the sphere form factor, shared by `form_factor_P` (SLS P(q) fit,
    parameterized by Rg) and `analysis.optical_model` (DLS RDG reweighting, parameterized
    by the geometric radius R directly) so the two paths can never silently drift apart.
    """
    u = np.asarray(u, dtype=float)
    small = np.abs(u) < _SPHERE_SERIES_U_MAX
    u_safe = np.where(small, 1.0, u)                 # avoid 0**-3 in the big branch
    big = (3.0 * (np.sin(u_safe) - u_safe * np.cos(u_safe)) / u_safe ** 3) ** 2
    u2 = u ** 2
    series = 1.0 - u2 / 5.0 + 3.0 * u2 ** 2 / 175.0
    return np.where(small, series, big)


def form_factor_P(model: str, q_nm_inv: np.ndarray, rg_nm: float) -> np.ndarray:
    """Vectorized form factor P(q) for a model, parameterized by Rg (nm).

    `q_nm_inv` is the scattering vector in nm^-1 (so q*Rg is dimensionless). For the
    sphere the geometric radius R = Rg*sqrt(5/3) is formed internally and u = qR; for
    the Debye coil x = (q Rg)^2. Returns P as a float array the same shape as q.

    Raises ValueError for an unknown model. Rg <= 0 yields P = 1 everywhere (the
    point-scatterer limit) so a degenerate seed never divides by zero upstream.
    """
    q = np.asarray(q_nm_inv, dtype=float)
    if model == ELLIPSOID:
        raise ValueError(
            "the ellipsoid P(q) needs an axial ratio and cannot be evaluated through the "
            "two-parameter form_factor_P; call ellipsoid_P(q, rg, epsilon) instead."
        )
    if model == MIE:
        raise ValueError(
            "the Mie P(q) needs the contrast m and the optical context and cannot be "
            "evaluated through the two-parameter form_factor_P; call "
            "mie_P(q, rg, m, wavelength_nm=..., n_medium=...) instead."
        )
    if model == CORE_SHELL:
        raise ValueError(
            "the core-shell P(q) needs two radii and two contrasts and cannot be evaluated "
            "through the two-parameter form_factor_P; call core_shell_P(q, r_core, r_outer, "
            "m1, m2, wavelength_nm=..., n_medium=...) instead."
        )
    if model not in _RG_ONLY_MODELS:
        raise ValueError(
            f"unknown form-factor model {model!r}; expected one of {_RG_ONLY_MODELS}."
        )
    if not (rg_nm > 0):
        return np.ones_like(q)
    # Vectorized (np.where masks the small-argument series limit and shields the 0/0
    # branch from a divide warning) — this runs on every least-squares iteration across
    # the multi-start fits, so a Python per-element loop of math.sin/cos/exp is wasteful.
    if model == SPHERE:
        return sphere_P_from_u(q * sphere_radius_from_rg(rg_nm))
    x = (q * rg_nm) ** 2
    small = np.abs(x) < _DEBYE_SERIES_X_MAX
    x_safe = np.where(small, 1.0, x)
    big = (2.0 / x_safe ** 2) * (np.exp(-x_safe) - 1.0 + x_safe)
    series = 1.0 - x / 3.0 + x ** 2 / 12.0
    return np.where(small, series, big)


def model_ordinate(P, inv_mw, a2, c):
    """THE form-factor model ordinate ``Kc/ΔR = inv_Mw / P(q) [+ 2·A₂·c]`` — one home.

    Shared by the fit residuals in ``analysis/sls.py`` (normalized ``inv_Mw``,
    per-point ``c``) and the plotting overlay (physical ``1/Mw``, one concentration
    at a time): the function is scale-agnostic, so fit and drawn curve are the same
    float ops in the same order. ``a2 = None`` omits the virial term (the
    apparent / single-concentration case). NaNs in ``P`` propagate — the view
    layer masks its display-only P-floor *before* calling; no display guard lives
    here (they shape what is drawn, never what is fitted).
    """
    ordinate = inv_mw / P
    if a2 is not None:
        ordinate = ordinate + 2.0 * a2 * c
    return ordinate


def model_P(result, qq):
    """P(q) on grid ``qq`` for a fitted form-factor result — the kernel dispatch.

    Duck-typed on the result's fields (``model`` plus that model's parameters:
    ``rg_nm``; Mie adds ``m``/``wavelength_nm``/``n_medium``; the ellipsoid adds
    ``axial_ratio``; core-shell carries ``r_core_nm``/``r_outer_nm``/``m1``/``m2``
    + optics), so the plotting overlay evaluates the SAME kernel with the SAME
    arguments as the fit that produced the result. A term or parameter added to a
    fit must land here too — pinned by the R²-reconstruction parity test
    (``tests/test_sls.py``), which rebuilds each fit's R² from ``model_P`` +
    ``model_ordinate`` + result fields alone.
    """
    if result.model == ELLIPSOID:
        return ellipsoid_P(qq, result.rg_nm, result.axial_ratio)
    if result.model == MIE:
        return mie_P(qq, result.rg_nm, result.m,
                     wavelength_nm=result.wavelength_nm, n_medium=result.n_medium)
    if result.model == CORE_SHELL:
        return core_shell_P(qq, result.r_core_nm, result.r_outer_nm,
                            result.m1, result.m2,
                            wavelength_nm=result.wavelength_nm,
                            n_medium=result.n_medium)
    return form_factor_P(result.model, qq, result.rg_nm)


def ellipsoid_equatorial_radius_from_rg(rg_nm: float, epsilon: float) -> float:
    """Equatorial radius R of an ellipsoid of revolution (semi-axes R, R, eps*R) from Rg.

    Rg^2 = (R^2 + R^2 + (eps R)^2)/5 = R^2 (2 + eps^2)/5  ->  R = Rg sqrt(5/(2 + eps^2)).
    At eps = 1 this gives R = Rg sqrt(5/3), the solid-sphere relation, so the ellipsoid
    model collapses exactly onto the sphere there.
    """
    return rg_nm * math.sqrt(5.0 / (2.0 + epsilon ** 2))


def ellipsoid_semi_axes_from_rg(rg_nm: float, epsilon: float) -> Tuple[float, float]:
    """(equatorial radius R, polar semi-axis eps*R) in nm from Rg and the axial ratio eps."""
    r_eq = ellipsoid_equatorial_radius_from_rg(rg_nm, epsilon)
    return r_eq, epsilon * r_eq


def ellipsoid_P(q_nm_inv: np.ndarray, rg_nm: float, epsilon: float) -> np.ndarray:
    """Ellipsoid-of-revolution form factor P(q) (Pedersen 1997 Eq. 61), parameterized by Rg.

    Semi-axes are (R, R, eps*R) with R the equatorial radius formed internally from Rg and
    the axial ratio `epsilon` (polar/equatorial: eps > 1 prolate, eps < 1 oblate). The
    orientation average

        P(q) = integral_0^{pi/2} F^2[q r(R,eps,a)] sin a da,
        r(R,eps,a) = R (sin^2 a + eps^2 cos^2 a)^(1/2)

    (F the sphere amplitude, Eq. 55; P = <F^2>) is evaluated after the substitution
    u = cos a, which turns it into the smooth, non-singular

        P(q) = integral_0^1 F^2[q R (1 + (eps^2 - 1) u^2)^(1/2)] du,

    integrated by the module's fixed 64-node Gauss-Legendre rule. F^2 is `sphere_P_from_u`
    (the SAME squared sphere amplitude the sphere model uses), so at eps = 1 the argument is
    q R for every node and P collapses exactly onto the sphere P(qR) (weights sum to 1).

    `q_nm_inv` is in nm^-1; returns P as a float array the same shape as q. Rg <= 0 yields
    P = 1 everywhere (point-scatterer limit) so a degenerate seed never divides by zero
    upstream; a non-finite/non-positive epsilon is treated as the sphere (eps = 1).
    """
    q = np.asarray(q_nm_inv, dtype=float)
    if not (rg_nm > 0):
        # Point-scatterer limit P = 1. Match the scalar/array contract of the normal return
        # below (a scalar q yields a Python float, not a 0-d array).
        return np.ones_like(q) if q.ndim else 1.0
    if not (epsilon > 0 and math.isfinite(epsilon)):
        epsilon = 1.0
    r_eq = ellipsoid_equatorial_radius_from_rg(rg_nm, epsilon)
    # r(u) = R sqrt(1 + (eps^2 - 1) u^2); 1 + (eps^2 - 1) u^2 = eps^2 u^2 + (1 - u^2) >= 0
    # for u in [0, 1] and any real eps, so the sqrt is always real.
    r_u = r_eq * np.sqrt(1.0 + (epsilon ** 2 - 1.0) * _ELLIPSOID_U ** 2)   # (n_nodes,)
    # F^2[q r] over the (q, node) grid, then contract the node axis with the weights.
    p_grid = sphere_P_from_u(np.outer(np.atleast_1d(q), r_u))             # (n_q, n_nodes)
    out = p_grid @ _ELLIPSOID_W                                          # (n_q,)
    return out.reshape(q.shape) if q.ndim else float(out[0])


def mie_P(q_nm_inv: np.ndarray, rg_nm: float, m: float, *,
          wavelength_nm: float, n_medium: float) -> np.ndarray:
    """Full-Mie sphere form factor P(q) = |S1(theta)|^2 / |S1(0)|^2 (Bohren & Huffman 1983),
    parameterized by the radius of gyration Rg and the relative refractive index m.

    Unlike the sphere/Debye ``form_factor_P`` (a function of q and Rg only), Mie needs the
    contrast ``m = n_solute/n_medium`` -- which enters the scattering *arithmetic*, not just a
    validity gate -- and the optical context (``wavelength_nm``, ``n_medium``) to form the size
    parameter ``x = 2*pi*n_medium*R/lambda`` and the scattering angle theta from q. The
    geometric radius ``R = Rg*sqrt(5/3)`` is formed internally (Mie is a solid sphere), so the
    fit stays parameterized by Rg for direct comparison with the other models.

    q = 2 k sin(theta/2) with the in-medium wavenumber k = 2*pi*n_medium/lambda, so
    ``cos(theta) = 1 - (q/k)^2/2`` and ``x = k R``. Returns P (P(0) = 1) as a float array the
    same shape as q. As ``m -> 1`` it collapses onto the sphere form factor
    ``[3(sin u - u cos u)/u^3]^2`` with u = qR (``sphere_P_from_u``); Rg <= 0 yields P = 1.

    Reference: Bohren & Huffman 1983 Eqs. 4.56/4.57 (a_n/b_n), 4.74 (S1); see analysis/mie.py.
    """
    q = np.asarray(q_nm_inv, dtype=float)
    if not (rg_nm > 0):
        return np.ones_like(q)
    r_geom = sphere_radius_from_rg(rg_nm)
    k = phys.optical_wavevector_k(wavelength_nm, n_medium)   # nm^-1
    x = k * r_geom
    sin_half = np.clip(q / (2.0 * k), 0.0, 1.0)
    cos_theta = 1.0 - 2.0 * sin_half ** 2
    return np.asarray(mie.mie_form_factor(x, m, cos_theta), dtype=float)


def mie_P_batch(q_nm_inv: np.ndarray, rg_nm: np.ndarray, m: float, *,
                wavelength_nm: float, n_medium: float) -> np.ndarray:
    """``mie_P`` for an ARRAY of Rg values at once (the profile-scan seeding path).

    Row ``i`` equals ``mie_P(q, rg_nm[i], ...)`` to ~machine precision (the batched
    recurrence in ``analysis.mie`` steps every Rg's own scalar series in lockstep --
    asserted in tests). Guard parity with ``mie_P`` exactly: rows with ``rg <= 0`` or
    NaN are the P = 1 point limit, while ``rg = +inf`` raises (the scalar path fails
    loudly in ``mie_n_max`` -- a silent P = 1 row for an overflowed radius could be
    SELECTED as a flat-P seed; code-review 2026-07-23). Returns shape
    ``(rg_nm.size, q.size)``; ``rg_nm`` is flattened like ``q``.
    """
    q = np.asarray(q_nm_inv, dtype=float).ravel()
    rg = np.asarray(rg_nm, dtype=float).ravel()
    k = phys.optical_wavevector_k(wavelength_nm, n_medium)   # nm^-1
    sin_half = np.clip(q / (2.0 * k), 0.0, 1.0)
    cos_theta = 1.0 - 2.0 * sin_half ** 2
    out = np.ones((rg.size, q.size), dtype=float)
    pos = rg > 0                     # scalar guard verbatim: <=0 / NaN -> point limit;
    if np.any(pos):                  # +inf passes and raises inside, like mie_P
        x = k * sphere_radius_from_rg(rg[pos])
        out[pos] = mie.mie_form_factor_batch(x, m, cos_theta)
    return out


# Relative floor below which the core-shell net contrast (the total excess scattering length,
# the denominator of the contrast-weighted Rg^2) counts as "matched" -> Rg is ill-defined and
# reported as nan. Compared against the sum of the absolute core/shell contrast-volume terms,
# so it is scale-free (independent of the units of R and the size of m-1). A documented module
# constant (not a function parameter).
_CORE_SHELL_CONTRAST_MATCH_REL = 1.0e-6


def core_shell_P(q_nm_inv: np.ndarray, r_core_nm: float, r_outer_nm: float, m1: float,
                 m2: float, *, wavelength_nm: float, n_medium: float) -> np.ndarray:
    """Coated (core-shell) sphere form factor ``P(q) = |S1(theta)|^2 / |S1(0)|^2`` (Aden-Kerker
    / Bohren & Huffman 1983 Eq. 8.2), parameterized by the two geometric radii and the two
    FIXED relative refractive indices.

    Unlike the sphere/Debye/Mie models (a single size parameter), the coated sphere carries a
    core radius ``r_core_nm`` inside a shell out to ``r_outer_nm``, with core/shell contrasts
    ``m1 = n_core/n_medium`` and ``m2 = n_shell/n_medium`` (both user-supplied, never fitted --
    invariant 3), which enter the scattering arithmetic. The optical context
    (``wavelength_nm``, ``n_medium``) forms the in-medium wavenumber ``k = 2*pi*n_medium/lambda``,
    the core/outer size parameters ``x = k*r_core``, ``y = k*r_outer``, and the angle from q
    (``q = 2 k sin(theta/2)`` -> ``cos(theta) = 1 - (q/k)^2 / 2``).

    ``q_nm_inv`` is in nm^-1; returns P (P(0) = 1) as a float array the same shape as q. As
    ``m2 -> m1`` it collapses onto the homogeneous Mie form factor of the OUTER sphere (radius
    r_outer). A degenerate geometry (r_outer <= 0, or r_core <= 0, or r_outer < r_core) yields
    P = 1 everywhere (point-scatterer limit) so a degenerate seed never raises upstream.

    Reference: Bohren & Huffman 1983 Eq. 8.2 + Appendix B (BHCOAT); see analysis/mie.py.
    """
    q = np.asarray(q_nm_inv, dtype=float)
    if not (r_outer_nm > 0 and r_core_nm > 0 and r_outer_nm >= r_core_nm):
        return np.ones_like(q)
    k = phys.optical_wavevector_k(wavelength_nm, n_medium)   # nm^-1
    x = k * r_core_nm
    y = k * r_outer_nm
    sin_half = np.clip(q / (2.0 * k), 0.0, 1.0)
    cos_theta = 1.0 - 2.0 * sin_half ** 2
    return np.asarray(mie.coated_mie_form_factor(x, y, m1, m2, cos_theta), dtype=float)


def core_shell_P_batch(q_nm_inv: np.ndarray, r_core_nm: np.ndarray, r_outer_nm: np.ndarray,
                       m1: float, m2: float, *, wavelength_nm: float,
                       n_medium: float) -> np.ndarray:
    """``core_shell_P`` for ARRAYS of (core, outer) radii at once (the profile-scan
    seeding path).

    Row ``i`` equals ``core_shell_P(q, r_core_nm[i], r_outer_nm[i], ...)`` to ~machine
    precision (the batched BHCOAT recurrence in ``analysis.mie`` -- asserted in tests).
    Guard parity with ``core_shell_P`` exactly: degenerate-geometry rows (a radius
    <= 0 or NaN, or ``ro < rc``) are the P = 1 point limit, while an INFINITE radius
    raises (the scalar path fails loudly -- a silent P = 1 row for an overflowed
    radius could be selected as a flat-P seed; code-review 2026-07-23). Returns shape
    ``(r_core_nm.size, q.size)``; the radius arrays are flattened like ``q``.
    """
    q = np.asarray(q_nm_inv, dtype=float).ravel()
    rc = np.asarray(r_core_nm, dtype=float)
    ro = np.asarray(r_outer_nm, dtype=float)
    # Shape check BEFORE flattening: two differently-shaped grids of equal size would
    # otherwise pair up silently misaligned after ravel (focused-review 2026-07-23).
    if rc.shape != ro.shape:
        raise ValueError(f"radius arrays must share a shape, got {rc.shape} vs {ro.shape}.")
    rc = rc.ravel()
    ro = ro.ravel()
    out = np.ones((rc.size, q.size), dtype=float)
    ok = (ro > 0) & (rc > 0) & (ro >= rc)   # scalar guard verbatim: NaN/degenerate ->
    if np.any(ok):                          # point limit; +inf passes and raises inside
        k = phys.optical_wavevector_k(wavelength_nm, n_medium)   # nm^-1
        sin_half = np.clip(q / (2.0 * k), 0.0, 1.0)
        cos_theta = 1.0 - 2.0 * sin_half ** 2
        out[ok] = mie.coated_mie_form_factor_batch(k * rc[ok], k * ro[ok], m1, m2,
                                                   cos_theta)
    return out


def core_shell_contrast_weighted_rg(r_core_nm: float, r_outer_nm: float, m1: float,
                                    m2: float) -> float:
    """Contrast-weighted radius of gyration Rg (nm) of a core-shell sphere -- the scattering
    Rg the coated P(q) exhibits in its small-q (Guinier) limit, and the size reported by the
    fit for comparison with the model-agnostic Zimm/Guinier Rg.

    A two-density object's Guinier Rg is the contrast-weighted second moment. With core
    (radius R1, contrast rho1 proportional to m1 - 1) and shell (R1..R2, contrast rho2
    proportional to m2 - 1), integrating r^2 over each region (integral r^2 dV over a solid
    sphere R = 4 pi R^5 / 5):

        Rg^2 = (3/5) [ (m1-1) R1^5 + (m2-1) (R2^5 - R1^5) ]
                     / [ (m1-1) R1^3 + (m2-1) (R2^3 - R1^3) ].

    The common n_medium factor in each (m_i - 1) = (n_i - n_medium)/n_medium cancels in the
    ratio. Returns ``nan`` when the object is **contrast-matched** (the denominator -- the net
    excess scattering length -- is negligible vs the magnitude of the two terms) or when
    Rg^2 <= 0 (an opposite-sign core/shell contrast profile whose Guinier Rg is imaginary):
    in both regimes a single scattering Rg is not physically defined, so none is reported.

    As ``m2 -> m1`` this collapses to ``sqrt(3/5) * R2`` -- the homogeneous solid-sphere Rg of
    the outer radius -- matching the ``core_shell_P`` -> Mie(outer) continuity.
    """
    r1 = float(r_core_nm)
    r2 = float(r_outer_nm)
    if not (r2 > 0 and r1 >= 0 and r2 >= r1 and math.isfinite(r1) and math.isfinite(r2)):
        return float('nan')
    c1 = float(m1) - 1.0
    c2 = float(m2) - 1.0
    # The powers are built by multiplication, never `**`. The guard above is about `r2`;
    # the expression needs `r2 ** 5`, which reaches the overflow tail at a radius five
    # times smaller -- and on a Python float `**` RAISES there rather than saturating, so
    # r_outer = 1e62 crashed before either finiteness check below could fire. With `*` the
    # overflow saturates to inf and the existing `isfinite(scale)` / `isfinite(rg_sq)`
    # guards withhold, which is what they were written to do. The radii are free
    # least_squares parameters with an infinite upper bound, so nothing upstream caps them.
    r1_3, r2_3 = r1 * r1 * r1, r2 * r2 * r2
    r1_5, r2_5 = r1_3 * r1 * r1, r2_3 * r2 * r2
    shell_vol = r2_3 - r1_3
    shell_r5 = r2_5 - r1_5
    num = c1 * r1_5 + c2 * shell_r5
    den = c1 * r1_3 + c2 * shell_vol
    scale = abs(c1) * r1_3 + abs(c2) * shell_vol          # sum of |contrast * volume| terms
    if (not (math.isfinite(scale) and scale > 0)
            or abs(den) < _CORE_SHELL_CONTRAST_MATCH_REL * scale):
        return float('nan')                               # contrast-matched -> Rg undefined
    rg_sq = 0.6 * num / den
    if not (rg_sq > 0 and math.isfinite(rg_sq)):
        return float('nan')                               # imaginary Guinier Rg (hollow-like)
    return math.sqrt(rg_sq)
