"""
analysis/mie.py
===============

Full **Mie** scattering solution for a homogeneous sphere (Phase 3 of the weighted-
distributions / form-factor spec). This is the *general* optical case -- large size
parameter ``x = 2*pi*n_medium*R/lambda0`` and non-negligible contrast ``m = n_particle /
n_medium`` -- that the Rayleigh (``x <~ 0.1``) and Rayleigh-Debye-Gans (``|m-1| <~ 0.1``)
limits cannot handle. It is what makes the classic aqueous-nanoparticle case work (e.g.
polystyrene latex in water, ``m ~ 1.19``, ``x`` large) instead of being refused.

A **vendored** solver (owner decision, 2026-07-20): a faithful implementation of the
published Bohren & Huffman / Wiscombe algorithm, NOT a home-grown model. Pure NumPy, no
new runtime dependency; ``miepython`` is used only as a *test-time* cross-check oracle
(``requirements-dev.txt``), never imported here. Vendoring keeps one optical core for the
homogeneous sphere and the planned core-shell follow-up, and keeps the packaging/freeze
story lean.

Algorithm (all forms read from the primary-source PDFs, Bohren & Huffman 1983 and
Wiscombe 1980):

* **Scattering coefficients** ``a_n, b_n`` -- Bohren & Huffman 1983 Eqs. 4.56/4.57
  (Riccati-Bessel, permeability-matched), rewritten with the logarithmic derivative
  ``D_n(mx) = psi_n'(mx)/psi_n(mx)`` (the BHMIE / Appendix-A form):
      a_n = [(D_n/m + n/x) psi_n(x) - psi_{n-1}(x)] / [(D_n/m + n/x) xi_n(x) - xi_{n-1}(x)]
      b_n = [(m D_n + n/x) psi_n(x) - psi_{n-1}(x)] / [(m D_n + n/x) xi_n(x) - xi_{n-1}(x)]
  with Riccati-Bessel ``psi_n(z) = z j_n(z)``, ``xi_n(z) = z h_n^(1)(z) = psi_n - i chi_n``
  (``chi_n(z) = -z y_n(z)``).
* **D_n** by the stable **downward** recurrence (Wiscombe 1980):
  ``D_{n-1} = n/y - 1/(D_n + n/y)`` from a high starting order where ``D ~ 0``.
* **psi_n, chi_n** by the upward recurrence from ``sin x``/``cos x`` (real argument x).
* **Series truncation** ``n_max = x + 4 x^(1/3) + 2`` (Wiscombe 1980).
* **Amplitude functions** ``S1(theta), S2(theta)`` -- B&H Eq. 4.74 -- with the angle
  functions ``pi_n, tau_n`` (Eqs. 4.46/4.47) by their upward recurrence.
* **Efficiencies** ``Q_ext, Q_sca, Q_back`` and asymmetry ``g`` -- B&H Eqs. 4.61-4.63
  (the published-table validation targets).

Per-particle angular intensity (the quantity the analysis consumes). A homogeneous sphere
does NOT depolarize, so under vertical incident light with no analyzer (VU
geometry) it scatters purely VV and the per-particle scattered intensity is ``|S1(theta)|^2``
(arbitrary relative units -- only ratios matter to both consumers). This is
* the DLS reweighting per-particle ``I1`` (``analysis.optical_model``), and
* the SLS form factor ``P(q) = |S1(theta)|^2 / |S1(0)|^2`` (``analysis.form_factor``).
As ``m -> 1`` these collapse onto the RDG ``I1 ∝ R^6 P(qR)`` and the sphere form factor
``[3(sin u - u cos u)/u^3]^2`` (``analysis.form_factor.sphere_P_from_u``) -- the continuity
that ties this new code to the already-validated Phase-1a/1b forms (asserted in tests).

**Coated (core-shell) sphere** -- the Aden-Kerker generalization -- is also provided
(``coated_mie_coefficients`` / ``coated_mie_phase_vv`` / ``coated_mie_form_factor`` /
``coated_mie_efficiencies``): a core of relative index ``m1`` and radius ``R_core`` inside a
concentric shell of index ``m2`` out to ``R_outer`` (size parameters ``x = k R_core``,
``y = k R_outer``). The scattered field has the SAME S1/S2/Q form as the homogeneous sphere
(B&H p.183) -- only ``a_n, b_n`` differ -- so the coated functions reuse the shared
``_efficiencies_from_ab`` / ``_phase_vv_from_ab`` / ``_form_factor_from_ab`` machinery. The
coated coefficients follow B&H Eq. (8.2), computed by the numerically stable Appendix-B
BHCOAT recurrence; see the block comment above ``coated_mie_coefficients``.

Design contract: pure NumPy/math, no GUI/IO/mutation (architecture invariant 1); no
material-, solvent-, or instrument-specific constant (invariant 3). ``x``, ``m`` and the
angle are supplied by the caller from user-owned physical parameters.

References
----------
  Bohren, C. F., & Huffman, D. R. (1983). Absorption and Scattering of Light by Small
    Particles, Sec. 4.3-4.4 (Eqs. 4.56/4.57 a_n/b_n, 4.61/4.62 Q, 4.74 S1/S2); Sec. 8.1 +
    Appendix B (coated a_n/b_n Eq. 8.2, BHCOAT algorithm).
  Aden, A. L., & Kerker, M. (1951). J. Appl. Phys. 22, 1242 (original coated-sphere solution).
  Wiscombe, W. J. (1980). Improved Mie scattering algorithms. Appl. Opt. 19, 1505
    (MIEV0: downward D_n recurrence, n_max = x + 4 x^(1/3) + 2 truncation).
  van de Hulst, H. C. (1981). Light Scattering by Small Particles (Mie cross-check).
"""

from __future__ import annotations

import cmath
import math
from typing import Tuple

import numpy as np


def mie_n_max(x: float) -> int:
    """Series-truncation order ``n_max = ceil(x + 4 x^(1/3) + 2)`` (Wiscombe 1980).

    The number of terms needed for the Mie series to converge at size parameter ``x``.
    A floor of 1 keeps the smallest particles (x -> 0, where only the n=1 dipole term
    survives) well-defined.
    """
    if not (x > 0 and math.isfinite(x)):
        raise ValueError(f"size parameter x must be a positive finite number, got {x!r}.")
    return max(1, int(math.ceil(x + 4.0 * x ** (1.0 / 3.0) + 2.0)))


def _log_deriv_downward(z: complex, n_max: int) -> np.ndarray:
    """Logarithmic derivative ``D_n(z) = psi_n'(z)/psi_n(z)``, orders 1..n_max, by the stable
    DOWNWARD recurrence (Wiscombe 1980): ``D_{n-1} = n/z - 1/(D_n + n/z)`` from a high starting
    order where ``D ~ 0`` (a margin above both n_max and |z|). Downward is stable even for
    n >> |z| and, unlike the upward form, never divides by zero at a Riccati-Bessel zero
    (a pole of D) -- essential when a coated fit probes core sizes across those zeros. Returns
    a complex array of length n_max (``D[k]`` is ``D_{k+1}``)."""
    z = complex(z)
    nmx = int(max(n_max, abs(z)) + 16)
    D = np.zeros(nmx + 1, dtype=complex)
    for n in range(nmx, 0, -1):
        D[n - 1] = (n / z) - 1.0 / (D[n] + n / z)
    return D[1:n_max + 1]


def mie_coefficients(x: float, m: complex, n_max: int | None = None
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Mie scattering coefficients ``a_n, b_n`` for a homogeneous sphere.

    Parameters
    ----------
    x : float
        Size parameter ``2*pi*n_medium*R/lambda0`` (> 0).
    m : complex or float
        Relative refractive index ``n_particle / n_medium``. Real for a non-absorbing
        particle (the usual light-scattering case); a complex ``m`` (absorbing) is handled
        by the same complex recurrences.
    n_max : int, optional
        Number of series terms; defaults to the Wiscombe truncation ``mie_n_max(x)``.

    Returns
    -------
    (a, b) : two complex ndarrays of length ``n_max`` (``a[k]`` is ``a_{k+1}``).

    Notes
    -----
    ``a_n, b_n`` are formed from the logarithmic derivative ``D_n(mx)`` (Bohren & Huffman
    Eqs. 4.56/4.57 in BHMIE form). ``D_n`` is built by the stable downward recurrence, the
    real Riccati-Bessel ``psi_n(x), chi_n(x)`` by the upward recurrence.
    """
    x = float(x)
    if not (x > 0 and math.isfinite(x)):
        raise ValueError(f"size parameter x must be a positive finite number, got {x!r}.")
    m = complex(m)
    if not (m.real > 0 and math.isfinite(m.real) and math.isfinite(m.imag)):
        raise ValueError(f"relative index m must have finite parts and Re(m) > 0, got {m!r}.")
    if n_max is None:
        n_max = mie_n_max(x)
    n_max = int(n_max)

    D = _log_deriv_downward(m * x, n_max)    # D_1 .. D_{n_max} of the argument m*x

    a = np.empty(n_max, dtype=complex)
    b = np.empty(n_max, dtype=complex)
    # Real Riccati-Bessel psi_n(x), chi_n(x) by upward recurrence from n = -1, 0.
    #   psi_{-1} = cos x, psi_0 = sin x ;  chi_{-1} = -sin x, chi_0 = cos x
    #   f_n = (2n-1)/x * f_{n-1} - f_{n-2}
    # xi_n = psi_n - i chi_n  (xi_n = z h_n^(1)(z), chi_n = -z y_n).
    psi_nm1, psi_n = math.cos(x), math.sin(x)        # psi_{-1}, psi_0
    chi_nm1, chi_n = -math.sin(x), math.cos(x)       # chi_{-1}, chi_0
    xi_nm1 = complex(psi_n, -chi_n)                  # xi_0
    for n in range(1, n_max + 1):
        psi_prev = psi_n
        chi_prev = chi_n
        psi_n = (2 * n - 1) / x * psi_prev - psi_nm1
        chi_n = (2 * n - 1) / x * chi_prev - chi_nm1
        xi_n = complex(psi_n, -chi_n)
        Dn = D[n - 1]
        fac_a = Dn / m + n / x
        fac_b = m * Dn + n / x
        a[n - 1] = (fac_a * psi_n - psi_prev) / (fac_a * xi_n - xi_nm1)
        b[n - 1] = (fac_b * psi_n - psi_prev) / (fac_b * xi_n - xi_nm1)
        psi_nm1, chi_nm1, xi_nm1 = psi_prev, chi_prev, xi_n
    return a, b


def _pi_tau(cos_theta: np.ndarray, n_max: int) -> Tuple[np.ndarray, np.ndarray]:
    """Angle functions ``pi_n, tau_n`` (B&H Eqs. 4.46/4.47) over an array of ``cos(theta)``.

    Returns two arrays of shape ``(n_max, *cos_theta.shape)``. ``pi_1 = 1``, ``pi_0 = 0``;
    ``pi_{n+1} = ((2n+1) mu pi_n - (n+1) pi_{n-1})/n``; ``tau_n = n mu pi_n - (n+1) pi_{n-1}``.
    """
    mu = np.asarray(cos_theta, dtype=float)
    pi = np.zeros((n_max,) + mu.shape, dtype=float)
    tau = np.zeros((n_max,) + mu.shape, dtype=float)
    pi_nm1 = np.zeros_like(mu)                # pi_0
    pi_n = np.ones_like(mu)                   # pi_1
    for n in range(1, n_max + 1):
        pi[n - 1] = pi_n
        tau[n - 1] = n * mu * pi_n - (n + 1) * pi_nm1
        pi_next = ((2 * n + 1) * mu * pi_n - (n + 1) * pi_nm1) / n
        pi_nm1, pi_n = pi_n, pi_next
    return pi, tau


# ---- pi/tau memoization -------------------------------------------------------------
# During a form-factor fit the angle set is FIXED while the optimizer varies only the size
# parameter, so ``_pi_tau`` -- the hottest loop in a fit -- is recomputed hundreds of times
# for identical (cos_theta, n_max) inputs. See ``_pi_tau_cached`` for the key, slicing and
# thread-safety rules.
_PI_TAU_CACHE: dict = {}
_PI_TAU_CACHE_MAX = 8      # a handful of angle sets (fit grid + forward-appended, DLS
                           # reweighting, plot overlays) -- FIFO-evicted beyond that.


def _pi_tau_cached(mu: np.ndarray, n_max: int) -> Tuple[np.ndarray, np.ndarray]:
    """``_pi_tau`` behind the module cache above. ``mu`` must already be a float ndarray
    (the internal call sites guarantee it). Returns read-only arrays (views on a hit).

    The key is an immutable byte snapshot of the angle array, so a caller mutating its
    array afterwards produces a different key (a correct miss, never a stale hit). One
    entry per angle set stores the masters at the LARGEST ``n_max`` seen; a smaller
    ``n_max`` (it drifts by a few orders as x moves during a fit) is served as a read-only
    slice -- the pi/tau recurrence for order n does not depend on ``n_max``, so the smaller
    result is exactly the top block of the larger. Masters are frozen
    (``writeable=False``) so an accidental consumer write raises instead of silently
    corrupting the cache; consumers only read (multiply/sum). No lock: dict get/set are
    individually atomic under the GIL, so a concurrent recompute is merely redundant, and
    the one check-then-act sequence (the eviction pop) is non-throwing (pop with default,
    ``StopIteration`` swallowed) so the worst outcome stays a redundant recompute."""
    key = (mu.tobytes(), mu.shape)
    hit = _PI_TAU_CACHE.get(key)
    if hit is not None and hit[0].shape[0] >= n_max:
        return hit[0][:n_max], hit[1][:n_max]
    pi, tau = _pi_tau(mu, n_max)
    pi.setflags(write=False)
    tau.setflags(write=False)
    if key not in _PI_TAU_CACHE and len(_PI_TAU_CACHE) >= _PI_TAU_CACHE_MAX:
        try:                                           # FIFO: dicts preserve insertion order
            _PI_TAU_CACHE.pop(next(iter(_PI_TAU_CACHE)), None)   # None: key already popped
        except (StopIteration, RuntimeError):
            # StopIteration: emptied between len() and iter(); RuntimeError: another
            # thread's insert/delete landed between iter() and next() ("dictionary
            # changed size during iteration"). Either way: skip this eviction (the
            # cache may transiently overshoot the cap by one -- harmless).
            pass
    _PI_TAU_CACHE[key] = (pi, tau)
    return pi, tau


def mie_S1_S2(x: float, m: complex, cos_theta) -> Tuple[np.ndarray, np.ndarray]:
    """Complex amplitude functions ``S1(theta), S2(theta)`` (Bohren & Huffman Eq. 4.74).

    ``cos_theta`` may be a scalar or an array; the return shape matches it (scalar -> a
    complex scalar). ``S1 = sum (2n+1)/(n(n+1)) (a_n pi_n + b_n tau_n)``,
    ``S2 = sum (2n+1)/(n(n+1)) (a_n tau_n + b_n pi_n)``.
    """
    a, b = mie_coefficients(x, m)
    n_max = a.size
    mu = np.asarray(cos_theta, dtype=float)
    scalar = (mu.ndim == 0)
    mu2 = np.atleast_1d(mu)
    pi, tau = _pi_tau_cached(mu2, n_max)      # (n_max, N)
    n = np.arange(1, n_max + 1)
    fn = (2 * n + 1) / (n * (n + 1))          # (n_max,)
    fa = (fn[:, None] * a[:, None])
    fb = (fn[:, None] * b[:, None])
    S1 = np.sum(fa * pi + fb * tau, axis=0)
    S2 = np.sum(fa * tau + fb * pi, axis=0)
    if scalar:
        return complex(S1[0]), complex(S2[0])
    return S1.reshape(mu.shape), S2.reshape(mu.shape)


def _efficiencies_from_ab(a: np.ndarray, b: np.ndarray, size_param: float
                          ) -> Tuple[float, float, float, float]:
    """``(Q_ext, Q_sca, Q_back, g)`` from precomputed coefficients ``a_n, b_n`` (B&H Eqs.
    4.61-4.63), normalised by ``size_param`` -- ``x`` for a homogeneous sphere, the OUTER
    ``y`` for a coated sphere (BHCOAT normalises the cross-sections by the outer size
    parameter). Shared by ``mie_efficiencies`` and ``coated_mie_efficiencies`` so the series
    is written once.
    """
    n = np.arange(1, a.size + 1)
    x2 = size_param * size_param
    qext = float((2.0 / x2) * np.sum((2 * n + 1) * np.real(a + b)))
    qsca = float((2.0 / x2) * np.sum((2 * n + 1) * (np.abs(a) ** 2 + np.abs(b) ** 2)))
    qback = float((1.0 / x2) * np.abs(np.sum((2 * n + 1) * ((-1.0) ** n) * (a - b))) ** 2)
    # Asymmetry parameter g = <cos theta>.
    if a.size >= 2:
        term1 = np.sum((n[:-1] * (n[:-1] + 2) / (n[:-1] + 1))
                       * np.real(a[:-1] * np.conj(a[1:]) + b[:-1] * np.conj(b[1:])))
    else:
        term1 = 0.0
    term2 = np.sum(((2 * n + 1) / (n * (n + 1))) * np.real(a * np.conj(b)))
    g = float((4.0 / x2) * (term1 + term2) / qsca) if (math.isfinite(qsca) and qsca > 0) else 0.0
    return qext, qsca, qback, g


def mie_efficiencies(x: float, m: complex) -> Tuple[float, float, float, float]:
    """Return ``(Q_ext, Q_sca, Q_back, g)`` for a homogeneous sphere (B&H Eqs. 4.61-4.63).

    ``Q_ext = (2/x^2) sum (2n+1) Re(a_n + b_n)``,
    ``Q_sca = (2/x^2) sum (2n+1) (|a_n|^2 + |b_n|^2)``,
    ``Q_back = (1/x^2) |sum (2n+1)(-1)^n (a_n - b_n)|^2``,
    ``g Q_sca = (4/x^2)[ sum n(n+2)/(n+1) Re(a_n a*_{n+1} + b_n b*_{n+1})
                          + sum (2n+1)/(n(n+1)) Re(a_n b*_n) ]``.
    These are the published-table validation targets (Wiscombe 1980 / B&H worked examples).
    """
    a, b = mie_coefficients(x, m)
    return _efficiencies_from_ab(a, b, x)


def _mie_S1(a: np.ndarray, b: np.ndarray, mu2: np.ndarray) -> np.ndarray:
    """Complex ``S1(theta)`` over an array ``mu2`` of cos(theta), from precomputed a_n/b_n.

    Only S1 is formed (a homogeneous sphere scatters purely via S1 under VU); S2 is never
    built. Shared by ``mie_phase_vv`` and ``mie_form_factor`` so the coefficient recurrence
    and the angle-function sweep run once for the whole angle set (incl. the forward direction).
    """
    n_max = a.size
    pi, tau = _pi_tau_cached(mu2, n_max)      # (n_max, N)
    n = np.arange(1, n_max + 1)
    fn = (2 * n + 1) / (n * (n + 1))          # (n_max,)
    return np.sum((fn[:, None] * a[:, None]) * pi
                  + (fn[:, None] * b[:, None]) * tau, axis=0)


def _phase_vv_from_ab(a: np.ndarray, b: np.ndarray, cos_theta) -> np.ndarray:
    """Per-particle VV intensity ``|S1(theta)|^2`` from precomputed a_n/b_n. ``cos_theta``
    scalar -> float; array -> array. Shared by the homogeneous and coated ``*_phase_vv``.
    """
    mu = np.asarray(cos_theta, dtype=float)
    S1 = _mie_S1(a, b, np.atleast_1d(mu))
    out = np.abs(S1) ** 2
    return float(out[0]) if mu.ndim == 0 else out.reshape(mu.shape)


def _form_factor_from_ab(a: np.ndarray, b: np.ndarray, cos_theta) -> np.ndarray:
    """Normalised form factor ``P = |S1(theta)|^2 / |S1(0)|^2`` from precomputed a_n/b_n.

    The forward direction (cos = 1) is appended to the angle set and used as the normaliser,
    so the angle-function sweep runs ONCE. ``P(0) = 1`` by construction. Shared by the
    homogeneous and coated ``*_form_factor``. A non-finite/zero forward amplitude (x -> 0
    underflow) returns the all-ones point limit.
    """
    mu = np.asarray(cos_theta, dtype=float)
    mu_all = np.append(np.atleast_1d(mu).ravel(), 1.0)   # angles + the forward direction
    s1_all = np.abs(_mie_S1(a, b, mu_all)) ** 2
    s1_theta, s1_fwd = s1_all[:-1], float(s1_all[-1])
    if not (s1_fwd > 0 and math.isfinite(s1_fwd)):
        return np.ones_like(mu) if mu.ndim else 1.0
    p = s1_theta / s1_fwd
    return float(p[0]) if mu.ndim == 0 else p.reshape(mu.shape)


def mie_phase_vv(x: float, m: complex, cos_theta) -> np.ndarray:
    """Per-particle VV angular intensity ``|S1(theta)|^2`` (arbitrary relative units).

    The quantity a homogeneous sphere scatters under vertical incident light with no
    analyzer (VU geometry -- spheres do not depolarize). Consumed as the DLS reweighting
    per-particle ``I1`` and, after normalising by its ``theta = 0`` value, as the SLS form
    factor. ``cos_theta`` scalar -> float; array -> array. Computes S1 only (not S2).
    """
    a, b = mie_coefficients(x, m)
    return _phase_vv_from_ab(a, b, cos_theta)


def mie_form_factor(x: float, m: complex, cos_theta) -> np.ndarray:
    """Normalised Mie form factor ``P = |S1(theta)|^2 / |S1(0)|^2`` for the SLS P(q) fit.

    ``P(0) = 1`` by construction, so it drops straight into the ordinate
    ``Kc/dR = 1/(Mw P(q)) + 2 A2 c``. As ``m -> 1`` it collapses onto the sphere form
    factor ``[3(sin u - u cos u)/u^3]^2`` with ``u = qR = 2 x sin(theta/2)``
    (``analysis.form_factor.sphere_P_from_u``) -- the continuity check.

    The Mie coefficients and angle-function sweep run ONCE (the forward direction is appended
    to the angle set as the normaliser).
    """
    a, b = mie_coefficients(x, m)
    return _form_factor_from_ab(a, b, cos_theta)


# ============================================================================================
# Batched (grid) homogeneous path -- vectorized over the SIZE-PARAMETER axis.
# ============================================================================================
#
# The per-order recurrences are SEQUENTIAL in n, so they cannot be reduced over n -- but a
# profile scan evaluates the SAME recurrence at many independent size parameters, and those
# are steppable in lockstep (the canonical batching: how miepython / PyMieScatt do it). The
# per-element rules are in ``mie_coefficients_batch``'s docstring.


def _mie_n_max_array(x: np.ndarray) -> np.ndarray:
    """``mie_n_max`` (Wiscombe truncation) over an array of size parameters.

    Deliberately a per-element loop over the SCALAR ``mie_n_max``, not a re-statement of
    the formula with numpy ufuncs: scalar-identical truncation BY CONSTRUCTION, on every
    platform. (A ``np.power``-based twin can differ from CPython's libm pow by a few ulp
    on some SIMD builds, and when the Wiscombe expression straddles an integer the ceil
    flips the batch n_max by 1 vs the scalar path -- code-review 2026-07-23. The loop is
    a few hundred trivial calls next to recurrences costing thousands of times more.)
    """
    x = np.asarray(x, dtype=float)
    if not (np.all(x > 0) and np.all(np.isfinite(x))):
        raise ValueError("all size parameters x must be positive finite numbers.")
    return np.array([mie_n_max(float(v)) for v in x.ravel()], dtype=int).reshape(x.shape)


def _log_deriv_downward_batch(z: np.ndarray, nmx: np.ndarray, n_cols: int) -> np.ndarray:
    """Batched ``_log_deriv_downward``: D_1..D_{n_cols} for an ARRAY of arguments ``z``,
    each element started at its OWN scalar start order ``nmx[i]`` (steps above it are
    masked no-ops, leaving the element at the scalar path's D = 0 starting condition).
    Returns shape ``(z.size, n_cols)``; entries an element's own recurrence never reached
    (``n_cols`` beyond its ``nmx[i]``) stay 0 and must be masked off by the caller."""
    z = np.asarray(z, dtype=complex)
    nmx = np.asarray(nmx)
    D = np.zeros(z.shape, dtype=complex)
    out = np.zeros((z.size, n_cols), dtype=complex)
    for n in range(int(nmx.max()), 0, -1):
        D = np.where(n <= nmx, (n / z) - 1.0 / (D + n / z), D)
        if 2 <= n <= n_cols + 1:
            out[:, n - 2] = D                  # after the step at n, D holds order n-1
    return out


def mie_coefficients_batch(x: np.ndarray, m: complex) -> Tuple[np.ndarray, np.ndarray]:
    """Mie ``a_n, b_n`` for an ARRAY of size parameters at one relative index ``m``.

    Returns two complex arrays of shape ``(x.size, n_cols)`` with
    ``n_cols = max(mie_n_max(x_i))``; row ``i`` equals the scalar
    ``mie_coefficients(x_i, m)`` (to ~machine precision) in its first
    ``mie_n_max(x_i)`` columns and is exactly 0.0 beyond them, so summing a full row
    is summing the element's own scalar series.

    Two per-element rules make that scalar identity hold (asserted at atol=1e-12 in
    ``tests/test_mie.py``):

    * PER-ELEMENT truncation, shared loop bound. Each element keeps its own scalar
      ``mie_n_max(x_i)`` -- its psi/chi/xi state stops advancing and its coefficients are
      exactly 0.0 beyond it -- while the loop runs to the grid maximum. Running a small-x
      element to the largest element's order would BOTH overflow its upward chi recurrence
      (chi_n(x) ~ (2n-1)!!/x^n) and perturb its P at ~1e-10 via real extra-order terms
      (``test_truncation_stability`` pins them below 1e-10, not at machine zero).
    * PER-ELEMENT downward start. ``_log_deriv_downward_batch`` seeds each element's D at
      its own scalar start order ``max(n_max_i, |z_i|) + 16`` (a masked no-op above it), so
      every element runs the same downward sequence as its scalar call -- no reliance on
      contraction from a shared higher start, which is weak where n ~ |z|.

    Frozen lanes recompute a one-step-ahead candidate each iteration (``np.where``
    evaluates both branches) but never compound it -- the discarded candidate is bounded,
    so no overflow.

    Precision caveat (focused-review 2026-07-23): at an order whose a_n/b_n Wronskian
    denominator is NEAR-SINGULAR (the argument close to a Riccati-Bessel zero), the
    condition number amplifies the ~1-ulp scalar-vs-vectorized transcendental
    differences into a per-COEFFICIENT deviation as large as ~1e-9 at isolated (x, m).
    The physically consumed sums (S1, P) stay at ~1e-14 there — the near-pole order's
    weighted contribution is negligible — so the form-factor contract is unaffected;
    only the raw coefficient arrays carry the caveat (no caller reads them directly).
    """
    x = np.asarray(x, dtype=float)
    m = complex(m)
    if not (m.real > 0 and math.isfinite(m.real) and math.isfinite(m.imag)):
        raise ValueError(f"relative index m must have finite parts and Re(m) > 0, got {m!r}.")
    if x.size == 0:                            # np.all(x > 0) is vacuously True for empty
        return np.zeros((0, 0), dtype=complex), np.zeros((0, 0), dtype=complex)
    n_max_arr = _mie_n_max_array(x)            # validates x
    n_cols = int(n_max_arr.max())
    z = m * x
    # Same start-order formula as the scalar path: int(max(n_max, |z|) + 16), per element.
    nmx = (np.maximum(n_max_arr, np.abs(z)) + 16).astype(int)
    D = _log_deriv_downward_batch(z, nmx, n_cols)

    a = np.zeros((x.size, n_cols), dtype=complex)
    b = np.zeros((x.size, n_cols), dtype=complex)
    psi_nm1, psi_n = np.cos(x), np.sin(x)      # psi_{-1}, psi_0
    chi_nm1, chi_n = -np.sin(x), np.cos(x)     # chi_{-1}, chi_0
    xi_nm1 = psi_n - 1j * chi_n                # xi_0
    for n in range(1, n_cols + 1):
        active = n <= n_max_arr
        psi_new = (2 * n - 1) / x * psi_n - psi_nm1
        chi_new = (2 * n - 1) / x * chi_n - chi_nm1
        xi_new = psi_new - 1j * chi_new
        Dn = D[:, n - 1]
        fac_a = Dn / m + n / x
        fac_b = m * Dn + n / x
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            a_col = (fac_a * psi_new - psi_n) / (fac_a * xi_new - xi_nm1)
            b_col = (fac_b * psi_new - psi_n) / (fac_b * xi_new - xi_nm1)
        a[:, n - 1] = np.where(active, a_col, 0.0)
        b[:, n - 1] = np.where(active, b_col, 0.0)
        # Advance the recurrence state only where active (per-element truncation).
        psi_nm1 = np.where(active, psi_n, psi_nm1)
        chi_nm1 = np.where(active, chi_n, chi_nm1)
        psi_n = np.where(active, psi_new, psi_n)
        chi_n = np.where(active, chi_new, chi_n)
        xi_nm1 = np.where(active, xi_new, xi_nm1)
    return a, b


def _mie_S1_batch(a: np.ndarray, b: np.ndarray, mu_all: np.ndarray) -> np.ndarray:
    """Complex ``S1`` for BATCHED coefficients: ``(N, n_cols) @ (n_cols, n_angles)`` --
    one ``pi/tau`` sweep at the shared column count and one matmul per term. Rows'
    zero-padding beyond each element's own ``n_max`` contributes exactly 0 to the sums."""
    n_cols = a.shape[1]
    pi, tau = _pi_tau_cached(mu_all, n_cols)   # (n_cols, n_angles)
    n = np.arange(1, n_cols + 1)
    fn = (2 * n + 1) / (n * (n + 1))           # (n_cols,)
    return (a * fn) @ pi + (b * fn) @ tau


def _form_factor_from_ab_batch(a: np.ndarray, b: np.ndarray, cos_theta) -> np.ndarray:
    """Batched ``_form_factor_from_ab``: normalised P rows from ``(N, n_cols)``
    coefficients. The forward direction is appended per row as the normaliser; a
    non-finite/zero forward amplitude yields that row as the all-ones point limit."""
    mu = np.asarray(cos_theta, dtype=float)
    mu_all = np.append(np.atleast_1d(mu).ravel(), 1.0)   # angles + forward direction
    s1_all = np.abs(_mie_S1_batch(a, b, mu_all)) ** 2    # (N, n_angles + 1)
    s1_theta, s1_fwd = s1_all[:, :-1], s1_all[:, -1]
    ok = np.isfinite(s1_fwd) & (s1_fwd > 0)
    p = np.ones_like(s1_theta)
    p[ok] = s1_theta[ok] / s1_fwd[ok, None]
    return p


# Memory cap for one batched call: rows are processed in chunks so no (rows x n_cols)
# work matrix exceeds this many complex elements (~16 MB each at 1e6). Without it, an
# angle-limited / low-q / mis-scaled dataset can push the largest grid node's n_max to
# 1e4-1e5 orders and a single whole-grid batch to GB-scale transient arrays — a hard
# regression vs. the O(n_max)-per-node scalar loop (code-review 2026-07-23). Chunking
# is exactly results-neutral: every batched quantity is computed per row with
# per-element truncation/starts (rows never mix), so splitting the row axis changes
# nothing but the padding width of the intermediate arrays.
_BATCH_MAX_ELEMENTS = 1_000_000


def _batch_row_chunks(n_rows: int, n_cols_est: int):
    """Row slices for the batched calls: chunks sized so rows x n_cols_est stays under
    ``_BATCH_MAX_ELEMENTS`` (always at least one row per chunk)."""
    rows = max(1, _BATCH_MAX_ELEMENTS // max(1, n_cols_est))
    return [slice(i, i + rows) for i in range(0, n_rows, rows)]


def mie_form_factor_batch(x: np.ndarray, m: complex, cos_theta) -> np.ndarray:
    """Normalised Mie form factor for an ARRAY of size parameters at one ``m``.

    Row ``i`` equals ``mie_form_factor(x_i, m, cos_theta)`` (to ~machine precision).
    Returns shape ``(x.size, n_angles)``. Large batches are processed in row chunks
    (see ``_BATCH_MAX_ELEMENTS``) — results-neutral, memory-bounded.
    """
    x = np.asarray(x, dtype=float)
    # Empty x falls through to mie_coefficients_batch, which still VALIDATES m before
    # its own empty return (an early return here would silently swallow a bad m for an
    # all-masked batch -- focused-review 2026-07-23); only the chunk estimate is skipped.
    if x.size:
        chunks = _batch_row_chunks(x.size, mie_n_max(float(np.max(x))))
        if len(chunks) > 1:
            return np.vstack([mie_form_factor_batch(x[s], m, cos_theta) for s in chunks])
    a, b = mie_coefficients_batch(x, m)
    return _form_factor_from_ab_batch(a, b, cos_theta)


# ============================================================================================
# Coated (core-shell) sphere -- the Aden-Kerker generalization of the Mie coefficients.
# ============================================================================================
#
# A core of relative index m1 and radius R_core inside a concentric shell of index m2 out to
# R_outer. S1/S2 and the efficiencies have the SAME form as the homogeneous sphere (Bohren &
# Huffman 1983 p.183, p.483) -- only a_n, b_n differ -- so the coated functions reuse the
# shared ``*_from_ab`` machinery. Recurrence: see ``coated_mie_coefficients``.


# BHCOAT inner-sphere convergence criterion (Bohren & Huffman 1983 Appendix B, p.484): once
# the core's contribution to the coated coefficients falls below this, the coefficients are
# frozen to the homogeneous outer sphere for all higher orders. The book's value; a documented
# module constant (not a function parameter).
_BHCOAT_DEL = 1.0e-8


def coated_mie_coefficients(x: float, y: float, m1: complex, m2: complex,
                            n_max: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Coated-sphere (core-shell) scattering coefficients ``a_n, b_n`` (Bohren & Huffman 1983
    Eq. 8.2, computed by the Appendix-B BHCOAT recurrence).

    Bohren & Huffman 1983 Eq. (8.2) is exact; it is used here in the numerically stable
    computation-ready form of Appendix B (the BHCOAT algorithm), read from the PDF page
    renders (Core-shell Aden-Kerker / BHCOAT). The A_n, B_n ratio functions depend on the CORE size
    parameter ``x`` only; the outer size parameter ``y`` sets the series length. All
    functions are built by upward recurrence (BHCOAT, unlike BHMIE's downward D_n), with
    the inner-sphere freeze test (DEL) that collapses the coated coefficients onto the
    homogeneous outer sphere once the core contribution is negligible. Exact reductions
    (asserted in ``tests/test_mie.py``): ``m1 = m2`` -> homogeneous sphere (size y, index
    m2); ``R_core -> 0`` -> homogeneous sphere (size y, index m2); ``m2 = 1`` ->
    homogeneous sphere (size x, index m1).

    Parameters
    ----------
    x : float
        CORE size parameter ``2*pi*n_medium*R_core/lambda`` (> 0).
    y : float
        OUTER size parameter ``2*pi*n_medium*R_outer/lambda`` (>= x). ``y = x`` is the zero-
        thickness (homogeneous core) limit.
    m1, m2 : complex or float
        Relative refractive indices of the core and shell (``n_core/n_medium``,
        ``n_shell/n_medium``). Real for non-absorbing particles (the usual light-scattering
        case, where every argument stays real); complex (absorbing) is handled by the same
        recurrences, subject to the BHCOAT caveat that ``Im(m*x)``, ``Im(m*y)`` should stay
        below ~30 (Bohren & Huffman p.485).
    n_max : int, optional
        Series terms; defaults to the Wiscombe truncation on the OUTER size parameter
        ``mie_n_max(y)`` (series convergence is set by the outer sphere, B&H p.484).

    Returns
    -------
    (a, b) : two complex ndarrays of length ``n_max`` (``a[k]`` is ``a_{k+1}``) -- the same
    shape and meaning as ``mie_coefficients``, so the shared S1/efficiency machinery consumes
    them unchanged.
    """
    x = float(x)
    y = float(y)
    if not (x > 0 and math.isfinite(x)):
        raise ValueError(f"core size parameter x must be a positive finite number, got {x!r}.")
    if not (math.isfinite(y) and y >= x):
        raise ValueError(
            f"outer size parameter y must be finite and >= core x ({x!r}), got {y!r}."
        )
    m1 = complex(m1)
    m2 = complex(m2)
    for name, mm in (("m1", m1), ("m2", m2)):
        if not (mm.real > 0 and math.isfinite(mm.real) and math.isfinite(mm.imag)):
            raise ValueError(
                f"relative index {name} must have finite parts and Re > 0, got {mm!r}."
            )
    if n_max is None:
        n_max = mie_n_max(y)
    n_max = int(n_max)

    x1 = m1 * x                       # core, core index
    x2 = m2 * x                       # core, shell index
    y2 = m2 * y                       # outer, shell index
    refrel = m2 / m1                  # "m" in the Appendix-B A_n/B_n (= m2/m1)

    a = np.empty(n_max, dtype=complex)
    b = np.empty(n_max, dtype=complex)

    # Log-derivatives D_n(m1 x), D_n(m2 x), D_n(m2 y) by the STABLE DOWNWARD recurrence
    # (Wiscombe), NOT BHCOAT's upward step: the upward form divides by zero at a Riccati-Bessel
    # zero of the argument (a pole of D_n), which a coated fit hits when it probes a core size
    # x1 = m1 k R_core through such a zero. Downward is pole-free and identical in value.
    D_x1 = _log_deriv_downward(x1, n_max)            # D_n(m1 x), D_x1[n-1] = D_n
    D_x2 = _log_deriv_downward(x2, n_max)            # D_n(m2 x)
    D_y2 = _log_deriv_downward(y2, n_max)            # D_n(m2 y)
    # Real Riccati-Bessel psi_n(y), chi_n(y): psi_{-1}=cos y, psi_0=sin y; chi seed -sin, cos.
    psi0y, psi1y = math.cos(y), math.sin(y)          # psi_{-1}(y), psi_0(y)
    chi0y, chi1y = -math.sin(y), math.cos(y)
    xi1y = complex(psi1y, -chi1y)                    # xi_0(y) = psi_0 - i chi_0
    chi0y2, chi1y2 = -cmath.sin(y2), cmath.cos(y2)   # chi seeds at m2 y (complex arg)
    chi0x2, chi1x2 = -cmath.sin(x2), cmath.cos(x2)   # chi seeds at m2 x

    iflag = 0
    brack = 0.0 + 0.0j       # A_n / psi_n(m2 y); frozen to 0 once the core is negligible
    crack = 0.0 + 0.0j       # B_n / psi_n(m2 y)
    for n in range(1, n_max + 1):
        rn = float(n)
        # psi_n(y), chi_n(y), xi_n(y) by upward recurrence (real argument y).
        psiy = (2.0 * rn - 1.0) / y * psi1y - psi0y
        chiy = (2.0 * rn - 1.0) / y * chi1y - chi0y
        xiy = complex(psiy, -chiy)
        d1y2 = D_y2[n - 1]                                     # D_n(m2 y) (downward, pole-free)
        if iflag != 1:
            d1x1 = D_x1[n - 1]                                # D_n(m1 x)
            d1x2 = D_x2[n - 1]                                # D_n(m2 x)
            chix2 = (2.0 * rn - 1.0) / x2 * chi1x2 - chi0x2   # chi_n(m2 x)
            chiy2 = (2.0 * rn - 1.0) / y2 * chi1y2 - chi0y2   # chi_n(m2 y)
            chipx2 = chi1x2 - rn * chix2 / x2                 # chi_n'(m2 x)
            chipy2 = chi1y2 - rn * chiy2 / y2                 # chi_n'(m2 y)
            # A_n (ANCAP): the third division is x psi_n(m2 x) via 1/psi_n = chi_n D_n - chi_n'
            # (B&H p.484 Wronskian) -- avoids evaluating psi_n at the complex argument m2 x.
            ancap = refrel * d1x1 - d1x2
            ancap = ancap / (refrel * d1x1 * chix2 - chipx2)
            ancap = ancap / (chix2 * d1x2 - chipx2)
            brack = ancap * (chiy2 * d1y2 - chipy2)           # A_n / psi_n(m2 y)
            # B_n (BNCAP), same structure.
            bncap = refrel * d1x2 - d1x1
            bncap = bncap / (refrel * chipx2 - d1x1 * chix2)
            bncap = bncap / (chix2 * d1x2 - chipx2)
            crack = bncap * (chiy2 * d1y2 - chipy2)           # B_n / psi_n(m2 y)
            # Inner-sphere freeze test (B&H p.484): once every core term is below DEL, drop the
            # core contribution (brack = crack = 0) for this and all higher orders.
            amess1 = brack * chipy2
            amess2 = brack * chiy2
            amess3 = crack * chipy2
            amess4 = crack * chiy2
            if not (abs(amess1) > _BHCOAT_DEL * abs(d1y2) or abs(amess2) > _BHCOAT_DEL
                    or abs(amess3) > _BHCOAT_DEL * abs(d1y2) or abs(amess4) > _BHCOAT_DEL):
                brack = 0.0 + 0.0j
                crack = 0.0 + 0.0j
                iflag = 1
        # Modified log-derivatives D-tilde_n, G-tilde_n (B&H p.483). brack/crack = 0 (frozen)
        # -> both collapse to D_n(m2 y), i.e. the homogeneous outer sphere.
        dnbar = (d1y2 - brack * chipy2) / (1.0 - brack * chiy2)
        gnbar = (d1y2 - crack * chipy2) / (1.0 - crack * chiy2)
        fac_a = dnbar / m2 + rn / y
        fac_b = m2 * gnbar + rn / y
        a[n - 1] = (fac_a * psiy - psi1y) / (fac_a * xiy - xi1y)
        b[n - 1] = (fac_b * psiy - psi1y) / (fac_b * xiy - xi1y)
        # Roll n-1 <- n for the upward psi/chi recurrences. (The D_n log-derivatives are
        # precomputed arrays, so they need no roll; once frozen, the chi rolls re-propagate
        # their last values, which are then unused.)
        psi0y, psi1y = psi1y, psiy
        chi0y, chi1y = chi1y, chiy
        xi1y = complex(psi1y, -chi1y)
        chi0x2, chi1x2 = chi1x2, chix2
        chi0y2, chi1y2 = chi1y2, chiy2
    return a, b


def coated_mie_efficiencies(x: float, y: float, m1: complex, m2: complex
                            ) -> Tuple[float, float, float, float]:
    """``(Q_ext, Q_sca, Q_back, g)`` for a coated sphere, normalised by the OUTER size
    parameter ``y`` (Bohren & Huffman 1983 Appendix B). Same series as the homogeneous
    ``mie_efficiencies`` with the coated coefficients; a published-table / oracle validation
    target (the product does not consume efficiencies -- only the form factor).
    """
    a, b = coated_mie_coefficients(x, y, m1, m2)
    return _efficiencies_from_ab(a, b, y)


def coated_mie_phase_vv(x: float, y: float, m1: complex, m2: complex, cos_theta) -> np.ndarray:
    """Per-particle VV angular intensity ``|S1(theta)|^2`` for a coated sphere (arbitrary
    relative units). A coated sphere, like a homogeneous one, does not depolarize, so under
    VU geometry it scatters purely via S1. ``cos_theta`` scalar -> float; array -> array.
    """
    a, b = coated_mie_coefficients(x, y, m1, m2)
    return _phase_vv_from_ab(a, b, cos_theta)


def coated_mie_form_factor(x: float, y: float, m1: complex, m2: complex, cos_theta) -> np.ndarray:
    """Normalised coated-sphere form factor ``P = |S1(theta)|^2 / |S1(0)|^2`` for the SLS P(q)
    fit. ``P(0) = 1`` by construction. As ``m2 -> m1`` it collapses onto the homogeneous Mie
    form factor ``mie_form_factor(y, m1, .)`` (continuity check); ``R_core -> 0`` collapses
    onto ``mie_form_factor(y, m2, .)``.
    """
    a, b = coated_mie_coefficients(x, y, m1, m2)
    return _form_factor_from_ab(a, b, cos_theta)


# ============================================================================================
# Batched (grid) coated path -- vectorized over the (x, y) size-parameter axis.
# ============================================================================================
#
# Same batching rules as the homogeneous grid path, plus the BHCOAT inner-sphere freeze as a
# per-element LATCH -- see ``coated_mie_coefficients_batch``.


def coated_mie_coefficients_batch(x: np.ndarray, y: np.ndarray, m1: complex, m2: complex
                                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Coated-sphere ``a_n, b_n`` for ARRAYS of core/outer size parameters at one contrast
    pair. Row ``i`` equals the scalar ``coated_mie_coefficients(x_i, y_i, m1, m2)`` (to
    ~machine precision) in its first ``mie_n_max(y_i)`` columns and is exactly 0.0 beyond
    them. Returns two complex arrays of shape ``(x.size, max(mie_n_max(y_i)))``.
    Same near-pole precision caveat as ``mie_coefficients_batch``: an isolated
    near-singular order can deviate well past 1e-12 per coefficient while the consumed
    S1/P stay at ~1e-14.

    Batching rules: per-element Wiscombe truncation on the OUTER y, per-element downward
    starts for the three D_n arrays, and the BHCOAT inner-sphere freeze as a per-element
    LATCH -- each element freezes (``brack = crack = 0``, for that and all higher orders)
    at exactly the order its scalar run would, because the freeze test consumes this
    order's live values. The freeze is NOT a droppable speed shortcut: it changes values at
    the ~1e-8 DEL threshold scale, and stopping the ``chi(m2 x)`` rolls at freeze is what
    keeps a small-core node (``x << y``, series length set by y) from overflowing the
    upward chi recurrence -- so the latch reproduces it exactly instead of omitting it.
    Frozen/inactive lanes stop advancing their recurrence state; the one-step-ahead
    candidates ``np.where`` still evaluates for them are bounded (no compounding).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} vs {y.shape}.")
    if not (np.all(x > 0) and np.all(np.isfinite(x))):
        raise ValueError("all core size parameters x must be positive finite numbers.")
    if not (np.all(np.isfinite(y)) and np.all(y >= x)):
        raise ValueError("all outer size parameters y must be finite and >= their core x.")
    m1 = complex(m1)
    m2 = complex(m2)
    for name, mm in (("m1", m1), ("m2", m2)):
        if not (mm.real > 0 and math.isfinite(mm.real) and math.isfinite(mm.imag)):
            raise ValueError(
                f"relative index {name} must have finite parts and Re > 0, got {mm!r}."
            )
    if x.size == 0:                            # np.all(...) is vacuously True for empty
        return np.zeros((0, 0), dtype=complex), np.zeros((0, 0), dtype=complex)
    n_max_arr = _mie_n_max_array(y)            # series length set by the OUTER sphere
    n_cols = int(n_max_arr.max())
    x1 = m1 * x
    x2 = m2 * x
    y2 = m2 * y
    refrel = m2 / m1

    def _D(z):
        # Per-element downward start, the scalar path's formula element-by-element.
        nmx = (np.maximum(n_max_arr, np.abs(z)) + 16).astype(int)
        return _log_deriv_downward_batch(z, nmx, n_cols)

    D_x1, D_x2, D_y2 = _D(x1), _D(x2), _D(y2)

    a = np.zeros((x.size, n_cols), dtype=complex)
    b = np.zeros((x.size, n_cols), dtype=complex)
    psi0y, psi1y = np.cos(y), np.sin(y)                    # psi_{-1}(y), psi_0(y)
    chi0y, chi1y = -np.sin(y), np.cos(y)
    xi1y = psi1y - 1j * chi1y                              # xi_0(y)
    y2c = y2.astype(complex)
    x2c = x2.astype(complex)
    chi0y2, chi1y2 = -np.sin(y2c), np.cos(y2c)             # chi seeds at m2 y (complex arg)
    chi0x2, chi1x2 = -np.sin(x2c), np.cos(x2c)             # chi seeds at m2 x

    frozen = np.zeros(x.shape, dtype=bool)
    brack = np.zeros(x.shape, dtype=complex)               # A_n / psi_n(m2 y); 0 once frozen
    crack = np.zeros(x.shape, dtype=complex)               # B_n / psi_n(m2 y)
    for n in range(1, n_cols + 1):
        rn = float(n)
        active = n <= n_max_arr
        live = active & ~frozen
        # psi_n(y), chi_n(y), xi_n(y) by upward recurrence (real argument y).
        psiy = (2.0 * rn - 1.0) / y * psi1y - psi0y
        chiy = (2.0 * rn - 1.0) / y * chi1y - chi0y
        xiy = psiy - 1j * chiy
        d1y2 = D_y2[:, n - 1]
        # Core block (meaningful where live; np.where evaluates all lanes, so suppress the
        # discarded lanes' float noise).
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            chix2 = (2.0 * rn - 1.0) / x2 * chi1x2 - chi0x2        # chi_n(m2 x)
            chiy2 = (2.0 * rn - 1.0) / y2 * chi1y2 - chi0y2        # chi_n(m2 y)
            chipx2 = chi1x2 - rn * chix2 / x2                      # chi_n'(m2 x)
            chipy2 = chi1y2 - rn * chiy2 / y2                      # chi_n'(m2 y)
            d1x1 = D_x1[:, n - 1]
            d1x2 = D_x2[:, n - 1]
            ancap = refrel * d1x1 - d1x2
            ancap = ancap / (refrel * d1x1 * chix2 - chipx2)
            ancap = ancap / (chix2 * d1x2 - chipx2)
            brack_new = ancap * (chiy2 * d1y2 - chipy2)            # A_n / psi_n(m2 y)
            bncap = refrel * d1x2 - d1x1
            bncap = bncap / (refrel * chipx2 - d1x1 * chix2)
            bncap = bncap / (chix2 * d1x2 - chipx2)
            crack_new = bncap * (chiy2 * d1y2 - chipy2)            # B_n / psi_n(m2 y)
            brack = np.where(live, brack_new, brack)               # frozen lanes keep 0
            crack = np.where(live, crack_new, crack)
            # Inner-sphere freeze test (B&H p.484), latched per element at exactly the
            # order the scalar run freezes (NaN compares False -> freezes, like the scalar
            # `if not (... > ...)`). Zeroing applies to THIS order's coefficients too.
            amess_small = ~((np.abs(brack * chipy2) > _BHCOAT_DEL * np.abs(d1y2))
                            | (np.abs(brack * chiy2) > _BHCOAT_DEL)
                            | (np.abs(crack * chipy2) > _BHCOAT_DEL * np.abs(d1y2))
                            | (np.abs(crack * chiy2) > _BHCOAT_DEL))
            newly_frozen = live & amess_small
            brack = np.where(newly_frozen, 0.0 + 0.0j, brack)
            crack = np.where(newly_frozen, 0.0 + 0.0j, crack)
            frozen = frozen | newly_frozen
            # Modified log-derivatives; brack/crack = 0 (frozen) -> both collapse to
            # D_n(m2 y), the homogeneous outer sphere (chi candidates stay finite for
            # frozen lanes -- their rolls are latched below -- so 0 * finite = 0).
            dnbar = (d1y2 - brack * chipy2) / (1.0 - brack * chiy2)
            gnbar = (d1y2 - crack * chipy2) / (1.0 - crack * chiy2)
            fac_a = dnbar / m2 + rn / y
            fac_b = m2 * gnbar + rn / y
            a_col = (fac_a * psiy - psi1y) / (fac_a * xiy - xi1y)
            b_col = (fac_b * psiy - psi1y) / (fac_b * xiy - xi1y)
        a[:, n - 1] = np.where(active, a_col, 0.0)
        b[:, n - 1] = np.where(active, b_col, 0.0)
        # Roll n-1 <- n: the outer-y recurrences where active (per-element truncation);
        # the core chi recurrences only where live (the scalar loop stops recomputing
        # them at freeze -- and a small-core lane rolled past its freeze would overflow).
        psi0y = np.where(active, psi1y, psi0y)
        chi0y = np.where(active, chi1y, chi0y)
        psi1y = np.where(active, psiy, psi1y)
        chi1y = np.where(active, chiy, chi1y)
        xi1y = psi1y - 1j * chi1y
        chi0x2 = np.where(live, chi1x2, chi0x2)
        chi1x2 = np.where(live, chix2, chi1x2)
        chi0y2 = np.where(live, chi1y2, chi0y2)
        chi1y2 = np.where(live, chiy2, chi1y2)
    return a, b


def coated_mie_form_factor_batch(x: np.ndarray, y: np.ndarray, m1: complex, m2: complex,
                                 cos_theta) -> np.ndarray:
    """Normalised coated-sphere form factor for ARRAYS of (core, outer) size parameters.

    Row ``i`` equals ``coated_mie_form_factor(x_i, y_i, m1, m2, cos_theta)`` (to ~machine
    precision). Returns shape ``(x.size, n_angles)``. Large batches are processed in row
    chunks (see ``_BATCH_MAX_ELEMENTS``) — results-neutral, memory-bounded.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # Empty/mismatched inputs fall through to coated_mie_coefficients_batch, which
    # validates m1/m2 and the x/y shapes before its own empty return (an early return
    # or an eager np.max(y) here would swallow a bad contrast for an all-masked batch,
    # or crash with a cryptic empty-reduction error on a shape mismatch --
    # focused-review 2026-07-23); only the chunk estimate is skipped.
    if x.size and y.size:
        chunks = _batch_row_chunks(x.size, mie_n_max(float(np.max(y))))
        if len(chunks) > 1:
            return np.vstack([coated_mie_form_factor_batch(x[s], y[s], m1, m2, cos_theta)
                              for s in chunks])
    a, b = coated_mie_coefficients_batch(x, y, m1, m2)
    return _form_factor_from_ab_batch(a, b, cos_theta)
