"""
analysis/uncertainty.py
=======================

Statistical-uncertainty toolkit for the analysis layer.

Scope and philosophy (see the Theory-and-Equations-Guide, "Uncertainty estimates"):
we report the **statistical** (regression) uncertainty of a fit, and only where the
fitted points are genuinely independent measurements (SLS angle/concentration series,
Gamma-vs-q^2 over angles, D-vs-c over concentrations, the cross-sample scaling fit).
We do NOT manufacture uncertainties where none is defensible: a single-correlogram DLS
dynamic fit (cumulant/single/double/KWW/lognormal) has strongly correlated lag channels
(Schaetzel 1990), so an ordinary-least-squares covariance from one correlogram
under-reports the truth; that uncertainty belongs to repeat measurements (ISO 22412),
handled by the replicate-averaging path (analysis/dls/replicate.py), not here. Ill-posed inversions (NNLS/CONTIN)
and single data points (single-angle Mw) get no uncertainty.

All functions are pure (NumPy only). Equations are documented here AND in the user guide.

Notation: for a linear model y = X b with n points and p parameters and residuals e,
the parameter covariance is the **heteroscedasticity-consistent HC3 "sandwich"**
estimator (MacKinnon & White 1985; Long & Ervin 2000):

    Cov(b) = (X^T X)^-1 [ sum_i x_i x_i^T e_i^2 / (1 - h_ii)^2 ] (X^T X)^-1,

with leverage h_ii = x_i^T (X^T X)^-1 x_i. HC3 does NOT assume the points share a
common error variance, so it does not under-report when point precision varies across
angles/concentrations (e.g. a per-angle Gamma whose precision changes with q) — Monte-
Carlo-verified to match the sampling spread where ordinary least squares under-reports.
It reduces to the usual s^2 (X^T X)^-1 under homoscedastic errors. A scalar f(b)
propagates as var(f) = J^T Cov(b) J with J the gradient of f (first-order delta method).

(X^T X)^-1 is obtained from an SVD of the column-equilibrated design and X^T X is never
formed -- see `_normal_equations`. That is a numerical choice, not a statistical one: the estimator
above is unchanged, but forming the normal equations explicitly squares the condition
number and costs two significant digits per decade instead of one.

The covariance estimator is selectable via an `estimator` argument on the fitters
('hc3', the default, vs 'ols', the classical s^2 (X^T X)^-1). HC3 is the default because
it never under-reports; classical OLS is an opt-in for comparability with legacy/literature
SEs and can under-report on short high-leverage designs (invariant 8 clause A). The choice
is recorded on the returned fit object (`.estimator`) so provenance travels with every SE.

**This module also owns the shared least-squares PRIMITIVES** — `_normal_equations`
(the single SVD), `_lstsq_coeffs`, and `polyfit_or_nan` — even though the last of those
produces no ± and so sits oddly beside the rest of the toolkit. That is deliberate, not
an accident of history: each of them pairs a raising numpy call with the finiteness and
conditioning guard sized to it, and the recurring defect in this subsystem is a guard
that has drifted onto a different object than the operation it protects (three
recorded instances). One home for the primitive means one home for its guard. Do not
tidy them out into a utilities module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Linear regression with parameter covariance
# ---------------------------------------------------------------------------

# Estimator vocabulary — one spelling, no drift. The default HC3 never under-reports
# under non-uniform precision; classical OLS is an opt-in for comparability with
# legacy/literature/spreadsheet SEs and can under-report (invariant 8 clause A).
HC3 = 'hc3'
OLS = 'ols'
_ESTIMATORS = (HC3, OLS)


# Reporting threshold: the condition-number ceiling for a healthy linear design X. A
# NUMERICAL-health criterion, not a physics threshold and not a magnitude of any fitted
# quantity -- it is measured on the COLUMN-NORMALIZED design (`_design_cond`), so it is
# unit-invariant and reflects genuine collinearity only, not "Mw too large" (invariant 4).
# A healthy design conditions at O(1-100) and float64 loses ~one digit per decade of cond,
# so 1e12 leaves ~4 of ~16 digits: near-singular, yet far above healthy. It targets
# NUMERICAL singularity, not statistical thinness -- sparse but well-separated
# concentrations condition fine. Overridable per-fit (`cond_limit`).
COND_LIMIT = 1e12


# COMPUTABILITY boundary for the COVARIANCE path, distinct from COND_LIMIT above: that one
# says "this extrapolation's VALUE is numerically unreliable, flag it", this one says "we
# cannot compute an honest covariance at all, so withhold the +/-". Its derivation is its
# own (see `_normal_equations`), and it merely coincides in magnitude. Deliberately NOT
# per-fit overridable: `cond_limit` is a reporting preference the caller may tune, whereas
# whether float64 can carry this covariance is not a matter of taste. How the two interact
# for a user, and how the shipped documentation reflects that, is on `_design_cond`.
COV_COND_LIMIT = 1e12


def _column_equilibrate(X: np.ndarray):
    """Scale each column of X to unit 2-norm; return `(Xn, norms)` with `X = Xn * norms`.

    The single source of the column scaling, shared by `_design_cond` (which reports the
    conditioning) and `_normal_equations` (which inverts). That sharing is the point: the guard and
    the operation it protects then measure THE SAME matrix by the same definition, which
    is precisely what failed before (a `matrix_rank(X)` guard in front of an inversion of
    X^T X, whose condition number is the square).

    A zero-norm column (an all-zero regressor) is itself degenerate; it is left unscaled
    so the matrix stays singular -> cond = inf, the correct verdict.

    The norm is computed SCALED -- each column divided by its own max |entry| before
    squaring -- rather than as a direct sqrt(sum(x^2)). Squaring first breaks at both ends
    of the float range, and both ends are the same recurring shape (a guard that an extreme but
    perfectly representable input walks straight past): a column of ~1e160 overflows to
    norm = inf, so a well-conditioned design is scaled to zeros, reads cond = inf and has
    its covariance withheld; a column of ~1e-200 underflows to norm = 0 and is then
    treated as an all-zero regressor. `np.linalg.norm` squares the same way and is NOT a
    fix (measured: it returns inf on the 1e160 column too).

    The property scaling buys is that the norm is computed without overflow or underflow
    for every column whose NORM is itself representable -- NOT bit-exactness, which neither
    form has, and not immunity for every finite column. A column whose entries approach
    DBL_MAX has a norm of |entry|*sqrt(n) that can exceed it (measured: 1e307 is fine,
    1e308 with n = 5 overflows), and it is then reported degenerate. That verdict is
    correct rather than a residual gap: the parameter variance for such a column is of
    order 1/norm^2, which underflows to exactly 0.0, so the alternative to withholding is
    an SE of precisely zero -- a spuriously perfect uncertainty, the cardinal invariant-8
    failure. Withholding is the honest answer at that end, where at 1e160 (the range the
    old form broke on) it was not. It agrees with the naive
    form to 5.0e-16 worst case (~2.3 eps, measured over 20 000 random designs spanning
    n = 2-39, p = 1-4 and per-column scales 1e-6...1e6; 1.8e-16 on the Zimm ladder). Since
    cond is a ratio of singular values of a matrix perturbed at that level, a COND_LIMIT or
    COV_COND_LIMIT verdict can only move for a design already within ~1e-16 of the
    threshold, so the documented "unit 2-norm" definition, COND_LIMIT's meaning and the
    unit-invariance property all stand (the pre-divisor is max|entry|, itself exactly
    proportional to the column's unit, so unit-invariance is preserved exactly).
    """
    scale = np.max(np.abs(X), axis=0) if X.size else np.zeros(X.shape[1])
    nz = scale > 0
    norms = np.zeros(X.shape[1], dtype=float)
    norms[nz] = scale[nz] * np.sqrt(np.sum((X[:, nz] / scale[nz]) ** 2, axis=0))
    safe = np.where(norms > 0, norms, 1.0)
    return X / safe, safe


def _normal_equations(X: np.ndarray):
    """The building blocks both covariance estimators need, or None when they cannot be
    computed: `(A, U)`, where `A = D^-1 V S^-1` (p x p) and `U` (n x p) come from ONE SVD
    of the COLUMN-EQUILIBRATED design. X^T X is never formed.

    Writing X = Xn D (D the diagonal matrix of column norms) and Xn = U S V^T, every
    quantity the estimators use is a product of these two:

        (X^T X)^-1 = A A^T,          the parameter dispersion
        X^+        = A U^T,          the pseudo-inverse (X^T X)^-1 X^T
        h_ii       = ||U_i||^2,      the leverage of point i (row i of U)

    all exact algebra. The leverage identity holds because h_ii is a diagonal entry of
    the orthogonal projector onto col(X), and col(X D) = col(X) for any invertible
    diagonal D -- so the equilibrated U gives the leverages of the ORIGINAL design.

    **Why factored rather than assembled.** Handing back A and U instead of (X^T X)^-1
    lets each estimator write its covariance as a GRAM product B B^T (see `_hc3_cov`,
    `_ols_cov`), which is positive semi-definite by construction no matter how the
    rounding falls. Assembling (X^T X)^-1 first and sandwiching it does not have that
    property: on a Zimm ladder at cond 2.9e9 the assembled form came back INDEFINITE,
    with negative variances on the diagonal, so `_diag_se` withheld the SE by accident
    (a negative float) rather than by any designed rule -- and HC3 withheld where OLS,
    on the very same design, still reported. The factored form also carries only one
    factor of S^-1 where the assembled one carries two, so it loses one digit per decade
    instead of two.

    The same two-digits-per-decade cancellation is why the leverages come from U. The
    triple product x_i^T (X^T X)^-1 x_i multiplies entries of order cond^2 by entries of
    X and must cancel back to a number in [0, 1]; measured on that same ladder it
    returned h_ii up to 32, summing to 96 where the true values lie in [0, 1] and sum to
    p = 3. From U they are in [0, 1] and sum to the rank by construction.

    Forming the normal equations explicitly "can lead to loss of information in
    finite-precision arithmetic" (Hansen 1998, p. 101, Sec. 5.1.1); concretely the singular values of
    X^T X are the squares of those of X, so cond(X^T X) = cond(X)^2. Equilibrating first
    also removes the arbitrary per-column UNIT scales (q^2 in nm^-2, c in g/mL, intercept
    1), which would otherwise inflate the raw condition number without any genuine
    collinearity -- the same reasoning as `_design_cond`.

    **Where `COV_COND_LIMIT` comes from.** This route loses roughly ONE significant digit
    per decade of cond, measured against an exact rational (`fractions.Fraction`) inverse
    of the same design: cond 2.9e6 -> 1.1e-12 relative error, 2.9e8 -> 7.4e-9, 2.9e9 ->
    1.7e-7, 2.9e11 -> 4.8e-7. Requiring >= 4 significant digits of the covariance -- the
    same usable-precision bar `COND_LIMIT` sets for the coefficients -- puts the ceiling at
    1e12. The two constants therefore agree in magnitude for a stated reason rather than by
    copying: both paths now lose one digit per decade, which is what the SVD route buys.

    Returns None -- which the caller turns into a NaN covariance, the honest "cannot
    compute here" of invariant 8 -- in four cases:

      - an EMPTY or UNDERDETERMINED design (no rows, or n < p). The thin SVD of such a
        design yields fewer than p singular values, so reading s[0]/s[-1] would raise
        IndexError on an empty one -- the guard against an uncaught exception throwing one
        itself, for the second time in this function's history. n < p also makes X^T X
        singular by construction, so withholding is the right answer on its own terms and
        not merely a shape defence; the dof checks in both estimators would reach the same
        verdict one step later.
      - a non-finite entry in X (an infinite/NaN intensity already propagated into the
        design). Checked explicitly: np.linalg.svd RAISES on non-finite input, and so did
        the np.linalg.matrix_rank guard this replaced, which is a second way the old guard
        could crash the analysis it was meant to protect.
      - the SVD does not converge (LinAlgError). Not reachable for finite input in
        practice; kept as a backstop so no path out of here can raise.
      - cond(Xn) > COV_COND_LIMIT, i.e. the covariance would carry fewer than ~4
        significant digits. A rank-deficient design (collinear columns -- a Zimm/Berry
        [1, q^2, c] fit at a single angle *and* a single concentration, or q^2
        proportional to c; or a nonlinear Jacobian with an unidentifiable parameter) is
        the limiting case of this, with cond = inf, so this subsumes the rank test it
        replaced rather than sitting beside it.

    We do NOT fall back to np.linalg.pinv: a pseudo-inverse would hand back a small,
    plausible, UNDER-reported SE for an unidentifiable parameter, the cardinal invariant-8
    failure. Withholding is the honest direction.
    """
    n, p = X.shape
    if min(X.shape) == 0 or n < p:
        return None
    if not np.all(np.isfinite(X)):
        return None
    Xn, norms = _column_equilibrate(X)
    try:
        U, s, Vt = np.linalg.svd(Xn, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    # isfinite: an infinite smallest singular value gives cond = nan, which the
    # `not (cond <= LIMIT)` test below would then treat as 'too ill-conditioned' by
    # accident rather than by measurement. inf is the honest verdict for an
    # unusable decomposition.
    cond = (float(s[0] / s[-1])
            if (math.isfinite(s[-1]) and s[-1] > 0) else float('inf'))
    if not (cond <= COV_COND_LIMIT):          # `not <=` also catches a NaN cond
        return None
    A = (Vt.T / s) / norms[:, None]           # D^-1 V S^-1
    return A, U


def _lstsq_coeffs_and_rank(X: np.ndarray, y: np.ndarray):
    """Least-squares coefficients AND the rank the solver actually used.

    The rank is reported because `np.linalg.lstsq` decides, on the RAW design, which
    singular directions to keep -- and that decision is invisible to `_design_cond`, which
    normalizes the columns first (deliberately, so the flag is unit-invariant). At an
    extreme column magnitude the raw singular values span hundreds of decades, `lstsq`
    truncates a direction, and the coefficients returned are the min-norm answer to a
    REDUCED problem. Measured on the Zimm-shaped design `[1e120*ones, 1..5]`: the truncated
    solve's residual sum of squares is 10 where a full-rank solve reaches ~1e-30, and
    nothing on the result said so.

    It is a SEPARATE field rather than a contribution to `cond`/`well_conditioned` (owner
    decision, 2026-08-01). Folding it in would make the health flag magnitude-dependent
    again, which is exactly what column equilibration removed and what
    `tests/test_covariance_contract_family.py` forbids -- conditioning is scale-free
    collinearity, and "the solver dropped a direction" is a different statement. Both are
    true at once; they are not the same flag.

    Returns `(beta, rank)`, with `rank = 0` alongside the NaN coefficients when the data
    are not finite.

    `np.linalg.lstsq` RAISES `LinAlgError: SVD did not converge` on a non-finite entry in
    the DESIGN -- an opaque crash for something that is simply unanswerable. (A non-finite
    RESPONSE does not raise; it already returns NaN coefficients. The check covers both so
    the two inputs behave alike, but only the design half prevents an exception.) Either
    way there is no least-squares solution, so we return NaN coefficients. Degrading to
    NaN or raising a CLEAR error -- never a bare library exception -- is the house rule
    for extreme-but-representable input.

    What the caller then reports differs between the two, and callers should know it: a
    non-finite DESIGN also drives `_design_cond` to `inf`, so the fit reports
    `well_conditioned = False`; a non-finite RESPONSE leaves the design healthy, so `cond`
    stays finite and `well_conditioned` stays True while every value is NaN. Neither can
    mis-report an uncertainty -- the covariance is all-NaN in both cases, so `se_or_none`
    yields None -- but only the first is described by the conditioning flag.

    The finite check lives here as well as in `_normal_equations` on purpose: the
    coefficient path and the covariance path are separate operations, and guarding only
    one of them is the very defect this module was corrected for.
    """
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(y))):
        return np.full(X.shape[1], np.nan), 0
    beta, _residues, rank, _sv = np.linalg.lstsq(X, y, rcond=None)
    return beta, int(rank)


def _lstsq_coeffs(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least-squares coefficients only, for callers that do not report a rank flag."""
    return _lstsq_coeffs_and_rank(X, y)[0]


def _rescale_weights(w: np.ndarray) -> np.ndarray:
    """Bring a weight vector into a sane binary exponent range, changing nothing else.

    A huge weight silently ZEROES a `np.polyfit`. Its internal column scaling is
    ``scale = sqrt((lhs*lhs).sum(axis=0))`` with ``lhs = vander * w[:, None]``; at
    ``w ~ 1e160`` the square overflows, ``scale`` becomes ``inf``, ``lhs /= inf`` zeroes
    the design, and the fit returns ``[0, 0, 0]``. In the cumulant path that is
    ``beta = exp(0) = 1`` and ``Gamma = 0``: finite, plausible and wrong, with every
    INPUT finite, so the finiteness guard above cannot see it. (The far tail,
    ``w ~ 1e-200``, instead raised ``LinAlgError`` -- caught below, so that end withheld.)

    Weights are scale-invariant in least squares: multiplying them all by a constant
    multiplies the objective by that constant and leaves the minimizer alone. So this
    normalization is a mathematical no-op, which is why it is applied unconditionally
    rather than behind a magnitude threshold.

    One boundary does remain, and this sheet's own thesis is to name it rather than claim
    there is none: the shift is by a single factor for the whole vector, so a weight vector
    whose own DYNAMIC RANGE exceeds the exponent budget (max/min ~ 2^1074) drives its
    smallest entries subnormal and eventually to zero. Measured: `[1e300, 1e-10, 1.0]`
    leaves the middle entry at ~7e-311, a handful of bits. It is still strictly better than
    not scaling -- there, raw `np.polyfit` returned a garbage fit where this now withholds
    NaN -- and no real `g2-1` amplitude vector spans anything like that range.

    The scale factor is a POWER OF TWO (`frexp`/`ldexp`), not ``max(|w|)``. Halving an
    exponent is exact in binary floating point, so the numbers the engine returns on real
    data are unchanged BIT FOR BIT; dividing by the max would have perturbed the last few
    ulps of every weighted fit in the platform to fix a 1e160 edge case.
    """
    m = float(np.max(np.abs(w))) if w.size else 0.0
    if not (math.isfinite(m) and m > 0):
        return w                       # all-zero (or unusable) weights: nothing to scale
    _, exponent = math.frexp(m)        # m == mantissa * 2**exponent, mantissa in [0.5, 1)
    return np.ldexp(w, -exponent)      # exact: only the binary exponents move


def r_squared_or_nan(ss_res: float, ss_tot: float) -> float:
    """Coefficient of determination 1 - ss_res/ss_tot, or NaN when it is unanswerable.

    The single place that ratio is formed anywhere in `analysis/`, the same
    one-call-node discipline `polyfit_or_nan` holds over `np.polyfit`: R2 is a
    criterion several fits report and compare verdicts on, so it has one body rather
    than one per caller (enforced by `tests/test_r_squared_home_family.py`).

    isfinite as well as > 0: around |y| ~ 1e155-1e160 there is a window where ss_tot
    overflows to inf while ss_res is still finite, which would report R2 = 1.0 off a
    denominator that is not a representable number. The true R2 in that window IS ~1.0
    (the window only exists when the fit is near-perfect), so this withholds a correct
    value in a physically absurd corner -- robustness, not a correctness fix.

    The guard is on the DENOMINATOR only, deliberately: `ss_res` is left unchecked so a
    NaN numerator rides through to a NaN R2. That is the path by which `polyfit_or_nan`'s
    NaN coefficients reach the result, and closing it would swallow a withheld fit.
    """
    return (1.0 - ss_res / ss_tot
            if (math.isfinite(ss_tot) and ss_tot > 0) else float('nan'))


def polyfit_or_nan(x: np.ndarray, y: np.ndarray, deg: int,
                   w: Optional[np.ndarray] = None) -> np.ndarray:
    """Polynomial coefficients for y ~ p(x), or a NaN vector when they are unanswerable.

    The `_lstsq_coeffs` of the polynomial fits, and the single place `np.polyfit` is
    called anywhere in `analysis/` (pinned by `tests/test_polyfit_guard_family.py`).
    Returns coefficients HIGHEST-degree-first, exactly as `np.polyfit` does, so a caller
    can be routed through here without touching how it unpacks the result.

    `np.polyfit` RAISES `LinAlgError: SVD did not converge` on a non-finite entry in the
    ABSCISSA or in the WEIGHTS -- an opaque crash for something that is simply
    unanswerable. (A non-finite RESPONSE does not raise; it already returns NaN
    coefficients. All three are checked so the channels behave alike, but only the first
    two prevent an exception.) Degrading to NaN or raising a CLEAR error -- never a bare
    library exception -- is the house rule for extreme-but-representable input.

    Two boundaries are worth stating because callers depend on them:

    - **What is caught.** `LinAlgError` is also caught as a backstop, so no data condition
      can raise out of here. `ValueError` and `TypeError` are NOT caught: a bad `deg` or a
      length mismatch is a programming error, and swallowing it into NaN would hide the
      very bugs this helper exists to make legible. Note `LinAlgError` SUBCLASSES
      `ValueError`, so the order of the `except` clauses at any call site matters -- an
      `except ValueError` written above it silently absorbs it.
    - **Degenerate shapes withhold.** With fewer than `deg + 1` points `np.polyfit` returns
      a min-norm fit and a `RankWarning`, and on an empty input it raises `TypeError` from
      inside numpy. Neither is an answer, so we return NaN instead -- matching
      `_normal_equations`' treatment of an underdetermined design, and sparing every caller
      its own `size` check.

    The check lives here as well as in `_lstsq_coeffs` on purpose: the polynomial path and
    the covariance path are separate operations, and guarding only one of them is the
    defect this module was corrected for once already.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n_coeffs = int(deg) + 1
    if x.size < n_coeffs:
        return np.full(n_coeffs, np.nan)
    finite = np.all(np.isfinite(x)) and np.all(np.isfinite(y))
    if w is not None:
        finite = finite and np.all(np.isfinite(np.asarray(w, dtype=float)))
    if not finite:
        return np.full(n_coeffs, np.nan)
    # ONE call node, not a `w is None` ternary over two: the family test asserts that
    # exactly one np.polyfit call survives in all of analysis/, so that the guard above and
    # the operation can never drift onto different call sites.
    kwargs = {} if w is None else {'w': _rescale_weights(np.asarray(w, dtype=float))}
    try:
        return np.polyfit(x, y, deg, **kwargs)
    except np.linalg.LinAlgError:
        return np.full(n_coeffs, np.nan)


def power_or_nan(x: float, n: int) -> float:
    """``x ** n`` for a small positive integer ``n``, or NaN unless it is a usable divisor.

    THE divisor of every reciprocal-power SE Jacobian in the engine — d(1/x)/dx = -1/x²,
    d(1/x²)/dx = -2/x³, d(slope/x)/dx = -slope/x². Eight sites wrote that divisor inline as
    ``x ** n`` under a predicate about ``x`` ALONE, which cannot answer the question asked:
    ``x`` being positive and finite says nothing about ``x ** n``. Measured at
    analysis/sls.py:550 before this existed — ``intercept = 1e200`` raised ``OverflowError``
    and ``intercept = 1e-200`` raised ``ZeroDivisionError``, from an SE path whose whole
    contract is to withhold rather than fail.

    Two things make one shared helper the right shape rather than eight corrected guards
    (CLAUDE.md rule 12 — remove the gap, don't police it):

    - **The square is built by repeated multiplication, never ``**``.** On a Python float
      ``**`` RAISES where ``*`` saturates: ``1e300 ** 2`` is an ``OverflowError`` while
      ``1e300 * 1e300`` is ``inf``. A saturating value is guardable; a raise happens before
      any guard can run. (NumPy is unaffected — this is a Python-scalar trap only.)
    - **NaN is a working withhold with no branch at the call site.** A NaN Jacobian entry
      makes ``propagate``'s variance NaN, which it already maps to NaN → ``se_or_none`` →
      ``None``. So a caller writes ``-1.0 / power_or_nan(x, 2)`` and gets the existing
      "no defensible uncertainty" path for free; there is no second predicate to mis-size.

    Sign is preserved (an odd power of a negative base stays negative) and **is not
    required to be positive**: the calibration-free A₂ site guards only ``intercept != 0``,
    and narrowing this helper to positive bases would silently stop reporting an SE that
    ships today. Match the predicate to the property under repair — the property here is
    "is the result a usable divisor", i.e. finite and non-zero (2026-07-31 ledger; the
    reciprocal-power ladder in tests/test_positivity_guard_family.py pins both tails).
    """
    if not math.isfinite(x):
        return float('nan')
    out = 1.0
    for _ in range(n):
        out *= x                 # `*` saturates to inf; `**` would RAISE -- see above
    # The test is on the RECIPROCAL, because that is the operation every divisor caller
    # performs; the one multiplicand caller (fit_power_law's prefactor, n = 1) keeps the
    # reciprocal test deliberately, as the conservative bound on a representable value.
    # `out != 0.0` is not the same question: across the subnormal window the result is
    # finite and non-zero yet its reciprocal is not -- measured, `power_or_nan(1e-160, 2)`
    # is 1e-320 and `1.0 / 1e-320` is `inf`, so the helper handed back an "answer" that
    # made every Jacobian entry built on it infinite. The predicate must be sized to the
    # operation the callers perform, not to the value this helper happens to hold.
    if not (math.isfinite(out) and out != 0.0 and math.isfinite(1.0 / out)):
        return float('nan')
    return out


def _hc3_cov(X: np.ndarray, resid: np.ndarray, A: np.ndarray,
             U: np.ndarray) -> np.ndarray:
    """HC3 heteroscedasticity-consistent covariance from a design X and residuals e.

    Cov = (X^T X)^-1 [sum_i x_i x_i^T e_i^2/(1-h_ii)^2] (X^T X)^-1.  Returns NaN when
    the residual degrees of freedom are too few to estimate a spread (n - p < 2): a
    fit with one residual dof cannot honestly support an uncertainty, so we report
    none rather than a wildly inflated value (HC3 over-inflates as dof -> 1). For a
    linear fit X is the design and e = y - X beta; for a converged NONLINEAR fit X is
    the model Jacobian at the optimum and e the fit residuals (see
    `covariance_from_jacobian`) -- the sandwich is identical once linearized.

    Evaluated in the algebraically identical GRAM form. With X^+ = (X^T X)^-1 X^T the
    pseudo-inverse and w_i = e_i^2/(1-h_ii)^2, the sandwich above is exactly

        Cov = [X^+ diag(sqrt(w))] [X^+ diag(sqrt(w))]^T = B B^T,

    so it is positive semi-definite by CONSTRUCTION and can never hand back a negative
    variance on the diagonal. `A` and `U` come from `_sandwich_cov` via
    `_normal_equations` (X^+ = A U^T, h_ii = ||U_i||^2), so the module has exactly one
    decomposition site and it cannot drift from its guard."""
    n, p = X.shape
    if n - p < 2:
        return np.full((p, p), np.nan)
    h = np.einsum('ij,ij->i', U, U)                   # leverages h_ii = ||U_i||^2
    w = resid ** 2 / np.clip((1.0 - h) ** 2, 1e-300, None)
    B = (A @ U.T) * np.sqrt(w)                        # X^+ diag(sqrt(w)), p x n
    return B @ B.T


def _ols_cov(X: np.ndarray, resid: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Classical homoscedastic OLS covariance Cov = s^2 (X^T X)^-1, s^2 = RSS/(n-p).

    The textbook estimator: assumes every point shares one error variance. Offered as
    an opt-in for comparability with classical software / literature tables (invariant 8
    clause A) — it under-reports on short, high-leverage designs where precision varies
    (~10% low on the 5-point calibration-free A2 ladder), which is why HC3,
    not this, is the default. Returns NaN when there are too few residual dof to estimate
    the spread (n - p < 1). X/resid follow the same linear-or-Jacobian convention as
    `_hc3_cov`, and `A` arrives the same way.

    Also evaluated as a Gram product: (X^T X)^-1 = A A^T, so Cov = (sA)(sA)^T is
    positive semi-definite by construction (s^2 = RSS/(n-p) >= 0)."""
    n, p = X.shape
    if n - p < 1:
        return np.full((p, p), np.nan)
    s2 = float(resid @ resid) / (n - p)
    B = A * math.sqrt(s2)
    return B @ B.T


def _sandwich_cov(X: np.ndarray, resid: np.ndarray, estimator: str = HC3) -> np.ndarray:
    """Parameter covariance from a design/Jacobian X and residuals e, by the selected
    estimator (default HC3 sandwich; OLS opt-in).

    The single dispatch point, and the single place the normal-equations quantities are
    obtained. When `_normal_equations` cannot produce them -- an empty, underdetermined,
    non-finite, rank-deficient, or too-ill-conditioned design (see its docstring for the
    boundary and the reasoning) -- we return a
    NaN-filled covariance, the honest "cannot compute here" (invariant 8), which flows
    through se_or_none -> None, a case every caller already handles. Never np.linalg.pinv:
    a pseudo-inverse would hand back a small, plausible, UNDER-reported SE for an
    unidentifiable parameter, the cardinal invariant-8 failure.

    The RESIDUALS are checked here for the same reason the design is checked in
    `_normal_equations`: both estimators multiply `resid` into the covariance, so a
    non-finite residual against a perfectly healthy design produced a covariance of mixed
    inf and NaN entries -- neither all-finite nor all-NaN, and so a violation of the
    contract below rather than an honest withholding. (No wrong +/- ever escaped it:
    `_diag_se` and `propagate` both reject a non-finite variance. It was the contract that
    was broken, not the reporting.)

    Nothing on any path out of here raises for a real-valued X, and every return is either
    all-finite or all-NaN: that contract is pinned by
    `tests/test_covariance_contract_family.py` across the whole conditioning range, both
    estimators, degenerate shapes, and non-finite designs, responses and residuals.
    """
    p = X.shape[1]
    if estimator not in _ESTIMATORS:
        raise ValueError(f"unknown estimator {estimator!r}; expected one of {_ESTIMATORS}")
    if not np.all(np.isfinite(resid)):
        return np.full((p, p), np.nan)
    normal = _normal_equations(X)
    if normal is None:
        return np.full((p, p), np.nan)
    A, U = normal
    cov = _ols_cov(X, resid, A) if estimator == OLS else _hc3_cov(X, resid, A, U)
    # Final structural check on the OUTPUT, not just the inputs. A design can be perfectly
    # well conditioned and still have a variance that float64 cannot hold -- a regressor of
    # order 1e-200 implies var ~ 1e400 -- and the result is then a covariance of mixed
    # finite and infinite entries. Partially non-finite is the one shape worse than either
    # answer, since a caller reading a single diagonal entry never sees the degeneracy. An
    # unrepresentable variance is a "cannot compute here", so it withholds like any other.
    if not np.all(np.isfinite(cov)):
        return np.full((p, p), np.nan)
    return cov


def _cov(X: np.ndarray, y: np.ndarray, beta: np.ndarray, estimator: str = HC3) -> np.ndarray:
    """Parameter covariance of a LINEAR fit y = X beta (default HC3; OLS opt-in).

    Thin wrapper: forms the residuals e = y - X beta and defers to `_sandwich_cov`.
    """
    return _sandwich_cov(X, np.asarray(y, dtype=float) - X @ beta, estimator)


def covariance_from_jacobian(jac: np.ndarray, resid: np.ndarray, estimator: str = HC3,
                             cond_limit: float = COND_LIMIT):
    """HC3 (default) or OLS covariance for a converged NONLINEAR least-squares fit.

    Given the model Jacobian `jac` (n x p, d f(x_i; theta)/d theta_j at the optimum) and
    the residuals `resid` (e_i = y_i - f(x_i; theta)), returns
    `(cov, cond, well_conditioned)`:

      - `cov`  : the p x p parameter covariance from the same sandwich the linear fits
                 use, applied to the local linearization f(x; theta) ~ J theta + const
                 about the optimum. HC3 keeps it from under-reporting under the
                 non-uniform precision of an angle/concentration series (invariant 8).
      - `cond` : condition number of the column-normalized Jacobian (`_design_cond`),
                 the numerical-health measure of parameter identifiability.
      - `well_conditioned` : `cond <= cond_limit`.

    A Jacobian too ill-conditioned to carry a covariance (an unidentifiable parameter is
    the limiting case) yields a NaN covariance, so a downstream `propagate`/`_diag_se` ->
    `se_or_none` returns None ("no defensible SE") rather than a spuriously tight one --
    see `_normal_equations` for that boundary, which is `COV_COND_LIMIT` and is deliberately NOT
    the `cond_limit` reported here. Because a nonlinear-fit SE is an asymptotic /
    linearized quantity, any SE built on it and propagated through a further nonlinear
    transform (e.g. Mw = 1/inv_Mw) must still be Monte-Carlo-validated before it ships
    (invariant 8).
    """
    jac = np.asarray(jac, dtype=float)
    resid = np.asarray(resid, dtype=float)
    cov = _sandwich_cov(jac, resid, estimator)
    cond = _design_cond(jac)
    return cov, cond, cond <= cond_limit

def _diag_se(var: float) -> float:
    """Square-root of a covariance-diagonal variance, guarding sign as well as finiteness.

    A near-singular design can make an HC3/OLS covariance diagonal come out as a tiny
    NEGATIVE number in float arithmetic (it should be >= 0). math.sqrt would then raise a
    bare ValueError('math domain error'); we return NaN instead (-> se_or_none -> None),
    so a degenerate fit yields "no defensible SE" rather than crashing the analysis.

    A variance of exactly ZERO is WITHHELD (NaN -> se_or_none -> None), the same rule
    `propagate` applies to its delta-method variance and `replicate_mean_se` applies to a
    zero replicate spread.

    The reasoning, which is an invariant-8 one. **An SE of exactly zero is not a
    measurement claim; it asserts infinite precision.**

    TWO distinct causes reach `var == 0`, and both are withheld for that one reason:

      1. An EXACT fit. Whether a residual-free design leaves bitwise-zero residuals or
         residuals at ~1e-15 depends on the design AND on the machine: numpy ships OpenBLAS
         built DYNAMIC_ARCH, so `lstsq` picks its kernel from the running CPU. Covariance
         diagonals measured on `y = 3x + 2`, exact for all four designs, on two builds --
         LEFT an AVX-512 desktop, RIGHT a Core Ultra 9 185H laptop without it (2026-08-03):

             x = [0, 1, 2, 3, 4]    [1.69e-30, 9.19e-31]  |  [0.0, 0.0]  <- DISAGREE
             x = [1, 2, 3, 4, 5]    [0.0,      0.0     ]  |  [0.0, 0.0]
             x = [2, 4, 6, 8, 10]   [3.13e-28, 7.28e-30]  |  [1.71e-28, 5.06e-30]
             x = [10, 20, ..., 50]  [8.75e-28, 2.09e-30]  |  [8.75e-28, 2.09e-30]

         A zero diagonal withholds and a positive one reports, so which branch a given exact
         design takes is NOT readable off this table -- it is evidence about these two
         machines, and two of the four rows already disagree. `test_linear_fit_exact_line`
         happened to use a design that was positive on the machine it was written on, which
         is how the "an exact fit never reaches this branch" premise survived being written
         down; it now accepts either outcome there and pins the withheld one on `x = [1..5]`,
         which is bitwise zero on both.
      2. UNDERFLOW. A noisy through-origin design at |x| ~ 1e-200, whose true SE is
         ~3.8e-5, produced a covariance diagonal of exactly 0.0 and reported SE = 0.0 --
         a genuine UNDER-report, the one direction invariant 8 forbids outright, reached
         by a float-boundary accident rather than by a decision.

    Cause 1 is honest-but-uninformative and cause 2 is an under-report, so telling them
    apart would buy a number only in the case where there is no measurement information
    to report anyway. Aligning the siblings rather than documenting the divergence is
    likewise deliberate: they answer the same question and nothing in the code connects
    them, which is exactly how they came to disagree unnoticed. The alternative considered
    and not taken was to trust a zero variance only when the residuals are genuinely zero;
    it is strictly more precise and strictly more code, and is the fallback if some caller
    ever turns out to need a number for an exact fit.

    What changes downstream: `format_pm` already renders 0.0 and None identically (its
    `has_se` requires `se > 0`), so no displayed value moves. A zero-variance SE now writes
    an EMPTY export cell instead of `0` -- the existing missing-value convention -- and the
    DDLS "± statistical" caveat line, which branches on `is not None`, is correctly
    suppressed where no ± is shown."""
    return math.sqrt(var) if (np.isfinite(var) and var > 0) else float('nan')


def _design_cond(X: np.ndarray) -> float:
    """2-norm condition number of a design matrix X after COLUMN-NORMALIZING it, robust
    to a singular design.

    We scale each column to unit 2-norm before taking the condition number. This is
    deliberate: the raw cond([1, q^2, c]) is dominated by the arbitrary UNIT scales of
    the columns (q^2 in nm^-2 ~ 1e-4, c in g/mL ~ 1e-3, intercept = 1), so it would
    change if concentration were expressed in mg/mL instead of g/mL -- making a "raw
    cond" flag depend on the unit choice, not the physics. Column-normalizing removes
    that per-column scaling and leaves the genuine collinearity (near-equal
    concentrations, clustered angles), so the criterion is invariant to the units the
    abscissas happen to be in -- the scale-invariance invariant 4 asks for. A healthy
    design then conditions at O(1-100); a degenerate one explodes.

    np.linalg.cond returns +inf for an exactly singular design; we also map a NaN (from
    a non-finite entry, e.g. an infinite intensity already propagated into X, or a
    zero-norm column) to +inf, so `cond <= cond_limit` cleanly reads False for any
    degenerate design rather than letting a NaN slip through the comparison.

    The column scaling comes from the shared `_column_equilibrate`, the same helper
    `_normal_equations` uses, so the reported conditioning and the covariance's own computability
    verdict can never be measured on different matrices.

    **`cond_limit` vs `COV_COND_LIMIT`.** The two are independent thresholds, which matters
    when reading the shipped documentation. `cond_limit` has no Settings, GUI or controller
    surface -- it is a function default that is always `COND_LIMIT` in the shipped program
    -- so for every configuration a user can reach the two coincide, and "the
    ill-conditioning flag fired" and "no +/- is reported" are the same statement. That is
    why the guide and the tooltip say exactly that and no more. They come apart only from a
    caller passing a different `cond_limit`, i.e. from test or library code: raising it
    suppresses the flag while the covariance is still withheld, lowering it fires the flag
    while the covariance is still computed. Pinned by
    `tests/test_uncertainty.py::test_a_tuned_cond_limit_does_not_change_whether_an_se_exists`.
    """
    X = np.asarray(X, dtype=float)
    if not np.all(np.isfinite(X)):
        return float('inf')          # np.linalg.cond RAISES on non-finite input
    Xn, _ = _column_equilibrate(X)
    try:
        c = float(np.linalg.cond(Xn))
    except np.linalg.LinAlgError:
        return float('inf')
    return c if math.isfinite(c) else float('inf')


@dataclass
class LinearFit:
    """OLS y = intercept + slope*x with standard errors and covariance.

    `cov` is the 2x2 covariance of (intercept, slope), in that order. SEs are NaN
    when there are too few points to estimate the residual variance: HC3 (p = 2)
    needs n - p >= 2 residual dof (n >= 4), classical OLS needs n - p >= 1 (n >= 3).
    `estimator` records which covariance estimator produced `cov`/the SEs ('hc3' | 'ols').

    `cond` is the condition number of the design X = [1, x] and `well_conditioned` is
    `cond <= cond_limit` (the numerical-health flag, default ceiling `COND_LIMIT`). A
    False here means the design is near-degenerate and any 1/coefficient extrapolation
    (Mw = 1/intercept, etc.) is numerically unreliable regardless of magnitude; callers
    that build such a quantity gate their reliability flag on it.

    `rank` is the rank `np.linalg.lstsq` used on the RAW design and `rank_deficient` is
    `rank < 2`, i.e. **the solver did not use the full design**. It is deliberately NOT
    part of `well_conditioned`: `cond` is a scale-free collinearity measure (see
    `_design_cond`), while this says the solve itself was reduced. Both can be true at
    once and neither implies the other.

    Two ways to be reduced, and the flag covers both rather than pretending to separate
    them: the solver TRUNCATED a direction (the interesting case), or the input was not
    finite so no solve happened at all and `rank` is 0. A caller that needs to tell them
    apart should read the coefficients, which are all-NaN in the second case.
    Both fitters use 0 as the "not recorded" default, so the two agree.

    **`rank_deficient` is a DIAGNOSTIC with no consumer yet, and that is deliberate.**
    Unlike `well_conditioned` — which ~12 result types copy and gate their `*_reliable`
    flags on — nothing downstream reads it: no result dataclass, no export column, no GUI
    label. Propagating it means deciding, per analysis, whether a truncated solve should
    suppress a reliability flag and what the note should say, and it is not reachable from
    any data a parser accepts (it needs column magnitudes ~1e120 apart), so it is recorded
    rather than half-propagated. Do not describe it as a user-visible flag until it reaches
    a surface.
    """
    slope: float
    intercept: float
    slope_se: float
    intercept_se: float
    cov: np.ndarray              # 2x2, order [intercept, slope]
    r_squared: float
    n: int
    estimator: str = HC3
    cond: float = float('nan')
    well_conditioned: bool = True
    cond_limit: float = COND_LIMIT
    rank: int = 0                # rank lstsq used on the RAW design (0 = not recorded)
    rank_deficient: bool = False  # rank < number of columns -> the full design was not used


def linear_fit(x: Sequence[float], y: Sequence[float], estimator: str = HC3,
               cond_limit: float = COND_LIMIT) -> LinearFit:
    """OLS y = a + b x with robust HC3 (default) or classical OLS standard errors.
    (Theory-and-Equations-Guide §6.1, HC3/OLS covariance & slope-intercept SE)

    Also reports the design condition number `cond = cond([1, x])` and the numerical-health
    flag `well_conditioned = cond <= cond_limit` (see `COND_LIMIT`).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = int(x.size)
    X = np.column_stack([np.ones_like(x), x])
    beta, rank = _lstsq_coeffs_and_rank(X, y)
    a, b = float(beta[0]), float(beta[1])
    ss_res = float(np.sum((y - X @ beta) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = r_squared_or_nan(ss_res, ss_tot)
    cov = _cov(X, y, beta, estimator)                # order [intercept, slope]
    a_se = _diag_se(cov[0, 0])
    b_se = _diag_se(cov[1, 1])
    cond = _design_cond(X)
    return LinearFit(b, a, b_se, a_se, cov, r2, n, estimator,
                     cond=cond, well_conditioned=cond <= cond_limit, cond_limit=cond_limit,
                     rank=rank, rank_deficient=rank < X.shape[1])


def linear_fit_through_origin(x: Sequence[float], y: Sequence[float], estimator: str = HC3):
    """OLS slope of y = b x (no intercept) with robust HC3 (default) or classical OLS SE.
    (Theory-and-Equations-Guide §6.1, through-origin slope SE)

    b = sum(xy)/sum(x^2). Returns (slope, slope_se); with p = 1 fitted parameter,
    slope_se is NaN for n < 3 under HC3 (needs n - p >= 2) and n < 2 under classical
    OLS (needs n - p >= 1).

    The slope comes from the shared `_lstsq_coeffs` so that this member handles awkward
    input EXACTLY as its siblings do. It previously computed sum(xy)/sum(x^2) inline and
    was the one path in the family never routed through that helper, which made it diverge
    from `linear_fit` in TWO regimes -- the same parity defect twice over:

      - a NON-FINITE response made `x @ y` infinite and returned slope = +inf, a
        finite-looking and wildly wrong number, where `linear_fit` returned NaN;
      - an extreme but perfectly FINITE abscissa (|x| >~ 1e160) overflowed `x @ x`, so the
        ratio evaluated to inf/inf = NaN and the slope was withheld by accident, where
        `linear_fit` returned the correct value. Measured: at |x| ~ 1e200 the inline form
        gave NaN and the shared path gives 3.0 for y = 3x.

    The two forms are algebraically identical and agree wherever both are representable;
    they differ only where forming `x @ x` overflows, which is the same overflow shape again (an
    extreme but representable input walking past a guard).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = int(x.size)
    # The question this gate asks is "is the abscissa all zeros?", where the slope is
    # genuinely undefined (`_lstsq_coeffs` would hand back a min-norm 0.0). Ask it of x
    # directly. The earlier form, `float(x @ x) > 0`, asked it of a COMPUTED quantity that
    # underflows: for |x| <~ 1e-165 the sum of squares reaches exactly 0.0 while every
    # entry is a perfectly good number, so the slope was refused on a representable design
    # (measured: |x| ~ 1e-200 gave NaN where the fit is exactly 3.0). That is the
    # same shape as the column-norm overflow this module fixed one function up -- the
    # OTHER end of the float range, left open by that fix and closed here.
    has_abscissa = bool(np.any(x != 0.0))
    b = float(_lstsq_coeffs(x.reshape(-1, 1), y)[0]) if has_abscissa else float('nan')
    if n > 1 and has_abscissa:
        cov = _cov(x.reshape(-1, 1), y, np.array([b]), estimator)
        b_se = _diag_se(cov[0, 0])
    else:
        b_se = float('nan')
    return b, b_se


@dataclass
class MultiFit:
    """Multilinear OLS y = X b (caller supplies the full design X, incl. intercept).

    `estimator` records which covariance estimator produced `cov` ('hc3' | 'ols').
    `cond` is the condition number of the supplied design X and `well_conditioned` is
    `cond <= cond_limit` (numerical-health flag, default ceiling `COND_LIMIT`) — a False
    means the Zimm/Berry extrapolation design is near-degenerate (clustered angles or
    near-equal concentrations) and the 1/intercept Mw is numerically unreliable."""
    coeffs: np.ndarray
    cov: np.ndarray             # p x p covariance of coeffs
    r_squared: float
    n: int
    estimator: str = HC3
    cond: float = float('nan')
    well_conditioned: bool = True
    cond_limit: float = COND_LIMIT
    rank: int = 0                 # rank lstsq used on the RAW design (0 = not recorded)
    rank_deficient: bool = False  # rank < p -> a direction was truncated; see LinearFit


def multilinear_fit(X: np.ndarray, y: Sequence[float], estimator: str = HC3,
                    cond_limit: float = COND_LIMIT) -> MultiFit:
    """OLS point estimate for a supplied design matrix X (n x p) with the selected covariance.

    The coefficients are ordinary least squares; the parameter covariance is, by default,
    the HC3 heteroscedasticity-consistent (sandwich) estimator (`_hc3_cov`), NOT the classical
    Cov(b) = s^2 (X^T X)^-1 -- so it does not under-report under non-uniform precision. Passing
    estimator='ols' selects the classical form for comparability (invariant 8 clause A).
    Used for the Zimm/Berry global fit ordinate = a + b q^2 + d c (X = [1, q^2, c]).
    Also reports `cond = cond(X)` and `well_conditioned = cond <= cond_limit` (see `COND_LIMIT`).
    (Theory-and-Equations-Guide §6.1, Zimm/Berry multilinear design)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    beta, rank = _lstsq_coeffs_and_rank(X, y)
    ss_res = float(np.sum((y - X @ beta) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = r_squared_or_nan(ss_res, ss_tot)
    cov = _cov(X, y, beta, estimator)
    cond = _design_cond(X)
    # Reported on BOTH fitters, not only the one the finding named: they are the same
    # shape, and this module's history is of a predicate corrected at one call site
    # (rule 10). The Zimm design is in fact where the truncation bites hardest.
    return MultiFit(beta, cov, r2, n, estimator,
                    cond=cond, well_conditioned=cond <= cond_limit, cond_limit=cond_limit,
                    rank=rank, rank_deficient=rank < p)


# ---------------------------------------------------------------------------
# Error propagation (first-order / delta method)
# ---------------------------------------------------------------------------

def se_or_none(x: Optional[float]) -> Optional[float]:
    """Normalize a standard error: return None unless it is a finite NON-NEGATIVE number
    (so 'no defensible uncertainty' is a clean None throughout, and NaN from a
    too-small fit never leaks into a displayed ±).

    Note the predicate admits exactly 0.0 and passes it through, which is why the
    zero-variance decision is taken at each PRODUCER -- at the operation that still knows
    what its zero means, rather than here, where the causes are already
    indistinguishable. The REGRESSION-side producers agree: `_diag_se`, `propagate` and
    `replicate_mean_se` all withhold a zero.

    That is a claim about those producers and NOT about the platform. The Monte-Carlo
    sites in `analysis/depolarization.py` hand this function a raw sample SD, and a
    collapsed MC spread still returns 0.0 through here -- measured with
    `rod_dimensions_from_diffusion(1e-11, 1e4, temperature_K=298.15,
    viscosity_Pa_s=8.9e-4, d_t_se=1e-30, d_r_se=1e-25, n_mc=50)` -> `diameter_se = 0.0`.
    Input SEs that small are not reachable from the GUI, so per rule 13 the class is not
    guarded here. The predicate is left admitting a zero so a future caller with a
    defensible one is not silently rewritten."""
    if x is None or not math.isfinite(x) or x < 0:
        return None
    return float(x)


def propagate(jac: Sequence[float], cov: np.ndarray) -> float:
    """Standard error of a scalar f(b): sqrt(J^T Cov J).
    (Theory-and-Equations-Guide §6.2, delta-method propagation)

    `jac` is the gradient of f w.r.t. the parameters; `cov` their covariance.
    Returns NaN if the covariance is not finite, or if the propagated variance is not
    strictly positive: a near-singular covariance can make J^T Cov J cancel to a tiny
    negative or exactly-zero value in float arithmetic, and reporting that as SE = 0.0
    would be a spuriously tight, UNDER-reported uncertainty (the cardinal invariant-8
    failure). We return NaN instead, which `se_or_none` maps to None ("no defensible
    uncertainty") — a case every caller already handles.
    """
    jac = np.asarray(jac, dtype=float)
    cov = np.asarray(cov, dtype=float)
    if not np.all(np.isfinite(cov)):
        return float('nan')
    var = float(jac @ cov @ jac)
    # isfinite: sqrt(inf) is inf, and an infinite SE is not an uncertainty. se_or_none
    # already rejects it downstream, so this is defence in depth -- but it keeps the
    # 'withhold rather than report a meaningless +/-' rule at the operation.
    return math.sqrt(var) if (math.isfinite(var) and var > 0) else float('nan')


def ratio_se(a: float, a_se: Optional[float],
             b: float, b_se: Optional[float]) -> Optional[float]:
    """SE of r = a/b for INDEPENDENT a, b: |a/b| sqrt((a_se/a)^2 + (b_se/b)^2).
    (Theory-and-Equations-Guide §6.2, ratio propagation)

    Returns None unless both SEs are available and a, b are non-zero (so the
    fractional terms are defined). Used for rho = Rg/Rh (Rg from SLS, Rh from DLS —
    independent measurements) and rho_v = I_VH/I_VV (depolarization).

    Invariant-8 carve-out (Monte-Carlo-characterized; see tests/test_uncertainty
    test_ratio_se_montecarlo_* and Theory-and-Equations-Guide §6.2). This first-order
    (delta-method) ratio SE tracks the sampling SD to within a few percent while the
    DENOMINATOR's fractional error is small, and is mildly ANTI-conservative (under-
    reports ~5-10%) once the denominator CV grows past ~0.15 — the heavy tail of 1/b as b
    approaches zero. The direction is set by which channel is noisy: rho = Rg/Rh at a
    viscous-solvent Rh CV of 0.10-0.15 under-reports ~4-10%; rho_v = I_VH/I_VV is IMMUNE
    (its noisy channel, I_VH, is the numerator). The honest remedy for a weak denominator
    is more points/replicates (a smaller denominator CV), not a wider algebraic SE."""
    if a_se is None or b_se is None:
        return None
    if a == 0 or b == 0 or not (math.isfinite(a_se) and math.isfinite(b_se)):
        return None
    # math.hypot, not sqrt(x**2 + y**2): the fractional errors are SQUARED, so a small
    # denominator overflows the square long before the fraction itself is remarkable --
    # `ratio_se(1e-160, 1.0, 1.0, 1.0)` raised OverflowError from inside an SE helper
    # whose contract is to return None when it cannot answer. hypot is the canonical
    # overflow-safe form of exactly this expression (it scales internally), so the fix
    # removes the intermediate rather than guarding it.
    out = abs(a / b) * math.hypot(a_se / a, b_se / b)
    return out if (math.isfinite(out) and out > 0) else None


def power_law_se(f_value: float, x_value: float, x_se: Optional[float],
                 exponent: float) -> Optional[float]:
    """SE of f = k * x^exponent from x's SE: |exponent| |f| (x_se/|x|).
    (Theory-and-Equations-Guide §6.2, power-law propagation)

    For Rh = kT/(6 pi eta D) ~ D^-1, exponent = -1 gives sigma_Rh/Rh = sigma_D/D.
    Returns None if x_se is None/non-finite or x is zero.

    Invariant-8 note for the reciprocal (exponent = -1, e.g. Rh = C/D and Rh0 = C/D0).
    This is the first-order (delta-method) SE of a nonlinear transform, NOT a linear
    pass-through: it is the exact fractional relation sigma_f/f = |exponent| sigma_x/x,
    Monte-Carlo-validated against the sampling SD of C/D (tests/test_uncertainty
    test_power_law_se_reciprocal_montecarlo). It tracks the sampling spread where a real
    multi-angle Gamma-q^2 fit lives (sigma_D/D <~ 0.05) and is mildly anti-conservative
    (~10% under) only once sigma_D/D reaches ~0.15, with the reciprocal tail dominating
    beyond ~0.3 — see the ratio_se carve-out and Theory-and-Equations-Guide §6.2. The
    sphere radius R_rot ∝ D_r^(-1/3) (analysis.depolarization) is a second reciprocal-tail
    case, at one third the slope of the D^-1 case above; it carries its own Monte-Carlo
    gate (tests/test_depolarization.py test_sphere_radius_monte_carlo_se). Any new
    fractional-exponent call site needs its own Monte-Carlo gate before it ships
    (invariant 8) — this analysis does not extend to an untested exponent."""
    if x_se is None or not math.isfinite(x_se) or x_value == 0:
        return None
    # The RESULT is tested, not only the inputs: at a subnormal x the fractional error
    # saturates and this returned `inf`, and unlike its siblings the value is assigned
    # straight onto RodShapeResult/DDLSResult without passing through `se_or_none`, so an
    # infinite "uncertainty" would reach a result object.
    out = abs(exponent) * abs(f_value) * (x_se / abs(x_value))
    return out if (math.isfinite(out) and out > 0) else None


# ---------------------------------------------------------------------------
# Replicate (repeat-measurement) statistics  --  the ONE licensed DLS uncertainty
# ---------------------------------------------------------------------------

@dataclass
class ReplicateStats:
    """Mean and standard error of a quantity measured across true replicates.

    This is the only honest uncertainty for a single-angle DLS dynamic result.
    A single correlogram cannot supply one (its lag channels are correlated,
    Schaetzel 1990 -> a within-curve OLS covariance under-reports). The defensible
    estimate is the spread of the per-replicate values from repeat measurements
    (ISO 22412): the standard error of the mean, sem = sd / sqrt(n), with the
    sample SD (ddof = 1). `sem` is None for n < 2, where no spread is defined,
    and also when the spread is exactly zero -- see `replicate_mean_se`.
    """
    mean: float
    sd: Optional[float]              # sample SD (ddof=1); None for n < 2 or a zero spread
    sem: Optional[float]            # sd / sqrt(n); None wherever `sd` is
    n: int


def replicate_mean_se(values: Sequence[float]) -> ReplicateStats:
    """Mean +/- standard error of a parameter across replicate measurements.
    (Theory-and-Equations-Guide §6.4, replicate mean/SEM)

    For n independent repeats x_1..x_n of one quantity (e.g. Rh fitted from each
    of an ALV 10-run replicate set):

        mean = (1/n) sum_i x_i
        sd   = sqrt( sum_i (x_i - mean)^2 / (n - 1) )       (sample SD)
        sem  = sd / sqrt(n)

    Non-finite values are dropped first. Returns sd = sem = None when fewer than
    two finite values remain (a single datum has no spread). This is the ISO 22412
    repeat-measurement uncertainty, and the only place the platform reports a DLS
    dynamic ± -- it is deliberately NOT derived from one correlogram.

    A spread of exactly ZERO is WITHHELD too (sd = sem = None), which reaches the user
    as an empty ± cell rather than `0` (owner decision). The zero
    is real -- identical replicates are a genuine measurement that really did agree to
    the last bit, unlike an exact fit's residual-free covariance -- but reporting it
    would still be a claim of infinite precision, and 0.0 is the maximal under-report,
    the one direction invariant 8 forbids in the default path. It also happens rarely for
    the reason you would want to know about: a zero-noise synthetic set, or the same run
    loaded twice. This aligns the platform's three zero-variance producers -- `_diag_se`,
    `propagate` and this function -- which had silently disagreed; `n` is still reported,
    so "how many repeats" is not lost with the ±.
    """
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return ReplicateStats(mean=float('nan'), sd=None, sem=None, n=0)
    mean = float(arr.mean())
    if n < 2:
        return ReplicateStats(mean=mean, sd=None, sem=None, n=n)
    sd = float(arr.std(ddof=1))
    sem = sd / math.sqrt(n) if math.isfinite(sd) else float('nan')
    # Two-sided, and the predicate names BOTH quantities (rule 12): `sd` reaches zero from
    # identical replicates at the ordinary scale, while `sem` can reach it on its own by
    # UNDERFLOW -- a subnormal `sd` divided by sqrt(n) rounds to 0.0 with `sd` still
    # positive. `math.isfinite(sd)` closes the other tail, where replicates ~1e300 apart
    # saturate the SD to inf.
    if not (math.isfinite(sd) and sd > 0 and math.isfinite(sem) and sem > 0):
        return ReplicateStats(mean=mean, sd=None, sem=None, n=n)
    return ReplicateStats(mean=mean, sd=se_or_none(sd), sem=se_or_none(sem), n=n)


# ---------------------------------------------------------------------------
# Display: value +/- uncertainty with significant figures driven by the SE
# ---------------------------------------------------------------------------

def _pdg_decimals(sigma: float) -> int:
    """The decimal place (as ``round()``'s ``ndigits``) the PDG-style convention
    assigns to an uncertainty: round sigma to 1 significant figure — 2 if its
    leading digit is 1 — and report the value at that same place. The single
    place-chooser shared by ``format_pm`` and ``round_to_uncertainty`` so the
    display convention and the autofill rounding can never drift apart."""
    # BOTH the exponent and the leading digit come from ONE read of the value -- the point
    # of this line, not an incidental economy. Sourcing them separately (`floor(log10)` for
    # the exponent, the rendering for the digit) makes two decade decisions that disagree
    # wherever the double sits a hair below a power of ten: `log10` rounds up while the
    # rendering shows `9.99...e(exp-1)`. The width matters too: `.17e` is the exact
    # round-trip width of a double, so the mantissa is never rounded across a decade, where
    # the default six-decimal `%e` renders 9.9999999e-5 as '1.000000e-04' -- a leading digit
    # of 1 where the true one is 9.
    mantissa, _, exponent = f'{sigma:.17e}'.partition('e')
    exp = int(exponent)
    lead = int(mantissa[0])                    # first significant digit of sigma
    sig = 2 if lead == 1 else 1
    return -(exp - (sig - 1))                  # decimals for round()


def round_to_uncertainty(value: Optional[float],
                         sigma: Optional[float]) -> Optional[float]:
    """Round ``value`` at the last digit ``sigma`` can stand behind (the PDG-style
    place of ``format_pm``) — e.g. value 1.3325541, sigma 6e-4 -> 1.3326.

    Used by the solvent-library autofill to strip false precision from proposed
    values before they are written (display-class perturbation, <= half a unit in
    the last confident digit; the sigma chooses the decimal place and travels no
    further — invariant #8). Idempotent. A missing/zero/non-finite sigma means
    there is nothing to stand behind, so the value is returned unrounded.
    """
    has_val = value is not None and math.isfinite(value)
    has_sig = sigma is not None and math.isfinite(sigma) and sigma > 0
    if not has_val or not has_sig:
        return value
    return round(value, _pdg_decimals(sigma))


def _format_sig(value: float, sig: int) -> str:
    """Render a finite ``value`` at ``sig`` significant figures in the house
    convention: scientific notation with a BARE integer exponent when
    ``|value| >= 1e5`` or ``0 < |value| < 1e-3`` — the SAME threshold and exponent
    style ``format_pm`` uses for a ± value — otherwise fixed-point. Trailing zeros
    are stripped: a value with no σ has nothing to make them confident (unlike
    ``format_value_at_uncertainty``, whose trailing zeros ARE σ-backed).

    This is the shared engine for every ±-less bare-value display, so a no-±
    result never renders '1.23e+06' next to a ± result's '(1.23 ± …)e6'.
    """
    if value == 0:
        return '0'
    # Round to `sig` significant figures FIRST (as format_pm rounds the value to the SE
    # place first), so a large value never shows MORE digits than `sig` — e.g. 3421.7 at
    # 3 sig -> 3420, not 3422 — then decide sci vs fixed on the ROUNDED magnitude (a value
    # that rounds up across 1e5, e.g. 99999 -> 1.00e5, then renders in scientific form).
    exp10 = math.floor(math.log10(abs(value)))
    ndig = sig - 1 - exp10                            # sig-fig rounding place (may be <= 0)
    r = round(value, ndig)
    if r == 0:
        return '0'
    mag = abs(r)
    if mag >= 1e5 or mag < 1e-3:
        # %e re-expresses the already-rounded value with correct carry (9.999e5 -> 1e6).
        mant, _, e = f'{r:.{sig - 1}e}'.partition('e')
        mant = mant.rstrip('0').rstrip('.')
        return f'{mant}e{int(e)}'                     # int() drops the '+' and zero-padding
    s = f'{r:.{max(0, ndig)}f}'                       # sig figs -> decimal places
    return s.rstrip('0').rstrip('.') if '.' in s else s


def format_pm(value: Optional[float], se: Optional[float], unit: str = '') -> str:
    """Format 'value +/- se' with the precision set by the uncertainty.

    Convention (PDG-style): round the SE to 1 significant figure, or 2 if its
    leading digit is 1; round the value to the same decimal place (the place
    comes from the shared ``_pdg_decimals``). Large/small magnitudes are shown
    with a shared power of ten, e.g. '(1.23 +/- 0.06)e6'. Falls back to a plain
    value when no usable SE is given.
    """
    has_val = value is not None and math.isfinite(value)
    has_se = se is not None and math.isfinite(se) and se > 0
    if not has_val:
        return 'n/a'
    if not has_se:
        s = _format_sig(value, 4)                   # ±-less fallback: house sig-fig render
        return f'{s} {unit}' if unit else s

    exp = math.floor(math.log10(se))            # SE's order (sci-notation fallback)
    dec = _pdg_decimals(se)
    last = -dec                                 # power-of-ten place of the last sig fig
    se_r = round(se, dec)
    val_r = round(value, dec)

    mag = max(abs(val_r), se_r)
    use_sci = mag != 0 and (mag >= 1e5 or mag < 1e-3)
    if use_sci:
        # isfinite: `!= 0` admits both inf and nan (nan != 0 is True), and
        # math.floor(math.log10(inf)) raises OverflowError. UNREACHABLE, though, and
        # recorded as such: format_pm returns 'n/a' at its `has_val` check above unless
        # `value` is finite, and rounding a finite value cannot produce a non-finite one.
        # Kept as defence in depth for a future caller that reaches this helper by another
        # route -- it was reported as a live crash on the display path, and it is not.
        E = (math.floor(math.log10(abs(val_r)))
             if (math.isfinite(val_r) and val_r != 0) else exp)
        mant_dec = max(0, E - last)
        v, s = val_r / 10 ** E, se_r / 10 ** E
        body = f'({v:.{mant_dec}f} ± {s:.{mant_dec}f})e{E}'
    else:
        d = max(0, dec)
        body = f'{val_r:.{d}f} ± {se_r:.{d}f}'
    return f'{body} {unit}' if unit else body


def pdg_erases_value(value: Optional[float], se: Optional[float]) -> bool:
    """True when the PDG place chosen by ``se`` rounds a NON-ZERO ``value`` to zero.

    The display convention sets the value's precision from its uncertainty, which is
    right until the uncertainty is so much larger that the value's own leading digit
    sits below the place sigma supports: ``30 +/- 900`` renders as ``0 +/- 900`` and the
    measurement is gone from the cell. The condition is not "sigma exceeds the value"
    (``30 +/- 90`` renders fine and is informative) and not any ratio -- pick a ratio and
    it is a magic number, since the boundary is a decade artifact of the rounding, not a
    property of the quantity. So this asks the actual question, through the same
    ``_pdg_decimals`` the renderer uses, and cannot drift from it.

    A caller that gets True should show the value at its own fixed precision and the
    uncertainty beside it, rather than through it (``format_fixed_sig`` on each) -- the
    number and its spread are both real and both belong on screen. Nothing here changes
    what an SE IS: the estimator's value is untouched, and exports, which write value and
    SE in separate full-precision columns, never reach this question.

    False whenever there is nothing to erase: no value, no usable SE, or a value that is
    itself zero (the render is then honest).
    """
    if value is None or se is None:
        return False
    if not (math.isfinite(value) and math.isfinite(se)) or se <= 0 or value == 0:
        return False
    return round(value, _pdg_decimals(se)) == 0


def format_fixed_sig(value: Optional[float], sig: int = 3) -> str:
    """Format ``value`` at a fixed number of significant figures — the display rule
    for **every result that has no honest uncertainty** (invariant #8).

    Used wherever an SE is genuinely unavailable, not merely absent from the display:
    a single-correlogram dynamic fit (correlated lag channels, Schaetzel 1990), an
    ill-posed inversion (NNLS/CONTIN), a single datum (single-angle Mw), and every
    other ±-less reported quantity (Γ, β, τ, PDI, ρ, δ², R², qRg, …). We must NOT
    fabricate a ± just to drive the σ-aware formatter (``format_value_at_uncertainty``);
    instead we show a modest, fixed precision that carries no confidence claim — the
    number's honesty is carried by its apparent/±-omitted qualifier (invariant #7),
    the digit count by this rule.

    Standing policy (see the Theory-and-Equations-Guide §6.6): a value that HAS an uncertainty is
    set by that uncertainty and never comes here; a value WITHOUT one is set by ``sig``,
    which is user-configurable (Settings → the no-uncertainty precision knob, default 3).
    Because any sig-fig choice for an unquantified number is inherently arbitrary, that is
    no invariant-8 violation — nothing is shown as more certain than its ± supports.
    Rendering (scientific-notation threshold + bare exponent) is shared with ``format_pm``
    via ``_format_sig``. Returns 'n/a' for a missing/non-finite value.
    """
    if value is None or not (isinstance(value, (int, float)) and math.isfinite(value)):
        return 'n/a'
    return _format_sig(float(value), sig)


def format_value_at_uncertainty(value: Optional[float],
                                sigma: Optional[float]) -> str:
    """Format ``value`` at the decimal place ``sigma`` stands behind, KEEPING the
    confident trailing zeros that place implies (e.g. value 1.33, sigma 6e-4 ->
    '1.3300') — the bare-value companion to :func:`format_pm`, with no '+/- se'.

    Used for a display-only readout of an already-rounded, uncertainty-tagged value
    (the Data-tab solvent-library cells): the value is shown to exactly the precision
    the library's per-condition sigma supports, so a rounded 1.3300 does not collapse
    to '1.33'. The place comes from the shared :func:`_pdg_decimals`, so it can never
    drift from :func:`round_to_uncertainty` (the autofill rounding). Falls back to a
    clean general format when sigma is missing/zero/non-finite (nothing to stand
    behind). The sigma chooses the decimal place and travels no further (invariant #8).
    """
    if value is None or not math.isfinite(value):
        return ''
    has_sig = sigma is not None and math.isfinite(sigma) and sigma > 0
    if not has_sig:
        return f'{value:.10g}'
    # Round at the σ place FIRST (honoring a negative ndigits for a coarse σ, e.g.
    # σ≈60 -> the tens place), exactly as round_to_uncertainty/format_pm do, THEN
    # show that place — so the two can never drift. `max(0, dec)` only clips the
    # DISPLAY decimals (a place at/left of the units digit shows no fractional part).
    dec = _pdg_decimals(sigma)
    return f'{round(value, dec):.{max(0, dec)}f}'
