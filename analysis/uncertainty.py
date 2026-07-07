"""
analysis/uncertainty.py
=======================

Statistical-uncertainty toolkit for the analysis layer.

Scope and philosophy (see the Advanced Guide, "Uncertainty estimates"):
we report the **statistical** (regression) uncertainty of a fit, and only where the
fitted points are genuinely independent measurements (SLS angle/concentration series,
Gamma-vs-q^2 over angles, D-vs-c over concentrations, the cross-sample scaling fit).
We do NOT manufacture uncertainties where none is defensible: a single-correlogram DLS
dynamic fit (cumulant/single/double/KWW/lognormal) has strongly correlated lag channels
(Schaetzel 1990), so an ordinary-least-squares covariance from one correlogram
under-reports the truth; that uncertainty belongs to repeat measurements (ISO 22412),
handled by the planned replicate-averaging, not here. Ill-posed inversions (NNLS/CONTIN)
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

The covariance estimator is selectable via an `estimator` argument on the fitters
('hc3', the default, vs 'ols', the classical s^2 (X^T X)^-1). HC3 is the default because
it never under-reports; classical OLS is an opt-in for comparability with legacy/literature
SEs and can under-report on short high-leverage designs (invariant 8 clause A). The choice
is recorded on the returned fit object (`.estimator`) so provenance travels with every SE.
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


def _hc3_cov(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """HC3 heteroscedasticity-consistent covariance of the OLS coefficients.

    Cov = (X^T X)^-1 [sum_i x_i x_i^T e_i^2/(1-h_ii)^2] (X^T X)^-1.  Returns NaN when
    the residual degrees of freedom are too few to estimate a spread (n - p < 2): a
    fit with one residual dof cannot honestly support an uncertainty, so we report
    none rather than a wildly inflated value (HC3 over-inflates as dof -> 1)."""
    n, p = X.shape
    if n - p < 2:
        return np.full((p, p), np.nan)
    XtX_inv = np.linalg.inv(X.T @ X)
    resid = y - X @ beta
    h = np.einsum('ij,jk,ik->i', X, XtX_inv, X)      # leverages h_ii
    w = resid ** 2 / np.clip((1.0 - h) ** 2, 1e-300, None)
    meat = (X * w[:, None]).T @ X                     # sum_i w_i x_i x_i^T
    return XtX_inv @ meat @ XtX_inv


def _ols_cov(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Classical homoscedastic OLS covariance Cov = s^2 (X^T X)^-1, s^2 = RSS/(n-p).

    The textbook estimator: assumes every point shares one error variance. Offered as
    an opt-in for comparability with classical software / literature tables (invariant 8
    clause A) — it under-reports on short, high-leverage designs where precision varies
    (Session 97: ~10% low on the 5-point calibration-free A2 ladder), which is why HC3,
    not this, is the default. Returns NaN when there are too few residual dof to estimate
    the spread (n - p < 1)."""
    n, p = X.shape
    if n - p < 1:
        return np.full((p, p), np.nan)
    XtX_inv = np.linalg.inv(X.T @ X)
    resid = y - X @ beta
    s2 = float(resid @ resid) / (n - p)
    return s2 * XtX_inv


def _cov(X: np.ndarray, y: np.ndarray, beta: np.ndarray, estimator: str = HC3) -> np.ndarray:
    """Parameter covariance by the selected estimator (default HC3 sandwich; OLS opt-in)."""
    if estimator == OLS:
        return _ols_cov(X, y, beta)
    if estimator == HC3:
        return _hc3_cov(X, y, beta)
    raise ValueError(f"unknown estimator {estimator!r}; expected one of {_ESTIMATORS}")

@dataclass
class LinearFit:
    """OLS y = intercept + slope*x with standard errors and covariance.

    `cov` is the 2x2 covariance of (intercept, slope), in that order. SEs are NaN
    when there are too few points to estimate the residual variance: HC3 (p = 2)
    needs n - p >= 2 residual dof (n >= 4), classical OLS needs n - p >= 1 (n >= 3).
    `estimator` records which covariance estimator produced `cov`/the SEs ('hc3' | 'ols').
    """
    slope: float
    intercept: float
    slope_se: float
    intercept_se: float
    cov: np.ndarray              # 2x2, order [intercept, slope]
    r_squared: float
    n: int
    estimator: str = HC3


def linear_fit(x: Sequence[float], y: Sequence[float], estimator: str = HC3) -> LinearFit:
    """OLS y = a + b x with robust HC3 (default) or classical OLS standard errors.  (Eq. 30, 30a, 32)"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = int(x.size)
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = float(beta[0]), float(beta[1])
    ss_res = float(np.sum((y - X @ beta) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    cov = _cov(X, y, beta, estimator)                # order [intercept, slope]
    a_se = math.sqrt(cov[0, 0]) if np.isfinite(cov[0, 0]) else float('nan')
    b_se = math.sqrt(cov[1, 1]) if np.isfinite(cov[1, 1]) else float('nan')
    return LinearFit(b, a, b_se, a_se, cov, r2, n, estimator)


def linear_fit_through_origin(x: Sequence[float], y: Sequence[float], estimator: str = HC3):
    """OLS slope of y = b x (no intercept) with robust HC3 (default) or classical OLS SE.  (Eq. 31)

    b = sum(xy)/sum(x^2). Returns (slope, slope_se); with p = 1 fitted parameter,
    slope_se is NaN for n < 3 under HC3 (needs n - p >= 2) and n < 2 under classical
    OLS (needs n - p >= 1).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = int(x.size)
    sxx = float(x @ x)
    b = float(x @ y) / sxx if sxx > 0 else float('nan')
    if n > 1 and sxx > 0:
        cov = _cov(x.reshape(-1, 1), y, np.array([b]), estimator)
        b_se = math.sqrt(cov[0, 0]) if np.isfinite(cov[0, 0]) else float('nan')
    else:
        b_se = float('nan')
    return b, b_se


@dataclass
class MultiFit:
    """Multilinear OLS y = X b (caller supplies the full design X, incl. intercept).

    `estimator` records which covariance estimator produced `cov` ('hc3' | 'ols')."""
    coeffs: np.ndarray
    cov: np.ndarray             # p x p covariance of coeffs
    r_squared: float
    n: int
    estimator: str = HC3


def multilinear_fit(X: np.ndarray, y: Sequence[float], estimator: str = HC3) -> MultiFit:
    """OLS point estimate for a supplied design matrix X (n x p) with the selected covariance.

    The coefficients are ordinary least squares; the parameter covariance is, by default,
    the HC3 heteroscedasticity-consistent (sandwich) estimator (`_hc3_cov`), NOT the classical
    Cov(b) = s^2 (X^T X)^-1 -- so it does not under-report under non-uniform precision. Passing
    estimator='ols' selects the classical form for comparability (invariant 8 clause A).
    Used for the Zimm/Berry global fit ordinate = a + b q^2 + d c (X = [1, q^2, c]).
    (Eq. 32)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    ss_res = float(np.sum((y - X @ beta) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    cov = _cov(X, y, beta, estimator)
    return MultiFit(beta, cov, r2, n, estimator)


# ---------------------------------------------------------------------------
# Error propagation (first-order / delta method)
# ---------------------------------------------------------------------------

def se_or_none(x: Optional[float]) -> Optional[float]:
    """Normalise a standard error: return None unless it is a finite positive number
    (so 'no defensible uncertainty' is a clean None throughout, and NaN from a
    too-small fit never leaks into a displayed ±)."""
    if x is None or not math.isfinite(x) or x < 0:
        return None
    return float(x)


def propagate(jac: Sequence[float], cov: np.ndarray) -> float:
    """Standard error of a scalar f(b): sqrt(J^T Cov J).  (Eq. 33)

    `jac` is the gradient of f w.r.t. the parameters; `cov` their covariance.
    Returns NaN if the covariance is not finite.
    """
    jac = np.asarray(jac, dtype=float)
    cov = np.asarray(cov, dtype=float)
    if not np.all(np.isfinite(cov)):
        return float('nan')
    var = float(jac @ cov @ jac)
    return math.sqrt(var) if var > 0 else 0.0


def ratio_se(a: float, a_se: Optional[float],
             b: float, b_se: Optional[float]) -> Optional[float]:
    """SE of r = a/b for INDEPENDENT a, b: |a/b| sqrt((a_se/a)^2 + (b_se/b)^2).  (Eq. 34)

    Returns None unless both SEs are available and a, b are non-zero (so the
    fractional terms are defined). Used for rho = Rg/Rh (Rg from SLS, Rh from DLS —
    independent measurements)."""
    if a_se is None or b_se is None:
        return None
    if a == 0 or b == 0 or not (math.isfinite(a_se) and math.isfinite(b_se)):
        return None
    return abs(a / b) * math.sqrt((a_se / a) ** 2 + (b_se / b) ** 2)


def power_law_se(f_value: float, x_value: float, x_se: Optional[float],
                 exponent: float) -> Optional[float]:
    """SE of f = k * x^exponent from x's SE: |exponent| |f| (x_se/|x|).  (Eq. 35)

    For Rh = kT/(6 pi eta D) ~ D^-1, exponent = -1 gives sigma_Rh/Rh = sigma_D/D.
    Returns None if x_se is None/non-finite or x is zero."""
    if x_se is None or not math.isfinite(x_se) or x_value == 0:
        return None
    return abs(exponent) * abs(f_value) * (x_se / abs(x_value))


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
    sample SD (ddof = 1). `sem` is None for n < 2, where no spread is defined.
    """
    mean: float
    sd: Optional[float]              # sample SD (ddof=1); None for n < 2
    sem: Optional[float]            # sd / sqrt(n); None for n < 2
    n: int


def replicate_mean_se(values: Sequence[float]) -> ReplicateStats:
    """Mean +/- standard error of a parameter across replicate measurements.  (Eq. 36)

    For n independent repeats x_1..x_n of one quantity (e.g. Rh fitted from each
    of an ALV 10-run replicate set):

        mean = (1/n) sum_i x_i
        sd   = sqrt( sum_i (x_i - mean)^2 / (n - 1) )       (sample SD)
        sem  = sd / sqrt(n)

    Non-finite values are dropped first. Returns sd = sem = None when fewer than
    two finite values remain (a single datum has no spread). This is the ISO 22412
    repeat-measurement uncertainty, and the only place the platform reports a DLS
    dynamic ± -- it is deliberately NOT derived from one correlogram.
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
    sem = sd / math.sqrt(n)
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
    exp = math.floor(math.log10(sigma))
    lead = int(sigma / 10 ** exp)              # first significant digit of sigma
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
        s = f'{value:.4g}'
        return f'{s} {unit}' if unit else s

    exp = math.floor(math.log10(se))            # SE's order (sci-notation fallback)
    dec = _pdg_decimals(se)
    last = -dec                                 # power-of-ten place of the last sig fig
    se_r = round(se, dec)
    val_r = round(value, dec)

    mag = max(abs(val_r), se_r)
    use_sci = mag != 0 and (mag >= 1e5 or mag < 1e-3)
    if use_sci:
        E = math.floor(math.log10(abs(val_r))) if val_r != 0 else exp
        mant_dec = max(0, E - last)
        v, s = val_r / 10 ** E, se_r / 10 ** E
        body = f'({v:.{mant_dec}f} ± {s:.{mant_dec}f})e{E}'
    else:
        d = max(0, dec)
        body = f'{val_r:.{d}f} ± {se_r:.{d}f}'
    return f'{body} {unit}' if unit else body
