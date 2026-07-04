"""Regression tests for ``analysis/uncertainty.py``.

The statistical-uncertainty toolkit: HC3 linear regression, delta-method
propagation, ratio / power-law SEs, ISO 22412 replicate statistics, and the
``format_pm`` display helper. Almost everything here is deterministic and checked
to tight tolerance against a hand computation; the one Monte-Carlo coverage check
is seeded and marked ``slow``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from analysis import uncertainty as unc


# ---------------------------------------------------------------------------
# linear_fit
# ---------------------------------------------------------------------------

def test_linear_fit_exact_line():
    x = [0.0, 1.0, 2.0, 3.0, 4.0]
    y = [2.0 + 3.0 * xi for xi in x]      # exact line: slope 3, intercept 2
    fit = unc.linear_fit(x, y)
    assert fit.slope == pytest.approx(3.0, rel=1e-12)
    assert fit.intercept == pytest.approx(2.0, rel=1e-12)
    assert fit.r_squared == pytest.approx(1.0, rel=1e-12)
    assert fit.n == 5
    # A perfect fit has zero residuals -> zero SE (finite, not NaN).
    assert math.isfinite(fit.slope_se) and fit.slope_se == pytest.approx(0.0, abs=1e-9)
    assert math.isfinite(fit.intercept_se) and fit.intercept_se == pytest.approx(0.0, abs=1e-9)


def test_linear_fit_hc3_dof_guard():
    # n - p = 3 - 2 = 1 < 2 -> HC3 cannot support an SE; SEs come back NaN.
    fit = unc.linear_fit([0.0, 1.0, 2.0], [0.1, 2.2, 3.9])
    assert math.isnan(fit.slope_se)
    assert math.isnan(fit.intercept_se)
    assert not np.all(np.isfinite(fit.cov))
    # The point estimates are still returned.
    assert math.isfinite(fit.slope)
    assert math.isfinite(fit.intercept)


def test_linear_fit_ols_matches_textbook_covariance():
    # Classical OLS SE = sqrt(diag(s^2 (X^T X)^-1)), s^2 = RSS/(n-p). Check the
    # 'ols' estimator reproduces the hand-computed textbook value to machine
    # precision (a noisy line so the residuals -- hence the SE -- are nonzero).
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.3, 4.1, 6.5, 8.0, 10.2])   # ~ 2 + 2x with structured residuals
    fit = unc.linear_fit(x, y, estimator='ols')
    assert fit.estimator == 'ols'
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    s2 = float(np.sum((y - X @ beta) ** 2)) / (x.size - 2)
    cov = s2 * np.linalg.inv(X.T @ X)          # order [intercept, slope]
    assert fit.intercept_se == pytest.approx(math.sqrt(cov[0, 0]), rel=1e-12)
    assert fit.slope_se == pytest.approx(math.sqrt(cov[1, 1]), rel=1e-12)


def test_linear_fit_default_is_hc3_and_tagged():
    # No estimator arg -> HC3, and the choice is recorded on the fit object.
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.3, 4.1, 6.5, 8.0, 10.2])
    default = unc.linear_fit(x, y)
    hc3 = unc.linear_fit(x, y, estimator='hc3')
    assert default.estimator == 'hc3'
    # Default and explicit HC3 are bit-identical (no behaviour change for the default).
    assert default.slope_se == hc3.slope_se
    assert default.intercept_se == hc3.intercept_se
    # On this heteroscedastic-ish design OLS and HC3 genuinely differ.
    ols = unc.linear_fit(x, y, estimator='ols')
    assert ols.slope_se != pytest.approx(hc3.slope_se, rel=1e-6)


def test_ols_through_origin_and_multilinear_tagged():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.1, 2.3, 2.8, 4.2, 4.9])
    slope, se = unc.linear_fit_through_origin(x, y, estimator='ols')
    # hand OLS through origin: var(b) = s^2 / sum(x^2), s^2 = RSS/(n-1)
    b = float(x @ y) / float(x @ x)
    s2 = float(np.sum((y - b * x) ** 2)) / (x.size - 1)
    assert se == pytest.approx(math.sqrt(s2 / float(x @ x)), rel=1e-12)
    X = np.column_stack([np.ones_like(x), x])
    mf = unc.multilinear_fit(X, y, estimator='ols')
    assert mf.estimator == 'ols'


def test_unknown_estimator_raises():
    with pytest.raises(ValueError):
        unc.linear_fit([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], estimator='bogus')


def test_linear_fit_through_origin_exact():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = 5.0 * x
    slope, slope_se = unc.linear_fit_through_origin(x, y)
    assert slope == pytest.approx(5.0, rel=1e-12)
    assert math.isfinite(slope_se) and slope_se == pytest.approx(0.0, abs=1e-9)


def test_linear_fit_through_origin_single_point_no_se():
    slope, slope_se = unc.linear_fit_through_origin([2.0], [6.0])
    assert slope == pytest.approx(3.0)
    assert math.isnan(slope_se)


def test_multilinear_fit_exact_plane():
    # Zimm-style design X = [1, q^2, c]; exact plane y = a + b q^2 + d c.
    a, b, d = 4.0, 2.5, -1.5
    q2 = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    c = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    X = np.column_stack([np.ones_like(q2), q2, c])
    y = a + b * q2 + d * c
    fit = unc.multilinear_fit(X, y)
    assert fit.coeffs[0] == pytest.approx(a, rel=1e-10)
    assert fit.coeffs[1] == pytest.approx(b, rel=1e-10)
    assert fit.coeffs[2] == pytest.approx(d, rel=1e-10)
    assert fit.r_squared == pytest.approx(1.0, rel=1e-10)
    assert fit.n == 6


# ---------------------------------------------------------------------------
# propagation helpers
# ---------------------------------------------------------------------------

def test_propagate_matches_hand_computation():
    cov = np.array([[4.0, 1.0], [1.0, 9.0]])
    jac = np.array([2.0, -1.0])
    expected = math.sqrt(float(jac @ cov @ jac))  # J^T Cov J
    assert unc.propagate(jac, cov) == pytest.approx(expected, rel=1e-12)


def test_propagate_nonfinite_cov_is_nan():
    cov = np.array([[np.nan, 0.0], [0.0, 1.0]])
    assert math.isnan(unc.propagate([1.0, 1.0], cov))


def test_ratio_se_independent():
    a, a_se, b, b_se = 6.0, 0.3, 2.0, 0.1
    expected = abs(a / b) * math.sqrt((a_se / a) ** 2 + (b_se / b) ** 2)
    assert unc.ratio_se(a, a_se, b, b_se) == pytest.approx(expected, rel=1e-12)


def test_ratio_se_missing_input_is_none():
    assert unc.ratio_se(6.0, None, 2.0, 0.1) is None
    assert unc.ratio_se(6.0, 0.3, 0.0, 0.1) is None   # b == 0 undefined


def test_power_law_se_exponent_minus_one():
    # Rh = kT/(6 pi eta D) ~ D^-1 -> sigma_Rh/Rh = sigma_D/D.
    D, D_se, Rh = 1.5e-11, 3.0e-13, 25.0
    se = unc.power_law_se(Rh, D, D_se, exponent=-1.0)
    assert se == pytest.approx(Rh * (D_se / D), rel=1e-12)
    assert (se / Rh) == pytest.approx(D_se / D, rel=1e-12)


def test_power_law_se_missing_is_none():
    assert unc.power_law_se(25.0, 1.5e-11, None, -1.0) is None
    assert unc.power_law_se(25.0, 0.0, 3e-13, -1.0) is None


def test_se_or_none():
    assert unc.se_or_none(2.5) == 2.5
    assert unc.se_or_none(None) is None
    assert unc.se_or_none(float('nan')) is None
    assert unc.se_or_none(-1.0) is None


# ---------------------------------------------------------------------------
# replicate statistics (ISO 22412)
# ---------------------------------------------------------------------------

def test_replicate_mean_se_basic():
    stats = unc.replicate_mean_se([10.0, 12.0, 14.0])
    assert stats.mean == pytest.approx(12.0)
    assert stats.sd == pytest.approx(2.0)              # sample SD, ddof=1
    assert stats.sem == pytest.approx(2.0 / math.sqrt(3.0))
    assert stats.n == 3


def test_replicate_mean_se_single_value_no_spread():
    stats = unc.replicate_mean_se([7.0])
    assert stats.mean == pytest.approx(7.0)
    assert stats.sd is None
    assert stats.sem is None
    assert stats.n == 1


def test_replicate_mean_se_drops_nonfinite():
    stats = unc.replicate_mean_se([10.0, float('nan'), 12.0, 14.0, float('inf')])
    assert stats.n == 3
    assert stats.mean == pytest.approx(12.0)
    assert stats.sd == pytest.approx(2.0)


def test_replicate_mean_se_empty():
    stats = unc.replicate_mean_se([])
    assert stats.n == 0
    assert math.isnan(stats.mean)
    assert stats.sd is None and stats.sem is None


# ---------------------------------------------------------------------------
# format_pm display
# ---------------------------------------------------------------------------

def test_format_pm_no_value():
    assert unc.format_pm(None, 0.1) == 'n/a'


def test_format_pm_no_se_plain_value():
    # No usable SE -> plain 4-sig-fig value, no +/-.
    out = unc.format_pm(5.0, None)
    assert '5' in out and '±' not in out


def test_format_pm_se_one_sigfig():
    # SE leading digit != 1 -> 1 sig fig on the SE; value rounded to match.
    out = unc.format_pm(1.23456, 0.0678)
    assert '±' in out
    assert '1.23' in out and '0.07' in out


def test_format_pm_se_leading_one_two_sigfigs():
    # SE leading digit == 1 -> 2 sig figs.
    out = unc.format_pm(1.23456, 0.0123)
    assert '1.235' in out and '0.012' in out


def test_format_pm_scientific_shared_power():
    out = unc.format_pm(1234567.0, 60000.0)
    assert 'e6' in out and '1.23' in out


# ---------------------------------------------------------------------------
# HC3 Monte-Carlo coverage (seeded, slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_hc3_coverage_heteroscedastic():
    """HC3 must not under-report under heteroscedastic errors: the SD of the fitted
    slope across many synthetic draws should match the reported HC3 SE."""
    rng = np.random.RandomState(20240601)
    x = np.linspace(1.0, 10.0, 12)
    true_slope, true_intercept = 3.0, 2.0
    sigma = 0.05 * x          # error grows with x (heteroscedastic)
    slopes = []
    reported_ses = []
    for _ in range(400):
        y = true_intercept + true_slope * x + rng.normal(0.0, sigma)
        fit = unc.linear_fit(x, y)
        slopes.append(fit.slope)
        reported_ses.append(fit.slope_se)
    empirical_sd = float(np.std(slopes, ddof=1))
    mean_reported = float(np.mean(reported_ses))
    # HC3 tracks the sampling spread (not a strict equality; within ~25%).
    assert mean_reported == pytest.approx(empirical_sd, rel=0.25)
