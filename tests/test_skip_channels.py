"""Regression tests for the DLS leading-channel skip (``skip_initial_channels``).

Ported from the retired channel-skip validator, groups A-E:
  * A  engine correctness / non-breaking no-op (skip=0 byte-identical to default),
  * B  tau_min interaction matrix via ``_apply_tau_window`` (skip vs tau_min,
       whichever cuts more wins),
  * C  boundary / error handling (min_points, negative/non-integer/oversized skip,
       afterpulse-spike removal),
  * D  distribution methods (NNLS / CONTIN / lognormal),
  * E  real-data regression on the committed SMALS Noisy050Latex0004 set.

Group F (controller + Settings + GUI persistence) is covered in the workspace /
GUI tiers, not here.

Tolerances and bounded ranges are copied verbatim from the source validator.
"""
from __future__ import annotations

import numpy as np
import pytest

from analysis import dls as E
from analysis.dls import _apply_tau_window
from fixtures.synthetic_dls import bimodal, gamma_for_rh, make_measurement, monomodal


# ==================================================== [A] engine correctness / no-op

def test_skip_zero_byte_identical_to_default():
    # skip=0 must be byte-identical to the default (no-arg) path across cumulant
    # orders and the distribution methods.
    maxd = 0.0
    for m in (monomodal(10.0), monomodal(30.0), monomodal(80.0)):
        for o in (1, 2, 3):
            a = E.fit_cumulants(m, order=o)
            b = E.fit_cumulants(m, order=o, skip_initial_channels=0)
            maxd = max(maxd, abs(a.gamma_s_inv - b.gamma_s_inv), abs(a.beta - b.beta),
                       float(np.max(np.abs(a.coefficients - b.coefficients))))
            assert a.n_points_used == b.n_points_used
        for fn in (E.fit_nnls, E.fit_lognormal):
            da = fn(m, rh_min_nm=2.0, rh_max_nm=2000.0, n_grid=120)
            db = fn(m, rh_min_nm=2.0, rh_max_nm=2000.0, n_grid=120, skip_initial_channels=0)
            maxd = max(maxd, float(np.max(np.abs(da.weights - db.weights))))
    assert maxd == 0.0, f"max delta {maxd:.3e}"


def test_skip_provenance_and_exact_slice():
    m = monomodal(30.0)
    tau = m.delay_times_s
    r = E.fit_cumulants(m, order=2, skip_initial_channels=7)
    assert r.n_skipped == 7
    twin, _ = _apply_tau_window(tau, m.correlogram, None, None,
                                min_points=3, skip_initial_channels=7)
    assert np.allclose(twin, tau[7:])
    d = E.fit_nnls(m, skip_initial_channels=4)
    assert d.n_skipped == 4


# ==================================================== [B] tau_min interaction matrix

def _start(tau, g, skip=0, tmin=None, tmax=None):
    t, _ = _apply_tau_window(tau, g, tmin, tmax, min_points=3, skip_initial_channels=skip)
    return t[0]


def test_tau_min_skip_interaction_matrix():
    m = monomodal(30.0)
    tau = m.delay_times_s
    g = m.correlogram
    N = 10
    assert _start(tau, g, skip=N) == tau[N]                        # B4 skip only
    assert _start(tau, g, tmin=tau[6]) == tau[6]                   # B5 tau_min only
    assert _start(tau, g, skip=N, tmin=tau[3]) == tau[N]           # B6 skip dominates
    assert _start(tau, g, skip=N, tmin=tau[20]) == tau[20]         # B7 tau_min dominates
    assert _start(tau, g, skip=N, tmin=tau[N]) == tau[N]           # B8 no double-drop


def test_effective_start_is_max_of_skip_and_searchsorted():
    m = monomodal(30.0)
    tau = m.delay_times_s
    g = m.correlogram
    rng = np.random.RandomState(0)
    for _ in range(40):
        skip = int(rng.randint(0, 50))
        tmin = float(tau[int(rng.randint(0, 150))])
        eff_idx = max(skip, int(np.searchsorted(tau, tmin, side="left")))
        if eff_idx >= tau.size - 4:   # keep enough points
            continue
        t, _ = _apply_tau_window(tau, g, tmin, None, min_points=3, skip_initial_channels=skip)
        assert t[0] == tau[eff_idx]


def test_tau_max_unaffected_by_skip():
    m = monomodal(30.0)
    tau = m.delay_times_s
    g = m.correlogram
    t, _ = _apply_tau_window(tau, g, None, tau[150], min_points=3, skip_initial_channels=5)
    assert t[0] == tau[5] and t[-1] == tau[150]


def test_skip_equals_tau_min_at_same_channel():
    m = monomodal(30.0)
    tau = m.delay_times_s
    rs = E.fit_cumulants(m, order=2, skip_initial_channels=8)
    rt = E.fit_cumulants(m, order=2, tau_min_s=tau[8])
    assert abs(rs.gamma_s_inv - rt.gamma_s_inv) < 1e-6


# ==================================================== [C] boundary / error handling

def test_window_min_points_boundary():
    m = monomodal(30.0)
    tau = m.delay_times_s
    g = m.correlogram
    n = tau.size
    twin, _ = _apply_tau_window(tau, g, None, None, min_points=3, skip_initial_channels=n - 3)
    assert twin.size == 3


def test_moderate_skip_still_fits():
    m = monomodal(30.0)
    r = E.fit_cumulants(m, order=2, skip_initial_channels=20)   # decay intact
    assert np.isfinite(r.rh_nm) and abs(r.rh_nm - 30) < 4


def test_window_under_min_points_raises_mentioning_skip():
    m = monomodal(30.0)
    tau = m.delay_times_s
    g = m.correlogram
    n = tau.size
    with pytest.raises(ValueError, match="skip"):
        _apply_tau_window(tau, g, None, None, min_points=3, skip_initial_channels=n - 1)


def test_skip_plus_tau_min_jointly_too_few_raises():
    m = monomodal(30.0)
    tau = m.delay_times_s
    g = m.correlogram
    n = tau.size
    with pytest.raises(ValueError):
        _apply_tau_window(tau, g, tau[n - 2], None, min_points=3, skip_initial_channels=n - 5)


def test_negative_skip_raises():
    with pytest.raises(ValueError):
        E.fit_cumulants(monomodal(30.0), skip_initial_channels=-1)


def test_non_integer_skip_raises():
    with pytest.raises(ValueError):
        E.fit_cumulants(monomodal(30.0), skip_initial_channels=2.5)


def test_skip_ge_length_raises():
    m = monomodal(30.0)
    n = m.delay_times_s.size
    with pytest.raises(ValueError):
        E.fit_cumulants(m, skip_initial_channels=n + 5)


def test_skip_removes_afterpulse_spike():
    m = monomodal(30.0)
    tau = m.delay_times_s
    g = m.correlogram.copy()
    g[0] = 0.9 * 2.0   # afterpulsing spike well above the true intercept
    m_spike = make_measurement(g, tau)
    rh0 = E.fit_cumulants(m_spike, order=2, skip_initial_channels=0).rh_nm
    rh1 = E.fit_cumulants(m_spike, order=2, skip_initial_channels=1).rh_nm
    assert abs(rh1 - 30.0) < 4.0 and abs(rh0 - 30.0) > abs(rh1 - 30.0), (
        f"rh(skip0)={rh0:.1f} rh(skip1)={rh1:.1f}")


# ==================================================== [D] distribution methods

@pytest.mark.parametrize("name,fn", [("nnls", E.fit_nnls), ("lognormal", E.fit_lognormal)])
def test_distribution_skip_zero_sane_peak(name, fn):
    m = monomodal(40.0)
    d0 = fn(m, rh_min_nm=2.0, rh_max_nm=2000.0, n_grid=120, skip_initial_channels=0)
    assert 20 < d0.peak_rh_nm < 80 and d0.n_skipped == 0


def test_contin_windowed_start_index():
    m = monomodal(40.0)
    tau = m.delay_times_s
    d = E.fit_contin(m, rh_min_nm=2.0, rh_max_nm=2000.0, n_grid=120, skip_initial_channels=6)
    assert d.distribution.fit_tau_s[0] == tau[6] and d.distribution.n_skipped == 6


def test_beta_estimation_improves_with_skip_on_spike():
    m = monomodal(40.0)
    tau = m.delay_times_s
    g = m.correlogram.copy()
    g[0] = 0.9 * 2.0
    msp = make_measurement(g, tau)
    b0 = E.fit_nnls(msp, rh_min_nm=2.0, rh_max_nm=2000.0, n_grid=120,
                    skip_initial_channels=0).beta
    b1 = E.fit_nnls(msp, rh_min_nm=2.0, rh_max_nm=2000.0, n_grid=120,
                    skip_initial_channels=1).beta
    assert abs(b1 - 0.9) < abs(b0 - 0.9), f"beta(skip0)={b0:.3f} beta(skip1)={b1:.3f}"


def test_cumulant_and_contin_share_windowed_start():
    m = monomodal(40.0)
    cum = E.fit_cumulants(m, order=2, skip_initial_channels=5)
    con = E.fit_contin(m, rh_min_nm=2.0, rh_max_nm=2000.0, n_grid=120, skip_initial_channels=5)
    assert cum.fit_tau_s[0] == con.distribution.fit_tau_s[0]


def test_over_skip_past_fast_decay_erases_small_rh_weight():
    mb = bimodal(3.0, 100.0, f_small=0.5)
    taub = mb.delay_times_s
    # skip past ~8 e-foldings of the small (fast) population so its signal is gone.
    g_small = gamma_for_rh(3.0)
    over = int(np.searchsorted(taub, 8.0 / (2 * g_small)))   # first lag where fast mode ~dead

    def weight_below(d, rh_cut=12.0):
        return float(d.weights[d.rh_grid_nm < rh_cut].sum())

    d_small = E.fit_nnls(mb, rh_min_nm=2.0, rh_max_nm=400.0, n_grid=140, skip_initial_channels=2)
    d_over = E.fit_nnls(mb, rh_min_nm=2.0, rh_max_nm=400.0, n_grid=140, skip_initial_channels=over)
    assert (weight_below(d_small) > weight_below(d_over)
            and d_over.mean_rh_nm > d_small.mean_rh_nm), (
        f"skip2: w<12nm={weight_below(d_small):.3f} mean={d_small.mean_rh_nm:.1f} | "
        f"skip{over}: w<12nm={weight_below(d_over):.3f} mean={d_over.mean_rh_nm:.1f}")


# ==================================================== [E] real-data regression (SMALS)

@pytest.mark.realdata
@pytest.mark.parametrize("ang,hi_expect,tol", [(117.0, 22.0, 3.0), (99.0, 28.0, 3.0), (147.0, 26.0, 3.0)])
def test_smals_bad_angles_recovered_by_skip(smals, ang, hi_expect, tol):
    if not smals:
        pytest.skip("SMALS data absent")
    m = smals[ang]
    r0 = E.fit_cumulants(m, order=2, skip_initial_channels=0)
    r9 = E.fit_cumulants(m, order=2, skip_initial_channels=9)
    assert abs(r9.rh_nm - hi_expect) < tol and r9.rh_nm > r0.rh_nm + 2, (
        f"skip0={r0.rh_nm:.1f} skip9={r9.rh_nm:.1f}")


@pytest.mark.realdata
@pytest.mark.parametrize("ang,tol", [(50.0, 1.0), (64.0, 1.0), (81.0, 1.0), (134.0, 2.5)])
def test_smals_good_angles_stable_under_skip(smals, ang, tol):
    # Clean angles barely move; 134 is borderline (PDI 0.30) and shifts slightly
    # toward the correct value, so it gets a looser bound.
    if not smals:
        pytest.skip("SMALS data absent")
    r0 = E.fit_cumulants(smals[ang], order=2, skip_initial_channels=0).rh_nm
    r9 = E.fit_cumulants(smals[ang], order=2, skip_initial_channels=9).rh_nm
    assert abs(r9 - r0) < tol, f"skip0={r0:.1f} skip9={r9:.1f}"
