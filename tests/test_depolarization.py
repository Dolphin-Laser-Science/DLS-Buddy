"""Regression tests for ``analysis/depolarization.py``.

Static DPLS (depolarization ratio, Cabannes split, optical anisotropy) and
dynamic DDLS (rotational diffusion D_r, translational D_t, rod/sphere shape
inverses). The dynamic tests fit the committed synthetic rod correlograms
(``test-data/Synthetic DPLS/``, DPLS Phase 2 ground truth) and recover D_t/D_r/Rh
to within the generator's stated tolerances.

Ground truth (parameters.txt): rigid rod L=60 nm, d=6 nm, p=10, water 25 C,
D_t = 2.183890e-11 m^2/s, D_r = 3.541112e4 rad^2/s, Rh_t = 11.2356 nm.
"""
from __future__ import annotations

import math

import pytest

from analysis import depolarization as dp
from analysis import dls as E
from fixtures.data_paths import SYNTH_DPLS_DIR
from fixtures.synthetic_dls import load_alv, nearest_angle


# Ground truth from test-data/Synthetic DPLS/parameters.txt.
_D_T = 2.183890e-11        # m^2/s
_D_R = 3.541112e4          # rad^2/s
_RH_T_NM = 11.2356         # nm
_WATER_ETA = 0.89e-3       # Pa s (25 C)
_T = 298.15                # K


# ===========================================================================
# Static depolarization
# ===========================================================================

def test_depolarization_ratio_with_dark_count():
    # rho_v = (I_VH - dark)/(I_VV - dark).
    rho_v = dp.depolarization_ratio(i_vv=1000.0, i_vh=50.0, dark_count=10.0)
    assert rho_v == pytest.approx((50.0 - 10.0) / (1000.0 - 10.0), rel=1e-12)


def test_depolarization_ratio_rejects_nonpositive_vv():
    with pytest.raises(ValueError):
        dp.depolarization_ratio(i_vv=10.0, i_vh=5.0, dark_count=10.0)  # VV-dark = 0


def test_analyze_depolarization_core_values():
    # rho_v = 0.346 -> known Cabannes / anisotropy values (module docstring math).
    res = dp.analyze_depolarization(i_vv=1000.0, i_vh=346.0)
    assert res.rho_v == pytest.approx(0.346, rel=1e-12)
    assert res.rho_u == pytest.approx(0.51412, abs=1e-4)
    assert res.optical_anisotropy_sq == pytest.approx(1.07054, abs=1e-4)
    assert res.cabannes_isotropic_factor == pytest.approx(0.53867, abs=1e-4)
    assert res.anisotropic_fraction == pytest.approx(1.0 - 0.53867, abs=1e-4)
    assert res.physically_valid is True
    assert res.rho_v_se is None                    # a single pair -> no SE (invariant 8)


def test_analyze_depolarization_out_of_range_flags_no_raise():
    # rho_v = 0.8 > 3/4 -> physically_valid False, delta^2 NaN, no exception.
    res = dp.analyze_depolarization(i_vv=1000.0, i_vh=800.0)
    assert res.rho_v == pytest.approx(0.8, rel=1e-12)
    assert res.physically_valid is False
    assert math.isnan(res.optical_anisotropy_sq)
    assert res.note != ''


def test_analyze_depolarization_replicate_se():
    # Both intensity SEs supplied -> a ratio SE is propagated.
    res = dp.analyze_depolarization(i_vv=1000.0, i_vh=346.0,
                                    i_vv_se=10.0, i_vh_se=5.0)
    assert res.rho_v_se is not None and res.rho_v_se > 0


def test_isotropic_rayleigh_ratio_cabannes():
    rho_v = 0.346
    r_vv = 2.5e-5
    r_iso = dp.isotropic_rayleigh_ratio(r_vv, rho_v)
    assert r_iso == pytest.approx(r_vv * (1.0 - (4.0 / 3.0) * rho_v), rel=1e-12)


# ===========================================================================
# Dynamic DDLS -- pure rate combination
# ===========================================================================

def test_rotational_diffusion_from_rates():
    # D_r = (Gamma_VH - Gamma_VV)/6.
    assert dp.rotational_diffusion_from_rates(1000.0, 1600.0) == pytest.approx(100.0)


def _rate_point(gamma_vv, gamma_vh, angle=90.0, q=2.0e7):
    return dp.DDLSRatePoint(angle_deg=angle, q_m_inv=q,
                            gamma_vv_s_inv=gamma_vv, gamma_vh_s_inv=gamma_vh)


def test_analyze_ddls_single_angle_no_se():
    # One VV/VH pair (single angle) -> no SE reported (invariant 8).
    res = dp.analyze_ddls([_rate_point(1000.0, 1000.0 + 6.0 * _D_R)],
                          temperature_K=_T, viscosity_Pa_s=_WATER_ETA)
    assert res.method == 'single-angle'
    assert res.d_t_se is None
    assert res.d_r_se is None
    assert res.n_angles == 1


# ===========================================================================
# Dynamic DDLS -- real synthetic rod correlograms
# ===========================================================================

def _ddls_points_from_files():
    vv = load_alv(str(SYNTH_DPLS_DIR / 'DDLS VV.ASC'))
    vh = load_alv(str(SYNTH_DPLS_DIR / 'DDLS VH.ASC'))
    if vv is None or vh is None:
        pytest.skip('Synthetic DPLS test files are not present.')
    points = []
    for mvv in vv:
        mvh = nearest_angle(vh, mvv.angle_deg)
        cvv = E.fit_cumulants(mvv, order=2)
        cvh = E.fit_cumulants(mvh, order=2)
        points.append(dp.DDLSRatePoint(
            angle_deg=mvv.angle_deg, q_m_inv=cvv.q_m_inv,
            gamma_vv_s_inv=cvv.gamma_s_inv, gamma_vh_s_inv=cvh.gamma_s_inv))
    return points


@pytest.mark.realdata
def test_analyze_ddls_recovers_rod_ground_truth():
    points = _ddls_points_from_files()
    assert len(points) == 7
    res = dp.analyze_ddls(points, temperature_K=_T, viscosity_Pa_s=_WATER_ETA,
                          rod_length_nm=60.0)
    assert res.method == 'multi-angle'
    # D_t within 5%, D_r within 2% (parameters.txt self-validation targets).
    assert abs(res.d_t_m2_s - _D_T) / _D_T < 0.05
    assert abs(res.d_r_rad2_s - _D_R) / _D_R < 0.02
    assert res.rh_t_nm == pytest.approx(_RH_T_NM, abs=0.1)
    # All qL < 3 in this set -> single-exponential regime is valid.
    assert res.single_exponential_valid is True
    # Multi-angle ensemble -> honest SEs are reported.
    assert res.d_t_se is not None
    assert res.d_r_se is not None


# ===========================================================================
# Shape-model inverses (Phase 3)
# ===========================================================================

def test_rod_dimensions_from_diffusion_recovers_rod():
    res = dp.rod_dimensions_from_diffusion(
        _D_T, _D_R, temperature_K=_T, viscosity_Pa_s=_WATER_ETA)
    assert res.converged is True
    assert res.in_valid_range is True
    assert res.length_nm == pytest.approx(60.0, abs=0.5)
    assert res.diameter_nm == pytest.approx(6.0, abs=0.1)
    assert res.aspect_ratio_p == pytest.approx(10.0, abs=0.1)
    # No SE inputs -> no dimension SEs.
    assert res.length_se is None
    assert res.diameter_se is None


def test_sphere_inverse_on_rod_flags_inconsistent():
    # A rod fed to the sphere model: R(D_r) >> R(D_t) -> sphericity fails.
    res = dp.sphere_dimensions_from_diffusion(
        _D_T, _D_R, temperature_K=_T, viscosity_Pa_s=_WATER_ETA)
    assert res.is_consistent is False
    assert res.sphericity_ratio > 1.2       # rod signature (ratio > 1)


@pytest.mark.slow
def test_rod_dimensions_monte_carlo_se():
    # With SE inputs the dimension SEs are propagated by Monte Carlo (slow).
    res = dp.rod_dimensions_from_diffusion(
        _D_T, _D_R, temperature_K=_T, viscosity_Pa_s=_WATER_ETA,
        d_t_se=0.02 * _D_T, d_r_se=0.02 * _D_R, n_mc=400, seed=12345)
    assert res.converged is True
    assert res.length_se is not None and res.length_se > 0
    assert res.diameter_se is not None and res.diameter_se > 0
    assert res.aspect_ratio_se is not None and res.aspect_ratio_se > 0
