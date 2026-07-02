"""Regression tests for physics/constants.py.

This module had NO test coverage before, yet everything downstream depends on
it (q, Stokes-Einstein, the geometry-aware toluene Rayleigh ratio, the optical
constant K, the depolarization/Cabannes helpers, and the rod/sphere shape
models). Every value here is analytically exact, so tolerances are tight.

Key physics enforced (CLAUDE.md invariants):
  - q = (4 pi n / lambda) sin(theta/2), n and lambda explicit
  - Rh = kT / (6 pi eta D), T and eta always required
  - toluene R_VV(532 nm, 25 C) = 2.34e-5 cm^-1; VU = VV(1+rho_v), VH = VV rho_v;
    temperature coefficient POSITIVE (~+0.43 %/C); geometry explicit
  - (n_solvent / n_standard)^2 refractive-index-mismatch correction
  - Cabannes isotropic factor 1 - (4/3) rho_v
"""
from __future__ import annotations

import math

import pytest

from physics import constants as C


# --------------------------------------------------------------- constants ---

def test_fundamental_constants_exact():
    assert C.BOLTZMANN_K == 1.380649e-23          # SI 2019 exact
    assert C.AVOGADRO_NA == 6.02214076e23         # SI 2019 exact


# ----------------------------------------------------- scattering vector q ---

def test_scattering_vector_q_value():
    # 4*pi*1.33/532 = pi/100 exactly, times sin(45 deg).
    q = C.scattering_vector_q(90.0, 532.0, 1.33)
    assert q == pytest.approx(0.0222144147, rel=1e-6)   # nm^-1


def test_scattering_vector_q_matches_closed_form():
    for angle, lam, n in [(30.0, 633.0, 1.33), (90.0, 532.0, 1.50), (150.0, 488.0, 1.40)]:
        expected = (4 * math.pi * n / lam) * math.sin(math.radians(angle) / 2)
        assert C.scattering_vector_q(angle, lam, n) == pytest.approx(expected, rel=1e-12)


def test_scattering_vector_q_m_is_q_times_1e9():
    q = C.scattering_vector_q(75.0, 532.0, 1.33)
    q_m = C.scattering_vector_q_m(75.0, 532.0, 1.33)
    assert q_m == pytest.approx(q * 1.0e9, rel=1e-12)


def test_scattering_vector_q_monotonic_in_angle():
    qs = [C.scattering_vector_q(a, 532.0, 1.33) for a in range(30, 160, 10)]
    assert all(b > a for a, b in zip(qs, qs[1:], strict=False))  # adjacent pairs: intentionally ragged


# ------------------------------------------------------- Stokes-Einstein ---

def test_stokes_einstein_round_trip():
    # Rh = 9.5 nm -> D -> Rh, exact to machine precision.
    rh_m = 9.5e-9
    T, eta = 298.15, 8.9e-4
    D = C.stokes_einstein_diffusion_coefficient(rh_m, T, eta)
    rh_back = C.stokes_einstein_rh(D, T, eta)
    assert rh_back == pytest.approx(rh_m, rel=1e-12)


def test_stokes_einstein_known_value():
    # D for Rh = 9.5 nm in water at 25 C: kT/(6 pi eta Rh).
    T, eta, rh_m = 298.15, 8.9e-4, 9.5e-9
    expected = C.BOLTZMANN_K * T / (6 * math.pi * eta * rh_m)
    assert C.stokes_einstein_diffusion_coefficient(rh_m, T, eta) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("bad", [
    dict(diffusion_coefficient_m2_per_s=-1e-12, temperature_K=298.0, viscosity_Pa_s=8.9e-4),
    dict(diffusion_coefficient_m2_per_s=1e-12, temperature_K=0.0, viscosity_Pa_s=8.9e-4),
    dict(diffusion_coefficient_m2_per_s=1e-12, temperature_K=298.0, viscosity_Pa_s=-1.0),
])
def test_stokes_einstein_rejects_nonpositive(bad):
    with pytest.raises(ValueError):
        C.stokes_einstein_rh(**bad)


def test_stokes_einstein_D_rejects_nonpositive():
    with pytest.raises(ValueError):
        C.stokes_einstein_diffusion_coefficient(-1e-9, 298.0, 8.9e-4)


# ------------------------------------------------- toluene Rayleigh ratio ---

def test_toluene_rvv_reference_values():
    # Authoritative polarised values at 25 C (dT = 0).
    assert C.rayleigh_ratio_toluene(532.0, 25.0, "VV") == pytest.approx(2.34e-5, rel=1e-12)
    assert C.rayleigh_ratio_toluene(660.0, 25.0, "VV") == pytest.approx(8.456e-6, rel=1e-12)


def test_toluene_geometry_conversions_532():
    # VU = VV (1 + rho_v), VH = VV rho_v, with rho_v(25 C) = 0.346.
    vv = C.rayleigh_ratio_toluene(532.0, 25.0, "VV")
    assert C.rayleigh_ratio_toluene(532.0, 25.0, "VU") == pytest.approx(vv * 1.346, rel=1e-9)
    assert C.rayleigh_ratio_toluene(532.0, 25.0, "VH") == pytest.approx(vv * 0.346, rel=1e-9)


def test_toluene_temperature_coefficient_positive():
    # +0.43 %/C: at 35 C, VV = VV(25) * (1 + 0.0043*10).
    vv25 = C.rayleigh_ratio_toluene(532.0, 25.0, "VV")
    vv35 = C.rayleigh_ratio_toluene(532.0, 35.0, "VV")
    assert vv35 == pytest.approx(vv25 * 1.043, rel=1e-9)
    assert vv35 > vv25          # sign must be positive


def test_toluene_user_supplied_depolarization_ratio():
    vv = C.rayleigh_ratio_toluene(532.0, 25.0, "VV")
    vu = C.rayleigh_ratio_toluene(532.0, 25.0, "VU", depolarization_ratio_v=0.30)
    assert vu == pytest.approx(vv * 1.30, rel=1e-12)


def test_toluene_wavelength_within_tolerance_ok():
    # 531 nm is within the +/-2 nm match window of 532 nm.
    assert C.rayleigh_ratio_toluene(531.0, 25.0, "VV") == pytest.approx(2.34e-5, rel=1e-12)


def test_toluene_unsupported_wavelength_raises():
    with pytest.raises(ValueError):
        C.rayleigh_ratio_toluene(633.0, 25.0, "VV")


def test_toluene_bad_geometry_raises():
    with pytest.raises(ValueError):
        C.rayleigh_ratio_toluene(532.0, 25.0, "HH")


# ------------------------------------- refractive-index mismatch correction ---

def test_refractive_index_correction_value():
    assert C.refractive_index_correction(1.33, 1.496) == pytest.approx((1.33 / 1.496) ** 2, rel=1e-12)
    assert C.refractive_index_correction(1.33, 1.496) == pytest.approx(0.79038756, rel=1e-6)


def test_refractive_index_correction_identity_is_one():
    assert C.refractive_index_correction(1.496, 1.496) == 1.0


@pytest.mark.parametrize("ns,nstd", [(-1.0, 1.496), (1.33, 0.0)])
def test_refractive_index_correction_rejects_nonpositive(ns, nstd):
    with pytest.raises(ValueError):
        C.refractive_index_correction(ns, nstd)


# ---------------------------------------------------------- optical constant K ---

def test_optical_constant_K_value():
    K = C.optical_constant_K(1.33, 0.135, 532.0)
    lam_cm = 532.0e-7
    expected = 4.0 * math.pi ** 2 * 1.33 ** 2 * 0.135 ** 2 / (C.AVOGADRO_NA * lam_cm ** 4)
    assert K == pytest.approx(expected, rel=1e-12)


def test_optical_constant_K_scaling():
    base = C.optical_constant_K(1.33, 0.135, 532.0)
    # doubling dn/dc quadruples K
    assert C.optical_constant_K(1.33, 0.270, 532.0) == pytest.approx(4 * base, rel=1e-12)
    # halving wavelength multiplies K by 16 (lambda^-4)
    assert C.optical_constant_K(1.33, 0.135, 266.0) == pytest.approx(16 * base, rel=1e-9)


# ---------------------------------------------------- depolarization ratios ---

def test_depolarization_ratio_unpolarized_value():
    assert C.depolarization_ratio_unpolarized(0.346) == pytest.approx(2 * 0.346 / 1.346, rel=1e-12)
    assert C.depolarization_ratio_unpolarized(0.346) == pytest.approx(0.51411590, rel=1e-6)


def test_depolarization_ratio_round_trip():
    for rho_v in (0.0, 0.1, 0.346, 0.5, 0.74):
        rho_u = C.depolarization_ratio_unpolarized(rho_v)
        assert C.depolarization_ratio_vertical(rho_u) == pytest.approx(rho_v, rel=1e-12, abs=1e-15)


def test_depolarization_sivokhin_table():
    # S&K 2021 Table 1 toluene: rho_v 0.364/0.346/0.310 -> rho_u 0.534/0.514/0.473.
    for rho_v, rho_u in [(0.364, 0.534), (0.346, 0.514), (0.310, 0.473)]:
        assert C.depolarization_ratio_unpolarized(rho_v) == pytest.approx(rho_u, abs=1e-3)


@pytest.mark.parametrize("rho_v", [-0.01, 0.76])
def test_depolarization_ratio_v_out_of_range_raises(rho_v):
    with pytest.raises(ValueError):
        C.depolarization_ratio_unpolarized(rho_v)


def test_depolarization_ratio_u_out_of_range_raises():
    with pytest.raises(ValueError):
        C.depolarization_ratio_vertical(0.90)   # > 6/7


# --------------------------------------------------------- Cabannes factor ---

def test_cabannes_vertical_value():
    assert C.cabannes_isotropic_factor(0.346) == pytest.approx(1 - (4 / 3) * 0.346, rel=1e-12)
    assert C.cabannes_isotropic_factor(0.0) == 1.0
    assert C.cabannes_isotropic_factor(0.75) == pytest.approx(0.0, abs=1e-12)


def test_cabannes_natural_value():
    # Coumou Table 3: benzene rho_u = 0.42 -> total/isotropic = 1/f = 2.78.
    f = C.cabannes_isotropic_factor_natural(0.42)
    assert 1.0 / f == pytest.approx(2.78, abs=0.01)


def test_cabannes_out_of_range_raises():
    with pytest.raises(ValueError):
        C.cabannes_isotropic_factor(0.80)


# ------------------------------------------------- optical anisotropy delta^2 ---

def test_optical_anisotropy_squared_value():
    assert C.optical_anisotropy_squared(0.346) == pytest.approx(5 * 0.346 / (3 - 4 * 0.346), rel=1e-12)
    assert C.optical_anisotropy_squared(0.0) == 0.0


def test_optical_anisotropy_squared_diverges_at_limit():
    with pytest.raises(ValueError):
        C.optical_anisotropy_squared(0.75)


# ------------------------------------------------------- rod / sphere models ---

def test_rod_end_corrections_p10():
    nu, delta_perp = C.rod_end_corrections(10.0)
    assert nu == pytest.approx(0.3675, rel=1e-12)
    assert delta_perp == pytest.approx(-0.5708, rel=1e-12)


def test_rod_translational_diffusion_value():
    # L=60 nm, d=6 nm, water 25 C: D_t = 2.183890e-11 m^2/s (Tirado 1984).
    D_t = C.rod_translational_diffusion(60e-9, 6e-9, 298.15, 8.9e-4)
    assert D_t == pytest.approx(2.183890e-11, rel=1e-4)


def test_rod_rotational_diffusion_value():
    D_r = C.rod_rotational_diffusion(60e-9, 6e-9, 298.15, 8.9e-4)
    assert D_r == pytest.approx(3.541112e4, rel=1e-4)


def test_rod_length_inverse_recovers_60nm():
    D_t = C.rod_translational_diffusion(60e-9, 6e-9, 298.15, 8.9e-4)
    L = C.rod_length_from_translational_diffusion(D_t, 10.0, 298.15, 8.9e-4)
    assert L == pytest.approx(60e-9, rel=1e-9)


@pytest.mark.parametrize("p,valid", [(1.5, False), (10.0, True), (40.0, False)])
def test_rod_aspect_ratio_valid_bounds(p, valid):
    assert C.rod_aspect_ratio_valid(p) is valid


def test_sphere_rotational_diffusion_round_trip():
    R = 11e-9
    D_r = C.sphere_rotational_diffusion(R, 298.15, 8.9e-4)
    R_back = C.sphere_radius_from_rotational_diffusion(D_r, 298.15, 8.9e-4)
    assert R_back == pytest.approx(R, rel=1e-12)


@pytest.mark.parametrize("fn,args", [
    (lambda: C.rod_translational_diffusion(-1e-9, 6e-9, 298.15, 8.9e-4), None),
    (lambda: C.rod_rotational_diffusion(60e-9, -6e-9, 298.15, 8.9e-4), None),
    (lambda: C.sphere_rotational_diffusion(-1e-9, 298.15, 8.9e-4), None),
    (lambda: C.rod_end_corrections(-1.0), None),
])
def test_shape_models_reject_nonpositive(fn, args):
    with pytest.raises(ValueError):
        fn()
