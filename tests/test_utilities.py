"""Regression tests for ``analysis/utilities.py``.

Covers the cross-analysis rho = Rg/Rh, the Rg/A2-Mw scaling power-law fit, the
provenance-aware result-candidate picker, the pure synthetic-correlogram
generator (ground-truth cumulant fields + validation guards), and the SLS
I*sin(theta) optical-quality diagnostic.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from analysis import utilities as util
from analysis.synthetic_dataset import DEFAULT_SERIES
from core.data_models import SLSMeasurement, SampleKey
from physics.constants import stokes_einstein_rh


# ---------------------------------------------------------------------------
# rho = Rg / Rh
# ---------------------------------------------------------------------------

def test_compute_rho_value_and_shape():
    res = util.compute_rho(rg_nm=55.0, rh_nm=37.0)
    assert res.rho == pytest.approx(55.0 / 37.0, rel=1e-12)   # ~1.486
    assert res.rho == pytest.approx(1.486, abs=0.005)
    assert 'coil' in res.shape.lower()                        # random-coil range
    assert res.keys_matched is None
    assert res.rho_se is None                                 # no input SEs


def test_compute_rho_propagates_se():
    res = util.compute_rho(rg_nm=55.0, rh_nm=37.0, rg_se=1.0, rh_se=0.5)
    assert res.rho_se is not None and res.rho_se > 0


def test_compute_rho_rejects_nonpositive():
    with pytest.raises(ValueError):
        util.compute_rho(rg_nm=0.0, rh_nm=37.0)
    with pytest.raises(ValueError):
        util.compute_rho(rg_nm=55.0, rh_nm=-1.0)


def _key(polymer='PVP', solvent='water', temp=298.15, conc=None):
    return SampleKey(polymer_name=polymer, solvent_name=solvent,
                     concentration_g_per_mL=conc, temperature_K=temp)


def test_compute_rho_keys_match():
    res = util.compute_rho(55.0, 37.0,
                           rg_sample_key=_key(conc=0.001),
                           rh_sample_key=_key(conc=None))
    assert res.keys_matched is True     # concentration is ignored


def test_compute_rho_key_mismatch_warns():
    with pytest.warns(UserWarning):
        res = util.compute_rho(55.0, 37.0,
                               rg_sample_key=_key(polymer='PVP'),
                               rh_sample_key=_key(polymer='P2VP'))
    assert res.keys_matched is False


def test_compute_rho_key_mismatch_raises_when_required():
    with pytest.raises(ValueError):
        util.compute_rho(55.0, 37.0,
                         rg_sample_key=_key(solvent='water'),
                         rh_sample_key=_key(solvent='toluene'),
                         require_match=True)


# ---------------------------------------------------------------------------
# scaling power-law fit
# ---------------------------------------------------------------------------

def test_fit_power_law_rg_vs_mw():
    mw = [d['mw'] for d in DEFAULT_SERIES.values()]
    rg = [d['rg'] for d in DEFAULT_SERIES.values()]
    res = util.fit_power_law(mw, rg)
    assert res.fit_valid is True
    assert res.n_points == 4
    assert res.exponent == pytest.approx(0.585, abs=0.03)     # good-solvent nu
    assert res.r_squared > 0.99
    assert res.exponent_se is not None


def test_fit_power_law_a2_vs_mw_negative():
    mw = [d['mw'] for d in DEFAULT_SERIES.values()]
    a2 = [d['a2'] for d in DEFAULT_SERIES.values()]
    res = util.fit_power_law(mw, a2)
    assert res.fit_valid is True
    assert res.exponent < 0.0                                  # A2 declines with Mw


def test_fit_power_law_too_few_points():
    res = util.fit_power_law([1.0e6], [55.0])
    assert res.fit_valid is False
    assert math.isnan(res.exponent)
    assert res.n_points == 1


def test_fit_power_law_ignores_nonpositive():
    # Non-finite / non-positive pairs are dropped before the log-log fit.
    res = util.fit_power_law([1e5, 3e5, -1.0, 1e6, 3e6],
                             [14.0, 27.0, 5.0, 55.0, 105.0])
    assert res.n_points == 4
    assert res.exponent == pytest.approx(0.585, abs=0.03)


# ---------------------------------------------------------------------------
# provenance-aware candidate picker
# ---------------------------------------------------------------------------

def _cand(value, tier=0, quality=None, label='c', kind='k'):
    return util.ResultCandidate(value=value, label=label, kind=kind,
                                is_apparent=False, tier=tier, quality=quality)


def test_select_default_candidate_tier_wins():
    lo = _cand(10.0, tier=0, quality=0.99)
    hi = _cand(11.0, tier=2, quality=0.10)
    assert util.select_default_candidate([lo, hi]) is hi


def test_select_default_candidate_quality_within_tier():
    a = _cand(10.0, tier=1, quality=0.90)
    b = _cand(11.0, tier=1, quality=0.99)
    assert util.select_default_candidate([a, b]) is b


def test_select_default_candidate_none_quality_sorts_below():
    good = _cand(10.0, tier=1, quality=0.5)
    unscored = _cand(11.0, tier=1, quality=None)
    assert util.select_default_candidate([unscored, good]) is good


def test_select_default_candidate_stable_tiebreak_first():
    first = _cand(10.0, tier=1, quality=0.9)
    second = _cand(11.0, tier=1, quality=0.9)
    assert util.select_default_candidate([first, second]) is first


def test_select_default_candidate_ignores_nonfinite():
    bad = _cand(float('nan'), tier=5, quality=1.0)
    ok = _cand(10.0, tier=0, quality=0.1)
    assert util.select_default_candidate([bad, ok]) is ok


def test_select_default_candidate_empty_is_none():
    assert util.select_default_candidate([]) is None
    assert util.select_default_candidate([_cand(float('inf'))]) is None


# ---------------------------------------------------------------------------
# synthetic correlogram generator
# ---------------------------------------------------------------------------

_GEO = dict(angle_deg=90.0, wavelength_nm=633.0, solvent_refractive_index=1.33,
            temperature_K=298.15, viscosity_Pa_s=0.89e-3)


def test_generate_correlogram_ground_truth_monodisperse():
    pop = util.SyntheticPopulation(rh_nm=30.0, weight=1.0)
    res = util.generate_synthetic_correlogram([pop], **_GEO, beta=0.8)
    # Monodisperse -> PDI ~ 0 and rh_eff recovers the input Rh exactly.
    assert res.pdi == pytest.approx(0.0, abs=1e-9)
    assert res.rh_eff_nm == pytest.approx(30.0, rel=1e-6)
    # gamma_bar = D q^2 with D from Stokes-Einstein.
    d_expected = res.gamma_bar_s_inv / res.q_m_inv ** 2
    rh_check = stokes_einstein_rh(d_expected, _GEO['temperature_K'],
                                  _GEO['viscosity_Pa_s']) * 1e9
    assert rh_check == pytest.approx(30.0, rel=1e-6)


def test_generate_correlogram_g1_normalized_at_zero():
    # g1(0) = 1 by construction; g2m1(0) = beta.
    pop = util.SyntheticPopulation(rh_nm=25.0, weight=1.0)
    beta = 0.7
    g1 = util.generate_synthetic_correlogram([pop], **_GEO, beta=beta,
                                             output_form='g1', delay_min_s=1e-9)
    assert g1.signal[0] == pytest.approx(1.0, rel=1e-3)


def test_generate_correlogram_output_form_relationships():
    pop = util.SyntheticPopulation(rh_nm=40.0, weight=1.0)
    common = dict(beta=0.75, delay_min_s=1e-8, delay_max_s=1.0, n_points=64, seed=0)
    g2m1 = util.generate_synthetic_correlogram([pop], **_GEO, output_form='g2m1', **common)
    g2 = util.generate_synthetic_correlogram([pop], **_GEO, output_form='g2', **common)
    g1 = util.generate_synthetic_correlogram([pop], **_GEO, output_form='g1', **common)
    # g2 = 1 + g2m1 ; g2m1 = beta * g1^2 (Siegert).
    np.testing.assert_allclose(g2.signal, 1.0 + g2m1.signal, rtol=1e-9)
    np.testing.assert_allclose(g2m1.signal, 0.75 * g1.signal ** 2, rtol=1e-9)


def test_generate_correlogram_guards():
    pop = util.SyntheticPopulation(rh_nm=30.0, weight=1.0)
    with pytest.raises(ValueError):
        util.generate_synthetic_correlogram([], **_GEO)                    # empty pops
    with pytest.raises(ValueError):
        util.generate_synthetic_correlogram([pop], **_GEO, beta=1.5)       # bad beta
    with pytest.raises(ValueError):
        util.generate_synthetic_correlogram([pop], **_GEO,
                                            delay_min_s=1.0, delay_max_s=1.0)  # min>=max
    with pytest.raises(ValueError):
        util.generate_synthetic_correlogram([pop], **_GEO, output_form='bogus')


# ---------------------------------------------------------------------------
# I*sin(theta) diagnostic
# ---------------------------------------------------------------------------

def _sls(intensities, angles, label='m'):
    angles = np.asarray(angles, dtype=float)
    return SLSMeasurement(
        angles_deg=angles, intensities=np.asarray(intensities, dtype=float),
        polymer_name='PS', solvent_name='toluene', concentration_g_per_mL=0.0,
        temperature_K=298.15, wavelength_nm=633.0, solvent_refractive_index=1.33,
        dn_dc_mL_per_g=0.1, sample_label=label)


def test_i_sin_theta_flat_for_isotropic():
    # Isotropic scatterer: measured I ~ 1/sin(theta) (scattering-volume factor),
    # so I*sin(theta) is flat across angle.
    angles = np.array([30.0, 50.0, 70.0, 90.0, 110.0, 130.0, 150.0])
    intensities = 100.0 / np.sin(np.radians(angles))
    res = util.i_sin_theta([_sls(intensities, angles)], mode='absolute')
    curve = res.curves[0].i_sin_theta
    np.testing.assert_allclose(curve, curve[0], rtol=1e-9)


def test_i_sin_theta_normalized_mean_unity():
    angles = np.array([40.0, 60.0, 90.0, 120.0, 140.0])
    intensities = np.array([120.0, 90.0, 80.0, 95.0, 130.0])
    res = util.i_sin_theta([_sls(intensities, angles)], mode='normalized')
    assert res.mode == 'normalized'
    assert res.curves[0].i_sin_theta.mean() == pytest.approx(1.0, rel=1e-12)


def test_i_sin_theta_guards():
    with pytest.raises(ValueError):
        util.i_sin_theta([], mode='absolute')                 # empty
    with pytest.raises(ValueError):
        util.i_sin_theta([_sls([1.0], [90.0])], mode='bogus')  # bad mode
