"""Regression tests for the workspace / session / controller behaviour.

Covers the framework-agnostic state layer (core/workspace.py) and the controller
(app/controller.py) that mediates it:

  * self-contained session JSON round-trips (floats to rtol 1e-9); wrong
    format/version rejected; a dict missing newer optional fields still loads
  * sample grouping by SampleKey (same identity -> one Sample; different
    temperature -> separate); assign/clear override re-bucketing that preserves
    SampleResults; new_sample_id minting
  * commit / working / dirty / revert (build() uses committed params)
  * manual-Mw provenance ('user' survives save/reload and is not overwritten by
    a recompute)
  * controller settings propagation (skip_initial_channels + cumulant_method
    default 'nonlinear', explicit override wins; distribution n_skipped)
  * the switch-clears-results guard (ported from
    the retired nonlinear-cumulant validator, group H)
  * SettingsState from_dict/to_dict tolerance + round-trip
  * run_zimm / run_ddls end-to-end into the SampleResult / result cache
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from core.workspace import (
    Workspace, LoadedMeasurement, SampleRhRow,
    sample_group_key,
)
from app.settings import SettingsState
from analysis import synthetic_dataset as synth
from fixtures.synthetic_dls import monomodal, q_m


# ------------------------------------------------------------ helpers ---

def _dls_raw(m):
    return {"delay_times_s": [float(x) for x in m.delay_times_s],
            "correlogram": [float(x) for x in m.correlogram]}


def _dls_params(*, polymer="P", solvent="water", T=298.15, angle=90.0,
                conc=None, geom=None):
    p = dict(polymer_name=polymer, solvent_name=solvent, temperature_K=T,
             angle_deg=angle, wavelength_nm=633.0, solvent_refractive_index=1.33,
             viscosity_Pa_s=8.9e-4, concentration_g_per_mL=conc)
    if geom is not None:
        p["analyzer_geometry"] = geom
    return p


def _add_dls(c, *, rh=30.0, angle=90.0, conc=None, geom=None, gamma=None, **pk):
    """Add a synthetic DLS correlogram to a controller and return its item_id."""
    if gamma is not None:
        # build a correlogram with an explicit field decay rate Gamma
        tau = monomodal(rh, angle_deg=angle).delay_times_s
        g = 0.9 * np.exp(-gamma * tau) ** 2
        raw = {"delay_times_s": [float(x) for x in tau],
               "correlogram": [float(x) for x in g]}
    else:
        raw = _dls_raw(monomodal(rh, angle_deg=angle))
    return c.add_loaded("dls", raw,
                        _dls_params(angle=angle, conc=conc, geom=geom, **pk))


def _build_zimm_sample(c, *, mw=1.0e6, rg=40.0, a2=9.0e-5, calibrated=True):
    """Load a synthetic Zimm SLS set (solvent ref + 4 concentrations) into the
    controller and set up the session calibration. Returns the sample_id."""

    angles = [40, 60, 80, 100, 120]
    concs = [2.0e-4, 4.0e-4, 6.0e-4, 8.0e-4]
    cal = synth.CalibrationSpec()
    sset = synth.build_sls_set(
        mw=mw, rg_nm=rg, a2_mol_mL_per_g2=a2, angles_deg=angles,
        concentrations_g_per_mL=concs, wavelength_nm=532.0, temperature_K=298.15,
        n_solvent=1.33, dn_dc=0.135, cal=cal, solvent_intensity_90=6000.0,
        polymer_name="PEG", solvent_name="water")

    def _sls_params(conc):
        return dict(polymer_name="PEG", solvent_name="water",
                    concentration_g_per_mL=conc, temperature_K=298.15,
                    wavelength_nm=532.0, solvent_refractive_index=1.33,
                    dn_dc_mL_per_g=0.135)

    for cc in sorted(sset.intensities):        # includes 0.0 (solvent ref)
        raw = {"angles_deg": [float(a) for a in sset.angles_deg],
               "intensities": [float(x) for x in sset.intensities[cc]]}
        c.add_loaded("sls", raw, _sls_params(cc))

    if calibrated:
        c.set_calibration_field("calibrant_intensity", cal.calibrant_intensity)
        c.set_calibration_field("calibrant_angle_deg", 90.0)
        c.set_calibration_field("standard_geometry", cal.geometry)
        c.set_calibration_field("standard_wavelength_nm", cal.wavelength_nm)
        c.set_calibration_field("standard_temperature_C", cal.temperature_C)
        c.set_calibration_field("standard_refractive_index", cal.n_standard)
    c.commit()
    return next(iter(c.workspace.samples))


# =========================================================== session JSON ===

def test_session_json_roundtrip_floats(controller):
    c = controller
    _add_dls(c, rh=30.0, angle=75.0, conc=1.3e-3)
    c.commit()
    sid = next(iter(c.workspace.samples))
    res = c.workspace.samples[sid].result_for(None)
    res.mw_g_per_mol, res.rg_nm, res.rh_nm = 1.234e6, 41.7, 28.9
    res.a2_mol_mL_per_g2 = 8.5e-5

    d = c.workspace.to_dict()
    reloaded = Workspace.from_dict(json.loads(json.dumps(d, allow_nan=True)))

    lm0 = next(iter(c.workspace.measurements.values()))
    lm1 = reloaded.measurements[lm0.item_id]
    np.testing.assert_allclose(lm1.raw["delay_times_s"],
                               lm0.raw["delay_times_s"], rtol=1e-9)
    np.testing.assert_allclose(lm1.raw["correlogram"],
                               lm0.raw["correlogram"], rtol=1e-9)
    assert lm1.committed_params["concentration_g_per_mL"] == pytest.approx(
        1.3e-3, rel=1e-9)
    assert lm1.committed_params["angle_deg"] == pytest.approx(75.0, rel=1e-9)

    r1 = reloaded.samples[sid].result_for(None)
    assert r1.mw_g_per_mol == pytest.approx(1.234e6, rel=1e-9)
    assert r1.rg_nm == pytest.approx(41.7, rel=1e-9)
    assert r1.rh_nm == pytest.approx(28.9, rel=1e-9)
    assert r1.a2_mol_mL_per_g2 == pytest.approx(8.5e-5, rel=1e-9)


def test_session_wrong_format_and_version_raise():
    with pytest.raises(ValueError):
        Workspace.from_dict({"format": "not_ours", "version": 1})
    with pytest.raises(ValueError):
        Workspace.from_dict({"format": "ls_session", "version": 999})


def test_session_missing_optional_fields_loads():
    # A minimal dict lacking traces / dls_result_rows / sample_rh_rows / samples
    # still loads: the absent lists take their defaults and samples regroup.
    lm = LoadedMeasurement(
        item_id="m0001", kind="dls",
        raw={"delay_times_s": [1e-6, 2e-6], "correlogram": [0.5, 0.4]},
        working_params=_dls_params(), committed_params=_dls_params())
    minimal = {"format": "ls_session", "version": 1,
               "measurements": [lm.to_dict()]}
    ws = Workspace.from_dict(minimal)
    assert "m0001" in ws.measurements
    assert ws.traces == {}
    assert ws.dls_result_rows == {}
    assert len(ws.samples) == 1              # regrouped from the one measurement


# =========================================================== grouping ===

def test_sample_group_key_temperature_rounding():
    assert (sample_group_key("PVP", "water", 298.15)
            == sample_group_key("PVP", "Water", 298.150001))
    assert (sample_group_key("PVP", "water", 308.15)
            != sample_group_key("PVP", "water", 298.15))


def test_grouping_same_identity_one_sample_diff_temperature_splits(controller):
    c = controller
    _add_dls(c, angle=60.0, T=298.15)
    _add_dls(c, angle=120.0, T=298.15)       # same identity -> same sample
    _add_dls(c, angle=90.0, T=308.15)        # different T -> its own sample
    c.commit()
    assert len(c.workspace.samples) == 2
    sizes = sorted(len(s.dls_item_ids) for s in c.workspace.samples.values())
    assert sizes == [1, 2]


def test_assign_and_clear_override_preserve_results(controller):
    c = controller
    a = _add_dls(c, angle=60.0)
    b = _add_dls(c, angle=120.0)             # a and b -> one sample
    c.commit()
    sid_ab = next(iter(c.workspace.samples))
    c.workspace.samples[sid_ab].result_for(None).mw_g_per_mol = 5.0e5

    # move measurement b into a fresh manual sample
    new_sid = c.new_sample_id()
    assert new_sid.startswith("override-")
    c.assign_to_sample(b, new_sid)

    # a's original sample still exists and KEEPS its SampleResult (sid stable)
    assert sid_ab in c.workspace.samples
    assert c.workspace.samples[sid_ab].result_for(None).mw_g_per_mol == pytest.approx(5.0e5)
    assert a in c.workspace.samples[sid_ab].dls_item_ids
    assert b in c.workspace.samples[new_sid].dls_item_ids

    # clearing the override re-buckets b back with a
    c.clear_override(b)
    assert b in c.workspace.samples[sid_ab].dls_item_ids


# =========================================================== commit / dirty ===

def test_commit_working_dirty_revert(controller):
    c = controller
    iid = _add_dls(c, angle=90.0, conc=1.0e-3)
    c.commit()
    lm = c.workspace.measurements[iid]
    assert not lm.is_dirty()

    c.set_param(iid, "concentration_g_per_mL", 2.0e-3)
    assert lm.is_dirty()
    assert "concentration_g_per_mL" in c.dirty_keys(iid)
    # build() uses COMMITTED params -> still the old value until commit
    assert lm.build().concentration_g_per_mL == pytest.approx(1.0e-3)

    c.undo_to_committed()                    # revert working <- committed
    assert not lm.is_dirty()
    assert lm.working_params["concentration_g_per_mL"] == pytest.approx(1.0e-3)

    c.set_param(iid, "concentration_g_per_mL", 2.0e-3)
    c.commit()
    assert lm.build().concentration_g_per_mL == pytest.approx(2.0e-3)


# =========================================================== settings ===

def test_settings_from_dict_missing_key_defaults():
    s = SettingsState.from_dict({"cumulant_order": 2})   # no cumulant_method / skip
    assert s.cumulant_method == "nonlinear"
    assert s.skip_initial_channels == 0


def test_settings_roundtrip_and_ignores_unknown():
    s = SettingsState(cumulant_method="linear", skip_initial_channels=5)
    s2 = SettingsState.from_dict(s.to_dict())
    assert s2.cumulant_method == "linear"
    assert s2.skip_initial_channels == 5
    # an unknown key from a newer build is ignored, not fatal
    s3 = SettingsState.from_dict({**s.to_dict(), "brand_new_setting": 42})
    assert s3.cumulant_method == "linear"


# =========================================================== controller: DLS ===

def test_run_cumulants_inherits_settings_and_override_wins(controller):
    c = controller
    assert c.settings.cumulant_method == "nonlinear"   # controller default
    iid = _add_dls(c, rh=30.0, angle=90.0, conc=None)
    c.commit()
    # inherits the nonlinear default from settings
    assert c.run_cumulants(iid).method == "nonlinear"
    # an explicit per-run override wins
    assert c.run_cumulants(iid, method="linear").method == "linear"


def test_run_cumulants_inherits_skip_channels(controller):
    c = controller
    c.apply_settings(SettingsState(skip_initial_channels=7), persist=False)
    iid = _add_dls(c, rh=30.0, angle=90.0, conc=None)
    c.commit()
    res = c.run_cumulants(iid)
    assert res.n_skipped == 7


def test_run_distribution_propagates_n_skipped(controller):
    c = controller
    c.apply_settings(SettingsState(skip_initial_channels=4), persist=False)
    iid = _add_dls(c, rh=40.0, angle=90.0, conc=None)
    c.commit()
    for method in ("nnls", "contin"):
        res = c.run_distribution(iid, method=method, n_grid=40)
        dist = getattr(res, "distribution", res)
        assert dist.n_skipped == 4


def test_switch_clears_cumulant_dependent_results(controller):
    # Ported from the retired nonlinear-cumulant validator (group H): changing the
    # cumulant method clears cumulant + gamma_q2 + cumulant replicate rows but
    # keeps distributions, a user Rh, and a single-exponential replicate row.
    c = controller
    iids = []
    for ang in (60.0, 120.0):
        iids.append(_add_dls(c, rh=30.0, angle=ang, conc=1.0e-3))
    c.commit()
    sid = next(iter(c.workspace.samples))

    c.run_cumulants(iids[0])                      # cumulant fit
    c.run_distribution(iids[0], method="nnls")   # distribution (must survive)
    c.run_gamma_q2(sid)                          # cumulant-sourced gamma-vs-q^2

    # a hand-entered user Rh on the sample (must survive)
    res = c.workspace.samples[sid].result_for(None)
    res.rh_nm, res.rh_source, res.rh_label = 42.0, "user", "hand entered"

    # a cumulant replicate average row + a single-exp one (only cumulant clears)
    c.workspace.upsert_sample_rh_row(SampleRhRow(
        sample_id=sid, source_kind="replicate_avg", source_set="cumulant|None",
        rh_nm=29.0, is_apparent=True, from_label="replicate avg (10 runs, cumulant)"))
    c.workspace.upsert_sample_rh_row(SampleRhRow(
        sample_id=sid, source_kind="replicate_avg", source_set="single|None",
        rh_nm=31.0, is_apparent=True, from_label="replicate avg (10 runs, single)"))

    assert c.cumulant_dependent_result_count() > 0
    dist_before = ("distribution", iids[0], "nnls") in c.results

    c.clear_cumulant_dependent_results()

    # cumulant-derived caches / rows are gone
    assert ("cumulants", iids[0]) not in c.results
    assert all(r.method != "cumulant" for r in c.workspace.dls_result_rows.values())
    assert ("gamma_q2", sid, None) not in c.results
    assert not any(r.source_kind == "gamma_q2"
                   for r in c.workspace.sample_rh_rows.values())
    # distribution survives
    assert dist_before and ("distribution", iids[0], "nnls") in c.results
    # user Rh untouched
    r = c.workspace.samples[sid].result_for(None)
    assert r.rh_nm == 42.0 and r.rh_source == "user"
    # cumulant replicate row removed, single-exp replicate row kept
    kinds = [(r.source_kind, str(r.source_set).split("|")[0])
             for r in c.workspace.sample_rh_rows.values()]
    assert ("replicate_avg", "cumulant") not in kinds
    assert ("replicate_avg", "single") in kinds
    assert c.cumulant_dependent_result_count() == 0


# =========================================================== controller: SLS ===

def test_run_zimm_writes_mw_rg_into_sample_result(controller):
    c = controller
    sid = _build_zimm_sample(c, mw=1.0e6, rg=40.0)
    res = c.run_zimm(sid, method="zimm")
    # the analysis recovers the ground truth
    assert res.mw_g_per_mol == pytest.approx(1.0e6, rel=0.05)
    assert res.rg_nm == pytest.approx(40.0, rel=0.05)
    # ...and it is written into the sample's SampleResult (computed provenance)
    sr = c.workspace.samples[sid].result_for(None)
    assert sr.mw_g_per_mol == pytest.approx(res.mw_g_per_mol, rel=1e-9)
    assert sr.rg_nm == pytest.approx(res.rg_nm, rel=1e-9)
    assert sr.mw_source == "computed"


def test_manual_mw_survives_reload_and_recompute(controller, tmp_path):
    c = controller
    sid = _build_zimm_sample(c, mw=1.0e6, rg=40.0)
    c.set_manual_mw(sid, 7.7e5)              # hand-entered (e.g. characterised in water)
    sr = c.workspace.samples[sid].result_for(None)
    assert sr.mw_source == "user"
    # a hand-entered Mw is a trusted (thermodynamic, calibrated) value
    assert sr.mw_apparent is False
    assert sr.calibrated is True

    # a re-analysis must NOT overwrite a user Mw
    c.run_zimm(sid, method="zimm")
    sr = c.workspace.samples[sid].result_for(None)
    assert sr.mw_g_per_mol == pytest.approx(7.7e5, rel=1e-9)
    assert sr.mw_source == "user"

    # and it survives a session save -> reload
    path = tmp_path / "session.lsjson"
    c.save_session(str(path))
    c.load_session(str(path))
    sr = c.workspace.samples[sid].result_for(None)
    assert sr.mw_g_per_mol == pytest.approx(7.7e5, rel=1e-9)
    assert sr.mw_source == "user"
    assert sr.calibrated is True


def test_save_load_session_reloads_equal(controller, tmp_path):
    c = controller
    iid = _add_dls(c, rh=30.0, angle=90.0, conc=1.0e-3)
    c.commit()
    lm0 = c.workspace.measurements[iid]

    path = tmp_path / "roundtrip.lsjson"
    c.save_session(str(path))
    c.load_session(str(path))

    lm1 = c.workspace.measurements[iid]
    np.testing.assert_allclose(lm1.raw["correlogram"], lm0.raw["correlogram"],
                               rtol=1e-9)
    assert lm1.committed_params == lm0.committed_params


# =========================================================== controller: DDLS ===

def test_run_ddls_recovers_dt_dr(controller):
    c = controller
    d_t, d_r = 1.5e-11, 4.0e4
    for ang in (40.0, 70.0, 100.0, 130.0):
        q = q_m(ang, 633.0, 1.33)
        g_vv = d_t * q * q
        g_vh = g_vv + 6.0 * d_r
        _add_dls(c, angle=ang, geom="VV", gamma=g_vv)
        _add_dls(c, angle=ang, geom="VH", gamma=g_vh)
    c.commit()
    sid = next(iter(c.workspace.samples))

    res, info = c.run_ddls(sid)
    assert info["paired_angles"] == [40.0, 70.0, 100.0, 130.0]
    assert res.d_r_rad2_s == pytest.approx(d_r, rel=0.05)
    assert res.d_t_m2_s == pytest.approx(d_t, rel=0.05)
    assert res.rh_t_nm > 0.0
    # result is cached for the sample
    assert ("ddls", sid) in c.results


# ============================================ controller: averaged-Rh candidate ===

def test_replicate_average_rh_wired_into_candidates(controller):
    """A replicate-averaged Rh (durable sample_rh_rows) surfaces as a Cross-Sample Rh
    candidate and is PREFERRED over a single cumulant, so an auto-refresh keeps the
    deliberately-averaged value rather than clobbering it (feedback 2026-07-02)."""
    from analysis.utilities import select_default_candidate

    c = controller
    iid = _add_dls(c, rh=30.0, angle=90.0, conc=1.0e-4)
    c.commit()
    sid = c.sample_id_of(iid)

    # Before averaging: a single cumulant candidate, no replicate_avg.
    base = c.dls_rh_candidates(sid, None)
    assert base and all(cd.kind != "dls_replicate_avg" for cd in base)

    c.workspace.upsert_sample_rh_row(SampleRhRow(
        sample_id=sid, source_kind="replicate_avg", source_set="cumulant|None",
        rh_nm=31.5, rh_se=0.4, is_apparent=True, rh_type_label="apparent",
        from_label="replicate avg (3 runs, cumulant)", fraction=None))

    cands = c.dls_rh_candidates(sid, None)
    avg = [cd for cd in cands if cd.kind == "dls_replicate_avg"]
    assert len(avg) == 1
    a = avg[0]
    assert a.value == pytest.approx(31.5) and a.value_se == pytest.approx(0.4)
    assert a.is_apparent and a.tier == 1
    # Preferred over the single cumulant -> auto-select keeps the averaged value.
    assert select_default_candidate(cands) is a

    # Fraction-scoped: not offered for a different Mw fraction.
    assert all(cd.kind != "dls_replicate_avg"
               for cd in c.dls_rh_candidates(sid, "A"))
