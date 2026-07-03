"""Headless GUI smoke tests for the DLS Buddy PySide6 shell.

Every test is marked ``@pytest.mark.gui`` and runs under Qt's offscreen
platform (the ``qt_app`` fixture from ``conftest.py`` sets
``QT_QPA_PLATFORM=offscreen`` before importing PySide6, importorskips PySide6,
and auto-confirms ``QMessageBox.question`` so modal dialogs never block).

These are *smoke* tests: they assert that the shell builds, that every module
tab constructs, that a scripted load -> commit -> analyze through the controller
lands a correct result with populated plot arrays (and that the embedded DLS
figure draws artists), and that the Settings seed/apply/restore round-trip works
against a temp ``settings.json``.

Run (with the project venv active):
    python -m pytest tests/test_gui_smoke.py -q
The ``qt_app`` fixture selects the offscreen platform automatically.
"""
from __future__ import annotations

import pytest

from fixtures.synthetic_dls import monomodal
from fixtures.synthetic_sls import make_sls_set


# The six module tabs, in the order the shell adds them (gui/main_window.py).
_EXPECTED_TABS = ['Data', 'DLS', 'SLS', 'Cross-Sample', 'Utilities', 'Settings']


def _dls_params(m):
    """The editable DLS parameter dict for a synthetic DLSMeasurement, matching
    the load path in gui/main_window._on_load_dls / validate_nonlinear_cumulant
    group G."""
    return dict(
        polymer_name=m.polymer_name, solvent_name=m.solvent_name,
        temperature_K=m.temperature_K, angle_deg=m.angle_deg,
        wavelength_nm=m.wavelength_nm,
        solvent_refractive_index=m.solvent_refractive_index,
        viscosity_Pa_s=m.viscosity_Pa_s, concentration_g_per_mL=None,
    )


def _inject_sls(controller):
    """Load a synthetic SLS set (solvent ref + a concentration series) through the
    controller, mirroring the load->commit flow. Returns a c>0 item_id."""
    meas, _cal = make_sls_set(mw=1.0e6, rg_nm=40.0, a2_mol_mL_per_g2=9.0e-5,
                              angles=[40.0, 90.0, 120.0], concs_g=[2.0e-4, 4.0e-4])
    last = None
    for m in meas:
        raw = {'angles_deg': [float(a) for a in m.angles_deg],
               'intensities': [float(x) for x in m.intensities]}
        params = dict(polymer_name=m.polymer_name, solvent_name=m.solvent_name,
                      concentration_g_per_mL=m.concentration_g_per_mL,
                      temperature_K=m.temperature_K, wavelength_nm=m.wavelength_nm,
                      solvent_refractive_index=m.solvent_refractive_index,
                      dn_dc_mL_per_g=m.dn_dc_mL_per_g)
        iid = controller.add_loaded('sls', raw, params)
        if m.concentration_g_per_mL:                # a c>0 row (not the solvent ref)
            last = iid
    controller.commit()
    return last


def _inject_dls(controller, rh_nm=30.0):
    """Add one synthetic monomodal correlogram to the controller and commit it,
    mirroring the parser -> add_loaded -> commit flow. Returns the item_id."""
    m = monomodal(rh_nm)
    raw = {
        'delay_times_s': [float(x) for x in m.delay_times_s],
        'correlogram': [float(x) for x in m.correlogram],
    }
    iid = controller.add_loaded('dls', raw, _dls_params(m))
    controller.commit()
    return iid


# --------------------------------------------------------------------------- #
# Shell                                                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.gui
def test_mainwindow_builds_with_six_named_tabs(qt_app, temp_settings_path):
    """The shell builds and exposes exactly the six agreed module tabs, named
    and ordered as the code declares them."""
    from gui.main_window import MainWindow

    mw = MainWindow()
    try:
        assert mw.tabs.count() == 6
        titles = [mw.tabs.tabText(i) for i in range(mw.tabs.count())]
        assert titles == _EXPECTED_TABS
    finally:
        mw.close()


@pytest.mark.gui
def test_mainwindow_exposes_each_module(qt_app, temp_settings_path):
    """The shell wires up one instance of every module type."""
    from gui.main_window import MainWindow
    from gui.data_module import DataModule
    from gui.dls_module import DLSModule
    from gui.sls_module import SLSModule
    from gui.cross_module import CrossSampleModule
    from gui.utilities_module import UtilitiesModule
    from gui.settings_module import SettingsModule

    mw = MainWindow()
    try:
        assert isinstance(mw.data_module, DataModule)
        assert isinstance(mw.dls_module, DLSModule)
        assert isinstance(mw.sls_module, SLSModule)
        assert isinstance(mw.cross_module, CrossSampleModule)
        assert isinstance(mw.utilities_module, UtilitiesModule)
        assert isinstance(mw.settings_module, SettingsModule)
    finally:
        mw.close()


# --------------------------------------------------------------------------- #
# Each module constructs standalone with a controller                         #
# --------------------------------------------------------------------------- #
@pytest.mark.gui
def test_each_module_constructs(qt_app, controller):
    """Every module tab constructs without raising, given only a controller
    (built here with a temp-settings controller so nothing touches the real
    settings.json)."""
    from gui.data_module import DataModule
    from gui.dls_module import DLSModule
    from gui.sls_module import SLSModule
    from gui.cross_module import CrossSampleModule
    from gui.utilities_module import UtilitiesModule
    from gui.settings_module import SettingsModule

    modules = [
        DataModule(controller),
        DLSModule(controller),
        SLSModule(controller),
        CrossSampleModule(controller),
        UtilitiesModule(controller),
        SettingsModule(controller),
    ]
    assert all(m is not None for m in modules)


# --------------------------------------------------------------------------- #
# Scripted load -> commit -> analyze through the controller                   #
# --------------------------------------------------------------------------- #
@pytest.mark.gui
def test_controller_load_commit_analyze(qt_app, controller):
    """A synthetic monomodal correlogram injected through the controller yields a
    correct cumulant result with populated plot arrays. (Rh recovers the input
    via the program's own Stokes-Einstein forward model.)"""
    iid = _inject_dls(controller, rh_nm=30.0)

    res = controller.run_cumulants(iid)

    # Rh recovers the synthetic input (~30 nm) to fit accuracy.
    assert 25.0 < res.rh_nm < 35.0
    # Plot arrays are populated and consistent in length (the GUI plots these).
    assert res.fit_tau_s.size > 0
    assert res.fitted_g2m1.size == res.fit_tau_s.size
    # And the durable Summary snapshot row was written.
    assert ('cumulants', iid) in controller.results


@pytest.mark.gui
def test_dls_module_draws_fit_artists(qt_app, controller):
    """Driving the DLS Correlogram sub-tab (tick the measurement -> Run fit)
    leaves a cached result and drawn artists on the embedded matplotlib axes."""
    from gui.dls_module import DLSModule

    iid = _inject_dls(controller, rh_nm=30.0)

    dls = DLSModule(controller)
    dls.set_measurement(iid)

    corr = dls.correlogram_tab
    # Tick the measurement for co-plotting (the checklist's shared selection),
    # then reload raw + redraw as the checklist signal would.
    dls.selection.set(iid, True)
    corr._on_selection_changed()

    # Raw data alone already draws lines on the main axis.
    assert len(corr.main_ax.lines) > 0

    # Run the parametric fit (default method = cumulant) and confirm it cached a
    # result for the ticked measurement and (re)drew the figure with artists.
    corr._on_run()
    assert corr._fit_for(iid) is not None
    assert len(corr.main_ax.lines) > 0


# --------------------------------------------------------------------------- #
# Shared selection layer (SelectionModel + MeasurementPicker)                  #
# --------------------------------------------------------------------------- #
@pytest.mark.gui
def test_measurement_picker_construct_and_toggle(qt_app, controller):
    """The shared MeasurementPicker builds real checkboxes from the workspace,
    toggles both directions (widget<->model), select-all/none works, and two
    pickers sharing one model stay in lock-step (the DLS overlay case)."""
    from PySide6 import QtCore
    from gui.widgets import SelectionModel, MeasurementPicker
    from gui.dls_module import _meas_label, _sample_header

    iid1 = _inject_dls(controller, rh_nm=30.0)
    iid2 = _inject_dls(controller, rh_nm=60.0)   # same params -> same sample

    def make():
        return MeasurementPicker(
            controller, model, kinds=('dls',),
            label_fn=_meas_label, header_fn=_sample_header,
            help_text='pick', help_bullets=['tick to include'])

    model = SelectionModel()
    p1 = make()
    p2 = make()                      # a sibling picker sharing the SAME model
    p1.refresh(); p2.refresh()

    # Both measurements are eligible rows; nothing ticked initially.
    assert set(p1._leaf_by_id) == {iid1, iid2}
    assert p1.selected_item_ids() == []

    # Widget-driven: check iid1's box -> model updates, sibling picker re-ticks it,
    # and both pickers emit selectionChanged.
    fired = {'p1': 0, 'p2': 0}
    p1.selectionChanged.connect(lambda: fired.__setitem__('p1', fired['p1'] + 1))
    p2.selectionChanged.connect(lambda: fired.__setitem__('p2', fired['p2'] + 1))
    p1._leaf_by_id[iid1].setCheckState(QtCore.Qt.CheckState.Checked)
    assert model.ids() == [iid1]
    assert p2._leaf_by_id[iid1].checkState() == QtCore.Qt.CheckState.Checked
    assert fired['p1'] == 1 and fired['p2'] == 1

    # Model-driven: ticking iid2 in the model flows to both pickers' checkboxes.
    model.set(iid2, True)
    assert p1._leaf_by_id[iid2].checkState() == QtCore.Qt.CheckState.Checked
    assert p1.selected_item_ids() == [iid1, iid2]

    # Select none / all through the picker buttons' handlers.
    p1._model.set_many(p1._leaf_by_id, False)
    assert model.ids() == []
    p1._model.set_many(p1._leaf_by_id, True)
    assert set(model.ids()) == {iid1, iid2}


def _inject_ddls(controller):
    """Load VV/VH correlogram pairs at four angles so the DDLS tab can pair them.
    Returns a VV item_id."""
    import numpy as np
    from fixtures.synthetic_dls import q_m

    d_t, d_r = 1.5e-11, 4.0e4
    first_vv = None
    for ang in (40.0, 70.0, 100.0, 130.0):
        q = q_m(ang, 633.0, 1.33)
        for geom, gamma in (('VV', d_t * q * q), ('VH', d_t * q * q + 6.0 * d_r)):
            tau = monomodal(30.0, angle_deg=ang).delay_times_s
            g = 0.9 * np.exp(-gamma * tau) ** 2
            raw = {'delay_times_s': [float(x) for x in tau],
                   'correlogram': [float(x) for x in g]}
            params = _dls_params(monomodal(30.0, angle_deg=ang))
            params['angle_deg'] = ang
            params['analyzer_geometry'] = geom
            iid = controller.add_loaded('dls', raw, params)
            if geom == 'VV' and first_vv is None:
                first_vv = iid
    controller.commit()
    return first_vv


@pytest.mark.gui
def test_ddls_include_checkboxes(qt_app, controller):
    """DDLS angle selection is include-checkboxes (Fork A): all paired angles start
    ticked; unticking one drops it from the fit (and its VV+VH sibling rows) and from
    the sidebar-mirror contract; 'Include all' restores them."""
    from PySide6 import QtCore
    from gui.dls_module import DLSModule

    vv = _inject_ddls(controller)
    sid = controller.sample_id_of(vv)

    dls = DLSModule(controller)
    dls.set_measurement(vv)
    for j in range(dls.tabs.count()):
        if dls.tabs.tabText(j) == 'DDLS':
            dls.tabs.setCurrentIndex(j)
    ddls = dls.ddls_tab

    # All four paired angles included by default; the mirror lights all 8 correlograms.
    assert ddls._included[sid] == {40.0, 70.0, 100.0, 130.0}
    assert len(set(dls.selected_item_ids())) == 8

    # Untick 40° via its checkbox -> excluded, dropped from the mirror (both VV+VH).
    for r in range(ddls.table.rowCount()):
        it = ddls.table.item(r, 0)
        if (it.data(QtCore.Qt.ItemDataRole.UserRole) == 40.0
                and it.flags() & QtCore.Qt.ItemFlag.ItemIsUserCheckable):
            it.setCheckState(QtCore.Qt.CheckState.Unchecked)
            break
    assert 40.0 not in ddls._included[sid]
    assert 40.0 in ddls._excluded_angles(sid)
    assert len(set(dls.selected_item_ids())) == 6

    ddls._on_include_all()
    assert ddls._included[sid] == {40.0, 70.0, 100.0, 130.0}


# --------------------------------------------------------------------------- #
# Utilities I·sinθ sample selector (the tab owns its sample)                   #
# --------------------------------------------------------------------------- #
@pytest.mark.gui
def test_isin_sample_selector_self_sources(qt_app, controller):
    """The I·sinθ tab picks its own sample via a SampleSelector: sidebar focus is a
    soft seed, and a non-SLS / cleared focus does NOT blank the chosen sample."""
    from gui.utilities_module import UtilitiesModule

    iid = _inject_sls(controller)
    sid = controller.sample_id_of(iid)

    u = UtilitiesModule(controller)
    u.set_measurement(iid)                          # sidebar focus seeds the selector
    assert u.isin_selector.has_sample(sid)
    assert u.isin_selector.current_sample_id() == sid
    assert u.sample_id == sid

    # Focus cleared (or a sample the tab can't plot) -> keep the current pick.
    u.set_measurement(None)
    assert u.sample_id == sid

    # A direct user pick in the selector drives the tab.
    u._on_isin_sample(sid)
    assert u.sample_id == sid


# --------------------------------------------------------------------------- #
# SLS sample selector (the tab owns its sample; Fork B)                        #
# --------------------------------------------------------------------------- #
@pytest.mark.gui
def test_sls_sample_selector_self_sources(qt_app, controller):
    """SLS picks its sample via its own SampleSelector. Focusing a DLS-only (or no)
    measurement in the sidebar does NOT blank the SLS sample — the old keep-last guard
    is gone, replaced by the tab owning its sample."""
    from gui.sls_module import SLSModule

    sls_iid = _inject_sls(controller)
    dls_iid = _inject_dls(controller, rh_nm=30.0)     # a separate DLS-only sample
    sid = controller.sample_id_of(sls_iid)

    m = SLSModule(controller)
    m.set_measurement(sls_iid)
    assert m.sls_selector.has_sample(sid)
    assert m.sls_selector.current_sample_id() == sid
    assert m.sample_id == sid

    m.set_measurement(dls_iid)     # incompatible focus -> keep the SLS sample
    assert m.sample_id == sid
    m.set_measurement(None)        # cleared focus -> still kept
    assert m.sample_id == sid


# --------------------------------------------------------------------------- #
# Sidebar selection mirror (the active tab's selection reflected on the tree)  #
# --------------------------------------------------------------------------- #
@pytest.mark.gui
def test_sidebar_mirrors_active_tab_selection(qt_app, temp_settings_path):
    """Ticking a measurement in the DLS tab tints its sidebar leaf with the
    `marker_selected` token + bold; switching to a tab with no selection concept
    (Settings) reverts the leaf. The tree is NOT rebuilt — the same leaf object is
    repainted (identity preserved)."""
    from gui.main_window import MainWindow
    from gui.theme import token

    mw = MainWindow()
    try:
        iid = _inject_dls(mw.controller, rh_nm=30.0)
        mw._refresh_sidebar()
        mw._set_current(iid)
        mw._show_module(mw.dls_module)          # DLS is the active tab
        leaf = mw._leaf_items[iid]

        # Nothing ticked yet -> not marked.
        assert not leaf.font(0).bold()

        # Tick it in the shared overlay -> the mirror lights the same leaf object.
        mw.dls_module.selection.set(iid, True)
        assert mw._leaf_items[iid] is leaf                      # not rebuilt
        assert leaf.font(0).bold()
        assert leaf.foreground(0).color().name() == token(mw.tree, 'marker_selected')

        # Switch to Settings (no selection concept) -> the mark reverts.
        mw._show_module(mw.settings_module)
        assert not leaf.font(0).bold()
        assert leaf.foreground(0).color().name() != token(mw.tree, 'marker_selected')
    finally:
        mw.close()


# --------------------------------------------------------------------------- #
# Cross-Sample tables follow the global Display-units setting                  #
# --------------------------------------------------------------------------- #
@pytest.mark.gui
def test_cross_sample_units_follow_setting(qt_app, controller):
    """The ρ-table headers/values and manual-entry fields honor the global
    Display-units choice, and a value typed in a non-default unit is stored canonical."""
    from gui.cross_module import CrossSampleModule

    controller.settings.plot_units = {'radius': 'µm', 'molar_mass': 'kg/mol'}
    m = CrossSampleModule(controller)

    # Headers + unit helpers reflect the setting.
    assert m._radius_unit() == 'µm' and m._mw_unit() == 'kg/mol'
    cols = m._columns()
    assert cols[1] == 'Rg (µm)' and cols[2] == 'Rh (µm)'
    # Display conversion: 40 nm -> 0.04 µm; missing -> n/a.
    assert m._disp_radius(40.0, 'µm') == '0.04'
    assert m._disp_radius(None, 'µm') == 'n/a'

    # Manual entry typed in the display unit round-trips to canonical storage.
    sid = controller.sample_id_of(_inject_sls(controller))
    m.refresh()
    m._current_unit = (sid, None)
    m.mw_manual.setText('1500')          # 1500 kg/mol
    m._on_manual('mw')
    r = controller.workspace.samples[sid].result_for(None)
    assert r.mw_g_per_mol == pytest.approx(1.5e6)   # 1500 kg/mol -> 1.5e6 g/mol
    assert r.mw_source == 'user'

    m.rh_manual.setText('0.03')          # 0.03 µm
    m._on_manual('rh')
    assert r.rh_nm == pytest.approx(30.0)           # 0.03 µm -> 30 nm

    m.close()


# --------------------------------------------------------------------------- #
# Settings seed / apply / restore                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.gui
def test_settings_seed_apply_restore(qt_app, controller):
    """SettingsModule seeds its skip-channels spinbox from the controller's
    settings, ``_apply`` persists an edit back to the controller, and
    ``_restore`` resets to the factory default (0). The qt_app fixture already
    stubs the switch-guard dialog so ``_restore`` never blocks."""
    from gui.settings_module import SettingsModule
    from app.settings import SettingsState

    # Seed: the spinbox initialises from the controller's current settings.
    sm = SettingsModule(controller)
    assert sm.skip_channels.value() == controller.settings.skip_initial_channels

    # Apply: an edit persists into the controller's SettingsState.
    sm.skip_channels.setValue(7)
    sm._apply()
    assert controller.settings.skip_initial_channels == 7

    # Restore: resets to the factory default (fresh SettingsState).
    sm._restore()
    assert (controller.settings.skip_initial_channels
            == SettingsState().skip_initial_channels)
