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
