"""Background-threading tests (gui/worker.py + the async Run paths).

Every test is ``@pytest.mark.gui`` (offscreen Qt via the ``qt_app`` fixture).
There is no pytest-qt in the suite, so event-loop spinning is done with a
local ``_spin_until`` helper; the autouse ``_idle_runner`` fixture asserts the
shared singleton runner is idle after each test so a wedged worker fails in
the test that caused it, not a later one.

Acceptance criteria covered (spec 2026-07-01):
- results are delivered on the main thread via the finished signal;
- one job in flight app-wide (second submit is refused);
- the failed path delivers the exception to ``on_fail`` and re-enables the UI;
- an async run returns numbers identical to the inline (synchronous) call;
- a seeded synthetic build is deterministic across the async boundary;
- ``app/`` and ``analysis/`` stay Qt-free (invariant 5, as a test).
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import pytest

from test_gui_smoke import _inject_dls


def _spin_until(qt_app, predicate, timeout_s: float = 15.0) -> bool:
    """Process events until ``predicate()`` is true or the timeout expires."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        qt_app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture(autouse=True)
def _idle_runner(qt_app):
    """Fail loudly (here, not in a later test) if a test leaves a job running
    on the process-wide singleton runner."""
    yield
    from gui import worker
    worker._reset_for_tests()


# --------------------------------------------------------------------------- #
# Runner unit tests                                                            #
# --------------------------------------------------------------------------- #
@pytest.mark.gui
def test_runner_delivers_result_on_main_thread(qt_app):
    """The finished payload arrives via on_done, on the GUI thread, with
    busy_changed toggling True -> False around it."""
    from PySide6 import QtCore
    from gui.worker import runner

    r = runner()
    seen = {}
    busy_events = []
    r.busy_changed.connect(lambda b, d: busy_events.append((b, d)))

    ok = r.try_submit(
        lambda: 41 + 1,
        on_done=lambda result: seen.update(
            result=result,
            on_main=QtCore.QThread.currentThread() is qt_app.thread()),
        description='unit test',
    )
    assert ok
    assert r.is_busy
    assert _spin_until(qt_app, lambda: 'result' in seen)
    assert seen['result'] == 42
    assert seen['on_main'] is True
    assert not r.is_busy
    assert busy_events == [(True, 'unit test'), (False, 'unit test')]


@pytest.mark.gui
def test_runner_refuses_second_submit_while_busy(qt_app):
    """Only one job in flight app-wide: a second try_submit returns False and
    runs nothing; after completion the runner accepts again."""
    from gui.worker import runner

    r = runner()
    done = []
    assert r.try_submit(lambda: time.sleep(0.2) or 'first',
                        on_done=done.append)
    assert r.try_submit(lambda: 'second', on_done=done.append) is False
    assert _spin_until(qt_app, lambda: done)
    assert done == ['first']
    # Idle again: a new submit is accepted.
    assert r.try_submit(lambda: 'third', on_done=done.append)
    assert _spin_until(qt_app, lambda: len(done) == 2)
    assert done == ['first', 'third']


@pytest.mark.gui
def test_runner_failed_path_reenables_widgets(qt_app):
    """A raising thunk routes the exception instance to on_fail on the main
    thread, and the busy widgets are re-enabled before it runs."""
    from PySide6 import QtWidgets
    from gui.worker import runner

    r = runner()
    button = QtWidgets.QPushButton('Run')
    boom = ValueError('synthetic failure')
    caught = []

    def on_fail(exc):
        caught.append((exc, button.isEnabled()))

    assert r.try_submit(lambda: (_ for _ in ()).throw(boom),
                        on_done=lambda _: pytest.fail('on_done on failure'),
                        on_fail=on_fail, busy_widgets=(button,))
    assert not button.isEnabled()
    assert _spin_until(qt_app, lambda: caught)
    exc, enabled_when_called = caught[0]
    assert exc is boom
    assert enabled_when_called is True
    assert button.isEnabled()


@pytest.mark.gui
def test_run_when_idle_defers_until_completion(qt_app):
    """run_when_idle fires immediately when idle, and exactly once on
    completion when a job is in flight."""
    from gui.worker import runner, run_when_idle

    r = runner()
    calls = []
    run_when_idle(lambda: calls.append('immediate'))
    assert calls == ['immediate']

    assert r.try_submit(lambda: time.sleep(0.2), on_done=lambda _: None)
    run_when_idle(lambda: calls.append('deferred'))
    assert calls == ['immediate']          # not yet
    assert r.wait_for_idle()
    assert calls == ['immediate', 'deferred']
    # A later completion must not re-fire the one-shot.
    assert r.try_submit(lambda: None, on_done=lambda _: None)
    assert r.wait_for_idle()
    assert calls == ['immediate', 'deferred']


@pytest.mark.gui
def test_drain_idle_does_not_livelock(qt_app):
    """A deferred callback that starts a new job, followed by one that re-defers
    itself while now-busy, must not spin _drain_idle forever: the re-deferred
    callback lands in the NEXT drain (after the new job), not the current one.
    (Regression for the snapshot-and-clear drain — a while-loop here hangs the GUI.)"""
    from gui.worker import runner, run_when_idle

    r = runner()
    log = []
    assert r.try_submit(lambda: time.sleep(0.05),
                        on_done=lambda _: log.append('jobA'))
    # cb1 starts a second job mid-drain, so is_busy becomes True for cb2.
    run_when_idle(lambda: (log.append('cb1'),
                           r.try_submit(lambda: time.sleep(0.05),
                                        on_done=lambda _: log.append('jobB'))))

    def cb2():
        log.append('cb2')
        run_when_idle(lambda: log.append('cb2-again'))   # re-defers while busy

    run_when_idle(cb2)
    assert r.wait_for_idle()          # must RETURN (no hang), waiting out jobB too
    qt_app.processEvents()
    assert log.count('cb1') == 1      # each ran exactly once in the first drain
    assert log.count('cb2') == 1
    assert 'cb2-again' in log         # the re-deferred one fired on jobB's drain


@pytest.mark.gui
def test_on_done_runs_before_deferred_idle_callbacks(qt_app):
    """The finishing job's own on_done runs BEFORE any run_when_idle callback
    registered while it was in flight — so a deferred refresh never pre-empts (or,
    if it raised, skipped) the completing analysis's render (worker.py ordering)."""
    from gui.worker import runner, run_when_idle

    r = runner()
    order = []
    assert r.try_submit(lambda: time.sleep(0.15),
                        on_done=lambda _: order.append('done'))
    run_when_idle(lambda: order.append('deferred'))   # busy → queued
    assert r.wait_for_idle()
    qt_app.processEvents()
    assert order == ['done', 'deferred']


# --------------------------------------------------------------------------- #
# Async == inline (spec acceptance 6) and determinism (acceptance 7)           #
# --------------------------------------------------------------------------- #
@pytest.mark.gui
def test_async_distribution_equals_inline(qt_app, controller):
    """The identical run_distribution call gives identical numbers whether run
    inline or dispatched through the worker. NNLS (not CONTIN) keeps this in
    the fast tier; run_distribution is routed whole, so the method is
    representative of the path every distribution method takes."""
    import numpy as np
    from gui.worker import runner

    iid = _inject_dls(controller, rh_nm=30.0)
    inline = controller.run_distribution(iid, 'nnls')

    got = []
    assert runner().try_submit(
        lambda: controller.run_distribution(iid, 'nnls'), got.append)
    assert _spin_until(qt_app, lambda: got)
    res = got[0]

    d_inline = getattr(inline, 'distribution', inline)
    d_async = getattr(res, 'distribution', res)
    assert np.array_equal(d_inline.rh_grid_nm, d_async.rh_grid_nm)
    assert np.array_equal(d_inline.weights, d_async.weights)
    assert np.array_equal(d_inline.residuals, d_async.residuals)


@pytest.mark.gui
def test_seeded_synth_deterministic_across_worker(qt_app, controller):
    """A seeded synthetic build returns bit-identical arrays inline vs via the
    worker (determinism is unchanged by the dispatch)."""
    import numpy as np
    from gui.worker import runner

    specs = [dict(rh_nm=30.0, weight=1.0)]
    kw = dict(angle_deg=90.0, wavelength_nm=532.0, solvent_refractive_index=1.33,
              temperature_K=298.15, viscosity_Pa_s=0.00089,
              noise_level=0.01, seed=42)
    inline = controller.synth_correlogram(specs, **kw)

    got = []
    assert runner().try_submit(
        lambda: controller.synth_correlogram(specs, **kw), got.append)
    assert _spin_until(qt_app, lambda: got)
    assert np.array_equal(inline.delay_times_s, got[0].delay_times_s)
    assert np.array_equal(inline.signal, got[0].signal)


# --------------------------------------------------------------------------- #
# The GUI Run path end-to-end (async slot -> worker -> redraw)                 #
# --------------------------------------------------------------------------- #
@pytest.mark.gui
def test_distribution_tab_runs_async(qt_app, controller):
    """Driving the Distribution sub-tab's Run slot dispatches to the worker
    (busy immediately after the call), then lands the results, enables export,
    and draws curves once the event loop delivers the finished signal."""
    from gui.dls_module import DLSModule
    from gui.worker import runner

    iid = _inject_dls(controller, rh_nm=30.0)
    dls = DLSModule(controller)
    dls.set_measurement(iid)
    dls.selection.set(iid, True)

    dist = dls.distribution_tab
    for key, cb in dist.method_checks.items():
        cb.setChecked(key == 'nnls')            # NNLS only: fast tier

    dist._on_run()
    assert runner().is_busy                     # dispatched, not run inline
    assert not dist._results                    # nothing landed yet
    assert _spin_until(qt_app, lambda: dist._results)
    assert len(dist._results) == 1
    assert dist._results[0][:2] == (iid, 'nnls')
    assert dist.export_button.isEnabled()
    assert len(dist.ax.lines) > 0               # drawn on completion


@pytest.mark.gui
def test_correlogram_fit_refused_while_worker_busy(qt_app, controller):
    """The synchronous Correlogram fit path writes controller.results, so it must
    refuse (not run) while a background fit is writing the same dict — otherwise
    two threads mutate controller.results concurrently (invariant 4)."""
    from gui.dls_module import DLSModule
    from gui.worker import runner, BUSY_NOTICE

    iid = _inject_dls(controller, rh_nm=30.0)
    dls = DLSModule(controller)
    dls.set_measurement(iid)
    dls.selection.set(iid, True)
    corr = dls.correlogram_tab
    corr._on_selection_changed()                # loads raw into the tab

    assert runner().try_submit(lambda: time.sleep(0.3), on_done=lambda _: None)
    assert runner().is_busy
    corr._on_run()                              # must refuse, not fit
    assert not corr._cache                      # no fit ran on the main thread
    assert corr.status.text() == BUSY_NOTICE
    assert runner().wait_for_idle()


@pytest.mark.gui
def test_distribution_tab_drops_superseded_result(qt_app, controller):
    """Changing the ticked selection while a fit is in flight bumps the epoch:
    the late-arriving result is discarded instead of drawing a stale plot
    (spec acceptance 4 — supersede, no interleave)."""
    from gui.dls_module import DLSModule
    from gui.worker import runner

    iid = _inject_dls(controller, rh_nm=30.0)
    dls = DLSModule(controller)
    dls.set_measurement(iid)
    dls.selection.set(iid, True)

    dist = dls.distribution_tab
    for key, cb in dist.method_checks.items():
        cb.setChecked(key == 'nnls')

    dist._on_run()
    dist._on_selection_changed()                # inputs changed mid-flight
    assert runner().wait_for_idle()
    qt_app.processEvents()
    assert dist._results == []                  # stale payload dropped
    assert not dist.export_button.isEnabled()


@pytest.mark.gui
def test_sls_module_runs_async(qt_app, controller):
    """Driving the SLS Run slot dispatches the compute phase to the worker, then
    the present phase plots + fills the result table on completion. Exercises the
    compute/present split and the _full_rr overlay cache end-to-end."""
    from gui.sls_module import SLSModule
    from gui.worker import runner

    truth = dict(mw=1.0e6, rg_nm=40.0, a2_mol_mL_per_g2=1.0e-4)
    sls_set = controller.synth_sls_set(
        **truth, angles_deg=[50.0, 90.0, 130.0],
        concentrations_g_per_mL=[0.0, 5e-4, 1e-3, 2e-3],
        wavelength_nm=532.0, temperature_K=298.15, n_solvent=1.33, dn_dc=0.15,
        calibrated=True, seed=1)
    controller.inject_sls_set(sls_set, polymer_name='PEG', solvent_name='water',
                              prefill_calibration=True)
    controller.commit()
    sid = controller.samples()[0].sample_id

    sls = SLSModule(controller)
    sls.set_measurement(controller.workspace.samples[sid].sls_item_ids[0])
    assert sls._runnable

    # Zimm is the default method; drive Run and let the worker deliver.
    sls._on_run()
    assert runner().is_busy
    assert _spin_until(qt_app, lambda: not runner().is_busy)
    qt_app.processEvents()
    assert sls._ran
    assert sls.export_button.isEnabled()
    assert (sid, sls._fraction) in sls._full_rr     # overlay cache populated
    assert sls.result_table.rowCount() > 0


# --------------------------------------------------------------------------- #
# Invariant 5: the layers below gui/ stay Qt-free (spec acceptance 5)          #
# --------------------------------------------------------------------------- #
def test_app_and_analysis_stay_qt_free():
    """No Qt *imports* below gui/ (docstring prose may mention PySide6; an
    import is the leak the invariant forbids)."""
    root = Path(__file__).resolve().parents[1]
    pattern = re.compile(
        r'^\s*(import|from)\s+(PySide6|PyQt\d|qtpy|shiboken)')
    offenders = [
        f'{py.relative_to(root)}:{lineno}'
        for pkg in ('app', 'analysis')
        for py in (root / pkg).rglob('*.py')
        for lineno, line in enumerate(py.read_text(encoding='utf-8')
                                      .splitlines(), 1)
        if pattern.match(line)
    ]
    assert not offenders, f'Qt leaked below gui/: {offenders}'
