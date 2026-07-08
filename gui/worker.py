"""
gui/worker.py
=============

Background dispatch for the heavy controller calls (threading enablement,
spec 2026-07-01).

The controller stays Qt-free (architecture invariant 5): widgets build a
zero-argument *thunk* on the main thread that binds a controller method to its
already-read arguments, then hand it to :func:`runner`'s ``try_submit``. The
thunk runs on a single pooled worker thread; ``on_done(result)`` /
``on_fail(exception)`` are invoked back on the GUI thread via queued signal
delivery, so widget code never touches Qt objects from the worker.

Deliberate non-features (spec boundaries):

- **One job in flight, app-wide.** ``try_submit`` refuses while busy — this is
  the global re-entry guard (invariant 4): controller ``run_*`` methods mutate
  shared state, so two concurrent runs are never allowed, whichever tabs they
  come from. No queue, no priorities.
- **No cancellation.** A running scipy call cannot be safely interrupted; a
  *stale* result (the widget's inputs changed while the job ran) is dropped by
  the caller's epoch check in ``on_done``, not by killing the computation.
- **One worker thread.** The goal is responsiveness, not multicore throughput:
  while the worker blocks in scipy, the Qt event loop keeps running.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Tuple

from PySide6 import QtCore, QtWidgets


class _Signals(QtCore.QObject):
    """Bridge from the worker thread back to the GUI thread.

    Created with the runner (on the main thread), so it has main-thread
    affinity: ``emit`` from the pooled thread crosses via Qt's automatic
    queued connection and the connected slots run on the GUI thread."""

    finished = QtCore.Signal(int, object)   # (job_id, result)
    failed = QtCore.Signal(int, object)     # (job_id, exception)


class _Job(QtCore.QRunnable):
    """One dispatched thunk. Runs on the pool thread — no widget access here."""

    def __init__(self, job_id: int, thunk: Callable[[], Any],
                 signals: _Signals) -> None:
        super().__init__()
        self._id = job_id
        self._thunk = thunk
        self._signals = signals

    def run(self) -> None:
        try:
            result = self._thunk()
        except BaseException as exc:
            self._signals.failed.emit(self._id, exc)
        else:
            self._signals.finished.emit(self._id, result)


class AnalysisRunner(QtCore.QObject):
    """Single-worker dispatcher with an app-wide one-in-flight guard."""

    #: (busy, human-readable description) — MainWindow drives the global busy
    #: affordance (override cursor + status bar) from this.
    busy_changed = QtCore.Signal(bool, str)

    def __init__(self, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._pool = QtCore.QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._signals = _Signals(self)
        self._signals.finished.connect(self._deliver_finished)
        self._signals.failed.connect(self._deliver_failed)
        self._job_id = 0
        self._current: Optional[Tuple[int, Callable[[Any], None],
                                      Optional[Callable[[BaseException], None]],
                                      str, Tuple[Any, ...]]] = None
        # Callbacks registered via run_when_idle() while a job was in flight.
        # Drained AFTER the finishing job's own on_done (see _deliver_finished),
        # so a deferred refresh never pre-empts the completing analysis's render.
        self._idle_queue: list = []

    @property
    def is_busy(self) -> bool:
        return self._current is not None

    def try_submit(self, thunk: Callable[[], Any],
                   on_done: Callable[[Any], None],
                   on_fail: Optional[Callable[[BaseException], None]] = None,
                   *, description: str = 'analysis',
                   busy_widgets: Sequence[Any] = ()) -> bool:
        """Dispatch ``thunk`` to the worker thread.

        Returns ``False`` (and does nothing) if a job is already in flight —
        the caller shows its own one-line busy notice. ``busy_widgets`` are
        disabled for the duration (typically the triggering Run button).
        ``on_done(result)`` / ``on_fail(exception)`` run on the main thread;
        if ``on_fail`` is omitted, a failure re-raises on the main thread and
        surfaces like today's uncaught slot exception."""
        if self._current is not None:
            return False
        self._job_id += 1
        widgets = tuple(busy_widgets)
        self._current = (self._job_id, on_done, on_fail, description, widgets)
        for w in widgets:
            w.setEnabled(False)
        self.busy_changed.emit(True, description)
        self._pool.start(_Job(self._job_id, thunk, self._signals))
        return True

    # -- delivery (GUI thread, via queued signals) --------------------------
    def _finish(self, job_id: int):
        """Clear the busy state and return the job record (None if stale).

        Busy state is cleared *before* the callback runs, so a completion
        handler may immediately trigger deferred work, and an exception inside
        it cannot leave the busy affordance stuck. ``busy_changed(False)`` here
        drives only the cursor/status affordance; deferred ``run_when_idle``
        callbacks are drained separately, *after* the job's own on_done."""
        if self._current is None or self._current[0] != job_id:
            # Defensive only: try_submit refuses new work while a job is in flight
            # and _current is cleared solely here, so exactly one job is ever live
            # and the delivered id always matches. This branch is not the staleness
            # mechanism — result staleness from a superseded analysis is handled by
            # each caller's _run_epoch guard, not by dropping the payload here.
            return None
        cur = self._current
        self._current = None
        for w in cur[4]:
            try:
                w.setEnabled(True)
            except RuntimeError:   # widget deleted while the job ran
                pass
        self.busy_changed.emit(False, cur[3])
        return cur

    def _drain_idle(self) -> None:
        """Run the callbacks registered while the just-finished job was in flight.

        Snapshot-and-clear: run exactly the callbacks queued *now*, ONE pass. A
        drained callback may start a new job (via try_submit) — a later callback
        that then re-defers via run_when_idle lands in the fresh queue and fires
        when THAT job finishes, not in this drain. Draining the live queue in a
        `while` loop instead would spin forever in that case (a GUI hang). Each
        callback is isolated so one raising can't drop a sibling (e.g. a precious
        deferred manual-Mw apply); the first error still surfaces."""
        pending, self._idle_queue = self._idle_queue, []
        first_exc = None
        for cb in pending:
            try:
                cb()
            except Exception as exc:          # noqa: BLE001 — isolate siblings
                if first_exc is None:
                    first_exc = exc
        if first_exc is not None:
            raise first_exc

    def _drain_after(self, primary_exc: Optional[BaseException]) -> None:
        """Drain the deferred idle queue after the finishing job's own callback.

        The primary callback (on_done / on_fail) is the *report*: if it already
        raised, a later drain error must not mask it (a plain `finally: drain()`
        would — the finally's exception replaces the original). So we always drain,
        but re-raise the primary exception if there was one; only when the primary
        succeeded does a drain error surface on its own."""
        try:
            self._drain_idle()
        except Exception:                     # noqa: BLE001
            if primary_exc is None:
                raise
            # primary_exc is the real failure; swallow the drain error so it wins.
        if primary_exc is not None:
            raise primary_exc

    @QtCore.Slot(int, object)
    def _deliver_finished(self, job_id: int, result: Any) -> None:
        cur = self._finish(job_id)
        if cur is None:
            return
        primary_exc: Optional[BaseException] = None
        try:
            cur[1](result)          # the finishing job's own on_done runs FIRST
        except BaseException as exc:   # noqa: BLE001 — re-raised after the drain
            primary_exc = exc
        self._drain_after(primary_exc)  # then the deferred refreshes it may depend on

    @QtCore.Slot(int, object)
    def _deliver_failed(self, job_id: int, exc: BaseException) -> None:
        cur = self._finish(job_id)
        if cur is None:
            return
        primary_exc: Optional[BaseException] = None
        try:
            if cur[2] is None:
                raise exc
            cur[2](exc)
        except BaseException as e:      # noqa: BLE001 — re-raised after the drain
            primary_exc = e
        self._drain_after(primary_exc)

    # -- test helper ---------------------------------------------------------
    def wait_for_idle(self, timeout_ms: int = 30000) -> bool:
        """Spin the event loop until idle (offscreen-safe). Test use only."""
        app = QtCore.QCoreApplication.instance()
        deadline = QtCore.QDeadlineTimer(timeout_ms)
        while self._current is not None and not deadline.hasExpired():
            app.processEvents()
            QtCore.QThread.msleep(5)
        return self._current is None


_runner: Optional[AnalysisRunner] = None


def runner() -> AnalysisRunner:
    """The process-wide runner (created lazily, on first use from the GUI
    thread). A singleton so every module and MainWindow share one busy state
    without threading a reference through six constructors."""
    global _runner
    if _runner is None:
        _runner = AnalysisRunner()
    return _runner


def run_when_idle(callback: Callable[[], None]) -> None:
    """Run ``callback`` now if no job is in flight, else once when the current
    job finishes — *after* that job's own on_done (the finishing analysis
    renders before any deferred refresh it might read). A deferred UI-refresh
    hook (e.g. the Cross-Sample refresh on tab switch, or a superseded refit),
    *not* a job queue; queued callbacks fire in registration order."""
    r = runner()
    if r.is_busy:
        r._idle_queue.append(callback)
    else:
        callback()


BUSY_NOTICE = 'Busy — wait for the running analysis to finish.'

#: Shared tooltip for every Run control that now dispatches to the worker.
BACKGROUND_RUN_TOOLTIP = (
    'Runs in the background — the window stays usable and the result '
    'appears when the fit finishes. One analysis runs at a time.')


def busy_notice(widget: QtWidgets.QWidget) -> None:
    """Show the standard 'still running' notice near ``widget``: on the main
    window's status bar when there is one, else as a transient tooltip (shown
    programmatically, so it works even with passive tooltips turned off)."""
    win = widget.window()
    if isinstance(win, QtWidgets.QMainWindow):
        win.statusBar().showMessage(BUSY_NOTICE, 3000)
    else:
        QtWidgets.QToolTip.showText(
            widget.mapToGlobal(QtCore.QPoint(0, 0)), BUSY_NOTICE, widget)


def _reset_for_tests() -> None:
    """Drop the singleton between tests. Asserts the worker is idle first so a
    wedged job fails loudly in the test that caused it."""
    global _runner
    if _runner is not None:
        assert _runner.wait_for_idle(), 'AnalysisRunner still busy at reset'
        _runner._idle_queue.clear()
        _runner = None
