"""
gui/worker.py
=============

Background dispatch for the heavy controller calls (threading enablement).

The controller stays Qt-free: widgets build a
zero-argument *thunk* on the main thread that binds a controller method to its
already-read arguments, then hand it to :func:`runner`'s ``try_submit``. The
thunk runs on a single pooled worker thread; ``on_done(result)`` /
``on_fail(exception)`` are invoked back on the GUI thread via queued signal
delivery, so widget code never touches Qt objects from the worker.

Every Run dispatches through its own :class:`RunSlot`, which meets a press made
while the worker is busy in one of two ways, chosen per Run: **queued**, the press's
target taking the slot's one pending place (the latest press wins) and retried once
the worker frees; or **refused**, nothing dispatched or retried. Either way the
press is told through the slot's ``notice`` hook, on the surface the Run's tab uses.
The one dispatch that is not a Run's, the Utilities tab's long-trace diagnostics
(``UtilitiesModule._ensure_focus_diag``), calls ``try_submit`` directly: each busy
call there queues its own continuation, which one pending place would not keep.

Deliberate non-features (spec boundaries):

- **One job in flight, app-wide.** ``try_submit`` refuses while busy — this is
  the global re-entry guard: controller ``run_*`` methods mutate
  shared state, so two concurrent runs are never allowed, whichever tabs they
  come from. No job queue, no priorities: a queued press waits in its own slot.
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
        except BaseException as exc:  # noqa: BLE001 — thread boundary, re-emitted
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
        except Exception:                     # noqa: BLE001 — a drain error must not mask the primary
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
        """Spin the event loop until idle (offscreen-safe). Test use only.

        IDLE MEANS BOTH CONDITIONS, and waiting on only the first is a real gap.
        ``_current`` is cleared before ``_drain_idle`` runs, and a drained
        callback may itself start a new job (see that method) -- so a caller that
        waited on ``_current`` alone could return at the moment the queue was
        about to dispatch, and hand the *next* test a busy runner. Both states
        are checked here rather than leaving each caller to remember the second,
        which is one predicate instead of one-per-call-site.

        No hang risk: ``_idle_queue`` is only appended to while a job is live and
        is snapshot-and-cleared in one pass, so at true rest it is empty; and the
        deadline bounds the wait either way, returning False rather than
        spinning.
        """
        app = QtCore.QCoreApplication.instance()
        deadline = QtCore.QDeadlineTimer(timeout_ms)
        while ((self._current is not None or self._idle_queue)
                and not deadline.hasExpired()):
            # Deliver queued cross-thread signals directly as well as via the
            # dispatcher: on Windows the dispatcher's posted-event wake-up can
            # be lost, after which processEvents() alone finds no pending
            # messages and a queued finished-signal delivery stalls for tens of
            # seconds. sendPostedEvents() drains the calling thread's posted-
            # event queue synchronously, wake-up or no wake-up.
            QtCore.QCoreApplication.sendPostedEvents()
            app.processEvents()
            QtCore.QThread.msleep(5)
        return self._current is None and not self._idle_queue


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


class RunSlot:
    """The dispatch state of one Run button: its staleness epoch, and the target of
    the latest press deferred behind a busy worker.

    ``submit`` dispatches a run through :func:`runner`'s ``try_submit``, which stays
    the single-flight gate. A press while the worker is busy never touches the epoch
    (that would drop the run in flight with nothing to replace it): with ``queue``
    the press's ``target`` takes the pending slot, the latest press winning, and one
    retry waits for idle; without it the press is refused. ``notice`` reports the busy
    press either way. The retry re-checks the target with ``available`` and, while it
    can run, hands it to ``rerun``; a slot built without ``rerun`` never queues.
    ``hold`` is that busy arm alone, for a Run that must turn a busy press away before
    it reads its inputs or asks the user anything.

    A dispatch with ``paint`` bumps the epoch and carries it as the run's token; with
    ``bump=False`` it carries the epoch without moving it; one without ``paint`` (a
    deferred Run whose target is no longer on screen) carries ``None``.
    ``done(result, token)`` and ``fail(exc, token)`` always run, so a run lands in its
    own slot whatever is shown; the caller paints only while ``current(token)``.
    Anything else that makes an in-flight run's display stale calls ``bump``.

    ``before_delivery``, when given, runs first in every delivery to the caller: before
    ``done`` and ``fail``, and before a retry hands its target to ``rerun`` (never on a
    retry that reruns nothing). A DLS sub-tab passes its ``flush_view`` here, so a
    repaint it owes runs before the landing paints. ``done``, ``fail`` and ``rerun``
    still run when the hook raises.
    """

    def __init__(self, *, available: Optional[Callable[[Any], bool]] = None,
                 rerun: Optional[Callable[[Any], None]] = None,
                 before_delivery: Optional[Callable[[], None]] = None) -> None:
        self.epoch = 0
        self.pending: Any = False
        self._available = available
        self._rerun = rerun
        self._before_delivery = before_delivery

    def bump(self) -> None:
        """Make every run in flight stale for display."""
        self.epoch += 1

    def current(self, token: Optional[int]) -> bool:
        """Whether a run dispatched with ``token`` may still paint."""
        return token is not None and token == self.epoch

    def hold(self, *, target: Any, queue: bool,
             notice: Optional[Callable[[], None]] = None) -> bool:
        """While the worker is busy, queue or refuse the press as ``submit`` does and
        return True; return False, doing nothing, while it is idle."""
        if not runner().is_busy:
            return False
        if queue:
            if self._rerun is None:
                raise ValueError('a RunSlot without a rerun hook cannot queue a press')
            first = not self.pending
            self.pending = target
            if first:
                run_when_idle(self.flush)
        if notice is not None:
            notice()
        return True

    def submit(self, thunk: Callable[[], Any],
               done: Callable[[Any, Optional[int]], None],
               fail: Callable[[BaseException, Optional[int]], None],
               *, target: Any, queue: bool, busy_widgets: Sequence[Any] = (),
               notice: Optional[Callable[[], None]] = None,
               description: str = 'analysis', paint: bool = True,
               bump: bool = True) -> bool:
        """Dispatch ``thunk``; returns whether it was dispatched now."""
        if self.hold(target=target, queue=queue, notice=notice):
            return False
        token: Optional[int] = None
        if paint:
            if bump:
                self.epoch += 1
            token = self.epoch
        hook = self._before_delivery

        def on_done(res: Any) -> None:
            try:
                if hook is not None:
                    hook()
            finally:
                done(res, token)

        def on_fail(exc: BaseException) -> None:
            try:
                if hook is not None:
                    hook()
            finally:
                fail(exc, token)

        return runner().try_submit(thunk, on_done, on_fail,
                                   description=description, busy_widgets=busy_widgets)

    def flush(self) -> None:
        """The deferred retry: empty the pending slot, and rerun its target while
        ``available`` says the target can still run."""
        target, self.pending = self.pending, False
        if (target and self._rerun is not None
                and (self._available is None or self._available(target))):
            try:
                if self._before_delivery is not None:
                    self._before_delivery()
            finally:
                self._rerun(target)


BUSY_NOTICE ='Busy — wait for the running analysis to finish.'

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
        # Collect the garbage left so far while no job runs, rather than leave it to a
        # later collection that may fall inside another runner's job.
        import gc
        gc.collect()
