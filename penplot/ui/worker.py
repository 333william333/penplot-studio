"""Background pipeline builder so the window never freezes while tracing."""

from __future__ import annotations

import copy
import traceback

from PySide6.QtCore import QObject, QThread, Signal, Slot

from ..core import autotune, pipeline
from ..core.drawing import SourceResult
from ..core.settings import AppSettings


#: Threads that refused to stop in time.  Keeping a reference alive is the only
#: safe option at shutdown: destroying a running QThread aborts the process,
#: and QThread.terminate() on a thread inside numpy or OpenCV can leave a lock
#: held and deadlock the main thread.  The process is exiting anyway.
_ORPHANED: list = []


def _retire(thread, worker) -> None:
    _ORPHANED.append((thread, worker))


class _BuildWorker(QObject):
    done = Signal(object, int, str)

    def __init__(self) -> None:
        super().__init__()
        # plain attributes set from the GUI thread: a queued slot could not be
        # delivered while a build is running, which is exactly when it matters
        self.abort = False
        self.wanted = 0

    @Slot(object, object, int)
    def build(self, sources, settings: AppSettings, generation: int) -> None:
        # Dragging a slider queues one request per tick.  Building all of them
        # is pure waste - the only one anybody will ever see is the last, and a
        # slow technique would otherwise pin the CPU for seconds after the user
        # has stopped moving.
        if generation != self.wanted:
            self.done.emit(None, generation, "")
            return
        try:
            job = pipeline.build_project(
                sources or {},
                settings,
                settings.library,
                should_cancel=lambda: self.abort or generation != self.wanted,
            )
            self.done.emit(job, generation, "")
        except pipeline.Cancelled:
            self.done.emit(None, generation, "")
        except Exception:  # pragma: no cover - defensive
            self.done.emit(None, generation, traceback.format_exc(limit=4))


class PipelineRunner(QObject):
    """Runs `build_plot` off the GUI thread and drops stale results."""

    finished = Signal(object)       # PlotJob
    failed = Signal(str)
    cancelled = Signal()
    busy_changed = Signal(bool)

    _request = Signal(object, object, int)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self.thread = QThread()
        self.thread.setObjectName("pipeline")
        self.worker = _BuildWorker()
        self.worker.moveToThread(self.thread)
        self._request.connect(self.worker.build)
        self.worker.done.connect(self._on_done)
        self.thread.start()

        self._generation = 0
        self._pending = 0

    def submit(self, sources, settings: AppSettings, draft: bool = False) -> None:
        self._generation += 1
        self._pending += 1
        # written before the request is queued so a build already running sees
        # it at its next cancellation check and gives up
        self.worker.wanted = self._generation
        if self._pending == 1:
            self.busy_changed.emit(True)
        # the GUI keeps editing `settings`, so the worker gets its own copy
        payload = copy.deepcopy(settings)
        if draft:
            # A draft is for the eye, not for the pen: drop the working
            # resolution and the tidying passes that cost the most and show the
            # least.  The full-quality pass follows a moment later.
            for item in payload.items:
                item.style.detail = max(int(item.style.detail * 0.45), 260)
            payload.optimize.tidy_tour = False
            payload.optimize.reorder = False
        self._request.emit(dict(sources or {}), payload, self._generation)

    def _on_done(self, job, generation: int, error: str) -> None:
        self._pending = max(self._pending - 1, 0)
        if self._pending == 0:
            self.busy_changed.emit(False)
        if generation != self._generation:
            return  # a newer request is already on its way
        if error:
            self.failed.emit(error)
        elif job is not None:
            self.finished.emit(job)
        else:
            self.cancelled.emit()

    @property
    def pending(self) -> int:
        return self._pending

    def shutdown(self) -> None:
        """Stop the build before the QThread object is destroyed.

        Without the abort flag a long render outlives wait() and Qt kills the
        whole process with "QThread: Destroyed while thread is still running".
        """
        if not self.thread.isRunning():
            return
        self.worker.abort = True
        self.thread.quit()
        if not self.thread.wait(4000):
            _retire(self.thread, self.worker)


class _TuneWorker(QObject):
    done = Signal(object, object, str)

    def __init__(self) -> None:
        super().__init__()
        self.abort = False

    @Slot(object, object, float, bool)
    def run(self, source, settings: AppSettings, minutes: float, choose: bool) -> None:
        if self.abort:
            self.done.emit(None, None, "")
            return
        try:
            result = autotune.auto_tune(
                settings,
                source,
                settings.library,
                target_minutes=minutes,
                choose_technique=choose,
            )
            self.done.emit(result, settings, "")
        except Exception:  # pragma: no cover - defensive
            self.done.emit(None, None, traceback.format_exc(limit=4))


class AutoTuneRunner(QObject):
    """Runs the closed-loop auto setup without blocking the window."""

    finished = Signal(object, object)   # TuneResult, tuned settings copy
    failed = Signal(str)
    busy_changed = Signal(bool)

    _request = Signal(object, object, float, bool)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self.thread = QThread()
        self.thread.setObjectName("autotune")
        self.worker = _TuneWorker()
        self.worker.moveToThread(self.thread)
        self._request.connect(self.worker.run)
        self.worker.done.connect(self._on_done)
        self.thread.start()
        self._busy = False

    @property
    def busy(self) -> bool:
        return self._busy

    def submit(self, source: SourceResult | None, settings: AppSettings, minutes: float, choose: bool) -> bool:
        if self._busy or source is None:
            return False
        self._busy = True
        self.busy_changed.emit(True)
        self._request.emit(source, copy.deepcopy(settings), float(minutes), bool(choose))
        return True

    def _on_done(self, result, settings, error: str) -> None:
        self._busy = False
        self.busy_changed.emit(False)
        if error:
            self.failed.emit(error)
        else:
            self.finished.emit(result, settings)

    def shutdown(self) -> None:
        if not self.thread.isRunning():
            return
        self.worker.abort = True
        self.thread.quit()
        if not self.thread.wait(4000):
            _retire(self.thread, self.worker)
