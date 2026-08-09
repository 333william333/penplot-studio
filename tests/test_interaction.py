"""Direct-manipulation tests: dragging and scaling the artwork on the bed.

These go through the real widget event handlers, so a regression in the
preview, the layout maths or the render scheduling shows up here instead of
in front of the printer.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QEventLoop, QPointF, Qt, QTimer  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from penplot.core.settings import AppSettings  # noqa: E402
from penplot.ui.main_window import MainWindow  # noqa: E402
from penplot.ui.theme import apply_theme  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'} {name} {detail}")
    if not condition:
        FAILURES.append(name)


def settle(ms: int = 900) -> None:
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def _event(kind, point: QPointF, buttons=Qt.LeftButton) -> QMouseEvent:
    return QMouseEvent(kind, point, point, Qt.LeftButton, buttons, Qt.NoModifier)


def drag(canvas, start: QPointF, end: QPointF, steps: int = 6) -> None:
    canvas.mousePressEvent(_event(QEvent.MouseButtonPress, start))
    for i in range(1, steps + 1):
        t = i / steps
        point = QPointF(start.x() + (end.x() - start.x()) * t, start.y() + (end.y() - start.y()) * t)
        canvas.mouseMoveEvent(_event(QEvent.MouseMove, point))
    canvas.mouseReleaseEvent(_event(QEvent.MouseButtonRelease, end, Qt.NoButton))


def centre_of(job):
    lo_x, lo_y, hi_x, hi_y = job.stats.bounds
    return ((lo_x + hi_x) / 2.0, (lo_y + hi_y) / 2.0)


def size_of(job):
    lo_x, lo_y, hi_x, hi_y = job.stats.bounds
    return (hi_x - lo_x, hi_y - lo_y)


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv[:1])
    apply_theme(app)

    settings = AppSettings()
    settings.style.detail = 500
    window = MainWindow(settings)
    window.resize(1500, 950)
    window.show()
    settle(400)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    window.source_panel.load_image(os.path.join(root, "samples", "portrait.png"))
    settle(1600)

    canvas = window.preview
    check("something was rendered", window.job is not None and not window.job.is_empty)
    check("preview starts up to date", window._up_to_date)

    # ---------------- drag to move ----------------
    print("\ndragging the artwork")
    before = centre_of(window.job)
    zoom = canvas._zoom
    start = canvas._to_screen(*before)
    dx_px, dy_px = 80.0, -50.0
    drag(canvas, start, QPointF(start.x() + dx_px, start.y() + dy_px))
    settle(1400)

    after = centre_of(window.job)
    expected = (before[0] + dx_px / zoom, before[1] - dy_px / zoom)
    moved_x = abs(after[0] - expected[0])
    moved_y = abs(after[1] - expected[1])
    check(
        f"the artwork actually moved ({before[0]:.1f},{before[1]:.1f} -> {after[0]:.1f},{after[1]:.1f} mm)",
        moved_x < 1.0 and moved_y < 1.0,
        f"expected {expected[0]:.1f},{expected[1]:.1f}",
    )
    check("the preview is not left waiting for anything", window._up_to_date)
    check(
        "the layout offsets were updated",
        abs(settings.layout.offset_x) > 1.0 and abs(settings.layout.offset_y) > 1.0,
        f"{settings.layout.offset_x:.1f}, {settings.layout.offset_y:.1f}",
    )

    # ---------------- arrow-key nudge ----------------
    print("\nnudging with the arrow keys")
    before = centre_of(window.job)
    window._on_preview_move(1.0, 0.0)
    settle(1200)
    after = centre_of(window.job)
    check(
        f"1 mm nudge moves 1 mm ({after[0] - before[0]:+.2f} mm)",
        abs((after[0] - before[0]) - 1.0) < 0.35,
    )

    # ---------------- drag a corner handle ----------------
    print("\nscaling with a corner handle")
    before_size = size_of(window.job)
    rect = canvas._artwork_rect()
    corner = rect.bottomRight()
    centre_px = rect.center()
    # move the corner 40 % further away from the centre
    target = QPointF(
        centre_px.x() + (corner.x() - centre_px.x()) * 1.4,
        centre_px.y() + (corner.y() - centre_px.y()) * 1.4,
    )
    drag(canvas, corner, target)
    settle(1600)
    after_size = size_of(window.job)
    ratio = after_size[0] / max(before_size[0], 1e-6)
    check(
        f"the artwork grew ({before_size[0]:.0f} -> {after_size[0]:.0f} mm, ×{ratio:.2f})",
        1.15 < ratio < 1.75,
    )
    check("size mode switched to exact", settings.layout.mode == "size")

    # ---------------- settings changes render on their own ----------------
    print("\nsettings apply without a button")
    paths_before = window.job.stats.path_count
    settings.style.technique_params()["coverage"] = 0.4
    window.source_panel.settings_changed.emit()
    settle(2500)
    check(
        f"a style change redraws by itself ({paths_before} -> {window.job.stats.path_count} strokes)",
        window.job.stats.path_count != paths_before,
    )
    check("and it settles up to date", window._up_to_date)

    print("\npen width drives the drawing")
    length_before = window.job.stats.draw_length
    settings.library[0].width = 1.5
    window.pens_panel.changed.emit()
    settle(2500)
    check(
        f"a wider pen redraws by itself ({length_before/1000:.1f} -> {window.job.stats.draw_length/1000:.1f} m)",
        window.job.stats.draw_length < length_before * 0.6,
    )

    print("\na fast drag does not queue a build per tick")
    import penplot.core.pipeline as _pipeline
    builds = {"n": 0}
    original = _pipeline.build_project
    def counted(*args, **kwargs):
        builds["n"] += 1
        return original(*args, **kwargs)
    import penplot.ui.worker as _worker
    _worker.pipeline.build_project = counted
    for tick in range(12):
        settings.style.technique_params()["coverage"] = 0.4 + tick * 0.02
        window.source_panel.settings_changed.emit()
        settle(30)
    settle(2500)
    _worker.pipeline.build_project = original
    check(f"12 slider ticks caused {builds['n']} build(s)", builds["n"] <= 3)

    window.close()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failing checks: {FAILURES}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
