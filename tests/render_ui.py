"""Render the window offscreen to PNGs so the layout can be inspected without a display.

Usage:  QT_QPA_PLATFORM=offscreen .venv/bin/python tests/render_ui.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from penplot.core.settings import AppSettings  # noqa: E402
from penplot.ui.main_window import MainWindow  # noqa: E402
from penplot.ui.theme import apply_theme  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "screens")


def settle(app: QApplication, ms: int = 900) -> None:
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()
    app.processEvents()


def shot(window: MainWindow, name: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    path = os.path.abspath(os.path.join(OUT, f"{name}.png"))
    window.grab().save(path)
    print("wrote", path)


def main() -> int:
    app = QApplication(sys.argv[:1])
    apply_theme(app)

    settings = AppSettings()
    window = MainWindow(settings)
    window.resize(1560, 980)
    window.show()
    settle(app, 400)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    portrait = os.path.join(root, "samples", "portrait.png")
    logo = os.path.join(root, "samples", "logo.png")
    pdf = os.path.join(root, "samples", "test.pdf")

    shot(window, "01-empty")

    window.source_panel.load_image(portrait)
    settle(app, 1400)
    shot(window, "02-image-hatch")

    settings.style.technique = "stipple"
    window.source_panel.refresh()
    window.render_now()
    settle(app, 1800)
    shot(window, "03-image-stipple")

    settings.style.technique = "spiral"
    window.source_panel.refresh()
    window.render_now()
    settle(app, 1600)
    shot(window, "04-image-spiral")

    settings.style.technique = "sketch"
    window.source_panel.refresh()
    window.render_now()
    settle(app, 1400)
    shot(window, "05-image-edges")

    # multi-pen colour work
    window.source_panel.load_image(logo)
    settings.library.apply_palette("Primary colours (5 pens)")
    settings.style.separation = "palette"
    settings.style.technique = "silhouette"
    settings.style.technique_params()["fill"] = 0.9
    window.pens_panel.rebuild()
    window.source_panel.refresh()
    window.render_now()
    settle(app, 1800)
    window.show_panel("Pens")
    settle(app, 300)
    shot(window, "06-logo-multipen")

    # text
    settings.source_kind = "text"
    settings.text.text = "Hej Ender 3!\nÅÄÖ 0123"
    settings.text.size_mm = 26.0
    settings.library.apply_palette("Single black pen")
    settings.style.separation = "mono"
    window.pens_panel.rebuild()
    window.source_panel.refresh()
    window.source_panel.rebuild_source()
    settle(app, 1200)
    window.show_panel("Technique")
    shot(window, "07-text")

    # pdf
    window.source_panel.load_pdf(pdf)
    settle(app, 1600)
    shot(window, "08-pdf-vector")

    # monitor stage
    window._set_stage(1)
    settle(app, 500)
    shot(window, "09-monitor")

    window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
