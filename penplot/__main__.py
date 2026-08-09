"""Entry point: python -m penplot"""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from .core.settings import AppSettings
from .ui.main_window import MainWindow
from .ui import theme
from .ui.theme import apply_theme


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    app = QApplication(argv)
    app.setApplicationName("PenPlot Studio")
    app.setApplicationDisplayName("PenPlot Studio")
    app.setOrganizationName("PenPlot")
    apply_theme(app)

    settings = AppSettings.load()
    if abs(settings.ui_scale - 1.0) > 1e-6:
        theme.set_scale(app, settings.ui_scale)
    window = MainWindow(settings)
    window.show()

    # a file passed on the command line is loaded straight away
    for argument in argv[1:]:
        if argument and not argument.startswith("-"):
            window.source_panel.load_any(argument)
            break

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
