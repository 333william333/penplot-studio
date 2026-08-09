"""Right column: where on the bed the drawing ends up, and how big it is."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ...core.settings import AppSettings
from ..widgets import Binder, Card, FieldRow


class LayoutPanel(QWidget):
    changed = Signal()

    def __init__(self, settings: AppSettings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.binder = Binder(self._emit, self, context=settings)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        card = Card("PLACEMENT")
        self.card = card
        card.add_reset(self.reset, "Centre it on the bed at its natural size")
        layout_settings = settings.layout

        self.mode_row = self.binder.segmented(
            "layout",
            "mode",
            [("fit", "Fit to bed"), ("size", "Exact size"), ("scale", "Scale"), ("asis", "As drawn")],
            label="Size mode",
        )
        card.add(self.mode_row)

        self.width_row = self.binder.slider(
            "layout", "width", 5.0, 400.0, label="Width", decimals=1, step=1.0, suffix="mm"
        )
        self.height_row = self.binder.slider(
            "layout", "height", 5.0, 400.0, label="Height", decimals=1, step=1.0, suffix="mm"
        )
        self.aspect_row = self.binder.check("layout", "keep_aspect", "Keep proportions")
        self.scale_row = self.binder.slider(
            "layout", "scale_percent", 1.0, 800.0, label="Scale", decimals=0, step=5, suffix="%"
        )
        card.add(self.width_row)
        card.add(self.height_row)
        card.add(self.aspect_row)
        card.add(self.scale_row)

        card.add(self.binder.slider("layout", "margin", 0.0, 60.0, label="Bed margin", decimals=1, step=1.0, suffix="mm"))

        card.add_heading("Position")
        self.centre_row = self.binder.check("layout", "center", "Centre on the bed")
        card.add(self.centre_row)
        self.offset_x_row = self.binder.slider(
            "layout", "offset_x", -200.0, 200.0, label="Offset X", decimals=1, step=1.0, suffix="mm"
        )
        self.offset_y_row = self.binder.slider(
            "layout", "offset_y", -200.0, 200.0, label="Offset Y", decimals=1, step=1.0, suffix="mm"
        )
        card.add(self.offset_x_row)
        card.add(self.offset_y_row)

        card.add_heading("Orientation")
        card.add(self.binder.slider("layout", "rotation", -180.0, 180.0, label="Rotation", decimals=0, step=5, suffix="°"))

        quick = QHBoxLayout()
        quick.setSpacing(6)
        for text, angle in (("↺ 90°", -90.0), ("↻ 90°", 90.0), ("180°", 180.0)):
            button = QPushButton(text)
            button.clicked.connect(lambda _checked=False, a=angle: self._rotate(a))
            quick.addWidget(button)
        card.add_layout(quick)

        mirrors = QHBoxLayout()
        mirrors.addWidget(self.binder.check("layout", "mirror_x", "Mirror X"))
        mirrors.addWidget(self.binder.check("layout", "mirror_y", "Mirror Y"))
        card.add_layout(mirrors)

        actions = QHBoxLayout()
        actions.setSpacing(6)
        fit_button = QPushButton("Fit to bed")
        fit_button.clicked.connect(self._fit)
        centre_button = QPushButton("Centre")
        centre_button.clicked.connect(self._centre)
        actions.addWidget(fit_button)
        actions.addWidget(centre_button)
        card.add_layout(actions)

        self.info = QLabel("")
        self.info.setObjectName("Hint")
        self.info.setWordWrap(True)
        card.add(self.info)

        layout.addWidget(card)
        layout.addStretch(1)
        self._sync()

    # ------------------------------------------------------------------
    def _emit(self) -> None:
        self._sync()
        self.changed.emit()

    def _sync(self) -> None:
        # "As drawn" only means anything for hand-drawn geometry, and a mode the
        # segmented control does not know would leave it silently showing the
        # wrong thing - so repair it here rather than lie about it.
        drawn_layer = self.settings.item.kind == "shapes"
        if self.settings.layout.mode == "asis" and not drawn_layer:
            self.settings.layout.mode = "fit"
            self.binder.refresh()
        segments = self.mode_row.control if hasattr(self.mode_row, "control") else self.mode_row
        for key, button in segments.buttons().items():
            if key == "asis":
                button.setVisible(drawn_layer)
        mode = self.settings.layout.mode
        self.width_row.setVisible(mode == "size")
        self.height_row.setVisible(mode == "size" and not self.settings.layout.keep_aspect)
        self.aspect_row.setVisible(mode == "size")
        self.scale_row.setVisible(mode == "scale")
        centred = self.settings.layout.center
        self.offset_x_row.label.setText("Offset X" if centred else "Left edge X")
        self.offset_y_row.label.setText("Offset Y" if centred else "Bottom edge Y")

    def _rotate(self, delta: float) -> None:
        value = self.settings.layout.rotation + delta
        while value > 180.0:
            value -= 360.0
        while value < -180.0:
            value += 360.0
        self.settings.layout.rotation = value
        self.binder.refresh()
        self._emit()

    def _fit(self) -> None:
        self.settings.layout.mode = "fit"
        self.settings.layout.center = True
        self.settings.layout.offset_x = 0.0
        self.settings.layout.offset_y = 0.0
        self.binder.refresh()
        self._emit()

    def _centre(self) -> None:
        self.settings.layout.center = True
        self.settings.layout.offset_x = 0.0
        self.settings.layout.offset_y = 0.0
        self.binder.refresh()
        self._emit()

    def reset(self) -> None:
        """Put the artwork back in the middle of the bed at a sensible size."""
        from ...core.settings import LayoutSettings

        keep_margin = self.settings.layout.margin
        drawn = self.settings.item.kind == "shapes"
        fresh = LayoutSettings()
        fresh.margin = keep_margin
        if drawn:
            fresh.mode = "asis"
            fresh.center = False
        for slot in fresh.__dataclass_fields__:
            setattr(self.settings.layout, slot, getattr(fresh, slot))
        self.binder.refresh()
        self._emit()

    def update_info(self, job) -> None:
        if job is None or job.is_empty:
            self.info.setText("")
            return
        bounds = job.stats.bounds
        if not bounds:
            self.info.setText("")
            return
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        native = ""
        if job.native_size:
            native = f"   ·   original {job.native_size[0]:.0f} × {job.native_size[1]:.0f} mm"
        self.info.setText(f"On the bed: {width:.1f} × {height:.1f} mm{native}")

    def refresh(self) -> None:
        self.binder.refresh()
        self._sync()
