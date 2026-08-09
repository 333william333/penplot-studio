"""Split a picture into spray stencils and turn each sheet into a cut layer.

The geometry lives in :mod:`penplot.core.stencil`; this is the window round it.
It shows every sheet, says what the stack would look like sprayed, and when you
accept it drops one cut layer per sheet into the project - hidden except the
first, because you cut them one sheet of card at a time.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..core import stencil as stencil_core
from ..core.pens import CUTTING_TOOLS
from ..core.settings import AppSettings
from .widgets import Card, FieldRow, SliderSpin, hint_label

THUMB = 96


def _to_pixmap(image: np.ndarray, width: int) -> QPixmap:
    """Float RGB 0..1 -> QPixmap, scaled to `width`."""
    array = np.ascontiguousarray((np.clip(image, 0.0, 1.0) * 255).astype(np.uint8))
    height, source_width = array.shape[:2]
    qimage = QImage(array.data, source_width, height, source_width * 3, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimage).scaledToWidth(width, Qt.SmoothTransformation)


class _SheetTile(QFrame):
    chosen = Signal(int)

    def __init__(self, index: int, parent=None):
        super().__init__(parent)
        self.index = index
        self.selected = False
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(QSize(THUMB + 12, THUMB + 30))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 3)
        layout.setSpacing(2)
        self.image = QLabel()
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setFixedHeight(THUMB)
        self.caption = QLabel("")
        self.caption.setAlignment(Qt.AlignCenter)
        self.caption.setObjectName("Hint")
        layout.addWidget(self.image)
        layout.addWidget(self.caption)
        self._restyle()

    def set_sheet(self, sheet) -> None:
        self.image.setPixmap(_to_pixmap(stencil_core.render_sheet(sheet), THUMB))
        self.caption.setText(f"Sheet {sheet.index + 1}")
        self.setToolTip(f"{sheet.label} - {len(sheet.cuts)} cuts, {sheet.cut_length:.0f} mm")

    def set_selected(self, value: bool) -> None:
        self.selected = value
        self._restyle()

    def _restyle(self) -> None:
        border = "#2C6FDB" if self.selected else "#DDE1E6"
        self.setStyleSheet(
            f"QFrame {{ background: #FFFFFF; border: {2 if self.selected else 1}px solid {border};"
            " border-radius: 6px; }"
        )

    def mousePressEvent(self, event) -> None:  # pragma: no cover - UI glue
        self.chosen.emit(self.index)


class StencilDialog(QDialog):
    """Preview the sheets, then hand them back as cut layers."""

    def __init__(self, rgb: np.ndarray, settings: AppSettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Make stencils")
        self.setMinimumSize(1000, 660)
        self.rgb = rgb
        self.settings = settings
        self.config = stencil_core.StencilSettings()
        # Sheets only line up if there is something on every sheet to line them
        # up by, and the border is that something: it is cut identically on all
        # of them, so you stack them on the same rectangle.
        self.config.frame = 6.0
        self.sheets: list = []
        self.selected = 0
        self.tiles: list[_SheetTile] = []

        bed = min(settings.machine.bed_x, settings.machine.bed_y)
        self.sheet_width = float(min(max(bed - 40.0, 40.0), 200.0))

        root = QHBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 12)
        root.setSpacing(12)

        # ---- left: the settings ------------------------------------------
        side = QWidget()
        side.setFixedWidth(340)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(10)

        sheet_card = Card("SHEET")
        width_row = FieldRow(
            "Sheet width",
            self._spin(20.0, max(bed, 40.0), self.sheet_width, 1, 5.0, " mm", self._set_width),
            "How wide the stencil is on the bed - the height follows the picture",
        )
        sheet_card.add(width_row)
        self.size_label = hint_label("")
        sheet_card.add(self.size_label)

        self.tool_box = QComboBox()
        for index, pen in enumerate(settings.library):
            if pen.cuts:
                self.tool_box.addItem(f"{pen.name} (pen {index + 1})", ("existing", index))
        for name in CUTTING_TOOLS:
            self.tool_box.addItem(f"Add: {name}", ("new", name))
        blade = settings.library.first_cutter()
        if blade is None:
            self.tool_box.setCurrentIndex(0)
        sheet_card.add(FieldRow("Cut with", self.tool_box, "The tool in the holder when these layers are drawn"))
        self.tool_note = hint_label("")
        sheet_card.add(self.tool_note)
        self.tool_box.currentIndexChanged.connect(self._update_tool_note)
        side_layout.addWidget(sheet_card)

        params_card = Card("SEPARATION")
        for param in stencil_core.STENCIL_PARAMS:
            params_card.add(self._param_row(param))
        params_card.add(hint_label(
            "The border is cut the same on every sheet - line the sheets up on it and "
            "the colours land where they should. Set it to 0 only if you have another "
            "way of registering them."
        ))
        side_layout.addWidget(params_card)
        side_layout.addStretch(1)

        side_scroll = QScrollArea()
        side_scroll.setWidgetResizable(True)
        side_scroll.setFrameShape(QFrame.NoFrame)
        side_scroll.setWidget(side)
        side_scroll.setFixedWidth(362)
        root.addWidget(side_scroll)

        # ---- right: the preview ------------------------------------------
        right = QVBoxLayout()
        right.setSpacing(8)

        self.big = QLabel()
        self.big.setAlignment(Qt.AlignCenter)
        self.big.setMinimumSize(460, 380)
        self.big.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.big.setStyleSheet("background: #F2F4F6; border: 1px solid #DDE1E6; border-radius: 6px;")
        right.addWidget(self.big, 1)

        self.strip = QHBoxLayout()
        self.strip.setSpacing(6)
        self.strip.addStretch(1)
        strip_host = QWidget()
        strip_host.setLayout(self.strip)
        strip_host.setFixedHeight(THUMB + 40)
        right.addWidget(strip_host)

        self.show_composite = QCheckBox("Show what the finished spray would look like")
        self.show_composite.toggled.connect(lambda _: self._update_big())
        right.addWidget(self.show_composite)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setObjectName("Hint")
        self.status.setMinimumHeight(46)
        right.addWidget(self.status)

        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.create_button = buttons.addButton("Create cut layers", QDialogButtonBox.AcceptRole)
        self.create_button.setObjectName("Primary")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        right.addWidget(buttons)
        root.addLayout(right, 1)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(220)
        self._timer.timeout.connect(self._rebuild)

        self._update_tool_note()
        self._rebuild()

    # ------------------------------------------------------------------
    def _spin(self, low, high, value, decimals, step, suffix, slot) -> SliderSpin:
        control = SliderSpin(low, high, decimals=decimals, step=step, suffix=suffix)
        control.setValue(value)
        control.valueChanged.connect(slot)
        return control

    def _param_row(self, param) -> QWidget:
        current = getattr(self.config, param.key, param.default)

        def store(value) -> None:
            setattr(self.config, param.key, value)
            self._schedule()

        if param.kind == "choice":
            control = QComboBox()
            for key, label in (param.choices or {}).items():
                control.addItem(label, key)
            control.setCurrentIndex(max(control.findData(current), 0))
            control.currentIndexChanged.connect(lambda i: store(control.itemData(i)))
            return FieldRow(param.label, control, param.hint)

        control = SliderSpin(
            param.minimum, param.maximum, decimals=param.decimals, step=param.step, suffix=param.suffix
        )
        control.setValue(float(current))
        control.valueChanged.connect(
            lambda value: store(int(round(value)) if param.decimals == 0 else float(value))
        )
        return FieldRow(param.label, control, param.hint)

    def _set_width(self, value: float) -> None:
        self.sheet_width = float(value)
        self._schedule()

    def _schedule(self) -> None:
        self.status.setText("Working out where the bridges go...")
        self._timer.start()

    # ------------------------------------------------------------------
    @property
    def px_per_mm(self) -> float:
        return max(self.rgb.shape[1] / max(self.sheet_width, 1.0), 0.2)

    def _rebuild(self) -> None:
        try:
            self.sheets = stencil_core.build_stencils(
                self.rgb, self.config, px_per_mm=self.px_per_mm, library=self.settings.library
            )
        except Exception as exc:  # pragma: no cover - defensive
            self.sheets = []
            self.status.setText(f"Could not build the stencils: {exc}")

        while len(self.tiles) > len(self.sheets):
            tile = self.tiles.pop()
            self.strip.removeWidget(tile)
            tile.deleteLater()
        while len(self.tiles) < len(self.sheets):
            tile = _SheetTile(len(self.tiles))
            tile.chosen.connect(self._select)
            self.strip.insertWidget(self.strip.count() - 1, tile)
            self.tiles.append(tile)
        for tile, sheet in zip(self.tiles, self.sheets):
            tile.set_sheet(sheet)

        self.selected = min(self.selected, max(len(self.sheets) - 1, 0))
        self._select(self.selected)
        self._update_status()
        self.create_button.setEnabled(bool(self.sheets))

    def _select(self, index: int) -> None:
        self.selected = index
        for tile in self.tiles:
            tile.set_selected(tile.index == index)
        self._update_big()

    def _update_big(self) -> None:
        if not self.sheets:
            self.big.setPixmap(QPixmap())
            return
        if self.show_composite.isChecked():
            image = stencil_core.render_composite(self.sheets)
        else:
            image = stencil_core.render_sheet(self.sheets[self.selected], scale=2)
        width = max(self.big.width() - 16, 200)
        height = max(self.big.height() - 16, 160)
        pixmap = _to_pixmap(image, width)
        if pixmap.height() > height:
            pixmap = pixmap.scaledToHeight(height, Qt.SmoothTransformation)
        self.big.setPixmap(pixmap)

    def _update_status(self) -> None:
        if not self.sheets:
            self.status.setText("Nothing to cut - try fewer sheets or a smaller minimum feature.")
            return
        height_mm = self.rgb.shape[0] / self.px_per_mm
        self.size_label.setText(f"{self.sheet_width:.0f} x {height_mm:.0f} mm on the bed")
        total = sum(sheet.cut_length for sheet in self.sheets) / 1000.0
        bridges = sum(sheet.bridges + sheet.tabs for sheet in self.sheets)
        parts = [
            f"{len(self.sheets)} sheets  ·  {total:.1f} m of cutting  ·  {bridges} bridges",
        ]
        unsafe = [s for s in self.sheets if not s.safe]
        if unsafe:
            parts.append(
                "Sheet "
                + ", ".join(str(s.index + 1) for s in unsafe)
                + " has a piece that would fall out. Widen the bridges or the smallest feature."
            )
        seen: list[str] = []
        for sheet in self.sheets:
            for warning in sheet.warnings:
                if warning not in seen:
                    seen.append(warning)
        if seen:
            parts.append("  ·  ".join(seen[:3]))
        self.status.setText("\n".join(parts))

    def _update_tool_note(self) -> None:
        data = self.tool_box.currentData()
        if not data:
            self.tool_note.setText("")
            return
        if data[0] == "existing":
            pen = self.settings.library[data[1]]
            passes = f"{pen.repeats} passes" if pen.repeats > 1 else "one pass"
            self.tool_note.setText(f"{pen.kind_label} · {passes} · {pen.note or 'already in the library'}")
        else:
            spec = CUTTING_TOOLS[data[1]]
            self.tool_note.setText(f"{spec['passes']} passes, {spec['pass_depth']:.2f} mm deeper each time. {spec['note']}")

    # ------------------------------------------------------------------
    def result_pen(self) -> int:
        """The pen the cut layers should use, adding it to the library if new."""
        data = self.tool_box.currentData()
        if data and data[0] == "existing":
            return int(data[1])
        if data and data[0] == "new":
            return self.settings.library.add_tool(data[1])
        return 0

    def sheet_strokes(self) -> list[tuple[str, list[list[list[float]]]]]:
        """(label, strokes) per sheet, in millimetres on the bed, Y up."""
        machine = self.settings.machine
        ppm = self.px_per_mm
        width_mm = self.rgb.shape[1] / ppm
        height_mm = self.rgb.shape[0] / ppm
        origin_x = max((machine.bed_x - width_mm) * 0.5, 0.0)
        origin_y = max((machine.bed_y - height_mm) * 0.5, 0.0)
        out = []
        for sheet in self.sheets:
            strokes = []
            for cut in sheet.cuts:
                points = np.asarray(cut, dtype=np.float64).reshape(-1, 2)
                if len(points) < 2:
                    continue
                # pixels are Y down from the top left; the bed is Y up from the
                # bottom left, and every sheet lands in the same place so they
                # register on top of each other
                strokes.append(
                    [
                        [origin_x + float(x) / ppm, origin_y + height_mm - float(y) / ppm]
                        for x, y in points
                    ]
                )
            out.append((f"Stencil {sheet.index + 1}/{len(self.sheets)} - {sheet.label.split(' - ')[-1]}", strokes))
        return out

    def resizeEvent(self, event) -> None:  # pragma: no cover - UI glue
        super().resizeEvent(event)
        self._update_big()
