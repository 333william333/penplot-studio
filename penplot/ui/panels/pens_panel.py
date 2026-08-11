"""The pen library: colours, widths, Z offsets, and when to stop for a swap."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ...core.pens import CUTTING_TOOLS, DEFAULT_PALETTES, PEN_KINDS, Pen, PenLibrary
from ...core.settings import AppSettings
from .. import theme
from ..widgets import Binder, Card, ColorButton, FieldRow, FlowLayout, hint_label, _DecimalSpin


class PenRow(QFrame):
    """One pen: colour, name, width, and the details behind a disclosure."""

    changed = Signal()
    remove_requested = Signal(int)
    move_requested = Signal(int, int)

    def __init__(self, index: int, pen: Pen, parent=None):
        super().__init__(parent)
        self.index = index
        self.pen = pen
        self.setObjectName("PenRow")
        self.setStyleSheet(
            f"QFrame#PenRow {{ background: {theme.PANEL_ALT}; border: 1px solid {theme.BORDER};"
            f" border-radius: 6px; }}"
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 7, 8, 7)
        outer.setSpacing(6)

        top = QHBoxLayout()
        top.setSpacing(7)

        self.number = QLabel(f"{index + 1}")
        self.number.setFixedWidth(14)
        self.number.setObjectName("Hint")

        self.colour = ColorButton(pen.color)
        self.colour.colorChanged.connect(self._on_colour)

        self.name = QLineEdit(pen.name)
        self.name.setMinimumWidth(70)
        self.name.setCursorPosition(0)   # show the start of the name, not its tail
        self.name.setToolTip(pen.name)
        self.name.textChanged.connect(self._on_name)

        self.width = _DecimalSpin()
        self.width.setRange(0.05, 5.0)
        self.width.setDecimals(2)
        self.width.setSingleStep(0.05)
        self.width.setSuffix(" mm")
        self.width.setValue(pen.width)
        self.width.setFixedWidth(92)
        self.width.setToolTip("Pen width. Line spacing and dot pitch scale with this.")
        self.width.valueChanged.connect(self._on_width)

        self.enabled = QCheckBox()
        self.enabled.setChecked(pen.enabled)
        self.enabled.setToolTip("Use this pen")
        self.enabled.toggled.connect(self._on_enabled)

        self.tip_badge = QLabel("")
        self.tip_badge.setToolTip("Ink keeps flowing when this pen rests - Dwell dots will work")
        self.tip_badge.setStyleSheet(f"color: {theme.ACCENT}; font-size: 11px;")

        self.expander = QToolButton()
        self.expander.setText("▸")
        self.expander.setCheckable(True)
        self.expander.setToolTip("More settings")
        self.expander.toggled.connect(self._on_expand)

        # Two lines, not one: at 386 px the single row clipped the name to
        # "Black finelin" and the width to "0,50 mı", which looked like a bug.
        top.addWidget(self.number)
        top.addWidget(self.colour)
        top.addWidget(self.name, 1)
        top.addWidget(self.tip_badge)
        top.addWidget(self.enabled)
        top.addWidget(self.expander)
        outer.addLayout(top)

        second = QHBoxLayout()
        second.setSpacing(7)
        width_label = QLabel("Width")
        width_label.setObjectName("Hint")
        second.addSpacing(18)
        second.addWidget(width_label)
        second.addWidget(self.width)
        second.addStretch(1)
        self.summary = QLabel("")
        self.summary.setObjectName("Hint")
        second.addWidget(self.summary)
        outer.addLayout(second)

        self.details = QWidget()
        details = QVBoxLayout(self.details)
        details.setContentsMargins(0, 2, 0, 0)
        details.setSpacing(5)

        self.kind = QComboBox()
        for key, label in PEN_KINDS.items():
            self.kind.addItem(label, key)
        index = self.kind.findData(pen.kind)
        self.kind.setCurrentIndex(max(index, 0))
        self.kind.setToolTip(
            "Fibre, fountain, gel and marker tips keep bleeding while the pen rests,\n"
            "which is what the Dwell dots technique needs. A ballpoint cannot do it."
        )
        self.kind.currentIndexChanged.connect(self._on_kind)
        details.addWidget(FieldRow("Tip", self.kind, label_width=86))

        self.z_offset = self._spin(-10.0, 10.0, 0.05, 2, " mm", pen.z_offset)
        self.z_offset.valueChanged.connect(self._on_z_offset)
        details.addWidget(FieldRow("Z offset", self.z_offset, label_width=86))

        self.feed_scale = self._spin(0.1, 3.0, 0.05, 2, " ×", pen.feed_scale)
        self.feed_scale.valueChanged.connect(self._on_feed)
        details.addWidget(FieldRow("Speed", self.feed_scale, label_width=86))

        self.sharpen = self._spin(0.0, 100.0, 0.5, 1, " m", pen.sharpen_every)
        self.sharpen.setToolTip("Stop for sharpening after this many metres. 0 = never.")
        self.sharpen.valueChanged.connect(self._on_sharpen)
        details.addWidget(FieldRow("Sharpen every", self.sharpen, label_width=86))

        # --- cutting and scoring ------------------------------------------
        self.passes = self._spin(1, 20, 1, 0, " ×", pen.passes)
        self.passes.setToolTip(
            "Go over every stroke this many times.\n"
            "A blade seldom gets through card in one go, and an ordinary pen\n"
            "will perforate paper if it repeats the same line often enough."
        )
        self.passes.valueChanged.connect(self._on_passes)
        self.passes_row = FieldRow("Passes", self.passes, label_width=86)
        details.addWidget(self.passes_row)

        self.pass_depth = self._spin(0.0, 2.0, 0.05, 2, " mm", pen.pass_depth)
        self.pass_depth.setToolTip("How much deeper each repeat goes.")
        self.pass_depth.valueChanged.connect(self._on_pass_depth)
        self.pass_depth_row = FieldRow("Deeper each", self.pass_depth, label_width=86)
        details.addWidget(self.pass_depth_row)

        self.blade_offset = self._spin(0.0, 2.0, 0.05, 2, " mm", pen.blade_offset)
        self.blade_offset.setToolTip(
            "How far the tip of a swivel blade trails behind its pivot.\n"
            "The corners are swung out by this much so they come out sharp\n"
            "instead of rounded. It is printed on the blade holder, usually 0.25 mm."
        )
        self.blade_offset.valueChanged.connect(self._on_blade_offset)
        self.blade_row = FieldRow("Blade offset", self.blade_offset, label_width=86)
        details.addWidget(self.blade_row)

        self.overcut = self._spin(0.0, 10.0, 0.5, 1, " mm", pen.overcut)
        self.overcut.setToolTip(
            "Carry a closed cut past its own start, so the bit that was cut\n"
            "while the blade was still turning gets cut again."
        )
        self.overcut.valueChanged.connect(self._on_overcut)
        self.overcut_row = FieldRow("Overcut", self.overcut, label_width=86)
        details.addWidget(self.overcut_row)

        buttons = QHBoxLayout()
        buttons.setSpacing(5)
        up = QPushButton("↑")
        up.setFixedWidth(30)
        up.clicked.connect(lambda: self.move_requested.emit(self.index, -1))
        down = QPushButton("↓")
        down.setFixedWidth(30)
        down.clicked.connect(lambda: self.move_requested.emit(self.index, 1))
        remove = QPushButton("Remove")
        remove.setObjectName("Danger")
        remove.clicked.connect(lambda: self.remove_requested.emit(self.index))
        buttons.addWidget(up)
        buttons.addWidget(down)
        buttons.addStretch(1)
        buttons.addWidget(remove)
        details.addLayout(buttons)

        self.details.setVisible(False)
        outer.addWidget(self.details)
        self._refresh_badge()
        self._sync_tool_rows()
        self._refresh_summary()

    @staticmethod
    def _spin(minimum, maximum, step, decimals, suffix, value) -> QDoubleSpinBox:
        spin = _DecimalSpin()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setSuffix(suffix)
        spin.setValue(value)
        return spin

    def _on_expand(self, expanded: bool) -> None:
        self.expander.setText("▾" if expanded else "▸")
        self.details.setVisible(expanded)

    def _on_colour(self, value: str) -> None:
        self.pen.color = value
        self.changed.emit()

    def _on_name(self, value: str) -> None:
        self.pen.name = value
        self.name.setToolTip(value)
        self.changed.emit()

    def _on_width(self, value: float) -> None:
        self.pen.width = float(value)
        self.changed.emit()

    def _on_enabled(self, value: bool) -> None:
        self.pen.enabled = bool(value)
        self.changed.emit()

    def _on_kind(self, index: int) -> None:
        self.pen.kind = self.kind.itemData(index) or "other"
        self._refresh_badge()
        self._refresh_summary()
        self._sync_tool_rows()
        self.changed.emit()

    def _refresh_summary(self) -> None:
        if self.pen.cuts:
            self.summary.setText(f"{self.pen.kind_label} · {self.pen.repeats}×")
        else:
            self.summary.setText(self.pen.kind_label)

    def _refresh_badge(self) -> None:
        if self.pen.cuts:
            self.tip_badge.setText("✂")
            self.tip_badge.setToolTip("This tool cuts or scores - it does not draw")
        else:
            self.tip_badge.setText("◍" if self.pen.bleeds else "")
            self.tip_badge.setToolTip("Ink keeps flowing when this pen rests - Dwell dots will work")

    def _sync_tool_rows(self) -> None:
        """Blade settings only make sense for something with a blade in it."""
        cutting = self.pen.cuts
        self.pass_depth_row.setVisible(cutting or self.pen.passes > 1)
        self.blade_row.setVisible(self.pen.kind == "knife")
        self.overcut_row.setVisible(cutting)

    def _on_passes(self, value: float) -> None:
        self.pen.passes = int(round(value))
        self._sync_tool_rows()
        self._refresh_summary()
        self.changed.emit()

    def _on_pass_depth(self, value: float) -> None:
        self.pen.pass_depth = float(value)
        self.changed.emit()

    def _on_blade_offset(self, value: float) -> None:
        self.pen.blade_offset = float(value)
        self.changed.emit()

    def _on_overcut(self, value: float) -> None:
        self.pen.overcut = float(value)
        self.changed.emit()

    def _on_z_offset(self, value: float) -> None:
        self.pen.z_offset = float(value)
        self.changed.emit()

    def _on_feed(self, value: float) -> None:
        self.pen.feed_scale = float(value)
        self.changed.emit()

    def _on_sharpen(self, value: float) -> None:
        self.pen.sharpen_every = float(value)
        self.changed.emit()


class PensPanel(QWidget):
    """Pen list plus the pause behaviour that goes with swapping pens."""

    changed = Signal()
    library_changed = Signal()

    def __init__(self, settings: AppSettings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.library: PenLibrary = settings.library
        self.binder = Binder(self._emit, self)

        # Flow, so a wider dock becomes columns instead of a longer scroll.
        outer = FlowLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(10)

        self.card = Card("PENS")
        self.card.add_reset(self._reset_pens, "Back to a single black 0.5 mm fineliner")
        self.preset = QComboBox()
        self.preset.addItem("Load a pen set…", "")
        for name in DEFAULT_PALETTES:
            self.preset.addItem(name, name)
        self.preset.currentIndexChanged.connect(self._apply_preset)
        self.card.add(self.preset)

        self.rows_host = QWidget()
        self.rows_layout = QVBoxLayout(self.rows_host)
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.setSpacing(6)
        self.card.add(self.rows_host)

        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        add = QPushButton("Add pen")
        add.clicked.connect(self._add_pen)
        order = QPushButton("Order light → dark")
        order.setToolTip("Draw the lightest pen first so dark ink is never smudged.")
        order.clicked.connect(self._sort_pens)
        buttons.addWidget(add)
        buttons.addWidget(order)
        self.card.add_layout(buttons)

        self.tool_preset = QComboBox()
        self.tool_preset.addItem("Add a cutting tool…", "")
        for name in CUTTING_TOOLS:
            self.tool_preset.addItem(name, name)
        self.tool_preset.setToolTip(
            "A blade - or an ordinary pen told to go over the same line until the\n"
            "paper gives - so the machine can cut stencils instead of drawing."
        )
        self.tool_preset.currentIndexChanged.connect(self._add_tool)
        self.card.add(self.tool_preset)

        self.card.add(hint_label(
            "Drawing order follows this list. The printer stops between pens so you can swap them."
        ))
        outer.addWidget(self.card)

        pause_card = Card("PAUSES", collapsible=True, expanded=True)
        pauses = settings.pauses
        pause_card.add(self.binder.check(pauses, "pause_between_pens", "Stop for pen changes"))
        pause_card.add(self.binder.check(pauses, "host_pause", "Pause from the computer",
                                         hint="The stream stops and a dialog appears here. Works on any firmware."))
        pause_card.add(self.binder.check(pauses, "emit_m0", "Also write M0 into the file",
                                         hint="Lets the pauses work when printing from an SD card too."))
        pause_card.add(self.binder.check(pauses, "park_for_pause", "Park the head before pausing"))
        pause_card.add_heading("Sharpening")
        pause_card.add(self.binder.check(pauses, "sharpen_enabled", "Stop to sharpen the pen"))
        self.sharpen_row = self.binder.slider(
            pauses, "sharpen_interval", 0.5, 60.0, label="Every", decimals=1, step=0.5, suffix="m"
        )
        pause_card.add(self.sharpen_row)
        pause_card.add(hint_label(
            "Pencils and clutch pencils go blunt as they draw. A per-pen value in the pen "
            "details overrides this one."
        ))
        outer.addWidget(pause_card)

        self.rebuild()

    # ------------------------------------------------------------------
    def _emit(self) -> None:
        self.changed.emit()

    def rebuild(self) -> None:
        while self.rows_layout.count():
            item = self.rows_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        for index, pen in enumerate(self.library):
            row = PenRow(index, pen)
            row.changed.connect(self._emit)
            row.remove_requested.connect(self._remove_pen)
            row.move_requested.connect(self._move_pen)
            self.rows_layout.addWidget(row)
        self.card.set_title(f"PENS  ({len(self.library)})")
        self.library_changed.emit()

    def _reset_pens(self) -> None:
        from ...core.pens import Pen

        self.library.pens = [Pen()]
        self.rebuild()
        self._emit()

    def _add_pen(self) -> None:
        self.library.add()
        self.rebuild()
        self._emit()

    def _add_tool(self, index: int) -> None:
        name = self.tool_preset.itemData(index)
        if not name:
            return
        self.library.add_tool(name)
        self.tool_preset.blockSignals(True)
        self.tool_preset.setCurrentIndex(0)
        self.tool_preset.blockSignals(False)
        self.rebuild()
        self._emit()

    def _remove_pen(self, index: int) -> None:
        if len(self.library) <= 1:
            return
        self.library.remove(index)
        self.rebuild()
        self._emit()

    def _move_pen(self, index: int, delta: int) -> None:
        self.library.move(index, delta)
        self.rebuild()
        self._emit()

    def _sort_pens(self) -> None:
        order = self.library.sorted_light_to_dark()
        self.library.pens = [self.library.pens[i] for i in order]
        self.rebuild()
        self._emit()

    def _apply_preset(self, index: int) -> None:
        name = self.preset.itemData(index)
        if not name:
            return
        self.library.apply_palette(name)
        self.preset.blockSignals(True)
        self.preset.setCurrentIndex(0)
        self.preset.blockSignals(False)
        self.rebuild()
        self._emit()

    def apply_colours(self, colours: list[str]) -> None:
        """Used by 'suggest pen colours from the picture'."""
        while len(self.library) < len(colours):
            self.library.add()
        while len(self.library) > len(colours) and len(self.library) > 1:
            self.library.remove(len(self.library) - 1)
        for pen, colour in zip(self.library.pens, colours):
            pen.color = colour
            pen.enabled = True
        self.rebuild()
        self._emit()

    def refresh(self) -> None:
        self.library = self.settings.library
        self.binder.refresh()
        self.rebuild()
