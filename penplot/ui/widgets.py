"""Reusable controls and a small settings-binding helper.

`Binder` connects a widget straight to a dataclass field, so a panel is a list
of one-line declarations instead of a pile of signal handlers.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

from PySide6.QtCore import QObject, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from . import theme

__all__ = [
    "Card",
    "FieldRow",
    "SliderSpin",
    "Segmented",
    "ColorButton",
    "Binder",
    "Separator",
    "swatch_icon",
    "heading",
    "hint_label",
]


def heading(text: str) -> QLabel:
    label = QLabel(text.upper())
    label.setObjectName("SectionLabel")
    return label


def hint_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("Hint")
    label.setWordWrap(True)
    return label


def swatch_icon(color: str, size: int = 12) -> QIcon:
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setBrush(QColor(color))
    painter.setPen(QColor(theme.BORDER_STRONG))
    painter.drawEllipse(1, 1, size - 2, size - 2)
    painter.end()
    return QIcon(pixmap)


class Separator(QFrame):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("Separator")
        self.setFixedHeight(1)


class Card(QFrame):
    """White rounded panel with a title and an optional collapse arrow."""

    toggled = Signal(bool)

    def __init__(self, title: str = "", collapsible: bool = False, expanded: bool = True, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(9, 7, 9, 9)
        outer.setSpacing(6)

        self._header = QFrame()
        self._header.setObjectName("CardHeader")
        header_layout = QHBoxLayout(self._header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(6)

        self._arrow = QLabel("")
        self._arrow.setObjectName("Hint")
        self._arrow.setFixedWidth(12)
        self._arrow.setStyleSheet(f"color: {theme.TEXT_MUTED}; font-size: 11px;")
        self._arrow.setVisible(collapsible)
        self._title = QLabel(title)
        self._title.setObjectName("CardTitle")
        header_layout.addWidget(self._arrow)
        header_layout.addWidget(self._title)
        header_layout.addStretch(1)
        self.header_extra = header_layout

        self._body = QWidget()
        self.body = QVBoxLayout(self._body)
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(5)

        outer.addWidget(self._header)
        outer.addWidget(self._body)

        self._collapsible = collapsible
        self._expanded = expanded
        if collapsible:
            self._header.setCursor(Qt.PointingHandCursor)
            self._header.mousePressEvent = self._on_header_click  # type: ignore[assignment]
            self._update_arrow()
            self._body.setVisible(expanded)
        if not title:
            self._header.hide()

    def _on_header_click(self, event) -> None:  # pragma: no cover - UI glue
        self.set_expanded(not self._expanded)

    def _update_arrow(self) -> None:
        self._arrow.setText("▼" if self._expanded else "▶")

    def set_expanded(self, value: bool) -> None:
        self._expanded = value
        self._body.setVisible(value)
        self._update_arrow()
        self.toggled.emit(value)

    def set_title(self, text: str) -> None:
        self._title.setText(text)

    def add(self, widget: QWidget) -> QWidget:
        self.body.addWidget(widget)
        return widget

    def add_layout(self, layout) -> None:
        self.body.addLayout(layout)

    def add_heading(self, text: str) -> QLabel:
        label = heading(text)
        self.body.addSpacing(2)
        self.body.addWidget(label)
        return label

    def add_hint(self, text: str) -> QLabel:
        return self.add(hint_label(text))

    def hide_header(self) -> None:
        self._header.hide()

    def add_reset(self, handler, tooltip: str = "Put this card back the way it was") -> QPushButton:
        """A reset in the card header, where the standard puts it.

        Per-card rather than one global reset: a user who has fiddled with the
        tone does not want to lose their pen library to get it back.
        """
        button = QPushButton("Reset")
        button.setObjectName("Ghost")
        button.setToolTip(tooltip)
        button.setCursor(Qt.PointingHandCursor)
        button.setFixedHeight(18)
        button.clicked.connect(handler)
        self.header_extra.addWidget(button)
        if not self._header.isVisible():
            self._header.show()
        return button


class FieldRow(QWidget):
    """Label on the left, control on the right, optional hint underneath."""

    def __init__(self, label: str, control: QWidget, hint: str = "", label_width: int = 86, parent=None):
        super().__init__(parent)
        self.control = control
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)

        line = QHBoxLayout()
        line.setContentsMargins(0, 0, 0, 0)
        line.setSpacing(8)
        if label:
            self.label = QLabel(label)
            self.label.setMinimumWidth(label_width)
            self.label.setMaximumWidth(label_width)
            self.label.setWordWrap(True)
            line.addWidget(self.label, 0, Qt.AlignVCenter)
        else:
            self.label = None
        line.addWidget(control, 1)
        outer.addLayout(line)

        if hint:
            outer.addWidget(hint_label(hint))

    def set_hidden(self, hidden: bool) -> None:
        self.setVisible(not hidden)


class SliderSpin(QWidget):
    """Slider plus numeric entry that stay in sync."""

    valueChanged = Signal(float)

    def __init__(
        self,
        minimum: float,
        maximum: float,
        *,
        decimals: int = 1,
        step: float = 0.1,
        suffix: str = "",
        show_slider: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self._decimals = decimals
        self._factor = 10**decimals
        self._guard = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(int(round(minimum * self._factor)), int(round(maximum * self._factor)))
        self.slider.setVisible(show_slider)

        if decimals == 0:
            self.spin: QSpinBox | QDoubleSpinBox = QSpinBox()
            self.spin.setRange(int(minimum), int(maximum))
            self.spin.setSingleStep(max(int(step), 1))
        else:
            self.spin = QDoubleSpinBox()
            self.spin.setRange(minimum, maximum)
            self.spin.setDecimals(decimals)
            self.spin.setSingleStep(step)
        if suffix:
            self.spin.setSuffix(f" {suffix}")
        # Size the box from the widest text it can ever hold.  A fixed width cut
        # "2400 mm/min" down to "2400 mm/r", which reads as a typo rather than
        # as a box that is too narrow.
        widest = f"-{int(abs(maximum)) or 1}" + (f".{'0' * decimals}" if decimals else "")
        if suffix:
            widest += f" {suffix}"
        self.spin.setFixedWidth(self.spin.fontMetrics().horizontalAdvance(widest) + 26)
        self.spin.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin, 0)

        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)

    def _from_slider(self, raw: int) -> None:
        if self._guard:
            return
        self._guard = True
        value = raw / self._factor
        self.spin.setValue(value)
        self._guard = False
        self.valueChanged.emit(value)

    def _from_spin(self, value: float) -> None:
        if self._guard:
            return
        self._guard = True
        self.slider.setValue(int(round(float(value) * self._factor)))
        self._guard = False
        self.valueChanged.emit(float(value))

    def value(self) -> float:
        return float(self.spin.value())

    def setValue(self, value: float) -> None:
        self._guard = True
        self.spin.setValue(value)
        self.slider.setValue(int(round(float(value) * self._factor)))
        self._guard = False


class Segmented(QWidget):
    """A row of connected toggle buttons, used for small enums."""

    changed = Signal(str)

    def __init__(self, options: Sequence[tuple[str, str]], parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: dict[str, QPushButton] = {}

        for index, (key, label) in enumerate(options):
            button = QPushButton(label)
            button.setCheckable(True)
            button.setCursor(Qt.PointingHandCursor)
            name = "Segment"
            if len(options) > 1:
                if index == 0:
                    name = "SegmentFirst"
                elif index == len(options) - 1:
                    name = "SegmentLast"
            button.setObjectName(name)
            button.setProperty("class", "Segment")
            button.setStyleSheet(_segment_style(index, len(options)))
            layout.addWidget(button, 1)
            self._group.addButton(button, index)
            self._buttons[key] = button
            button.clicked.connect(lambda _checked=False, k=key: self.changed.emit(k))

    def value(self) -> str:
        for key, button in self._buttons.items():
            if button.isChecked():
                return key
        return ""

    def setValue(self, key: str) -> None:
        button = self._buttons.get(key)
        if button:
            button.setChecked(True)

    def buttons(self) -> dict:
        """So a panel can hide a segment that does not apply right now."""
        return dict(self._buttons)


def _segment_style(index: int, count: int) -> str:
    radius_left = theme.RADIUS if index == 0 else "0px"
    radius_right = theme.RADIUS if index == count - 1 else "0px"
    return f"""
    QPushButton {{
        background: {theme.PANEL};
        border: 1px solid {theme.BORDER_STRONG};
        border-left-width: {1 if index == 0 else 0}px;
        border-top-left-radius: {radius_left};
        border-bottom-left-radius: {radius_left};
        border-top-right-radius: {radius_right};
        border-bottom-right-radius: {radius_right};
        padding: 6px 8px;
        color: {theme.TEXT_MUTED};
    }}
    QPushButton:hover {{ color: {theme.TEXT}; }}
    QPushButton:checked {{
        background: {theme.ACCENT_SOFT};
        color: {theme.ACCENT};
        border-color: {theme.ACCENT};
        font-weight: 600;
    }}
    """


class ColorButton(QPushButton):
    """Swatch button that opens the system colour picker."""

    colorChanged = Signal(str)

    def __init__(self, color: str = "#1A1A1A", parent=None):
        super().__init__(parent)
        self._color = color
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(QSize(28, 20))
        self.clicked.connect(self._pick)
        self._refresh()

    def _refresh(self) -> None:
        self.setStyleSheet(
            f"QPushButton {{ background: {self._color}; border: 1px solid {theme.BORDER_STRONG};"
            f" border-radius: 5px; }}"
            f"QPushButton:hover {{ border: 2px solid {theme.ACCENT}; }}"
        )

    def color(self) -> str:
        return self._color

    def setColor(self, value: str) -> None:
        self._color = value
        self._refresh()

    def _pick(self) -> None:
        chosen = QColorDialog.getColor(QColor(self._color), self, "Pen colour")
        if chosen.isValid():
            self.setColor(chosen.name().upper())
            self.colorChanged.emit(self._color)


# --------------------------------------------------------------------------
class Binder(QObject):
    """Creates widgets wired straight to dataclass attributes."""

    def __init__(self, on_change: Callable[[], None], parent: QObject | None = None, context: Any = None):
        super().__init__(parent)
        self._on_change = on_change
        self._refreshers: list[Callable[[], None]] = []
        self.blocked = False
        #: when a target is given as a string it is looked up on this object at
        #: every read and write, so a control automatically follows whichever
        #: layer is selected instead of clinging to the one it was built with
        self.context = context

    def _resolve(self, target: Any):
        if isinstance(target, str):
            return getattr(self.context, target)
        return target

    # ---- internals ----
    def _changed(self) -> None:
        if not self.blocked:
            self._on_change()

    def _register(self, refresher: Callable[[], None]) -> None:
        self._refreshers.append(refresher)

    def refresh(self) -> None:
        self.blocked = True
        try:
            for refresher in self._refreshers:
                refresher()
        finally:
            self.blocked = False

    # ---- factories ----
    def slider(
        self,
        target: Any,
        attr: str,
        minimum: float,
        maximum: float,
        *,
        label: str = "",
        decimals: int = 1,
        step: float = 0.1,
        suffix: str = "",
        hint: str = "",
        show_slider: bool = True,
    ) -> FieldRow:
        control = SliderSpin(minimum, maximum, decimals=decimals, step=step, suffix=suffix, show_slider=show_slider)
        control.setValue(float(getattr(self._resolve(target), attr)))

        def apply(value: float) -> None:
            setattr(self._resolve(target), attr, int(round(value)) if decimals == 0 else float(value))
            self._changed()

        control.valueChanged.connect(apply)
        self._register(lambda: control.setValue(float(getattr(self._resolve(target), attr))))
        return FieldRow(label, control, hint)

    def check(self, target: Any, attr: str, text: str, hint: str = "") -> QCheckBox | FieldRow:
        control = QCheckBox(text)
        control.setChecked(bool(getattr(self._resolve(target), attr)))

        def apply(state: bool) -> None:
            setattr(self._resolve(target), attr, bool(state))
            self._changed()

        control.toggled.connect(apply)
        self._register(lambda: control.setChecked(bool(getattr(self._resolve(target), attr))))
        if hint:
            row = FieldRow("", control, hint)
            row.control = control
            return row
        return control

    def combo(
        self,
        target: Any,
        attr: str,
        options: dict[str, str] | Iterable[str],
        *,
        label: str = "",
        hint: str = "",
    ) -> FieldRow:
        control = QComboBox()
        items = options.items() if isinstance(options, dict) else [(o, o) for o in options]
        for key, text in items:
            control.addItem(text, key)
        index = control.findData(getattr(self._resolve(target), attr))
        control.setCurrentIndex(max(index, 0))

        def apply(idx: int) -> None:
            setattr(self._resolve(target), attr, control.itemData(idx))
            self._changed()

        control.currentIndexChanged.connect(apply)

        def refresh() -> None:
            idx = control.findData(getattr(self._resolve(target), attr))
            control.setCurrentIndex(max(idx, 0))

        self._register(refresh)
        return FieldRow(label, control, hint)

    def segmented(
        self, target: Any, attr: str, options: Sequence[tuple[str, str]], *, label: str = "", hint: str = ""
    ) -> FieldRow:
        control = Segmented(options)
        control.setValue(str(getattr(self._resolve(target), attr)))

        def apply(key: str) -> None:
            setattr(self._resolve(target), attr, key)
            self._changed()

        control.changed.connect(apply)
        self._register(lambda: control.setValue(str(getattr(self._resolve(target), attr))))
        return FieldRow(label, control, hint)

    def line(self, target: Any, attr: str, *, label: str = "", placeholder: str = "", hint: str = "") -> FieldRow:
        control = QLineEdit(str(getattr(self._resolve(target), attr)))
        control.setPlaceholderText(placeholder)

        def apply(text: str) -> None:
            setattr(self._resolve(target), attr, text)
            self._changed()

        control.textChanged.connect(apply)
        self._register(lambda: control.setText(str(getattr(self._resolve(target), attr))))
        return FieldRow(label, control, hint)

    def multiline(self, target: Any, attr: str, *, height: int = 96, placeholder: str = "") -> QPlainTextEdit:
        control = QPlainTextEdit(str(getattr(self._resolve(target), attr)))
        control.setPlaceholderText(placeholder)
        control.setFixedHeight(height)
        control.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        def apply() -> None:
            setattr(self._resolve(target), attr, control.toPlainText())
            self._changed()

        control.textChanged.connect(apply)

        def refresh() -> None:
            value = str(getattr(self._resolve(target), attr))
            if control.toPlainText() != value:
                control.setPlainText(value)

        self._register(refresh)
        return control
