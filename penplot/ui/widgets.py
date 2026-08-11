"""Reusable controls and a small settings-binding helper.

`Binder` connects a widget straight to a dataclass field, so a panel is a list
of one-line declarations instead of a pile of signal handlers.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

from PySide6.QtCore import QLocale, QObject, QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QLayout,
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
    "fit_text",
    "FlowLayout",
    "_DecimalSpin",
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


def fit_text(label: QLabel, minimum_points: float = 8.0) -> None:
    """Shrink a label's type until its text fits, rather than cutting it off.

    A clipped word reads as a broken program; the same word a point smaller
    reads as a tight layout.  Only ever shrinks, never grows past the size the
    label was given.
    """
    available = label.width() or label.maximumWidth()
    if available <= 0 or not label.text():
        return
    font = QFont(label.font())
    base = font.pointSizeF() if font.pointSizeF() > 0 else 11.0
    size = base
    while size > minimum_points:
        metrics = QFontMetricsF(font)
        if metrics.horizontalAdvance(label.text()) <= available:
            break
        size -= 0.5
        font.setPointSizeF(size)
    label.setFont(font)


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
            # two lines are fine; a third would be cut off, so shrink instead
            if len(label) > 18:
                fit_text(self.label, minimum_points=9.0)
            line.addWidget(self.label, 0, Qt.AlignVCenter)
        else:
            self.label = None
        line.addWidget(control, 1)
        outer.addLayout(line)

        if hint:
            outer.addWidget(hint_label(hint))

    def set_hidden(self, hidden: bool) -> None:
        self.setVisible(not hidden)


class _DecimalSpin(QDoubleSpinBox):
    """A number box that behaves the same wherever you live.

    Two problems, both reported as "the live controls do not work":

    * The system locale here is en_SE, whose decimal separator is a comma, so
      typing `-0.50` into an English-language interface produced `-0.00` - the
      point was refused and the digits after it thrown away.  Both separators
      are accepted now, and the point is what gets shown.
    * Qt reinterprets the text on every keystroke by default, so `-0.50` passed
      through `-`, `-0`, `-0.` ... and the intermediate values were committed
      and sent to the printer.  The value is now taken when the field is left
      or Enter is pressed; the arrow keys and the slider still act at once.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setLocale(QLocale.c())
        self.setKeyboardTracking(False)

    def validate(self, text: str, pos: int):
        return super().validate(text.replace(",", "."), pos)

    def valueFromText(self, text: str) -> float:
        return super().valueFromText(text.replace(",", "."))


class _IntSpin(QSpinBox):
    """Integers, with the same "do not commit half a number" rule."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setLocale(QLocale.c())
        self.setKeyboardTracking(False)


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
            self.spin: QSpinBox | QDoubleSpinBox = _IntSpin()
            self.spin.setRange(int(minimum), int(maximum))
            self.spin.setSingleStep(max(int(step), 1))
        else:
            self.spin = _DecimalSpin()
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


class FlowLayout(QLayout):
    """Cards flow into as many columns as the width allows.

    The settings panels were a single tall column inside a scroll area, so the
    Printer panel was 1147 px of content in a 406 px viewport - three screens of
    scrolling to reach the pen-change height.  Widening the dock did nothing,
    because a vertical box layout has no use for horizontal space.

    This lays the same cards out left to right, wrapping when it runs out of
    width, so a wide dock becomes two or three columns and the scrolling stops.
    Narrow it again and it collapses back to one column on its own.
    """

    def __init__(self, parent=None, spacing: int = 10):
        super().__init__(parent)
        self._items: list = []
        self._spacing = spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item) -> None:      # noqa: N802 - Qt
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):         # noqa: N802 - Qt
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int):         # noqa: N802 - Qt
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):        # noqa: N802 - Qt
        return Qt.Orientations(Qt.Orientation(0))

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt
        return True

    def heightForWidth(self, width: int) -> int:   # noqa: N802 - Qt
        return self._layout(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect) -> None:  # noqa: N802 - Qt
        super().setGeometry(rect)
        self._layout(rect, apply=True)

    def sizeHint(self) -> QSize:          # noqa: N802 - Qt
        return self.minimumSize()

    def minimumSize(self) -> QSize:       # noqa: N802 - Qt
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QSize(margins.left() + margins.right(), margins.top() + margins.bottom())

    def _gap(self) -> int:
        set_by_caller = self.spacing()
        return set_by_caller if set_by_caller >= 0 else self._spacing

    def _layout(self, rect, apply: bool) -> int:
        gap = self._gap()
        margins = self.contentsMargins()
        left = rect.x() + margins.left()
        top = rect.y() + margins.top()
        right = rect.right() - margins.right()

        # How many columns fit?  Every card gets the same width so the panel
        # reads as a grid rather than a ragged pile.
        widest = max((item.sizeHint().width() for item in self._items), default=1)
        available = max(right - left + 1, 1)
        columns = max(int((available + gap) // (widest + gap)), 1)
        column_width = max((available - (columns - 1) * gap) // columns, 1)

        x, y, row_height = left, top, 0
        column = 0
        for item in self._items:
            height = item.heightForWidth(column_width) if item.hasHeightForWidth() else item.sizeHint().height()
            if apply:
                item.setGeometry(QRect(x, y, column_width, height))
            row_height = max(row_height, height)
            column += 1
            if column >= columns:
                column = 0
                x = left
                y += row_height + gap
                row_height = 0
            else:
                x += column_width + gap
        if column:
            y += row_height
        else:
            y -= gap
        return y - rect.y() + margins.bottom()
