"""The layer stack: several things on one sheet, each with its own settings."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ...core.settings import AppSettings
from .. import theme
from ..widgets import Card, hint_label

KIND_LABELS = {"image": "Image", "text": "Text", "pdf": "PDF", "shapes": "Drawing"}


class LayerRow(QFrame):
    """One entry in the stack."""

    selected = Signal(int)
    changed = Signal()
    visibility_changed = Signal()

    def __init__(self, index: int, item, active: bool, parent=None):
        super().__init__(parent)
        self.index = index
        self.item = item
        self.setCursor(Qt.PointingHandCursor)
        self._refresh_style(active)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(6)

        self.eye = QToolButton()
        self.eye.setCheckable(True)
        self.eye.setChecked(item.visible)
        self.eye.setFixedWidth(26)
        self.eye.setToolTip("Draw this layer")
        self.eye.setText("◉" if item.visible else "○")
        self.eye.toggled.connect(self._on_visible)

        self.name = QLineEdit(item.label())
        self.name.setFrame(False)
        self.name.setStyleSheet("background: transparent;")
        self.name.textEdited.connect(self._on_name)

        self.kind = QLabel(KIND_LABELS.get(item.kind, item.kind))
        self.kind.setStyleSheet(f"color: {theme.TEXT_FAINT}; font-size: 10px;")

        layout.addWidget(self.eye)
        layout.addWidget(self.name, 1)
        layout.addWidget(self.kind)

    def _refresh_style(self, active: bool) -> None:
        if active:
            self.setStyleSheet(
                f"QFrame {{ background: {theme.ACCENT_SOFT}; border: 1px solid {theme.ACCENT};"
                f" border-radius: 6px; }}"
            )
        else:
            self.setStyleSheet(
                f"QFrame {{ background: {theme.PANEL_ALT}; border: 1px solid {theme.BORDER};"
                f" border-radius: 6px; }}"
                f"QFrame:hover {{ border-color: {theme.BORDER_STRONG}; }}"
            )

    def _on_visible(self, value: bool) -> None:
        self.item.visible = bool(value)
        self.eye.setText("◉" if value else "○")
        self.visibility_changed.emit()

    def _on_name(self, text: str) -> None:
        self.item.name = text
        self.changed.emit()

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() == Qt.LeftButton:
            self.selected.emit(self.index)


class LayersPanel(QWidget):
    """Add, reorder, hide and select the things on the sheet."""

    selection_changed = Signal(int)
    structure_changed = Signal()
    redraw_needed = Signal()
    add_requested = Signal(str)

    def __init__(self, settings: AppSettings, parent=None):
        super().__init__(parent)
        self.settings = settings

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(10)

        self.card = Card("LAYERS")
        self.rows_host = QWidget()
        self.rows_layout = QVBoxLayout(self.rows_host)
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.setSpacing(4)
        self.card.add(self.rows_host)

        add_row = QHBoxLayout()
        add_row.setSpacing(5)
        for label, kind, tip in (
            ("+ Image", "image", "Add a picture"),
            ("+ Text", "text", "Add a line of text"),
            ("+ PDF", "pdf", "Add a PDF page"),
            ("+ Draw", "shapes", "Draw lines by hand on the bed"),
        ):
            button = QPushButton(label)
            button.setToolTip(tip)
            button.clicked.connect(lambda _checked=False, k=kind: self.add_requested.emit(k))
            add_row.addWidget(button)
        self.card.add_layout(add_row)

        tools = QHBoxLayout()
        tools.setSpacing(5)
        for label, handler, tip in (
            ("↑", lambda: self._move(-1), "Move up"),
            ("↓", lambda: self._move(1), "Move down"),
            ("Duplicate", self._duplicate, "Copy this layer"),
            ("Delete", self._delete, "Remove this layer"),
        ):
            button = QPushButton(label)
            button.setToolTip(tip)
            if label in ("↑", "↓"):
                button.setObjectName("Compact")
                button.setFixedWidth(34)
            if label == "Delete":
                button.setObjectName("Danger")
                # a red button that silently does nothing reads as broken
                self.delete_button = button
            button.clicked.connect(handler)
            tools.addWidget(button)
        self.card.add_layout(tools)
        self.card.add(hint_label(
            "Each layer keeps its own picture, technique and placement. Click one to edit it, "
            "or click it on the bed."
        ))
        outer.addWidget(self.card)
        outer.addStretch(1)
        self.rebuild()

    # ------------------------------------------------------------------
    def rebuild(self) -> None:
        while self.rows_layout.count():
            entry = self.rows_layout.takeAt(0)
            widget = entry.widget()
            if widget:
                widget.deleteLater()
        for index, item in enumerate(self.settings.items):
            row = LayerRow(index, item, index == self.settings.active)
            row.selected.connect(self.selection_changed)
            row.changed.connect(self.structure_changed)
            row.visibility_changed.connect(self.redraw_needed)
            self.rows_layout.addWidget(row)
        self.card.set_title(f"LAYERS  ({len(self.settings.items)})")
        only_one = len(self.settings.items) <= 1
        self.delete_button.setEnabled(not only_one)
        self.delete_button.setToolTip(
            "A project needs at least one layer" if only_one else "Remove this layer"
        )

    def _move(self, delta: int) -> None:
        before = self.settings.active
        after = self.settings.move_item(before, delta)
        if after != before:
            self.structure_changed.emit()
            self.selection_changed.emit(after)

    def _duplicate(self) -> None:
        if self.settings.duplicate_item(self.settings.active):
            self.structure_changed.emit()
            self.selection_changed.emit(self.settings.active)

    def _delete(self) -> None:
        if len(self.settings.items) <= 1:
            return
        item = self.settings.items[self.settings.active]
        has_work = bool(item.source_path or item.strokes)
        if has_work:
            answer = QMessageBox.question(
                self,
                "Delete this layer?",
                f"\u201c{item.label()}\u201d and everything on it will be removed. "
                "There is no undo.",
                QMessageBox.Cancel | QMessageBox.Yes,
                QMessageBox.Cancel,
            )
            if answer != QMessageBox.Yes:
                return
        self.settings.remove_item(self.settings.active)
        self.structure_changed.emit()
        self.selection_changed.emit(self.settings.active)
