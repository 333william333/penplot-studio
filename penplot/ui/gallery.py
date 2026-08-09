"""Technique gallery: see how every drawing style would look before choosing one."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, QSize, Qt, QThread, Signal, Slot
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..core import techniques
from . import theme

THUMB = 96

#: see ui/worker.py - a thread that will not stop is kept alive rather than
#: destroyed, because destroying it aborts the whole process
_ORPHANED: list = []


class _ThumbWorker(QObject):
    done = Signal(str, object, int)

    def __init__(self) -> None:
        super().__init__()
        # plain attributes, written from the GUI thread and read here: a slot
        # cannot be delivered while this loop is busy, so a queued signal would
        # never arrive in time to stop it
        self.abort = False
        self.wanted = 0

    @Slot(object, object, int, float, float)
    def build(self, image, keys, generation: int, width_mm: float, pen_width: float) -> None:
        for key in keys:
            if self.abort or generation != self.wanted:
                return
            try:
                thumbnail = techniques.render_thumbnail(
                    image, key, None, size=THUMB * 2, width_mm=width_mm, pen_width=pen_width
                )
            except Exception:
                thumbnail = None
            if self.abort or generation != self.wanted:
                return
            self.done.emit(key, thumbnail, generation)


class TechniqueTile(QFrame):
    """One clickable preview."""

    chosen = Signal(str)

    def __init__(self, key: str, label: str, parent=None):
        super().__init__(parent)
        self.key = key
        self.selected = False
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(QSize(THUMB + 10, THUMB + 24))
        self.setToolTip(techniques.REGISTRY[key].description)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 2)
        layout.setSpacing(2)

        self.image = QLabel()
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setFixedSize(THUMB, THUMB)
        self.image.setStyleSheet(f"background: #FFFFFF; border: 1px solid {theme.BORDER}; border-radius: 4px;")
        layout.addWidget(self.image, 0, Qt.AlignHCenter)

        self.caption = QLabel(label)
        self.caption.setAlignment(Qt.AlignCenter)
        self.caption.setStyleSheet(f"color: {theme.TEXT_MUTED}; font-size: 10px;")
        layout.addWidget(self.caption)

        self._refresh_style()

    def set_thumbnail(self, array: np.ndarray | None) -> None:
        if array is None:
            self.image.setText("—")
            return
        buffer = np.ascontiguousarray(array)
        # A technique that draws almost nothing on this picture - hatching a
        # page of handwriting, say - renders a blank card, and a row of blank
        # cards reads as a broken program rather than as an answer.
        ink = float(1.0 - buffer.mean() / 255.0)
        if ink < 0.004:
            self.image.setPixmap(QPixmap())
            self.image.setText("nothing\nto draw")
            self.setToolTip(f"{self.caption.text()} finds almost nothing in this picture")
            self.setEnabled(False)
            return
        self.setEnabled(True)
        height, width = buffer.shape
        image = QImage(buffer.data, width, height, width, QImage.Format_Grayscale8).copy()
        self.image.setPixmap(
            QPixmap.fromImage(image).scaled(THUMB, THUMB, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        self.image.setText("")

    def set_tile_size(self, size: int) -> None:
        self.setFixedSize(QSize(size + 10, size + 24))
        self.image.setFixedSize(size, size)

    def restyle(self) -> None:
        """Re-read the theme colours after a light/dark switch."""
        self.image.setStyleSheet(
            f"background: #FFFFFF; border: 1px solid {theme.BORDER}; border-radius: 4px;"
        )
        self._refresh_style()

    def set_selected(self, value: bool) -> None:
        if value == self.selected:
            return
        self.selected = value
        self._refresh_style()

    def _refresh_style(self) -> None:
        if self.selected:
            self.setStyleSheet(
                f"QFrame {{ background: {theme.ACCENT_SOFT}; border: 2px solid {theme.ACCENT};"
                f" border-radius: 7px; }}"
            )
            self.caption.setStyleSheet(f"color: {theme.ACCENT}; font-size: 10px; font-weight: 700;")
        else:
            self.setStyleSheet(
                f"QFrame {{ background: transparent; border: 2px solid transparent; border-radius: 7px; }}"
                f"QFrame:hover {{ background: {theme.PANEL_ALT}; border-color: {theme.BORDER_STRONG}; }}"
            )
            self.caption.setStyleSheet(f"color: {theme.TEXT_MUTED}; font-size: 10px;")

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() == Qt.LeftButton:
            self.chosen.emit(self.key)


class TechniqueGallery(QWidget):
    """Grid of live previews, filled in on a background thread."""

    technique_chosen = Signal(str)

    def __init__(self, columns: int = 2, parent=None):
        super().__init__(parent)
        self._generation = 0
        self._image: np.ndarray | None = None
        self._width_mm = 120.0
        self._pen_width = 0.5

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(4)
        layout.setVerticalSpacing(4)

        self.tiles: dict[str, TechniqueTile] = {}
        for index, (key, technique) in enumerate(techniques.REGISTRY.items()):
            tile = TechniqueTile(key, technique.label)
            tile.chosen.connect(self._on_chosen)
            layout.addWidget(tile, index // columns, index % columns)
            self.tiles[key] = tile
        self.columns = columns
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        self.thread = QThread()
        self.thread.setObjectName("gallery")
        self.worker = _ThumbWorker()
        self.worker.moveToThread(self.thread)
        self.worker.done.connect(self._on_thumbnail)
        self._request.connect(self.worker.build)
        self.thread.start()

    _request = Signal(object, object, int, float, float)

    def _on_chosen(self, key: str) -> None:
        self.set_current(key)
        self.technique_chosen.emit(key)

    def set_current(self, key: str) -> None:
        for tile_key, tile in self.tiles.items():
            tile.set_selected(tile_key == key)

    def set_tile_size(self, size: int) -> None:
        for tile in self.tiles.values():
            tile.set_tile_size(size)

    def restyle(self) -> None:
        for tile in self.tiles.values():
            tile.restyle()

    def refresh(self, image: np.ndarray | None, width_mm: float, pen_width: float) -> None:
        """Re-render every tile from the current picture."""
        self._generation += 1
        self.worker.wanted = self._generation  # makes the running loop bail out
        if image is None:
            for tile in self.tiles.values():
                tile.set_thumbnail(None)
            return
        self._image = image
        self._width_mm = max(width_mm, 10.0)
        self._pen_width = pen_width
        self._request.emit(image, list(self.tiles.keys()), self._generation, self._width_mm, pen_width)

    def _on_thumbnail(self, key: str, array, generation: int) -> None:
        if generation != self._generation:
            return
        tile = self.tiles.get(key)
        if tile is not None:
            tile.set_thumbnail(array)

    def shutdown(self) -> None:
        """Stop the render loop before the QThread object is destroyed.

        A tile can take a few hundred milliseconds, so quitting the event loop
        is not enough - the loop has to be told to stop, or Qt aborts the whole
        process with "QThread: Destroyed while thread is still running".
        """
        if not self.thread.isRunning():
            return
        self.worker.abort = True
        self.thread.quit()
        if not self.thread.wait(4000):
            _ORPHANED.append((self.thread, self.worker))
