"""The plot preview: bed, artwork, travel moves, and direct manipulation.

This is where the user checks *where* and *how* the pen will move before
anything touches paper, so it draws the real generated tool path - not a
picture of the source image.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
    QTransform,
)
from PySide6.QtWidgets import QWidget

from ..core.gcode import format_duration
from ..core.pens import PenLibrary
from ..core.pipeline import PlotJob
from . import theme


def _polygon(path: np.ndarray) -> QPolygonF:
    """One stroke as a polyline Qt can draw in a single call.

    A single-point path is a dot: give it a hair of length so a round cap
    actually paints something.
    """
    points = np.asarray(path, dtype=np.float64)[:, :2]
    if len(points) == 1:
        x, y = float(points[0][0]), float(points[0][1])
        return QPolygonF([QPointF(x, y), QPointF(x + 0.001, y)])
    return QPolygonF([QPointF(float(x), float(y)) for x, y in points])

HANDLE_SIZE = 9
MIN_ZOOM = 0.15
MAX_ZOOM = 120.0


class PreviewCanvas(QWidget):
    """Interactive view of the generated tool path on the printer bed."""

    move_committed = Signal(float, float)     # dx, dy in mm
    scale_committed = Signal(float)           # multiplier
    files_dropped = Signal(list)
    hover_position = Signal(float, float)
    selection_changed = Signal(bool)
    zoom_changed = Signal(float)
    item_selected = Signal(int)
    stroke_drawn = Signal(object)      # a finished stroke, in bed millimetres

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(420, 340)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.StrongFocus)

        self.job: PlotJob | None = None
        self.library = PenLibrary()
        self.bed = (220.0, 220.0)
        self.margin = 10.0
        self.park = (5.0, 200.0)

        self._zoom = 2.0
        self._origin = QPointF(40.0, 400.0)
        self._panning = False
        self._pan_anchor = QPointF()

        self._flat: list[tuple[int, np.ndarray]] = []
        #: group -> list of QPolygonF, one per stroke.  NOT one big QPainterPath:
        #: Qt's stroker is superlinear in the number of subpaths inside a single
        #: path, so a 5000-stroke drawing took 259 ms as one path and 17 ms as
        #: separate polylines.  That difference is the whole live preview.
        self._pen_paths: dict[tuple, list] = {}
        self._group_weight: dict[tuple, float] = {}
        self._travel_path = QPainterPath()
        self._done_paths: dict[tuple, list] = {}
        self._done_count = 0
        self._bounds: tuple[float, float, float, float] | None = None

        # the artwork is rendered once into a pixmap; panning and dragging then
        # only blit it, which keeps big stipple jobs interactive
        self._cache: QPixmap | None = None
        self._cache_key: tuple | None = None
        self._cache_origin = QPointF()
        self._cache_zoom = 1.0
        self._job_serial = 0

        # a short settle timer keeps zooming smooth: the cached image is scaled
        # during the burst and re-rendered crisply once the wheel stops
        self._stale_timer = QTimer(self)
        self._stale_timer.setSingleShot(True)
        self._stale_timer.setInterval(400)
        self._stale_timer.timeout.connect(self._show_stale)

        self._zoom_settle = QTimer(self)
        self._zoom_settle.setSingleShot(True)
        self._zoom_settle.setInterval(140)
        self._zoom_settle.timeout.connect(self.update)

        self._progress = 1.0
        self._live_position: tuple[float, float] | None = None
        self._hidden_pens: set[int] = set()
        self._legend_rects: list[tuple[QRectF, int]] = []

        self.show_travels = True
        self.show_grid = True
        self.show_stats = True
        self.show_handles = True
        self.stale = False
        self.stale_message = "Updating…"

        self._drag_mode: str | None = None
        self._drag_start = QPointF()
        self._temp_offset = QPointF(0.0, 0.0)
        self._temp_scale = 1.0
        self._hover_handle: int | None = None
        self._selected = True
        self._space_held = False
        self._item_bounds: dict[int, tuple] = {}
        self._active_item = 0
        #: "" | "free" | "dot" | "line" | "rect" | "ellipse"
        self.draw_tool = ""
        self._sketch: list[tuple[float, float]] = []

    # ------------------------------------------------------------------
    # data
    # ------------------------------------------------------------------
    def set_job(self, job: PlotJob | None, library: PenLibrary, bed: tuple[float, float], margin: float, park) -> None:
        self.job = job
        self.library = library
        self.bed = bed
        self.margin = margin
        self.park = park
        self._temp_offset = QPointF(0.0, 0.0)
        self._temp_scale = 1.0
        self._rebuild_cache()
        self.update()

    def _rebuild_cache(self) -> None:
        self._job_serial += 1
        self._cache = None
        self._flat = []
        self._pen_paths = {}
        self._group_weight = {}
        self._travel_path = QPainterPath()
        self._done_paths = {}
        self._done_count = 0
        self._bounds = None
        if self.job is None or self.job.drawing.is_empty():
            return

        # A dwell layer is drawn with a fatter pen: on paper a 300 ms dot really
        # is bigger and darker than a 20 ms one, and the preview should say so.
        dwells = [l.dwell_ms for l in self.job.drawing.layers if l.dwell_ms > 0]
        dwell_lo = min(dwells) if dwells else 0.0
        dwell_hi = max(dwells) if dwells else 1.0

        cursor = QPointF(self.park[0], self.park[1])
        for layer in self.job.drawing.layers:
            weight = 0.0
            if layer.dwell_ms > 0:
                weight = (layer.dwell_ms - dwell_lo) / max(dwell_hi - dwell_lo, 1e-6)
            modulated = bool(layer.modulation)
            group = (layer.pen, round(weight, 2))
            self._group_weight[group] = weight
            strokes = self._pen_paths.setdefault(group, [])

            for path in layer.paths:
                if len(path) == 0:
                    continue
                if modulated and path.shape[1] > 2 and len(path) > 1:
                    # Split the stroke into a few weight bands and draw each with
                    # its own pen width, so the preview shows the line getting
                    # heavier where the machine will press harder or slow down.
                    for band, piece in self._weight_bands(path, layer.modulation_amount):
                        band_group = (layer.pen, round(band, 2))
                        self._group_weight[band_group] = band
                        self._pen_paths.setdefault(band_group, []).append(_polygon(piece))
                    self._flat.append((group, path))
                else:
                    self._flat.append((group, path))
                    strokes.append(_polygon(path))
                self._travel_path.moveTo(cursor)
                self._travel_path.lineTo(float(path[0][0]), float(path[0][1]))
                cursor = QPointF(float(path[-1][0]), float(path[-1][1]))
        self._bounds = self.job.stats.bounds
        self._item_bounds = self.job.drawing.item_bounds()

    @staticmethod
    def _weight_bands(path: np.ndarray, amount: float, bands: int = 4):
        """Cut a modulated stroke into runs of similar weight."""
        weights = np.clip(path[:, 2], 0.0, 1.0)
        level = np.clip((weights * bands).astype(np.int32), 0, bands - 1)
        edges = np.flatnonzero(np.diff(level)) + 1
        start = 0
        for stop in list(edges) + [len(path)]:
            piece = path[max(start - 1, 0) : stop]
            if len(piece) >= 2:
                band = float(level[start] + 0.5) / bands * max(min(amount, 1.0), 0.0)
                yield band, piece
            start = stop


    # ------------------------------------------------------------------
    def clear_temporary_transform(self) -> None:
        """Drop a drag that will never be confirmed by a new job."""
        if self._temp_offset.isNull() and abs(self._temp_scale - 1.0) < 1e-9:
            return
        self._temp_offset = QPointF(0.0, 0.0)
        self._temp_scale = 1.0
        self.update()

    def set_progress(self, fraction: float) -> None:
        fraction = max(0.0, min(1.0, float(fraction)))
        if abs(fraction - self._progress) < 1e-4:
            return
        self._progress = fraction
        self.update()

    def set_live_position(self, x: float | None, y: float | None = None) -> None:
        self._live_position = None if x is None else (float(x), float(y or 0.0))
        self.update()

    def toggle_pen(self, index: int) -> None:
        if index in self._hidden_pens:
            self._hidden_pens.discard(index)
        else:
            self._hidden_pens.add(index)
        self.update()

    def visible_pens(self) -> set[int]:
        return {group[0] for group, _ in self._flat} - self._hidden_pens

    # ------------------------------------------------------------------
    # view transform
    # ------------------------------------------------------------------
    def _bed_transform(self) -> QTransform:
        matrix = QTransform()
        matrix.translate(self._origin.x(), self._origin.y())
        matrix.scale(self._zoom, -self._zoom)
        return matrix

    def _to_world(self, point: QPointF) -> QPointF:
        return QPointF(
            (point.x() - self._origin.x()) / self._zoom,
            (self._origin.y() - point.y()) / self._zoom,
        )

    def _to_screen(self, x: float, y: float) -> QPointF:
        return QPointF(self._origin.x() + x * self._zoom, self._origin.y() - y * self._zoom)

    def fit_view(self) -> None:
        bed_w, bed_h = self.bed
        available_w = max(self.width() - 90, 60)
        available_h = max(self.height() - 90, 60)
        self._zoom = min(available_w / max(bed_w, 1.0), available_h / max(bed_h, 1.0))
        used_w = bed_w * self._zoom
        used_h = bed_h * self._zoom
        self._origin = QPointF((self.width() - used_w) / 2.0, (self.height() + used_h) / 2.0 - 6)
        self.zoom_changed.emit(self._zoom)
        self.update()

    def zoom_to_artwork(self) -> None:
        if not self._bounds:
            self.fit_view()
            return
        lo_x, lo_y, hi_x, hi_y = self._bounds
        width = max(hi_x - lo_x, 5.0)
        height = max(hi_y - lo_y, 5.0)
        available_w = max(self.width() - 120, 60)
        available_h = max(self.height() - 120, 60)
        self._zoom = min(available_w / width, available_h / height)
        centre = self._to_screen((lo_x + hi_x) / 2.0, (lo_y + hi_y) / 2.0)
        delta = QPointF(self.width() / 2.0 - centre.x(), self.height() / 2.0 - centre.y())
        self._origin += delta
        self.zoom_changed.emit(self._zoom)
        self.update()

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        old = event.oldSize()
        if old.width() <= 0 or old.height() <= 0:
            self.fit_view()
            return
        # keep whatever the user was looking at in the middle of the view
        self._origin += QPointF(
            (event.size().width() - old.width()) / 2.0,
            (event.size().height() - old.height()) / 2.0,
        )

    # ------------------------------------------------------------------
    # painting
    # ------------------------------------------------------------------
    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(theme.CANVAS))

        self._paint_bed(painter)
        self._paint_paths(painter)
        self._paint_handles(painter)
        self._paint_live(painter)
        self._paint_legend(painter)
        self._paint_stats(painter)
        self._paint_sketch(painter)
        self._paint_stale(painter)
        painter.end()

    def _paint_sketch(self, painter: QPainter) -> None:
        """The stroke being drawn right now, before it is committed."""
        if not self._sketch or self._drag_mode != "sketch":
            return
        if self.draw_tool == "dot":
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(theme.ACCENT))
            radius = max(self.library[0].width * self._zoom * 0.5, 2.0)
            for x, y in self._sketch:
                painter.drawEllipse(self._to_screen(x, y), radius, radius)
            return
        points = self._sketch_points(self._sketch[0], self._sketch[-1])
        if len(points) < 2:
            return
        pen = QPen(QColor(theme.ACCENT), 1.6)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        screen = [self._to_screen(x, y) for x, y in points]
        for a, b in zip(screen[:-1], screen[1:]):
            painter.drawLine(a, b)

    def set_stale(self, value: bool, message: str | None = None) -> None:
        """A render is on its way.

        The badge is deliberately late: most rebuilds finish in a few tens of
        milliseconds, and flashing a banner on every keystroke would be worse
        than saying nothing.  Only a render slow enough to notice announces
        itself.
        """
        if message:
            self.stale_message = message
        if value == self.stale:
            return
        self.stale = value
        if value:
            self._stale_timer.start()
        else:
            self._stale_timer.stop()
            self.update()

    def _show_stale(self) -> None:
        if self.stale:
            self.update()

    def _paint_stale(self, painter: QPainter) -> None:
        """Quiet badge saying the picture is catching up with the settings."""
        if not self.stale or self._stale_timer.isActive():
            return
        font = QFont(painter.font())
        font.setPointSizeF(10.5)
        font.setBold(True)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        width = metrics.horizontalAdvance(self.stale_message) + 30
        rect = QRectF((self.width() - width) / 2.0, 14, width, 28)

        chip = QColor(theme.PANEL)
        chip.setAlpha(236)
        painter.setPen(QPen(QColor(theme.BORDER), 1))
        painter.setBrush(QBrush(chip))
        painter.drawRoundedRect(rect, 14, 14)
        painter.setPen(QPen(QColor(theme.TEXT_MUTED)))
        painter.drawText(rect, Qt.AlignCenter, self.stale_message)

    def _paint_bed(self, painter: QPainter) -> None:
        bed_w, bed_h = self.bed
        top_left = self._to_screen(0, bed_h)
        bottom_right = self._to_screen(bed_w, 0)
        rect = QRectF(top_left, bottom_right)

        shadow = QColor(0, 0, 0, 22)
        painter.setPen(Qt.NoPen)
        painter.setBrush(shadow)
        painter.drawRoundedRect(rect.adjusted(3, 4, 3, 4), 3, 3)

        painter.setBrush(QColor(theme.BED))
        painter.setPen(QPen(QColor(theme.BED_EDGE), 1))
        painter.drawRect(rect)

        if self.show_grid and self._zoom > 0.6:
            painter.save()
            painter.setClipRect(rect)
            minor = QPen(QColor(theme.GRID_MINOR), 1)
            major = QPen(QColor(theme.GRID_MAJOR), 1)
            step = 10.0 if self._zoom > 1.6 else 50.0
            value = 0.0
            while value <= bed_w + 0.001:
                painter.setPen(major if abs(value % 50.0) < 0.01 else minor)
                x = self._to_screen(value, 0).x()
                painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
                value += step
            value = 0.0
            while value <= bed_h + 0.001:
                painter.setPen(major if abs(value % 50.0) < 0.01 else minor)
                y = self._to_screen(0, value).y()
                painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
                value += step
            painter.restore()

        # printable area
        if self.margin > 0:
            inner_tl = self._to_screen(self.margin, bed_h - self.margin)
            inner_br = self._to_screen(bed_w - self.margin, self.margin)
            pen = QPen(QColor(theme.ACCENT), 1, Qt.DashLine)
            pen.setDashPattern([4, 4])
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRectF(inner_tl, inner_br))

        # origin marker + axis labels
        painter.setPen(QPen(QColor(theme.TEXT_FAINT), 1))
        font = QFont(painter.font())
        font.setPointSizeF(9.0)
        painter.setFont(font)
        origin = self._to_screen(0, 0)
        painter.drawText(QPointF(origin.x() + 3, origin.y() + 13), "0,0")
        painter.drawText(QPointF(rect.right() - 34, rect.bottom() + 14), f"X {bed_w:.0f}")
        painter.save()
        painter.translate(rect.left() - 8, rect.top() + 34)
        painter.rotate(-90)
        painter.drawText(QPointF(0, 0), f"Y {bed_h:.0f}")
        painter.restore()

        # park position
        park = self._to_screen(self.park[0], self.park[1])
        painter.setPen(QPen(QColor(theme.TEXT_FAINT), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(park, 3, 3)

    # ------------------------------------------------------------------
    # artwork caching
    # ------------------------------------------------------------------
    def _content_key(self) -> tuple:
        """Everything that changes the *pixels* of the artwork, not its placement."""
        return (
            self.width(),
            self.height(),
            self._job_serial,
            round(self._progress, 4),
            self.show_travels,
            tuple(sorted(self._hidden_pens)),
            round(self.devicePixelRatioF(), 2),
            # a pen edit changes the drawn thickness and colour, so the cached
            # pixmap has to go even though the geometry is untouched
            tuple((round(p.width, 3), p.color) for p in self.library),
        )

    def _interacting(self) -> bool:
        return self._panning or self._drag_mode is not None or self._zoom_settle.isActive()

    def _ensure_cache(self) -> None:
        key = self._content_key()
        needs_render = self._cache is None or key != self._cache_key
        moved = self._cache_origin != self._origin or abs(self._cache_zoom - self._zoom) > 1e-9
        if not needs_render and moved and not self._interacting():
            needs_render = True  # settle down and redraw it crisply
        if not needs_render:
            return

        ratio = self.devicePixelRatioF()
        pixmap = QPixmap(max(int(self.width() * ratio), 1), max(int(self.height() * ratio), 1))
        pixmap.setDevicePixelRatio(ratio)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)
        self._render_artwork(painter)
        painter.end()

        self._cache = pixmap
        self._cache_key = key
        self._cache_origin = QPointF(self._origin)
        self._cache_zoom = self._zoom

    def _paint_paths(self, painter: QPainter) -> None:
        if not self._flat:
            self._paint_placeholder(painter)
            return

        self._ensure_cache()
        if self._cache is None:
            return

        # While the user pans, zooms or drags the artwork the cached image is
        # simply moved and scaled.  Thousands of strokes are only re-rasterised
        # once the mouse settles, which is what keeps this smooth.
        factor = self._zoom / max(self._cache_zoom, 1e-9)
        painter.save()
        chosen = self._selection_bounds()
        if abs(self._temp_scale - 1.0) > 1e-6 and chosen:
            centre = self._to_screen((chosen[0] + chosen[2]) / 2.0, (chosen[1] + chosen[3]) / 2.0)
            painter.translate(centre)
            painter.scale(self._temp_scale, self._temp_scale)
            painter.translate(-centre)
        painter.translate(
            self._origin.x() - factor * self._cache_origin.x() + self._temp_offset.x() * self._zoom,
            self._origin.y() - factor * self._cache_origin.y() - self._temp_offset.y() * self._zoom,
        )
        painter.scale(factor, factor)
        painter.drawPixmap(0, 0, self._cache)
        painter.restore()

    def _render_artwork(self, painter: QPainter) -> None:
        painter.save()
        painter.setTransform(self._bed_transform(), True)
        painter.setBrush(Qt.NoBrush)

        hairline = 1.0 / max(self._zoom, 0.01)

        if self.show_travels:
            pen = QPen(QColor(theme.TRAVEL), hairline * 0.9)
            pen.setCosmetic(False)
            painter.setPen(pen)
            painter.drawPath(self._travel_path)

        partial = self._progress < 0.999
        if partial:
            ghost = QPen(QColor(theme.GHOST), hairline)
            painter.setPen(ghost)
            for index, polygons in self._pen_paths.items():
                pen_index = index[0] if isinstance(index, tuple) else index
                if pen_index in self._hidden_pens:
                    continue
                for polygon in polygons:
                    painter.drawPolyline(polygon)
            self._ensure_done_paths()
            source = self._done_paths
        else:
            source = self._pen_paths

        for index, polygons in source.items():
            pen_index = index[0] if isinstance(index, tuple) else index
            if pen_index in self._hidden_pens:
                continue
            pen_def = self.library[pen_index]
            weight = self._group_weight.get(index, 0.0)
            width = max(pen_def.width * (1.0 + weight * 1.6), hairline)
            pen = QPen(QColor(pen_def.color), width)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(pen)
            for polygon in polygons:
                painter.drawPolyline(polygon)

        painter.restore()

    def _ensure_done_paths(self) -> None:
        target = int(round(self._progress * len(self._flat)))
        if target < self._done_count:
            self._done_paths = {}
            self._done_count = 0
        for index in range(self._done_count, target):
            group, path = self._flat[index]
            self._done_paths.setdefault(group, []).append(_polygon(path))
        self._done_count = target

    def _paint_placeholder(self, painter: QPainter) -> None:
        painter.setPen(QPen(QColor(theme.TEXT_FAINT)))
        font = QFont(painter.font())
        font.setPointSizeF(13.0)
        painter.setFont(font)
        painter.drawText(
            self.rect().adjusted(0, 0, 0, -40),
            Qt.AlignCenter,
            "Drop an image or PDF here\nor pick a source on the left",
        )

    def set_selected_item(self, index: int) -> None:
        if index == self._active_item:
            return
        self._active_item = index
        self.update()

    def _selection_bounds(self):
        """The frame follows the selected layer, or the whole sheet if it is alone."""
        if len(self._item_bounds) > 1:
            return self._item_bounds.get(self._active_item) or self._bounds
        return self._bounds

    def _artwork_rect(self) -> QRectF | None:
        chosen = self._selection_bounds()
        if not chosen:
            return None
        lo_x, lo_y, hi_x, hi_y = chosen
        centre_x = (lo_x + hi_x) / 2.0
        centre_y = (lo_y + hi_y) / 2.0
        half_w = (hi_x - lo_x) / 2.0 * self._temp_scale
        half_h = (hi_y - lo_y) / 2.0 * self._temp_scale
        top_left = self._to_screen(centre_x - half_w, centre_y + half_h)
        bottom_right = self._to_screen(centre_x + half_w, centre_y - half_h)
        rect = QRectF(top_left, bottom_right)
        return rect.translated(self._temp_offset.x(), -self._temp_offset.y())

    def _handle_points(self) -> list[QPointF]:
        rect = self._artwork_rect()
        if rect is None:
            return []
        return [rect.topLeft(), rect.topRight(), rect.bottomRight(), rect.bottomLeft()]

    def _paint_handles(self, painter: QPainter) -> None:
        if not self.show_handles or not self._selected:
            return
        rect = self._artwork_rect()
        if rect is None:
            return
        out_of_bounds = self.job is not None and self.job.stats.out_of_bounds
        colour = QColor(theme.DANGER if out_of_bounds else theme.ACCENT)
        pen = QPen(colour, 1, Qt.DashLine)
        pen.setDashPattern([3, 3])
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(rect)

        painter.setPen(QPen(colour, 1))
        painter.setBrush(QBrush(QColor("#FFFFFF")))
        for index, point in enumerate(self._handle_points()):
            size = HANDLE_SIZE + (2 if self._hover_handle == index else 0)
            painter.drawRect(QRectF(point.x() - size / 2, point.y() - size / 2, size, size))

        # size label
        chosen = self._selection_bounds()
        if chosen:
            lo_x, lo_y, hi_x, hi_y = chosen
            width = (hi_x - lo_x) * self._temp_scale
            height = (hi_y - lo_y) * self._temp_scale
            text = f"{width:.1f} × {height:.1f} mm"
            painter.setPen(QPen(colour))
            font = QFont(painter.font())
            font.setPointSizeF(9.5)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(QPointF(rect.left(), rect.top() - 7), text)

    def _paint_live(self, painter: QPainter) -> None:
        if not self._live_position:
            return
        point = self._to_screen(*self._live_position)
        painter.setPen(QPen(QColor(theme.DANGER), 1.4))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(point, 6, 6)
        painter.drawLine(QPointF(point.x() - 10, point.y()), QPointF(point.x() + 10, point.y()))
        painter.drawLine(QPointF(point.x(), point.y() - 10), QPointF(point.x(), point.y() + 10))

    def _paint_legend(self, painter: QPainter) -> None:
        self._legend_rects = []
        if self.job is None or self.job.drawing.is_empty():
            return
        used = self.job.drawing.used_pens()
        if len(used) <= 1 and not self._hidden_pens:
            return

        font = QFont(painter.font())
        font.setPointSizeF(10.0)
        painter.setFont(font)
        metrics = painter.fontMetrics()

        rows = []
        for pen_index in used:
            pen = self.library[pen_index]
            stats = self.job.stats.per_pen.get(pen_index, {})
            label = f"{pen.name}  ·  {stats.get('length', 0.0)/1000:.1f} m"
            rows.append((pen_index, label))
        width = max(metrics.horizontalAdvance(text) for _, text in rows) + 44
        height = len(rows) * 20 + 14
        box = QRectF(14, 14, width, height)

        chip = QColor(theme.PANEL)
        chip.setAlpha(238)
        painter.setPen(QPen(QColor(theme.BORDER), 1))
        painter.setBrush(QBrush(chip))
        painter.drawRoundedRect(box, 7, 7)

        y = box.top() + 7
        for pen_index, label in rows:
            row_rect = QRectF(box.left() + 4, y, box.width() - 8, 19)
            self._legend_rects.append((row_rect, pen_index))
            hidden = pen_index in self._hidden_pens
            colour = QColor(self.library[pen_index].color)
            if hidden:
                colour.setAlpha(70)
            painter.setPen(QPen(QColor(theme.BORDER_STRONG), 1))
            painter.setBrush(QBrush(colour))
            painter.drawEllipse(QPointF(row_rect.left() + 12, row_rect.center().y()), 6, 6)
            painter.setPen(QPen(QColor(theme.TEXT_FAINT if hidden else theme.TEXT)))
            painter.drawText(
                QRectF(row_rect.left() + 26, row_rect.top(), row_rect.width() - 26, row_rect.height()),
                Qt.AlignVCenter | Qt.AlignLeft,
                label,
            )
            y += 20

    def _paint_stats(self, painter: QPainter) -> None:
        if not self.show_stats or self.job is None or self.job.drawing.is_empty():
            return
        stats = self.job.stats
        lines = [
            f"{stats.path_count:,} strokes   ·   {stats.draw_length/1000:.1f} m drawn",
            f"{stats.travel_length/1000:.1f} m travel   ·   {format_duration(stats.estimated_seconds)}",
        ]
        if stats.bounds:
            lo_x, lo_y, hi_x, hi_y = stats.bounds
            lines.append(
                f"{hi_x-lo_x:.1f} × {hi_y-lo_y:.1f} mm at X {lo_x:.1f}  Y {lo_y:.1f}"
            )

        font = QFont(painter.font())
        font.setPointSizeF(10.0)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        width = max(metrics.horizontalAdvance(line) for line in lines) + 22
        height = len(lines) * 17 + 14
        box = QRectF(14, self.height() - height - 14, width, height)

        chip = QColor(theme.PANEL)
        chip.setAlpha(238)
        painter.setPen(QPen(QColor(theme.BORDER), 1))
        painter.setBrush(QBrush(chip))
        painter.drawRoundedRect(box, 7, 7)

        painter.setPen(QPen(QColor(theme.TEXT_MUTED)))
        y = box.top() + 8
        for index, line in enumerate(lines):
            painter.setPen(QPen(QColor(theme.TEXT if index == 0 else theme.TEXT_MUTED)))
            painter.drawText(QRectF(box.left() + 11, y, box.width(), 17), Qt.AlignVCenter | Qt.AlignLeft, line)
            y += 17

        if stats.out_of_bounds:
            warning = "⚠  Sticks out past the bed"
            w = metrics.horizontalAdvance(warning) + 22
            rect = QRectF(box.left(), box.top() - 30, w, 24)
            alarm = QColor(theme.PANEL)
            alarm.setAlpha(242)
            painter.setPen(QPen(QColor(theme.DANGER), 1))
            painter.setBrush(QBrush(alarm))
            painter.drawRoundedRect(rect, 6, 6)
            painter.setPen(QPen(QColor(theme.DANGER)))
            painter.drawText(rect, Qt.AlignCenter, warning)

    # ------------------------------------------------------------------
    # interaction
    # ------------------------------------------------------------------
    # ---- zooming and panning -------------------------------------------
    def zoom_at(self, factor: float, anchor: QPointF | None = None) -> None:
        anchor = anchor or QPointF(self.width() / 2.0, self.height() / 2.0)
        before = self._to_world(anchor)
        self._zoom = max(MIN_ZOOM, min(self._zoom * factor, MAX_ZOOM))
        after = self._to_world(anchor)
        self._origin += QPointF(
            (after.x() - before.x()) * self._zoom, -(after.y() - before.y()) * self._zoom
        )
        self.zoom_changed.emit(self._zoom)
        self.update()

    def zoom_in(self) -> None:
        self._zoom_settle.start()
        self.zoom_at(1.25)

    def zoom_out(self) -> None:
        self._zoom_settle.start()
        self.zoom_at(1 / 1.25)

    @property
    def zoom(self) -> float:
        return self._zoom

    def wheelEvent(self, event):  # noqa: N802
        delta_y = event.angleDelta().y()
        delta_x = event.angleDelta().x()
        modifiers = event.modifiers()

        # shift or alt scrolls the view instead of zooming, which is what a
        # trackpad user expects when they are trying to look around
        if modifiers & (Qt.ShiftModifier | Qt.AltModifier):
            # alt scrolls sideways, shift scrolls the view; both reuse the
            # cached artwork instead of re-rasterising on every tick
            self._zoom_settle.start()
            if modifiers & Qt.AltModifier:
                self._origin += QPointF(float(delta_y + delta_x), 0.0)
            else:
                self._origin += QPointF(float(delta_x), float(delta_y))
            self.update()
            return
        if delta_y == 0:
            if delta_x:
                self._origin += QPointF(float(delta_x), 0.0)
                self.update()
            return
        self._zoom_settle.start()
        self.zoom_at(1.0015 ** delta_y, event.position())

    def event(self, event):
        """Trackpad pinch zoom."""
        if event.type() == QEvent.NativeGesture:
            if event.gestureType() == Qt.ZoomNativeGesture:
                self._zoom_settle.start()
                self.zoom_at(1.0 + event.value(), event.position())
                return True
            if event.gestureType() == Qt.SmartZoomNativeGesture:
                self.zoom_to_artwork()
                return True
        return super().event(event)

    # ------------------------------------------------------------------
    # drawing tools
    # ------------------------------------------------------------------
    def set_draw_tool(self, tool: str) -> None:
        self.draw_tool = tool or ""
        self._sketch = []
        self.setCursor(Qt.CrossCursor if self.draw_tool else Qt.ArrowCursor)
        self.update()

    def _sketch_points(self, start, current) -> list[tuple[float, float]]:
        """Turn the drag into the shape the current tool makes."""
        x0, y0 = start
        x1, y1 = current
        if self.draw_tool == "line":
            return [(x0, y0), (x1, y1)]
        if self.draw_tool == "rect":
            return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
        if self.draw_tool == "ellipse":
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            rx, ry = abs(x1 - x0) / 2.0, abs(y1 - y0) / 2.0
            angles = np.linspace(0.0, 2 * math.pi, 64)
            return [(cx + rx * math.cos(a), cy + ry * math.sin(a)) for a in angles]
        return self._sketch

    def mousePressEvent(self, event):  # noqa: N802
        position = event.position()
        if self.draw_tool and event.button() == Qt.LeftButton and not self._space_held:
            world = self._to_world(position)
            self._sketch = [(world.x(), world.y())]
            self._drag_mode = "sketch"
            return
        if event.button() == Qt.MiddleButton or self._space_held or (
            event.button() == Qt.LeftButton and event.modifiers() & Qt.AltModifier
        ):
            self._panning = True
            self._pan_anchor = position
            self.setCursor(Qt.ClosedHandCursor)
            return
        if event.button() == Qt.LeftButton:
            for rect, pen_index in self._legend_rects:
                if rect.contains(position):
                    self.toggle_pen(pen_index)
                    return
            handle = self._handle_at(position)
            if handle is not None:
                self._drag_mode = "scale"
                self._drag_start = position
                self._selected = True
                return
            # clicking a different layer picks it up instead of moving this one
            world = self._to_world(position)
            for index, box in self._item_bounds.items():
                if index == self._active_item or box is None:
                    continue
                if box[0] - 1 <= world.x() <= box[2] + 1 and box[1] - 1 <= world.y() <= box[3] + 1:
                    inside_current = self._artwork_rect()
                    if inside_current is None or not inside_current.contains(position):
                        self.item_selected.emit(index)
                        return

            rect = self._artwork_rect()
            if rect and rect.adjusted(-4, -4, 4, 4).contains(position):
                self._drag_mode = "move"
                self._drag_start = position
                self._selected = True
                self.setCursor(Qt.SizeAllCursor)
                return
            # a click on empty bed just deselects; clicking the artwork again
            # picks it back up, which is what every other editor does
            if self._selected:
                self._selected = False
                self.selection_changed.emit(False)
                self.update()

    def mouseMoveEvent(self, event):  # noqa: N802
        position = event.position()
        world = self._to_world(position)
        self.hover_position.emit(world.x(), world.y())

        if self._panning:
            delta = position - self._pan_anchor
            self._origin += delta
            self._pan_anchor = position
            self.update()
            return

        if self._drag_mode == "sketch":
            world = self._to_world(position)
            if self.draw_tool in ("free", "dot"):
                last = self._sketch[-1]
                if math.hypot(world.x() - last[0], world.y() - last[1]) * self._zoom > 2.0:
                    self._sketch.append((world.x(), world.y()))
            else:
                self._sketch = [self._sketch[0], (world.x(), world.y())]
            self.update()
            return

        if self._drag_mode == "move":
            delta = position - self._drag_start
            self._temp_offset = QPointF(delta.x() / self._zoom, -delta.y() / self._zoom)
            self.update()
            return

        if self._drag_mode == "scale":
            rect = self._artwork_rect()
            if rect is None:
                return
            centre = rect.center()
            start_distance = math.hypot(self._drag_start.x() - centre.x(), self._drag_start.y() - centre.y())
            now_distance = math.hypot(position.x() - centre.x(), position.y() - centre.y())
            if start_distance > 4:
                self._temp_scale = max(0.05, min(now_distance / start_distance, 20.0))
                self.update()
            return

        handle = self._handle_at(position)
        if handle != self._hover_handle:
            self._hover_handle = handle
            self.update()
        if handle is not None:
            self.setCursor(Qt.SizeFDiagCursor if handle in (0, 2) else Qt.SizeBDiagCursor)
        else:
            rect = self._artwork_rect()
            inside = rect is not None and rect.contains(position)
            self.setCursor(Qt.SizeAllCursor if inside and self._selected else Qt.ArrowCursor)

    def mouseReleaseEvent(self, event):  # noqa: N802
        if self._drag_mode == "sketch":
            self._drag_mode = None
            if self.draw_tool == "dot":
                # one click, one tap: a single point, which the G-code turns
                # into pen down / pen up with no move in between
                x, y = self._sketch[0]
                self.stroke_drawn.emit([(float(x), float(y))])
            elif len(self._sketch) >= 2:
                points = self._sketch_points(self._sketch[0], self._sketch[-1])
                if len(points) >= 2:
                    self.stroke_drawn.emit([(float(x), float(y)) for x, y in points])
            self._sketch = []
            self.update()
            return

        if self._panning:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            self.update()
            return
        if self._drag_mode == "move":
            offset = self._temp_offset
            self._drag_mode = None
            self.setCursor(Qt.ArrowCursor)
            if abs(offset.x()) > 0.01 or abs(offset.y()) > 0.01:
                # the offset stays applied until the re-placed job arrives,
                # otherwise the artwork visibly snaps back for a moment
                self.move_committed.emit(offset.x(), offset.y())
            else:
                self._temp_offset = QPointF(0.0, 0.0)
            self.update()
            return
        if self._drag_mode == "scale":
            factor = self._temp_scale
            self._drag_mode = None
            if abs(factor - 1.0) > 0.005:
                self.scale_committed.emit(factor)
            else:
                self._temp_scale = 1.0
            self.update()

    def mouseDoubleClickEvent(self, event):  # noqa: N802
        self.zoom_to_artwork()

    def keyPressEvent(self, event):  # noqa: N802
        step = 0.1 if event.modifiers() & Qt.ShiftModifier else 1.0
        moves = {
            Qt.Key_Left: (-step, 0.0),
            Qt.Key_Right: (step, 0.0),
            Qt.Key_Up: (0.0, step),
            Qt.Key_Down: (0.0, -step),
        }
        if event.key() in moves and self._selected:
            dx, dy = moves[event.key()]
            self.move_committed.emit(dx, dy)
            return
        if event.key() in (Qt.Key_F, Qt.Key_0):
            self.fit_view()
            return
        if event.key() in (Qt.Key_Plus, Qt.Key_Equal):
            self.zoom_in()
            return
        if event.key() == Qt.Key_Minus:
            self.zoom_out()
            return
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_held = True
            self.setCursor(Qt.OpenHandCursor)
            return
        super().keyPressEvent(event)

    def focusOutEvent(self, event):  # noqa: N802
        # otherwise a Space pressed here and released elsewhere leaves the
        # canvas stuck in pan mode, and dragging the artwork stops working
        self._space_held = False
        self.setCursor(Qt.ArrowCursor)
        super().focusOutEvent(event)

    def keyReleaseEvent(self, event):  # noqa: N802
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_held = False
            self.setCursor(Qt.ArrowCursor)
            return
        super().keyReleaseEvent(event)

    def _handle_at(self, position: QPointF) -> int | None:
        if not self.show_handles or not self._selected:
            return None
        for index, point in enumerate(self._handle_points()):
            box = QRectF(point.x() - HANDLE_SIZE, point.y() - HANDLE_SIZE, HANDLE_SIZE * 2, HANDLE_SIZE * 2)
            if box.contains(position):
                return index
        return None

    # ------------------------------------------------------------------
    def dragEnterEvent(self, event):  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):  # noqa: N802
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            self.files_dropped.emit(paths)
            event.acceptProposedAction()
