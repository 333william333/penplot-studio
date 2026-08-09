"""Render every drawing technique to a contact sheet so the results can be judged by eye."""

from __future__ import annotations

import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
from PySide6.QtCore import QPointF, QRectF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from penplot.core import geometry as geo, raster, techniques  # noqa: E402

TILE = 420
LABEL = 34
COLUMNS = 5


def draw_paths(paths, size_px: int, source_px: int, pen_px: float) -> QImage:
    image = QImage(size_px, size_px, QImage.Format_RGB32)
    image.fill(QColor("#FFFFFF"))
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    scale = size_px / max(source_px, 1)
    painter.scale(scale, scale)
    pen = QPen(QColor("#111418"), max(pen_px, 0.8))
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)

    path = QPainterPath()
    for points in paths:
        if len(points) == 1:
            x, y = float(points[0][0]), float(points[0][1])
            path.moveTo(x, y)
            path.lineTo(x + 0.01, y)
            continue
        path.moveTo(float(points[0][0]), float(points[0][1]))
        for point in points[1:]:
            path.lineTo(float(point[0]), float(point[1]))
    painter.drawPath(path)
    painter.end()
    return image


def main() -> int:
    QApplication.instance() or QApplication(sys.argv[:1])
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "samples", "portrait.png")
    name = os.path.splitext(os.path.basename(source))[0]

    rgb = raster.load_rgb(source)
    prepared = raster.prepare(rgb, detail=700, auto_levels=False)
    gray = raster.to_gray(prepared)
    height, width = gray.shape
    long_edge = max(width, height)

    # pretend the drawing is 160 mm wide with a 0.5 mm pen
    ctx = techniques.Context(px_per_mm=long_edge / 160.0, pen_width=0.5)
    pen_px = ctx.pen_px

    keys = list(techniques.REGISTRY)
    rows = math.ceil(len(keys) / COLUMNS)
    sheet = QImage(COLUMNS * TILE, rows * (TILE + LABEL), QImage.Format_RGB32)
    sheet.fill(QColor("#F2F4F6"))
    painter = QPainter(sheet)
    font = QFont()
    font.setPointSize(15)
    font.setBold(True)
    painter.setFont(font)

    for index, key in enumerate(keys):
        started = time.perf_counter()
        paths = techniques.render(key, gray, None, ctx)
        elapsed = time.perf_counter() - started
        tile = draw_paths(paths, TILE - 8, long_edge, pen_px * (TILE - 8) / long_edge)

        column = index % COLUMNS
        row = index // COLUMNS
        x = column * TILE
        y = row * (TILE + LABEL)
        painter.drawImage(x + 4, y + LABEL, tile)
        painter.setPen(QColor("#1F2933"))
        length = sum(geo.path_length(p) for p in paths) / ctx.px_per_mm / 1000.0
        painter.drawText(
            QRectF(x + 6, y + 2, TILE - 12, LABEL - 4),
            Qt.AlignVCenter | Qt.AlignLeft,
            f"{techniques.REGISTRY[key].label}   ({len(paths):,} strokes · {length:.1f} m · {elapsed*1000:.0f} ms)",
        )
    painter.end()

    out = os.path.join(root, "build", "screens", f"techniques-{name}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sheet.save(out)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
