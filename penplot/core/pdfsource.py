"""PDF import.

Two ways in:

* **Vector** - read the real line work out of the page (lines, curves,
  rectangles) plus the text, and plot it directly.  Crisp at any size and it
  keeps the page's colours, which map onto pens.
* **Render** - rasterise the page at a chosen DPI and run it through the normal
  image styles.  Use this for scans, photos and complicated artwork.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pymupdf as fitz
import numpy as np

from . import strokefont

__all__ = ["PdfDocument", "PdfSettings", "VectorArt", "PT_TO_MM"]

PT_TO_MM = 25.4 / 72.0

TEXT_MODES = {
    "outline": "Outline (follow the real font)",
    "stroke": "Single-stroke font (best for pens)",
    "skip": "Skip text",
}


@dataclass
class VectorArt:
    """Coloured polylines in millimetres, Y down, origin at the page corner."""

    paths: list[np.ndarray] = field(default_factory=list)
    colors: list[tuple[float, float, float]] = field(default_factory=list)
    widths: list[float] = field(default_factory=list)
    closed: list[bool] = field(default_factory=list)

    def add(self, path: np.ndarray, color, width: float, closed: bool = False) -> None:
        if len(path) < 2:
            return
        self.paths.append(path)
        self.colors.append(tuple(float(c) for c in (color or (0.0, 0.0, 0.0))))
        self.widths.append(float(width or 0.0))
        self.closed.append(bool(closed))

    def extend(self, other: "VectorArt") -> None:
        self.paths.extend(other.paths)
        self.colors.extend(other.colors)
        self.widths.extend(other.widths)
        self.closed.extend(other.closed)

    def __len__(self) -> int:
        return len(self.paths)


@dataclass
class PdfSettings:
    page: int = 0
    mode: str = "vector"          # vector | render
    dpi: int = 300                # render mode
    text_mode: str = "outline"
    keep_fills: bool = True       # outline filled shapes too
    fill_hatch: float = 0.0       # >0 hatches filled shapes, mm spacing
    min_length: float = 0.15      # mm - drops dust
    flatness: float = 0.08        # mm - curve flattening tolerance
    crop_to_content: bool = False


def _flatten_cubic(p0, p1, p2, p3, tolerance: float) -> np.ndarray:
    """Adaptive-ish cubic Bezier flattening based on control polygon length."""
    poly = (
        math.dist(p0, p1) + math.dist(p1, p2) + math.dist(p2, p3)
    )
    steps = int(max(4, min(80, math.sqrt(poly / max(tolerance, 1e-3)) * 2)))
    t = np.linspace(0.0, 1.0, steps)[:, None]
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    p3 = np.asarray(p3, dtype=np.float64)
    mt = 1.0 - t
    return (mt**3) * p0 + 3 * (mt**2) * t * p1 + 3 * mt * (t**2) * p2 + (t**3) * p3


class PdfDocument:
    """Thin wrapper around a fitz.Document with the bits the app needs."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.doc = fitz.open(file_path)

    def close(self) -> None:
        try:
            self.doc.close()
        except Exception:
            pass

    @property
    def page_count(self) -> int:
        return self.doc.page_count

    def page_size_mm(self, index: int) -> tuple[float, float]:
        rect = self.doc[max(0, min(index, self.page_count - 1))].rect
        return (rect.width * PT_TO_MM, rect.height * PT_TO_MM)

    # ---------------- raster ------------------------------------------------
    def render(self, index: int, dpi: int) -> np.ndarray:
        page = self.doc[max(0, min(index, self.page_count - 1))]
        pix = page.get_pixmap(dpi=int(max(30, min(dpi, 1200))), alpha=False, colorspace=fitz.csRGB)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        return np.ascontiguousarray(arr.astype(np.float32) / 255.0)

    # ---------------- vector ------------------------------------------------
    def vectors(self, settings: PdfSettings) -> tuple[VectorArt, list[str]]:
        page = self.doc[max(0, min(settings.page, self.page_count - 1))]
        art = VectorArt()
        warnings: list[str] = []
        tolerance_pt = max(settings.flatness, 0.01) / PT_TO_MM

        for drawing in page.get_drawings():
            kind = drawing.get("type", "s")
            stroke_color = drawing.get("color")
            fill_color = drawing.get("fill")
            color = stroke_color if stroke_color is not None else fill_color
            if color is None:
                color = (0.0, 0.0, 0.0)
            if kind == "f" and not settings.keep_fills:
                continue
            width = float(drawing.get("width") or 0.0)
            closed_flag = bool(drawing.get("closePath"))

            current: list[np.ndarray] = []

            def flush(force_closed: bool = False) -> None:
                nonlocal current
                if not current:
                    return
                pts = np.vstack(current)
                if len(pts) >= 2:
                    if force_closed and not np.allclose(pts[0], pts[-1]):
                        pts = np.vstack([pts, pts[:1]])
                    art.add(pts * PT_TO_MM, color, width * PT_TO_MM, force_closed)
                current = []

            for item in drawing.get("items", []):
                op = item[0]
                if op == "l":
                    p1 = np.array([item[1].x, item[1].y], dtype=np.float64)
                    p2 = np.array([item[2].x, item[2].y], dtype=np.float64)
                    if current and np.allclose(current[-1][-1], p1, atol=1e-6):
                        current.append(p2[None, :])
                    else:
                        flush()
                        current = [p1[None, :], p2[None, :]]
                elif op == "c":
                    pts = _flatten_cubic(
                        (item[1].x, item[1].y),
                        (item[2].x, item[2].y),
                        (item[3].x, item[3].y),
                        (item[4].x, item[4].y),
                        tolerance_pt,
                    )
                    if current and np.allclose(current[-1][-1], pts[0], atol=1e-6):
                        current.append(pts[1:])
                    else:
                        flush()
                        current = [pts]
                elif op == "re":
                    flush()
                    rect = item[1]
                    box = np.array(
                        [
                            [rect.x0, rect.y0],
                            [rect.x1, rect.y0],
                            [rect.x1, rect.y1],
                            [rect.x0, rect.y1],
                            [rect.x0, rect.y0],
                        ],
                        dtype=np.float64,
                    )
                    art.add(box * PT_TO_MM, color, width * PT_TO_MM, True)
                elif op == "qu":
                    flush()
                    quad = item[1]
                    pts = np.array(
                        [
                            [quad.ul.x, quad.ul.y],
                            [quad.ur.x, quad.ur.y],
                            [quad.lr.x, quad.lr.y],
                            [quad.ll.x, quad.ll.y],
                            [quad.ul.x, quad.ul.y],
                        ],
                        dtype=np.float64,
                    )
                    art.add(pts * PT_TO_MM, color, width * PT_TO_MM, True)
            flush(closed_flag)

        if settings.text_mode != "skip":
            try:
                art.extend(self._text_art(page, settings))
            except Exception as exc:  # pragma: no cover - defensive
                warnings.append(f"Could not read PDF text: {exc}")

        if not len(art):
            warnings.append(
                "No vector content found on this page - switch to “Render page” mode."
            )
        return art, warnings

    # ---------------- text --------------------------------------------------
    def _text_art(self, page, settings: PdfSettings) -> VectorArt:
        art = VectorArt()
        data = page.get_text("rawdict")
        use_stroke_font = settings.text_mode == "stroke"

        outline_cache: dict[tuple[str, bool, bool], object] = {}

        for block in data.get("blocks", []):
            if block.get("type", 0) != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    color_int = span.get("color", 0)
                    color = (
                        ((color_int >> 16) & 255) / 255.0,
                        ((color_int >> 8) & 255) / 255.0,
                        (color_int & 255) / 255.0,
                    )
                    size = float(span.get("size", 10.0))
                    font_name = str(span.get("font", ""))
                    bold = "bold" in font_name.lower() or "black" in font_name.lower()
                    italic = "italic" in font_name.lower() or "oblique" in font_name.lower()
                    for char in span.get("chars", []):
                        ch = char.get("c", "")
                        if not ch.strip():
                            continue
                        origin = char.get("origin", (0.0, 0.0))
                        bbox = char.get("bbox", None)
                        if use_stroke_font:
                            art.extend(_stroke_glyph(ch, origin, size, color))
                        else:
                            art.extend(
                                _outline_glyph(ch, origin, bbox, size, font_name, bold, italic, color, outline_cache)
                            )
        return art


def _stroke_glyph(ch: str, origin, size: float, color) -> VectorArt:
    art = VectorArt()
    advance, strokes = strokefont.glyph(ch)
    scale = (size * 0.70) / strokefont.CAP  # cap height ~70% of the point size
    ox, oy = origin[0], origin[1]
    for stroke in strokes:
        p = stroke.copy()
        p[:, 0] = ox + p[:, 0] * scale
        p[:, 1] = oy - p[:, 1] * scale
        art.add(p * PT_TO_MM, color, 0.0)
    return art


def _outline_glyph(ch, origin, bbox, size, font_name, bold, italic, color, cache) -> VectorArt:
    """Trace the glyph outline using the closest installed font."""
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QFont, QPainterPath

    art = VectorArt()
    key = (_substitute_family(font_name), bold, italic)
    font = cache.get(key)
    if font is None:
        font = QFont(key[0])
        font.setPixelSize(256)
        font.setBold(bold)
        font.setItalic(italic)
        cache[key] = font

    path = QPainterPath()
    path.addText(QPointF(0.0, 0.0), font, ch)
    polygons = path.toSubpathPolygons()
    if not polygons:
        return art

    scale = size / 256.0
    ox, oy = origin[0], origin[1]
    for polygon in polygons:
        pts = np.array([[p.x(), p.y()] for p in polygon], dtype=np.float64)
        if len(pts) < 3:
            continue
        pts *= scale
        pts[:, 0] += ox
        pts[:, 1] += oy
        if not np.allclose(pts[0], pts[-1]):
            pts = np.vstack([pts, pts[:1]])
        art.add(pts * PT_TO_MM, color, 0.0, True)
    return art


_FONT_SUBSTITUTIONS = (
    ("times", "Times New Roman"),
    ("serif", "Times New Roman"),
    ("georgia", "Georgia"),
    ("garamond", "Times New Roman"),
    ("courier", "Courier New"),
    ("mono", "Menlo"),
    ("arial", "Arial"),
    ("helvetica", "Helvetica"),
    ("calibri", "Helvetica"),
    ("verdana", "Verdana"),
)


def _substitute_family(font_name: str) -> str:
    lowered = font_name.lower()
    for needle, family in _FONT_SUBSTITUTIONS:
        if needle in lowered:
            return family
    return "Helvetica"
