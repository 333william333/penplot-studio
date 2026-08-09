"""Text -> pen paths.

Two very different kinds of type are supported:

* the built-in **single-stroke font**, where the pen follows the centre line of
  each letter - the right choice for small text and for pencils;
* any **installed TrueType/OpenType font**, where the pen traces the glyph
  outline, optionally hatch-filled so big letters look solid.

Output is in millimetres with Y pointing down, matching every other source.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QPointF
from PySide6.QtGui import QFont, QFontDatabase, QFontMetricsF, QPainterPath

from . import strokefont
from . import styles
from .drawing import Layer

__all__ = ["TextSettings", "build_text", "available_families", "STROKE_FONTS"]

STROKE_FONTS = {
    "Plotter Sans (single line)": "sans",
    "Plotter Mono (single line)": "mono",
}

PEN_MODES = {
    "single": "All text with one pen",
    "line": "New pen for every line",
    "word": "New pen for every word",
    "char": "New pen for every character",
}


@dataclass
class TextSettings:
    text: str = "Hello Ender 3"
    font: str = "Plotter Sans (single line)"
    style_name: str = "Regular"
    size_mm: float = 18.0            # cap height
    letter_spacing: float = 0.0      # extra mm between glyphs
    word_spacing: float = 0.0        # extra mm on spaces
    line_spacing: float = 1.6        # multiple of size_mm
    align: str = "left"              # left | center | right
    fill: bool = False               # hatch-fill outline fonts
    fill_spacing: float = 0.45       # mm between fill lines
    fill_angle: float = 45.0
    keep_outline: bool = True        # draw the outline as well as the fill
    double_stroke: bool = False      # trace stroke fonts twice for bolder lines
    double_offset: float = 0.25      # mm offset of the second pass
    pen_mode: str = "single"
    pen: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def is_stroke_font(self) -> bool:
        return self.font in STROKE_FONTS


def available_families() -> list[str]:
    """Single-stroke fonts first, then everything installed on the machine."""
    families = sorted(QFontDatabase.families())
    return list(STROKE_FONTS.keys()) + families


def styles_for(family: str) -> list[str]:
    if family in STROKE_FONTS:
        return ["Regular"]
    found = QFontDatabase.styles(family)
    return list(found) if found else ["Regular"]


# --------------------------------------------------------------------------
# single-stroke rendering
# --------------------------------------------------------------------------
def _stroke_font_line(text: str, settings: TextSettings, monospace: bool) -> tuple[list[np.ndarray], list[int], float]:
    """Return (strokes, group index per stroke, line width in mm)."""
    scale = settings.size_mm / strokefont.CAP
    tracking = settings.letter_spacing / scale if scale else 0.0
    space_extra = settings.word_spacing / scale if scale else 0.0
    fixed_advance = strokefont.GLYPHS["M"][0] if monospace else None

    out: list[np.ndarray] = []
    groups: list[int] = []
    cursor = 0.0
    for index, ch in enumerate(text):
        advance, glyph_strokes = strokefont.glyph(ch)
        if fixed_advance is not None:
            offset = (fixed_advance - advance) / 2.0
        else:
            offset = 0.0
        for stroke in glyph_strokes:
            moved = stroke.copy()
            moved[:, 0] += cursor + offset
            out.append(moved)
            groups.append(index)
        cursor += (fixed_advance if fixed_advance is not None else advance) + tracking
        if ch == " ":
            cursor += space_extra
    width = max(cursor - tracking, 0.0)
    return out, groups, width * scale


# --------------------------------------------------------------------------
# outline (TrueType) rendering
# --------------------------------------------------------------------------
_REFERENCE_PIXEL_SIZE = 512


def _qt_font(settings: TextSettings) -> tuple[QFont, float]:
    font = QFontDatabase.font(settings.font, settings.style_name, _REFERENCE_PIXEL_SIZE)
    font.setPixelSize(_REFERENCE_PIXEL_SIZE)
    metrics = QFontMetricsF(font)
    cap = metrics.capHeight()
    if cap <= 1.0:
        cap = metrics.ascent() * 0.72
    return font, float(cap)


def _outline_line(text: str, settings: TextSettings, font: QFont, unit: float) -> tuple[list[np.ndarray], list[int], float]:
    """Build glyph outlines for one line.  *unit* = mm per font pixel."""
    metrics = QFontMetricsF(font)
    out: list[np.ndarray] = []
    groups: list[int] = []
    cursor = 0.0
    extra = settings.letter_spacing / unit if unit else 0.0
    space_extra = settings.word_spacing / unit if unit else 0.0

    for index, ch in enumerate(text):
        advance = metrics.horizontalAdvance(ch)
        if ch.strip():
            path = QPainterPath()
            path.addText(QPointF(cursor, 0.0), font, ch)
            for polygon in path.toSubpathPolygons():
                pts = np.array([[p.x(), p.y()] for p in polygon], dtype=np.float64)
                if len(pts) >= 3:
                    if not np.allclose(pts[0], pts[-1]):
                        pts = np.vstack([pts, pts[:1]])
                    out.append(pts * unit)
                    groups.append(index)
        cursor += advance + extra
        if ch == " ":
            cursor += space_extra
    width = max(cursor - extra, 0.0) * unit
    return out, groups, width


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------
def build_text(settings: TextSettings, pen_count: int = 1) -> list[Layer]:
    """Render *settings.text* into millimetre paths, split per pen."""
    lines = settings.text.split("\n") if settings.text else []
    if not any(line.strip() for line in lines):
        return []

    line_height = settings.size_mm * max(settings.line_spacing, 0.4)
    rendered: list[tuple[list[np.ndarray], list[int], float, int]] = []

    if settings.is_stroke_font:
        monospace = STROKE_FONTS[settings.font] == "mono"
        scale = settings.size_mm / strokefont.CAP
        for row, line in enumerate(lines):
            strokes, groups, width = _stroke_font_line(line, settings, monospace)
            # font space (Y up, cap = 100) -> millimetres with Y down
            scaled = []
            for s in strokes:
                p = s.copy()
                p[:, 0] *= scale
                p[:, 1] *= -scale
                p[:, 1] += row * line_height
                scaled.append(p)
            rendered.append((scaled, groups, width, row))
    else:
        font, cap_pixels = _qt_font(settings)
        unit = settings.size_mm / cap_pixels if cap_pixels else 0.01
        for row, line in enumerate(lines):
            outlines, groups, width = _outline_line(line, settings, font, unit)
            shifted = []
            for p in outlines:
                q = p.copy()
                q[:, 1] += row * line_height
                shifted.append(q)
            rendered.append((shifted, groups, width, row))

    widest = max((width for _, _, width, _ in rendered), default=0.0)

    # horizontal alignment
    aligned: list[tuple[np.ndarray, int, int]] = []
    for strokes, groups, width, row in rendered:
        if settings.align == "center":
            dx = (widest - width) / 2.0
        elif settings.align == "right":
            dx = widest - width
        else:
            dx = 0.0
        for stroke, group in zip(strokes, groups):
            moved = stroke.copy()
            moved[:, 0] += dx
            aligned.append((moved, group, row))

    if not aligned:
        return []

    # optional hatch fill for outline fonts
    filled: list[tuple[np.ndarray, int, int]] = []
    if settings.fill and not settings.is_stroke_font:
        by_key: dict[tuple[int, int], list[np.ndarray]] = {}
        for stroke, group, row in aligned:
            by_key.setdefault((row, group), []).append(stroke)
        for (row, group), polys in by_key.items():
            for seg in styles.hatch_polygons(polys, max(settings.fill_spacing, 0.05), settings.fill_angle):
                filled.append((seg, group, row))
        if settings.keep_outline:
            filled.extend(aligned)
        aligned = filled

    # stroke fonts can be emboldened with a second offset pass
    if settings.double_stroke and settings.is_stroke_font:
        offset = settings.double_offset
        doubled = list(aligned)
        for stroke, group, row in aligned:
            shifted = stroke.copy()
            shifted[:, 0] += offset
            doubled.append((shifted, group, row))
        aligned = doubled

    # assign pens
    pen_count = max(1, pen_count)
    buckets: dict[int, list[np.ndarray]] = {}
    word_index_cache: dict[int, dict[int, int]] = {}
    if settings.pen_mode == "word":
        for row, line in enumerate(lines):
            mapping: dict[int, int] = {}
            word = 0
            in_word = False
            for i, ch in enumerate(line):
                if ch.isspace():
                    in_word = False
                else:
                    if not in_word:
                        word += 1
                        in_word = True
                mapping[i] = word - 1
            word_index_cache[row] = mapping

    for stroke, group, row in aligned:
        if settings.pen_mode == "line":
            slot = row
        elif settings.pen_mode == "char":
            slot = group
        elif settings.pen_mode == "word":
            slot = word_index_cache.get(row, {}).get(group, 0)
        else:
            slot = 0
        pen = settings.pen if settings.pen_mode == "single" else slot % pen_count
        buckets.setdefault(pen, []).append(stroke)

    return [Layer(pen=pen, paths=paths, name="Text") for pen, paths in sorted(buckets.items())]
