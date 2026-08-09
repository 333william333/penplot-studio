"""Multi-layer spray stencils.

One picture becomes N sheets.  You spray the lightest colour through sheet 1,
register sheet 2 on top, spray again, and the image builds up like screen
printing.  This module is only the geometry: the tool, the number of passes and
the G-code are somebody else's problem.

Conventions
-----------
* the input is a float32 RGB (or grayscale) image, 0 = black, 1 = paper, exactly
  what :mod:`raster` produces;
* output paths are polylines in *pixel* coordinates with Y pointing down, like
  every other source stage;
* every spatial setting is in **millimetres on the paper** and is converted with
  the same ``max(mm * px_per_mm, 0.4)`` rule as :class:`techniques.Context`.

Why bridges exist
-----------------
A stencil that is cut wrong falls apart.  The inside of an "O" drops out and the
picture becomes a hole.  Every enclosed piece of material therefore has to stay
attached to the rest of the sheet, and the only way to do that is to leave short
strips of material - bridges, or tabs - crossing the opening.

Two different things are called a bridge, and they do different jobs:

*tie bars*
    material carved *across* an opening so an island (the middle of the O) is
    physically joined to the surround.  These are what stop the stencil falling
    apart, and they are found on the raster mask, where connectivity is
    something you can actually compute.

*tabs*
    short un-cut stretches of a cut contour, so the waste drop stays attached
    while the blade is still moving and does not slide under it.  These turn
    each closed contour into a series of open arcs.

Tie bars are inserted first, by modifying the opening itself; the cut contours
are then traced from the repaired mask, so a tie bar simply appears as a notch
in the outline.  Tabs are added afterwards by :func:`add_bridges`.

Every sheet is finally *verified by rasterising*: the arcs are drawn onto a full
sheet of material, the waste is flood-filled away from the border, and if any
piece of material is left floating the sheet is rebuilt with fatter ties and
less smoothing.  A sheet that passes that check is one that survives being
picked up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import cv2
import numpy as np

from . import geometry as geo
from .pens import PenLibrary
from .raster import to_gray
from .separation import separate
from .techniques import Context, Param

__all__ = [
    "StencilSheet",
    "StencilSettings",
    "STENCIL_PARAMS",
    "STENCIL_MODES",
    "build_stencils",
    "add_bridges",
    "render_sheet",
    "render_composite",
]


STENCIL_MODES = {
    "tone": "Tone levels (one grey per sheet)",
    "colour": "One sheet per paint colour",
}

#: how a level is described in a sheet label
_TONE_NAMES = (
    (0.18, "pale grey"),
    (0.34, "light grey"),
    (0.52, "mid grey"),
    (0.70, "dark grey"),
    (0.88, "charcoal"),
    (2.00, "black"),
)

_CROSS3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

#: a component this small is a rasterising artefact, not a piece of paper
_ARTEFACT_PX = 3.0


# --------------------------------------------------------------------------
# containers
# --------------------------------------------------------------------------
@dataclass
class StencilSheet:
    """One physical sheet of the stack."""

    index: int                                  # spray order, 0 first
    label: str                                  # e.g. "1 of 4 - mid grey"
    tone: float                                 # 0..1 ink level this sheet carries
    color: tuple[float, float, float] | None    # when separated by colour
    cuts: list[np.ndarray]                      # what the blade follows, image px, Y down
    bridges: int                                # how many tabs were inserted
    warnings: list[str] = field(default_factory=list)

    # --- extras the interface and the preview want ------------------------
    #: the closed contours the cuts were derived from (before tabs were cut in)
    contours: list[np.ndarray] = field(default_factory=list)
    #: the sheet outline, when ``frame`` is set.  Also the last entry of *cuts*.
    outline: np.ndarray | None = None
    #: final opening mask, 1 where the spray goes through (uint8, HxW)
    mask: np.ndarray | None = None
    #: material strips carved across an opening to hold an island: (x0,y0,x1,y1)
    ties: list[tuple[float, float, float, float]] = field(default_factory=list)
    #: un-cut stretches left in the contours
    tabs: int = 0
    #: features dropped for being under ``min_feature``
    dropped: int = 0
    #: True when the island-safety check passed
    safe: bool = True

    @property
    def size(self) -> tuple[int, int]:
        """(height, width) in pixels."""
        return self.mask.shape[:2] if self.mask is not None else (0, 0)

    @property
    def cut_length(self) -> float:
        """Total blade travel, in pixels."""
        return geo.total_length(self.cuts)


@dataclass
class StencilSettings:
    """Everything spatial is in millimetres on the paper."""

    levels: int = 3
    mode: str = "tone"               # "tone" | "colour"
    smooth: float = 0.4              # mm - contour simplification tolerance
    min_feature: float = 2.0         # mm - anything smaller is removed, with a warning
    grow: float = 0.0                # mm - dilate/erode the cut to tune spray bleed
    bridge_width: float = 2.0        # mm of material left in each tab
    bridge_spacing: float = 25.0     # mm of cut between tabs
    min_bridges: int = 2             # per closed contour, whatever the spacing says
    frame: float = 0.0               # mm - rectangular border inset from the sheet edge
    # --- the rest have sensible defaults and rarely need touching ---------
    margin: float = 4.0              # mm of guaranteed un-cut material round the edge
    threshold_mode: str = "auto"     # auto | even | area - how the levels are chosen
    colour_threshold: float = 0.30   # coverage a pen needs before it gets an opening
    kerf: float = 0.3                # mm - blade width, used by the safety simulation
    max_repair_rounds: int = 4       # rebuilds allowed before a sheet is declared unsafe

    def clamped(self) -> "StencilSettings":
        out = StencilSettings(**self.__dict__)
        out.levels = int(max(1, min(int(out.levels), 8)))
        out.mode = out.mode if out.mode in STENCIL_MODES else "tone"
        out.smooth = float(max(0.0, out.smooth))
        out.min_feature = float(max(0.0, out.min_feature))
        out.bridge_width = float(max(0.2, out.bridge_width))
        out.bridge_spacing = float(max(out.bridge_width * 2.0, out.bridge_spacing))
        out.min_bridges = int(max(0, min(int(out.min_bridges), 24)))
        out.margin = float(max(0.5, out.margin))
        out.frame = float(max(0.0, out.frame))
        out.kerf = float(max(0.05, out.kerf))
        out.max_repair_rounds = int(max(0, min(int(out.max_repair_rounds), 8)))
        return out


#: control descriptions, so the interface can build its panel from this file
STENCIL_PARAMS: list[Param] = [
    Param("levels", "Sheets", 3, 1, 8, 1, 0, "", "How many stencils the picture is split into", kind="int"),
    Param("mode", "Separate by", "tone", kind="choice", choices=STENCIL_MODES,
          hint="Tone levels stack up like screen printing; colour makes one sheet per paint"),
    Param("smooth", "Smoothing", 0.4, 0.0, 3.0, 0.1, 2, " mm", "Straightens the cut - too much and a tab can be shaved off"),
    Param("min_feature", "Smallest feature", 2.0, 0.0, 10.0, 0.5, 1, " mm", "Slivers and islands thinner than this are removed"),
    Param("grow", "Bleed offset", 0.0, -3.0, 3.0, 0.1, 2, " mm", "Shrink the opening to allow for spray creeping under the edge"),
    Param("bridge_width", "Bridge width", 2.0, 0.5, 10.0, 0.5, 1, " mm", "Material left in each tab"),
    Param("bridge_spacing", "Bridge spacing", 25.0, 5.0, 200.0, 5.0, 0, " mm", "Cut length between tabs"),
    Param("min_bridges", "Bridges per shape", 2, 0, 12, 1, 0, "", "However short the contour is", kind="int"),
    Param("frame", "Border", 0.0, 0.0, 50.0, 1.0, 1, " mm", "Rectangular registration outline inset from the sheet edge"),
    Param("margin", "Edge margin", 4.0, 0.5, 50.0, 0.5, 1, " mm", "Un-cut band round the sheet that holds everything"),
]


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _disk(radius: int) -> np.ndarray:
    r = max(1, int(radius))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _sample(field_img: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Nearest-pixel lookup of a scalar field at (x, y) points."""
    if len(points) == 0:
        return np.zeros(0, dtype=np.float64)
    x = np.clip(np.round(points[:, 0]).astype(np.int32), 0, field_img.shape[1] - 1)
    y = np.clip(np.round(points[:, 1]).astype(np.int32), 0, field_img.shape[0] - 1)
    return field_img[y, x].astype(np.float64)


def _dedupe(points: np.ndarray) -> np.ndarray:
    """Drop repeated points; zero-length segments break arc-length maths."""
    if len(points) < 2:
        return points
    keep = np.ones(len(points), dtype=bool)
    keep[1:] = np.any(np.abs(np.diff(points, axis=0)) > 1e-9, axis=1)
    return points[keep]


def _cumulative(points: np.ndarray) -> np.ndarray:
    d = np.hypot(*np.diff(points, axis=0).T)
    return np.concatenate([[0.0], np.cumsum(d)])


def _tone_name(tone: float) -> str:
    for limit, name in _TONE_NAMES:
        if tone <= limit:
            return name
    return "black"


def _count_lost(before: np.ndarray, after: np.ndarray) -> int:
    """How many connected pieces of *before* are more than half gone in *after*."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(before, 8)
    if count <= 1:
        return 0
    survived = np.bincount(labels.ravel(), weights=(after > 0).ravel().astype(np.float64), minlength=count)
    areas = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
    lost = (areas >= _ARTEFACT_PX) & (survived < 0.5 * areas)
    return int(lost[1:].sum())


def _drop_small(mask: np.ndarray, area_floor: float, connectivity: int = 8) -> np.ndarray:
    """Erase connected components under *area_floor* pixels."""
    if area_floor <= 1.0 or not mask.any():
        return mask
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity)
    if count <= 1:
        return mask
    small = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] < area_floor)
    small = small[small != 0]
    if len(small) == 0:
        return mask
    kill = np.zeros(count, dtype=bool)
    kill[small] = True
    return np.where(kill[labels], np.uint8(0), mask)


# --------------------------------------------------------------------------
# level separation
# --------------------------------------------------------------------------
def _lloyd_thresholds(ink: np.ndarray, levels: int) -> np.ndarray:
    """Posterise into ``levels + 1`` tones (paper counts as one) and return the
    boundaries between them.

    A 1-D Lloyd relaxation over the 256-bin histogram: cheap, deterministic, and
    it follows the picture instead of slicing it into equal thirds, which on a
    portrait is the difference between a readable face and three grey blobs.
    """
    hist = np.bincount((np.clip(ink, 0.0, 1.0) * 255.0).astype(np.uint8).ravel(), minlength=256).astype(np.float64)
    values = np.arange(256, dtype=np.float64) / 255.0
    k = levels + 1
    cdf = np.cumsum(hist)
    if cdf[-1] <= 0:
        return np.linspace(0.0, 1.0, k + 1)[1:-1]
    cdf /= cdf[-1]
    centres = np.interp((np.arange(k) + 0.5) / k, cdf, values)
    for _ in range(40):
        edges = 0.5 * (centres[:-1] + centres[1:])
        bins = np.searchsorted(edges, values)
        weight = np.bincount(bins, weights=hist, minlength=k)
        total = np.bincount(bins, weights=hist * values, minlength=k)
        moved = np.where(weight > 0, total / np.maximum(weight, 1e-12), centres)
        moved = np.sort(moved)
        if np.allclose(moved, centres, atol=1e-6):
            centres = moved
            break
        centres = moved
    thresholds = 0.5 * (centres[:-1] + centres[1:])
    return thresholds


def _level_thresholds(ink: np.ndarray, levels: int, mode: str) -> np.ndarray:
    if mode == "even":
        thresholds = np.linspace(0.0, 1.0, levels + 2)[1:-1]
    elif mode == "area":
        quantiles = np.linspace(0.0, 1.0, levels + 2)[1:-1]
        thresholds = np.quantile(ink.ravel(), 1.0 - quantiles[::-1])[::-1]
    else:
        thresholds = _lloyd_thresholds(ink, levels)

    thresholds = np.asarray(thresholds, dtype=np.float64).ravel()
    # strictly increasing, and never zero: paper has to stay paper
    thresholds = np.clip(thresholds, 0.01, 0.999)
    for i in range(1, len(thresholds)):
        thresholds[i] = max(thresholds[i], thresholds[i - 1] + 1e-3)
    if len(thresholds) != levels or not np.all(np.isfinite(thresholds)):
        thresholds = np.linspace(0.0, 1.0, levels + 2)[1:-1]
    return thresholds


def _tone_layers(rgb: np.ndarray, settings: StencilSettings) -> list[dict]:
    ink = 1.0 - to_gray(rgb)
    thresholds = _level_thresholds(ink, settings.levels, settings.threshold_mode)
    layers = []
    for k, limit in enumerate(thresholds):
        # sheet k opens everything at or darker than level k, so the sprays stack
        mask = (ink >= limit).astype(np.uint8)
        tone = (k + 1) / float(settings.levels)
        layers.append({"mask": mask, "tone": tone, "color": None, "name": _tone_name(tone)})
    return layers


def _colour_layers(rgb: np.ndarray, settings: StencilSettings, library: PenLibrary) -> tuple[list[dict], list[str]]:
    warnings: list[str] = []
    maps = separate(rgb, library, mode="palette")
    layers = []
    for index, ink_map in enumerate(maps):
        if ink_map is None:
            continue
        mask = (ink_map >= settings.colour_threshold).astype(np.uint8)
        if not mask.any():
            continue
        pen = library[index]
        lightness = float(np.clip(pen.lightness / 100.0, 0.0, 1.0))
        layers.append({
            "mask": mask,
            "tone": float(np.clip(1.0 - lightness, 0.0, 1.0)),
            "color": pen.rgb,
            "name": pen.name or f"pen {index + 1}",
            "lightness": lightness,
        })
    # lightest paint first, so nothing is sprayed over a colour that hides it
    layers.sort(key=lambda item: -item["lightness"])
    if not layers:
        warnings.append("no pen matched enough of the picture to need a sheet")
    return layers, warnings


# --------------------------------------------------------------------------
# mask conditioning
# --------------------------------------------------------------------------
def _apply_grow(mask: np.ndarray, grow_px: float) -> np.ndarray:
    radius = int(round(abs(grow_px)))
    if radius < 1:
        return mask
    kernel = _disk(radius)
    return cv2.dilate(mask, kernel) if grow_px > 0 else cv2.erode(mask, kernel)


def _min_feature_clean(mask: np.ndarray, min_feature_px: float) -> tuple[np.ndarray, int]:
    """Remove openings and material thinner or smaller than *min_feature_px*.

    A 0.5 mm web of paper is not real: it tears the first time the sheet is
    lifted, and a 0.5 mm slot never sprays cleanly either.  Both sides of the
    cut therefore get the same treatment - open the mask to kill thin *slots*,
    close it to kill thin *webs*, then drop anything too small to be a feature
    at all in either direction.

    The returned count is the number of separate pieces that were more than half
    destroyed.  It is deliberately conservative: a hairline web joined to a big
    shape at both ends is absorbed into that shape rather than counted on its
    own, so the number never overstates the damage.
    """
    if min_feature_px < 1.5:
        return mask, 0

    before = mask
    radius = max(1, int(round(min_feature_px / 2.0)))
    kernel = _disk(radius)
    out = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)

    area_floor = 0.6 * min_feature_px * min_feature_px
    out = _drop_small(out, area_floor)                      # specks of opening
    material = _drop_small((out == 0).astype(np.uint8), area_floor)
    out = (material == 0).astype(np.uint8)                  # specks of material

    removed = _count_lost(before, out) + _count_lost((before == 0).astype(np.uint8), (out == 0).astype(np.uint8))
    return out, int(removed)


def _clear_margin(mask: np.ndarray, margin_px: int) -> tuple[np.ndarray, bool]:
    """Force an un-cut band round the sheet.

    Without it a dark area that runs off the edge cuts the sheet in half, and
    there is nothing left to hold on to.
    """
    m = max(1, int(margin_px))
    if mask.shape[0] <= 2 * m or mask.shape[1] <= 2 * m:
        return np.zeros_like(mask), bool(mask.any())
    touched = bool(mask[:m, :].any() or mask[-m:, :].any() or mask[:, :m].any() or mask[:, -m:].any())
    if touched:
        mask = mask.copy()
        mask[:m, :] = 0
        mask[-m:, :] = 0
        mask[:, :m] = 0
        mask[:, -m:] = 0
    return mask, touched


# --------------------------------------------------------------------------
# tie bars - the part that stops the stencil falling apart
# --------------------------------------------------------------------------
def _nearest_set_pixel(target: np.ndarray, point: np.ndarray, hint: float) -> np.ndarray | None:
    """Closest non-zero pixel of *target* to (y, x) *point*, searching outwards."""
    height, width = target.shape
    radius = int(hint) + 3
    for _ in range(8):
        y0, y1 = max(0, point[0] - radius), min(height, point[0] + radius + 1)
        x0, x1 = max(0, point[1] - radius), min(width, point[1] + radius + 1)
        window = target[y0:y1, x0:x1]
        found = np.argwhere(window)
        if len(found):
            found = found + (y0, x0)
            d = np.hypot(found[:, 0] - point[0], found[:, 1] - point[1])
            return found[int(np.argmin(d))]
        radius *= 3
        if radius > max(height, width):
            break
    return None


def _tie_island(
    mask: np.ndarray,
    island: np.ndarray,
    held: np.ndarray,
    distance: np.ndarray,
    *,
    width_px: int,
    spacing_px: float,
    minimum: int,
) -> list[tuple[float, float, float, float]]:
    """Carve material strips from *island* to the *held* material.

    The anchors are the points of the island boundary closest to something that
    is already attached, so a tie is the shortest possible strip across the
    opening; extra ties are pushed apart so a long island cannot pivot about a
    single one.
    """
    edge = np.argwhere(island - cv2.erode(island, _CROSS3))
    if len(edge) == 0:
        edge = np.argwhere(island)
    if len(edge) == 0:
        return []

    perimeter = float(len(edge))
    wanted = minimum if spacing_px <= 0 else max(minimum, int(math.ceil(perimeter / spacing_px)))
    room = int(perimeter // max(4.0 * width_px, 1.0))
    count = max(1, min(wanted, 6, max(1, room)))

    scores = distance[edge[:, 0], edge[:, 1]]
    order = np.argsort(scores, kind="stable")[:4000]
    separation = max(4.0 * width_px, perimeter / (3.0 * count))

    chosen: list[np.ndarray] = []
    for index in order:
        point = edge[index]
        if any(math.hypot(float(point[0] - c[0]), float(point[1] - c[1])) < separation for c in chosen):
            continue
        chosen.append(point)
        if len(chosen) >= count:
            break
    if not chosen:
        chosen = [edge[order[0]]]

    ties: list[tuple[float, float, float, float]] = []
    for point in chosen:
        anchor = _nearest_set_pixel(held, point, float(distance[point[0], point[1]]))
        if anchor is None:
            continue
        a = (int(point[1]), int(point[0]))
        b = (int(anchor[1]), int(anchor[0]))
        cv2.line(mask, a, b, 0, thickness=width_px, lineType=cv2.LINE_8)
        ties.append((float(a[0]), float(a[1]), float(b[0]), float(b[1])))
    return ties


def _tie_islands(
    mask: np.ndarray,
    *,
    seed: tuple[int, int],
    width_px: float,
    spacing_px: float,
    minimum: int,
    area_floor: float,
    extra: Sequence[np.ndarray] = (),
    limit: int = 16,
) -> tuple[np.ndarray, list[tuple[float, float, float, float]]]:
    """Repeat until every piece of material reaches the sheet border.

    Iterating matters: an island inside an island only becomes reachable once
    its parent has been tied down.
    """
    mask = mask.copy()
    width = max(1, int(round(width_px)))
    ties: list[tuple[float, float, float, float]] = []
    forced = [np.asarray(e, dtype=np.uint8) for e in extra]

    for _ in range(limit):
        material = (mask == 0).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(material, 4)
        held_label = int(labels[seed])
        loose = [
            i for i in range(1, count)
            if i != held_label and stats[i, cv2.CC_STAT_AREA] >= max(area_floor, _ARTEFACT_PX)
        ]
        if forced:
            for region in forced:
                hit = np.bincount(labels[region > 0].ravel(), minlength=count)
                hit[0] = 0
                if held_label < len(hit):
                    hit[held_label] = 0
                if hit.any() and int(np.argmax(hit)) not in loose:
                    loose.append(int(np.argmax(hit)))
            forced = []
        if not loose:
            break

        held = (labels == held_label).astype(np.uint8)
        distance = cv2.distanceTransform((1 - held).astype(np.uint8), cv2.DIST_L2, 5)
        for label in loose:
            island = (labels == label).astype(np.uint8)
            ties.extend(_tie_island(
                mask, island, held, distance,
                width_px=width, spacing_px=spacing_px, minimum=minimum,
            ))
    return mask, ties


# --------------------------------------------------------------------------
# contours and tabs
# --------------------------------------------------------------------------
def _trace(mask: np.ndarray, smooth_px: float) -> list[np.ndarray]:
    """Closed outlines of every opening, holes included."""
    found, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    out: list[np.ndarray] = []
    for contour in found:
        points = contour.reshape(-1, 2).astype(np.float64)
        if len(points) < 3 or abs(cv2.contourArea(contour)) < 2.0:
            continue
        loop = np.vstack([points, points[:1]])
        if smooth_px > 0:
            simplified = geo.rdp(loop, smooth_px)
            if len(simplified) >= 4:
                loop = simplified
        loop = _dedupe(loop)
        if len(loop) < 4:
            continue
        if np.any(np.abs(loop[0] - loop[-1]) > 1e-9):
            loop = np.vstack([loop, loop[:1]])
        out.append(loop)
    return out


def _arc(points: np.ndarray, cumulative: np.ndarray, start: float, end: float) -> np.ndarray:
    """The stretch of a closed polyline between two arc-length positions."""
    total = cumulative[-1]
    body = points[:-1]
    doubled = np.vstack([body, body, points[:1]])
    doubled_cum = np.concatenate([cumulative[:-1], cumulative[:-1] + total, [2.0 * total]])

    start = float(start) % total
    end = float(end)
    if end <= start:
        end += total
    end = min(end, start + total)

    i = int(np.searchsorted(doubled_cum, start, side="right")) - 1
    j = int(np.searchsorted(doubled_cum, end, side="left"))
    i = max(0, min(i, len(doubled) - 2))
    j = max(i + 1, min(j, len(doubled) - 1))

    def at(index: int, position: float) -> np.ndarray:
        span = doubled_cum[index + 1] - doubled_cum[index]
        if span <= 1e-12:
            return doubled[index]
        t = (position - doubled_cum[index]) / span
        return doubled[index] + (doubled[index + 1] - doubled[index]) * float(np.clip(t, 0.0, 1.0))

    head = at(i, start)
    tail = at(j - 1, end)
    middle = doubled[i + 1 : j]
    return _dedupe(np.vstack([head[None, :], middle, tail[None, :]]))


def add_bridges(
    contour: np.ndarray,
    *,
    width_px: float,
    spacing_px: float,
    minimum: int,
    avoid: Callable[[np.ndarray], np.ndarray] | None = None,
    count: int | None = None,
    straight_weight: float = 1.0,
) -> list[np.ndarray]:
    """Cut a closed contour into open arcs, leaving un-cut tabs between them.

    *width_px* is the material left in each tab, *spacing_px* the cut length
    between tabs, *minimum* the number of tabs a contour gets however short it
    is.  *avoid* is an optional penalty field, sampled at (x, y) points, used to
    keep tabs away from places where two contours nearly touch - a tab there
    would sit on a web of paper too thin to hold it.

    Tabs are pulled towards straight, low-curvature stretches: a tab on a corner
    is both ugly and weak, because the material either side of it is being
    pulled in two directions.
    """
    points = _dedupe(np.asarray(contour, dtype=np.float64)[:, :2])
    if len(points) < 3:
        return [np.asarray(contour, dtype=np.float64)]
    if np.any(np.abs(points[0] - points[-1]) > 1e-9):
        points = np.vstack([points, points[:1]])

    cumulative = _cumulative(points)
    perimeter = float(cumulative[-1])
    if perimeter <= 1e-9 or width_px <= 0:
        return [points]

    if count is None:
        wanted = int(minimum)
        if spacing_px > 0:
            wanted = max(wanted, int(math.ceil(perimeter / spacing_px)))
    else:
        wanted = int(count)
    if wanted <= 0:
        return [points]

    # every tab eats material, and an arc shorter than the tab is not worth
    # lifting the blade for
    room = int(perimeter // (width_px * 2.5))
    if room < 1:
        # too small to tab properly: leave a single tab of whatever fits
        width_px = min(width_px, perimeter * 0.35)
        wanted = 1
    else:
        wanted = max(1, min(wanted, room))

    # candidate positions, one every pixel or so
    step = max(perimeter / 2048.0, 1.0)
    targets = np.arange(0.0, perimeter, step)
    if len(targets) < 8:
        targets = np.linspace(0.0, perimeter, 8, endpoint=False)
        step = perimeter / 8.0
    samples = np.stack([
        np.interp(targets, cumulative, points[:, 0]),
        np.interp(targets, cumulative, points[:, 1]),
    ], axis=1)

    # the curvature window has to stay well inside the contour, or a short loop
    # ends up comparing a point with itself
    window = max(1, min(int(round(max(width_px, 3.0) / step)), max(1, len(samples) // 4)))
    back = samples - np.roll(samples, window, axis=0)
    forward = np.roll(samples, -window, axis=0) - samples
    cross = back[:, 0] * forward[:, 1] - back[:, 1] * forward[:, 0]
    dot = (back * forward).sum(axis=1)
    turn = np.abs(np.arctan2(cross, dot)) / math.pi

    # judge a stretch, not a point: a tab needs its whole width to be calm
    span = max(1, min(window, len(samples) // 2))
    kernel = np.ones(span, dtype=np.float64) / span
    padded = np.concatenate([turn[-span:], turn, turn[:span]])
    turn = np.convolve(padded, kernel, mode="same")[span : span + len(samples)]

    cost = straight_weight * turn
    if avoid is not None:
        penalty = np.asarray(avoid(samples), dtype=np.float64)
        if penalty.shape == cost.shape:
            cost = cost + 2.0 * penalty

    # even spacing first, then slide each tab to the calmest nearby spot
    ideal = np.arange(wanted) * (perimeter / wanted)
    slack = 0.35 * perimeter / wanted
    separation = max(width_px * 1.6, perimeter / (wanted * 3.0))
    chosen: list[float] = []
    for centre in ideal:
        low = min(int(np.searchsorted(targets, max(0.0, centre - slack))), len(cost))
        high = min(int(np.searchsorted(targets, min(perimeter, centre + slack))), len(cost))
        if high <= low:
            best = float(centre)
        else:
            window_cost = cost[low:high].copy()
            for taken in chosen:
                delta = np.abs(targets[low:high] - taken)
                delta = np.minimum(delta, perimeter - delta)
                window_cost[delta < separation] += 10.0
            best = float(targets[low + int(np.argmin(window_cost))])
        chosen.append(best)

    chosen.sort()
    half = width_px * 0.5
    arcs: list[np.ndarray] = []
    for index, centre in enumerate(chosen):
        if len(chosen) == 1:
            gap = perimeter
        else:
            nxt = chosen[(index + 1) % len(chosen)]
            # the last tab wraps round to the first one
            gap = nxt - centre if index + 1 < len(chosen) else nxt + perimeter - centre
        cut = gap - width_px
        # two tabs that ended up nearly on top of each other leave no arc worth
        # cutting between them; measure the gap directly rather than modulo the
        # perimeter, which would turn a negative gap into a full extra lap
        if cut < width_px * 0.25 and len(chosen) > 1:
            continue
        arc = _arc(points, cumulative, centre + half, centre + half + cut)
        if len(arc) >= 2 and geo.path_length(arc) > 1e-6:
            arcs.append(arc)
    return arcs or [points]


# --------------------------------------------------------------------------
# island safety
# --------------------------------------------------------------------------
def _verify(
    arcs: Sequence[np.ndarray],
    mask: np.ndarray,
    *,
    seed: tuple[int, int],
    kerf_px: int,
    area_floor: float,
) -> list[np.ndarray]:
    """Cut the sheet for real and see what falls out.

    A full sheet of material, the arcs drawn on as severed pixels, then a flood
    fill from the border.  Anything left over that is made of *material* rather
    than waste is a piece that would drop on the floor.  Four-connectivity is
    deliberate: an eight-connected line separates four-connected regions, so a
    single-pixel kerf really does cut.
    """
    canvas = np.ones(mask.shape, dtype=np.uint8)
    for arc in arcs:
        if len(arc) < 2:
            continue
        points = np.round(np.asarray(arc, dtype=np.float64)[:, :2]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [points], False, 0, thickness=max(1, int(kerf_px)), lineType=cv2.LINE_8)

    count, labels, _, _ = cv2.connectedComponentsWithStats(canvas, 4)
    if count <= 1:
        return []
    held = int(labels[seed])
    material = (mask == 0).ravel().astype(np.float64)
    per_label = np.bincount(labels.ravel(), weights=material, minlength=count)

    orphans: list[np.ndarray] = []
    for label in range(1, count):
        if label == held:
            continue
        if per_label[label] >= max(area_floor, _ARTEFACT_PX):
            orphans.append(((labels == label) & (mask == 0)).astype(np.uint8))
    return orphans


def _loose_material(mask: np.ndarray, seed: tuple[int, int], area_floor: float) -> list[np.ndarray]:
    """Material that is not joined to the border at all - checked on the mask,
    before any contour or simplification could hide the problem."""
    material = (mask == 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(material, 4)
    if count <= 1:
        return []
    held = int(labels[seed])
    return [
        (labels == label).astype(np.uint8)
        for label in range(1, count)
        if label != held and stats[label, cv2.CC_STAT_AREA] >= max(area_floor, _ARTEFACT_PX)
    ]


# --------------------------------------------------------------------------
# one sheet
# --------------------------------------------------------------------------
def _clearance_field(mask: np.ndarray, width_px: float) -> np.ndarray:
    """How thick the material is near each pixel, so tabs can avoid thin webs."""
    material = (mask == 0).astype(np.uint8)
    thickness = cv2.distanceTransform(material, cv2.DIST_L2, 3)
    radius = max(1, int(round(width_px)))
    return cv2.dilate(thickness, _disk(radius))


def _build_sheet(
    opening: np.ndarray,
    meta: dict,
    settings: StencilSettings,
    ctx: Context,
    *,
    index: int,
    total: int,
) -> StencilSheet:
    warnings: list[str] = []
    height, width = opening.shape[:2]

    min_feature_px = ctx.px(settings.min_feature) if settings.min_feature > 0 else 0.0
    bridge_px = ctx.px(settings.bridge_width)
    spacing_px = ctx.px(settings.bridge_spacing)
    smooth_px = ctx.px(settings.smooth) if settings.smooth > 0 else 0.0
    kerf_px = max(1, int(round(ctx.px(settings.kerf))))
    frame_px = ctx.px(settings.frame) if settings.frame > 0 else 0.0
    # the border is a cut too, so the un-cut band has to reach past it - otherwise
    # the trim line would be the only thing holding the sheet together
    margin_px = max(1, int(round(max(ctx.px(settings.margin), frame_px + max(min_feature_px, 2.0)))))
    area_floor = max(0.6 * min_feature_px * min_feature_px, 4.0)

    mask = np.ascontiguousarray(opening.astype(np.uint8))
    if abs(settings.grow) > 1e-6:
        mask = _apply_grow(mask, math.copysign(ctx.px(abs(settings.grow)), settings.grow))

    mask, dropped = _min_feature_clean(mask, min_feature_px)
    if dropped:
        warnings.append(f"{dropped} feature{'s' if dropped != 1 else ''} under {settings.min_feature:g} mm removed")

    mask, clipped = _clear_margin(mask, margin_px)
    if clipped:
        warnings.append(f"opening reached the sheet edge, trimmed back to the {settings.margin:g} mm margin")

    # somewhere in the un-cut band, and inside the picture however small it is
    seed = (min(margin_px // 2, height - 1), min(margin_px // 2, width - 1))
    base = mask
    extra_regions: list[np.ndarray] = []
    ties: list[tuple[float, float, float, float]] = []
    contours: list[np.ndarray] = []
    arcs: list[np.ndarray] = []
    rounds = 0
    tabs = 0
    safe = True

    for attempt in range(settings.max_repair_rounds + 1):
        rounds = attempt
        last = attempt == settings.max_repair_rounds
        tie_width = bridge_px * (1.35 ** attempt)
        # the final attempt cuts exactly on the mask boundary: with no
        # simplification the blade cannot stray across a tie bar
        round_smooth = 0.0 if last else smooth_px * (0.45 ** attempt)

        mask, ties = _tie_islands(
            base,
            seed=seed,
            width_px=tie_width,
            spacing_px=spacing_px,
            minimum=max(1, settings.min_bridges),
            area_floor=area_floor,
            extra=extra_regions,
        )
        # a tie can pinch an opening into a sliver that will never spray
        mask = _drop_small(mask, area_floor)

        contours = _trace(mask, round_smooth)
        clearance = _clearance_field(mask, bridge_px)
        need = max(bridge_px, 1.0)

        def avoid(points: np.ndarray, _field=clearance, _need=need) -> np.ndarray:
            return np.clip(1.0 - _sample(_field, points) / _need, 0.0, 1.0)

        arcs = []
        tabs = 0
        for contour in contours:
            pieces = add_bridges(
                contour,
                width_px=bridge_px,
                spacing_px=spacing_px,
                minimum=settings.min_bridges,
                avoid=avoid,
            )
            # a contour handed back still closed was too small to tab at all;
            # otherwise k open arcs means k tabs
            closed = len(pieces) == 1 and np.allclose(pieces[0][0], pieces[0][-1])
            tabs += 0 if closed else len(pieces)
            arcs.extend(pieces)

        orphans = _loose_material(mask, seed, area_floor)
        orphans += _verify(arcs, mask, seed=seed, kerf_px=kerf_px, area_floor=area_floor)
        if not orphans:
            break
        extra_regions = orphans
        if last:
            safe = False
            warnings.append(
                f"{len(orphans)} piece{'s' if len(orphans) != 1 else ''} could not be held "
                f"even with {settings.bridge_width * (1.35 ** attempt):.1f} mm ties - "
                "raise the bridge width or the smallest feature"
            )

    if rounds:
        warnings.append(
            f"island check needed {rounds} extra pass{'es' if rounds != 1 else ''} "
            f"(ties widened to {settings.bridge_width * (1.35 ** rounds):.1f} mm)"
        )

    cuts = [np.clip(np.ascontiguousarray(a, dtype=np.float64), (0.0, 0.0), (width - 1.0, height - 1.0)) for a in arcs]
    cuts = [c for c in cuts if len(c) >= 2 and np.all(np.isfinite(c))]

    outline = None
    if frame_px > 0:
        inset = float(min(frame_px, min(width, height) * 0.5 - 2.0))
        outline = np.array([
            [inset, inset],
            [width - 1 - inset, inset],
            [width - 1 - inset, height - 1 - inset],
            [inset, height - 1 - inset],
            [inset, inset],
        ], dtype=np.float64)
        # the trim line is the sheet edge, not part of the picture: it is never
        # tabbed and never counted as a piece that could fall out
        contours = contours + [outline]
        cuts = cuts + [outline]

    label = f"{index + 1} of {total} - {meta['name']}"
    return StencilSheet(
        index=index,
        label=label,
        tone=float(meta["tone"]),
        color=meta.get("color"),
        cuts=cuts,
        bridges=len(ties) + tabs,
        warnings=warnings,
        contours=contours,
        outline=outline,
        mask=mask,
        ties=ties,
        tabs=tabs,
        dropped=dropped,
        safe=safe,
    )


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------
def build_stencils(
    rgb: np.ndarray,
    settings: StencilSettings,
    *,
    px_per_mm: float,
    library: PenLibrary | None = None,
) -> list[StencilSheet]:
    """Split a picture into spray stencils, lightest sheet first."""
    settings = settings.clamped()
    ctx = Context(px_per_mm=max(float(px_per_mm), 1e-3), scale_with_pen=False)

    image = np.asarray(rgb, dtype=np.float32)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    image = np.clip(image, 0.0, 1.0)

    shared: list[str] = []
    if settings.mode == "colour":
        if library is None or len(library) == 0:
            shared.append("no pen library given, falling back to tone levels")
            layers = _tone_layers(image, settings)
        else:
            layers, colour_warnings = _colour_layers(image, settings, library)
            shared.extend(colour_warnings)
            if not layers:
                layers = _tone_layers(image, settings)
    else:
        layers = _tone_layers(image, settings)

    sheets: list[StencilSheet] = []
    total = len(layers)
    for index, meta in enumerate(layers):
        sheet = _build_sheet(meta["mask"], meta, settings, ctx, index=index, total=total)
        sheet.warnings = shared + sheet.warnings if index == 0 else sheet.warnings
        if sheet.mask is not None and not sheet.mask.any():
            sheet.warnings.append("nothing left to cut on this sheet")
        # a sheet that opens the same area as the one before it is a wasted sheet:
        # the picture simply does not hold that many distinct levels
        if index and sheets[-1].mask is not None and sheet.mask is not None:
            union = np.count_nonzero(sheets[-1].mask | sheet.mask)
            same = np.count_nonzero(sheets[-1].mask & sheet.mask)
            if union and same / union > 0.97:
                sheet.warnings.append(
                    f"opens almost the same area as sheet {index}"
                    + (" - the picture has fewer distinct tones than sheets"
                       if settings.mode == "tone" else " - these two paints overlap")
                )
        sheets.append(sheet)
    return sheets


# --------------------------------------------------------------------------
# preview rendering
# --------------------------------------------------------------------------
def render_sheet(
    sheet: StencilSheet,
    *,
    material: float = 0.72,
    scale: int = 1,
) -> np.ndarray:
    """Float RGB picture of one sheet: material grey, openings white, cuts black."""
    if sheet.mask is None:
        return np.ones((1, 1, 3), dtype=np.float32)
    height, width = sheet.mask.shape[:2]
    scale = max(1, int(scale))
    canvas = np.where(sheet.mask[:, :, None] > 0, np.float32(1.0), np.float32(material))
    canvas = np.repeat(canvas, 3, axis=2).astype(np.float32)
    if scale > 1:
        canvas = cv2.resize(canvas, (width * scale, height * scale), interpolation=cv2.INTER_NEAREST)
    for arc in sheet.cuts:
        points = np.round(np.asarray(arc, dtype=np.float64)[:, :2] * scale).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [points], False, (0.0, 0.0, 0.0), thickness=max(1, scale), lineType=cv2.LINE_AA)
    for x0, y0, x1, y1 in sheet.ties:
        cv2.line(canvas, (int(x0 * scale), int(y0 * scale)), (int(x1 * scale), int(y1 * scale)),
                 (0.85, 0.25, 0.15), thickness=max(1, scale), lineType=cv2.LINE_AA)
    return np.clip(canvas, 0.0, 1.0)


def render_composite(sheets: Sequence[StencilSheet]) -> np.ndarray:
    """What the stack would look like sprayed: each sheet's openings filled with
    its own tone, in spray order.  Bridges show up as unsprayed strips, which is
    exactly what happens on the paper."""
    live = [s for s in sheets if s.mask is not None]
    if not live:
        return np.ones((1, 1, 3), dtype=np.float32)
    height, width = live[0].mask.shape[:2]
    canvas = np.ones((height, width, 3), dtype=np.float32)
    for sheet in live:
        colour = sheet.color if sheet.color is not None else (1.0 - sheet.tone,) * 3
        paint = np.asarray(colour, dtype=np.float32).reshape(1, 1, 3)
        where = sheet.mask[:, :, None] > 0
        canvas = np.where(where, paint, canvas)
    return np.clip(canvas, 0.0, 1.0)
