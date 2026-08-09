"""Drawing techniques.

Each technique turns a prepared grayscale image into pen strokes.  They are
registered in one table with their own parameters, so the interface can build
its controls - and its preview gallery - straight from this file.

Conventions
-----------
* the input image is float32, 0 = black ink, 1 = white paper;
* output paths are polylines in *pixel* coordinates with Y pointing down;
* every spatial parameter is in **millimetres on the paper** and converted with
  the `Context`, which also knows how wide the pen is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from . import geometry as geo
from .styles import (
    contour_paths,
    hatch_mask,
    sample_bilinear,
    split_by_mask,
    trace_binary,
)

__all__ = [
    "Cancelled",
    "Param",
    "Technique",
    "Context",
    "REGISTRY",
    "GROUPS",
    "render",
    "render_groups",
    "defaults_for",
    "resolve",
]

class Cancelled(Exception):
    """Raised inside a technique when the caller asks it to stop early."""


REFERENCE_PEN = 0.5  # every default is written for a 0.5 mm pen

#: A plotter cannot usefully place more dots than this, and beyond it the path
#: ordering alone takes minutes.
MAX_DOTS = 120_000


# --------------------------------------------------------------------------
@dataclass
class Param:
    key: str
    label: str
    default: float | bool | str
    minimum: float = 0.0
    maximum: float = 1.0
    step: float = 0.1
    decimals: int = 2
    suffix: str = ""
    hint: str = ""
    kind: str = "float"            # float | int | bool | choice | text
    choices: dict[str, str] | None = None
    pen_scaled: bool = False       # multiply by pen width / 0.5 mm


@dataclass
class Technique:
    key: str
    label: str
    group: str
    description: str
    params: list[Param]
    render: Callable[[np.ndarray, dict, "Context"], list[np.ndarray]]
    slow: bool = False
    #: True when render() returns [(dwell_ms, paths), ...] instead of paths
    grouped: bool = False
    #: hint the interface shows as a warning, e.g. that a ballpoint will not work
    requires: str = ""
    #: True when consecutive strokes may be joined to avoid a pen lift.  Right
    #: for parallel fill strokes, wrong for outlines and for anything where the
    #: gaps are the drawing (dashes, dots).
    stitchable: bool = False

    def defaults(self) -> dict:
        return {p.key: p.default for p in self.params}


@dataclass
class Context:
    """Everything a technique needs to translate millimetres into pixels."""

    px_per_mm: float = 4.0
    pen_width: float = 0.5
    scale_with_pen: bool = True
    seed: int = 1234
    #: polled inside the long Python loops so shutting down is prompt
    should_cancel: object = None

    def check(self) -> None:
        if self.should_cancel is not None and self.should_cancel():
            raise Cancelled()

    @property
    def pen_scale(self) -> float:
        return max(self.pen_width, 0.05) / REFERENCE_PEN if self.scale_with_pen else 1.0

    def px(self, millimetres: float, pen_scaled: bool = False) -> float:
        value = millimetres * (self.pen_scale if pen_scaled else 1.0)
        return max(value * self.px_per_mm, 0.4)

    @property
    def pen_px(self) -> float:
        return max(self.pen_width * self.px_per_mm, 0.5)

    def rng(self, salt: int = 0) -> np.random.Generator:
        return np.random.default_rng(self.seed + salt)


GROUPS = {
    "line": "Line work",
    "shading": "Shading",
    "dots": "Dots & halftone",
    "geometric": "Geometric",
}


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------
def _ink(image: np.ndarray) -> np.ndarray:
    return np.clip(1.0 - image, 0.0, 1.0)


def _tone(ink: np.ndarray, gamma: float) -> np.ndarray:
    if abs(gamma - 1.0) < 1e-3:
        return ink
    return np.power(np.clip(ink, 0.0, 1.0), max(gamma, 0.05))


def _blue_noise(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Cheap high-frequency noise used to break up tonal banding."""
    noise = rng.random(shape).astype(np.float32)
    return noise - cv2.GaussianBlur(noise, (0, 0), 2.0)


def _jitter_path(path: np.ndarray, amount: float, rng: np.random.Generator) -> np.ndarray:
    """Wobble a polyline so it looks drawn by a hand rather than a machine."""
    if amount <= 1e-6 or len(path) < 2:
        return path
    steps = len(path)
    wave = np.cumsum(rng.normal(0.0, 1.0, (steps, 2)), axis=0)
    wave -= np.linspace(wave[0], wave[-1], steps)
    span = np.abs(wave).max()
    if span < 1e-9:
        return path
    return path + wave * (amount / span)


def _overshoot(path: np.ndarray, amount: float, rng: np.random.Generator,
               bounds: tuple[int, int] | None = None) -> np.ndarray:
    """Extend both ends of a stroke a little, the way a sketching hand does."""
    if amount <= 1e-6 or len(path) < 2:
        return path
    start_dir = path[0] - path[1]
    end_dir = path[-1] - path[-2]
    for direction in (start_dir, end_dir):
        norm = math.hypot(direction[0], direction[1])
        if norm > 1e-9:
            direction /= norm
    head = path[0] + start_dir * rng.uniform(0.2, 1.0) * amount
    tail = path[-1] + end_dir * rng.uniform(0.2, 1.0) * amount
    if bounds is not None:
        width, height = bounds
        head = np.clip(head, (0.0, 0.0), (width - 1.0, height - 1.0))
        tail = np.clip(tail, (0.0, 0.0), (width - 1.0, height - 1.0))
    return np.vstack([head, path, tail])


def _occupancy_stop(grid: np.ndarray, cell: float, x: float, y: float, mark: bool) -> bool:
    gy = int(y / cell)
    gx = int(x / cell)
    if gy < 0 or gx < 0 or gy >= grid.shape[0] or gx >= grid.shape[1]:
        return True
    if grid[gy, gx]:
        return True
    if mark:
        grid[gy, gx] = True
    return False


def lloyd_relax(points: np.ndarray, weight: np.ndarray, iterations: int) -> np.ndarray:
    """Weighted Lloyd relaxation - the classic way to get even stipple dots.

    The Voronoi regions come from OpenCV's distance transform and the weighted
    centroids from `np.bincount`, so an iteration is a handful of array ops
    instead of a nearest-neighbour search per pixel.
    """
    if iterations <= 0 or len(points) < 2:
        return points
    height, width = weight.shape
    ys, xs = np.mgrid[0:height, 0:width]
    xs = xs.ravel().astype(np.float64)
    ys = ys.ravel().astype(np.float64)
    flat_weight = np.clip(weight, 1e-4, None).ravel().astype(np.float64)

    for _ in range(iterations):
        mask = np.full((height, width), 255, dtype=np.uint8)
        px = np.clip(points[:, 0].astype(np.int32), 0, width - 1)
        py = np.clip(points[:, 1].astype(np.int32), 0, height - 1)
        mask[py, px] = 0
        _, labels = cv2.distanceTransformWithLabels(
            mask, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL
        )
        labels = labels.astype(np.int64)
        seed_labels = labels[py, px]
        size = int(labels.max()) + 1
        flat = labels.ravel()
        total = np.bincount(flat, weights=flat_weight, minlength=size)
        sum_x = np.bincount(flat, weights=flat_weight * xs, minlength=size)
        sum_y = np.bincount(flat, weights=flat_weight * ys, minlength=size)
        good = total[seed_labels] > 1e-6
        moved = points.copy()
        idx = seed_labels[good]
        moved[good, 0] = sum_x[idx] / total[idx]
        moved[good, 1] = sum_y[idx] / total[idx]
        points = moved
    return points


def _dither_points(ink: np.ndarray, pitch: float, min_ink: float, rng: np.random.Generator) -> np.ndarray:
    """Floyd-Steinberg on a coarse grid - a good starting set for relaxation."""
    height, width = ink.shape
    gw = max(int(round(width / pitch)), 2)
    gh = max(int(round(height / pitch)), 2)
    small = cv2.resize(ink, (gw, gh), interpolation=cv2.INTER_AREA).astype(np.float32)
    small[small < min_ink] = 0.0

    work = small.copy()
    dots = np.zeros((gh, gw), dtype=bool)
    for y in range(gh):
        for x in range(gw):
            old = work[y, x]
            new = 1.0 if old > 0.5 else 0.0
            dots[y, x] = new > 0.5
            error = old - new
            if x + 1 < gw:
                work[y, x + 1] += error * 7 / 16
            if y + 1 < gh:
                if x > 0:
                    work[y + 1, x - 1] += error * 3 / 16
                work[y + 1, x] += error * 5 / 16
                if x + 1 < gw:
                    work[y + 1, x + 1] += error * 1 / 16

    ys, xs = np.nonzero(dots)
    if len(xs) == 0:
        return np.zeros((0, 2))
    sx = width / gw
    sy = height / gh
    points = np.stack([(xs + 0.5) * sx, (ys + 0.5) * sy], axis=1).astype(np.float64)
    points += rng.uniform(-0.3, 0.3, points.shape) * np.array([sx, sy])
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    return points


def _edge_tangents(image: np.ndarray, smoothing: float) -> tuple[np.ndarray, np.ndarray]:
    """Direction field that follows the shapes in the picture.

    This is the minor eigenvector of the smoothed structure tensor, i.e. the
    direction in which the image changes least - along edges and around forms.
    """
    blurred = cv2.GaussianBlur(image, (0, 0), max(smoothing * 0.4, 0.6))
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    sigma = max(smoothing, 1.0)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma)
    theta = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy + 1e-9)
    # rotate 90 degrees: follow the edge instead of crossing it
    return np.cos(theta + math.pi / 2).astype(np.float32), np.sin(theta + math.pi / 2).astype(np.float32)


# --------------------------------------------------------------------------
# 1. sketch - traced edges
# --------------------------------------------------------------------------
def _render_sketch(image: np.ndarray, p: dict, ctx: Context) -> list[np.ndarray]:
    u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    # a wide nib cannot draw hair-fine edges, so stop looking for them
    blur = cv2.GaussianBlur(u8, (0, 0), max(p["softness"] * max(ctx.pen_scale, 0.2), 0.1))

    if p["auto_threshold"]:
        # Thresholding on the *gradient* is what makes this work on any picture.
        # The usual "median of the image" trick fails badly on artwork with a
        # white background, where the median is simply 255.
        # Sobel with the same aperture Canny uses, so the numbers are comparable
        gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.hypot(gx, gy)
        peak = float(np.percentile(magnitude, 99.5)) or 1.0
        scaled = np.clip(magnitude / peak * 255.0, 0, 255).astype(np.uint8)
        otsu, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        strength = max(otsu, 12.0) * peak / 255.0
        sensitivity = max(p["sensitivity"], 1.0) / 50.0   # 1.0 = neutral
        high = float(np.clip(strength / sensitivity, 4.0, 4000.0))
        low = high * 0.4
    else:
        low, high = p["low"], p["high"]

    edges = cv2.Canny(blur, float(low), float(high), L2gradient=True)
    paths = trace_binary(edges)

    minimum = ctx.px(p["min_length"], pen_scaled=True)
    paths = [path for path in paths if geo.path_length(path) >= minimum]
    paths = geo.simplify_paths(paths, max(ctx.pen_px * 0.25, 0.4))

    rng = ctx.rng(3)
    jitter = ctx.px(p["wobble"], pen_scaled=True) if p["wobble"] > 0 else 0.0
    passes = int(p["passes"])
    out: list[np.ndarray] = []
    for path in paths:
        for index in range(passes):
            stroke = path if index == 0 and jitter <= 0 else _jitter_path(path, jitter, rng)
            if p["overshoot"] > 0:
                stroke = _overshoot(
                    stroke, ctx.px(p["overshoot"], pen_scaled=True), rng, (image.shape[1], image.shape[0])
                )
            out.append(stroke)
    return out


# --------------------------------------------------------------------------
# 2. contour lines - topographic map of the brightness
# --------------------------------------------------------------------------
def _render_contours(image: np.ndarray, p: dict, ctx: Context) -> list[np.ndarray]:
    ink = _tone(_ink(image), p["tone"])
    ink = cv2.GaussianBlur(ink, (0, 0), max(p["smoothing"], 0.1))
    # A fat pen cannot hold twelve separate contour lines in the same gradient -
    # they merge into a black band.  Fewer, further apart, is what it can draw.
    levels = max(int(round(p["levels"] / max(ctx.pen_scale, 0.05))), 2) if ctx.scale_with_pen else max(int(p["levels"]), 1)
    smooth = max(ctx.px(p["simplify"], pen_scaled=True), 0.3)
    min_area = max(ctx.px(p["min_size"], pen_scaled=True) ** 2, 3.0)

    out: list[np.ndarray] = []
    for index in range(levels):
        threshold = (index + 0.5) / levels
        mask = (ink >= threshold).astype(np.uint8)
        if not mask.any():
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue
            points = geo.rdp(contour.reshape(-1, 2).astype(np.float64), smooth)
            if len(points) < 3:
                continue
            out.append(np.vstack([points, points[:1]]))
    return out


# --------------------------------------------------------------------------
# 3. crosshatch
# --------------------------------------------------------------------------
def _render_crosshatch(image: np.ndarray, p: dict, ctx: Context) -> list[np.ndarray]:
    ink = _tone(_ink(image), p["tone"])
    layers = max(int(p["layers"]), 1)

    # The key relationship: with `layers` passes at `spacing`, the darkest area
    # gets layers * pen_width / spacing of ink.  Solving for full coverage keeps
    # crosshatching from turning into a solid black block.
    coverage = max(p["coverage"], 0.05)
    spacing = ctx.pen_px * layers / coverage

    rng = ctx.rng(7)
    noise = _blue_noise(ink.shape, rng) * p["dither"]
    out: list[np.ndarray] = []
    minimum = ctx.px(p["min_length"])
    overshoot = ctx.px(p["sketchiness"]) if p["sketchiness"] > 0 else 0.0

    for index in range(layers):
        threshold = (index + 1) / (layers + 1.0)
        mask = ((ink + noise) >= threshold).astype(np.uint8)
        if not mask.any():
            continue
        angle = p["angle"] + p["angle_step"] * index
        segments = hatch_mask(mask, spacing, angle, phase=index * spacing / max(layers, 1))
        for segment in segments:
            if geo.path_length(segment) < minimum:
                continue
            if overshoot > 0:
                segment = _overshoot(segment, overshoot, rng, (ink.shape[1], ink.shape[0]))
            out.append(segment)
    return out


# --------------------------------------------------------------------------
# 4. dashes - broken hatching, like a pencil sketch
# --------------------------------------------------------------------------
def _render_dashes(image: np.ndarray, p: dict, ctx: Context) -> list[np.ndarray]:
    ink = _tone(_ink(image), p["tone"])
    spacing = ctx.px(p["spacing"], pen_scaled=True)
    mask = (ink > p["min_ink"]).astype(np.uint8)
    if not mask.any():
        return []

    segments = hatch_mask(mask, spacing, p["angle"])
    dash = ctx.px(p["dash"], pen_scaled=True)
    gap = ctx.px(p["gap"], pen_scaled=True)
    rng = ctx.rng(11)
    out: list[np.ndarray] = []

    for segment in segments:
        start, end = segment[0], segment[-1]
        length = math.hypot(end[0] - start[0], end[1] - start[1])
        if length < 1e-6:
            continue
        direction = (end - start) / length
        position = rng.uniform(0.0, dash + gap)
        while position < length:
            local = sample_bilinear(
                ink,
                np.array([start[0] + direction[0] * position]),
                np.array([start[1] + direction[1] * position]),
            )[0]
            if local > p["min_ink"]:
                run = dash * (0.25 + 0.75 * float(local)) * rng.uniform(0.7, 1.3)
                run = min(run, length - position)
                if run > ctx.pen_px:
                    out.append(
                        np.array([start + direction * position, start + direction * (position + run)])
                    )
                position += run + gap * rng.uniform(0.6, 1.4)
            else:
                position += gap
    return out


# --------------------------------------------------------------------------
# 5. stipple
# --------------------------------------------------------------------------
def _render_stipple(image: np.ndarray, p: dict, ctx: Context) -> list[np.ndarray]:
    ink = _tone(_ink(image), p["tone"])
    pitch = ctx.px(p["pitch"], pen_scaled=True)
    rng = ctx.rng(5)

    points = _dither_points(ink, pitch, p["min_ink"], rng)
    if len(points) == 0:
        return []
    if len(points) > MAX_DOTS:
        keep = rng.choice(len(points), MAX_DOTS, replace=False)
        points = points[np.sort(keep)]
    points = lloyd_relax(points, ink ** max(p["weight"], 0.1), int(p["even_out"]))

    keep = sample_bilinear(ink, points[:, 0], points[:, 1]) > p["min_ink"]
    points = points[keep]
    if len(points) == 0:
        return []

    size = p["dot_size"]
    if size <= 0.01:
        return [point.reshape(1, 2) for point in points]

    radius = ctx.px(size, pen_scaled=True) / 2.0
    density = sample_bilinear(ink, points[:, 0], points[:, 1])
    height, width = ink.shape
    out = []
    for point, value in zip(points, density):
        r = radius * (0.35 + 0.65 * float(value)) if p["vary_size"] else radius
        r = min(r, point[0], point[1], width - 1 - point[0], height - 1 - point[1])
        if r < ctx.pen_px * 0.4:
            out.append(point.reshape(1, 2))
        else:
            out.append(geo.circle_path(float(point[0]), float(point[1]), r, 10))
    return out


def _render_dots(image: np.ndarray, p: dict, ctx: Context) -> list[np.ndarray]:
    """Dots whose spacing is the control: tight where it is dark, open where it is light.

    Stipple gets its density out of a dithering pass, which works but gives the
    user nothing to hold on to - the only way to make the darks darker is to
    change a pitch that also changes everything else.  Here the two ends of the
    range *are* the settings: how close dots get in the blackest area, and where
    they give up entirely.  Everything in between is interpolated along a curve.

    Placement is greedy dart-throwing with a spatially varying radius, darkest
    first, so the dark areas are laid down while there is still room for them.
    """
    ink = _tone(_ink(image), p["tone"])
    height, width = ink.shape

    tight = max(ctx.px(p["dark_spacing"], pen_scaled=True), ctx.pen_px * 0.8)
    loose = max(ctx.px(p["light_spacing"], pen_scaled=True), tight * 1.05)
    min_ink = float(p["min_ink"])
    curve = max(float(p["curve"]), 0.05)

    strength = np.clip((ink - min_ink) / max(1.0 - min_ink, 1e-6), 0.0, 1.0) ** curve
    # spacing runs from `loose` in the palest area it still draws, to `tight`
    # in the blackest one
    spacing = loose + (tight - loose) * strength

    step = max(tight * 0.62, 1.0)
    ys, xs = np.mgrid[0:height:step, 0:width:step]
    candidates = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)
    rng = ctx.rng(23)
    jitter = float(p["jitter"]) * step * 0.5
    if jitter > 0:
        candidates += rng.uniform(-jitter, jitter, candidates.shape)
    candidates[:, 0] = np.clip(candidates[:, 0], 0, width - 1)
    candidates[:, 1] = np.clip(candidates[:, 1], 0, height - 1)

    value = sample_bilinear(ink, candidates[:, 0], candidates[:, 1])
    keep = value > min_ink
    candidates = candidates[keep]
    value = value[keep]
    if len(candidates) == 0:
        return []
    # darkest first: the shadows get their dots while there is still room
    order = np.argsort(-value)
    candidates = candidates[order]

    want = sample_bilinear(spacing, candidates[:, 0], candidates[:, 1])
    cell = max(tight, 1.0)
    grid: dict[tuple[int, int], list[tuple[float, float]]] = {}
    accepted: list[np.ndarray] = []
    ctx.check()
    for index, (point, radius) in enumerate(zip(candidates, want)):
        if index % 4096 == 0:
            ctx.check()
        gx, gy = int(point[0] / cell), int(point[1] / cell)
        reach = int(radius / cell) + 1
        clash = False
        for cx in range(gx - reach, gx + reach + 1):
            for cy in range(gy - reach, gy + reach + 1):
                for other in grid.get((cx, cy), ()):  # noqa: B007
                    if (other[0] - point[0]) ** 2 + (other[1] - point[1]) ** 2 < radius * radius:
                        clash = True
                        break
                if clash:
                    break
            if clash:
                break
        if clash:
            continue
        grid.setdefault((gx, gy), []).append((point[0], point[1]))
        accepted.append(point)
        if len(accepted) >= MAX_DOTS:
            break

    if not accepted:
        return []
    size = float(p["dot_size"])
    if size <= 0.01:
        # a tap: one point, which the G-code turns into pen down / pen up
        return [np.asarray(point).reshape(1, 2) for point in accepted]

    radius_px = ctx.px(size, pen_scaled=True) / 2.0
    out: list[np.ndarray] = []
    for point in accepted:
        weight = float(sample_bilinear(ink, np.array([point[0]]), np.array([point[1]]))[0])
        r = radius_px * (0.4 + 0.6 * weight) if p["grow_with_tone"] else radius_px
        r = min(r, point[0], point[1], width - 1 - point[0], height - 1 - point[1])
        if r < ctx.pen_px * 0.4:
            out.append(np.asarray(point).reshape(1, 2))
        else:
            out.append(geo.circle_path(float(point[0]), float(point[1]), r, 10))
    return out


# --------------------------------------------------------------------------
# 6. halftone - a grid of circles
# --------------------------------------------------------------------------
def _render_halftone(image: np.ndarray, p: dict, ctx: Context) -> list[np.ndarray]:
    ink = _tone(_ink(image), p["tone"])
    height, width = ink.shape
    cell = ctx.px(p["cell"], pen_scaled=True)
    angle = math.radians(p["angle"])
    ca, sa = math.cos(angle), math.sin(angle)
    diagonal = math.hypot(width, height)
    steps = int(diagonal / cell) + 2
    centre = np.array([width / 2.0, height / 2.0])
    rings = max(int(p["rings"]), 1)
    shape = p["shape"]

    out: list[np.ndarray] = []
    for row in range(-steps // 2, steps // 2 + 1):
        ctx.check()
        for column in range(-steps // 2, steps // 2 + 1):
            local = np.array([column * cell, row * cell])
            x = centre[0] + local[0] * ca - local[1] * sa
            y = centre[1] + local[0] * sa + local[1] * ca
            if not (0 <= x < width and 0 <= y < height):
                continue
            value = float(sample_bilinear(ink, np.array([x]), np.array([y]))[0])
            if value <= p["min_ink"]:
                continue
            full = cell * 0.5 * math.sqrt(2) * p["max_fill"]
            radius = full * value
            # keep the whole dot on the paper, not just its centre
            radius = min(radius, x, y, width - 1 - x, height - 1 - y)
            if radius <= 0:
                continue
            if radius < ctx.pen_px * 0.35:
                out.append(np.array([[x, y]]))
                continue
            count = max(1, min(rings, int(radius / max(ctx.pen_px, 0.4))))
            for ring in range(count):
                r = radius * (ring + 1) / count
                if shape == "square":
                    out.append(
                        np.array([[x - r, y - r], [x + r, y - r], [x + r, y + r], [x - r, y + r], [x - r, y - r]])
                    )
                else:
                    out.append(geo.circle_path(x, y, r, max(8, int(6 + r))))
    return out


# --------------------------------------------------------------------------
# 7. spiral
# --------------------------------------------------------------------------
def _render_spiral(image: np.ndarray, p: dict, ctx: Context) -> list[np.ndarray]:
    ink = _tone(_ink(image), p["tone"])
    height, width = ink.shape
    cx, cy = width / 2.0, height / 2.0
    pitch = ctx.px(p["pitch"], pen_scaled=True)
    amplitude = ctx.px(p["amplitude"], pen_scaled=True)
    r_max = max(math.hypot(x - cx, y - cy) for x, y in ((0, 0), (width, 0), (0, height), (width, height)))

    turns = r_max / pitch
    if turns < 0.2:
        return []
    total = turns * 2 * math.pi
    step = max(pitch / 12.0, 0.5)
    thetas = [0.0]
    theta = 0.0
    while theta < total:
        radius = pitch * theta / (2 * math.pi)
        theta += step / max(radius, pitch / 6.0)
        thetas.append(theta)
    theta = np.asarray(thetas)
    radius = pitch * theta / (2 * math.pi)

    xs = cx + radius * np.cos(theta)
    ys = cy + radius * np.sin(theta)
    density = np.clip(sample_bilinear(ink, xs, ys), 0.0, 1.0)
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])
    wavelength = max(pitch / max(p["frequency"], 0.05), 1.0)
    wobble = amplitude * density * np.sin(2 * math.pi * arc / wavelength)
    r2 = radius + wobble
    xs2 = cx + r2 * np.cos(theta)
    ys2 = cy + r2 * np.sin(theta)

    inside = (xs2 >= 0) & (xs2 <= width - 1) & (ys2 >= 0) & (ys2 <= height - 1)
    if p["min_ink"] > 0:
        inside &= density > p["min_ink"]
    return split_by_mask(xs2, ys2, inside)


# --------------------------------------------------------------------------
# 8. rings - concentric circles modulated by the picture
# --------------------------------------------------------------------------
def _render_rings(image: np.ndarray, p: dict, ctx: Context) -> list[np.ndarray]:
    ink = _tone(_ink(image), p["tone"])
    height, width = ink.shape
    cx, cy = width / 2.0, height / 2.0
    pitch = ctx.px(p["pitch"], pen_scaled=True)
    amplitude = ctx.px(p["amplitude"], pen_scaled=True)
    r_max = max(math.hypot(x - cx, y - cy) for x, y in ((0, 0), (width, 0), (0, height), (width, height)))

    out: list[np.ndarray] = []
    radius = pitch
    while radius < r_max:
        steps = max(24, int(2 * math.pi * radius / 1.5))
        angles = np.linspace(0.0, 2 * math.pi, steps)
        xs = cx + radius * np.cos(angles)
        ys = cy + radius * np.sin(angles)
        density = np.clip(sample_bilinear(ink, xs, ys), 0.0, 1.0)
        waves = max(int(p["frequency"] * radius / pitch), 3)
        wobble = amplitude * density * np.sin(angles * waves)
        r2 = radius + wobble
        xs2 = cx + r2 * np.cos(angles)
        ys2 = cy + r2 * np.sin(angles)
        inside = (xs2 >= 0) & (xs2 <= width - 1) & (ys2 >= 0) & (ys2 <= height - 1)
        if p["min_ink"] > 0:
            inside &= density > p["min_ink"]
        out.extend(split_by_mask(xs2, ys2, inside))
        radius += pitch
    return out


# --------------------------------------------------------------------------
# 9. waves - parallel squiggles
# --------------------------------------------------------------------------
def _render_waves(image: np.ndarray, p: dict, ctx: Context) -> list[np.ndarray]:
    ink = _tone(_ink(image), p["tone"])
    height, width = ink.shape
    spacing = ctx.px(p["spacing"], pen_scaled=True)
    amplitude = ctx.px(p["amplitude"], pen_scaled=True)
    angle = math.radians(p["angle"])
    ca, sa = math.cos(angle), math.sin(angle)
    cx, cy = width / 2.0, height / 2.0
    diagonal = math.hypot(width, height)

    lines = int(diagonal / spacing) + 1
    step = max(spacing / 8.0, 0.4)
    count = int(diagonal / step) + 2
    t = (np.arange(count) * step) - diagonal / 2.0
    wavelength = max(spacing * 2.0 / max(p["frequency"], 0.05), 1.0)

    out: list[np.ndarray] = []
    for index in range(lines):
        offset = (index - lines / 2.0) * spacing
        base_x = cx + t * ca - offset * sa
        base_y = cy + t * sa + offset * ca
        density = np.clip(sample_bilinear(ink, base_x, base_y), 0.0, 1.0)
        wobble = amplitude * density * np.sin(2 * math.pi * t / wavelength)
        xs = base_x - wobble * sa
        ys = base_y + wobble * ca
        if index % 2 == 1:
            xs, ys, density = xs[::-1], ys[::-1], density[::-1]
        inside = (xs >= 0) & (xs <= width - 1) & (ys >= 0) & (ys <= height - 1) & (density > p["min_ink"])
        out.extend(split_by_mask(xs, ys, inside))
    return out


# --------------------------------------------------------------------------
# 10. flow field - strokes that follow the shapes
# --------------------------------------------------------------------------
def _render_flow(image: np.ndarray, p: dict, ctx: Context) -> list[np.ndarray]:
    ink = _tone(_ink(image), p["tone"])
    height, width = ink.shape
    fx, fy = _edge_tangents(image, p["coherence"] * 4.0)
    if p["across"]:
        fx, fy = -fy.copy(), fx.copy()

    spacing = max(ctx.px(p["spacing"], pen_scaled=True), 1.0)
    step = max(ctx.px(p["step"], pen_scaled=True), 0.7)
    max_steps = int(max(ctx.px(p["max_length"]) / step, 4))
    min_ink = p["min_ink"]

    cell = spacing
    grid = np.zeros((int(height / cell) + 2, int(width / cell) + 2), dtype=bool)

    # seed the darkest places first so the important detail gets the strokes
    rng = ctx.rng(13)
    # one seed per pixel at the smallest spacing meant 800 000 Python-loop
    # seeds; the pen cannot resolve that anyway
    # the 2.0 px floor used to be absolute, which quietly made a fine pen seed
    # far more strokes than a fat one for the same picture
    seed_step = max(spacing * 0.55, ctx.pen_px * 0.75, 1.0)
    ys, xs = np.mgrid[0 : height : seed_step, 0 : width : seed_step]
    seeds = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)
    seeds += rng.uniform(-seed_step * 0.4, seed_step * 0.4, seeds.shape)
    seeds[:, 0] = np.clip(seeds[:, 0], 0, width - 1)
    seeds[:, 1] = np.clip(seeds[:, 1], 0, height - 1)
    strength = sample_bilinear(ink, seeds[:, 0], seeds[:, 1])
    seeds = seeds[np.argsort(-strength)]
    strength = np.sort(strength)[::-1]

    out: list[np.ndarray] = []
    for index, (seed, value) in enumerate(zip(seeds, strength)):
        if index % 64 == 0:
            ctx.check()
        if value <= min_ink:
            break
        gy, gx = int(seed[1] / cell), int(seed[0] / cell)
        if grid[gy, gx]:
            continue
        points = [seed.copy()]
        for direction in (1.0, -1.0):
            x, y = float(seed[0]), float(seed[1])
            trail = []
            for _ in range(max_steps):
                ix, iy = int(round(x)), int(round(y))
                if not (0 <= ix < width and 0 <= iy < height):
                    break
                vx = float(fx[iy, ix]) * direction
                vy = float(fy[iy, ix]) * direction
                norm = math.hypot(vx, vy)
                if norm < 1e-6:
                    break
                # midpoint step keeps the curve smooth
                mx = x + vx / norm * step * 0.5
                my = y + vy / norm * step * 0.5
                imx, imy = int(round(mx)), int(round(my))
                if 0 <= imx < width and 0 <= imy < height:
                    vx2 = float(fx[imy, imx]) * direction
                    vy2 = float(fy[imy, imx]) * direction
                    if vx * vx2 + vy * vy2 < 0:
                        vx2, vy2 = -vx2, -vy2
                    norm2 = math.hypot(vx2, vy2)
                    if norm2 > 1e-6:
                        vx, vy, norm = vx2, vy2, norm2
                x += vx / norm * step
                y += vy / norm * step
                if not (0 <= x < width and 0 <= y < height):
                    break
                if float(sample_bilinear(ink, np.array([x]), np.array([y]))[0]) <= min_ink:
                    break
                if _occupancy_stop(grid, cell, x, y, mark=False):
                    break
                trail.append((x, y))
            if direction > 0:
                points.extend(trail)
            else:
                points = list(reversed(trail)) + points
        if len(points) < 3:
            continue
        path = np.asarray(points, dtype=np.float64)
        for x, y in path:
            _occupancy_stop(grid, cell, x, y, mark=True)
        out.append(path)
    return out


# --------------------------------------------------------------------------
# 11. scribble - one long chaotic line that shades the picture
# --------------------------------------------------------------------------
def _render_scribble(image: np.ndarray, p: dict, ctx: Context) -> list[np.ndarray]:
    """A wandering line that keeps steering into whatever ink is still unused.

    Each step looks a whole segment ahead - not just the end point - so the
    line follows shapes instead of rattling around inside them, and the ink it
    covers is subtracted so it spreads out over the whole picture.
    """
    ink = _tone(_ink(image), p["tone"]).astype(np.float32).copy()
    height, width = ink.shape
    step = max(ctx.px(p["step"], pen_scaled=True), 1.5)
    max_steps = int(p["length"] * 1000)
    rng = ctx.rng(17)
    turn_limit = math.radians(max(p["straightness"], 1.0))
    min_ink = p["min_ink"]

    # look ahead along each candidate segment, not just at its end point
    candidates = np.linspace(-turn_limit, turn_limit, 13)
    cos_d = np.cos(candidates)
    sin_d = np.sin(candidates)
    tap_steps = np.linspace(0.35, 1.0, 3) * step
    brush = max(ctx.px(p["ink_use"]), 1.0)
    brush_px = max(int(round(brush)), 1)
    margin = brush_px + 2

    # Finding the darkest remaining spot used to blur the whole image on every
    # jump, which dominated the run time.  Shrinking it instead is ~25x faster
    # and picks the same region.
    coarse_w = max(int(width / max(step * 2.0, 4.0)), 8)
    coarse_h = max(int(height / max(step * 2.0, 4.0)), 8)

    # The coarse map is rebuilt only now and then; between rebuilds the next
    # best cell is taken from the cached ranking and verified against the live
    # ink.  Rebuilding on every jump made resize() the single hottest call.
    cache: dict[str, object] = {"cells": [], "cursor": 0}
    floor = max(min_ink, 0.02)

    def rebuild_cache() -> None:
        coarse = cv2.resize(ink, (coarse_w, coarse_h), interpolation=cv2.INTER_AREA)
        flat = coarse.ravel()
        top = np.argsort(flat)[::-1][:48]
        cache["cells"] = [int(c) for c in top if flat[c] >= floor]
        cache["cursor"] = 0

    def refine(cell: int) -> tuple[float, float] | None:
        cy, cx = divmod(cell, coarse_w)
        x0 = int(cx * width / coarse_w)
        x1 = max(int((cx + 1) * width / coarse_w), x0 + 1)
        y0 = int(cy * height / coarse_h)
        y1 = max(int((cy + 1) * height / coarse_h), y0 + 1)
        patch = ink[y0:y1, x0:x1]
        if not patch.size:
            return None
        local = int(np.argmax(patch))
        py, px = divmod(local, patch.shape[1])
        if patch[py, px] < floor:
            return None
        return float(x0 + px), float(y0 + py)

    def darkest_start() -> tuple[float, float] | None:
        for attempt in range(2):
            cells = cache["cells"]
            while cache["cursor"] < len(cells):
                cell = cells[cache["cursor"]]
                cache["cursor"] += 1
                found = refine(cell)
                if found is not None:
                    return found
            if attempt == 0:
                rebuild_cache()
        return None

    start = darkest_start()
    if start is None:
        return []
    x, y = start
    heading = rng.uniform(0, 2 * math.pi)
    path = [(x, y)]
    out: list[np.ndarray] = []
    stalled = 0

    for _ in range(max_steps):
        # Rotate the precomputed candidate directions by the current heading
        # with the angle-addition identities: no trigonometry per step, and
        # nearest-neighbour sampling instead of bilinear, which is invisible at
        # this scale but several times cheaper.
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        dir_x = cos_d * cos_h - sin_d * sin_h
        dir_y = sin_d * cos_h + cos_d * sin_h

        score = np.zeros(len(candidates))
        reachable = np.ones(len(candidates), dtype=bool)
        for tap in tap_steps:
            sx = x + dir_x * tap
            sy = y + dir_y * tap
            reachable &= (sx >= 1) & (sx < width - 1) & (sy >= 1) & (sy < height - 1)
            np.clip(sx, 0, width - 1, out=sx)
            np.clip(sy, 0, height - 1, out=sy)
            score += ink[sy.astype(np.int32), sx.astype(np.int32)]
        score /= len(tap_steps)
        angles = heading + candidates
        score = np.where(reachable, score, -1.0)
        # a gentle pull towards going straight keeps the line from knotting up
        score -= np.abs(candidates) / max(turn_limit, 1e-6) * p["straight_bias"]
        score += rng.normal(0.0, p["chaos"], score.shape)

        best = int(np.argmax(score))
        if not reachable[best] or score[best] < min_ink * 0.5:
            # before breaking the line, try turning all the way round - a
            # continuous scribble looks far better than a heap of fragments
            wide = heading + np.linspace(-math.pi, math.pi, 25)
            wx = x + np.cos(wide) * step
            wy = y + np.sin(wide) * step
            ok = (wx >= 1) & (wx < width - 1) & (wy >= 1) & (wy < height - 1)
            wide_score = np.where(
                ok,
                sample_bilinear(ink, np.clip(wx, 0, width - 1), np.clip(wy, 0, height - 1)),
                -1.0,
            )
            alternative = int(np.argmax(wide_score))
            if ok[alternative] and wide_score[alternative] > min_ink * 0.6:
                heading = float(wide[alternative])
                angles = wide
                best = alternative
                score = wide_score
                reachable = ok
            elif len(path) > 2:
                out.append(np.asarray(path, dtype=np.float64))
                path = []

            if best is None or not reachable[best] or score[best] < min_ink * 0.5:
                start = darkest_start()
                if start is None:
                    break
                stalled += 1
                if stalled > 200:
                    # the picture cannot take any more line (or is smaller than
                    # one step); burning the remaining iterations helps nobody
                    break
                x, y = start
                heading = rng.uniform(0, 2 * math.pi)
                path = [(x, y)]
                continue

        heading = float(angles[best])
        next_x = x + math.cos(heading) * step
        next_y = y + math.sin(heading) * step

        # subtract the ink this stroke covers, but only inside its own little
        # bounding box - clearing a full-size buffer every step is what made
        # this technique crawl
        lo_x = max(int(min(x, next_x)) - margin, 0)
        hi_x = min(int(max(x, next_x)) + margin, width)
        lo_y = max(int(min(y, next_y)) - margin, 0)
        hi_y = min(int(max(y, next_y)) + margin, height)
        if hi_x > lo_x and hi_y > lo_y:
            patch = np.zeros((hi_y - lo_y, hi_x - lo_x), dtype=np.float32)
            cv2.line(
                patch,
                (int(x) - lo_x, int(y) - lo_y),
                (int(next_x) - lo_x, int(next_y) - lo_y),
                1.0,
                brush_px,
            )
            region = ink[lo_y:hi_y, lo_x:hi_x]
            np.clip(region - patch * p["fade"], 0.0, 1.0, out=region)

        x, y = next_x, next_y
        path.append((x, y))
        stalled = 0

    if len(path) > 2:
        out.append(np.asarray(path, dtype=np.float64))
    return out


# --------------------------------------------------------------------------
# 12. hilbert - one space-filling line, denser where the picture is dark
# --------------------------------------------------------------------------
def _hilbert_points(x0, y0, xi, xj, yi, yj, depth, ink, thresholds, base, out) -> None:
    """Adaptive Hilbert curve in the classic vector formulation.

    (x0, y0) is a corner, (xi, xj) the major axis and (yi, yj) the minor axis of
    the current square.  Recursing only where the picture is dark keeps the
    curve continuous while packing more line into the shadows.
    """
    centre = (x0 + (xi + yi) / 2.0, y0 + (xj + yj) / 2.0)
    if depth <= 0:
        out.append(centre)
        return

    size = math.hypot(xi, xj)
    if size < 3.0:
        out.append(centre)
        return

    height, width = ink.shape
    lo_x = int(max(0, min(width - 1, min(x0, x0 + xi + yi))))
    hi_x = int(max(1, min(width, max(x0, x0 + xi + yi) + 1)))
    lo_y = int(max(0, min(height - 1, min(y0, y0 + xj + yj))))
    hi_y = int(max(1, min(height, max(y0, y0 + xj + yj) + 1)))
    patch = ink[lo_y:hi_y, lo_x:hi_x]
    value = float(patch.mean()) if patch.size else 0.0
    level = len(thresholds) - depth
    if level >= base and 0 <= level < len(thresholds) and value < thresholds[level]:
        out.append(centre)
        return

    _hilbert_points(x0, y0, yi / 2, yj / 2, xi / 2, xj / 2, depth - 1, ink, thresholds, base, out)
    _hilbert_points(x0 + xi / 2, y0 + xj / 2, xi / 2, xj / 2, yi / 2, yj / 2, depth - 1, ink, thresholds, base, out)
    _hilbert_points(
        x0 + xi / 2 + yi / 2, y0 + xj / 2 + yj / 2, xi / 2, xj / 2, yi / 2, yj / 2, depth - 1, ink, thresholds, base, out
    )
    _hilbert_points(
        x0 + xi / 2 + yi, y0 + xj / 2 + yj, -yi / 2, -yj / 2, -xi / 2, -xj / 2, depth - 1, ink, thresholds, base, out
    )


def _render_hilbert(image: np.ndarray, p: dict, ctx: Context) -> list[np.ndarray]:
    ink = _tone(_ink(image), p["tone"])
    height, width = ink.shape
    size = float(max(width, height))
    # The finest fold wants to be about one pen wide - any finer and the ink
    # merges into a block.  That sets the sensible depth; the slider then biases
    # it up or down from there.  (Deriving it outright, as this did, made the
    # slider do nothing at all.)
    if ctx.scale_with_pen:
        natural = int(math.ceil(math.log2(max(size / max(ctx.pen_px * 2.0, 1.0), 2.0))))
        depth = int(np.clip(natural + int(p["depth"]) - 6, int(p["base"]) + 1, 11))
    else:
        depth = int(p["depth"])
    thresholds = [p["threshold"] * (index + 1) / depth for index in range(depth)]

    points: list[tuple[float, float]] = []
    _hilbert_points(0.0, 0.0, size, 0.0, 0.0, size, depth, ink, thresholds, int(p["base"]), points)
    if len(points) < 2:
        return []

    path = np.asarray(points, dtype=np.float64)
    inside = (path[:, 0] >= 0) & (path[:, 0] <= width - 1) & (path[:, 1] >= 0) & (path[:, 1] <= height - 1)
    if p["min_ink"] > 0:
        density = sample_bilinear(ink, np.clip(path[:, 0], 0, width - 1), np.clip(path[:, 1], 0, height - 1))
        inside &= density > p["min_ink"]
    return split_by_mask(path[:, 0], path[:, 1], inside)


# --------------------------------------------------------------------------
# 13. mesh - Delaunay triangulation of stipple points
# --------------------------------------------------------------------------
def _render_mesh(image: np.ndarray, p: dict, ctx: Context) -> list[np.ndarray]:
    ink = _tone(_ink(image), p["tone"])
    height, width = ink.shape
    pitch = ctx.px(p["pitch"], pen_scaled=True)
    rng = ctx.rng(19)

    points = _dither_points(ink, pitch, p["min_ink"], rng)
    if len(points) < 3:
        return []
    points = lloyd_relax(points, ink ** 1.5, int(p["even_out"]))

    subdiv = cv2.Subdiv2D((0, 0, width, height))
    for point in points:
        x = float(np.clip(point[0], 0, width - 1))
        y = float(np.clip(point[1], 0, height - 1))
        subdiv.insert((x, y))

    max_edge = ctx.px(p["max_edge"], pen_scaled=True)
    seen: set[tuple[int, int, int, int]] = set()
    out: list[np.ndarray] = []
    for x0, y0, x1, y1 in subdiv.getEdgeList():
        if not (0 <= x0 < width and 0 <= y0 < height and 0 <= x1 < width and 0 <= y1 < height):
            continue
        if math.hypot(x1 - x0, y1 - y0) > max_edge:
            continue
        key = tuple(sorted(((int(x0), int(y0)), (int(x1), int(y1)))))
        flat = (key[0][0], key[0][1], key[1][0], key[1][1])
        if flat in seen:
            continue
        seen.add(flat)
        out.append(np.array([[x0, y0], [x1, y1]], dtype=np.float64))
    return out


# --------------------------------------------------------------------------
# 14. silhouette - traced (and optionally filled) dark shapes
# --------------------------------------------------------------------------
def _render_silhouette(image: np.ndarray, p: dict, ctx: Context) -> list[np.ndarray]:
    return contour_paths(
        image,
        method=p["threshold_mode"],
        threshold=p["level"],
        block=int(p["local_area"]) | 1,
        offset=p["local_offset"],
        despeckle=int(p["despeckle"]),
        min_area_px=max(ctx.px(p["min_size"]) ** 2, 3.0),
        smooth=max(ctx.px(p["simplify"]), 0.3),
        fill_spacing=ctx.pen_px / max(p["fill"], 0.01) if p["fill"] > 0 else 0.0,
        fill_angle=p["fill_angle"],
    )


# --------------------------------------------------------------------------
# 15. dwell dots - darkness from how long the pen rests, not from dot size
# --------------------------------------------------------------------------
def _render_dwell_dots(image: np.ndarray, p: dict, ctx: Context) -> list[tuple[float, list[np.ndarray]]]:
    """Even grid of dots whose *darkness* comes from pen dwell time.

    With a fibre or fountain tip the ink keeps wicking into the paper for as
    long as the pen rests on it, so a 400 ms dot is visibly darker and fatter
    than a 20 ms one.  That gives a proper grey scale from a single pen without
    changing the dot geometry at all.  A ballpoint cannot do this - it needs
    movement to write - which is why this technique is flagged in the interface.
    """
    ink = _tone(_ink(image), p["tone"])
    pitch = ctx.px(p["pitch"], pen_scaled=True)
    rng = ctx.rng(23)
    height, width = ink.shape

    # a jittered grid keeps the spacing even; the tone lives in the dwell time
    step = max(pitch, 1.0)
    ys, xs = np.mgrid[step / 2 : height : step, step / 2 : width : step]
    points = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)
    if len(points) == 0:
        return []
    jitter = p["jitter"] * step * 0.5
    if jitter > 0:
        points += rng.uniform(-jitter, jitter, points.shape)
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)

    # optionally let the density follow the tone as well, like classic stippling
    follow = float(p["density_follows_tone"])
    if follow > 0.01:
        weight = np.clip(ink, 1e-3, None) ** (2.0 * follow)
        points = lloyd_relax(points, weight, 2)
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)

    density = np.clip(sample_bilinear(ink, points[:, 0], points[:, 1]), 0.0, 1.0)
    keep = density > p["min_ink"]
    points = points[keep]
    density = density[keep]
    if len(points) == 0:
        return []

    curve = max(p["dwell_curve"], 0.05)
    shortest = min(p["min_dwell"], p["max_dwell"])
    longest = max(p["min_dwell"], p["max_dwell"])
    dwell = shortest + (longest - shortest) * np.power(density, curve)

    # quantise into a handful of dwell steps so the G-code stays compact
    levels = max(int(p["levels"]), 1)
    if longest - shortest < 1e-6:
        buckets = np.zeros(len(dwell), dtype=np.int64)
        edges = np.array([shortest])
    else:
        edges = np.linspace(shortest, longest, levels)
        buckets = np.clip(
            np.round((dwell - shortest) / (longest - shortest) * (levels - 1)).astype(np.int64), 0, levels - 1
        )

    groups: list[tuple[float, list[np.ndarray]]] = []
    for level in range(len(edges)):
        chosen = points[buckets == level]
        if not len(chosen):
            continue
        groups.append((float(edges[level]), [point.reshape(1, 2) for point in chosen]))
    return groups


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------
def _tone_param(default: float = 1.0) -> Param:
    return Param(
        "tone", "Tone curve", default, 0.3, 3.0, 0.05, 2,
        hint="Below 1 lightens the midtones, above 1 darkens them.",
    )


def _min_ink_param(default: float = 0.06) -> Param:
    return Param("min_ink", "Skip lighter than", default, 0.0, 0.6, 0.01, 2)


REGISTRY: dict[str, Technique] = {}


def _add(technique: Technique) -> None:
    REGISTRY[technique.key] = technique


_add(Technique(
    "sketch", "Sketch", "line",
    "Traces the edges in the picture as single strokes - the classic line-art look.",
    [
        Param("auto_threshold", "Find edges automatically", True, kind="bool"),
        Param("sensitivity", "Sensitivity", 50.0, 5.0, 100.0, 1.0, 0,
              hint="Higher finds more, fainter edges."),
        Param("low", "Weak edges", 60.0, 1.0, 250.0, 5.0, 0),
        Param("high", "Strong edges", 150.0, 1.0, 400.0, 5.0, 0),
        Param("softness", "Denoise", 1.2, 0.1, 6.0, 0.1, 1, "px"),
        Param("min_length", "Ignore shorter than", 1.0, 0.1, 15.0, 0.1, 2, "mm", pen_scaled=True),
        Param("wobble", "Hand wobble", 0.0, 0.0, 2.0, 0.05, 2, "mm",
              hint="Adds a gentle waver so the line looks drawn, not plotted."),
        Param("overshoot", "Overshoot", 0.0, 0.0, 5.0, 0.1, 2, "mm", pen_scaled=True),
        Param("passes", "Passes", 1, 1, 3, 1, 0,
              hint="Draw every line more than once for a sketchy, built-up look."),
    ],
    _render_sketch,
))

_add(Technique(
    "contours", "Contour lines", "line",
    "Draws the picture as a height map - one closed line per brightness level.",
    [
        Param("levels", "Levels", 12, 2, 40, 1, 0),
        Param("smoothing", "Smoothing", 2.5, 0.1, 12.0, 0.1, 1, "px"),
        Param("simplify", "Simplify", 0.2, 0.02, 2.0, 0.02, 2, "mm", pen_scaled=True),
        Param("min_size", "Ignore smaller than", 1.5, 0.2, 20.0, 0.1, 1, "mm", pen_scaled=True),
        _tone_param(),
    ],
    _render_contours,
))

_add(Technique(
    "crosshatch", "Crosshatch", "shading",
    "Layers of parallel lines at different angles - the most photographic shading.",
    [
        Param("layers", "Layers", 4, 1, 6, 1, 0),
        Param("coverage", "Ink coverage", 0.85, 0.15, 1.4, 0.05, 2,
              hint="How black the darkest areas get. 1.0 means the pen covers the paper completely."),
        Param("angle", "First angle", 45.0, 0.0, 180.0, 5.0, 0, "°"),
        Param("angle_step", "Angle per layer", 45.0, 0.0, 90.0, 5.0, 0, "°"),
        # 0.35 shattered a continuous hatch into a few hundred extra fragments -
        # measured 818 strokes against 304 for the same picture - which read as
        # scratchy dashes rather than tone.  0.12 still softens the tone steps.
        Param("dither", "Soften banding", 0.12, 0.0, 1.5, 0.05, 2,
              hint="Breaks up the hard edges between tone steps."),
        Param("sketchiness", "Sketchiness", 0.0, 0.0, 4.0, 0.1, 2, "mm"),
        Param("min_length", "Ignore shorter than", 0.6, 0.0, 10.0, 0.1, 2, "mm"),
        _tone_param(),
    ],
    _render_crosshatch,
    stitchable=True,
))

_add(Technique(
    "dashes", "Pencil dashes", "shading",
    "Broken strokes whose length follows the tone - looks like quick pencil shading.",
    [
        Param("spacing", "Line spacing", 1.4, 0.2, 8.0, 0.05, 2, "mm", pen_scaled=True),
        Param("angle", "Angle", 55.0, 0.0, 180.0, 5.0, 0, "°"),
        Param("dash", "Dash length", 4.0, 0.5, 30.0, 0.5, 1, "mm", pen_scaled=True),
        Param("gap", "Gap", 1.2, 0.2, 10.0, 0.1, 2, "mm", pen_scaled=True),
        _min_ink_param(0.08),
        _tone_param(),
    ],
    _render_dashes,
))

_add(Technique(
    "dots", "Dot shading", "dots",
    "Nothing but taps - the pen touches the paper and lifts. The dots crowd "
    "together where the picture is dark and thin out where it is light, and "
    "the two spacings are the settings.",
    [
        Param("dark_spacing", "Closest dots", 0.6, 0.2, 6.0, 0.05, 2, "mm", pen_scaled=True,
              hint="How close the dots get in the blackest area."),
        Param("light_spacing", "Furthest apart", 2.6, 0.5, 20.0, 0.1, 2, "mm", pen_scaled=True,
              hint="Spacing in the palest area that still gets any dots at all."),
        Param("curve", "Density curve", 1.0, 0.2, 3.0, 0.05, 2,
              hint="Below 1 spreads the dots into the midtones; above 1 keeps them in the shadows."),
        Param("jitter", "Scatter", 0.8, 0.0, 1.0, 0.05, 2,
              hint="0 lines the dots up on a grid; 1 scatters them like hand stippling."),
        Param("dot_size", "Dot size", 0.0, 0.0, 4.0, 0.05, 2, "mm", pen_scaled=True,
              hint="0 taps a single point - the fastest and the cleanest. Larger draws a circle."),
        Param("grow_with_tone", "Bigger dots in the dark", True, kind="bool"),
        _min_ink_param(),
        _tone_param(),
    ],
    _render_dots,
    slow=True,
))

_add(Technique(
    "stipple", "Stipple", "dots",
    "Dots placed by weighted relaxation, so the spacing is even and the tone is right.",
    [
        Param("pitch", "Dot pitch", 1.0, 0.25, 8.0, 0.05, 2, "mm", pen_scaled=True),
        Param("even_out", "Even out", 2, 0, 8, 1, 0,
              hint="Relaxation passes. More gives an evenly spread look, but softens the tone."),
        Param("weight", "Weighting", 2.5, 0.2, 5.0, 0.1, 2,
              hint="How strongly the dots crowd into the dark areas."),
        Param("dot_size", "Dot size", 0.0, 0.0, 4.0, 0.05, 2, "mm", pen_scaled=True,
              hint="0 dabs a single point. Larger draws a small circle."),
        Param("vary_size", "Vary size with tone", True, kind="bool"),
        _min_ink_param(),
        _tone_param(),
    ],
    _render_stipple,
    slow=True,
))

_add(Technique(
    "dwell", "Dwell dots", "dots",
    "Even dots that get darker the longer the pen rests. Needs an ink pen that keeps "
    "flowing when it stands still - a fibre tip, fountain pen, gel pen or marker.",
    [
        Param("pitch", "Dot pitch", 1.6, 0.4, 10.0, 0.05, 2, "mm", pen_scaled=True),
        Param("min_dwell", "Lightest dot", 15.0, 0.0, 500.0, 5.0, 0, "ms",
              hint="Time the pen rests for the palest tone."),
        Param("max_dwell", "Darkest dot", 320.0, 10.0, 2000.0, 10.0, 0, "ms",
              hint="Time the pen rests for solid black. Longer means a bigger, darker blot."),
        Param("levels", "Dwell steps", 6, 2, 16, 1, 0,
              hint="How many different dwell times to use."),
        Param("dwell_curve", "Tone curve", 1.0, 0.2, 3.0, 0.05, 2),
        Param("density_follows_tone", "Density follows tone", 0.0, 0.0, 1.0, 0.05, 2,
              hint="0 keeps a perfectly even grid. Higher also crowds the dots into the shadows."),
        Param("jitter", "Jitter", 0.25, 0.0, 1.0, 0.05, 2),
        _min_ink_param(0.05),
        _tone_param(),
    ],
    _render_dwell_dots,
    grouped=True,
    requires="bleeding_pen",
))

_add(Technique(
    "halftone", "Halftone", "dots",
    "A rotated grid of circles that grow with the tone, like a newspaper print.",
    [
        Param("cell", "Cell size", 3.0, 0.6, 20.0, 0.1, 2, "mm", pen_scaled=True),
        Param("angle", "Screen angle", 45.0, 0.0, 90.0, 5.0, 0, "°"),
        Param("max_fill", "Largest dot", 0.95, 0.2, 1.4, 0.05, 2),
        Param("rings", "Rings per dot", 3, 1, 8, 1, 0,
              hint="How many circles fill a dark dot."),
        Param("shape", "Shape", "circle", kind="choice",
              choices={"circle": "Circles", "square": "Squares"}),
        _min_ink_param(0.04),
        _tone_param(),
    ],
    _render_halftone,
))

_add(Technique(
    "flow", "Flow field", "shading",
    "Strokes that follow the shapes in the picture, like brush marks or engraving.",
    [
        Param("spacing", "Stroke spacing", 1.2, 0.3, 8.0, 0.05, 2, "mm", pen_scaled=True),
        Param("step", "Smoothness", 0.6, 0.2, 4.0, 0.05, 2, "mm", pen_scaled=True),
        Param("max_length", "Max stroke length", 40.0, 3.0, 300.0, 1.0, 1, "mm"),
        Param("coherence", "Follow the form", 1.5, 0.2, 6.0, 0.1, 2,
              hint="Higher makes the strokes flow together more calmly."),
        Param("across", "Cross the shapes instead", False, kind="bool"),
        _min_ink_param(0.10),
        _tone_param(),
    ],
    _render_flow,
    stitchable=True,
    slow=True,
))

_add(Technique(
    "scribble", "Scribble", "shading",
    "One long wandering line that keeps returning to the dark areas.",
    [
        Param("length", "Line length", 20.0, 1.0, 80.0, 1.0, 0, "k steps",
              hint="How long the line may get. It stops early once the picture is covered."),
        Param("step", "Step", 1.2, 0.3, 8.0, 0.1, 2, "mm", pen_scaled=True),
        Param("straightness", "Turn limit", 70.0, 10.0, 170.0, 5.0, 0, "°"),
        Param("straight_bias", "Prefer straight", 0.10, 0.0, 0.5, 0.01, 2),
        Param("chaos", "Chaos", 0.05, 0.0, 0.4, 0.01, 2),
        Param("ink_use", "Stroke width", 1.4, 0.3, 8.0, 0.1, 2, "mm",
              hint="How wide a band each stroke counts as covered."),
        Param("fade", "Ink used per pass", 0.55, 0.05, 1.0, 0.05, 2,
              hint="Lower lets the line come back over the same area again."),
        _min_ink_param(0.08),
        _tone_param(),
    ],
    _render_scribble,
    stitchable=True,
    slow=True,
))

_add(Technique(
    "spiral", "Spiral", "geometric",
    "A single spiral from the centre that wobbles where the picture is dark.",
    [
        Param("pitch", "Turn spacing", 1.8, 0.4, 8.0, 0.05, 2, "mm", pen_scaled=True),
        Param("amplitude", "Wobble", 0.8, 0.0, 5.0, 0.05, 2, "mm", pen_scaled=True),
        Param("frequency", "Wobble rate", 1.0, 0.2, 6.0, 0.1, 2),
        _min_ink_param(0.0),
        _tone_param(),
    ],
    _render_spiral,
))

_add(Technique(
    "rings", "Concentric rings", "geometric",
    "Closed rings instead of one spiral - a cleaner, more graphic version.",
    [
        Param("pitch", "Ring spacing", 2.0, 0.4, 10.0, 0.05, 2, "mm", pen_scaled=True),
        Param("amplitude", "Wobble", 0.9, 0.0, 5.0, 0.05, 2, "mm", pen_scaled=True),
        Param("frequency", "Waves per ring", 1.5, 0.2, 8.0, 0.1, 2),
        _min_ink_param(0.05),
        _tone_param(),
    ],
    _render_rings,
))

_add(Technique(
    "waves", "Wave lines", "geometric",
    "Parallel lines whose amplitude follows the tone.",
    [
        Param("spacing", "Line spacing", 1.8, 0.4, 8.0, 0.05, 2, "mm", pen_scaled=True),
        Param("amplitude", "Wobble", 0.8, 0.0, 5.0, 0.05, 2, "mm", pen_scaled=True),
        Param("frequency", "Wobble rate", 1.0, 0.2, 6.0, 0.1, 2),
        Param("angle", "Angle", 0.0, 0.0, 180.0, 5.0, 0, "°"),
        _min_ink_param(0.03),
        _tone_param(),
    ],
    _render_waves,
    stitchable=True,
))

_add(Technique(
    "hilbert", "Hilbert curve", "geometric",
    "One continuous space-filling line that folds tighter where the picture is dark.",
    [
        Param("base", "Base grid", 3, 1, 6, 1, 0,
              hint="How fine the curve is even where the paper is white."),
        Param("depth", "Detail depth", 6, 2, 9, 1, 0,
              hint="Each step doubles the resolution in the dark areas - and the drawing time."),
        Param("threshold", "Subdivide from", 0.35, 0.05, 1.0, 0.05, 2,
              hint="How dark an area has to be before the line folds tighter."),
        _min_ink_param(0.0),
        _tone_param(),
    ],
    _render_hilbert,
))

_add(Technique(
    "mesh", "Triangle mesh", "geometric",
    "Points spread by tone, then joined into a Delaunay web - a low-poly look.",
    [
        Param("pitch", "Point spacing", 3.0, 0.8, 15.0, 0.1, 2, "mm", pen_scaled=True),
        Param("even_out", "Even out", 3, 0, 8, 1, 0),
        Param("max_edge", "Longest edge", 12.0, 2.0, 60.0, 0.5, 1, "mm", pen_scaled=True),
        _min_ink_param(0.08),
        _tone_param(),
    ],
    _render_mesh,
    slow=True,
))

_add(Technique(
    "silhouette", "Silhouette", "line",
    "Traces the outline of the dark areas, and can fill them in.",
    [
        Param("threshold_mode", "Threshold", "otsu", kind="choice",
              choices={"otsu": "Automatic", "adaptive": "Local", "manual": "Manual"}),
        Param("level", "Level", 0.5, 0.0, 1.0, 0.02, 2),
        Param("local_area", "Local area", 25, 3, 199, 2, 0),
        Param("local_offset", "Local offset", 8.0, -30.0, 30.0, 1.0, 0),
        Param("despeckle", "Despeckle", 1, 0, 5, 1, 0),
        Param("min_size", "Ignore smaller than", 1.0, 0.1, 20.0, 0.1, 1, "mm"),
        Param("simplify", "Simplify", 0.15, 0.02, 2.0, 0.01, 2, "mm"),
        Param("fill", "Fill coverage", 0.0, 0.0, 1.4, 0.05, 2,
              hint="0 leaves the shapes as outlines. 1.0 fills them solid."),
        Param("fill_angle", "Fill angle", 45.0, 0.0, 180.0, 5.0, 0, "°"),
    ],
    _render_silhouette,
))


# --------------------------------------------------------------------------
def defaults_for(key: str) -> dict:
    technique = REGISTRY.get(key)
    return technique.defaults() if technique else {}


def resolve(key: str, stored: dict | None) -> dict:
    """Merge stored values over the defaults, ignoring anything unknown."""
    technique = REGISTRY.get(key)
    if technique is None:
        return {}
    values = technique.defaults()
    for param in technique.params:
        if stored and param.key in stored:
            raw = stored[param.key]
            try:
                if param.kind == "bool":
                    values[param.key] = bool(raw)
                elif param.kind == "int":
                    values[param.key] = int(raw)
                elif param.kind == "text":
                    values[param.key] = str(raw)
                elif param.kind == "choice":
                    values[param.key] = str(raw) if not param.choices or str(raw) in param.choices else param.default
                else:
                    values[param.key] = float(raw)
            except (TypeError, ValueError):
                pass
    return values


def render_groups(key: str, image: np.ndarray, stored: dict | None, ctx: Context):
    """Always returns [(dwell_ms, paths), ...] - the shape the pipeline wants."""
    technique = REGISTRY.get(key)
    if technique is None:
        return []
    result = technique.render(image, resolve(key, stored), ctx)
    if technique.grouped:
        return [(float(dwell), paths) for dwell, paths in result if paths]
    return [(0.0, result)] if result else []


def render(key: str, image: np.ndarray, stored: dict | None, ctx: Context) -> list[np.ndarray]:
    """Flat list of paths, ignoring any dwell grouping (previews, thumbnails)."""
    out: list[np.ndarray] = []
    for _dwell, paths in render_groups(key, image, stored, ctx):
        out.extend(paths)
    return out


def render_thumbnail(
    image: np.ndarray,
    key: str,
    stored: dict | None,
    size: int = 190,
    width_mm: float = 120.0,
    pen_width: float = 0.5,
) -> np.ndarray:
    """Small preview of a technique, drawn with OpenCV only.

    Deliberately Qt-free so the gallery can render every tile on a background
    thread and hand back plain arrays.
    """
    height, source_width = image.shape[:2]
    long_edge = max(source_width, height)
    # A very fine pen would ask for ten times the line at thumbnail size, which
    # nobody can see anyway.  The floor is the tile's own resolution rather than
    # a fixed 0.35 mm - that made 0.1, 0.2 and 0.35 mm pens render identically,
    # so a fineliner set looked like one pen.
    px_per_mm = long_edge / max(width_mm, 1.0)
    pen_width = max(float(pen_width), 0.9 / max(px_per_mm, 1e-6))
    width_mm = min(max(float(width_mm), 40.0), 400.0)
    context = Context(px_per_mm=long_edge / max(width_mm, 1.0), pen_width=pen_width)
    try:
        groups = render_groups(key, image, stored, context)
    except Exception:
        groups = []

    canvas = np.full((size, size), 255, dtype=np.uint8)
    if not groups:
        return canvas
    scale = (size - 4) / max(long_edge, 1)
    offset_x = (size - source_width * scale) / 2.0
    offset_y = (size - height * scale) / 2.0

    # a dwell technique gets its tone from time, not geometry, so the preview
    # has to show it as dot size or every tile would look like a plain grid
    dwells = [d for d, _ in groups if d > 0]
    lo = min(dwells) if dwells else 0.0
    hi = max(dwells) if dwells else 1.0

    polylines = []
    for dwell, paths in groups:
        if dwell > 0:
            weight = (dwell - lo) / max(hi - lo, 1e-6)
            radius = int(round(1 + weight * max(size / 90.0, 1.5)))
        else:
            radius = 0
        for path in paths:
            points = np.round(np.asarray(path) * scale + (offset_x, offset_y)).astype(np.int32)
            if len(points) == 1:
                cv2.circle(canvas, (int(points[0][0]), int(points[0][1])), radius, 40, -1)
            else:
                polylines.append(points)
    if polylines:
        cv2.polylines(canvas, polylines, False, 40, 1, cv2.LINE_AA)
    return canvas


# The extra techniques live in their own module and register themselves here.
# The import sits at the very bottom so the registry above is already built.
from .techniques_extra import register as _register_extra  # noqa: E402

_register_extra()
