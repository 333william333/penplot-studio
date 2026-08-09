"""The pipeline: source -> style -> layout -> optimisation -> plot job.

This is the only place that knows how all the pieces fit together, which keeps
the UI thin and makes the whole chain testable without a window on screen.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from . import geometry as geo
from . import knife, raster, separation, techniques
from .drawing import Drawing, Layer, SourceResult
from .pens import PenLibrary, srgb_to_lab
from .settings import AppSettings

__all__ = ["PlotJob", "PlotStats", "build_plot", "build_project", "REFERENCE_PEN_WIDTH", "Cancelled"]


#: re-exported so callers only need one import
Cancelled = techniques.Cancelled

# Spacing settings are authored for a 0.5 mm pen; wider pens scale up from here.
REFERENCE_PEN_WIDTH = 0.5


@dataclass
class PlotStats:
    draw_length: float = 0.0
    travel_length: float = 0.0
    estimated_seconds: float = 0.0
    path_count: int = 0
    pen_lifts: int = 0
    dot_count: int = 0
    bounds: tuple[float, float, float, float] | None = None
    out_of_bounds: bool = False
    per_pen: dict[int, dict] = field(default_factory=dict)
    build_seconds: float = 0.0


@dataclass
class PlotJob:
    drawing: Drawing = field(default_factory=Drawing)
    stats: PlotStats = field(default_factory=PlotStats)
    warnings: list[str] = field(default_factory=list)
    target_size: tuple[float, float] = (0.0, 0.0)
    native_size: tuple[float, float] | None = None

    @property
    def is_empty(self) -> bool:
        return self.drawing.is_empty()


# --------------------------------------------------------------------------
# Stage memos.
#
# Dragging a slider re-runs the whole chain many times a second, and most of it
# is provably unchanged: moving the artwork on the bed, or picking a different
# pen colour, cannot alter the prepared raster.  Each memo keeps a couple of
# entries and holds a reference to the array it was keyed on, so the id() in the
# key cannot be recycled onto a different array.
_MEMO_SIZE = 3
_prepared_memo: dict = {}
_ink_memo: dict = {}
_paths_memo: dict = {}


def _memo_get(memo: dict, key):
    entry = memo.get(key)
    return entry[1] if entry is not None else None


def _memo_put(memo: dict, key, keep, value) -> None:
    if len(memo) >= _MEMO_SIZE:
        memo.pop(next(iter(memo)))
    memo[key] = (keep, value)


def clear_memos() -> None:
    """Drop every cached stage - used by the tests and when a source is closed."""
    _prepared_memo.clear()
    _ink_memo.clear()
    _paths_memo.clear()


def _scale_xy(path: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Pixels to millimetres, keeping any modulation column untouched."""
    xy = np.column_stack([path[:, 0] * sx, path[:, 1] * sy])
    return np.hstack([xy, path[:, 2:]]) if path.shape[1] > 2 else xy


def _pen_scale(width: float, enabled: bool) -> float:
    if not enabled:
        return 1.0
    return max(width, 0.05) / REFERENCE_PEN_WIDTH


def _source_aspect(source: SourceResult, prepared: np.ndarray | None) -> float:
    if prepared is not None:
        h, w = prepared.shape[:2]
        return w / max(h, 1)
    paths = []
    for layer in source.layers or []:
        paths.extend(layer.paths)
    if source.vector is not None:
        paths.extend(source.vector.paths)
    bounds = geo.bounds(paths)
    if not bounds:
        return 1.0
    width = max(bounds[2] - bounds[0], 1e-6)
    height = max(bounds[3] - bounds[1], 1e-6)
    return width / height


def _as_drawn(settings: AppSettings, source: SourceResult, native) -> bool:
    """Is this really geometry that is already in bed millimetres?

    Only hand-drawn strokes and cut sheets are.  A picture has no such frame -
    applying the as-drawn matrix to one pins it to the top-left of the bed at
    whatever size the fit branch happened to return, which is how a dropped
    image ended up hanging off the sheet.
    """
    return (
        settings.layout.mode == "asis"
        and getattr(settings.item, "kind", "") == "shapes"
        and source.kind == "shapes"
        and bool(native)
    )


def _target_size(settings: AppSettings, source: SourceResult, aspect: float, native: tuple[float, float] | None):
    layout = settings.layout
    machine = settings.machine
    usable_w = max(machine.bed_x - 2 * layout.margin, 10.0)
    usable_h = max(machine.bed_y - 2 * layout.margin, 10.0)

    if _as_drawn(settings, source, native):
        # hand-drawn strokes are already in bed millimetres
        return native
    if layout.mode == "scale" and native:
        factor = max(layout.scale_percent, 1.0) / 100.0
        return native[0] * factor, native[1] * factor
    if layout.mode == "size":
        width = max(layout.width, 1.0)
        height = max(layout.height, 1.0) if not layout.keep_aspect else width / max(aspect, 1e-6)
        return width, height

    # Fit: a rotated rectangle needs its *rotated* bounding box to fit, not the
    # upright one.  At 45 degrees that box is up to 1.41x wider, which used to
    # push the artwork straight off the bed.
    angle = math.radians(layout.rotation)
    cos_a, sin_a = abs(math.cos(angle)), abs(math.sin(angle))
    unit_w = max(aspect, 1e-6)
    unit_h = 1.0
    box_w = unit_w * cos_a + unit_h * sin_a
    box_h = unit_w * sin_a + unit_h * cos_a
    scale = min(usable_w / max(box_w, 1e-6), usable_h / max(box_h, 1e-6))
    return unit_w * scale, unit_h * scale


def _layout_matrix(settings: AppSettings, paths_mm: list[np.ndarray], as_drawn: bool = False) -> np.ndarray:
    """Mirror + flip Y + rotate, then place the artwork on the bed."""
    layout = settings.layout
    machine = settings.machine

    mirror_x = -1.0 if layout.mirror_x else 1.0
    mirror_y = -1.0 if layout.mirror_y else 1.0
    base = geo.affine(scale_x=mirror_x, scale_y=-mirror_y, rotation_deg=layout.rotation)

    if as_drawn:
        # The source stored these as bed_y - y, so flipping and adding the bed
        # height puts every point back exactly where it was drawn.
        flip = geo.affine(scale_y=-1.0)
        placed = geo.affine(translate=(layout.offset_x, machine.bed_y + layout.offset_y)) @ flip
        if abs(layout.rotation) < 1e-9 and not layout.mirror_x and not layout.mirror_y:
            return placed
        # Rotating about the bed origin would swing a drawing clean off the
        # sheet, so turn it about its own centre and leave it where it is.
        bounds = geo.bounds(geo.transform_paths(paths_mm, placed))
        if not bounds:
            return placed
        pivot = ((bounds[0] + bounds[2]) / 2.0, (bounds[1] + bounds[3]) / 2.0)
        spin = geo.affine(
            scale_x=mirror_x, scale_y=mirror_y, rotation_deg=layout.rotation, pivot=pivot
        )
        return spin @ placed

    moved = geo.transform_paths(paths_mm, base)
    bounds = geo.bounds(moved)
    if not bounds:
        return base

    lo_x, lo_y, hi_x, hi_y = bounds
    width = hi_x - lo_x
    height = hi_y - lo_y
    if layout.center:
        target_cx = machine.bed_x / 2.0 + layout.offset_x
        target_cy = machine.bed_y / 2.0 + layout.offset_y
        dx = target_cx - (lo_x + width / 2.0)
        dy = target_cy - (lo_y + height / 2.0)
    else:
        dx = layout.margin + layout.offset_x - lo_x
        dy = layout.margin + layout.offset_y - lo_y

    shift = geo.affine(translate=(dx, dy))
    return shift @ base


# --------------------------------------------------------------------------
def _style_paths(image: np.ndarray, settings: AppSettings, px_per_mm: float, pen_width: float) -> list[np.ndarray]:
    """Run the chosen technique.  Every millimetre setting is converted inside."""
    context = techniques.Context(
        px_per_mm=px_per_mm,
        pen_width=pen_width,
        scale_with_pen=settings.pen.scale_with_pen_width,
    )
    style = settings.style
    return techniques.render(style.technique, image, style.params.get(style.technique), context)


def _style_groups(image: np.ndarray, settings: AppSettings, px_per_mm: float, pen_width: float, should_cancel=None):
    """Same as `_style_paths` but keeps the dwell grouping some techniques use."""
    context = techniques.Context(
        px_per_mm=px_per_mm,
        pen_width=pen_width,
        scale_with_pen=settings.pen.scale_with_pen_width,
        should_cancel=should_cancel,
    )
    style = settings.style
    return techniques.render_groups(style.technique, image, style.params.get(style.technique), context)


def _attach_modulation(paths: list[np.ndarray], ink_map: np.ndarray, resample_px: float) -> list[np.ndarray]:
    """Append a per-point darkness column, sampled from the ink map.

    This is what lets the *machine* carry tone: the same stroke can be pressed
    harder or drawn slower where the picture is dark, so a single pen produces
    a range of line weights.  Long segments are subdivided first, otherwise the
    modulation could only change at existing corners.
    """
    from .styles import sample_bilinear

    height, width = ink_map.shape
    out: list[np.ndarray] = []
    for path in paths:
        points = np.asarray(path, dtype=np.float64)
        if len(points) < 2:
            out.append(points)
            continue
        if resample_px > 0.5:
            dense = [points[0]]
            for a, b in zip(points[:-1], points[1:]):
                distance = float(np.hypot(b[0] - a[0], b[1] - a[1]))
                steps = int(distance // resample_px)
                if steps > 0:
                    for t in np.linspace(0.0, 1.0, steps + 2)[1:-1]:
                        dense.append(a + (b - a) * t)
                dense.append(b)
            points = np.asarray(dense, dtype=np.float64)
        weights = np.clip(
            sample_bilinear(ink_map, np.clip(points[:, 0], 0, width - 1), np.clip(points[:, 1], 0, height - 1)),
            0.0,
            1.0,
        )
        out.append(np.hstack([points[:, :2], weights[:, None]]))
    return out


def _assign_vector_pens(art, library: PenLibrary, mode: str) -> dict[int, list[np.ndarray]]:
    """Group coloured PDF line work onto the closest matching pens."""
    buckets: dict[int, list[np.ndarray]] = {}
    enabled = [i for i, p in enumerate(library) if p.enabled] or [0]
    if mode == "mono" or len(enabled) == 1:
        buckets[enabled[0]] = list(art.paths)
        return buckets

    pen_labs = {i: np.asarray(library[i].lab, dtype=np.float64) for i in enabled}
    paper = np.asarray([100.0, 0.0, 0.0])
    for path, color in zip(art.paths, art.colors):
        lab = np.asarray(srgb_to_lab(color), dtype=np.float64)
        if float(np.linalg.norm(lab - paper)) < 12.0:
            continue  # essentially white - leave the paper alone
        best = min(enabled, key=lambda i: float(np.linalg.norm(lab - pen_labs[i])))
        buckets.setdefault(best, []).append(path)
    return buckets


def _expand_dots(paths: list[np.ndarray], diameter: float) -> list[np.ndarray]:
    if diameter <= 0.01:
        return paths
    radius = diameter / 2.0
    out = []
    for p in paths:
        if len(p) == 1:
            out.append(geo.circle_path(float(p[0][0]), float(p[0][1]), radius, 10))
        else:
            out.append(p)
    return out


def _warn_if_too_fine(result, layers, library: PenLibrary) -> None:
    """Say so when the nib is fatter than the detail it is being asked to draw.

    A thesis page at A4 has body text about 0.7 mm tall.  A 0.5 mm nib is 69 % of
    that: every counter fills in and the page plots as grey mud.  The machine
    will happily do it and waste an hour, so the number is worth saying before
    it does - with the two things that actually fix it.
    """
    heights = []
    for layer in layers:
        pen_width = library[layer.pen].width
        for path in layer.paths:
            if len(path) < 2:
                continue
            points = np.asarray(path)[:, :2]
            size = max(float(points[:, 0].max() - points[:, 0].min()),
                       float(points[:, 1].max() - points[:, 1].min()))
            if 0.05 < size < 40.0:
                heights.append((size, pen_width))
    if len(heights) < 40:
        return
    sizes = np.asarray([h[0] for h in heights])
    pen = float(np.median([h[1] for h in heights]))
    fine = float(np.percentile(sizes, 25))
    if fine <= 0 or pen / fine < 0.34:
        return
    bigger = pen / (fine * 0.34)
    remedy = f"fit a pen under {fine * 0.34:.2f} mm"
    if bigger >= 1.2:
        remedy = f"draw it {bigger:.1f}x bigger, or {remedy}"
    result.warnings.append(
        f"A quarter of this drawing is under {fine:.1f} mm across and the pen is "
        f"{pen:.2f} mm - the fine detail will fill in. To keep it: {remedy}."
    )


def _estimate_seconds(job_layers: list[Layer], settings: AppSettings, library: PenLibrary) -> tuple[float, float]:
    machine = settings.machine
    pen = settings.pen
    accel = max(machine.acceleration, 50.0)
    travel_accel = max(machine.travel_acceleration, accel)

    # The firmware clamps Z to its own maximum feedrate - stock Ender 3 firmware
    # allows only 5 mm/s - so asking for 900 mm/min does not make the lift any
    # faster.  Estimating with the real ceiling is the difference between a
    # believable number and one that is out by a factor of three.
    z_ceiling = machine.z_limit_target if machine.raise_z_limit else machine.z_max_feed
    z_effective = min(max(machine.z_feed, 1.0), max(z_ceiling, 1.0))
    # A 2.5 mm lift never gets near its top speed: at 100 mm/s^2 it is pure
    # acceleration from start to finish.  Estimating it with the XY figure made
    # the biggest single cost in the job come out at less than half its length.
    z_accel = max(
        machine.z_acceleration_target if machine.raise_z_limit else machine.z_acceleration,
        1.0,
    )

    def move_time(distance: float, feed_mm_min: float, acceleration: float | None = None) -> float:
        if distance <= 1e-9:
            return 0.0
        a = acceleration or accel
        v = max(feed_mm_min, 1.0) / 60.0
        if distance < v * v / a:
            return 2.0 * math.sqrt(distance / a)
        return distance / v + v / a

    travel_total = 0.0
    seconds = 0.0
    cursor = np.array([machine.park_x, machine.park_y], dtype=np.float64)
    full_lift = move_time(max(pen.lift, 0.1), z_effective, z_accel) * 2.0
    small_lift = move_time(max(min(pen.short_lift, pen.lift), 0.05), z_effective, z_accel) * 2.0
    delays = (pen.down_delay + pen.up_delay) / 1000.0
    change_point = np.array([pen.change_x, pen.change_y], dtype=np.float64)
    previous_pen: int | None = None

    for layer in job_layers:
        tool = library[layer.pen]
        feed = machine.draw_feed * max(tool.feed_scale, 0.05)
        dwell = max(layer.dwell_ms, 0.0) / 1000.0
        # a blade or a scoring pen goes over every stroke several times
        repeats = 1 if layer.modulation in ("pressure", "speed") else tool.repeats
        plunge = move_time(max(tool.pass_depth, 0.05), z_effective, z_accel) * (repeats - 1)
        # Speed modulation slows the pen down wherever the picture is dark, and
        # the G-code really does emit those lower feed rates; ignoring them made
        # the estimate up to 30 % short.
        slowdown = (
            min(max(layer.modulation_amount, 0.0), 0.95) * 0.8
            if layer.modulation == "speed"
            else 0.0
        )
        # Pressure modulation adds a Z move at every interior point.
        pressure_step = (
            move_time(max(layer.modulation_amount, 0.0) * 0.6, z_effective, z_accel)
            if layer.modulation == "pressure"
            else 0.0
        )
        if (
            previous_pen is not None
            and layer.pen != previous_pen
            and settings.pauses.pause_between_pens
            and settings.pauses.park_for_pause
        ):
            # the trip out to the pen-change position and back is real time
            hop = float(np.hypot(change_point[0] - cursor[0], change_point[1] - cursor[1]))
            travel_total += hop
            seconds += move_time(hop, machine.travel_feed, travel_accel)
            cursor = change_point.copy()
        previous_pen = layer.pen
        for path in layer.paths:
            if len(path) == 0:
                continue
            hop = float(np.hypot(path[0][0] - cursor[0], path[0][1] - cursor[1]))
            travel_total += hop
            lift = small_lift if hop <= pen.short_hop else full_lift
            seconds += move_time(hop, machine.travel_feed, travel_accel) + lift + delays + dwell
            path_feed = feed
            if slowdown > 0.0 and path.shape[1] > 2:
                mean_weight = float(np.clip(path[:, 2], 0.0, 1.0).mean())
                path_feed = max(feed * (1.0 - slowdown * mean_weight), 60.0)
            seconds += move_time(geo.path_length(path), path_feed) * repeats + plunge
            if pressure_step > 0.0 and path.shape[1] > 2:
                seconds += pressure_step * max(len(path) - 1, 0)
            cursor = path[-1]

    # and the trip home at the end
    park = np.array([machine.park_x, machine.park_y], dtype=np.float64)
    hop = float(np.hypot(park[0] - cursor[0], park[1] - cursor[1]))
    travel_total += hop
    seconds += move_time(hop, machine.travel_feed, travel_accel)
    return seconds, travel_total


# --------------------------------------------------------------------------
def build_plot(
    source: SourceResult | None,
    settings: AppSettings,
    library: PenLibrary,
    should_cancel=None,
) -> PlotJob:
    """Turn the current source + settings into everything the app needs.

    `should_cancel` is polled between stages so the window can shut down
    promptly instead of destroying a thread that is still working.
    """
    started = time.perf_counter()

    def check() -> None:
        if should_cancel is not None and should_cancel():
            raise Cancelled()

    job = PlotJob()
    if source is None:
        return job
    job.warnings.extend(source.warnings)

    prepared = None
    if source.is_raster:
        style = settings.style
        prepared_key = (
            id(source.rgb), source.rgb.shape, style.detail, style.enhance,
            style.brightness, style.contrast, style.gamma, style.blur, style.invert,
            style.auto_levels, style.black_point, style.white_point, style.saturation,
        )
        prepared = _memo_get(_prepared_memo, prepared_key)
        if prepared is None:
            base = source.rgb
            if style.enhance == "subject":
                # Local contrast on the subject and a background that falls back
                # towards paper - before the global curve, not after it: CLAHE
                # flattens the histogram, and stretching a flattened histogram
                # puts the hatching straight back onto the wall behind the head.
                base = raster.enhance_subject(base, local=0.60, background=0.45)
            elif style.enhance == "subject-light":
                # the same idea, gently: safe on any photograph even when the
                # app has no idea what the subject is
                base = raster.enhance_subject(base, local=0.40, background=0.22)
            prepared = raster.prepare(
                base,
                detail=style.detail,
                brightness=style.brightness,
                contrast=style.contrast,
                gamma=style.gamma,
                blur=style.blur,
                invert=style.invert,
                auto_levels=style.auto_levels,
                black_point=style.black_point,
                white_point=style.white_point,
                saturation=style.saturation,
            )
            _memo_put(_prepared_memo, prepared_key, source.rgb, prepared)

    aspect = _source_aspect(source, prepared)
    native = source.native_size_mm()
    as_drawn = _as_drawn(settings, source, native)
    target_w, target_h = _target_size(settings, source, aspect, native)
    job.target_size = (target_w, target_h)
    job.native_size = native

    layers: list[Layer] = []

    if prepared is not None:
        px_per_mm = prepared.shape[1] / max(target_w, 1e-6)
        ink_key = (
            id(prepared),
            settings.style.separation,
            settings.style.paper_lightness,
            settings.style.ink_gamma,
            # deliberately not pen.width: the separation is about colour only
            tuple((p.color, p.enabled) for p in library),
        )
        ink_maps = _memo_get(_ink_memo, ink_key)
        if ink_maps is None:
            ink_maps = separation.separate(
                prepared,
                library,
                mode=settings.style.separation,
                paper_lightness=settings.style.paper_lightness,
                ink_gamma=settings.style.ink_gamma,
            )
            _memo_put(_ink_memo, ink_key, prepared, ink_maps)
        for pen_index, ink_map in enumerate(ink_maps):
            check()
            if ink_map is None or not library[pen_index].enabled:
                continue
            if not np.any(ink_map > 0.01):
                continue
            image = np.clip(1.0 - ink_map, 0.0, 1.0)
            scale = 1.0 / px_per_mm
            # Exact size with "keep proportions" off has to stretch vertically
            # too, or the Height box does nothing at all - which is what it did.
            scale_y = target_h / max(prepared.shape[0], 1) if target_h > 0 else scale
            modulation = settings.style.modulation
            style_key = (
                id(ink_map),
                settings.style.technique,
                tuple(sorted(settings.style.technique_params().items())),
                round(px_per_mm, 6),
                round(library[pen_index].width, 4),
                settings.pen.scale_with_pen_width,
            )
            groups = _memo_get(_paths_memo, style_key)
            if groups is None:
                groups = list(
                    _style_groups(image, settings, px_per_mm, library[pen_index].width, should_cancel)
                )
                _memo_put(_paths_memo, style_key, ink_map, groups)
            for dwell_ms, paths_px in groups:
                if not paths_px:
                    continue
                if modulation in ("pressure", "speed") and dwell_ms <= 0:
                    paths_px = _attach_modulation(paths_px, ink_map, max(px_per_mm * 1.5, 2.0))
                layers.append(
                    Layer(
                        pen=pen_index,
                        paths=[_scale_xy(p, scale, scale_y) for p in paths_px],
                        name=library[pen_index].name,
                        dwell_ms=dwell_ms,
                        modulation=modulation if modulation != "none" and dwell_ms <= 0 else "",
                        modulation_amount=settings.style.modulation_amount,
                    )
                )
    else:
        scale = 1.0
        if native and native[0] > 0:
            scale = target_w / native[0]
        if source.layers:
            for layer in source.layers:
                pen_index = max(0, min(int(layer.pen), len(library) - 1))
                if not library[pen_index].enabled:
                    continue
                layers.append(
                    Layer(
                        pen=pen_index,
                        paths=[p * scale for p in layer.paths],
                        name=layer.name,
                        dwell_ms=layer.dwell_ms,
                    )
                )
        if source.vector is not None and len(source.vector):
            buckets = _assign_vector_pens(source.vector, library, settings.style.separation)
            for pen_index, paths in sorted(buckets.items()):
                if not library[pen_index].enabled:
                    continue
                layers.append(
                    Layer(pen=pen_index, paths=[p * scale for p in paths], name=library[pen_index].name)
                )

    if not layers:
        job.stats.build_seconds = time.perf_counter() - started
        return job

    # ---- layout ---------------------------------------------------------
    check()
    all_paths = [p for layer in layers for p in layer.paths]
    matrix = _layout_matrix(settings, all_paths, as_drawn)
    for layer in layers:
        layer.paths = [geo.apply_matrix(matrix, p) for p in layer.paths]

    # ---- optimise -------------------------------------------------------
    opt = settings.optimize
    technique = techniques.REGISTRY.get(settings.style.technique)
    technique_stitchable = bool(technique.stitchable) if technique else False
    if source.kind != "image" and not source.is_raster:
        technique_stitchable = False
    ordered_layers: list[Layer] = []
    cursor = (settings.machine.park_x, settings.machine.park_y)
    for pen_index in range(len(library)):
        for layer in sorted(
            (l for l in layers if l.pen == pen_index), key=lambda l: l.dwell_ms
        ):
            check()
            pen_width = library[pen_index].width
            # one switch governs everything pen-relative: the technique's own
            # spacings and the tidying tolerances alike
            scale = _pen_scale(pen_width, settings.pen.scale_with_pen_width)
            paths = layer.paths
            dots = [p for p in paths if len(p) == 1]
            strokes = [p for p in paths if len(p) > 1]
            # A modulated stroke carries its tone in the interior points.  RDP
            # only looks at X and Y, so on a straight hatch line it collapses
            # the stroke back to its two ends and throws every tone sample away
            # - the pressure and speed features quietly did nothing at all.
            toned = [p for p in strokes if p.shape[1] > 2]
            if toned:
                strokes = [p for p in strokes if p.shape[1] <= 2]
            if opt.simplify > 0:
                strokes = geo.simplify_paths(strokes, opt.simplify * scale)
            strokes.extend(toned)
            if opt.join > 0:
                strokes = geo.join_paths(strokes, opt.join * scale)
            if opt.min_length > 0:
                strokes = geo.drop_short(strokes, opt.min_length * scale)
            paths = strokes + dots
            if opt.reorder:
                paths = geo.reorder_paths(paths, cursor, opt.allow_reverse)
                if opt.tidy_tour:
                    paths = geo.improve_tour(paths, cursor, time_budget=0.6)
            # Stitching is what actually makes a drawing fast: pen lifts are
            # most of the job, and parallel fill strokes almost always end a
            # millimetre from where the next one starts.
            # Stitching draws a line between two strokes.  With a blade in the
            # holder that line is a cut across the middle of the stencil, so
            # cutting tools never get stitched.
            if opt.stitch > 0 and technique_stitchable and not library[pen_index].cuts:
                paths = geo.stitch_paths(paths, opt.stitch * scale)
            paths = _expand_dots(paths, settings.pen.dot_diameter)
            # A swivel blade trails behind the carriage, so its corners have to
            # be swung out - after simplification, or they would be thrown away.
            paths = knife.prepare(paths, library[pen_index])
            if paths:
                cursor = tuple(paths[-1][-1])
                ordered_layers.append(layer.copy_with(paths))

    drawing = Drawing(layers=ordered_layers, warnings=job.warnings, source_label=source.label)
    job.drawing = drawing

    # ---- stats ----------------------------------------------------------
    stats = job.stats
    stats.path_count = drawing.path_count
    stats.pen_lifts = drawing.path_count
    stats.dot_count = sum(1 for layer in ordered_layers for p in layer.paths if len(p) <= 2 and geo.path_length(p) < 0.05)
    stats.draw_length = drawing.draw_length
    stats.bounds = drawing.bounds()
    seconds, travel = _estimate_seconds(ordered_layers, settings, library)
    stats.estimated_seconds = seconds
    stats.travel_length = travel
    for layer in ordered_layers:
        entry = stats.per_pen.setdefault(layer.pen, {"paths": 0, "length": 0.0})
        entry["paths"] += len(layer.paths)
        entry["length"] += layer.draw_length

    if stats.bounds:
        lo_x, lo_y, hi_x, hi_y = stats.bounds
        machine = settings.machine
        stats.out_of_bounds = lo_x < -0.01 or lo_y < -0.01 or hi_x > machine.bed_x + 0.01 or hi_y > machine.bed_y + 0.01
        if stats.out_of_bounds:
            job.warnings.append("The drawing sticks out past the bed - move or scale it down.")

    stats.build_seconds = time.perf_counter() - started
    return job


# --------------------------------------------------------------------------
def build_project(
    sources: dict,
    settings: AppSettings,
    library: PenLibrary,
    should_cancel=None,
) -> PlotJob:
    """Draw every visible item and merge them onto one sheet.

    `sources` maps an item index to its prepared SourceResult.  Each item is
    built with its own technique and placement - `settings.style` and friends
    already point at whichever item is selected - and the results are then
    grouped by pen so the machine still changes pens as few times as possible.
    """
    started = time.perf_counter()
    was_active = settings.active
    merged: list[Layer] = []
    warnings: list[str] = []
    native: tuple[float, float] | None = None
    target = (0.0, 0.0)
    built = 0

    try:
        for index, item in enumerate(settings.items):
            if not item.visible:
                continue
            source = sources.get(index)
            if source is None:
                continue
            settings.active = index
            job = build_plot(source, settings, library, should_cancel)
            if job.drawing.is_empty():
                warnings.extend(job.warnings)
                continue
            built += 1
            label = item.label()
            for layer in job.drawing.layers:
                copy_of = layer.copy_with(layer.paths)
                copy_of.item = index
                copy_of.name = f"{label} - {layer.name}" if layer.name else label
                merged.append(copy_of)
            warnings.extend(job.warnings)
            if native is None:
                native = job.native_size
            target = job.target_size
    finally:
        settings.active = was_active

    result = PlotJob(warnings=warnings, target_size=target, native_size=native)
    if not merged:
        result.stats.build_seconds = time.perf_counter() - started
        return result

    # one pen at a time, items in their layer order within each pen
    ordered: list[Layer] = []
    for pen_index in range(len(library)):
        ordered.extend(layer for layer in merged if layer.pen == pen_index)
    ordered.extend(layer for layer in merged if layer.pen >= len(library))

    label = settings.items[was_active].label() if settings.items else ""
    if built > 1:
        label = f"{built} layers"
    result.drawing = Drawing(layers=ordered, warnings=warnings, source_label=label)

    stats = result.stats
    stats.path_count = result.drawing.path_count
    stats.pen_lifts = result.drawing.path_count
    stats.dot_count = sum(
        1 for layer in ordered for p in layer.paths if len(p) <= 2 and geo.path_length(p) < 0.05
    )
    stats.draw_length = result.drawing.draw_length
    stats.bounds = result.drawing.bounds()
    seconds, travel = _estimate_seconds(ordered, settings, library)
    stats.estimated_seconds = seconds
    stats.travel_length = travel
    for layer in ordered:
        entry = stats.per_pen.setdefault(layer.pen, {"paths": 0, "length": 0.0})
        entry["paths"] += len(layer.paths)
        entry["length"] += layer.draw_length

    if stats.bounds:
        lo_x, lo_y, hi_x, hi_y = stats.bounds
        machine = settings.machine
        stats.out_of_bounds = (
            lo_x < -0.01 or lo_y < -0.01 or hi_x > machine.bed_x + 0.01 or hi_y > machine.bed_y + 0.01
        )
        if stats.out_of_bounds:
            result.warnings.append("Part of the drawing sticks out past the bed.")
    _warn_if_too_fine(result, ordered, library)
    if stats.path_count > 60000 or stats.estimated_seconds > 6 * 3600:
        result.warnings.append(
            f"That is {stats.path_count:,} strokes and about "
            f"{stats.estimated_seconds / 3600:.1f} hours of plotting. A wider pen or a "
            f"coarser setting will bring it down quickly."
        )
    stats.build_seconds = time.perf_counter() - started
    return result
