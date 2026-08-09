"""Calibration patterns: the sheets that stop you guessing.

Everything in here builds a normal `PlotJob` out of ordinary `Drawing` /
`Layer` objects in bed millimetres, so a pattern can be handed to
`gcode.generate` exactly like an imported picture.  Nothing needs a source
file, and nothing goes through `pipeline.build_plot` - the geometry is already
laid out on the bed, already optimised, already the right way up (Y-up).

    pen_test(...)       one sheet, five swatch types at five line spacings
    z_ladder(...)       find the pen height that just kisses the paper
    speed_ladder(...)   find the fastest feed the pen can still keep up with
    registration(...)   corner crosses and rulers: scale and squareness

Per-layer machine settings
--------------------------
The Z ladder and the speed ladder are only useful if each row is actually
drawn with a different Z / feed.  `Layer` already carries one such per-layer
machine value - `dwell_ms`, which `gcode.generate` reads straight off the
layer - but there is no field for Z or feed and this module does not add one.

So the value travels in `Layer.name`, which no other stage reads (the G-code
writer labels its layers from `library[layer.pen].name`, not from
`layer.name`).  The encoding is one short tag:

    penplot:z=-0.30;feed=1200;label=Z -0.30

`parse_layer_tag(layer.name)` turns that back into a `LayerOverride`.  What the
G-code side has to do with it, precisely - all of it inside the per-layer block
of `gcode.generate`, right where `down_z`, `up_z` and `draw_feed` are worked
out today:

* `z_offset` - millimetres, *relative* to the Z that layer would otherwise be
  drawn at.  With `shift = override.z_offset or 0.0`:

      down_z = _num(pen_z(layer.pen) + shift)
      up_z   = _num(travel_z(layer.pen) + shift)

  Shifting both keeps the lift the same height above the paper, so only the
  drawing height changes.  The `program.z_at[index] = ("down"|"up", ...)`
  entries must record `pen_z(layer.pen) + shift` as well - they are what the
  live streamer rewrites when the user nudges the pen height mid-plot, and
  without the shift that nudge would flatten the ladder as it plots.  If a
  pause happens inside a shifted layer, `do_pause` should come back down to
  `travel_z(resume_pen) + shift` for the same reason.
* `draw_feed` - millimetres per minute, *absolute*, replacing
  `machine.draw_feed * max(pen.feed_scale, 0.05)` for that layer only:

      feed = override.draw_feed if override.draw_feed is not None else <today's value>
      draw_feed = _num(max(feed, 60.0))

  Travel and Z feeds are untouched.
* `label` - free text.  Worth appending to the `;` comment already emitted
  above each layer, next to the dwell note, so the G-code reads back sensibly.

Both fields are `None` when the layer does not ask for an override, which is
every layer produced anywhere else in the app, so the change is inert for
normal drawings.  Layers that carry text labels deliberately have *no*
override: a caption plotted at +0.6 mm would be invisible, which is exactly
what the ladder is trying to find out.

One caveat worth repeating: do not run these jobs back through
`pipeline.build_plot`.  It rebuilds every `Layer` with `name=library[...].name`
and the tags would be dropped.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from . import geometry as geo
from . import strokefont
from .drawing import Drawing, Layer
from .pens import PenLibrary
from .pipeline import PlotJob, PlotStats
from .settings import MachineSettings, PenSetup

__all__ = [
    "pen_test",
    "z_ladder",
    "speed_ladder",
    "registration",
    "LayerOverride",
    "make_layer_tag",
    "parse_layer_tag",
    "TAG_PREFIX",
]


#: marks a `Layer.name` that carries per-layer machine settings
TAG_PREFIX = "penplot:"


@dataclass(frozen=True)
class LayerOverride:
    """Per-layer machine settings decoded from `Layer.name`."""

    z_offset: float | None = None   # mm, added to this layer's drawing Z
    draw_feed: float | None = None  # mm/min, replaces the layer's drawing feed
    label: str = ""

    @property
    def is_empty(self) -> bool:
        return self.z_offset is None and self.draw_feed is None


def make_layer_tag(
    *, z_offset: float | None = None, draw_feed: float | None = None, label: str = ""
) -> str:
    """Build the `Layer.name` tag understood by `parse_layer_tag`."""
    parts: list[str] = []
    if z_offset is not None:
        parts.append(f"z={z_offset:.3f}")
    if draw_feed is not None:
        parts.append(f"feed={draw_feed:.1f}")
    if label:
        parts.append(f"label={str(label).replace(';', ',').replace('=', '-')}")
    return TAG_PREFIX + ";".join(parts)


def parse_layer_tag(name: str) -> LayerOverride:
    """Decode a tagged `Layer.name`.  Anything else gives an empty override."""
    text = str(name or "")
    if not text.startswith(TAG_PREFIX):
        return LayerOverride()
    z_offset: float | None = None
    draw_feed: float | None = None
    label = ""
    for chunk in text[len(TAG_PREFIX) :].split(";"):
        key, _, value = chunk.partition("=")
        key = key.strip()
        try:
            if key == "z":
                z_offset = float(value)
            elif key == "feed":
                draw_feed = float(value)
            elif key == "label":
                label = value
        except ValueError:
            continue
    return LayerOverride(z_offset=z_offset, draw_feed=draw_feed, label=label)


# --------------------------------------------------------------------------
# text
# --------------------------------------------------------------------------
def _text_width(text: str, size: float, tracking: float = 0.0) -> float:
    scale = size / strokefont.CAP
    width = strokefont.text_width(text) * scale
    return width + max(len(text) - 1, 0) * tracking


def _text(
    text: str,
    x: float,
    y: float,
    size: float,
    *,
    anchor: str = "left",
    tracking: float = 0.0,
) -> list[np.ndarray]:
    """Single-stroke text, baseline at *y*, cap height *size*, Y-up bed mm."""
    if not text or size <= 0.0:
        return []
    scale = size / strokefont.CAP
    width = _text_width(text, size, tracking)
    if anchor == "center":
        x -= width / 2.0
    elif anchor == "right":
        x -= width

    out: list[np.ndarray] = []
    cursor = 0.0
    for ch in text:
        advance, strokes = strokefont.glyph(ch)
        for stroke in strokes:
            moved = np.asarray(stroke, dtype=np.float64) * scale
            moved[:, 0] += x + cursor
            moved[:, 1] += y
            out.append(moved)
        cursor += advance * scale + tracking
    return out


# --------------------------------------------------------------------------
# geometry bits the swatches are made of
# --------------------------------------------------------------------------
def _rect(x0: float, y0: float, w: float, h: float) -> np.ndarray:
    return np.asarray(
        [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h), (x0, y0)], dtype=np.float64
    )


def _serpentine(x0: float, y0: float, w: float, h: float, spacing: float, vertical: bool = False) -> np.ndarray:
    """A solid fill as one continuous boustrophedon stroke - no pen lifts."""
    span = h if not vertical else w
    run = w if not vertical else h
    steps = max(int(round(span / max(spacing, 1e-3))), 1)
    points: list[tuple[float, float]] = []
    for k in range(steps + 1):
        offset = min(k * spacing, span)
        near, far = (0.0, run) if k % 2 == 0 else (run, 0.0)
        if vertical:
            points.append((x0 + offset, y0 + near))
            points.append((x0 + offset, y0 + far))
        else:
            points.append((x0 + near, y0 + offset))
            points.append((x0 + far, y0 + offset))
    return np.asarray(points, dtype=np.float64)


def _diagonal_diamond(cx: float, cy: float, size: float, spacing: float) -> np.ndarray:
    """45-degree hatching, as a serpentine over the square rotated into the cell.

    Filling a square with diagonal lines means clipping every line; filling a
    *rotated* square with straight lines means rotating one serpentine.  Same
    picture, one stroke, no clipping.
    """
    side = size / math.sqrt(2.0)
    path = _serpentine(-side / 2.0, -side / 2.0, side, side, spacing)
    angle = math.radians(45.0)
    ca, sa = math.cos(angle), math.sin(angle)
    out = np.empty_like(path)
    out[:, 0] = path[:, 0] * ca - path[:, 1] * sa + cx
    out[:, 1] = path[:, 0] * sa + path[:, 1] * ca + cy
    return out


def _dot_grid(cx: float, cy: float, size: float, pitch: float) -> list[np.ndarray]:
    """Single-point paths - the same shape a stipple layer produces."""
    count = max(int(size / max(pitch, 1e-3)), 1)
    count = min(count, 60)
    span = count * pitch
    x0 = cx - span / 2.0
    y0 = cy - span / 2.0
    return [
        np.asarray([[x0 + i * pitch, y0 + j * pitch]], dtype=np.float64)
        for j in range(count + 1)
        for i in range(count + 1)
    ]


def _spiral(cx: float, cy: float, r_max: float, pitch: float) -> np.ndarray:
    turns = max(r_max / max(pitch, 1e-3), 1.0)
    total = 2.0 * math.pi * turns
    steps = int(max(96.0, total / 0.15))
    t = np.linspace(0.0, total, steps)
    r = pitch * t / (2.0 * math.pi)
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], axis=1)


def _zigzag(x0: float, y0: float, w: float, amplitude: float, teeth: int) -> np.ndarray:
    xs = np.linspace(x0, x0 + w, teeth * 2 + 1)
    ys = np.array([y0 + (amplitude if i % 2 else 0.0) for i in range(len(xs))])
    return np.stack([xs, ys], axis=1)


def _cross(x: float, y: float, arm: float) -> list[np.ndarray]:
    return [
        np.asarray([(x - arm, y), (x + arm, y)], dtype=np.float64),
        np.asarray([(x, y - arm), (x, y + arm)], dtype=np.float64),
    ]


# --------------------------------------------------------------------------
# stats
# --------------------------------------------------------------------------
def _estimate(
    layers: list[Layer], machine: MachineSettings, pen_setup: PenSetup, library: PenLibrary
) -> tuple[float, float]:
    """Seconds and travel millimetres, using the same model as `pipeline`.

    Written out again here rather than borrowed because these jobs change the
    drawing feed per layer, which the pipeline estimator has no way to know
    about - and a speed ladder whose estimate ignores the speeds would be a
    strange thing to hand a user.
    """
    accel = max(machine.acceleration, 50.0)

    def move_time(distance: float, feed_mm_min: float) -> float:
        if distance <= 1e-9:
            return 0.0
        v = max(feed_mm_min, 1.0) / 60.0
        if distance < v * v / accel:
            return 2.0 * math.sqrt(distance / accel)
        return distance / v + v / accel

    travel_total = 0.0
    seconds = 0.0
    cursor = np.array([machine.park_x, machine.park_y], dtype=np.float64)
    z_travel = move_time(max(pen_setup.lift, 0.1), machine.z_feed) * 2.0
    delays = (pen_setup.down_delay + pen_setup.up_delay) / 1000.0

    for layer in layers:
        override = parse_layer_tag(layer.name)
        base = machine.draw_feed * max(library[layer.pen].feed_scale, 0.05)
        feed = override.draw_feed if override.draw_feed is not None else base
        dwell = max(layer.dwell_ms, 0.0) / 1000.0
        for path in layer.paths:
            if len(path) == 0:
                continue
            hop = float(np.hypot(path[0][0] - cursor[0], path[0][1] - cursor[1]))
            travel_total += hop
            seconds += move_time(hop, machine.travel_feed) + z_travel + delays + dwell
            seconds += move_time(geo.path_length(path), feed)
            cursor = path[-1]

    park = np.array([machine.park_x, machine.park_y], dtype=np.float64)
    hop = float(np.hypot(park[0] - cursor[0], park[1] - cursor[1]))
    travel_total += hop
    seconds += move_time(hop, machine.travel_feed)
    return seconds, travel_total


def _build_job(
    label: str,
    layers: list[Layer],
    machine: MachineSettings,
    pen_setup: PenSetup,
    library: PenLibrary,
    warnings: list[str] | None = None,
    started: float | None = None,
) -> PlotJob:
    """Wrap finished layers in a `PlotJob` with realistic statistics."""
    started = started if started is not None else time.perf_counter()
    layers = [l for l in layers if any(len(p) for p in l.paths)]
    warnings = list(warnings or [])

    drawing = Drawing(layers=layers, warnings=warnings, source_label=label)
    job = PlotJob(drawing=drawing, warnings=warnings)

    stats: PlotStats = job.stats
    stats.path_count = drawing.path_count
    stats.dot_count = sum(
        1 for layer in layers for p in layer.paths if len(p) <= 2 and geo.path_length(p) < 0.05
    )
    stats.draw_length = drawing.draw_length
    stats.bounds = drawing.bounds()
    seconds, travel = _estimate(layers, machine, pen_setup, library)
    stats.estimated_seconds = seconds
    stats.travel_length = travel
    for layer in layers:
        entry = stats.per_pen.setdefault(layer.pen, {"paths": 0, "length": 0.0})
        entry["paths"] += len(layer.paths)
        entry["length"] += layer.draw_length

    if stats.bounds:
        lo_x, lo_y, hi_x, hi_y = stats.bounds
        stats.out_of_bounds = (
            lo_x < -0.01 or lo_y < -0.01 or hi_x > machine.bed_x + 0.01 or hi_y > machine.bed_y + 0.01
        )
        if stats.out_of_bounds:
            message = "The test pattern does not fit the bed - reduce the margin or the number of rows."
            warnings.append(message)
            job.warnings = warnings
            drawing.warnings = warnings
        job.target_size = (hi_x - lo_x, hi_y - lo_y)
        job.native_size = job.target_size

    stats.build_seconds = time.perf_counter() - started
    return job


def _pen_index(library: PenLibrary, pen: int) -> int:
    return max(0, min(int(pen), max(len(library) - 1, 0)))


def _usable(machine: MachineSettings, margin: float) -> tuple[float, float]:
    return (
        max(machine.bed_x - 2.0 * margin, 20.0),
        max(machine.bed_y - 2.0 * margin, 20.0),
    )


def _fit_factor(need_w: float, need_h: float, usable_w: float, usable_h: float) -> float:
    """How much a nominal layout has to shrink to sit inside the usable bed."""
    return min(1.0, usable_w / max(need_w, 1e-6), usable_h / max(need_h, 1e-6))


def _round_down(value: float, choices: tuple[float, ...]) -> float:
    fits = [c for c in choices if c <= value]
    return max(fits) if fits else 0.0


def _fit_text(text: str, size: float, budget: float, floor: float = 1.8) -> float:
    """Cap height that makes *text* fit *budget* mm, or 0 when that is unreadable.

    Returning 0 lets the caller drop a caption rather than run it off the sheet
    or print it at a size nobody can read - `_text` draws nothing at size 0.
    """
    width = _text_width(text, size)
    if width <= budget or width <= 0.0:
        return size
    shrunk = size * budget / width
    return shrunk if shrunk >= floor else 0.0


# --------------------------------------------------------------------------
# 1. pen test sheet
# --------------------------------------------------------------------------
_SWATCHES = ("fill", "cross", "dots", "spiral", "diag 45")


def pen_test(
    machine: MachineSettings,
    pen_setup: PenSetup,
    library: PenLibrary,
    *,
    pen: int = 0,
    spacings: tuple[float, ...] | None = None,
    cell: float = 18.0,
    gap: float = 4.0,
    margin: float = 10.0,
    label_size: float = 3.2,
    title: str | None = None,
) -> PlotJob:
    """A one-sheet character study of a pen.

    Five swatches - solid fill, crosshatch, dot grid, spiral, 45-degree hatch -
    repeated at five line spacings, with the spacing printed beside every row.
    Read down a column to see where the pen stops filling solid and starts
    showing stripes; read across a row to see whether it is the spacing or the
    direction that the pen dislikes.

    `spacings` defaults to 0.5x, 0.8x, 1.2x, 2.0x and 3.0x the pen's nominal
    width, which is the range where the answer always turns out to be.
    """
    started = time.perf_counter()
    index = _pen_index(library, pen)
    nib = library[index]

    if spacings is None:
        width = max(nib.width, 0.05)
        spacings = tuple(round(width * m, 2) for m in (0.5, 0.8, 1.2, 2.0, 3.0))
    values = sorted({max(float(s), 0.08) for s in spacings})
    rows = len(values)
    columns = len(_SWATCHES)

    label_column = 15.0
    usable_w, usable_h = _usable(machine, margin)
    warnings: list[str] = []

    # Everything below is linear in (cell, gap, label_column, label_size), so a
    # single scale factor is enough to make any bed fit exactly.
    def measure():
        block = label_column + columns * cell + (columns - 1) * gap
        grid = rows * cell + (rows - 1) * gap
        head = label_size * 0.55                              # column headings
        sub = head + label_size * (0.85 + 0.8)                # subtitle above them
        top = sub + label_size * (1.35 + 0.8)                 # title above that
        # a few glyphs reach past the cap height, so budget the ascender
        crown = label_size * 1.35 * strokefont.ASCENDER / strokefont.CAP
        return block, grid, head, sub, top, top + crown

    block_w, grid_h, head_base, sub_base, title_base, top_band = measure()
    fit = _fit_factor(block_w, grid_h + top_band, usable_w, usable_h)
    if fit < 1.0:
        cell *= fit
        gap *= fit
        label_column *= fit
        label_size *= fit
        block_w, grid_h, head_base, sub_base, title_base, top_band = measure()
        if label_size < 2.0:
            warnings.append(
                "The pen test sheet had to shrink to fit the bed and its captions are now very "
                "small - use fewer spacings."
            )

    x0 = (machine.bed_x - block_w) / 2.0
    y0 = (machine.bed_y - (grid_h + top_band)) / 2.0
    grid_top = y0 + grid_h

    marks: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    heading = title or f"PEN TEST - {nib.name} {nib.width:.2f} mm"
    subtitle = f"line spacing in mm - feed {machine.draw_feed:.0f} mm/min - Z {pen_setup.draw_z:+.2f} mm"
    labels += _text(
        heading,
        machine.bed_x / 2.0,
        grid_top + title_base,
        _fit_text(heading, label_size * 1.35, block_w),
        anchor="center",
    )
    labels += _text(
        subtitle,
        machine.bed_x / 2.0,
        grid_top + sub_base,
        _fit_text(subtitle, label_size * 0.85, block_w),
        anchor="center",
    )
    labels += _text("mm", x0 + label_column - 3.0, grid_top + head_base, label_size * 0.85, anchor="right")

    for j, name in enumerate(_SWATCHES):
        cx = x0 + label_column + j * (cell + gap) + cell / 2.0
        labels += _text(name, cx, grid_top + head_base, label_size * 0.85, anchor="center")

    for i, spacing in enumerate(values):
        cell_y = grid_top - (i + 1) * cell - i * gap
        labels += _text(
            f"{spacing:.2f}",
            x0 + label_column - 3.0,
            cell_y + cell / 2.0 - label_size * 0.35,
            label_size,
            anchor="right",
        )
        for j, kind in enumerate(_SWATCHES):
            cell_x = x0 + label_column + j * (cell + gap)
            cx = cell_x + cell / 2.0
            cy = cell_y + cell / 2.0
            marks.append(_rect(cell_x, cell_y, cell, cell))
            inset = cell * 0.033
            if kind == "fill":
                marks.append(_serpentine(cell_x + inset, cell_y + inset, cell - 2 * inset, cell - 2 * inset, spacing))
            elif kind == "cross":
                marks.append(_serpentine(cell_x + inset, cell_y + inset, cell - 2 * inset, cell - 2 * inset, spacing))
                marks.append(
                    _serpentine(cell_x + inset, cell_y + inset, cell - 2 * inset, cell - 2 * inset, spacing, vertical=True)
                )
            elif kind == "dots":
                # a dot costs a pen lift, so the patch is capped at 12 mm - the
                # difference between spacings is obvious long before the cell is full
                marks.extend(_dot_grid(cx, cy, min(cell - 2 * inset, 12.0), spacing * 3.0))
            elif kind == "spiral":
                marks.append(_spiral(cx, cy, cell / 2.0 - inset, spacing))
            else:
                marks.append(_diagonal_diamond(cx, cy, cell - 2 * inset, spacing))

    layers = [
        Layer(pen=index, paths=labels, name="", dwell_ms=0.0),
        Layer(pen=index, paths=marks, name="", dwell_ms=0.0),
    ]
    return _build_job("Pen test sheet", layers, machine, pen_setup, library, warnings, started)


# --------------------------------------------------------------------------
# 2. Z ladder
# --------------------------------------------------------------------------
def z_ladder(
    machine: MachineSettings,
    pen_setup: PenSetup,
    library: PenLibrary,
    *,
    pen: int = 0,
    z_min: float = -0.6,
    z_max: float = 0.6,
    step: float = 0.1,
    strokes: int = 3,
    stroke_pitch: float = 1.2,
    stroke_length: float | None = None,
    margin: float = 10.0,
    label_size: float = 3.2,
    title: str | None = None,
) -> PlotJob:
    """A row of strokes per pen height, labelled with the offset it was drawn at.

    Draw it once, look at which row is crisp without embossing the paper, add
    that number to the drawing Z, and never think about pen height again.

    Each row is a `Layer` tagged `penplot:z=<offset>` - see the module
    docstring for what the G-code writer has to do with that.  The captions are
    in their own untagged layer so they are always drawn at the working height:
    a caption plotted at +0.6 mm would not be there to read.
    """
    started = time.perf_counter()
    index = _pen_index(library, pen)
    step = max(abs(float(step)), 0.01)
    lo, hi = (z_min, z_max) if z_min <= z_max else (z_max, z_min)
    count = int(round((hi - lo) / step)) + 1
    offsets = [round(lo + k * step, 4) for k in range(max(count, 1))]

    label_column = 18.0
    usable_w, usable_h = _usable(machine, margin)
    warnings: list[str] = []

    # row pitch and the title band are both proportional, so one factor fits
    def measure():
        pitch = strokes * stroke_pitch + label_size * 1.7
        head = label_size * (1.35 + 0.8 + 0.85 + 1.4)
        return pitch, head, len(offsets) * pitch + head

    row_pitch, title_h, total_h = measure()
    fit = _fit_factor(1.0, total_h, usable_w, usable_h)
    if fit < 1.0:
        stroke_pitch *= fit
        label_size *= fit
        row_pitch, title_h, total_h = measure()
        if label_size < 2.0:
            warnings.append(
                f"{len(offsets)} rows only fit the bed at a caption size that is hard to read - "
                "use a coarser step or a narrower Z range."
            )

    # width: the strokes take whatever is left after the caption column and the
    # room reserved for the "current" marker; on a narrow bed those shrink too
    reserve = 22.0
    if usable_w - label_column - reserve < 20.0:
        narrow = usable_w / max(label_column + 20.0 + reserve, 1e-6)
        label_column *= narrow
        reserve *= narrow
    length = usable_w - label_column - reserve
    if stroke_length is not None:
        length = min(length, stroke_length)
    length = max(min(length, 150.0), 5.0)
    block_w = label_column + length + reserve
    x0 = (machine.bed_x - block_w) / 2.0
    top = (machine.bed_y + total_h) / 2.0

    labels: list[np.ndarray] = []
    heading = title or "Z LADDER"
    subtitle = f"offset from drawing Z {pen_setup.draw_z:+.2f} mm - pick the crispest row"
    labels += _text(
        heading,
        machine.bed_x / 2.0,
        top - label_size * 1.35,
        _fit_text(heading, label_size * 1.35, block_w),
        anchor="center",
    )
    labels += _text(
        subtitle,
        machine.bed_x / 2.0,
        top - label_size * (1.35 + 0.8 + 0.85),
        _fit_text(subtitle, label_size * 0.85, block_w),
        anchor="center",
    )

    rows: list[Layer] = []
    for i, offset in enumerate(offsets):
        row_top = top - title_h - i * row_pitch
        block_bottom = row_top - strokes * stroke_pitch
        middle = block_bottom + (strokes - 1) * stroke_pitch / 2.0
        labels += _text(
            f"{offset:+.2f}",
            x0 + label_column - 3.0,
            middle - label_size * 0.35,
            label_size,
            anchor="right",
        )
        if abs(offset) < 1e-9:
            labels += _text(
                "current",
                x0 + label_column + length + 3.0,
                middle - label_size * 0.3,
                _fit_text("current", label_size * 0.8, reserve - 4.0),
            )
        # alternate direction: the pen ends each stroke where the next one
        # starts, which saves a 150 mm travel per stroke over the whole ladder
        paths = []
        for k in range(strokes):
            y = block_bottom + k * stroke_pitch
            ends = [(x0 + label_column, y), (x0 + label_column + length, y)]
            if k % 2:
                ends.reverse()
            paths.append(np.asarray(ends, dtype=np.float64))
        rows.append(
            Layer(
                pen=index,
                paths=paths,
                name=make_layer_tag(z_offset=offset, label=f"Z {offset:+.2f} mm"),
                dwell_ms=0.0,
            )
        )

    layers = [Layer(pen=index, paths=labels, name="", dwell_ms=0.0)] + rows
    return _build_job("Z height ladder", layers, machine, pen_setup, library, warnings, started)


# --------------------------------------------------------------------------
# 3. speed ladder
# --------------------------------------------------------------------------
def speed_ladder(
    machine: MachineSettings,
    pen_setup: PenSetup,
    library: PenLibrary,
    *,
    pen: int = 0,
    feeds: tuple[float, ...] = (600.0, 900.0, 1200.0, 1800.0, 2400.0, 3000.0, 4000.0, 6000.0),
    margin: float = 10.0,
    label_size: float = 3.2,
    title: str | None = None,
) -> PlotJob:
    """The same idea as `z_ladder`, for the drawing feed rate.

    Every row is a straight line, a zigzag and a circle drawn at one feed.  The
    straight line shows ink starvation, the zigzag shows what the pen does when
    the machine reverses under it, and the circle shows backlash.  A pen that
    skips at 3000 mm/min is often perfect at 1200.

    Each row is a `Layer` tagged `penplot:feed=<mm/min>`; the captions carry no
    tag and are drawn at the machine's normal feed.
    """
    started = time.perf_counter()
    index = _pen_index(library, pen)
    values = sorted({max(float(f), 60.0) for f in feeds})

    label_column = 20.0
    usable_w, usable_h = _usable(machine, margin)
    warnings: list[str] = []

    def measure():
        pitch = label_size * 3.75
        head = label_size * (1.35 + 0.8 + 0.85 + 1.9)
        return pitch, head, len(values) * pitch + head

    row_pitch, title_h, total_h = measure()
    fit = _fit_factor(1.0, total_h, usable_w, usable_h)
    if fit < 1.0:
        label_size *= fit
        row_pitch, title_h, total_h = measure()
        if label_size < 2.0:
            warnings.append(
                f"{len(values)} feed rows only fit the bed at a caption size that is hard to "
                "read - drop a few feeds."
            )
    amplitude = row_pitch * 0.25
    radius = row_pitch * 0.33

    # width: a caption column, a straight line, a zigzag and a circle.  The
    # line absorbs spare width; when there is none, everything shrinks together.
    zig_w = 34.0
    inner_gap = 6.0
    circle_w = 2.0 * radius + 4.0
    nominal_w = label_column + 90.0 + inner_gap + zig_w + inner_gap + circle_w
    narrow = min(1.0, usable_w / nominal_w)
    if narrow < 1.0:
        label_column *= narrow
        zig_w *= narrow
        inner_gap *= narrow
        radius *= narrow
        circle_w *= narrow
    line_w = max(usable_w * narrow - label_column - zig_w - circle_w - 2.0 * inner_gap, 5.0)
    line_w = min(line_w, 90.0)
    block_w = label_column + line_w + inner_gap + zig_w + inner_gap + circle_w
    x0 = (machine.bed_x - block_w) / 2.0
    top = (machine.bed_y + total_h) / 2.0

    labels: list[np.ndarray] = []
    heading = title or "SPEED LADDER"
    subtitle = "drawing feed in mm/min - pick the fastest row with no gaps"
    labels += _text(
        heading,
        machine.bed_x / 2.0,
        top - label_size * 1.35,
        _fit_text(heading, label_size * 1.35, block_w),
        anchor="center",
    )
    labels += _text(
        subtitle,
        machine.bed_x / 2.0,
        top - label_size * (1.35 + 0.8 + 0.85),
        _fit_text(subtitle, label_size * 0.85, block_w),
        anchor="center",
    )

    rows: list[Layer] = []
    for i, feed in enumerate(values):
        baseline = top - title_h - (i + 1) * row_pitch + row_pitch * 0.33
        labels += _text(
            f"{feed:.0f}",
            x0 + label_column - 3.0,
            baseline - label_size * 0.35,
            label_size,
            anchor="right",
        )
        line_x = x0 + label_column
        zig_x = line_x + line_w + inner_gap
        circle_x = zig_x + zig_w + inner_gap + radius
        paths = [
            np.asarray([(line_x, baseline), (line_x + line_w, baseline)], dtype=np.float64),
            _zigzag(zig_x, baseline - amplitude / 2.0, zig_w, amplitude, 6),
            geo.circle_path(circle_x, baseline, radius, 48),
        ]
        if i % 2:  # every other row runs right to left, so no long hop back
            paths = [p[::-1] for p in reversed(paths)]
        rows.append(
            Layer(
                pen=index,
                paths=paths,
                name=make_layer_tag(draw_feed=feed, label=f"{feed:.0f} mm/min"),
                dwell_ms=0.0,
            )
        )

    labels += _text(
        "mm/min", x0 + label_column - 3.0, top - title_h + label_size * 0.6, label_size * 0.8, anchor="right"
    )

    layers = [Layer(pen=index, paths=labels, name="", dwell_ms=0.0)] + rows
    return _build_job("Feed rate ladder", layers, machine, pen_setup, library, warnings, started)


# --------------------------------------------------------------------------
# 4. registration / scale
# --------------------------------------------------------------------------
def registration(
    machine: MachineSettings,
    pen_setup: PenSetup | None = None,
    library: PenLibrary | None = None,
    *,
    pen: int = 0,
    margin: float = 10.0,
    tick_label_step: float = 20.0,
    label_size: float = 3.0,
    title: str | None = None,
) -> PlotJob:
    """Corner crosses, a frame, two millimetre rulers and two reference shapes.

    Lay a real ruler along the drawn one: if 100 printed millimetres are not
    100 real millimetres the machine's steps per millimetre are wrong.  Measure
    the frame's two diagonals against each other for squareness, the 20 mm
    square with a caliper for scale in both axes at once, and look at the
    40 mm circle for backlash.
    """
    started = time.perf_counter()
    pen_setup = pen_setup or PenSetup()
    library = library or PenLibrary()
    index = _pen_index(library, pen)

    fx0, fy0 = margin, margin
    fx1, fy1 = machine.bed_x - margin, machine.bed_y - margin
    width, height = fx1 - fx0, fy1 - fy0
    cx, cy = machine.bed_x / 2.0, machine.bed_y / 2.0

    warnings: list[str] = []
    if width < 60.0 or height < 60.0:
        warnings.append("The bed minus the margin is too small for a useful registration sheet.")

    marks: list[np.ndarray] = [_rect(fx0, fy0, width, height)]
    labels: list[np.ndarray] = []

    # ---- corner crosses -------------------------------------------------
    # They sit on round bed coordinates *inside* the frame, so the whole sheet
    # stays within the margin and the distance between two crosses is a number
    # you can put a ruler on without doing arithmetic first.
    arm = min(8.0, max(margin * 0.8, 3.0))
    gx0 = math.ceil((fx0 + arm) / 10.0) * 10.0
    gx1 = math.floor((fx1 - arm) / 10.0) * 10.0
    gy0 = math.ceil((fy0 + arm) / 10.0) * 10.0
    gy1 = math.floor((fy1 - arm) / 10.0) * 10.0
    if gx1 <= gx0 or gy1 <= gy0:  # a very small bed: fall back to the frame inset
        gx0, gx1, gy0, gy1 = fx0 + arm, fx1 - arm, fy0 + arm, fy1 - arm

    # ---- title, kept clear of everything the rulers reach ---------------
    heading = title or "REGISTRATION"
    subtitle = f"crosses {gx1 - gx0:.0f} x {gy1 - gy0:.0f} mm apart, frame {width:.0f} x {height:.0f} mm"
    title_base = fy1 - 7.0
    sub_base = title_base - label_size * 1.4 - 2.2
    # half the frame width keeps the captions clear of the corner labels, which
    # start about a fifth of the way in from each edge
    labels += _text(heading, cx, title_base, _fit_text(heading, label_size * 1.4, width * 0.5), anchor="center")
    labels += _text(subtitle, cx, sub_base, _fit_text(subtitle, label_size * 0.85, width * 0.5), anchor="center")

    # The coordinate captions sit *beside* their cross on the cross's own line,
    # not diagonally inward: that keeps them out of the quadrants where the
    # reference shapes live, however small the bed is.
    for x, y, anchor in ((gx0, gy0, "left"), (gx1, gy0, "right"), (gx0, gy1, "left"), (gx1, gy1, "right")):
        marks += _cross(x, y, arm)
        beside = x + (arm + 2.0) * (1.0 if anchor == "left" else -1.0)
        labels += _text(
            f"{x:.0f}, {y:.0f}", beside, y - label_size * 0.3, label_size * 0.85, anchor=anchor
        )

    # ---- rulers ---------------------------------------------------------
    def ruler(horizontal: bool, start: float, end: float, line: float) -> None:
        """One comb: baseline and ticks in a single stroke.

        Drawn as separate segments this would be 200 pen lifts and several
        minutes of hopping.  Running up each tick and back down means the pen
        never leaves the paper - the connecting moves *are* the ruler line.
        """
        first, last = math.ceil(start), math.floor(end)
        if last <= first:
            return
        points: list[tuple[float, float]] = []
        step = int(max(tick_label_step, 5.0))
        for value in range(first, last + 1):
            if value % 10 == 0:
                size = 4.0
            elif value % 5 == 0:
                size = 2.5
            else:
                size = 1.4
            if horizontal:
                points += [(value, line), (value, line + size), (value, line)]
            else:
                points += [(line, value), (line + size, value), (line, value)]

            if value % step or value in (first, last):
                continue
            # the two rulers cross in the middle; keep their captions apart
            if abs(value - (cx if horizontal else cy)) < 14.0:
                continue
            if horizontal:
                labels.extend(_text(f"{value}", value, line - 5.2, label_size * 0.8, anchor="center"))
            else:
                labels.extend(_text(f"{value}", line + 5.5, value - label_size * 0.3, label_size * 0.8))
        marks.append(np.asarray(points, dtype=np.float64))

    ruler(True, fx0, fx1, cy)
    # the vertical ruler stops below the title instead of running through it
    ruler(False, fy0, min(fy1, sub_base - 3.0), cx)

    # ---- two shapes to put a caliper on ---------------------------------
    # Both live in the middle of a lower quadrant, at a round size chosen for
    # the bed.  On a bed too small to hold them clear of the rulers they are
    # simply left out - the frame, crosses and rulers are the sheet.
    quadrant = min(cx - fx0, cy - fy0)
    if quadrant >= 45.0:
        square = _round_down(quadrant * 0.35, (5.0, 10.0, 20.0, 25.0, 50.0))
        sx = fx0 + (cx - fx0 - square) / 2.0
        sy = fy0 + (cy - fy0 - square) / 2.0
        marks.append(_rect(sx, sy, square, square))
        labels += _text(f"{square:.0f} mm square", sx, sy + square + 2.5, label_size * 0.8)

        diameter = _round_down(quadrant * 0.45, (10.0, 20.0, 40.0, 50.0, 100.0))
        ccx = cx + (fx1 - cx) / 2.0
        ccy = fy0 + (cy - fy0) / 2.0
        marks.append(geo.circle_path(ccx, ccy, diameter / 2.0, 96))
        labels += _text(
            f"{diameter:.0f} mm circle", ccx, ccy - diameter / 2.0 - 4.5, label_size * 0.8, anchor="center"
        )

    layers = [
        Layer(pen=index, paths=marks, name="", dwell_ms=0.0),
        Layer(pen=index, paths=labels, name="", dwell_ms=0.0),
    ]
    return _build_job("Registration sheet", layers, machine, pen_setup, library, warnings, started)
