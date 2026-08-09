"""Write a finished plot out as SVG.

The G-code writer in `gcode.py` is the machine-facing end of the pipeline; this
is the paper-facing one.  The file it produces is meant to be opened in
Inkscape or Illustrator at *true size*, so:

* the document is measured in millimetres (`width="220mm"`, and a `viewBox`
  whose user units are millimetres too - one user unit = one millimetre);
* Y is flipped.  Bed coordinates are Y-up, the way the printer thinks; SVG is
  Y-down.  Every point is written as ``bed_y - y`` so the drawing lands on the
  page the same way round as it lands on the paper;
* one group per pen, carrying the pen colour and the real pen width, so a
  0.2 mm fineliner looks like a 0.2 mm line and not a hairline.  The groups are
  tagged as Inkscape layers, which is what the usual plotter extensions expect.

Nothing here changes the geometry - the same numbers that go into the G-code go
into the path data, only rounded to 3 decimals (a micron) so the file does not
bloat.  That means the total length measured back out of the path data matches
`job.stats.draw_length` to well under a tenth of a percent.
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np

from .pens import PenLibrary
from .pipeline import PlotJob
from .settings import MachineSettings

__all__ = ["to_svg", "save_svg"]


#: colour used for the (hidden by default) travel-move layer
TRAVEL_COLOR = "#C8D8E8"


# --------------------------------------------------------------------------
# small formatting helpers
# --------------------------------------------------------------------------
def _num(value: float) -> str:
    """3 decimals, trailing zeros stripped - a micron of resolution is plenty."""
    text = f"{float(value):.3f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-", "-0") else text


def _escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _comment_safe(text: str) -> str:
    """`--` may not appear inside an XML comment, and it may not end with `-`."""
    out = str(text).replace("--", "- -").replace("\r", " ").replace("\n", " ")
    while out.endswith("-"):
        out = out[:-1] + "_"
    return out


def _slug(text: str, fallback: str) -> str:
    keep = [c if (c.isalnum() or c in "-_") else "-" for c in str(text).strip()]
    out = "".join(keep).strip("-")
    while "--" in out:
        out = out.replace("--", "-")
    if not out or not (out[0].isalpha() or out[0] == "_"):
        out = f"{fallback}-{out}" if out else fallback
    return out


def _format_duration(seconds: float) -> str:
    """Same wording as `gcode.format_duration`, kept local so this module stays light."""
    seconds = max(0.0, float(seconds))
    hours, rest = divmod(int(round(seconds)), 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours} h {minutes} min"
    if minutes:
        return f"{minutes} min {secs} s"
    return f"{secs} s"


# --------------------------------------------------------------------------
# path data
# --------------------------------------------------------------------------
def _path_data(path: np.ndarray, bed_y: float) -> str:
    """One polyline as `M x,y L x,y ...` with Y flipped into SVG space.

    A single-point path is a stipple dot.  The G-code writes it as a G1 to the
    point it is already standing on; the SVG equivalent is a zero-length
    subpath, which with `stroke-linecap="round"` paints exactly one round dot
    of the pen's width.
    """
    points = np.asarray(path, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0:
        return ""

    xs = [_num(p[0]) for p in points]
    ys = [_num(bed_y - p[1]) for p in points]
    start = f"M {xs[0]},{ys[0]}"
    if len(points) == 1:
        return f"{start} L {xs[0]},{ys[0]}"

    parts = [start]
    previous = (xs[0], ys[0])
    for x, y in zip(xs[1:], ys[1:]):
        if (x, y) == previous:  # identical *after rounding* - nothing to draw
            continue
        parts.append(f"L {x},{y}")
        previous = (x, y)
    if len(parts) == 1:  # every point collapsed onto one spot: a dot
        return f"{start} L {xs[0]},{ys[0]}"
    return " ".join(parts)


def _travel_data(job: PlotJob, machine: MachineSettings) -> str:
    """Every pen-up hop, in plot order, as one multi-subpath `d` string."""
    bed_y = float(machine.bed_y)
    cursor = np.array([machine.park_x, machine.park_y], dtype=np.float64)
    parts: list[str] = []
    for layer in job.drawing.layers:
        for path in layer.paths:
            points = np.asarray(path, dtype=np.float64).reshape(-1, 2)
            if len(points) == 0:
                continue
            parts.append(
                f"M {_num(cursor[0])},{_num(bed_y - cursor[1])} "
                f"L {_num(points[0][0])},{_num(bed_y - points[0][1])}"
            )
            cursor = points[-1]
    parts.append(
        f"M {_num(cursor[0])},{_num(bed_y - cursor[1])} "
        f"L {_num(machine.park_x)},{_num(bed_y - machine.park_y)}"
    )
    return " ".join(parts)


# --------------------------------------------------------------------------
def to_svg(
    job: PlotJob,
    library: PenLibrary,
    machine: MachineSettings,
    *,
    layers_as_groups: bool = True,
    include_travels: bool = False,
) -> str:
    """Render *job* as an SVG document string.

    `layers_as_groups` (default) puts each pen in its own `<g>` - an Inkscape
    layer named after the pen, carrying that pen's colour and width.  Turn it
    off to get a single flat group where every `<path>` carries its own stroke,
    which some downstream tools prefer.

    `include_travels` adds the pen-up moves as an extra group.  It is written
    with `display:none` so it never prints by accident; switch the layer on in
    the editor when you want to see where the time goes.
    """
    drawing = job.drawing
    stats = job.stats
    bed_x = max(float(machine.bed_x), 1.0)
    bed_y = max(float(machine.bed_y), 1.0)

    source = drawing.source_label or "untitled"
    header = [
        "PenPlot Studio SVG export",
        f"Source: {source}",
        f"Machine: {machine.name}  bed {_num(bed_x)} x {_num(bed_y)} mm",
        f"Drawn length: {stats.draw_length / 1000.0:.2f} m in {stats.path_count} strokes",
        f"Estimated time: {_format_duration(stats.estimated_seconds)}",
        f"Written: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "1 user unit = 1 mm.  Y is flipped: SVG y = bed_y - bed y.",
    ]

    out: list[str] = ['<?xml version="1.0" encoding="UTF-8" standalone="no"?>']
    out.append("<!--")
    for line in header:
        out.append(f"  {_comment_safe(line)}")
    out.append("-->")

    namespaces = 'xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"'
    if layers_as_groups:
        namespaces += ' xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"'
    out.append(
        f'<svg {namespaces} version="1.1" '
        f'width="{_num(bed_x)}mm" height="{_num(bed_y)}mm" '
        f'viewBox="0 0 {_num(bed_x)} {_num(bed_y)}">'
    )
    out.append(f"  <title>{_escape(source)}</title>")
    out.append(f"  <desc>{_escape(' | '.join(header[1:]))}</desc>")

    if layers_as_groups:
        for pen_index in drawing.used_pens():
            pen = library[pen_index]
            paths = [p for layer in drawing.layers if layer.pen == pen_index for p in layer.paths]
            if not paths:
                continue
            label = pen.name or f"Pen {pen_index + 1}"
            out.append(
                f'  <g id="{_escape(_slug(label, f"pen{pen_index + 1}"))}" '
                f'inkscape:groupmode="layer" inkscape:label="{_escape(label)}" '
                f'fill="none" stroke="{_escape(pen.color)}" '
                f'stroke-width="{_num(max(pen.width, 0.01))}" '
                f'stroke-linecap="round" stroke-linejoin="round">'
            )
            for path in paths:
                data = _path_data(path, bed_y)
                if data:
                    out.append(f'    <path d="{data}"/>')
            out.append("  </g>")
    else:
        out.append(
            '  <g id="drawing" fill="none" stroke-linecap="round" stroke-linejoin="round">'
        )
        for layer in drawing.layers:
            pen = library[layer.pen]
            for path in layer.paths:
                data = _path_data(path, bed_y)
                if data:
                    out.append(
                        f'    <path stroke="{_escape(pen.color)}" '
                        f'stroke-width="{_num(max(pen.width, 0.01))}" d="{data}"/>'
                    )
        out.append("  </g>")

    if include_travels:
        data = _travel_data(job, machine)
        if data:
            group = (
                '  <g id="travel-moves" fill="none" '
                f'stroke="{TRAVEL_COLOR}" stroke-width="0.1" '
                'stroke-dasharray="1,1" style="display:none"'
            )
            if layers_as_groups:
                group += ' inkscape:groupmode="layer" inkscape:label="Travel moves"'
            out.append(group + ">")
            out.append(f'    <path d="{data}"/>')
            out.append("  </g>")

    out.append("</svg>")
    return "\n".join(out) + "\n"


def save_svg(path, job: PlotJob, library: PenLibrary, machine: MachineSettings, **kwargs) -> None:
    """Write `to_svg(...)` to *path* (str or Path), UTF-8, creating the folder."""
    text = to_svg(job, library, machine, **kwargs)
    folder = os.path.dirname(os.fspath(path))
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(os.fspath(path), "w", encoding="utf-8") as handle:
        handle.write(text)
