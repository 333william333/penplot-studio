"""Headless smoke tests for the whole non-UI pipeline."""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QGuiApplication  # noqa: E402

_app = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])

from penplot.core import gcode, geometry as geo, pipeline, raster, strokefont, styles, techniques, textsource  # noqa: E402
from penplot.core.drawing import Drawing, Layer, SourceResult  # noqa: E402
from penplot.core.pens import Pen, PenLibrary  # noqa: E402
from penplot.core.settings import AppSettings  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILURES.append(name)


def make_test_image(size: int = 480) -> np.ndarray:
    """A colourful test card: gradient, circle, square and thin lines."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    img = np.ones((size, size, 3), dtype=np.float32)
    img[:, :, 0] = 1.0 - (x / size) * 0.9
    img[:, :, 1] = 1.0 - (y / size) * 0.7
    img[:, :, 2] = 0.9
    circle = ((x - size * 0.35) ** 2 + (y - size * 0.4) ** 2) < (size * 0.22) ** 2
    img[circle] = (0.85, 0.1, 0.15)
    square = (abs(x - size * 0.7) < size * 0.15) & (abs(y - size * 0.7) < size * 0.15)
    img[square] = (0.1, 0.25, 0.75)
    for offset in range(0, size, 37):
        img[offset : offset + 2, :, :] *= 0.35
    return np.clip(img, 0, 1)


def test_geometry() -> None:
    print("\ngeometry")
    line = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)
    check("path_length", abs(geo.path_length(line) - 3.0) < 1e-9)
    simplified = geo.rdp(line, 0.01)
    check("rdp collapses collinear points", len(simplified) == 2, f"got {len(simplified)}")

    zig = np.array([[0, 0], [1, 1], [2, 0], [3, 1]], dtype=float)
    check("rdp keeps corners", len(geo.rdp(zig, 0.1)) == 4)

    a = np.array([[0, 0], [10, 0]], dtype=float)
    b = np.array([[10.1, 0], [20, 0]], dtype=float)
    joined = geo.join_paths([a, b], 0.5)
    check(
        "join_paths merges touching ends",
        len(joined) == 1 and len(joined[0]) == 3 and abs(geo.path_length(joined[0]) - 20.0) < 0.2,
        f"got {len(joined)} paths / {len(joined[0])} points",
    )

    far = geo.join_paths([a, np.array([[50, 50], [60, 60]], dtype=float)], 0.5)
    check("join_paths leaves distant paths alone", len(far) == 2)

    scattered = [
        np.array([[100, 100], [101, 100]], dtype=float),
        np.array([[1, 1], [2, 1]], dtype=float),
        np.array([[50, 50], [51, 50]], dtype=float),
    ]
    before = geo.travel_length(scattered, (0, 0))
    after = geo.travel_length(geo.reorder_paths(scattered, (0, 0)), (0, 0))
    check("reorder_paths shortens travel", after < before, f"{after:.1f} vs {before:.1f}")

    matrix = geo.affine(scale_x=2.0, scale_y=2.0, translate=(5.0, 0.0))
    moved = geo.apply_matrix(matrix, np.array([[1.0, 1.0]]))
    check("affine scale+translate", np.allclose(moved, [[7.0, 2.0]]), str(moved))


def test_strokefont() -> None:
    print("\nsingle-stroke font")
    missing = [ch for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,:;!?-+()[]/ åäöÅÄÖ" if not strokefont.has_glyph(ch)]
    check("all expected glyphs present", not missing, f"missing {missing}")
    for ch, (advance, strokes) in strokefont.GLYPHS.items():
        if advance <= 0 and ch != " ":
            check(f"advance for {ch!r}", False)
        for stroke in strokes:
            if stroke.ndim != 2 or stroke.shape[1] != 2 or len(stroke) < 2:
                check(f"stroke shape for {ch!r}", False, str(stroke.shape))
                break
            if not np.isfinite(stroke).all():
                check(f"finite points for {ch!r}", False)
                break
    check("glyph geometry sane", True)
    width = strokefont.text_width("Hello")
    check("text_width positive", width > 0, str(width))


def test_text_source() -> None:
    print("\ntext source")
    settings = textsource.TextSettings(text="Hej Ender 3!\nÅÄÖ 123", size_mm=10.0, align="center")
    layers = textsource.build_text(settings, pen_count=1)
    check("stroke font produces paths", layers and len(layers[0].paths) > 20, f"{len(layers)} layers")
    bounds = geo.bounds([p for layer in layers for p in layer.paths])
    height = bounds[3] - bounds[1]
    check("two lines are about 26 mm tall", 20 < height < 32, f"{height:.1f} mm")

    families = textsource.available_families()
    check("font list includes system fonts", len(families) > 5, f"{len(families)}")
    outline_name = next((f for f in families if f in ("Helvetica", "Arial", "Times New Roman")), None)
    if outline_name:
        outline = textsource.TextSettings(text="AB", font=outline_name, size_mm=20.0, fill=True, fill_spacing=0.5)
        layers = textsource.build_text(outline, pen_count=1)
        check("outline font + fill produces paths", layers and len(layers[0].paths) > 10, f"{layers and len(layers[0].paths)}")

    multi = textsource.TextSettings(text="AB\nCD", pen_mode="line")
    layers = textsource.build_text(multi, pen_count=3)
    check("per-line pen assignment", len({layer.pen for layer in layers}) == 2)


def test_styles() -> None:
    print("\nstyles")
    rgb = make_test_image()
    prepared = raster.prepare(rgb, detail=400)
    gray = raster.to_gray(prepared)
    checks = {
        "edges": lambda: styles.edges_paths(gray, low=60, high=150, min_length_px=2),
        "contour": lambda: styles.contour_paths(gray, method="otsu"),
        "hatch": lambda: styles.hatch_paths(gray, levels=3, spacing=5),
        "stipple": lambda: styles.stipple_paths(gray, spacing=6),
        "spiral": lambda: styles.spiral_paths(gray, pitch=8, amplitude=3),
        "wave": lambda: styles.wave_paths(gray, spacing=8, amplitude=3),
    }
    for name, function in checks.items():
        started = time.perf_counter()
        paths = function()
        elapsed = time.perf_counter() - started
        finite = all(np.isfinite(p).all() for p in paths)
        check(f"{name} produces finite paths ({len(paths)} in {elapsed*1000:.0f} ms)", bool(paths) and finite)

    square = [np.array([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]], dtype=float)]
    fill = styles.hatch_polygons(square, 1.0, 0.0)
    check("hatch_polygons fills a square", 8 <= len(fill) <= 12, f"{len(fill)}")
    total = sum(geo.path_length(p) for p in fill)
    check("hatch length is about 10 mm per line", abs(total - len(fill) * 10) < 1.0, f"{total:.1f}")

    ring = [
        np.array([[0, 0], [20, 0], [20, 20], [0, 20], [0, 0]], dtype=float),
        np.array([[5, 5], [15, 5], [15, 15], [5, 15], [5, 5]], dtype=float),
    ]
    donut = styles.hatch_polygons(ring, 2.0, 0.0)
    row = [p for p in donut if abs(p[0][1] - 9.0) < 0.6]
    gap_is_empty = all(not (min(p[0][0], p[1][0]) < 8.0 < max(p[0][0], p[1][0])) for p in row)
    check(
        "even-odd rule leaves the hole empty",
        len(row) == 2 and gap_is_empty,
        f"{len(row)} segments across the hole row",
    )


def test_pipeline_and_gcode() -> None:
    print("\npipeline + g-code")
    rgb = make_test_image()
    source = SourceResult(kind="image", rgb=rgb, label="testcard")

    settings = AppSettings()
    settings.style.detail = 400
    library = PenLibrary()

    from penplot.core import techniques

    for style in techniques.REGISTRY:
        settings.style.technique = style
        job = pipeline.build_plot(source, settings, library)
        bounds = job.stats.bounds
        inside = bounds and bounds[0] >= -0.01 and bounds[1] >= -0.01 and bounds[2] <= settings.machine.bed_x and bounds[3] <= settings.machine.bed_y
        check(
            f"{style}: {job.stats.path_count} paths, {job.stats.draw_length/1000:.1f} m, "
            f"{gcode.format_duration(job.stats.estimated_seconds)}, built in {job.stats.build_seconds*1000:.0f} ms",
            job.stats.path_count > 0 and bool(inside) and not job.stats.out_of_bounds and job.stats.build_seconds < 8.0,
        )

    settings.style.technique = "crosshatch"
    library.apply_palette("CMYK (4 pens)")
    settings.style.separation = "cmyk"
    job = pipeline.build_plot(source, settings, library)
    check("cmyk separation uses several pens", len(job.drawing.used_pens()) >= 3, str(job.drawing.used_pens()))

    settings.style.separation = "palette"
    job = pipeline.build_plot(source, settings, library)
    check("palette separation uses several pens", len(job.drawing.used_pens()) >= 2, str(job.drawing.used_pens()))

    # wider pens must produce fewer, more widely spaced lines
    settings.style.separation = "mono"
    library = PenLibrary([Pen(name="thin", width=0.3)])
    thin = pipeline.build_plot(source, settings, library)
    library = PenLibrary([Pen(name="thick", width=1.2)])
    thick = pipeline.build_plot(source, settings, library)
    check(
        f"pen width changes hatch density ({thin.stats.draw_length:.0f} mm vs {thick.stats.draw_length:.0f} mm)",
        thin.stats.draw_length > thick.stats.draw_length * 1.5,
    )

    # g-code
    library = PenLibrary()
    library.apply_palette("Primary colours (5 pens)")
    settings.style.separation = "palette"
    settings.pauses.sharpen_enabled = True
    settings.pauses.sharpen_interval = 2.0
    job = pipeline.build_plot(source, settings, library)
    program = gcode.generate(job, settings, library)
    text = program.text()
    check("g-code has a header", text.startswith("; PenPlot Studio"))
    extruding = [
        line
        for line in program.lines
        if line.startswith(("G0", "G1"))
        and any(token[:1] == "E" and token[1:].replace("-", "").replace(".", "").isdigit() for token in line.split())
    ]
    heaters = [line for line in program.lines if line.startswith(("M104", "M109", "M140", "M190", "M303"))]
    check("no extrusion anywhere", not extruding and not heaters, f"{extruding[:2]} {heaters[:2]}")
    check("homes X and Y only", "G28 X Y" in text)
    check("pen changes emitted", program.pen_changes >= 1, str(program.pen_changes))
    check("sharpening stops emitted", program.sharpen_stops >= 1, str(program.sharpen_stops))
    check("pause markers match pause_at", len(program.pause_at) == program.pen_changes + program.sharpen_stops)
    check("drawn length tracked per line", len(program.drawn_at) == len(program.lines))
    check("ends with motors off", program.lines[-1].startswith("M84"))

    xs, ys = [], []
    for line in program.lines:
        if line.startswith(("G0", "G1")):
            for token in line.split():
                if token.startswith("X"):
                    xs.append(float(token[1:]))
                elif token.startswith("Y"):
                    ys.append(float(token[1:]))
    check(
        f"all moves inside the bed (X {min(xs):.1f}..{max(xs):.1f}, Y {min(ys):.1f}..{max(ys):.1f})",
        min(xs) >= -0.01 and max(xs) <= settings.machine.bed_x and min(ys) >= -0.01 and max(ys) <= settings.machine.bed_y,
    )


def test_tools() -> None:
    print("\ntest patterns + svg")
    import xml.etree.ElementTree as ET

    from penplot.core import svgexport, testpattern

    settings = AppSettings()
    machine, pen, library = settings.machine, settings.pen, settings.library

    patterns = {
        "pen height ladder": testpattern.z_ladder(machine, pen, library),
        "speed ladder": testpattern.speed_ladder(machine, pen, library),
        "pen test sheet": testpattern.pen_test(machine, pen, library),
        "registration": testpattern.registration(machine, pen, library),
    }
    for name, job in patterns.items():
        bounds = job.stats.bounds
        inside = bounds and bounds[0] >= -0.01 and bounds[1] >= -0.01 and \
            bounds[2] <= machine.bed_x + 0.01 and bounds[3] <= machine.bed_y + 0.01
        check(f"{name} fits the bed ({job.stats.path_count} paths, {job.stats.draw_length/1000:.1f} m)",
              bool(inside) and job.stats.path_count > 0)

    program = gcode.generate(patterns["pen height ladder"], settings, library)
    heights = sorted({round(tag[1], 2) for tag in program.z_at.values() if tag[0] == "down"})
    check(f"the ladder really draws at {len(heights)} different heights", len(heights) >= 10, str(heights[:4]))

    program = gcode.generate(patterns["speed ladder"], settings, library)
    feeds = sorted({float(line.rsplit("F", 1)[1]) for line in program.lines
                    if line.startswith("G1 X") and " F" in line})
    check(f"the speed ladder really uses {len(feeds)} feed rates", len(feeds) >= 6, str(feeds[:4]))

    rgb = make_test_image()
    settings.style.detail = 300
    job = pipeline.build_plot(SourceResult(kind="image", rgb=rgb, label="card"), settings, library)
    svg = svgexport.to_svg(job, library, machine)
    root = ET.fromstring(svg)
    check("svg parses", root.tag.endswith("svg"))
    check("svg is in millimetres at bed size",
          root.get("width") == f"{machine.bed_x:g}mm" and root.get("viewBox", "").endswith(f"{machine.bed_y:g}"),
          f"{root.get('width')} / {root.get('viewBox')}")

    # measure the exported geometry back and compare with the job
    total = 0.0
    for element in root.iter():
        if not element.tag.endswith("path"):
            continue
        if "display:none" in (element.get("style") or ""):
            continue
        for chunk in (element.get("d") or "").split("M")[1:]:
            numbers = [float(v) for v in chunk.replace("L", " ").replace(",", " ").split()]
            points = np.array(numbers).reshape(-1, 2)
            if len(points) > 1:
                total += float(np.hypot(*np.diff(points, axis=0).T).sum())
    error = abs(total - job.stats.draw_length) / max(job.stats.draw_length, 1e-6)
    check(f"svg geometry matches the job ({total:.1f} vs {job.stats.draw_length:.1f} mm)", error < 0.001)


def _castor(pivot_path, offset: float, step: float = 0.01):
    """Where the tip of a trailing blade really goes, integrated finely.

    The tip is dragged like a castor wheel: it ends up `offset` behind the
    pivot, on the line from where the tip was to where the pivot now is.  This
    is the physical model the compensation has to satisfy, and it is the only
    honest way to check it - comparing against the carriage path itself would
    just be restating the code.
    """
    points = [pivot_path[0]]
    for index in range(1, len(pivot_path)):
        a, b = pivot_path[index - 1], pivot_path[index]
        steps = int(float(np.hypot(*(b - a))) / step) + 1
        points.extend(a + (b - a) * (k / steps) for k in range(1, steps + 1))
    tip = points[0] - np.array([offset, 0.0])
    trace = []
    for point in points:
        delta = point - tip
        length = float(np.hypot(*delta))
        if length > 1e-12:
            tip = point - delta / length * offset
        trace.append(tip.copy())
    return np.asarray(trace)


def test_cutting() -> None:
    print("\ncutting and multi-pass")
    from penplot.core import knife
    from penplot.core.pens import Pen, PenLibrary

    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]], dtype=float)

    def off_square(point) -> float:
        best = 1e9
        for i in range(4):
            a, b = square[i], square[i + 1]
            ab = b - a
            t = float(np.clip(np.dot(point - a, ab) / np.dot(ab, ab), 0.0, 1.0))
            best = min(best, float(np.hypot(*(point - (a + t * ab)))))
        return best

    carriage = knife.compensate(square, 0.25)
    on_arc = True
    for cx, cy in ((10, 0), (10, 10), (0, 10), (0, 0)):
        near = carriage[1:][np.hypot(carriage[1:, 0] - cx, carriage[1:, 1] - cy) < 0.26]
        if len(near) < 3 or float(np.abs(np.hypot(near[:, 0] - cx, near[:, 1] - cy) - 0.25).max()) > 1e-6:
            on_arc = False
    check("every corner is swung around on an offset-radius arc", on_arc)

    error = max(off_square(p) for p in _castor(carriage, 0.25)[40:])
    plain = max(off_square(p) for p in _castor(square, 0.25)[40:])
    check(f"a trailing blade cuts the square ({error * 1000:.1f} um off the line)", error < 0.01)
    check(f"and without compensation it would not ({plain * 1000:.0f} um)", plain > 0.05)

    circle = np.array([[5 * math.cos(t), 5 * math.sin(t)] for t in np.linspace(0, 2 * math.pi, 60)])
    check("a smooth curve is left alone", len(knife.compensate(circle, 0.25)) <= len(circle) + 2)

    over = knife.overcut_closed(square, 3.0)
    extra = sum(float(np.hypot(*(over[i + 1] - over[i]))) for i in range(len(square) - 1, len(over) - 1))
    check(f"overcut adds exactly the length asked for ({extra:.3f} mm)", abs(extra - 3.0) < 1e-6)
    check("an open cut is not overcut", len(knife.overcut_closed(square[:3], 3.0)) == 3)

    settings = AppSettings()
    settings.pen.zero_z_at_start = False
    library = PenLibrary(pens=[Pen(name="Score", kind="scoring", passes=4, pass_depth=0.1)])
    triangle = np.array([[20, 20], [60, 20], [40, 50], [20, 20]], dtype=float)
    line = np.array([[20, 80], [80, 80]], dtype=float)
    job = pipeline.PlotJob()
    job.drawing = Drawing(layers=[Layer(pen=0, paths=[triangle, line], name="Score")])
    program = gcode.generate(job, settings, library)
    depths = sorted({round(float(l.split("Z")[1].split()[0]), 3) for l in program.lines if l.startswith("G1 Z")})
    depths = [z for z in depths if z <= settings.pen.draw_z + 1e-6]
    check(f"four depths a tenth apart {depths}", len(depths) == 4 and abs(depths[-1] - depths[0] - 0.3) < 1e-6)
    check("the extra passes are counted", program.extra_passes == 6)
    # two strokes, so two approaches and one trip home - the repeats add nothing
    check("repeats cost no repositioning", program.text().count("G0 X") == 3)
    single = PenLibrary(pens=[Pen(name="Pen")])
    plain_job = pipeline.PlotJob()
    plain_job.drawing = Drawing(layers=[Layer(pen=0, paths=[triangle, line], name="Pen")])
    plain_program = gcode.generate(plain_job, settings, single)
    check(
        f"four passes draw four times the line ({program.total_drawn:.0f} vs {plain_program.total_drawn:.0f} mm)",
        abs(program.total_drawn - 4 * plain_program.total_drawn) < 0.5,
    )


def test_stencils() -> None:
    print("\nstencils")
    from penplot.core import stencil

    rgb = make_test_image(360)
    settings = stencil.StencilSettings(levels=3, bridge_width=2.0, bridge_spacing=20.0, frame=6.0)
    sheets = stencil.build_stencils(rgb, settings, px_per_mm=360 / 120.0)
    check(f"one sheet per level ({len(sheets)})", len(sheets) == 3)
    # tone is how much ink the sheet lays down, so it climbs as the sheets darken
    check("sheets are sprayed light to dark", all(sheets[i].tone < sheets[i + 1].tone for i in range(len(sheets) - 1)))
    check("every sheet has something to cut", all(sheet.cuts for sheet in sheets))
    check("every sheet holds together", all(sheet.safe for sheet in sheets))
    check("bridges were added", sum(sheet.bridges + sheet.tabs for sheet in sheets) > 0)
    check("each sheet carries the same registration outline", all(sheet.outline is not None for sheet in sheets))

    # what actually matters: no cut ever crosses another cut on the same sheet
    # and nothing falls out
    composite = stencil.render_composite(sheets)
    check("the stack still looks like the picture", composite.min() < 0.4 and composite.max() > 0.9)


def test_looks_and_dots() -> None:
    print("\nlooks and dots")
    from penplot.core import autotune, looks, raster

    logo = raster.load_rgb(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples", "logo.png"))
    card = make_test_image(420)
    check("line art is recognised", looks.choose(autotune.analyse(logo)) == "drawing")
    check("a picture is not called line art", looks.choose(autotune.analyse(card)) == "photo")

    settings = AppSettings()
    stats = autotune.analyse(card)
    for key in looks.LOOKS:
        style = AppSettings().style
        applied = looks.apply_look(key, style, stats)
        check(f"{key} sets a real technique ({style.technique})", style.technique in techniques.REGISTRY)
        check(f"{key} explains itself", len(looks.describe(key, stats)) > 20)
        if applied.enhance:
            check(f"{key} names a known enhancement", applied.enhance in ("subject", "subject-light"))

    # The subject map has to find a sharp thing in the middle of a soft frame -
    # that is the whole job.  A flat test card has detail everywhere and would
    # pass by accident, so build the case the map exists for.
    size = 300
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    frame = 0.55 + 0.08 * np.sin(xx / 40.0)          # soft, out-of-focus surround
    detail = (((xx // 4 + yy // 4) % 2) * 0.5 + 0.25)  # a crisp checker in the middle
    inside = (np.abs(xx - size / 2) < size * 0.18) & (np.abs(yy - size * 0.46) < size * 0.18)
    subject = np.where(inside, detail, frame).astype(np.float32)
    weight = raster.subject_weight(subject)
    middle = float(weight[inside].mean())
    corners = float(np.concatenate([
        weight[: size // 8, : size // 8].ravel(), weight[: size // 8, -size // 8 :].ravel()
    ]).mean())
    check(
        f"the subject map finds the sharp middle ({middle:.2f} vs {corners:.2f} in the corners)",
        middle > corners + 0.25,
    )

    # a dot is a tap: down, up, no move in between
    settings.pen.zero_z_at_start = False
    dot = np.array([[40.0, 60.0]])
    job = pipeline.PlotJob()
    job.drawing = Drawing(layers=[Layer(pen=0, paths=[dot, np.array([[80.0, 90.0]])], name="Dots")])
    program = gcode.generate(job, settings, settings.library)
    moves = [l for l in program.lines if l.startswith(("G0 X", "G1 X", "G1 Z"))]
    first = moves[moves.index("G0 X40 Y60 F6000"):][:3]
    check(f"a dot taps without drawing {first}", not any(l.startswith("G1 X") for l in first))
    check("both dots are emitted", sum(1 for l in program.lines if l.startswith("G0 X")) >= 2)


def test_ai_engine() -> None:
    print("\nai engine")
    from penplot.core import ai, autotune, looks

    logo = raster.load_rgb(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples", "logo.png"))
    reading = ai.read(logo)
    check(f"the engine answers ({reading.backend})", reading.backend in ("classical", "neural"))
    check("and says what it found", len(reading.summary()) > 10)
    check("a logo is not a person", not reading.knows_it_is_a_person)
    check("the classical engine never claims to know", ai.read(logo, use_neural=False).knows_it_is_a_person is False)

    stats = autotune.analyse(logo)
    check("a logo is still line art", looks.choose(stats, reading) == "drawing")

    if ai.have_model():
        # with a detector installed, a picture with no face must not be called a
        # portrait, and the nudge must stay quiet rather than second-guess it
        card = make_test_image(420)
        card_reading = ai.read(card)
        check(f"no face in the test card ({len(card_reading.faces)} found)", not card_reading.faces)
        check("and no portrait is suggested over a real answer",
              looks.suggestion(autotune.analyse(card), card_reading) == "")
    else:
        print("  --   no model installed, neural path not exercised")

    weight = ai.read(logo).weight
    check("the reading carries a subject map", weight is not None and weight.ndim == 2)


def test_dots_and_fit() -> None:
    print("\ndots and fitting")
    from penplot.core import fit, techniques

    card = make_test_image(420)
    gray = raster.to_gray(raster.prepare(card, detail=420, auto_levels=False))
    ctx = techniques.Context(px_per_mm=max(gray.shape) / 150.0, pen_width=0.5)

    # dots: the whole point is that density follows tone
    dots = techniques.render("dots", gray, None, ctx)
    check(f"dot shading emits taps ({len(dots)} dots)", dots and all(len(p) == 1 for p in dots))
    ink = 1.0 - gray
    points = np.array([p[0] for p in dots])
    band = np.clip((ink * 3).astype(int), 0, 2)
    area = np.array([(band == b).sum() for b in range(3)], dtype=float)
    hit = band[points[:, 1].astype(int), points[:, 0].astype(int)]
    density = np.array([(hit == b).sum() for b in range(3)]) / np.maximum(area, 1)
    check(
        f"dots crowd into the dark ({density[0]*1e4:.0f} -> {density[2]*1e4:.0f} per 10k px)",
        density[2] > density[0] * 1.5,
    )
    # Tightening the dark end only adds dots in the dark, which is the point of
    # having two ends: count them where the change is supposed to land.
    tighter = techniques.render("dots", gray, {"dark_spacing": 0.3}, ctx)
    def dark_dots(paths):
        pts = np.array([p[0] for p in paths])
        return int((band[pts[:, 1].astype(int), pts[:, 0].astype(int)] == 2).sum())
    check(
        f"closer dark spacing packs the shadows ({dark_dots(dots)} -> {dark_dots(tighter)} dots)",
        # halving the dark spacing adds about a quarter more dots to the
        # shadows on this card - the guard is against it doing nothing
        dark_dots(tighter) > dark_dots(dots) * 1.15,
    )
    opened = techniques.render("dots", gray, {"light_spacing": 8.0}, ctx)
    check(f"opening the light end thins the picture ({len(dots)} -> {len(opened)})", len(opened) < len(dots) * 0.8)

    # hilbert's depth slider must actually do something (it once did not)
    coarse = techniques.render("hilbert", gray, {"depth": 4}, ctx)
    fine = techniques.render("hilbert", gray, {"depth": 8}, ctx)
    check(
        f"hilbert detail changes the curve ({sum(len(p) for p in coarse)} -> {sum(len(p) for p in fine)} points)",
        sum(len(p) for p in fine) > sum(len(p) for p in coarse) * 1.5,
    )

    # the fitter must never return something it scores as worse than the start
    base = techniques.render("crosshatch", gray, None, ctx)
    target = 1.0 - gray
    before, _ = fit.score_drawing(fit._rasterise(base, gray.shape, ctx.pen_px), target)
    result = fit.fit_technique(gray, "crosshatch", {}, ctx, budget_seconds=4.0)
    after_paths = techniques.render("crosshatch", gray, dict(result.params), ctx)
    after, _ = fit.score_drawing(fit._rasterise(after_paths, gray.shape, ctx.pen_px), target)
    check(f"fitting does not make the tone worse ({before:.4f} -> {after:.4f})", after <= before + 0.005)
    check(f"and it tried more than one setting ({result.tried})", result.tried > 1)
    check("every knob it returns is a real parameter of the technique",
          set(result.params) <= {p.key for p in techniques.REGISTRY["crosshatch"].params})


def test_too_fine_warning() -> None:
    print("\nfine detail versus the nib")
    from penplot.core.drawing import Drawing, Layer
    from penplot.core.pens import Pen, PenLibrary

    # a page of small lettering: lots of strokes about a millimetre across
    rng = np.random.default_rng(4)
    paths = []
    for _ in range(300):
        x, y = rng.uniform(20, 180), rng.uniform(20, 180)
        paths.append(np.array([[x, y], [x + 0.9, y + 0.6]], dtype=float))
    settings = AppSettings()
    job = pipeline.PlotJob()
    job.drawing = Drawing(layers=[Layer(pen=0, paths=paths, name="text")])
    fat = PenLibrary(pens=[Pen(width=0.5)])
    pipeline._warn_if_too_fine(job, job.drawing.layers, fat)
    check(f"a fat nib on fine work is called out ({len(job.warnings)} warning)", len(job.warnings) == 1)
    if job.warnings:
        check("and it says what to do about it",
              "bigger" in job.warnings[0] or "pen under" in job.warnings[0])

    quiet = pipeline.PlotJob()
    pipeline._warn_if_too_fine(quiet, job.drawing.layers, PenLibrary(pens=[Pen(width=0.1)]))
    check("a fine nib on the same work says nothing", not quiet.warnings)

    big = pipeline.PlotJob()
    wide = [p * 8.0 for p in paths]
    big.drawing = Drawing(layers=[Layer(pen=0, paths=wide, name="text")])
    pipeline._warn_if_too_fine(big, big.drawing.layers, fat)
    check("drawn bigger, the same work is fine", not big.warnings)


def test_ender3_tuning() -> None:
    print("\nender 3 tuning")
    from penplot.core import profiles
    from penplot.core.drawing import Drawing, Layer

    settings = AppSettings()
    profiles.apply_profile(settings.machine, profiles.find_profile("Ender 3 / Pro / V2"), settings.pen)
    settings.pen.zero_z_at_start = False
    machine = settings.machine
    check(f"the Z ceiling is raised for a pen ({machine.z_limit_target:.0f} mm/min)", machine.raise_z_limit)
    check(
        f"and the Z acceleration with it ({machine.z_acceleration:.0f} -> {machine.z_acceleration_target:.0f})",
        machine.z_acceleration_target > machine.z_acceleration,
    )
    # a pen lift never reaches its feedrate, so acceleration is what it costs
    strokes = [np.array([[20.0 + i, 20.0], [20.0 + i, 60.0]]) for i in range(0, 120, 2)]
    job = pipeline.PlotJob()
    job.drawing = Drawing(layers=[Layer(pen=0, paths=strokes, name="hatch")])
    fast, _ = pipeline._estimate_seconds(job.drawing.layers, settings, settings.library)
    machine.raise_z_limit = False
    slow, _ = pipeline._estimate_seconds(job.drawing.layers, settings, settings.library)
    check(f"raising them shortens the estimate ({slow/60:.1f} -> {fast/60:.1f} min)", fast < slow * 0.9)

    machine.raise_z_limit = True
    program = gcode.generate(job, settings, settings.library)
    text = program.text()
    check("M203 raises the Z speed", "M203 Z12" in text)
    check("M201 raises the Z acceleration", f"M201 Z{machine.z_acceleration_target:.0f}" in text)
    check("junction deviation is set for the corners", "M205 J" in text)
    # whatever we change, we hand the machine back as we found it
    check("the Z speed is put back", text.rstrip().count("M203 Z") == 2)
    check("the Z acceleration is put back", f"M201 Z{machine.z_acceleration:.0f}" in text)

    # Klipper has neither M201 nor M203.  Check the commands, not the text: the
    # header explains in a comment why it left them alone, and that comment
    # naturally contains the word M203.
    settings.machine.firmware = "klipper"
    klipper = gcode.generate(job, settings, settings.library)
    commands = [l.split(";")[0].strip() for l in klipper.lines if l and not l.startswith(";")]
    check("Klipper is not sent M201", not any(c.startswith("M201") for c in commands))
    check("Klipper is not sent M203", not any(c.startswith("M203") for c in commands))
    check("but it is told why", "does not take M203" in klipper.text())


def test_settings_roundtrip() -> None:
    print("\nsettings")
    settings = AppSettings()
    settings.machine.bed_x = 235.0
    settings.style.technique = "spiral"
    settings.style.technique_params()["pitch"] = 2.4
    settings.library.apply_palette("CMYK (4 pens)")
    settings.text.text = "Åäö"
    restored = AppSettings.from_dict(settings.to_dict())
    check("machine round-trip", restored.machine.bed_x == 235.0)
    check("style round-trip", restored.style.technique == "spiral")
    check("pens round-trip", len(restored.library) == 4 and restored.library[0].name == "Yellow")
    check("technique params round-trip", restored.style.params.get("spiral", {}).get("pitch") == 2.4)
    check("text round-trip", restored.text.text == "Åäö")
    check("unknown keys ignored", AppSettings.from_dict({"nope": 1}).machine.bed_x == 220.0)


def main() -> int:
    test_geometry()
    test_strokefont()
    test_text_source()
    test_styles()
    test_pipeline_and_gcode()
    test_tools()
    test_cutting()
    test_stencils()
    test_looks_and_dots()
    test_ai_engine()
    test_dots_and_fit()
    test_too_fine_warning()
    test_ender3_tuning()
    test_settings_roundtrip()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failing checks: {FAILURES}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
