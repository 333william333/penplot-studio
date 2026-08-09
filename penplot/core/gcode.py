"""G-code generation for pen plotting on a Cartesian FDM printer.

Nothing is ever extruded and no heater is touched.  The pen is raised and
lowered with the Z axis, and the program carries machine-readable markers so
the app can pause the USB stream for pen changes and sharpening.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from . import geometry as geo
from .pens import PenLibrary
from .profiles import firmware_of
from .testpattern import parse_layer_tag
from .pipeline import PlotJob
from .settings import AppSettings

__all__ = ["GCodeProgram", "generate", "format_duration", "PAUSE_MARKER"]

PAUSE_MARKER = ";PENPLOT_PAUSE:"
LAYER_MARKER = ";PENPLOT_LAYER:"


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours, rest = divmod(int(round(seconds)), 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours} h {minutes} min"
    if minutes:
        return f"{minutes} min {secs} s"
    return f"{secs} s"


def _fit_circle(path, tolerance: float):
    """Recognise a closed circular path so it can go out as a single arc.

    Halftone dots, stipple dots and circle packing all emit polygons with a
    dozen or more vertices each.  As G2 they become one line of G-code that the
    firmware interpolates smoothly, which shrinks the file by an order of
    magnitude and removes a deceleration at every vertex.
    """
    if len(path) < 6:
        return None
    points = np.asarray(path)[:, :2]
    if float(np.hypot(*(points[0] - points[-1]))) > tolerance:
        return None
    ring = points[:-1]
    centre = ring.mean(axis=0)
    radii = np.hypot(ring[:, 0] - centre[0], ring[:, 1] - centre[1])
    radius = float(radii.mean())
    if radius < 0.05 or float(np.abs(radii - radius).max()) > tolerance:
        return None
    # signed area gives the direction of travel
    x, y = ring[:, 0], ring[:, 1]
    area = float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return float(centre[0]), float(centre[1]), area < 0


def _num(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text if text not in ("-0", "") else "0"


@dataclass
class GCodeProgram:
    lines: list[str] = field(default_factory=list)
    pause_at: dict[int, str] = field(default_factory=dict)
    path_at: dict[int, tuple[int, int]] = field(default_factory=dict)
    # line index -> ("down"|"up", draw Z for that pen, lift height); lets the
    # streamer rewrite pen height and hop live while the drawing is running
    z_at: dict[int, tuple[str, float, float]] = field(default_factory=dict)
    drawn_at: list[float] = field(default_factory=list)
    total_drawn: float = 0.0
    estimated_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)
    pen_changes: int = 0
    sharpen_stops: int = 0
    short_lifts: int = 0
    arcs: int = 0
    extra_passes: int = 0

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"

    def __len__(self) -> int:
        return len(self.lines)


class _Emitter:
    def __init__(self, program: GCodeProgram):
        self.program = program
        self.drawn = 0.0

    def add(self, line: str) -> int:
        self.program.lines.append(line)
        self.program.drawn_at.append(self.drawn)
        return len(self.program.lines) - 1

    def comment(self, text: str) -> None:
        self.add(f"; {text}")

    def blank(self) -> None:
        self.add("")


def generate(job: PlotJob, settings: AppSettings, library: PenLibrary) -> GCodeProgram:
    program = GCodeProgram()
    emit = _Emitter(program)

    machine = settings.machine
    pen_setup = settings.pen
    pauses = settings.pauses
    layers = job.drawing.layers

    firmware = firmware_of(machine)
    travel_feed = _num(machine.travel_feed)
    z_feed = _num(machine.z_feed)
    # how far a polygon may sit from a perfect circle and still be sent as one
    arc_tolerance = max(min(library.narrowest() * 0.35, 0.25), 0.02)

    # ---------------- header --------------------------------------------
    emit.comment(f"PenPlot Studio - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    emit.comment(f"Machine: {machine.name}  bed {_num(machine.bed_x)} x {_num(machine.bed_y)} mm")
    emit.comment(f"Source: {job.drawing.source_label or 'untitled'}")
    emit.comment(f"Drawn length: {job.stats.draw_length / 1000.0:.2f} m in {job.stats.path_count} strokes")
    emit.comment(f"Estimated time: {format_duration(job.stats.estimated_seconds)}")
    for pen_index in job.drawing.used_pens():
        pen = library[pen_index]
        offset = f" - Z offset {pen.z_offset:+.2f} mm" if abs(pen.z_offset) > 1e-6 else ""
        emit.comment(f"Pen {pen_index + 1}: {pen.name} - {pen.width:.2f} mm - {pen.color}{offset}")
    emit.comment("No filament is extruded and no heater is enabled by this file.")

    # The pen-change and park points live in the pen setup, not in the machine
    # profile, so a machine change can leave them pointing off the bed.
    for label, x, y in (
        ("The pen-change position", pen_setup.change_x, pen_setup.change_y),
        ("The parking position", machine.park_x, machine.park_y),
    ):
        if not (0.0 <= x <= machine.bed_x and 0.0 <= y <= machine.bed_y):
            program.warnings.append(
                f"{label} (X{x:.0f} Y{y:.0f}) is outside this bed ({machine.bed_x:.0f} x "
                f"{machine.bed_y:.0f} mm). The head will hit its limits."
            )
    if not pen_setup.zero_z_at_start and not (-50.0 <= pen_setup.draw_z <= machine.max_z):
        # Negative is normal: with Z never homed, the machine's zero is wherever
        # it powered up, so the paper often sits below it.  Wildly out is not.
        program.warnings.append(
            f"The stored pen height (Z{pen_setup.draw_z:.2f}) is nowhere near this machine's "
            f"Z range. Set the pen height again."
        )
    if pen_setup.change_z > machine.max_z:
        program.warnings.append(
            f"The pen-change height (Z{pen_setup.change_z:.0f}) is above this machine's "
            f"maximum Z ({machine.max_z:.0f} mm)."
        )
    emit.blank()

    if firmware.notes:
        emit.comment(firmware.notes)
    emit.add("G21 ; millimetres")
    emit.add("G90 ; absolute positioning")
    emit.add("M107 ; fan off")
    if machine.send_acceleration and firmware.acceleration:
        # make the machine match the estimate instead of whatever the last
        # print left behind
        if firmware.acceleration == "S":
            emit.add(f"M204 S{_num(machine.acceleration)} ; acceleration")
        else:
            emit.add(
                f"M204 P{_num(machine.acceleration)} T{_num(machine.travel_acceleration)}"
                " ; drawing / travel acceleration"
            )
    if machine.raise_z_limit:
        # A bare pen carriage is far lighter than a hot end, so the stock Z
        # speed ceiling can be lifted for the job.  Every pen lift is paid for
        # at this rate, and lifts are most of the drawing time.  The units of
        # M203 differ between firmwares - getting that wrong is a factor of 60.
        if firmware.z_speed_limit == "mm/s":
            emit.add(f"M203 Z{_num(machine.z_limit_target / 60.0)} ; faster pen lifts (mm/s)")
        elif firmware.z_speed_limit == "mm/min":
            emit.add(f"M203 Z{_num(machine.z_limit_target)} ; faster pen lifts (mm/min)")
        else:
            emit.comment(f"Z speed limit left alone - {firmware.label} does not take M203")
    if machine.raise_z_limit and firmware.axis_acceleration:
        # The real cost of a pen lift is acceleration, not top speed: 2.5 mm at
        # the stock 100 mm/s^2 is 0.32 s each way and never gets near the
        # feedrate.  Raising this is worth more than raising M203, and a bare
        # pen carriage weighs a fraction of the hot end the number was set for.
        emit.add(
            f"M201 Z{_num(machine.z_acceleration_target)} ; faster pen lifts (mm/s^2)"
        )
    if machine.junction_deviation > 0 and firmware.junction_deviation:
        # Hatching is thousands of corners.  Marlin ships junction deviation at
        # 0.013 mm, which brings the head almost to a stop at every one.
        emit.add(
            f"M205 J{machine.junction_deviation:.3f} ; keep speed through corners"
        )
    if machine.use_bed_mesh and firmware.bed_mesh:
        emit.add(f"{firmware.bed_mesh} ; use the stored bed mesh so the pen presses evenly")

    home_mode = machine.home_mode
    if pen_setup.zero_z_at_start and home_mode == "all":
        # Homing Z would drive the pen into the bed and throw away the G92
        # paper reference we just set - the two settings cannot both apply.
        home_mode = "xy"
        program.warnings.append(
            "Homing was limited to X and Y: a G28 would have destroyed the pen height "
            "you set with G92 and pushed the pen into the bed."
        )
        emit.comment("Z homing skipped on purpose - the pen height comes from G92 below")

    safe_z = pen_setup.draw_z + max(pen_setup.lift, 0.5) + 3.0
    if pen_setup.zero_z_at_start:
        # Touch-off mode: whatever height the pen is at right now is declared
        # to be the paper.  Only correct if the operator really has just set it
        # there, which is why the app refuses to send in this mode without a
        # confirmation - see MainWindow.send_to_printer.
        emit.comment("The pen must be touching the paper right now - this becomes Z0")
        emit.add(f"G92 Z{_num(pen_setup.draw_z)}")
        emit.add(f"G1 Z{_num(safe_z)} F{z_feed} ; lift before homing")
    else:
        # Measured mode: the drawing height is an absolute machine coordinate,
        # so nothing here may redefine the frame.  A relative hop clears the
        # paper without needing to know where we are.
        emit.comment(f"Pen touches the paper at machine Z{_num(pen_setup.draw_z)}")
        emit.add("G91 ; relative")
        emit.add(f"G1 Z{_num(max(pen_setup.lift, 0.5) + 3.0)} F{z_feed} ; lift before homing")
        emit.add("G90 ; absolute")

    if home_mode == "all":
        emit.add("G28 ; home all axes")
    elif home_mode == "xy":
        emit.add("G28 X Y ; home the carriage, leave Z alone")
    else:
        emit.comment("Homing skipped")
    # G28 is the one command that can leave the machine in a mode we did not
    # choose, and every Z number after this point is absolute.  Saying so again
    # costs one line and removes a whole class of runaway.
    emit.add("G90 ; absolute - every Z below is a machine coordinate")
    emit.blank()

    # ---------------- body ----------------------------------------------
    cursor = np.array([machine.park_x, machine.park_y], dtype=np.float64)
    current_pen: int | None = None
    since_sharpen = 0.0
    first_layer = True

    def pen_z(index: int) -> float:
        return pen_setup.draw_z + library[index].z_offset

    def travel_z(index: int) -> float:
        return pen_z(index) + max(pen_setup.lift, 0.2)

    def do_pause(message: str, resume_pen: int) -> None:
        """Park high, ask for the pen, then come back down to the new pen's travel height."""
        emit.add(f"G1 Z{_num(max(pen_setup.change_z, travel_z(resume_pen)))} F{z_feed}")
        if pauses.park_for_pause:
            emit.add(f"G0 X{_num(pen_setup.change_x)} Y{_num(pen_setup.change_y)} F{travel_feed}")
        if firmware.status_message:
            emit.add(f"M117 {message[:24]}")
        index = emit.add(f"{PAUSE_MARKER}{message}")
        if pauses.host_pause:
            program.pause_at[index] = message
        if pauses.emit_m0:
            emit.add(f"{firmware.pause_command} {message}".strip())
        emit.add(f"G1 Z{_num(travel_z(resume_pen))} F{z_feed}")

    for layer_index, layer in enumerate(layers):
        if not any(len(path) for path in layer.paths):
            continue
        pen = library[layer.pen]
        # Test patterns put a per-layer Z offset and feed in the layer name, so
        # one job can draw the same stroke at a dozen different pen heights.
        # Every normal layer parses to all-None and is unaffected.
        override = parse_layer_tag(layer.name)
        z_shift = override.z_offset or 0.0
        emit.blank()
        emit.add(f"{LAYER_MARKER}{layer_index}:{layer.pen}:{pen.name}")
        dwell_note = f" - dwell {layer.dwell_ms:.0f} ms/dot" if layer.dwell_ms > 0 else ""
        if override.label:
            dwell_note += f" - {override.label}"
        emit.comment(
            f"Pen {layer.pen + 1} - {pen.name} ({pen.width:.2f} mm) - "
            f"{layer.draw_length / 1000:.2f} m{dwell_note}"
        )

        if current_pen is not None and layer.pen != current_pen and pauses.pause_between_pens:
            message = f"{pauses.pen_change_message} {layer.pen + 1}: {pen.name}"
            do_pause(message, layer.pen)
            program.pen_changes += 1
            since_sharpen = 0.0
        elif first_layer or (current_pen is not None and layer.pen != current_pen) or z_shift:
            # pens have different lengths, so the travel height has to be reset
            # even when we do not stop - otherwise a longer pen scrapes along
            emit.add(f"G1 Z{_num(travel_z(layer.pen) + z_shift)} F{z_feed}")
            if current_pen is not None and layer.pen != current_pen:
                since_sharpen = 0.0
        current_pen = layer.pen
        first_layer = False

        lift = max(pen_setup.lift, 0.2)
        modulated = layer.modulation in ("pressure", "speed")
        base_feed = max(machine.draw_feed * max(pen.feed_scale, 0.05), 60.0)
        depth = max(layer.modulation_amount, 0.0) * 0.6   # mm of extra pen pressure
        slowdown = min(max(layer.modulation_amount, 0.0), 0.95) * 0.8
        # a dwell layer rests on each dot instead of moving; that is where its
        # tone comes from, so it replaces the usual settle delay
        layer_dwell = int(round(max(layer.dwell_ms, 0.0)))
        draw_feed = _num(
            override.draw_feed
            if override.draw_feed
            else max(machine.draw_feed * max(pen.feed_scale, 0.05), 60.0)
        )
        down_z = _num(pen_z(layer.pen) + z_shift)
        up_z = _num(travel_z(layer.pen) + z_shift)

        # A blade seldom gets through card in one go, and an ordinary pen will
        # perforate paper if it goes over the same line often enough.  Both are
        # the same instruction to the machine: run each stroke `repeats` times,
        # `pass_depth` deeper every time.
        repeats = 1 if modulated else pen.repeats
        pass_depth = max(pen.pass_depth, 0.0) if repeats > 1 else 0.0
        keep_direction = pen.swivels
        deepest = pass_depth * (repeats - 1)
        if deepest > 2.0:
            # a blade this far under the paper is into the mat, or the bed
            program.warnings.append(
                f"{pen.name} ends {deepest:.1f} mm below the paper after {repeats} passes. "
                "Check that there is a cutting mat under the sheet, or reduce the depth per pass."
            )

        interval = pen.sharpen_every if pen.sharpen_every > 0 else (
            pauses.sharpen_interval if pauses.sharpen_enabled else 0.0
        )

        def trace(run: np.ndarray, base_z: float) -> None:
            """One complete pass over `run`, pen already down."""
            first = run[0]
            arc = (
                _fit_circle(run, arc_tolerance)
                if machine.use_arcs and not modulated and len(run) >= 6
                else None
            )
            if len(run) == 1:
                # A dot is a tap.  The pen is already down on the spot, so a
                # move to where it already is is one wasted line per dot -
                # 7 339 of them on a stipple drawing, and the firmware has to
                # parse and acknowledge every one.
                return
            if arc is not None:
                cx, cy, clockwise = arc
                emit.drawn += geo.path_length(run)
                emit.add(
                    f"{'G2' if clockwise else 'G3'} X{_num(first[0])} Y{_num(first[1])} "
                    f"I{_num(cx - first[0])} J{_num(cy - first[1])} F{draw_feed}"
                )
                program.arcs += 1
                return
            if modulated and run.shape[1] > 2:
                # The machine itself carries the tone here: either the pen is
                # pressed a little deeper, or it slows down, wherever the
                # picture is dark.  Both make a visibly heavier line.
                previous = first
                for point in run[1:]:
                    if abs(point[0] - previous[0]) < 0.005 and abs(point[1] - previous[1]) < 0.005:
                        continue
                    emit.drawn += float(np.hypot(point[0] - previous[0], point[1] - previous[1]))
                    weight = float(min(max(point[2], 0.0), 1.0))
                    if layer.modulation == "pressure":
                        z = base_z - depth * weight
                        index = emit.add(
                            f"G1 X{_num(point[0])} Y{_num(point[1])} Z{_num(z)} F{draw_feed}"
                        )
                        # tag it so the live pressure slider reaches these moves
                        # too, not just the pen up/down ones
                        program.z_at[index] = ("draw", z, 0.0)
                    else:
                        feed = max(base_feed * (1.0 - slowdown * weight), 60.0)
                        emit.add(f"G1 X{_num(point[0])} Y{_num(point[1])} F{_num(feed)}")
                    previous = point
                return
            previous = first
            for point in run[1:]:
                if abs(point[0] - previous[0]) < 0.005 and abs(point[1] - previous[1]) < 0.005:
                    continue
                emit.drawn += float(np.hypot(point[0] - previous[0], point[1] - previous[1]))
                emit.add(f"G1 X{_num(point[0])} Y{_num(point[1])} F{draw_feed}")
                previous = point

        for path_index, path in enumerate(layer.paths):
            if len(path) == 0:
                continue
            # How far is the next stroke?  A short hop only needs to clear the
            # paper, and a small lift is several times quicker on a leadscrew Z.
            next_start = None
            for candidate in layer.paths[path_index + 1 :]:
                if len(candidate):
                    next_start = candidate[0]
                    break
            if interval > 0 and since_sharpen >= interval * 1000.0:
                message = f"{pauses.sharpen_message} ({pen.name})"
                do_pause(message, layer.pen)
                program.sharpen_stops += 1
                since_sharpen = 0.0

            wait = max(layer_dwell, int(pen_setup.down_delay))
            base = pen_z(layer.pen) + z_shift
            closed = len(path) > 2 and float(np.hypot(path[0][0] - path[-1][0], path[0][1] - path[-1][1])) <= 0.05
            for pass_index in range(repeats):
                # A blade has to keep going the same way round or its swivel
                # compensation points the wrong way; a pen scoring a line is
                # happy to come straight back and saves the trip home.
                backwards = bool(pass_index % 2) and not keep_direction and not closed
                run = path[::-1] if backwards else path
                z_here = base - pass_depth * pass_index
                start = run[0]
                if pass_index == 0:
                    emit.add(f"G0 X{_num(start[0])} Y{_num(start[1])} F{travel_feed}")
                    index = emit.add(f"G1 Z{_num(z_here)} F{z_feed}")
                    program.z_at[index] = ("down", z_here, lift)
                    if wait > 0:
                        emit.add(f"G4 P{wait}")
                else:
                    program.extra_passes += 1
                    gap = float(np.hypot(start[0] - previous_end[0], start[1] - previous_end[1]))
                    if gap > 0.05:
                        # lift just clear of the work, hop back, plunge again
                        emit.add(f"G1 Z{_num(z_here + max(min(pen_setup.short_lift, lift), 0.05))} F{z_feed}")
                        emit.add(f"G0 X{_num(start[0])} Y{_num(start[1])} F{travel_feed}")
                    index = emit.add(f"G1 Z{_num(z_here)} F{z_feed}")
                    program.z_at[index] = ("down", z_here, lift)
                trace(run, z_here)
                previous_end = run[-1]

            hop = (
                float(np.hypot(next_start[0] - previous_end[0], next_start[1] - previous_end[1]))
                if next_start is not None
                else float("inf")
            )
            if hop <= pen_setup.short_hop:
                this_lift = max(min(pen_setup.short_lift, lift), 0.05)
                program.short_lifts += 1
            else:
                this_lift = lift
            index = emit.add(f"G1 Z{_num(pen_z(layer.pen) + z_shift + this_lift)} F{z_feed}")
            program.z_at[index] = ("up", pen_z(layer.pen) + z_shift, this_lift)
            if pen_setup.up_delay > 0:
                emit.add(f"G4 P{int(pen_setup.up_delay)}")
            index = len(program.lines) - 1
            program.path_at[index] = (layer_index, path_index)

            length = geo.path_length(path) * repeats
            since_sharpen += length
            cursor = previous_end

    # ---------------- footer --------------------------------------------
    emit.blank()
    emit.comment("finished")
    emit.add(f"G1 Z{_num(max(pen_setup.change_z, safe_z))} F{z_feed}")
    emit.add(f"G0 X{_num(machine.park_x)} Y{_num(machine.park_y)} F{travel_feed}")
    if machine.raise_z_limit and firmware.z_speed_limit:
        restore = machine.z_max_feed / 60.0 if firmware.z_speed_limit == "mm/s" else machine.z_max_feed
        emit.add(f"M203 Z{_num(restore)} ; put the Z speed limit back")
    if machine.raise_z_limit and firmware.axis_acceleration:
        emit.add(f"M201 Z{_num(machine.z_acceleration)} ; put the Z acceleration back")
    if firmware.status_message:
        emit.add("M117 Drawing done")
    if machine.disable_motors_at_end and firmware.motors_off:
        emit.add(f"{firmware.motors_off} ; motors off")

    program.total_drawn = emit.drawn
    program.estimated_seconds = job.stats.estimated_seconds
    return program
