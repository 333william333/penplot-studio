"""Making a cut follow the line you actually drew.

Two tools end up in the holder and they behave differently.

A **swivel blade** (a drag knife) is not held in a fixed direction: the tip
hangs a fraction of a millimetre *behind* the pivot and is dragged round like a
castor wheel.  Cut a square with the carriage following the square exactly and
you get rounded corners, because the blade needs those first few tenths of a
millimetre after the corner to swing into the new direction - and it cuts while
it swings.  The fix is to move the carriage, not the tip: run past the corner
by exactly the blade offset, swing round the corner point on a small arc, and
only then set off down the next side.  The tip stays parked on the corner for
the whole arc, so the corner comes out sharp.

A **fixed blade** (a scalpel) cannot turn at all, so it only scores.  It gets
no compensation - just repeated passes, which is what :mod:`gcode` does.

`overcut` is the other half of the job: on a closed shape the very start of the
cut is made while the blade is still swinging into line, so the cut runs a
little past the start point to go over that stretch again.
"""

from __future__ import annotations

import math

import numpy as np

from .pens import Pen

__all__ = ["compensate", "overcut_closed", "prepare", "prepare_layer"]

#: Corners gentler than this are left alone - the blade follows a curve on its
#: own, and inserting an arc at every vertex of a traced photograph would turn
#: a few thousand points into a few hundred thousand.
CORNER_DEGREES = 20.0


def _unit(vector: np.ndarray) -> np.ndarray | None:
    length = float(np.hypot(vector[0], vector[1]))
    if length < 1e-9:
        return None
    return vector / length


def _is_closed(points: np.ndarray, tolerance: float = 0.05) -> bool:
    return len(points) > 2 and float(np.hypot(*(points[0, :2] - points[-1, :2]))) <= tolerance


def _swing(centre: np.ndarray, start_dir: np.ndarray, end_dir: np.ndarray, radius: float) -> list[np.ndarray]:
    """Points along the arc the carriage takes while the tip sits on `centre`."""
    a0 = math.atan2(start_dir[1], start_dir[0])
    a1 = math.atan2(end_dir[1], end_dir[0])
    sweep = a1 - a0
    while sweep > math.pi:
        sweep -= 2 * math.pi
    while sweep < -math.pi:
        sweep += 2 * math.pi
    steps = max(2, int(abs(sweep) / 0.35) + 1)
    return [
        centre + radius * np.array([math.cos(a0 + sweep * i / steps), math.sin(a0 + sweep * i / steps)])
        for i in range(1, steps + 1)
    ]


def compensate(path: np.ndarray, offset: float) -> np.ndarray:
    """Carriage path whose *tip* traces `path`, for a blade trailing by `offset`.

    Straight runs are untouched, because a point offset along the tangent of a
    straight line is still on that line.  Only corners gain anything.
    """
    points = np.asarray(path, dtype=np.float64)
    if offset <= 1e-6 or len(points) < 3:
        return points
    xy = points[:, :2]
    closed = _is_closed(xy)
    ring = xy[:-1] if closed else xy

    count = len(ring)
    # a turn gentler than CORNER_DEGREES has cos above this and is left alone
    limit = math.cos(math.radians(CORNER_DEGREES))

    # The carriage starts on the first point with the blade hanging in whatever
    # direction the last cut left it, so the first `offset` mm is cut crooked.
    # That is what `overcut` is for on a closed shape.
    out: list[np.ndarray] = [ring[0].copy()]
    # walk the corners; on a closed ring the return to the first point is a
    # corner too, so it gets the same treatment as the rest
    stops = list(range(1, count)) + ([0] if closed else [])

    for index in stops:
        here = ring[index]
        incoming = _unit(here - ring[index - 1])
        outgoing = _unit(ring[(index + 1) % count] - here) if (closed or index < count - 1) else None
        if incoming is None:
            continue
        if outgoing is None:
            # end of an open cut: run on so the tip reaches the last point
            out.append(here + incoming * offset)
            break
        if float(np.dot(incoming, outgoing)) > limit:
            # near enough straight ahead: the blade tracks it without help
            out.append(here.copy())
            continue
        # run past the corner so the tip lands exactly on it, swing, carry on
        out.append(here + incoming * offset)
        out.extend(_swing(here, incoming, outgoing, offset))

    result = np.asarray(out, dtype=np.float64)
    if points.shape[1] > 2:   # keep any modulation column, flat across the arcs
        pad = np.full((len(result), points.shape[1] - 2), points[0, 2:], dtype=np.float64)
        result = np.hstack([result, pad])
    return result


def overcut_closed(path: np.ndarray, distance: float) -> np.ndarray:
    """Carry a closed cut `distance` mm past its own start point."""
    points = np.asarray(path, dtype=np.float64)
    if distance <= 1e-6 or not _is_closed(points):
        return points
    extra: list[np.ndarray] = []
    remaining = float(distance)
    cursor = points[-1]
    index = 1
    while remaining > 1e-6 and index < len(points):
        target = points[index]
        step = float(np.hypot(*(target[:2] - cursor[:2])))
        if step < 1e-9:
            index += 1
            continue
        if step >= remaining:
            extra.append(cursor + (target - cursor) * (remaining / step))
            break
        extra.append(target.copy())
        remaining -= step
        cursor = target
        index += 1
    if not extra:
        return points
    return np.vstack([points, np.asarray(extra, dtype=np.float64)])


def prepare(paths: list[np.ndarray], pen: Pen) -> list[np.ndarray]:
    """Apply whatever this tool needs before the cut is turned into G-code."""
    if not pen.cuts or (pen.blade_offset <= 1e-6 and pen.overcut <= 1e-6):
        return paths
    out: list[np.ndarray] = []
    for path in paths:
        points = np.asarray(path, dtype=np.float64)
        if len(points) < 2:
            out.append(points)
            continue
        # overcut first, while the shape is still closed and its start point is
        # still the place the blade was crooked
        if pen.overcut > 1e-6:
            points = overcut_closed(points, pen.overcut)
        if pen.swivels:
            points = compensate(points, pen.blade_offset)
        out.append(points)
    return out


def prepare_layer(layer, library) -> None:
    """In-place blade preparation for one layer, if its pen is a blade."""
    pen = library[layer.pen]
    if not pen.cuts:
        return
    layer.paths = prepare(layer.paths, pen)
