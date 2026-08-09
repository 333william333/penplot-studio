"""Six more drawing techniques.

They follow exactly the conventions of :mod:`penplot.core.techniques` - the
input is float32 with 0 = black ink and 1 = white paper, the output is a list of
polylines in pixel coordinates with Y pointing down, and every spatial setting
is a length in millimetres that the :class:`~penplot.core.techniques.Context`
turns into pixels.

The file is self-contained on purpose: it only *reads* from ``techniques`` and
adds its entries to ``REGISTRY`` from :func:`register`.  Hook it up with two
lines at the *end* of ``techniques.py`` - after the registry is built, so the
import is not circular::

    from .techniques_extra import register as _register_extra
    _register_extra()
"""

from __future__ import annotations

import math
import time
from collections import deque

import cv2
import numpy as np

from . import geometry as geo
from .strokefont import CAP, glyph
from .styles import sample_bilinear, split_by_mask
from .techniques import (
    MAX_DOTS,
    REGISTRY,
    Context,
    Param,
    Technique,
    _dither_points,
    _edge_tangents,
    _ink,
    _tone,
    lloyd_relax,
)

__all__ = ["register"]


# --------------------------------------------------------------------------
# small shared helpers
# --------------------------------------------------------------------------
def _finish(paths, width: int, height: int, min_points: int = 2) -> list[np.ndarray]:
    """Last line of defence: finite, float64, inside the paper."""
    out: list[np.ndarray] = []
    hi_x = float(width - 1)
    hi_y = float(height - 1)
    for raw in paths:
        path = np.asarray(raw, dtype=np.float64).reshape(-1, 2)
        if len(path) == 0:
            continue
        if not np.isfinite(path).all():
            path = path[np.isfinite(path).all(axis=1)]
        if len(path) < min_points:
            continue
        np.clip(path[:, 0], 0.0, hi_x, out=path[:, 0])
        np.clip(path[:, 1], 0.0, hi_y, out=path[:, 1])
        out.append(path)
    return out


def _quant(x: float, y: float) -> tuple[int, int]:
    """Lattice key so segments that share an end point really do share it."""
    return (int(round(x * 16.0)), int(round(y * 16.0)))


def _chain_segments(segments) -> list[np.ndarray]:
    """Weld unit segments into long polylines, preferring to carry straight on.

    Both the maze walls and the Voronoi borders come out as thousands of tiny
    two-point segments.  Walking the graph they form turns them into a handful
    of long strokes, which is what keeps the pen on the paper.
    """
    coords: dict[tuple[int, int], tuple[float, float]] = {}
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    adjacency: dict[tuple[int, int], list[int]] = {}

    for (x0, y0), (x1, y1) in segments:
        a = _quant(x0, y0)
        b = _quant(x1, y1)
        if a == b:
            continue
        coords.setdefault(a, (x0, y0))
        coords.setdefault(b, (x1, y1))
        index = len(edges)
        edges.append((a, b))
        adjacency.setdefault(a, []).append(index)
        adjacency.setdefault(b, []).append(index)

    if not edges:
        return []

    used = bytearray(len(edges))
    # start at the loose ends first, so open runs come out as one stroke
    order = [node for node, links in adjacency.items() if len(links) % 2 == 1]
    order.extend(adjacency.keys())

    out: list[np.ndarray] = []
    for node in order:
        while True:
            start_edge = -1
            for index in adjacency[node]:
                if not used[index]:
                    start_edge = index
                    break
            if start_edge < 0:
                break

            points = [coords[node]]
            current = node
            edge = start_edge
            direction = (0.0, 0.0)
            while edge >= 0:
                used[edge] = 1
                a, b = edges[edge]
                other = b if a == current else a
                px, py = coords[current]
                nx, ny = coords[other]
                length = math.hypot(nx - px, ny - py)
                direction = ((nx - px) / length, (ny - py) / length) if length > 1e-9 else direction
                points.append((nx, ny))
                current = other

                best = -1
                best_score = -2.0
                for index in adjacency[current]:
                    if used[index]:
                        continue
                    a2, b2 = edges[index]
                    other2 = b2 if a2 == current else a2
                    ox, oy = coords[other2]
                    cx, cy = coords[current]
                    length2 = math.hypot(ox - cx, oy - cy)
                    if length2 < 1e-9:
                        continue
                    score = (direction[0] * (ox - cx) + direction[1] * (oy - cy)) / length2
                    if score > best_score:
                        best_score = score
                        best = index
                edge = best
            out.append(np.asarray(points, dtype=np.float64))
    return out


def _grid_neighbours(points: np.ndarray, count: int, cell: float) -> np.ndarray:
    """Deterministic approximate k-nearest-neighbour table from a uniform grid.

    scipy is not a dependency and FLANN's forests are randomised, so this does
    the job with plain array operations: bucket the points, then look at the
    3x3 block of buckets around each one.
    """
    total = len(points)
    count = max(1, min(count, total - 1))
    gx = np.floor(points[:, 0] / cell).astype(np.int64)
    gy = np.floor(points[:, 1] / cell).astype(np.int64)
    # one empty ring of buckets all round, so the 3x3 block never has to be
    # folded back on itself - folding hands the same point in several times and
    # quietly halves the neighbour list along the border
    gx = gx - gx.min() + 1
    gy = gy - gy.min() + 1
    width = int(gx.max()) + 2
    height = int(gy.max()) + 2
    cell_id = gy * width + gx

    order = np.argsort(cell_id, kind="stable")
    sorted_id = cell_id[order]
    starts = np.searchsorted(sorted_id, np.arange(width * height))
    rank = np.arange(total) - starts[sorted_id]
    per_cell = max(int(np.bincount(cell_id).max()), 1)
    per_cell = min(per_cell, 24)

    table = np.full((width * height, per_cell), -1, dtype=np.int32)
    keep = rank < per_cell
    table[sorted_id[keep], rank[keep]] = order[keep]

    # the 3x3 block of buckets around every point; the padding ring above means
    # none of these indices can leave the table
    offsets = [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    blocks = [table[(gy + dy) * width + (gx + dx)] for dx, dy in offsets]
    candidates = np.concatenate(blocks, axis=1)

    valid = candidates >= 0
    safe = np.where(valid, candidates, 0)
    dx = points[safe, 0] - points[:, 0:1]
    dy = points[safe, 1] - points[:, 1:2]
    dist = np.hypot(dx, dy)
    dist[~valid] = np.inf
    dist[safe == np.arange(total)[:, None]] = np.inf

    take = min(count, dist.shape[1])
    part = np.argpartition(dist, take - 1, axis=1)[:, :take]
    rows = np.arange(total)[:, None]
    picked = np.take_along_axis(dist, part, axis=1)
    order2 = np.argsort(picked, axis=1)
    part = np.take_along_axis(part, order2, axis=1)
    picked = np.take_along_axis(picked, order2, axis=1)
    result = np.where(np.isfinite(picked), safe[rows, part], -1)
    return result.astype(np.int32)


# --------------------------------------------------------------------------
# 1. tsp - one continuous line through every stipple point
# --------------------------------------------------------------------------
def _nearest_neighbour_tour(points: np.ndarray, cell: float, ctx: Context) -> list[int]:
    """Greedy nearest-neighbour tour over a uniform grid with removals."""
    total = len(points)
    if total < 2:
        return list(range(total))
    xs = points[:, 0].tolist()
    ys = points[:, 1].tolist()

    buckets: dict[tuple[int, int], list[int]] = {}
    keys: list[tuple[int, int]] = []
    for index in range(total):
        key = (int(xs[index] // cell), int(ys[index] // cell))
        buckets.setdefault(key, []).append(index)
        keys.append(key)
    lo_x = min(k[0] for k in buckets)
    hi_x = max(k[0] for k in buckets)
    lo_y = min(k[1] for k in buckets)
    hi_y = max(k[1] for k in buckets)

    def take(index: int) -> None:
        bucket = buckets.get(keys[index])
        if bucket is not None:
            bucket.remove(index)
            if not bucket:
                del buckets[keys[index]]

    # start bottom-left-ish so the stroke has an obvious beginning
    start = int(np.argmin(points[:, 0] + points[:, 1]))
    take(start)
    tour = [start]
    cx, cy = xs[start], ys[start]

    for step in range(total - 1):
        if step % 64 == 0:
            ctx.check()
        gx = int(cx // cell)
        gy = int(cy // cell)
        best = -1
        best_d = math.inf
        ring = 0
        limit = max(abs(gx - lo_x), abs(gx - hi_x), abs(gy - lo_y), abs(gy - hi_y)) + 1
        while ring <= limit:
            if ring == 0:
                shell = ((gx, gy),)
            else:
                shell = []
                top = gy - ring
                bottom = gy + ring
                for x in range(gx - ring, gx + ring + 1):
                    shell.append((x, top))
                    shell.append((x, bottom))
                for y in range(gy - ring + 1, gy + ring):
                    shell.append((gx - ring, y))
                    shell.append((gx + ring, y))
            for key in shell:
                bucket = buckets.get(key)
                if not bucket:
                    continue
                for index in bucket:
                    d = math.hypot(xs[index] - cx, ys[index] - cy)
                    if d < best_d:
                        best_d = d
                        best = index
            if best >= 0 and best_d <= ring * cell:
                break
            ring += 1
        if best < 0:
            break
        take(best)
        tour.append(best)
        cx, cy = xs[best], ys[best]
    return tour


def _improve_tour(tour: list[int], points: np.ndarray, neighbours: np.ndarray,
                  budget: float, ctx: Context) -> list[int]:
    """2-opt and Or-opt with neighbour lists, don't-look bits and a deadline."""
    total = len(tour)
    if total < 5 or budget <= 0.0:
        return tour

    xs = points[:, 0].tolist()
    ys = points[:, 1].tolist()
    links = [row[row >= 0].tolist() for row in neighbours]

    position = [0] * total
    for slot, city in enumerate(tour):
        position[city] = slot

    def distance(a: int, b: int) -> float:
        return math.hypot(xs[a] - xs[b], ys[a] - ys[b])

    def reverse(i: int, j: int) -> None:
        span = (j - i) % total + 1
        if span * 2 > total:
            i, j = (j + 1) % total, (i - 1) % total
            span = total - span
        for _ in range(span // 2):
            a, b = tour[i], tour[j]
            tour[i] = b
            tour[j] = a
            position[b] = i
            position[a] = j
            i += 1
            if i == total:
                i = 0
            j -= 1
            if j < 0:
                j = total - 1

    queue = deque(tour)
    queued = bytearray([1]) * total
    deadline = time.perf_counter() + budget
    counter = 0

    def wake(city: int) -> None:
        if not queued[city]:
            queued[city] = 1
            queue.append(city)

    def long_edge_pass() -> bool:
        """Pair up the few remaining giant hops and try to cancel them out.

        A neighbour-list 2-opt cannot see these: when the line has to cross a
        white hole it does so twice, and each crossing can only be repaired
        against the *other* crossing, which is nowhere near it on the paper.
        There are never many of them, so simply trying every pair is cheap.
        """
        order = points[tour]
        lengths = np.hypot(*(np.roll(order, -1, axis=0) - order).T)
        picks = np.argsort(lengths)[::-1][:64]
        picks = picks[lengths[picks] > lengths.mean() * 3.0]
        if len(picks) < 2:
            return False
        best_gain = 1e-6
        move = None
        for n, i in enumerate(picks.tolist()):
            a = tour[i]
            b = tour[(i + 1) % total]
            for j in picks[n + 1:].tolist():
                c = tour[j]
                d = tour[(j + 1) % total]
                gain = lengths[i] + lengths[j] - distance(a, c) - distance(b, d)
                if gain > best_gain:
                    best_gain = gain
                    move = (i, j, a, b, c, d)
        if move is None:
            return False
        i, j, a, b, c, d = move
        reverse((i + 1) % total, j)
        for city in (a, b, c, d):
            wake(city)
        return True

    while True:
        while queue:
            counter += 1
            if counter % 64 == 0:
                ctx.check()
                if time.perf_counter() > deadline:
                    break

            a = queue.popleft()
            queued[a] = 0
            improved = False

            # ---- 2-opt --------------------------------------------------
            for forward in (True, False):
                slot = position[a]
                other = (slot + 1) % total if forward else (slot - 1) % total
                b = tour[other]
                d_ab = distance(a, b)
                for c in links[a]:
                    d_ac = distance(a, c)
                    if d_ac >= d_ab:
                        break
                    slot_c = position[c]
                    slot_d = (slot_c + 1) % total if forward else (slot_c - 1) % total
                    d = tour[slot_d]
                    if d == a:
                        continue
                    gain = d_ab + distance(c, d) - d_ac - distance(b, d)
                    if gain > 1e-7:
                        if forward:
                            reverse((slot + 1) % total, slot_c)
                        else:
                            reverse(slot_c, (slot - 1) % total)
                        for city in (a, b, c, d):
                            wake(city)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                wake(a)
                continue

            # ---- Or-opt: move a run of 1..3 cities somewhere better ------
            slot = position[a]
            for length in (1, 2, 3):
                if slot + length >= total:
                    break
                head = tour[slot]
                tail = tour[slot + length - 1]
                if slot == 0:
                    break
                before = tour[slot - 1]
                after = tour[slot + length]
                removed = distance(before, head) + distance(tail, after) - distance(before, after)
                if removed <= 1e-7:
                    continue
                best_gain = 1e-7
                best_at = -1
                best_flip = False
                for c in links[head] + links[tail]:
                    slot_c = position[c]
                    if slot - 1 <= slot_c <= slot + length:
                        continue
                    if slot_c + 1 >= total:
                        continue
                    d = tour[slot_c + 1]
                    base = distance(c, d)
                    straight = removed - (distance(c, head) + distance(tail, d) - base)
                    flipped = removed - (distance(c, tail) + distance(head, d) - base)
                    if straight > best_gain:
                        best_gain = straight
                        best_at = slot_c
                        best_flip = False
                    if flipped > best_gain:
                        best_gain = flipped
                        best_at = slot_c
                        best_flip = True
                if best_at < 0:
                    continue

                anchor = tour[best_at]
                segment = tour[slot:slot + length]
                if best_flip:
                    segment = segment[::-1]
                del tour[slot:slot + length]
                target = best_at + 1 - (length if best_at > slot else 0)
                tour[target:target] = segment
                low = min(slot, target)
                high = max(slot + length, target + length)
                for index in range(low, high):
                    position[tour[index]] = index
                for city in (head, tail, before, after, anchor):
                    wake(city)
                improved = True
                break

            if improved:
                wake(a)

        if time.perf_counter() > deadline:
            break
        if not long_edge_pass():
            break

    return tour


def _drop_strays(points: np.ndarray, reach: float) -> np.ndarray:
    """Throw away the specks that sit on their own.

    A single dot marooned in a pale corner forces the line to fly right across
    the picture and back, which is the one thing that spoils this technique.
    """
    for _ in range(3):
        if len(points) < 8:
            break
        near = _grid_neighbours(points, 2, reach)
        index = np.maximum(near, 0)
        far = np.hypot(points[index, 0] - points[:, 0:1], points[index, 1] - points[:, 1:2])
        far[near < 0] = np.inf
        keep = far.max(axis=1) < reach
        if keep.all():
            break
        points = points[keep]
    return points


def _render_tsp(image: np.ndarray, p: dict, ctx: Context) -> list[np.ndarray]:
    ink = _tone(_ink(image), p["tone"])
    height, width = ink.shape
    pitch = max(ctx.px(p["spacing"], pen_scaled=True), 1.5)
    rng = ctx.rng(31)

    floor = float(np.clip(p["background"], 0.0, 1.0))
    field = np.clip(floor + (1.0 - floor) * ink, 0.0, 1.0) if floor > 0.0 else ink

    points = _dither_points(field, pitch, p["min_ink"], rng)
    if len(points) < 4:
        return []
    cap = min(MAX_DOTS, 24_000)
    if len(points) > cap:
        keep = np.sort(rng.choice(len(points), cap, replace=False))
        points = points[keep]
    points = lloyd_relax(points, np.clip(field, 1e-3, None) ** 1.6, int(p["even_out"]))
    points[:, 0] = np.clip(points[:, 0], 0.0, width - 1.0)
    points[:, 1] = np.clip(points[:, 1], 0.0, height - 1.0)

    # two points on the same pixel would make a zero-length hop and confuse the
    # tour, and a lone speck would make an enormous one
    unique, index = np.unique(np.round(points, 3), axis=0, return_index=True)
    if len(unique) < len(points):
        points = points[np.sort(index)]
    points = _drop_strays(points, pitch * 3.0)
    if len(points) < 4:
        return []

    tour = _nearest_neighbour_tour(points, max(pitch * 1.2, 2.0), ctx)
    if len(tour) < 4:
        return []

    neighbours = _grid_neighbours(points, 10, max(pitch * 2.0, 4.0))
    tour = _improve_tour(tour, points, neighbours, float(p["effort"]), ctx)

    # the improver works on a closed loop; cutting its longest hop gives the
    # open stroke the famous one-line portraits are drawn with
    ordered = points[tour]
    steps = np.hypot(*(np.roll(ordered, -1, axis=0) - ordered).T)
    cut = int(np.argmax(steps))
    path = np.vstack([ordered[cut + 1:], ordered[:cut + 1]])
    return _finish([path], width, height)


# --------------------------------------------------------------------------
# 2. voronoi - the cell walls of a weighted point set
# --------------------------------------------------------------------------
def _clip_halfplane(polygon, nx: float, ny: float, offset: float):
    """Sutherland-Hodgman against nx*x + ny*y <= offset."""
    out = []
    count = len(polygon)
    if count == 0:
        return out
    px, py = polygon[-1]
    pv = nx * px + ny * py - offset
    for cx, cy in polygon:
        cv = nx * cx + ny * cy - offset
        if cv <= 0.0:
            if pv > 0.0:
                t = pv / (pv - cv)
                out.append((px + (cx - px) * t, py + (cy - py) * t))
            out.append((cx, cy))
        elif pv <= 0.0:
            t = pv / (pv - cv)
            out.append((px + (cx - px) * t, py + (cy - py) * t))
        px, py, pv = cx, cy, cv
    return out


def _render_voronoi(image: np.ndarray, p: dict, ctx: Context) -> list[np.ndarray]:
    ink = _tone(_ink(image), p["tone"])
    height, width = ink.shape
    pitch = max(ctx.px(p["spacing"], pen_scaled=True), 2.0)
    rng = ctx.rng(37)

    # a density floor keeps the pale areas tiled with big calm cells instead of
    # one enormous polygon swallowing the whole background
    floor = float(np.clip(p["background"], 0.0, 1.0))
    field = np.clip(floor + (1.0 - floor) * ink, 0.0, 1.0)

    points = _dither_points(field, pitch, 0.0, rng)
    if len(points) < 3:
        return []
    if len(points) > 40_000:
        keep = np.sort(rng.choice(len(points), 40_000, replace=False))
        points = points[keep]
    points = lloyd_relax(points, np.clip(field, 1e-3, None) ** 1.4, int(p["even_out"]))
    points[:, 0] = np.clip(points[:, 0], 0.5, width - 1.5)
    points[:, 1] = np.clip(points[:, 1], 0.5, height - 1.5)

    subdiv = cv2.Subdiv2D((0, 0, width, height))
    seen: set[tuple[int, int]] = set()
    kept: list[np.ndarray] = []
    for point in points:
        key = (int(point[0] * 4), int(point[1] * 4))
        if key in seen:
            continue
        seen.add(key)
        subdiv.insert((float(point[0]), float(point[1])))
        kept.append(point)
    if len(kept) < 3:
        return []
    centres = np.asarray(kept, dtype=np.float64)

    facets, _ = subdiv.getVoronoiFacetList([])
    density = sample_bilinear(ink, centres[:, 0], centres[:, 1])
    min_ink = float(p["min_ink"])
    border = bool(p["border"])
    edge_length = max(ctx.pen_px * 0.6, 0.8)

    hi_x = width - 1.0
    hi_y = height - 1.0
    segments = []
    edges: set[tuple[int, int, int, int]] = set()
    for index, facet in enumerate(facets):
        if index % 64 == 0:
            ctx.check()
        if index >= len(density) or (min_ink > 0.0 and density[index] <= min_ink):
            continue
        polygon = [(float(x), float(y)) for x, y in facet]
        polygon = _clip_halfplane(polygon, -1.0, 0.0, 0.0)
        polygon = _clip_halfplane(polygon, 1.0, 0.0, hi_x)
        polygon = _clip_halfplane(polygon, 0.0, -1.0, 0.0)
        polygon = _clip_halfplane(polygon, 0.0, 1.0, hi_y)
        if len(polygon) < 3:
            continue
        for i in range(len(polygon)):
            ax, ay = polygon[i - 1]
            bx, by = polygon[i]
            if math.hypot(bx - ax, by - ay) < edge_length:
                continue
            if not border:
                on_x = (ax <= 0.01 and bx <= 0.01) or (ax >= hi_x - 0.01 and bx >= hi_x - 0.01)
                on_y = (ay <= 0.01 and by <= 0.01) or (ay >= hi_y - 0.01 and by >= hi_y - 0.01)
                if on_x or on_y:
                    continue
            a = _quant(ax, ay)
            b = _quant(bx, by)
            key = a + b if a <= b else b + a
            if key in edges:
                continue
            edges.add(key)
            segments.append(((ax, ay), (bx, by)))

    return _finish(_chain_segments(segments), width, height)


# --------------------------------------------------------------------------
# 3. packing - non-overlapping circles sized by the tone
# --------------------------------------------------------------------------
def _render_packing(image: np.ndarray, p: dict, ctx: Context) -> list[np.ndarray]:
    ink = _tone(_ink(image), p["tone"])
    height, width = ink.shape
    rng = ctx.rng(41)

    small = max(ctx.px(p["min_radius"], pen_scaled=True), ctx.pen_px * 0.6)
    large = max(ctx.px(p["max_radius"], pen_scaled=True), small * 1.2)
    gap = ctx.pen_px * float(p["gap"])
    min_ink = float(p["min_ink"])
    invert = bool(p["invert"])

    attempts = int(max(p["attempts"], 1) * 1000)
    attempts = min(attempts, 400_000)

    # one big deterministic draw is far cheaper than a call per attempt
    xs = rng.uniform(0.0, width - 1.0, attempts)
    ys = rng.uniform(0.0, height - 1.0, attempts)
    values = np.clip(sample_bilinear(ink, xs, ys), 0.0, 1.0)
    shaped = values if invert else 1.0 - values
    targets = small + (large - small) * shaped
    xs_list = xs.tolist()
    ys_list = ys.tolist()
    target_list = targets.tolist()
    value_list = values.tolist()

    cell = large * 2.0 + gap
    buckets: dict[tuple[int, int], list[tuple[float, float, float]]] = {}
    circles: list[tuple[float, float, float]] = []

    for index in range(attempts):
        if index % 64 == 0:
            ctx.check()
        if value_list[index] <= min_ink:
            continue
        x = xs_list[index]
        y = ys_list[index]
        radius = target_list[index]
        # never let a circle run off the paper
        radius = min(radius, x, y, width - 1.0 - x, height - 1.0 - y)
        if radius < small:
            continue
        gx = int(x // cell)
        gy = int(y // cell)
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                for cx, cy, cr in buckets.get((gx + ox, gy + oy), ()):
                    allowed = math.hypot(cx - x, cy - y) - cr - gap
                    if allowed < radius:
                        radius = allowed
                        if radius < small:
                            break
                if radius < small:
                    break
            if radius < small:
                break
        if radius < small:
            continue
        entry = (x, y, radius)
        buckets.setdefault((gx, gy), []).append(entry)
        circles.append(entry)

    rings = max(int(p["rings"]), 1)
    out: list[np.ndarray] = []
    for x, y, radius in circles:
        for ring in range(rings):
            r = radius * (ring + 1) / rings
            if r < ctx.pen_px * 0.4:
                continue
            steps = int(np.clip(r * 1.4, 8, 48))
            out.append(geo.circle_path(x, y, r, steps))
    return _finish(out, width, height)


# --------------------------------------------------------------------------
# 4. maze - a perfect maze whose walls appear only where there is ink
# --------------------------------------------------------------------------
_PERMUTATIONS = [
    (0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 1, 3), (0, 2, 3, 1), (0, 3, 1, 2), (0, 3, 2, 1),
    (1, 0, 2, 3), (1, 0, 3, 2), (1, 2, 0, 3), (1, 2, 3, 0), (1, 3, 0, 2), (1, 3, 2, 0),
    (2, 0, 1, 3), (2, 0, 3, 1), (2, 1, 0, 3), (2, 1, 3, 0), (2, 3, 0, 1), (2, 3, 1, 0),
    (3, 0, 1, 2), (3, 0, 2, 1), (3, 1, 0, 2), (3, 1, 2, 0), (3, 2, 0, 1), (3, 2, 1, 0),
]
_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _carve_maze(rows: int, columns: int, rng: np.random.Generator, ctx: Context):
    """Randomised depth-first search - the classic 'perfect maze' carve."""
    visited = np.zeros((rows, columns), dtype=bool)
    vertical = np.ones((rows, columns + 1), dtype=bool)     # between (r,c-1) and (r,c)
    horizontal = np.ones((rows + 1, columns), dtype=bool)   # between (r-1,c) and (r,c)

    choices = rng.integers(0, len(_PERMUTATIONS), size=rows * columns).tolist()
    stack = [(0, 0)]
    visited[0, 0] = True
    counter = 0
    while stack:
        counter += 1
        if counter % 64 == 0:
            ctx.check()
        row, column = stack[-1]
        moved = False
        for step in _PERMUTATIONS[choices[(row * columns + column) % len(choices)]]:
            dr, dc = _STEPS[step]
            r2 = row + dr
            c2 = column + dc
            if r2 < 0 or c2 < 0 or r2 >= rows or c2 >= columns or visited[r2, c2]:
                continue
            if dr == 0:
                vertical[row, column + (1 if dc > 0 else 0)] = False
            else:
                horizontal[row + (1 if dr > 0 else 0), column] = False
            visited[r2, c2] = True
            stack.append((r2, c2))
            moved = True
            break
        if not moved:
            stack.pop()
    return vertical, horizontal


def _render_maze(image: np.ndarray, p: dict, ctx: Context) -> list[np.ndarray]:
    ink = _tone(_ink(image), p["tone"])
    height, width = ink.shape
    cell = max(ctx.px(p["cell"], pen_scaled=True), 2.0)
    columns = int(max(min((width - 1) / cell, 400), 3))
    rows = int(max(min((height - 1) / cell, 400), 3))
    step_x = (width - 1.0) / columns
    step_y = (height - 1.0) / rows

    rng = ctx.rng(43)
    vertical, horizontal = _carve_maze(rows, columns, rng, ctx)

    min_ink = float(p["min_ink"])
    solid = max(float(p["solid"]), min_ink + 1e-3)

    # sample the picture at the middle of every wall in one go
    v_rows, v_columns = np.nonzero(vertical)
    v_x = v_columns * step_x
    v_y = (v_rows + 0.5) * step_y
    h_rows, h_columns = np.nonzero(horizontal)
    h_x = (h_columns + 0.5) * step_x
    h_y = h_rows * step_y

    v_value = sample_bilinear(ink, np.clip(v_x, 0, width - 1), np.clip(v_y, 0, height - 1))
    h_value = sample_bilinear(ink, np.clip(h_x, 0, width - 1), np.clip(h_y, 0, height - 1))

    def survivors(value: np.ndarray, salt: int) -> np.ndarray:
        chance = np.clip((value - min_ink) / (solid - min_ink), 0.0, 1.0)
        return (value > min_ink) & (ctx.rng(salt).random(len(value)) < chance)

    v_keep = survivors(v_value, 44)
    h_keep = survivors(h_value, 45)

    segments = []
    for column, row in zip(v_columns[v_keep], v_rows[v_keep]):
        x = column * step_x
        segments.append(((x, row * step_y), (x, (row + 1) * step_y)))
    for column, row in zip(h_columns[h_keep], h_rows[h_keep]):
        y = row * step_y
        segments.append(((column * step_x, y), ((column + 1) * step_x, y)))

    # a wall on its own in a pale corner reads as dirt, not as a maze
    shortest = min(step_x, step_y) * 1.6
    runs = [run for run in _chain_segments(segments) if geo.path_length(run) >= shortest]
    return _finish(runs, width, height)


# --------------------------------------------------------------------------
# 5. mosaic - the picture typed out in the built-in single-stroke font
# --------------------------------------------------------------------------
def _glyph_coverage(ch: str) -> float:
    """Ink a character puts down per unit of the box it occupies."""
    advance, strokes = glyph(ch)
    length = sum(geo.path_length(np.asarray(s, dtype=np.float64)) for s in strokes)
    return length / max(advance * CAP, 1.0)


def _ramp(text: str) -> list[str]:
    """Order the characters from palest to blackest, measured not guessed."""
    unique = []
    for ch in text:
        if ch not in unique:
            unique.append(ch)
    if not unique:
        unique = [" "]
    return sorted(unique, key=_glyph_coverage)


def _render_mosaic(image: np.ndarray, p: dict, ctx: Context) -> list[np.ndarray]:
    ink = _tone(_ink(image), p["tone"])
    height, width = ink.shape
    cell = max(ctx.px(p["cell"], pen_scaled=True), 3.0)
    columns = int(max(min(width / cell, 400), 1))
    rows = int(max(min(height / cell, 400), 1))
    step_x = width / columns
    step_y = height / rows

    tiles = cv2.resize(ink, (columns, rows), interpolation=cv2.INTER_AREA)
    ramp = _ramp(str(p["charset"]))
    min_ink = float(p["min_ink"])

    # Matching each cell to the character whose *measured* coverage comes
    # closest to the tone it wants gives a far smoother ramp than stepping
    # through the string, because the glyphs are nowhere near evenly spaced.
    coverage = np.array([_glyph_coverage(ch) for ch in ramp], dtype=np.float64)
    darkest = float(coverage.max()) or 1.0
    wanted = np.clip(tiles, 0.0, 1.0) * darkest
    index = np.argmin(np.abs(coverage[None, None, :] - wanted[:, :, None]), axis=2)

    scale = min(step_x, step_y) * float(p["size"]) / CAP
    cache: dict[str, tuple[float, list[np.ndarray]]] = {}
    out: list[np.ndarray] = []
    for row in range(rows):
        ctx.check()
        for column in range(columns):
            if tiles[row, column] <= min_ink:
                continue
            ch = ramp[index[row, column]]
            item = cache.get(ch)
            if item is None:
                item = glyph(ch)
                cache[ch] = item
            advance, strokes = item
            if not strokes:
                continue
            origin_x = (column + 0.5) * step_x - advance * scale * 0.5
            baseline = (row + 0.5) * step_y + CAP * scale * 0.5
            for stroke in strokes:
                points = np.asarray(stroke, dtype=np.float64)
                out.append(
                    np.stack(
                        [origin_x + points[:, 0] * scale, baseline - points[:, 1] * scale],
                        axis=1,
                    )
                )
    return _finish(out, width, height)


# --------------------------------------------------------------------------
# 6. crosscontour - engraving lines that run across the form
# --------------------------------------------------------------------------
def _cross_field(image: np.ndarray, coherence: float, fallback_deg: float, ctx: Context):
    """Smooth direction field pointing *across* the shapes.

    The structure tensor gives an orientation but no confidence, and a flat wall
    of grey has no orientation at all.  Blending the measured direction with a
    fixed fallback angle - in double-angle space, where directions add properly -
    keeps the lines calm where the picture is featureless and lets them bend
    around the forms where it is not.
    """
    smoothing = max(coherence, 0.2) * 4.0
    tangent_x, tangent_y = _edge_tangents(image, smoothing)
    # across the form, not along it
    across = np.arctan2(-tangent_x, tangent_y)

    blurred = cv2.GaussianBlur(image, (0, 0), max(smoothing * 0.4, 0.6))
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    energy = cv2.GaussianBlur(gx * gx + gy * gy, (0, 0), max(smoothing, 1.0))
    scale = float(np.percentile(energy, 92)) or 1.0
    confidence = (energy / (energy + scale * 0.25)).astype(np.float32)

    fallback = math.radians(fallback_deg)
    cos2 = confidence * np.cos(2.0 * across) + (1.0 - confidence) * math.cos(2.0 * fallback)
    sin2 = confidence * np.sin(2.0 * across) + (1.0 - confidence) * math.sin(2.0 * fallback)
    cos2 = cv2.GaussianBlur(cos2.astype(np.float32), (0, 0), max(smoothing * 0.5, 0.8))
    sin2 = cv2.GaussianBlur(sin2.astype(np.float32), (0, 0), max(smoothing * 0.5, 0.8))
    angle = 0.5 * np.arctan2(sin2, cos2)
    ctx.check()
    return np.cos(angle).astype(np.float32), np.sin(angle).astype(np.float32)


def _render_crosscontour(image: np.ndarray, p: dict, ctx: Context) -> list[np.ndarray]:
    ink = _tone(_ink(image), p["tone"])
    height, width = ink.shape
    field_x, field_y = _cross_field(image, p["coherence"], p["angle"], ctx)

    spacing = max(ctx.px(p["spacing"], pen_scaled=True), 1.2)
    step = max(ctx.px(p["step"], pen_scaled=True), 0.8)
    max_steps = int(max(ctx.px(p["max_length"]) / step, 6))
    min_ink = float(p["min_ink"])
    layers = max(int(p["layers"]), 1)

    def grow(cell: float, floor: float, placed: list[np.ndarray]) -> list[np.ndarray]:
        """Evenly spaced streamlines at *cell* apart, over everything darker
        than *floor*, threading between the lines already on the paper."""
        grid = np.zeros((int(height / cell) + 2, int(width / cell) + 2), dtype=bool)
        for done in placed:
            gx = np.clip((done[:, 0] / cell).astype(np.int32), 0, grid.shape[1] - 1)
            gy = np.clip((done[:, 1] / cell).astype(np.int32), 0, grid.shape[0] - 1)
            grid[gy, gx] = True

        def occupied(x: float, y: float) -> bool:
            gy = int(y / cell)
            gx = int(x / cell)
            if gy < 0 or gx < 0 or gy >= grid.shape[0] or gx >= grid.shape[1]:
                return True
            return bool(grid[gy, gx])

        rng = ctx.rng(47 + int(cell))
        seed_step = max(cell * 0.5, 1.2)
        ys, xs = np.mgrid[0:height:seed_step, 0:width:seed_step]
        seeds = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)
        seeds += rng.uniform(-seed_step * 0.35, seed_step * 0.35, seeds.shape)
        seeds[:, 0] = np.clip(seeds[:, 0], 0.0, width - 1.0)
        seeds[:, 1] = np.clip(seeds[:, 1], 0.0, height - 1.0)
        strength = sample_bilinear(ink, seeds[:, 0], seeds[:, 1])
        order = np.argsort(-strength)
        seeds = seeds[order]
        strength = strength[order]

        made: list[np.ndarray] = []
        for count, (seed, value) in enumerate(zip(seeds, strength)):
            if count % 64 == 0:
                ctx.check()
            if value <= floor:
                break
            if occupied(seed[0], seed[1]):
                continue
            points = [(float(seed[0]), float(seed[1]))]
            for way in (1.0, -1.0):
                x, y = float(seed[0]), float(seed[1])
                dir_x = float(field_x[int(y), int(x)]) * way
                dir_y = float(field_y[int(y), int(x)]) * way
                trail = []
                for _ in range(max_steps):
                    ix = int(round(x))
                    iy = int(round(y))
                    if not (0 <= ix < width and 0 <= iy < height):
                        break
                    vx = float(field_x[iy, ix])
                    vy = float(field_y[iy, ix])
                    if vx * dir_x + vy * dir_y < 0.0:      # the field has no sign
                        vx, vy = -vx, -vy
                    dir_x, dir_y = vx, vy
                    x += vx * step
                    y += vy * step
                    if not (0 <= x < width and 0 <= y < height):
                        break
                    if float(sample_bilinear(ink, np.array([x]), np.array([y]))[0]) <= floor:
                        break
                    if occupied(x, y):
                        break
                    trail.append((x, y))
                if way > 0:
                    points.extend(trail)
                else:
                    points = list(reversed(trail)) + points
            if len(points) < 3:
                continue
            path = np.asarray(points, dtype=np.float64)
            gx = np.clip((path[:, 0] / cell).astype(np.int32), 0, grid.shape[1] - 1)
            gy = np.clip((path[:, 1] / cell).astype(np.int32), 0, grid.shape[0] - 1)
            grid[gy, gx] = True
            # a two-pixel dash left over between two neighbours reads as dirt
            if geo.path_length(path) >= cell:
                made.append(path)
        return made

    # Each pass halves the spacing and only covers what is darker than its own
    # threshold, so the tone comes from how tightly the lines are packed - the
    # way an engraver lays in a second and third set of lines over the shadows.
    lines: list[tuple[np.ndarray, float]] = []
    placed: list[np.ndarray] = []
    previous = math.inf
    for layer in range(layers):
        # below one pen width the lines merely overprint each other, and the
        # occupancy grid would grow to hundreds of megabytes for nothing
        cell = max(spacing / (2 ** layer), ctx.pen_px)
        if cell >= previous:
            break
        previous = cell
        floor = min_ink if layer == 0 else min_ink + (1.0 - min_ink) * layer / layers
        fresh = grow(cell, floor, placed)
        placed.extend(fresh)
        lines.extend((path, cell) for path in fresh)

    # ---- optional swell, so a single pen can still fake a thicker line ---
    weight = float(p["weight"])
    wave = float(p["wave"])
    out: list[np.ndarray] = []
    for path, cell in lines:
        amplitude = cell * 0.5 * weight
        wavelength = max(cell * wave, 1.5)
        if len(path) < 3:
            continue
        if amplitude <= 0.05:
            out.append(path)
            continue
        delta = np.diff(path, axis=0)
        seg = np.hypot(delta[:, 0], delta[:, 1])
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        tangent = np.vstack([delta[:1], (delta[:-1] + delta[1:]) * 0.5, delta[-1:]])
        norm = np.hypot(tangent[:, 0], tangent[:, 1])
        norm[norm < 1e-9] = 1.0
        nx = -tangent[:, 1] / norm
        ny = tangent[:, 0] / norm
        density = np.clip(sample_bilinear(ink, path[:, 0], path[:, 1]), 0.0, 1.0)
        # fade the swell in and out so the stroke does not end on a spike
        ends = np.minimum(arc, arc[-1] - arc) / max(wavelength, 1e-6)
        taper = np.clip(ends, 0.0, 1.0)
        swell = amplitude * density * taper * np.sin(2.0 * math.pi * arc / wavelength)
        xs2 = path[:, 0] + nx * swell
        ys2 = path[:, 1] + ny * swell
        inside = (xs2 >= 0) & (xs2 <= width - 1) & (ys2 >= 0) & (ys2 <= height - 1)
        out.extend(split_by_mask(xs2, ys2, inside))
    return _finish(out, width, height)


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------
def _tone_param(default: float = 1.0) -> Param:
    return Param(
        "tone", "Tone curve", default, 0.3, 3.0, 0.05, 2,
        hint="Below 1 lightens the midtones, above 1 darkens them.",
    )


_RAMPS = {
    " .:-=+*#%@": "Classic   . : - = + * # % @",
    " .,:;i1tfLC0@": "Fine      . , : ; i 1 t f L C 0 @",
    " .oO0#@": "Bold      . o O 0 # @",
    " 123456789": "Digits    1 2 3 4 5 6 7 8 9",
    " -+*#@": "Short     - + * # @",
    "01": "Binary    0 1",
}


def register() -> None:
    """Add the six techniques to the shared registry (called once)."""
    REGISTRY["tsp"] = Technique(
        "tsp", "Single line", "line",
        "Every tone point threaded onto one unbroken stroke - the classic "
        "TSP-art portrait the pen draws without ever lifting.",
        [
            Param("spacing", "Point spacing", 1.15, 0.4, 6.0, 0.05, 2, "mm", pen_scaled=True,
                  hint="Closer points mean a finer picture and a much longer line."),
            Param("even_out", "Even out", 3, 0, 8, 1, 0,
                  hint="Relaxation passes. More gives an evenly spread line."),
            Param("effort", "Improve for", 1.5, 0.0, 8.0, 0.1, 1, "s",
                  hint="Time spent untangling the tour. Longer means fewer crossings."),
            Param("background", "Pale-area density", 0.0, 0.0, 1.0, 0.02, 2,
                  hint="Above 0 the line also wanders over the white paper, which is how "
                       "the classic full-page portraits are drawn."),
            Param("min_ink", "Skip lighter than", 0.05, 0.0, 0.6, 0.01, 2),
            _tone_param(1.15),
        ],
        _render_tsp,
        slow=True,
    )

    REGISTRY["voronoi"] = Technique(
        "voronoi", "Voronoi cells", "geometric",
        "Points spread by tone, then the walls between their territories - "
        "a cracked-glass mosaic that tightens in the shadows.",
        [
            Param("spacing", "Cell size", 2.2, 0.6, 15.0, 0.05, 2, "mm", pen_scaled=True,
                  hint="The size of a cell in the darkest area."),
            Param("even_out", "Even out", 4, 0, 10, 1, 0,
                  hint="Relaxation passes. More makes the cells rounder and more even."),
            Param("background", "Pale-area density", 0.08, 0.0, 1.0, 0.02, 2,
                  hint="How many cells the white paper still gets. Lower means bigger, "
                       "calmer cells there and more contrast with the shadows."),
            Param("border", "Draw the edge of the paper", True, kind="bool"),
            Param("min_ink", "Skip lighter than", 0.0, 0.0, 0.6, 0.01, 2),
            _tone_param(),
        ],
        _render_voronoi,
    )

    REGISTRY["packing"] = Technique(
        "packing", "Circle packing", "dots",
        "Circles pressed together until they touch, small and crowded where the "
        "picture is dark, wide and open where it is pale.",
        [
            Param("min_radius", "Smallest circle", 0.5, 0.15, 5.0, 0.05, 2, "mm", pen_scaled=True),
            Param("max_radius", "Largest circle", 5.0, 0.5, 20.0, 0.1, 2, "mm", pen_scaled=True),
            Param("attempts", "Density", 80.0, 5.0, 400.0, 5.0, 0, "k tries",
                  hint="How hard to try to squeeze more circles in."),
            Param("gap", "Gap", 0.5, 0.0, 4.0, 0.05, 2, "pen widths"),
            Param("rings", "Rings per circle", 1, 1, 4, 1, 0,
                  hint="Draw each circle more than once to darken it."),
            Param("invert", "Big circles in the shadows", False, kind="bool"),
            Param("min_ink", "Skip lighter than", 0.04, 0.0, 0.6, 0.01, 2),
            _tone_param(),
        ],
        _render_packing,
    )

    REGISTRY["maze"] = Technique(
        "maze", "Maze", "geometric",
        "A perfect maze carved over the whole sheet, with its walls left "
        "standing only where the picture has ink.",
        [
            Param("cell", "Cell size", 1.6, 0.5, 12.0, 0.05, 2, "mm", pen_scaled=True),
            Param("solid", "Solid from", 0.45, 0.05, 1.0, 0.05, 2,
                  hint="Tone at which every wall is drawn. Paler areas keep fewer of them."),
            Param("min_ink", "Skip lighter than", 0.05, 0.0, 0.6, 0.01, 2),
            _tone_param(1.1),
        ],
        _render_maze,
    )

    REGISTRY["mosaic"] = Technique(
        "mosaic", "Glyph mosaic", "shading",
        "The picture typed out in the built-in engraving font, one character "
        "per cell, chosen by how much ink it puts on the paper.",
        [
            Param("cell", "Cell size", 4.5, 0.8, 20.0, 0.1, 2, "mm", pen_scaled=True),
            Param("charset", "Characters", " .:-=+*#%@", kind="choice", choices=_RAMPS,
                  hint="Matched to the tone by measured ink, not by their order in the list."),
            Param("size", "Character size", 0.86, 0.3, 1.3, 0.02, 2,
                  hint="Height of a capital as a fraction of the cell."),
            Param("min_ink", "Skip lighter than", 0.03, 0.0, 0.6, 0.01, 2),
            _tone_param(),
        ],
        _render_mosaic,
    )

    REGISTRY["crosscontour"] = Technique(
        "crosscontour", "Form lines", "shading",
        "Lines that run across the form instead of along it, packed tighter over "
        "the shadows - the way a copperplate engraver models a sphere.",
        [
            Param("spacing", "Line spacing", 2.2, 0.4, 12.0, 0.05, 2, "mm", pen_scaled=True,
                  hint="Spacing in the palest area. Each shading pass halves it."),
            Param("layers", "Shading passes", 3, 1, 4, 1, 0,
                  hint="Extra passes laid between the lines over the darker areas."),
            Param("weight", "Swell", 0.0, 0.0, 1.0, 0.05, 2,
                  hint="Waver each line to fake a thicker one. 0 draws clean lines."),
            Param("wave", "Swell rate", 1.2, 0.4, 4.0, 0.05, 2,
                  hint="Length of one swell, as a multiple of the line spacing."),
            Param("coherence", "Follow the form", 1.6, 0.2, 6.0, 0.1, 2,
                  hint="Higher makes the lines calmer and less willing to bend."),
            Param("angle", "Angle on flat areas", 45.0, 0.0, 180.0, 5.0, 0, "°",
                  hint="Direction the lines take where the picture has no shape to follow."),
            Param("step", "Smoothness", 0.6, 0.2, 4.0, 0.05, 2, "mm", pen_scaled=True),
            Param("max_length", "Max line length", 60.0, 3.0, 300.0, 1.0, 1, "mm"),
            Param("min_ink", "Skip lighter than", 0.05, 0.0, 0.6, 0.01, 2),
            _tone_param(),
        ],
        _render_crosscontour,
        stitchable=True,
        slow=True,
    )
