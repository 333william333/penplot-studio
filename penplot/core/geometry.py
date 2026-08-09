"""Polyline geometry helpers.

A *path* is an (N, 2) float64 numpy array of points.  Everything downstream of
the source stage (image tracing, text, PDF) speaks this language, which keeps
layout, optimisation, preview and G-code generation completely decoupled from
where the geometry came from.

Source stages emit paths in a Y-down coordinate system (image/screen
convention).  The layout stage flips Y once, so bed coordinates are Y-up like
the printer expects.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

Path = np.ndarray
Paths = list

__all__ = [
    "as_path",
    "path_length",
    "total_length",
    "bounds",
    "identity",
    "affine",
    "apply_matrix",
    "transform_paths",
    "rdp",
    "simplify_paths",
    "drop_short",
    "join_paths",
    "reorder_paths",
    "travel_length",
    "stitch_paths",
    "improve_tour",
    "resample",
    "circle_path",
    "cumulative_lengths",
]


# --------------------------------------------------------------------------
# basics
# --------------------------------------------------------------------------
def as_path(points: Sequence) -> Path:
    """Coerce anything point-like into an (N, 2) float64 array."""
    arr = np.asarray(points, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return arr.reshape(-1, 2)


def path_length(path: Path) -> float:
    if len(path) < 2:
        return 0.0
    d = np.diff(path[:, :2], axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


def total_length(paths: Iterable[Path]) -> float:
    return float(sum(path_length(p) for p in paths))


def cumulative_lengths(paths: Sequence[Path]) -> np.ndarray:
    """Running total of drawn length after each path (used for animation)."""
    out = np.zeros(len(paths), dtype=np.float64)
    acc = 0.0
    for i, p in enumerate(paths):
        acc += path_length(p)
        out[i] = acc
    return out


def bounds(paths: Sequence[Path]) -> tuple[float, float, float, float] | None:
    """(min_x, min_y, max_x, max_y) over every point, or None when empty."""
    lo_x = lo_y = math.inf
    hi_x = hi_y = -math.inf
    found = False
    for p in paths:
        if len(p) == 0:
            continue
        found = True
        low = p.min(axis=0)
        high = p.max(axis=0)
        lo_x = min(lo_x, float(low[0]))
        lo_y = min(lo_y, float(low[1]))
        hi_x = max(hi_x, float(high[0]))
        hi_y = max(hi_y, float(high[1]))
    if not found:
        return None
    return lo_x, lo_y, hi_x, hi_y


# --------------------------------------------------------------------------
# affine transforms (3x3, row vectors are points)
# --------------------------------------------------------------------------
def identity() -> np.ndarray:
    return np.eye(3, dtype=np.float64)


def affine(
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    rotation_deg: float = 0.0,
    translate: tuple[float, float] = (0.0, 0.0),
    pivot: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Scale about *pivot*, then rotate about *pivot*, then translate."""
    px, py = pivot
    to_origin = np.array([[1, 0, -px], [0, 1, -py], [0, 0, 1]], dtype=np.float64)
    scale = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]], dtype=np.float64)
    a = math.radians(rotation_deg)
    ca, sa = math.cos(a), math.sin(a)
    rot = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], dtype=np.float64)
    back = np.array([[1, 0, px + translate[0]], [0, 1, py + translate[1]], [0, 0, 1]], dtype=np.float64)
    return back @ rot @ scale @ to_origin


def apply_matrix(matrix: np.ndarray, path: Path) -> Path:
    """Transform the XY of a path, leaving any extra columns untouched.

    A third column carries a per-point weight (used for pressure and speed
    modulation), and it must survive layout and optimisation unchanged.
    """
    if len(path) == 0:
        return path
    xy = np.empty((len(path), 3), dtype=np.float64)
    xy[:, :2] = path[:, :2]
    xy[:, 2] = 1.0
    out = (xy @ matrix.T)[:, :2]
    if path.shape[1] > 2:
        return np.hstack([out, path[:, 2:]])
    return out


def transform_paths(paths: Sequence[Path], matrix: np.ndarray) -> list[Path]:
    return [apply_matrix(matrix, p) for p in paths if len(p)]


# --------------------------------------------------------------------------
# simplification
# --------------------------------------------------------------------------
def rdp(points: Path, epsilon: float) -> Path:
    """Ramer-Douglas-Peucker simplification (iterative, no recursion limit)."""
    n = len(points)
    if n < 3 or epsilon <= 0.0:
        return points

    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack: list[tuple[int, int]] = [(0, n - 1)]

    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        p0 = points[i, :2]
        seg = points[j, :2] - p0
        seg_len = math.hypot(seg[0], seg[1])
        rel = points[i + 1 : j, :2] - p0
        if seg_len < 1e-12:
            dist = np.hypot(rel[:, 0], rel[:, 1])
        else:
            dist = np.abs(seg[0] * rel[:, 1] - seg[1] * rel[:, 0]) / seg_len
        k = int(np.argmax(dist))
        if dist[k] > epsilon:
            idx = i + 1 + k
            keep[idx] = True
            stack.append((i, idx))
            stack.append((idx, j))

    return points[keep]


def simplify_paths(paths: Sequence[Path], epsilon: float) -> list[Path]:
    """Ramer-Douglas-Peucker over a list, leaving small shapes intact.

    A 1.2 mm dot drawn as a ten-sided polygon has a sagitta of 0.03 mm, so a
    0.12 mm tolerance collapses it to a zero-length line - the dot silently
    disappears.  Anything whose own bounding box is comparable to the tolerance
    is therefore left alone.
    """
    if epsilon <= 0:
        return list(paths)
    out = []
    floor = epsilon * 6.0
    for p in paths:
        if len(p) >= 3:
            extent = p[:, :2].max(axis=0) - p[:, :2].min(axis=0)
            if float(max(extent[0], extent[1])) <= floor:
                out.append(p)
                continue
        simplified = rdp(p, epsilon)
        if len(simplified) >= 2 or (len(p) == 1):
            out.append(simplified)
    return out


def drop_short(paths: Sequence[Path], min_length: float, keep_dots: bool = True) -> list[Path]:
    """Remove specks.  Single-point "dots" survive when *keep_dots* is set."""
    out = []
    for p in paths:
        if len(p) < 2:
            if keep_dots and len(p) == 1:
                out.append(p)
            continue
        if path_length(p) >= min_length:
            out.append(p)
        elif keep_dots and min_length > 0 and len(p) >= 2 and path_length(p) < 1e-9:
            out.append(p[:1])
    return out


# --------------------------------------------------------------------------
# endpoint index used by join / reorder
# --------------------------------------------------------------------------
class _EndpointIndex:
    """Uniform-grid nearest-neighbour index with removals."""

    def __init__(self, cell: float):
        self.cell = max(cell, 1e-6)
        self.cells: dict[tuple[int, int], set] = {}
        self.points: dict[object, np.ndarray] = {}
        self._extent: tuple[int, int, int, int] | None = None

    def _key(self, pt) -> tuple[int, int]:
        return (int(math.floor(pt[0] / self.cell)), int(math.floor(pt[1] / self.cell)))

    def add(self, key, pt) -> None:
        pt = np.asarray(pt, dtype=np.float64)
        self.points[key] = pt
        self.cells.setdefault(self._key(pt), set()).add(key)
        self._extent = None

    def remove(self, key) -> None:
        pt = self.points.pop(key, None)
        if pt is None:
            return
        ck = self._key(pt)
        bucket = self.cells.get(ck)
        if bucket:
            bucket.discard(key)
            if not bucket:
                del self.cells[ck]

    def __len__(self) -> int:
        return len(self.points)

    def nearest(self, pt, max_dist: float | None = None):
        """Return (key, distance) of the closest stored point, or (None, inf).

        Only the perimeter of each ring is visited.  Scanning the full square
        and filtering it was costing over a million comparisons on a stipple
        job, and it is pure waste - the interior was already searched.
        """
        count = len(self.points)
        if not count:
            return None, math.inf

        px, py = pt[0], pt[1]

        # once only a handful are left, walking rings costs more than looking
        if count <= 24:
            best_key, best_d = None, math.inf
            for key, point in self.points.items():
                d = math.hypot(point[0] - px, point[1] - py)
                if d < best_d:
                    best_d, best_key = d, key
            if max_dist is not None and best_d > max_dist:
                return None, math.inf
            return best_key, best_d

        cx, cy = self._key(pt)
        best_key = None
        best_d = math.inf
        cell = self.cell
        ring = 0
        max_ring = self._max_ring(cx, cy)

        while ring <= max_ring:
            if ring == 0:
                shell = ((cx, cy),)
            else:
                shell = []
                top, bottom = cy - ring, cy + ring
                for gx in range(cx - ring, cx + ring + 1):
                    shell.append((gx, top))
                    shell.append((gx, bottom))
                for gy in range(cy - ring + 1, cy + ring):
                    shell.append((cx - ring, gy))
                    shell.append((cx + ring, gy))

            for key_cell in shell:
                bucket = self.cells.get(key_cell)
                if not bucket:
                    continue
                for key in bucket:
                    point = self.points[key]
                    d = math.hypot(point[0] - px, point[1] - py)
                    if d < best_d:
                        best_d = d
                        best_key = key

            # a hit inside ring R is only optimal once every cell that could be
            # closer has been searched as well
            if best_key is not None and best_d <= ring * cell:
                break
            if max_dist is not None and (ring - 1) * cell > max_dist and best_key is None:
                return None, math.inf
            ring += 1

        if max_dist is not None and best_d > max_dist:
            return None, math.inf
        return best_key, best_d

    def _max_ring(self, cx: int, cy: int) -> int:
        """How far the search could ever need to go, from the occupied extent."""
        if self._extent is None:
            keys = self.cells.keys()
            xs = [k[0] for k in keys]
            ys = [k[1] for k in keys]
            self._extent = (min(xs), min(ys), max(xs), max(ys)) if xs else (0, 0, 0, 0)
        lo_x, lo_y, hi_x, hi_y = self._extent
        return max(abs(cx - lo_x), abs(cx - hi_x), abs(cy - lo_y), abs(cy - hi_y)) + 1


# --------------------------------------------------------------------------
# joining & ordering
# --------------------------------------------------------------------------
def join_paths(paths: Sequence[Path], tolerance: float) -> list[Path]:
    """Chain paths whose endpoints nearly touch into longer strokes.

    Fewer pen lifts means faster, cleaner drawings - this is the single most
    effective optimisation for traced line art.
    """
    if tolerance <= 0:
        return list(paths)

    items = [np.asarray(p, dtype=np.float64) for p in paths if len(p) >= 2]
    dots = [np.asarray(p, dtype=np.float64) for p in paths if len(p) == 1]
    if not items:
        return dots

    index = _EndpointIndex(max(tolerance, 1e-3))
    for i, p in enumerate(items):
        index.add((i, 0), p[0])
        index.add((i, 1), p[-1])

    used = [False] * len(items)
    result: list[Path] = []

    for i in range(len(items)):
        if used[i]:
            continue
        used[i] = True
        index.remove((i, 0))
        index.remove((i, 1))
        chain = items[i]

        # grow forward, then flip and grow again so both ends get extended
        for _ in range(2):
            while True:
                key, dist = index.nearest(chain[-1], tolerance)
                if key is None or dist > tolerance:
                    break
                j, end = key
                index.remove((j, 0))
                index.remove((j, 1))
                used[j] = True
                other = items[j] if end == 0 else items[j][::-1]
                chain = np.vstack([chain, other[1:]])
            chain = chain[::-1]

        result.append(chain)

    result.extend(dots)
    return result


def reorder_paths(
    paths: Sequence[Path],
    start: tuple[float, float] = (0.0, 0.0),
    allow_reverse: bool = True,
) -> list[Path]:
    """Greedy nearest-neighbour ordering to cut down travel moves."""
    items = [np.asarray(p, dtype=np.float64) for p in paths if len(p)]
    if len(items) < 3:
        return items

    # Sizing the grid from the bounding box lets a single far-away speck
    # collapse every real path into one bucket, turning the nearest-neighbour
    # search into a linear scan (measured 430x slower).  The typical *spread*
    # of the endpoints is a far more robust scale.
    starts = np.array([p[0] for p in items], dtype=np.float64)
    middle = np.median(starts, axis=0)
    spread = np.median(np.abs(starts - middle), axis=0).max() * 4.0
    if not np.isfinite(spread) or spread <= 1e-6:
        bb = bounds(items)
        spread = max(bb[2] - bb[0], bb[3] - bb[1], 1.0) if bb else 1.0
    cell = max(float(spread) / max(math.sqrt(len(items)), 1.0), 0.5)

    index = _EndpointIndex(cell)
    for i, p in enumerate(items):
        index.add((i, 0), p[0])
        if allow_reverse and len(p) > 1:
            index.add((i, 1), p[-1])

    ordered: list[Path] = []
    cursor = np.asarray(start, dtype=np.float64)
    remaining = len(items)
    while remaining:
        key, _ = index.nearest(cursor)
        if key is None:
            break
        i, end = key
        index.remove((i, 0))
        index.remove((i, 1))
        remaining -= 1
        p = items[i] if end == 0 else items[i][::-1]
        ordered.append(p)
        cursor = p[-1]

    if len(ordered) != len(items):  # safety net, should not happen
        seen = {id(p) for p in ordered}
        ordered.extend(p for p in items if id(p) not in seen)
    return ordered


def travel_length(paths: Sequence[Path], start: tuple[float, float] = (0.0, 0.0)) -> float:
    cursor = np.asarray(start, dtype=np.float64)
    total = 0.0
    for p in paths:
        if len(p) == 0:
            continue
        total += float(math.hypot(p[0][0] - cursor[0], p[0][1] - cursor[1]))
        cursor = p[-1]
    return total


# --------------------------------------------------------------------------
# misc generators
# --------------------------------------------------------------------------
def resample(path: Path, max_segment: float) -> Path:
    """Insert points so no segment is longer than *max_segment*."""
    if len(path) < 2 or max_segment <= 0 or path.shape[1] > 2:
        return path
    out = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        n = int(d // max_segment)
        if n > 0:
            ts = np.linspace(0.0, 1.0, n + 2)[1:-1]
            for t in ts:
                out.append(a + (b - a) * t)
        out.append(b)
    return np.asarray(out, dtype=np.float64)


def circle_path(cx: float, cy: float, radius: float, segments: int = 12) -> Path:
    a = np.linspace(0.0, 2 * math.pi, segments + 1)
    return np.stack([cx + radius * np.cos(a), cy + radius * np.sin(a)], axis=1)


# --------------------------------------------------------------------------
# fewer pen lifts, shorter tour
# --------------------------------------------------------------------------
def stitch_paths(paths: Sequence[Path], max_gap: float) -> list[Path]:
    """Join consecutive strokes that end close together into one long stroke.

    On this machine a pen lift costs far more than the ink it saves: the Z
    leadscrew is slow, and lifting plus coming back down is most of a second.
    A hatched drawing has thousands of them - measured at 52 to 79 % of the
    whole job.  Where the next stroke starts only a millimetre or two away -
    exactly what boustrophedon hatching produces - drawing straight through is
    far faster, and the connection runs along the edge of the shaded area where
    it reads as part of the texture.

    Must run *after* ordering, when consecutive strokes are already neighbours.
    Dots are never stitched: there the connection would become the drawing.
    """
    if max_gap <= 0:
        return list(paths)

    out: list[Path] = []
    current: Path | None = None
    for path in paths:
        if len(path) == 0:
            continue
        if len(path) == 1:
            if current is not None:
                out.append(current)
                current = None
            out.append(path)
            continue
        if current is None:
            current = path
            continue
        gap = math.hypot(path[0][0] - current[-1][0], path[0][1] - current[-1][1])
        if gap <= max_gap:
            current = np.vstack([current, path])
        else:
            out.append(current)
            current = path
    if current is not None:
        out.append(current)
    return out


def improve_tour(
    paths: Sequence[Path],
    start: tuple[float, float] = (0.0, 0.0),
    *,
    time_budget: float = 1.0,
) -> list[Path]:
    """Or-opt pass over an ordered list of strokes.

    Greedy nearest-neighbour leaves long "go back for the one I missed" edges.
    Taking a single stroke out and re-inserting it next to a spatial neighbour
    removes most of them, and unlike 2-opt it never has to reverse a run, so
    each move is O(1) on a linked list.  Only nearby candidates are considered,
    and the whole thing runs under a time budget.
    """
    import time as _time

    items = [np.asarray(p, dtype=np.float64) for p in paths if len(p)]
    count = len(items)
    if count < 8 or time_budget <= 0:
        return items

    deadline = _time.perf_counter() + time_budget
    heads = np.array([p[0] for p in items], dtype=np.float64)
    tails = np.array([p[-1] for p in items], dtype=np.float64)

    # doubly linked list over tour positions, in the given order
    nxt = list(range(1, count)) + [-1]
    prv = [-1] + list(range(count - 1))
    first = 0

    def hop(a: int, b: int) -> float:
        return float(math.hypot(tails[a][0] - heads[b][0], tails[a][1] - heads[b][1]))

    origin = np.asarray(start, dtype=np.float64)

    def hop_from_origin(b: int) -> float:
        return float(math.hypot(origin[0] - heads[b][0], origin[1] - heads[b][1]))

    # candidates: strokes whose head is near this stroke's tail
    spread = float(np.median(np.abs(heads - np.median(heads, axis=0))).max()) * 4.0
    if not math.isfinite(spread) or spread <= 1e-6:
        spread = 100.0
    index = _EndpointIndex(max(spread / max(math.sqrt(count), 1.0), 0.5))
    for i in range(count):
        index.add(i, heads[i])

    moved = 0
    for i in range(count):
        if _time.perf_counter() > deadline:
            break
        before_i, after_i = prv[i], nxt[i]
        if after_i == -1:
            continue

        # cost of leaving i where it is
        old = (hop(before_i, i) if before_i != -1 else hop_from_origin(i)) + hop(i, after_i)
        bridge = hop(before_i, after_i) if before_i != -1 else hop_from_origin(after_i)
        removal_gain = old - bridge
        if removal_gain <= 1e-9:
            continue

        target, _distance = index.nearest(tails[i])
        if target is None or target == i or target == before_i:
            continue
        target_next = nxt[target]
        if target_next == i:
            continue

        insert_cost = hop(target, i) + (hop(i, target_next) if target_next != -1 else 0.0)
        insert_cost -= hop(target, target_next) if target_next != -1 else 0.0
        if insert_cost >= removal_gain - 1e-9:
            continue

        # unlink i
        if before_i != -1:
            nxt[before_i] = after_i
        else:
            first = after_i
        prv[after_i] = before_i
        # relink after target
        nxt[i] = target_next
        prv[i] = target
        nxt[target] = i
        if target_next != -1:
            prv[target_next] = i
        moved += 1

    if not moved:
        return items

    out: list[Path] = []
    node = first
    seen = 0
    while node != -1 and seen <= count:
        out.append(items[node])
        node = nxt[node]
        seen += 1
    return out if len(out) == count else items
