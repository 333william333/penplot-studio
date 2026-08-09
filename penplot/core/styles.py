"""Image -> pen path conversion ("drawing styles").

Every function takes a prepared float32 image (0 = black, 1 = white) and
returns a list of polylines in *pixel* coordinates with Y pointing down.  The
layout stage scales and flips them onto the bed later, so nothing here needs
to know about millimetres.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from . import geometry as geo

__all__ = [
    "edges_paths",
    "contour_paths",
    "hatch_paths",
    "stipple_paths",
    "spiral_paths",
    "wave_paths",
    "hatch_polygons",
    "hatch_mask",
    "trace_binary",
    "mask_from_paths",
    "sample_bilinear",
    "split_by_mask",
]

_NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


# --------------------------------------------------------------------------
# binary skeleton tracing (used by the edge style)
# --------------------------------------------------------------------------
def trace_binary(mask: np.ndarray, prefer_straight: bool = True) -> list[np.ndarray]:
    """Follow every 1-pixel-wide run in *mask* and return it as a polyline.

    cv2.findContours would walk *around* a thin line and come back, drawing
    everything twice.  Walking the pixels directly gives single strokes, which
    is what a pen wants.
    """
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if not m.any():
        return []

    h, w = m.shape
    pw = w + 2
    padded = np.zeros((h + 2, pw), dtype=np.uint8)
    padded[1:-1, 1:-1] = m

    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    counts = cv2.filter2D(padded, cv2.CV_16S, kernel, borderType=cv2.BORDER_CONSTANT)
    counts = counts * padded

    on = padded.ravel().copy()
    counts_flat = counts.ravel()
    offsets = [dy * pw + dx for dy, dx in _NEIGHBOURS]
    diag = [abs(dy) + abs(dx) == 2 for dy, dx in _NEIGHBOURS]
    dirs = [(float(dx), float(dy)) for dy, dx in _NEIGHBOURS]

    ends = np.flatnonzero((counts_flat == 1) & (on > 0))
    junctions = np.flatnonzero((counts_flat >= 3) & (on > 0))
    seeds = list(ends) + list(junctions)

    paths: list[np.ndarray] = []

    def walk(start: int) -> None:
        on[start] = 0
        pts = [start]
        prev_dir = None
        cur = start
        while True:
            best = -1
            best_score = 1e9
            best_dir = None
            for oi, off in enumerate(offsets):
                n = cur + off
                if not on[n]:
                    continue
                dx, dy = dirs[oi]
                score = 0.25 if diag[oi] else 0.0
                if prefer_straight and prev_dir is not None:
                    dot = prev_dir[0] * dx + prev_dir[1] * dy
                    norm = math.hypot(dx, dy) * math.hypot(prev_dir[0], prev_dir[1])
                    score += 1.0 - (dot / norm if norm else 0.0)
                if score < best_score:
                    best_score = score
                    best = n
                    best_dir = (dx, dy)
            if best < 0:
                break
            on[best] = 0
            pts.append(best)
            prev_dir = best_dir
            cur = best
        if len(pts) >= 2:
            idx = np.asarray(pts, dtype=np.int64)
            ys, xs = np.divmod(idx, pw)
            paths.append(np.stack([xs - 1 + 0.5, ys - 1 + 0.5], axis=1).astype(np.float64))

    for s in seeds:
        if on[s]:
            walk(int(s))

    # whatever is left are closed loops without endpoints
    leftovers = np.flatnonzero(on > 0)
    for s in leftovers:
        if on[s]:
            walk(int(s))

    return paths


# --------------------------------------------------------------------------
# masks & hatching
# --------------------------------------------------------------------------
def _rotate_mask(mask: np.ndarray, angle_deg: float):
    """Rotate a mask so hatch lines become image rows.  Returns (rot, inverse)."""
    h, w = mask.shape
    diag = int(math.ceil(math.hypot(w, h))) + 4
    cx, cy = w / 2.0, h / 2.0
    matrix = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    matrix[0, 2] += diag / 2.0 - cx
    matrix[1, 2] += diag / 2.0 - cy
    rot = cv2.warpAffine(
        mask.astype(np.uint8),
        matrix,
        (diag, diag),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )
    inverse = cv2.invertAffineTransform(matrix)
    return rot, inverse


def _segments_from_rows(rot: np.ndarray, spacing: float, phase: float = 0.0) -> list[tuple[float, float, float]]:
    """Scan every *spacing*-th row and return (y, x_start, x_end) runs."""
    out = []
    h, w = rot.shape
    y = phase % max(spacing, 1.0)
    while y < h:
        row = rot[int(y)]
        if row.any():
            padded = np.concatenate(([0], (row > 0).astype(np.int8), [0]))
            change = np.diff(padded)
            starts = np.flatnonzero(change == 1)
            stops = np.flatnonzero(change == -1)
            for x0, x1 in zip(starts, stops):
                # a run of a single pixel would give x_start == x_end: a pen
                # down/up cycle that draws nothing but still costs a second
                if x1 - x0 >= 2:
                    out.append((float(y), float(x0), float(x1 - 1)))
        y += spacing
    return out


def _map_back(points: np.ndarray, inverse: np.ndarray) -> np.ndarray:
    xy = np.empty((len(points), 3), dtype=np.float64)
    xy[:, :2] = points
    xy[:, 2] = 1.0
    return xy @ inverse.T


def hatch_mask(
    mask: np.ndarray,
    spacing: float,
    angle_deg: float,
    *,
    phase: float = 0.0,
    boustrophedon: bool = True,
) -> list[np.ndarray]:
    """Fill a binary mask with parallel lines at *angle_deg*."""
    if not mask.any():
        return []
    spacing = max(spacing, 0.7)
    rot, inverse = _rotate_mask(mask, angle_deg)
    runs = _segments_from_rows(rot, spacing, phase)
    if not runs:
        return []

    paths = []
    flip = False
    last_y = None
    for y, x0, x1 in runs:
        if last_y is not None and y != last_y:
            flip = not flip
        last_y = y
        pts = np.array([[x0, y], [x1, y]], dtype=np.float64)
        if boustrophedon and flip:
            pts = pts[::-1]
        paths.append(_map_back(pts, inverse))
    return paths


def mask_from_paths(paths, width: int, height: int, scale: float = 1.0) -> np.ndarray:
    """Rasterise closed polygons with the even-odd rule (holes stay empty)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    tmp = np.zeros_like(mask)
    for p in paths:
        if len(p) < 3:
            continue
        tmp[:] = 0
        poly = np.round(np.asarray(p, dtype=np.float64) * scale).astype(np.int32)
        cv2.fillPoly(tmp, [poly], 1)
        mask ^= tmp
    return mask


def hatch_polygons(
    polys,
    spacing: float,
    angle_deg: float,
    *,
    boustrophedon: bool = True,
) -> list[np.ndarray]:
    """Exact vector scanline fill of closed polygons using the even-odd rule.

    Used for filled text and filled silhouettes, where rasterising first would
    lose crispness at small sizes.
    """
    edges = []
    ang = math.radians(-angle_deg)
    ca, sa = math.cos(ang), math.sin(ang)
    rot = np.array([[ca, -sa], [sa, ca]], dtype=np.float64)
    inv = rot.T

    for p in polys:
        p = np.asarray(p, dtype=np.float64)
        if len(p) < 3:
            continue
        q = p @ rot.T
        if not np.allclose(q[0], q[-1]):
            q = np.vstack([q, q[:1]])
        edges.append(q)
    if not edges:
        return []

    allpts = np.vstack(edges)
    y_min, y_max = float(allpts[:, 1].min()), float(allpts[:, 1].max())
    x0s, y0s, x1s, y1s = [], [], [], []
    for q in edges:
        x0s.append(q[:-1, 0])
        y0s.append(q[:-1, 1])
        x1s.append(q[1:, 0])
        y1s.append(q[1:, 1])
    x0 = np.concatenate(x0s)
    y0 = np.concatenate(y0s)
    x1 = np.concatenate(x1s)
    y1 = np.concatenate(y1s)

    spacing = max(spacing, 1e-3)
    n_lines = int((y_max - y_min) / spacing)
    if n_lines <= 0:
        return []

    out: list[np.ndarray] = []
    flip = False
    for i in range(n_lines + 1):
        y = y_min + spacing * (i + 0.5)
        if y > y_max:
            break
        crossing = ((y0 <= y) & (y1 > y)) | ((y1 <= y) & (y0 > y))
        if not crossing.any():
            continue
        ys0 = y0[crossing]
        ys1 = y1[crossing]
        xs0 = x0[crossing]
        xs1 = x1[crossing]
        t = (y - ys0) / (ys1 - ys0)
        xs = np.sort(xs0 + t * (xs1 - xs0))
        pairs = xs.reshape(-1, 2) if len(xs) % 2 == 0 else xs[: len(xs) - 1].reshape(-1, 2)
        segs = []
        for a, b in pairs:
            if b - a < 1e-9:
                continue
            segs.append(np.array([[a, y], [b, y]], dtype=np.float64))
        if boustrophedon and flip:
            segs = [s[::-1] for s in reversed(segs)]
        flip = not flip
        out.extend(s @ inv.T for s in segs)
    return out


# --------------------------------------------------------------------------
# styles
# --------------------------------------------------------------------------
def edges_paths(
    img: np.ndarray,
    *,
    low: float = 60.0,
    high: float = 150.0,
    aperture: int = 3,
    thicken: int = 0,
    min_length_px: float = 3.0,
) -> list[np.ndarray]:
    """Canny edge detection followed by stroke tracing."""
    u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    edges = cv2.Canny(u8, float(low), float(high), apertureSize=int(aperture) | 1, L2gradient=True)
    if thicken > 0:
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=int(thicken))
        edges = cv2.ximgproc.thinning(edges) if hasattr(cv2, "ximgproc") else _thin(edges)
    paths = trace_binary(edges)
    return [p for p in paths if geo.path_length(p) >= min_length_px]


def _thin(binary: np.ndarray) -> np.ndarray:
    """Zhang-Suen style thinning fallback when opencv-contrib is unavailable."""
    img = (binary > 0).astype(np.uint8)
    prev = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    skel = np.zeros_like(img)
    work = img.copy()
    for _ in range(64):
        eroded = cv2.erode(work, element)
        opened = cv2.dilate(eroded, element)
        skel |= work - opened
        work = eroded
        if not work.any():
            break
        if np.array_equal(prev, work):
            break
        prev = work.copy()
    return (skel > 0).astype(np.uint8) * 255


def _binarise(
    img: np.ndarray,
    method: str,
    threshold: float,
    block: int,
    offset: float,
) -> np.ndarray:
    u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    if method == "otsu":
        _, binary = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif method == "adaptive":
        block = max(3, int(block) | 1)
        binary = cv2.adaptiveThreshold(
            u8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, float(offset)
        )
    else:  # manual
        _, binary = cv2.threshold(u8, float(threshold * 255.0), 255, cv2.THRESH_BINARY_INV)
    return binary


def contour_paths(
    img: np.ndarray,
    *,
    method: str = "otsu",
    threshold: float = 0.5,
    block: int = 25,
    offset: float = 8.0,
    despeckle: int = 2,
    min_area_px: float = 12.0,
    smooth: float = 0.6,
    fill_spacing: float = 0.0,
    fill_angle: float = 45.0,
) -> list[np.ndarray]:
    """Trace the outline of dark regions; optionally hatch-fill them too."""
    binary = _binarise(img, method, threshold, block, offset)
    if despeckle > 0:
        k = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=int(despeckle))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=int(despeckle))

    contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    paths: list[np.ndarray] = []
    for c in contours:
        if cv2.contourArea(c) < min_area_px:
            continue
        pts = c.reshape(-1, 2).astype(np.float64)
        if len(pts) < 3:
            continue
        if smooth > 0:
            pts = geo.rdp(pts, smooth)
        if len(pts) < 3:
            continue
        pts = np.vstack([pts, pts[:1]])
        paths.append(pts)

    if fill_spacing > 0:
        paths.extend(hatch_mask(binary > 0, fill_spacing, fill_angle))
    return paths


def hatch_paths(
    img: np.ndarray,
    *,
    levels: int = 4,
    spacing: float = 4.0,
    angle: float = 45.0,
    angle_step: float = 45.0,
    outline: bool = False,
    min_length_px: float = 2.0,
) -> list[np.ndarray]:
    """Tonal shading: each darkness level adds another layer of hatching."""
    levels = max(1, int(levels))
    darkness = 1.0 - img
    paths: list[np.ndarray] = []
    for i in range(levels):
        limit = (i + 1) / (levels + 1.0)
        mask = (darkness >= limit).astype(np.uint8)
        if not mask.any():
            continue
        layer_angle = angle + angle_step * i
        segs = hatch_mask(mask, spacing, layer_angle, phase=(i * spacing) / levels)
        paths.extend(s for s in segs if geo.path_length(s) >= min_length_px)
        if outline and i == 0:
            contours, _ = cv2.findContours(mask * 255, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            for c in contours:
                if cv2.contourArea(c) < 20:
                    continue
                pts = geo.rdp(c.reshape(-1, 2).astype(np.float64), 0.8)
                if len(pts) >= 3:
                    paths.append(np.vstack([pts, pts[:1]]))
    return paths


def stipple_paths(
    img: np.ndarray,
    *,
    spacing: float = 4.0,
    jitter: float = 0.35,
    gamma: float = 1.0,
    min_ink: float = 0.06,
    seed: int = 12345,
) -> list[np.ndarray]:
    """Error-diffusion stippling.  Returns single-point paths (dots)."""
    spacing = max(float(spacing), 1.0)
    h, w = img.shape
    gw = max(int(round(w / spacing)), 2)
    gh = max(int(round(h / spacing)), 2)
    small = cv2.resize(img, (gw, gh), interpolation=cv2.INTER_AREA)
    darkness = np.clip(1.0 - small, 0.0, 1.0)
    if abs(gamma - 1.0) > 1e-3:
        darkness = np.power(darkness, max(gamma, 0.05))
    darkness[darkness < min_ink] = 0.0

    # Floyd-Steinberg error diffusion over the coarse grid
    work = darkness.astype(np.float32).copy()
    dots = np.zeros((gh, gw), dtype=bool)
    for y in range(gh):
        for x in range(gw):
            old = work[y, x]
            new = 1.0 if old > 0.5 else 0.0
            dots[y, x] = new > 0.5
            err = old - new
            if x + 1 < gw:
                work[y, x + 1] += err * 7 / 16
            if y + 1 < gh:
                if x > 0:
                    work[y + 1, x - 1] += err * 3 / 16
                work[y + 1, x] += err * 5 / 16
                if x + 1 < gw:
                    work[y + 1, x + 1] += err * 1 / 16

    ys, xs = np.nonzero(dots)
    if len(xs) == 0:
        return []
    rng = np.random.default_rng(seed)
    sx = w / gw
    sy = h / gh
    px = (xs + 0.5) * sx
    py = (ys + 0.5) * sy
    if jitter > 0:
        px = px + rng.uniform(-jitter, jitter, len(px)) * sx
        py = py + rng.uniform(-jitter, jitter, len(py)) * sy
    px = np.clip(px, 0, w - 1)
    py = np.clip(py, 0, h - 1)
    return [np.array([[x, y]], dtype=np.float64) for x, y in zip(px, py)]


def sample_bilinear(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    h, w = img.shape
    x = np.clip(xs, 0, w - 1.001)
    y = np.clip(ys, 0, h - 1.001)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = x - x0
    fy = y - y0
    top = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
    bot = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
    return top * (1 - fy) + bot * fy


def spiral_paths(
    img: np.ndarray,
    *,
    pitch: float = 6.0,
    amplitude: float = 2.4,
    frequency: float = 1.0,
    center: tuple[float, float] | None = None,
    gamma: float = 1.0,
    min_ink: float = 0.0,
) -> list[np.ndarray]:
    """One continuous spiral whose radius wobbles where the image is dark."""
    h, w = img.shape
    cx, cy = center if center else (w / 2.0, h / 2.0)
    corners = [(0, 0), (w, 0), (0, h), (w, h)]
    r_max = max(math.hypot(x - cx, y - cy) for x, y in corners)
    pitch = max(float(pitch), 1.0)

    turns = r_max / pitch
    if turns < 0.2:
        return []
    # constant-ish arc-length stepping: more samples further out
    total_theta = turns * 2 * math.pi
    step_arc = max(pitch / 10.0, 0.6)
    thetas = [0.0]
    theta = 0.0
    while theta < total_theta:
        r = pitch * theta / (2 * math.pi)
        dtheta = step_arc / max(r, pitch / 6.0)
        theta += dtheta
        thetas.append(theta)
    theta_arr = np.asarray(thetas, dtype=np.float64)
    radius = pitch * theta_arr / (2 * math.pi)

    xs = cx + radius * np.cos(theta_arr)
    ys = cy + radius * np.sin(theta_arr)
    darkness = np.clip(1.0 - sample_bilinear(img, xs, ys), 0.0, 1.0)
    if abs(gamma - 1.0) > 1e-3:
        darkness = np.power(darkness, max(gamma, 0.05))

    arc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])
    wavelength = max(pitch / max(frequency, 0.05), 1.0)
    wobble = amplitude * darkness * np.sin(2 * math.pi * arc / wavelength)
    r2 = radius + wobble
    xs2 = cx + r2 * np.cos(theta_arr)
    ys2 = cy + r2 * np.sin(theta_arr)

    inside = (xs2 >= 0) & (xs2 <= w - 1) & (ys2 >= 0) & (ys2 <= h - 1)
    if min_ink > 0:
        inside &= darkness > min_ink
    return split_by_mask(xs2, ys2, inside)


def wave_paths(
    img: np.ndarray,
    *,
    spacing: float = 5.0,
    amplitude: float = 2.0,
    frequency: float = 1.0,
    angle: float = 0.0,
    gamma: float = 1.0,
) -> list[np.ndarray]:
    """Parallel squiggle lines whose amplitude follows image darkness."""
    h, w = img.shape
    spacing = max(float(spacing), 1.0)
    diag = math.hypot(w, h)
    ang = math.radians(angle)
    ca, sa = math.cos(ang), math.sin(ang)
    cx, cy = w / 2.0, h / 2.0

    n_lines = int(diag / spacing) + 1
    step = max(spacing / 6.0, 0.5)
    n_pts = int(diag / step) + 2
    t = (np.arange(n_pts) * step) - diag / 2.0
    wavelength = max(spacing * 2.0 / max(frequency, 0.05), 1.0)

    out: list[np.ndarray] = []
    for i in range(n_lines):
        offset = (i - n_lines / 2.0) * spacing
        base_x = cx + t * ca - offset * sa
        base_y = cy + t * sa + offset * ca
        darkness = np.clip(1.0 - sample_bilinear(img, base_x, base_y), 0.0, 1.0)
        if abs(gamma - 1.0) > 1e-3:
            darkness = np.power(darkness, max(gamma, 0.05))
        wob = amplitude * darkness * np.sin(2 * math.pi * t / wavelength)
        xs = base_x - wob * sa
        ys = base_y + wob * ca
        if i % 2 == 1:
            xs = xs[::-1]
            ys = ys[::-1]
            darkness = darkness[::-1]
        inside = (xs >= 0) & (xs <= w - 1) & (ys >= 0) & (ys <= h - 1) & (darkness > 0.02)
        out.extend(split_by_mask(xs, ys, inside))
    return out


def split_by_mask(xs: np.ndarray, ys: np.ndarray, keep: np.ndarray) -> list[np.ndarray]:
    """Cut a sampled curve into runs where *keep* is True."""
    out: list[np.ndarray] = []
    if not keep.any():
        return out
    idx = np.flatnonzero(keep)
    splits = np.flatnonzero(np.diff(idx) > 1)
    start = 0
    for s in list(splits) + [len(idx) - 1]:
        seg = idx[start : s + 1]
        start = s + 1
        if len(seg) >= 2:
            out.append(np.stack([xs[seg], ys[seg]], axis=1))
    return out
