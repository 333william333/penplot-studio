"""Colour separation: one ink map per pen.

The result of every mode is a list of "ink maps" - float32 images where 0 means
"leave the paper alone" and 1 means "this pen at full coverage".  Because an
ink map looks exactly like an inverted grayscale image, every drawing style
(hatch, stipple, spiral, ...) works unchanged on a colour separation.
"""

from __future__ import annotations

import cv2
import numpy as np

from .pens import Pen, PenLibrary, rgb_to_hex, srgb_to_lab
from .raster import to_gray

__all__ = ["separate", "suggest_palette", "SEPARATION_MODES"]

SEPARATION_MODES = {
    "mono": "Single pen (grayscale)",
    "palette": "Match my pen colours",
    "cmyk": "CMYK separation",
}

_IDEAL_CMYK = {
    "C": (0.0, 0.62, 0.88),
    "M": (0.90, 0.0, 0.50),
    "Y": (0.95, 0.76, 0.0),
    "K": (0.10, 0.10, 0.10),
}


def _lab_image(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.float32), cv2.COLOR_RGB2Lab)


def suggest_palette(rgb: np.ndarray, count: int, paper_lightness: float = 90.0) -> list[str]:
    """K-means the image (ignoring paper-white) into *count* pen colours."""
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[:, :, None], 3, axis=2)
    small = cv2.resize(rgb, (160, max(1, int(160 * rgb.shape[0] / max(rgb.shape[1], 1)))), interpolation=cv2.INTER_AREA)
    lab = _lab_image(small).reshape(-1, 3)
    flat = small.reshape(-1, 3)
    keep = lab[:, 0] < paper_lightness
    if keep.sum() < count * 4:
        keep = np.ones(len(lab), dtype=bool)
    samples = lab[keep].astype(np.float32)

    count = max(1, min(int(count), 12, len(samples)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, centers = cv2.kmeans(samples, count, None, criteria, 4, cv2.KMEANS_PP_CENTERS)

    colours = []
    src = flat[keep]
    labels = labels.ravel()
    for i in range(count):
        members = src[labels == i]
        if len(members) == 0:
            continue
        mean = members.mean(axis=0)
        colours.append((srgb_to_lab(mean)[0], rgb_to_hex(mean)))
    colours.sort(key=lambda item: -item[0])  # light pens first, black last
    return [hexcolour for _, hexcolour in colours]


def _coverage_from_lightness(lightness: np.ndarray, pen_lightness: float, paper_lightness: float) -> np.ndarray:
    """Linear tint ramp: paper-light -> 0 ink, pen-dark -> full ink."""
    span = max(paper_lightness - pen_lightness, 6.0)
    return np.clip((paper_lightness - lightness) / span, 0.0, 1.0)


def separate(
    rgb: np.ndarray,
    library: PenLibrary,
    *,
    mode: str = "mono",
    paper_lightness: float = 96.0,
    ink_gamma: float = 1.0,
    mono_pen: int = 0,
    min_coverage: float = 0.03,
) -> list[np.ndarray | None]:
    """Return one ink map per pen in *library* (None where the pen is unused)."""
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[:, :, None], 3, axis=2)
    height, width = rgb.shape[:2]
    layers: list[np.ndarray | None] = [None] * len(library)

    def finish(ink_map: np.ndarray) -> np.ndarray:
        out = np.clip(ink_map, 0.0, 1.0)
        if abs(ink_gamma - 1.0) > 1e-3:
            out = np.power(out, max(ink_gamma, 0.05))
        out[out < min_coverage] = 0.0
        return out.astype(np.float32)

    enabled = [i for i, pen in enumerate(library) if pen.enabled]
    if mode == "mono" or len(enabled) <= 1:
        # fall back to the first *enabled* pen: drawing everything onto a pen
        # the user switched off means nothing comes out at all
        index = mono_pen if mono_pen in enabled else (enabled[0] if enabled else 0)
        layers[index] = finish(1.0 - to_gray(rgb))
        return layers

    if mode == "cmyk":
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        k = 1.0 - np.max(rgb, axis=2)
        denominator = np.maximum(1.0 - k, 1e-4)
        c = (1.0 - r - k) / denominator
        m = (1.0 - g - k) / denominator
        y = (1.0 - b - k) / denominator
        # K carries almost all of the tonal detail, so it claims its pen first;
        # assigning in CMYK order used to hand the black pen to magenta and
        # throw the key channel away whenever fewer than four pens were loaded
        channels = {"K": k, "C": c, "M": m, "Y": y}
        used: set[int] = set()
        for key, channel in channels.items():
            ideal = _IDEAL_CMYK[key]
            ideal_lab = np.asarray(srgb_to_lab(ideal), dtype=np.float32)
            best, best_distance = None, 1e9
            for i, pen in enumerate(library):
                if i in used or not pen.enabled:
                    continue
                distance = float(np.linalg.norm(np.asarray(pen.lab, dtype=np.float32) - ideal_lab))
                if distance < best_distance:
                    best, best_distance = i, distance
            if best is None:
                continue
            used.add(best)
            layers[best] = finish(np.clip(channel, 0.0, 1.0))
        return layers

    # --- palette matching -------------------------------------------------
    lab = _lab_image(rgb)
    lightness = lab[:, :, 0]
    candidates = [i for i, pen in enumerate(library) if pen.enabled]
    if not candidates:
        return layers

    pen_labs = np.asarray([library[i].lab for i in candidates], dtype=np.float32)
    # paper is a candidate too, so light pixels simply get no ink at all
    paper_lab = np.asarray([[100.0, 0.0, 0.0]], dtype=np.float32)
    all_labs = np.vstack([pen_labs, paper_lab])

    flat = lab.reshape(-1, 3)
    distances = np.empty((len(flat), len(all_labs)), dtype=np.float32)
    for i, ref in enumerate(all_labs):
        delta = flat - ref
        # weight chroma a little higher so hue wins over brightness
        distances[:, i] = np.sqrt(delta[:, 0] ** 2 * 0.6 + delta[:, 1] ** 2 + delta[:, 2] ** 2)
    assignment = np.argmin(distances, axis=1).reshape(height, width)

    for slot, pen_index in enumerate(candidates):
        mask = assignment == slot
        if not mask.any():
            continue
        pen = library[pen_index]
        coverage = np.zeros((height, width), dtype=np.float32)
        coverage[mask] = _coverage_from_lightness(lightness[mask], pen.lightness, paper_lightness)
        layers[pen_index] = finish(coverage)
    return layers
