"""Image loading and pre-processing.

Images travel through the app as float32 arrays in the 0..1 range, either
single channel (grayscale) or HxWx3 RGB.  0 = black, 1 = white/paper.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image, ImageOps

__all__ = ["load_rgb", "to_gray", "prepare", "ink", "to_qimage_bytes", "subject_weight", "enhance_subject"]

MAX_WORK_PIXELS = 4_000_000


def load_rgb(file_path: str) -> np.ndarray:
    """Load any Pillow-readable image as float32 RGB (0..1).

    Transparency is composited onto white, so transparent = paper.
    """
    with Image.open(file_path) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, rgba).convert("RGB")
        else:
            img = img.convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0

    if arr.shape[0] * arr.shape[1] > MAX_WORK_PIXELS:
        scale = math.sqrt(MAX_WORK_PIXELS / (arr.shape[0] * arr.shape[1]))
        arr = cv2.resize(arr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(arr)


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return np.ascontiguousarray(
        0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
    ).astype(np.float32)


def resize_long_edge(img: np.ndarray, long_edge: int) -> np.ndarray:
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest == 0:
        return img
    scale = long_edge / float(longest)
    if abs(scale - 1.0) < 0.01:
        return img
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(img, (max(int(round(w * scale)), 8), max(int(round(h * scale)), 8)), interpolation=interp)


def prepare(
    img: np.ndarray,
    *,
    detail: int = 900,
    brightness: float = 0.0,
    contrast: float = 0.0,
    gamma: float = 1.0,
    blur: float = 0.0,
    invert: bool = False,
    auto_levels: bool = False,
    black_point: float = 0.0,
    white_point: float = 1.0,
    saturation: float = 1.0,
) -> np.ndarray:
    """Shared pre-processing chain; works on grayscale or RGB.

    brightness/contrast are -100..100 sliders, gamma 0.2..3.0, blur is in
    pixels of the working image, black/white point are 0..1 level clamps.
    """
    out = resize_long_edge(np.asarray(img, dtype=np.float32), max(int(detail), 64))

    if auto_levels:
        lum = to_gray(out)
        lo = float(np.percentile(lum, 1.0))
        hi = float(np.percentile(lum, 99.0))
        if hi - lo > 1e-3:
            out = (out - lo) / (hi - lo)

    if black_point > 0.0 or white_point < 1.0:
        lo = min(black_point, white_point - 1e-3)
        hi = max(white_point, lo + 1e-3)
        out = (out - lo) / (hi - lo)

    out = np.clip(out, 0.0, 1.0)

    if abs(brightness) > 1e-6:
        out = out + brightness / 100.0
    if abs(contrast) > 1e-6:
        factor = (100.0 + contrast) / 100.0
        factor = factor * factor if contrast > 0 else factor
        out = (out - 0.5) * factor + 0.5
    out = np.clip(out, 0.0, 1.0)

    if abs(gamma - 1.0) > 1e-3:
        out = np.power(out, max(gamma, 0.05))

    if out.ndim == 3 and abs(saturation - 1.0) > 1e-3:
        lum = to_gray(out)[:, :, None]
        out = np.clip(lum + (out - lum) * saturation, 0.0, 1.0)

    if blur > 0.05:
        k = int(max(3, round(blur * 3) | 1))
        out = cv2.GaussianBlur(out, (k, k), blur)

    if invert:
        out = 1.0 - out

    return np.clip(out, 0.0, 1.0).astype(np.float32)


def subject_weight(gray: np.ndarray) -> np.ndarray:
    """Where the picture is worth spending ink, 0..1.

    A snapshot of a person is mostly not the person: there is a wall, a window,
    a jumper.  Hatching all of that at the same density is what makes a plot
    look like wallpaper with a face in it.

    Two cheap signals agree surprisingly well on where the subject is, and
    neither needs a face detector (OpenCV 5 no longer ships the cascades):

    *sharpness* - a camera focuses on the subject, so the subject carries the
    high-frequency energy;
    *position* - people put the subject near the middle, a little above centre.
    """
    detail = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 4.0))
    # Spread the detail over the region that *contains* it, not just the pixels
    # that carry it.  A face is mostly smooth - the sharpness lives in the hair,
    # the eyes and the edge of the cheek - so a tight map scores the cheek as
    # background and the background lift bleaches the face away.
    reach = max(min(gray.shape) * 0.10, 8.0)
    detail = cv2.GaussianBlur(detail, (0, 0), reach)
    detail = np.clip(detail / max(float(np.percentile(detail, 97)), 1e-6), 0.0, 1.0)

    height, width = gray.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    radius = np.sqrt(
        ((xx - width / 2.0) / (width * 0.60)) ** 2
        + ((yy - height * 0.46) / (height * 0.60)) ** 2
    )
    centre = np.clip(1.30 - radius, 0.0, 1.0)

    weight = np.clip(0.40 * centre + 0.90 * detail, 0.0, 1.0)
    weight = cv2.GaussianBlur(weight, (0, 0), max(min(height, width) * 0.02, 2.0))
    span = float(weight.max() - weight.min())
    return (weight - float(weight.min())) / max(span, 1e-6)


def enhance_subject(rgb: np.ndarray, local: float = 0.7, background: float = 0.55) -> np.ndarray:
    """Local contrast on the subject, and let the background fall back to paper.

    `local` drives CLAHE, which is what brings a flat-lit face back: a global
    curve cannot lift the modelling in the cheek without also crushing the
    shadow under the chin.  `background` lifts everything the subject map does
    not care about towards blank paper.
    """
    gray = to_gray(rgb)
    if local > 0.0:
        u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=1.2 + 2.8 * float(local), tileGridSize=(8, 8))
        boosted = clahe.apply(u8).astype(np.float32) / 255.0
        gray = gray * (1.0 - local) + boosted * local

    if background > 0.0:
        weight = subject_weight(gray)
        gray = gray + float(background) * (1.0 - weight) * (1.0 - gray)

    gray = np.clip(gray, 0.0, 1.0)
    return np.dstack([gray, gray, gray]).astype(np.float32)


def ink(img: np.ndarray) -> np.ndarray:
    """Darkness map: 0 = paper, 1 = full ink."""
    return 1.0 - to_gray(img)


def to_qimage_bytes(img: np.ndarray) -> tuple[bytes, int, int]:
    """uint8 RGB buffer for cheap thumbnail display."""
    rgb = img if img.ndim == 3 else np.repeat(img[:, :, None], 3, axis=2)
    buf = np.ascontiguousarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8))
    return buf.tobytes(), buf.shape[1], buf.shape[0]
