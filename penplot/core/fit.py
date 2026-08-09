"""Tune a technique's parameters until the drawing looks like the picture.

Every technique has knobs, and most of them interact: raising the ink coverage
of a crosshatch changes what the line spacing should be, and both change how
long the job takes.  Asking a person to find that corner by dragging sliders is
asking them to solve a small optimisation problem by hand, which is exactly the
sort of thing they should not have to do.

So the app solves it.  It renders the technique, rasterises the result the way
the pen will actually lay ink down, and scores it against the picture:

*tone*      - the drawing blurred to the distance the eye reads it from should
              have the same light and dark as the photograph blurred the same
              way.  This is what makes a drawing "look like" its subject.
*structure* - the edges that survive should be the edges that were there.  Tone
              alone will happily smear a face into an even grey of the right
              average.
*the face*  - when a face was found in the photograph, the error inside it
              counts for several times as much as the error in the wall behind
              it.  A drawing is judged the way it is looked at, and a portrait
              is looked at in the eyes.

              (Running the detector on the *drawing* and requiring it to still
              find a face was the obvious idea and it does not work: measured
              across three techniques at three resolutions, YuNet found a face
              in none of the nine renderings, so the test answers "no" for every
              candidate and steers nothing.  A check that cannot fail either way
              is not a check.)
*cost*      - a drawing that takes six hours is not a better answer than one
              that takes forty minutes and looks nearly the same.

The search is a coordinate pattern search: cheap, deterministic, and it stops
when it stops improving.  A technique has two or three parameters that actually
matter, and they are the ones it moves.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import geometry as geo, raster, techniques

__all__ = ["FitResult", "TUNABLE", "score_drawing", "fit_technique"]


#: The parameters worth searching, per technique, most important first.  Taken
#: from a sensitivity sweep: these are the ones that move the result, and
#: nothing is gained by searching a knob whose effect is under a percent.
TUNABLE: dict[str, tuple[str, ...]] = {
    "crosshatch": ("coverage", "layers", "dither"),
    "sketch": ("sensitivity", "min_length"),
    "contours": ("levels", "smoothing"),
    "dashes": ("spacing", "dash"),
    "stipple": ("pitch", "weight"),
    "dots": ("dark_spacing", "light_spacing", "curve"),
    "dwell": ("pitch",),
    "halftone": ("pitch", "angle"),
    "flow": ("spacing", "max_length"),
    "scribble": ("straightness", "ink_use"),
    "spiral": ("pitch", "amplitude"),
    "rings": ("pitch", "amplitude"),
    "waves": ("spacing", "amplitude"),
    "hilbert": ("depth", "threshold"),
    "mesh": ("density",),
    "silhouette": ("threshold", "fill"),
    "tsp": ("points",),
    "voronoi": ("points", "relax"),
    "packing": ("min_radius", "max_radius"),
    "maze": ("cell",),
    "mosaic": ("cell", "size"),
    "crosscontour": ("spacing", "layers", "step"),
}


@dataclass
class FitResult:
    params: dict = field(default_factory=dict)
    score: float = 0.0
    tone_error: float = 0.0
    structure: float = 0.0
    minutes: float = 0.0
    face_kept: bool | None = None
    rounds: int = 0
    tried: int = 0
    seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"tone within {self.tone_error * 100:.1f}%", f"structure {self.structure * 100:.0f}%"]
        parts.append(f"{self.minutes:.0f} min")
        return f"Tried {self.tried} settings: " + ", ".join(parts)


def _rasterise(paths, shape: tuple[int, int], pen_px: float) -> np.ndarray:
    """Ink on paper, as the pen will actually leave it."""
    canvas = np.zeros(shape, dtype=np.uint8)
    width = max(int(round(pen_px)), 1)
    for path in paths:
        points = np.asarray(path)[:, :2]
        if len(points) == 1:
            cv2.circle(canvas, (int(points[0][0]), int(points[0][1])), max(width // 2, 1), 255, -1)
        elif len(points) > 1:
            cv2.polylines(canvas, [points.astype(np.int32)], False, 255, width, cv2.LINE_AA)
    return canvas.astype(np.float32) / 255.0


def score_drawing(
    ink_drawn: np.ndarray, ink_target: np.ndarray, weight: np.ndarray | None = None
) -> tuple[float, float]:
    """(tone error, structure agreement) - both 0..1, lower and higher are better."""
    blur = max(int(min(ink_target.shape) * 0.02) | 1, 3)
    a = cv2.GaussianBlur(ink_drawn, (blur, blur), 0)
    b = cv2.GaussianBlur(ink_target, (blur, blur), 0)
    # a pen cannot lay down more ink than solid black, so compare shapes, not
    # absolute levels: fit the one free gain the drawing really has
    gain = float((a * b).sum() / max((a * a).sum(), 1e-9))
    gain = float(np.clip(gain, 0.2, 5.0))
    squared = (a * gain - b) ** 2
    if weight is not None:
        tone = float(np.sqrt((squared * weight).sum() / max(weight.sum(), 1e-9)))
    else:
        tone = float(np.sqrt(squared.mean()))

    fine = max(int(min(ink_target.shape) * 0.006) | 1, 3)
    ea = cv2.GaussianBlur(np.abs(cv2.Sobel(a, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(a, cv2.CV_32F, 0, 1)), (fine, fine), 0)
    eb = cv2.GaussianBlur(np.abs(cv2.Sobel(b, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(b, cv2.CV_32F, 0, 1)), (fine, fine), 0)
    denominator = math.sqrt(float((ea * ea).sum()) * float((eb * eb).sum()))
    structure = float((ea * eb).sum() / denominator) if denominator > 1e-9 else 0.0
    return tone, structure


def face_weight_map(shape: tuple[int, int], faces) -> np.ndarray | None:
    """Weight the error by where a person actually looks: 5x inside the face."""
    if not faces:
        return None
    height, width = shape
    weight = np.ones((height, width), dtype=np.float32)
    for face in faces:
        x0 = int(max(face.x - face.width * 0.15, 0.0) * width)
        x1 = int(min(face.x + face.width * 1.15, 1.0) * width)
        y0 = int(max(face.y - face.height * 0.20, 0.0) * height)
        y1 = int(min(face.y + face.height * 1.20, 1.0) * height)
        if x1 > x0 and y1 > y0:
            weight[y0:y1, x0:x1] = 5.0
    return cv2.GaussianBlur(weight, (0, 0), max(min(height, width) * 0.01, 2.0))


def fit_technique(
    gray: np.ndarray,
    key: str,
    params: dict,
    ctx: techniques.Context,
    *,
    target_minutes: float | None = None,
    minutes_of: callable | None = None,
    faces=(),
    budget_seconds: float = 8.0,
    should_cancel=None,
) -> FitResult:
    """Search this technique's parameters for the closest drawing to `gray`.

    The search runs on a smaller copy of the picture.  Judging tone and
    structure does not need full resolution - they are measured through a blur
    anyway - and a quarter of the pixels buys four times as many candidates,
    which matters far more than the last decimal of the score.
    """
    started = time.perf_counter()
    technique = techniques.REGISTRY.get(key)
    if technique is None:
        return FitResult(params=dict(params))

    definitions = {p.key: p for p in technique.params}
    knobs = [k for k in TUNABLE.get(key, ()) if k in definitions and definitions[k].kind not in ("bool", "choice", "text")]
    if not knobs:
        return FitResult(params=dict(params), notes=["nothing worth searching for this technique"])

    full_long = max(gray.shape)
    # The copy has to be big enough that the pen is still several pixels wide,
    # or every fine spacing rasterises to the same picture and the search is
    # optimising a proxy that cannot see the thing it is changing.  Measured:
    # at 260 px a 0.5 mm pen is one pixel, and the "best" dot spacing found
    # there was 11 % *worse* than the default at full size.
    wanted = int(math.ceil(2.6 / max(ctx.pen_px / full_long, 1e-9)))
    search_long = int(min(full_long, max(wanted, 320)))
    if search_long < full_long:
        scale = search_long / float(full_long)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ctx = techniques.Context(
            px_per_mm=ctx.px_per_mm * search_long / full_long,
            pen_width=ctx.pen_width,
            scale_with_pen=ctx.scale_with_pen,
        )
    target = np.clip(1.0 - gray, 0.0, 1.0)
    shape = gray.shape
    weight = face_weight_map(shape, faces)
    tried = {"n": 0}

    def evaluate(candidate: dict) -> tuple[float, dict]:
        if should_cancel is not None and should_cancel():
            raise techniques.Cancelled()
        tried["n"] += 1
        paths = techniques.render(key, gray, candidate, ctx)
        drawn = _rasterise(paths, shape, ctx.pen_px)
        tone, structure = score_drawing(drawn, target, weight)
        minutes = 0.0
        if minutes_of is not None:
            minutes = float(minutes_of(paths))
        penalty = 0.0
        if target_minutes and minutes > target_minutes:
            # Squared, and not capped: a linear, capped penalty let the search
            # buy a 70 % tonal gain for seven hours of plotting, which is not a
            # trade any person would make.  At twice the budget this costs 0.3,
            # at four times it costs 2.7 - more than the whole tonal term.
            over = minutes / max(target_minutes, 1.0) - 1.0
            penalty = 0.3 * over * over
        cost = tone * 2.2 - structure * 0.8 + penalty
        return cost, {
            "tone": tone, "structure": structure, "minutes": minutes,
            "paths": paths, "drawn": drawn, "penalty": penalty,
        }

    current = dict(params)
    for knob in knobs:
        current.setdefault(knob, definitions[knob].default)
    best_cost, best_extra = evaluate(current)
    best = dict(current)
    start_tone, start_structure = best_extra["tone"], best_extra["structure"]

    rounds = 0
    steps = {k: 0.42 for k in knobs}          # fraction of the range, halved as it settles
    while time.perf_counter() - started < budget_seconds and rounds < 6:
        rounds += 1
        improved = False
        for knob in knobs:
            definition = definitions[knob]
            span = definition.maximum - definition.minimum
            step = span * steps[knob]
            for direction in (1.0, -1.0):
                if time.perf_counter() - started >= budget_seconds:
                    break
                value = best[knob] + direction * step
                value = float(np.clip(value, definition.minimum, definition.maximum))
                if definition.decimals == 0:
                    value = float(round(value))
                if abs(value - best[knob]) < 1e-9:
                    continue
                candidate = dict(best)
                candidate[knob] = value
                try:
                    cost, extra = evaluate(candidate)
                except techniques.Cancelled:
                    raise
                except Exception:
                    continue
                if cost < best_cost - 1e-4:
                    best_cost, best_extra, best = cost, extra, candidate
                    improved = True
                    break
            steps[knob] *= 0.62 if not improved else 0.85
        if not improved:
            break

    notes: list[str] = []
    if weight is not None:
        notes.append("weighted towards the face")
    if target_minutes and best_extra["minutes"] > target_minutes * 1.25:
        notes.append(f"could not reach {target_minutes:.0f} min without losing the picture")

    if best_extra["tone"] > start_tone + 1e-4 and target_minutes:
        notes.append(f"traded some likeness to fit {target_minutes:.0f} min")

    return FitResult(
        params={k: best[k] for k in knobs},
        score=best_cost,
        tone_error=best_extra["tone"],
        structure=best_extra["structure"],
        minutes=best_extra["minutes"],
        face_kept=None,
        rounds=rounds,
        tried=tried["n"],
        seconds=time.perf_counter() - started,
        notes=notes,
    )
