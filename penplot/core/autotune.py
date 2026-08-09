"""Automatic setup: look at the picture, then choose settings that suit it.

Two things go wrong most often when people set this up by hand:

* the tonal range is wrong, so the drawing comes out either washed out or
  solid black;
* the line density is wrong, so the plot either looks empty or takes nine hours.

This module fixes both.  The tonal part is measured straight from the
histogram.  The density part is closed-loop: it renders the drawing, measures
how long it would actually take, and adjusts until it lands on target.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import raster, techniques
from .drawing import SourceResult
from .pens import PenLibrary
from .settings import AppSettings

__all__ = ["ImageStats", "analyse", "tune_levels", "rank_techniques", "auto_tune", "TuneResult"]

#: How to make a technique draw more or less.  Each entry is an ordered list of
#: knobs; the first is the fine control and the rest are the coarse ones the
#: tuner escalates to when the fine one runs out of range.  The flag is True
#: when *raising* the value produces *less* ink - true of every spacing.
DENSITY_KEYS: dict[str, list[tuple[str, bool]]] = {
    "crosshatch": [("coverage", False), ("layers", False)],
    "dashes": [("spacing", True), ("gap", True)],
    "stipple": [("pitch", True)],
    "dwell": [("pitch", True)],
    "halftone": [("cell", True), ("rings", False)],
    "flow": [("spacing", True), ("max_length", False)],
    "scribble": [("fade", True), ("step", True)],
    "spiral": [("pitch", True), ("frequency", False)],
    "rings": [("pitch", True), ("frequency", False)],
    "waves": [("spacing", True), ("frequency", False)],
    "mesh": [("pitch", True)],
    "contours": [("levels", False)],
    "hilbert": [("threshold", True), ("depth", False)],
    "sketch": [("sensitivity", False), ("passes", False)],
    "silhouette": [("fill", False)],
    "tsp": [("spacing", True)],
    "voronoi": [("spacing", True)],
    "packing": [("min_radius", True), ("max_radius", True)],
    "maze": [("cell", True)],
    "mosaic": [("cell", True)],
    "crosscontour": [("spacing", True), ("layers", False)],
}


@dataclass
class ImageStats:
    mean_ink: float = 0.0          # 0 = blank paper, 1 = solid black
    p05: float = 0.0               # 5th percentile of ink
    p50: float = 0.0
    p95: float = 0.0
    contrast: float = 0.0          # standard deviation of the ink
    subject_p50: float = 0.0       # median ink of the parts that are not blank paper
    subject_share: float = 0.0     # how much of the frame the sharp, central part fills
    edge_density: float = 0.0      # fraction of pixels on an edge
    detail: float = 0.0            # relative amount of fine detail
    flatness: float = 0.0          # 1 = flat graphic art, 0 = continuous tone
    has_colour: bool = False
    distinct_colours: int = 0

    @property
    def is_flat_art(self) -> bool:
        return self.flatness > 0.55

    @property
    def is_line_art(self) -> bool:
        return self.is_flat_art and self.mean_ink < 0.22


def analyse(rgb: np.ndarray) -> ImageStats:
    """Measure the things that decide which settings will look good."""
    small = raster.resize_long_edge(np.asarray(rgb, dtype=np.float32), 420)
    gray = raster.to_gray(small)
    ink = np.clip(1.0 - gray, 0.0, 1.0)

    stats = ImageStats()
    stats.mean_ink = float(ink.mean())
    stats.p05, stats.p50, stats.p95 = (float(v) for v in np.percentile(ink, [5, 50, 95]))
    stats.contrast = float(ink.std())
    # The midtone of the *subject*, not of the frame.  A picture on white paper
    # has a whole-frame median of nearly zero, and aiming the gamma at that
    # crushes everything that is actually drawn into a black mass.
    subject_ink = ink[ink > 0.04]
    stats.subject_p50 = float(np.median(subject_ink)) if subject_ink.size >= 64 else stats.p50

    u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(u8, (0, 0), 1.2)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    peak = float(np.percentile(magnitude, 99.5)) or 1.0
    edges = cv2.Canny(blurred, peak * 0.25, peak * 0.6, L2gradient=True)
    stats.edge_density = float((edges > 0).mean())

    # fine detail = how much energy survives a high-pass filter
    low = cv2.GaussianBlur(gray, (0, 0), 3.0)
    stats.detail = float(np.abs(gray - low).mean() * 6.0)

    # Flat art has very few distinct levels *within the subject*.  Measuring
    # the whole frame would just report "mostly white paper" for any photo on a
    # light background, so the blank areas are excluded first.
    subject = ink[ink > 0.04]
    if subject.size < 64:
        stats.flatness = 1.0
    else:
        histogram = np.bincount((subject * 31).astype(np.int32), minlength=32).astype(np.float64)
        histogram /= max(histogram.sum(), 1.0)
        occupied = int((histogram > 0.01).sum())
        biggest = float(np.sort(histogram)[-3:].sum())
        stats.flatness = float(np.clip(biggest * 0.55 + (1.0 - occupied / 32.0) * 0.75, 0.0, 1.0))

    # How much of the frame is "the subject" - sharp and near the middle.  A
    # portrait sits between a fifth and two thirds; a landscape or a flat
    # graphic does not concentrate like that.
    try:
        weight = raster.subject_weight(gray)
        stats.subject_share = float((weight > 0.55).mean())
    except Exception:  # pragma: no cover - defensive
        stats.subject_share = 0.0

    if small.ndim == 3:
        channel_spread = float(np.abs(small - raster.to_gray(small)[:, :, None]).mean())
        stats.has_colour = channel_spread > 0.03
        reduced = (small * 8).astype(np.int32)
        packed = reduced[:, :, 0] * 81 + reduced[:, :, 1] * 9 + reduced[:, :, 2]
        stats.distinct_colours = int(len(np.unique(packed)))
    return stats


def tune_levels(stats: ImageStats) -> dict:
    """Pick the tonal adjustments so the pen has a full range to work with.

    The target is simple and specific to drawing rather than screens: the
    lightest few percent must end up as blank paper, and the darkest few
    percent must reach roughly 90 % ink - never a solid slab, which just
    shreds the paper and wastes an hour.
    """
    values = {
        "auto_levels": False,
        "brightness": 0.0,
        "contrast": 0.0,
        "gamma": 1.0,
        "black_point": 0.0,
        "white_point": 1.0,
        "saturation": 1.0,
    }

    # stretch whatever range the picture actually uses onto 0..1
    light = 1.0 - stats.p95     # brightest ink -> paper
    dark = 1.0 - stats.p05
    if dark - light > 0.08:
        values["black_point"] = float(max(light - 0.02, 0.0))
        values["white_point"] = float(min(dark + 0.02, 1.0))

    # Aim the midtone at 0.30 of ink after the stretch.  Paper is not a screen:
    # a drawing wants its light half to stay blank so the darks read as darks -
    # at 0.45 an ordinary photograph comes out as an even mesh with no subject.
    span = max(stats.p95 - stats.p05, 1e-3)
    mid = float(np.clip((stats.subject_p50 - stats.p05) / span, 0.02, 0.98))
    target_mid = 0.30
    gamma = math.log(max(target_mid, 1e-3)) / math.log(max(mid, 1e-3))
    values["gamma"] = float(np.clip(1.0 / gamma if gamma > 1e-3 else 1.0, 0.45, 2.2))

    if stats.contrast < 0.12:
        values["contrast"] = float(np.clip((0.16 - stats.contrast) * 320.0, 0.0, 45.0))
    if stats.is_line_art:
        # A logo has two tones and no midtone to aim at; leave it alone and just
        # make the edges crisp.  A soft photograph also scores as "flat", which
        # is why this tests for line art rather than flatness.
        values["gamma"] = 1.0
        values["contrast"] = max(values["contrast"], 10.0)
    return values


def rank_techniques(stats: ImageStats) -> list[tuple[str, float, str]]:
    """Score every technique for this picture.  Highest score first."""
    scores: list[tuple[str, float, str]] = []

    tone_range = float(np.clip(stats.p95 - stats.p05, 0.0, 1.0))
    edges = float(np.clip(stats.edge_density * 12.0, 0.0, 1.0))
    detail = float(np.clip(stats.detail, 0.0, 1.0))
    flat = stats.flatness
    weight = float(np.clip(stats.mean_ink * 2.2, 0.0, 1.0))

    def add(key: str, score: float, reason: str) -> None:
        scores.append((key, float(np.clip(score, 0.0, 1.0)), reason))

    add("sketch", 0.35 + edges * 0.5 + flat * 0.2 - weight * 0.2,
        "clear outlines to follow")
    add("silhouette", 0.15 + flat * 0.75 - detail * 0.3,
        "large flat areas with hard edges")
    add("contours", 0.3 + tone_range * 0.45 + (1.0 - detail) * 0.25 - flat * 0.3,
        "smooth tonal gradients")
    add("crosshatch", 0.45 + tone_range * 0.4 + weight * 0.2 - flat * 0.25,
        "a full range of midtones to shade")
    add("dashes", 0.3 + tone_range * 0.3 - flat * 0.2, "soft shading")
    add("stipple", 0.3 + tone_range * 0.35 + detail * 0.2 - flat * 0.25,
        "fine tonal detail")
    add("dwell", 0.25 + tone_range * 0.35 - flat * 0.2,
        "even dots with tone from dwell time")
    add("halftone", 0.3 + tone_range * 0.3 + flat * 0.15, "an even tonal grid")
    add("flow", 0.35 + detail * 0.35 + tone_range * 0.3 - flat * 0.2,
        "shapes with a clear direction to follow")
    add("scribble", 0.2 + weight * 0.4 + tone_range * 0.2 - flat * 0.2,
        "dense dark areas to fill")
    add("spiral", 0.25 + tone_range * 0.25, "any picture, as a graphic poster")
    add("rings", 0.2 + tone_range * 0.25, "any picture, as a graphic poster")
    add("waves", 0.2 + tone_range * 0.25, "any picture, as a graphic poster")
    add("hilbert", 0.2 + tone_range * 0.3 - flat * 0.15, "one continuous line")
    add("mesh", 0.15 + flat * 0.2 + tone_range * 0.2, "a low-poly look")
    add("tsp", 0.3 + tone_range * 0.35 + weight * 0.2 - flat * 0.2,
        "a subject that reads as one continuous line")
    add("voronoi", 0.2 + tone_range * 0.3 - flat * 0.1, "an organic cell texture")
    add("packing", 0.25 + tone_range * 0.35 - flat * 0.15, "tone from circle size")
    add("maze", 0.2 + flat * 0.25 + edges * 0.2, "bold shapes to fill with a maze")
    add("mosaic", 0.15 + tone_range * 0.3 - flat * 0.25, "a typographic screen")
    add("crosscontour", 0.3 + detail * 0.3 + tone_range * 0.35 - flat * 0.25,
        "rounded forms to wrap engraving lines around")

    scores.sort(key=lambda item: -item[1])
    return scores


@dataclass
class TuneResult:
    technique: str = ""
    levels: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    minutes: float = 0.0
    iterations: int = 0
    notes: list[str] = field(default_factory=list)


def _quick_job(settings: AppSettings, source: SourceResult, library: PenLibrary, detail: int):
    from . import pipeline

    trial = copy.deepcopy(settings)
    trial.style.detail = detail
    return pipeline.build_plot(source, trial, trial.library)


def auto_tune(
    settings: AppSettings,
    source: SourceResult,
    library: PenLibrary,
    *,
    target_minutes: float = 25.0,
    choose_technique: bool = False,
    adjust_levels: bool = True,
    detail: int = 420,
    max_iterations: int = 8,
    time_budget: float = 9.0,
) -> TuneResult:
    """Set the levels, pick the density, and land near *target_minutes*.

    Mutates `settings` in place and returns what it decided.
    """
    result = TuneResult()
    if source is None or not source.is_raster or source.rgb is None:
        result.notes.append("Auto-tune needs a picture.")
        return result

    stats = analyse(source.rgb)

    if adjust_levels:
        levels = tune_levels(stats)
        for key, value in levels.items():
            setattr(settings.style, key, value)
        result.levels = levels

    if choose_technique:
        ranked = rank_techniques(stats)
        settings.style.technique = ranked[0][0]
        result.notes.append(
            f"Chose {techniques.REGISTRY[ranked[0][0]].label} - {ranked[0][2]}."
        )
    result.technique = settings.style.technique

    knobs = DENSITY_KEYS.get(settings.style.technique) or []
    values = settings.style.technique_params()
    technique = techniques.REGISTRY[settings.style.technique]
    by_key = {p.key: p for p in technique.params}
    knobs = [(k, flag) for k, flag in knobs if k in by_key]
    if not knobs:
        job = _quick_job(settings, source, library, settings.style.detail)
        result.minutes = job.stats.estimated_seconds / 60.0
        result.params = dict(values)
        return result

    target_seconds = max(target_minutes, 0.5) * 60.0

    # heavy techniques get a coarser search so the button still feels instant
    if technique.slow:
        detail = min(detail, 320)
    started = time.perf_counter()

    def out_of_time() -> bool:
        return time.perf_counter() - started > time_budget

    def read(key: str) -> float:
        param = by_key[key]
        return float(values.get(key, param.default))

    def write(key: str, value: float) -> float:
        param = by_key[key]
        value = float(np.clip(value, param.minimum, param.maximum))
        if param.decimals == 0:
            value = float(int(round(value)))
        values[key] = value if param.decimals else int(round(value))
        return value

    def at_limit(key: str, want_more_ink: bool) -> bool:
        param = by_key[key]
        _, raise_to_reduce = next(k for k in knobs if k[0] == key)
        # more ink means moving towards the minimum for a spacing knob
        edge = param.minimum if (raise_to_reduce == want_more_ink) else param.maximum
        return abs(read(key) - edge) < 1e-6

    def nudge(key: str, ratio: float, coarse: bool) -> bool:
        """Move one knob towards the target. Returns False if it could not move."""
        param = by_key[key]
        _, raise_to_reduce = next(k for k in knobs if k[0] == key)
        current = read(key)
        step = ratio if raise_to_reduce else 1.0 / ratio
        # damped so a noisy technique cannot oscillate between two extremes
        step = float(np.clip(step, 0.3, 3.2)) ** (0.8 if not coarse else 0.45)

        span = max(param.maximum - param.minimum, 1e-6)
        if abs(current) < 1e-6 or (param.decimals == 0 and abs(current) < 0.5):
            want_more = ratio < 1.0
            seed = param.minimum + span * (0.35 if raise_to_reduce == want_more else 0.15)
            proposed = seed
        else:
            proposed = current * step
        if param.decimals == 0:
            rounded = int(round(float(np.clip(proposed, param.minimum, param.maximum))))
            if rounded == int(round(current)):
                rounded = int(round(current)) + (1 if proposed > current else -1)
            proposed = rounded
        written = write(key, proposed)
        return abs(written - current) > 1e-9

    def measure(detail_px: int) -> float:
        job = _quick_job(settings, source, library, detail_px)
        return job.stats.estimated_seconds

    # --- coarse search at reduced resolution -----------------------------
    for iteration in range(max_iterations):
        if out_of_time():
            break
        seconds = measure(detail)
        result.iterations = iteration + 1
        result.minutes = seconds / 60.0
        if seconds <= 1.0:
            break
        ratio = seconds / target_seconds
        if 0.82 <= ratio <= 1.22:
            break
        want_more = ratio < 1.0
        moved = False
        for index, (key, _flag) in enumerate(knobs):
            if at_limit(key, want_more):
                continue
            moved = nudge(key, ratio, coarse=index > 0)
            if moved and index > 0:
                # a coarse knob changes everything, so re-centre the fine one
                fine = by_key[knobs[0][0]]
                write(knobs[0][0], (fine.minimum + fine.maximum) * 0.4)
            if moved:
                break
        if not moved:
            break

    # --- confirm at the real working resolution --------------------------
    seconds = measure(settings.style.detail)
    result.minutes = seconds / 60.0
    for _ in range(2):
        if out_of_time():
            break
        ratio = seconds / target_seconds
        if 0.87 <= ratio <= 1.15:
            break
        want_more = ratio < 1.0
        moved = False
        for index, (key, _flag) in enumerate(knobs):
            if at_limit(key, want_more):
                continue
            moved = nudge(key, ratio, coarse=index > 0)
            if moved:
                break
        if not moved:
            break
        seconds = measure(settings.style.detail)
        result.minutes = seconds / 60.0
        result.iterations += 1

    result.params = dict(values)
    off = result.minutes / max(target_minutes, 0.01)
    if off > 1.25 or off < 0.75:
        result.notes.append(
            f"{technique.label} bottoms out at about {result.minutes:.0f} min for this "
            f"picture rather than {target_minutes:.0f} - change the size, the pen, or the technique."
        )
    return result
