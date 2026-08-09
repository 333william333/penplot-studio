"""Looks: what the app should do with a picture, decided from the picture.

A technique is a rendering algorithm.  A *look* is an answer to "what kind of
picture is this, and what would a person want out of it" - which technique, at
which settings, with which preparation.  There are twenty-one techniques and
nobody should have to audition them.

The rule the rest of the app follows: the look is chosen automatically when a
file is loaded, it says in one line what it decided, and the user can overrule
it with a single control.  Every parameter a look sets is a normal setting
afterwards, so overruling one value does not throw the rest away.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["Look", "LOOKS", "look_labels", "choose", "apply_look", "describe"]


@dataclass(frozen=True)
class Look:
    key: str
    label: str
    blurb: str                       # what it does, in the user's words
    technique: str
    params: dict = field(default_factory=dict)
    #: local contrast and background fade, see raster.enhance_subject
    enhance: str = ""
    #: tonal targets applied on top of what the picture asks for
    tone: dict = field(default_factory=dict)


LOOKS: dict[str, Look] = {}


def _add(look: Look) -> None:
    LOOKS[look.key] = look


_add(Look(
    "portrait", "Portrait",
    "For people. Lifts the modelling in the face, lets the background fade to "
    "paper, and hatches along the form instead of straight across it.",
    technique="crosscontour",
    params={"spacing": 1.35, "levels": 3, "smooth": 5.0, "min_length": 1.4, "tone": 0.92},
    enhance="subject",
))

_add(Look(
    "photo", "Photograph",
    "Even crosshatch for a general picture: tone from the density of the lines.",
    technique="crosshatch",
    params={"layers": 4, "coverage": 0.85, "dither": 0.12, "min_length": 0.6},
    enhance="subject-light",
))

_add(Look(
    "drawing", "Line drawing",
    "Traces the lines that are already there. For logos, plans, diagrams and "
    "anything drawn rather than photographed.",
    technique="sketch",
    params={"sensitivity": 55.0, "min_length": 1.0, "wobble": 0.0, "passes": 1},
    tone={"contrast": 12.0},
))

_add(Look(
    "sketch", "Loose sketch",
    "The same lines with a shaking hand and overshot corners, so it reads as "
    "drawn rather than printed.",
    technique="sketch",
    params={"sensitivity": 45.0, "wobble": 0.35, "overshoot": 1.2, "passes": 2},
))

_add(Look(
    "dots", "Dots only",
    "No lines at all: the pen taps the paper and lifts. Tone comes from how "
    "close together the dots are.",
    technique="dots",
    params={"dark_spacing": 0.6, "light_spacing": 2.6, "curve": 1.0, "dot_size": 0.0},
    enhance="subject-light",
))

_add(Look(
    "engrave", "Engraving",
    "Long flowing strokes that follow the picture, like a banknote engraving.",
    technique="flow",
    # a photograph carries grain, and a flow field will happily trace every
    # speck of it; smoothing first is what makes the strokes follow the picture
    params={"spacing": 1.1, "max_length": 40.0, "coherence": 0.9},
    enhance="subject-light",
    tone={"blur": 1.2},
))


def look_labels() -> dict[str, str]:
    return {key: look.label for key, look in LOOKS.items()}


def choose(stats, reading=None) -> str:
    """Pick a look from what the picture actually is.

    Only two answers are decided automatically, because only two can be decided
    honestly.  Line art versus photograph is a measurement.  "Is this a person"
    is not: OpenCV 5 ships no face cascade, and sharpness-plus-centre fires just
    as happily on a bowl of fruit.  Guessing wrong there costs an hour of
    plotting, so Portrait is offered rather than assumed - and because the
    photograph look already carries a gentler version of the same subject
    enhancement, being wrong is never expensive.
    """
    if getattr(stats, "is_line_art", False):
        return "drawing"
    # A detector actually found a face: that is knowledge, not a guess, so the
    # portrait look is chosen rather than offered.
    if reading is not None and reading.knows_it_is_a_person:
        return "portrait"
    return "photo"


def suggestion(stats, reading=None) -> str:
    """A nudge, not a decision - shown next to the chosen look.

    Only the classical engine ever nudges.  When a face detector has run, its
    answer is the answer: no face means no face, and a nudge on top of that
    would just be the app second-guessing its own evidence.
    """
    if reading is not None and reading.backend == "neural":
        return ""
    subject = getattr(stats, "subject_share", 0.0)
    if 0.10 <= subject <= 0.72 and stats.contrast < 0.32:
        return "portrait"
    return ""


def apply_look(key: str, style, stats=None) -> Look | None:
    """Write a look's decisions into the style settings."""
    look = LOOKS.get(key)
    if look is None:
        return None
    style.technique = look.technique
    style.enhance = look.enhance
    style.technique_params(look.technique).update(look.params)
    for field_name, value in look.tone.items():
        setattr(style, field_name, value)
    return look


def describe(key: str, stats=None, reading=None) -> str:
    """One line telling the user what was decided and why."""
    look = LOOKS.get(key)
    if look is None:
        return ""
    if stats is None:
        return look.blurb
    why = []
    if key == "portrait":
        if reading is not None and reading.knows_it_is_a_person:
            why.append(reading.summary().lower())
        else:
            why.append(f"subject fills {getattr(stats, 'subject_share', 0.0) * 100:.0f}% of the frame")
    if key == "drawing":
        why.append("flat colours and clean edges")
    if stats.contrast < 0.12:
        why.append("low contrast, so the tone was stretched")
    reason = f" ({', '.join(why)})" if why else ""
    return f"{look.label}{reason}. {look.blurb}"
