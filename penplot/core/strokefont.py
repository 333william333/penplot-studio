"""A built-in single-stroke ("engraving") font.

Regular TrueType fonts describe the *outline* of a letter, so a pen traces the
edge and leaves the middle empty - fine for big display text, bad for small
text.  This font describes the *centre line* of every stroke instead, which is
what a pen actually wants to follow.

Coordinate system (Y up):
    baseline        y = 0
    x-height        y = 66
    cap height      y = 100
    ascender        y = 105
    descender       y = -30
"""

from __future__ import annotations

import math

import numpy as np

CAP = 100.0
X_HEIGHT = 66.0
ASCENDER = 108.0
DESCENDER = -30.0
DEFAULT_LINE_HEIGHT = 150.0

Stroke = list


def _pts(*points) -> np.ndarray:
    return np.asarray(points, dtype=np.float64)


def _arc(cx: float, cy: float, rx: float, ry: float, a0: float, a1: float, steps: int | None = None) -> np.ndarray:
    """Elliptical arc from a0 to a1 (degrees, CCW positive, Y up)."""
    sweep = abs(a1 - a0)
    if steps is None:
        steps = max(4, int(sweep / 12) + 2)
    a = np.radians(np.linspace(a0, a1, steps))
    return np.stack([cx + rx * np.cos(a), cy + ry * np.sin(a)], axis=1)


def _ellipse(cx: float, cy: float, rx: float, ry: float, steps: int = 32) -> np.ndarray:
    return _arc(cx, cy, rx, ry, 0.0, 360.0, steps)


def _dot(cx: float, cy: float, r: float = 3.2) -> np.ndarray:
    return _ellipse(cx, cy, r, r, 9)


def _join(*parts) -> np.ndarray:
    """Concatenate arcs/point lists into one continuous stroke."""
    chunks = [np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in parts if len(p)]
    return np.vstack(chunks) if chunks else np.zeros((0, 2))


# --------------------------------------------------------------------------
# glyph table: char -> (advance width, [strokes])
# --------------------------------------------------------------------------
def _build() -> dict[str, tuple[float, list[np.ndarray]]]:
    g: dict[str, tuple[float, list[np.ndarray]]] = {}

    # ---- uppercase -------------------------------------------------------
    g["A"] = (72, [_pts((4, 0), (36, 100), (68, 0)), _pts((16, 32), (56, 32))])
    g["B"] = (
        72,
        [
            _pts((10, 0), (10, 100)),
            _join(_pts((10, 100), (42, 100)), _arc(42, 75, 24, 25, 90, -90), _pts((10, 50))),
            _join(_pts((10, 50), (44, 50)), _arc(44, 25, 26, 25, 90, -90), _pts((10, 0))),
        ],
    )
    g["C"] = (72, [_arc(38, 50, 32, 50, 50, 310)])
    g["D"] = (74, [_join(_pts((10, 0), (10, 100), (38, 100)), _arc(38, 50, 32, 50, 90, -90), _pts((10, 0)))])
    g["E"] = (64, [_pts((58, 100), (10, 100), (10, 0), (58, 0)), _pts((10, 52), (46, 52))])
    g["F"] = (62, [_pts((58, 100), (10, 100), (10, 0)), _pts((10, 52), (46, 52))])
    # The bowl must sweep the long way round, through 180 degrees, exactly like
    # C.  Ending at -55 took the 95 degree short cut and left a bare hook.
    g["G"] = (78, [_join(_arc(38, 50, 32, 50, 40, 305), _pts((70, 9), (70, 46), (46, 46)))])
    g["H"] = (74, [_pts((10, 0), (10, 100)), _pts((64, 0), (64, 100)), _pts((10, 52), (64, 52))])
    g["I"] = (32, [_pts((16, 0), (16, 100))])
    g["J"] = (58, [_join(_pts((46, 100), (46, 26)), _arc(24, 26, 22, 26, 0, -180))])
    g["K"] = (70, [_pts((10, 0), (10, 100)), _pts((64, 100), (10, 44)), _pts((28, 60), (66, 0))])
    g["L"] = (58, [_pts((10, 100), (10, 0), (54, 0))])
    g["M"] = (86, [_pts((10, 0), (10, 100), (43, 40), (76, 100), (76, 0))])
    g["N"] = (76, [_pts((10, 0), (10, 100), (66, 4), (66, 100))])
    g["O"] = (80, [_ellipse(40, 50, 33, 50)])
    g["P"] = (70, [_pts((10, 0), (10, 100)), _join(_pts((10, 100), (40, 100)), _arc(40, 74, 26, 26, 90, -90), _pts((10, 48)))])
    g["Q"] = (80, [_ellipse(40, 50, 33, 50), _pts((50, 22), (76, -8))])
    g["R"] = (
        72,
        [
            _pts((10, 0), (10, 100)),
            _join(_pts((10, 100), (40, 100)), _arc(40, 74, 26, 26, 90, -90), _pts((10, 48))),
            _pts((38, 48), (68, 0)),
        ],
    )
    g["S"] = (68, [_join(_arc(33, 72, 25, 28, -30, 210), _arc(33, 28, 25, 28, 30, -150))])
    g["T"] = (64, [_pts((4, 100), (60, 100)), _pts((32, 100), (32, 0))])
    g["U"] = (74, [_join(_pts((10, 100), (10, 28)), _arc(37, 28, 27, 28, 180, 360), _pts((64, 100)))])
    g["V"] = (72, [_pts((4, 100), (36, 0), (68, 100))])
    g["W"] = (98, [_pts((4, 100), (24, 0), (49, 74), (74, 0), (94, 100))])
    g["X"] = (70, [_pts((6, 100), (64, 0)), _pts((6, 0), (64, 100))])
    g["Y"] = (70, [_pts((6, 100), (35, 50), (64, 100)), _pts((35, 50), (35, 0))])
    g["Z"] = (68, [_pts((8, 100), (60, 100), (8, 0), (60, 0))])

    # ---- lowercase -------------------------------------------------------
    g["a"] = (58, [_ellipse(25, 33, 20, 33), _pts((45, 66), (45, 0))])
    g["b"] = (58, [_pts((10, 100), (10, 0)), _ellipse(32, 33, 22, 33)])
    g["c"] = (54, [_arc(28, 33, 21, 33, 45, 315)])
    g["d"] = (58, [_pts((48, 100), (48, 0)), _ellipse(26, 33, 22, 33)])
    g["e"] = (56, [_join(_pts((6, 33), (50, 33)), _arc(28, 33, 22, 33, 0, 320)[1:])])
    g["f"] = (38, [_join(_arc(24, 88, 13, 14, 10, 175), _pts((11, 88), (11, 0))), _pts((0, 66), (34, 66))])
    g["g"] = (58, [_ellipse(26, 33, 22, 33), _join(_pts((48, 66), (48, -8)), _arc(26, -8, 22, 22, 0, -165))])
    g["h"] = (58, [_pts((10, 100), (10, 0)), _join(_arc(30, 44, 20, 22, 180, 0), _pts((50, 0)))])
    g["i"] = (26, [_pts((13, 66), (13, 0)), _dot(13, 86)])
    g["j"] = (28, [_join(_pts((17, 66), (17, -10)), _arc(3, -10, 14, 16, 0, -170)), _dot(17, 86)])
    g["k"] = (54, [_pts((10, 100), (10, 0)), _pts((48, 66), (14, 32)), _pts((24, 42), (50, 0))])
    g["l"] = (26, [_pts((13, 100), (13, 0))])
    g["m"] = (
        88,
        [
            _pts((10, 66), (10, 0)),
            _join(_arc(29, 45, 19, 21, 180, 0), _pts((48, 0))),
            _join(_arc(67, 45, 19, 21, 180, 0), _pts((86, 0))),
        ],
    )
    g["n"] = (58, [_pts((10, 66), (10, 0)), _join(_arc(30, 44, 20, 22, 180, 0), _pts((50, 0)))])
    g["o"] = (58, [_ellipse(29, 33, 23, 33)])
    g["p"] = (58, [_pts((10, 66), (10, -30)), _ellipse(32, 33, 22, 33)])
    g["q"] = (58, [_pts((48, 66), (48, -30)), _ellipse(26, 33, 22, 33)])
    g["r"] = (44, [_pts((10, 66), (10, 0)), _arc(29, 45, 19, 21, 180, 55)])
    g["s"] = (52, [_join(_arc(26, 49, 19, 17, -30, 210), _arc(26, 17, 19, 17, 30, -150))])
    # The foot must swing *down* to the baseline before it turns right, so the
    # sweep goes the other way round from r's shoulder: 180 -> 305, not 180 -> 55.
    g["t"] = (40, [_join(_pts((14, 92), (14, 12)), _arc(26, 12, 12, 12, 180, 305)), _pts((2, 66), (34, 66))])
    g["u"] = (58, [_join(_pts((10, 66), (10, 22)), _arc(30, 22, 20, 22, 180, 360), _pts((50, 66))), _pts((50, 22), (50, 0))])
    g["v"] = (54, [_pts((4, 66), (27, 0), (50, 66))])
    g["w"] = (78, [_pts((4, 66), (21, 0), (39, 48), (57, 0), (74, 66))])
    g["x"] = (52, [_pts((4, 66), (48, 0)), _pts((4, 0), (48, 66))])
    g["y"] = (54, [_pts((4, 66), (28, 4)), _pts((50, 66), (14, -30))])
    g["z"] = (52, [_pts((6, 66), (46, 66), (6, 0), (46, 0))])

    # ---- digits ----------------------------------------------------------
    g["0"] = (68, [_ellipse(34, 50, 26, 50)])
    g["1"] = (48, [_pts((12, 78), (30, 100), (30, 0)), _pts((10, 0), (50, 0))])
    g["2"] = (66, [_join(_arc(33, 74, 24, 26, 170, -50), _pts((8, 0), (60, 0)))])
    g["3"] = (66, [_join(_arc(33, 74, 22, 26, 165, -70), _arc(33, 26, 25, 26, 75, -175))])
    g["4"] = (68, [_pts((46, 0), (46, 100)), _pts((46, 100), (6, 32), (62, 32))])
    g["5"] = (66, [_join(_pts((56, 100), (18, 100), (14, 58)), _arc(33, 30, 26, 30, 100, -160))])
    g["6"] = (66, [_join(_arc(33, 66, 26, 34, 25, 180), _pts((7, 28)), _arc(33, 28, 26, 28, 180, -180))])
    g["7"] = (64, [_pts((6, 100), (58, 100), (26, 0))])
    g["8"] = (68, [_ellipse(34, 74, 23, 26), _ellipse(34, 26, 27, 26)])
    g["9"] = (66, [_ellipse(33, 72, 26, 28), _join(_pts((59, 72), (59, 34)), _arc(33, 34, 26, 34, 0, -150))])

    # ---- punctuation & symbols ------------------------------------------
    g[" "] = (40, [])
    g["."] = (28, [_dot(14, 5)])
    g[","] = (28, [_pts((17, 8), (14, 0), (8, -14))])
    g[":"] = (28, [_dot(14, 5), _dot(14, 46)])
    g[";"] = (28, [_dot(14, 46), _pts((17, 8), (14, 0), (8, -14))])
    g["!"] = (28, [_pts((14, 100), (14, 26)), _dot(14, 6)])
    g["?"] = (58, [_join(_arc(29, 76, 21, 22, 180, -35), _pts((29, 42), (29, 30))), _dot(29, 6)])
    g["-"] = (52, [_pts((8, 44), (44, 44))])
    g["–"] = (64, [_pts((6, 44), (58, 44))])
    g["—"] = (80, [_pts((4, 44), (76, 44))])
    g["_"] = (58, [_pts((2, -14), (56, -14))])
    g["+"] = (62, [_pts((8, 46), (54, 46)), _pts((31, 23), (31, 69))])
    g["="] = (62, [_pts((8, 56), (54, 56)), _pts((8, 34), (54, 34))])
    g["*"] = (52, [_pts((26, 100), (26, 56)), _pts((7, 89), (45, 67)), _pts((7, 67), (45, 89))])
    g["/"] = (48, [_pts((4, -10), (44, 104))])
    g["\\"] = (48, [_pts((4, 104), (44, -10))])
    g["|"] = (26, [_pts((13, -10), (13, 104))])
    g["("] = (32, [_arc(34, 48, 27, 58, 148, 212)])
    g[")"] = (32, [_arc(-2, 48, 27, 58, 32, -32)])
    g["["] = (32, [_pts((26, 104), (8, 104), (8, -8), (26, -8))])
    g["]"] = (32, [_pts((6, 104), (24, 104), (24, -8), (6, -8))])
    g["{"] = (36, [_join(_pts((30, 104), (18, 96), (18, 56), (6, 48), (18, 40), (18, -0), (30, -8)))])
    g["}"] = (36, [_join(_pts((6, 104), (18, 96), (18, 56), (30, 48), (18, 40), (18, -0), (6, -8)))])
    g["'"] = (22, [_pts((11, 100), (11, 72))])
    g['"'] = (36, [_pts((11, 100), (11, 72)), _pts((25, 100), (25, 72))])
    g["#"] = (68, [_pts((20, 0), (28, 100)), _pts((42, 0), (50, 100)), _pts((6, 32), (62, 32)), _pts((8, 68), (64, 68))])
    g["%"] = (
        80,
        [
            _pts((8, 0), (72, 100)),
            _ellipse(21, 79, 13, 15, 14),
            _ellipse(59, 21, 13, 15, 14),
        ],
    )
    g["&"] = (
        80,
        [
            _join(
                _arc(31, 74, 15, 16, -45, 235),
                _arc(30, 24, 24, 24, 150, 340),
                _pts((70, 44)),
            ),
            _pts((22, 60), (58, 4)),
        ],
    )
    g["@"] = (
        90,
        [
            _ellipse(42, 44, 16, 18, 18),
            _join(_pts((58, 44), (58, 24)), _arc(46, 44, 38, 44, -18, 300)),
        ],
    )
    g["<"] = (58, [_pts((48, 78), (10, 46), (48, 14))])
    g[">"] = (58, [_pts((10, 78), (48, 46), (10, 14))])
    g["~"] = (60, [_join(_arc(18, 46, 12, 10, 180, 0), _arc(42, 46, 12, 10, 180, 360))])
    g["^"] = (58, [_pts((8, 74), (29, 100), (50, 74))])
    g["°"] = (40, [_ellipse(20, 84, 12, 12, 14)])
    g["·"] = (28, [_dot(14, 40)])

    # ---- accented / Nordic ----------------------------------------------
    def _accented(base: str, accents: list[np.ndarray], extra_width: float = 0.0):
        w, strokes = g[base]
        return (w + extra_width, list(strokes) + list(accents))

    g["Å"] = _accented("A", [_ellipse(36, 122, 10, 10, 12)])
    g["Ä"] = _accented("A", [_dot(24, 118), _dot(48, 118)])
    g["Ö"] = _accented("O", [_dot(28, 118), _dot(52, 118)])
    g["Ü"] = _accented("U", [_dot(25, 118), _dot(49, 118)])
    g["É"] = _accented("E", [_pts((22, 112), (40, 130))])
    g["È"] = _accented("E", [_pts((40, 112), (22, 130))])
    g["å"] = _accented("a", [_ellipse(25, 90, 9, 9, 12)])
    g["ä"] = _accented("a", [_dot(16, 86), _dot(34, 86)])
    g["ö"] = _accented("o", [_dot(20, 86), _dot(38, 86)])
    g["ü"] = _accented("u", [_dot(20, 86), _dot(40, 86)])
    g["é"] = _accented("e", [_pts((22, 82), (36, 100))])
    g["è"] = _accented("e", [_pts((36, 82), (22, 100))])
    g["ñ"] = _accented("n", [_join(_arc(20, 88, 9, 7, 180, 0), _arc(38, 88, 9, 7, 180, 360))])
    g["ø"] = (58, [_ellipse(29, 33, 23, 33), _pts((4, 0), (54, 66))])
    g["Ø"] = (80, [_ellipse(40, 50, 33, 50), _pts((4, -6), (76, 106))])
    g["æ"] = (84, [_ellipse(25, 33, 20, 33), _pts((45, 66), (45, 0)), _join(_pts((32, 33), (76, 33)), _arc(54, 33, 22, 33, 0, 320)[1:])])
    g["Æ"] = (100, [_pts((4, 0), (36, 100)), _pts((36, 100), (94, 100)), _pts((36, 100), (36, 0), (94, 0)), _pts((36, 52), (78, 52))])

    # currency
    g["€"] = (72, [_arc(42, 50, 30, 50, 50, 310), _pts((4, 60), (52, 60)), _pts((4, 40), (52, 40))])
    g["$"] = (
        68,
        [
            _join(_arc(33, 66, 23, 22, -30, 210), _arc(33, 28, 23, 22, 30, -150)),
            _pts((33, 108), (33, -10)),
        ],
    )
    g["£"] = (68, [_join(_arc(40, 76, 22, 22, 20, 200), _pts((18, 62), (18, 12), (12, 0), (62, 0))), _pts((6, 40), (46, 40))])

    return g


GLYPHS = _build()
MISSING = (56.0, [np.array([[8, 0], [48, 0], [48, 92], [8, 92], [8, 0]], dtype=np.float64)])


def glyph(ch: str) -> tuple[float, list[np.ndarray]]:
    """Advance width and strokes (Y up, cap height = 100) for one character."""
    if ch in GLYPHS:
        return GLYPHS[ch]
    lowered = ch.lower()
    if lowered in GLYPHS:
        return GLYPHS[lowered]
    return MISSING


def has_glyph(ch: str) -> bool:
    return ch in GLYPHS


def text_width(text: str, tracking: float = 0.0) -> float:
    total = 0.0
    for ch in text:
        total += glyph(ch)[0] + tracking
    return max(total - tracking, 0.0) if text else 0.0
