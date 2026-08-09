"""Draw the application icon and compile it to build/icon.icns.

Kept as a script rather than a checked-in binary so the icon can never drift
from the palette the app itself uses, and so a fresh clone can build the bundle
without hunting for a missing .icns.

    python packaging/make_icon.py
"""

from __future__ import annotations

import math
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"

# the app's own Spectrum dark values
BG_TOP = (34, 34, 34)
BG_BOTTOM = (20, 20, 20)
INK = (240, 240, 240)
ACCENT = (20, 115, 230)


def _rounded_mask(size: int, radius_ratio: float = 0.2237) -> Image.Image:
    """macOS' squircle is close enough to a rounded rect at icon sizes."""
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, size - 1, size - 1), radius=int(size * radius_ratio), fill=255
    )
    return mask


def render(size: int) -> Image.Image:
    ss = 4  # supersample, then shrink: the plotted line has to stay crisp
    big = size * ss
    image = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    for y in range(big):
        t = y / max(big - 1, 1)
        draw.line(
            [(0, y), (big, y)],
            fill=tuple(int(a + (b - a) * t) for a, b in zip(BG_TOP, BG_BOTTOM)) + (255,),
        )

    # One continuous stroke, the way the plotter would draw it: a spiral that
    # tightens towards the middle.  Reads as "pen on paper" at 32 px too.
    centre = big / 2
    points = []
    turns = 3.15
    steps = 1400
    for i in range(steps + 1):
        t = i / steps
        angle = t * turns * 2 * math.pi - math.pi / 2
        radius = big * 0.40 * (1.0 - t * 0.86)
        points.append((centre + radius * math.cos(angle), centre + radius * math.sin(angle)))
    draw.line(points, fill=INK + (255,), width=max(int(big * 0.035), 1), joint="curve")

    # the pen itself, parked at the end of the line
    tip = points[-1]
    draw.ellipse(
        [tip[0] - big * 0.045, tip[1] - big * 0.045, tip[0] + big * 0.045, tip[1] + big * 0.045],
        fill=ACCENT + (255,),
    )

    image = image.resize((size, size), Image.LANCZOS)
    image.putalpha(_rounded_mask(size))
    return image


def main() -> int:
    iconset = BUILD / "icon.iconset"
    if iconset.exists():
        shutil.rmtree(iconset)
    iconset.mkdir(parents=True)

    for base in (16, 32, 128, 256, 512):
        render(base).save(iconset / f"icon_{base}x{base}.png")
        render(base * 2).save(iconset / f"icon_{base}x{base}@2x.png")

    icns = BUILD / "icon.icns"
    result = subprocess.run(
        ["iconutil", "-c", "icns", str(iconset), "-o", str(icns)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        return result.returncode
    print(f"wrote {icns} ({icns.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
