"""Pen library.

A drawing is split into one layer per pen.  Each pen carries the properties
that actually change how geometry is generated (line width) and how it is
plotted (Z offset because pens have different lengths, feed rate, and how
often a pencil needs re-sharpening).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

__all__ = [
    "Pen",
    "PenLibrary",
    "hex_to_rgb",
    "rgb_to_hex",
    "srgb_to_lab",
    "DEFAULT_PALETTES",
    "PEN_KINDS",
    "DWELL_CAPABLE",
    "CUTTING_KINDS",
]

# What sort of tip is in the holder.  This is not cosmetic: with a fibre or
# fountain tip the ink keeps flowing while the pen rests on the paper, so the
# dot gets darker the longer it waits.  A ballpoint needs movement to write at
# all and simply will not do that.
PEN_KINDS = {
    "fineliner": "Fineliner / fibre tip",
    "fountain": "Fountain pen",
    "gel": "Gel pen",
    "marker": "Marker",
    "ballpoint": "Ballpoint",
    "pencil": "Pencil",
    "knife": "Drag knife / swivel blade",
    "scalpel": "Fixed blade / scalpel",
    "scoring": "Scoring / embossing tip",
    "other": "Other",
}

#: tips where resting on the paper actually deposits more ink
DWELL_CAPABLE = {"fineliner", "fountain", "gel", "marker"}

#: tools that cut or crease instead of leaving ink
CUTTING_KINDS = {"knife", "scalpel", "scoring"}


def hex_to_rgb(value: str) -> tuple[float, float, float]:
    v = value.strip().lstrip("#")
    if len(v) == 3:
        v = "".join(c * 2 for c in v)
    if len(v) != 6:
        return (0.0, 0.0, 0.0)
    return tuple(int(v[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def rgb_to_hex(rgb) -> str:
    r, g, b = (int(round(max(0.0, min(1.0, c)) * 255)) for c in rgb)
    return f"#{r:02X}{g:02X}{b:02X}"


def _f(x: float) -> float:
    return x ** (1 / 3) if x > 0.008856 else (7.787 * x) + (16 / 116)


def srgb_to_lab(rgb) -> tuple[float, float, float]:
    """Single-colour sRGB (0..1) -> CIE Lab, D65."""
    r, g, b = rgb

    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = lin(r), lin(g), lin(b)
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883
    fx, fy, fz = _f(x), _f(y), _f(z)
    return (116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))


@dataclass
class Pen:
    """One physical pen in the holder."""

    name: str = "Black fineliner"
    color: str = "#1A1A1A"
    kind: str = "fineliner"         # see PEN_KINDS
    width: float = 0.5              # mm - drives hatch spacing, dot pitch, simplify
    z_offset: float = 0.0           # mm added to the global draw Z for this pen
    feed_scale: float = 1.0         # multiplies the drawing feed rate
    sharpen_every: float = 0.0      # metres of drawn line between sharpening pauses
    enabled: bool = True
    note: str = ""

    # --- cutting / scoring -------------------------------------------------
    # A blade rarely goes through card in one go, and a plain pen can perforate
    # paper if it goes over the same line often enough.  Both are the same
    # thing to the machine: run every stroke `passes` times, a little deeper
    # each time.
    passes: int = 1                 # how many times each stroke is repeated
    pass_depth: float = 0.0         # mm deeper on each repeat
    blade_offset: float = 0.0       # mm the tip of a swivel blade trails behind
    overcut: float = 0.0            # mm past the start of a closed cut

    #: The finest real drawing pen is about 0.03 mm.  Below that every spacing
    #: derived from the width collapses towards zero, and a technique that lays
    #: one stroke per pen width tries to fill the page - a quarter of a million
    #: strokes and an eighteen-hour plot, from one slider.
    MIN_WIDTH = 0.03

    def __setattr__(self, name: str, value) -> None:
        # on assignment, not just on construction: the width is set from the
        # panel, from a preset and from a settings file, and only one of those
        # goes through __init__
        if name == "width":
            try:
                value = max(float(value), self.MIN_WIDTH)
            except (TypeError, ValueError):
                value = 0.5
        object.__setattr__(self, name, value)

    @property
    def rgb(self) -> tuple[float, float, float]:
        return hex_to_rgb(self.color)

    @property
    def lab(self) -> tuple[float, float, float]:
        return srgb_to_lab(self.rgb)

    @property
    def lightness(self) -> float:
        return self.lab[0]

    @property
    def bleeds(self) -> bool:
        """True when holding the pen still makes a darker mark."""
        return self.kind in DWELL_CAPABLE

    @property
    def cuts(self) -> bool:
        """True when this tool is a blade rather than something that draws."""
        return self.kind in CUTTING_KINDS

    @property
    def swivels(self) -> bool:
        """A drag knife whose tip needs the corners swinging out for it."""
        return self.kind == "knife" and self.blade_offset > 1e-3

    @property
    def repeats(self) -> int:
        return max(1, int(self.passes))

    @property
    def kind_label(self) -> str:
        return PEN_KINDS.get(self.kind, PEN_KINDS["other"])

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Pen":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


DEFAULT_PALETTES: dict[str, list[tuple[str, str, float]]] = {
    "Single black pen": [("Black fineliner", "#1A1A1A", 0.5)],
    "CMYK (4 pens)": [
        ("Yellow", "#F2C200", 0.7),
        ("Magenta", "#E5007E", 0.7),
        ("Cyan", "#009EE0", 0.7),
        ("Black", "#1A1A1A", 0.5),
    ],
    "Primary colours (5 pens)": [
        ("Yellow", "#F5C518", 0.7),
        ("Red", "#E4321B", 0.7),
        ("Green", "#2FA84F", 0.7),
        ("Blue", "#1F5FBF", 0.7),
        ("Black", "#1A1A1A", 0.5),
    ],
    "Pencil set (3 grades)": [
        ("2H light", "#9AA0A6", 0.7),
        ("HB medium", "#5F6368", 0.7),
        ("4B dark", "#202124", 0.9),
    ],
    "Fineliner set (3 widths)": [
        ("Fineliner 0.2", "#1A1A1A", 0.2),
        ("Fineliner 0.5", "#1A1A1A", 0.5),
        ("Fineliner 0.8", "#1A1A1A", 0.8),
    ],
}

#: Tool presets for cutting stencils.  These are not colours, they are how the
#: blade behaves: how many times it goes round, and how much deeper each time.
CUTTING_TOOLS: dict[str, dict] = {
    "Drag knife (vinyl / thin card)": dict(
        name="Drag knife", color="#B0343C", kind="knife", width=0.3,
        passes=1, pass_depth=0.0, blade_offset=0.25, overcut=1.0, feed_scale=0.5,
        note="Swivel blade. Set the depth so it cuts the sheet but not the backing.",
    ),
    "Scalpel (card, several passes)": dict(
        name="Scalpel", color="#8A3FFC", kind="scalpel", width=0.4,
        passes=3, pass_depth=0.25, blade_offset=0.0, overcut=0.5, feed_scale=0.35,
        note="A fixed blade cannot turn - it scores. Repeat passes cut deeper.",
    ),
    "Ordinary pen, scored N times": dict(
        name="Scoring pen", color="#1A1A1A", kind="scoring", width=0.5,
        passes=6, pass_depth=0.06, blade_offset=0.0, overcut=0.0, feed_scale=0.6,
        note="No blade at all: the same line over and over until the paper gives.",
    ),
    "Embossing / creasing tip": dict(
        name="Creaser", color="#0F7B6C", kind="scoring", width=0.8,
        passes=2, pass_depth=0.1, blade_offset=0.0, overcut=0.0, feed_scale=0.5,
        note="Presses a fold line without breaking the surface.",
    ),
}


@dataclass
class PenLibrary:
    """The pens currently loaded, in the order they will be drawn."""

    pens: list[Pen] = field(default_factory=lambda: [Pen()])

    def __len__(self) -> int:
        return len(self.pens)

    def __getitem__(self, index: int) -> Pen:
        if not self.pens:
            self.pens.append(Pen())
        return self.pens[max(0, min(index, len(self.pens) - 1))]

    def __iter__(self):
        return iter(self.pens)

    @property
    def enabled_indices(self) -> list[int]:
        return [i for i, p in enumerate(self.pens) if p.enabled]

    def add(self, pen: Pen | None = None) -> Pen:
        pen = pen or Pen(name=f"Pen {len(self.pens) + 1}", color="#1A1A1A")
        self.pens.append(pen)
        return pen

    def remove(self, index: int) -> None:
        if 0 <= index < len(self.pens) and len(self.pens) > 1:
            self.pens.pop(index)

    def move(self, index: int, delta: int) -> int:
        target = index + delta
        if 0 <= index < len(self.pens) and 0 <= target < len(self.pens):
            self.pens[index], self.pens[target] = self.pens[target], self.pens[index]
            return target
        return index

    def widest(self) -> float:
        return max((p.width for p in self.pens), default=0.5)

    def narrowest(self) -> float:
        return min((p.width for p in self.pens), default=0.5)

    def sorted_light_to_dark(self) -> list[int]:
        """Plot order that avoids dragging a dark pen through wet light ink."""
        return sorted(range(len(self.pens)), key=lambda i: -self.pens[i].lightness)

    def lab_array(self) -> np.ndarray:
        return np.asarray([p.lab for p in self.pens], dtype=np.float32)

    def apply_palette(self, name: str) -> None:
        preset = DEFAULT_PALETTES.get(name)
        if not preset:
            return
        kind = "pencil" if "Pencil" in name else "fineliner"
        self.pens = [Pen(name=n, color=c, width=w, kind=kind) for n, c, w in preset]

    def any_bleeds(self) -> bool:
        return any(pen.bleeds for pen in self.pens if pen.enabled)

    def any_cuts(self) -> bool:
        return any(pen.cuts for pen in self.pens if pen.enabled)

    def first_cutter(self) -> int | None:
        for index, pen in enumerate(self.pens):
            if pen.cuts:
                return index
        return None

    def add_tool(self, preset: str) -> int:
        """Add one of the CUTTING_TOOLS presets and return its index."""
        spec = CUTTING_TOOLS.get(preset)
        if not spec:
            return max(len(self.pens) - 1, 0)
        self.pens.append(Pen(**spec))
        return len(self.pens) - 1

    def to_dict(self) -> dict:
        return {"pens": [p.to_dict() for p in self.pens]}

    @classmethod
    def from_dict(cls, data: dict) -> "PenLibrary":
        pens = [Pen.from_dict(d) for d in data.get("pens", [])]
        return cls(pens=pens or [Pen()])
