"""All user-facing settings, plus JSON load/save.

Every spatial setting is in millimetres on the paper - never pixels - so the
same numbers keep meaning the same thing when the artwork is resized or when a
different pen width is selected.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from .pdfsource import PdfSettings
from .pens import Pen, PenLibrary
from .textsource import TextSettings

__all__ = [
    "MachineSettings",
    "PenSetup",
    "StyleSettings",
    "LayoutSettings",
    "OptimizeSettings",
    "PauseSettings",
    "AppSettings",
    "config_dir",
    "technique_labels",
]

def technique_labels() -> dict[str, str]:
    from .techniques import REGISTRY

    return {key: tech.label for key, tech in REGISTRY.items()}


def config_dir() -> Path:
    base = Path(os.path.expanduser("~/Library/Application Support"))
    if not base.exists():
        base = Path(os.path.expanduser("~/.config"))
    path = base / "PenPlotStudio"
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------
@dataclass
class MachineSettings:
    name: str = "Ender 3 / Pro / V2"
    firmware: str = "marlin"       # see core/profiles.FIRMWARES
    bed_x: float = 220.0
    bed_y: float = 220.0
    max_z: float = 250.0
    draw_feed: float = 2400.0      # mm/min while the pen is down
    travel_feed: float = 6000.0    # mm/min while the pen is up
    z_feed: float = 900.0          # mm/min for pen up/down moves
    z_max_feed: float = 300.0      # mm/min the firmware will actually allow
    raise_z_limit: bool = True     # lift the Z ceiling for the job
    z_limit_target: float = 720.0  # mm/min to raise it to (12 mm/s)
    #: A pen lift is short, so it never reaches its top speed - the whole move
    #: is acceleration.  On a stock Ender 3 the Z accelerates at 100 mm/s^2,
    #: which is what actually decides how long a lift takes.
    z_acceleration: float = 100.0        # mm/s^2 the firmware ships with
    z_acceleration_target: float = 300.0 # what to raise it to for a pen
    junction_deviation: float = 0.0      # mm; 0 leaves the firmware alone
    acceleration: float = 500.0    # mm/s^2 while drawing
    travel_acceleration: float = 1200.0
    send_acceleration: bool = True # emit M204 so the machine matches the estimate
    use_bed_mesh: bool = False     # M420 S1: keep pen pressure even on a warped bed
    use_arcs: bool = True          # emit G2/G3 for circles instead of many G1 segments
    port: str = ""
    baud: int = 115200
    use_checksums: bool = True     # numbered lines + checksums (safer on long jobs)
    home_mode: str = "xy"          # xy | all | none
    park_x: float = 5.0
    park_y: float = 200.0
    disable_motors_at_end: bool = True


@dataclass
class PenSetup:
    draw_z: float = 0.0            # Z where the pen touches the paper
    lift: float = 2.5              # how far above draw_z the pen travels
    change_z: float = 60.0         # Z used while swapping pens
    change_x: float = 110.0
    change_y: float = 200.0
    down_delay: int = 0            # ms to settle after lowering
    up_delay: int = 0              # ms to settle after lifting
    scale_with_pen_width: bool = True
    dot_diameter: float = 0.0      # >0 draws stipple dots as tiny circles
    zero_z_at_start: bool = True   # treat the pen's current height as Z0 (G92)
    short_hop: float = 3.0         # mm - hops shorter than this use the small lift
    short_lift: float = 0.6        # mm - enough to clear the paper, much faster


@dataclass
class StyleSettings:
    technique: str = "crosshatch"
    #: which Look decided these settings; "" once the user overrides by hand
    look: str = ""
    #: extra preparation before the technique runs; see raster.enhance_subject
    enhance: str = ""              # "" | "subject"
    detail: int = 900              # working resolution, long edge in pixels

    # shared image adjustments
    brightness: float = 0.0
    contrast: float = 0.0
    gamma: float = 1.0
    blur: float = 0.0
    invert: bool = False
    auto_levels: bool = False
    black_point: float = 0.0
    white_point: float = 1.0
    saturation: float = 1.0

    # tone from the machine itself, on top of whatever the technique draws
    modulation: str = "none"       # none | pressure | speed
    modulation_amount: float = 0.5

    # colour separation
    separation: str = "mono"       # mono | palette | cmyk
    paper_lightness: float = 96.0
    ink_gamma: float = 1.0

    # per-technique settings, keyed by technique id
    params: dict = field(default_factory=dict)

    def technique_params(self, key: str | None = None) -> dict:
        return self.params.setdefault(key or self.technique, {})


@dataclass
class LayoutSettings:
    mode: str = "fit"              # fit | size | scale
    width: float = 150.0           # mm
    height: float = 150.0          # mm
    keep_aspect: bool = True
    scale_percent: float = 100.0
    rotation: float = 0.0
    mirror_x: bool = False
    mirror_y: bool = False
    margin: float = 10.0           # mm of unusable border around the bed
    center: bool = True
    offset_x: float = 0.0          # mm from the centre (or from bed origin)
    offset_y: float = 0.0


@dataclass
class OptimizeSettings:
    simplify: float = 0.12         # mm
    join: float = 0.35             # mm
    min_length: float = 0.5        # mm
    reorder: bool = True
    allow_reverse: bool = True
    #: kept so old settings files still load; PenSetup.scale_with_pen_width is
    #: the single switch the pipeline reads
    scale_with_pen_width: bool = True
    stitch: float = 1.5        # mm - join strokes that end this close (0 = off)
    tidy_tour: bool = True     # extra pass to shorten the travel


@dataclass
class PauseSettings:
    pause_between_pens: bool = True
    park_for_pause: bool = True
    host_pause: bool = True        # pause the USB stream and show a dialog
    emit_m0: bool = True           # also emit M0 so SD prints pause
    sharpen_enabled: bool = False
    sharpen_interval: float = 6.0  # metres of drawn line
    pen_change_message: str = "Insert pen"
    sharpen_message: str = "Sharpen the pen"


@dataclass
class Item:
    """One object on the bed.

    The old single-source app is just a project with one of these.  Each item
    keeps its own source, its own technique settings and its own placement, so
    a sheet can hold a photo in crosshatch next to a line of text next to a
    hand-drawn shape.
    """

    name: str = "Layer 1"
    kind: str = "image"            # image | text | pdf | shapes
    visible: bool = True
    locked: bool = False
    source_path: str = ""
    #: freehand and shape geometry, in millimetres on the bed
    strokes: list = field(default_factory=list)
    #: which pen or tool draws a shapes layer; picture layers get their pens
    #: from the colour separation instead
    pen: int = 0
    style: StyleSettings = field(default_factory=StyleSettings)
    layout: LayoutSettings = field(default_factory=LayoutSettings)
    text: TextSettings = field(default_factory=TextSettings)
    pdf: PdfSettings = field(default_factory=PdfSettings)

    def label(self) -> str:
        if self.name:
            return self.name
        return {"image": "Image", "text": "Text", "pdf": "PDF", "shapes": "Drawing"}.get(self.kind, "Layer")


@dataclass
class AppSettings:
    machine: MachineSettings = field(default_factory=MachineSettings)
    pen: PenSetup = field(default_factory=PenSetup)
    optimize: OptimizeSettings = field(default_factory=OptimizeSettings)
    pauses: PauseSettings = field(default_factory=PauseSettings)
    library: PenLibrary = field(default_factory=PenLibrary)
    items: list = field(default_factory=lambda: [Item()])
    active: int = 0
    #: how big the interface is drawn, 1.0 = as designed
    ui_scale: float = 1.0
    #: the dock arrangement, base64 of QMainWindow.saveState
    window_state: str = ""
    last_image: str = ""
    last_pdf: str = ""
    last_export_dir: str = ""

    # ---------------- the active item, exposed under the old names ----------
    # Everything in the app used to read settings.style / .layout / .text /
    # .pdf directly.  Routing those to the selected item keeps all of that
    # working while the project underneath holds any number of items.
    @property
    def item(self) -> Item:
        if not self.items:
            self.items.append(Item())
        self.active = max(0, min(self.active, len(self.items) - 1))
        return self.items[self.active]

    @property
    def style(self) -> StyleSettings:
        return self.item.style

    @property
    def layout(self) -> LayoutSettings:
        return self.item.layout

    @property
    def text(self) -> TextSettings:
        return self.item.text

    @property
    def pdf(self) -> PdfSettings:
        return self.item.pdf

    @property
    def source_kind(self) -> str:
        return self.item.kind

    @source_kind.setter
    def source_kind(self, value: str) -> None:
        self.item.kind = value

    # ---------------- project editing --------------------------------------
    def add_item(self, kind: str = "image", name: str = "") -> Item:
        item = Item(kind=kind, name=name or f"Layer {len(self.items) + 1}")
        if kind == "shapes":
            # hand-drawn strokes stay exactly where they were put
            item.layout.mode = "asis"
            item.layout.center = False
        # A new item starts near the current one but not exactly on top of it -
        # two coincident layers are invisible and unselectable on the canvas.
        if self.items and kind != "shapes":
            current = self.item.layout
            item.layout.margin = current.margin
            item.layout.center = current.center
            item.layout.offset_x = current.offset_x + 10.0
            item.layout.offset_y = current.offset_y - 10.0
        else:
            item.layout.margin = self.item.layout.margin if self.items else 10.0
        self.items.append(item)
        self.active = len(self.items) - 1
        return item

    def duplicate_item(self, index: int) -> Item | None:
        import copy as _copy

        if not (0 <= index < len(self.items)):
            return None
        clone = _copy.deepcopy(self.items[index])
        clone.name = f"{clone.label()} copy"
        clone.layout.offset_x += 10.0
        clone.layout.offset_y -= 10.0
        self.items.insert(index + 1, clone)
        self.active = index + 1
        return clone

    def remove_item(self, index: int) -> None:
        if len(self.items) <= 1 or not (0 <= index < len(self.items)):
            return
        self.items.pop(index)
        self.active = min(self.active, len(self.items) - 1)

    def move_item(self, index: int, delta: int) -> int:
        target = index + delta
        if 0 <= index < len(self.items) and 0 <= target < len(self.items):
            self.items[index], self.items[target] = self.items[target], self.items[index]
            self.active = target
            return target
        return index

    # ---------------- serialisation ------------------------------------
    def to_dict(self) -> dict:
        return _encode(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AppSettings":
        return _decode(cls, data or {})

    def save(self, path: Path | None = None) -> Path:
        """Write atomically and keep one backup.

        A half-written settings file used to silently reset the pen height and
        the whole pen list back to factory values on the next start.
        """
        path = path or (config_dir() / "settings.json")
        payload = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(payload, encoding="utf-8")
        if path.exists():
            try:
                backup = path.with_suffix(".json.bak")
                backup.write_bytes(path.read_bytes())
            except OSError:
                pass
        os.replace(temporary, path)
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "AppSettings":
        path = path or (config_dir() / "settings.json")
        for candidate in (path, path.with_suffix(".json.bak")):
            if not candidate.exists():
                continue
            try:
                return cls.from_dict(json.loads(candidate.read_text(encoding="utf-8")))
            except Exception:
                continue
        return cls()


# --------------------------------------------------------------------------
def _encode(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _encode(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, PenLibrary):
        return value.to_dict()
    if isinstance(value, (list, tuple)):
        return [_encode(v) for v in value]
    if isinstance(value, dict):
        return {k: _encode(v) for k, v in value.items()}
    return value


def _coerce(raw, _default, field):
    """Force a stored value to the type the dataclass declares.

    Without this a hand-edited or truncated file could put a string into
    `bed_x`, which only blows up much later inside G-code generation.
    """
    annotation = field.type if isinstance(field.type, str) else getattr(field.type, "__name__", "")
    try:
        if annotation.startswith("bool"):
            return bool(raw)
        if annotation.startswith("int"):
            return int(raw)
        if annotation.startswith("float"):
            return float(raw)
        if annotation.startswith("str"):
            return str(raw)
        if annotation.startswith("dict"):
            return raw if isinstance(raw, dict) else {}
        if annotation.startswith("list"):
            return raw if isinstance(raw, list) else []
    except (TypeError, ValueError):
        return field.default if field.default is not MISSING else None
    return raw


def _decode(cls, data: dict):
    if cls is PenLibrary:
        return PenLibrary.from_dict(data)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        raw = data[f.name]
        target = f.type
        if isinstance(target, str):  # postponed annotations
            target = {
                "MachineSettings": MachineSettings,
                "PenSetup": PenSetup,
                "StyleSettings": StyleSettings,
                "LayoutSettings": LayoutSettings,
                "OptimizeSettings": OptimizeSettings,
                "PauseSettings": PauseSettings,
                "TextSettings": TextSettings,
                "PdfSettings": PdfSettings,
                "PenLibrary": PenLibrary,
            }.get(target.split("[")[0].strip(), None)
        if target is PenLibrary:
            kwargs[f.name] = PenLibrary.from_dict(raw or {})
        elif target is not None and is_dataclass(target) and isinstance(raw, dict):
            kwargs[f.name] = _decode(target, raw)
        else:
            kwargs[f.name] = _coerce(raw, getattr(cls, f.name, None), f)
    if cls is AppSettings:
        raw_items = data.get("items")
        if isinstance(raw_items, list) and raw_items:
            kwargs["items"] = [_decode(Item, entry) for entry in raw_items if isinstance(entry, dict)]
        else:
            # a settings file from before the project model: fold the old
            # top-level style/layout/text/pdf into a single item
            legacy = Item()
            for name, target in (("style", StyleSettings), ("layout", LayoutSettings),
                                 ("text", TextSettings), ("pdf", PdfSettings)):
                if isinstance(data.get(name), dict):
                    setattr(legacy, name, _decode(target, data[name]))
            legacy.kind = str(data.get("source_kind", "image"))
            kwargs["items"] = [legacy]
        kwargs.pop("style", None)
        kwargs.pop("layout", None)
        kwargs.pop("text", None)
        kwargs.pop("pdf", None)
        kwargs.pop("source_kind", None)

    try:
        return cls(**kwargs)
    except TypeError:
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in kwargs.items() if k in allowed})
