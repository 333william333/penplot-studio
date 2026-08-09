"""Containers shared by every stage of the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import geometry as geo

__all__ = ["Layer", "Drawing", "SourceResult"]


@dataclass
class Layer:
    """All strokes drawn with one pen."""

    pen: int = 0
    paths: list[np.ndarray] = field(default_factory=list)
    name: str = ""
    dwell_ms: float = 0.0   # how long the pen rests on each dot in this layer
    #: "" | "pressure" | "speed" - what a third path column means, if present
    modulation: str = ""
    modulation_amount: float = 0.0
    #: index of the project item this came from, or -1
    item: int = -1

    def copy_with(self, paths: list[np.ndarray]) -> "Layer":
        return Layer(
            pen=self.pen,
            paths=paths,
            name=self.name,
            dwell_ms=self.dwell_ms,
            modulation=self.modulation,
            modulation_amount=self.modulation_amount,
            item=self.item,
        )

    @property
    def draw_length(self) -> float:
        return geo.total_length(self.paths)

    def __len__(self) -> int:
        return len(self.paths)


@dataclass
class Drawing:
    """A complete plot: layers in drawing order plus some bookkeeping."""

    layers: list[Layer] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    source_label: str = ""

    def all_paths(self) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for layer in self.layers:
            out.extend(layer.paths)
        return out

    def bounds(self):
        return geo.bounds(self.all_paths())

    @property
    def path_count(self) -> int:
        return sum(len(layer.paths) for layer in self.layers)

    @property
    def draw_length(self) -> float:
        return sum(layer.draw_length for layer in self.layers)

    def is_empty(self) -> bool:
        return self.path_count == 0

    def item_bounds(self) -> dict:
        """Bounding box per project item, for selecting things on the canvas."""
        from . import geometry as geo

        grouped: dict[int, list] = {}
        for layer in self.layers:
            if layer.item >= 0:
                grouped.setdefault(layer.item, []).extend(layer.paths)
        return {index: geo.bounds(paths) for index, paths in grouped.items() if paths}

    def used_pens(self) -> list[int]:
        seen: list[int] = []
        for layer in self.layers:
            if layer.paths and layer.pen not in seen:
                seen.append(layer.pen)
        return seen


@dataclass
class SourceResult:
    """What a source stage (image / text / PDF) hands to the pipeline."""

    kind: str = "image"
    rgb: np.ndarray | None = None          # raster sources: float32 HxWx3, 0..1
    layers: list[Layer] | None = None      # vector sources: paths already in mm
    vector: object | None = None           # pdfsource.VectorArt for coloured line work
    mm_per_unit: float = 0.0               # >0 when the source has a physical size
    label: str = ""
    page_count: int = 1
    warnings: list[str] = field(default_factory=list)

    @property
    def is_raster(self) -> bool:
        return self.rgb is not None

    def native_size_mm(self) -> tuple[float, float] | None:
        if self.mm_per_unit <= 0:
            return None
        if self.is_raster:
            h, w = self.rgb.shape[:2]
            return (w * self.mm_per_unit, h * self.mm_per_unit)
        paths = [p for layer in (self.layers or []) for p in layer.paths]
        if self.vector is not None:
            paths = paths + list(getattr(self.vector, "paths", []))
        bounds = geo.bounds(paths)
        if not bounds:
            return None
        return (bounds[2] - bounds[0], bounds[3] - bounds[1])
