"""The engine that looks at a picture and decides what to do with it.

Two backends, and the difference between them is the difference between knowing
and guessing:

*classical* is always there and needs nothing downloaded.  Sharpness, centre
weighting, histograms, edge density.  It can tell a drawing from a photograph
and it can tell you where the sharp, central part of a frame is.  It cannot tell
you that the sharp central part is a person - a bowl of fruit scores the same.

*neural* runs YuNet through OpenCV's own ONNX inference (`cv2.FaceDetectorYN`)
and actually finds faces: how many, where, and how much of the frame they fill.
That turns "this might be a person" into "there is a face here, 41% of the
frame", which is the difference between offering Portrait and choosing it.

The model is not shipped and is never fetched behind the user's back.  The app
asks, says how big it is and where it comes from, and works exactly as well as
before if the answer is no - the classical backend is the floor, not a fallback
that breaks things.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import raster

__all__ = [
    "Face",
    "Reading",
    "MODEL",
    "model_path",
    "have_model",
    "download_model",
    "read",
    "detail_map",
]


#: YuNet face detector: 232 KB, from the OpenCV Zoo, Apache-2.0.  Small enough
#: that the download is over before the dialog has finished appearing.
MODEL = {
    "name": "face_detection_yunet_2023mar.onnx",
    # the media endpoint, not raw: the file is stored with Git LFS, and raw
    # hands back a 131-byte pointer that looks like a successful download
    "url": (
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx"
    ),
    "bytes": 232589,
    "licence": "Apache-2.0, OpenCV Zoo",
}


@dataclass
class Face:
    x: float
    y: float
    width: float
    height: float
    confidence: float

    @property
    def centre(self) -> tuple[float, float]:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)


@dataclass
class Reading:
    """What the engine believes about a picture, and how sure it is."""

    backend: str = "classical"          # classical | neural
    faces: list[Face] = field(default_factory=list)
    face_share: float = 0.0             # fraction of the frame the faces fill
    subject_share: float = 0.0
    #: 0..1 per pixel, at the analysis resolution
    weight: np.ndarray | None = None

    @property
    def knows_it_is_a_person(self) -> bool:
        """True only when a detector actually found a face."""
        return self.backend == "neural" and bool(self.faces)

    def summary(self) -> str:
        if self.knows_it_is_a_person:
            count = len(self.faces)
            who = "a face" if count == 1 else f"{count} faces"
            return f"Found {who}, filling {self.face_share * 100:.0f}% of the picture"
        if self.backend == "neural":
            return "No face found - treating it as a general picture"
        return f"Subject fills about {self.subject_share * 100:.0f}% of the frame"


def model_path() -> Path:
    from .settings import config_dir

    return Path(config_dir()) / MODEL["name"]


def have_model() -> bool:
    path = model_path()
    return path.exists() and path.stat().st_size > 10_000


def download_model(progress=None) -> tuple[bool, str]:
    """Fetch the detector.  Only ever called from an explicit user action."""
    import urllib.request

    path = model_path()
    try:
        with urllib.request.urlopen(MODEL["url"], timeout=30) as response:
            data = response.read()
        if len(data) < 10_000:
            return False, (
                "That was not the model - the server sent "
                f"{len(data)} bytes. Check the network and try again."
            )
        temporary = path.with_suffix(".part")
        temporary.write_bytes(data)
        os.replace(temporary, path)
    except Exception as exc:
        return False, f"Could not download the model: {exc}"
    # prove it loads before claiming success
    try:
        _detector((320, 320))
    except Exception as exc:  # pragma: no cover - defensive
        path.unlink(missing_ok=True)
        return False, f"The model downloaded but would not load: {exc}"
    return True, f"Face detection ready ({len(data) / 1024:.0f} KB)"


_DETECTOR = None


def _detector(size: tuple[int, int]):
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = cv2.FaceDetectorYN.create(str(model_path()), "", size, 0.6, 0.3, 5000)
    _DETECTOR.setInputSize(size)
    return _DETECTOR


def _find_faces(rgb: np.ndarray) -> list[Face]:
    small = raster.resize_long_edge(np.asarray(rgb, dtype=np.float32), 640)
    bgr = np.clip(small[:, :, ::-1] * 255.0, 0, 255).astype(np.uint8)
    height, width = bgr.shape[:2]
    detector = _detector((width, height))
    _, raw = detector.detect(bgr)
    if raw is None:
        return []
    scale_x = 1.0 / width
    scale_y = 1.0 / height
    faces = []
    for row in raw:
        x, y, w, h = (float(v) for v in row[:4])
        faces.append(
            Face(x * scale_x, y * scale_y, w * scale_x, h * scale_y, float(row[-1]))
        )
    return faces


def read(rgb: np.ndarray, use_neural: bool = True) -> Reading:
    """Look at a picture with the best engine available."""
    gray = raster.to_gray(raster.resize_long_edge(np.asarray(rgb, dtype=np.float32), 420))
    weight = raster.subject_weight(gray)
    reading = Reading(
        backend="classical",
        subject_share=float((weight > 0.55).mean()),
        weight=weight,
    )

    if use_neural and have_model():
        try:
            faces = _find_faces(rgb)
        except Exception:  # pragma: no cover - a broken model must not break the app
            return reading
        reading.backend = "neural"
        reading.faces = faces
        reading.face_share = float(sum(f.width * f.height for f in faces))
        if faces:
            # A found face is worth more than any guess about where the subject
            # is, so it replaces the centre weighting rather than adding to it.
            reading.weight = _face_weight(weight.shape, faces)
            reading.subject_share = float((reading.weight > 0.55).mean())
    return reading


def _face_weight(shape: tuple[int, int], faces: list[Face]) -> np.ndarray:
    """A subject map built around real faces: head, shoulders, then falloff."""
    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    weight = np.zeros((height, width), dtype=np.float32)
    for face in faces:
        cx, cy = face.centre
        # the head is taller than the detected box, and the shoulders below it
        # are part of the subject too
        rx = max(face.width * 1.5, 0.06)
        ry = max(face.height * 1.9, 0.08)
        distance = np.sqrt(
            ((xx / width - cx) / rx) ** 2 + ((yy / height - (cy + face.height * 0.25)) / ry) ** 2
        )
        weight = np.maximum(weight, np.clip(1.4 - distance, 0.0, 1.0))
    return cv2.GaussianBlur(weight, (0, 0), max(min(height, width) * 0.03, 2.0))


def detail_map(reading: Reading, shape: tuple[int, int]) -> np.ndarray | None:
    """Where the drawing should spend its detail, 0..1 at `shape`.

    Eyes and mouth carry a face; cheeks carry tone.  With a real detection this
    is a genuine map of where to look, which is what a portrait needs and what
    no amount of centre weighting can give you.
    """
    if reading.weight is None:
        return None
    return cv2.resize(reading.weight, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
