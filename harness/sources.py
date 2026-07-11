from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Union

import numpy as np
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


@dataclass
class Frame:
    """One RGB frame plus provenance for the engine/sinks."""

    id: int
    pil: Image.Image
    kind: str                 # "image" | "video" | "camera"
    total: Optional[int]      # known frame/image count, else None (camera)


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    """OpenCV BGR ndarray -> RGB PIL image."""
    rgb = bgr[:, :, ::-1]
    return Image.fromarray(np.ascontiguousarray(rgb))


def _iter_capture(
    cap_or_index: Union[int, "cv2.VideoCapture"],  # noqa: F821 - cv2 imported lazily
    *,
    kind: str,
    total: Optional[int],
    max_frames: Optional[int],
    already_open: bool = False,
) -> Iterator[Frame]:
    import cv2

    cap = cap_or_index if already_open else cv2.VideoCapture(cap_or_index)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {kind} source: {cap_or_index!r}")
    i = 0
    try:
        while True:
            if max_frames is not None and i >= max_frames:
                break
            ok, bgr = cap.read()
            if not ok:
                break
            yield Frame(i, _bgr_to_pil(bgr), kind, total)
            i += 1
    finally:
        cap.release()


def open_source(source: Union[str, Path], max_frames: Optional[int] = None) -> Iterator[Frame]:
    """Yield :class:`Frame` objects from an image / folder / video / camera source."""
    s = str(source)
    p = Path(s)

    # 1. camera index, e.g. "0"
    if s.isdigit():
        yield from _iter_capture(int(s), kind="camera", total=None, max_frames=max_frames)
        return

    # 2. directory of images
    if p.is_dir():
        files = sorted(q for q in p.iterdir() if q.suffix.lower() in IMAGE_EXTS)
        if not files:
            raise ValueError(f"No images ({sorted(IMAGE_EXTS)}) in directory: {p}")
        limit = len(files) if max_frames is None else min(len(files), max_frames)
        for i, f in enumerate(files[:limit]):
            yield Frame(i, Image.open(f).convert("RGB"), "image", len(files))
        return

    # 3./4. a file: video or single image
    if p.is_file():
        ext = p.suffix.lower()
        if ext in VIDEO_EXTS:
            import cv2

            cap = cv2.VideoCapture(str(p))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
            yield from _iter_capture(
                cap, kind="video", total=total, max_frames=max_frames, already_open=True
            )
            return
        # image extension, or fall back to trying PIL for unknown extensions
        try:
            img = Image.open(p).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                f"Unrecognized source (not a known image/video/dir/camera): {source}"
            ) from exc
        yield Frame(0, img, "image", 1)
        return

    raise ValueError(f"Source does not exist: {source}")
