"""Shared detection container + the per-model adapter seam.

These live in their own module (not ``compare``) so the single-backend runners
(``infer_torch`` / ``infer_ort`` / ``infer_engine``) can import them without pulling in
the comparison engine — which would create an import cycle.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Protocol, runtime_checkable

import numpy as np
import torch


@dataclass
class Detections:
    """Decoded detections in original-image pixel space."""

    labels: np.ndarray  # (N,) int class ids (contiguous)
    boxes: np.ndarray   # (N, 4) float xyxy
    scores: np.ndarray  # (N,) float

    def __len__(self) -> int:
        return int(np.asarray(self.labels).shape[0])

    def filter(self, score_thr: float) -> "Detections":
        scores = np.asarray(self.scores)
        keep = scores >= score_thr
        return Detections(np.asarray(self.labels)[keep], np.asarray(self.boxes)[keep], scores[keep])


@runtime_checkable
class DetectorAdapter(Protocol):
    """The per-model seam. See ``models/rtdetr/infer.py`` for a concrete adapter.

    A ``run_torch(model, x) -> dict[str, np.ndarray]`` method is optional; when absent
    the torch backend uses ``infer_torch.run_torch_default``, which maps the eager
    forward result onto ``output_names``.
    """

    name: str
    class_names: List[str]
    input_name: str
    output_names: List[str]
    onnx_path: Path
    device: str

    def build_torch(self) -> torch.nn.Module: ...
    def preprocess(self, frame) -> "tuple[torch.Tensor, dict]": ...
    def postprocess(self, named: Dict[str, np.ndarray], meta: dict) -> Detections: ...
