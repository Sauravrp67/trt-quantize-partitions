"""The model-agnostic half of the adapter seam.

Nothing in this module knows what a detector is made of. It defines what every model
must provide (:class:`BaseAdapter`), what every model returns (:class:`Detections`), and
the structural type the backend runners accept (:class:`DetectorAdapter`).

These live here rather than in ``compare`` so the single-backend runners
(``infer_torch`` / ``infer_ort`` / ``infer_engine``) can import them without pulling in
the comparison engine — which would create an import cycle.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol, runtime_checkable

import numpy as np
import torch

from ..config import ModelSpec, load_spec


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
    """What the backend runners require of an adapter, structurally.

    :class:`BaseAdapter` satisfies this, but the protocol is deliberately kept separate:
    a runner should accept anything shaped like an adapter, not only this repo's subclasses.

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


class BaseAdapter(ABC):
    """Base class for every model adapter.

    Subclasses supply the three model-specific behaviours — build the eager model,
    turn a frame into a tensor, turn raw outputs into :class:`Detections` — and inherit
    the spec plumbing, which is identical for every model because a
    :class:`~harness.config.ModelSpec` is the only place identity is declared.

    ``spec_name`` names the ``configs/<name>.yaml`` used when a caller constructs the
    adapter without a spec, and is the key this adapter is registered under.
    """

    spec_name: str = ""

    def __init__(self, spec: Optional[ModelSpec] = None, *, onnx=None,
                 device: Optional[str] = None) -> None:
        if spec is None:
            if not self.spec_name:
                raise ValueError(f"{type(self).__name__} has no spec_name; pass a ModelSpec")
            spec = load_spec(self.spec_name)
        self.spec = spec
        self.name = spec.name
        self.class_names = spec.class_names
        self.input_name = spec.input_name
        self.output_names = list(spec.output_names)
        self.img_size = spec.img_size
        self.onnx_path = spec.onnx_path(onnx)
        self.device = device or spec.defaults.get("device", "cuda")

    @abstractmethod
    def build_torch(self) -> torch.nn.Module:
        """The eager model, on ``self.device``, in eval/deploy mode."""

    @abstractmethod
    def preprocess(self, frame) -> "tuple[torch.Tensor, dict]":
        """A frame -> (network input tensor, meta needed to decode its outputs)."""

    @abstractmethod
    def postprocess(self, named: Dict[str, np.ndarray], meta: dict) -> Detections:
        """Raw named outputs -> pixel-space :class:`Detections`."""


def resolve_device(spec: ModelSpec, override: Optional[str] = None,
                   tag: Optional[str] = None) -> str:
    """Requested device, downgraded to CPU (with a notice) when CUDA is unavailable."""
    device = override or spec.defaults.get("device", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[{tag or spec.name}] CUDA unavailable; falling back to CPU")
        return "cpu"
    return device
