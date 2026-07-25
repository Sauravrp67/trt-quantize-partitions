"""Single-backend eager-PyTorch inference — a :class:`Backend` over real image/video
sources, mirroring ``infer_engine`` (TensorRT) and ``infer_ort`` (ONNX Runtime).

The backend owns the eager model and returns raw named outputs; preprocessing and
postprocessing stay in the adapter (shared across all backends). ``run_inference`` is a
one-line convenience that drives this single backend through ``harness.runner``.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from .adapter import DetectorAdapter
from .parity import _flatten_torch_output


def run_torch_default(
    model: torch.nn.Module, x: torch.Tensor, output_names: List[str], device: str
) -> Dict[str, np.ndarray]:
    """Default eager forward: no-grad, detach, map outputs onto ``output_names``.

    Reuses ``parity._flatten_torch_output`` so dict / tuple / single-tensor forward
    results are all normalized before matching by name (then by position). The final
    ``.cpu()`` also synchronizes the device, so a wall-clock timer around this call is a
    fair, fully-resolved latency measurement.
    """
    with torch.no_grad():
        out = model(x.to(device))
    named = _flatten_torch_output(out)
    items = list(named.items())
    result: Dict[str, np.ndarray] = {}
    for i, name in enumerate(output_names):
        if name in named:
            tensor = named[name]
        elif i < len(items):
            tensor = items[i][1]
        else:
            raise KeyError(f"eager output has no tensor for ONNX output '{name}'")
        result[name] = tensor.detach().cpu().numpy()
    return result


class TorchBackend:
    """Eager PyTorch backend. ``infer(x)`` returns raw named outputs (numpy)."""

    key = "torch"

    def __init__(self, adapter: DetectorAdapter, *, device: Optional[str] = None) -> None:
        self.label = "PyTorch"
        self.device = device or adapter.device
        self.output_names = adapter.output_names
        self.model = adapter.build_torch()
        self._custom: Optional[Callable] = getattr(adapter, "run_torch", None)

    def infer(self, x: torch.Tensor) -> Dict[str, np.ndarray]:
        if self._custom is not None:
            return self._custom(self.model, x)
        return run_torch_default(self.model, x, self.output_names, self.device)

    def close(self) -> None:  # nothing to release
        pass


def run_inference(adapter, source, **kw):
    """Single-backend eager-PyTorch run (own window/summary). See harness.runner.run_backends."""
    from .runner import run_backends
    return run_backends(adapter, source, [TorchBackend(adapter)], **kw)
