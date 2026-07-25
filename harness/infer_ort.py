"""Single-backend ONNX Runtime inference — a :class:`Backend` over real image/video
sources, mirroring ``infer_engine`` (TensorRT) and ``infer_torch`` (eager).

Owns the ORT session; ``infer(x)`` returns raw named outputs. ``run_inference`` drives
this single backend through ``harness.runner``.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import onnxruntime as ort

from .adapter import DetectorAdapter


def _validate_io(session: ort.InferenceSession, adapter: DetectorAdapter) -> None:
    in_names = [i.name for i in session.get_inputs()]
    if adapter.input_name not in in_names:
        raise ValueError(f"ONNX input '{adapter.input_name}' not found; graph inputs = {in_names}")
    out_names = {o.name for o in session.get_outputs()}
    missing = [n for n in adapter.output_names if n not in out_names]
    if missing:
        raise ValueError(f"ONNX graph missing outputs {missing}; has {sorted(out_names)}")


class OrtBackend:
    """ONNX Runtime backend. ``infer(x)`` returns raw named outputs (numpy)."""

    key = "ort"

    def __init__(self, adapter: DetectorAdapter, *, providers: Optional[List[str]] = None) -> None:
        self.providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.input_name = adapter.input_name
        self.output_names = adapter.output_names
        self.session = ort.InferenceSession(str(adapter.onnx_path), providers=self.providers)
        _validate_io(self.session, adapter)
        active = self.session.get_providers()
        ep = "CUDA" if "CUDAExecutionProvider" in active[:1] else active[0].replace("ExecutionProvider", "")
        self.label = f"ONNX Runtime ({ep})"

    def infer(self, x) -> Dict[str, np.ndarray]:
        x_np = np.ascontiguousarray(x.detach().cpu().numpy(), dtype=np.float32)
        out = self.session.run(self.output_names, {self.input_name: x_np})
        return dict(zip(self.output_names, out))

    def close(self) -> None:
        pass


def run_inference(adapter, source, *, providers=None, **kw):
    """Single-backend ONNX Runtime run (own window/summary). See harness.runner.run_backends."""
    from .runner import run_backends
    return run_backends(adapter, source, [OrtBackend(adapter, providers=providers)], **kw)
