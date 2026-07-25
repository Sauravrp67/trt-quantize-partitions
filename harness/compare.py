"""Backend orchestration: build one-or-more backends and run them (separate or compared).

The per-backend inference logic now lives in the single-backend modules
(``infer_torch`` / ``infer_ort`` / ``infer_engine``); this module just wires them together
and hands off to :func:`harness.runner.run_backends`. ``Detections`` / ``DetectorAdapter``
are re-exported from :mod:`harness.adapter` so existing ``from harness.compare import
Detections`` imports keep working.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from .adapter import Detections, DetectorAdapter        # re-exported for back-compat
from .infer_engine import TrtBackend
from .infer_ort import OrtBackend
from .infer_torch import TorchBackend, run_torch_default  # noqa: F401  re-exported
from .runner import match_detections, run_backends        # noqa: F401  re-exported

__all__ = ["Detections", "DetectorAdapter", "build_backend", "run", "run_backends",
           "match_detections", "run_torch_default"]


def build_backend(name: str, adapter: DetectorAdapter, *, providers: Optional[List[str]] = None,
                  engine=None, onnx_path=None, tf32: bool = True):
    """Construct one backend by short name: ``torch`` | ``ort`` | ``trt``."""
    key = name.strip().lower()
    if key == "torch":
        return TorchBackend(adapter)
    if key == "ort":
        return OrtBackend(adapter, providers=providers)
    if key == "trt":
        return TrtBackend(adapter, engine=engine, onnx_path=onnx_path, tf32=tf32)
    raise ValueError(f"unknown backend {name!r}; choose from: torch, ort, trt")


def run(adapter: DetectorAdapter, source, backend_names: Sequence[str] = ("torch", "ort"), *,
        compare: bool = False, providers: Optional[List[str]] = None, engine=None,
        onnx_path=None, tf32: bool = True, **kw):
    """Build ``backend_names`` and run them over ``source`` (separate sinks, or one compare sink).

    Extra kwargs (score_thr, iou_thr, show, save, out_dir, warmup, max_frames, source_label)
    pass straight through to :func:`harness.runner.run_backends`.
    """
    backends = [build_backend(n, adapter, providers=providers, engine=engine,
                              onnx_path=onnx_path, tf32=tf32) for n in backend_names]
    return run_backends(adapter, source, backends, compare=compare, **kw)
