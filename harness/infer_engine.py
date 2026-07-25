"""Single-backend TensorRT inference — a :class:`Backend` over real image/video sources,
alongside ``infer_torch`` (eager) and ``infer_ort`` (ONNX Runtime).

``TrtBackend`` resolves an engine (prebuilt ``.engine`` or built from an ONNX), owns the
``TRTSession``, and returns raw named outputs; preprocess/postprocess stay in the adapter.
``run_inference`` drives this single backend through ``harness.runner`` (which owns the
frame loop, timing, sinks, and summary table).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .adapter import DetectorAdapter
from .trt_runner import TRTSession, build_engine, load_engine, plan_fingerprint


def resolve_engine(*, engine=None, onnx_path=None, tf32: bool = True) -> bytes:
    """Serialized engine bytes from (precedence) a prebuilt engine file, else an ONNX to build."""
    if engine is not None:
        return load_engine(str(engine))
    if onnx_path is not None:
        return build_engine(str(onnx_path), tf32=tf32)
    raise ValueError("resolve_engine needs either engine= (prebuilt) or onnx_path= (to build)")


def _validate_io(sess: TRTSession, adapter: DetectorAdapter) -> None:
    missing = [n for n in adapter.output_names if n not in sess.output_names]
    if missing:
        raise ValueError(
            f"engine is missing outputs {missing}; adapter wants {adapter.output_names}, "
            f"engine has {sess.output_names}")


class TrtBackend:
    """TensorRT backend. ``infer(x)`` returns raw named outputs (numpy)."""

    key = "trt"

    def __init__(self, adapter: DetectorAdapter, *, engine=None, onnx_path=None,
                 tf32: bool = True, label: str = "TensorRT") -> None:
        engine_bytes = resolve_engine(engine=engine, onnx_path=onnx_path, tf32=tf32)
        self.plan = plan_fingerprint(engine_bytes)
        self.size_mb = len(engine_bytes) / 2**20
        self.sess = TRTSession(engine_bytes)
        _validate_io(self.sess, adapter)
        self.label = label
        print(f"[trt] {label}  plan={self.plan}  engine={self.size_mb:.1f} MiB  "
              f"in={self.sess.input_name}  out={self.sess.output_names}")

    def infer(self, x) -> Dict[str, np.ndarray]:
        return self.sess.run(x.detach().cpu().numpy())

    def close(self) -> None:
        self.sess.close()


def run_inference(adapter, source, *, engine=None, onnx_path=None, tf32: bool = True,
                  backend_label: str = "TensorRT", **kw):
    """Single-backend TensorRT run (own window/summary). See harness.runner.run_backends."""
    from .runner import run_backends
    backend = TrtBackend(adapter, engine=engine, onnx_path=onnx_path, tf32=tf32, label=backend_label)
    return run_backends(adapter, source, [backend], **kw)
