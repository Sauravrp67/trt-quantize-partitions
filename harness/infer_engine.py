"""Single-backend TensorRT inference engine over real image / video sources.

The deployment-style counterpart to ``harness/compare.py`` (which compares torch vs ORT).
It resolves a TensorRT engine, iterates any source with :func:`harness.sources.open_source`,
and per frame runs ``preprocess -> TRTSession.run -> postprocess -> filter -> draw``. A
model plugs in through the same :class:`harness.compare.DetectorAdapter` seam; only
``preprocess``/``postprocess``/``class_names``/``output_names`` are used (``build_torch`` is
not — this path never touches the eager model).

Seam for a future TRT-vs-TRT diagnostic (design spec §②): the engine-resolution helper
:func:`resolve_engine` and the per-frame :func:`infer_frame` are factored out so a later
``compare_engines(adapter, source, engine_a, engine_b, ...)`` can reuse them and render
with :func:`harness.visualize.draw_side_by_side`. That mode is intentionally not built yet.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from .compare import Detections, DetectorAdapter
from .metrics import StreamTimer
from .sources import open_source
from .trt_runner import TRTSession, build_engine, load_engine, plan_fingerprint
from .visualize import draw_single, make_sink


def resolve_engine(*, engine=None, onnx_path=None, tf32: bool = True) -> bytes:
    """Serialized engine bytes from (precedence) a prebuilt engine file, else an ONNX to build."""
    if engine is not None:
        return load_engine(str(engine))
    if onnx_path is not None:
        return build_engine(str(onnx_path), tf32=tf32)
    raise ValueError("resolve_engine needs either engine= (prebuilt) or onnx_path= (to build)")


def infer_frame(sess: TRTSession, adapter: DetectorAdapter, pil, score_thr: float) -> Detections:
    """One frame: preprocess -> TRT run -> postprocess -> score filter."""
    x, meta = adapter.preprocess(pil)
    named = sess.run(x.detach().cpu().numpy())
    return adapter.postprocess(named, meta).filter(score_thr)


def _validate_io(sess: TRTSession, adapter: DetectorAdapter) -> None:
    missing = [n for n in adapter.output_names if n not in sess.output_names]
    if missing:
        raise ValueError(
            f"engine is missing outputs {missing}; adapter wants {adapter.output_names}, "
            f"engine has {sess.output_names}"
        )


class InferenceAggregator:
    """Accumulate per-frame reports into an end-of-run summary (single stream)."""

    def __init__(self) -> None:
        self.frames = 0
        self.total_dets = 0

    def add(self, report: dict) -> None:
        self.frames += 1
        self.total_dets += report["n"]

    def summary(self, timer: StreamTimer, backend: str) -> str:
        a = timer.agg
        return "\n".join(
            [
                "",
                "===================== TensorRT inference summary =====================",
                f" backend                : {backend}",
                f" frames processed       : {self.frames}",
                f" detections total/mean  : {self.total_dets} / "
                f"{(self.total_dets / self.frames) if self.frames else 0.0:.1f} per frame",
                f" latency (ms)           : mean {a.mean():.2f}  median {a.median():.2f}  p90 {a.p90():.2f}",
                f" throughput             : {a.fps():.1f} FPS (mean, end-to-end wall-clock)",
                " note: authoritative latency/throughput is trtexec; this is a display measure.",
                "======================================================================",
            ]
        )


def run_inference(
    adapter: DetectorAdapter,
    source,
    *,
    engine=None,
    onnx_path=None,
    tf32: bool = True,
    score_thr: float = 0.6,
    show: bool = False,
    save: bool = False,
    out_dir="results/figures/infer",
    warmup: int = 1,
    max_frames: Optional[int] = None,
    source_label: str = "source",
    backend_label: str = "TRT · FP32",
) -> InferenceAggregator:
    """Run single-backend TensorRT inference over ``source`` and print a report."""
    engine_bytes = resolve_engine(engine=engine, onnx_path=onnx_path, tf32=tf32)
    print(f"[infer] backend={backend_label}  plan={plan_fingerprint(engine_bytes)}  "
          f"engine={len(engine_bytes) / 2**20:.1f} MiB")

    sess = TRTSession(engine_bytes)
    _validate_io(sess, adapter)
    print(f"[infer] input={sess.input_name}  outputs={sess.output_names}")

    timer = StreamTimer(warmup=warmup)
    agg = InferenceAggregator()
    sink = None
    try:
        for frame in open_source(source, max_frames=max_frames):
            x, meta = adapter.preprocess(frame.pil)
            with timer.time():
                named = sess.run(x.detach().cpu().numpy())
            det = adapter.postprocess(named, meta).filter(score_thr)

            report = {"id": frame.id, "n": len(det), "backend": backend_label, "ms": timer.last_ms}
            agg.add(report)
            print(f"[frame {frame.id:>5}] {len(det)} det  "
                  f"| {timer.last_ms:.1f}ms  {(1000.0 / timer.last_ms if timer.last_ms else 0.0):.1f} FPS")

            if sink is None:
                sink = make_sink(frame.kind, show=show, save=save, out_dir=out_dir,
                                 source_stem=source_label, stem_suffix="_trt")
            panel = draw_single(frame.pil, det, report, adapter.class_names, score_thr=score_thr)
            if not sink.consume(frame, panel, report):
                break
    finally:
        if sink is not None:
            sink.close()
        sess.close()

    print(agg.summary(timer, backend_label))
    return agg
