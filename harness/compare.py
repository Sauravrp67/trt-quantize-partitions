
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, runtime_checkable

import numpy as np
import onnxruntime as ort
import torch

from .metrics import LatencyMeter
from .parity import _flatten_torch_output
from .sources import open_source
from .visualize import draw_side_by_side, make_sink


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
    the engine uses :func:`run_torch_default`, which maps the eager forward result onto
    ``output_names``.
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


def run_torch_default(
    model: torch.nn.Module, x: torch.Tensor, output_names: List[str], device: str
) -> Dict[str, np.ndarray]:
    """Default eager forward: no-grad, detach, map outputs onto ``output_names``.

    Reuses ``parity._flatten_torch_output`` so dict / tuple / single-tensor forward
    results are all normalized before matching by name (then by position).
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


def _validate_io(session: ort.InferenceSession, adapter: DetectorAdapter) -> None:
    in_names = [i.name for i in session.get_inputs()]
    if adapter.input_name not in in_names:
        raise ValueError(f"ONNX input '{adapter.input_name}' not found; graph inputs = {in_names}")
    out_names = {o.name for o in session.get_outputs()}
    missing = [n for n in adapter.output_names if n not in out_names]
    if missing:
        raise ValueError(f"ONNX graph missing outputs {missing}; has {sorted(out_names)}")


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of one box ``a`` (4,) against many boxes ``b`` (M, 4)."""
    x1 = np.maximum(a[0], b[:, 0])
    y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2])
    y2 = np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0)


def match_detections(t_det: Detections, o_det: Detections, iou_thr: float, frame_id: int) -> dict:
    """Greedy same-label IoU match between torch and ORT detections.

    Same-label + IoU matching is robust to the top-K reshuffle (a positional zip would
    misalign whenever a background query straddles the cutoff). Reports counts and the
    worst box/score disagreement over matched pairs.
    """
    used_o: set = set()
    matched = 0
    max_box_diff = 0.0
    max_score_diff = 0.0
    for i in range(len(t_det)):
        tb = np.asarray(t_det.boxes[i], dtype=float)
        tl = int(t_det.labels[i])
        cand = [j for j in range(len(o_det)) if j not in used_o and int(o_det.labels[j]) == tl]
        if not cand:
            continue
        cand_boxes = np.asarray(o_det.boxes)[cand].astype(float)
        ious = _iou(tb, cand_boxes)
        k = int(np.argmax(ious))
        if ious[k] >= iou_thr:
            j = cand[k]
            used_o.add(j)
            matched += 1
            max_box_diff = max(max_box_diff, float(np.max(np.abs(tb - np.asarray(o_det.boxes[j], dtype=float)))))
            max_score_diff = max(max_score_diff, float(abs(float(t_det.scores[i]) - float(o_det.scores[j]))))
    return {
        "id": frame_id,
        "n_torch": len(t_det),
        "n_ort": len(o_det),
        "matched": matched,
        "unmatched_torch": len(t_det) - matched,
        "unmatched_ort": len(o_det) - matched,
        "max_box_diff": max_box_diff,
        "max_score_diff": max_score_diff,
    }


class ParityAggregator:
    """Accumulate per-frame reports into an end-of-run summary."""

    def __init__(self) -> None:
        self.frames = 0
        self.tot_torch = 0
        self.tot_ort = 0
        self.matched = 0
        self.mismatch_frames = 0
        self.box_diffs: List[float] = []
        self.score_diffs: List[float] = []

    def add(self, r: dict) -> None:
        self.frames += 1
        self.tot_torch += r["n_torch"]
        self.tot_ort += r["n_ort"]
        self.matched += r["matched"]
        if r["unmatched_torch"] or r["unmatched_ort"]:
            self.mismatch_frames += 1
        if r["matched"]:
            self.box_diffs.append(r["max_box_diff"])
            self.score_diffs.append(r["max_score_diff"])

    def summary(self, lat: LatencyMeter) -> str:
        def stats(xs: List[float]):
            return (0.0, 0.0) if not xs else (float(np.mean(xs)), float(np.max(xs)))

        mean_box, max_box = stats(self.box_diffs)
        mean_score, max_score = stats(self.score_diffs)
        agree = (self.matched / self.tot_torch * 100.0) if self.tot_torch else 100.0
        return "\n".join(
            [
                "",
                "================= torch vs ONNX Runtime summary =================",
                f" frames processed       : {self.frames}",
                f" detections torch/ORT   : {self.tot_torch} / {self.tot_ort}",
                f" matched detections     : {self.matched}  ({agree:.1f}% of torch)",
                f" frames with mismatch   : {self.mismatch_frames}",
                f" box diff   mean/max    : {mean_box:.3f} / {max_box:.3f} px",
                f" score diff mean/max    : {mean_score:.4f} / {max_score:.4f}",
                f" latency torch (ms)     : mean {lat.torch_ms.mean():.2f}  median {lat.torch_ms.median():.2f}  p90 {lat.torch_ms.p90():.2f}",
                f" latency ORT   (ms)     : mean {lat.ort_ms.mean():.2f}  median {lat.ort_ms.median():.2f}  p90 {lat.ort_ms.p90():.2f}",
                f" throughput torch/ORT   : {lat.torch_ms.fps():.1f} / {lat.ort_ms.fps():.1f} FPS (mean)",
                "=================================================================",
            ]
        )


def _print_frame(r: dict, last: dict) -> None:
    tms = last.get("torch_ms")
    oms = last.get("ort_ms")
    tstr = f"{tms:.1f}" if tms is not None else "  -"
    ostr = f"{oms:.1f}" if oms is not None else "  -"
    print(
        f"[frame {r['id']:>5}] T:{r['n_torch']} O:{r['n_ort']} matched:{r['matched']} "
        f"dbox:{r['max_box_diff']:.2f}px dscore:{r['max_score_diff']:.3f} "
        f"| torch {tstr}ms  ort {ostr}ms"
    )


def run(
    adapter: DetectorAdapter,
    source,
    *,
    score_thr: float = 0.6,
    iou_thr: float = 0.5,
    providers: Optional[List[str]] = None,
    show: bool = False,
    save: bool = False,
    out_dir="results/figures/parity",
    warmup: int = 1,
    max_frames: Optional[int] = None,
    source_label: str = "source",
) -> ParityAggregator:
    """Run the side-by-side torch/ORT comparison over ``source`` and print a report."""
    providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
    model = adapter.build_torch()
    session = ort.InferenceSession(str(adapter.onnx_path), providers=providers)
    _validate_io(session, adapter)
    print(f"[compare] torch device={adapter.device}  ORT providers={session.get_providers()}")

    lat = LatencyMeter(device=adapter.device, warmup=warmup)
    agg = ParityAggregator()
    custom_run_torch: Optional[Callable] = getattr(adapter, "run_torch", None)
    sink = None

    try:
        for frame in open_source(source, max_frames=max_frames):
            x, meta = adapter.preprocess(frame.pil)

            with lat.torch_timer():
                if custom_run_torch is None:
                    t_named = run_torch_default(model, x, adapter.output_names, adapter.device)
                else:
                    t_named = custom_run_torch(model, x)

            x_np = x.detach().cpu().numpy()
            with lat.ort_timer():
                o_list = session.run(adapter.output_names, {adapter.input_name: x_np})
            o_named = dict(zip(adapter.output_names, o_list))

            t_det = adapter.postprocess(t_named, meta).filter(score_thr)
            o_det = adapter.postprocess(o_named, meta).filter(score_thr)

            report = match_detections(t_det, o_det, iou_thr, frame.id)
            report.update(lat.last())  # torch_ms / ort_ms -> per-panel FPS overlay
            agg.add(report)
            _print_frame(report, lat.last())

            if sink is None:
                sink = make_sink(frame.kind, show=show, save=save, out_dir=out_dir, source_stem=source_label)
            panel = draw_side_by_side(
                frame.pil, t_det, o_det, report, adapter.class_names, score_thr=score_thr
            )
            if not sink.consume(frame, panel, report):
                break
    finally:
        if sink is not None:
            sink.close()

    print(agg.summary(lat))
    return agg
