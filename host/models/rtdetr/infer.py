
from __future__ import annotations

import argparse
from pathlib import Path

if __package__:                    # python -m models.rtdetr.infer
    from . import _bootstrap       # noqa: F401  puts repo root + this dir on sys.path
else:                              # python host/models/rtdetr/infer.py
    import _bootstrap              # noqa: F401

from harness import run  # noqa: E402  build named backends + run (separate or compare)
from harness.adapter import load_adapter, resolve_device  # noqa: E402
from harness.config import load_spec  # noqa: E402


def _source_label(source: str) -> str:
    if source.isdigit():
        return f"camera{source}"
    p = Path(source)
    if p.is_dir():
        return p.name or "folder"
    return p.stem


def main() -> None:
    parser = argparse.ArgumentParser(description="RT-DETR multi-backend inference (torch / ort / trt)")
    parser.add_argument("--source", required=True,
                        help="image file | folder of images | video file | camera index (e.g. 0)")
    parser.add_argument("--spec", default="rtdetr", help="model spec name or configs/*.yaml path")
    parser.add_argument("--backends", default=None,
                        help="comma list from {torch,ort,trt} (default: spec defaults.backends)")
    parser.add_argument("--compare", action="store_true",
                        help="draw all backends side-by-side in ONE window, with agreement vs the first")
    parser.add_argument("--precision", default=None,
                        help="ONNX variant read by the ort/trt backends, e.g. fp32/fp16/int8 "
                             "(default: the spec's onnx.default). The torch backend builds the "
                             "eager checkpoint and ignores this, so it is always the pre-export "
                             "FP32 reference — never an int8/fp16 arm")
    parser.add_argument("--tf32", dest="tf32", action="store_true", default=None,
                        help="[trt] allow TF32 tensor cores when building")
    parser.add_argument("--no-tf32", dest="tf32", action="store_false",
                        help="[trt] strict IEEE FP32 build")
    parser.add_argument("--no-timing-cache", action="store_true",
                        help="[trt] build without the spec's TensorRT timing cache")
    parser.add_argument("--out", default=None, help="output dir (default: spec outputs.figures)")
    parser.add_argument("--show", action="store_true", help="live cv2 window(s) (auto-disabled if headless)")
    parser.add_argument("--save", action="store_true", help="write annotated output video(s)")
    parser.add_argument("--score-thr", type=float, default=None,
                        help="detection score threshold (default: spec defaults.score_thr)")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    args = parser.parse_args()

    # Model identity — which backbone, which checkpoint, which ONNX — comes from the spec
    # alone. --backends and --precision select *how* to run it, which is the axis this CLI
    # exists to sweep; everything else is read from the YAML so a run is reproducible from
    # the committed spec plus the command line, with no third source of truth.
    spec = load_spec(args.spec)
    device = resolve_device(spec, tag="infer")

    backends = (args.backends.split(",") if args.backends else spec.defaults.get("backends", ["torch"]))
    backends = [b.strip() for b in backends if str(b).strip()]
    providers = spec.defaults.get("providers", ["CUDAExecutionProvider", "CPUExecutionProvider"])
    providers = [str(p).strip() for p in providers if str(p).strip()]

    onnx_path = spec.onnx_path(args.precision)
    out_dir = args.out or spec.outputs.get("figures", onnx_path.parent / "figures")
    adapter = load_adapter(spec, onnx=args.precision, device=device)

    run(
        adapter,
        args.source,
        backends,
        compare=args.compare,
        providers=providers,
        onnx_path=str(onnx_path),
        tf32=spec.default("tf32", args.tf32),
        timing_cache=None if args.no_timing_cache else spec.timing_cache,
        score_thr=spec.default("score_thr", args.score_thr),
        iou_thr=spec.default("iou_thr"),
        show=args.show,
        save=args.save,
        out_dir=str(out_dir),
        warmup=spec.default("warmup", args.warmup),
        max_frames=args.max_frames,
        source_label=_source_label(args.source),
    )


if __name__ == "__main__":
    main()
