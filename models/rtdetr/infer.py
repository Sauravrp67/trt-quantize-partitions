
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401  repo root on sys.path (must precede harness imports)

from adapter import RTDETRAdapter, resolve_device  # noqa: E402
from harness import run  # noqa: E402  build named backends + run (separate or compare)
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
    parser.add_argument("--onnx", default=None,
                        help="ONNX for the ort/trt backends: a variant name (fp32/fp16) or a path")
    parser.add_argument("--engine", default=None,
                        help="[trt] prebuilt .engine to load; if omitted, build from --onnx")
    parser.add_argument("--tf32", dest="tf32", action="store_true", default=None,
                        help="[trt] allow TF32 tensor cores when building")
    parser.add_argument("--no-tf32", dest="tf32", action="store_false",
                        help="[trt] strict IEEE FP32 build")
    parser.add_argument("--device", default=None)
    parser.add_argument("--score-thr", type=float, default=None)
    parser.add_argument("--iou-thr", type=float, default=None, help="IoU for --compare agreement")
    parser.add_argument("--providers", default=None, help="comma list of ONNX Runtime providers")
    parser.add_argument("--out", default=None, help="output dir (default: spec outputs.figures)")
    parser.add_argument("--show", action="store_true", help="live cv2 window(s) (auto-disabled if headless)")
    parser.add_argument("--save", action="store_true", help="write annotated output video(s)")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    args = parser.parse_args()

    spec = load_spec(args.spec)
    device = resolve_device(spec, args.device, tag="infer")

    backends = (args.backends.split(",") if args.backends else spec.defaults.get("backends", ["torch"]))
    backends = [b.strip() for b in backends if str(b).strip()]
    providers = (args.providers.split(",") if args.providers
                 else spec.defaults.get("providers", ["CUDAExecutionProvider", "CPUExecutionProvider"]))
    providers = [p.strip() for p in providers if str(p).strip()]

    onnx_path = spec.onnx_path(args.onnx)
    out_dir = args.out or spec.outputs.get("figures", spec.onnx_path().parent / "figures")
    adapter = RTDETRAdapter(spec, onnx=args.onnx, device=device)

    run(
        adapter,
        args.source,
        backends,
        compare=args.compare,
        providers=providers,
        engine=args.engine,
        onnx_path=str(onnx_path),
        tf32=spec.default("tf32", args.tf32),
        score_thr=spec.default("score_thr", args.score_thr),
        iou_thr=spec.default("iou_thr", args.iou_thr),
        show=args.show,
        save=args.save,
        out_dir=str(out_dir),
        warmup=spec.default("warmup", args.warmup),
        max_frames=args.max_frames,
        source_label=_source_label(args.source),
    )


if __name__ == "__main__":
    main()
