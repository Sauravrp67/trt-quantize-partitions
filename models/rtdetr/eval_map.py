from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401  repo root on sys.path (must precede harness imports)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from adapter import RTDETRAdapter, resolve_device  # noqa: E402
from harness.coco_eval import evaluate  # noqa: E402
from harness.compare import build_backend  # noqa: E402
from harness.config import load_spec  # noqa: E402
from harness.paths import COCO_VAL_ANN, COCO_VAL_IMAGES, require  # noqa: E402


def _run_fn(backend):
    """Adapt a harness backend to ``coco_eval.evaluate``'s ``run_fn(x_np) -> named dict``.

    ``evaluate`` hands out numpy (it is backend-neutral) while the backends take the
    adapter's tensor, so the array is re-wrapped without a copy.
    """
    def run(x_np: np.ndarray):
        return backend.infer(torch.from_numpy(np.ascontiguousarray(x_np)))
    return run


def _check_dataset() -> None:
    hint = ("expected data/coco/val2017/*.jpg and data/coco/annotations/"
            "instances_val2017.json — or set $TRTQP_COCO to an existing COCO root")
    require(COCO_VAL_IMAGES, "COCO val2017 images", hint)
    require(COCO_VAL_ANN, "COCO val2017 annotations", hint)


def _write_table(rows, out_path: Path, *, spec, backend_label: str, n_images: int) -> Path:
    """Markdown table; Δ columns are measured against the first row (the reference model)."""
    ref = rows[0]
    lines = [f"# COCO mAP — {spec.name}", "",
             f"backend: `{backend_label}` · images: {n_images} of val2017", "",
             "| model | mAP@50-95 | mAP@50 | Δ mAP@50-95 | Δ mAP@50 |",
             "|---|---|---|---|---|"]
    for i, r in enumerate(rows):
        if i == 0:
            d95 = d50 = "ref"
        else:
            d95 = f"{r['mAP5095'] - ref['mAP5095']:+.4f}"
            d50 = f"{r['mAP50'] - ref['mAP50']:+.4f}"
        lines.append(f"| `{r['model']}` | {r['mAP5095']:.4f} | {r['mAP50']:.4f} | {d95} | {d50} |")
    lines += ["", "_Accuracy only — no score threshold applied (COCOeval sweeps them). "
              "For latency see `models/rtdetr/infer.py` or `trtexec` with locked clocks._", ""]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="COCO mAP for RT-DETR precision variants")
    parser.add_argument("--spec", default="rtdetr", help="model spec name or configs/*.yaml path")
    parser.add_argument("--onnx", nargs="+", default=None,
                        help="variant names (fp32 fp16 ...) or paths; the first is the Δ reference")
    parser.add_argument("--backend", default="ort", choices=["ort", "trt", "torch"],
                        help="ort/trt read the ONNX; torch evaluates the eager checkpoint "
                             "(the pre-export reference — ignores --onnx)")
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate only the first N val images (default: all 5000)")
    parser.add_argument("--tf32", dest="tf32", action="store_true", default=None,
                        help="[trt] allow TF32 tensor cores when building")
    parser.add_argument("--no-tf32", dest="tf32", action="store_false", help="[trt] strict IEEE FP32")
    parser.add_argument("--providers", default=None, help="comma list of ONNX Runtime providers")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None, help="markdown table path (default: spec outputs.table)")
    args = parser.parse_args()

    _check_dataset()
    spec = load_spec(args.spec)
    device = resolve_device(spec, args.device, tag="eval")

    providers = (args.providers.split(",") if args.providers
                 else spec.defaults.get("providers", ["CUDAExecutionProvider", "CPUExecutionProvider"]))
    providers = [p.strip() for p in providers if str(p).strip()]
    tf32 = spec.default("tf32", args.tf32)

    refs = args.onnx or [spec.default_onnx]
    if args.backend == "torch":
        refs = refs[:1]   # eager weights are the same regardless of --onnx
    onnx_paths = [spec.onnx_path(r) for r in refs]
    if args.backend != "torch":
        for path in onnx_paths:
            require(path, "ONNX model", "build it with models/rtdetr/export.py")

    # one adapter (loads the checkpoint + postprocessor once); the ONNX it points at is
    # swapped per model, since that is what the ort/trt backends read on construction
    adapter = RTDETRAdapter(spec, onnx=refs[0], device=device)

    rows = []
    backend_label = args.backend
    n_images = 0
    for ref, path in zip(refs, onnx_paths):
        adapter.onnx_path = path
        label = "eager checkpoint" if args.backend == "torch" else path.name
        print(f"\n[eval] {args.backend} · {label} · {args.limit or 'all'} images")
        backend = build_backend(args.backend, adapter, providers=providers,
                                onnx_path=str(path), tf32=tf32)
        backend_label = backend.label
        try:
            stats = evaluate(_run_fn(backend), adapter, limit=args.limit,
                             desc=f"{args.backend}·{spec.variant_of(path)}")
        finally:
            backend.close()
        n_images = stats["n_images"]
        rows.append({"model": label, "mAP5095": stats["mAP5095"], "mAP50": stats["mAP50"]})
        print(f"[eval] {label}: mAP@50-95={stats['mAP5095']:.4f}  "
              f"mAP@50={stats['mAP50']:.4f}  ({n_images} images)")

    print("\n===================== mAP summary =====================")
    print(f" backend: {backend_label}   images: {n_images}")
    for i, r in enumerate(rows):
        delta = "" if i == 0 else f"   Δ={r['mAP5095'] - rows[0]['mAP5095']:+.4f}"
        print(f" {r['model']:24s} mAP@50-95 {r['mAP5095']:.4f}   mAP@50 {r['mAP50']:.4f}{delta}")
    print("=======================================================")

    out = Path(args.out) if args.out else spec.outputs["table"]
    print(f"[eval] wrote table -> {_write_table(rows, out, spec=spec, backend_label=backend_label, n_images=n_images)}")


if __name__ == "__main__":
    main()
