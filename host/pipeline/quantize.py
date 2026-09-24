"""INT8 post-training quantization via NVIDIA ModelOpt into a Q/DQ ONNX graph.

Under TensorRT 11 strong typing the Q/DQ nodes ARE the INT8 declaration: there is no
INT8 builder flag and `trt.IInt8Calibrator` does not exist. Calibration happens here,
in ONNX-land, and TensorRT then compiles the graph verbatim.

Run this in the modelopt conda env — ModelOpt conflicts with the cu13 onnxruntime-gpu
build used everywhere else in this repo.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from modelopt.onnx.quantization import quantize as moq_quantize
from harness.adapter import load_adapter
from harness.calibration import calib_images
from harness.config import load_spec
from harness.paths import require

def int8_path_for(fp32_path: Path) -> Path:
    """`model.onnx` -> `model.int8.onnx`, mirroring the `.fp16.onnx` convention."""
    return fp32_path.with_suffix("").with_suffix(".int8.onnx") \
        if fp32_path.suffixes[:-1] else fp32_path.with_name(f"{fp32_path.stem}.int8.onnx")


def main() -> None:
    ap = argparse.ArgumentParser(description="INT8 PTQ (ModelOpt) -> Q/DQ ONNX")
    ap.add_argument("--spec", default="rtdetr")
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--calib", type=int, default=200, help="calibration images")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="output ONNX (default: <fp32 stem>.int8.onnx)")
    args = ap.parse_args()

    spec = load_spec(args.spec, args.backbone)
    fp32 = require(spec.onnx_path("fp32"), "FP32 ONNX", "export it with host/models/rtdetr/export.py")
    out = Path(args.out) if args.out else int8_path_for(fp32)

    adapter = load_adapter(spec, onnx="fp32", device=args.device)
    print(f"[ptq] calibrating on {args.calib} COCO images")
    calib = np.concatenate(calib_images(adapter, n=args.calib), axis=0)  # (N,3,H,W)
    print(f"[ptq] calibration tensor {calib.shape} {calib.dtype}")

    moq_quantize(
        onnx_path=str(fp32),
        calibration_data={spec.input_name: calib},
        output_path=str(out),
        quantize_mode="int8",
    )
    print(f"[ptq] wrote {out}")
    print(f"[ptq] add to configs/{spec.name}.yaml under backbones.{spec.backbone}.onnx:")
    print(f"        int8: ${{models}}/{out.relative_to(out.parents[1])}")

if __name__ == "__main__":
    main()
