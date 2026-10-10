"""Export official Paddle RT-DETRv3 to the host raw-output ONNX contract."""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HOST_ROOT = Path(__file__).resolve().parents[2]
if str(HOST_ROOT) not in sys.path:
    sys.path.insert(0, str(HOST_ROOT))

from harness.config import load_spec  # noqa: E402
from harness.paths import RTDETRV3_ROOT, require  # noqa: E402


def _shape(value_info) -> tuple[int | None, ...]:
    return tuple(d.dim_value if d.HasField("dim_value") else None
                 for d in value_info.type.tensor_type.shape.dim)


def normalize_raw_onnx(source: Path, destination: Path, spec) -> None:
    """Name v3's raw boxes/logits and fix batch 1 at the ONNX interface."""
    import onnx

    model = onnx.load(str(source))
    graph = model.graph
    if len(graph.input) != 1 or len(graph.output) != 2:
        raise ValueError("v3 raw export must have one image input and two raw outputs; "
                         f"got {[x.name for x in graph.input]} -> {[x.name for x in graph.output]}")
    image = graph.input[0]
    image_shape = _shape(image)
    if len(image_shape) != 4 or any(actual not in (None, expected)
                                    for actual, expected in zip(
                                        image_shape, (1, 3, spec.img_size, spec.img_size))):
        raise ValueError(f"unexpected v3 image shape: {image_shape}")
    original_input = image.name
    image.name = spec.input_name
    for node in graph.node:
        node.input[:] = [spec.input_name if name == original_input else name
                         for name in node.input]
    for dim, size in zip(image.type.tensor_type.shape.dim,
                         (1, 3, spec.img_size, spec.img_size)):
        dim.Clear()
        dim.dim_value = size

    by_last_dim = {}
    for output in graph.output:
        shape = _shape(output)
        if len(shape) != 3 or shape[-1] not in (4, spec.num_classes):
            raise ValueError(f"unexpected v3 raw output {output.name}: {shape}")
        if any(actual not in (None, expected) for actual, expected in
               zip(shape[:2], (1, 300))):
            raise ValueError(f"unexpected v3 batch/query dimensions in {output.name}: {shape}")
        if output.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise ValueError(f"v3 raw output {output.name} is not FP32")
        by_last_dim[shape[-1]] = output.name
    if set(by_last_dim) != {4, spec.num_classes}:
        raise ValueError("v3 raw export does not contain distinct boxes and logits")

    del graph.output[:]
    for target, old, shape in (
        ("pred_logits", by_last_dim[spec.num_classes], [1, 300, spec.num_classes]),
        ("pred_boxes", by_last_dim[4], [1, 300, 4]),
    ):
        graph.node.append(onnx.helper.make_node("Identity", [old], [target],
                                                name=f"rename_{target}"))
        graph.output.append(onnx.helper.make_tensor_value_info(
            target, onnx.TensorProto.FLOAT, shape))
    if spec.output_names != ["pred_logits", "pred_boxes"]:
        raise ValueError("v3 requires the raw pred_logits,pred_boxes contract")
    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(destination))


def export(spec, out_path: Path, *, report: bool = True) -> None:
    if spec.name != "rtdetrv3":
        raise ValueError(f"RT-DETRv3 exporter received {spec.name!r}")
    require(RTDETRV3_ROOT / "tools" / "export_model.py", "RT-DETRv3 source",
            "run git submodule update --init third_party/RT-DETRv3")
    require(spec.torch_config, "RT-DETRv3 Paddle config")
    require(spec.ckpt, "RT-DETRv3 Paddle checkpoint")
    if importlib.util.find_spec("paddle") is None or shutil.which("paddle2onnx") is None:
        raise RuntimeError("RT-DETRv3 export needs PaddlePaddle and the paddle2onnx CLI "
                           "in this Python environment; see host/models/rtdetrv3/README.md")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rtdetrv3-export-", dir=out_path.parent) as temp:
        stage = Path(temp)
        command = [sys.executable, str(RTDETRV3_ROOT / "tools" / "export_model.py"),
                   "-c", str(spec.torch_config), "-o", f"weights={spec.ckpt}",
                   "exclude_post_process=True", "use_gpu=False", "trt=True",
                   "--output_dir", str(stage)]
        subprocess.run(command, cwd=RTDETRV3_ROOT, check=True)
        paddle_dir = stage / spec.torch_config.stem
        require(paddle_dir / "model.pdmodel", "exported Paddle inference graph")
        require(paddle_dir / "model.pdiparams", "exported Paddle inference parameters")
        raw_onnx = stage / "raw.onnx"
        subprocess.run([
            "paddle2onnx", "--model_dir", str(paddle_dir),
            "--model_filename", "model.pdmodel",
            "--params_filename", "model.pdiparams",
            "--opset_version", "16", "--save_file", str(raw_onnx),
        ], cwd=RTDETRV3_ROOT, check=True)
        normalize_raw_onnx(raw_onnx, out_path, spec)
    print(f"[rtdetrv3] exported {spec.backbone} -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export Paddle RT-DETRv3 to raw-output ONNX")
    ap.add_argument("--backbone", default=None, choices=load_spec("rtdetrv3").backbones)
    ap.add_argument("--out", default=None, help="registered ONNX variant or output path")
    args = ap.parse_args()
    spec = load_spec("rtdetrv3", args.backbone)
    export(spec, spec.onnx_dest(args.out))


if __name__ == "__main__":
    main()
