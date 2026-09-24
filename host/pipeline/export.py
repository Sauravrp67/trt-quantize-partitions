"""Export a model's registered FP32 ONNX artifact.

The model-specific ``models/<name>/export.py`` owns checkpoint loading and
PyTorch export. This stage checks the shared detector contract and records the
artifact that later pipeline stages consume.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib import import_module
import json
import os
from pathlib import Path
import re
import sys
import tempfile

HOST_ROOT = Path(__file__).resolve().parents[1]
if str(HOST_ROOT) not in sys.path:
    sys.path.insert(0, str(HOST_ROOT))

from harness.config import ModelSpec, load_spec  # noqa: E402


def _exporter_for(spec: ModelSpec):
    """Find the model's export function without importing it for other models."""
    if not re.fullmatch(r"[a-z][a-z0-9_]*", spec.name):
        raise ValueError(f"invalid model name for export: {spec.name!r}")
    source = HOST_ROOT / "models" / spec.name / "export.py"
    if not source.is_file():
        raise ValueError(f"no exporter for model {spec.name!r}; expected {source}")
    exporter = getattr(import_module(f"models.{spec.name}.export"), "export", None)
    if not callable(exporter):
        raise TypeError(f"{source} must expose export(spec, out_path, *, report=...)")
    return exporter


def _shape(value_info) -> list[int]:
    dims = value_info.type.tensor_type.shape.dim
    if not dims or any(not d.HasField("dim_value") or d.dim_value <= 0 for d in dims):
        raise ValueError(f"{value_info.name}: expected a fully static, positive ONNX shape")
    return [d.dim_value for d in dims]


def _inspect_onnx(path: Path, spec: ModelSpec) -> dict:
    import onnx

    onnx.checker.check_model(str(path))
    model = onnx.load(str(path), load_external_data=False)
    inputs = list(model.graph.input)
    outputs = list(model.graph.output)
    if len(inputs) != 1 or inputs[0].name != spec.input_name:
        raise ValueError(f"expected one input named {spec.input_name!r}; got {[x.name for x in inputs]}")
    if [x.name for x in outputs] != spec.output_names:
        raise ValueError(f"expected outputs {spec.output_names}; got {[x.name for x in outputs]}")

    expected_input_shape = [1, 3, spec.img_size, spec.img_size]
    if _shape(inputs[0]) != expected_input_shape:
        raise ValueError(f"{spec.input_name}: expected {expected_input_shape}; got {_shape(inputs[0])}")
    for tensor in [*inputs, *outputs]:
        shape = _shape(tensor)
        if shape[0] != 1:
            raise ValueError(f"{tensor.name}: expected batch dimension 1; got {shape[0]}")
        if tensor.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise ValueError(f"{tensor.name}: expected FP32 tensor I/O")

    return {
        "opsets": {entry.domain or "ai.onnx": entry.version for entry in model.opset_import},
        "input": {"name": inputs[0].name, "shape": _shape(inputs[0]), "dtype": "float32"},
        "outputs": [
            {"name": tensor.name, "shape": _shape(tensor), "dtype": "float32"}
            for tensor in outputs
        ],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(spec: ModelSpec, *, report: bool = True) -> Path:
    """Export FP32, check the ONNX contract, and write adjacent JSON metadata."""
    if spec.batch != 1:
        raise ValueError(f"export requires static batch=1; {spec.backbone} specifies {spec.batch}")
    if not spec.output_names or len(set(spec.output_names)) != len(spec.output_names):
        raise ValueError("the model spec needs distinct ONNX output names")
    if spec.input_name in spec.output_names:
        raise ValueError("ONNX input and output names must be distinct")
    path = spec.onnx_path("fp32")
    path.parent.mkdir(parents=True, exist_ok=True)

    exporter = _exporter_for(spec)
    exporter(spec, path, report=report)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"{spec.name} exporter did not write a nonempty ONNX file: {path}")

    contract = _inspect_onnx(path, spec)
    metadata = {
        "schema_version": 1,
        "model": spec.name,
        "backbone": spec.backbone,
        "precision": "fp32",
        "spec": str(spec.source) if spec.source else None,
        "checkpoint": str(spec.ckpt),
        "onnx": str(path),
        "onnx_sha256": _sha256(path),
        "classes": spec.class_names,
        **contract,
    }
    manifest = path.with_name(f"{path.stem}.export.json")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=manifest.parent, prefix=f".{manifest.name}.",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(metadata, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, manifest)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"[export] validated {path}; metadata -> {manifest}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a registered model to static FP32 ONNX")
    parser.add_argument("--spec", default="rtdetr", help="model name or host/configs/*.yaml path")
    parser.add_argument("--backbone", default=None, help="backbone in the model spec")
    parser.add_argument("--no-report", action="store_true", help="skip the model exporter's report")
    args = parser.parse_args()
    run(load_spec(args.spec, args.backbone), report=not args.no_report)


if __name__ == "__main__":
    main()
