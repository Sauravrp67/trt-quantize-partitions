"""Prepare a separate, traceable FP32 ONNX graph for later pipeline steps.

Graph surgery here means removing unreachable nodes, sorting live nodes, and
giving unnamed nodes deterministic names. It does not fold constants or change
precision. Operator lists in the manifest are inspection candidates, not
measured sensitivity or an automatic mixed-precision partition.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

HOST_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = HOST_ROOT.parent
if str(HOST_ROOT) not in sys.path:
    sys.path.insert(0, str(HOST_ROOT))

from harness.config import ModelSpec, load_spec  # noqa: E402

NUMERICAL_CANDIDATE_OPS = ("GridSample", "LayerNormalization", "Softmax")
LINEAR_CANDIDATE_OPS = ("MatMul", "Gemm")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _signature(model) -> tuple:
    """Tensor names, static shapes, and types at the public ONNX boundary."""
    def describe(value):
        shape = []
        for dim in value.type.tensor_type.shape.dim:
            if not dim.HasField("dim_value") or dim.dim_value <= 0:
                raise ValueError(f"{value.name}: expected a fully static tensor shape")
            shape.append(int(dim.dim_value))
        if not shape:
            raise ValueError(f"{value.name}: expected a tensor with a batch dimension")
        return value.name, tuple(shape), value.type.tensor_type.elem_type

    return tuple(map(describe, model.graph.input)), tuple(map(describe, model.graph.output))


def _assign_missing_names(nodes) -> int:
    """Preserve exporter names; fill gaps by topological position and op type."""
    existing = [node.name for node in nodes if node.name]
    if len(existing) != len(set(existing)):
        raise ValueError("exported graph has duplicate node names; cannot build a unique manifest")
    taken = set(existing)
    assigned = 0
    for position, node in enumerate(nodes):
        if node.name:
            continue
        base = f"node_{node.op}_{position:05d}"
        name = base
        suffix = 2
        while name in taken:
            name = f"{base}_{suffix}"
            suffix += 1
        node.name = name
        taken.add(name)
        assigned += 1
    return assigned


def _read_export_record(source: Path, spec: ModelSpec, source_hash: str) -> str | None:
    """Reject stale export metadata when present; older exports lack it."""
    record_path = source.with_name(f"{source.stem}.export.json")
    if not record_path.exists():
        return None
    record = json.loads(record_path.read_text())
    if (record.get("onnx_sha256") != source_hash or record.get("model") != spec.name
            or record.get("backbone") != spec.backbone or record.get("precision") != "fp32"):
        raise ValueError(f"stale or mismatched export metadata: {record_path}")
    return str(record_path)


def _write_json(path: Path, record: dict) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(record, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run(spec: ModelSpec, *, out: Path | None = None, force: bool = False) -> Path:
    """Write a cleaned ONNX and JSON node manifest; never change the FP32 source."""
    import onnx
    import onnx_graphsurgeon as gs

    source = spec.onnx_path("fp32")
    if not source.is_file():
        raise FileNotFoundError(f"FP32 ONNX not found: {source}; run export.py first")
    out = Path(out) if out is not None else (
        REPO_ROOT / "artifacts" / "graphs" / f"{spec.name}_{spec.backbone}.fp32.onnx"
    )
    if out.suffix != ".onnx":
        raise ValueError(f"prepared graph output must end in .onnx: {out}")
    if out.resolve() == source.resolve():
        raise ValueError("graph surgery output must differ from the FP32 export")
    if out.exists() and not force:
        raise FileExistsError(f"{out} already exists; pass --force to replace it")
    out.parent.mkdir(parents=True, exist_ok=True)

    source_hash = _sha256(source)
    export_record = _read_export_record(source, spec, source_hash)
    onnx.checker.check_model(str(source))
    model = onnx.load(str(source))
    before_io = _signature(model)
    if len(before_io[0]) != 1 or before_io[0][0][0] != spec.input_name:
        raise ValueError(f"FP32 ONNX input does not match {spec.name} model spec")
    if [item[0] for item in before_io[1]] != spec.output_names:
        raise ValueError(f"FP32 ONNX outputs do not match {spec.name} model spec")
    if before_io[0][0][1] != (1, 3, spec.img_size, spec.img_size):
        raise ValueError(f"FP32 ONNX input shape does not match {spec.name} model spec")
    if any(item[1][0] != 1 or item[2] != onnx.TensorProto.FLOAT
           for item in (*before_io[0], *before_io[1])):
        raise ValueError("graph surgery requires static batch-1 FP32 tensor I/O")

    original_nodes = len(model.graph.node)
    graph = gs.import_onnx(model)
    graph.cleanup().toposort()
    assigned = _assign_missing_names(graph.nodes)
    cleaned = gs.export_onnx(graph)
    if _signature(cleaned) != before_io:
        raise RuntimeError("graph cleanup changed the model input or output contract")

    # Validate the serialized artifact before it replaces any previous output.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=out.parent, prefix=f".{out.stem}.",
                                         suffix=".onnx", delete=False) as stream:
            temporary = Path(stream.name)
        onnx.save(cleaned, str(temporary))
        onnx.checker.check_model(str(temporary))
        prepared_hash = _sha256(temporary)
        os.replace(temporary, out)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    nodes = [
        {"name": node.name, "op": node.op, "outputs": [tensor.name for tensor in node.outputs]}
        for node in graph.nodes
    ]
    candidates = {
        "numerical_ops": [node["name"] for node in nodes
                          if node["op"] in NUMERICAL_CANDIDATE_OPS],
        "linear_ops": [node["name"] for node in nodes
                       if node["op"] in LINEAR_CANDIDATE_OPS],
    }
    manifest = out.with_name(f"{out.stem}.graph.json")
    _write_json(manifest, {
        "schema_version": 1,
        "model": spec.name,
        "backbone": spec.backbone,
        "source": str(source),
        "source_sha256": source_hash,
        "stage_00_metadata": export_record,
        "prepared": str(out),
        "prepared_sha256": prepared_hash,
        "input": {"name": before_io[0][0][0], "shape": before_io[0][0][1]},
        "outputs": [{"name": name, "shape": shape} for name, shape, _ in before_io[1]],
        "normalization": {
            "removed_unreachable_nodes": original_nodes - len(nodes),
            "assigned_node_names": assigned,
            "constant_folding": False,
            "numerical_parity_checked": False,
        },
        "operator_counts": dict(sorted(Counter(node["op"] for node in nodes).items())),
        "inspection_candidates": candidates,
        "nodes": nodes,
    })
    print(f"[graph_surgery] {len(nodes)} live nodes ({original_nodes - len(nodes)} removed, "
          f"{assigned} named) -> {out}")
    print(f"[graph_surgery] node manifest -> {manifest}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare FP32 ONNX and record its node map")
    parser.add_argument("--spec", default="rtdetr", help="model name or host/configs/*.yaml path")
    parser.add_argument("--backbone", default=None, help="backbone in the model spec")
    parser.add_argument("--out", type=Path, help="prepared ONNX path (default: artifacts/graphs/)")
    parser.add_argument("--force", action="store_true", help="replace an existing prepared ONNX")
    args = parser.parse_args()
    run(load_spec(args.spec, args.backbone), out=args.out, force=args.force)


if __name__ == "__main__":
    main()
