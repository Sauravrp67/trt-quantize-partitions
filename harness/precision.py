"""Precision lives in the ONNX graph, not in a TensorRT builder flag (TRT 11 is
strongly typed). This module produces the precision-variant graphs that
harness.trt_runner.build_engine then compiles verbatim."""

from __future__ import annotations
from pathlib import Path

import onnx
import onnx_graphsurgeon as gs
from onnxruntime.transformers.float16 import convert_float_to_float16


def _node_sig(n) -> tuple:
    """Structural identity of a node: op, inputs, outputs, and (int) attributes."""
    return (n.op_type, tuple(n.input), tuple(n.output),
            tuple((a.name, a.i) for a in n.attribute))


def _sanitize(model: onnx.ModelProto) -> onnx.ModelProto:
    """Repair two defects the ORT float16 converter can emit on real graphs.

    1. Duplicate cast nodes. When one tensor feeds several nodes that must stay FP32,
       the converter inserts an identical ``<name>_cast_to_fp32`` node *per consumer*,
       all sharing the same output name. Two producers of one tensor make the graph
       impossible to topologically sort -- TensorRT's parser rejects it with
       "Output name is not unique" / "Failed to sort the model topologically".
       (RT-DETR trips this: one upsample-scales tensor feeds two Resize ops.)
    2. Non-topological node order. The converter appends its boundary Cast nodes at the
       end of the node list, so the graph is a valid DAG but not stored in sorted order
       -- which onnx.checker and stricter consumers reject.

    Fix: drop structurally-identical duplicate-output nodes (keep one), fail loudly if a
    name collision is between *different* nodes (that would be real corruption), then
    topologically sort.
    """
    g = model.graph
    seen: set = set()
    kept = []
    for n in g.node:
        key = (tuple(n.output), _node_sig(n))
        if key in seen:
            continue  # exact duplicate of a node already kept
        seen.add(key)
        kept.append(n)
    del g.node[:]
    g.node.extend(kept)

    # Any output name still produced by >1 node is a genuine collision, not a dup.
    produced: dict = {}
    for n in g.node:
        for o in n.output:
            produced.setdefault(o, []).append(n.name)
    collisions = {o: names for o, names in produced.items() if len(names) > 1}
    if collisions:
        raise ValueError(f"FP16 graph has non-identical duplicate outputs: {collisions}")

    graph = gs.import_onnx(model)
    graph.toposort()
    return gs.export_onnx(graph)


def to_fp16(onnx_path: str, out_path, *, node_block_list: list[str] | None = None) -> Path:
    """FP32 ONNX -> FP16 ONNX, FP32 kept at the I/O boundary, output sanitized.

    ``keep_io_types=True`` leaves graph inputs/outputs FP32 (Cast nodes wrap the graph)
    so TRTSession / trtexec bindings are identical across precisions. ``node_block_list``
    names nodes to leave in FP32 -- None for uniform FP16 (Plan 1); the sensitivity-driven
    list for the partition (Plan 2). The result is passed through ``_sanitize`` because the
    ORT converter can emit duplicate cast nodes and an unsorted node list that TensorRT
    rejects (see that helper).
    """
    model = convert_float_to_float16(
        onnx.load(str(onnx_path)),
        keep_io_types=True,
        node_block_list=node_block_list,
    )
    model = _sanitize(model)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_path))
    return Path(out_path)
