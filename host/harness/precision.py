"""Precision lives in the ONNX graph, not in a TensorRT builder flag (TRT 11 is
strongly typed). This module produces the precision-variant graphs that
harness.trt_runner.build_engine then compiles verbatim."""

from __future__ import annotations
from pathlib import Path

import onnx
import onnx_graphsurgeon as gs
from onnx import TensorProto
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


def _cast_target(node) -> int | None:
    """The ``to`` dtype of a Cast node, or None for anything else."""
    if node.op_type != "Cast":
        return None
    return next((a.i for a in node.attribute if a.name == "to"), None)


def _fuse_roundtrip_casts(model: onnx.ModelProto) -> onnx.ModelProto:
    """Collapse ``FP32 -> Cast(FP16) -> Cast(FP32)`` round trips.

    The ORT converter emits its boundary casts per *blocked node* rather than per
    precision *transition*, so two adjacent FP32 nodes hand over through FP16: the first
    result is rounded down and immediately raised again. That costs bandwidth, discards
    mantissa bits for no reason, and -- the reason this matters here -- pins the cast
    count at ``2N + 2`` for any block list of size N. Boundary count would then measure
    how many nodes are held in FP32 and never where the boundaries fall, which is exactly
    the distinction stages 03/04 and the H1 pilot are built on.

    Each up-cast is redirected to the down-cast's FP32 source, then down-casts left
    without consumers are dropped. A cast whose output is a graph output always survives:
    ``keep_io_types=True`` puts it there deliberately.
    """
    g = model.graph
    graph_outputs = {o.name for o in g.output}
    producer = {out: n for n in g.node for out in n.output}

    # up-cast output -> the FP32 tensor it should have read all along
    rewire: dict[str, str] = {}
    for node in g.node:
        if _cast_target(node) != TensorProto.FLOAT or node.output[0] in graph_outputs:
            continue
        src = producer.get(node.input[0])
        if src is not None and _cast_target(src) == TensorProto.FLOAT16:
            rewire[node.output[0]] = src.input[0]
    if not rewire:
        return model

    kept = [n for n in g.node if n.output[0] not in rewire]
    for node in kept:
        for i, name in enumerate(node.input):
            while name in rewire:        # chained round trips collapse to the origin
                name = rewire[name]
            node.input[i] = name

    # a down-cast whose only consumer was a fused up-cast is now dead
    live_tensors = {i for n in kept for i in n.input} | graph_outputs
    kept = [n for n in kept
            if _cast_target(n) != TensorProto.FLOAT16 or n.output[0] in live_tensors]

    del g.node[:]
    g.node.extend(kept)

    reachable = {o for n in g.node for o in n.output} | {i.name for i in g.input}
    stale = [v for v in g.value_info if v.name not in reachable]
    for v in stale:
        g.value_info.remove(v)
    return model


def to_fp16(onnx_path: str, out_path, *, node_block_list: list[str] | None = None) -> Path:
    """FP32 ONNX -> FP16 ONNX, FP32 kept at the I/O boundary, output sanitized.

    ``keep_io_types=True`` leaves graph inputs/outputs FP32 (Cast nodes wrap the graph)
    so TRTSession / trtexec bindings are identical across precisions. ``node_block_list``
    names nodes to leave in FP32 -- None for uniform FP16 (Plan 1); the sensitivity-driven
    list for the partition (Plan 2). The result is passed through ``_sanitize`` because the
    ORT converter can emit duplicate cast nodes and an unsorted node list that TensorRT
    rejects, then through ``_fuse_roundtrip_casts`` so adjacent FP32 nodes hand over in
    FP32 instead of round-tripping through FP16 (see those helpers).
    """
    model = convert_float_to_float16(
        onnx.load(str(onnx_path)),
        keep_io_types=True,
        node_block_list=node_block_list,
    )
    model = _fuse_roundtrip_casts(_sanitize(model))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_path))
    return Path(out_path)
