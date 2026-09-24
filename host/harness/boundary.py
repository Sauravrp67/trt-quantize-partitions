"""Precision boundaries as a graph property.

TensorRT 11 is strongly typed, so a mixed-precision partition materializes every
precision transition as a `Cast` node. The count of those nodes is the partition's
boundary count, readable from the ONNX without building an engine. H1 asks whether
that count — not just how many nodes are held at high precision — predicts latency.

Two tiers live here. `count_casts` / `float_node_names` / the block builders need only the
ONNX. `fusion_groups` needs a built engine, because which nodes TensorRT welds together is
a decision only the builder makes — and a partition that splits one of those groups can
fail to compile outright, so the groups are hard constraints on the partition space rather
than another term in the cost.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import onnx
import tensorrt as trt
from onnx import TensorProto
from polygraphy.backend.trt import engine_from_bytes

_FLOAT_CAST_TARGETS = {TensorProto.FLOAT, TensorProto.FLOAT16}

# Ops that carry no float compute; holding one at a different precision means nothing.
_NON_COMPUTE = {"Cast", "Shape", "Constant", "ConstantOfShape", "Reshape",
                "Squeeze", "Unsqueeze", "Transpose", "Identity", "Gather"}


def count_casts(onnx_path) -> int:
    model = onnx.load(str(onnx_path))
    total = 0
    for node in model.graph.node:
        if node.op_type != "Cast":
            continue
        for attr in node.attribute:
            if attr.name == "to" and attr.i in _FLOAT_CAST_TARGETS:
                total += 1
    return total


def float_node_names(onnx_path) -> list[str]:

    model = onnx.load(str(onnx_path))
    return [n.name for n in model.graph.node
            if n.name and n.op_type not in _NON_COMPUTE]


def contiguous_block(names: list[str], n: int) -> list[str]:
    if n > len(names):
        raise ValueError(f"asked for {n} nodes but only {len(names)} available")
    start = (len(names) - n) // 2
    return names[start:start + n]


def scattered_block(names: list[str], n: int) -> list[str]:

    if n > len(names):
        raise ValueError(f"asked for {n} nodes but only {len(names)} available")
    if n == 0:
        return []
    stride = len(names) / n
    return [names[min(int(i * stride), len(names) - 1)] for i in range(n)]


# --- fusion regions (engine-derived) ------------------------------------------------

# Myelin names every layer of a compiled region `<something>_myl<REGION>_<SEQ>`, which is
# the only reliable way to tell that two layers came from one NVRTC kernel.
_MYELIN_SUFFIX = re.compile(r"_myl(\d+)_\d+$")
# torch.onnx names every exported node `node_<op>_<counter>`; TensorRT keeps those names
# inside fused layer names (`a + b`, `PWN(PWN(a), PWN(b))`, `a+b_myl36_3`).
_ONNX_NODE = re.compile(r"node_[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*")


@dataclass(frozen=True)
class FusionGroup:
    """ONNX nodes TensorRT compiled into a single unit.

    Holding only *some* of these at FP32 puts a precision boundary inside one kernel.
    For ``kind == "myelin"`` that is not merely expensive: TensorRT emits the region as one
    NVRTC kernel and a split can fail codegen with no fallback, making the partition
    unbuildable.
    """
    kind: str                    # "myelin" (one NVRTC kernel) | "fused" (classic fusion)
    nodes: frozenset[str]
    layers: tuple[str, ...]      # the engine layers the group was recovered from


def fusion_groups_from_layer_names(layer_names, *, known=None) -> list[FusionGroup]:
    """Group engine layer names into the ONNX-node sets they fused. Pure, so it is testable
    against captured names without building an engine.

    ``known`` restricts results to node names that exist in the source graph, which drops
    false positives from synthesized kernel names.

    **The Myelin groups are a lower bound.** Layers such as ``__myl_MaxrSubExpSumDivMul_myl36_5``
    carry no reference to the ONNX nodes they came from, so a region can legitimately span
    more nodes than are recoverable from its layer names.
    """
    regions: dict[str, tuple[str, set[str], list[str]]] = {}
    for i, name in enumerate(layer_names):
        # A reformat is a precision/format boundary, not a fusion. Its name may quote a
        # ForeignNode's endpoints, which would fabricate a group out of two unrelated nodes.
        if name.startswith("Reformatting CopyNode"):
            continue
        match = _MYELIN_SUFFIX.search(name)
        body = name[:match.start()] if match else name
        nodes = set(_ONNX_NODE.findall(body))
        if known is not None:
            nodes &= set(known)
        if not nodes:
            continue
        key, kind = (f"myl{match.group(1)}", "myelin") if match else (f"layer{i}", "fused")
        _, seen, layers = regions.setdefault(key, (kind, set(), []))
        seen.update(nodes)
        layers.append(name)
    return [FusionGroup(kind, frozenset(nodes), tuple(layers))
            for kind, nodes, layers in regions.values() if len(nodes) > 1]


def fusion_groups(engine_bytes: bytes, *, known=None) -> list[FusionGroup]:
    """Fusion groups of a built engine — the node sets that must share a precision label."""
    engine = engine_from_bytes(engine_bytes)
    inspector = engine.create_engine_inspector()
    names = [json.loads(inspector.get_layer_information(i, trt.LayerInformationFormat.JSON))
             .get("Name", "") for i in range(engine.num_layers)]
    return fusion_groups_from_layer_names(names, known=known)


def split_fusions(groups: list[FusionGroup], block_list) -> list[FusionGroup]:
    """Groups the block list cuts through — held at FP32 in part but not whole.

    A non-empty result means the partition puts a boundary inside a fused kernel, which is
    the condition that precedes the observed NVRTC build failures.
    """
    blocked = set(block_list)
    return [g for g in groups if g.nodes & blocked and not g.nodes <= blocked]