import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from harness.boundary import (
    contiguous_block,
    count_casts,
    float_node_names,
    fusion_groups_from_layer_names,
    scattered_block,
    split_fusions,
)

# Verbatim layer names from an r18vd FP32 engine (TensorRT 11.0.0.114), the shapes the
# parser has to survive: Myelin region members, classic '+' fusion, PWN trees, reformats.
REAL_LAYERS = [
    "__myl_ReshTran_myl36_0",
    "__myl_Add_myl36_2",
    "node_MatMul_301+node_MatMul_299_myl36_3",
    "node_bmm_myl36_4",
    "__myl_MaxrSubExpSumDivMul_myl36_5",
    "node_MatMul_303_myl36_7",
    "node_Conv_1427 + node_relu_11",
    "PWN(PWN(PWN(node_Sigmoid_418), PWN(node_silu_11)), PWN(node_add_12))",
    "node_Conv_1447",
    "Reformatting CopyNode for Input Tensor 0 to "
    "{ForeignNode[model.encoder.encoder.0.layers.0.self_attn.in_proj_weight"
    "...node_permute_1 + node_view_7]}",
]


def _chain_model(path, depth=8):
    """A linear MatMul chain: `depth` named float nodes, one after another."""
    inits, nodes = [], []
    prev = "x"
    for i in range(depth):
        inits.append(numpy_helper.from_array(np.eye(4, dtype=np.float32), f"W{i}"))
        nodes.append(helper.make_node("MatMul", [prev, f"W{i}"], [f"h{i}"], name=f"mm{i}"))
        prev = f"h{i}"
    graph = helper.make_graph(
        nodes, "chain",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info(prev, TensorProto.FLOAT, [1, 4])],
        inits,
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)]), str(path))


def test_float_node_names_are_topological(tmp_path):
    src = tmp_path / "m.onnx"
    _chain_model(src)
    assert float_node_names(src) == [f"mm{i}" for i in range(8)]


def test_contiguous_block_is_consecutive():
    names = [f"n{i}" for i in range(10)]
    block = contiguous_block(names, 4)
    assert len(block) == 4
    start = names.index(block[0])
    assert block == names[start:start + 4]


def test_scattered_block_is_spread_out():
    names = [f"n{i}" for i in range(10)]
    block = scattered_block(names, 4)
    assert len(block) == 4
    idx = sorted(names.index(b) for b in block)
    gaps = [b - a for a, b in zip(idx, idx[1:])]
    assert min(gaps) > 1, "scattered block must not contain adjacent nodes"


def test_equal_size_blocks_differ_in_cast_count(tmp_path):
    """The premise of H1: same N, different boundary count."""
    src = tmp_path / "m.onnx"
    _chain_model(src, depth=12)
    names = float_node_names(src)

    from harness.precision import to_fp16
    contig = tmp_path / "contig.onnx"
    scat = tmp_path / "scat.onnx"
    to_fp16(src, contig, node_block_list=contiguous_block(names, 4))
    to_fp16(src, scat, node_block_list=scattered_block(names, 4))

    assert count_casts(scat) > count_casts(contig)


def _by_kind(groups):
    return {g.kind: g for g in groups}


def test_myelin_layers_of_one_region_become_one_group():
    """`_myl36_*` is TensorRT's own region id -- the only reliable fusion evidence."""
    groups = _by_kind(fusion_groups_from_layer_names(REAL_LAYERS))
    myelin = groups["myelin"]
    # the region suffix must be stripped before node extraction, or `node_bmm_myl36_4`
    # yields the layer name instead of the ONNX node `node_bmm`
    assert myelin.nodes == {"node_MatMul_301", "node_MatMul_299", "node_bmm", "node_MatMul_303"}


def test_classic_fusions_and_pwn_trees_are_grouped_separately():
    groups = fusion_groups_from_layer_names(REAL_LAYERS)
    fused = {g.nodes for g in groups if g.kind == "fused"}
    assert {"node_Conv_1427", "node_relu_11"} in fused
    assert {"node_Sigmoid_418", "node_silu_11", "node_add_12"} in fused
    # a lone unfused layer constrains nothing, so it is not a group
    assert not any("node_Conv_1447" in g.nodes for g in groups)


def test_reformat_layers_are_not_treated_as_fusions():
    """A reformat names a ForeignNode's endpoints; grouping them would invent a constraint."""
    groups = fusion_groups_from_layer_names(REAL_LAYERS)
    assert not any({"node_permute_1", "node_view_7"} & g.nodes for g in groups)


def test_known_filter_drops_names_absent_from_the_source_graph():
    groups = fusion_groups_from_layer_names(
        REAL_LAYERS, known={"node_MatMul_301", "node_MatMul_299", "node_Conv_1427"})
    assert {g.nodes for g in groups} == {frozenset({"node_MatMul_301", "node_MatMul_299"})}


def test_split_fusions_flags_partially_blocked_groups_only():
    groups = fusion_groups_from_layer_names(REAL_LAYERS)
    # whole group held FP32 -> no boundary inside the kernel
    assert split_fusions(groups, ["node_Conv_1427", "node_relu_11"]) == []
    # half of it -> a boundary lands inside a fused layer
    assert [g.nodes for g in split_fusions(groups, ["node_Conv_1427"])] == \
        [frozenset({"node_Conv_1427", "node_relu_11"})]
    assert split_fusions(groups, []) == []