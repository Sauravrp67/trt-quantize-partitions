import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from harness.precision import to_fp16


def _tiny_model(path):
    w = numpy_helper.from_array(np.ones((4, 4), np.float32), "W")
    node = helper.make_node("MatMul", ["x", "W"], ["y"], name="mm")
    graph = helper.make_graph(
        [node], "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
        [w],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)]), str(path))


def test_to_fp16_halves_weights_but_keeps_io_fp32(tmp_path):
    src, dst = tmp_path / "m.onnx", tmp_path / "m16.onnx"
    _tiny_model(src)
    to_fp16(src, dst)
    m = onnx.load(str(dst))
    # I/O stays FP32 so TRTSession / trtexec bindings are unchanged; Casts wrap the graph.
    assert m.graph.input[0].type.tensor_type.elem_type == TensorProto.FLOAT
    assert m.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
    assert {i.data_type for i in m.graph.initializer} == {TensorProto.FLOAT16}
    assert "Cast" in [n.op_type for n in m.graph.node]


def test_node_block_list_keeps_named_node_in_fp32(tmp_path):
    """The Plan 2 partition mechanism: blocked nodes keep FP32 weights."""
    src, dst = tmp_path / "m.onnx", tmp_path / "m16.onnx"
    _tiny_model(src)
    to_fp16(src, dst, node_block_list=["mm"])
    m = onnx.load(str(dst))
    assert {i.data_type for i in m.graph.initializer} == {TensorProto.FLOAT}


def _chain_model(path, depth):
    """A linear MatMul chain: `depth` named float nodes, one after another."""
    inits, nodes, prev = [], [], "x"
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


def _casts(model) -> list:
    return [n for n in model.graph.node if n.op_type == "Cast"]


def test_adjacent_fp32_nodes_hand_over_without_a_fp16_round_trip(tmp_path):
    """Two neighbouring blocked nodes must exchange FP32 directly.

    The ORT converter emits boundary casts per *blocked node*, not per precision
    transition, so it rounds an FP32 result down to FP16 and immediately raises it back
    for the next FP32 node. That is wasted bandwidth and a real precision loss, and it
    makes the cast count depend only on how many nodes are held in FP32 -- never on
    where the boundaries fall, which is the quantity stages 03/04 reason about.
    """
    src, dst = tmp_path / "m.onnx", tmp_path / "m16.onnx"
    _chain_model(src, 6)
    to_fp16(src, dst, node_block_list=["mm2", "mm3"])     # adjacent
    m = onnx.load(str(dst))

    produced_by_cast = {n.output[0] for n in _casts(m)}
    assert not [n for n in _casts(m) if n.input[0] in produced_by_cast], \
        "a Cast feeding another Cast is a round trip that should have been collapsed"

    # 2 graph-I/O casts + 1 down-boundary + 1 up-boundary; the mm2->mm3 edge stays FP32
    assert len(_casts(m)) == 4
    by_name = {n.name: n for n in m.graph.node}
    assert by_name["mm3"].input[0] == by_name["mm2"].output[0], \
        "mm3 must read mm2's FP32 output directly, with no cast in between"


def test_cast_count_tracks_boundary_placement_not_block_size(tmp_path):
    """Equal N, different placement -> different cast count. The premise of H1."""
    src = tmp_path / "m.onnx"
    _chain_model(src, 12)
    contig, scat = tmp_path / "c.onnx", tmp_path / "s.onnx"
    to_fp16(src, contig, node_block_list=["mm4", "mm5", "mm6", "mm7"])
    to_fp16(src, scat, node_block_list=["mm0", "mm3", "mm6", "mm9"])
    assert len(_casts(onnx.load(str(scat)))) > len(_casts(onnx.load(str(contig))))