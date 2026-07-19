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