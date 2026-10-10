"""Check the shared raw-output contract without downloading model weights."""
from __future__ import annotations

import numpy as np
import onnx
import pytest
from PIL import Image

from harness.adapter import load_adapter
from harness.config import load_spec
from models.rtdetrv3.export import normalize_raw_onnx


@pytest.mark.parametrize("name,backbone", [
    ("rtdetrv2", None), ("rtdetrv3", None), ("rtdetrv4", None),
    ("rtdetrv2", "r18vd_voc"), ("rtdetrv3", "r18vd_lvis"),
])
def test_family_specs_and_decode(name, backbone):
    spec = load_spec(name, backbone)
    adapter = load_adapter(spec, device="cpu")
    image = Image.new("RGB", (1280, 720), (100, 150, 200))
    tensor, meta = adapter.preprocess(image)
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype.is_floating_point
    assert np.allclose(tensor.numpy()[0, :, 0, 0], [100, 150, 200] / np.array(255))

    logits = np.full((1, 300, spec.num_classes), -10, dtype=np.float32)
    logits[0, 0, 5] = 10
    boxes = np.zeros((1, 300, 4), dtype=np.float32)
    boxes[0, 0] = [0.5, 0.5, 0.25, 0.5]
    found = adapter.postprocess({"pred_logits": logits, "pred_boxes": boxes}, meta)
    assert found.labels[0] == 5
    assert np.allclose(found.boxes[0], [480, 180, 800, 540])
    assert found.scores[0] > 0.99


def _raw_onnx(path, *, queries=300, classes=80):
    image = onnx.helper.make_tensor_value_info("image", onnx.TensorProto.FLOAT,
                                               [1, 3, 640, 640])
    boxes = onnx.helper.make_tensor_value_info("bbox", onnx.TensorProto.FLOAT,
                                               [1, queries, 4])
    logits = onnx.helper.make_tensor_value_info("bbox_num", onnx.TensorProto.FLOAT,
                                                [1, queries, classes])
    graph = onnx.helper.make_graph([
        onnx.helper.make_node("ReduceMean", ["image"], ["mean"], keepdims=0),
        onnx.helper.make_node("ConstantOfShape", ["shape_boxes"], ["bbox"]),
        onnx.helper.make_node("ConstantOfShape", ["shape_logits"], ["bbox_num"]),
    ], "raw", [image], [boxes, logits], initializer=[
        onnx.helper.make_tensor("shape_boxes", onnx.TensorProto.INT64, [3], [1, queries, 4]),
        onnx.helper.make_tensor("shape_logits", onnx.TensorProto.INT64, [3], [1, queries, classes]),
    ])
    onnx.save(onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 16)]), path)


@pytest.mark.parametrize("backbone", ["r18vd", "r18vd_lvis"])
def test_v3_normalizes_raw_onnx_interface(tmp_path, backbone):
    source, destination = tmp_path / "raw.onnx", tmp_path / "normalized.onnx"
    spec = load_spec("rtdetrv3", backbone)
    _raw_onnx(source, classes=spec.num_classes)
    normalize_raw_onnx(source, destination, spec)
    graph = onnx.load(destination).graph
    assert [i.name for i in graph.input] == ["images"]
    assert [o.name for o in graph.output] == ["pred_logits", "pred_boxes"]
    onnx.checker.check_model(str(destination))


def test_v3_rejects_unexpected_query_count(tmp_path):
    source = tmp_path / "raw.onnx"
    _raw_onnx(source, queries=100)
    with pytest.raises(ValueError, match="batch/query"):
        normalize_raw_onnx(source, tmp_path / "normalized.onnx", load_spec("rtdetrv3"))
