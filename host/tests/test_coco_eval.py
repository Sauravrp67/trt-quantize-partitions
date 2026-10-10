# tests/test_coco_eval.py
import numpy as np
import pytest
from types import SimpleNamespace
from harness.config import load_spec
from harness.compare import Detections
from harness.coco_eval import detections_to_coco, COCO80_TO_CATID
from harness.coco_eval import evaluate


@pytest.mark.parametrize("family,backbone", [("rtdetrv2", "r18vd_voc"),
                                           ("rtdetrv3", "r50vd_lvis")])
def test_coco_eval_rejects_other_class_vocabularies_before_loading_data(family, backbone):
    with pytest.raises(ValueError, match="COCO80 class order"):
        evaluate(None, SimpleNamespace(spec=load_spec(family, backbone)))

def test_detections_to_coco_format():
    det = Detections(np.array([0]), np.array([[10., 20., 30., 50.]]), np.array([0.9]))
    out = detections_to_coco(42, det)
    assert out[0]["image_id"] == 42
    assert out[0]["category_id"] == COCO80_TO_CATID[0]   # person -> 1
    assert out[0]["bbox"] == [10.0, 20.0, 20.0, 30.0]    # xyxy -> xywh
    assert abs(out[0]["score"] - 0.9) < 1e-6
