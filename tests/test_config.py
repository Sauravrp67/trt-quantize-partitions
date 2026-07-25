# tests/test_config.py
import pytest

from harness import paths
from harness.config import load_spec, spec_path


def _write(tmp_path, body: str):
    p = tmp_path / "spec.yaml"
    p.write_text(body)
    return p


BODY = """
name: toy
classes: [person, cat]
model:
  config: ${rtdetr_pytorch}/configs/toy.yml
  ckpt: ${models}/toy/toy.pth
onnx:
  input_name: images
  output_names: [pred_logits, pred_boxes]
  img_size: 320
  default: fp16
  variants:
    fp32: ${models}/toy/model.onnx
    fp16: ${models}/toy/model_fp16.onnx
defaults:
  device: cuda
  score_thr: 0.6
outputs:
  table: ${tables}/toy.md
"""


def test_roots_are_interpolated_and_absolute(tmp_path):
    spec = load_spec(_write(tmp_path, BODY))
    assert spec.ckpt == paths.MODELS_DIR / "toy" / "toy.pth"
    assert spec.torch_config == paths.RTDETR_PYTORCH_ROOT / "configs" / "toy.yml"
    assert spec.outputs["table"] == paths.TABLES_DIR / "toy.md"
    assert spec.ckpt.is_absolute()


def test_onnx_path_takes_variant_name_or_path(tmp_path):
    spec = load_spec(_write(tmp_path, BODY))
    assert spec.onnx_path() == spec.onnx["fp16"]          # onnx.default
    assert spec.onnx_path("fp32") == spec.onnx["fp32"]
    assert spec.onnx_path("/somewhere/other.onnx").name == "other.onnx"
    assert spec.variant_of(spec.onnx["fp32"]) == "fp32"


def test_defaults_are_overridable(tmp_path):
    spec = load_spec(_write(tmp_path, BODY))
    assert spec.default("score_thr") == 0.6      # no override -> YAML value
    assert spec.default("score_thr", 0.9) == 0.9  # CLI wins
    assert spec.img_size == 320 and spec.class_names == ["person", "cat"]


def test_unknown_root_is_rejected(tmp_path):
    with pytest.raises(KeyError, match="nope"):
        load_spec(_write(tmp_path, BODY.replace("${models}/toy/toy.pth", "${nope}/toy.pth")))


def test_bad_default_variant_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="onnx.default"):
        load_spec(_write(tmp_path, BODY.replace("default: fp16", "default: int8")))


def test_spec_name_resolves_into_configs_dir():
    assert spec_path("rtdetr") == paths.CONFIGS_DIR / "rtdetr.yaml"


def test_shipped_rtdetr_spec_loads():
    spec = load_spec("rtdetr")
    assert spec.name == "rtdetr"
    assert len(spec.class_names) == 80 and spec.class_names[0] == "person"
    assert spec.output_names == ["pred_logits", "pred_boxes"]
    assert spec.img_size == 640 and spec.batch == 1
    assert set(spec.onnx) >= {"fp32", "fp16"}
