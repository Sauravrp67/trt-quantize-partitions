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
backbone: small
backbones:
  small:
    config: ${rtdetr_pytorch}/configs/toy.yml
    ckpt: ${models}/toy/toy.pth
    onnx:
      fp32: ${models}/toy/model.onnx
      fp16: ${models}/toy/model_fp16.onnx
  big:
    config: ${rtdetr_pytorch}/configs/big.yml
    ckpt: ${models}/toy/big.pth
    img_size: 800
    onnx:
      fp32: ${models}/toy/big.onnx
      fp16: ${models}/toy/big_fp16.onnx
onnx:
  input_name: images
  output_names: [pred_logits, pred_boxes]
  img_size: 320
  default: fp16
defaults:
  device: cuda
  score_thr: 0.6
outputs:
  table: ${tables}/toy_{backbone}.md
"""


def test_roots_are_interpolated_and_absolute(tmp_path):
    spec = load_spec(_write(tmp_path, BODY))
    assert spec.ckpt == paths.MODELS_DIR / "toy" / "toy.pth"
    assert spec.torch_config == paths.RTDETR_PYTORCH_ROOT / "configs" / "toy.yml"
    assert spec.outputs["table"] == paths.TABLES_DIR / "toy_small.md"
    assert spec.ckpt.is_absolute()


def test_backbone_selection_switches_every_derived_path(tmp_path):
    path = _write(tmp_path, BODY)
    small, big = load_spec(path), load_spec(path, "big")
    assert (small.backbone, big.backbone) == ("small", "big")
    assert small.backbones == big.backbones == ["big", "small"]
    assert big.ckpt.name == "big.pth" and big.onnx["fp16"].name == "big_fp16.onnx"
    # {backbone} expands per selection, so runs cannot overwrite each other
    assert small.outputs["table"] != big.outputs["table"]
    assert big.outputs["table"] == paths.TABLES_DIR / "toy_big.md"


def test_backbone_entry_overrides_the_shared_onnx_contract(tmp_path):
    path = _write(tmp_path, BODY)
    assert load_spec(path).img_size == 320          # shared contract
    assert load_spec(path, "big").img_size == 800   # entry override
    assert load_spec(path, "big").input_name == "images"  # unset keys still inherited


def test_backbone_can_override_the_class_vocabulary(tmp_path):
    body = BODY.replace("    img_size: 800", "    img_size: 800\n    classes: [bird, dog, horse]")
    path = _write(tmp_path, body)
    assert load_spec(path).class_names == ["person", "cat"]
    assert load_spec(path, "big").class_names == ["bird", "dog", "horse"]


@pytest.mark.parametrize("family,directory", [
    ("rtdetr", paths.RTDETR_PYTORCH_ROOT / "configs/rtdetr"),
    ("rtdetrv2", paths.RTDETRV2_PYTORCH_ROOT / "configs/rtdetrv2"),
    ("rtdetrv3", paths.RTDETRV3_ROOT / "configs/rtdetrv3"),
    ("rtdetrv4", paths.RTDETRV4_ROOT / "configs/rtv4"),
])
def test_every_upstream_model_config_has_a_distinct_registry_entry(family, directory):
    specs = [load_spec(family, name) for name in load_spec(family).backbones]
    assert {spec.torch_config for spec in specs} == set(directory.glob("*.yml"))
    assert len({spec.torch_config for spec in specs}) == len(specs)
    assert len({spec.ckpt for spec in specs}) == len(specs)
    assert len({spec.onnx_path("fp32") for spec in specs}) == len(specs)
    for spec in specs:
        assert spec.num_classes == (20 if spec.backbone.endswith("_voc") else
                                    1203 if spec.backbone.endswith("_lvis") else 80)
        assert spec.onnx_path("fp32") != spec.onnx_path("fp16")
        assert spec.raw["backbones"][spec.backbone]["checkpoint_status"] in {
            "released", "user_supplied"}


def test_unknown_backbone_lists_the_registry(tmp_path):
    with pytest.raises(ValueError, match=r"unknown backbone 'nope'.*big.*small"):
        load_spec(_write(tmp_path, BODY), "nope")


def test_onnx_path_takes_registered_variants_only(tmp_path):
    spec = load_spec(_write(tmp_path, BODY))
    assert spec.onnx_path() == spec.onnx["fp16"]          # onnx.default
    assert spec.onnx_path("fp32") == spec.onnx["fp32"]
    assert spec.variant_of(spec.onnx["fp32"]) == "fp32"


def test_onnx_path_rejects_a_path_so_weights_cannot_leave_the_backbone(tmp_path):
    """A path here would pair one backbone's spec with another's weights, silently."""
    spec = load_spec(_write(tmp_path, BODY))
    with pytest.raises(ValueError, match=r"'/somewhere/other.onnx'.*small.*fp16.*fp32"):
        spec.onnx_path("/somewhere/other.onnx")
    with pytest.raises(ValueError, match=r"int8"):
        spec.onnx_path("int8")


def test_onnx_dest_allows_an_explicit_path(tmp_path):
    """Export writes new files, so its destination is not restricted to the registry."""
    spec = load_spec(_write(tmp_path, BODY))
    assert spec.onnx_dest() == spec.onnx["fp16"]          # onnx.default
    assert spec.onnx_dest("fp32") == spec.onnx["fp32"]    # registry name still wins
    assert spec.onnx_dest("/somewhere/other.onnx").name == "other.onnx"


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


@pytest.mark.parametrize("family,root", [
    ("rtdetr", paths.RTDETR_ROOT),
    ("rtdetrv2", paths.RTDETR_ROOT),
    ("rtdetrv3", paths.RTDETRV3_ROOT),
    ("rtdetrv4", paths.RTDETRV4_ROOT),
])
def test_source_driven_specs_use_registered_upstream_roots(family, root):
    spec = load_spec(family)
    assert spec.torch_config.is_relative_to(root)


def test_shipped_rtdetr_spec_loads():
    spec = load_spec("rtdetr")
    assert spec.name == "rtdetr"
    assert len(spec.class_names) == 80 and spec.class_names[0] == "person"
    assert spec.output_names == ["pred_logits", "pred_boxes"]
    assert spec.img_size == 640 and spec.batch == 1
    assert set(spec.onnx) >= {"fp32", "fp16"}


def test_shipped_rtdetr_backbones_all_resolve():
    """Every registry entry must name an upstream config that actually exists."""
    for name in load_spec("rtdetr").backbones:
        spec = load_spec("rtdetr", name)
        assert spec.torch_config.exists(), f"{name}: missing {spec.torch_config}"
        assert set(spec.onnx) >= {"fp32", "fp16"}, name


# --- TensorRT timing cache ----------------------------------------------------------

def test_timing_cache_is_absent_when_the_spec_does_not_declare_one(tmp_path):
    assert load_spec(_write(tmp_path, BODY)).timing_cache is None


def test_timing_cache_null_disables_caching(tmp_path):
    body = BODY.replace("  score_thr: 0.6", "  score_thr: 0.6\n  timing_cache: null")
    assert load_spec(_write(tmp_path, body)).timing_cache is None


def test_timing_cache_resolves_roots_and_is_absolute(tmp_path):
    body = BODY.replace("  score_thr: 0.6",
                        "  score_thr: 0.6\n  timing_cache: ${engines}/timing.cache")
    cache = load_spec(_write(tmp_path, body)).timing_cache
    assert cache == paths.ENGINES_DIR / "timing.cache"
    assert cache.is_absolute()


def test_timing_cache_is_scoped_per_backbone(tmp_path):
    """One backbone's tactic timings must not shape another backbone's build."""
    body = BODY.replace("  score_thr: 0.6",
                        "  score_thr: 0.6\n  timing_cache: ${engines}/timing_{backbone}.cache")
    path = _write(tmp_path, body)
    small, big = load_spec(path), load_spec(path, "big")
    assert small.timing_cache.name == "timing_small.cache"
    assert big.timing_cache.name == "timing_big.cache"


def test_shipped_rtdetr_spec_declares_a_per_backbone_timing_cache():
    for name in load_spec("rtdetr").backbones:
        spec = load_spec("rtdetr", name)
        assert spec.timing_cache == paths.ENGINES_DIR / f"timing_{name}.cache", name
