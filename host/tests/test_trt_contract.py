# tests/test_trt_contract.py
"""Pins the TensorRT 11 facts this project's precision strategy depends on.

If any of these fail, precision handling must be revisited before trusting a single
number in results/tables/. See the "Critical finding" section of the Phase 1 plan.
"""
import tensorrt as trt


def _builder():
    return trt.Builder(trt.Logger(trt.Logger.ERROR))


def test_precision_builder_flags_do_not_exist():
    for flag in ("FP16", "INT8", "BF16", "FP8"):
        assert not hasattr(trt.BuilderFlag, flag), f"BuilderFlag.{flag} reappeared"


def test_networks_are_always_strongly_typed():
    net = _builder().create_network(0)  # 0 == "no flags"; a weak mode no longer exists
    assert net.get_flag(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)


def test_per_layer_precision_api_is_gone():
    assert not hasattr(trt.ILayer, "precision")
    assert not hasattr(trt.ILayer, "set_output_type")
    assert not hasattr(trt, "IInt8Calibrator")


def test_tf32_is_on_by_default_in_tensorrt():
    # Polygraphy's CreateConfig() silently clears this. build_engine sets it explicitly.
    assert _builder().create_builder_config().get_flag(trt.BuilderFlag.TF32)