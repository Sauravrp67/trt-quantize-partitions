"""The adapter seam: a base class, a registry, and no model names in generic code.

Every test here runs on CPU without a checkpoint, an ONNX file, or the RT-DETR
submodule — the seam is the part of the system that must stay cheap to reason about.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from harness.adapter import (
    ADAPTERS,
    BaseAdapter,
    Detections,
    DetectorAdapter,
    adapter_class,
    load_adapter,
    resolve_device,
)
from harness.config import ModelSpec

REPO_ROOT = Path(__file__).resolve().parents[1]


def _spec(name: str = "stub") -> ModelSpec:
    """A ModelSpec built in memory — no YAML, no filesystem."""
    return ModelSpec(
        name=name,
        class_names=["a", "b"],
        input_name="images",
        output_names=["pred_logits", "pred_boxes"],
        img_size=640,
        batch=1,
        torch_config=Path("/nonexistent/config.yml"),
        ckpt=Path("/nonexistent/ckpt.pth"),
        onnx={"fp32": Path("/nonexistent/m.onnx"), "fp16": Path("/nonexistent/m.fp16.onnx")},
        default_onnx="fp32",
        defaults={"device": "cpu"},
    )


class _StubAdapter(BaseAdapter):
    """Minimal concrete adapter — the contract a model plugin must satisfy."""

    spec_name = "stub"

    def build_torch(self):
        raise NotImplementedError

    def preprocess(self, frame):
        raise NotImplementedError

    def postprocess(self, named, meta):
        raise NotImplementedError


# --- the base class -----------------------------------------------------------------

def test_base_adapter_populates_the_spec_derived_fields():
    adapter = _StubAdapter(_spec(), onnx="fp16", device="cpu")
    assert adapter.name == "stub"
    assert adapter.class_names == ["a", "b"]
    assert adapter.input_name == "images"
    assert adapter.output_names == ["pred_logits", "pred_boxes"]
    assert adapter.img_size == 640
    assert adapter.onnx_path == Path("/nonexistent/m.fp16.onnx")
    assert adapter.device == "cpu"


def test_base_adapter_defaults_to_the_specs_default_variant():
    assert _StubAdapter(_spec(), device="cpu").onnx_path == Path("/nonexistent/m.onnx")


def test_base_adapter_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseAdapter(_spec(), device="cpu")


def test_base_adapter_satisfies_the_detector_adapter_protocol():
    assert isinstance(_StubAdapter(_spec(), device="cpu"), DetectorAdapter)


# --- the registry -------------------------------------------------------------------

def test_registry_knows_rtdetr():
    assert "rtdetr" in ADAPTERS


def test_adapter_class_resolves_rtdetr_to_a_base_adapter_subclass():
    cls = adapter_class("rtdetr")
    assert issubclass(cls, BaseAdapter)
    assert cls.__name__ == "RTDETRAdapter"


def test_adapter_class_rejects_an_unknown_model_and_names_the_known_ones():
    with pytest.raises(ValueError, match="rtdetr"):
        adapter_class("does-not-exist")


def test_load_adapter_rejects_an_unknown_spec():
    with pytest.raises(ValueError, match="does-not-exist"):
        load_adapter(_spec("does-not-exist"))


def test_load_adapter_accepts_a_registered_stub(monkeypatch):
    """`load_adapter` dispatches on spec.name — callers never name a model class."""
    monkeypatch.setitem(ADAPTERS, "stub", f"{__name__}:_StubAdapter")
    adapter = load_adapter(_spec("stub"), onnx="fp16")
    assert isinstance(adapter, _StubAdapter)
    assert adapter.onnx_path == Path("/nonexistent/m.fp16.onnx")


def test_importing_the_package_does_not_import_a_model_module():
    """Laziness is the point: a nanodet run must not drag in the RT-DETR submodule shim."""
    code = ("import sys, harness.adapter; "
            "print('harness.adapter.rtdetr' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT,
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


# --- device resolution --------------------------------------------------------------

def test_resolve_device_falls_back_to_cpu_without_cuda(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device(_spec(), "cuda") == "cpu"


def test_resolve_device_keeps_an_explicit_cpu_request():
    assert resolve_device(_spec(), "cpu") == "cpu"


def test_resolve_device_defaults_to_the_specs_device():
    assert resolve_device(_spec()) == "cpu"


# --- back-compat surface ------------------------------------------------------------

def test_detections_is_still_importable_from_the_package_and_from_harness():
    from harness import Detections as top_level
    assert top_level is Detections
    assert len(Detections(labels=[], boxes=[], scores=[])) == 0
