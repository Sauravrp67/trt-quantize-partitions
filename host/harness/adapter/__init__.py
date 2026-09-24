"""The adapter seam: generic code asks for an adapter, it never names a model.

``pipeline/*.py`` steps, ``harness.bench``, and the sensitivity/partition drivers all
work on *whatever model the spec names*. They get it from :func:`load_adapter`, which
dispatches on ``spec.name`` through :data:`ADAPTERS`. Nothing outside this package
imports a concrete adapter class, so adding a model is one module plus one registry line
— no edits to any stage.

Model modules are imported lazily, by string. Importing ``harness.adapter`` therefore
costs nothing model-specific: a NanoDet run never executes RT-DETR's submodule shim.
"""
from __future__ import annotations

from importlib import import_module
from typing import Optional, Type, Union

from ..config import ModelSpec, load_spec
from .base import BaseAdapter, Detections, DetectorAdapter, resolve_device

# spec name (configs/<name>.yaml) -> "module:class", imported on first use
ADAPTERS: dict[str, str] = {
    "rtdetr": "harness.adapter.rtdetr:RTDETRAdapter",
}


def adapter_class(name: str) -> Type[BaseAdapter]:
    """The adapter class registered for a spec name."""
    try:
        target = ADAPTERS[name]
    except KeyError:
        raise ValueError(
            f"no adapter registered for spec '{name}'; known: {sorted(ADAPTERS)}. "
            f"Add one to harness/adapter/ and register it in harness.adapter.ADAPTERS."
        ) from None
    module_name, _, class_name = target.partition(":")
    return getattr(import_module(module_name), class_name)


def load_adapter(spec: Union[ModelSpec, str], *, onnx=None,
                 device: Optional[str] = None) -> BaseAdapter:
    """The adapter for ``spec``, chosen by ``spec.name``.

    ``device`` goes through :func:`resolve_device`, so a caller can pass a ``--device``
    flag straight through (``None`` means the spec's default) and still get the CPU
    fallback when CUDA is missing.
    """
    if not isinstance(spec, ModelSpec):
        spec = load_spec(spec)
    cls = adapter_class(spec.name)
    return cls(spec, onnx=onnx, device=resolve_device(spec, device))


__all__ = [
    "ADAPTERS",
    "BaseAdapter",
    "Detections",
    "DetectorAdapter",
    "adapter_class",
    "load_adapter",
    "resolve_device",
]
