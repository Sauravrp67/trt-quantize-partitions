"""RT-DETR adapter — the model-specific half of the seam, plus the submodule shim.

Two things live here that every RT-DETR entrypoint (``export``, ``infer``, ``eval_map``)
needs, and that no entrypoint should be re-deriving:

1. :func:`install_src_package` — the workaround for the read-only submodule. RT-DETR's
   ``src/__init__.py`` eagerly imports data modules tied to old torchvision beta APIs, so
   a fake ``src`` package is registered in ``sys.modules`` before importing only the parts
   that matter (``src.core``, and ``src.nn`` / ``src.zoo`` for their class registrations).
2. :class:`RTDETRAdapter` — preprocess / postprocess / class names, shared by all backends.

Both are driven by a :class:`~harness.config.ModelSpec` (``configs/rtdetr.yaml``); nothing
here hard-codes a path, a class list, or an image size.

This module is imported lazily by :func:`harness.adapter.load_adapter`, so importing
``harness.adapter`` never touches the RT-DETR submodule.
"""
from __future__ import annotations

import sys
import types
from typing import Optional

import numpy as np
import torch
from PIL import Image

from ..config import ModelSpec
from ..paths import RTDETR_PYTORCH_ROOT, RTDETR_SRC_ROOT, require
from .base import BaseAdapter, Detections

_SRC_INSTALLED = False


def install_src_package() -> None:
    """Make ``import src`` resolve to the RT-DETR submodule, skipping its data stack."""
    global _SRC_INSTALLED
    if _SRC_INSTALLED:
        return
    require(RTDETR_SRC_ROOT, "RT-DETR PyTorch checkout",
            "run: git submodule update --init --recursive  (or set $RTDETR_ROOT)")
    if str(RTDETR_PYTORCH_ROOT) not in sys.path:
        sys.path.insert(0, str(RTDETR_PYTORCH_ROOT))
    pkg = types.ModuleType("src")
    pkg.__file__ = str(RTDETR_SRC_ROOT / "__init__.py")
    pkg.__path__ = [str(RTDETR_SRC_ROOT)]
    sys.modules["src"] = pkg
    _SRC_INSTALLED = True


def build_config(spec: ModelSpec, *, weights: bool = True):
    """``YAMLConfig`` for ``spec``, weights loaded unless ``weights=False``.

    ``weights=False`` touches neither the checkpoint nor ``config.model``: the
    postprocessor is stateless decode logic, so an ORT/TensorRT run — where the network
    lives in the ONNX — has no reason to build and populate an eager model it never calls.
    """
    install_src_package()
    from src.core import YAMLConfig   # noqa: E402  (needs the shim above)
    import src.nn    # noqa: F401  registers backbones
    import src.zoo   # noqa: F401  registers RT-DETR + postprocessors

    require(spec.torch_config, "RT-DETR model config")
    config = YAMLConfig(str(spec.torch_config), resume=str(spec.ckpt))
    if weights:
        load_weights(config, spec)
    return config


def load_weights(config, spec: ModelSpec) -> None:
    """Load the checkpoint into ``config.model`` (EMA if present).

    ``YAMLConfig(resume=…)`` only *records* the path — without this the model silently
    keeps its init weights, detectable only by logits at the focal-loss prior bias.
    """
    require(spec.ckpt, "RT-DETR checkpoint",
            "download it into host/models/rtdetr/checkpoints/ (see host/models/rtdetr/README.md)")
    ckpt = torch.load(str(spec.ckpt), map_location="cpu")
    state = ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]
    config.model.load_state_dict(state)


class RTDETRAdapter(BaseAdapter):
    """RT-DETR's :class:`~harness.adapter.BaseAdapter`."""

    spec_name = "rtdetr"

    def __init__(self, spec: Optional[ModelSpec] = None, *, onnx=None,
                 device: Optional[str] = None) -> None:
        super().__init__(spec, onnx=onnx, device=device)

        # decode-only: the checkpoint is loaded on first build_torch(), not here, so an
        # ORT/TensorRT run never pays for an eager model it will not call
        self._config = build_config(self.spec, weights=False)
        self._weights_loaded = False
        self._postprocessor = self._config.postprocessor.deploy().to(self.device).eval()

    def build_torch(self) -> torch.nn.Module:
        if not self._weights_loaded:
            load_weights(self._config, self.spec)
            self._weights_loaded = True
        return self._config.model.deploy().to(self.device).eval()

    def preprocess(self, frame: Image.Image):
        """RGB PIL -> ((1, 3, S, S) float32 CPU tensor, meta). No mean/std normalization."""
        w, h = frame.size
        size = self.img_size
        resized = frame.resize((size, size), Image.BILINEAR)
        x = (
            torch.from_numpy(np.asarray(resized, dtype=np.float32))
            .permute(2, 0, 1)   # HWC -> CHW
            .div_(255.0)        # [0, 255] -> [0, 1]
            .unsqueeze(0)       # -> (1, 3, S, S)
            .contiguous()
        )
        return x, {"orig_size": (w, h)}

    def postprocess(self, named, meta) -> Detections:
        """Raw named outputs (numpy) -> pixel-space Detections via the shared postprocessor."""
        w, h = meta["orig_size"]
        logits = torch.from_numpy(np.asarray(named[self.output_names[0]])).to(self.device)
        boxes = torch.from_numpy(np.asarray(named[self.output_names[1]])).to(self.device)
        orig_size = torch.tensor([[w, h]], device=self.device)
        with torch.no_grad():
            labels, out_boxes, scores = self._postprocessor(
                {"pred_logits": logits, "pred_boxes": boxes}, orig_size
            )
        return Detections(
            labels[0].detach().cpu().numpy(),
            out_boxes[0].detach().cpu().numpy(),
            scores[0].detach().cpu().numpy(),
        )
