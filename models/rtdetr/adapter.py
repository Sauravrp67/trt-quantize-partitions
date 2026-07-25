"""RT-DETR adapter — the model-specific half of the harness seam, plus the submodule shim.

Two things live here that every RT-DETR entrypoint (``export``, ``infer``, ``eval_map``)
needs, and that no entrypoint should be re-deriving:

1. :func:`install_src_package` — the workaround for the read-only submodule. RT-DETR's
   ``src/__init__.py`` eagerly imports data modules tied to old torchvision beta APIs, so
   a fake ``src`` package is registered in ``sys.modules`` before importing only the parts
   that matter (``src.core``, and ``src.nn`` / ``src.zoo`` for their class registrations).
2. :class:`RTDETRAdapter` — preprocess / postprocess / class names, shared by all backends.

Both are driven by a :class:`~harness.config.ModelSpec` (``configs/rtdetr.yaml``); nothing
here hard-codes a path, a class list, or an image size.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from harness.adapter import Detections
from harness.config import ModelSpec, load_spec
from harness.paths import RTDETR_PYTORCH_ROOT, RTDETR_SRC_ROOT, require

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


def build_config(spec: ModelSpec):
    """``YAMLConfig`` with the checkpoint's weights loaded (EMA if present)."""
    install_src_package()
    from src.core import YAMLConfig   # noqa: E402  (needs the shim above)
    import src.nn    # noqa: F401  registers backbones
    import src.zoo   # noqa: F401  registers RT-DETR + postprocessors

    require(spec.torch_config, "RT-DETR model config")
    require(spec.ckpt, "RT-DETR checkpoint",
            "download it into models/rtdetr/checkpoints/ (see models/rtdetr/README.md)")
    config = YAMLConfig(str(spec.torch_config), resume=str(spec.ckpt))
    ckpt = torch.load(str(spec.ckpt), map_location="cpu")
    state = ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]
    config.model.load_state_dict(state)
    return config


class RTDETRAdapter:
    """DetectorAdapter for RT-DETR. See ``harness.adapter.DetectorAdapter``."""

    def __init__(self, spec: Optional[ModelSpec] = None, *, onnx=None,
                 device: Optional[str] = None) -> None:
        self.spec = spec if spec is not None else load_spec("rtdetr")
        self.name = self.spec.name
        self.class_names = self.spec.class_names
        self.input_name = self.spec.input_name
        self.output_names = list(self.spec.output_names)
        self.img_size = self.spec.img_size
        self.onnx_path = self.spec.onnx_path(onnx)
        self.device = device or self.spec.defaults.get("device", "cuda")

        self._config = build_config(self.spec)
        self._postprocessor = self._config.postprocessor.deploy().to(self.device).eval()

    def build_torch(self) -> torch.nn.Module:
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


def resolve_device(spec: ModelSpec, override: Optional[str] = None, tag: str = "rtdetr") -> str:
    """Requested device, downgraded to CPU (with a notice) when CUDA is unavailable."""
    device = override or spec.defaults.get("device", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[{tag}] CUDA unavailable; falling back to CPU")
        return "cpu"
    return device
