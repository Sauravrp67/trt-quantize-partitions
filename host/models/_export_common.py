"""Shared raw-output ONNX export for the PyTorch RT-DETR family.

Model-specific modules own upstream configuration and checkpoint loading. This
helper only enforces the ONNX I/O contract used by the host pipeline.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class RawDetector(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model.deploy().eval()

    def forward(self, images):
        output = self.model(images)
        return output["pred_logits"], output["pred_boxes"]


def load_checkpoint(model: nn.Module, checkpoint: Path) -> None:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    saved = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state = saved["ema"]["module"] if "ema" in saved else saved["model"]
    model.load_state_dict(state, strict=True)


def export_raw(model: nn.Module, spec, out_path: Path) -> None:
    if spec.batch != 1 or spec.output_names != ["pred_logits", "pred_boxes"]:
        raise ValueError("raw RT-DETR export requires batch 1 and pred_logits,pred_boxes outputs")
    wrapped = RawDetector(model)
    sample = torch.rand(1, 3, spec.img_size, spec.img_size)
    with torch.inference_mode():
        logits, boxes = wrapped(sample)
    if logits.shape != (1, 300, spec.num_classes) or boxes.shape != (1, 300, 4):
        raise ValueError(f"unexpected raw output shapes: logits={tuple(logits.shape)}, "
                         f"boxes={tuple(boxes.shape)}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped, (sample,), str(out_path),
        input_names=[spec.input_name], output_names=spec.output_names,
        opset_version=17, dynamo=False, do_constant_folding=True,
    )
    # The v4 decoder traces a static query count through a reshape that ONNX
    # annotates symbolically. The eager shape above is checked, and this export
    # deliberately fixes batch and image size, so record its concrete I/O shape.
    import onnx

    graph = onnx.load(str(out_path))
    for output, expected in zip(graph.graph.output,
                                ((1, 300, spec.num_classes), (1, 300, 4))):
        dims = output.type.tensor_type.shape.dim
        if len(dims) != 3:
            raise ValueError(f"unexpected ONNX output rank: {output.name}")
        for dim, size in zip(dims, expected):
            if dim.HasField("dim_value") and dim.dim_value != size:
                raise ValueError(f"unexpected ONNX output shape: {output.name}")
            dim.Clear()
            dim.dim_value = size
    onnx.checker.check_model(graph)
    onnx.save(graph, str(out_path))
