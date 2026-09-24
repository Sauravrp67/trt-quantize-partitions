"""Parity and tolerance-reporting harness placeholder.

Planned role:
    Wrap Polygraphy comparisons for ONNX Runtime, TensorRT, and intermediate
    debug artifacts.

TODO:
    - Define model-specific tolerance profiles.
    - Report absolute, relative, and task-level output drift.
    - Save parity summaries for FP32 ONNX and TensorRT engines.
"""
from typing import Optional

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

RTOL = 1e-2
ATOL = 1e-3

# ONNX Runtime tensor element types -> numpy dtypes for input synthesis.
_ORT_DTYPE_TO_NUMPY = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(uint8)": np.uint8,
}


def _resolve_shape(shape) -> tuple:
    """Replace dynamic axes (strings / None) with 1 for a concrete sample shape."""
    return tuple(dim if isinstance(dim, int) and dim > 0 else 1 for dim in shape)


def _flatten_torch_output(output) -> "dict[Optional[str], torch.Tensor]":
    """Normalize an eager forward output into name -> tensor pairs.

    Handles a single tensor, a (possibly nested) tuple/list of tensors, and a
    dict of tensors (RT-DETR returns e.g. ``{"pred_logits": ..., "pred_boxes": ...}``).
    Names are kept when available so they can be matched against ONNX output names;
    positional outputs are keyed by ``None`` and matched by order.
    """
    named: "dict[Optional[str], torch.Tensor]" = {}
    if isinstance(output, torch.Tensor):
        named[None] = output
    elif isinstance(output, dict):
        for key, value in output.items():
            if isinstance(value, torch.Tensor):
                named[key] = value
    elif isinstance(output, (tuple, list)):
        for value in output:
            if isinstance(value, torch.Tensor):
                named[None if None in named else f"__pos_{len(named)}"] = value
    else:
        raise TypeError(f"Unsupported eager output type for parity check: {type(output)}")
    return named


def verify_parity(
    ort_session: ort.InferenceSession,
    model: nn.Module,
    sample_input: Optional[torch.Tensor] = None,
    rtol: float = RTOL,
    atol: float = ATOL,
    seed: int = 0,
) -> bool:
    """Verify ONNX Runtime outputs match the model's eager forward pass.

    Synthesizes (or accepts) a single input, runs both the eager ``model`` and the
    ONNX ``ort_session`` on it, and reports per-output absolute/relative drift. The
    eager output dictates dtype/device; the same numpy input feeds ONNX Runtime so
    only the export/runtime numerics differ.

    Args:
        ort_session: ONNX Runtime session for the exported model.
        model: PyTorch module the ONNX model was exported from.
        sample_input: Optional input tensor. If omitted, a random tensor is built
            from the ONNX graph's declared input shape and dtype (static batch=1
            export, so dynamic axes collapse to 1).
        rtol, atol: Tolerances forwarded to ``numpy.allclose`` / ``testing``.
        seed: RNG seed for reproducible synthetic input.

    Returns:
        ``True`` if every matched output is within tolerance, else ``False``.
    """
    ort_inputs = ort_session.get_inputs()
    if len(ort_inputs) != 1:
        raise NotImplementedError(
            f"verify_parity supports single-input models; got {len(ort_inputs)} inputs."
        )
    input_meta = ort_inputs[0]

    model.eval()
    device = next(model.parameters()).device

    # Build the shared input, letting the eager output define the canonical dtype.
    if sample_input is None:
        np_dtype = _ORT_DTYPE_TO_NUMPY.get(input_meta.type, np.float32)
        shape = _resolve_shape(input_meta.shape)
        torch.manual_seed(seed)
        if np.issubdtype(np_dtype, np.floating):
            sample_input = torch.rand(*shape, dtype=torch.float32)
        else:
            sample_input = torch.zeros(*shape, dtype=torch.from_numpy(np.zeros(1, np_dtype)).dtype)
    sample_input = sample_input.to(device)

    with torch.no_grad():
        eager_output = model(sample_input)
    eager_named = _flatten_torch_output(eager_output)

    ort_input_np = sample_input.detach().cpu().numpy().astype(
        _ORT_DTYPE_TO_NUMPY.get(input_meta.type, np.float32)
    )
    ort_outputs = ort_session.run(None, {input_meta.name: ort_input_np})
    ort_meta = ort_session.get_outputs()

    # Match ONNX outputs to eager tensors by name when possible, else by order.
    eager_list = list(eager_named.items())
    all_close = True
    for idx, (onnx_meta, onnx_value) in enumerate(zip(ort_meta, ort_outputs)):
        if onnx_meta.name in eager_named:
            eager_tensor = eager_named[onnx_meta.name]
            label = onnx_meta.name
        elif idx < len(eager_list):
            label, eager_tensor = eager_list[idx]
            label = onnx_meta.name if label is None else str(label)
        else:
            print(f"[parity] {onnx_meta.name}: no matching eager output, skipped")
            continue

        eager_np = eager_tensor.detach().cpu().numpy()
        if eager_np.shape != onnx_value.shape:
            print(
                f"[parity] {label}: SHAPE MISMATCH eager={eager_np.shape} onnx={onnx_value.shape}"
            )
            all_close = False
            continue

        diff = np.abs(eager_np.astype(np.float64) - onnx_value.astype(np.float64))
        denom = np.maximum(np.abs(eager_np.astype(np.float64)), atol)
        max_abs = float(diff.max()) if diff.size else 0.0
        max_rel = float((diff / denom).max()) if diff.size else 0.0
        passed = np.allclose(eager_np, onnx_value, rtol=rtol, atol=atol)
        all_close = all_close and passed
        status = "OK" if passed else "FAIL"
        print(f"[parity] {label}: {status} max_abs={max_abs:.3e} max_rel={max_rel:.3e}")

    print(f"[parity] overall: {'PASS' if all_close else 'FAIL'} (rtol={rtol}, atol={atol})")
    return all_close


