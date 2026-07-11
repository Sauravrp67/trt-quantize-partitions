"""Stage 00: export PyTorch detectors to FP32 ONNX.

Planned role:
    Convert NanoDet and RT-DETR checkpoints into static batch=1 FP32 ONNX
    artifacts for parity checking and downstream TensorRT compilation.

TODO:
    - Dispatch to model-specific export stubs under `models/`.
    - Enforce static batch=1 and explicit input/output tensor names.
    - Save export metadata for calibration, evaluation, and graph surgery.
"""
