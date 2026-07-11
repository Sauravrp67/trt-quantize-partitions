"""NanoDet model export entrypoint placeholder.

Planned role:
    Export a NanoDet PyTorch checkpoint to a static batch=1 FP32 ONNX model
    consumed by the shared pipeline.

TODO:
    - Load the selected NanoDet configuration and checkpoint.
    - Freeze preprocessing and decode assumptions.
    - Export FP32 ONNX with stable input/output names.
    - Emit metadata needed by graph surgery and benchmark stages.
"""
