"""Stage 03: build layerwise sensitivity maps.

Planned role:
    Use Polygraphy layerwise comparisons and debug-reduce style isolation to
    generate divergence tables that guide precision partitioning.

TODO:
    - Compare FP32 ONNX, Q/DQ ONNX, and TensorRT engine outputs.
    - Produce layerwise activation divergence summaries.
    - Identify layers that should remain FP16 or FP32.
    - Export markdown and machine-readable sensitivity tables.
"""
