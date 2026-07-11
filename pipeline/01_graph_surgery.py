"""Stage 01: normalize and annotate ONNX graphs.

Planned role:
    Use ONNX GraphSurgeon to fold constants, clean graph structure, and assign
    stable names to precision-sensitive layers such as layer norm, softmax, and
    projection blocks.

TODO:
    - Fold constants and remove unused tensors.
    - Name RT-DETR transformer subgraphs for partition specs.
    - Name NanoDet backbone, neck, head, and decode boundaries.
    - Emit an annotated FP32 ONNX model and layer-name manifest.
"""
