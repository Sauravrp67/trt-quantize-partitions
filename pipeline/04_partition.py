"""Stage 04: generate TensorRT precision-partition build configs.

Planned role:
    Convert per-model sensitivity maps into TensorRT build constraints using
    `precisionConstraints=obey` and explicit `layerPrecisions` assignments.

TODO:
    - Read YAML partition specs from `configs/`.
    - Validate that layer names match annotated ONNX graphs.
    - Generate TensorRT builder configuration artifacts.
    - Track four benchmark configs for each model.
"""
