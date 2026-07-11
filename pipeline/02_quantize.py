"""Stage 02: quantize annotated ONNX models with ModelOpt INT8 PTQ.

Planned role:
    Produce TensorRT-compatible Q/DQ ONNX models using NVIDIA ModelOpt
    post-training quantization.

TODO:
    - Define representative calibration dataset loaders.
    - Configure INT8 PTQ recipes per model.
    - Export Q/DQ ONNX artifacts.
    - Record calibration set hashes and quantization metadata.
"""
