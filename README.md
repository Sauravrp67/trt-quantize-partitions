# TensorRT Quantization and Precision Partitioning

Skeleton for a TensorRT quantize -> graph-surgery -> precision-partition -> compile pipeline.

Target scope:

- Models: NanoDet CNN detector and RT-DETR transformer detector.
- Hardware: power-constrained Ada GPU, RTX 4050, `sm_89`, 55 W.
- Deployment analogy: mirror the Vitis AI `vai_q` -> `vai_c` workflow on the GPU side.
- Precision path: FP32 ONNX parity, INT8 PTQ Q/DQ export, layer sensitivity mapping, mixed-precision TensorRT build, and benchmark reporting.

This repository currently contains only structure, TODO stubs, and documentation placeholders.

## Pipeline

1. Export PyTorch checkpoints to FP32 ONNX with static batch size 1.
2. Normalize and annotate the ONNX graph with ONNX GraphSurgeon.
3. Quantize with NVIDIA ModelOpt INT8 PTQ into Q/DQ ONNX.
4. Build a layerwise sensitivity map with Polygraphy and debug-reduce style isolation.
5. Partition precision with TensorRT `precisionConstraints=obey` and explicit layer precision specs.
6. Compile, inspect, and visualize TensorRT engines with layer info and TREX.
7. Benchmark latency, throughput, mAP, and perf-per-watt across four configs and two models.

## Status

Implementation is intentionally deferred. See `docs/` and each stage script for planned deliverables.
