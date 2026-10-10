# Jetson Orin Nano precision benchmark

RT-DETR R101-vd, TensorRT 10.16.2, Jetson Linux R39.2.1, 8 GB Orin Nano, `MAXN_SUPER` power mode. The runner processed the same 468-frame, 1360 × 764 video for each run, with 30 warmup iterations, score threshold 0.5, drawing/video writing disabled, and `tegrastats` power sampling enabled. Values below are medians of three runs.

| Graph / engine precision | Inference p50 (ms) | Inference p90 (ms) | End-to-end p50 (ms) | Pipeline FPS | Mean VDD_IN (W) | Energy (J/frame) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 graph, TF32 allowed | 76.78 | 77.35 | 110.39 | 8.69 | 15.98 | 1.838 |
| FP32, TF32 disabled | 168.73 | 169.12 | 216.10 | 4.51 | 16.84 | 3.738 |
| FP16 | 40.15 | 40.34 | 68.09 | 13.86 | 11.95 | 0.862 |
| INT8 | 25.93 | 26.17 | 50.59 | 18.37 | 10.02 | 0.546 |
| Mixed BF16/FP32 | 90.93 | 91.44 | 122.69 | 7.84 | 15.83 | 2.018 |
| FP8 | Not supported | — | — | — | — | — |

TensorRT permits TF32 by default for FP32 graph operations. The first row is the existing FP32 ONNX engine built with that default. The second is a separate engine built with `--noTF32`; use it as the strict FP32 comparison. This setting makes a large difference on this board.

The INT8 engine had the lowest measured latency and energy use. Relative to the FP32 graph with TF32 allowed, it delivered about 2.1× the pipeline throughput and used about 70% less energy per frame. The measured BF16 engine was a mixed-precision experiment, not a pure BF16 model: NVIDIA ModelOpt converted 1,162 of 1,340 nodes (86.72%) to BF16 and left the rest in FP32. Two Resize nodes were kept in FP32 with casts around them so TensorRT could build the graph. It was slower than both FP32 engine configurations, so BF16 is not a useful deployment target for this model as tested.

TensorRT rejected the FP8 Q/DQ ONNX with `Networks with FP8 Q/DQ layers require hardware with FP8 support`. No FP8 engine or performance result is available for Orin Nano.

This is a runtime benchmark, not an accuracy comparison. The board did not have COCO val2017 or `pycocotools` installed, so mAP was not measured. The per-frame detection counts in the raw run log are diagnostic only. Run [`deploy/eval_map.py`](../../deploy/README.md#evaluate-the-deployed-engines) with COCO before choosing a precision based on accuracy.

Run logs are in [`precision_benchmark_2026-09-runs.log`](precision_benchmark_2026-09-runs.log) and [`precision_benchmark_2026-09-strict-fp32-runs.log`](precision_benchmark_2026-09-strict-fp32-runs.log). Engine build logs are in [`builds/`](builds/).
