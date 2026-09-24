# COCO mAP — rtdetr

backend: `ONNX Runtime (CUDA)` · images: 5000 of val2017

| model | mAP@50-95 | mAP@50 | Δ mAP@50-95 | Δ mAP@50 |
|---|---|---|---|---|
| `model_r101.onnx` | 0.5624 | 0.7454 | ref | ref |
| `model_r101.fp16.onnx` | 0.5620 | 0.7457 | -0.0004 | +0.0003 |
| `model_r101.int8.onnx` | 0.5394 | 0.7325 | -0.0230 | -0.0129 |
| `model_r101.fp8.onnx` | 0.5569 | 0.7417 | -0.0055 | -0.0037 |

_Accuracy only — no score threshold applied (COCOeval sweeps them). For latency see `models/rtdetr/infer.py` or `trtexec` with locked clocks._

_R101 FP8 was calibrated with 200 COCO images using ModelOpt 0.45.0 and evaluated with ONNX Runtime CUDA on the host. It is an accuracy experiment, not a Jetson Orin Nano deployment result: Orin Nano (SM 8.7) has no FP8 Tensor Core support._
