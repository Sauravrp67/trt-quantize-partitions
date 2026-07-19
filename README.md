# trt-quantize-partition

Sensitivity-guided precision partitioning for a transformer detector (RT-DETR r18vd) on **TensorRT 11 / RTX 4050 Laptop (sm_89)**, measured for accuracy, latency, power, and perf-per-watt.

**Question:** does a sensitivity-guided INT8/FP16 partition recover INT8's efficiency *and* FP16's accuracy, or must one be traded for the other?

Work in progress. mAP is not yet measured; no accuracy claim is made.

## Status

- [x] ONNX export — static batch=1, opset 20, parity-verified against eager
- [x] PyTorch (eager) runner
- [x] ONNX Runtime runner
- [x] Torch↔ORT decoded-detection parity (100% agreement, `harness/compare.py`)
- [x] Detection visualizer — per-class hue, score-continuous shading (`harness/visualize.py`)
- [x] Frame sources — image / folder / video / camera (`harness/sources.py`)
- [x] Latency metering — CUDA events (`harness/metrics.py`)
- [x] NVML power sampler (`harness/power.py`)
- [x] TensorRT engine builder + per-layer precision readback (`harness/trt_runner.py`)
- [x] TensorRT 11 API contract guard (`tests/test_trt_contract.py`)
- [x] COCO mAP evaluation
- [ ] FP16 graph conversion + ModelOpt INT8 PTQ (Q/DQ ONNX)
- [ ] Benchmark driver → `results/tables/baselines.md`
- [ ] Layerwise sensitivity map → partition N-sweep → Pareto
- [ ] Fused MSDeformAttn CUDA plugin (sm_89) + Nsight profile

## TensorRT 11 constraints

Verified against `tensorrt==11.0.0.114`; pinned by `tests/test_trt_contract.py`.

- `trt.BuilderFlag` exposes no `FP16` / `INT8` / `BF16` / `FP8`. Only `TF32`.
- `builder.create_network(0)` is always `STRONGLY_TYPED`. No weakly-typed mode.
- `trt.ILayer.precision` / `.set_output_type` and `trt.IInt8Calibrator` do not exist.
- `trtexec --stronglyTyped` is a no-op; there is no `--fp16` / `--int8` / `--layerPrecisions`.
- `polygraphy.CreateConfig(fp16=True)` raises.

Consequence: **precision is declared in the ONNX graph, not by a builder flag.** `build_engine()` takes no `precision` argument.

| Config | Declared by |
| --- | --- |
| FP32 | graph as exported |
| FP16 | FP16 weights + `Cast` at the I/O boundary |
| INT8 | `QuantizeLinear` / `DequantizeLinear` nodes |
| Partitioned | FP16/Q-DQ with a node block list (sensitive layers held higher) |

`TF32` is on by default in TensorRT and silently cleared by `polygraphy.CreateConfig()`. `build_engine()` builds `IBuilderConfig` by hand and takes `tf32` explicitly; it is reported as a benchmark variable.

## Reproduce

```bash
git clone --recurse-submodules <repo> && cd trt-quantize-partition
pip install -r requirements.txt
```

Checkpoint → `models/rtdetr/checkpoints/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth`
([upstream](https://github.com/lyuwenyu/RT-DETR)).

**Export** → `models/rtdetr/model.onnx` (`images [1,3,640,640]` → `pred_logits [1,300,80]`, `pred_boxes [1,300,4]`):

```bash
python models/rtdetr/export.py \
  --config RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_r18vd_6x_coco.yml \
  --ckpt   models/rtdetr/checkpoints/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth
```

**Torch vs ONNX Runtime parity** — same tensor into both backends, same postprocessor, detections matched by same-label IoU:

```bash
python models/rtdetr/infer.py --source images/1.jpg
python models/rtdetr/infer.py --source images/
python models/rtdetr/infer.py --source clip.mp4 --save
python models/rtdetr/infer.py --source 0 --show
```

**Build engine + verify realized precision:**

```python
import numpy as np
from harness.trt_runner import build_engine, save_engine, TRTSession, layer_precisions

engine = build_engine("models/rtdetr/model.onnx", tf32=True)   # precision comes from the graph
save_engine(engine, "results/engines/rtdetr_fp32.engine")
layer_precisions(engine)                                       # {layer: {"Float"|"Half"|"Int8"}}

sess = TRTSession(engine)
sess.run(np.random.rand(1, 3, 640, 640).astype("float32"))
sess.close()
```

**Benchmark** (build in Python so the builder config is pinned; time with `trtexec`):

```bash
bash scripts/lock_clocks.sh
trtexec --loadEngine=results/engines/rtdetr_fp32.engine --useCudaGraph --noDataTransfers
bash scripts/unlock_clocks.sh
```

`TrtRunner` wall-clock and `trtexec` GPU-compute time are not comparable; do not mix them.

## Results

FP32 engine · RTX 4050 Laptop (sm_89) · batch=1 · 640×640 · CUDA graph · transfers excluded.

| | |
| --- | --- |
| GPU compute latency p50 | 8.23 ms |
| p90 / p99 | 8.47 / 8.87 ms |
| Throughput | 121 qps |
| Engine size | 89.0 MiB |
| Engine layers (post-fusion) | 286 |
| Torch ↔ ORT detection agreement | 100% |
| mAP@0.5:0.95 | not measured |

## Layout

```
models/rtdetr/   export.py (ONNX) · infer.py (DetectorAdapter + CLI)
models/nanodet/  reserved adapter seam                          [stub]
harness/         compare · visualize · sources · metrics · power · parity
                 · paths · trt_runner            (model-agnostic)
pipeline/        00..06 stages, one per NPU-compiler step       [stubs]
configs/         precision-partition specs                      [stubs]
tests/           power math · TensorRT 11 API contract
scripts/         COCO download · GPU clock lock/unlock
results/tables/  committed result tables
RT-DETR/         upstream submodule — read-only, never modified
```

`*.onnx`, `*.engine`, `*.pth`, checkpoints, and `results/figures/` are gitignored.

A detector plugs in by implementing `harness.compare.DetectorAdapter`
(`build_torch` / `preprocess` / `postprocess`); everything in `harness/` is model-agnostic.

## Implementation notes

Non-obvious behaviors that fail silently if unhandled:

- `YAMLConfig(resume=…)` records a checkpoint path but **does not load weights**. `models/rtdetr/export.py` and `infer.py` load `ckpt["ema"]["module"]` explicitly. An unloaded model is detectable only by logits clustered at the focal-loss prior bias (−log 99 ≈ −4.6).
- RT-DETR's top-300 query selection has ties that break differently in torch vs ORT, so sub-threshold background queries reshuffle. Raw-logit `allclose` therefore fails by design; **decoded-detection agreement is the parity signal**, and matching must be same-label IoU, not positional.
- Preprocessing is a plain 640×640 bilinear resize to [0,1]. **No ImageNet mean/std.**
- `layer_precisions()` unions Inputs+Constants+Outputs: TensorRT fuses regions into single Myelin layers whose outputs sit on the FP32 I/O boundary even when the interior computes in FP16.
- MSDeformAttn materializes in the exported ONNX as **9 `GridSample` nodes** — simultaneously the decoder latency hotspot and a precision-sensitive op.

## Tests

```bash
pytest -q     # 6 passed
```

`tests/test_trt_contract.py` is an environment guard: it fails if a TensorRT upgrade restores the flag-based precision API, whose absence the precision strategy above depends on.

## Environment

Python 3.13 (conda) · PyTorch cu132 · TensorRT 11.0.0.114 · Polygraphy 0.50.3 · ONNX Runtime 1.27 (CUDA EP) · NVIDIA ModelOpt 0.44 · CUDA 12 · RTX 4050 Laptop (sm_89), ~55 W envelope.

RT-DETR is vendored as a submodule from [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) and consumed read-only.
