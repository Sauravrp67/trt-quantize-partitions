# trt-quantize-partition

Sensitivity-guided precision partitioning for a transformer detector (RT-DETR) on **TensorRT 11 / RTX 4050 (sm_89)**, measured for accuracy, latency, power, and perf-per-watt. Backbone and checkpoint are selected in `configs/rtdetr.yaml`; results below are r18vd.

Work in progress.

## Status

- [x] ONNX export — static batch=1, opset 20, parity-verified against eager
- [x] PyTorch (eager) runner
- [x] ONNX Runtime runner
- [x] Torch↔ORT decoded-detection parity (100% agreement, `harness/compare.py`)
- [x] Detection visualizer — per-class hue, score-continuous shading (`harness/visualize.py`)
- [x] Frame sources — image / folder / video / camera (`harness/sources.py`)
- [x] Latency metering — CUDA events (`harness/metrics.py`)
- [x] NVML power sampler (`harness/power.py`)
- [x] TensorRT engine builder — reproducible (timing cache + tactic-plan fingerprint), per-layer precision readback (`harness/trt_runner.py`)
- [x] TensorRT 11 API contract guard (`tests/test_trt_contract.py`)
- [x] Decoupled backends — torch / ORT / TensorRT as peers behind one runner (`harness/infer_{torch,ort,engine}.py`, `harness/runner.py`)
- [x] COCO mAP evaluation — any backend, any precision variant (`harness/coco_eval.py`, `models/rtdetr/eval_map.py`)
- [x] YAML model specs — static settings in `configs/*.yaml`, machine paths resolved in `harness/paths.py`
- [x] FP16 graph conversion — FP32→FP16 ONNX, I/O kept FP32, converter-bug sanitized (`harness/precision.py`)
- [ ] ModelOpt INT8 PTQ (Q/DQ ONNX)
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

Checkpoint → `models/rtdetr/checkpoints/` ([upstream](https://github.com/lyuwenyu/RT-DETR)).

**Configure.** Which checkpoint, which upstream config, ONNX filenames per precision, tensor
names, image size, and CLI defaults all live in `configs/rtdetr.yaml`. Paths there are written
against roots (`${models}`, `${rtdetr_pytorch}`, `${figures}`, …) resolved per machine by
`harness/paths.py`, so nothing local is checked in. Every command below reads it; flags only
override it, and `--onnx` takes a variant name (`fp32`, `fp16`) or a path.

```bash
python -c "from harness.config import load_spec; print(load_spec('rtdetr'))"   # what resolved where
```

Roots are overridable per machine: `TRTQP_ROOT`, `TRTQP_DATA`, `TRTQP_COCO`, `TRTQP_RESULTS`, `RTDETR_ROOT`.

**Export** → the spec's `fp32` variant (`images [1,3,640,640]` → `pred_logits [1,300,80]`, `pred_boxes [1,300,4]`):

```bash
python models/rtdetr/export.py                       # or: --onnx <variant|path> --no-report
```

**Inference** — each backend in its own window, or side-by-side with same-label IoU agreement
against the first:

```bash
python models/rtdetr/infer.py --source images/1.jpg                        # spec's default backends
python models/rtdetr/infer.py --source images/ --backends torch,ort --compare
python models/rtdetr/infer.py --source clip.mp4 --backends trt --onnx fp16 --save
python models/rtdetr/infer.py --source 0 --show                            # camera
```

TensorRT builds from the selected ONNX unless `--engine <file>` is given.

**COCO mAP** — several precision variants in one pass; Δ columns are against the first:

```bash
python models/rtdetr/eval_map.py --backend torch                  # eager reference
python models/rtdetr/eval_map.py --onnx fp32 fp16 --limit 500     # ORT
python models/rtdetr/eval_map.py --backend trt --onnx fp16        # what actually ships
```

**FP16 graph** — precision is declared in the ONNX (see the table above), not by a flag:

```bash
python -c "from harness.precision import to_fp16; to_fp16('models/rtdetr/model.onnx', 'models/rtdetr/model_fp16.onnx')"
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

FP32 engine · RTX 4050 (sm_89) · batch=1 · 640×640 · CUDA graph · transfers excluded.

| | |
| --- | --- |
| GPU compute latency p50 | 8.23 ms |
| p90 / p99 | 8.47 / 8.87 ms |
| Throughput | 121 qps |
| Engine size | 89.0 MiB |
| Engine layers (post-fusion) | 286 |
| Torch ↔ ORT detection agreement | 100% |
| mAP@0.5:0.95 (eager, val2017 5000) | 0.4640 |
| mAP@0.5 (eager, val2017 5000) | 0.6372 |

Eager mAP matches the upstream r18vd number (46.5), so the checkpoint and postprocessor are
wired correctly — it is the reference every exported/quantized variant is measured against
(`results/tables/rtdetr_map.md`).

## Layout

```
models/rtdetr/   adapter.py (DetectorAdapter + submodule shim)
                 export.py · infer.py · eval_map.py            (thin CLIs)
models/nanodet/  reserved adapter seam                          [stub]
harness/         adapter (seam) · infer_torch/infer_ort/infer_engine (backends)
                 · runner · compare · visualize · sources · metrics · power
                 · parity · coco_eval · precision · trt_runner
                 · config (YAML specs) · paths (machine roots)  (model-agnostic)
pipeline/        00..06 stages, one per NPU-compiler step       [stubs]
configs/         rtdetr.yaml (model spec) · classes/coco80.yaml
                 · precision-partition specs                    [stubs]
tests/           power · precision · config · TensorRT 11 API contract
scripts/         COCO download · GPU clock lock/unlock
results/tables/  committed result tables
RT-DETR/         upstream submodule — read-only, never modified
```

`*.onnx`, `*.engine`, `*.pth`, checkpoints, and `results/figures/` are gitignored.

A detector plugs in by writing a `configs/<model>.yaml` and implementing
`harness.adapter.DetectorAdapter` (`build_torch` / `preprocess` / `postprocess`);
everything in `harness/` is model-agnostic. Backends are peers — each exposes
`label` / `infer(x) -> {name: ndarray}` / `close()`, so N of them run over one source with
no per-backend branching.

## Implementation notes

Non-obvious behaviors that fail silently if unhandled:

- `YAMLConfig(resume=…)` records a checkpoint path but **does not load weights**. `models/rtdetr/adapter.py::build_config` loads `ckpt["ema"]["module"]` explicitly. An unloaded model is detectable only by logits clustered at the focal-loss prior bias (−log 99 ≈ −4.6).
- RT-DETR's `src/__init__.py` eagerly imports data modules tied to old torchvision beta APIs. `adapter.py::install_src_package` registers a fake `src` package in `sys.modules` first, then imports only `src.core` / `src.nn` / `src.zoo` — the submodule stays pristine.
- RT-DETR's top-300 query selection has ties that break differently in torch vs ORT, so sub-threshold background queries reshuffle. Raw-logit `allclose` therefore fails by design; **decoded-detection agreement is the parity signal**, and matching must be same-label IoU, not positional.
- Preprocessing is a plain 640×640 bilinear resize to [0,1]. **No ImageNet mean/std.**
- `layer_precisions()` unions Inputs+Constants+Outputs: TensorRT fuses regions into single Myelin layers whose outputs sit on the FP32 I/O boundary even when the interior computes in FP16.
- MSDeformAttn materializes in the exported ONNX as **9 `GridSample` nodes** — simultaneously the decoder latency hotspot and a precision-sensitive op.

## Tests

```bash
pytest -q tests/test_power.py tests/test_precision.py tests/test_trt_contract.py tests/test_config.py   # 15 passed, no data needed
```

`tests/test_trt_contract.py` is an environment guard: it fails if a TensorRT upgrade restores the flag-based precision API, whose absence the precision strategy above depends on. `tests/test_coco_eval.py` additionally needs COCO val2017 (`scripts/download_coco_val.sh`).

## Environment

Python 3.13 (conda) · PyTorch cu132 · TensorRT 11.0.0.114 · Polygraphy 0.50.3 · ONNX Runtime 1.27 (cu13 CUDA EP) · NVIDIA ModelOpt 0.44 (INT8 PTQ; isolated env) · CUDA 13 · RTX 4050 Laptop (sm_89), ~55 W envelope.

RT-DETR is vendored as a submodule from [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) and consumed read-only.
