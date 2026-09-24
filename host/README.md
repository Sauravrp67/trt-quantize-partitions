# Host model preparation

Use this workflow to export custom checkpoints, convert precision, calibrate INT8 models, or work on sensitivity and partitioning. To run a prepared ONNX model, use [Jetson deployment](../deploy/README.md).

## Environment

Run the commands below from the repository root unless a block explicitly changes directory.

```bash
git submodule update --init --recursive
```

Use a separate host environment with a compatible PyTorch/torchvision installation. `requirements.txt` records the existing host tooling, including TensorRT 11; it is not a Jetson dependency list. Install the remaining host dependencies into that environment:

```bash
python -m pip install -r host/requirements.txt
```

ModelOpt quantization has a separate dependency file, `host/requirements-quantization.txt`, because its ONNX Runtime dependencies can conflict with the host evaluation environment. Install it in a separate environment with compatible PyTorch/torchvision. The host dependency sets are retained from development and have not been validated as a universal installer.

## Configure and export

`host/configs/rtdetr.yaml` selects the upstream model configuration, checkpoint, and precision variants. Place the matching checkpoint in `host/models/rtdetr/checkpoints/`. The R101 entry uses the Objects365-pretrained release.

```bash
python host/models/rtdetr/export.py --backbone r101vd
```

FP16 conversion uses the host precision utility:

```bash
cd host
python -c "from harness.precision import to_fp16; to_fp16('models/rtdetr/model_r101.onnx', 'models/rtdetr/model_r101.fp16.onnx')"
cd ..
```

In the ModelOpt environment, calibrate INT8 using COCO images:

```bash
python host/pipeline/quantize.py --backbone r101vd --calib 200
```

The quantization stage explicitly requests INT8 Q/DQ and writes `model_r101.int8.onnx`. Calibration data must be available at `data/coco/` or through `TRTQP_COCO`. Model export and quantization have not been rerun as part of the directory migration.

## Transfer to Jetson

```bash
ssh jetson 'mkdir -p ~/rtdetr/artifacts/onnx'
scp host/models/rtdetr/model_r101.fp16.onnx jetson:~/rtdetr/artifacts/onnx/
```

Copy external tensor data alongside the ONNX when present. On the board, use `deploy/build_engines.sh` and `deploy/infer.py`. If the repo is not cloned there, also copy the complete `deploy/` directory as described in its README.

## Host evaluation and development

```bash
python host/models/rtdetr/eval_map.py \
  --backbone r101vd --backend ort --precision fp32 fp16 int8 \
  --out results/tables/host_r101_map.md
```

`--backend trt` builds engines in the host environment; it does not measure existing Jetson plans. For deployment accuracy, use `deploy/eval_map.py` on the board.

| Directory | Responsibility |
| --- | --- |
| `models/rtdetr/` | Export and host inference/evaluation entry points. |
| `harness/` | Host adapters, runtime backends, precision and graph utilities. |
| `configs/` | Checkpoints, upstream configurations, and model variants. |
| [`pipeline/`](pipeline/README.md) | Shared ONNX preparation and analysis steps. |
| `scripts/` | Host experiments and clock controls. |
| `tests/` | Host tooling tests, including the TensorRT 11 contract guard. |

Datasets, results, and the upstream `RT-DETR/` submodule remain at the repository root. `TRTQP_ROOT` refers to that root; `TRTQP_COCO`, `TRTQP_DATA`, `TRTQP_RESULTS`, and `RTDETR_ROOT` still override their respective locations. To import `harness` interactively, run Python from `host/` or set `PYTHONPATH=host` from the repository root.

From the repository root, run the host tests with `python -m pytest host/tests`. Tests that need models, COCO, TensorRT, or CUDA require those resources; the standalone deployment tests are separate under `deploy/tests/`.

## Directory migration

| Previous location | Current location |
| --- | --- |
| `harness/`, `models/`, `configs/`, `pipeline/`, `tests/` | The same directories under `host/`. |
| `requirements.txt` | `host/requirements.txt`; Jetson has separate requirements under `deploy/`. |
| `scripts/jetson/` or `jetson/scripts/jetson/` | `deploy/`; `infer_jetson.py` is now `infer.py`. |
| `jetson/model_r101*.onnx` | `artifacts/onnx/`. |
| `jetson/engines/` | `artifacts/engines/`; historical logs are in `results/jetson/builds/`. |
| `jetson/results/tables/jetson_r101.md` | `results/jetson/benchmark.md`. |
| `jetson/sample/` | `samples/jetson/` (local only). |
| Host clock scripts and `h1_boundary_pilot.py` | `host/scripts/`. |

Local notebooks and historical design notes moved under `host/`. Duplicate Jetson configuration/script snapshots were preserved locally under `artifacts/legacy-jetson/`. Existing custom shell scripts and notebooks may need their old paths updated using this table.
