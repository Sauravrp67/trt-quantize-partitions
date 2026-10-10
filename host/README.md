# Host model preparation

Use this workflow to export custom checkpoints, convert precision, calibrate INT8 models, or work on sensitivity and partitioning. To run a prepared ONNX model, use [Jetson deployment](../deploy/README.md).

## Environment

Run the commands below from the repository root unless a block explicitly changes directory.

```bash
git submodule update --init third_party/RT-DETR
```

Use a separate host environment with compatible PyTorch/torchvision. For FP32 ONNX export alone, install the [locked export dependencies](requirements-export.lock) with `python -m pip install -r host/requirements-export.lock` or `uv pip install -r host/requirements-export.lock`. The broader `requirements.txt` records host analysis tooling, including TensorRT 11; it is not a Jetson dependency list. Install that broader set only when you need it:

```bash
python -m pip install -r host/requirements.txt
```

ModelOpt quantization has a separate dependency file, `host/requirements-quantization.txt`, because its ONNX Runtime dependencies can conflict with the host evaluation environment. Install it in a separate environment with compatible PyTorch/torchvision. The host dependency sets are retained from development and have not been validated as a universal installer.

## Configure and export

Each model has a config in `host/configs/` selecting the upstream configuration, checkpoint, and precision variants. The [complete ONNX export guide](models/README.md) covers source setup, downloads, and commands for RT-DETR, RT-DETRv2, RT-DETRv3, and RT-DETRv4. The [variant catalog](models/variants.md) lists all 32 registered model configs and checkpoint requirements. The original R101 entry uses the Objects365-pretrained release.

```bash
python host/models/rtdetr/export.py --backbone r101vd
# List another family's variants, checkpoint downloads, and output paths:
python host/pipeline/export.py --spec rtdetrv2 --list
```

The shared entry point accepts any registered family, for example `python host/pipeline/export.py --spec rtdetrv2 --backbone r18vd`. v3 requires PaddlePaddle and `paddle2onnx`; v2 and v4 use PyTorch. Exports must satisfy the static `images` to `pred_logits`, `pred_boxes` ONNX contract before downstream host stages can use them.

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

The existing host inference and COCO evaluation CLIs accept the new specs too:

```bash
python host/models/rtdetr/infer.py --spec rtdetrv2 --backbone r18vd --source path/to/image.jpg --backends ort
python host/models/rtdetr/eval_map.py --spec rtdetrv4 --backbone s --backend ort --precision fp32
```

These commands require the selected ONNX export. RT-DETRv3 uses its Paddle checkpoint only for export, so select `ort` or `trt` for inference and evaluation.

| Directory | Responsibility |
| --- | --- |
| `models/rtdetr*/` | Family-specific checkpoint loading and export; see each model README. |
| `harness/` | Host adapters, runtime backends, precision and graph utilities. |
| `configs/` | Checkpoints, upstream configurations, and model variants. |
| [`pipeline/`](pipeline/README.md) | Shared ONNX preparation and analysis steps. |
| `scripts/` | Host experiments and clock controls. |
| `tests/` | Host tooling tests, including the TensorRT 11 contract guard. |

Datasets and results remain at the repository root. Pinned model sources live in [`third_party/`](../third_party/); initialize only the submodules needed for your chosen model. `TRTQP_ROOT` refers to the repository root; `TRTQP_COCO`, `TRTQP_DATA`, `TRTQP_RESULTS`, and the model-specific `RTDETR_ROOT`, `RTDETRV3_ROOT`, and `RTDETRV4_ROOT` variables override their respective locations. To import `harness` interactively, run Python from `host/` or set `PYTHONPATH=host` from the repository root.

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
