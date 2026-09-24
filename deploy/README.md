# Jetson deployment

Run prepared RT-DETR ONNX models on the Jetson Orin Nano. This folder is self-contained: it can be used inside a clone or copied to the board on its own. It does not import the host harness, PyTorch, torchvision, ONNX Runtime, ModelOpt, or the upstream RT-DETR repository.

## Setup

The board must already provide TensorRT Python bindings, `trtexec`, OpenCV, and Python's `venv` module. The recorded runs used TensorRT 10.16.2. Setup checks the system bindings before creating an environment; it does not replace the board's CUDA or TensorRT installation.

From the repository root:

```bash
bash deploy/setup.sh
source .venv-jetson/bin/activate
```

Use `bash deploy/setup.sh --eval` to include `pycocotools`. Override `JETSON_PYTHON`, `JETSON_VENV`, or `TRTEXEC` when those tools live elsewhere. If `venv` is missing, install the board's `python3-venv` package first.

## Obtain a model

Prebuilt ONNX release downloads have not been published. Supply an existing export, or follow the [host preparation workflow](../host/README.md). FP32, FP16, and INT8 are separate ONNX files; changing a filename does not change precision.

```bash
python deploy/import_model.py \
  --source /path/to/model_r101.fp16.onnx \
  --out artifacts/onnx/model_r101.fp16.onnx
```

The import tool also accepts an HTTPS source with a required `--sha256` checksum. It refuses to overwrite an existing model. For exports that use ONNX external data, copy the ONNX file and all referenced data files together, retaining their relative paths; the import tool transfers only the supplied file.

The current runner supports RT-DETR exports with input `images [1,3,640,640]` and outputs `pred_logits [1,300,80]`, `pred_boxes [1,300,4]`. Prepared models must use that contract. Other detector families require their own preprocessing and decoding implementation; they are not currently supported.

## Build on the board

```bash
bash deploy/build_engines.sh -o artifacts/engines \
  artifacts/onnx/model_r101.fp16.onnx
```

Pass multiple ONNX paths to build several variants. The script saves `<model>.plan`, build logs, and a timing cache. It uses `--stronglyTyped`, a 2048 MiB workspace limit, optimization level 3, and detailed profiling. Override workspace with `-w 1024`. Arguments after `--` go to `trtexec`.

Transfer ONNX models between machines and build plans on the target board. Keep timing caches local to their GPU and TensorRT environment. Build again after changing the model or deployment environment; inference loads the saved plan and never rebuilds it.

## Video, camera, and streams

```bash
python deploy/infer.py \
  --engine artifacts/engines/model_r101.fp16.plan \
  --source /path/to/input.mp4 \
  --save artifacts/output/fp16.mp4 \
  --table artifacts/output/benchmark.md \
  --score-thr 0.5 --warmup 10 --power
```

Use `--source 0 --show` for a camera with local display, or `--source 'rtsp://host/path'` for a stream. Stream and camera support depend on the board's OpenCV video backends. Press `q` to stop a displayed run; use `--max-frames` to bound a headless stream run. Omit `--show` for headless operation.

| Option | Behavior |
| --- | --- |
| `--save FILE` | Save video at the source frame rate. |
| `--power` | Sample `tegrastats` power rails after warmup. |
| `--no-draw` | Disable boxes and the statistics overlay. |
| `--no-hud` | Draw boxes without the statistics overlay. |
| `--classes FILE` | Override the bundled COCO class names. |
| `--max-frames N` | Stop after N frames. |

Preprocessing, decoding, and TensorRT execution live in `runtime.py` and are shared with evaluation. Visualization and power sampling live in `infer.py`.

## Evaluate the deployed engines

COCO validation requires `val2017/` and `annotations/instances_val2017.json`. In a full clone, `bash scripts/download_coco_val.sh` downloads them under `data/coco/`. You can also copy an existing dataset to the board.

```bash
bash deploy/setup.sh --eval
source .venv-jetson/bin/activate
python deploy/eval_map.py \
  --engines artifacts/engines/model_r101.plan \
            artifacts/engines/model_r101.fp16.plan \
            artifacts/engines/model_r101.int8.plan \
  --coco data/coco \
  --out artifacts/evaluation/coco_map.json
```

The evaluator loads each existing plan, runs the same images, and writes JSON plus a Markdown table. The first engine is the delta reference. It uses all top-300 decoded predictions without the demo confidence threshold. `--limit 100` is useful for a smoke test; omit it for the full validation set.

## Copy only the runtime from a host

The same commands work without cloning on the Jetson:

```bash
ssh jetson 'mkdir -p ~/rtdetr/artifacts/onnx'
scp -r deploy jetson:~/rtdetr/
scp /path/to/model_r101.fp16.onnx jetson:~/rtdetr/artifacts/onnx/
```

On the board, `cd ~/rtdetr`, run setup, build the engine, and run inference using the commands above. Copy any ONNX external data files as well. COCO and video inputs can be stored separately and passed by path.
