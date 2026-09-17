# RT-DETR on Jetson Orin Nano

TensorRT deployment and precision benchmarking for RT-DETR on **NVIDIA Jetson Orin Nano**. The project compares FP32, FP16, and INT8 inference on UAV video, measuring latency, pipeline throughput, board power, and energy per frame.

The measured model is **RT-DETR R101-vd**, using the Objects365-pretrained checkpoint, with batch size 1 and a 640 × 640 input. The repository includes ONNX export and precision tooling, a standalone Jetson inference runner, engine build scripts, and recorded results.

## Demo

Annotated recordings of the UAV sequence. FP32 and INT8 show all 468 frames; the FP16 sample is a 174-frame excerpt.

**FP32**

https://github.com/user-attachments/assets/0a343da5-3e62-446c-9345-5e2c73258c4e

**FP16**

https://github.com/user-attachments/assets/cf8699f0-ccb5-40eb-bb9c-a7b373d4a8f1

**INT8**

https://github.com/user-attachments/assets/0c26c084-7622-4f27-8fcc-29ad178c9006

The recordings show detections and runtime statistics. Video playback uses the source frame rate; measured inference throughput is reported below.

## Results

Jetson Orin Nano · TensorRT 10.16.2 · RT-DETR R101-vd · batch 1 · 640 × 640 input · 468 frames per run.

| Precision | Run | Inference p50 (ms) | Inference p90 (ms) | End-to-end p50 (ms) | Pipeline FPS | Mean board power (W) | Energy (J/frame) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 | 1 | 79.86 | 83.34 | 129.46 | 7.48 | 15.29 | 2.044 |
| FP32 | 2 | 79.85 | 87.16 | 136.81 | 7.06 | 14.79 | 2.095 |
| FP16 | 1 | 42.83 | 44.98 | 101.10 | 9.54 | 10.34 | 1.084 |
| INT8 | 1 | 31.58 | 35.75 | 88.42 | 10.83 | 8.33 | 0.769 |
| INT8 | 2 | 31.85 | 35.71 | 89.34 | 10.76 | 8.29 | 0.770 |

Source: [recorded results](jetson/results/tables/jetson_r101.md). All recorded runs are shown; run numbers follow their order within each precision variant.

- **Inference latency** is wall-clock time around the TensorRT runner, including host/device transfers and runner overhead.
- **End-to-end latency** includes preprocessing, inference, postprocessing, and drawing/video writing when enabled. It excludes frame capture/decode.
- **Pipeline FPS** is processed frames divided by total loop time, including frame capture/decode.
- **Board power** is the mean `VDD_IN` reading from `tegrastats`. Energy per frame is mean board power divided by pipeline FPS.

The results table does not record the power mode, clock settings, or full run commands. These are recorded runs rather than a controlled accuracy benchmark. Mean detections per frame were 17.65 for FP32, 17.67 for FP16, and 15.48 for INT8; detection counts alone do not establish accuracy. Jetson mAP results are not included.

## Requirements

For inference on the Jetson:

- TensorRT with Python bindings and `trtexec`; the supplied build logs report **10.16.2**.
- Python with NumPy, OpenCV, Pillow, Polygraphy, and PyYAML.
- `tegrastats` for power measurement.
- Exported RT-DETR ONNX models for engine building, or compatible TensorRT plans built on the target board.

The standalone Jetson runner does not require PyTorch or the RT-DETR submodule. Use a Python environment that can import the board's TensorRT bindings:

```bash
python3 -m venv --system-site-packages .venv-jetson
source .venv-jetson/bin/activate
python -m pip install numpy pillow pyyaml polygraphy
python -c "import tensorrt, cv2, PIL, numpy, polygraphy, yaml; print(tensorrt.__version__)"
```

This assumes OpenCV and TensorRT are already installed on the board. The root `requirements.txt` describes the model preparation environment and pins a different TensorRT version; use the dependencies above for Jetson inference.

## Build engines

Run commands from the repository root. Place the exported models at:

```text
jetson/model_r101.onnx
jetson/model_r101.fp16.onnx
jetson/model_r101.int8.onnx
```

ONNX models, checkpoints, and compiled engines are excluded from Git. Supply the exported models separately. Build the engines on the target Jetson:

```bash
bash jetson/scripts/jetson/build_engines.sh -o jetson/engines \
  jetson/model_r101.onnx \
  jetson/model_r101.fp16.onnx \
  jetson/model_r101.int8.onnx
```

The script writes a `.plan` and build log for each model, plus a shared timing cache. It uses strongly typed networks, a 2048 MiB workspace limit, and builder optimization level 3. Precision is specified in the ONNX graph: FP16 tensor types or INT8 quantize/dequantize nodes. Use `-w` to change the workspace limit or set `TRTEXEC` to override `/usr/src/tensorrt/bin/trtexec`.

## Run inference

Provide an unannotated video as `input.mp4`. This example saves an FP16 recording and appends measurements to a new results table:

```bash
python jetson/scripts/jetson/infer_jetson.py \
  --engine jetson/engines/model_r101.fp16.plan \
  --source input.mp4 \
  --classes jetson/configs/classes/coco80.yaml \
  --score-thr 0.5 \
  --warmup 10 \
  --power \
  --save jetson/output/fp16.mp4 \
  --table jetson/output/benchmark.md
```

Use `model_r101.plan` for FP32 or `model_r101.int8.plan` for INT8. Keep the source, confidence threshold, warmup, drawing, and recording settings consistent across comparisons.

| Option | Purpose |
| --- | --- |
| `--source 0` | Read from a camera. |
| `--max-frames 300` | Limit the number of processed frames. |
| `--no-draw` | Disable detection boxes and the statistics overlay. |
| `--no-hud` | Keep detection boxes but hide the statistics overlay. |
| `--power` | Sample power rails through `tegrastats`. |
| `--save <path>` | Write an annotated MP4. Omit to disable video writing. |
| `--table <path>` | Append a Markdown results row. |

Preprocessing uses RGB, PIL bilinear resizing to 640 × 640, and scaling to `[0, 1]`. Postprocessing selects the top 300 class/query scores, applies the confidence threshold, and maps boxes to the original frame size.

For repeatable measurements, record the active `nvpmodel` mode and clock settings, and keep them fixed across runs. Power sampling starts after warmup. Disabling drawing and video writing changes pipeline throughput; record those options alongside the results.

## Repository layout

| Path | Contents |
| --- | --- |
| [jetson/scripts/jetson/](jetson/scripts/jetson/) | Standalone inference runner and engine build script. |
| [jetson/sample/](jetson/sample/) | FP32, FP16, and INT8 demonstration videos. |
| [jetson/results/tables/](jetson/results/tables/) | Recorded Jetson latency, throughput, and power measurements. |
| [jetson/engines/](jetson/engines/) | Recorded build logs; generated plans and timing caches stay local. |
| [jetson/configs/](jetson/configs/) | Model configuration snapshot and COCO class names. |
| [models/rtdetr/](models/rtdetr/) | Model export, inference, and COCO evaluation entry points. |
| [harness/](harness/) | Model adapters, precision conversion, evaluation, and runtime utilities. |
| [pipeline/](pipeline/) | Model preparation and precision-partitioning stages under development. |
| [tests/](tests/) | Tests for the shared tooling. |

The measured Jetson workflow covers FP32, FP16, and INT8 deployment. Sensitivity-guided partitioning remains under development and is not part of the results above.

RT-DETR source is included through the [RT-DETR](RT-DETR/) submodule. Initialize it with `git submodule update --init --recursive` when working on model export or evaluation.
