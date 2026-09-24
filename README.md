# RT-DETR on Jetson Orin Nano

TensorRT deployment and precision benchmarking for RT-DETR on **NVIDIA Jetson Orin Nano**. The project compares FP32, FP16, and INT8 inference on UAV video, measuring latency, pipeline throughput, board power, and energy per frame.

The measured model is **RT-DETR R101-vd**, using the Objects365-pretrained checkpoint, with batch size 1 and a 640 × 640 input. Choose the [Jetson deployment workflow](deploy/README.md) to run prepared models, or the [host preparation workflow](host/README.md) to export and quantize custom weights.

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

Source: [recorded results](results/jetson/benchmark.md). All recorded runs are shown; run numbers follow their order within each precision variant.

- **Inference latency** is wall-clock time around the TensorRT runner, including host/device transfers and runner overhead.
- **End-to-end latency** includes preprocessing, inference, postprocessing, and drawing/video writing when enabled. It excludes frame capture/decode.
- **Pipeline FPS** is processed frames divided by total loop time, including frame capture/decode.
- **Board power** is the mean `VDD_IN` reading from `tegrastats`. Energy per frame is mean board power divided by pipeline FPS.

The results table does not record the power mode, clock settings, or full run commands. These are recorded runs rather than a controlled accuracy benchmark. Mean detections per frame were 17.65 for FP32, 17.67 for FP16, and 15.48 for INT8; detection counts alone do not establish accuracy. Jetson mAP results are not included.

## Run on Jetson

Clone directly onto the board. The deployment workflow does not require the RT-DETR submodule, PyTorch, torchvision, or ONNX Runtime.

```bash
git clone https://github.com/Sauravrp67/trt-quantize-partitions.git
cd trt-quantize-partitions
bash deploy/setup.sh
source .venv-jetson/bin/activate
```

Supply a prepared RT-DETR ONNX model. Model release downloads are not published yet; use an existing export or prepare one with the host tools.

```bash
python deploy/import_model.py \
  --source /path/to/model_r101.fp16.onnx \
  --out artifacts/onnx/model_r101.fp16.onnx

bash deploy/build_engines.sh -o artifacts/engines \
  artifacts/onnx/model_r101.fp16.onnx

python deploy/infer.py \
  --engine artifacts/engines/model_r101.fp16.plan \
  --source /path/to/input.mp4 \
  --save artifacts/output/fp16.mp4 \
  --score-thr 0.5 --power
```

Use `--source 0 --show` for a camera or a stream URL for network video. Engine plans are built on the target board and reused for inference. The current runtime supports static RT-DETR exports with batch 1 and a 640 × 640 input.

See [deployment instructions](deploy/README.md) for dependencies, model import, saved-engine COCO evaluation, power measurement, and copying only the runtime folder to a board.

## Export and quantize on a host

Model preparation runs on the host. Set up PyTorch, torchvision, and the [host dependencies](host/README.md#environment), initialize the RT-DETR submodule, and place the matching R101-vd checkpoint in `host/models/rtdetr/checkpoints/`. The model paths and selected checkpoint are defined in `host/configs/rtdetr.yaml`.

Export the FP32 ONNX model, then convert it to FP16:

```bash
git submodule update --init --recursive
python host/models/rtdetr/export.py --backbone r101vd
PYTHONPATH=host python -c "from harness.precision import to_fp16; to_fp16('host/models/rtdetr/model_r101.onnx', 'host/models/rtdetr/model_r101.fp16.onnx')"
```

For INT8, use a separate host environment with PyTorch, torchvision, and [ModelOpt dependencies](host/requirements-quantization.txt). Download COCO val2017 for calibration, then quantize the **FP32** export with 200 images:

```bash
python -m pip install -r host/requirements-quantization.txt
bash scripts/download_coco_val.sh
python host/pipeline/quantize.py --backbone r101vd --calib 200
```

The three outputs are `host/models/rtdetr/model_r101.onnx`, `model_r101.fp16.onnx`, and `model_r101.int8.onnx`. The INT8 script calibrates with COCO images and writes quantize/dequantize nodes into the ONNX graph. FP16 types are also encoded in its ONNX graph. Copy the desired ONNX files to `artifacts/onnx/` on the Jetson, then build their engines there with `deploy/build_engines.sh`; each engine keeps the precision specified by its graph. The Jetson runtime only builds and executes these prepared models.

See [host preparation](host/README.md) for the full environment and transfer workflow. To compare deployed accuracy, evaluate the saved Jetson engines with [COCO evaluation](deploy/README.md#evaluate-the-deployed-engines).

## Repository layout

| Path | Contents |
| --- | --- |
| [deploy/](deploy/) | Self-contained Jetson setup, model import, engine build, inference, and COCO evaluation. |
| [host/](host/) | Model export, quantization, host harness, configurations, experiments, and tests. |
| `artifacts/` | Local deployment ONNX models, engines, recordings, and evaluation outputs; excluded from Git. |
| `data/` | Local datasets; excluded from Git. |
| [results/jetson/](results/jetson/) | Recorded Jetson measurements and engine build logs. |
| [samples/](samples/) | Sample recording information; large local videos are excluded from Git. |
| [scripts/](scripts/) | Dataset download helpers. |
| [RT-DETR/](RT-DETR/) | Upstream submodule, needed only for host preparation. |

The measured Jetson workflow covers FP32, FP16, and INT8 deployment. Sensitivity-guided partitioning remains under development and is not part of the results above.
