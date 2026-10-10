# RT-DETR TensorRT C++ inference and Jetson benchmarks

TensorRT C++ inference for developer-supplied RT-DETR ONNX models, with recorded precision benchmarks on **NVIDIA Jetson Orin Nano**. The runtime builds a TensorRT plan on the target machine and runs video inference through the TensorRT C++ API. ONNX export, quantization, and model training are separate preparation steps.

The recorded Jetson precision study used **RT-DETR R101-vd**, an Objects365-pretrained checkpoint, batch size 1, and a 640 × 640 input. For current deployment, use the [C++ runtime on `main`](https://github.com/Sauravrp67/trt-quantize-partitions/blob/main/cpp/README.md). The [host experiments](host/README.md) remain available for model preparation and analysis.

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

Source: [recorded demo runs](results/jetson/benchmark.md); see the [controlled precision benchmark](results/jetson/precision_benchmark_2026-09.md) for repeated FP32, FP16, mixed-BF16, and INT8 timings, plus the FP8 hardware-support check.

- **Inference latency** is wall-clock time around the TensorRT runner, including host/device transfers and runner overhead.
- **End-to-end latency** includes preprocessing, inference, postprocessing, and drawing/video writing when enabled. It excludes frame capture/decode.
- **Pipeline FPS** is processed frames divided by total loop time, including frame capture/decode.
- **Board power** is the mean `VDD_IN` reading from `tegrastats`. Energy per frame is mean board power divided by pipeline FPS.

The results table does not record the power mode, clock settings, or full run commands. These are recorded runs rather than a controlled accuracy benchmark. Mean detections per frame were 17.65 for FP32, 17.67 for FP16, and 15.48 for INT8; detection counts alone do not establish accuracy. Jetson mAP results are not included.

## Run C++ inference

Clone onto the target machine. The C++ runtime does not require the RT-DETR submodule, PyTorch, or ONNX Runtime. Install TensorRT C++ headers and libraries, CUDA, OpenCV with video I/O, CMake, and a C++17 compiler.

```bash
git clone https://github.com/Sauravrp67/trt-quantize-partitions.git
cd trt-quantize-partitions
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j
ctest --test-dir cpp/build --output-on-failure
```

Supply a prepared RT-DETR ONNX model. Select a matching tensor contract from `cpp/models/`, then build a plan on the machine that will run it:

```bash
./cpp/build_engine.sh /path/to/model_r101.fp16.onnx \
  artifacts/engines/model_r101.fp16.plan

cpp/build/rtdetr-infer \
  --model cpp/models/v1-r101.json \
  --engine artifacts/engines/model_r101.fp16.plan \
  --source /path/to/input.mp4 \
  --save artifacts/output/fp16.mp4 --score-thr 0.5 --power
```

Add `--show` to display video; `--save` alone writes an MP4 headlessly. Use `--source 0` for a camera. Engine plans are built on the target GPU and reused for inference. The runtime supports batch 1 and fixed image dimensions defined in the contract; an arbitrary ONNX export is not automatically compatible.

The supplied contracts cover these export interfaces:

| Contract | Output format |
| --- | --- |
| `cpp/models/v1-r101.json` | Raw query logits and boxes |
| `cpp/models/v2-r50.json` | Decoded labels, boxes, and scores |
| `cpp/models/v3-r18.json` | Paddle detection rows |
| `cpp/models/v4-s.json` | Decoded labels, boxes, and scores |

Other backbones or precision variants can reuse a contract when tensor names, shapes, preprocessing, and output semantics match. Internal FP32, FP16, INT8, or other precision is supplied by the ONNX graph and must be supported by the target TensorRT installation and GPU. The runtime does not convert precision.

See the [C++ instructions](https://github.com/Sauravrp67/trt-quantize-partitions/blob/main/cpp/README.md) for TensorRT shape profiles, model contracts, runtime options, and a same-plan Python timing comparison. The historical Jetson table above came from the previous Python deployment workflow; the [C++ validation record](https://github.com/Sauravrp67/trt-quantize-partitions/blob/main/cpp/results/validation.md) reports separate same-plan comparisons.

## Export and quantize on a host

Model preparation runs on the host. Set up PyTorch, torchvision, and the [host dependencies](host/README.md#environment), initialize the `third_party/RT-DETR` submodule, and place the matching R101-vd checkpoint in `host/models/rtdetr/checkpoints/`. The model paths and selected checkpoint are defined in `host/configs/rtdetr.yaml`.

Export the FP32 ONNX model, then convert it to FP16:

```bash
git submodule update --init third_party/RT-DETR
python host/models/rtdetr/export.py --backbone r101vd
PYTHONPATH=host python -c "from harness.precision import to_fp16; to_fp16('host/models/rtdetr/model_r101.onnx', 'host/models/rtdetr/model_r101.fp16.onnx')"
```

For INT8, use a separate host environment with PyTorch, torchvision, and [ModelOpt dependencies](host/requirements-quantization.txt). Download COCO val2017 for calibration, then quantize the **FP32** export with 200 images:

```bash
python -m pip install -r host/requirements-quantization.txt
bash scripts/download_coco_val.sh
python host/pipeline/quantize.py --backbone r101vd --calib 200
```

The three outputs are `host/models/rtdetr/model_r101.onnx`, `model_r101.fp16.onnx`, and `model_r101.int8.onnx`. The INT8 script calibrates with COCO images and writes quantize/dequantize nodes into the ONNX graph. FP16 types are also encoded in its ONNX graph. Copy the desired ONNX files to the target machine, then build their engines there with `cpp/build_engine.sh`; each engine keeps the precision specified by its graph.

See [host preparation](host/README.md) for its environment and export workflow. The host experiments are separate from the C++ deployment runtime.

## Repository layout

| Path | Contents |
| --- | --- |
| [cpp/](https://github.com/Sauravrp67/trt-quantize-partitions/tree/main/cpp) | TensorRT C++ library and CLI, ONNX-to-plan builder, model contracts, native tests, and Python timing comparison. |
| [host/](host/) | Model export, quantization, host harness, configurations, experiments, and tests. |
| `artifacts/` | Local deployment ONNX models, engines, recordings, and evaluation outputs; excluded from Git. |
| `data/` | Local datasets; excluded from Git. |
| [results/jetson/](results/jetson/) | Recorded Jetson measurements and engine build logs. |
| [samples/](samples/) | Sample recording information; large local videos are excluded from Git. |
| [scripts/](scripts/) | Dataset download helpers for host experiments. |
| [third_party/](third_party/) | Pinned upstream model source, needed only for host preparation. |

The measured Jetson workflow covers FP32, FP16, and INT8 deployment. Sensitivity-guided partitioning remains under development and is not part of the results above.
