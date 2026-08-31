# RT-DETR TensorRT C++ inference

This directory builds a TensorRT plan from a **developer-supplied ONNX file** and
runs that plan through the **TensorRT C++ API**. It does not export models,
download checkpoints, train, calibrate, or quantize. ONNX export and any
precision conversion happen outside the C++ runtime.

The C++ code creates an `nvinfer1::IRuntime`, deserializes an
`nvinfer1::ICudaEngine`, creates an `nvinfer1::IExecutionContext`, and calls
`enqueueV3` on a CUDA stream. It handles video input, preprocessing, RT-DETR
output decoding, annotations, and timing. Python is optional and used only
for a same-engine FPS/E2E comparison.

Run the commands below from this `cpp/` directory (`cd cpp` from the repository
root).

## Build the C++ inference binary

Requires CMake, C++17, CUDA, TensorRT C++ headers/libraries, and OpenCV with
video I/O.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DTENSORRT_ROOT=/path/to/TensorRT
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Omit `TENSORRT_ROOT` when your system packages provide TensorRT. The reusable
library is `rtdetr_runtime`; `Detector::infer` accepts a BGR `cv::Mat` and
returns class IDs, scores, and pixel-space boxes.

## Build a plan from ONNX

Create or obtain a compatible RT-DETR ONNX graph elsewhere. Build its plan on
the target GPU with the target TensorRT installation:

```bash
./build_engine.sh /path/to/model.onnx artifacts/model.plan
```

For dynamic inputs, pass TensorRT shape profiles through to `trtexec`. For a
decoded-output model with the usual 640 × 640 image and original-size input:

```bash
./build_engine.sh /path/to/model.onnx artifacts/model.plan \
  --minShapes=images:1x3x640x640,orig_target_sizes:1x2 \
  --optShapes=images:1x3x640x640,orig_target_sizes:1x2 \
  --maxShapes=images:1x3x640x640,orig_target_sizes:1x2
```

The script preserves the ONNX graph's tensor types. TensorRT may use TF32
compute for FP32 layers by default; add `--noTF32` to disable it. FP16,
INT8, and INT4 require an ONNX graph already prepared for those modes and
support from the installed TensorRT version, GPU, and operators. The C++ runtime
does not perform a precision conversion. An INT8 or INT4 graph can still have
FP32 or FP16 image inputs and outputs. The current C++ preprocessor accepts
FP32 or FP16 image I/O; other image I/O types need explicit quantization
parameters and code changes. Keep ONNX external data beside its graph when
building. Plans are tied to the GPU and TensorRT installation.

Set `TRTEXEC=/path/to/trtexec` if it is not on `PATH`. Set
`TRT_WORKSPACE_MIB=4096` to change the builder workspace from its 2048 MiB
default.

## Run C++ inference

The ONNX graph does not describe how to resize a video frame or decode model
outputs. Supply a small runtime contract JSON whose tensor names, dimensions,
class count, resize mode, and output format match the graph. Four examples are
in [`models/`](models/): `v1-r101.json` (raw queries), `v2-r50.json` and
`v4-s.json` (decoded outputs), and `v3-r18.json` (Paddle rows). Copy and edit
one for another model or export. The runtime supports batch 1 and fixed image
height/width from the contract.

```bash
build/rtdetr-infer \
  --model models/v1-r101.json \
  --engine artifacts/model.plan \
  --source /path/to/video.mp4 \
  --show --save artifacts/annotated.mp4
```

`--show` displays the video; `--save` writes an annotated MP4. Without either,
the CLI runs headlessly and prints timings. `--source 0` selects a camera.
For COCO contracts, class names load from `models/coco80.yaml` automatically;
use `--classes FILE` for another label set. `--no-draw` excludes boxes and HUD
for model-path benchmarks. `--max-frames N`, `--warmup N`, `--score-thr X`,
`--table FILE`, and `--profile FILE` control measurement. Engine time includes
host/device transfers and stream synchronization. Processing E2E starts after
frame read; pipeline FPS includes frame read.

## Compare with Python TensorRT inference

The only Python script under `scripts/` runs the **same TensorRT plan** for
FPS and E2E comparison. It requires Python TensorRT, Polygraphy, NumPy,
OpenCV, and Pillow. Use the same source, contract, warmup, frame count, and
threshold for both commands:

```bash
build/rtdetr-infer --model models/v1-r101.json --engine artifacts/model.plan \
  --source /path/to/video.mp4 --no-draw --warmup 10 --max-frames 30

python scripts/benchmark_engine.py --model models/v1-r101.json \
  --engine artifacts/model.plan --source /path/to/video.mp4 \
  --warmup 10 --max-frames 30
```

Both commands print engine p50/p90 latency, processing E2E p50 latency, and
pipeline FPS. The Python numbers include its own TensorRT runner overhead.
The [validation record](results/validation.md) preserves measured results
from the original reference engines; it is historical evidence, not a model
export or quantization workflow.
