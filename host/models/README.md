# Export RT-DETR models to ONNX

Run these commands on a host machine from the repository root. The export scripts write FP32 ONNX files and adjacent `.export.json` manifests. They do not build TensorRT engines; build those on the Jetson from the ONNX files.

All 32 upstream RT-DETR model configs are registered. See the [variant catalog](variants.md) for every config, checkpoint filename, and download reference, including HGNet, sampling variants, VOC, and LVIS. List a family's paths without loading weights:

```bash
python host/pipeline/export.py --spec rtdetrv2 --list
```

## 1. Get the source and set up Python

If you already cloned the repository, initialize the pinned source submodules:

```bash
git submodule update --init --recursive
```

RT-DETR and RT-DETRv2 share `third_party/RT-DETR`; v3 and v4 have their own source trees under `third_party/`. The source trees contain configs and model code, not the trained checkpoints.

For RT-DETR, v2, and v4, use a Python 3.11 environment. [`host/requirements-export.lock`](../requirements-export.lock) pins direct and transitive dependencies for Linux x86-64. Install it with pip or uv:

```bash
python3.11 -m venv .venv-export
source .venv-export/bin/activate
python -m pip install -r host/requirements-export.lock
# Or, with uv installed: uv pip install -r host/requirements-export.lock
```

[`host/requirements-export.txt`](../requirements-export.txt) records the direct version choices; regenerate the lockfile after changing them with `uv pip compile host/requirements-export.txt --python-version 3.11 --python-platform x86_64-unknown-linux-gnu -o host/requirements-export.lock`. The installed package versions were exercised locally with Python 3.13; the Python 3.11 lock resolves but has not been runtime-tested here. It uses the default PyTorch wheels. If you need a different CUDA wheel variant, follow the [PyTorch installer](https://pytorch.org/get-started/locally/) and resolve a separate lock for that environment. The broader `host/requirements.txt` includes TensorRT and other analysis tools; it is not needed solely to export FP32 ONNX. RT-DETRv3 uses PaddlePaddle and needs a separate environment described below.

## 2. RT-DETR (v1)

The default `r101vd` spec uses the COCO checkpoint initialized from Objects365, not the plain COCO-only R101 checkpoint. Download it using the exact configured filename:

```bash
mkdir -p host/models/rtdetr/checkpoints
curl -fL https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_2x_coco_objects365_from_paddle.pth \
  -o host/models/rtdetr/checkpoints/rtdetr_r101vd_2x_coco_objects365_from_paddle.pth
python host/pipeline/export.py --spec rtdetr --backbone r101vd --no-report
```

This writes `host/models/rtdetr/model_r101.onnx`. Other configured backbones use the same export command with a different `--backbone`; save the corresponding [official checkpoint](https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetr_pytorch/README.md) under `host/models/rtdetr/checkpoints/` with the filename below.

| `--backbone` | Official download | Checkpoint filename | FP32 output |
| --- | --- | --- | --- |
| `r18vd` | [R18](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth) | `rtdetr_r18vd_dec3_6x_coco_from_paddle.pth` | `model.onnx` |
| `r34vd` | [R34](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r34vd_dec4_6x_coco_from_paddle.pth) | `rtdetr_r34vd_dec4_6x_coco_from_paddle.pth` | `rtdetr_r34vd.onnx` |
| `r50vd` | [R50](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth) | `rtdetr_r50vd_6x_coco_from_paddle.pth` | `rtdetr_r50vd.onnx` |
| `r50vd_m` | [R50-M](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_m_6x_coco_from_paddle.pth) | `rtdetr_r50vd_m_6x_coco_from_paddle.pth` | `rtdetr_r50vd_m.onnx` |
| `r101vd` | [R101 Objects365](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_2x_coco_objects365_from_paddle.pth) | `rtdetr_r101vd_2x_coco_objects365_from_paddle.pth` | `model_r101.onnx` |

## 3. RT-DETRv2

The default variant is `r18vd`:

```bash
mkdir -p host/models/rtdetrv2/checkpoints
curl -fL https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth \
  -o host/models/rtdetrv2/checkpoints/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth
python host/pipeline/export.py --spec rtdetrv2 --backbone r18vd
```

This writes `host/models/rtdetrv2/rtdetrv2_r18vd.onnx`. For `r101vd`, download the [official COCO checkpoint](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r101vd_6x_coco_from_paddle.pth) as `host/models/rtdetrv2/checkpoints/rtdetrv2_r101vd_6x_coco_from_paddle.pth`, then run:

```bash
python host/pipeline/export.py --spec rtdetrv2 --backbone r101vd
```

Its output is `host/models/rtdetrv2/rtdetrv2_r101vd.onnx`. Do not substitute the separate Objects365-pretrained v2 R101 checkpoint without changing the spec.

R34, R50, R50-M, HGNetv2 L/X/H, discrete-sampling (`*_dsp`), and R18 sampling-point (`*_sp1`/`*_sp2`/`*_sp3`) variants use the same command with their catalog entry name. The VOC entry needs your own trained checkpoint and uses 20 classes.

## 4. RT-DETRv3 (PaddlePaddle)

Use Python 3.10 in a separate environment. Run `deactivate` first if `.venv-export` is active. Upstream documents the older PaddlePaddle and Paddle2ONNX toolchain used by this export route. The commands below have not yet been validated end to end in this repository.

```bash
python3.10 -m venv .venv-export-v3
source .venv-export-v3/bin/activate
python -m pip install --upgrade pip
SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True \
  python -m pip install -r third_party/RT-DETRv3/requirements.txt
python -m pip install 'paddlepaddle==2.4.2' -i https://mirror.baidu.com/pypi/simple
python -m pip install 'paddle2onnx==1.0.5' 'onnx==1.13.0'
```

PaddlePaddle 2.4.2 is absent from the default PyPI index; this mirror command comes from [Paddle's 2.4 installation guide](https://www.paddlepaddle.org.cn/documentation/docs/en/2.4/install/index_en.html). That guide also lists `https://pypi.tuna.tsinghua.edu.cn/simple` as an alternative index.

Download the official checkpoint from the [v3 model zoo](https://github.com/clxia12/RT-DETRv3#model-zoo-on-coco). The upstream Google Drive links are [R18](https://drive.google.com/file/d/1zIDOjn1qDccC3TBsDlGQHOjVrehd26bk/view?usp=drive_link) and [R50](https://drive.google.com/file/d/1wfJE-QgdgqKE0IkiTuoD5HEbZwwZg3sQ/view?usp=drive_link). Save the downloaded `.pdparams` file at the configured location, using the exact name:

```bash
mkdir -p host/models/rtdetrv3/checkpoints
```

| `--backbone` | Checkpoint destination | FP32 output |
| --- | --- | --- |
| `r18vd` | `host/models/rtdetrv3/checkpoints/rtdetrv3_r18vd_6x_coco.pdparams` | `host/models/rtdetrv3/rtdetrv3_r18vd.onnx` |
| `r50vd` | `host/models/rtdetrv3/checkpoints/rtdetrv3_r50vd_6x_coco.pdparams` | `host/models/rtdetrv3/rtdetrv3_r50vd.onnx` |

R34 is also registered as `r34vd`, with its official download in the [catalog](variants.md). The `r18vd_lvis` and `r50vd_lvis` entries require your own LVIS-trained checkpoints and export 1,203-class logits.

Then run, selecting the variant you downloaded:

```bash
python host/pipeline/export.py --spec rtdetrv3 --backbone r18vd
# or: python host/pipeline/export.py --spec rtdetrv3 --backbone r50vd
```

The script runs the upstream Paddle inference export, converts it with `paddle2onnx`, and validates the raw detector outputs. It requires CPU PaddlePaddle for export. If the older Paddle packages do not provide wheels for your platform, use an upstream-compatible Linux/Python 3.10 environment; do not install them into the PyTorch export environment.

## 5. RT-DETRv4

Reactivate `.venv-export` for v4. The shared export requirements above already include its upstream dependencies. Run `deactivate` first if the v3 environment is active:

```bash
source .venv-export/bin/activate
```

Download an [official v4 checkpoint](https://github.com/RT-DETRs/RT-DETRv4#-performance) from Google Drive. Save it under `host/models/rtdetrv4/checkpoints/` with the filename required by the selected spec:

```bash
mkdir -p host/models/rtdetrv4/checkpoints
```

| `--backbone` | Official download | Checkpoint filename | FP32 output |
| --- | --- | --- | --- |
| `s` | [S checkpoint](https://drive.google.com/file/d/1jDAVxblqRPEWed7Hxm6GwcEl7zn72U6z) | `rtv4_hgnetv2_s_coco.pth` | `rtdetrv4_s.onnx` |
| `m` | [M checkpoint](https://drive.google.com/file/d/1O-YpP4X-quuOXbi96y2TKkztbjroP5mX) | `rtv4_hgnetv2_m_coco.pth` | `rtdetrv4_m.onnx` |
| `l` | [L checkpoint](https://drive.google.com/file/d/1shO9EzZvXZyKedE2urLsN4dwEv8Jqa_8) | `rtv4_hgnetv2_l_coco.pth` | `rtdetrv4_l.onnx` |
| `x` | [X checkpoint](https://drive.google.com/file/d/19gnkMTgFveJsrOvSmEPQXCTG6v9oQHN3) | `rtv4_hgnetv2_x_coco.pth` | `rtdetrv4_x.onnx` |

For example, after saving S as `host/models/rtdetrv4/checkpoints/rtv4_hgnetv2_s_coco.pth`:

```bash
python host/pipeline/export.py --spec rtdetrv4 --backbone s
```

The output is `host/models/rtdetrv4/rtdetrv4_s.onnx`. The S export and ONNX Runtime execution have been checked with synthetic weights; official checkpoint accuracy has not been measured here.

## Check the output and prepare other precisions

The registered exports target a static FP32 ONNX contract: `images [1,3,640,640]` to `pred_logits [1,300,C]` and `pred_boxes [1,300,4]`, where `C` is 80 for COCO, 20 for VOC, or 1,203 for LVIS. The shared exporter validates tensor names, FP32 types, and static batch-1 shapes, then writes a `.export.json` manifest next to the ONNX. The paths for every variant are defined in [`host/configs/`](../configs/). Host mAP evaluation currently supports COCO80 only; VOC/LVIS entries need their respective upstream evaluators.

To create FP16 from one exported model, use the registered FP32 and FP16 paths:

```bash
PYTHONPATH=host python - <<'PY'
from harness.config import load_spec
from harness.precision import to_fp16
spec = load_spec('rtdetrv2', 'r18vd')  # change family and backbone
to_fp16(str(spec.onnx_path('fp32')), str(spec.onnx_dest('fp16')))
PY
```

INT8 quantization uses a separate ModelOpt environment and COCO calibration images; see the [host pipeline](../pipeline/README.md). Copy the desired ONNX file to the Jetson and build its TensorRT engine there with [`deploy/build_engines.sh`](../../deploy/README.md#build-on-the-board). The engine is hardware-specific and is not created by these export commands.
