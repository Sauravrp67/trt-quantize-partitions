# Registered model variants

Every top-level detector config in the pinned RT-DETR family directories is registered. Dataset, runtime, include/base configs, and the separate D-FINE/DEIM models bundled with v4 are not RT-DETR variants.

Use `--backbone` to select a registry entry. This includes dataset and sampling variants, not just backbone architectures. Existing default models and ONNX paths are preserved.

```bash
python host/pipeline/export.py --spec rtdetrv2 --list
python host/pipeline/export.py --spec rtdetrv2 --backbone hgnetv2_l
```

Place the matching checkpoint in `host/models/<family>/checkpoints/` using the destination filename below. Exports go to `host/models/<family>/`; the registry and `--list` give the exact ONNX paths. A download link is an upstream checkpoint reference, not a claim that its export or accuracy was tested here. Entries marked **user supplied** need a checkpoint trained with that exact upstream model config.

For environment setup, see the [export guide](README.md).

| Version | Registry entries | Upstream model directory |
| --- | ---: | --- |
| v1 (`rtdetr`) | 7 | [rtdetr](../../third_party/RT-DETR/rtdetr_pytorch/configs/rtdetr) |
| v2 (`rtdetrv2`) | 16 | [rtdetrv2](../../third_party/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2) |
| v3 (`rtdetrv3`) | 5 | [rtdetrv3](../../third_party/RT-DETRv3/configs/rtdetrv3) |
| v4 (`rtdetrv4`) | 4 | [rtv4](../../third_party/RT-DETRv4/configs/rtv4) |

## RT-DETR v1

| `--backbone` | Upstream config | Checkpoint destination filename | Download |
| --- | --- | --- | --- |
| `r18vd` | [rtdetr_r18vd_6x_coco.yml](../../third_party/RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_r18vd_6x_coco.yml) | `rtdetr_r18vd_dec3_6x_coco_from_paddle.pth` | [checkpoint](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth) |
| `r101vd` | [rtdetr_r101vd_6x_coco.yml](../../third_party/RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_r101vd_6x_coco.yml) | `rtdetr_r101vd_2x_coco_objects365_from_paddle.pth` | [checkpoint](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_2x_coco_objects365_from_paddle.pth) |
| `r34vd` | [rtdetr_r34vd_6x_coco.yml](../../third_party/RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_r34vd_6x_coco.yml) | `rtdetr_r34vd_dec4_6x_coco_from_paddle.pth` | [checkpoint](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r34vd_dec4_6x_coco_from_paddle.pth) |
| `r50vd` | [rtdetr_r50vd_6x_coco.yml](../../third_party/RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_r50vd_6x_coco.yml) | `rtdetr_r50vd_6x_coco_from_paddle.pth` | [checkpoint](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth) |
| `r50vd_m` | [rtdetr_r50vd_m_6x_coco.yml](../../third_party/RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_r50vd_m_6x_coco.yml) | `rtdetr_r50vd_m_6x_coco_from_paddle.pth` | [checkpoint](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_m_6x_coco_from_paddle.pth) |
| `dla34` | [rtdetr_dla34_6x_coco.yml](../../third_party/RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_dla34_6x_coco.yml) | `rtdetr_dla34_6x_coco.pth` | **user supplied** |
| `regnet` | [rtdetr_regnet_6x_coco.yml](../../third_party/RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_regnet_6x_coco.yml) | `rtdetr_regnet_6x_coco.pth` | **user supplied** |

## RT-DETR v2

| `--backbone` | Upstream config | Checkpoint destination filename | Download |
| --- | --- | --- | --- |
| `r18vd` | [rtdetrv2_r18vd_120e_coco.yml](../../third_party/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml) | `rtdetrv2_r18vd_120e_coco_rerun_48.1.pth` | [checkpoint](https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth) |
| `r101vd` | [rtdetrv2_r101vd_6x_coco.yml](../../third_party/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml) | `rtdetrv2_r101vd_6x_coco_from_paddle.pth` | [checkpoint](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r101vd_6x_coco_from_paddle.pth) |
| `hgnetv2_h` | [rtdetrv2_hgnetv2_h_6x_coco.yml](../../third_party/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_hgnetv2_h_6x_coco.yml) | `rtdetrv2_hgnetv2_h_6x_coco_from_paddle.pth` | [checkpoint](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_hgnetv2_h_6x_coco_from_paddle.pth) |
| `hgnetv2_l` | [rtdetrv2_hgnetv2_l_6x_coco.yml](../../third_party/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_hgnetv2_l_6x_coco.yml) | `rtdetrv2_hgnetv2_l_6x_coco_from_paddle.pth` | [checkpoint](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_hgnetv2_l_6x_coco_from_paddle.pth) |
| `hgnetv2_x` | [rtdetrv2_hgnetv2_x_6x_coco.yml](../../third_party/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_hgnetv2_x_6x_coco.yml) | `rtdetrv2_hgnetv2_x_6x_coco_from_paddle.pth` | [checkpoint](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_hgnetv2_x_6x_coco_from_paddle.pth) |
| `r18vd_voc` | [rtdetrv2_r18vd_120e_voc.yml](../../third_party/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_120e_voc.yml) | `rtdetrv2_r18vd_120e_voc.pth` | **user supplied** |
| `r18vd_dsp` | [rtdetrv2_r18vd_dsp_3x_coco.yml](../../third_party/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_dsp_3x_coco.yml) | `rtdetrv2_r18vd_dsp_3x_coco.pth` | [checkpoint](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r18vd_dsp_3x_coco.pth) |
| `r18vd_sp1` | [rtdetrv2_r18vd_sp1_120e_coco.yml](../../third_party/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_sp1_120e_coco.yml) | `rtdetrv2_r18vd_sp1_120e_coco.pth` | [checkpoint](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r18vd_sp1_120e_coco.pth) |
| `r18vd_sp2` | [rtdetrv2_r18vd_sp2_120e_coco.yml](../../third_party/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_sp2_120e_coco.yml) | `rtdetrv2_r18vd_sp2_120e_coco.pth` | [checkpoint](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r18vd_sp2_120e_coco.pth) |
| `r18vd_sp3` | [rtdetrv2_r18vd_sp3_120e_coco.yml](../../third_party/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_sp3_120e_coco.yml) | `rtdetrv2_r18vd_sp3_120e_coco.pth` | [checkpoint](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r18vd_sp3_120e_coco.pth) |
| `r34vd` | [rtdetrv2_r34vd_120e_coco.yml](../../third_party/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r34vd_120e_coco.yml) | `rtdetrv2_r34vd_120e_coco_ema.pth` | [checkpoint](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r34vd_120e_coco_ema.pth) |
| `r34vd_dsp` | [rtdetrv2_r34vd_dsp_1x_coco.yml](../../third_party/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r34vd_dsp_1x_coco.yml) | `rtdetrv2_r34vd_dsp_1x_coco.pth` | [checkpoint](https://github.com/lyuwenyu/storage/releases/download/v0.1/rrtdetrv2_r34vd_dsp_1x_coco.pth) |
| `r50vd` | [rtdetrv2_r50vd_6x_coco.yml](../../third_party/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml) | `rtdetrv2_r50vd_6x_coco_ema.pth` | [checkpoint](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_6x_coco_ema.pth) |
| `r50vd_dsp` | [rtdetrv2_r50vd_dsp_1x_coco.yml](../../third_party/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r50vd_dsp_1x_coco.yml) | `rtdetrv2_r50vd_dsp_1x_coco.pth` | [checkpoint](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_dsp_1x_coco.pth) |
| `r50vd_m` | [rtdetrv2_r50vd_m_7x_coco.yml](../../third_party/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml) | `rtdetrv2_r50vd_m_7x_coco_ema.pth` | [checkpoint](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_m_7x_coco_ema.pth) |
| `r50vd_m_dsp` | [rtdetrv2_r50vd_m_dsp_3x_coco.yml](../../third_party/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r50vd_m_dsp_3x_coco.yml) | `rtdetrv2_r50vd_m_dsp_3x_coco.pth` | [checkpoint](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_m_dsp_3x_coco.pth) |

## RT-DETR v3

| `--backbone` | Upstream config | Checkpoint destination filename | Download |
| --- | --- | --- | --- |
| `r18vd` | [rtdetrv3_r18vd_6x_coco.yml](../../third_party/RT-DETRv3/configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml) | `rtdetrv3_r18vd_6x_coco.pdparams` | [checkpoint](https://drive.google.com/file/d/1zIDOjn1qDccC3TBsDlGQHOjVrehd26bk/view) |
| `r50vd` | [rtdetrv3_r50vd_6x_coco.yml](../../third_party/RT-DETRv3/configs/rtdetrv3/rtdetrv3_r50vd_6x_coco.yml) | `rtdetrv3_r50vd_6x_coco.pdparams` | [checkpoint](https://drive.google.com/file/d/1wfJE-QgdgqKE0IkiTuoD5HEbZwwZg3sQ/view) |
| `r18vd_lvis` | [rtdetrv3_r18vd_6x_lvis.yml](../../third_party/RT-DETRv3/configs/rtdetrv3/rtdetrv3_r18vd_6x_lvis.yml) | `rtdetrv3_r18vd_6x_lvis.pdparams` | **user supplied** |
| `r34vd` | [rtdetrv3_r34vd_6x_coco.yml](../../third_party/RT-DETRv3/configs/rtdetrv3/rtdetrv3_r34vd_6x_coco.yml) | `rtdetrv3_r34vd_6x_coco.pdparams` | [checkpoint](https://drive.google.com/file/d/12-wqAF8i67eqbocaWPK33d4tFkN2wGi2/view) |
| `r50vd_lvis` | [rtdetrv3_r50vd_6x_lvis.yml](../../third_party/RT-DETRv3/configs/rtdetrv3/rtdetrv3_r50vd_6x_lvis.yml) | `rtdetrv3_r50vd_6x_lvis.pdparams` | **user supplied** |

## RT-DETR v4

| `--backbone` | Upstream config | Checkpoint destination filename | Download |
| --- | --- | --- | --- |
| `s` | [rtv4_hgnetv2_s_coco.yml](../../third_party/RT-DETRv4/configs/rtv4/rtv4_hgnetv2_s_coco.yml) | `rtv4_hgnetv2_s_coco.pth` | [checkpoint](https://drive.google.com/file/d/1jDAVxblqRPEWed7Hxm6GwcEl7zn72U6z) |
| `m` | [rtv4_hgnetv2_m_coco.yml](../../third_party/RT-DETRv4/configs/rtv4/rtv4_hgnetv2_m_coco.yml) | `rtv4_hgnetv2_m_coco.pth` | [checkpoint](https://drive.google.com/file/d/1O-YpP4X-quuOXbi96y2TKkztbjroP5mX) |
| `l` | [rtv4_hgnetv2_l_coco.yml](../../third_party/RT-DETRv4/configs/rtv4/rtv4_hgnetv2_l_coco.yml) | `rtv4_hgnetv2_l_coco.pth` | [checkpoint](https://drive.google.com/file/d/1shO9EzZvXZyKedE2urLsN4dwEv8Jqa_8) |
| `x` | [rtv4_hgnetv2_x_coco.yml](../../third_party/RT-DETRv4/configs/rtv4/rtv4_hgnetv2_x_coco.yml) | `rtv4_hgnetv2_x_coco.pth` | [checkpoint](https://drive.google.com/file/d/19gnkMTgFveJsrOvSmEPQXCTG6v9oQHN3) |

## Dataset contracts

COCO variants use 80 classes, v2 `r18vd_voc` uses 20 Pascal VOC classes, and v3 `r18vd_lvis` / `r50vd_lvis` use 1,203 LVIS v1 classes. The selected entry controls class names and ONNX logits width. VOC checkpoints must use the supplied standard class order; edit the entry's `classes` file if training used another `label_list.txt`. LVIS names follow category IDs 1–1203 from [Detectron2's LVIS v1 metadata](https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/datasets/lvis_v1_categories.py).

The host mAP evaluator currently evaluates COCO80 models only and rejects VOC/LVIS variants. Evaluate those with their upstream dataset evaluator. Quantization and engine building consume the selected ONNX; choose representative calibration images for the target dataset.

For Jetson inference, build a TensorRT plan from the ONNX and supply a C++ runtime contract matching its raw query outputs. For the 20-class VOC export, copy `cpp/models/v1-r101.json` and set `classes` to `20`, `name` to the VOC model name, and tensor names and input dimensions to those in the export. Then pass the matching class YAML:

```bash
./cpp/build_engine.sh host/models/rtdetrv2/rtdetrv2_r18vd_voc.onnx \
  artifacts/engines/rtdetrv2_r18vd_voc.plan
cpp/build/rtdetr-infer --model path/to/voc-raw.json \
  --engine artifacts/engines/rtdetrv2_r18vd_voc.plan \
  --source samples/input.mp4 --classes host/configs/classes/voc20.yaml
```

## Validation

All 32 registry entries are checked against their upstream model directories. All 27 PyTorch variants construct and run a 640×640 forward pass with synthetic weights. HGNetv2-L, VOC R18, DLA34, and RegNet additionally export and execute in ONNX Runtime. Synthetic detector weights can produce nearly tied query scores and different query ordering across runtimes; those runs establish shape and execution, not checkpoint accuracy.

The official v2 R18 discrete-sampling checkpoint exports and matches PyTorch in ONNX Runtime on a random input, with maximum absolute differences of `1.72e-5` for logits and `1.56e-5` for boxes. v2 R18's default checkpoint has also been checked previously. These checks do not establish Jetson/TensorRT compatibility or dataset mAP. v3 requires a compatible Paddle environment and has not yet been exported end to end here.

v1 DLA34 and RegNet need user-trained detector checkpoints. The downstream RegNet constructor preserves upstream parameter names and the RegNet-Y-040 architecture while avoiding an unnecessary ImageNet download. v2 HGNet checkpoints are present in the upstream release assets even though their rows are commented out in the pinned README. The R34 DSP download filename starts with `rrtdetrv2`; save it under the single-`r` destination filename in the table.
