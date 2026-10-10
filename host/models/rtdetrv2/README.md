# RT-DETRv2

Uses the official PyTorch implementation in `third_party/RT-DETR`. All 16 upstream v2 configs are registered: R18/R34/R50/R50-M/R101, HGNetv2 L/X/H, discrete-sampling and sampling-point variants, and R18 VOC. See the [variant catalog](../variants.md) for checkpoints and [`../../configs/rtdetrv2.yaml`](../../configs/rtdetrv2.yaml) for ONNX destinations. `python host/pipeline/export.py --spec rtdetrv2 --list` lists every variant.

From the repository root:

```bash
git submodule update --init third_party/RT-DETR
python -m pip install -r host/requirements-export.lock
```

Download the matching official checkpoint into `host/models/rtdetrv2/checkpoints/`. For R18:

```bash
mkdir -p host/models/rtdetrv2/checkpoints
curl -fL https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth \
  -o host/models/rtdetrv2/checkpoints/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth
python host/pipeline/export.py --spec rtdetrv2 --backbone r18vd
```

The exporter loads the official checkpoint and writes a static FP32 ONNX with `images` input and `pred_logits`/`pred_boxes` outputs. Run the export command in a fresh Python process if you have already imported the original RT-DETR `src` package in an interactive session.

For R101, use the [official checkpoint release](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch) and the filename in the spec, then select `--backbone r101vd`.
