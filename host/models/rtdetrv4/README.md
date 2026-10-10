# RT-DETRv4

Uses the [official PyTorch implementation](https://github.com/RT-DETRs/RT-DETRv4). The registered HGNetv2 variants are `s`, `m`, `l`, and `x`; checkpoint names and ONNX paths are in [`../../configs/rtdetrv4.yaml`](../../configs/rtdetrv4.yaml).

From the repository root:

```bash
git submodule update --init third_party/RT-DETRv4
python -m pip install -r host/requirements-export.lock
```

Download a matching COCO checkpoint from upstream's README and put it under `host/models/rtdetrv4/checkpoints/` using the filename in the spec. Then run:

```bash
python host/pipeline/export.py --spec rtdetrv4 --backbone s
```

The exporter loads model weights and emits raw `pred_logits` and `pred_boxes`, without the upstream postprocessor. The common host adapter decodes these into COCO detections. A synthetic checkpoint was used to validate the S variant's export and ONNX Runtime execution; official checkpoint accuracy remains unmeasured here.
