# RT-DETRv3

RT-DETRv3's official implementation and checkpoints use PaddlePaddle. Export needs its source, PaddlePaddle, `paddle2onnx`, and the matching `.pdparams` checkpoint. Keep these in a separate export environment if their dependency versions conflict with the PyTorch host environment.

All five upstream v3 configs are registered: `r18vd`, `r34vd`, `r50vd`, `r18vd_lvis`, and `r50vd_lvis`. The [catalog](../variants.md) lists checkpoint filenames and release links; LVIS variants require your own trained checkpoints.

Follow the [complete export guide](../README.md#4-rt-detrv3-paddlepaddle) for the Python 3.10 environment, upstream checkpoint downloads, and exact file destinations. Put the checkpoint at the path listed in [`../../configs/rtdetrv3.yaml`](../../configs/rtdetrv3.yaml), then run:

```bash
python host/pipeline/export.py --spec rtdetrv3 --backbone r18vd
```

The exporter invokes upstream's Paddle inference export with postprocessing excluded, converts the graph with `paddle2onnx`, then validates and renames the raw 300-query boxes/logits to this repo's ONNX contract. Its host adapter uses the upstream OpenCV bilinear resize. PyTorch inference is unavailable for this Paddle checkpoint; ONNX Runtime and TensorRT use the exported ONNX instead.

This path still needs an end-to-end run with an official checkpoint and a compatible Paddle environment. The exporter rejects unexpected input and output shapes rather than silently treating postprocessed detections as raw queries.
