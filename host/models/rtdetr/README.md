# RT-DETR preparation

The maintained preparation workflow is in [host/README.md](../../README.md).

- `export.py`: checkpoint to ONNX.
- `infer.py`: host backend comparisons.
- `eval_map.py`: host ONNX Runtime, TensorRT, or eager COCO evaluation.
- `checkpoints/`: local checkpoint files, excluded from Git.

Model paths and variants are configured in `host/configs/rtdetr.yaml`. Jetson users run the separate `deploy/` workflow with prepared ONNX models and saved engines.

All seven upstream v1 configs are registered: R18/R34/R50/R50-M/R101, DLA34, and RegNet. DLA34 and RegNet require user-trained detector checkpoints. See the [variant catalog](../variants.md) for exact filenames and downloads.
