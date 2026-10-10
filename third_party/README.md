# Upstream model source

This directory holds pinned, unmodified model implementations used during host-side checkpoint loading and ONNX export. The RT-DETR submodule contains both the original PyTorch model and RT-DETRv2; RT-DETRv3 and RT-DETRv4 have separate submodules. Their revisions are recorded by this repository.

From the repository root, initialize the source required by your model:

```bash
git submodule update --init third_party/RT-DETR
git submodule update --init third_party/RT-DETRv3
git submodule update --init third_party/RT-DETRv4
```

Each [`host/models/`](../host/models/) family directory owns the corresponding export adapter and setup instructions. Checkpoints and generated ONNX files stay there or under `artifacts/`; they are not part of the submodules. Host quantization consumes ONNX, and Jetson engine building consumes ONNX, so neither step needs the upstream source checkout on the board.
