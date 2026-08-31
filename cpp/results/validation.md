# Validation on 2026-09-25

## COCO 2017 validation

These historical measurements used a separate COCO evaluation workflow over
all 5,000 validation images with score threshold zero; standard bbox AP was
calculated with `pycocotools`. Model export and accuracy evaluation tools are
outside this inference-focused repository.
These results use an RTX 4050 Laptop GPU and TensorRT 11.0.0.114. The four
original rows use the manifests and checkpoints listed in the project README;
v1-R18 uses the official `rtdetr_r18vd_dec3_6x_coco_from_paddle.pth` checkpoint.

The original v1-R101 plan came from the existing prepared FP16 ONNX graph.
The additional v1-R18 plan was exported from its PyTorch checkpoint outside
the current repository workflow, then built with TensorRT 11.

| Variant | Plan precision | AP@[.50:.95] | AP50 |
| --- | --- | ---: | ---: |
| v1 R18-vd | FP32 graph | 0.464 | 0.637 |
| v1 R101-vd | FP16 graph | 0.541 | 0.737 |
| v2 R50-vd | FP32 graph | 0.534 | 0.716 |
| v3 R18-vd | FP32 graph | 0.481 | 0.656 |
| v4 HGNetv2-S | FP32 graph, bilinear gather export | 0.498 | 0.671 |

For v4, the official exporter graph with native `GridSample` yielded 0.382 AP
under TensorRT 11. The same checkpoint with the project's bilinear gather
export yielded 0.498 AP. ONNX Runtime and PyTorch agreed closely on a checked
image before the replacement; the TensorRT plan diverged. The replacement
matched PyTorch sampling on randomized tensors, and the resulting plan passed
the full COCO run.

## CLI speed

Both CLIs used the same v1 TensorRT plan, source video, `--no-draw`, 10 warmup
iterations, and the first 30 decoded frames on each machine. E2E timing starts
after frame read; pipeline FPS includes frame read. These are short sequential
runs, not sustained throughput claims. The Jetson plan was built with TensorRT
10.16.2.

| Machine and source | CLI | Engine p50 ms | E2E p50 ms | Pipeline FPS |
| --- | --- | ---: | ---: | ---: |
| RTX 4050 Laptop, 4K MP4 | C++ | 8.47 | 14.51 | 43.52 |
| RTX 4050 Laptop, same MP4 | existing Python | 9.12 | 78.39 | 11.47 |
| Jetson Orin Nano, 4K MP4 | C++ | 36.81 | 53.94 | 11.46 |
| Jetson Orin Nano, same MP4 | existing Python | 39.47 | 160.85 | 5.16 |

Desktop C++ timings for the same 30 frames and settings:

| Variant | Engine p50 ms | E2E p50 ms | Pipeline FPS |
| --- | ---: | ---: | ---: |
| v1 R101-vd | 8.47 | 14.51 | 43.52 |
| v2 R50-vd | 15.71 | 21.53 | 34.37 |
| v3 R18-vd | 7.89 | 9.75 | 58.22 |
| v4 HGNetv2-S, gather export | 5.73 | 11.78 | 51.85 |

The desktop source was a previously annotated output MP4; Jetson used the
original unannotated clip. Cross-machine times should be interpreted
independently. All video timings include the
model's preprocessing and output decoding. `--no-draw` excludes annotation,
display, and video encoding.

In a separate 30-frame Jetson v1 run with `--power`, the sampled mean VDD_IN
was 12.02 W. The corrected v4 FP16 plan on Jetson ran 10 frames at 14.36 ms
engine p50 and 31.28 ms E2E p50; this short run is a smoke benchmark.

## Parity and unit checks

Desktop C++ detections matched a Python runner using each respective TensorRT
plan on 10 video frames for v2, v3, and the corrected v4 plan; v1 matched on
30 frames. Jetson v1–v4 also matched their same-plan Python runners on 10
frames each. The corrected Jetson v4 FP16 plan achieved 0.553 AP on the first
200 COCO images, versus 0.554 for the desktop FP32 plan on those same images.
Jetson v3 FP16 achieved 0.537 AP on that subset, versus 0.536 for desktop FP32.
The newly exported v1-R18 plan matched the Python TensorRT decoder on 10 video
frames with zero detection mismatches.
These checks verify the C++ I/O and decoding against the plan,
while COCO AP checks detection quality against labeled images.

`ctest` passed both native tests on desktop and Jetson. The native Pillow-like
resize was within one byte of reference pixels in the fixture set.
The v4 bilinear gather parity check passed during the original export study;
that exporter and its test are no longer part of this runtime repository.
