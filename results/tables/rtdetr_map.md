# COCO mAP — RT-DETR r18vd

backend: `PyTorch` · images: 5000 of val2017

| model | mAP@50-95 | mAP@50 | Δ mAP@50-95 | Δ mAP@50 |
|---|---|---|---|---|
| `eager checkpoint` | 0.4640 | 0.6372 | ref | ref |

_Accuracy only — no score threshold applied (COCOeval sweeps them). For latency see `models/rtdetr/infer.py` or `trtexec` with locked clocks._
