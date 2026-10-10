# RT-DETR on Jetson Orin Nano — Completed Run + What's Left + The Stretch

**Status:** RT-DETR ran with TensorRT on the Orin Nano; latency figures exist. **Missing: the compute-vs-memory attribution** — where the milliseconds actually go between kernels, DRAM traffic, and copies. That gap is the difference between "I ran a benchmark" and the thing you're known for (the 1.82 ms compute vs 41.76 ms e2e Versal analysis).

This folder:
- `01-finish-the-breakdown.md` — the half-day plan to get compute vs memory r/w timing, with exact commands. **Do this before either Option A or B starts** — it reuses the engine you already built and completes a publishable chapter.
- `02-stretch-rfdetr-yolo26-sota.md` — how this one-model run stretches into the 2026 edge-detector shootout (RF-DETR, YOLO26, D-FINE/DEIM lineage). Being finalized against a live research pass of the Sept-2026 landscape.
- `03-publication.md` — the shared write-up template + CV/LinkedIn lines for **all three** scaffolds in `project_scaffolds/`.

## Why finishing this matters more than starting something new

A benchmark with only end-to-end latency invites exactly one question from every NVIDIA/Zoox/robotics interviewer: *"where does the time go?"* The attribution table answers it before it's asked, and it's ~4 hours of work on an engine that already exists. Ship the chapter, then start the week's main project.

## What "done" looks like (the attribution table)

| Component | Time (ms) | % | DRAM read (MB) | DRAM write (MB) | Bound by |
|---|---|---|---|---|---|
| Preprocess (CPU/GPU) | | | | | |
| H2D copies / zero-copy | | | | | |
| Backbone convs | | | | | |
| Encoder (AIFI) attention | | | | | |
| Decoder + deformable/grid-sample ops | | | | | |
| Postprocess + D2H | | | | | |
| **Sum vs measured e2e** | must reconcile within ~5% | | | | |

Plus: per-layer roofline placement (arithmetic intensity vs the Orin's LPDDR5 bandwidth ceiling), and the one-paragraph verdict — *which third of RT-DETR is bandwidth-bound on this SoC and what that predicts for every other DETR-family model.*
