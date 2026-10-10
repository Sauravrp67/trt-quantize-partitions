# Stretch: From One Model to the 2026 Edge-Detector Shootout

**Landscape verified 2026-09-15/16** by an adversarially-checked research pass (3-vote verification; sources in §6). Vendor numbers below are self-reported — treating them as claims-to-test, not truth, is the whole point of this project.

---

## 1. The idea: your RT-DETR run is row one of a table, not a finished project

The attribution harness from `01-finish-the-breakdown.md` — per-layer time, DRAM traffic, roofline placement, zero-copy analysis, power, sustained thermals — is model-agnostic. Every additional detector costs roughly half a day. What makes it citable: **no verified 2025–26 detector paper contains a single Jetson number, INT8 result, or per-layer attribution.** Grep-verified: the RF-DETR ICLR paper has zero hits for Jetson/Orin/edge/INT8/roofline/DRAM; RT-DETRv4's zoo latencies are all "Latency (T4)"; Ultralytics' YOLO26 docs contain no Jetson benchmarks at all. The gap you'd fill is real and confirmed.

Better still: `rf-detr` issue **#1043** reports latency *rankings vs D-FINE flip on different hardware* — vendor T4 Pareto ordering demonstrably does not transfer. "Does the T4 ranking survive on Orin INT8?" is a falsifiable question nobody has answered.

## 2. The verified Sept-2026 roster

### YOLO26 (Ultralytics) — SHIPPED, January 2026
- Five detection sizes n/s/m/l/x; task variants (-seg etc.); YOLO27 already teased.
- Claimed COCO mAP50-95: 40.9 / 48.6 / 53.1 / 55.0 / 57.5; T4 TensorRT: 1.7 / 2.5 / 4.7 / 6.2 / 11.8 ms. **All vendor, all T4.**
- **The architectural hook for you:** DFL removed, and NMS-free inference comes from an **optional one-to-one head that is NOT the default** — default is one-to-many + NMS; `nms=False` switches. ⇒ **Benchmark BOTH head modes of the same weights on Orin INT8.** The one-to-many-vs-one-to-one latency/accuracy split on a bandwidth-limited SoC is a measurement that exists nowhere.
- License: **AGPL-3.0** / paid Enterprise. Benchmarking is fine; note deployability in the license column.
- Paper: arXiv 2606.03748. Docs: docs.ultralytics.com/models/yolo26.

### RF-DETR (Roboflow + CMU) — now an ICLR 2026 paper, six variants
- "RF-DETR: Neural Architecture Search for Real-Time Detection Transformers" — arXiv **2511.09554** (v2 2026-02-03), accepted **ICLR 2026** (OpenReview camera-ready confirmed).
- Lineup: Nano 30.5M / 2.3 ms / 48.0 AP · Small 32.1M / 3.5 ms / 52.9 · Medium 33.7M / 4.4 ms / 54.7 · up to 2X-Large 126.9M / 17.2 ms / **60.1 AP** — claimed first real-time detector past 60 AP on COCO; beats D-FINE-nano by 5.3 AP at similar latency. All T4 + TRT 10.4 + CUDA 12.4, FP16/FP32.
- ⚠️ **License UNKNOWN.** The pre-existing "Apache-2.0" assumption was **refuted 0-3** in verification. Check `github.com/roboflow/rf-detr` LICENSE the day you start, before publishing anything.
- DINOv2-derived backbone ⇒ its INT8 PTQ behavior on Orin is an open, interesting question (ViT backbones quantize differently than CNNs — you know this from Kria).

### RT-DETRv4 — your baseline's direct successor, ECCV 2026
- "Painlessly Furthering Real-Time Object Detection with Vision Foundation Models" — arXiv **2510.25257**; code at `github.com/RT-DETRs/RT-DETRv4` (official continuation, announced in lyuwenyu/RT-DETR). Frozen **DINOv3** ViT-B teacher distilled into lightweight detectors; repo builds on and reproduces **DEIM, D-FINE, RT-DETRv2**.
- Zoo claims: 49.8 AP @ 3.66 ms (S) → 57.0 @ 12.90 ms (X) — T4 only, no Jetson, no INT8 (absence grep-verified).
- **The lineage hook:** RT-DETR (yours, done) → RT-DETRv4-S on the same harness = "what did three years of DETR evolution buy on the same $249 board?" One repo also gives you D-FINE/DEIM configs if you extend later.

### Recommended scope for the remaining window
**RT-DETR (done) + YOLO26n/s in both head modes + RF-DETR Nano/Small.** Three architecture families, ~1.5 days with the scripted harness. RT-DETRv4-S is the first stretch beyond that; D-FINE/DEIM/LW-DETR go in the public future-work list (an outreach hook: "happy to add yours").

## 3. Fairness protocol — adopt RF-DETR's, then out-honest it

The RF-DETR paper (Section 4) explicitly calls out that *"YOLO-based models often omit non-maximal suppression (NMS) when computing latency, leading to unfair comparisons"*, standardizes NMS-inclusive latency, uses a 200 ms inter-pass buffer against thermal throttling, and admits failing to reproduce YOLOv8/v11 mAP in TensorRT (multi-class vs single-class NMS). Adopt all of it, cite it, then add what they didn't do:

1. **NMS-inclusive "boxes-out" latency** on the same device, *plus* engine-only ms as a separate column. For YOLO26, both head modes make this a three-way comparison.
2. **One TensorRT version, one board, one power mode, one input size** per table — version pinned in every CSV row (see §4 — this bit is now sharper than it was).
3. **INT8 per model via each model's real calibration path**, accuracy re-measured post-quant through one shared eval script — vendor mAP never reused. (Their paper is FP16/FP32-only; INT8 is your differentiator, and ViT-backbone-vs-CNN INT8 behavior is the analysis layer.)
4. **Attribution** — per-layer DRAM roofline. Zero prior art in any of the three papers (grep-verified). This column is why your table gets cited instead of skimmed.
5. Their 200 ms thermal buffering **plus** your minute-10 sustained section — burst *and* steady-state.

## 4. ⚠️ The JetPack trap (new, verified, affects your existing numbers)

**JetPack 7.2 (2026-06-02; 7.2.1 on 2026-08-11) now supports the Orin Nano** — Ubuntu 24.04, CUDA 13.2.1, **TensorRT 10.16.2** — and there is **no SD-card image anymore**; flashing is SDK Manager/ISO, with a JetPack-6.x-era firmware prerequisite. Your existing RT-DETR figures were almost certainly taken on JetPack 6.x TensorRT.

Consequences:
- **Do not reflash the borrowed board.** Mid-week, on a loaner, through the new no-SD-card path, with a firmware prerequisite — that is how the week dies. Pin what's installed, report it, done.
- Every published number carries its JetPack/TRT version; old and new figures are **not comparable across the 6.x→7.2 boundary**, and your write-up should say so explicitly (it's a credibility feature: you're the benchmark that states its TRT version — F4's verified finding is that vendors mixing TRT versions is one of the standing fairness problems).
- "Re-run under JetPack 7.2 / TRT 10.16" goes in future work — a natural follow-up post *after* the board goes back, on any Orin you get next.

## 5. License column (verified where possible)

| Model | License | Status |
|---|---|---|
| YOLO26 | **AGPL-3.0** / Enterprise | verified 2026-09-15 |
| RF-DETR | **UNKNOWN — check LICENSE file before publishing**; Apache-2.0 assumption refuted 0-3 | open question |
| RT-DETR / RT-DETRv4 | check repo at start (v1 was Apache-2.0; v4 unverified) | check |

Deployability-by-license next to results is information your employer-audience actually uses; keep the column.

## 6. Sources (primary, fetched 2026-09-15/16)

YOLO26: docs.ultralytics.com/models/yolo26 · docs.ultralytics.com/guides/end2end-detection · arXiv 2606.03748 — RF-DETR: arXiv 2511.09554 · OpenReview ICLR 2026 camera-ready · blog.roboflow.com/rf-detr-nano-small-medium · github.com/roboflow/rf-detr (+ issue #1043) — RT-DETRv4: arXiv 2510.25257 · github.com/RT-DETRs/RT-DETRv4 · ECCV 2026 poster page — JetPack: developer.nvidia.com/embedded/jetpack (archive 7.2) · Orin Nano devkit quick-start docs. Full verification log: the deep-research run of 2026-09-15 (23 confirmed / 2 refuted claims, 3-vote adversarial).
