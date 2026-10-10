# Finish RT-DETR: Compute vs Memory R/W Attribution (half a day)

Prereq for every measurement: `sudo nvpmodel -m 0 && sudo jetson_clocks`, fan on, log temps. Same engine, same input size as your existing figures — the new data must be attributable to the old runs.

## Step 1 — Per-layer time (30 min): trtexec profile

```bash
trtexec --loadEngine=rtdetr.engine \
  --iterations=200 --warmUp=500 --useSpinWait --useCudaGraph \
  --dumpProfile --exportProfile=results/layers.json \
  --exportTimes=results/times.json
```
- `layers.json` → per-layer ms (post-fusion names; map fused blocks back to model stages: backbone / AIFI encoder / decoder / heads).
- Run once more **without** `--useCudaGraph` — the delta is your launch-overhead line item, itself worth a sentence on an SoC.

## Step 2 — Timeline truth (1 h): Nsight Systems on the *real* pipeline

trtexec profiles the engine in isolation; your pipeline adds preprocessing, copies, postprocess:

```bash
nsys profile -t cuda,nvtx,osrt -o results/pipeline --stats=true python3 infer.py --n 100
```
Wrap pipeline stages in NVTX ranges (`torch.cuda.nvtx.range_push/pop` or `nvtx` pkg) so the report groups by *your* stages. Read from the stats/GUI:
- kernel time vs `cudaMemcpy`(H2D/D2H) vs gaps (CPU-bound stretches);
- whether copies overlap compute (streams) or serialize;
- preprocess share — on Orin the resize/normalize on CPU is a classic hidden tax (your NEON instincts apply; or move it to GPU with NPP/DALI and measure the delta).

**Unified-memory experiment (30 min):** allocate input/output with `cudaHostAllocMapped` (zero-copy) vs explicit `cudaMemcpy`, measure both. On discrete GPUs copies are mandatory; on Orin they're a choice — quantify what the choice costs. This is the most SoC-specific finding in the whole chapter.

## Step 3 — DRAM traffic per kernel (1.5 h): Nsight Compute

```bash
sudo ncu --target-processes all \
  --metrics dram__bytes_read.sum,dram__bytes_write.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed \
  --launch-count 200 -o results/ncu_rtdetr python3 infer.py --n 1
```
(You know this tool from the sm_89 SGEMM work; same discipline, sm_87 target. The research pass could NOT verify NCU dram__bytes counter availability on current JetPack builds — spend 10 minutes confirming `ncu --query-metrics | grep dram__bytes` on the board before planning the afternoon around it; if the counters are missing, fall back to `lts__t_bytes`/L2 metrics plus EMC% from tegrastats and say so in the write-up.)

Per kernel → arithmetic intensity = FLOPs / (dram bytes r+w); place on the roofline vs the Orin Nano's LPDDR5 ceiling (~102 GB/s is NVIDIA's Super-mode marketing figure — the research pass did not independently verify it, so measure your own ceiling with a STREAM-style triad or `bandwidthTest` and use the measured number on the roofline) and its FP16/INT8 peak. Kernels sitting on the bandwidth roof are the "no kernel can save you" set — name them.

## Step 4 — System corroboration (15 min): EMC utilization

```bash
tegrastats --interval 100 --logfile results/tegrastats.log   # during a 60 s steady run
```
EMC% near saturation while SM% is modest = system-level confirmation of the roofline story. Two independent instruments agreeing is what makes the chapter credible.

## Step 5 — Reconcile and write (1 h)

- Attribution table (README) — components must sum to measured e2e within ~5%; if they don't, find the missing time (usually CPU gaps or sync points) — the hunt itself is good content.
- One roofline figure, one timeline screenshot, the verdict paragraph.
- Append INT8-vs-FP16 per-layer deltas if you have both engines: which layers *stop* being bandwidth-bound at INT8 (weights halve; bandwidth-bound layers speed up ~2×, compute-bound ones don't — the divergence is the lesson).

## Output → feeds

`results/` CSVs + the chapter markdown → becomes Section 1 of whichever write-up ships Sunday (`03-publication.md`), and the shared harness for the stretch (`02-stretch-...`): every future detector gets this same table, which is exactly what makes the shootout different from every mAP-vs-FPS blog post.
