"""H1 pilot — does precision-boundary placement change latency?

Two FP16 partitions of the same graph hold the SAME number of nodes at FP32 (N),
differing only in whether those nodes are contiguous or scattered. Under strong
typing, scattered means more materialized Cast nodes and more broken fusion regions.

If latency does not separate beyond the run-to-run noise floor, boundary cost is not
a real effect on this hardware and the boundary-aware research direction is dead.
Measuring the noise floor first is what makes the comparison meaningful: without it,
any difference could be tactic jitter.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.boundary import contiguous_block, count_casts, float_node_names, scattered_block
from harness.config import load_spec
from harness.paths import ENGINES_DIR, TABLES_DIR
from harness.precision import to_fp16
from harness.trt_runner import EngineBuildError, build_engine, plan_fingerprint, save_engine
from harness.trtexec import bench_latency


def _measure(onnx_path, tag: str, *, repeats: int, iterations: int, timing_cache=None) -> dict:
    casts = count_casts(onnx_path)
    try:
        engine = build_engine(onnx_path, timing_cache=timing_cache)
    except EngineBuildError:
        # Not every labeling of the graph compiles. TensorRT fuses transformer blocks into
        # Myelin regions and emits one NVRTC kernel per region; a precision split *inside*
        # one fails codegen with no fallback implementation. That makes the partition
        # infeasible -- a property of the partition, and evidence neither for nor against
        # H1, so it is recorded rather than allowed to abort the sweep.
        return {"tag": tag, "casts": casts, "feasible": False, "p50_ms": float("nan"),
                "spread_ms": float("nan"), "fingerprint": "-", "engine_mb": float("nan")}
    engine_path = ENGINES_DIR / f"h1_{tag}.engine"
    save_engine(engine, engine_path)
    samples = [bench_latency(engine_path, iterations=iterations)["lat_p50_ms"]
               for _ in range(repeats)]
    return {
        "tag": tag,
        "casts": casts,
        "feasible": True,
        "p50_ms": statistics.median(samples),
        "spread_ms": max(samples) - min(samples),
        "fingerprint": plan_fingerprint(engine),
        "engine_mb": len(engine) / 2 ** 20,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="H1: does boundary placement change latency?")
    ap.add_argument("--spec", default="rtdetr")
    ap.add_argument("--backbone", default=None)
    ap.add_argument("-n", type=int, default=16, help="nodes held at FP32 in both partitions")
    ap.add_argument("--repeats", type=int, default=5, help="trtexec runs per engine")
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--no-timing-cache", action="store_true",
                    help="build without the spec's TensorRT timing cache")
    args = ap.parse_args()

    spec = load_spec(args.spec, args.backbone)
    # Both variants share one cache, so a tactic is timed once and the contiguous vs
    # scattered delta is boundary cost rather than tactic-timing jitter.
    timing_cache = None if args.no_timing_cache else spec.timing_cache
    src = spec.onnx_path("fp32")
    names = float_node_names(src)
    print(f"[h1] {len(names)} float nodes in {src.name}; N={args.n}")

    variants = {
        "contig": contiguous_block(names, args.n),
        "scatter": scattered_block(names, args.n),
    }
    rows = []
    for tag, block in variants.items():
        onnx_out = ENGINES_DIR / f"h1_{tag}.onnx"
        to_fp16(src, onnx_out, node_block_list=block)
        row = _measure(onnx_out, tag, repeats=args.repeats, iterations=args.iterations,
                       timing_cache=timing_cache)
        rows.append(row)
        if row["feasible"]:
            print(f"[h1] {tag:8s} casts={row['casts']:4d}  "
                  f"p50={row['p50_ms']:.3f} ms  spread={row['spread_ms']:.3f} ms")
        else:
            print(f"[h1] {tag:8s} casts={row['casts']:4d}  INFEASIBLE — TensorRT could not "
                  f"build this partition")

    infeasible = [r for r in rows if not r["feasible"]]
    if infeasible:
        # Inconclusive is NOT the same as unsupported: the arms were never comparable, so
        # this must not read as evidence against H1 (hence its own exit code).
        noise = delta = float("nan")
        d_casts = rows[1]["casts"] - rows[0]["casts"]
        verdict = ("H1 INCONCLUSIVE — " + ", ".join(r["tag"] for r in infeasible)
                   + " could not be built, so the two arms are not comparable")
        status = 2
        print(f"\n[h1] {verdict}")
    else:
        # Noise floor: the worst within-config spread. A difference smaller than this is
        # indistinguishable from tactic jitter, no matter how suggestive it looks.
        noise = max(r["spread_ms"] for r in rows)
        contig, scatter = rows[0], rows[1]
        delta = scatter["p50_ms"] - contig["p50_ms"]
        d_casts = scatter["casts"] - contig["casts"]
        holds = abs(delta) > 3 * noise and delta > 0
        verdict = "H1 HOLDS" if holds else "H1 NOT SUPPORTED"
        status = 0 if holds else 1
        print(f"\n[h1] Δlatency = {delta:+.3f} ms over Δcasts = {d_casts:+d}  "
              f"(noise floor {noise:.3f} ms)\n[h1] {verdict}")

    lines = [
        "# H1 — boundary-cost pilot", "",
        f"spec `{spec.name}` · backbone `{spec.backbone}` · N={args.n} nodes held FP32 · "
        f"{args.repeats} trtexec runs × {args.iterations} iterations · "
        f"timing cache {'shared across both arms' if timing_cache else 'disabled'}", "",
        "| partition | FP32 nodes | cast nodes | p50 (ms) | within-config spread (ms) | engine (MiB) | plan |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r["feasible"]:
            lines.append(f"| {r['tag']} | {args.n} | {r['casts']} | {r['p50_ms']:.3f} | "
                         f"{r['spread_ms']:.3f} | {r['engine_mb']:.1f} | `{r['fingerprint'][:12]}` |")
        else:
            lines.append(f"| {r['tag']} | {args.n} | {r['casts']} | — | — | — | build failed |")
    if infeasible:
        lines += ["",
                  f"TensorRT could not build: **{', '.join(r['tag'] for r in infeasible)}**.",
                  "",
                  "A precision boundary *inside* a Myelin-fused region — TensorRT compiles "
                  "each transformer attention block to a single NVRTC kernel — fails codegen "
                  "with no fallback implementation. Not every labeling of the graph is "
                  "buildable, so the partition space carries hard feasibility constraints "
                  "and these two arms were never comparable.", "",
                  f"**{verdict}**", ""]
    else:
        lines += ["",
                  f"Δlatency **{delta:+.3f} ms** over Δcasts **{d_casts:+d}**; "
                  f"noise floor **{noise:.3f} ms**.", "",
                  f"**{verdict}** — decision rule: |Δlatency| > 3× noise floor and scattered slower.", ""]
    # per-backbone, so an r101 run cannot silently overwrite the r18 evidence
    out = TABLES_DIR / f"h1_boundary_pilot_{spec.backbone}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"[h1] wrote {out}")

    sys.exit(status)   # 0 = H1 holds, 1 = not supported, 2 = inconclusive (unbuildable arm)


if __name__ == "__main__":
    main()