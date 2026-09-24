"""Latency/throughput from the official `trtexec` tool.

`TrtRunner` wall-clock and `trtexec` GPU-compute time are not comparable: trtexec
uses a CUDA graph, excludes H2D/D2H, and reports pure device time. Every latency
number in results/ comes from here so the numbers are homogeneous.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

_KEYS = ("lat_mean_ms", "lat_p50_ms", "lat_p90_ms", "lat_p99_ms", "throughput_qps")


def _grab(text: str, stat: str) -> float:
    """One statistic out of trtexec's `GPU Compute Time:` summary line."""
    m = re.search(rf"GPU Compute Time:.*?{re.escape(stat)}\s*=\s*([\d.]+)\s*ms", text)
    return float(m.group(1)) if m else float("nan")


def parse_trtexec(stdout: str) -> dict:
    """Pure parser, so the regexes are testable without a GPU or an engine."""
    thr = re.search(r"Throughput:\s*([\d.]+)\s*qps", stdout)
    return {
        "lat_mean_ms": _grab(stdout, "mean"),
        "lat_p50_ms": _grab(stdout, "median"),
        "lat_p90_ms": _grab(stdout, "percentile(90%)"),
        "lat_p99_ms": _grab(stdout, "percentile(99%)"),
        "throughput_qps": float(thr.group(1)) if thr else float("nan"),
    }


def trtexec_bin() -> str:
    exe = shutil.which("trtexec")
    if exe:
        return exe
    for cand in Path.home().glob("sdks/TensorRT-*/bin/trtexec"):
        if cand.exists():
            return str(cand)
    raise FileNotFoundError(
        "trtexec not found on PATH or in ~/sdks/TensorRT-*/bin — "
        "install the TensorRT tarball or add its bin/ to PATH"
    )


def bench_latency(engine_path, *, iterations: int = 2000, use_cuda_graph: bool = True) -> dict:
    """Benchmark a saved engine. `--loadEngine` never rebuilds, so the builder config
    pinned by `build_engine` (notably TF32) cannot be silently changed here."""
    cmd = [
        trtexec_bin(),
        f"--loadEngine={engine_path}",
        f"--iterations={iterations}",
        "--noDataTransfers",
    ]
    if use_cuda_graph:
        cmd.append("--useCudaGraph")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return parse_trtexec(proc.stdout)