# tests/test_trtexec.py
import math

from harness.trtexec import parse_trtexec

SAMPLE = """
[I] === Performance summary ===
[I] Throughput: 121.478 qps
[I] Latency: min = 8.09 ms, max = 9.44 ms, mean = 8.29 ms
[I] GPU Compute Time: min = 8.05 ms, max = 9.31 ms, mean = 8.16 ms, median = 8.13 ms, percentile(90%) = 8.31 ms, percentile(99%) = 8.95 ms
[I] Total Host Walltime: 3.01 s
"""


def test_parse_trtexec_extracts_gpu_compute_stats():
    got = parse_trtexec(SAMPLE)
    assert got["lat_mean_ms"] == 8.16
    assert got["lat_p50_ms"] == 8.13
    assert got["lat_p90_ms"] == 8.31
    assert got["lat_p99_ms"] == 8.95
    assert got["throughput_qps"] == 121.478


def test_parse_trtexec_missing_fields_are_nan():
    got = parse_trtexec("[I] nothing useful here")
    assert all(math.isnan(v) for v in got.values())