"""Metrics, latency, and power-sampling harness.

TODO:
    - mAP evaluation adapters.
    - Throughput / perf-per-watt aggregation.
    - RTX 4050 power-cap sampling.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List

import torch


@dataclass
class Aggregate:
    """Collect per-call timings (milliseconds) and summarize.

    The first ``warmup`` samples are excluded from every statistic so cold-start
    kernel compilation / cuDNN autotuning does not skew the numbers.
    """

    warmup: int = 0
    samples: List[float] = field(default_factory=list)

    def add(self, ms: float) -> None:
        self.samples.append(float(ms))

    def _kept(self) -> List[float]:
        # Drop warmup samples once there are enough; otherwise report what we have
        # (e.g. a single-image run would otherwise summarize to 0.0).
        if len(self.samples) > self.warmup:
            return self.samples[self.warmup:]
        return list(self.samples)

    @property
    def count(self) -> int:
        return len(self._kept())

    def mean(self) -> float:
        kept = self._kept()
        return sum(kept) / len(kept) if kept else 0.0

    def median(self) -> float:
        kept = sorted(self._kept())
        if not kept:
            return 0.0
        n = len(kept)
        mid = n // 2
        return kept[mid] if n % 2 else (kept[mid - 1] + kept[mid]) / 2.0

    def p90(self) -> float:
        kept = sorted(self._kept())
        if not kept:
            return 0.0
        idx = min(len(kept) - 1, int(round(0.9 * (len(kept) - 1))))
        return kept[idx]

    def fps(self) -> float:
        m = self.mean()
        return 1000.0 / m if m > 0 else 0.0


class LatencyMeter:
    """Time the torch forward and the ONNX Runtime call per frame.

    ``torch_timer()`` uses ``torch.cuda.Event`` (+ ``synchronize``) on CUDA for
    accurate device timing, else ``time.perf_counter``; ``ort_timer()`` always uses
    ``perf_counter`` (ORT returns host numpy, so wall time is the honest measure).
    """

    def __init__(self, device: str = "cuda", warmup: int = 1) -> None:
        self.device = device
        self.torch_ms = Aggregate(warmup=warmup)
        self.ort_ms = Aggregate(warmup=warmup)
        self._last = {"torch_ms": None, "ort_ms": None}

    @property
    def _use_cuda(self) -> bool:
        return self.device.startswith("cuda") and torch.cuda.is_available()

    @contextmanager
    def torch_timer(self):
        if self._use_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            yield
            end.record()
            torch.cuda.synchronize()
            ms = start.elapsed_time(end)
        else:
            t0 = time.perf_counter()
            yield
            ms = (time.perf_counter() - t0) * 1000.0
        self.torch_ms.add(ms)
        self._last["torch_ms"] = ms

    @contextmanager
    def ort_timer(self):
        t0 = time.perf_counter()
        yield
        ms = (time.perf_counter() - t0) * 1000.0
        self.ort_ms.add(ms)
        self._last["ort_ms"] = ms

    def last(self) -> dict:
        """Most recent (torch_ms, ort_ms) pair for per-frame reporting."""
        return dict(self._last)


class StreamTimer:
    """Backend-neutral single-stream wall-clock timer for a one-backend inference loop.

    ``time()`` wraps a call with ``time.perf_counter``; the measured milliseconds feed an
    :class:`Aggregate` (warmup-skipping) and are cached in ``.last_ms``. Use for the TRT
    inference viewer, where the runner (Polygraphy ``TrtRunner``) synchronizes internally,
    so wall time is the honest *end-to-end* per-frame latency (H2D + compute + D2H).

    Note: this is a *display* measure. The authoritative latency/throughput benchmark is
    ``trtexec`` (CUDA-graph, pure-GPU-compute, percentiles) — see ``harness/trtexec.py``.
    Do not quote these numbers as the benchmark.
    """

    def __init__(self, warmup: int = 1) -> None:
        self.agg = Aggregate(warmup=warmup)
        self.last_ms: float | None = None

    @contextmanager
    def time(self):
        t0 = time.perf_counter()
        yield
        ms = (time.perf_counter() - t0) * 1000.0
        self.agg.add(ms)
        self.last_ms = ms
