from __future__ import annotations
from pathlib import Path
import hashlib
import numpy as np
import tensorrt as trt
import json
from polygraphy.backend.trt import (
    network_from_onnx_path, CreateConfig, engine_bytes_from_network, engine_from_bytes, TrtRunner
)

from .paths import ENGINES_DIR

# One cache per (GPU, TensorRT version). Shared by every build in the study.
DEFAULT_TIMING_CACHE = ENGINES_DIR / "timing.cache"

def build_engine(
    onnx_path,
    *,
    tf32: bool = True,
    opt_level: int = 3,
    timing_cache=DEFAULT_TIMING_CACHE,
) -> bytes:

    builder, network, parser = network_from_onnx_path(str(onnx_path))  # parser must stay alive
    config = builder.create_builder_config()
    (config.set_flag if tf32 else config.clear_flag)(trt.BuilderFlag.TF32)
    config.builder_optimization_level = opt_level
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED  # required by layer_precisions

    cache = None
    if timing_cache is not None:
        cache_path = Path(timing_cache)
        blob = cache_path.read_bytes() if cache_path.exists() else b""
        # ignore_mismatch=False: reject a cache from a different GPU/TRT build rather
        # than silently trusting timings that were never measured on this device.
        cache = config.create_timing_cache(blob)
        config.set_timing_cache(cache, ignore_mismatch=False)

    engine_bytes = bytes(builder.build_serialized_network(network, config))

    if cache is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(bytes(cache.serialize()))

    return engine_bytes

def save_engine(engine_bytes: bytes, path: str) -> None:
    Path(path).parent.mkdir(parents = True, exist_ok = True)
    Path(path).write_bytes(engine_bytes)

def load_engine(path: str) -> bytes:
    return Path(path).read_bytes()

def layer_precisions(engine_bytes: bytes) -> dict[str, set[str]]:
    engine = engine_from_bytes(engine_bytes)
    inspector = engine.create_engine_inspector()
    out: dict[str, set[str]] = {}

    for i in range(engine.num_layers):
        info = json.loads(inspector.get_layer_information(i, trt.LayerInformationFormat.JSON))
        dtypes = {
            t["Datatype"]

            for section in ("Inputs", "Constants", "Outputs")
            for t in info.get(section, [])
            if t.get("Datatype")
        }
        out[info["Name"]] = dtypes
    return out


def tactic_plan(engine_bytes: bytes) -> list[tuple]:

    engine = engine_from_bytes(engine_bytes)
    inspector = engine.create_engine_inspector()
    plan = []
    for i in range(engine.num_layers):
        info = json.loads(inspector.get_layer_information(i, trt.LayerInformationFormat.JSON))
        dtypes = sorted({
            t["Datatype"]
            for section in ("Inputs", "Constants", "Outputs")
            for t in info.get(section, [])
            if t.get("Datatype")
        })
        plan.append((info["Name"], info.get("LayerType"), info.get("TacticName"), tuple(dtypes)))
    return plan


def plan_fingerprint(engine_bytes: bytes) -> str:
    """Short stable hash of tactic_plan(); equal iff two engines made the same decisions."""
    return hashlib.sha256(repr(tactic_plan(engine_bytes)).encode()).hexdigest()[:16]

class TRTSession:
    """Reusable TensorRT runner; Polygraphy's TrtRunner owns device buffers + stream."""

    def __init__(self, engine_bytes: bytes) -> None:
        engine = engine_from_bytes(engine_bytes)
        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        self.input_name = next(n for n in names
                               if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.output_names = [n for n in names
                             if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        # Read the declared input dtype instead of assuming float32: an all-FP16-IO
        # engine would silently receive misinterpreted bytes otherwise.
        self._in_dtype = trt.nptype(engine.get_tensor_dtype(self.input_name))
        self._runner = TrtRunner(engine)
        self._runner.activate()

    def run(self, x_np: np.ndarray) -> dict[str, np.ndarray]:
        x_np = np.ascontiguousarray(x_np, dtype=self._in_dtype)
        out = self._runner.infer({self.input_name: x_np})
        return {k: np.asarray(v) for k, v in out.items()}  # copy out of runner buffers

    def close(self) -> None:
        self._runner.deactivate()