from __future__ import annotations
from pathlib import Path
import numpy as np
import tensorrt as trt
import json
from polygraphy.backend.trt import (
    network_from_onnx_path, CreateConfig, engine_bytes_from_network, engine_from_bytes, TrtRunner
)

def build_engine(onnx_path, *, tf32: bool = True, opt_level: int = 3) -> bytes:
    builder, network, parser = network_from_onnx_path(str(onnx_path))
    config = builder.create_builder_config()
    (config.set_flag if tf32 else config.clear_flag)(trt.BuilderFlag.TF32)
    config.builder_optimization_level = opt_level
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    return bytes(builder.build_serialized_network(network,config))

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