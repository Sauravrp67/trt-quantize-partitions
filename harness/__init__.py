from .adapter import Detections, DetectorAdapter
from .compare import build_backend, match_detections, run, run_backends
from .config import ModelSpec, load_spec
from .infer_engine import TrtBackend, run_inference
from .infer_ort import OrtBackend
from .infer_torch import TorchBackend
from .metrics import Aggregate, LatencyMeter, StreamTimer
from .parity import verify_parity
from .paths import *
from .power import PowerSampler, PowerStats, summarize_power
from .trt_runner import TRTSession, build_engine, load_engine, save_engine

__all__ = [
    # detections / adapter seam
    "Detections",
    "DetectorAdapter",
    # config (static YAML spec) + paths (machine-dependent roots, re-exported above)
    "ModelSpec",
    "load_spec",
    # backends
    "TorchBackend",
    "OrtBackend",
    "TrtBackend",
    "build_backend",
    # drivers
    "run",            # build named backends + run (separate or compare)
    "run_backends",   # run pre-built backends
    "run_inference",  # single TensorRT backend convenience
    "match_detections",
    # timing / eval / build
    "LatencyMeter",
    "StreamTimer",
    "Aggregate",
    "PowerSampler",
    "PowerStats",
    "summarize_power",
    "TRTSession",
    "build_engine",
    "load_engine",
    "save_engine",
    "verify_parity",
]
