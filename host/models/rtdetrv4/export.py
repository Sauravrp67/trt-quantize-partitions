"""Export official RT-DETRv4 PyTorch checkpoints as raw-output ONNX."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HOST_ROOT = Path(__file__).resolve().parents[2]
if str(HOST_ROOT) not in sys.path:
    sys.path.insert(0, str(HOST_ROOT))

from harness.config import load_spec  # noqa: E402
from harness.paths import RTDETRV4_ROOT, require  # noqa: E402
from models._export_common import export_raw, load_checkpoint  # noqa: E402


def build_model(spec):
    if spec.name != "rtdetrv4":
        raise ValueError(f"RT-DETRv4 exporter received {spec.name!r}")
    require(RTDETRV4_ROOT / "engine", "RT-DETRv4 source",
            "run git submodule update --init third_party/RT-DETRv4")
    require(spec.torch_config, "RT-DETRv4 config")
    sys.path.insert(0, str(RTDETRV4_ROOT))
    import engine  # noqa: F401,E402  register the official model components
    from engine.core import YAMLConfig  # noqa: E402

    cfg = YAMLConfig(str(spec.torch_config))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model
    load_checkpoint(model, spec.ckpt)
    return model


def export(spec, out_path: Path, *, report: bool = True) -> None:
    model = build_model(spec)
    export_raw(model, spec, out_path)
    print(f"[rtdetrv4] exported {spec.backbone} -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export RT-DETRv4 to static raw-output ONNX")
    ap.add_argument("--backbone", default=None, choices=load_spec("rtdetrv4").backbones)
    ap.add_argument("--out", default=None, help="registered ONNX variant or output path")
    args = ap.parse_args()
    spec = load_spec("rtdetrv4", args.backbone)
    export(spec, spec.onnx_dest(args.out))


if __name__ == "__main__":
    main()
