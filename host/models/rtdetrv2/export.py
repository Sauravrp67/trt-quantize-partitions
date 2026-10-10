"""Export official RT-DETRv2 PyTorch checkpoints as raw-output ONNX."""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

HOST_ROOT = Path(__file__).resolve().parents[2]
if str(HOST_ROOT) not in sys.path:
    sys.path.insert(0, str(HOST_ROOT))

from harness.config import load_spec  # noqa: E402
from harness.paths import RTDETRV2_PYTORCH_ROOT, require  # noqa: E402
from models._export_common import export_raw, load_checkpoint  # noqa: E402


def _install_upstream() -> None:
    source = require(RTDETRV2_PYTORCH_ROOT / "src", "RT-DETRv2 PyTorch source",
                     "run git submodule update --init third_party/RT-DETR")
    existing = sys.modules.get("src")
    if existing is not None and str(source) not in getattr(existing, "__path__", []):
        raise RuntimeError("another RT-DETR 'src' package is loaded; export v2 in a fresh Python process")
    if existing is None:
        sys.path.insert(0, str(RTDETRV2_PYTORCH_ROOT))
        package = types.ModuleType("src")
        package.__file__ = str(source / "__init__.py")
        package.__path__ = [str(source)]
        sys.modules["src"] = package


def build_model(spec):
    if spec.name != "rtdetrv2":
        raise ValueError(f"RT-DETRv2 exporter received {spec.name!r}")
    _install_upstream()
    from src.core import YAMLConfig
    import src.nn  # noqa: F401  register modules
    import src.zoo  # noqa: F401  register models

    require(spec.torch_config, "RT-DETRv2 config")
    cfg = YAMLConfig(str(spec.torch_config))
    # A complete detector checkpoint supplies the backbone weights too.
    for backbone in ("PResNet", "HGNetv2"):
        if backbone in cfg.yaml_cfg:
            cfg.yaml_cfg[backbone]["pretrained"] = False
    model = cfg.model
    load_checkpoint(model, spec.ckpt)
    return model


def export(spec, out_path: Path, *, report: bool = True) -> None:
    model = build_model(spec)
    export_raw(model, spec, out_path)
    print(f"[rtdetrv2] exported {spec.backbone} -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export RT-DETRv2 to static raw-output ONNX")
    ap.add_argument("--backbone", default=None, choices=load_spec("rtdetrv2").backbones)
    ap.add_argument("--out", default=None, help="registered ONNX variant or output path")
    args = ap.parse_args()
    spec = load_spec("rtdetrv2", args.backbone)
    export(spec, spec.onnx_dest(args.out))


if __name__ == "__main__":
    main()
