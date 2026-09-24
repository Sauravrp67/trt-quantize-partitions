from __future__ import annotations

import argparse

if __package__:                    # python -m models.rtdetr.export
    from . import _bootstrap       # noqa: F401  puts repo root + this dir on sys.path
else:                              # python host/models/rtdetr/export.py
    import _bootstrap              # noqa: F401

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from harness.adapter.rtdetr import build_config  # noqa: E402
from harness.config import load_spec  # noqa: E402


class RTDETRExportModel(nn.Module):
    """Deploy-mode wrapper: the export graph is exactly what inference runs."""

    def __init__(self, config) -> None:
        super().__init__()
        self.model = config.model.deploy()

    def forward(self, images):
        return self.model(images)


def export(spec, out_path, *, report: bool = True) -> None:
    config = build_config(spec)
    model = RTDETRExportModel(config)
    data = torch.rand(spec.batch, 3, spec.img_size, spec.img_size)
    _ = model(data)   # trace-warm the deploy path before exporting

    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts = spec.outputs.get("export_artifacts")
    onnx_program = torch.onnx.export(
        model,
        (data,),
        input_names=[spec.input_name],
        output_names=list(spec.output_names),
        dynamo=True,
        verbose=False,
        verify=True,
        profile=report,
        keep_initializers_as_inputs=False,
        report=report,
        dump_exported_program=report,
        artifacts_dir=str(artifacts) if artifacts else None,
    )
    onnx_program.save(str(out_path))
    print(f"[export] {spec.name}: batch={spec.batch} size={spec.img_size} "
          f"in={spec.input_name} out={list(spec.output_names)} -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RT-DETR checkpoint -> static batch=1 FP32 ONNX")
    parser.add_argument("--spec", default="rtdetr", help="model spec name or configs/*.yaml path")
    parser.add_argument("--backbone", default=None,
                        help="entry of the spec's backbones registry (default: its `backbone:` key)")
    parser.add_argument("--out", default=None,
                        help="destination: a variant name from the spec (default: its onnx.default) "
                             "or an explicit path")
    parser.add_argument("--no-report", dest="report", action="store_false", default=True,
                        help="skip the profile/report/exported-program artifacts")
    args = parser.parse_args()

    spec = load_spec(args.spec, args.backbone)
    export(spec, spec.onnx_dest(args.out), report=args.report)


if __name__ == "__main__":
    main()
