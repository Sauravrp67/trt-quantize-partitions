"""Measure ONNX activation divergence across precision variants.

The default scan compares FP32, FP16 and INT8 Q/DQ tensor outputs at selected
nodes on the same images. Large activation error is a screening signal, not proof
that keeping that node in FP32 improves detection mAP. Optional one-node FP32
rescues measure whether an FP16 graph's *final raw output* error improves; neither
measure substitutes for COCO mAP or a TensorRT build on the target board.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile

HOST_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = HOST_ROOT.parent
if str(HOST_ROOT) not in sys.path:
    sys.path.insert(0, str(HOST_ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from harness.adapter import load_adapter  # noqa: E402
from harness.config import ModelSpec, load_spec  # noqa: E402
from harness.paths import COCO_VAL_IMAGES  # noqa: E402

DEFAULT_OPS = ("GridSample", "LayerNormalization", "Softmax")


@dataclass
class ErrorStats:
    """Stream paired arrays across images without retaining variant activations."""

    elements: int = 0
    squared_reference: float = 0.0
    squared_error: float = 0.0
    absolute_error: float = 0.0
    maximum_error: float = 0.0

    def add(self, reference: np.ndarray, candidate: np.ndarray, *, name: str) -> None:
        if reference.shape != candidate.shape:
            raise ValueError(f"{name}: FP32 and variant shapes differ: "
                             f"{reference.shape} vs {candidate.shape}")
        a = np.asarray(reference, dtype=np.float32)
        b = np.asarray(candidate, dtype=np.float32)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f"{name}: non-finite activation encountered")
        difference = a.astype(np.float64) - b.astype(np.float64)
        self.elements += a.size
        self.squared_reference += float(np.sum(np.square(a, dtype=np.float64)))
        self.squared_error += float(np.sum(np.square(difference)))
        self.absolute_error += float(np.sum(np.abs(difference)))
        self.maximum_error = max(self.maximum_error, float(np.max(np.abs(difference))))

    def result(self) -> dict:
        return {
            "elements": self.elements,
            "relative_l2": math.sqrt(self.squared_error / max(self.squared_reference, 1e-24)),
            "mean_absolute_error": self.absolute_error / self.elements,
            "max_absolute_error": self.maximum_error,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reference_path(spec: ModelSpec) -> Path:
    """Prefer the graph_surgery output when present; reject stale prepared graphs."""
    raw = spec.onnx_path("fp32")
    prepared = REPO_ROOT / "artifacts" / "graphs" / f"{spec.name}_{spec.backbone}.fp32.onnx"
    if not prepared.exists():
        return raw
    manifest = prepared.with_name(f"{prepared.stem}.graph.json")
    if not manifest.is_file():
        raise ValueError(f"prepared ONNX has no graph_surgery manifest: {manifest}")
    record = json.loads(manifest.read_text())
    if (record.get("model") != spec.name or record.get("backbone") != spec.backbone
            or record.get("source_sha256") != _sha256(raw)
            or record.get("prepared_sha256") != _sha256(prepared)):
        raise ValueError(f"prepared ONNX or source changed since graph_surgery: {manifest}")
    return prepared


def _select_tensors(model, *, ops: tuple[str, ...], node_names: tuple[str, ...],
                    max_nodes: int | None) -> list[dict]:
    if node_names:
        wanted = set(node_names)
        selected = [node for node in model.graph.node if node.name in wanted]
        missing = wanted - {node.name for node in selected}
        if missing:
            raise ValueError(f"unknown node names: {sorted(missing)}")
    else:
        selected = [node for node in model.graph.node if node.op_type in ops]
    if max_nodes is not None:
        selected = selected[:max_nodes]
    if not selected:
        raise ValueError("no nodes selected; choose --ops or --node-names from graph_surgery output")
    if any(not node.name for node in selected):
        raise ValueError("selected nodes need names; run graph_surgery.py first")
    return [{"node": node.name, "op": node.op_type, "tensor": tensor}
            for node in selected for tensor in node.output if tensor]


def _augment(model, tensor_names: list[str]):
    """Expose intermediate tensors as graph outputs using their existing type data."""
    import onnx

    known = {item.name: item for item in [*model.graph.value_info, *model.graph.output]}
    produced = {name for node in model.graph.node for name in node.output}
    for name in tensor_names:
        if name not in produced:
            raise ValueError(f"tensor {name!r} is absent from this precision variant")
        if name not in known:
            raise ValueError(f"tensor {name!r} lacks ONNX type information")
        if name not in {item.name for item in model.graph.output}:
            model.graph.output.append(known[name])
    onnx.checker.check_model(model)
    return model


def _session(path: Path, provider: str, threads: int):
    import onnxruntime as ort

    if provider not in ort.get_available_providers():
        raise RuntimeError(f"{provider} is not available; installed providers: "
                           f"{ort.get_available_providers()}")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = threads
    session = ort.InferenceSession(str(path), sess_options=options, providers=[provider])
    if session.get_providers()[0] != provider:
        raise RuntimeError(f"requested {provider}, but ONNX Runtime activated "
                           f"{session.get_providers()}")
    return session


def _inputs(spec: ModelSpec, image_dir: Path, count: int):
    files = sorted(image_dir.glob("*.jpg"))[:count]
    if not files:
        raise FileNotFoundError(f"no calibration images found in {image_dir}")
    adapter = load_adapter(spec, onnx="fp32", device="cpu")
    samples = []
    for path in files:
        with Image.open(path) as image:
            x, _ = adapter.preprocess(image.convert("RGB"))
        samples.append((path.name, np.ascontiguousarray(x.detach().cpu().numpy())))
    return samples


def _collect(session, samples, output_names: list[str]) -> list[dict]:
    input_names = [item.name for item in session.get_inputs()]
    if len(input_names) != 1:
        raise ValueError(f"expected one ONNX input, found {input_names}")
    result = []
    for _, x in samples:
        values = session.run(output_names, {input_names[0]: x})
        result.append(dict(zip(output_names, values)))
    return result


def _compare(reference: list[dict], session, samples, names: list[str]) -> dict[str, ErrorStats]:
    input_name = session.get_inputs()[0].name
    stats = {name: ErrorStats() for name in names}
    for (_, x), expected in zip(samples, reference):
        values = session.run(names, {input_name: x})
        for name, candidate in zip(names, values):
            stats[name].add(expected[name], candidate, name=name)
    return stats


def _final_error(stats: dict[str, ErrorStats], outputs: list[str]) -> float:
    squared_reference = sum(stats[name].squared_reference for name in outputs)
    squared_error = sum(stats[name].squared_error for name in outputs)
    return math.sqrt(squared_error / max(squared_reference, 1e-24))


def run(spec: ModelSpec, *, variants: tuple[str, ...] = ("fp16", "int8"),
        image_dir: Path = COCO_VAL_IMAGES, limit: int = 4,
        ops: tuple[str, ...] = DEFAULT_OPS, node_names: tuple[str, ...] = (),
        max_nodes: int | None = None, provider: str = "CUDAExecutionProvider",
        threads: int = 2, rescue_top: int = 0, out: Path | None = None) -> dict:
    """Compare matched intermediate tensors on identical preprocessed images."""
    import onnx

    if limit <= 0 or threads <= 0 or (max_nodes is not None and max_nodes <= 0):
        raise ValueError("--limit, --threads, and --max-nodes must be positive")
    if rescue_top < 0:
        raise ValueError("--rescue-top must be nonnegative")
    if not variants or len(set(variants)) != len(variants) or "fp32" in variants:
        raise ValueError("choose distinct non-FP32 precision variants")
    if rescue_top and "fp16" not in variants:
        raise ValueError("--rescue-top requires the fp16 variant")
    reference_path = _reference_path(spec)
    if not reference_path.is_file():
        raise FileNotFoundError(f"FP32 ONNX not found: {reference_path}")
    model = onnx.load(str(reference_path), load_external_data=False)
    if [item.name for item in model.graph.output] != spec.output_names:
        raise ValueError("FP32 ONNX output names differ from the selected model spec")
    selected = _select_tensors(model, ops=ops, node_names=node_names, max_nodes=max_nodes)
    tensor_names = list(dict.fromkeys(item["tensor"] for item in selected))
    names = list(dict.fromkeys([*spec.output_names, *tensor_names]))
    paths = {variant: spec.onnx_path(variant) for variant in variants}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"precision variant not found: {path}")
    samples = _inputs(spec, Path(image_dir), limit)
    out = Path(out) if out is not None else (
        REPO_ROOT / "results" / "tables" / f"{spec.name}_{spec.backbone}_sensitivity.json"
    )
    if out.suffix != ".json":
        raise ValueError(f"output path must end in .json: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="trtqp-sensitivity-") as work:
        work = Path(work)
        augmented = work / "fp32.onnx"
        onnx.save(_augment(onnx.load(str(reference_path)), tensor_names), str(augmented))
        baseline_session = _session(augmented, provider, threads)
        try:
            reference = _collect(baseline_session, samples, names)
        finally:
            del baseline_session

        report = {
            "schema_version": 1,
            "method": "matched_onnx_tensor_output_divergence",
            "model": spec.name,
            "backbone": spec.backbone,
            "provider": provider,
            "reference": {"path": str(reference_path), "sha256": _sha256(reference_path)},
            "images": [name for name, _ in samples],
            "operators": list(ops) if not node_names else [],
            "selected_tensors": len(tensor_names),
            "variants": {},
            "rescue": [],
        }
        for variant, path in paths.items():
            augmented = work / f"{variant}.onnx"
            onnx.save(_augment(onnx.load(str(path)), tensor_names), str(augmented))
            session = _session(augmented, provider, threads)
            try:
                stats = _compare(reference, session, samples, names)
            finally:
                del session
            rows = [
                {**selected_item, **stats[selected_item["tensor"]].result()}
                for selected_item in selected
            ]
            rows.sort(key=lambda row: row["relative_l2"], reverse=True)
            report["variants"][variant] = {
                "path": str(path),
                "sha256": _sha256(path),
                "final_output_relative_l2": _final_error(stats, spec.output_names),
                "final_outputs": {name: stats[name].result() for name in spec.output_names},
                "nodes": rows,
            }
            print(f"[sensitivity] {variant}: {len(rows)} tensors compared on "
                  f"{len(samples)} images; final output relative L2 "
                  f"{report['variants'][variant]['final_output_relative_l2']:.6f}", flush=True)

        if rescue_top:
            from harness.precision import to_fp16

            # The rescue and its baseline must come from the same converter.
            # An existing FP16 artifact might have different conversion settings.
            rescue_baseline_path = work / "rescue_baseline.fp16.onnx"
            to_fp16(str(reference_path), rescue_baseline_path)
            session = _session(rescue_baseline_path, provider, threads)
            try:
                baseline_stats = _compare(reference, session, samples, list(spec.output_names))
            finally:
                del session
            baseline_error = _final_error(baseline_stats, spec.output_names)
            report["rescue_baseline_final_output_relative_l2"] = baseline_error
            unique_nodes = list(dict.fromkeys(
                row["node"] for row in report["variants"]["fp16"]["nodes"
            ]))[:rescue_top]
            for index, node_name in enumerate(unique_nodes):
                rescue_path = work / f"rescue_{index}.onnx"
                try:
                    to_fp16(str(reference_path), rescue_path, node_block_list=[node_name])
                    session = _session(rescue_path, provider, threads)
                    try:
                        stats = _compare(reference, session, samples, list(spec.output_names))
                    finally:
                        del session
                    error = _final_error(stats, spec.output_names)
                    row = {"node": node_name, "final_output_relative_l2": error,
                           "reduction_vs_fp16": baseline_error - error}
                except (RuntimeError, ValueError, OSError) as exc:
                    row = {"node": node_name, "error": f"{type(exc).__name__}: {exc}"}
                report["rescue"].append(row)
                print(f"[sensitivity] FP32 rescue {node_name}: "
                      f"{row.get('reduction_vs_fp16', row.get('error'))}", flush=True)

    out.write_text(json.dumps(report, indent=2) + "\n")
    table = [f"# Activation divergence: {spec.name} / {spec.backbone}", "",
             f"{len(samples)} images; ONNX Runtime `{provider}`. Relative L2 is measured "
             "against the FP32 tensor of the same name.", "",
             "Higher divergence identifies tensors to inspect; it does not establish "
             "that retaining the named node in FP32 improves mAP. Validate any proposed "
             "partition with COCO evaluation and a TensorRT build on the target board.", ""]
    for variant, data in report["variants"].items():
        table.extend([f"## {variant.upper()}", "",
                      f"Final raw output relative L2: {data['final_output_relative_l2']:.6f}", "",
                      "| Node | Op | Tensor | Relative L2 | Mean abs. error | Max abs. error |",
                      "| --- | --- | --- | ---: | ---: | ---: |"])
        for row in data["nodes"]:
            table.append(f"| `{row['node']}` | {row['op']} | `{row['tensor']}` | "
                         f"{row['relative_l2']:.6g} | {row['mean_absolute_error']:.6g} | "
                         f"{row['max_absolute_error']:.6g} |")
        table.append("")
    if report["rescue"]:
        table.extend(["## One-node FP32 rescues from FP16", "",
                      "Positive reduction means lower final raw-output error than FP16 "
                      "generated with the same converter. "
                      "A rescue may still fail a TensorRT build or hurt detection mAP.", "",
                      "Rescue baseline final relative L2: "
                      f"{report['rescue_baseline_final_output_relative_l2']:.6f}", "",
                      "| Node held FP32 | Final relative L2 | Reduction vs FP16 |",
                      "| --- | ---: | ---: |"])
        for row in report["rescue"]:
            if "error" in row:
                table.append(f"| `{row['node']}` | error | {row['error']} |")
            else:
                table.append(f"| `{row['node']}` | {row['final_output_relative_l2']:.6g} | "
                             f"{row['reduction_vs_fp16']:+.6g} |")
        table.append("")
    markdown = out.with_suffix(".md")
    markdown.write_text("\n".join(table) + "\n")
    print(f"[sensitivity] wrote {out} and {markdown}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen ONNX precision sensitivity by activation")
    parser.add_argument("--spec", default="rtdetr")
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--variants", nargs="+", default=["fp16", "int8"])
    parser.add_argument("--images-dir", type=Path, default=COCO_VAL_IMAGES)
    parser.add_argument("--limit", type=int, default=4, help="number of sorted images (default: 4)")
    parser.add_argument("--ops", nargs="+", default=list(DEFAULT_OPS))
    parser.add_argument("--node-names", nargs="+", default=[])
    parser.add_argument("--max-nodes", type=int, help="limit selected nodes in graph order")
    parser.add_argument("--provider", default="CUDAExecutionProvider",
                        help="ONNX Runtime provider; default CUDAExecutionProvider")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--rescue-top", type=int, default=0,
                        help="try holding the N highest-divergence FP16 nodes in FP32")
    parser.add_argument("--out", type=Path, help="JSON path; a Markdown table is written alongside")
    args = parser.parse_args()
    run(load_spec(args.spec, args.backbone), variants=tuple(args.variants),
        image_dir=args.images_dir, limit=args.limit, ops=tuple(args.ops),
        node_names=tuple(args.node_names), max_nodes=args.max_nodes,
        provider=args.provider, threads=args.threads, rescue_top=args.rescue_top,
        out=args.out)


if __name__ == "__main__":
    main()
