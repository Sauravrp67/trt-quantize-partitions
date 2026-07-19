from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
RTDETR_PYTORCH_ROOT = REPO_ROOT / "RT-DETR" / "rtdetr_pytorch"
RTDETR_SRC_ROOT = RTDETR_PYTORCH_ROOT / "src"
if not RTDETR_SRC_ROOT.is_dir():
    raise FileNotFoundError(f"RT-DETR PyTorch checkout not found: {RTDETR_PYTORCH_ROOT}")

sys.path.insert(0, str(REPO_ROOT))            # so `import harness` resolves
sys.path.insert(0, str(RTDETR_PYTORCH_ROOT))  # so `import src` resolves

src_pkg = types.ModuleType("src")
src_pkg.__file__ = str(RTDETR_SRC_ROOT / "__init__.py")
src_pkg.__path__ = [str(RTDETR_SRC_ROOT)]
sys.modules["src"] = src_pkg

from src.core import YAMLConfig  # noqa: E402
import src.nn  
import src.zoo 

from harness import run, run_inference, verify_parity  # noqa: E402
from harness.compare import Detections  # noqa: E402

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

DEFAULT_CONFIG = RTDETR_PYTORCH_ROOT / "configs/rtdetr/rtdetr_r18vd_6x_coco.yml"
DEFAULT_CKPT = REPO_ROOT / "models/rtdetr/checkpoints/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth"
DEFAULT_ONNX = REPO_ROOT / "models/rtdetr/model.onnx"
IMG_SIZE = 640

class RTDETRAdapter:
    """DetectorAdapter for RT-DETR r18vd. See harness.compare.DetectorAdapter."""

    name = "rtdetr"
    class_names = COCO_CLASSES
    input_name = "images"
    output_names = ["pred_logits", "pred_boxes"]

    def __init__(self, config_path, ckpt_path, onnx_path, device: str = "cuda") -> None:
        self.onnx_path = Path(onnx_path)
        self.device = device

        self._config = YAMLConfig(str(config_path), resume=str(ckpt_path))
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        state = ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]
        self._config.model.load_state_dict(state)

        self._postprocessor = self._config.postprocessor.deploy().to(device).eval()

    def build_torch(self) -> torch.nn.Module:
        return self._config.model.deploy().to(self.device).eval()

    def preprocess(self, frame: Image.Image):
        """RGB PIL -> ((1,3,640,640) float32 CPU tensor, meta). No mean/std normalization."""
        w, h = frame.size
        resized = frame.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        x = (
            torch.from_numpy(np.asarray(resized, dtype=np.float32))
            .permute(2, 0, 1)   # HWC -> CHW
            .div_(255.0)        # [0, 255] -> [0, 1]
            .unsqueeze(0)       # -> (1, 3, 640, 640)
            .contiguous()
        )
        return x, {"orig_size": (w, h)}

    def postprocess(self, named, meta) -> Detections:
        """Raw named outputs (numpy) -> pixel-space Detections via the shared postprocessor."""
        w, h = meta["orig_size"]
        logits = torch.from_numpy(np.asarray(named["pred_logits"])).to(self.device)
        boxes = torch.from_numpy(np.asarray(named["pred_boxes"])).to(self.device)
        orig_size = torch.tensor([[w, h]], device=self.device)
        with torch.no_grad():
            labels, out_boxes, scores = self._postprocessor(
                {"pred_logits": logits, "pred_boxes": boxes}, orig_size
            )
        return Detections(
            labels[0].detach().cpu().numpy(),
            out_boxes[0].detach().cpu().numpy(),
            scores[0].detach().cpu().numpy(),
        )


def _source_label(source: str) -> str:
    if source.isdigit():
        return f"camera{source}"
    p = Path(source)
    if p.is_dir():
        return p.name or "folder"
    return p.stem


def main() -> None:
    parser = argparse.ArgumentParser(description="RT-DETR torch vs ONNX Runtime side-by-side inference")
    parser.add_argument("--source", required=True,
                        help="image file | folder of images | video file | camera index (e.g. 0)")
    parser.add_argument("--backend", choices=["torch-ort", "trt"], default="torch-ort",
                        help="torch-ort: eager-vs-ONNX parity (default); trt: single-backend TensorRT")
    parser.add_argument("--engine", default=None,
                        help="[trt] prebuilt .engine to load; if omitted, build from --onnx")
    parser.add_argument("--tf32", dest="tf32", action="store_true", default=True,
                        help="[trt] allow TF32 tensor cores when building (default on)")
    parser.add_argument("--no-tf32", dest="tf32", action="store_false",
                        help="[trt] strict IEEE FP32 build")
    parser.add_argument("--onnx", default=str(DEFAULT_ONNX))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-thr", type=float, default=0.6)
    parser.add_argument("--iou-thr", type=float, default=0.5, help="IoU for torch/ORT agreement matching")
    parser.add_argument("--out", default=str(REPO_ROOT / "results/figures/parity/rtdetr"))
    parser.add_argument("--show", action="store_true", help="live cv2 window (auto-disabled if headless)")
    parser.add_argument("--save", action="store_true", help="write an annotated output video (video/camera)")
    parser.add_argument("--providers", default="CUDAExecutionProvider,CPUExecutionProvider")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--check", action="store_true",
                        help="run harness.verify_parity on a synthetic input before iterating")
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[infer] CUDA unavailable; falling back to CPU")
        device = "cpu"

    adapter = RTDETRAdapter(args.config, args.ckpt, args.onnx, device=device)

    if args.backend == "trt":
        if args.check:
            print("[infer] --check (verify_parity) applies only to --backend torch-ort; ignoring.")
        tf32 = getattr(args, "tf32", True)
        run_inference(
            adapter,
            args.source,
            engine=args.engine,
            onnx_path=args.onnx,
            tf32=tf32,
            score_thr=args.score_thr,
            show=args.show,
            save=args.save,
            out_dir=args.out,
            warmup=args.warmup,
            max_frames=args.max_frames,
            source_label=_source_label(args.source),
            backend_label=f"TRT · {'TF32' if tf32 else 'FP32-strict'}",
        )
        return

    providers = [p.strip() for p in args.providers.split(",") if p.strip()]

    if args.check:
        import onnxruntime as ort

        print("[infer] verify_parity (synthetic input, RAW pred_logits/pred_boxes):")
        session = ort.InferenceSession(str(adapter.onnx_path), providers=providers)
        verify_parity(session, adapter.build_torch())
        print(
            "[infer] NOTE: raw-output parity is EXPECTED to FAIL for RT-DETR. Its decoder\n"
            "        selects the top-300 queries with a topk whose ties break differently in\n"
            "        torch vs ORT, so a few sub-threshold background queries diverge. The\n"
            "        authoritative signal is the per-detection summary below (above threshold).\n"
        )

    run(
        adapter,
        args.source,
        score_thr=args.score_thr,
        iou_thr=args.iou_thr,
        providers=providers,
        show=args.show,
        save=args.save,
        out_dir=args.out,
        warmup=args.warmup,
        max_frames=args.max_frames,
        source_label=_source_label(args.source),
    )


if __name__ == "__main__":
    main()
