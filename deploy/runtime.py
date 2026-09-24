"""RT-DETR decoding and saved-engine execution shared by inference and COCO evaluation.

No PyTorch, ONNX Runtime, host harness, or upstream RT-DETR imports.
TensorRT and Polygraphy are loaded only when opening an engine.
"""
from pathlib import Path

import numpy as np
from PIL import Image

IMG_SIZE = 640
NUM_TOP_QUERIES = 300


class TRTSession:
    def __init__(self, plan_path: Path) -> None:
        import tensorrt as trt
        from polygraphy.backend.trt import TrtRunner, engine_from_bytes

        engine = engine_from_bytes(plan_path.read_bytes())
        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        self.input_name = next(n for n in names
                               if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.output_names = [n for n in names
                             if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        if self.input_name != "images" or set(self.output_names) != {"pred_logits", "pred_boxes"}:
            raise ValueError("Expected an RT-DETR engine with images, pred_logits, pred_boxes tensors")
        if tuple(engine.get_tensor_shape(self.input_name)) != (1, 3, IMG_SIZE, IMG_SIZE):
            raise ValueError("Expected a static RT-DETR input shape of (1, 3, 640, 640)")
        self._in_dtype = trt.nptype(engine.get_tensor_dtype(self.input_name))
        self._runner = TrtRunner(engine)
        self._runner.activate()

    def run(self, x: np.ndarray) -> dict:
        x = np.ascontiguousarray(x, dtype=self._in_dtype)
        return {k: np.asarray(v) for k, v in self._runner.infer({self.input_name: x}).items()}

    def close(self) -> None:
        self._runner.deactivate()


def preprocess(frame_rgb: np.ndarray) -> np.ndarray:
    """HWC uint8 RGB -> (1, 3, 640, 640) float32 in [0, 1].

    PIL, not cv2.resize: PIL's filters are antialiased and INTER_LINEAR is not, which
    changes small-object scores when a 1080p frame is downscaled to 640. Must stay in
    step with RTDETRAdapter.preprocess or Jetson and desktop runs are not comparable.
    """
    resized = Image.fromarray(frame_rgb).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    x = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return np.ascontiguousarray(x[None])


def _sigmoid(z: np.ndarray) -> np.ndarray:
    exp = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + exp), exp / (1.0 + exp))


def postprocess(logits: np.ndarray, boxes: np.ndarray, orig_wh: tuple[int, int]):
    """Engine outputs -> (labels, xyxy pixels, scores), score-sorted.

    numpy port of RTDETRPostProcessor.forward in deploy mode (focal loss).
    """
    w, h = orig_wh
    scores = _sigmoid(logits[0].astype(np.float32))
    num_classes = scores.shape[-1]
    flat = scores.reshape(-1)

    k = min(NUM_TOP_QUERIES, flat.size)
    part = np.argpartition(-flat, k - 1)[:k]
    index = part[np.argsort(-flat[part])]
    labels = (index % num_classes).astype(np.int32)
    queries = index // num_classes

    cx, cy, bw, bh = np.split(boxes[0].astype(np.float32)[queries], 4, axis=-1)
    xyxy = np.concatenate([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=-1)
    xyxy *= np.array([w, h, w, h], dtype=np.float32)
    return labels, xyxy, flat[index]


