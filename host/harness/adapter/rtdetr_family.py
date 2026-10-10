"""Raw-output RT-DETRv2/v3/v4 adapters for host ONNX evaluation and calibration."""
from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image

from .base import BaseAdapter, Detections


class RawRTDETRAdapter(BaseAdapter):
    """Decode 300 raw queries without applying a demo score threshold."""

    def _resize(self, frame: Image.Image) -> np.ndarray:
        return np.asarray(frame.resize((self.img_size, self.img_size), Image.BILINEAR))

    def preprocess(self, frame: Image.Image):
        w, h = frame.size
        resized = self._resize(frame)
        x = np.ascontiguousarray(
            resized.astype(np.float32).transpose(2, 0, 1)[None] / 255.0)
        return torch.from_numpy(x), {"orig_size": (w, h)}

    def postprocess(self, named, meta) -> Detections:
        logits = np.asarray(named["pred_logits"])[0].astype(np.float32)
        boxes = np.asarray(named["pred_boxes"])[0].astype(np.float32)
        if logits.shape != (300, self.spec.num_classes) or boxes.shape != (300, 4):
            raise ValueError(f"unexpected raw shapes: {logits.shape}, {boxes.shape}")
        positive = np.exp(-np.abs(logits))
        scores = np.where(logits >= 0, 1.0 / (1.0 + positive),
                          positive / (1.0 + positive)).reshape(-1)
        index = np.argpartition(-scores, 299)[:300]
        index = index[np.argsort(-scores[index])]
        labels = (index % self.spec.num_classes).astype(np.int32)
        selected = boxes[index // self.spec.num_classes]
        cx, cy, bw, bh = selected.T
        w, h = meta["orig_size"]
        xyxy = np.stack([cx - bw / 2, cy - bh / 2,
                         cx + bw / 2, cy + bh / 2], axis=1)
        xyxy *= np.asarray([w, h, w, h], dtype=np.float32)
        return Detections(labels, xyxy, scores[index])


class RTDETRv2Adapter(RawRTDETRAdapter):
    spec_name = "rtdetrv2"

    def build_torch(self):
        from models._export_common import RawDetector
        from models.rtdetrv2.export import build_model
        return RawDetector(build_model(self.spec)).to(self.device).eval()


class RTDETRv3Adapter(RawRTDETRAdapter):
    spec_name = "rtdetrv3"

    def _resize(self, frame: Image.Image) -> np.ndarray:
        return cv2.resize(np.asarray(frame), (self.img_size, self.img_size),
                          interpolation=cv2.INTER_LINEAR)

    def build_torch(self):
        raise RuntimeError("RT-DETRv3's official checkpoint is Paddle; use ORT or TensorRT")


class RTDETRv4Adapter(RawRTDETRAdapter):
    spec_name = "rtdetrv4"

    def build_torch(self):
        from models._export_common import RawDetector
        from models.rtdetrv4.export import build_model
        return RawDetector(build_model(self.spec)).to(self.device).eval()
