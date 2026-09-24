from __future__ import annotations
import numpy as np
from PIL import Image
from harness.paths import COCO_VAL_IMAGES

def calib_images(adapter, n: int = 200) -> list[np.ndarray]:
    files = sorted(COCO_VAL_IMAGES.glob("*.jpg"))[:n]
    out = []
    for f in files:
        x, _ = adapter.preprocess(Image.open(f).convert("RGB"))
        out.append(np.ascontiguousarray(x.detach().cpu().numpy(), dtype=np.float32))
    return out