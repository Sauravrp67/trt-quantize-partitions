from __future__ import annotations
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image
from harness.compare import Detections
from harness.paths import COCO_VAL_IMAGES, COCO_VAL_ANN

COCO80_TO_CATID = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,
    27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,
    56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90]

def detections_to_coco(image_id: int, det: Detections) -> list[dict]:
    out = []
    for lab, box, sc in zip(det.labels, det.boxes, det.scores):
        x1, y1, x2, y2 = (float(v) for v in box)
        out.append({
            "image_id": int(image_id),
            "category_id": int(COCO80_TO_CATID[int(lab)]),
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "score": float(sc),
        })
    return out


def evaluate(run_fn, adapter, *, limit=None) -> dict:
    coco = COCO(str(COCO_VAL_ANN))
    img_ids = coco.getImgIds()
    if limit:
        img_ids = img_ids[:limit]
    results = []
    for image_id in img_ids:
        info = coco.loadImgs(image_id)[0]
        pil = Image.open(COCO_VAL_IMAGES / info["file_name"]).convert("RGB")
        x, meta = adapter.preprocess(pil)
        named = run_fn(x.detach().cpu().numpy())
        det = adapter.postprocess(named, meta)  # all queries; COCOeval handles thresholds
        results.extend(detections_to_coco(image_id, det))
    if not results:
        return {"mAP50": 0.0, "mAP5095": 0.0, "n_images": len(img_ids)}
    coco_dt = coco.loadRes(results)
    ev = COCOeval(coco, coco_dt, "bbox")
    ev.params.imgIds = img_ids
    ev.evaluate(); ev.accumulate(); ev.summarize()
    return {"mAP5095": float(ev.stats[0]), "mAP50": float(ev.stats[1]), "n_images": len(img_ids)}