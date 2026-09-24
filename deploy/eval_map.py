#!/usr/bin/env python3
"""Evaluate existing RT-DETR TensorRT plans on COCO; no export or engine rebuild."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

if __package__:
    from .runtime import TRTSession, preprocess, postprocess
else:
    from runtime import TRTSession, preprocess, postprocess

COCO80_TO_CATID = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
    23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
    44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
    63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85,
    86, 87, 88, 89, 90,
)


def detections_to_coco(image_id, labels, boxes, scores):
    """Map contiguous model labels and xyxy boxes to COCO category IDs and xywh."""
    predictions = []
    for label, (x1, y1, x2, y2), score in zip(labels, boxes, scores):
        predictions.append({
            "image_id": int(image_id),
            "category_id": COCO80_TO_CATID[int(label)],
            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            "score": float(score),
        })
    return predictions


def evaluate_engine(engine, coco, image_dir, image_ids, *, session_factory=TRTSession):
    from pycocotools.cocoeval import COCOeval

    session = session_factory(engine)
    predictions = []
    try:
        for index, image_id in enumerate(image_ids, 1):
            info = coco.loadImgs(image_id)[0]
            with Image.open(image_dir / info["file_name"]) as image:
                image = image.convert("RGB")
                outputs = session.run(preprocess(np.asarray(image)))
                logits = outputs["pred_logits"]
                if logits.shape[-1] != len(COCO80_TO_CATID):
                    raise ValueError("COCO evaluation requires the 80-class RT-DETR model")
                # Retain all top-300 predictions: COCOeval applies its own ranking.
                labels, boxes, scores = postprocess(logits, outputs["pred_boxes"], image.size)
            predictions.extend(detections_to_coco(image_id, labels, boxes, scores))
            if index % 100 == 0:
                print(f"[{engine.name}] {index}/{len(image_ids)} images", flush=True)
    finally:
        session.close()

    if not predictions:
        return {"engine": str(engine), "images": len(image_ids), "mAP50_95": 0.0, "mAP50": 0.0}
    evaluator = COCOeval(coco, coco.loadRes(predictions), "bbox")
    evaluator.params.imgIds = image_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {"engine": str(engine), "images": len(image_ids),
            "mAP50_95": float(evaluator.stats[0]), "mAP50": float(evaluator.stats[1])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engines", nargs="+", type=Path, required=True,
                        help="saved .plan/.engine files, baseline first")
    parser.add_argument("--coco", type=Path, required=True,
                        help="root containing val2017/ and annotations/instances_val2017.json")
    parser.add_argument("--limit", type=int, help="evaluate the first N sorted image IDs")
    parser.add_argument("--out", type=Path, default=Path("artifacts/evaluation/coco_map.json"))
    args = parser.parse_args()
    if args.out.suffix.lower() != ".json":
        parser.error("--out must end in .json (the Markdown table is written alongside it)")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    annotation = args.coco / "annotations" / "instances_val2017.json"
    image_dir = args.coco / "val2017"
    for path in [annotation, *args.engines]:
        if not path.is_file():
            parser.error(f"file not found: {path}")
    if not image_dir.is_dir():
        parser.error(f"image directory not found: {image_dir}")

    from pycocotools.coco import COCO
    coco = COCO(str(annotation))
    image_ids = sorted(coco.getImgIds())
    if args.limit:
        image_ids = image_ids[:args.limit]
    if not image_ids:
        parser.error("annotation file contains no images")
    for image_id in image_ids:
        path = image_dir / coco.imgs[image_id]["file_name"]
        if not path.is_file():
            parser.error(f"image not found: {path}")

    rows = [evaluate_engine(engine, coco, image_dir, image_ids) for engine in args.engines]
    report = {"annotations": str(annotation.resolve()), "image_ids": image_ids, "results": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    lines = ["| Engine | Images | mAP@50:95 | mAP@50 | Delta mAP@50:95 |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        delta = row["mAP50_95"] - rows[0]["mAP50_95"]
        lines.append(f"| {Path(row['engine']).name} | {row['images']} | "
                     f"{row['mAP50_95']:.4f} | {row['mAP50']:.4f} | {delta:+.4f} |")
    table = "\n".join(lines) + "\n"
    args.out.with_suffix(".md").write_text(table)
    print(table)
    print(f"Saved {args.out} and {args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
