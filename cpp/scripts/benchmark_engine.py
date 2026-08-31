#!/usr/bin/env python3
"""Run a TensorRT plan in Python for FPS/E2E comparison with rtdetr-infer.

Use the same engine, model contract, source, frame count, and warmup as the C++
CLI with --no-draw. Engine time includes Python TensorRT input/output transfers.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from polygraphy.backend.trt import TrtRunner, engine_from_bytes


def prepare_inputs(frame: np.ndarray, spec: dict, metadata) -> dict[str, np.ndarray]:
    height, width = frame.shape[:2]
    target = (spec['width'], spec['height'])
    mode = spec.get('resize_mode', 'pillow_bilinear')
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if mode == 'pillow_bilinear':
        resized = np.asarray(Image.fromarray(rgb).resize(target, Image.Resampling.BILINEAR))
    else:
        interpolation = {
            'linear': cv2.INTER_LINEAR,
            'cubic': cv2.INTER_CUBIC,
            'area_down_linear_up': cv2.INTER_AREA if width > target[0] or height > target[1]
                                   else cv2.INTER_LINEAR,
        }.get(mode)
        if interpolation is None:
            raise ValueError(f'unsupported resize_mode: {mode}')
        resized = cv2.resize(rgb, target, interpolation=interpolation)

    image_name = spec['image_input']
    dtype = metadata[image_name].dtype.numpy()
    if dtype not in (np.dtype('float32'), np.dtype('float16')):
        raise ValueError(f'image I/O dtype {dtype} is unsupported by the C++ preprocessor')
    image = np.ascontiguousarray(resized.transpose(2, 0, 1)[None].astype(dtype) / 255)
    inputs = {image_name: image}
    values = {
        'original_wh': (width, height),
        'original_hw': (height, width),
        'input_hw': (spec['height'], spec['width']),
        'resize_scale_yx': (spec['height'] / height, spec['width'] / width),
        'unit_scale': (1, 1),
    }
    for aux in spec.get('auxiliary_inputs', []):
        name = aux['name']
        inputs[name] = np.asarray([values[aux['value']]], dtype=metadata[name].dtype.numpy())
    if set(inputs) != set(metadata):
        raise ValueError(f'contract inputs {sorted(inputs)} differ from engine inputs {sorted(metadata)}')
    return inputs


def decode(outputs: dict, spec: dict, width: int, height: int, threshold: float) -> list:
    kind = spec['output_kind']
    detections = []
    if kind == 'raw_queries':
        logits = np.asarray(outputs[spec['logits_output']], dtype=np.float32).reshape(-1, spec['classes'])
        boxes = np.asarray(outputs[spec['boxes_output']], dtype=np.float32).reshape(-1, 4)
        scores = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
        order = np.argsort(-scores.ravel(), kind='stable')[:spec.get('top_k', 300)]
        for index in order:
            score = float(scores.ravel()[index])
            if score < threshold:
                continue
            query, label = divmod(int(index), spec['classes'])
            cx, cy, bw, bh = boxes[query]
            detections.append((label, score,
                               ((cx - bw / 2) * width, (cy - bh / 2) * height,
                                (cx + bw / 2) * width, (cy + bh / 2) * height)))
    elif kind == 'decoded':
        labels = np.asarray(outputs[spec['labels_output']]).reshape(-1)
        boxes = np.asarray(outputs[spec['boxes_output']], dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(outputs[spec['scores_output']], dtype=np.float32).reshape(-1)
        if len(labels) != len(boxes) or len(labels) != len(scores):
            raise ValueError('decoded output shapes mismatch')
        sx, sy = (width, height) if spec.get('boxes_normalized', False) else (1, 1)
        for label, score, box in zip(labels, scores, boxes):
            if score >= threshold:
                detections.append((int(label), float(score),
                                   (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy)))
    elif kind == 'paddle_rows':
        rows = np.asarray(outputs[spec['boxes_output']], dtype=np.float32).reshape(-1, 6)
        count = max(0, int(np.asarray(outputs[spec['count_output']]).reshape(-1)[0]))
        sx, sy = ((width / spec['width'], height / spec['height'])
                  if spec.get('paddle_boxes_input_pixels', False) else (1, 1))
        for label, score, x0, y0, x1, y1 in rows[:count]:
            if score >= threshold:
                detections.append((int(label), float(score),
                                   (x0 * sx, y0 * sy, x1 * sx, y1 * sy)))
    else:
        raise ValueError(f'unsupported output_kind: {kind}')
    detections.sort(key=lambda item: item[1], reverse=True)
    return detections


def percentile(values: list[float], level: float) -> float:
    return float(np.percentile(values, level))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path, help='runtime tensor contract JSON')
    parser.add_argument('--engine', required=True, type=Path)
    parser.add_argument('--source', required=True, help='video path, stream URL, or camera index')
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--score-thr', type=float, default=0.6)
    args = parser.parse_args()
    if args.max_frames < 0 or args.warmup < 0 or not 0 <= args.score_thr <= 1:
        parser.error('invalid frame count, warmup, or threshold')
    spec = json.loads(args.model.read_text())
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        parser.error(f'cannot open source: {args.source}')
    engine_ms, e2e_ms = [], []
    detections = frames = 0
    try:
        with TrtRunner(engine_from_bytes(args.engine.read_bytes())) as runner:
            metadata = runner.get_input_metadata(use_numpy_dtypes=False)
            black = np.zeros((spec['height'], spec['width'], 3), dtype=np.uint8)
            warm_inputs = prepare_inputs(black, spec, metadata)
            for _ in range(args.warmup):
                decode(runner.infer(warm_inputs), spec, black.shape[1], black.shape[0], 1.0)
            wall_start = time.perf_counter()
            while not args.max_frames or frames < args.max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                after_read = time.perf_counter()
                inputs = prepare_inputs(frame, spec, metadata)
                before_engine = time.perf_counter()
                output = runner.infer(inputs)
                after_engine = time.perf_counter()
                found = decode(output, spec, frame.shape[1], frame.shape[0], args.score_thr)
                done = time.perf_counter()
                engine_ms.append((after_engine - before_engine) * 1000)
                e2e_ms.append((done - after_read) * 1000)
                detections += len(found)
                frames += 1
            wall_seconds = time.perf_counter() - wall_start
    finally:
        cap.release()
    if not frames:
        parser.error('no frames decoded from source')
    print(f'model={spec["name"]} version={spec["version"]} frames={frames} '
          f'engine_p50_ms={percentile(engine_ms, 50):.2f} '
          f'engine_p90_ms={percentile(engine_ms, 90):.2f} '
          f'e2e_p50_ms={percentile(e2e_ms, 50):.2f} '
          f'pipeline_fps={frames / wall_seconds:.2f} '
          f'detections_per_frame={detections / frames:.2f}')


if __name__ == '__main__':
    main()
