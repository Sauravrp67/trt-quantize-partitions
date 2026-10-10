#!/usr/bin/env python3
"""RT-DETR TensorRT inference on Jetson. No torch, no RT-DETR submodule.

Requires: tensorrt, polygraphy, numpy, opencv, pillow.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

if __package__:
    from .runtime import IMG_SIZE, TRTSession, preprocess, postprocess
else:
    from runtime import IMG_SIZE, TRTSession, preprocess, postprocess
FONT = cv2.FONT_HERSHEY_DUPLEX
STAGES = (
    ("read", "Frame read/decode"),
    ("preprocess", "Preprocessing"),
    ("engine", "Engine inference"),
    ("postprocess", "Postprocessing/filter"),
    ("boxes", "Box drawing"),
    ("hud", "HUD drawing"),
    ("video_write", "Video encode/write"),
    ("display", "Display"),
)


def class_palette(n: int) -> np.ndarray:
    """n visually distinct BGR colors; golden-angle hues so neighbouring ids differ."""
    hues = ((np.arange(n) * 0.6180339887) % 1.0 * 179).astype(np.uint8)
    sat = np.full(n, 225, np.uint8)
    val = np.full(n, 255, np.uint8)
    hsv = np.stack([hues, sat, val], axis=1)[None]
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0]


def _blend(img: np.ndarray, box, color, alpha: float) -> None:
    x0, y0, x1, y1 = box
    roi = img[y0:y1, x0:x1]
    if roi.size:
        cv2.addWeighted(roi, 1.0 - alpha, np.full_like(roi, color), alpha, 0.0, roi)


def _text(img, s, org, scale, color, thick=1) -> None:
    cv2.putText(img, s, (org[0] + 1, org[1] + 1), FONT, scale, (0, 0, 0), thick + 1, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


class Overlay:
    """Detection boxes plus a live HUD, sized relative to the frame."""

    ACCENT = (120, 220, 255)

    def __init__(self, w: int, h: int, names: list[str], engine: str) -> None:
        self.w, self.h = w, h
        self.names = names
        self.engine = engine
        self.colors = class_palette(max(len(names), 1))
        self.s = max(0.45, h / 1080.0)
        self.thick = max(1, round(1.6 * self.s))
        self.fs = 0.55 * self.s
        self.small = round(56 * self.s)
        self.hist: deque[float] = deque(maxlen=120)
        self.fps = 0.0

    def _line(self, img, a, b, color, thick) -> None:
        cv2.line(img, a, b, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.line(img, a, b, color, thick, cv2.LINE_AA)

    def _chip(self, img, x, y_top, text, color) -> None:
        (tw, th), base = cv2.getTextSize(text, FONT, self.fs, 1)
        pad = max(3, round(5 * self.s))
        cw, ch = tw + 2 * pad, th + base + pad
        cx = int(np.clip(x, 0, self.w - cw))
        cy = y_top - ch if y_top - ch >= 0 else y_top
        cy = int(np.clip(cy, 0, self.h - ch))
        cv2.rectangle(img, (cx - 1, cy - 1), (cx + cw + 1, cy + ch + 1), (0, 0, 0), -1, cv2.LINE_AA)
        cv2.rectangle(img, (cx, cy), (cx + cw, cy + ch), color, -1, cv2.LINE_AA)
        lum = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
        ink = (0, 0, 0) if lum > 150 else (255, 255, 255)
        cv2.putText(img, text, (cx + pad, cy + ch - base - 1), FONT, self.fs, ink, 1, cv2.LINE_AA)

    def boxes(self, img, labels, boxes, scores) -> None:
        for lab, (fx0, fy0, fx1, fy1), sc in zip(labels, boxes, scores):
            x0 = int(np.clip(fx0, 0, self.w - 1)); x1 = int(np.clip(fx1, 0, self.w - 1))
            y0 = int(np.clip(fy0, 0, self.h - 1)); y1 = int(np.clip(fy1, 0, self.h - 1))
            if x1 <= x0 or y1 <= y0:
                continue
            color = tuple(int(c) for c in self.colors[lab % len(self.colors)])
            side = min(x1 - x0, y1 - y0)
            _blend(img, (x0, y0, x1, y1), color, 0.12)
            cv2.rectangle(img, (x0, y0), (x1, y1), color,
                          self.thick if side >= self.small else self.thick + 1, cv2.LINE_AA)

            # corner arms on a small box meet in the middle and read as a dashed line
            if side >= self.small:
                arm = int(np.clip(side * 0.3, 7, 44 * self.s))
                heavy = self.thick + round(3 * self.s)
                for px, py, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1),
                                       (x0, y1, 1, -1), (x1, y1, -1, -1)):
                    self._line(img, (px, py), (px + dx * arm, py), color, heavy)
                    self._line(img, (px, py), (px, py + dy * arm), color, heavy)

            name = self.names[lab] if lab < len(self.names) else str(lab)
            self._chip(img, x0, y0 - max(2, round(3 * self.s)), f"{name} {sc:.2f}", color)

    def _sparkline(self, img, box) -> None:
        x0, y0, x1, y1 = box
        if len(self.hist) < 2:
            return
        v = np.asarray(self.hist, np.float32)
        xs = np.linspace(x0, x1, len(v))
        ys = y1 - (v / max(v.max(), 1.0)) * (y1 - y0)
        pts = np.stack([xs, ys], 1).astype(np.int32)
        cv2.polylines(img, [pts], False, self.ACCENT, max(1, self.thick), cv2.LINE_AA)
        cv2.circle(img, tuple(pts[-1]), max(2, self.thick + 1), self.ACCENT, -1, cv2.LINE_AA)

    def hud(self, img, *, n_det, engine_ms, e2e_ms, frame_i, total, watts=None) -> None:
        self.hist.append(1000.0 / max(e2e_ms, 1e-3))
        self.fps = self.hist[-1] if self.fps == 0 else 0.9 * self.fps + 0.1 * self.hist[-1]

        head = f"{self.fps:.1f} FPS"
        rows = [f"engine {engine_ms:.1f} ms   e2e {e2e_ms:.1f} ms",
                f"{n_det} objects   frame {frame_i}" + (f"/{total}" if total else "")]
        if watts is not None:
            rows.append(f"{watts:.1f} W   {watts / max(self.fps, 1e-3):.2f} J/frame")
        rows.append(self.engine)

        hs, rs = 1.1 * self.s, 0.5 * self.s
        pad = round(14 * self.s)
        gap = round(9 * self.s)
        spark = round(96 * self.s)
        (hw, hh), _ = cv2.getTextSize(head, FONT, hs, 2)
        rw = max(cv2.getTextSize(r, FONT, rs, 1)[0][0] for r in rows)
        rh = cv2.getTextSize(rows[0], FONT, rs, 1)[0][1]
        lh = rh + gap

        pw = max(hw + gap + spark, rw) + 2 * pad
        ph = hh + gap + lh * len(rows) + 2 * pad
        x0, y0 = pad, pad
        _blend(img, (x0, y0, x0 + pw, y0 + ph), (16, 16, 16), 0.66)
        cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), (72, 72, 72), 1, cv2.LINE_AA)
        cv2.rectangle(img, (x0, y0), (x0 + max(3, round(4 * self.s)), y0 + ph), self.ACCENT, -1)

        x, y = x0 + pad, y0 + pad + hh
        _text(img, head, (x, y), hs, (255, 255, 255), 2)
        self._sparkline(img, (x0 + pw - pad - spark, y0 + pad,
                              x0 + pw - pad, y0 + pad + hh))
        for k, row in enumerate(rows):
            y += lh
            shade = (150, 150, 150) if row is rows[-1] else (218, 218, 218)
            _text(img, row, (x, y), rs, shade, 1)


class TegraPower:
    """Board power from tegrastats. NVML (harness/power.py) does not exist on Jetson."""

    RAIL = re.compile(r"\b(VDD_IN|VDD_CPU_GPU_CV|VDD_SOC|VDD_GPU_SOC|POM_5V_IN)\s+(\d+)mW")

    def __init__(self, interval_ms: int = 100) -> None:
        self.interval_ms = interval_ms
        self.samples: dict[str, list[float]] = {}
        self._proc = None
        self._thread = None

    def _loop(self) -> None:
        for line in self._proc.stdout:
            for rail, mw in self.RAIL.findall(line):
                self.samples.setdefault(rail, []).append(int(mw) / 1000.0)

    def latest(self, rail: str = "VDD_IN") -> float | None:
        for key in (rail, *self.samples):
            v = self.samples.get(key)
            if v:
                return v[-1]
        return None

    def __enter__(self):
        try:
            self._proc = subprocess.Popen(
                ["tegrastats", "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except FileNotFoundError:
            print("[power] tegrastats not found; skipping power sampling")
            return self
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> bool:
        if self._proc is not None:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        return False

    def summary(self) -> dict:
        return {rail: {"mean_w": float(np.mean(v)), "peak_w": float(max(v)), "n": len(v)}
                for rail, v in self.samples.items() if v}


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *exc): return False


def load_class_names(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return [str(i) for i in range(80)]
    try:
        import yaml
        return list(yaml.safe_load(path.read_text())["names"])
    except Exception as exc:
        print(f"[classes] could not read {path} ({exc}); using numeric labels")
        return [str(i) for i in range(80)]


def write_row(table: Path, row: dict) -> None:
    cols = [k for k in row if k not in ("power", "stages_mean_ms")]
    table.parent.mkdir(parents=True, exist_ok=True)
    if not table.exists():
        table.write_text("| " + " | ".join(cols) + " |\n| "
                         + " | ".join("---" for _ in cols) + " |\n")
    with table.open("a") as f:
        f.write("| " + " | ".join(str(row[c]) for c in cols) + " |\n")


def write_profile(path: Path, row: dict, samples: dict[str, list[float]],
                  source: str, save: str | None) -> None:
    """Append an additive per-frame timing breakdown for one engine run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            "# RT-DETR video pipeline profile\n\n"
            "Stage means sum to the measured frame-loop mean. Per-stage medians "
            "do not generally sum to the frame median. Frame read/decode is "
            "excluded from processing e2e but included in pipeline FPS. "
            "Engine inference includes runner transfers and synchronization. "
            "Video writer finalization after the loop is excluded.\n")

    frame = np.asarray(samples["frame"], dtype=np.float64)
    frame_mean = float(np.mean(frame))
    lines = [
        f"\n## {Path(source).name} · {row['engine']}\n",
        f"Source: `{source}`  ",
        f"Saved video: `{save}`  " if save else "Saved video: none  ",
        f"Frames: {row['frames']}  ",
        f"Detections/frame: {row['det_per_frame']}  ",
        f"Engine p50/p90: {row['engine_p50_ms']}/{row['engine_p90_ms']} ms  ",
        f"Processing e2e p50: {row['e2e_p50_ms']} ms  ",
        f"Pipeline FPS: {row['pipeline_fps']}\n",
        "| Stage | Mean (ms) | p50 (ms) | p90 (ms) | Frame share |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key, label in STAGES:
        values = np.asarray(samples[key], dtype=np.float64)
        mean = float(np.mean(values))
        lines.append(f"| {label} | {mean:.2f} | {np.median(values):.2f} | "
                     f"{np.percentile(values, 90):.2f} | "
                     f"{100 * mean / frame_mean:.1f}% |")
    lines.append(f"| **Frame loop** | **{frame_mean:.2f}** | "
                 f"**{np.median(frame):.2f}** | "
                 f"**{np.percentile(frame, 90):.2f}** | **100%** |")
    if "vdd_in_mean_w" in row:
        lines.append(f"\nMean board power: {row['vdd_in_mean_w']} W; "
                     f"energy: {row['j_per_frame']} J/frame.\n")
    with path.open("a") as f:
        f.write("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="RT-DETR TensorRT inference on Jetson")
    ap.add_argument("--engine", required=True, help="a .plan built on this board")
    ap.add_argument("--source", required=True, help="video file, stream URL, or camera index")
    ap.add_argument("--save", default=None, help="write an annotated .mp4 here")
    ap.add_argument("--table", default=None, help="append a markdown summary row here")
    ap.add_argument("--profile", default=None,
                    help="append a per-stage timing breakdown to this markdown file")
    ap.add_argument("--classes", default=str(Path(__file__).resolve().parent / "configs" / "coco80.yaml"))
    ap.add_argument("--score-thr", type=float, default=0.6)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--power", action="store_true", help="sample board power via tegrastats")
    ap.add_argument("--no-draw", action="store_true", help="skip drawing boxes and the statistics overlay")
    ap.add_argument("--show", action="store_true", help="display detections; press q to stop")
    ap.add_argument("--no-hud", action="store_true", help="draw boxes but no stats panel")
    args = ap.parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        ap.error("--max-frames must be positive")
    if args.warmup < 0:
        ap.error("--warmup must be nonnegative")
    if not 0 <= args.score_thr <= 1:
        ap.error("--score-thr must be between 0 and 1")
    return args


def main() -> None:
    args = parse_args()

    names = load_class_names(Path(args.classes))
    sess = TRTSession(Path(args.engine))
    cap = None
    writer = None
    try:
        print(f"[trt] {args.engine}  in={sess.input_name}  out={sess.output_names}")

        src = int(args.source) if args.source.isdigit() else args.source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise SystemExit(f"cannot open source: {args.source}")
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        if args.max_frames:
            total = min(total, args.max_frames) if total else args.max_frames
        writer = None
        overlay = None

        dummy = np.zeros((1, 3, IMG_SIZE, IMG_SIZE), np.float32)
        for _ in range(args.warmup):
            sess.run(dummy)

        lat_ms: list[float] = []
        e2e_ms: list[float] = []
        stage_ms: dict[str, list[float]] = {key: [] for key, _ in STAGES}
        stage_ms["frame"] = []
        n_det = 0
        power = TegraPower() if args.power else None

        with (power or _NullCtx()):
            wall0 = time.perf_counter()
            i = 0
            while args.max_frames is None or i < args.max_frames:
                frame_start = time.perf_counter()
                ok, bgr = cap.read()
                read_end = time.perf_counter()
                if not ok:
                    break
                x = preprocess(np.ascontiguousarray(bgr[:, :, ::-1]))
                preprocess_end = time.perf_counter()
                out = sess.run(x)
                engine_end = time.perf_counter()

                labels, boxes, scores = postprocess(out["pred_logits"], out["pred_boxes"],
                                                    (bgr.shape[1], bgr.shape[0]))
                keep = scores >= args.score_thr
                labels, boxes, scores = labels[keep], boxes[keep], scores[keep]
                n_det += len(labels)
                postprocess_end = time.perf_counter()
                engine_ms = (engine_end - preprocess_end) * 1e3

                if not args.no_draw:
                    if overlay is None:
                        overlay = Overlay(bgr.shape[1], bgr.shape[0], names,
                                          Path(args.engine).name)
                    overlay.boxes(bgr, labels, boxes, scores)
                boxes_end = time.perf_counter()
                if not args.no_draw and not args.no_hud:
                    prev = e2e_ms[-1] if e2e_ms else engine_ms
                    overlay.hud(bgr, n_det=len(labels), engine_ms=engine_ms, e2e_ms=prev,
                                frame_i=i + 1, total=total,
                                watts=power.latest() if power else None)
                hud_end = time.perf_counter()
                if args.save:
                    if writer is None:
                        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
                        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                                 fps_in, (bgr.shape[1], bgr.shape[0]))
                        if not writer.isOpened():
                            raise RuntimeError(f"Cannot write video: {args.save}")
                    writer.write(bgr)
                write_end = time.perf_counter()

                if args.show:
                    cv2.imshow("RT-DETR", bgr)
                    stop = cv2.waitKey(1) & 0xFF == ord("q")
                else:
                    stop = False
                display_end = time.perf_counter()

                boundaries = (frame_start, read_end, preprocess_end, engine_end,
                              postprocess_end, boxes_end, hud_end, write_end, display_end)
                for (key, _), start, end in zip(STAGES, boundaries, boundaries[1:]):
                    stage_ms[key].append((end - start) * 1e3)
                stage_ms["frame"].append((display_end - frame_start) * 1e3)
                lat_ms.append(engine_ms)
                e2e_ms.append((display_end - read_end) * 1e3)
                i += 1
                if stop:
                    break
                if i % 50 == 0:
                    print(f"  {i} frames  engine p50 {np.median(lat_ms):.2f} ms", flush=True)
            wall = time.perf_counter() - wall0

        if not lat_ms:
            raise SystemExit("No frames decoded from the source")

        a = np.asarray(lat_ms)
        row = {
            "source": Path(args.source).name,
            "engine": Path(args.engine).name,
            "frames": len(a),
            "det_per_frame": round(n_det / max(len(a), 1), 2),
            "engine_p50_ms": round(float(np.median(a)), 2),
            "engine_p90_ms": round(float(np.percentile(a, 90)), 2),
            "e2e_p50_ms": round(float(np.median(e2e_ms)), 2),
            "pipeline_fps": round(len(a) / wall, 2),
            "stages_mean_ms": {key: round(float(np.mean(stage_ms[key])), 2)
                               for key, _ in STAGES},
        }
        if power is not None:
            summary = power.summary()
            row["power"] = summary
            for rail, s in summary.items():
                row[f"{rail.lower()}_mean_w"] = round(s["mean_w"], 2)
            if "VDD_IN" in summary:
                row["j_per_frame"] = round(summary["VDD_IN"]["mean_w"] * wall / max(len(a), 1), 3)

        print(json.dumps(row, indent=2))
        if args.table:
            write_row(Path(args.table), row)
            print(f"[table] appended to {args.table}")
        if args.profile:
            write_profile(Path(args.profile), row, stage_ms, args.source, args.save)
            print(f"[profile] appended to {args.profile}")

    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        sess.close()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
