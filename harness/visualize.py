from __future__ import annotations

import colorsys
import os
from functools import lru_cache
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_PALETTE = [
    (239, 71, 111), (255, 140, 66), (255, 196, 61), (214, 214, 0), (149, 213, 89),
    (46, 204, 139), (0, 201, 167), (34, 189, 224), (66, 153, 245), (99, 110, 250),
    (149, 97, 255), (194, 96, 245), (247, 92, 199), (255, 122, 145), (120, 144, 156),
    (0, 176, 255), (124, 179, 66), (255, 167, 38),
]

_BG = (15, 17, 23)          # canvas / HUD background
_HAIRLINE = (255, 255, 255, 22)
_TORCH = (255, 138, 76)     # warm accent -> PyTorch panel
_ORT = (86, 156, 255)       # cool accent -> ONNX Runtime panel
_MUTED = (150, 156, 168)
_NAME = (240, 242, 245)
_TRACK = (255, 255, 255, 30)  # meter/track fill on dark


@lru_cache(maxsize=32)
def _font(size: int, bold: bool = False):
    """Cached TrueType loader with a graceful fall back to Pillow's default."""
    for name in (("DejaVuSans-Bold.ttf",) if bold else ("DejaVuSans.ttf",)):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size)
    except TypeError:  # very old Pillow
        return ImageFont.load_default()

_SHADE_SAT = (0.30, 1.00)
_SHADE_VAL = (0.55, 1.00)
_SHADE_ALPHA = (16, 64)     # translucent box fill, min -> max confidence

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _lerp(lo, hi, t):
    return lo + (hi - lo) * t


def _color_for(idx: int):
    return _PALETTE[int(idx) % len(_PALETTE)]


def _confidence(score: float, score_thr: float = 0.0) -> float:
    """Map a score onto [0, 1] over the *displayed* range ``[score_thr, 1.0]``.

    Anchoring to the threshold (rather than to the frame's own min/max) keeps the
    mapping absolute: the same detection gets the same shade in every frame and in
    both the torch and ORT panels, so the two are directly comparable.
    """
    lo = _clamp(float(score_thr), 0.0, 0.99)
    return _clamp((float(score) - lo) / (1.0 - lo), 0.0, 1.0)


def _shade(rgb, t: float):
    """Tint a palette colour by confidence ``t``: same hue, scaled saturation/value."""
    h, s, v = colorsys.rgb_to_hsv(*(c / 255.0 for c in rgb))
    r, g, b = colorsys.hsv_to_rgb(
        h,
        _clamp(s * _lerp(*_SHADE_SAT, t), 0.0, 1.0),
        _clamp(v * _lerp(*_SHADE_VAL, t), 0.0, 1.0),
    )
    return (round(r * 255), round(g * 255), round(b * 255))


def _text_on(rgb):
    """Black or white ink, whichever reads on ``rgb`` (Rec. 601 luma)."""
    r, g, b = rgb[:3]
    return (14, 16, 20) if (0.299 * r + 0.587 * g + 0.114 * b) > 140 else (245, 247, 250)


def _fps(ms) -> float:
    return 1000.0 / ms if ms else 0.0

def _draw_boxes(base: Image.Image, det, class_names, *, score_thr: float = 0.0) -> Image.Image:
    """Modern detection overlay: per-class hue, soft translucent fill, rounded border,
    and a filled label chip with contrast-aware text.

    The colour's *shade* varies continuously with the score over ``[score_thr, 1.0]``:
    a barely-above-threshold box is dim and desaturated, a confident one is fully vivid.
    """
    im = base.convert("RGBA")
    W, H = im.size
    scale = _clamp(min(W, H) / 520.0, 1.0, 2.2)
    stroke = max(2, round(2 * scale))
    font = _font(int(13 * scale), bold=True)

    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    labels = np.asarray(det.labels).reshape(-1)
    boxes = np.asarray(det.boxes, dtype=float).reshape(-1, 4) if len(labels) else np.empty((0, 4))
    scores = np.asarray(det.scores, dtype=float).reshape(-1)
    # Draw larger boxes first so small boxes and their chips end up on top.
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if len(boxes) else np.empty(0)
    order = np.argsort(-areas)

    for i in order:
        x1, y1, x2, y2 = (float(v) for v in boxes[i])
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        t = _confidence(scores[i], score_thr)
        color = _shade(_color_for(labels[i]), t)
        alpha = int(round(_lerp(*_SHADE_ALPHA, t)))
        radius = int(_clamp(min(x2 - x1, y2 - y1) * 0.14, 4, 16 * scale))

        d.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=color + (alpha,))
        d.rounded_rectangle([x1, y1, x2, y2], radius=radius, outline=color + (255,), width=stroke)

        idx = int(labels[i])
        name = class_names[idx] if 0 <= idx < len(class_names) else str(idx)
        text = f"{name}  {scores[i]:.2f}"
        tb = d.textbbox((0, 0), text, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        pad_x, pad_y = int(7 * scale), int(4 * scale)
        chip_w, chip_h = tw + 2 * pad_x, th + 2 * pad_y

        cx = _clamp(x1 - stroke / 2, 0, max(0, W - chip_w))
        cy = y1 - chip_h - int(2 * scale)
        if cy < 0:                       # no room above -> tuck inside the box
            cy = y1 + int(2 * scale)
        chip_r = int(_clamp(chip_h * 0.28, 3, 8))
        d.rounded_rectangle([cx, cy, cx + chip_w, cy + chip_h], radius=chip_r, fill=color + (240,))
        d.text((cx + chip_w / 2, cy + chip_h / 2), text, font=font,
               fill=_text_on(color) + (255,), anchor="mm")

    return Image.alpha_composite(im, overlay).convert("RGB")

def _pill(d, right_x, cy, text, font, bg, fg, s):
    """Right-anchored rounded pill; returns its left edge x."""
    tb = d.textbbox((0, 0), text, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    px, py = int(9 * s), int(4 * s)
    w, h = tw + 2 * px, th + 2 * py
    x0, y0 = right_x - w, cy - h / 2
    d.rounded_rectangle([x0, y0, right_x, y0 + h], radius=h / 2, fill=bg)
    d.text((right_x - px, cy), text, font=font, fill=fg, anchor="rm")
    return x0


def _meter(d, x, y, w, h, frac, accent):
    """Slim rounded progress meter (track + accent fill)."""
    d.rounded_rectangle([x, y, x + w, y + h], radius=h / 2, fill=_TRACK)
    fw = max(h, w * _clamp(frac, 0.0, 1.0))
    d.rounded_rectangle([x, y, x + fw, y + h], radius=h / 2, fill=accent + (255,))


def _hud_panel(d, x0, width, header_h, s, *, accent, name, subtitle, fps, fps_ref, n):
    y1 = header_h * 0.30
    y2 = header_h * 0.62
    dot_r = 4 * s
    dcx = x0 + 15 * s
    d.ellipse([dcx - dot_r, y1 - dot_r, dcx + dot_r, y1 + dot_r], fill=accent + (255,))
    d.text((dcx + dot_r + 7 * s, y1), name, font=_font(int(15 * s), bold=True),
           fill=_NAME + (255,), anchor="lm")

    right_x = x0 + width - 12 * s
    _pill(d, right_x, y1, f"{fps:5.1f} FPS", _font(int(13 * s), bold=True),
          accent + (255,), _text_on(accent) + (255,), s)

    d.text((dcx, y2), subtitle, font=_font(int(11 * s)), fill=_MUTED + (255,), anchor="lm")
    bar_x = dcx + 62 * s
    bar_w = (x0 + width - 12 * s) - bar_x - 46 * s
    if bar_w > 20 * s:
        _meter(d, bar_x, y2 - 3 * s, bar_w, 6 * s, fps / fps_ref if fps_ref else 0.0, accent)
    d.text((x0 + width - 12 * s, y2), f"{n} obj", font=_font(int(11 * s)),
           fill=_MUTED + (255,), anchor="rm")


def _frame_readout(d, W, header_h, s, report):
    n_t = report.get("n_torch", 0)
    matched = report.get("matched", 0)
    agree = (matched / n_t * 100.0) if n_t else 100.0
    ac = ((120, 215, 130) if agree >= 99.5 else (255, 190, 80) if agree >= 90 else (255, 110, 110))
    f = _font(int(11 * s))
    parts = [
        (f"frame {report.get('id', 0)}", _MUTED),
        (f"match {matched}/{n_t} ({agree:.0f}%)", ac),
        (f"Δbox {report.get('max_box_diff', 0.0):.2f}px", _MUTED),
        (f"Δscore {report.get('max_score_diff', 0.0):.3f}", _MUTED),
    ]
    sep = "   ·   "
    widths = [d.textbbox((0, 0), t, font=f)[2] for t, _ in parts]
    sep_w = d.textbbox((0, 0), sep, font=f)[2]
    total = sum(widths) + sep_w * (len(parts) - 1)
    x = W / 2 - total / 2
    y = header_h - 12 * s
    for i, (t, c) in enumerate(parts):
        d.text((x, y), t, font=f, fill=c + (255,), anchor="lm")
        x += widths[i]
        if i < len(parts) - 1:
            d.text((x, y), sep, font=f, fill=(90, 96, 108, 255), anchor="lm")
            x += sep_w


def draw_side_by_side(
    orig_pil: Image.Image, t_det, o_det, report: dict, class_names, *, score_thr: float = 0.0
) -> Image.Image:
    """Join torch (left) and ORT (right) overlays under a modern stats HUD.

    Each panel header shows the backend, its inference FPS as a pill plus a relative
    speed meter (both bars share a scale, so the faster backend visibly wins), and its
    object count. A centered readout reports per-frame agreement and torch-vs-ORT drift.

    ``score_thr`` anchors the low end of the box-colour confidence ramp; passing the
    same threshold used to filter detections makes the weakest visible box the dimmest.
    """
    left = _draw_boxes(orig_pil, t_det, class_names, score_thr=score_thr)
    right = _draw_boxes(orig_pil, o_det, class_names, score_thr=score_thr)
    gap = 3
    W = left.width + right.width + gap
    s = _clamp(W / 1000.0, 0.85, 1.6)
    header_h = int(round(78 * s))
    H = header_h + max(left.height, right.height)

    canvas = Image.new("RGB", (W, H), _BG)
    canvas.paste(left, (0, header_h))
    canvas.paste(right, (left.width + gap, header_h))
    d = ImageDraw.Draw(canvas, "RGBA")

    seam_x = left.width + gap // 2
    d.line([(seam_x, header_h), (seam_x, H)], fill=_HAIRLINE, width=1)
    d.line([(0, header_h - 1), (W, header_h - 1)], fill=_HAIRLINE, width=1)

    t_fps, o_fps = _fps(report.get("torch_ms")), _fps(report.get("ort_ms"))
    ref = max(t_fps, o_fps, 1e-6)
    _hud_panel(d, 0, left.width, header_h, s, accent=_TORCH, name="PyTorch",
               subtitle="eager", fps=t_fps, fps_ref=ref, n=report.get("n_torch", len(t_det)))
    _hud_panel(d, left.width + gap, right.width, header_h, s, accent=_ORT, name="ONNX Runtime",
               subtitle="runtime", fps=o_fps, fps_ref=ref, n=report.get("n_ort", len(o_det)))
    _frame_readout(d, W, header_h, s, report)
    return canvas

class FigureSink:
    """Save each side-by-side panel as a numbered JPG under ``out_dir``."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.n = 0

    def consume(self, frame, panel: Image.Image, report: dict) -> bool:
        panel.save(self.out_dir / f"{frame.id:06d}.jpg")
        self.n += 1
        return True

    def close(self) -> None:
        if self.n:
            print(f"[viz] wrote {self.n} panel(s) to {self.out_dir}")


class VideoSink:
    """Write annotated panels to an mp4 via ``cv2.VideoWriter`` (lazy-sized)."""

    def __init__(self, out_path: Path, fps: float = 20.0) -> None:
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.writer = None

    def consume(self, frame, panel: Image.Image, report: dict) -> bool:
        import cv2

        arr = cv2.cvtColor(np.asarray(panel), cv2.COLOR_RGB2BGR)
        if self.writer is None:
            h, w = arr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(self.out_path), fourcc, self.fps, (w, h))
        self.writer.write(arr)
        return True

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            print(f"[viz] wrote video to {self.out_path}")


class WindowSink:
    """Live preview via ``cv2.imshow``; press 'q' to stop the run."""

    def __init__(self, title: str = "torch vs ONNX Runtime") -> None:
        self.title = title

    def consume(self, frame, panel: Image.Image, report: dict) -> bool:
        import cv2

        arr = cv2.cvtColor(np.asarray(panel), cv2.COLOR_RGB2BGR)
        cv2.imshow(self.title, arr)
        return (cv2.waitKey(1) & 0xFF) != ord("q")

    def close(self) -> None:
        import cv2

        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


class MultiSink:
    """Fan a panel out to several sinks; stops when any sink asks to stop."""

    def __init__(self, sinks: List[object]) -> None:
        self.sinks = sinks

    def consume(self, frame, panel: Image.Image, report: dict) -> bool:
        keep = True
        for s in self.sinks:
            keep = bool(s.consume(frame, panel, report)) and keep
        return keep

    def close(self) -> None:
        for s in self.sinks:
            s.close()


def _gui_available() -> bool:
    """True only if OpenCV can actually open a window.

    Checks both a display AND that this cv2 build has HighGUI support — the common
    ``opencv-python-headless`` wheel has no GTK/Cocoa backend, so ``imshow`` raises
    ``cv2.error: The function is not implemented`` even when ``$DISPLAY`` is set.
    """
    if not (os.environ.get("DISPLAY") or os.name == "nt"):
        return False
    try:
        import cv2

        cv2.namedWindow("__gui_probe__", cv2.WINDOW_AUTOSIZE)
        cv2.destroyWindow("__gui_probe__")
        return True
    except Exception:  # noqa: BLE001 - cv2.error (no GUI) or anything else -> no window
        return False


def make_sink(kind: str, *, show: bool, save: bool, out_dir, source_stem: str, fps: float = 20.0):
    """Choose output sinks for a source kind, degrading gracefully when headless."""
    out_dir = Path(out_dir)
    sinks: List[object] = []

    if kind == "image":
        sinks.append(FigureSink(out_dir / source_stem))
        return MultiSink(sinks)

    # video / camera
    want_window = show and _gui_available()
    if show and not want_window:
        print("[viz] live window unavailable (no display, or OpenCV built without GUI "
              "support, e.g. opencv-python-headless); saving an annotated video instead")
        save = True
    if want_window:
        sinks.append(WindowSink())
    # Always leave an artifact for video/camera unless we are showing a live window.
    if save or not want_window:
        sinks.append(VideoSink(out_dir / f"{source_stem}_compare.mp4", fps=fps))

    if not sinks:
        sinks.append(FigureSink(out_dir / source_stem))
    return MultiSink(sinks)
