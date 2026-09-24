"""Shared multi-backend driver.

One pass over a source, every backend run per frame, then either fanned out to a
**separate sink per backend** (independent windows / videos) or drawn into a **single
compare sink** (side-by-side, with agreement-vs-reference). Backends are passed in as a
list of objects exposing ``label``, ``infer(x) -> {name: np.ndarray}``, and ``close()``
(see ``infer_torch`` / ``infer_ort`` / ``infer_engine``) — this module imports none of
them, so there is no import cycle.

Timing is a uniform end-to-end wall-clock (``StreamTimer``) for every backend, which
makes them directly comparable — each backend's ``infer`` returns host numpy, so the call
is fully synchronized when timed. It is a *display* measure; the authoritative benchmark
is ``trtexec`` with locked clocks (running N backends at once also makes them contend for
the shared laptop power/thermal envelope).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

import numpy as np

from .adapter import DetectorAdapter, Detections
from .metrics import StreamTimer
from .sources import open_source
from .visualize import _ACCENTS, draw_compare, draw_single, make_sink

_GREEN, _AMBER, _RED, _MUTED = (120, 215, 130), (255, 190, 80), (255, 110, 110), (150, 156, 168)


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of one box ``a`` (4,) against many boxes ``b`` (M, 4)."""
    x1 = np.maximum(a[0], b[:, 0]); y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2]); y2 = np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0)


def match_detections(ref: Detections, other: Detections, iou_thr: float) -> dict:
    """Greedy same-label IoU match of ``other`` against reference ``ref``.

    Same-label + IoU matching is robust to top-K reshuffle (a positional zip misaligns
    whenever a background query straddles the cutoff). Reports counts and the worst
    box/score disagreement over matched pairs.
    """
    used: set = set()
    matched = 0
    max_box = 0.0
    max_score = 0.0
    for i in range(len(ref)):
        rb = np.asarray(ref.boxes[i], dtype=float)
        rl = int(ref.labels[i])
        cand = [j for j in range(len(other)) if j not in used and int(other.labels[j]) == rl]
        if not cand:
            continue
        ious = _iou(rb, np.asarray(other.boxes)[cand].astype(float))
        k = int(np.argmax(ious))
        if ious[k] >= iou_thr:
            j = cand[k]
            used.add(j)
            matched += 1
            max_box = max(max_box, float(np.max(np.abs(rb - np.asarray(other.boxes[j], dtype=float)))))
            max_score = max(max_score, float(abs(float(ref.scores[i]) - float(other.scores[j]))))
    return {"matched": matched, "n_ref": len(ref), "n_other": len(other),
            "mismatch": (len(ref) - matched) or (len(other) - matched), "box": max_box, "score": max_score}


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or "backend"


def _agree_color(pct: float):
    return _GREEN if pct >= 99.5 else _AMBER if pct >= 90 else _RED


def run_backends(
    adapter: DetectorAdapter,
    source,
    backends: List,
    *,
    compare: bool = False,
    score_thr: float = 0.6,
    iou_thr: float = 0.5,
    show: bool = False,
    save: bool = False,
    out_dir="results/figures/infer",
    warmup: int = 1,
    max_frames: Optional[int] = None,
    source_label: str = "source",
) -> dict:
    """Run every backend over ``source``; separate sinks, or one compare sink."""
    if not backends:
        raise ValueError("run_backends needs at least one backend")
    out_dir = Path(out_dir) / source_label
    timers = [StreamTimer(warmup=warmup) for _ in backends]
    stats = [{"frames": 0, "dets": 0} for _ in backends]
    # comparison accumulators, one per non-reference backend (reference = backends[0])
    cmp = [{"matched": 0, "n_ref": 0, "n_other": 0, "mismatch_frames": 0, "box": [], "score": []}
           for _ in backends] if compare else None
    sinks: dict = {}
    stop = False
    print(f"[run] backends={[b.label for b in backends]}  mode={'compare' if compare else 'separate'}")

    try:
        for frame in open_source(source, max_frames=max_frames):
            x, meta = adapter.preprocess(frame.pil)
            dets: List[Detections] = []
            for i, b in enumerate(backends):
                with timers[i].time():
                    named = b.infer(x)
                det = adapter.postprocess(named, meta).filter(score_thr)
                dets.append(det)
                stats[i]["frames"] += 1
                stats[i]["dets"] += len(det)

            if compare:
                ref = dets[0]
                readout = []
                for i in range(1, len(backends)):
                    r = match_detections(ref, dets[i], iou_thr)
                    c = cmp[i]
                    c["matched"] += r["matched"]; c["n_ref"] += r["n_ref"]; c["n_other"] += r["n_other"]
                    if r["mismatch"]:
                        c["mismatch_frames"] += 1
                    if r["matched"]:
                        c["box"].append(r["box"]); c["score"].append(r["score"])
                    pct = (r["matched"] / r["n_ref"] * 100.0) if r["n_ref"] else 100.0
                    readout.append((f"{backends[i].label}≈{backends[0].label} {pct:.0f}%", _agree_color(pct)))
                entries = [{"label": backends[i].label, "det": dets[i], "ms": timers[i].last_ms,
                            "accent": _ACCENTS[i % len(_ACCENTS)]} for i in range(len(backends))]
                panel = draw_compare(frame.pil, entries, adapter.class_names,
                                     score_thr=score_thr, readout=readout)
                if "compare" not in sinks:
                    sinks["compare"] = make_sink(frame.kind, show=show, save=save, out_dir=out_dir,
                                                 source_stem="compare", stem_suffix="", title="compare")
                if not sinks["compare"].consume(frame, panel, {"id": frame.id, "n": len(ref)}):
                    stop = True
            else:
                for i, b in enumerate(backends):
                    rep = {"id": frame.id, "n": len(dets[i]), "backend": b.label, "ms": timers[i].last_ms}
                    panel = draw_single(frame.pil, dets[i], rep, adapter.class_names, score_thr=score_thr)
                    if b.label not in sinks:
                        sinks[b.label] = make_sink(frame.kind, show=show, save=save, out_dir=out_dir,
                                                   source_stem=_slug(b.label), stem_suffix="", title=b.label)
                    if not sinks[b.label].consume(frame, panel, rep):
                        stop = True

            print("[frame {:>5}] ".format(frame.id) + "  ".join(
                f"{b.label}:{len(dets[i])}det/{(timers[i].last_ms or 0):.1f}ms" for i, b in enumerate(backends)))
            if stop:
                break
    finally:
        for s in sinks.values():
            s.close()
        for b in backends:
            b.close()

    print(_summary(backends, stats, timers, cmp))
    _write_table(out_dir, backends, stats, timers, cmp, source_label)
    return {"stats": stats, "timers": timers, "cmp": cmp}


def _rows(backends, stats, timers, cmp):
    """Assemble table rows as dicts (shared by the printed summary and the .md file)."""
    rows = []
    for i, b in enumerate(backends):
        a = timers[i].agg
        n = stats[i]["frames"]
        row = {"backend": b.label, "frames": n,
               "det": stats[i]["dets"], "det_per_frame": (stats[i]["dets"] / n) if n else 0.0,
               "lat_mean": a.mean(), "lat_p50": a.median(), "lat_p90": a.p90(), "fps": a.fps()}
        if cmp is not None:
            if i == 0:
                row.update(agree="ref", box_max=0.0, score_max=0.0)
            else:
                c = cmp[i]
                row.update(
                    agree=(c["matched"] / c["n_ref"] * 100.0) if c["n_ref"] else 100.0,
                    box_max=(max(c["box"]) if c["box"] else 0.0),
                    score_max=(max(c["score"]) if c["score"] else 0.0))
        rows.append(row)
    return rows


def _summary(backends, stats, timers, cmp) -> str:
    rows = _rows(backends, stats, timers, cmp)
    lines = ["", "===================== inference summary =====================",
             f" frames: {stats[0]['frames'] if stats else 0}   mode: {'compare' if cmp else 'separate'}"]
    for r in rows:
        extra = ""
        if cmp is not None:
            extra = "  agree=ref" if r["agree"] == "ref" else \
                    f"  agree={r['agree']:.1f}%  Δbox={r['box_max']:.2f}px  Δscore={r['score_max']:.3f}"
        lines.append(f" {r['backend']:22s} {r['det_per_frame']:4.1f} det/f  "
                     f"lat {r['lat_mean']:6.2f} ms (p50 {r['lat_p50']:.2f}, p90 {r['lat_p90']:.2f})  "
                     f"{r['fps']:6.1f} FPS{extra}")
    lines += [" note: display wall-clock under N-backend contention; benchmark with trtexec + locked clocks.",
              "============================================================="]
    return "\n".join(lines)


def _write_table(out_dir, backends, stats, timers, cmp, source_label) -> Path:
    rows = _rows(backends, stats, timers, cmp)
    cols = ["backend", "frames", "det", "det_per_frame", "lat_mean", "lat_p50", "lat_p90", "fps"]
    heads = ["backend", "frames", "det", "det/frame", "lat_mean_ms", "lat_p50_ms", "lat_p90_ms", "FPS"]
    if cmp is not None:
        cols += ["agree", "box_max", "score_max"]
        heads += ["agree_%", "box_max_px", "score_max"]

    def fmt(v):
        return f"{v:.2f}" if isinstance(v, float) else str(v)

    lines = [f"# Inference summary — `{source_label}`", "",
             "| " + " | ".join(heads) + " |",
             "|" + "|".join(["---"] * len(heads)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    lines += ["", "_Display wall-clock under N-backend contention; not a benchmark "
              "(use trtexec + locked clocks)._", ""]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / "summary.md"
    path.write_text("\n".join(lines))
    print(f"[run] wrote summary table -> {path}")
    return path
