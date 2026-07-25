from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

__all__ = [
    "REPO_ROOT", "CONFIGS_DIR", "MODELS_DIR", "HARNESS_DIR", "PIPELINE_DIR",
    "DATA_DIR", "COCO_DIR", "COCO_VAL_IMAGES", "COCO_VAL_ANN",
    "RESULTS_DIR", "FIGURES_DIR", "TABLES_DIR", "ENGINES_DIR",
    "RTDETR_ROOT", "RTDETR_PYTORCH_ROOT", "RTDETR_SRC_ROOT",
    "ROOTS", "find_repo_root", "resolve", "require",
]

_ROOT_MARKERS = ("harness", "configs")

def find_repo_root(start: Optional[Path] = None) -> Path:
    """Walk up from ``start`` for the checkout root; ``$TRTQP_ROOT`` wins if set."""
    env = os.environ.get("TRTQP_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(start or __file__).resolve()
    for candidate in (here, *here.parents):
        if candidate.is_dir() and all((candidate / m).is_dir() for m in _ROOT_MARKERS):
            return candidate
    return Path(__file__).resolve().parents[1]   # installed oddly; fall back to layout


def _env_dir(var: str, default: Path) -> Path:
    value = os.environ.get(var)
    return Path(value).expanduser().resolve() if value else default


REPO_ROOT = find_repo_root()

# in-tree source layout
CONFIGS_DIR = REPO_ROOT / "configs"
MODELS_DIR = REPO_ROOT / "models"
HARNESS_DIR = REPO_ROOT / "harness"
PIPELINE_DIR = REPO_ROOT / "pipeline"

# datasets (often on a different disk -> overridable)
DATA_DIR = _env_dir("TRTQP_DATA", REPO_ROOT / "data")
COCO_DIR = _env_dir("TRTQP_COCO", DATA_DIR / "coco")
COCO_VAL_IMAGES = COCO_DIR / "val2017"
COCO_VAL_ANN = COCO_DIR / "annotations" / "instances_val2017.json"

# artifacts
RESULTS_DIR = _env_dir("TRTQP_RESULTS", REPO_ROOT / "results")
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
ENGINES_DIR = RESULTS_DIR / "engines"

# submodules (read-only; see the hard rule in CLAUDE.md)
RTDETR_ROOT = _env_dir("RTDETR_ROOT", REPO_ROOT / "RT-DETR")
RTDETR_PYTORCH_ROOT = RTDETR_ROOT / "rtdetr_pytorch"
RTDETR_SRC_ROOT = RTDETR_PYTORCH_ROOT / "src"

#: names usable as ``${...}`` inside ``configs/*.yaml`` (see :mod:`harness.config`)
ROOTS: Dict[str, Path] = {
    "repo": REPO_ROOT,
    "configs": CONFIGS_DIR,
    "models": MODELS_DIR,
    "pipeline": PIPELINE_DIR,
    "data": DATA_DIR,
    "coco": COCO_DIR,
    "results": RESULTS_DIR,
    "figures": FIGURES_DIR,
    "tables": TABLES_DIR,
    "engines": ENGINES_DIR,
    "rtdetr": RTDETR_ROOT,
    "rtdetr_pytorch": RTDETR_PYTORCH_ROOT,
    "rtdetr_src": RTDETR_SRC_ROOT,
}


def resolve(value, base: Optional[Path] = None) -> Path:
    """``value`` -> absolute Path; ``~`` expanded, relative paths taken against ``base``."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (Path(base) if base else REPO_ROOT) / path
    return path


def require(path, what: str, hint: str = "") -> Path:
    """Return ``path`` if it exists, else raise with what was missing and how to get it."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}" + (f"\n  {hint}" if hint else ""))
    return path
