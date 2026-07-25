from __future__ import annotations

import sys
from pathlib import Path


def _repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "harness").is_dir() and (candidate / "configs").is_dir():
            return candidate
    raise RuntimeError(f"repo root (a dir with harness/ and configs/) not found above {__file__}")


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
