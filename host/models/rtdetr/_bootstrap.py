from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

def _repo_root() -> Path:
    for candidate in HERE.parents:
        if (candidate / "harness").is_dir() and (candidate / "configs").is_dir():
            return candidate
    raise RuntimeError(f"repo root (a dir with harness/ and configs/) not found above {__file__}")


REPO_ROOT = _repo_root()
for _entry in (str(HERE), str(REPO_ROOT)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)
