
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .paths import CONFIGS_DIR, ROOTS, require, resolve

_PLACEHOLDER = re.compile(r"\$\{(\w+)\}")


def _interpolate(value: Any) -> Any:
    """Recursively replace ``${root}`` in every string with its resolved absolute path."""
    if isinstance(value, str):
        def sub(m: re.Match) -> str:
            key = m.group(1)
            if key not in ROOTS:
                raise KeyError(f"unknown path root '${{{key}}}' in config; "
                               f"known roots: {sorted(ROOTS)}")
            return str(ROOTS[key])
        return _PLACEHOLDER.sub(sub, value)
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    return value


def _load_yaml(path: Path) -> dict:
    with open(require(path, "config file")) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    return data


def _load_class_names(value) -> List[str]:
    """``classes:`` is either an inline list or a path to a YAML holding ``names: [...]``."""
    if isinstance(value, list):
        return [str(v) for v in value]
    data = _load_yaml(resolve(value))
    names = data.get("names", data)
    if not isinstance(names, list):
        raise ValueError(f"{value}: expected a list under 'names'")
    return [str(n) for n in names]


@dataclass(frozen=True)
class ModelSpec:
    """One model's static settings, with every path already resolved for this machine."""

    name: str
    class_names: List[str]
    input_name: str
    output_names: List[str]
    img_size: int
    batch: int
    torch_config: Path                      # the upstream model .yml (in the submodule)
    ckpt: Path
    onnx: Dict[str, Path]                   # precision variant -> file
    default_onnx: str                       # which variant CLIs use when none is given
    defaults: Dict[str, Any] = field(default_factory=dict)   # device, thresholds, ...
    outputs: Dict[str, Path] = field(default_factory=dict)    # figures dir, table file
    source: Optional[Path] = None           # the YAML this came from
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def onnx_path(self, ref: Optional[str] = None) -> Path:
        """Resolve a variant name (``fp16``) or a filesystem path to an absolute Path."""
        if ref is None:
            ref = self.default_onnx
        if ref in self.onnx:
            return self.onnx[ref]
        return resolve(ref)

    def variant_of(self, path) -> str:
        """Reverse lookup: the variant name for a path, else its filename."""
        path = Path(path)
        for key, value in self.onnx.items():
            if value == path:
                return key
        return path.stem

    def default(self, key: str, override=None):
        """CLI override if given, else the YAML default, else ``None``."""
        return override if override is not None else self.defaults.get(key)


def spec_path(name_or_path) -> Path:
    """``"rtdetr"`` -> ``configs/rtdetr.yaml``; anything path-like is used as given."""
    candidate = Path(name_or_path)
    if candidate.suffix in (".yaml", ".yml") or candidate.exists():
        return resolve(candidate)
    return CONFIGS_DIR / f"{candidate}.yaml"


def load_spec(name_or_path) -> ModelSpec:
    """Load and resolve a model spec by name (``rtdetr``) or by path."""
    path = spec_path(name_or_path)
    raw = _interpolate(_load_yaml(path))

    onnx_cfg = raw.get("onnx", {})
    variants = {k: resolve(v) for k, v in (onnx_cfg.get("variants") or {}).items()}
    if not variants:
        raise ValueError(f"{path}: onnx.variants must list at least one precision variant")
    default_onnx = onnx_cfg.get("default", next(iter(variants)))
    if default_onnx not in variants:
        raise ValueError(f"{path}: onnx.default '{default_onnx}' is not in "
                         f"onnx.variants ({sorted(variants)})")

    model_cfg = raw.get("model", {})
    return ModelSpec(
        name=raw.get("name", path.stem),
        class_names=_load_class_names(raw["classes"]),
        input_name=onnx_cfg.get("input_name", "images"),
        output_names=list(onnx_cfg.get("output_names", [])),
        img_size=int(onnx_cfg.get("img_size", 640)),
        batch=int(onnx_cfg.get("batch", 1)),
        torch_config=resolve(model_cfg["config"]),
        ckpt=resolve(model_cfg["ckpt"]),
        onnx=variants,
        default_onnx=default_onnx,
        defaults=dict(raw.get("defaults", {})),
        outputs={k: resolve(v) for k, v in (raw.get("outputs") or {}).items()},
        source=path,
        raw=raw,
    )
