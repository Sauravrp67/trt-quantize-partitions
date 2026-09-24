
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .paths import CONFIGS_DIR, ROOTS, require, resolve

# One spec file per model family; variants of it (backbones) are a registry inside that
# file, selected by name:
#
#     spec = load_spec("rtdetr")                  # the spec's own `backbone:` choice
#     spec = load_spec("rtdetr", "r101vd")        # override it
#     spec.backbones                              # -> every entry the registry offers
#     spec.onnx_path("fp16")                      # that backbone's fp16 export
#
# ONNX selection is by variant name, never by path: the weights a CLI runs and the spec
# it reads preprocessing/postprocessing from must describe the same model.
#
# Paths may contain `{backbone}`, expanded after selection, so per-backbone runs write to
# distinct figure dirs and result tables instead of overwriting each other.

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


#: keys a backbone entry may override on the shared ``onnx:`` contract
_CONTRACT_KEYS = ("input_name", "output_names", "img_size", "batch", "default")


def _fill(value: Any, **fields: str) -> Any:
    """Recursively expand ``{backbone}``-style fields, known only after selection."""
    if isinstance(value, str):
        for key, replacement in fields.items():
            value = value.replace("{" + key + "}", replacement)
        return value
    if isinstance(value, dict):
        return {k: _fill(v, **fields) for k, v in value.items()}
    if isinstance(value, list):
        return [_fill(v, **fields) for v in value]
    return value


def _select_backbone(raw: dict, path: Path, override: Optional[str]):
    """Pick an entry from the ``backbones:`` registry (name, entry, all names).

    A spec with no registry — one model, one checkpoint — is treated as a single
    unnamed entry, so both shapes load through the same path.
    """
    registry = raw.get("backbones")
    if not registry:
        registry = {"default": {**raw.get("model", {}),
                                "onnx": (raw.get("onnx") or {}).get("variants", {})}}
    name = override or raw.get("backbone") or next(iter(registry))
    if name not in registry:
        raise ValueError(f"{path}: unknown backbone '{name}'; "
                         f"available: {sorted(registry)}")
    return name, registry[name], sorted(registry)


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
    backbone: str = "default"               # selected entry of the backbones registry
    backbones: List[str] = field(default_factory=list)   # every entry the spec offers
    defaults: Dict[str, Any] = field(default_factory=dict)   # device, thresholds, ...
    outputs: Dict[str, Path] = field(default_factory=dict)    # figures dir, table file
    source: Optional[Path] = None           # the YAML this came from
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def onnx_path(self, ref: Optional[str] = None) -> Path:
        """Resolve a registered variant name (``fp16``) to an absolute Path.

        Registry names only, deliberately. The spec supplies preprocessing, the
        postprocessor and the class names for *this* backbone; letting a caller pass an
        arbitrary path would pair those with some other backbone's weights, and nothing
        downstream can detect it — box agreement stays high because both models are
        trained on the same data. Export destinations, which may legitimately name a file
        that does not exist yet, go through ``onnx_dest``.
        """
        if ref is None:
            ref = self.default_onnx
        if ref not in self.onnx:
            raise ValueError(f"unknown onnx variant '{ref}' for backbone "
                             f"'{self.backbone}'; available: {sorted(self.onnx)}")
        return self.onnx[ref]

    def onnx_dest(self, ref: Optional[str] = None) -> Path:
        """Export destination: a registered variant, or an explicit path for a one-off."""
        if ref is None or ref in self.onnx:
            return self.onnx_path(ref)
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

    @property
    def timing_cache(self) -> Optional[Path]:
        """TensorRT timing cache for this backbone, or ``None`` when caching is off.

        Reused across builds so a tactic is timed once instead of on every build. The
        path is scoped per backbone by ``{backbone}`` in the spec; correctness across
        machines is enforced at build time, where ``ignore_mismatch=False`` rejects a
        cache written by a different GPU or TensorRT version rather than trusting
        timings that were never measured on this device.
        """
        value = self.defaults.get("timing_cache")
        return resolve(value) if value else None


def spec_path(name_or_path) -> Path:
    """``"rtdetr"`` -> ``configs/rtdetr.yaml``; anything path-like is used as given."""
    candidate = Path(name_or_path)
    if candidate.suffix in (".yaml", ".yml") or candidate.exists():
        return resolve(candidate)
    return CONFIGS_DIR / f"{candidate}.yaml"


def load_spec(name_or_path, backbone: Optional[str] = None) -> ModelSpec:
    """Load and resolve a model spec by name (``rtdetr``) or path, for one backbone.

    ``backbone`` overrides the spec's own ``backbone:`` choice — everything downstream
    (checkpoint, ONNX variants, output paths) follows from that one selection.
    """
    path = spec_path(name_or_path)
    raw = _interpolate(_load_yaml(path))
    bb_name, entry, bb_names = _select_backbone(raw, path, backbone)
    raw = _fill(raw, backbone=bb_name)
    entry = _fill(entry, backbone=bb_name)

    # shared onnx contract, with any per-backbone override applied on top
    onnx_cfg = {**(raw.get("onnx") or {}),
                **{k: entry[k] for k in _CONTRACT_KEYS if k in entry}}
    variants = {k: resolve(v) for k, v in (entry.get("onnx") or {}).items()}
    if not variants:
        raise ValueError(f"{path}: backbone '{bb_name}' declares no onnx variants")
    default_onnx = onnx_cfg.get("default", next(iter(variants)))
    if default_onnx not in variants:
        raise ValueError(f"{path}: onnx.default '{default_onnx}' is not a variant of "
                         f"backbone '{bb_name}' ({sorted(variants)})")

    return ModelSpec(
        name=raw.get("name", path.stem),
        class_names=_load_class_names(raw["classes"]),
        input_name=onnx_cfg.get("input_name", "images"),
        output_names=list(onnx_cfg.get("output_names", [])),
        img_size=int(onnx_cfg.get("img_size", 640)),
        batch=int(onnx_cfg.get("batch", 1)),
        torch_config=resolve(entry["config"]),
        ckpt=resolve(entry["ckpt"]),
        onnx=variants,
        default_onnx=default_onnx,
        backbone=bb_name,
        backbones=bb_names,
        defaults=dict(raw.get("defaults", {})),
        outputs={k: resolve(v) for k, v in (raw.get("outputs") or {}).items()},
        source=path,
        raw=raw,
    )
