"""Load YAML configs with inheritance and dump the resolved object."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from quant_fund.config.models import AppConfig


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge overlay onto base. Overlay wins on conflicts."""
    out = dict(base)
    for key, value in overlay.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config {path} must be a mapping")
    return raw


def load_config(
    path: str | Path,
    *,
    _seen: frozenset[Path] | None = None,
    _root: Path | None = None,
) -> AppConfig:
    """Load a YAML file, following `inherit:` relative to the file's directory."""
    path = Path(path).resolve()
    root = path.parent if _root is None else _root
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Config inheritance escapes config root: {path}") from exc
    seen = _seen or frozenset()
    if path in seen:
        raise ValueError(f"Config inheritance cycle at {path}")
    data = _read_yaml(path)
    inherit = data.pop("inherit", None)
    if inherit:
        parent_path = (path.parent / inherit).resolve()
        parent = load_config(parent_path, _seen=seen | {path}, _root=root)
        merged = deep_merge(parent.model_dump(mode="python"), data)
        return AppConfig.model_validate(merged)
    return AppConfig.model_validate(data)


def dump_resolved(config: AppConfig, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(config.dump(), indent=2, sort_keys=True))
