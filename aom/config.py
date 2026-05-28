from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def sha256_text(text: str) -> str:
    h = hashlib.sha256()
    h.update(str(text).encode("utf-8"))
    return h.hexdigest()


def load_config(path: str | Path) -> dict[str, Any]:
    """
    Load a JSON or YAML config file.

    YAML parsing uses PyYAML (`yaml.safe_load`). This dependency is already required by
    Transformers, but we keep the import local for clarity.
    """
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        obj = json.loads(raw)
    else:
        import yaml  # type: ignore[import-not-found]

        obj = yaml.safe_load(raw)
    if obj is None:
        return {}
    if not isinstance(obj, Mapping):
        raise ValueError(f"Config file must parse to an object/mapping, got {type(obj).__name__}")
    return {str(k): v for k, v in dict(obj).items()}


def validate_config_keys(*, config: Mapping[str, Any], allowed: set[str]) -> None:
    bad = sorted(str(k) for k in config.keys() if str(k) not in allowed)
    if bad:
        raise ValueError(f"Unknown config keys: {bad!r}")


def resolve_relative_paths(config: Mapping[str, Any], *, base_dir: str | Path) -> dict[str, Any]:
    """
    Resolve relative filesystem paths inside a config.

    Heuristic: keys ending with `_path` plus `results_dir` are treated as paths when they are strings.
    """
    base = Path(base_dir)
    out: dict[str, Any] = dict(config)
    for k, v in list(out.items()):
        if not isinstance(v, str) or not v.strip():
            continue
        key = str(k)
        if not (key.endswith("_path") or key in {"results_dir"}):
            continue
        p = Path(v)
        if p.is_absolute():
            continue
        out[key] = str((base / p).resolve())
    return out
