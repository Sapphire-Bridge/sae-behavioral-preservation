#!/usr/bin/env python3
from __future__ import annotations

import math
import shlex
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CommandSpec:
    name: str
    argv: tuple[str, ...]
    artifact_kind: str = "core"


@dataclass(frozen=True)
class CommandRecord:
    name: str
    argv: tuple[str, ...]
    display_command: str
    started_at_utc: str
    ended_at_utc: str
    exit_code: int
    generated_files: tuple[str, ...]
    artifact_kind: str = "core"
    stdout_log: str = ""
    stderr_log: str = ""
    stderr_tail: tuple[str, ...] = ()


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    observed: Any
    expected: Any
    reference_path: str
    note: str = ""
    artifact_kind: str = "core"


@dataclass(frozen=True)
class MissingArtifact:
    path: Path
    artifact_kind: str = "core"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _walk_files(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}


def _display_arg(arg: str, run_root: Path) -> str:
    if arg == sys.executable:
        return "python"
    path = Path(arg)
    if path.is_absolute():
        try:
            return f"$RUN_ROOT/{path.relative_to(run_root).as_posix()}"
        except ValueError:
            pass
        try:
            return path.relative_to(ROOT).as_posix()
        except ValueError:
            return arg
    return arg


def _display_command(argv: Iterable[str], run_root: Path) -> str:
    return " ".join(shlex.quote(_display_arg(arg, run_root)) for arg in argv)


def _status_label(status: str) -> str:
    mapping = {
        "pass": "PASS",
        "fail": "FAIL",
        "warn": "WARN",
    }
    normalized = str(status).lower()
    if normalized in mapping:
        return mapping[normalized]
    raw = str(status).strip()
    if not raw:
        return "INVALID"
    return f"INVALID({raw})"


def _is_failure_status(status: str) -> bool:
    return str(status).lower() not in {"pass", "warn"}


def _per_layer_entry(summary: dict[str, Any], layer: int) -> dict[str, Any]:
    per_layer = summary["per_layer"]
    if isinstance(per_layer, dict):
        return per_layer[str(layer)]
    for row in per_layer:
        if int(row["layer"]) == int(layer):
            return row
    raise KeyError(f"Layer {layer} missing")


def _compare_exact(name: str, observed: Any, expected: Any, reference_path: str) -> CheckResult:
    return CheckResult(
        name=name,
        status="pass" if observed == expected else "fail",
        observed=observed,
        expected=expected,
        reference_path=reference_path,
    )


def _compare_close(
    name: str,
    observed: float,
    expected: float,
    reference_path: str,
    *,
    atol: float,
) -> CheckResult:
    ok = math.isclose(float(observed), float(expected), abs_tol=atol, rel_tol=0.0)
    return CheckResult(
        name=name,
        status="pass" if ok else "fail",
        observed=observed,
        expected=expected,
        reference_path=reference_path,
        note=f"abs_tol={atol}",
    )
