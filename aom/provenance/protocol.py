from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

from aom.config import load_config


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_sha256_hex(sha: str) -> str:
    s = str(sha or "").strip().lower()
    if not s:
        return ""
    if len(s) != 64 or any(c not in "0123456789abcdef" for c in s):
        raise ValueError("--protocol_sha256 must be 64 lowercase hex characters")
    return s


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return bool(v)
    if isinstance(v, str):
        s = str(v).strip().lower()
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off"}:
            return False
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return bool(v)
    raise ValueError(f"Could not coerce value to bool: {v!r}")


def _get_nested(cfg: Mapping[str, Any], path: Sequence[str]) -> tuple[bool, Any]:
    cur: Any = cfg
    for key in path:
        if not isinstance(cur, Mapping) or str(key) not in cur:
            return False, None
        cur = cur[str(key)]
    return True, cur


@dataclass(frozen=True)
class ProtocolProvenance:
    protocol_path: str
    protocol_sha256: str
    protocol_sha256_verified: bool
    protocol_sha256_source: Literal["computed_from_path", "provided_unverified", "empty"]
    protocol_name: str
    protocol_version: str
    protocol_prereg_tag: str
    protocol_config: dict[str, Any]


@dataclass(frozen=True)
class ProtocolArgBinding:
    arg_name: str
    protocol_path: tuple[str, ...]
    coerce: Callable[[Any], Any] | None = None
    required: bool = True


def resolve_protocol_provenance(
    *,
    protocol_path_raw: str,
    protocol_sha256_raw: str,
    require_path_for_sha: bool = False,
    require_frozen: bool = False,
) -> ProtocolProvenance:
    """
    Resolve and verify protocol provenance for run manifests.

    Behavior:
    - Validates hash format if provided.
    - Validates protocol path exists and is a file when provided.
    - If both path+hash are provided, requires hash(path) == provided hash.
    - If path is provided and hash is empty, computes hash(path).
    - If hash is provided without a path:
      - allowed when require_path_for_sha=False (marked unverified)
      - rejected when require_path_for_sha=True.
    """
    protocol_path = str(protocol_path_raw or "").strip()
    provided_sha = _validate_sha256_hex(str(protocol_sha256_raw or ""))
    protocol_config: dict[str, Any] = {}
    protocol_name = ""
    protocol_version = ""
    protocol_prereg_tag = ""
    protocol_status = ""

    if not protocol_path:
        if bool(require_frozen):
            raise ValueError("--protocol_path is required when require_frozen=True")
        if provided_sha and bool(require_path_for_sha):
            raise ValueError("--protocol_sha256 requires --protocol_path (hash cannot be verified without a file path)")
        protocol_sha256_source: Literal["computed_from_path", "provided_unverified", "empty"] = (
            "provided_unverified" if provided_sha else "empty"
        )
        return ProtocolProvenance(
            protocol_path="",
            protocol_sha256=str(provided_sha),
            protocol_sha256_verified=False,
            protocol_sha256_source=protocol_sha256_source,
            protocol_name="",
            protocol_version="",
            protocol_prereg_tag="",
            protocol_config={},
        )

    p = Path(protocol_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"--protocol_path not found: {str(p)}")
    if p.is_dir():
        raise IsADirectoryError(f"--protocol_path must be a file, got directory: {str(p)}")

    computed_sha = _sha256_file(p)
    if provided_sha and provided_sha != computed_sha:
        raise ValueError(
            f"--protocol_sha256 mismatch for {str(p)}: provided={provided_sha} computed={computed_sha}"
        )
    final_sha = str(provided_sha or computed_sha)

    try:
        cfg = load_config(p)
    except ValueError as e:
        msg = str(e)
        if "object/mapping" in msg or "mapping" in msg:
            raise ValueError(f"Protocol config must be a mapping/object: {e}") from e
        raise
    if not isinstance(cfg, Mapping):
        raise ValueError(f"Protocol config must be a mapping/object, got {type(cfg).__name__}")
    protocol_config = {str(k): v for k, v in dict(cfg).items()}
    proto_block = cfg.get("protocol", None)
    if isinstance(proto_block, Mapping):
        protocol_name = str(proto_block.get("name", "") or "").strip()
        protocol_version = str(proto_block.get("version", "") or "").strip()
        protocol_prereg_tag = str(proto_block.get("prereg_tag", "") or "").strip()
        protocol_status = str(proto_block.get("status", "") or "").strip().lower()
    if bool(require_frozen):
        if not isinstance(proto_block, Mapping):
            raise ValueError("Protocol config missing required 'protocol' mapping for require_frozen=True")
        missing: list[str] = []
        if not protocol_name:
            missing.append("protocol.name")
        if not protocol_version:
            missing.append("protocol.version")
        if not protocol_prereg_tag:
            missing.append("protocol.prereg_tag")
        if not protocol_status:
            missing.append("protocol.status")
        if missing:
            raise ValueError("Frozen protocol is missing required fields: " + ", ".join(missing))
        if protocol_status != "frozen":
            raise ValueError(f"Frozen protocol requires protocol.status='frozen', got {protocol_status!r}")

    return ProtocolProvenance(
        protocol_path=str(p),
        protocol_sha256=str(final_sha),
        protocol_sha256_verified=True,
        protocol_sha256_source="computed_from_path",
        protocol_name=str(protocol_name),
        protocol_version=str(protocol_version),
        protocol_prereg_tag=str(protocol_prereg_tag),
        protocol_config=protocol_config,
    )


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_seq(v: Any) -> bool:
    return isinstance(v, Sequence) and not isinstance(v, (str, bytes, bytearray))


def _values_equal(actual: Any, expected: Any) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return isinstance(actual, bool) and isinstance(expected, bool) and actual is expected
    if isinstance(actual, str) and isinstance(expected, str):
        return actual.strip() == expected.strip()
    if _is_number(actual) and _is_number(expected):
        if isinstance(actual, float) or isinstance(expected, float):
            return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12)
        return int(actual) == int(expected)
    if _is_seq(actual) and _is_seq(expected):
        a_seq = list(actual)
        e_seq = list(expected)
        if len(a_seq) != len(e_seq):
            return False
        return all(_values_equal(a, e) for a, e in zip(a_seq, e_seq))
    return actual == expected


def _typed(v: Any) -> str:
    return f"{v!r} (type={type(v).__name__})"


def enforce_protocol_bindings(
    *,
    args: Any,
    protocol_config: Mapping[str, Any],
    bindings: Sequence[ProtocolArgBinding],
    context: str,
    protocol_path: str = "",
    protocol_sha256: str = "",
) -> None:
    """
    Enforce protocol-configured knobs against runtime args.

    For each binding, runtime arg value must match protocol value.
    Missing protocol keys are errors unless binding.required=False.
    """
    if not bindings:
        return

    mismatches: list[str] = []
    for b in bindings:
        ok, expected_raw = _get_nested(protocol_config, b.protocol_path)
        if not ok:
            if bool(b.required):
                mismatches.append(
                    f"--{b.arg_name} requires protocol key {'.'.join(b.protocol_path)} but it is missing"
                )
            continue
        if not hasattr(args, str(b.arg_name)):
            mismatches.append(
                f"--{b.arg_name} missing from args (protocol {'.'.join(b.protocol_path)}={_typed(expected_raw)})"
            )
            continue
        actual_raw = getattr(args, str(b.arg_name))
        try:
            if b.coerce is None:
                expected = expected_raw
                actual = actual_raw
            else:
                expected = b.coerce(expected_raw)
                actual = b.coerce(actual_raw)
        except Exception as e:
            mismatches.append(
                f"--{b.arg_name} coercion failed for protocol {'.'.join(b.protocol_path)} "
                f"(actual={_typed(actual_raw)}, protocol={_typed(expected_raw)}): {type(e).__name__}: {e}"
            )
            continue
        if not _values_equal(actual, expected):
            mismatches.append(
                f"--{b.arg_name} mismatch: actual={_typed(actual)} "
                f"!= protocol {'.'.join(b.protocol_path)}={_typed(expected)}"
            )

    if mismatches:
        header = f"Protocol enforcement failed ({context})"
        if str(protocol_path).strip() or str(protocol_sha256).strip():
            header += f" [protocol_path={str(protocol_path)!r}, protocol_sha256={str(protocol_sha256)!r}]"
        joined = "; ".join(mismatches)
        raise ValueError(f"{header}: {joined}")


# Re-export bool coercer for callers.
coerce_bool = _as_bool
