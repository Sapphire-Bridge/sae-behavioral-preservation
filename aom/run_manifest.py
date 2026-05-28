from __future__ import annotations

import json
import math
import numbers
from collections.abc import Mapping as ABCMapping, Sequence as ABCSequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


MANIFEST_VERSION = "1.0"
DEFAULT_REDACTED_FLAGS: set[str] = {"--system_prompt", "--system_prompt_file"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def redact_argv(argv: Sequence[str], *, redacted_flags: set[str] | None = None) -> list[str]:
    """
    Redact sensitive command-line arguments.

    This is intentionally small and explicit: only flags in `redacted_flags` are redacted.
    """
    flags = DEFAULT_REDACTED_FLAGS if redacted_flags is None else set(redacted_flags)
    out: list[str] = []
    i = 0
    while i < len(argv):
        a = str(argv[i])
        if any(a.startswith(f"{flag}=") for flag in flags):
            flag, _eq, _val = a.partition("=")
            out.append(f"{flag}=<redacted>")
            i += 1
            continue
        if a in flags:
            out.append(a)
            if i + 1 < len(argv):
                out.append("<redacted>")
                i += 2
                continue
            i += 1
            continue
        out.append(a)
        i += 1
    return out


def build_run_manifest(
    *,
    argv: Sequence[str],
    results_row: Mapping[str, Any],
    dataset_manifest_path: str | None = None,
    csv_path: str | None = None,
    csv_sha256: str | None = None,
    csv_n_rows: int | None = None,
    redacted_flags: set[str] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "created_at_utc": _utc_now_iso(),
        "argv_redacted": redact_argv(argv, redacted_flags=redacted_flags),
        "results_row": dict(results_row),
        "dataset_manifest_path": dataset_manifest_path,
    }
    if csv_path is not None:
        out["csv_path"] = str(csv_path)
    if csv_sha256 is not None:
        out["csv_sha256"] = str(csv_sha256)
    if csv_n_rows is not None:
        out["csv_n_rows"] = int(csv_n_rows)
    return out


def _to_jsonable(x: Any) -> Any:
    if x is None or isinstance(x, (str, bool)):
        return x
    if isinstance(x, numbers.Integral) and not isinstance(x, bool):
        return int(x)
    if isinstance(x, numbers.Real) and not isinstance(x, bool):
        xf = float(x)
        return xf if math.isfinite(xf) else None
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, ABCMapping):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, ABCSequence) and not isinstance(x, (str, bytes, bytearray)):
        return [_to_jsonable(v) for v in x]
    return str(x)


def write_run_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Keep manifests strict-JSON: replace NaN/Inf with null to avoid non-standard JSON tokens.
    safe = _to_jsonable(dict(manifest))
    p.write_text(json.dumps(safe, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def read_run_manifest(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Run manifest must be a JSON object, got {type(obj).__name__}")
    return obj


def validate_run_manifest(manifest: Mapping[str, Any]) -> None:
    if not isinstance(manifest, Mapping):
        raise ValueError(f"Run manifest must be a mapping, got {type(manifest).__name__}")

    version = manifest.get("manifest_version", None)
    if version != MANIFEST_VERSION:
        raise ValueError(f"Unsupported run manifest version: {version!r} (expected {MANIFEST_VERSION!r})")

    created_at = manifest.get("created_at_utc", None)
    if not isinstance(created_at, str) or not created_at.strip():
        raise ValueError("Run manifest missing non-empty 'created_at_utc' string")

    argv_redacted = manifest.get("argv_redacted", None)
    if not isinstance(argv_redacted, list) or not all(isinstance(x, str) for x in argv_redacted):
        raise ValueError("Run manifest 'argv_redacted' must be a list[str]")

    results_row = manifest.get("results_row", None)
    if not isinstance(results_row, dict):
        raise ValueError("Run manifest 'results_row' must be an object")

    dataset_manifest_path = manifest.get("dataset_manifest_path", None)
    if dataset_manifest_path is not None and not isinstance(dataset_manifest_path, str):
        raise ValueError("Run manifest 'dataset_manifest_path' must be a string or null")

    csv_path = manifest.get("csv_path", None)
    if csv_path is not None and not isinstance(csv_path, str):
        raise ValueError("Run manifest 'csv_path' must be a string or null")

    csv_sha256 = manifest.get("csv_sha256", None)
    if csv_sha256 is not None and not isinstance(csv_sha256, str):
        raise ValueError("Run manifest 'csv_sha256' must be a string or null")

    csv_n_rows = manifest.get("csv_n_rows", None)
    if csv_n_rows is not None and not isinstance(csv_n_rows, int):
        raise ValueError("Run manifest 'csv_n_rows' must be an int or null")

    run_status = manifest.get("run_status", None)
    if run_status not in {"PASS", "WARN", "FAIL"}:
        raise ValueError("Run manifest missing valid 'run_status' (PASS/WARN/FAIL)")

    run_status_reasons = manifest.get("run_status_reasons", None)
    if not isinstance(run_status_reasons, list) or not all(isinstance(x, str) for x in run_status_reasons):
        raise ValueError("Run manifest 'run_status_reasons' must be a list[str]")

    run_summary = manifest.get("run_summary", None)
    if not isinstance(run_summary, dict):
        raise ValueError("Run manifest 'run_summary' must be an object")

    def _req_int(key: str) -> int:
        v = run_summary.get(key, None)
        if not isinstance(v, int):
            raise ValueError(f"Run manifest 'run_summary.{key}' must be an int")
        return v

    _req_int("attempted")
    _req_int("succeeded")
    _req_int("failed")
    _req_int("skipped")
    _req_int("invalid")

    for rate_key in ("fail_rate", "skip_rate", "invalid_rate"):
        rv = run_summary.get(rate_key, None)
        if rv is not None and not isinstance(rv, (int, float)):
            raise ValueError(f"Run manifest 'run_summary.{rate_key}' must be a number or null")

    def _validate_top_list(key: str, *, required_keys: set[str]) -> None:
        xs = run_summary.get(key, None)
        if not isinstance(xs, list):
            raise ValueError(f"Run manifest 'run_summary.{key}' must be a list")
        for x in xs:
            if not isinstance(x, dict):
                raise ValueError(f"Run manifest 'run_summary.{key}' items must be objects")
            missing = required_keys - set(x.keys())
            if missing:
                raise ValueError(f"Run manifest 'run_summary.{key}' item missing keys: {sorted(missing)!r}")

    _validate_top_list("top_failure_types", required_keys={"type", "count", "example"})
    _validate_top_list("top_skip_types", required_keys={"type", "count", "example"})
    _validate_top_list("top_invalid_reasons", required_keys={"reason", "count"})

    inv = run_summary.get("invariant_problems", None)
    if not isinstance(inv, list) or not all(isinstance(x, str) for x in inv):
        raise ValueError("Run manifest 'run_summary.invariant_problems' must be a list[str]")

    attempt_unit = manifest.get("attempt_unit", None)
    if attempt_unit is not None and not isinstance(attempt_unit, str):
        raise ValueError("Run manifest 'attempt_unit' must be a string or null")

    attempted_expected = manifest.get("attempted_expected", None)
    if attempted_expected is not None and not isinstance(attempted_expected, int):
        raise ValueError("Run manifest 'attempted_expected' must be an int or null")

    data_error_policy = manifest.get("data_error_policy", None)
    if data_error_policy is not None and data_error_policy not in {"raise", "warn_skip"}:
        raise ValueError("Run manifest 'data_error_policy' must be one of raise/warn_skip or null")

    strict_data = manifest.get("strict_data", None)
    if strict_data is not None and not isinstance(strict_data, bool):
        raise ValueError("Run manifest 'strict_data' must be a bool or null")

    device_backend = manifest.get("device_backend", None)
    if device_backend is not None and not isinstance(device_backend, str):
        raise ValueError("Run manifest 'device_backend' must be a string or null")

    versions = manifest.get("versions", None)
    if versions is not None:
        if not isinstance(versions, dict):
            raise ValueError("Run manifest 'versions' must be an object or null")
        for k, v in versions.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise ValueError("Run manifest 'versions' must be a mapping of string keys to string values")

    repro = manifest.get("repro", None)
    if repro is not None:
        if not isinstance(repro, dict):
            raise ValueError("Run manifest 'repro' must be an object or null")

        seed = repro.get("seed", None)
        if not isinstance(seed, int):
            raise ValueError("Run manifest 'repro.seed' must be an int")

        determinism_requested = repro.get("determinism_requested", None)
        if determinism_requested not in {"strict", "best_effort", "off"}:
            raise ValueError("Run manifest 'repro.determinism_requested' must be one of strict/best_effort/off")

        determinism_enforced = repro.get("determinism_enforced", None)
        if not isinstance(determinism_enforced, bool):
            raise ValueError("Run manifest 'repro.determinism_enforced' must be a bool")

        determinism_reason = repro.get("determinism_reason", None)
        if determinism_reason is not None and not isinstance(determinism_reason, str):
            raise ValueError("Run manifest 'repro.determinism_reason' must be a string or null")

        repro_device_backend = repro.get("device_backend", None)
        if repro_device_backend is not None and not isinstance(repro_device_backend, str):
            raise ValueError("Run manifest 'repro.device_backend' must be a string or null")
        if device_backend is not None and repro_device_backend is not None and str(device_backend) != str(repro_device_backend):
            raise ValueError("Run manifest 'device_backend' must match 'repro.device_backend' when both are present")

        seeds = repro.get("seeds", None)
        if seeds is not None:
            if not isinstance(seeds, list) or not all(isinstance(x, int) for x in seeds):
                raise ValueError("Run manifest 'repro.seeds' must be a list[int] or null")

        cublas_workspace_config = repro.get("cublas_workspace_config", None)
        if cublas_workspace_config is not None and not isinstance(cublas_workspace_config, str):
            raise ValueError("Run manifest 'repro.cublas_workspace_config' must be a string or null")

        for opt_key in (
            "torch_deterministic_algorithms",
            "torch_deterministic_warn_only",
            "cudnn_deterministic",
            "cudnn_benchmark",
        ):
            ov = repro.get(opt_key, None)
            if ov is not None and not isinstance(ov, bool):
                raise ValueError(f"Run manifest 'repro.{opt_key}' must be a bool or null")

    datasets = manifest.get("datasets", None)
    if datasets is not None:
        if not isinstance(datasets, dict):
            raise ValueError("Run manifest 'datasets' must be an object or null")
        for dataset_key, dm in datasets.items():
            if not isinstance(dataset_key, str) or not dataset_key.strip():
                raise ValueError("Run manifest 'datasets' keys must be non-empty strings")
            if not isinstance(dm, dict):
                raise ValueError("Run manifest 'datasets' values must be objects")

            role = dm.get("role", None)
            if not isinstance(role, str) or not role.strip():
                raise ValueError("Run manifest dataset missing non-empty 'role' string")
            if role != dataset_key:
                raise ValueError("Run manifest dataset 'role' must match its key in 'datasets'")

            path = dm.get("path", None)
            if not isinstance(path, str) or not path.strip():
                raise ValueError("Run manifest dataset missing non-empty 'path' string")

            sha256 = dm.get("sha256", None)
            if not isinstance(sha256, str):
                raise ValueError("Run manifest dataset 'sha256' must be a string")
            if sha256 and len(sha256) != 64:
                raise ValueError("Run manifest dataset 'sha256' must be 64 hex chars (or empty)")

            size_bytes = dm.get("size_bytes", None)
            if not isinstance(size_bytes, int):
                raise ValueError("Run manifest dataset 'size_bytes' must be an int")

            n_total = dm.get("n_rows_total", None)
            n_valid = dm.get("n_rows_valid", None)
            n_invalid = dm.get("n_rows_invalid", None)
            if not isinstance(n_total, int) or not isinstance(n_valid, int) or not isinstance(n_invalid, int):
                raise ValueError("Run manifest dataset row counts must be ints")
            if n_total != n_valid + n_invalid:
                raise ValueError("Run manifest dataset must satisfy n_rows_total == n_rows_valid + n_rows_invalid")

            schema_name = dm.get("schema_name", None)
            if not isinstance(schema_name, str) or not schema_name.strip():
                raise ValueError("Run manifest dataset missing non-empty 'schema_name' string")

            schema_version = dm.get("schema_version", None)
            if schema_version is not None and not isinstance(schema_version, str):
                raise ValueError("Run manifest dataset 'schema_version' must be a string or null")

            ep = dm.get("error_policy", None)
            if ep not in {"raise", "warn_skip"}:
                raise ValueError("Run manifest dataset 'error_policy' must be one of raise/warn_skip")

            invalid_samples = dm.get("invalid_samples", None)
            if not isinstance(invalid_samples, list):
                raise ValueError("Run manifest dataset 'invalid_samples' must be a list")
            for s in invalid_samples:
                if not isinstance(s, dict):
                    raise ValueError("Run manifest dataset 'invalid_samples' items must be objects")
                line = s.get("line", None)
                if line is not None and not isinstance(line, int):
                    raise ValueError("Run manifest dataset invalid sample 'line' must be an int or null")
                et = s.get("error_type", None)
                if not isinstance(et, str) or not et.strip():
                    raise ValueError("Run manifest dataset invalid sample missing non-empty 'error_type'")
                msg = s.get("message", None)
                if not isinstance(msg, str):
                    raise ValueError("Run manifest dataset invalid sample 'message' must be a string")
