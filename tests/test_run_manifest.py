from __future__ import annotations

import json
from pathlib import Path

from aom.run_manifest import MANIFEST_VERSION, build_run_manifest, read_run_manifest, redact_argv, validate_run_manifest, write_run_manifest


def _add_minimal_run_fields(manifest: dict) -> dict:
    manifest["run_status"] = "PASS"
    manifest["run_status_reasons"] = []
    manifest["run_summary"] = {
        "attempted": 1,
        "succeeded": 1,
        "failed": 0,
        "skipped": 0,
        "invalid": 0,
        "fail_rate": 0.0,
        "skip_rate": 0.0,
        "invalid_rate": 0.0,
        "top_failure_types": [],
        "top_skip_types": [],
        "top_invalid_reasons": [],
        "invariant_problems": [],
    }
    return manifest


def test_redact_argv_redacts_system_prompt_value():
    argv = ["python", "scripts/clt_raw_comparability.py", "--system_prompt", "SECRET", "--x", "y"]
    redacted = redact_argv(argv)
    assert "SECRET" not in redacted
    assert redacted[0:4] == ["python", "scripts/clt_raw_comparability.py", "--system_prompt", "<redacted>"]


def test_redact_argv_redacts_equals_form():
    argv = ["python", "scripts/clt_raw_comparability.py", "--system_prompt=SECRET"]
    redacted = redact_argv(argv)
    assert redacted == ["python", "scripts/clt_raw_comparability.py", "--system_prompt=<redacted>"]


def test_redact_argv_redacts_system_prompt_file():
    argv = ["python", "scripts/clt_raw_comparability.py", "--system_prompt_file", "/tmp/secret.txt"]
    redacted = redact_argv(argv)
    assert "/tmp/secret.txt" not in redacted
    assert redacted[0:4] == ["python", "scripts/clt_raw_comparability.py", "--system_prompt_file", "<redacted>"]


def test_redact_argv_redacts_system_prompt_file_equals_form():
    argv = ["python", "scripts/clt_raw_comparability.py", "--system_prompt_file=/tmp/secret.txt"]
    redacted = redact_argv(argv)
    assert redacted == ["python", "scripts/clt_raw_comparability.py", "--system_prompt_file=<redacted>"]


def test_run_manifest_roundtrip(tmp_path: Path):
    argv = ["python", "scripts/clt_raw_comparability.py", "--system_prompt", "SECRET", "--seed", "0"]
    row = {"model": "local", "seed": 0}
    manifest = _add_minimal_run_fields(build_run_manifest(argv=argv, results_row=row, dataset_manifest_path=None))
    assert manifest["manifest_version"] == MANIFEST_VERSION
    assert isinstance(manifest.get("created_at_utc"), str)
    assert isinstance(manifest.get("argv_redacted"), list)
    assert isinstance(manifest.get("results_row"), dict)

    s = json.dumps(manifest, sort_keys=True)
    assert "SECRET" not in s

    out_path = tmp_path / "run.manifest.json"
    write_run_manifest(out_path, manifest)
    loaded = read_run_manifest(out_path)
    validate_run_manifest(loaded)


def test_write_run_manifest_serializes_non_jsonable_values(tmp_path: Path):
    class _Obj:
        def __str__(self) -> str:
            return "OBJ"

    manifest = _add_minimal_run_fields(build_run_manifest(argv=["python"], results_row={"x": _Obj()}, dataset_manifest_path=None))
    out_path = tmp_path / "run.manifest.json"
    write_run_manifest(out_path, manifest)
    loaded = read_run_manifest(out_path)
    validate_run_manifest(loaded)
    assert loaded["results_row"]["x"] == "OBJ"


def test_write_run_manifest_replaces_nan_and_inf_with_null(tmp_path: Path):
    manifest = _add_minimal_run_fields(build_run_manifest(
        argv=["python"],
        results_row={"x": float("nan"), "y": float("inf"), "z": -float("inf")},
        dataset_manifest_path=None,
    ))
    out_path = tmp_path / "run.manifest.json"
    write_run_manifest(out_path, manifest)
    raw = out_path.read_text(encoding="utf-8")
    assert "NaN" not in raw
    assert "Infinity" not in raw

    loaded = read_run_manifest(out_path)
    validate_run_manifest(loaded)
    assert loaded["results_row"]["x"] is None
    assert loaded["results_row"]["y"] is None
    assert loaded["results_row"]["z"] is None
