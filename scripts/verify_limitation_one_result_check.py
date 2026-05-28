#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.limitation_requirements import (
    limitation_comparability_summary_path,
    load_public_comparability_reference,
    relative_to_root,
)
from scripts.limitation_surface import _portable_repo_path, load_json
from scripts.reproduction_common import (
    CheckResult,
    MissingArtifact,
    _compare_close,
    _compare_exact,
    _is_failure_status,
    _status_label,
)


EXPECTED_OUTPUTS = ("comparability/l4/comparability.summary.json",)
NUMERIC_METRICS = (
    "fidelity_cosine",
    "fidelity_rel_mse",
    "raw_effect",
    "sae_effect",
    "sae_minus_raw",
    "pca_effect",
)


def _nested_get(obj: Any, dotted: str) -> Any:
    cur = obj
    for part in dotted.split("."):
        cur = cur[part]
    return cur


def _nested_get_optional(obj: Any, dotted: str, default: Any = None) -> Any:
    cur = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _is_comparability_argv(argv: object) -> bool:
    if not isinstance(argv, list):
        return False
    for arg in argv:
        raw = str(arg).strip()
        if not raw:
            continue
        if raw.endswith("scripts/clt_raw_comparability.py"):
            return True
        if Path(raw).name == "clt_raw_comparability.py":
            return True
    return False


def _run_device(run_root: Path) -> str:
    log_path = run_root / "one_result_check_log.json"
    if not log_path.exists():
        return "cpu"
    try:
        payload = json.loads(log_path.read_text(encoding="utf-8"))
        records = payload.get("records", [])
        if not records:
            return "cpu"
        for record in reversed(records):
            argv = record.get("argv", [])
            if not _is_comparability_argv(argv):
                continue
            if "--device" not in argv:
                continue
            device = str(argv[argv.index("--device") + 1]).strip().lower()
            return device or "cpu"
        return "cpu"
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return "cpu"


def _comparison_value(summary: dict[str, Any], field: str) -> Any:
    if field == "sae":
        sae = dict(_nested_get_optional(summary, field, {}) or {})
        sae["repo"] = _portable_repo_path(sae.get("repo", ""))
        return sae
    if field == "analysis_policy":
        return dict(summary.get("analysis_policy") or {})
    if field in {"run.device", "run.torch_dtype", "run.source_schema_version"}:
        return str(_nested_get_optional(summary, field, "") or "")
    return _nested_get(summary, field)


def verify_run(run_root: Path) -> tuple[list[MissingArtifact], list[CheckResult]]:
    missing = [MissingArtifact(run_root / rel, artifact_kind="core") for rel in EXPECTED_OUTPUTS if not (run_root / rel).exists()]
    if missing:
        return missing, []

    run_device = _run_device(run_root)
    run_summary = load_json(run_root / EXPECTED_OUTPUTS[0])
    ref_summary = load_public_comparability_reference(4)
    reference_path = relative_to_root(limitation_comparability_summary_path(4))

    checks: list[CheckResult] = []
    # Quickcheck identity is intentionally narrower than the full source surface.
    # The public oracle is policy-/profile-driven: analysis policy, build
    # identity, and the public evaluation-cardinality field stay exact.
    # `counts.n_pairs_analysis_included` is identity because it defines the
    # estimand surface, while row-level counts and invariant-fail counts remain
    # implementation diagnostics that may legitimately drift across backends or
    # numerical reductions.
    exact_fields = (
        "summary_schema_version",
        "publication_profile",
        "execution_profile_id",
        "layer",
        "model",
        "dataset",
        "sae",
        "analysis_policy",
        "run.seed",
        "run.bootstrap_seed",
        "run.bootstrap_n",
        "run.ci",
        "run.device",
        "run.torch_dtype",
        "run.source_schema_version",
        "counts.n_pairs_analysis_included",
    )
    for field in exact_fields:
        check = (
            _compare_exact(
                f"limitation {field}",
                _comparison_value(run_summary, field),
                _comparison_value(ref_summary, field),
                reference_path,
            )
        )
        if field in {"run.device", "run.torch_dtype"} and run_device != "cpu" and check.status == "fail":
            check = replace(
                check,
                status="warn",
                note=f"accelerator run on {run_device}; build-field drift is advisory only",
            )
        checks.append(check)

    for metric_name in NUMERIC_METRICS:
        for part in ("mean", "ci_low", "ci_high"):
            check = (
                _compare_close(
                    f"limitation metrics.{metric_name}.{part}",
                    float(run_summary["metrics"][metric_name][part]),
                    float(ref_summary["metrics"][metric_name][part]),
                    reference_path,
                    atol=1e-4,
                )
            )
            if run_device != "cpu" and check.status == "fail":
                check = replace(
                    check,
                    status="warn",
                    note=f"accelerator run on {run_device}; numeric drift is advisory only",
                )
            checks.append(check)

    return missing, checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the public layer-4 limitation quickcheck against the checked-in reference.")
    parser.add_argument("--run_root", required=True, help="Directory containing the quickcheck outputs.")
    args = parser.parse_args(argv)

    run_root = Path(args.run_root).expanduser().resolve()
    missing, checks = verify_run(run_root)
    for artifact in missing:
        print(f"[missing] {artifact}")
    for check in checks:
        note = f"; {check.note}" if check.note else ""
        print(
            f"[{_status_label(check.status).lower()}] {check.name}: observed={check.observed!r}; "
            f"expected={check.expected!r}{note}; reference={check.reference_path}"
        )

    if missing or any(_is_failure_status(check.status) for check in checks):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
