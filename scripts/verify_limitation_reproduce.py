#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.limitation_requirements as limitation_requirements
from scripts.limitation_analysis_policy import ANALYSIS_POLICY_VERSION, LIMITATION_REFERENCE_BUILD_PROFILE
from scripts.limitation_surface import (
    build_limitation_robustness,
    build_limitation_robustness_from_case_target,
    load_json,
    read_csv_rows,
)


EXACT_FIELDS = (
    "test",
    "comparison",
    "layer",
    "unit",
    "heldout_target",
    "n_units",
    "n_pairs",
    "n_targets",
    "n_permutations",
    "permutation_mode",
    "seed",
    "target_means",
    "leave_one_target_means",
    "loto_negative_count",
    "loto_n",
)
FLOAT_FIELDS = (
    "observed_mean",
    "p_two_sided",
    "loto_min",
    "loto_max",
)


def _normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            str(row["test"]),
            str(row["comparison"]),
            str(row["unit"]),
            str(row.get("heldout_target", "")),
        ),
    )


def _compare_rows(name: str, observed: list[dict[str, Any]], expected: list[dict[str, str]]) -> list[str]:
    observed_rows = _normalize_rows(observed)
    expected_rows = _normalize_rows(expected)
    failures: list[str] = []
    if len(observed_rows) != len(expected_rows):
        failures.append(f"{name}: row count mismatch observed={len(observed_rows)} expected={len(expected_rows)}")
        return failures
    for idx, (obs, exp) in enumerate(zip(observed_rows, expected_rows), start=1):
        row_label = f"{name} row {idx} ({exp.get('test', '')}/{exp.get('heldout_target', '')})"
        for field in EXACT_FIELDS:
            observed_value = str(obs[field])
            expected_value = str(exp[field])
            if observed_value != expected_value:
                failures.append(
                    f"{row_label}: {field} mismatch observed={observed_value!r} expected={expected_value!r}"
                )
        for field in FLOAT_FIELDS:
            observed_value = float(obs[field])
            expected_value = float(exp[field])
            if observed_value != expected_value:
                failures.append(
                    f"{row_label}: {field} mismatch observed={observed_value!r} expected={expected_value!r}"
                )
    return failures


def _read_csv_exact(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return tuple(reader.fieldnames or ()), list(reader)


def _compare_csv_file(name: str, observed_path: Path, expected_path: Path) -> list[str]:
    observed_fields, observed_rows = _read_csv_exact(observed_path)
    expected_fields, expected_rows = _read_csv_exact(expected_path)
    failures: list[str] = []
    if observed_fields != expected_fields:
        failures.append(
            f"{name}: header mismatch observed={observed_fields!r} expected={expected_fields!r}"
        )
        return failures
    if len(observed_rows) != len(expected_rows):
        failures.append(
            f"{name}: row count mismatch observed={len(observed_rows)} expected={len(expected_rows)}"
        )
        return failures
    for idx, (observed, expected) in enumerate(zip(observed_rows, expected_rows), start=1):
        if observed != expected:
            failures.append(f"{name}: row {idx} mismatch observed={observed!r} expected={expected!r}")
    return failures


def _manifest_stable_subset(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": manifest.get("schema_version", ""),
        "publication_profile": manifest.get("publication_profile", ""),
        "execution_profile_id": manifest.get("execution_profile_id", ""),
        "analysis_policy_version": manifest.get("analysis_policy", {}).get("version", ""),
        "build_profile": manifest.get("build_profile", ""),
        "identity": manifest.get("identity", {}),
        "seeds": manifest.get("seeds", {}),
    }


def verify_run(run_root: Path) -> list[str]:
    committed_summary_path = limitation_requirements.limitation_robustness_summary_table_path()
    committed_input_path = limitation_requirements.limitation_robustness_input_table_path()
    committed_manifest_path = limitation_requirements.limitation_release_manifest_path()
    source_csv_path = limitation_requirements.limitation_source_comparability_csv_path(run_root)
    release_tables_root = limitation_requirements.limitation_release_tables_root(run_root)
    run_manifest_path = limitation_requirements.limitation_release_manifest_path(
        root=release_tables_root
    )
    csv_artifacts = (
        (
            "gate_diagnostics_summary.csv",
            limitation_requirements.limitation_gate_diagnostics_summary_table_path(),
            limitation_requirements.limitation_gate_diagnostics_summary_table_path(root=release_tables_root),
        ),
        (
            "gate_diagnostics_rows.csv",
            limitation_requirements.limitation_gate_diagnostics_rows_table_path(),
            limitation_requirements.limitation_gate_diagnostics_rows_table_path(root=release_tables_root),
        ),
        (
            "strict_gate_sensitivity.csv",
            limitation_requirements.limitation_strict_gate_sensitivity_table_path(),
            limitation_requirements.limitation_strict_gate_sensitivity_table_path(root=release_tables_root),
        ),
    )
    missing = [
        path
        for path in (
            committed_summary_path,
            committed_input_path,
            committed_manifest_path,
            source_csv_path,
            run_manifest_path,
            *(path for _, committed_path, run_path in csv_artifacts for path in (committed_path, run_path)),
        )
        if not path.exists()
    ]
    if missing:
        return [f"Missing required artifact: {limitation_requirements.relative_to_root(path)}" for path in missing]

    committed_summary = read_csv_rows(committed_summary_path)
    source_rows = read_csv_rows(source_csv_path)
    committed_input_rows = read_csv_rows(committed_input_path)
    committed_manifest = load_json(committed_manifest_path)
    run_manifest = load_json(run_manifest_path)

    failures: list[str] = []
    for name, manifest in (("committed release manifest", committed_manifest), ("run release manifest", run_manifest)):
        policy_version = str(manifest.get("analysis_policy", {}).get("version", ""))
        if policy_version != ANALYSIS_POLICY_VERSION:
            failures.append(
                f"{name}: analysis_policy.version mismatch observed={policy_version!r} expected={ANALYSIS_POLICY_VERSION!r}"
            )
        build_profile = str(manifest.get("build_profile", ""))
        if build_profile != LIMITATION_REFERENCE_BUILD_PROFILE:
            failures.append(
                f"{name}: build_profile mismatch observed={build_profile!r} expected={LIMITATION_REFERENCE_BUILD_PROFILE!r}"
            )
    committed_manifest_subset = _manifest_stable_subset(committed_manifest)
    run_manifest_subset = _manifest_stable_subset(run_manifest)
    if run_manifest_subset != committed_manifest_subset:
        failures.append(
            "release manifest stable subset mismatch "
            f"observed={run_manifest_subset!r} expected={committed_manifest_subset!r}"
        )

    failures.extend(_compare_rows(
        "full-source robustness",
        build_limitation_robustness(source_rows, layers=limitation_requirements.LIMITATION_PROFILE.public_layers),
        committed_summary,
    ))
    failures.extend(
        _compare_rows(
            "public-input robustness",
            build_limitation_robustness_from_case_target(
                committed_input_rows,
                layers=limitation_requirements.LIMITATION_PROFILE.public_layers,
            ),
            committed_summary,
        )
    )
    for name, committed_path, run_path in csv_artifacts:
        failures.extend(_compare_csv_file(name, run_path, committed_path))
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the limitation release surface from the full source rows, "
            "the public case-target CSV, and the committed governance artifacts."
        )
    )
    parser.add_argument("--run_root", required=True, help="Full limitation rerun root containing source/comparability CSV.")
    args = parser.parse_args(argv)

    failures = verify_run(Path(args.run_root).expanduser().resolve())
    for failure in failures:
        print(f"[fail] {failure}")
    if failures:
        return 2
    print(
        "[pass] full-source robustness, public-input robustness, gate diagnostics, "
        "strict sensitivity, and release manifest match the committed release surface"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
