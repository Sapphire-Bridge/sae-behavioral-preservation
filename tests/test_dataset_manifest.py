from __future__ import annotations

from pathlib import Path

import pytest

from aom.data.dataset_manifest import DatasetLoadError, sha256_file
from aom.data.loaders import load_disamb_pairs_with_manifest
from aom.run_manifest import build_run_manifest, validate_run_manifest


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


def test_load_disamb_pairs_with_manifest_counts_invalid_rows(tmp_path: Path) -> None:
    disamb_path = tmp_path / "disamb_pairs.jsonl"
    disamb_path.write_text(
        "\n".join(
            [
                (
                    '{"pair_id":"bank-0","target":"bank","target_occurrence":0,'
                    '"a":{"prompt":"I sat by the bank and watched the","expected_label":"river"},'
                    '"b":{"prompt":"I went to the bank to discuss the","expected_label":"loan"},'
                    '"choices":{"river":[" river"],"loan":[" loan"]}}'
                ),
                "{",  # invalid JSON
                '{"pair_id":"bad","target":"bank"}',  # schema invalid
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    items, dm = load_disamb_pairs_with_manifest(str(disamb_path), role="disamb", error_policy="warn_skip")
    assert len(items) == 1
    assert dm.role == "disamb"
    assert dm.path == str(disamb_path)
    assert dm.sha256 == sha256_file(disamb_path)
    assert dm.size_bytes == int(disamb_path.stat().st_size)
    assert dm.n_rows_total == 3
    assert dm.n_rows_valid == 1
    assert dm.n_rows_invalid == 2
    assert len(dm.invalid_samples) == 2

    m = _add_minimal_run_fields(build_run_manifest(argv=["python"], results_row={"x": 1}, dataset_manifest_path=None))
    m["datasets"] = {"disamb": dm.as_dict()}
    validate_run_manifest(m)


def test_load_disamb_pairs_with_manifest_raises_on_invalid_rows_when_error_policy_raise(tmp_path: Path) -> None:
    disamb_path = tmp_path / "disamb_pairs.jsonl"
    disamb_path.write_text(
        "\n".join(
            [
                (
                    '{"pair_id":"bank-0","target":"bank","target_occurrence":0,'
                    '"a":{"prompt":"I sat by the bank and watched the","expected_label":"river"},'
                    '"b":{"prompt":"I went to the bank to discuss the","expected_label":"loan"},'
                    '"choices":{"river":[" river"],"loan":[" loan"]}}'
                ),
                "{",  # invalid JSON
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(DatasetLoadError) as ei:
        load_disamb_pairs_with_manifest(str(disamb_path), role="disamb", error_policy="raise")
    dm = ei.value.manifest
    assert dm.role == "disamb"
    assert dm.n_rows_total == 2
    assert dm.n_rows_valid == 1
    assert dm.n_rows_invalid == 1
