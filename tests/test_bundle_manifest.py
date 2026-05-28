from __future__ import annotations

import json
from pathlib import Path

import pytest

from aom.data.bundle_manifest import compute_bundle_id, validate_bundle_manifest
from aom.data.dataset_manifest import sha256_file


def test_compute_bundle_id_stable_across_metadata() -> None:
    files = {
        "disamb_pairs.jsonl": {"sha256": "a" * 64},
    }
    m1 = {"name": "paper_hardened_v2", "generated_at_utc": "t1", "files": files}
    m2 = {"name": "paper_hardened_v2", "generated_at_utc": "t2", "git_commit": "x", "files": files}
    assert compute_bundle_id(m1) == compute_bundle_id(m2)


def test_compute_bundle_id_changes_when_any_file_hash_changes() -> None:
    files1 = {
        "disamb_pairs.jsonl": {"sha256": "a" * 64},
    }
    files2 = dict(files1)
    files2["disamb_pairs.jsonl"] = {"sha256": "d" * 64}
    assert compute_bundle_id({"files": files1}) != compute_bundle_id({"files": files2})


def test_compute_bundle_id_ignores_unrecognized_legacy_files() -> None:
    base = {
        "files": {
            "disamb_pairs.jsonl": {"sha256": "a" * 64},
        }
    }
    with_extra = {
        "files": {
            "disamb_pairs.jsonl": {"sha256": "a" * 64},
            "legacy_extra.jsonl": {"sha256": "b" * 64},
            "unused_auxiliary.jsonl": {"sha256": "c" * 64},
        }
    }
    assert compute_bundle_id(base) == compute_bundle_id(with_extra)


def test_validate_bundle_manifest_returns_bundle_id_and_checks_hashes(tmp_path: Path) -> None:
    dis = tmp_path / "disamb_pairs.jsonl"
    dis.write_text("x\n", encoding="utf-8")

    manifest = {
        "name": "paper_hardened_v2",
        "files": {
            "disamb_pairs.jsonl": {"sha256": sha256_file(dis)},
        },
    }
    mp = tmp_path / "DATASET_MANIFEST.json"
    mp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    out = validate_bundle_manifest(mp, disamb_path=str(dis))
    assert out["dataset_bundle_manifest_name"] == "paper_hardened_v2"
    assert out["dataset_bundle_id"] == compute_bundle_id(manifest)

    dis.write_text("x \n", encoding="utf-8")  # 1-byte change (trailing space)
    with pytest.raises(ValueError) as ei:
        validate_bundle_manifest(mp, disamb_path=str(dis))
    msg = str(ei.value)
    assert "role=disamb" in msg
    assert str(dis) in msg


def test_validate_bundle_manifest_rejects_mismatched_bundle_id(tmp_path: Path) -> None:
    dis = tmp_path / "disamb_pairs.jsonl"
    dis.write_text("x\n", encoding="utf-8")

    manifest = {
        "name": "paper_hardened_v2",
        "bundle_id": "0" * 64,
        "files": {"disamb_pairs.jsonl": {"sha256": sha256_file(dis)}},
    }
    mp = tmp_path / "DATASET_MANIFEST.json"
    mp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        validate_bundle_manifest(mp, disamb_path=str(dis))
