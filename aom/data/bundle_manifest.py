from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .dataset_manifest import sha256_file


ROLE_TO_FILENAME: dict[str, str] = {
    "disamb": "disamb_pairs.jsonl",
}


def read_bundle_manifest(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Dataset bundle manifest must be a JSON object, got {type(obj).__name__}")
    return obj


def _get_file_sha(manifest: Mapping[str, Any], filename: str) -> str:
    files = manifest.get("files", None)
    if not isinstance(files, Mapping):
        raise ValueError("Dataset bundle manifest missing mapping 'files'")
    entry = files.get(str(filename), None)
    if not isinstance(entry, Mapping):
        raise ValueError(f"Dataset bundle manifest missing file entry for {filename!r}")
    sha = entry.get("sha256", None)
    if not isinstance(sha, str) or not sha.strip():
        raise ValueError(f"Dataset bundle manifest file {filename!r} missing non-empty 'sha256'")
    if len(sha.strip()) != 64:
        raise ValueError(f"Dataset bundle manifest file {filename!r} sha256 must be 64 hex chars")
    return sha.strip()


def compute_bundle_id(manifest: Mapping[str, Any]) -> str:
    """
    Compute a stable, content-addressed dataset bundle identifier from per-file SHA-256s.

    Unlike the manifest file's SHA (which can change when metadata like timestamps changes),
    this bundle ID depends only on the underlying dataset JSONL file hashes.
    """
    file_map: dict[str, str] = {}
    for filename in sorted(set(ROLE_TO_FILENAME.values())):
        try:
            file_map[str(filename)] = _get_file_sha(manifest, str(filename))
        except Exception:
            # Allow partial manifests (callers may validate only a subset of roles).
            continue
    if not file_map:
        raise ValueError("Dataset bundle manifest contains no recognized dataset file entries")
    payload = {"files": file_map}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_bundle_manifest(
    manifest_path: str | Path,
    *,
    disamb_path: str | None = None,
) -> dict[str, str]:
    """
    Validate that dataset paths match a bundle manifest (by SHA-256).

    The expected format is the `scripts/run_paper.py` bundle manifest:
      {"files": {"disamb_pairs.jsonl": {"sha256": ...}, ...}}

    Returns both:
    - `dataset_bundle_manifest_sha256`: SHA-256 of the manifest JSON file itself (may change if metadata changes)
    - `dataset_bundle_id`: stable, content-addressed ID computed from the dataset JSONL file hashes
    """
    mp = Path(manifest_path)
    if not mp.exists():
        raise ValueError(f"Dataset bundle manifest does not exist: {str(mp)}")
    manifest = read_bundle_manifest(mp)
    manifest_sha256 = sha256_file(mp)
    bundle_id = compute_bundle_id(manifest)
    if "bundle_id" in manifest:
        got = manifest.get("bundle_id", None)
        if not isinstance(got, str) or not got.strip():
            raise ValueError("Dataset bundle manifest 'bundle_id' must be a non-empty string when present")
        if str(got).strip() != str(bundle_id):
            raise ValueError(
                "Dataset bundle manifest 'bundle_id' does not match computed bundle ID "
                f"(got {str(got).strip()}, computed {bundle_id})"
            )

    role_to_path: dict[str, str | None] = {"disamb": None if disamb_path is None else str(disamb_path)}

    for role, path in role_to_path.items():
        if path is None or not str(path).strip():
            continue
        filename = ROLE_TO_FILENAME.get(role)
        if filename is None:
            raise ValueError(f"Unknown dataset role: {role!r}")
        expected = _get_file_sha(manifest, filename)
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"Dataset SHA256 mismatch for role={role} path={path!r}: expected {expected} (from {str(mp)}), got {actual}"
            )

    name = manifest.get("name", "")
    return {
        "dataset_bundle_manifest_path": str(mp),
        "dataset_bundle_manifest_sha256": str(manifest_sha256),
        "dataset_bundle_manifest_name": str(name) if isinstance(name, str) else "",
        "dataset_bundle_id": str(bundle_id),
    }
