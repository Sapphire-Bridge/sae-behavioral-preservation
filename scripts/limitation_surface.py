#!/usr/bin/env python3
"""
Public release surface builder for the SAE behavioral preservation result.

This module is governance/derivation code: it does not run new model
experiments. It reads source artifacts and exposes builders used by the release
script to produce public summaries, tables, diagnostics, figures, and manifest
payloads.

Primary source inputs:
  comparability_summary  : JSON output of scripts/clt_raw_comparability.py
                           (per-layer bootstrap aggregates, run config, counts)
  topk_summary           : JSON output of analysis/aom_clt_topk_recovery.py
                           (top-k latent concentration curves)
  comparability_rows     : CSV rows from the comparability run

The surrounding release script calls these builders to produce:
  public JSON summaries  : one per layer, publishable comparability + top-k entries
  public CSV tables      : centerpiece, top-k, STRESS, gate, and robustness tables
  public SVG figures     : centerpiece and top-k figures
  release manifest JSON  : provenance, artifact hashes, public layer entries,
                           identity, seeds, execution profile, and source pointers
  derived numbers JSON   : centerpiece, PCA, top-k, and STRESS summaries

The builders are deterministic given fixed source artifacts, fixed builder
seeds, and a fixed runtime environment. Runtime metadata fields in the release
manifest intentionally record the environment used to build the artifacts.
"""

from __future__ import annotations

import csv
import json
import math
import platform
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aom.stats import bootstrap_ci_pair_cluster
from scripts.limitation_analysis_policy import (
    ANALYSIS_POLICY_VERSION,
    limitation_build_profile,
    limitation_analysis_policy_metadata,
)
import scripts.limitation_requirements as limitation_requirements
from scripts.reproduction_common import _per_layer_entry


# ---------------------------------------------------------------------------
# Identity and output container dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LimitationIdentity:
    model_id: str
    model_revision: str
    tokenizer_revision: str
    dataset_bundle_id: str
    dataset_manifest_sha256: str
    sae_bundle_id: str
    sae_bundle_manifest_sha256: str


@dataclass(frozen=True)
class DerivedOutputs:
    numbers: dict[str, Any]
    stress_rows: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# I/O helpers and small utilities
# ---------------------------------------------------------------------------


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(rows: Iterable[dict[str, Any]], out_path: Path) -> None:
    materialized = list(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(materialized[0].keys()) if materialized else []
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialized)


def write_csv_rows_with_fields(
    rows: Iterable[dict[str, Any]],
    out_path: Path,
    *,
    fieldnames: Iterable[str],
) -> None:
    materialized = list(rows)
    fields = list(fieldnames)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialized)


def _source_artifact_fields(
    key: str,
    path: Path,
    *,
    context: str,
    source_artifact_root: Path,
) -> dict[str, str]:
    if Path(source_artifact_root).expanduser().resolve() == limitation_requirements.ROOT.resolve():
        artifact = limitation_requirements.public_repo_artifact(path, context=context)
    else:
        artifact = limitation_requirements.portable_artifact(
            path,
            context=context,
            root=source_artifact_root,
        )
    return {key: artifact["path"], f"{key}_sha256": artifact["sha256"]}


def configured_identity(
    profile: limitation_requirements.LimitationProfile | None = None,
) -> LimitationIdentity:
    p = limitation_requirements.ensure_profile_configured(profile, require_runs=False)
    return LimitationIdentity(
        model_id=str(p.model_id),
        model_revision=str(p.model_revision),
        tokenizer_revision=str(p.tokenizer_revision),
        dataset_bundle_id=str(p.dataset_bundle_id),
        dataset_manifest_sha256=str(p.dataset_manifest_sha256),
        sae_bundle_id=str(p.sae_bundle_id),
        sae_bundle_manifest_sha256=str(p.sae_bundle_manifest_sha256),
    )


def _metric_triplet(entry: dict[str, Any], prefix: str) -> dict[str, float]:
    return {
        "mean": float(entry[f"{prefix}_mean"]),
        "ci_low": float(entry[f"{prefix}_ci_low"]),
        "ci_high": float(entry[f"{prefix}_ci_high"]),
    }


def _has_metric_triplet(entry: dict[str, Any], prefix: str) -> bool:
    return all(f"{prefix}_{suffix}" in entry for suffix in ("mean", "ci_low", "ci_high"))


def _required_dict(obj: Any, name: str) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise TypeError(f"{name} must be an object")
    return dict(obj)


def _parse_model_id(raw: str) -> str:
    value = str(raw).strip()
    if value.startswith("hf://"):
        value = value[len("hf://") :]
    if "@" in value:
        return value.split("@", 1)[0]
    return value


def _parse_model_revision(raw: str) -> str:
    value = str(raw).strip()
    if value.startswith("hf://"):
        value = value[len("hf://") :]
    if "@" in value:
        return value.split("@", 1)[1]
    return ""


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _safe_float(value: object, *, field_name: str) -> float:
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"Could not parse float for {field_name}: {value!r}") from exc


def _canonicalize_repo_path(value: object) -> tuple[str, str]:
    text = str(value).strip()
    if not text:
        return "", ""
    path = Path(text).expanduser()
    if path.is_absolute():
        resolved = path.resolve()
        try:
            rel = resolved.relative_to(ROOT).as_posix()
        except ValueError:
            rel = ""
        return str(resolved), rel
    normalized = path.as_posix()
    resolved = (ROOT / path).resolve()
    try:
        rel = resolved.relative_to(ROOT).as_posix()
    except ValueError:
        rel = normalized
    return str(resolved), rel


def _portable_repo_path(value: object) -> str:
    abs_path, rel_path = _canonicalize_repo_path(value)
    if rel_path:
        return rel_path
    text = str(value).strip().replace("\\", "/")
    # Historical public summaries accidentally carried a user-specific checkout
    # prefix; keep public evidence paths anchored at repo artifact roots.
    for anchor in ("clt_bundles/", "data_paper_hardened_v2/"):
        idx = text.find(anchor)
        if idx >= 0:
            return text[idx:]
    return abs_path


def _validate_expected_repo_path(*, observed: object, expected: object, field_name: str) -> None:
    observed_abs, observed_rel = _canonicalize_repo_path(observed)
    expected_abs, expected_rel = _canonicalize_repo_path(expected)
    observed_portable = _portable_repo_path(observed)
    expected_portable = _portable_repo_path(expected)
    if not observed_abs:
        raise ValueError(f"{field_name} missing from source summary")
    if observed_abs != expected_abs and observed_rel != expected_rel and observed_portable != expected_portable:
        raise ValueError(
            f"{field_name} mismatch: expected {expected_portable or expected_rel or expected_abs!r}, "
            f"got {observed_portable or observed_rel or observed_abs!r}"
        )


# ---------------------------------------------------------------------------
# Source validation: check input summaries before building public artifacts
# ---------------------------------------------------------------------------


def validate_comparability_source_summary(
    comparability_summary: dict[str, Any],
    *,
    identity: LimitationIdentity,
    layers: Iterable[int],
    expected_disamb_path: str | Path | None = None,
    expected_clt_repo: str | Path | None = None,
) -> None:
    """Validate a comparability source summary before building public artifacts.

    Checks that all required layers are present, analysis policy version matches,
    and identity fields (model, dataset, SAE bundle) agree with `identity`.
    Raises KeyError or ValueError on any mismatch.
    """
    layer_list = tuple(int(layer) for layer in layers)
    comp_layers = {int(layer) for layer in comparability_summary.get("layers", [])}
    if not set(layer_list).issubset(comp_layers):
        raise KeyError(f"Comparability summary missing required layers: {sorted(set(layer_list) - comp_layers)}")

    analysis_policy = _required_dict(
        comparability_summary.get("analysis_policy"),
        "comparability_summary.analysis_policy",
    )
    policy_version = str(analysis_policy.get("version", "")).strip()
    if policy_version != ANALYSIS_POLICY_VERSION:
        raise ValueError(
            f"Comparability analysis_policy mismatch: expected {ANALYSIS_POLICY_VERSION!r}, got {policy_version!r}"
        )

    comp_model_raw = str(comparability_summary.get("model_name_or_path", ""))
    comp_model_id = _parse_model_id(comp_model_raw)
    if comp_model_id != identity.model_id:
        raise ValueError(f"Comparability model_id mismatch: expected {identity.model_id!r}, got {comp_model_id!r}")

    run_config = _required_dict(comparability_summary.get("run_config"), "comparability_summary.run_config")
    comp_model_revision = str(run_config.get("model_revision") or _parse_model_revision(comp_model_raw)).strip()
    if comp_model_revision and comp_model_revision != identity.model_revision:
        raise ValueError(
            f"Comparability model_revision mismatch: expected {identity.model_revision!r}, got {comp_model_revision!r}"
        )

    comp_tokenizer_revision = str(run_config.get("tokenizer_revision") or "").strip()
    if comp_tokenizer_revision and comp_tokenizer_revision != identity.tokenizer_revision:
        raise ValueError(
            "Comparability tokenizer_revision mismatch: "
            f"expected {identity.tokenizer_revision!r}, got {comp_tokenizer_revision!r}"
        )

    if expected_disamb_path is not None:
        _validate_expected_repo_path(
            observed=comparability_summary.get("disamb_path", ""),
            expected=expected_disamb_path,
            field_name="Comparability disamb_path",
        )

    if expected_clt_repo is not None:
        provenance = _required_dict(comparability_summary.get("provenance", {}), "comparability_summary.provenance")
        _validate_expected_repo_path(
            observed=provenance.get("clt_repo", ""),
            expected=expected_clt_repo,
            field_name="Comparability clt_repo",
        )


def validate_topk_source_summary(
    topk_summary: dict[str, Any],
    *,
    identity: LimitationIdentity,
    layers: Iterable[int],
    compact_ks: Iterable[int] | None = None,
) -> None:
    """Validate a top-k source summary before building public artifacts.

    Checks that required layers and compact_ks are present and identity fields
    match. Raises KeyError or ValueError on any mismatch.
    """
    layer_list = tuple(int(layer) for layer in layers)
    compact_k_list = tuple(int(k) for k in (compact_ks or limitation_requirements.LIMITATION_COMPACT_KS))

    topk_spec = _required_dict(topk_summary.get("spec"), "topk_summary.spec")
    topk_layers = {int(layer) for layer in topk_spec.get("layers", [])}
    if not set(layer_list).issubset(topk_layers):
        raise KeyError(f"Top-k summary missing required layers: {sorted(set(layer_list) - topk_layers)}")

    available_ks = {int(k) for k in topk_spec.get("ks", [])}
    if not set(compact_k_list).issubset(available_ks):
        raise KeyError(f"Top-k summary missing compact ks: {sorted(set(compact_k_list) - available_ks)}")

    topk_run = _required_dict(topk_summary.get("run"), "topk_summary.run")
    topk_model_id = _parse_model_id(str(topk_run.get("model_name_or_path", "")))
    if topk_model_id != identity.model_id:
        raise ValueError(f"Top-k model_id mismatch: expected {identity.model_id!r}, got {topk_model_id!r}")

    topk_model_revision = str(topk_run.get("model_revision", "")).strip()
    if topk_model_revision and topk_model_revision != identity.model_revision:
        raise ValueError(
            f"Top-k model_revision mismatch: expected {identity.model_revision!r}, got {topk_model_revision!r}"
        )

    topk_tokenizer_revision = str(topk_run.get("tokenizer_revision", "")).strip()
    if topk_tokenizer_revision and topk_tokenizer_revision != identity.tokenizer_revision:
        raise ValueError(
            "Top-k tokenizer_revision mismatch: "
            f"expected {identity.tokenizer_revision!r}, got {topk_tokenizer_revision!r}"
        )


def validate_source_summaries(
    comparability_summary: dict[str, Any],
    topk_summary: dict[str, Any],
    *,
    identity: LimitationIdentity,
    layers: Iterable[int],
    compact_ks: Iterable[int] | None = None,
    expected_disamb_path: str | Path | None = None,
    expected_clt_repo: str | Path | None = None,
) -> None:
    validate_comparability_source_summary(
        comparability_summary,
        identity=identity,
        layers=layers,
        expected_disamb_path=expected_disamb_path,
        expected_clt_repo=expected_clt_repo,
    )
    validate_topk_source_summary(
        topk_summary,
        identity=identity,
        layers=layers,
        compact_ks=compact_ks,
    )


# ---------------------------------------------------------------------------
# Public summary builders: derive publishable JSON entries per layer
# ---------------------------------------------------------------------------


def build_public_comparability_summary(
    comparability_summary: dict[str, Any],
    *,
    identity: LimitationIdentity,
    layer: int,
    source_path: Path,
    source_artifact_root: Path,
) -> dict[str, Any]:
    """Build a single-layer public comparability summary entry.

    Extracts the bootstrap aggregates, counts, and provenance from the source
    summary and returns a dict suitable for the public release JSON. Does not
    recompute statistics — only selects and renames fields.
    """
    entry = _per_layer_entry(comparability_summary, layer)
    counts = _required_dict(comparability_summary.get("counts"), "comparability_summary.counts")
    run_config = _required_dict(comparability_summary.get("run_config"), "comparability_summary.run_config")
    provenance = _required_dict(comparability_summary.get("provenance", {}), "comparability_summary.provenance")

    return {
        "summary_schema_version": limitation_requirements.LIMITATION_COMPARABILITY_SCHEMA_VERSION,
        "publication_profile": limitation_requirements.LIMITATION_PUBLICATION_PROFILE,
        "execution_profile_id": limitation_requirements.LIMITATION_EXECUTION_PROFILE_ID,
        "layer": int(layer),
        "model": {
            "id": identity.model_id,
            "revision": identity.model_revision,
            "tokenizer_revision": identity.tokenizer_revision,
        },
        "dataset": {
            "bundle_id": identity.dataset_bundle_id,
            "manifest_sha256": identity.dataset_manifest_sha256,
            "disamb_path": _portable_repo_path(comparability_summary.get("disamb_path", "")),
        },
        "sae": {
            "bundle_id": identity.sae_bundle_id,
            "bundle_manifest_sha256": identity.sae_bundle_manifest_sha256,
            "repo": _portable_repo_path(provenance.get("clt_repo", "")),
        },
        "run": {
            "seed": int(run_config.get("seed", 0)),
            "bootstrap_seed": int(run_config.get("bootstrap_seed", 0)),
            "bootstrap_n": int(run_config.get("bootstrap_n", 0)),
            "ci": float(run_config.get("ci", 0.95)),
            "device": str(run_config.get("device", "")),
            "torch_dtype": str(run_config.get("torch_dtype", "")),
            "git_commit": str(provenance.get("git_commit", "")),
            "source_schema_version": str(comparability_summary.get("schema_version", "")),
        },
        "analysis_policy": limitation_analysis_policy_metadata(),
        "counts": {
            "n_rows_total": int(counts.get("n_rows_total", 0)),
            "n_rows_analysis_included": int(counts.get("n_rows_analysis_included", 0)),
            "n_pairs_analysis_included": int(counts.get("n_pairs_analysis_included", 0)),
            "n_invariant_fail_rows": int(counts.get("n_invariant_fail_rows", 0)),
        },
        "metrics": {
            "fidelity_cosine": _metric_triplet(entry, "fidelity_cosine"),
            "fidelity_rel_mse": _metric_triplet(entry, "fidelity_rel_mse"),
            **(
                {"fidelity_fvu": _metric_triplet(entry, "fidelity_fvu")}
                if _has_metric_triplet(entry, "fidelity_fvu")
                else {}
            ),
            "raw_effect": _metric_triplet(entry, "effect_A"),
            "sae_effect": _metric_triplet(entry, "effect_C"),
            "sae_minus_raw": _metric_triplet(entry, "d_CA"),
            "crr": _metric_triplet(entry, "crr_C_over_A"),
            "pca_effect": _metric_triplet(entry, "effect_PRJ_PCA"),
        },
        "source_artifacts": _source_artifact_fields(
            "comparability_summary",
            source_path,
            context="Limitation comparability source summary",
            source_artifact_root=source_artifact_root,
        ),
    }


def build_public_topk_summary(
    topk_summary: dict[str, Any],
    *,
    identity: LimitationIdentity,
    layer: int,
    compact_ks: Iterable[int] | None = None,
    source_path: Path,
    source_artifact_root: Path,
) -> dict[str, Any]:
    """Build a single-layer public top-k concentration summary entry.

    Extracts concentration curves and run metadata from the top-k source
    summary and returns a dict suitable for the public release JSON.
    """
    run = _required_dict(topk_summary.get("run"), "topk_summary.run")
    spec = _required_dict(topk_summary.get("spec"), "topk_summary.spec")
    concentration_by_layer = _required_dict(topk_summary.get("concentration_by_layer"), "topk_summary.concentration_by_layer")
    curves_by_layer = _required_dict(topk_summary.get("curves_by_layer"), "topk_summary.curves_by_layer")

    layer_key = str(int(layer))
    layer_curves = _required_dict(curves_by_layer.get(layer_key), f"topk_summary.curves_by_layer[{layer_key}]")
    effects = _required_dict(layer_curves.get("effects"), f"topk_summary.curves_by_layer[{layer_key}].effects")
    full_effects = _required_dict(effects.get("full"), f"topk_summary.curves_by_layer[{layer_key}].effects.full")
    topk_effects = _required_dict(effects.get("topk"), f"topk_summary.curves_by_layer[{layer_key}].effects.topk")

    full_k = max(int(k) for k in full_effects)
    compact_payload: dict[str, dict[str, float]] = {}
    compact_k_list = tuple(int(k) for k in (compact_ks or limitation_requirements.LIMITATION_COMPACT_KS))
    for k in compact_k_list:
        compact_payload[str(k)] = {
            "mean": float(topk_effects[str(k)]["mean"]),
            "ci_low": float(topk_effects[str(k)]["ci_low"]),
            "ci_high": float(topk_effects[str(k)]["ci_high"]),
        }

    concentration = _required_dict(concentration_by_layer.get(layer_key), f"topk_summary.concentration_by_layer[{layer_key}]")

    return {
        "summary_schema_version": limitation_requirements.LIMITATION_TOPK_SCHEMA_VERSION,
        "publication_profile": limitation_requirements.LIMITATION_PUBLICATION_PROFILE,
        "execution_profile_id": limitation_requirements.LIMITATION_EXECUTION_PROFILE_ID,
        "layer": int(layer),
        "model": {
            "id": identity.model_id,
            "revision": identity.model_revision,
            "tokenizer_revision": identity.tokenizer_revision,
        },
        "dataset": {
            "bundle_id": identity.dataset_bundle_id,
            "manifest_sha256": identity.dataset_manifest_sha256,
        },
        "sae": {
            "bundle_id": identity.sae_bundle_id,
            "bundle_manifest_sha256": identity.sae_bundle_manifest_sha256,
        },
        "run": {
            "seed": int(run.get("seed", 0)),
            "bootstrap_n": int(spec.get("bootstrap_B", 0)),
            "split_seed": int(spec.get("split_seed", 0)),
            "frac_selection": float(spec.get("frac_selection", 0.0)),
            "ci": float(spec.get("ci", 0.95)),
            "git_commit": str(run.get("git_commit", "")),
            "source_schema_version": str(topk_summary.get("schema_version", "")),
        },
        "counts": {
            "n_eval_cases": int(layer_curves.get("n_eval_cases", 0)),
            "n_total_directions": int(layer_curves.get("n_total_directions", 0)),
            "n_skipped_misaligned": int(layer_curves.get("n_skipped_misaligned", 0)),
        },
        "full_effect": {
            "mean": float(full_effects[str(full_k)]["mean"]),
            "ci_low": float(full_effects[str(full_k)]["ci_low"]),
            "ci_high": float(full_effects[str(full_k)]["ci_high"]),
        },
        "compact_topk_effects": compact_payload,
        "concentration": {
            "gini": float(concentration.get("gini", 0.0)),
            "mass_at_20": float(concentration.get("mass_at_20", 0.0)),
            "mass_at_50": float(concentration.get("mass_at_50", 0.0)),
            "mass_at_100": float(concentration.get("mass_at_100", 0.0)),
        },
        "source_artifacts": _source_artifact_fields(
            "topk_summary",
            source_path,
            context="Limitation top-k source summary",
            source_artifact_root=source_artifact_root,
        ),
    }


# ---------------------------------------------------------------------------
# Release manifest: provenance, hashes, public layer entries
# ---------------------------------------------------------------------------


def build_limitation_release_manifest(
    *,
    comparability_summary: dict[str, Any],
    topk_summary: dict[str, Any],
    identity: LimitationIdentity,
    profile: limitation_requirements.LimitationProfile,
    results_root: Path,
    comparability_summary_path: Path,
    comparability_csv_path: Path,
    topk_summary_path: Path,
    source_artifact_root: Path,
) -> dict[str, Any]:
    """Build the complete release manifest for the limitation result.

    Assembles provenance (model, dataset, SAE bundle IDs and hashes),
    public layer entries, and pointers to the source artifacts used by the
    public release surface.
    This is the authoritative machine-readable record of what was published
    and from which source artifacts it was derived.
    """
    run_config = _required_dict(comparability_summary.get("run_config"), "comparability_summary.run_config")
    provenance = _required_dict(comparability_summary.get("provenance", {}), "comparability_summary.provenance")
    topk_run = _required_dict(topk_summary.get("run"), "topk_summary.run")
    topk_spec = _required_dict(topk_summary.get("spec"), "topk_summary.spec")
    try:
        import torch as _torch

        torch_version = str(_torch.__version__)
    except Exception:
        torch_version = ""

    return {
        "schema_version": "sae_writeback_limitation_release_manifest_v1",
        "publication_profile": limitation_requirements.LIMITATION_PUBLICATION_PROFILE,
        "execution_profile_id": limitation_requirements.LIMITATION_EXECUTION_PROFILE_ID,
        "build_profile": limitation_build_profile(
            device=run_config.get("device", ""),
            torch_dtype=run_config.get("torch_dtype", ""),
        ),
        "analysis_policy": limitation_analysis_policy_metadata(),
        "environment": {
            "device": str(run_config.get("device", "")),
            "torch_dtype": str(run_config.get("torch_dtype", "")),
            "topk_device": str(topk_run.get("device", "")),
            "topk_torch_dtype": str(topk_run.get("torch_dtype", profile.topk_torch_dtype or "")),
            "python_version": sys.version.split()[0],
            "torch_version": torch_version,
            "platform": platform.platform(),
        },
        "seeds": {
            "comparability_seed": int(run_config.get("seed", profile.seed)),
            "comparability_bootstrap_seed": int(run_config.get("bootstrap_seed", profile.bootstrap_seed)),
            "comparability_bootstrap_n": int(run_config.get("bootstrap_n", profile.bootstrap_n)),
            "topk_seed": int(topk_run.get("seed", profile.topk_seed)),
            "topk_split_seed": int(topk_spec.get("split_seed", profile.topk_split_seed)),
            "topk_bootstrap_B": int(topk_spec.get("bootstrap_B", profile.topk_bootstrap_B)),
        },
        "public_layer_entries": [
            {
                "layer": int(layer),
                "source_artifacts": {
                    **_source_artifact_fields(
                        "comparability_summary",
                        limitation_requirements.limitation_comparability_summary_path(layer, root=results_root),
                        context=f"Limitation public L{int(layer)} comparability summary",
                        source_artifact_root=source_artifact_root,
                    ),
                    **_source_artifact_fields(
                        "topk_summary",
                        limitation_requirements.limitation_topk_summary_path(layer, root=results_root),
                        context=f"Limitation public L{int(layer)} top-k summary",
                        source_artifact_root=source_artifact_root,
                    ),
                },
            }
            for layer in tuple(int(layer) for layer in profile.public_layers)
        ],
        "source_five_layer_fvu": [
            {
                "layer": int(entry["layer"]),
                "fidelity_fvu_mean": float(entry["fidelity_fvu_mean"]),
                "fidelity_fvu_ci_low": float(entry["fidelity_fvu_ci_low"]),
                "fidelity_fvu_ci_high": float(entry["fidelity_fvu_ci_high"]),
            }
            for entry in sorted(
                (
                    item
                    for item in comparability_summary.get("per_layer", [])
                    if int(item.get("layer", -1)) in set(int(layer) for layer in profile.paper_layers)
                ),
                key=lambda item: int(item["layer"]),
            )
        ],
        "identity": {
            "model_id": identity.model_id,
            "model_revision": identity.model_revision,
            "tokenizer_revision": identity.tokenizer_revision,
            "dataset_bundle_id": identity.dataset_bundle_id,
            "dataset_manifest_sha256": identity.dataset_manifest_sha256,
            "sae_bundle_id": identity.sae_bundle_id,
            "sae_bundle_manifest_sha256": identity.sae_bundle_manifest_sha256,
            "sae_repo": _portable_repo_path(provenance.get("clt_repo", "")),
        },
        "source_artifacts": {
            **_source_artifact_fields(
                "comparability_summary",
                comparability_summary_path,
                context="Limitation comparability source summary",
                source_artifact_root=source_artifact_root,
            ),
            **_source_artifact_fields(
                "comparability_csv",
                comparability_csv_path,
                context="Limitation comparability source CSV",
                source_artifact_root=source_artifact_root,
            ),
            **_source_artifact_fields(
                "topk_summary",
                topk_summary_path,
                context="Limitation top-k source summary",
                source_artifact_root=source_artifact_root,
            ),
        },
        "git_commit": str(provenance.get("git_commit", topk_run.get("git_commit", ""))),
    }


def write_centerpiece_table(summaries: Iterable[dict[str, Any]], out_path: Path) -> None:
    materialized = sorted(list(summaries), key=lambda item: int(item["layer"]))
    all_have_fvu = materialized and all(
        "fidelity_fvu" in _required_dict(summary.get("metrics"), "comparability.metrics")
        for summary in materialized
    )
    metric_names = [
        "fidelity_cosine",
        "fidelity_rel_mse",
        *(["fidelity_fvu"] if all_have_fvu else []),
        "raw_effect",
        "sae_effect",
        "sae_minus_raw",
        "crr",
        "pca_effect",
    ]
    rows = []
    for summary in materialized:
        metrics = _required_dict(summary.get("metrics"), "comparability.metrics")
        counts = _required_dict(summary.get("counts"), "comparability.counts")
        row = {
            "layer": int(summary["layer"]),
            "n_pairs_analysis_included": int(counts["n_pairs_analysis_included"]),
            "n_invariant_fail_rows": int(counts["n_invariant_fail_rows"]),
        }
        for metric_name in metric_names:
            metric = _required_dict(metrics.get(metric_name), f"comparability.metrics.{metric_name}")
            row[f"{metric_name}_mean"] = metric["mean"]
            row[f"{metric_name}_ci_low"] = metric["ci_low"]
            row[f"{metric_name}_ci_high"] = metric["ci_high"]
        rows.append(row)

    write_csv_rows(rows, out_path)


def write_topk_table(summaries: Iterable[dict[str, Any]], out_path: Path) -> None:
    rows = []
    for summary in sorted(summaries, key=lambda item: int(item["layer"])):
        concentration = _required_dict(summary.get("concentration"), "topk.concentration")
        counts = _required_dict(summary.get("counts"), "topk.counts")
        full_effect = _required_dict(summary.get("full_effect"), "topk.full_effect")
        compact = _required_dict(summary.get("compact_topk_effects"), "topk.compact_topk_effects")
        for k in limitation_requirements.LIMITATION_COMPACT_KS:
            metric = _required_dict(compact.get(str(k)), f"topk.compact_topk_effects[{k}]")
            rows.append(
                {
                    "layer": int(summary["layer"]),
                    "k": int(k),
                    "mass_at_k": float(concentration[f"mass_at_{int(k)}"]),
                    "n_eval_cases": int(counts["n_eval_cases"]),
                    "full_effect_mean": float(full_effect["mean"]),
                    "full_effect_ci_low": float(full_effect["ci_low"]),
                    "full_effect_ci_high": float(full_effect["ci_high"]),
                    "topk_effect_mean": float(metric["mean"]),
                    "topk_effect_ci_low": float(metric["ci_low"]),
                    "topk_effect_ci_high": float(metric["ci_high"]),
                }
            )

    write_csv_rows(rows, out_path)


def _svg_text(x: float, y: float, text: str, *, size: int = 12, weight: str = "normal") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-family="monospace" font-weight="{weight}">{text}</text>'


def _scale_y(value: float, minimum: float, maximum: float, top: float, height: float) -> float:
    if math.isclose(maximum, minimum):
        return top + height / 2.0
    return top + height * (1.0 - ((value - minimum) / (maximum - minimum)))


def render_centerpiece_figure(summaries: Iterable[dict[str, Any]], out_path: Path) -> None:
    entries = sorted(summaries, key=lambda item: int(item["layer"]))
    colors = {
        "raw_effect": "#1f77b4",
        "sae_effect": "#d62728",
        "pca_effect": "#2ca02c",
    }
    metric_names = ("raw_effect", "sae_effect", "pca_effect")
    values = []
    for summary in entries:
        for metric_name in metric_names:
            metric = summary["metrics"][metric_name]
            values.extend([float(metric["ci_low"]), float(metric["ci_high"])])
    minimum = min(values + [0.0])
    maximum = max(values + [0.0])
    width = 720
    height = 420
    left = 80
    top = 40
    plot_width = 580
    plot_height = 280
    group_width = plot_width / max(len(entries), 1)
    bar_width = group_width / 4.0
    zero_y = _scale_y(0.0, minimum, maximum, top, plot_height)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#222222" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#222222" stroke-width="1"/>',
        f'<line x1="{left}" y1="{zero_y:.1f}" x2="{left + plot_width}" y2="{zero_y:.1f}" stroke="#aaaaaa" stroke-width="1" stroke-dasharray="4 4"/>',
        _svg_text(left, 24, "SAE behavioral preservation: L4/L8 centerpiece", size=16, weight="bold"),
    ]
    for idx, summary in enumerate(entries):
        group_left = left + idx * group_width + bar_width * 0.5
        parts.append(_svg_text(group_left + bar_width, top + plot_height + 28, f"L{int(summary['layer'])}", size=12, weight="bold"))
        for metric_idx, metric_name in enumerate(metric_names):
            metric = summary["metrics"][metric_name]
            x = group_left + metric_idx * bar_width * 1.2
            mean = float(metric["mean"])
            ci_low = float(metric["ci_low"])
            ci_high = float(metric["ci_high"])
            y_mean = _scale_y(mean, minimum, maximum, top, plot_height)
            y_low = _scale_y(ci_low, minimum, maximum, top, plot_height)
            y_high = _scale_y(ci_high, minimum, maximum, top, plot_height)
            bar_top = min(y_mean, zero_y)
            bar_height = max(abs(zero_y - y_mean), 1.0)
            parts.append(f'<rect x="{x:.1f}" y="{bar_top:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{colors[metric_name]}"/>')
            parts.append(f'<line x1="{x + bar_width/2:.1f}" y1="{y_low:.1f}" x2="{x + bar_width/2:.1f}" y2="{y_high:.1f}" stroke="#111111" stroke-width="1.5"/>')
            parts.append(f'<line x1="{x + 4:.1f}" y1="{y_low:.1f}" x2="{x + bar_width - 4:.1f}" y2="{y_low:.1f}" stroke="#111111" stroke-width="1.5"/>')
            parts.append(f'<line x1="{x + 4:.1f}" y1="{y_high:.1f}" x2="{x + bar_width - 4:.1f}" y2="{y_high:.1f}" stroke="#111111" stroke-width="1.5"/>')

    legend_y = top + plot_height + 58
    legend_x = left
    for offset, (metric_name, color) in enumerate(colors.items()):
        x = legend_x + offset * 150
        label = metric_name.replace("_", " ")
        parts.append(f'<rect x="{x:.1f}" y="{legend_y - 10:.1f}" width="14" height="14" fill="{color}"/>')
        parts.append(_svg_text(x + 22, legend_y + 2, label, size=11))

    parts.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def render_topk_figure(summaries: Iterable[dict[str, Any]], out_path: Path) -> None:
    entries = sorted(summaries, key=lambda item: int(item["layer"]))
    width = 720
    height = 420
    left = 80
    top = 40
    plot_width = 580
    plot_height = 280
    all_values = []
    for summary in entries:
        full_effect = summary["full_effect"]
        all_values.extend([float(full_effect["ci_low"]), float(full_effect["ci_high"])])
        for k in limitation_requirements.LIMITATION_COMPACT_KS:
            metric = summary["compact_topk_effects"][str(k)]
            all_values.extend([float(metric["ci_low"]), float(metric["ci_high"])])
    minimum = min(all_values + [0.0])
    maximum = max(all_values + [0.0])
    zero_y = _scale_y(0.0, minimum, maximum, top, plot_height)
    x_positions = {
        k: left + (idx / max(len(limitation_requirements.LIMITATION_COMPACT_KS) - 1, 1)) * plot_width
        for idx, k in enumerate(limitation_requirements.LIMITATION_COMPACT_KS)
    }
    colors = {4: "#8c564b", 8: "#9467bd"}

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#222222" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#222222" stroke-width="1"/>',
        f'<line x1="{left}" y1="{zero_y:.1f}" x2="{left + plot_width}" y2="{zero_y:.1f}" stroke="#aaaaaa" stroke-width="1" stroke-dasharray="4 4"/>',
        _svg_text(left, 24, "Compact top-k versus full SAE effect", size=16, weight="bold"),
    ]

    for k, x in x_positions.items():
        parts.append(_svg_text(x - 6, top + plot_height + 28, str(k), size=12, weight="bold"))

    for summary in entries:
        layer = int(summary["layer"])
        color = colors.get(layer, "#444444")
        points = []
        for k in limitation_requirements.LIMITATION_COMPACT_KS:
            metric = summary["compact_topk_effects"][str(k)]
            x = x_positions[k]
            y = _scale_y(float(metric["mean"]), minimum, maximum, top, plot_height)
            points.append(f"{x:.1f},{y:.1f}")
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>')
        joined_points = " ".join(points)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{joined_points}"/>')
        full_effect = summary["full_effect"]
        full_y = _scale_y(float(full_effect["mean"]), minimum, maximum, top, plot_height)
        parts.append(f'<line x1="{left}" y1="{full_y:.1f}" x2="{left + plot_width}" y2="{full_y:.1f}" stroke="{color}" stroke-width="1.5" stroke-dasharray="6 4"/>')

    legend_y = top + plot_height + 58
    legend_x = left
    for idx, summary in enumerate(entries):
        layer = int(summary["layer"])
        color = colors.get(layer, "#444444")
        x = legend_x + idx * 160
        parts.append(f'<rect x="{x:.1f}" y="{legend_y - 10:.1f}" width="14" height="14" fill="{color}"/>')
        parts.append(_svg_text(x + 22, legend_y + 2, f"L{layer} compact-k; dashed = full", size=11))

    parts.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _field_triplet_from_rows(
    rows: list[dict[str, str]],
    *,
    field_name: str,
    bootstrap_n: int,
    ci: float,
    seed: int,
) -> dict[str, float]:
    pair_ids: list[str] = []
    values: list[float] = []
    for row in rows:
        raw_value = str(row.get(field_name, "")).strip()
        if not raw_value:
            continue
        value = _safe_float(raw_value, field_name=field_name)
        if not math.isfinite(value):
            continue
        pair_ids.append(str(row["pair_id"]))
        values.append(value)
    mean, ci_low, ci_high = bootstrap_ci_pair_cluster(
        pair_ids,
        values,
        n_bootstrap=int(bootstrap_n),
        ci=float(ci),
        seed=int(seed),
    )
    return {
        "mean": float(mean),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def _difference_triplet_from_rows(
    rows: list[dict[str, str]],
    *,
    minuend_field: str,
    subtrahend_field: str,
    bootstrap_n: int,
    ci: float,
    seed: int,
) -> dict[str, float]:
    pair_ids: list[str] = []
    values: list[float] = []
    for row in rows:
        raw_left = str(row.get(minuend_field, "")).strip()
        raw_right = str(row.get(subtrahend_field, "")).strip()
        if not raw_left or not raw_right:
            continue
        left = _safe_float(raw_left, field_name=minuend_field)
        right = _safe_float(raw_right, field_name=subtrahend_field)
        diff = float(left - right)
        if not math.isfinite(diff):
            continue
        pair_ids.append(str(row["pair_id"]))
        values.append(diff)
    mean, ci_low, ci_high = bootstrap_ci_pair_cluster(
        pair_ids,
        values,
        n_bootstrap=int(bootstrap_n),
        ci=float(ci),
        seed=int(seed),
    )
    return {
        "mean": float(mean),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def _filter_comparability_rows(rows: list[dict[str, str]], *, layer: int) -> list[dict[str, str]]:
    out = []
    for row in rows:
        if int(row.get("layer", "-1")) != int(layer):
            continue
        if not _truthy(row.get("analysis_included", False)):
            continue
        out.append(row)
    if not out:
        raise ValueError(f"No analysis-included comparability rows for layer {layer}")
    return out


def _finite_mean_values(values: Iterable[float], *, field_name: str) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        raise ValueError(f"No finite values for {field_name}")
    return float(sum(finite) / len(finite))


def _row_target(row: dict[str, Any]) -> str:
    target = str(row.get("target", "")).strip()
    if target:
        return target
    pair_id = str(row.get("pair_id", "")).strip()
    if "-" in pair_id:
        return pair_id.split("-", 1)[0]
    if "__" in pair_id:
        return pair_id.split("__", 1)[0]
    if pair_id:
        return pair_id
    raise ValueError(f"Cannot infer target for row without target or pair_id: {row!r}")


ROBUSTNESS_INPUT_CASE_TARGET_FIELDS = (
    "layer",
    "pair_id",
    "target",
    "effect_A",
    "effect_C",
    "sae_minus_raw",
    "included_in_robustness",
)

GATE_DIAGNOSTICS_SUMMARY_FIELDS = (
    "layer",
    "n_all_arms_success",
    "n_invariant_gate_fail",
    "n_gate_activation_fail",
    "n_gate_margin_fail",
    "n_gate_score_fail",
    "n_gate_clt_equiv_fail",
    "n_gate_identity_fail",
    "max_gate_margin_abs_diff",
    "max_gate_score_abs_diff_exp",
    "max_gate_score_abs_diff_other",
    "max_gate_clt_effect_abs_diff_C",
    "max_gate_clt_effect_abs_diff_D",
    "max_gate_identity_abs_effect_raw",
    "max_gate_identity_abs_effect_clt",
)

GATE_DIAGNOSTICS_ROW_FIELDS = (
    "layer",
    "pair_id",
    "direction",
    "target",
    "gate_activation_pass",
    "gate_margin_pass",
    "gate_score_pass",
    "gate_clt_equiv_pass",
    "gate_identity_pass",
    "gate_activation_ratio",
    "gate_margin_abs_diff",
    "gate_score_abs_diff_exp",
    "gate_score_abs_diff_other",
    "gate_clt_effect_abs_diff_C",
    "gate_clt_effect_abs_diff_D",
    "gate_identity_abs_effect_raw",
    "gate_identity_abs_effect_clt",
)

STRICT_GATE_SENSITIVITY_FIELDS = (
    "layer",
    "n_rows_strict_gate",
    "n_pairs_strict_gate",
    "d_CA_mean",
    "d_CA_ci_low",
    "d_CA_ci_high",
    "target_sign_flip_observed_mean",
    "target_sign_flip_p_two_sided",
    "target_sign_flip_n_units",
    "target_sign_flip_n_permutations",
    "target_sign_flip_permutation_mode",
    "seed",
)


def _parse_bool_strict(value: object, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Could not parse boolean for {field_name}: {value!r}")


def _finite_or_blank(value: object, *, field_name: str) -> float | str:
    if value is None or str(value).strip() == "":
        return ""
    parsed = _safe_float(value, field_name=field_name)
    return float(parsed) if math.isfinite(parsed) else ""


def _max_finite(rows: list[dict[str, str]], field_name: str) -> float | str:
    values: list[float] = []
    for row in rows:
        raw = str(row.get(field_name, "")).strip()
        if not raw:
            continue
        value = _safe_float(raw, field_name=field_name)
        if math.isfinite(value):
            values.append(abs(float(value)))
    return float(max(values)) if values else ""


# ---------------------------------------------------------------------------
# Gate diagnostics and robustness analyses
# ---------------------------------------------------------------------------


def build_gate_diagnostics(
    comparability_rows: list[dict[str, str]],
    *,
    layers: Iterable[int] = (4, 8),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build invariant-gate diagnostic tables from comparability CSV rows.

    Returns (summary_rows, detail_rows):
      summary_rows : one row per layer with all-arms-success counts,
                     invariant failure counts, per-gate failure counts, and
                     max gate magnitudes.
      detail_rows  : one row per failed directed row, including pair_id,
                     direction, target, per-gate pass flags, and magnitudes.
    Only rows with all_arms_success=True are included in the gate analysis.
    """
    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for layer in tuple(int(layer) for layer in layers):
        success_rows = [
            row
            for row in comparability_rows
            if int(row.get("layer", "-1")) == int(layer)
            and _truthy(row.get("all_arms_success", False))
        ]
        gate_fail_rows = [
            row
            for row in success_rows
            if not _truthy(row.get("invariant_all_pass", False))
        ]
        summary_rows.append(
            {
                "layer": int(layer),
                "n_all_arms_success": int(len(success_rows)),
                "n_invariant_gate_fail": int(len(gate_fail_rows)),
                "n_gate_activation_fail": int(sum(not _truthy(row.get("gate_activation_pass", False)) for row in gate_fail_rows)),
                "n_gate_margin_fail": int(sum(not _truthy(row.get("gate_margin_pass", False)) for row in gate_fail_rows)),
                "n_gate_score_fail": int(sum(not _truthy(row.get("gate_score_pass", False)) for row in gate_fail_rows)),
                "n_gate_clt_equiv_fail": int(sum(not _truthy(row.get("gate_clt_equiv_pass", False)) for row in gate_fail_rows)),
                "n_gate_identity_fail": int(sum(not _truthy(row.get("gate_identity_pass", False)) for row in gate_fail_rows)),
                "max_gate_margin_abs_diff": _max_finite(success_rows, "gate_margin_abs_diff"),
                "max_gate_score_abs_diff_exp": _max_finite(success_rows, "gate_score_abs_diff_exp"),
                "max_gate_score_abs_diff_other": _max_finite(success_rows, "gate_score_abs_diff_other"),
                "max_gate_clt_effect_abs_diff_C": _max_finite(success_rows, "gate_clt_effect_abs_diff_C"),
                "max_gate_clt_effect_abs_diff_D": _max_finite(success_rows, "gate_clt_effect_abs_diff_D"),
                "max_gate_identity_abs_effect_raw": _max_finite(success_rows, "gate_identity_abs_effect_raw"),
                "max_gate_identity_abs_effect_clt": _max_finite(success_rows, "gate_identity_abs_effect_clt"),
            }
        )
        for row in gate_fail_rows:
            detail_rows.append(
                {
                    "layer": int(layer),
                    "pair_id": str(row.get("pair_id", "")),
                    "direction": str(row.get("direction", "")),
                    "target": _row_target(dict(row)),
                    "gate_activation_pass": bool(_truthy(row.get("gate_activation_pass", False))),
                    "gate_margin_pass": bool(_truthy(row.get("gate_margin_pass", False))),
                    "gate_score_pass": bool(_truthy(row.get("gate_score_pass", False))),
                    "gate_clt_equiv_pass": bool(_truthy(row.get("gate_clt_equiv_pass", False))),
                    "gate_identity_pass": bool(_truthy(row.get("gate_identity_pass", False))),
                    "gate_activation_ratio": _finite_or_blank(row.get("gate_activation_ratio", ""), field_name="gate_activation_ratio"),
                    "gate_margin_abs_diff": _finite_or_blank(row.get("gate_margin_abs_diff", ""), field_name="gate_margin_abs_diff"),
                    "gate_score_abs_diff_exp": _finite_or_blank(row.get("gate_score_abs_diff_exp", ""), field_name="gate_score_abs_diff_exp"),
                    "gate_score_abs_diff_other": _finite_or_blank(row.get("gate_score_abs_diff_other", ""), field_name="gate_score_abs_diff_other"),
                    "gate_clt_effect_abs_diff_C": _finite_or_blank(row.get("gate_clt_effect_abs_diff_C", ""), field_name="gate_clt_effect_abs_diff_C"),
                    "gate_clt_effect_abs_diff_D": _finite_or_blank(row.get("gate_clt_effect_abs_diff_D", ""), field_name="gate_clt_effect_abs_diff_D"),
                    "gate_identity_abs_effect_raw": _finite_or_blank(row.get("gate_identity_abs_effect_raw", ""), field_name="gate_identity_abs_effect_raw"),
                    "gate_identity_abs_effect_clt": _finite_or_blank(row.get("gate_identity_abs_effect_clt", ""), field_name="gate_identity_abs_effect_clt"),
                }
            )
    return summary_rows, detail_rows


def build_strict_gate_sensitivity(
    comparability_rows: list[dict[str, str]],
    *,
    layers: Iterable[int] = (4, 8),
    bootstrap_n: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Compute sensitivity restricted to rows passing existing invariant gates.

    Filters rows to all_arms_success=True and invariant_all_pass=True, then
    recomputes d_CA bootstrap intervals and target-level sign-flip robustness.
    Returns one row per layer.
    """
    rows_out: list[dict[str, Any]] = []
    for layer in tuple(int(layer) for layer in layers):
        strict_rows = [
            row
            for row in comparability_rows
            if int(row.get("layer", "-1")) == int(layer)
            and _truthy(row.get("all_arms_success", False))
            and _truthy(row.get("invariant_all_pass", False))
            and math.isfinite(_safe_float(row.get("effect_A", ""), field_name="effect_A"))
            and math.isfinite(_safe_float(row.get("effect_C", ""), field_name="effect_C"))
        ]
        if not strict_rows:
            raise ValueError(f"No strict-gate sensitivity rows for layer {layer}")
        triplet = _difference_triplet_from_rows(
            strict_rows,
            minuend_field="effect_C",
            subtrahend_field="effect_A",
            bootstrap_n=bootstrap_n,
            ci=ci,
            seed=seed,
        )
        pair_payload = _pair_level_effect_inputs(strict_rows, layer=layer)
        values_by_pair = {
            pair_id: float(payload["sae_minus_raw"])
            for pair_id, payload in pair_payload.items()
        }
        pair_targets = {
            pair_id: str(payload["target"])
            for pair_id, payload in pair_payload.items()
        }
        target_row = next(
            row
            for row in _comparison_robustness_rows(
                comparison=f"l{layer}_strict_gate_sae_minus_raw",
                layer=layer,
                values_by_pair=values_by_pair,
                pair_targets=pair_targets,
                n_permutations=10000,
                seed=seed,
            )
            if row["test"] == "target_sign_flip"
        )
        rows_out.append(
            {
                "layer": int(layer),
                "n_rows_strict_gate": int(len(strict_rows)),
                "n_pairs_strict_gate": int(len(pair_payload)),
                "d_CA_mean": float(triplet["mean"]),
                "d_CA_ci_low": float(triplet["ci_low"]),
                "d_CA_ci_high": float(triplet["ci_high"]),
                "target_sign_flip_observed_mean": float(target_row["observed_mean"]),
                "target_sign_flip_p_two_sided": float(target_row["p_two_sided"]),
                "target_sign_flip_n_units": int(target_row["n_units"]),
                "target_sign_flip_n_permutations": int(target_row["n_permutations"]),
                "target_sign_flip_permutation_mode": str(target_row["permutation_mode"]),
                "seed": int(seed),
            }
        )
    return rows_out


def _mean_effect_payload(
    rows: list[dict[str, str]],
    *,
    pair_id: str,
) -> dict[str, float]:
    effects_a: list[float] = []
    effects_c: list[float] = []
    sae_minus_raw: list[float] = []
    for row in rows:
        effect_a = _safe_float(row.get("effect_A", ""), field_name="effect_A")
        effect_c = _safe_float(row.get("effect_C", ""), field_name="effect_C")
        if not (math.isfinite(effect_a) and math.isfinite(effect_c)):
            continue
        effects_a.append(effect_a)
        effects_c.append(effect_c)
        sae_minus_raw.append(float(effect_c - effect_a))
    return {
        "effect_A": _finite_mean_values(effects_a, field_name=f"{pair_id}.effect_A"),
        "effect_C": _finite_mean_values(effects_c, field_name=f"{pair_id}.effect_C"),
        "sae_minus_raw": _finite_mean_values(sae_minus_raw, field_name=f"{pair_id}.sae_minus_raw"),
    }


def _pair_level_effect_inputs(
    comparability_rows: list[dict[str, str]],
    *,
    layer: int,
) -> dict[str, dict[str, Any]]:
    # This intentionally mirrors _pair_level_sae_minus_raw while retaining the
    # two input effects needed by the public sufficient-statistic CSV.
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in _filter_comparability_rows(comparability_rows, layer=layer):
        pair_id = str(row.get("pair_id", "")).strip()
        if not pair_id:
            raise ValueError(f"Missing pair_id in comparability row for layer {layer}: {row!r}")
        grouped.setdefault(pair_id, []).append(row)

    out: dict[str, dict[str, Any]] = {}
    for pair_id, pair_rows in sorted(grouped.items()):
        targets = {_row_target(dict(row)) for row in pair_rows}
        if len(targets) != 1:
            raise ValueError(
                f"Conflicting target values for pair_id={pair_id!r}, layer={layer}: {sorted(targets)!r}"
            )
        effect_payload = _mean_effect_payload(pair_rows, pair_id=pair_id)
        out[pair_id] = {
            "target": targets.pop(),
            **effect_payload,
            "included_in_robustness": True,
        }
    if not out:
        raise ValueError(f"No finite pair-level effect rows for layer {layer}")
    return out


def _pair_level_sae_minus_raw(
    comparability_rows: list[dict[str, str]],
    *,
    layer: int,
) -> dict[str, dict[str, Any]]:
    pair_values = _pair_level_effect_inputs(comparability_rows, layer=layer)
    return {
        pair_id: {
            "target": payload["target"],
            "sae_minus_raw": payload["sae_minus_raw"],
        }
        for pair_id, payload in pair_values.items()
    }


def _values_by_target(
    pair_values: dict[str, dict[str, Any]],
    *,
    pair_ids: Iterable[str] | None = None,
    value_key: str = "sae_minus_raw",
) -> dict[str, list[float]]:
    selected = set(str(pair_id) for pair_id in pair_ids) if pair_ids is not None else set(pair_values)
    out: dict[str, list[float]] = {}
    for pair_id in sorted(selected):
        payload = pair_values[pair_id]
        target = str(payload["target"])
        value = float(payload[value_key])
        if math.isfinite(value):
            out.setdefault(target, []).append(value)
    return out


def build_robustness_input_case_target(
    comparability_rows: list[dict[str, str]],
    *,
    layers: Iterable[int] = (4, 8),
) -> list[dict[str, Any]]:
    """Extract per-pair sufficient statistics for limitation robustness tests.

    The output is intentionally one row per ``(layer, pair_id)`` and contains
    all source-visible pair rows. ``included_in_robustness`` is the public
    inference boundary: rows may remain audit-visible while being excluded from
    the robustness tests.
    """
    layer_list = tuple(int(layer) for layer in layers)
    by_layer: dict[int, dict[str, dict[str, Any]]] = {layer: {} for layer in layer_list}
    grouped: dict[tuple[int, str], list[dict[str, str]]] = {}
    for row in comparability_rows:
        try:
            layer = int(row.get("layer", "-1"))
        except Exception as exc:
            raise ValueError(f"Invalid layer in comparability row: {row!r}") from exc
        if layer not in by_layer:
            continue
        pair_id = str(row.get("pair_id", "")).strip()
        if not pair_id:
            raise ValueError(f"Missing pair_id in comparability row for layer {layer}: {row!r}")
        grouped.setdefault((layer, pair_id), []).append(row)

    for (layer, pair_id), pair_rows in sorted(grouped.items()):
        targets = {_row_target(dict(row)) for row in pair_rows}
        if len(targets) != 1:
            raise ValueError(
                f"Conflicting target values for pair_id={pair_id!r}, layer={layer}: {sorted(targets)!r}"
            )
        included_rows = [
            row
            for row in pair_rows
            if _truthy(row.get("analysis_included", False))
            and math.isfinite(_safe_float(row.get("effect_A", ""), field_name="effect_A"))
            and math.isfinite(_safe_float(row.get("effect_C", ""), field_name="effect_C"))
        ]
        included_in_robustness = bool(included_rows)
        effect_rows = included_rows if included_in_robustness else pair_rows
        effect_payload = _mean_effect_payload(effect_rows, pair_id=pair_id)
        by_layer[layer][pair_id] = {
            "layer": int(layer),
            "pair_id": pair_id,
            "target": targets.pop(),
            "effect_A": float(effect_payload["effect_A"]),
            "effect_C": float(effect_payload["effect_C"]),
            "sae_minus_raw": float(effect_payload["sae_minus_raw"]),
            "included_in_robustness": bool(included_in_robustness),
        }

    pair_sets = {layer: set(by_layer[layer]) for layer in layer_list}
    if any(not pairs for pairs in pair_sets.values()):
        missing_layers = [layer for layer, pairs in pair_sets.items() if not pairs]
        raise ValueError(f"No robustness input rows for layers: {missing_layers}")
    first_layer = layer_list[0]
    first_pairs = pair_sets[first_layer]
    for layer in layer_list[1:]:
        if pair_sets[layer] != first_pairs:
            missing = sorted(first_pairs - pair_sets[layer])
            extra = sorted(pair_sets[layer] - first_pairs)
            raise ValueError(
                f"Robustness input pair set mismatch for layer {layer}: missing={missing}, extra={extra}"
            )

    out: list[dict[str, Any]] = []
    for layer in layer_list:
        for pair_id in sorted(by_layer[layer]):
            out.append(by_layer[layer][pair_id])
    return out


def _coerce_case_target_rows(
    rows: list[dict[str, Any]],
    *,
    layers: Iterable[int],
) -> dict[int, dict[str, dict[str, Any]]]:
    expected = set(ROBUSTNESS_INPUT_CASE_TARGET_FIELDS)
    layer_list = tuple(int(layer) for layer in layers)
    layer_set = set(layer_list)
    out: dict[int, dict[str, dict[str, Any]]] = {int(layer): {} for layer in layer_set}
    for raw_row in rows:
        observed = set(raw_row)
        if observed != expected:
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise ValueError(f"Robustness input row schema mismatch: missing={missing}, extra={extra}")
        layer = int(raw_row["layer"])
        if layer not in layer_set:
            continue
        pair_id = str(raw_row["pair_id"]).strip()
        if not pair_id:
            raise ValueError(f"Robustness input row has empty pair_id: {raw_row!r}")
        target = _row_target(dict(raw_row))
        effect_a = _safe_float(raw_row["effect_A"], field_name="effect_A")
        effect_c = _safe_float(raw_row["effect_C"], field_name="effect_C")
        sae_minus_raw = _safe_float(raw_row["sae_minus_raw"], field_name="sae_minus_raw")
        included = _parse_bool_strict(raw_row["included_in_robustness"], field_name="included_in_robustness")
        if not (math.isfinite(effect_a) and math.isfinite(effect_c) and math.isfinite(sae_minus_raw)):
            raise ValueError(f"Robustness input row has non-finite effect value: {raw_row!r}")
        recomputed = float(effect_c - effect_a)
        if not math.isclose(recomputed, sae_minus_raw, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"sae_minus_raw mismatch for layer={layer}, pair_id={pair_id!r}: "
                f"effect_C - effect_A = {recomputed!r}, row has {sae_minus_raw!r}"
            )
        if pair_id in out[layer]:
            raise ValueError(f"Duplicate robustness input row for layer={layer}, pair_id={pair_id!r}")
        out[layer][pair_id] = {
            "target": target,
            "effect_A": effect_a,
            "effect_C": effect_c,
            "sae_minus_raw": sae_minus_raw,
            "included_in_robustness": included,
        }
    for layer in layer_list:
        payload = out[layer]
        if not payload:
            raise ValueError(f"No robustness input rows for layer {layer}")
    first_layer = layer_list[0]
    first_pairs = set(out[first_layer])
    for layer in layer_list[1:]:
        layer_pairs = set(out[layer])
        if layer_pairs != first_pairs:
            missing = sorted(first_pairs - layer_pairs)
            extra = sorted(layer_pairs - first_pairs)
            raise ValueError(
                f"Robustness input pair set mismatch for layer {layer}: missing={missing}, extra={extra}"
            )
    for pair_id in sorted(first_pairs):
        targets = {str(out[layer][pair_id]["target"]) for layer in layer_list}
        if len(targets) != 1:
            raise ValueError(f"Target mismatch for shared pair_id={pair_id!r}: {sorted(targets)!r}")
    return out


def _mean(values: Iterable[float]) -> float:
    vals = [float(value) for value in values]
    if not vals:
        raise ValueError("Cannot compute mean of empty values")
    return float(sum(vals) / len(vals))


def _sign_flip_p_two_sided(
    values: list[float],
    *,
    n_permutations: int,
    seed: int,
) -> tuple[float, int, str]:
    vals = [float(value) for value in values if math.isfinite(float(value))]
    if not vals:
        raise ValueError("Cannot run sign-flip test with no finite values")
    observed_abs_sum = abs(sum(vals))
    n_units = len(vals)
    if n_units <= 20:
        total = 1 << n_units
        extreme = 0
        for mask in range(total):
            signed_sum = 0.0
            for idx, value in enumerate(vals):
                signed_sum += value if ((mask >> idx) & 1) else -value
            if abs(signed_sum) >= observed_abs_sum - 1e-15:
                extreme += 1
        return float(extreme / total), int(total), "exact"

    rng = random.Random(int(seed))
    draws = int(n_permutations)
    extreme = 0
    for _ in range(draws):
        signed_sum = sum(value if rng.random() < 0.5 else -value for value in vals)
        if abs(signed_sum) >= observed_abs_sum - 1e-15:
            extreme += 1
    return float((extreme + 1) / (draws + 1)), int(draws), "monte_carlo"


def _robustness_summary_row(
    *,
    test: str,
    unit: str,
    comparison: str,
    heldout_target: str,
    values: list[float],
    n_pairs: int,
    n_targets: int,
    n_permutations: int,
    seed: int,
    target_means: dict[str, float],
    leave_one_target_means: dict[str, float],
) -> dict[str, Any]:
    p_two_sided, actual_permutations, permutation_mode = _sign_flip_p_two_sided(
        values,
        n_permutations=n_permutations,
        seed=seed,
    )
    loto_values = [float(value) for value in leave_one_target_means.values()]
    return {
        "test": test,
        "comparison": comparison,
        "layer": "",
        "unit": unit,
        "heldout_target": heldout_target,
        "observed_mean": _mean(values),
        "p_two_sided": p_two_sided,
        "n_units": int(len(values)),
        "n_pairs": int(n_pairs),
        "n_targets": int(n_targets),
        "n_permutations": int(actual_permutations),
        "permutation_mode": permutation_mode,
        "seed": int(seed),
        "target_means": json.dumps(target_means, sort_keys=True, separators=(",", ":")),
        "leave_one_target_means": json.dumps(leave_one_target_means, sort_keys=True, separators=(",", ":")),
        "loto_min": (float(min(loto_values)) if loto_values else float("nan")),
        "loto_max": (float(max(loto_values)) if loto_values else float("nan")),
        "loto_negative_count": int(sum(1 for value in loto_values if value < 0.0)),
        "loto_n": int(len(loto_values)),
    }


def _comparison_robustness_rows(
    *,
    comparison: str,
    layer: int | None,
    values_by_pair: dict[str, float],
    pair_targets: dict[str, str],
    n_permutations: int,
    seed: int,
) -> list[dict[str, Any]]:
    if not values_by_pair:
        raise ValueError(f"No included pairs for robustness comparison {comparison!r}")
    target_values: dict[str, list[float]] = {}
    for pair_id, value in sorted(values_by_pair.items()):
        target_values.setdefault(pair_targets[pair_id], []).append(float(value))
    target_means = {target: _mean(values) for target, values in sorted(target_values.items())}
    leave_one_target_means = {
        target: _mean([value for other, value in target_means.items() if other != target])
        for target in sorted(target_means)
        if len(target_means) > 1
    }

    rows_out = [
        _robustness_summary_row(
            test="case_sign_flip",
            comparison=comparison,
            unit="pair",
            heldout_target="",
            values=[values_by_pair[pair_id] for pair_id in sorted(values_by_pair)],
            n_pairs=len(values_by_pair),
            n_targets=len(target_means),
            n_permutations=n_permutations,
            seed=seed,
            target_means=target_means,
            leave_one_target_means=leave_one_target_means,
        ),
        _robustness_summary_row(
            test="target_sign_flip",
            comparison=comparison,
            unit="target",
            heldout_target="",
            values=[target_means[target] for target in sorted(target_means)],
            n_pairs=len(values_by_pair),
            n_targets=len(target_means),
            n_permutations=n_permutations,
            seed=seed,
            target_means=target_means,
            leave_one_target_means=leave_one_target_means,
        ),
    ]
    for row in rows_out:
        row["layer"] = "" if layer is None else int(layer)

    for heldout_target in sorted(leave_one_target_means):
        kept_targets = [target for target in sorted(target_means) if target != heldout_target]
        kept_values = [target_means[target] for target in kept_targets]
        kept_pairs = sum(len(target_values[target]) for target in kept_targets)
        row = _robustness_summary_row(
            test="leave_one_target_out",
            comparison=comparison,
            unit="target",
            heldout_target=heldout_target,
            values=kept_values,
            n_pairs=kept_pairs,
            n_targets=len(kept_targets),
            n_permutations=n_permutations,
            seed=seed,
            target_means=target_means,
            leave_one_target_means=leave_one_target_means,
        )
        row["layer"] = "" if layer is None else int(layer)
        rows_out.append(row)
    return rows_out


def build_limitation_robustness_from_case_target(
    rows: list[dict[str, Any]],
    *,
    layers: Iterable[int] = (4, 8),
    n_permutations: int = 10000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Build robustness summaries from pair-level case-target rows.

    For each layer, tests sae_minus_raw = effect_C - effect_A at pair and
    target aggregation levels. For the two-layer comparison, tests the paired
    interaction contrast sae_minus_raw_low - sae_minus_raw_high. The target
    rows support the paper's target-level heterogeneity and leave-one-target-out
    robustness analysis. Requires exactly two layers (low, high).
    """
    layer_list = tuple(int(layer) for layer in layers)
    if len(layer_list) != 2:
        raise ValueError(f"Robustness tests require exactly two layers, got {layer_list!r}")
    low_layer, high_layer = layer_list
    by_layer = _coerce_case_target_rows(rows, layers=layer_list)
    included_by_layer = {
        layer: {
            pair_id: payload
            for pair_id, payload in by_layer[layer].items()
            if bool(payload["included_in_robustness"])
        }
        for layer in layer_list
    }
    shared_pair_ids = sorted(set(included_by_layer[low_layer]) & set(included_by_layer[high_layer]))
    if not shared_pair_ids:
        raise ValueError(f"No shared pair_ids across robustness layers {layer_list!r}")

    rows_out: list[dict[str, Any]] = []
    for layer in layer_list:
        layer_values = {
            pair_id: float(payload["sae_minus_raw"])
            for pair_id, payload in included_by_layer[layer].items()
        }
        layer_targets = {
            pair_id: str(by_layer[layer][pair_id]["target"])
            for pair_id in layer_values
        }
        rows_out.extend(
            _comparison_robustness_rows(
                comparison=f"l{layer}_sae_minus_raw",
                layer=layer,
                values_by_pair=layer_values,
                pair_targets=layer_targets,
                n_permutations=n_permutations,
                seed=seed,
            )
        )

    contrast_values: dict[str, float] = {}
    contrast_targets: dict[str, str] = {}
    for pair_id in shared_pair_ids:
        low_payload = by_layer[low_layer][pair_id]
        high_payload = by_layer[high_layer][pair_id]
        if str(low_payload["target"]) != str(high_payload["target"]):
            raise ValueError(
                f"Target mismatch for shared pair_id={pair_id!r}: "
                f"L{low_layer}={low_payload['target']!r}, L{high_layer}={high_payload['target']!r}"
            )
        contrast_targets[pair_id] = str(low_payload["target"])
        contrast_values[pair_id] = float(low_payload["sae_minus_raw"]) - float(high_payload["sae_minus_raw"])
    rows_out.extend(
        _comparison_robustness_rows(
            comparison=f"l{low_layer}_minus_l{high_layer}_sae_minus_raw",
            layer=None,
            values_by_pair=contrast_values,
            pair_targets=contrast_targets,
            n_permutations=n_permutations,
            seed=seed,
        )
    )
    return rows_out


def build_limitation_robustness(
    comparability_rows: list[dict[str, str]],
    *,
    layers: Iterable[int] = (4, 8),
    n_permutations: int = 10000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    case_target_rows = build_robustness_input_case_target(comparability_rows, layers=layers)
    return build_limitation_robustness_from_case_target(
        case_target_rows,
        layers=layers,
        n_permutations=n_permutations,
        seed=seed,
    )


def _topk_triplet(topk_summary: dict[str, Any], *, layer: int, field_family: str, k: int) -> dict[str, float]:
    layer_curves = topk_summary["curves_by_layer"][str(int(layer))]
    effects = layer_curves["effects"][field_family][str(int(k))]
    return {
        "mean": float(effects["mean"]),
        "ci_low": float(effects["ci_low"]),
        "ci_high": float(effects["ci_high"]),
    }


# ---------------------------------------------------------------------------
# Derived numbers and auxiliary public-table payload builder
# ---------------------------------------------------------------------------


def build_limitation_derived_outputs(
    *,
    comparability_summary: dict[str, Any],
    comparability_rows: list[dict[str, str]],
    topk_summary: dict[str, Any],
    source_run_root: Path,
) -> DerivedOutputs:
    """Derive the main public numbers payload and auxiliary public tables.

    Assumes source summaries have already been validated by the caller. Extracts
    Gemma-3 centerpiece values plus L4 PCA/STRESS/top-k summaries. It does not
    run experiments and does not write files.
    """
    profile = limitation_requirements.LIMITATION_PROFILE
    bootstrap_n = int(comparability_summary["run_config"]["bootstrap_n"])
    bootstrap_seed = int(comparability_summary["run_config"]["bootstrap_seed"])
    ci = float(comparability_summary["run_config"]["ci"])

    gemma3_centerpiece_layers = tuple(int(layer) for layer in profile.paper_layers)
    gemma3_centerpiece: dict[str, Any] = {}
    stress_rows: list[dict[str, Any]] = []

    for layer in gemma3_centerpiece_layers:
        entry = _per_layer_entry(comparability_summary, layer)
        payload = {
            "layer": int(layer),
            "n_pairs": int(entry.get("n_pairs", comparability_summary["counts"]["n_pairs_analysis_included"])),
            "fidelity_cosine": _metric_triplet(entry, "fidelity_cosine"),
            "fidelity_rel_mse": _metric_triplet(entry, "fidelity_rel_mse"),
            "raw_effect": _metric_triplet(entry, "effect_A"),
            "sae_effect": _metric_triplet(entry, "effect_C"),
            "sae_minus_raw": _metric_triplet(entry, "d_CA"),
            "crr": _metric_triplet(entry, "crr_C_over_A"),
        }
        if _has_metric_triplet(entry, "fidelity_fvu"):
            payload["fidelity_fvu"] = _metric_triplet(entry, "fidelity_fvu")
        gemma3_centerpiece[str(layer)] = payload

    layer4_entry = _per_layer_entry(comparability_summary, 4)
    layer4_rows = _filter_comparability_rows(comparability_rows, layer=4)
    gemma3_pca_l4 = {
        "pca_effect": _metric_triplet(layer4_entry, "effect_PRJ_PCA"),
        "pca_minus_raw": _metric_triplet(layer4_entry, "d_PRJ_PCA_A"),
        "pca_minus_sae": _difference_triplet_from_rows(
            layer4_rows,
            minuend_field="effect_PRJ_PCA",
            subtrahend_field="effect_C",
            bootstrap_n=bootstrap_n,
            ci=ci,
            seed=bootstrap_seed,
        ),
        "random_mean_effect": _field_triplet_from_rows(
            layer4_rows,
            field_name="effect_PRJ_RAND_mean",
            bootstrap_n=bootstrap_n,
            ci=ci,
            seed=bootstrap_seed,
        ),
    }

    gemma3_stress_l4 = {
        "recon_effect": _metric_triplet(layer4_entry, "effect_STRESS_RECON"),
        "resid_effect": _metric_triplet(layer4_entry, "effect_STRESS_RESID"),
        "resid_minus_recon": _metric_triplet(layer4_entry, "d_STRESS_RESID_RECON"),
        "stress_delta_additivity_rel_err": _metric_triplet(layer4_entry, "stress_delta_additivity_rel_err"),
    }
    stress_rows.append(
        {
            "layer": 4,
            "recon_effect_mean": gemma3_stress_l4["recon_effect"]["mean"],
            "recon_effect_ci_low": gemma3_stress_l4["recon_effect"]["ci_low"],
            "recon_effect_ci_high": gemma3_stress_l4["recon_effect"]["ci_high"],
            "resid_effect_mean": gemma3_stress_l4["resid_effect"]["mean"],
            "resid_effect_ci_low": gemma3_stress_l4["resid_effect"]["ci_low"],
            "resid_effect_ci_high": gemma3_stress_l4["resid_effect"]["ci_high"],
            "resid_minus_recon_mean": gemma3_stress_l4["resid_minus_recon"]["mean"],
            "resid_minus_recon_ci_low": gemma3_stress_l4["resid_minus_recon"]["ci_low"],
            "resid_minus_recon_ci_high": gemma3_stress_l4["resid_minus_recon"]["ci_high"],
            "stress_delta_additivity_rel_err_mean": gemma3_stress_l4["stress_delta_additivity_rel_err"]["mean"],
            "stress_delta_additivity_rel_err_ci_low": gemma3_stress_l4["stress_delta_additivity_rel_err"]["ci_low"],
            "stress_delta_additivity_rel_err_ci_high": gemma3_stress_l4["stress_delta_additivity_rel_err"]["ci_high"],
        }
    )

    gemma3_topk_l4 = {
        "counts": {
            "n_eval_cases": int(topk_summary["curves_by_layer"]["4"]["n_eval_cases"]),
            "n_total_directions": int(topk_summary["curves_by_layer"]["4"]["n_total_directions"]),
            "n_skipped_misaligned": int(topk_summary["curves_by_layer"]["4"]["n_skipped_misaligned"]),
        },
        "full_effect": _topk_triplet(
            topk_summary,
            layer=4,
            field_family="full",
            k=max(int(k) for k in topk_summary["curves_by_layer"]["4"]["effects"]["full"]),
        ),
        "topk": {
            str(k): _topk_triplet(topk_summary, layer=4, field_family="topk", k=k)
            for k in profile.compact_ks
        },
        "concentration": {
            "gini": float(topk_summary["concentration_by_layer"]["4"]["gini"]),
            "mass_at_20": float(topk_summary["concentration_by_layer"]["4"]["mass_at_20"]),
            "mass_at_50": float(topk_summary["concentration_by_layer"]["4"]["mass_at_50"]),
            "mass_at_100": float(topk_summary["concentration_by_layer"]["4"]["mass_at_100"]),
        },
    }

    numbers = {
        "publication_profile": limitation_requirements.LIMITATION_PUBLICATION_PROFILE,
        "execution_profile_id": limitation_requirements.LIMITATION_EXECUTION_PROFILE_ID,
        "profile": limitation_requirements.limitation_profile_summary(profile),
        "source_artifacts": {
            **_source_artifact_fields(
                "comparability_summary",
                limitation_requirements.limitation_source_comparability_summary_path(source_run_root),
                context="Limitation derived comparability source summary",
                source_artifact_root=source_run_root,
            ),
            **_source_artifact_fields(
                "comparability_csv",
                limitation_requirements.limitation_source_comparability_csv_path(source_run_root),
                context="Limitation derived comparability source CSV",
                source_artifact_root=source_run_root,
            ),
            **_source_artifact_fields(
                "topk_summary",
                limitation_requirements.limitation_source_topk_summary_path(source_run_root),
                context="Limitation derived top-k source summary",
                source_artifact_root=source_run_root,
            ),
            **_source_artifact_fields(
                "topk_csv",
                limitation_requirements.limitation_source_topk_csv_path(source_run_root),
                context="Limitation derived top-k source CSV",
                source_artifact_root=source_run_root,
            ),
        },
        "gemma3_centerpiece": gemma3_centerpiece,
        "gemma3_pca_l4": gemma3_pca_l4,
        "gemma3_stress_l4": gemma3_stress_l4,
        "gemma3_topk_l4": gemma3_topk_l4,
    }
    return DerivedOutputs(
        numbers=numbers,
        stress_rows=stress_rows,
    )
