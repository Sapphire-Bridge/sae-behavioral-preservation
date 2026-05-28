#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
UNSET = "__SET_ME__"

LIMITATION_PUBLICATION_PROFILE = "sae_writeback_limitation"
LIMITATION_EXECUTION_PROFILE_ID = "sae_writeback_limitation_v1"
LIMITATION_COMPARABILITY_SCHEMA_VERSION = "sae_writeback_limitation_comparability_v1"
LIMITATION_TOPK_SCHEMA_VERSION = "sae_writeback_limitation_topk_v1"

LIMITATION_LOCAL_CLT_BUNDLE_PATH = ROOT / "clt_bundles" / f"{LIMITATION_PUBLICATION_PROFILE}_release"
LIMITATION_RESULTS_ROOT = ROOT / "results" / f"{LIMITATION_PUBLICATION_PROFILE}_release"
LIMITATION_TABLES_ROOT = ROOT / "tables" / f"{LIMITATION_PUBLICATION_PROFILE}_release"
LIMITATION_FIGURES_ROOT = ROOT / "figures" / f"{LIMITATION_PUBLICATION_PROFILE}_release"

LIMITATION_SOURCE_COMPARABILITY_BASENAME = "gemma3_4b_comparability"
LIMITATION_SOURCE_TOPK_BASENAME = "gemma3_4b_topk"


@dataclass(frozen=True)
class LimitationSaeSourceEntry:
    source_entry: str
    expected_l0: int
    bundle_run_name: str | None = None

    def resolved_bundle_run_name(self) -> str:
        raw = str(self.bundle_run_name or "").strip()
        if raw:
            return raw
        return Path(str(self.source_entry)).name


@dataclass(frozen=True)
class LimitationProfile:
    model_id: str = "google/gemma-3-4b-pt"
    model_revision: str = "cc012e0a6d0787b4adcc0fa2c4da74402494554d"
    tokenizer_revision: str = "cc012e0a6d0787b4adcc0fa2c4da74402494554d"
    dataset_bundle_id: str = "db5cb11757fce2cc6f803baedb11173c0ccee311c839fe8103cb4f974bcba0c6"
    dataset_manifest_sha256: str = "81bc6129775ff29a650a1eebfbfe9dded6a3923e877b740fa744989440d564d0"
    sae_bundle_id: str = "c50c24b2b45322e28d2eae9f7f7cf82f55cb523653010d06977fde3b60a79f84"
    sae_bundle_manifest_sha256: str = "03bff6c4498fdcb1b5298dbae7d95992a74f08ccbef5a155837108d7a9b04199"
    sae_repo_id: str = "google/gemma-scope-2-4b-pt"  # Historical repo name; local scope configs under this snapshot declare model_name=google/gemma-3-4b-pt.
    sae_repo_revision: str = "a0ffd6132a985bc84077a66d1a1033e10b604fa8"
    sae_width: str = "16k"
    sae_source_entries: Mapping[int, LimitationSaeSourceEntry] = field(
        default_factory=lambda: {
            4: LimitationSaeSourceEntry(
                source_entry="resid_post_all/layer_4_width_16k_l0_big",
                expected_l0=81,
            ),
            5: LimitationSaeSourceEntry(
                source_entry="resid_post_all/layer_5_width_16k_l0_big",
                expected_l0=86,
            ),
            8: LimitationSaeSourceEntry(
                source_entry="resid_post_all/layer_8_width_16k_l0_big",
                expected_l0=102,
            ),
            11: LimitationSaeSourceEntry(
                source_entry="resid_post_all/layer_11_width_16k_l0_big",
                expected_l0=118,
            ),
            16: LimitationSaeSourceEntry(
                source_entry="resid_post_all/layer_16_width_16k_l0_big",
                expected_l0=120,
            ),
        }
    )
    dataset_disamb_path: str = "data_paper_hardened_v2/disamb_pairs.jsonl"
    paper_layers: tuple[int, ...] = (4, 5, 8, 11, 16)
    public_layers: tuple[int, ...] = (4, 8)
    topk_layers: tuple[int, ...] = (4, 8)
    compact_ks: tuple[int, ...] = (20, 50, 100)
    seed: int = 42
    bootstrap_seed: int = 42
    bootstrap_n: int = 1000
    ci: float = 0.95
    comparability_attn_implementation: str = "eager"
    comparability_torch_dtype: str | None = "float32"
    comparability_hard_fail_invariant: bool = False
    comparability_hard_fail_primary_logodds: bool = False
    comparability_primary_logodds_residual_tol: float = 5e-6
    comparability_extra_args: tuple[str, ...] = ()
    topk_seed: int = 0
    topk_split_seed: int = 0
    topk_frac_selection: float = 0.5
    topk_bootstrap_B: int = 1000
    topk_ci: float = 0.95
    topk_attn_implementation: str = "eager"
    topk_torch_dtype: str | None = "float32"
    topk_ks: tuple[int, ...] = (20, 50, 100)
    topk_logz_ks: tuple[int, ...] = (20, 50, 100)
    topk_random_control_mode: str | None = None
    topk_matched_bin_n_bins: int | None = None
    topk_extra_args: tuple[str, ...] = ()


LIMITATION_PROFILE = LimitationProfile()
LIMITATION_LAYERS = LIMITATION_PROFILE.public_layers
LIMITATION_COMPACT_KS = LIMITATION_PROFILE.compact_ks


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def relative_to_root(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    resolved = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_artifact(path: Path, *, context: str, root: Path) -> dict[str, str]:
    artifact_root = Path(root).expanduser().resolve()
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{context} does not exist: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"{context} must be a file: {resolved}")
    try:
        relative_path = resolved.relative_to(artifact_root).as_posix()
    except ValueError:
        raise ValueError(f"{context} must be inside artifact root {artifact_root}: {resolved}")
    return {"path": relative_path, "sha256": sha256_file(resolved)}


def public_repo_artifact(path: Path, *, context: str) -> dict[str, str]:
    try:
        return portable_artifact(path, context=context, root=ROOT)
    except ValueError as exc:
        message = str(exc).replace("inside artifact root", "inside repo root")
        raise ValueError(message)


def _is_unset(value: object) -> bool:
    text = str(value or "").strip()
    return text in {"", UNSET}


def compute_limitation_sae_bundle_id(manifest: Mapping[str, Any]) -> str:
    source = manifest.get("sae_source")
    if not isinstance(source, Mapping):
        raise ValueError("SAE bundle manifest missing mapping 'sae_source'")
    layers = manifest.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("SAE bundle manifest missing non-empty list 'layers'")

    canonical_layers: list[dict[str, Any]] = []
    for raw_row in layers:
        if not isinstance(raw_row, Mapping):
            raise ValueError("SAE bundle manifest layer rows must be JSON objects")
        canonical_layers.append(
            {
                "layer": int(raw_row.get("layer", -1)),
                "source_entry": str(raw_row.get("source_entry", "")).strip(),
                "expected_l0": int(raw_row.get("expected_l0", -1)),
                "source_params_sha256": str(raw_row.get("source_params_sha256", "")).strip(),
            }
        )
    canonical_layers.sort(key=lambda row: int(row["layer"]))
    payload = {
        "model": {
            "id": str(_required_manifest_mapping(manifest, "model").get("id", "")).strip(),
            "revision": str(_required_manifest_mapping(manifest, "model").get("revision", "")).strip(),
        },
        "sae_source": {
            "repo_id": str(source.get("repo_id", "")).strip(),
            "revision": str(source.get("revision", "")).strip(),
            "width": str(source.get("width", "")).strip(),
        },
        "layers": canonical_layers,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _required_manifest_mapping(manifest: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = manifest.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"SAE bundle manifest missing mapping {key!r}")
    return value


def limitation_profile_summary(profile: LimitationProfile | None = None) -> dict[str, Any]:
    p = profile or LIMITATION_PROFILE
    return {
        "model_id": p.model_id,
        "model_revision": p.model_revision,
        "tokenizer_revision": p.tokenizer_revision,
        "sae_repo_id": p.sae_repo_id,
        "sae_repo_revision": p.sae_repo_revision,
        "sae_width": p.sae_width,
        "paper_layers": list(p.paper_layers),
        "public_layers": list(p.public_layers),
        "topk_layers": list(p.topk_layers),
        "compact_ks": list(p.compact_ks),
        "comparability_hard_fail_invariant": bool(p.comparability_hard_fail_invariant),
        "comparability_hard_fail_primary_logodds": bool(p.comparability_hard_fail_primary_logodds),
        "comparability_primary_logodds_residual_tol": float(p.comparability_primary_logodds_residual_tol),
        "sae_source_entries": {
            str(layer): {
                "source_entry": entry.source_entry,
                "expected_l0": int(entry.expected_l0),
                "bundle_run_name": entry.resolved_bundle_run_name(),
            }
            for layer, entry in sorted(normalize_sae_source_entries(p).items())
        },
    }


def coerce_sae_source_entry(value: LimitationSaeSourceEntry | Mapping[str, Any]) -> LimitationSaeSourceEntry:
    if isinstance(value, LimitationSaeSourceEntry):
        return value
    if isinstance(value, Mapping):
        source_entry = str(value.get("source_entry", "")).strip()
        expected_l0 = value.get("expected_l0", UNSET)
        bundle_run_name = value.get("bundle_run_name")
        if _is_unset(source_entry) or _is_unset(expected_l0):
            raise ValueError(f"Invalid SAE source entry mapping: {value!r}")
        bundle_name = None if _is_unset(bundle_run_name) else str(bundle_run_name).strip()
        return LimitationSaeSourceEntry(
            source_entry=source_entry,
            expected_l0=int(expected_l0),
            bundle_run_name=bundle_name,
        )
    raise TypeError(f"Unsupported SAE source entry: {value!r}")


def normalize_sae_source_entries(profile: LimitationProfile | None = None) -> dict[int, LimitationSaeSourceEntry]:
    p = profile or LIMITATION_PROFILE
    return {int(layer): coerce_sae_source_entry(entry) for layer, entry in p.sae_source_entries.items()}


def limitation_sae_source_entry(profile: LimitationProfile | None, layer: int) -> LimitationSaeSourceEntry:
    entries = normalize_sae_source_entries(profile)
    try:
        return entries[int(layer)]
    except KeyError as exc:
        raise KeyError(f"Limitation profile missing SAE source entry for layer {layer}") from exc


def missing_profile_fields(profile: LimitationProfile | None = None, *, require_runs: bool) -> list[str]:
    p = profile or LIMITATION_PROFILE
    missing: list[str] = []
    for field_name in (
        "model_id",
        "model_revision",
        "tokenizer_revision",
        "dataset_bundle_id",
        "dataset_manifest_sha256",
        "sae_bundle_id",
        "sae_bundle_manifest_sha256",
        "sae_repo_id",
        "sae_repo_revision",
    ):
        if _is_unset(getattr(p, field_name)):
            missing.append(field_name)
    if require_runs:
        required_layers = set(int(layer) for layer in p.paper_layers)
        entry_map = normalize_sae_source_entries(p)
        for layer in sorted(required_layers):
            entry = entry_map.get(layer)
            if entry is None:
                missing.append(f"sae_source_entries[{layer}]")
                continue
            if _is_unset(entry.source_entry):
                missing.append(f"sae_source_entries[{layer}].source_entry")
            if _is_unset(entry.expected_l0):
                missing.append(f"sae_source_entries[{layer}].expected_l0")
    return missing


def missing_bundle_profile_fields(profile: LimitationProfile | None = None) -> list[str]:
    p = profile or LIMITATION_PROFILE
    missing: list[str] = []
    for field_name in (
        "model_id",
        "model_revision",
        "sae_repo_id",
        "sae_repo_revision",
    ):
        if _is_unset(getattr(p, field_name)):
            missing.append(field_name)
    entry_map = normalize_sae_source_entries(p)
    for layer in sorted(int(layer) for layer in p.paper_layers):
        entry = entry_map.get(layer)
        if entry is None:
            missing.append(f"sae_source_entries[{layer}]")
            continue
        if _is_unset(entry.source_entry):
            missing.append(f"sae_source_entries[{layer}].source_entry")
        if _is_unset(entry.expected_l0):
            missing.append(f"sae_source_entries[{layer}].expected_l0")
    return missing


def ensure_profile_configured(profile: LimitationProfile | None = None, *, require_runs: bool) -> LimitationProfile:
    p = profile or LIMITATION_PROFILE
    missing = missing_profile_fields(p, require_runs=require_runs)
    if missing:
        detail = ", ".join(missing)
        raise ValueError(
            "Limitation profile is incomplete. Fill the frozen Gemma-3 pins in "
            f"scripts/limitation_requirements.py first. Missing: {detail}"
        )
    return p


def ensure_bundle_profile_configured(profile: LimitationProfile | None = None) -> LimitationProfile:
    p = profile or LIMITATION_PROFILE
    missing = missing_bundle_profile_fields(p)
    if missing:
        detail = ", ".join(missing)
        raise ValueError(
            "Limitation bundle config is incomplete. Fill the frozen model/SAE pins in "
            f"scripts/limitation_requirements.py first. Missing: {detail}"
        )
    return p


def resolve_cached_snapshot(repo_id: str, revision: str | None) -> Path:
    from huggingface_hub import scan_cache_dir

    target_revision = str(revision or "").strip()
    for repo in scan_cache_dir().repos:
        if repo.repo_id != repo_id:
            continue
        if target_revision:
            for cached_revision in repo.revisions:
                if cached_revision.commit_hash == target_revision:
                    return Path(str(cached_revision.snapshot_path))
            raise FileNotFoundError(f"Cached revision {target_revision!r} not found for {repo_id}")
        if repo.revisions:
            chosen = sorted(repo.revisions, key=lambda item: item.commit_hash)[-1]
            return Path(str(chosen.snapshot_path))
    raise FileNotFoundError(f"No cached snapshot found for {repo_id}")


def limitation_comparability_summary_path(layer: int, *, root: Path | None = None) -> Path:
    base = Path(root) if root is not None else LIMITATION_RESULTS_ROOT
    return base / "comparability" / f"l{int(layer)}" / "comparability.summary.json"


def limitation_topk_summary_path(layer: int, *, root: Path | None = None) -> Path:
    base = Path(root) if root is not None else LIMITATION_RESULTS_ROOT
    return base / "topk" / f"l{int(layer)}" / "topk.summary.json"


def limitation_centerpiece_table_path(*, root: Path | None = None) -> Path:
    base = Path(root) if root is not None else LIMITATION_TABLES_ROOT
    return base / "centerpiece_summary.csv"


def limitation_topk_table_path(*, root: Path | None = None) -> Path:
    base = Path(root) if root is not None else LIMITATION_TABLES_ROOT
    return base / "topk_summary.csv"


def limitation_robustness_input_table_path(*, root: Path | None = None) -> Path:
    base = Path(root) if root is not None else LIMITATION_TABLES_ROOT
    # TODO: If public robustness expands beyond L4/L8, keep this schema one row
    # per analysis-included (layer, pair_id) and extend layers consistently.
    return base / "robustness_input_case_target.csv"


def limitation_robustness_summary_table_path(*, root: Path | None = None) -> Path:
    base = Path(root) if root is not None else LIMITATION_TABLES_ROOT
    return base / "robustness_summary.csv"


def limitation_gate_diagnostics_summary_table_path(*, root: Path | None = None) -> Path:
    base = Path(root) if root is not None else LIMITATION_TABLES_ROOT
    return base / "gate_diagnostics_summary.csv"


def limitation_gate_diagnostics_rows_table_path(*, root: Path | None = None) -> Path:
    base = Path(root) if root is not None else LIMITATION_TABLES_ROOT
    return base / "gate_diagnostics_rows.csv"


def limitation_strict_gate_sensitivity_table_path(*, root: Path | None = None) -> Path:
    base = Path(root) if root is not None else LIMITATION_TABLES_ROOT
    return base / "strict_gate_sensitivity.csv"


def limitation_release_manifest_path(*, root: Path | None = None) -> Path:
    base = Path(root) if root is not None else LIMITATION_TABLES_ROOT
    return base / "release_manifest.json"


def limitation_centerpiece_figure_path(*, root: Path | None = None) -> Path:
    base = Path(root) if root is not None else LIMITATION_FIGURES_ROOT
    return base / "centerpiece_summary.svg"


def limitation_topk_figure_path(*, root: Path | None = None) -> Path:
    base = Path(root) if root is not None else LIMITATION_FIGURES_ROOT
    return base / "topk_summary.svg"


def limitation_source_comparability_dir(run_root: Path) -> Path:
    return Path(run_root) / "source" / "comparability"


def limitation_source_topk_dir(run_root: Path) -> Path:
    return Path(run_root) / "source" / "topk"


def limitation_source_comparability_summary_path(run_root: Path) -> Path:
    return limitation_source_comparability_dir(run_root) / f"{LIMITATION_SOURCE_COMPARABILITY_BASENAME}.summary.json"


def limitation_source_comparability_csv_path(run_root: Path) -> Path:
    return limitation_source_comparability_dir(run_root) / f"{LIMITATION_SOURCE_COMPARABILITY_BASENAME}.csv"


def limitation_source_topk_summary_path(run_root: Path) -> Path:
    return limitation_source_topk_dir(run_root) / f"{LIMITATION_SOURCE_TOPK_BASENAME}.summary.json"


def limitation_source_topk_csv_path(run_root: Path) -> Path:
    return limitation_source_topk_dir(run_root) / f"{LIMITATION_SOURCE_TOPK_BASENAME}.csv"


def limitation_release_source_root(*, root: Path | None = None) -> Path:
    base = Path(root) if root is not None else LIMITATION_RESULTS_ROOT
    return base / "source"


def limitation_release_source_comparability_dir(*, root: Path | None = None) -> Path:
    return limitation_release_source_root(root=root) / "comparability"


def limitation_release_source_topk_dir(*, root: Path | None = None) -> Path:
    return limitation_release_source_root(root=root) / "topk"


def limitation_release_source_comparability_summary_path(*, root: Path | None = None) -> Path:
    return limitation_release_source_comparability_dir(root=root) / f"{LIMITATION_SOURCE_COMPARABILITY_BASENAME}.summary.json"


def limitation_release_source_comparability_csv_path(*, root: Path | None = None) -> Path:
    return limitation_release_source_comparability_dir(root=root) / f"{LIMITATION_SOURCE_COMPARABILITY_BASENAME}.csv"


def limitation_release_source_topk_summary_path(*, root: Path | None = None) -> Path:
    return limitation_release_source_topk_dir(root=root) / f"{LIMITATION_SOURCE_TOPK_BASENAME}.summary.json"


def limitation_derived_dir(run_root: Path) -> Path:
    return Path(run_root) / "derived"


def limitation_derived_numbers_path(run_root: Path) -> Path:
    return limitation_derived_dir(run_root) / "limitation_paper_numbers.json"


def limitation_stress_arm_summary_path(run_root: Path) -> Path:
    return limitation_derived_dir(run_root) / "stress_arm_summary.csv"


def limitation_release_results_root(run_root: Path) -> Path:
    return Path(run_root) / "release" / "results"


def limitation_release_tables_root(run_root: Path) -> Path:
    return Path(run_root) / "release" / "tables"


def limitation_release_figures_root(run_root: Path) -> Path:
    return Path(run_root) / "release" / "figures"


def limitation_report_path(run_root: Path) -> Path:
    return Path(run_root) / "limitation_paper_report.md"


def limitation_json_log_path(run_root: Path) -> Path:
    return Path(run_root) / "limitation_paper_log.json"


def limitation_bundle_manifest_path(bundle_root: Path | None = None) -> Path:
    base = Path(bundle_root) if bundle_root is not None else LIMITATION_LOCAL_CLT_BUNDLE_PATH
    return base / "BUNDLE_MANIFEST.json"


def load_public_comparability_reference(layer: int = 4) -> dict[str, Any]:
    path = limitation_comparability_summary_path(layer)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing public limitation reference summary: {relative_to_root(path)}. "
            "Build the public surface first with scripts/build_limitation_release_surface.py."
        )
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise TypeError(f"Invalid public limitation reference payload: {path}")
    return payload
