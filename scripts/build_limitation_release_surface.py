#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.limitation_requirements as limitation_requirements
from scripts.limitation_surface import (
    GATE_DIAGNOSTICS_ROW_FIELDS,
    GATE_DIAGNOSTICS_SUMMARY_FIELDS,
    STRICT_GATE_SENSITIVITY_FIELDS,
    build_gate_diagnostics,
    build_limitation_release_manifest,
    build_limitation_robustness_from_case_target,
    build_robustness_input_case_target,
    build_public_comparability_summary,
    build_public_topk_summary,
    build_strict_gate_sensitivity,
    configured_identity,
    load_json,
    read_csv_rows,
    render_centerpiece_figure,
    render_topk_figure,
    validate_source_summaries,
    write_centerpiece_table,
    write_csv_rows,
    write_csv_rows_with_fields,
    write_json,
    write_topk_table,
)


def _resolve_source_paths(
    *,
    source_run_root: str,
    comparability_summary: str,
    comparability_csv: str,
    topk_summary: str,
) -> tuple[Path, Path, Path]:
    if source_run_root:
        run_root = Path(source_run_root).expanduser().resolve()
        return (
            limitation_requirements.limitation_source_comparability_summary_path(run_root),
            limitation_requirements.limitation_source_comparability_csv_path(run_root),
            limitation_requirements.limitation_source_topk_summary_path(run_root),
        )
    if not comparability_summary or not comparability_csv or not topk_summary:
        raise ValueError(
            "Pass either --source_run_root or --comparability_summary, --comparability_csv, and --topk_summary."
        )
    return (
        Path(comparability_summary).expanduser().resolve(),
        Path(comparability_csv).expanduser().resolve(),
        Path(topk_summary).expanduser().resolve(),
    )


def _copy_file(src: Path, dst: Path) -> Path:
    src_resolved = Path(src).expanduser().resolve()
    dst_resolved = Path(dst).expanduser().resolve()
    if not src_resolved.exists():
        raise FileNotFoundError(f"Source artifact does not exist: {src_resolved}")
    dst_resolved.parent.mkdir(parents=True, exist_ok=True)
    if src_resolved != dst_resolved:
        shutil.copy2(src_resolved, dst_resolved)
    return dst_resolved


def _portable_public_source_path(value: str) -> str:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        return value

    resolved = candidate.resolve()
    repo_root = limitation_requirements.ROOT.resolve()
    try:
        rel = resolved.relative_to(repo_root)
        return rel.as_posix() if rel.as_posix() != "." else "."
    except ValueError:
        pass

    text = resolved.as_posix()
    for anchor in ("clt_bundles/", "data_paper_hardened_v2/", "source/"):
        idx = text.find(anchor)
        if idx >= 0:
            return text[idx:]
    return resolved.name or "."


def _sanitize_public_source_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _sanitize_public_source_json(item)
            for key, item in value.items()
            if key not in {"manifest_path", "out_csv", "out_summary"}
        }
    if isinstance(value, list):
        return [_sanitize_public_source_json(item) for item in value]
    if isinstance(value, str):
        return _portable_public_source_path(value)
    return value


def _copy_public_source_json(src: Path, dst: Path) -> Path:
    src_resolved = Path(src).expanduser().resolve()
    dst_resolved = Path(dst).expanduser().resolve()
    if not src_resolved.exists():
        raise FileNotFoundError(f"Source artifact does not exist: {src_resolved}")
    write_json(dst_resolved, _sanitize_public_source_json(load_json(src_resolved)))
    return dst_resolved


def _copy_public_source_csv(src: Path, dst: Path) -> Path:
    src_resolved = Path(src).expanduser().resolve()
    dst_resolved = Path(dst).expanduser().resolve()
    if not src_resolved.exists():
        raise FileNotFoundError(f"Source artifact does not exist: {src_resolved}")
    dst_resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = src_resolved.read_bytes().replace(b"\r\n", b"\n")
    dst_resolved.write_bytes(payload)
    return dst_resolved


def _copy_release_source_artifacts(
    *,
    comparability_summary_path: Path,
    comparability_csv_path: Path,
    topk_summary_path: Path,
    results_root: Path,
) -> tuple[Path, Path, Path]:
    return (
        _copy_public_source_json(
            comparability_summary_path,
            limitation_requirements.limitation_release_source_comparability_summary_path(root=results_root),
        ),
        _copy_public_source_csv(
            comparability_csv_path,
            limitation_requirements.limitation_release_source_comparability_csv_path(root=results_root),
        ),
        _copy_public_source_json(
            topk_summary_path,
            limitation_requirements.limitation_release_source_topk_summary_path(root=results_root),
        ),
    )


def _remove_stale_release_source_artifacts(*, results_root: Path) -> None:
    stale_topk_csv = (
        limitation_requirements.limitation_release_source_topk_dir(root=results_root)
        / f"{limitation_requirements.LIMITATION_SOURCE_TOPK_BASENAME}.csv"
    )
    stale_topk_csv.unlink(missing_ok=True)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _release_artifact_root(*paths: Path) -> Path:
    resolved = [Path(path).expanduser().resolve() for path in paths]
    repo_root = limitation_requirements.ROOT.resolve()
    inside_repo = [_inside(path, repo_root) for path in resolved]
    if all(inside_repo):
        return repo_root
    if any(inside_repo):
        raise ValueError(
            "Limitation release output roots must be either all inside the repo or all outside the repo; "
            f"got {', '.join(str(path) for path in resolved)}"
        )
    parent_roots = {path.parent for path in resolved}
    if len(parent_roots) != 1:
        raise ValueError(
            "Limitation release output roots outside the repo must be sibling directories under one release root; "
            f"got {', '.join(str(path) for path in resolved)}"
        )
    return next(iter(parent_roots))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the public SAE behavioral preservation release surface.")
    parser.add_argument(
        "--source_run_root",
        default="",
        help="Run root containing source/comparability and source/topk outputs.",
    )
    parser.add_argument(
        "--comparability_summary",
        default="",
        help="Optional direct path to a source comparability summary JSON.",
    )
    parser.add_argument(
        "--comparability_csv",
        default="",
        help="Optional direct path to a source comparability row CSV.",
    )
    parser.add_argument(
        "--topk_summary",
        default="",
        help="Optional direct path to a source top-k summary JSON.",
    )
    parser.add_argument("--results_root", default=str(limitation_requirements.LIMITATION_RESULTS_ROOT))
    parser.add_argument("--tables_root", default=str(limitation_requirements.LIMITATION_TABLES_ROOT))
    parser.add_argument("--figures_root", default=str(limitation_requirements.LIMITATION_FIGURES_ROOT))
    args = parser.parse_args(argv)

    profile = limitation_requirements.ensure_profile_configured(require_runs=False)
    identity = configured_identity(profile)
    comparability_source_path, comparability_csv_path, topk_source_path = _resolve_source_paths(
        source_run_root=str(args.source_run_root),
        comparability_summary=str(args.comparability_summary),
        comparability_csv=str(args.comparability_csv),
        topk_summary=str(args.topk_summary),
    )
    results_root = Path(args.results_root).expanduser().resolve()
    tables_root = Path(args.tables_root).expanduser().resolve()
    figures_root = Path(args.figures_root).expanduser().resolve()
    source_artifact_root = _release_artifact_root(results_root, tables_root, figures_root)

    comparability_source = load_json(comparability_source_path)
    comparability_rows = read_csv_rows(comparability_csv_path)
    topk_source = load_json(topk_source_path)
    validate_source_summaries(
        comparability_source,
        topk_source,
        identity=identity,
        layers=profile.public_layers,
        compact_ks=profile.compact_ks,
        expected_disamb_path=profile.dataset_disamb_path,
        expected_clt_repo=limitation_requirements.LIMITATION_LOCAL_CLT_BUNDLE_PATH,
    )
    _remove_stale_release_source_artifacts(results_root=results_root)
    (
        public_comparability_source_path,
        public_comparability_csv_path,
        public_topk_source_path,
    ) = _copy_release_source_artifacts(
        comparability_summary_path=comparability_source_path,
        comparability_csv_path=comparability_csv_path,
        topk_summary_path=topk_source_path,
        results_root=results_root,
    )

    public_comparability = []
    public_topk = []
    for layer in profile.public_layers:
        comp_summary = build_public_comparability_summary(
            comparability_source,
            identity=identity,
            layer=layer,
            source_path=public_comparability_source_path,
            source_artifact_root=source_artifact_root,
        )
        topk_summary = build_public_topk_summary(
            topk_source,
            identity=identity,
            layer=layer,
            compact_ks=profile.compact_ks,
            source_path=public_topk_source_path,
            source_artifact_root=source_artifact_root,
        )
        public_comparability.append(comp_summary)
        public_topk.append(topk_summary)
        write_json(limitation_requirements.limitation_comparability_summary_path(layer, root=results_root), comp_summary)
        write_json(limitation_requirements.limitation_topk_summary_path(layer, root=results_root), topk_summary)

    write_centerpiece_table(public_comparability, limitation_requirements.limitation_centerpiece_table_path(root=tables_root))
    write_topk_table(public_topk, limitation_requirements.limitation_topk_table_path(root=tables_root))
    robustness_input = build_robustness_input_case_target(comparability_rows, layers=profile.public_layers)
    write_csv_rows(
        robustness_input,
        limitation_requirements.limitation_robustness_input_table_path(root=tables_root),
    )
    write_csv_rows(
        build_limitation_robustness_from_case_target(robustness_input, layers=profile.public_layers),
        limitation_requirements.limitation_robustness_summary_table_path(root=tables_root),
    )
    gate_summary, gate_rows = build_gate_diagnostics(comparability_rows, layers=profile.paper_layers)
    write_csv_rows_with_fields(
        gate_summary,
        limitation_requirements.limitation_gate_diagnostics_summary_table_path(root=tables_root),
        fieldnames=GATE_DIAGNOSTICS_SUMMARY_FIELDS,
    )
    write_csv_rows_with_fields(
        gate_rows,
        limitation_requirements.limitation_gate_diagnostics_rows_table_path(root=tables_root),
        fieldnames=GATE_DIAGNOSTICS_ROW_FIELDS,
    )
    write_csv_rows_with_fields(
        build_strict_gate_sensitivity(
            comparability_rows,
            layers=profile.public_layers,
            bootstrap_n=profile.bootstrap_n,
            ci=profile.ci,
            seed=profile.bootstrap_seed,
        ),
        limitation_requirements.limitation_strict_gate_sensitivity_table_path(root=tables_root),
        fieldnames=STRICT_GATE_SENSITIVITY_FIELDS,
    )
    write_json(
        limitation_requirements.limitation_release_manifest_path(root=tables_root),
        build_limitation_release_manifest(
            comparability_summary=comparability_source,
            topk_summary=topk_source,
            identity=identity,
            profile=profile,
            results_root=results_root,
            comparability_summary_path=public_comparability_source_path,
            comparability_csv_path=public_comparability_csv_path,
            topk_summary_path=public_topk_source_path,
            source_artifact_root=source_artifact_root,
        ),
    )
    render_centerpiece_figure(public_comparability, limitation_requirements.limitation_centerpiece_figure_path(root=figures_root))
    render_topk_figure(public_topk, limitation_requirements.limitation_topk_figure_path(root=figures_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
