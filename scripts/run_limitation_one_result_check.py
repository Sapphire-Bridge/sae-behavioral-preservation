#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.limitation_requirements as limitation_requirements
from scripts.limitation_requirements import (
    LIMITATION_LOCAL_CLT_BUNDLE_PATH,
    limitation_comparability_summary_path,
    load_public_comparability_reference,
)
from scripts.limitation_surface import (
    build_public_comparability_summary,
    configured_identity,
    load_json,
    validate_comparability_source_summary,
    write_json,
)
from scripts.reproduction_common import (
    CommandRecord,
    CommandSpec,
    _display_arg,
    _display_command,
    _is_failure_status,
    _utc_now_iso,
    _walk_files,
)
from scripts.verify_limitation_one_result_check import verify_run


def _repository_commit_label() -> str:
    git_dir = _git_dir(ROOT)
    if git_dir is not None:
        commit = _read_git_head(git_dir)
        if commit:
            return commit
    source_head_path = ROOT / "SOURCE_HEAD.txt"
    if source_head_path.exists():
        text = source_head_path.read_text(encoding="utf-8").strip()
        if text:
            return f"{text} (SOURCE_HEAD.txt)"
    return "snapshot-without-git"


def _git_dir(root: Path) -> Path | None:
    marker = root / ".git"
    if marker.is_dir():
        return marker
    if marker.is_file():
        text = marker.read_text(encoding="utf-8").strip()
        prefix = "gitdir:"
        if text.startswith(prefix):
            path = Path(text.removeprefix(prefix).strip())
            if not path.is_absolute():
                path = (root / path).resolve()
            return path
    return None


def _read_git_head(git_dir: Path) -> str | None:
    head_path = git_dir / "HEAD"
    if not head_path.exists():
        return None
    head = head_path.read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head or None
    ref = head.removeprefix("ref: ").strip()
    ref_path = git_dir / ref
    if ref_path.exists():
        commit = ref_path.read_text(encoding="utf-8").strip()
        return commit or None
    packed_refs = git_dir / "packed-refs"
    if packed_refs.exists():
        for line in packed_refs.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#") or line.startswith("^"):
                continue
            commit, _, packed_ref = line.partition(" ")
            if packed_ref == ref and commit:
                return commit
    return None


def _run_one(spec: CommandSpec, *, cwd: Path, run_root: Path, dry_run: bool) -> CommandRecord:
    before = _walk_files(run_root)
    started = _utc_now_iso()
    exit_code = 0
    if dry_run:
        print(_display_command(spec.argv, run_root))
    else:
        result = subprocess.run(spec.argv, cwd=str(cwd), check=False)
        exit_code = int(result.returncode)
    ended = _utc_now_iso()
    after = _walk_files(run_root)
    generated = tuple(sorted(after - before))
    return CommandRecord(
        name=spec.name,
        argv=tuple(_display_arg(arg, run_root) for arg in spec.argv),
        display_command=_display_command(spec.argv, run_root),
        started_at_utc=started,
        ended_at_utc=ended,
        exit_code=exit_code,
        generated_files=generated,
        artifact_kind=spec.artifact_kind,
    )


def _resolve_quickcheck_torch_dtype(*, device: str) -> str | None:
    profile = limitation_requirements.ensure_profile_configured(require_runs=True)
    configured = str(profile.comparability_torch_dtype or "").strip()
    if configured:
        return configured
    if str(device).strip().lower() == "mps":
        return "float32"
    return None


def _one_result_command_spec(
    run_root: Path,
    *,
    local_files_only: bool,
    device: str = "cpu",
) -> CommandSpec:
    profile = limitation_requirements.ensure_profile_configured(require_runs=True)
    source_root = run_root / "comparability" / "l4"
    argv = [
        sys.executable,
        "scripts/clt_raw_comparability.py",
        "--model_name_or_path",
        str(profile.model_id),
        "--revision",
        str(profile.model_revision),
        "--disamb_path",
        str(profile.dataset_disamb_path),
        "--clt_repo",
        str(LIMITATION_LOCAL_CLT_BUNDLE_PATH),
        "--layers",
        "4",
        "--device",
        str(device),
        "--attn_implementation",
        str(profile.comparability_attn_implementation),
        "--ci",
        str(profile.ci),
        "--seed",
        str(profile.seed),
        "--bootstrap_n",
        str(profile.bootstrap_n),
        "--bootstrap_seed",
        str(profile.bootstrap_seed),
        "--primary_logodds_residual_tol",
        str(profile.comparability_primary_logodds_residual_tol),
        "--run_pca_baseline",
        "--out_csv",
        str(source_root / "comparability.source.csv"),
        "--out_json",
        str(source_root / "comparability.source.summary.json"),
    ]
    if not bool(profile.comparability_hard_fail_invariant):
        argv.append("--no-hard_fail_invariant")
    if not bool(profile.comparability_hard_fail_primary_logodds):
        argv.append("--no-hard_fail_primary_logodds")
    torch_dtype = _resolve_quickcheck_torch_dtype(device=str(device))
    if torch_dtype:
        argv.extend(["--torch_dtype", str(torch_dtype)])
    if local_files_only:
        argv.append("--local_files_only")
    argv.extend(profile.comparability_extra_args)
    return CommandSpec(name="Layer-4 limitation quickcheck", argv=tuple(argv))


def _render_report(
    *,
    run_root: Path,
    created_run_root: bool,
    records: list[CommandRecord],
    missing: list[Path],
    checks: list,
    dry_run: bool,
) -> str:
    lines = [
        "# Limitation One Result Check Report",
        "",
        f"- run root: `$RUN_ROOT` (`{run_root.name}`)",
        f"- run root created by script: `{'yes' if created_run_root else 'no'}`",
        f"- repository commit: `{_repository_commit_label()}`",
        f"- mode: `{'dry_run' if dry_run else 'execute'}`",
        "",
        "## Command Log",
        "",
    ]
    for index, record in enumerate(records, start=1):
        status = "PLAN_ONLY" if dry_run else ("PASS" if record.exit_code == 0 else "FAIL")
        lines.append(f"### {index}. {record.name}")
        lines.append("")
        lines.append(f"- status: `{status}`")
        lines.append(f"- started_at_utc: `{record.started_at_utc}`")
        lines.append(f"- ended_at_utc: `{record.ended_at_utc}`")
        lines.append(f"- exit_code: `{record.exit_code}`")
        lines.append("- command:")
        lines.append("")
        lines.append("```bash")
        lines.append(record.display_command)
        lines.append("```")
        lines.append("")
        if record.generated_files:
            lines.append("- generated files:")
            for rel_path in record.generated_files:
                lines.append(f"  - `$RUN_ROOT/{rel_path}`")
        else:
            lines.append("- generated files: `none`")
        lines.append("")

    if dry_run:
        lines.extend(["## Overall", "", "- overall_status: `PLAN_ONLY`"])
        return "\n".join(lines) + "\n"

    if missing:
        lines.extend(["## Missing Outputs", ""])
        for path in missing:
            lines.append(f"- `$RUN_ROOT/{path.relative_to(run_root).as_posix()}`")
        lines.extend(["", "## Overall", "", "- overall_status: `FAIL`"])
        return "\n".join(lines) + "\n"

    lines.extend(["## Numeric Checks", ""])
    for check in checks:
        note = f" ({check.note})" if check.note else ""
        lines.append(f"- [{check.status.upper()}] {check.name}: observed=`{check.observed}` expected=`{check.expected}`{note}")

    overall = "PASS"
    if any(record.exit_code != 0 for record in records) or any(_is_failure_status(check.status) for check in checks):
        overall = "FAIL"
    lines.extend(["", "## Overall", "", f"- overall_status: `{overall}`"])
    return "\n".join(lines) + "\n"


def _write_json_log(path: Path, *, records: list[CommandRecord], missing: list[Path], checks: list) -> None:
    payload = {
        "records": [asdict(record) for record in records],
        "missing": [str(item) for item in missing],
        "checks": [asdict(check) for check in checks],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_public_summary(run_root: Path) -> None:
    source_summary_path = run_root / "comparability" / "l4" / "comparability.source.summary.json"
    source_summary = load_json(source_summary_path)
    identity = configured_identity()
    validate_comparability_source_summary(
        source_summary,
        identity=identity,
        layers=(4,),
        expected_disamb_path=limitation_requirements.LIMITATION_PROFILE.dataset_disamb_path,
        expected_clt_repo=LIMITATION_LOCAL_CLT_BUNDLE_PATH,
    )
    public_summary = build_public_comparability_summary(
        source_summary,
        identity=identity,
        layer=4,
        source_path=source_summary_path,
        source_artifact_root=run_root,
    )
    write_json(limitation_comparability_summary_path(4, root=run_root), public_summary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute the public CPU-only layer-4 limitation quickcheck.")
    parser.add_argument(
        "--run_root",
        default="",
        help="Optional run directory. Defaults to a fresh temp directory outside the repo.",
    )
    parser.add_argument(
        "--report_path",
        default="",
        help="Markdown report path. Defaults to `$RUN_ROOT/one_result_check_report.md`.",
    )
    parser.add_argument(
        "--json_log_path",
        default="",
        help="Machine-readable command/check log path. Defaults to `$RUN_ROOT/one_result_check_log.json`.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Use only local HF cache for the quickcheck run.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print and log the exact command without executing it.",
    )
    args = parser.parse_args(argv)

    try:
        limitation_requirements.ensure_profile_configured(require_runs=True)
        load_public_comparability_reference(4)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    if not LIMITATION_LOCAL_CLT_BUNDLE_PATH.exists():
        print(
            f"[error] Missing local limitation bundle: {LIMITATION_LOCAL_CLT_BUNDLE_PATH}",
            file=sys.stderr,
        )
        return 2

    created_run_root = False
    if args.run_root:
        run_root = Path(args.run_root).expanduser().resolve()
        run_root.mkdir(parents=True, exist_ok=True)
    else:
        run_root = Path(tempfile.mkdtemp(prefix="limitation_one_result_"))
        created_run_root = True

    if any(run_root.iterdir()):
        print(f"[error] Run root must start empty: {run_root}", file=sys.stderr)
        return 2

    report_path = Path(args.report_path).expanduser().resolve() if args.report_path else run_root / "one_result_check_report.md"
    json_log_path = Path(args.json_log_path).expanduser().resolve() if args.json_log_path else run_root / "one_result_check_log.json"

    records: list[CommandRecord] = []
    record = _run_one(
        _one_result_command_spec(run_root, local_files_only=bool(args.local_files_only)),
        cwd=ROOT,
        run_root=run_root,
        dry_run=bool(args.dry_run),
    )
    records.append(record)

    missing: list[Path] = []
    checks = []
    if not args.dry_run and record.exit_code == 0:
        _write_public_summary(run_root)
        _write_json_log(json_log_path, records=records, missing=[], checks=[])
        missing_artifacts, checks = verify_run(run_root)
        missing = [artifact.path for artifact in missing_artifacts]

    report = _render_report(
        run_root=run_root,
        created_run_root=created_run_root,
        records=records,
        missing=missing,
        checks=checks,
        dry_run=bool(args.dry_run),
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    _write_json_log(json_log_path, records=records, missing=missing, checks=checks)

    stream = sys.stdout if (args.dry_run or (record.exit_code == 0 and not missing and not any(_is_failure_status(check.status) for check in checks))) else sys.stderr
    print(report, file=stream)

    if args.dry_run:
        return 0
    if record.exit_code != 0 or missing or any(_is_failure_status(check.status) for check in checks):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
