#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import threading
from dataclasses import asdict
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aom.utils import get_best_device
import scripts.limitation_requirements as limitation_requirements
from scripts import build_limitation_release_surface
from scripts.limitation_surface import (
    build_limitation_derived_outputs,
    load_json,
    read_csv_rows,
    write_csv_rows,
    write_json,
)
from scripts.reproduction_common import (
    CommandRecord,
    CommandSpec,
    _display_arg,
    _display_command,
    _utc_now_iso,
    _walk_files,
)


def _csv(raw_values: tuple[int, ...] | tuple[str, ...]) -> str:
    return ",".join(str(value) for value in raw_values)


def _resolve_torch_dtype(configured: str | None, *, device: str) -> str | None:
    if str(configured or "").strip():
        return str(configured)
    if str(device) == "mps":
        # Gemma-3 limitation runs hit non-finite PCA inputs on the implicit MPS fp16 path.
        return "float32"
    return None


def _log_stem(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9]+", "_", str(name).strip().lower()).strip("_")
    return stem or "command"


def _relative_log_path(path: Path, run_root: Path) -> str:
    try:
        return f"$RUN_ROOT/{path.relative_to(run_root).as_posix()}"
    except ValueError:
        return str(path)


def _tail_text_lines(path: Path, *, n: int = 40) -> tuple[str, ...]:
    if not path.exists():
        return ()
    text = path.read_text(encoding="utf-8", errors="replace")
    return tuple(text.splitlines()[-int(n) :])


def _tee_pipe(pipe, log_file, console_buffer) -> None:
    try:
        for chunk in iter(lambda: pipe.readline(), b""):
            log_file.write(chunk)
            log_file.flush()
            if console_buffer is not None:
                console_buffer.write(chunk)
                console_buffer.flush()
    finally:
        pipe.close()


def _run_subprocess_with_logs(spec: CommandSpec, *, cwd: Path, run_root: Path) -> tuple[int, Path, Path]:
    logs_root = run_root / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    stem = _log_stem(spec.name)
    stdout_path = logs_root / f"{stem}.stdout.log"
    stderr_path = logs_root / f"{stem}.stderr.log"
    stdout_buffer = getattr(sys.stdout, "buffer", None)
    stderr_buffer = getattr(sys.stderr, "buffer", None)
    with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
        proc = subprocess.Popen(
            spec.argv,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert proc.stdout is not None
        assert proc.stderr is not None
        stdout_thread = threading.Thread(
            target=_tee_pipe,
            args=(proc.stdout, stdout_file, stdout_buffer),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_tee_pipe,
            args=(proc.stderr, stderr_file, stderr_buffer),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        exit_code = int(proc.wait())
        stdout_thread.join()
        stderr_thread.join()
    return exit_code, stdout_path, stderr_path


def _run_one(spec: CommandSpec, *, cwd: Path, run_root: Path, dry_run: bool) -> CommandRecord:
    before = _walk_files(run_root)
    started = _utc_now_iso()
    exit_code = 0
    stdout_path: Path | None = None
    stderr_path: Path | None = None
    if dry_run:
        print(_display_command(spec.argv, run_root))
    else:
        exit_code, stdout_path, stderr_path = _run_subprocess_with_logs(spec, cwd=cwd, run_root=run_root)
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
        stdout_log=_relative_log_path(stdout_path, run_root) if stdout_path is not None else "",
        stderr_log=_relative_log_path(stderr_path, run_root) if stderr_path is not None else "",
        stderr_tail=_tail_text_lines(stderr_path) if stderr_path is not None and int(exit_code) != 0 else (),
    )


def _run_internal_step(
    *,
    name: str,
    display_command: str,
    run_root: Path,
    dry_run: bool,
    fn,
) -> CommandRecord:
    before = _walk_files(run_root)
    started = _utc_now_iso()
    exit_code = 0
    if dry_run:
        print(display_command)
    else:
        try:
            fn()
        except Exception as exc:
            print(f"[error] {name}: {exc}", file=sys.stderr)
            exit_code = 1
    ended = _utc_now_iso()
    after = _walk_files(run_root)
    generated = tuple(sorted(after - before))
    return CommandRecord(
        name=name,
        argv=(),
        display_command=display_command,
        started_at_utc=started,
        ended_at_utc=ended,
        exit_code=exit_code,
        generated_files=generated,
        artifact_kind="core",
    )


def _effective_device(device_raw: str) -> str:
    if str(device_raw) == "auto":
        return str(get_best_device().type)
    return str(device_raw)


def _validate_device(*, device_raw: str, require_accelerator: bool, dry_run: bool) -> str:
    effective = _effective_device(device_raw)
    if require_accelerator and effective == "cpu":
        raise ValueError("This reproduction path requires an accelerator, but --device resolved to CPU.")
    if dry_run:
        return effective
    if effective == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested, but torch.cuda.is_available() is false.")
    if effective == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise ValueError("MPS requested, but torch.backends.mps.is_available() is false.")
    return effective


def _source_outputs_exist(run_root: Path) -> bool:
    return (
        limitation_requirements.limitation_source_comparability_summary_path(run_root).exists()
        and limitation_requirements.limitation_source_comparability_csv_path(run_root).exists()
        and limitation_requirements.limitation_source_topk_summary_path(run_root).exists()
        and limitation_requirements.limitation_source_topk_csv_path(run_root).exists()
    )


def _comparability_command_spec(run_root: Path, *, device: str, local_files_only: bool) -> CommandSpec:
    profile = limitation_requirements.ensure_profile_configured(require_runs=True)
    torch_dtype = _resolve_torch_dtype(profile.comparability_torch_dtype, device=device)
    argv = [
        sys.executable,
        "scripts/clt_raw_comparability.py",
        "--model_name_or_path",
        str(profile.model_id),
        "--revision",
        str(profile.model_revision),
        "--tokenizer_revision",
        str(profile.tokenizer_revision),
        "--disamb_path",
        str(profile.dataset_disamb_path),
        "--clt_repo",
        str(limitation_requirements.LIMITATION_LOCAL_CLT_BUNDLE_PATH),
        "--clt_width",
        str(profile.sae_width),
        "--layers",
        _csv(profile.paper_layers),
        "--device",
        str(device),
        "--attn_implementation",
        str(profile.comparability_attn_implementation),
        "--seed",
        str(profile.seed),
        "--bootstrap_n",
        str(profile.bootstrap_n),
        "--bootstrap_seed",
        str(profile.bootstrap_seed),
        "--ci",
        str(profile.ci),
        "--primary_logodds_residual_tol",
        str(profile.comparability_primary_logodds_residual_tol),
        "--run_pca_baseline",
        "--run_random_orth_baseline",
        "--run_faithfulness_decomposition_arms",
        "--out_csv",
        str(limitation_requirements.limitation_source_comparability_csv_path(run_root)),
        "--out_json",
        str(limitation_requirements.limitation_source_comparability_summary_path(run_root)),
    ]
    if not bool(profile.comparability_hard_fail_invariant):
        argv.append("--no-hard_fail_invariant")
    if not bool(profile.comparability_hard_fail_primary_logodds):
        argv.append("--no-hard_fail_primary_logodds")
    if torch_dtype:
        argv.extend(["--torch_dtype", str(torch_dtype)])
    if local_files_only:
        argv.append("--local_files_only")
    argv.extend(profile.comparability_extra_args)
    return CommandSpec(name="Gemma-3 comparability source run", argv=tuple(argv))


def _topk_command_spec(run_root: Path, *, device: str, local_files_only: bool) -> CommandSpec:
    profile = limitation_requirements.ensure_profile_configured(require_runs=True)
    torch_dtype = _resolve_torch_dtype(profile.topk_torch_dtype, device=device)
    argv = [
        sys.executable,
        "analysis/aom_clt_topk_recovery.py",
        "--model_name_or_path",
        str(profile.model_id),
        "--revision",
        str(profile.model_revision),
        "--tokenizer_revision",
        str(profile.tokenizer_revision),
        "--clt_repo",
        str(limitation_requirements.LIMITATION_LOCAL_CLT_BUNDLE_PATH),
        "--clt_width",
        str(profile.sae_width),
        "--layers",
        _csv(profile.topk_layers),
        "--ks",
        _csv(profile.topk_ks),
        "--logz_ks",
        _csv(profile.topk_logz_ks),
        "--device",
        str(device),
        "--attn_implementation",
        str(profile.topk_attn_implementation),
        "--disamb_path",
        str(profile.dataset_disamb_path),
        "--seed",
        str(profile.topk_seed),
        "--split_seed",
        str(profile.topk_split_seed),
        "--frac_selection",
        str(profile.topk_frac_selection),
        "--bootstrap_B",
        str(profile.topk_bootstrap_B),
        "--ci",
        str(profile.topk_ci),
        "--out_csv",
        str(limitation_requirements.limitation_source_topk_csv_path(run_root)),
        "--out_summary",
        str(limitation_requirements.limitation_source_topk_summary_path(run_root)),
        "--overwrite",
    ]
    if torch_dtype:
        argv.extend(["--torch_dtype", str(torch_dtype)])
    if profile.topk_random_control_mode:
        argv.extend(["--random_control_mode", str(profile.topk_random_control_mode)])
    if profile.topk_matched_bin_n_bins is not None:
        argv.extend(["--matched_bin_n_bins", str(profile.topk_matched_bin_n_bins)])
    if local_files_only:
        argv.append("--local_files_only")
    argv.extend(profile.topk_extra_args)
    return CommandSpec(name="Gemma-3 top-k source run", argv=tuple(argv))


def _derive_outputs(run_root: Path) -> None:
    comparability_summary = load_json(limitation_requirements.limitation_source_comparability_summary_path(run_root))
    comparability_rows = read_csv_rows(limitation_requirements.limitation_source_comparability_csv_path(run_root))
    topk_summary = load_json(limitation_requirements.limitation_source_topk_summary_path(run_root))
    derived = build_limitation_derived_outputs(
        comparability_summary=comparability_summary,
        comparability_rows=comparability_rows,
        topk_summary=topk_summary,
        source_run_root=run_root,
    )
    write_json(limitation_requirements.limitation_derived_numbers_path(run_root), derived.numbers)
    write_csv_rows(derived.stress_rows, limitation_requirements.limitation_stress_arm_summary_path(run_root))


def _release_outputs(run_root: Path) -> None:
    code = build_limitation_release_surface.main(
        [
            "--source_run_root",
            str(run_root),
            "--results_root",
            str(limitation_requirements.limitation_release_results_root(run_root)),
            "--tables_root",
            str(limitation_requirements.limitation_release_tables_root(run_root)),
            "--figures_root",
            str(limitation_requirements.limitation_release_figures_root(run_root)),
        ]
    )
    if int(code) != 0:
        raise RuntimeError(f"build_limitation_release_surface exited with code {code}")


def _expected_outputs_for_mode(mode: str) -> tuple[Path, ...]:
    def rel(path: Path, run_root: Path) -> Path:
        return path.relative_to(run_root)

    run_root = Path("/tmp/placeholder")
    outputs: list[Path] = []
    if mode in {"comparability", "full"}:
        outputs.extend(
            [
                rel(limitation_requirements.limitation_source_comparability_summary_path(run_root), run_root),
                rel(limitation_requirements.limitation_source_comparability_csv_path(run_root), run_root),
            ]
        )
    if mode in {"topk", "full"}:
        outputs.extend(
            [
                rel(limitation_requirements.limitation_source_topk_summary_path(run_root), run_root),
                rel(limitation_requirements.limitation_source_topk_csv_path(run_root), run_root),
            ]
        )
    if mode in {"derive", "full"}:
        outputs.extend(
            [
                rel(limitation_requirements.limitation_derived_numbers_path(run_root), run_root),
                rel(limitation_requirements.limitation_stress_arm_summary_path(run_root), run_root),
            ]
        )
    if mode in {"release", "full"}:
        release_results_root = limitation_requirements.limitation_release_results_root(run_root)
        release_tables_root = limitation_requirements.limitation_release_tables_root(run_root)
        release_figures_root = limitation_requirements.limitation_release_figures_root(run_root)
        for layer in limitation_requirements.LIMITATION_PROFILE.public_layers:
            outputs.append(rel(limitation_requirements.limitation_comparability_summary_path(layer, root=release_results_root), run_root))
            outputs.append(rel(limitation_requirements.limitation_topk_summary_path(layer, root=release_results_root), run_root))
        outputs.extend(
            [
                rel(limitation_requirements.limitation_release_source_comparability_summary_path(root=release_results_root), run_root),
                rel(limitation_requirements.limitation_release_source_comparability_csv_path(root=release_results_root), run_root),
                rel(limitation_requirements.limitation_release_source_topk_summary_path(root=release_results_root), run_root),
                rel(limitation_requirements.limitation_centerpiece_table_path(root=release_tables_root), run_root),
                rel(limitation_requirements.limitation_topk_table_path(root=release_tables_root), run_root),
                rel(limitation_requirements.limitation_robustness_input_table_path(root=release_tables_root), run_root),
                rel(limitation_requirements.limitation_robustness_summary_table_path(root=release_tables_root), run_root),
                rel(limitation_requirements.limitation_gate_diagnostics_summary_table_path(root=release_tables_root), run_root),
                rel(limitation_requirements.limitation_gate_diagnostics_rows_table_path(root=release_tables_root), run_root),
                rel(limitation_requirements.limitation_strict_gate_sensitivity_table_path(root=release_tables_root), run_root),
                rel(limitation_requirements.limitation_release_manifest_path(root=release_tables_root), run_root),
                rel(limitation_requirements.limitation_centerpiece_figure_path(root=release_figures_root), run_root),
                rel(limitation_requirements.limitation_topk_figure_path(root=release_figures_root), run_root),
            ]
        )
    return tuple(outputs)


def _render_report(
    *,
    run_root: Path,
    created_run_root: bool,
    mode: str,
    requested_device: str,
    effective_device: str,
    local_files_only: bool,
    records: list[CommandRecord],
    missing: list[Path],
    dry_run: bool,
) -> str:
    overall = "PLAN_ONLY" if dry_run else "PASS"
    if not dry_run and (missing or any(int(record.exit_code) != 0 for record in records)):
        overall = "FAIL"
    lines = [
        "# Limitation Paper Reproduction Report",
        "",
        f"- run root: `$RUN_ROOT` (`{run_root.name}`)",
        f"- run root created by script: `{'yes' if created_run_root else 'no'}`",
        f"- mode: `{mode}`",
        f"- requested_device: `{requested_device}`",
        f"- effective_device: `{effective_device}`",
        f"- local_files_only: `{'yes' if local_files_only else 'no'}`",
        f"- overall_status: `{overall}`",
        "",
        "## Command Log",
        "",
    ]
    for idx, record in enumerate(records, start=1):
        status = "PLAN_ONLY" if dry_run else ("PASS" if int(record.exit_code) == 0 else "FAIL")
        lines.append(f"### {idx}. {record.name}")
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
        if record.stdout_log:
            lines.append(f"- stdout_log: `{record.stdout_log}`")
        if record.stderr_log:
            lines.append(f"- stderr_log: `{record.stderr_log}`")
        if record.stderr_tail:
            lines.append("- stderr_tail:")
            lines.append("")
            lines.append("```text")
            lines.extend(record.stderr_tail)
            lines.append("```")
            lines.append("")
        if record.generated_files:
            lines.append("- generated files:")
            for rel_path in record.generated_files:
                lines.append(f"  - `$RUN_ROOT/{rel_path}`")
        else:
            lines.append("- generated files: `none`")
        lines.append("")

    if missing:
        lines.extend(["## Missing Outputs", ""])
        for rel_path in missing:
            lines.append(f"- `$RUN_ROOT/{rel_path.as_posix()}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def _write_json_log(
    path: Path,
    *,
    mode: str,
    requested_device: str,
    effective_device: str,
    local_files_only: bool,
    records: list[CommandRecord],
    missing: list[Path],
) -> None:
    payload = {
        "mode": str(mode),
        "requested_device": str(requested_device),
        "effective_device": str(effective_device),
        "local_files_only": bool(local_files_only),
        "profile": limitation_requirements.limitation_profile_summary(),
        "records": [asdict(record) for record in records],
        "missing": [item.as_posix() for item in missing],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the full reproduction flow for the SAE limitation paper.")
    parser.add_argument(
        "--run_root",
        default="",
        help="Run directory for fresh source, derived, and release outputs. Defaults to a temp directory.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--require_accelerator", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--mode", default="full", choices=["full", "comparability", "topk", "derive", "release"])
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv)

    try:
        limitation_requirements.ensure_profile_configured(require_runs=True)
        effective_device = _validate_device(
            device_raw=str(args.device),
            require_accelerator=bool(args.require_accelerator),
            dry_run=bool(args.dry_run),
        )
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    created_run_root = False
    if args.run_root:
        run_root = Path(args.run_root).expanduser().resolve()
        run_root.mkdir(parents=True, exist_ok=True)
    else:
        run_root = Path(tempfile.mkdtemp(prefix="limitation_paper_"))
        created_run_root = True

    report_path = limitation_requirements.limitation_report_path(run_root)
    json_log_path = limitation_requirements.limitation_json_log_path(run_root)
    mode = str(args.mode)

    if not args.dry_run and mode in {"full", "comparability", "topk"}:
        if not limitation_requirements.LIMITATION_LOCAL_CLT_BUNDLE_PATH.exists():
            print(
                f"[error] Missing local limitation bundle: {limitation_requirements.LIMITATION_LOCAL_CLT_BUNDLE_PATH}. "
                "Run scripts/prepare_limitation_bundle.py first.",
                file=sys.stderr,
            )
            return 2

    specs: list[CommandSpec] = []
    if mode in {"full", "comparability"}:
        specs.append(
            _comparability_command_spec(
                run_root,
                device=effective_device,
                local_files_only=bool(args.local_files_only),
            )
        )
    if mode in {"full", "topk"}:
        specs.append(
            _topk_command_spec(
                run_root,
                device=effective_device,
                local_files_only=bool(args.local_files_only),
            )
        )

    records: list[CommandRecord] = []
    for spec in specs:
        record = _run_one(spec, cwd=ROOT, run_root=run_root, dry_run=bool(args.dry_run))
        records.append(record)
        if not args.dry_run and int(record.exit_code) != 0:
            break

    external_failed = any(int(record.exit_code) != 0 for record in records)
    if mode in {"full", "derive"} and (not external_failed):
        records.append(
            _run_internal_step(
                name="Derive paper outputs",
                display_command="internal: derive limitation paper outputs from source summaries",
                run_root=run_root,
                dry_run=bool(args.dry_run),
                fn=lambda: _derive_outputs(run_root),
            )
        )

    if mode in {"full", "release"} and (not any(int(record.exit_code) != 0 for record in records)):
        records.append(
            _run_internal_step(
                name="Build release surface",
                display_command="internal: build limitation release surface from source_run_root",
                run_root=run_root,
                dry_run=bool(args.dry_run),
                fn=lambda: _release_outputs(run_root),
            )
        )

    missing: list[Path] = []
    if not args.dry_run:
        for rel_path in _expected_outputs_for_mode(mode):
            candidate = run_root / rel_path
            if not candidate.exists():
                missing.append(rel_path)

    report = _render_report(
        run_root=run_root,
        created_run_root=created_run_root,
        mode=mode,
        requested_device=str(args.device),
        effective_device=effective_device,
        local_files_only=bool(args.local_files_only),
        records=records,
        missing=missing,
        dry_run=bool(args.dry_run),
    )
    report_path.write_text(report, encoding="utf-8")
    _write_json_log(
        json_log_path,
        mode=mode,
        requested_device=str(args.device),
        effective_device=effective_device,
        local_files_only=bool(args.local_files_only),
        records=records,
        missing=missing,
    )

    if args.dry_run:
        print(report, file=sys.stdout)
        return 0
    if missing or any(int(record.exit_code) != 0 for record in records):
        print(report, file=sys.stderr)
        return 2
    print(report, file=sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
