#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aom.utils import get_best_device
from scripts.run_limitation_one_result_check import (
    LIMITATION_LOCAL_CLT_BUNDLE_PATH,
    _one_result_command_spec,
    _render_report,
    _run_one,
    _write_json_log,
    _write_public_summary,
    verify_run,
)
from scripts.reproduction_common import _is_failure_status


def _device_available(device: str) -> bool:
    kind = str(device).lower()
    if kind == "cpu":
        return True
    if kind == "cuda":
        return bool(torch.cuda.is_available())
    if kind == "mps":
        return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    return False


def _resolve_requested_device(device: str) -> str:
    requested = str(device).strip().lower()
    if requested == "auto":
        return str(get_best_device().type)
    return requested


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Execute the accelerated layer-4 limitation quickcheck and write a report."
    )
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
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Execution device for the accelerated quickcheck. Default resolves automatically.",
    )
    parser.add_argument(
        "--require_accelerator",
        action="store_true",
        help="Fail unless the resolved device is a GPU accelerator (CUDA or MPS).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print and log the exact command without executing it.",
    )
    args = parser.parse_args(argv)

    resolved_device = _resolve_requested_device(str(args.device))
    if not _device_available(resolved_device):
        print(f"[error] Requested device is unavailable: {resolved_device}", file=sys.stderr)
        return 2
    if bool(args.require_accelerator) and resolved_device == "cpu":
        print("[error] No accelerator available; resolved device is cpu.", file=sys.stderr)
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
        run_root = Path(tempfile.mkdtemp(prefix="limitation_one_result_gpu_"))
        created_run_root = True

    if any(run_root.iterdir()):
        print(f"[error] Run root must start empty: {run_root}", file=sys.stderr)
        return 2

    report_path = Path(args.report_path).expanduser().resolve() if args.report_path else run_root / "one_result_check_report.md"
    json_log_path = Path(args.json_log_path).expanduser().resolve() if args.json_log_path else run_root / "one_result_check_log.json"

    records = []
    record = _run_one(
        _one_result_command_spec(
            run_root,
            local_files_only=bool(args.local_files_only),
            device=resolved_device,
        ),
        cwd=ROOT,
        run_root=run_root,
        dry_run=bool(args.dry_run),
    )
    records.append(record)

    missing = []
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

    stream = sys.stdout if (
        args.dry_run or (record.exit_code == 0 and not missing and not any(_is_failure_status(check.status) for check in checks))
    ) else sys.stderr
    print(report, file=stream)

    if args.dry_run:
        return 0
    if record.exit_code != 0 or missing or any(_is_failure_status(check.status) for check in checks):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
