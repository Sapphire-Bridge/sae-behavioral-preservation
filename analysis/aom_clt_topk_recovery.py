from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import torch

from aom.data.loaders import load_disamb_pairs
from aom.interventions.clt_adapter import CLTInputTransform, CLTPatchConfig
from aom.interventions.clt_loader import load_clt
from aom.metrics.clt_cpt import TopKRecoverySpec, run_clt_topk_feature_recovery
from aom.models.loader import load_causal_lm
from aom.provenance.protocol import (
    ProtocolArgBinding,
    coerce_bool as _coerce_bool,
    enforce_protocol_bindings,
    resolve_protocol_provenance,
)
from aom.repro import collect_versions, get_git_commit_hash
from aom.run_manifest import build_run_manifest, write_run_manifest
from aom.utils import configure_logprob_computation, get_best_device, set_seed


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({str(k) for r in rows for k in r.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _parse_int_csv(arg: str) -> List[int]:
    out: List[int] = []
    for part in str(arg).split(","):
        s = str(part).strip()
        if not s:
            continue
        out.append(int(s))
    if not out:
        raise ValueError("Expected at least one integer")
    return out


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Run CLT top-k feature recovery on DISAMB.")
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--tokenizer_revision", type=str, default=None)
    p.add_argument("--clt_repo", type=str, required=True, help="HF repo id or local path to CLT bundle.")
    p.add_argument("--clt_width", type=str, default="16k")
    p.add_argument("--clt_run_name", type=str, default=None)
    p.add_argument("--clt_l0_target", type=int, default=None)
    p.add_argument("--layers", type=str, default="4,8,12")
    p.add_argument("--ks", type=str, default="1,5,10,20,50,100,200,500,1000,2000,4000,8000,16384")
    p.add_argument("--logz_ks", type=str, default="20,50,200,16384")

    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--torch_dtype", type=str, default=None)
    p.add_argument("--attn_implementation", type=str, default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true")

    p.add_argument("--clt_scale", type=float, default=1.0)
    p.add_argument("--clt_dtype", type=str, default="float32")
    p.add_argument("--clt_decode_strategy", type=str, default="delta_1decode", choices=["safe_2decode", "delta_1decode"])
    p.add_argument("--clt_dtype_policy", type=str, default="clt", choices=["clt", "model"])
    p.add_argument("--clt_eps_active", type=float, default=1e-6)

    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--frac_selection", type=float, default=0.5)
    p.add_argument("--random_k_seeds", type=str, default="0,1,2,3,4")
    p.add_argument(
        "--random_control_mode",
        type=str,
        default="complement",
        choices=["complement", "matched_bin"],
        help="Random-k control sampler.",
    )
    p.add_argument("--matched_bin_n_bins", type=int, default=10, help="Bin count per stat for matched-bin controls.")
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--bootstrap_B", type=int, default=1000)
    p.add_argument("--ci", type=float, default=0.95)
    p.add_argument("--with_logz", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--protocol_path",
        type=str,
        default="",
        help="Optional frozen protocol file path; hash is recorded in summary + run manifest.",
    )
    p.add_argument(
        "--protocol_sha256",
        type=str,
        default="",
        help="Optional explicit protocol SHA256 (64 hex). If protocol_path is set and this is empty, hash is computed.",
    )
    p.add_argument(
        "--require_frozen_protocol",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require protocol.{name,version,prereg_tag,status=frozen}.",
    )
    p.add_argument(
        "--require_git",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require a non-empty git commit hash for provenance.",
    )

    p.add_argument("--disamb_path", type=str, default=str(root / "data" / "disamb_pairs.jsonl"))
    p.add_argument("--no_length_norm", action="store_true")
    p.add_argument("--patch_allow_token_id_mismatch", action="store_true")

    p.add_argument("--logprobs_dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16", "float64"])
    p.add_argument(
        "--strict_finite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail fast on NaN/Inf during logprob scoring.",
    )

    p.add_argument("--out_csv", type=str, default=str(root / "results" / "topk_recovery_l4_l8_l12.csv"))
    p.add_argument(
        "--out_summary",
        type=str,
        default=str(root / "results" / "topk_recovery_l4_l8_l12.summary.json"),
    )
    p.add_argument("--manifest_path", type=str, default="", help="Optional run manifest output path.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    started_at_utc = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    args = parse_args()
    set_seed(int(args.seed))
    configure_logprob_computation(
        logprobs_dtype=getattr(torch, str(args.logprobs_dtype)),
        strict_finite=bool(args.strict_finite),
    )

    protocol_prov = resolve_protocol_provenance(
        protocol_path_raw=str(getattr(args, "protocol_path", "") or ""),
        protocol_sha256_raw=str(getattr(args, "protocol_sha256", "") or ""),
        require_path_for_sha=False,
        require_frozen=bool(getattr(args, "require_frozen_protocol", False)),
    )
    setattr(args, "protocol_path", str(protocol_prov.protocol_path))
    setattr(args, "protocol_sha256", str(protocol_prov.protocol_sha256))
    setattr(args, "protocol_sha256_verified", bool(protocol_prov.protocol_sha256_verified))
    setattr(args, "protocol_sha256_source", str(protocol_prov.protocol_sha256_source))
    setattr(args, "protocol_name", str(protocol_prov.protocol_name))
    setattr(args, "protocol_version", str(protocol_prov.protocol_version))
    setattr(args, "protocol_prereg_tag", str(protocol_prov.protocol_prereg_tag))

    if str(protocol_prov.protocol_path):
        enforce_protocol_bindings(
            args=args,
            protocol_config=protocol_prov.protocol_config,
            bindings=[
                ProtocolArgBinding("bootstrap_B", ("bootstrap", "n"), int),
                ProtocolArgBinding("ci", ("bootstrap", "ci"), float),
                ProtocolArgBinding("split_seed", ("splits", "random_seed"), int, required=False),
                ProtocolArgBinding("frac_selection", ("splits", "selection_fraction"), float, required=False),
                ProtocolArgBinding("require_git", ("repro", "require_git"), _coerce_bool, required=False),
            ],
            context="aom_clt_topk_recovery",
            protocol_path=str(protocol_prov.protocol_path),
            protocol_sha256=str(protocol_prov.protocol_sha256),
        )

        controls = protocol_prov.protocol_config.get("controls", None)
        if isinstance(controls, dict):
            mrk = controls.get("matched_random_k", None)
            if isinstance(mrk, dict):
                enabled_raw = mrk.get("enabled", False)
                sampling_raw = str(mrk.get("sampling", "") or "").strip().lower()
                enabled = _coerce_bool(enabled_raw)
                expected_mode = "matched_bin" if enabled and sampling_raw == "matched_bin" else "complement"
                actual_mode = str(getattr(args, "random_control_mode", "complement")).strip().lower()
                if actual_mode != expected_mode:
                    raise ValueError(
                        "Protocol enforcement failed (aom_clt_topk_recovery): "
                        f"--random_control_mode={actual_mode!r} != protocol controls.matched_random_k "
                        f"(enabled={enabled!r}, sampling={sampling_raw!r}) -> expected {expected_mode!r}"
                    )

    out_csv = Path(str(args.out_csv))
    out_summary = Path(str(args.out_summary))
    manifest_path = (
        Path(str(args.manifest_path))
        if str(getattr(args, "manifest_path", "")).strip()
        else out_summary.with_suffix(".manifest.json")
    )
    if (out_csv.exists() or out_summary.exists()) and not bool(args.overwrite):
        raise FileExistsError(
            f"Output exists. Use --overwrite to replace files: csv={str(out_csv)}, summary={str(out_summary)}"
        )
    if manifest_path.exists() and not bool(args.overwrite):
        raise FileExistsError(f"Output exists. Use --overwrite to replace file: manifest={str(manifest_path)}")

    if args.device == "auto":
        device = get_best_device()
    else:
        device = torch.device({"cpu": "cpu", "cuda": "cuda", "mps": "mps"}[str(args.device)])

    items = load_disamb_pairs(str(args.disamb_path))
    if not items:
        raise ValueError("DISAMB dataset is empty")

    loaded = load_causal_lm(
        str(args.model_name_or_path),
        device=device,
        torch_dtype=str(args.torch_dtype) if args.torch_dtype else None,
        revision=str(args.revision) if args.revision else None,
        tokenizer_revision=str(args.tokenizer_revision) if args.tokenizer_revision else None,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=bool(args.trust_remote_code),
        attn_implementation=str(args.attn_implementation),
        device_map=None,
    )

    layers = _parse_int_csv(str(args.layers))
    ks = _parse_int_csv(str(args.ks))
    random_k_seeds = _parse_int_csv(str(args.random_k_seeds))
    logz_ks = _parse_int_csv(str(args.logz_ks))
    if int(args.matched_bin_n_bins) < 1:
        raise ValueError("--matched_bin_n_bins must be >= 1")

    clt_by_layer = {}
    transform_by_layer = {}
    for l in layers:
        clt, _meta = load_clt(
            str(args.clt_repo),
            layer=int(l),
            width=str(args.clt_width),
            run_name=args.clt_run_name,
            l0_target=args.clt_l0_target,
            device=str(device),
            dtype=str(args.clt_dtype),
            local_files_only=bool(args.local_files_only),
        )
        clt_by_layer[int(l)] = clt
        transform_by_layer[int(l)] = CLTInputTransform(scale=float(args.clt_scale))

    spec = TopKRecoverySpec(
        layers=tuple(int(x) for x in layers),
        ks=tuple(int(x) for x in ks),
        split_seed=int(args.split_seed),
        frac_selection=float(args.frac_selection),
        random_k_seeds=tuple(int(x) for x in random_k_seeds),
        eps=float(args.eps),
        bootstrap_B=int(args.bootstrap_B),
        ci=float(args.ci),
        token_reduce="mean",
        include_logz=bool(args.with_logz),
        logz_ks=tuple(int(x) for x in logz_ks),
        random_control_mode=str(args.random_control_mode),
        matched_bin_n_bins=int(args.matched_bin_n_bins),
    )

    cfg = CLTPatchConfig(
        decode_strategy=str(args.clt_decode_strategy),
        dtype_policy=str(args.clt_dtype_policy),
        eps_active=float(args.clt_eps_active),
    )

    rows, summary = run_clt_topk_feature_recovery(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        items=items,
        device=device,
        clt_by_layer=clt_by_layer,
        transform_by_layer=transform_by_layer,
        spec=spec,
        config=cfg,
        normalize_by_length=not bool(args.no_length_norm),
        require_token_id_match=not bool(args.patch_allow_token_id_mismatch),
    )

    _write_csv(rows, out_csv)
    csv_sha256 = _sha256_file(out_csv) if out_csv.exists() else ""
    git_commit = get_git_commit_hash(repo_root=Path(__file__).resolve().parent, required=bool(args.require_git))
    ended_at_utc = datetime.now(timezone.utc).isoformat()
    wall_time_sec = float(time.perf_counter() - t0)
    summary["run"] = {
        "model_name_or_path": str(args.model_name_or_path),
        "model_revision": str(args.revision or ""),
        "tokenizer_revision": str(args.tokenizer_revision or args.revision or ""),
        "device": str(device),
        "seed": int(args.seed),
        "git_commit": str(git_commit),
        "started_at_utc": str(started_at_utc),
        "ended_at_utc": str(ended_at_utc),
        "wall_time_sec": float(wall_time_sec),
        "protocol_path": str(getattr(args, "protocol_path", "") or ""),
        "protocol_sha256": str(getattr(args, "protocol_sha256", "") or ""),
        "protocol_sha256_verified": bool(getattr(args, "protocol_sha256_verified", False)),
        "protocol_sha256_source": str(getattr(args, "protocol_sha256_source", "") or ""),
        "protocol_name": str(getattr(args, "protocol_name", "") or ""),
        "protocol_version": str(getattr(args, "protocol_version", "") or ""),
        "protocol_prereg_tag": str(getattr(args, "protocol_prereg_tag", "") or ""),
        "out_csv": str(out_csv),
        "out_summary": str(out_summary),
        "manifest_path": str(manifest_path),
    }
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result_row = {
        "model_name_or_path": str(args.model_name_or_path),
        "model_revision": str(args.revision or ""),
        "tokenizer_revision": str(args.tokenizer_revision or args.revision or ""),
        "device": str(device),
        "seed": int(args.seed),
        "layers": ",".join(str(x) for x in layers),
        "ks": ",".join(str(x) for x in ks),
        "random_control_mode": str(args.random_control_mode),
        "matched_bin_n_bins": int(args.matched_bin_n_bins),
        "n_rows": int(len(rows)),
        "n_items": int(len(items)),
        "csv_path": str(out_csv),
        "summary_path": str(out_summary),
        "csv_sha256": str(csv_sha256),
        "git_commit": str(git_commit),
        "protocol_sha256": str(getattr(args, "protocol_sha256", "") or ""),
    }
    manifest = build_run_manifest(
        argv=sys.argv,
        results_row=result_row,
        dataset_manifest_path=str(args.disamb_path),
        csv_path=str(out_csv),
        csv_sha256=str(csv_sha256),
        csv_n_rows=int(len(rows)),
    )
    manifest["run_status"] = "PASS"
    manifest["run_status_reasons"] = []
    manifest["run_summary"] = {
        "attempted": int(len(items) * 2),
        "succeeded": int(len(items) * 2),
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
    manifest["versions"] = collect_versions()
    manifest["started_at_utc"] = str(started_at_utc)
    manifest["ended_at_utc"] = str(ended_at_utc)
    manifest["wall_time_sec"] = float(wall_time_sec)
    protocol_path = str(getattr(args, "protocol_path", "") or "").strip()
    protocol_sha = str(getattr(args, "protocol_sha256", "") or "").strip()
    if protocol_path:
        manifest["protocol_path"] = str(protocol_path)
    if protocol_sha:
        manifest["protocol_sha256"] = str(protocol_sha)
    manifest["protocol_sha256_verified"] = bool(getattr(args, "protocol_sha256_verified", False))
    protocol_sha_source = str(getattr(args, "protocol_sha256_source", "") or "").strip()
    if protocol_sha_source:
        manifest["protocol_sha256_source"] = str(protocol_sha_source)
    protocol_name = str(getattr(args, "protocol_name", "") or "").strip()
    if protocol_name:
        manifest["protocol_name"] = str(protocol_name)
    protocol_version = str(getattr(args, "protocol_version", "") or "").strip()
    if protocol_version:
        manifest["protocol_version"] = str(protocol_version)
    protocol_prereg_tag = str(getattr(args, "protocol_prereg_tag", "") or "").strip()
    if protocol_prereg_tag:
        manifest["protocol_prereg_tag"] = str(protocol_prereg_tag)
    write_run_manifest(manifest_path, manifest)

    print(f"Wrote telemetry CSV: {str(out_csv)}", flush=True)
    print(f"Wrote summary JSON: {str(out_summary)}", flush=True)
    print(f"Wrote run manifest: {str(manifest_path)}", flush=True)


if __name__ == "__main__":
    main()
