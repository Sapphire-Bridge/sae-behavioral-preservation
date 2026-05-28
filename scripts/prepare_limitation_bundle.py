#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file as load_safetensors_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.limitation_requirements as limitation_requirements


WIDTH_TO_D_LATENT = {
    "16k": 16384,
    "32k": 32768,
    "65k": 65536,
    "131k": 131072,
    "262k": 262144,
    "524k": 524288,
    "1m": 1048576,
}

SOURCE_PARAM_KEY_MAP = (
    ("w_enc", "W_enc"),
    ("w_dec", "W_dec"),
    ("b_enc", "b_enc"),
    ("b_dec", "b_dec"),
)
OPTIONAL_SOURCE_PARAM_KEY_MAP = (("threshold", "threshold"),)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_model_dims(snapshot: Path) -> tuple[int, int]:
    config_path = snapshot / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing model config in snapshot: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        dim_source = text_config
    else:
        dim_source = config
    hidden_size = int(dim_source.get("hidden_size", 0))
    num_hidden_layers = int(dim_source.get("num_hidden_layers", 0))
    if hidden_size < 1 or num_hidden_layers < 1:
        raise ValueError(f"Model config missing hidden_size/num_hidden_layers in {config_path}")
    return hidden_size, num_hidden_layers


def _ensure_snapshot(repo_id: str, revision: str, *, local_files_only: bool) -> Path:
    if local_files_only:
        return limitation_requirements.resolve_cached_snapshot(repo_id, revision)
    from huggingface_hub import snapshot_download

    return Path(str(snapshot_download(repo_id=repo_id, revision=revision)))


def _resolve_expected_latent(width: str) -> int | None:
    return WIDTH_TO_D_LATENT.get(str(width))


def _load_source_config(config_path: Path) -> dict[str, Any]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Invalid Gemma Scope config payload: {config_path}")
    return payload


def _resolve_source_artifacts(
    *,
    repo_id: str,
    revision: str,
    source_entry: str,
    local_files_only: bool,
) -> tuple[Path, Path, Path]:
    source_rel = Path(str(source_entry))
    config_rel = source_rel / "config.json"
    params_rel = source_rel / "params.safetensors"
    if local_files_only:
        snapshot = limitation_requirements.resolve_cached_snapshot(repo_id, revision)
        return source_rel, snapshot / config_rel, snapshot / params_rel
    from huggingface_hub import hf_hub_download

    return (
        source_rel,
        Path(str(hf_hub_download(repo_id=repo_id, filename=str(config_rel), revision=revision))),
        Path(str(hf_hub_download(repo_id=repo_id, filename=str(params_rel), revision=revision))),
    )


def _validate_source_config(
    source_cfg: dict[str, Any],
    *,
    layer: int,
    model_id: str,
    width: str,
    expected_l0: int,
    config_path: Path,
) -> int:
    observed_model_name = str(source_cfg.get("model_name", "")).strip()
    if observed_model_name != str(model_id):
        raise ValueError(
            f"{config_path} model_name mismatch: expected {model_id!r}, got {observed_model_name!r}"
        )

    expected_width = _resolve_expected_latent(width)
    observed_width = source_cfg.get("width")
    if expected_width is not None and int(observed_width) != int(expected_width):
        raise ValueError(
            f"{config_path} width mismatch: expected {expected_width} for width {width!r}, got {observed_width!r}"
        )

    expected_hook = f"model.layers.{int(layer)}.output"
    observed_in = str(source_cfg.get("hf_hook_point_in", "")).strip()
    observed_out = str(source_cfg.get("hf_hook_point_out", "")).strip()
    if observed_in != expected_hook or observed_out != expected_hook:
        raise ValueError(
            f"{config_path} hook mismatch: expected in/out={expected_hook!r}, "
            f"got in={observed_in!r}, out={observed_out!r}"
        )

    observed_l0 = int(source_cfg.get("l0", -1))
    if int(observed_l0) != int(expected_l0):
        raise ValueError(
            f"{config_path} l0 mismatch: expected {expected_l0}, got {observed_l0}"
        )
    return observed_l0


def _load_source_arrays(params_path: Path) -> dict[str, np.ndarray]:
    arrays = load_safetensors_file(str(params_path))
    return {str(key): np.asarray(value) for key, value in arrays.items()}


def _map_source_arrays(
    arrays: dict[str, np.ndarray],
    *,
    source_params_path: Path,
) -> dict[str, np.ndarray]:
    mapped: dict[str, np.ndarray] = {}
    for source_key, target_key in SOURCE_PARAM_KEY_MAP:
        if source_key not in arrays:
            raise ValueError(
                f"{source_params_path} missing {source_key!r}; found keys={sorted(arrays.keys())}"
            )
        mapped[target_key] = np.asarray(arrays[source_key])
    for source_key, target_key in OPTIONAL_SOURCE_PARAM_KEY_MAP:
        if source_key in arrays:
            mapped[target_key] = np.asarray(arrays[source_key])
    return mapped


def _build_cfg_from_arrays(
    arrays: dict[str, np.ndarray],
    *,
    hidden_size: int,
    width: str,
    params_sha256: str,
    source_repo_id: str,
) -> dict[str, Any]:
    for key in ("W_enc", "W_dec", "b_enc", "b_dec"):
        if key not in arrays:
            raise ValueError(f"Mapped arrays missing {key}; found keys={sorted(arrays.keys())}")

    d_latent = int(np.asarray(arrays["b_enc"]).reshape(-1).shape[0])
    expected_latent = _resolve_expected_latent(width)
    if expected_latent is not None and int(d_latent) != int(expected_latent):
        raise ValueError(
            f"Latent width mismatch: expected {expected_latent} from width {width}, got {d_latent}"
        )

    w_enc = np.asarray(arrays["W_enc"])
    w_dec = np.asarray(arrays["W_dec"])
    b_enc = np.asarray(arrays["b_enc"]).reshape(-1)
    b_dec = np.asarray(arrays["b_dec"]).reshape(-1)

    if b_enc.shape != (d_latent,):
        raise ValueError(f"b_enc shape mismatch: got {b_enc.shape}, expected {(d_latent,)}")
    if b_dec.shape != (hidden_size,):
        raise ValueError(
            f"b_dec shape mismatch: got {b_dec.shape}, expected {(hidden_size,)}"
        )
    if w_enc.shape not in {(hidden_size, d_latent), (d_latent, hidden_size)}:
        raise ValueError(
            f"W_enc shape mismatch: got {w_enc.shape}, "
            f"expected {(hidden_size, d_latent)} or transpose"
        )
    if w_dec.shape not in {(d_latent, hidden_size), (hidden_size, d_latent)}:
        raise ValueError(
            f"W_dec shape mismatch: got {w_dec.shape}, "
            f"expected {(d_latent, hidden_size)} or transpose"
        )

    has_threshold = "threshold" in arrays
    if has_threshold:
        threshold = np.asarray(arrays["threshold"]).reshape(-1)
        if threshold.shape != (d_latent,):
            raise ValueError(
                f"threshold shape mismatch: got {threshold.shape}, expected {(d_latent,)}"
            )

    cfg = {
        "d_in": int(hidden_size),
        "d_latent": int(d_latent),
        "d_out": int(hidden_size),
        "d_sae": int(d_latent),
        "activation": "jumprelu" if has_threshold else "relu",
        "pre_encoder_bias": False,
        "encode_site": "resid_post",
        "decode_site": "resid_post",
        "writeback_site": "resid_post",
        "site_mode": "same_site_v1",
        "source": str(source_repo_id),
        "source_format": "jumprelu_sae" if has_threshold else "sae",
        "params_sha256": str(params_sha256),
    }
    return cfg


def _write_cfg(run_dir: Path, cfg: dict[str, Any]) -> None:
    cfg_path = run_dir / "cfg.json"
    cfg_path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_params_npz(dest_path: Path, arrays: dict[str, np.ndarray]) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dest_path, **arrays)


def _load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(str(path), allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _verify_round_trip_exact(
    *,
    source_arrays: dict[str, np.ndarray],
    materialized_params_path: Path,
) -> None:
    materialized = _load_npz_arrays(materialized_params_path)
    if sorted(materialized.keys()) != sorted(source_arrays.keys()):
        raise ValueError(
            f"Round-trip key mismatch for {materialized_params_path}: "
            f"expected {sorted(source_arrays.keys())}, got {sorted(materialized.keys())}"
        )
    for key in sorted(source_arrays.keys()):
        src = np.asarray(source_arrays[key])
        dst = np.asarray(materialized[key])
        if src.shape != dst.shape:
            raise ValueError(
                f"Round-trip shape mismatch for {materialized_params_path} key {key}: "
                f"expected {src.shape}, got {dst.shape}"
            )
        if src.dtype != dst.dtype:
            raise ValueError(
                f"Round-trip dtype mismatch for {materialized_params_path} key {key}: "
                f"expected {src.dtype}, got {dst.dtype}"
            )
        if not np.array_equal(src, dst):
            raise ValueError(
                f"Round-trip value mismatch for {materialized_params_path} key {key}: "
                "materialized NPZ differs from source safetensors arrays"
            )


def _prepare_output_dir(out_dir: Path) -> None:
    if out_dir.exists():
        if not out_dir.is_dir():
            raise ValueError(f"Bundle output path exists and is not a directory: {out_dir}")
        if any(out_dir.iterdir()):
            raise ValueError(f"Bundle output directory must be empty before materialization: {out_dir}")
        out_dir.rmdir()
    out_dir.parent.mkdir(parents=True, exist_ok=True)


def _manifest_path_value(value: object) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return limitation_requirements.ROOT / path


def _existing_bundle_is_complete(out_dir: Path, profile: limitation_requirements.LimitationProfile) -> bool:
    manifest_path = limitation_requirements.limitation_bundle_manifest_path(out_dir)
    if not manifest_path.exists() or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if not isinstance(manifest, dict):
        return False
    try:
        expected_bundle_id = limitation_requirements.compute_limitation_sae_bundle_id(manifest)
    except (TypeError, ValueError):
        return False
    if str(manifest.get("bundle_id", "")) != expected_bundle_id:
        return False
    if str(manifest.get("publication_profile", "")) != limitation_requirements.LIMITATION_PUBLICATION_PROFILE:
        return False

    model = manifest.get("model")
    sae_source = manifest.get("sae_source")
    layers = manifest.get("layers")
    if not isinstance(model, dict) or not isinstance(sae_source, dict) or not isinstance(layers, list):
        return False
    if str(model.get("id", "")) != str(profile.model_id):
        return False
    if str(model.get("revision", "")) != str(profile.model_revision):
        return False
    if str(sae_source.get("repo_id", "")) != str(profile.sae_repo_id):
        return False
    if str(sae_source.get("revision", "")) != str(profile.sae_repo_revision):
        return False
    if str(sae_source.get("width", "")) != str(profile.sae_width):
        return False

    rows_by_layer: dict[int, dict[str, Any]] = {}
    for raw_row in layers:
        if not isinstance(raw_row, dict):
            return False
        try:
            layer = int(raw_row.get("layer", -1))
        except (TypeError, ValueError):
            return False
        rows_by_layer[layer] = raw_row
    expected_layers = sorted(int(layer) for layer in profile.paper_layers)
    if sorted(rows_by_layer) != expected_layers:
        return False

    for layer in expected_layers:
        row = rows_by_layer[layer]
        entry = limitation_requirements.limitation_sae_source_entry(profile, layer)
        if str(row.get("width", "")) != str(profile.sae_width):
            return False
        if str(row.get("source_entry", "")) != str(entry.source_entry):
            return False
        try:
            expected_l0 = int(row.get("expected_l0", -1))
        except (TypeError, ValueError):
            return False
        if expected_l0 != int(entry.expected_l0):
            return False
        for key in ("params_path", "cfg_path"):
            path = _manifest_path_value(row.get(key, ""))
            if not path.exists() or not path.is_file():
                return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialize the local CLT bundle for the SAE limitation paper.")
    parser.add_argument(
        "--out_dir",
        default=str(limitation_requirements.LIMITATION_LOCAL_CLT_BUNDLE_PATH),
        help="Bundle output directory.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Use only local HF cache entries for the frozen model and Gemma Scope runs.",
    )
    args = parser.parse_args(argv)

    profile = limitation_requirements.ensure_bundle_profile_configured()
    out_dir = Path(args.out_dir).expanduser().resolve()

    model_snapshot = _ensure_snapshot(
        str(profile.model_id),
        str(profile.model_revision),
        local_files_only=bool(args.local_files_only),
    )
    hidden_size, num_hidden_layers = _load_model_dims(model_snapshot)

    if _existing_bundle_is_complete(out_dir, profile):
        print(f"Using existing limitation bundle at {out_dir}", flush=True)
        return 0

    resolved_sources: list[dict[str, Any]] = []
    for layer in sorted(int(layer) for layer in profile.paper_layers):
        if layer < 0 or layer >= int(num_hidden_layers):
            raise ValueError(f"Configured layer {layer} is out of range for model with {num_hidden_layers} layers")
        entry = limitation_requirements.limitation_sae_source_entry(profile, layer)
        source_rel, source_config_path, source_params_path = _resolve_source_artifacts(
            repo_id=str(profile.sae_repo_id),
            revision=str(profile.sae_repo_revision),
            source_entry=str(entry.source_entry),
            local_files_only=bool(args.local_files_only),
        )
        if not source_config_path.exists():
            raise FileNotFoundError(
                f"Missing Gemma Scope config for layer {layer}: {source_config_path}"
            )
        if not source_params_path.exists():
            raise FileNotFoundError(
                f"Missing Gemma Scope params.safetensors for layer {layer}: {source_params_path}"
            )
        observed_l0 = _validate_source_config(
            _load_source_config(source_config_path),
            layer=int(layer),
            model_id=str(profile.model_id),
            width=str(profile.sae_width),
            expected_l0=int(entry.expected_l0),
            config_path=source_config_path,
        )
        resolved_sources.append(
            {
                "layer": int(layer),
                "entry": entry,
                "source_rel": source_rel,
                "source_config_path": source_config_path,
                "source_params_path": source_params_path,
                "observed_l0": int(observed_l0),
                "bundle_run_name": entry.resolved_bundle_run_name(),
            }
        )

    _prepare_output_dir(out_dir)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.tmp-", dir=str(out_dir.parent)))
    try:
        manifest_rows: list[dict[str, Any]] = []
        for item in resolved_sources:
            layer = int(item["layer"])
            entry = item["entry"]
            source_params_path = Path(item["source_params_path"])
            source_arrays = _map_source_arrays(
                _load_source_arrays(source_params_path),
                source_params_path=source_params_path,
            )
            bundle_rel = Path(f"layer_{layer}") / f"width_{profile.sae_width}" / str(item["bundle_run_name"])
            run_dir = staging_dir / bundle_rel
            params_path = run_dir / "params.npz"
            _write_params_npz(params_path, source_arrays)
            _verify_round_trip_exact(
                source_arrays=source_arrays,
                materialized_params_path=params_path,
            )
            materialized_sha = _sha256(params_path)
            cfg = _build_cfg_from_arrays(
                source_arrays,
                hidden_size=hidden_size,
                width=str(profile.sae_width),
                params_sha256=materialized_sha,
                source_repo_id=str(profile.sae_repo_id),
            )
            _write_cfg(run_dir, cfg)
            manifest_rows.append(
                {
                    "layer": int(layer),
                    "width": str(profile.sae_width),
                    "run_name": str(item["bundle_run_name"]),
                    "source_entry": str(item["source_rel"].as_posix()),
                    "expected_l0": int(entry.expected_l0),
                    "observed_l0": int(item["observed_l0"]),
                    "params_path": limitation_requirements.relative_to_root(out_dir / bundle_rel / "params.npz"),
                    "cfg_path": limitation_requirements.relative_to_root(out_dir / bundle_rel / "cfg.json"),
                    "source_params_sha256": _sha256(source_params_path),
                    "materialized_params_sha256": materialized_sha,
                }
            )

        manifest = {
            "publication_profile": limitation_requirements.LIMITATION_PUBLICATION_PROFILE,
            "bundle_root": limitation_requirements.relative_to_root(out_dir),
            "model": {
                "id": str(profile.model_id),
                "revision": str(profile.model_revision),
                "hidden_size": int(hidden_size),
                "num_hidden_layers": int(num_hidden_layers),
            },
            "sae_source": {
                "repo_id": str(profile.sae_repo_id),
                "revision": str(profile.sae_repo_revision),
                "width": str(profile.sae_width),
            },
            "layers": manifest_rows,
        }
        manifest["bundle_id"] = limitation_requirements.compute_limitation_sae_bundle_id(manifest)
        limitation_requirements.limitation_bundle_manifest_path(staging_dir).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        shutil.move(str(staging_dir), str(out_dir))
        print(
            f"Prepared limitation bundle at {out_dir} for layers "
            f"{','.join(str(item['layer']) for item in manifest_rows)}",
            flush=True,
        )
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
