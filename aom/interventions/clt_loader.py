from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn

from aom.interventions.activation_patching import get_decoder_blocks
from aom.interventions.clt_adapter import CLTInputTransform, CLTProtocol, CLTSiteSpec


@dataclass(frozen=True)
class CLTMetadata:
    repo_id_or_path: str
    layer: int
    width: str
    run_name: str
    d_in: int
    d_latent: int
    d_out: int
    encode_site: str
    decode_site: str
    writeback_site: str
    site_mode: str
    params_path: str
    cfg_path: Optional[str]
    inferred_from_weights: bool = False


class LinearCLT(nn.Module):
    """
    Minimal linear CLT module suitable for inference-time patching experiments.

    By default this mirrors SAE-style behavior:
      z = relu(x @ W_enc + b_enc)
      y = z @ W_dec + b_dec
    """

    def __init__(
        self,
        *,
        W_enc: torch.Tensor,
        W_dec: torch.Tensor,
        b_enc: torch.Tensor,
        b_dec: torch.Tensor,
        cfg: Dict[str, Any],
        threshold: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        d_in = int(cfg["d_in"])
        d_latent = int(cfg["d_latent"])
        d_out = int(cfg.get("d_out", d_in))

        if W_enc.shape != (d_in, d_latent):
            raise ValueError(f"W_enc must have shape {(d_in, d_latent)}; got {tuple(W_enc.shape)}")
        if W_dec.shape != (d_latent, d_out):
            raise ValueError(f"W_dec must have shape {(d_latent, d_out)}; got {tuple(W_dec.shape)}")
        if b_enc.shape != (d_latent,):
            raise ValueError(f"b_enc must have shape {(d_latent,)}; got {tuple(b_enc.shape)}")
        if b_dec.shape != (d_out,):
            raise ValueError(f"b_dec must have shape {(d_out,)}; got {tuple(b_dec.shape)}")

        self.W_enc = nn.Parameter(W_enc, requires_grad=False)
        self.W_dec = nn.Parameter(W_dec, requires_grad=False)
        self.b_enc = nn.Parameter(b_enc, requires_grad=False)
        self.b_dec = nn.Parameter(b_dec, requires_grad=False)
        self.cfg = dict(cfg)

        self._d_in = d_in
        self._d_latent = d_latent
        self._d_out = d_out
        act_raw = str(cfg.get("activation", "relu")).lower()
        self._activation = act_raw.replace("_", "").replace("-", "")
        if self._activation not in {"relu", "identity", "gelu", "jumprelu"}:
            raise ValueError(
                f"Unsupported activation={self._activation!r}; expected one of relu|identity|gelu|jumprelu"
            )

        if threshold is not None:
            if threshold.shape != (d_latent,):
                raise ValueError(f"threshold must have shape ({d_latent},); got {tuple(threshold.shape)}")
            self._threshold = nn.Parameter(threshold, requires_grad=False)
        else:
            self._threshold = None

        if self._activation == "jumprelu" and self._threshold is None:
            raise ValueError("activation='jumprelu' requires a threshold tensor")

        # Backward-safe default: do not apply pre-encoder bias unless explicitly enabled.
        self._pre_encoder_bias = bool(cfg.get("pre_encoder_bias", False))
        if self._pre_encoder_bias and self._d_out != self._d_in:
            raise ValueError(
                "pre_encoder_bias=True requires d_out == d_in so b_dec can be subtracted from encoder inputs"
            )

    @property
    def d_in(self) -> int:  # noqa: D401
        """Input dimension for `encode`."""
        return self._d_in

    @property
    def d_latent(self) -> int:  # noqa: D401
        """Latent dimension."""
        return self._d_latent

    @property
    def d_out(self) -> int:  # noqa: D401
        """Output dimension for `decode`."""
        return self._d_out

    def _apply_activation(self, z: torch.Tensor) -> torch.Tensor:
        if self._activation == "relu":
            return torch.relu(z)
        if self._activation == "gelu":
            return torch.nn.functional.gelu(z)
        if self._activation == "jumprelu":
            assert self._threshold is not None
            mask = (z > self._threshold).to(z.dtype)
            return mask * torch.relu(z)
        return z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"encode expects (B,S,H); got {tuple(x.shape)}")
        if int(x.size(-1)) != self._d_in:
            raise ValueError(f"encode expected hidden dim {self._d_in}; got {int(x.size(-1))}")
        x2 = x.reshape(-1, self._d_in)
        if self._pre_encoder_bias:
            x2 = x2 - self.b_dec
        z2 = x2 @ self.W_enc + self.b_enc
        z2 = self._apply_activation(z2)
        return z2.reshape(x.shape[0], x.shape[1], self._d_latent)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 3:
            raise ValueError(f"decode expects (B,S,D); got {tuple(latents.shape)}")
        if int(latents.size(-1)) != self._d_latent:
            raise ValueError(f"decode expected d_latent {self._d_latent}; got {int(latents.size(-1))}")
        z2 = latents.reshape(-1, self._d_latent)
        y2 = z2 @ self.W_dec + self.b_dec
        return y2.reshape(latents.shape[0], latents.shape[1], self._d_out)


def _as_torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    return getattr(torch, str(dtype))


def _load_cfg(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Accept SAE-style d_sae alias for compatibility.
    if "d_latent" not in cfg and "d_sae" in cfg:
        cfg["d_latent"] = int(cfg["d_sae"])
    if "d_in" not in cfg or "d_latent" not in cfg:
        raise ValueError("cfg.json must contain keys 'd_in' and 'd_latent' (or 'd_sae')")
    if "d_out" not in cfg:
        cfg["d_out"] = int(cfg["d_in"])
    if "encode_site" not in cfg:
        cfg["encode_site"] = "resid_post"
    if "decode_site" not in cfg:
        cfg["decode_site"] = str(cfg["encode_site"])
    if "writeback_site" not in cfg:
        cfg["writeback_site"] = str(cfg["decode_site"])
    if "site_mode" not in cfg:
        cfg["site_mode"] = "same_site_v1"
    if "activation" not in cfg:
        cfg["activation"] = "relu"
    act = str(cfg["activation"]).lower().replace("_", "").replace("-", "")
    if act == "jumprelu":
        cfg["activation"] = "jumprelu"
    elif act in {"relu", "identity", "gelu"}:
        cfg["activation"] = act
    else:
        # Keep unknown activation as-is; LinearCLT validates and raises a clear error.
        cfg["activation"] = str(cfg["activation"])
    return cfg


def _load_params_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(str(path), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _infer_cfg_from_weights(arrays: Dict[str, np.ndarray]) -> Dict[str, Any]:
    if "W_enc" not in arrays or "W_dec" not in arrays or "b_enc" not in arrays or "b_dec" not in arrays:
        raise ValueError(
            "Cannot infer CLT cfg: params.npz must include 'W_enc', 'W_dec', 'b_enc', 'b_dec' "
            f"(found keys={sorted(arrays.keys())})"
        )

    W_enc_np = arrays["W_enc"]
    W_dec_np = arrays["W_dec"]
    b_enc_np = np.asarray(arrays["b_enc"]).reshape(-1)
    b_dec_np = np.asarray(arrays["b_dec"]).reshape(-1)

    if W_enc_np.ndim != 2 or W_dec_np.ndim != 2:
        raise ValueError(
            f"Expected W_enc/W_dec to be 2D; got W_enc={tuple(W_enc_np.shape)}, W_dec={tuple(W_dec_np.shape)}"
        )
    if b_enc_np.ndim != 1 or b_enc_np.size < 1:
        raise ValueError(f"Expected b_enc to be 1D with len>0; got shape={tuple(arrays['b_enc'].shape)}")
    if b_dec_np.ndim != 1 or b_dec_np.size < 1:
        raise ValueError(f"Expected b_dec to be 1D with len>0; got shape={tuple(arrays['b_dec'].shape)}")

    d_latent = int(b_enc_np.shape[0])
    d_out = int(b_dec_np.shape[0])

    if W_enc_np.shape[1] == d_latent and W_enc_np.shape[0] != d_latent:
        d_in = int(W_enc_np.shape[0])
    elif W_enc_np.shape[0] == d_latent and W_enc_np.shape[1] != d_latent:
        d_in = int(W_enc_np.shape[1])
    elif W_enc_np.shape[0] == d_latent and W_enc_np.shape[1] == d_latent:
        # Ambiguous square matrix; defer to W_dec orientation.
        if W_dec_np.shape[0] == d_latent and W_dec_np.shape[1] == d_out:
            d_in = int(d_out)
        elif W_dec_np.shape[1] == d_latent and W_dec_np.shape[0] == d_out:
            d_in = int(d_out)
        else:
            d_in = int(d_latent)
    else:
        raise ValueError(
            "Cannot infer d_in from W_enc/b_enc: expected one W_enc dimension to equal "
            f"d_latent={d_latent}, got W_enc shape={tuple(W_enc_np.shape)}"
        )

    if W_dec_np.shape not in ((d_latent, d_out), (d_out, d_latent)):
        raise ValueError(
            f"Cannot infer W_dec orientation for inferred dims (d_latent={d_latent}, d_out={d_out}); "
            f"got W_dec shape={tuple(W_dec_np.shape)}"
        )

    activation = "jumprelu" if "threshold" in arrays else "relu"
    return {
        "d_in": d_in,
        "d_latent": d_latent,
        "d_out": d_out,
        "dtype": str(W_enc_np.dtype),
        "inferred_from_weights": True,
        "encode_site": "resid_post",
        "decode_site": "resid_post",
        "writeback_site": "resid_post",
        "site_mode": "same_site_v1",
        "activation": activation,
        "pre_encoder_bias": False,
    }


def _canonicalize_weights(
    *,
    cfg: Dict[str, Any],
    arrays: Dict[str, np.ndarray],
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    for k in ("W_enc", "W_dec", "b_enc", "b_dec"):
        if k not in arrays:
            raise ValueError(f"params.npz missing key {k!r}; found keys={sorted(arrays.keys())}")

    d_in = int(cfg["d_in"])
    d_latent = int(cfg["d_latent"])
    d_out = int(cfg.get("d_out", d_in))

    W_enc = torch.from_numpy(arrays["W_enc"]).to(device=device, dtype=dtype)
    W_dec = torch.from_numpy(arrays["W_dec"]).to(device=device, dtype=dtype)
    b_enc = torch.from_numpy(arrays["b_enc"]).to(device=device, dtype=dtype).reshape(-1)
    b_dec = torch.from_numpy(arrays["b_dec"]).to(device=device, dtype=dtype).reshape(-1)

    if W_enc.shape == (d_latent, d_in):
        W_enc = W_enc.t()
    if W_dec.shape == (d_out, d_latent):
        W_dec = W_dec.t()

    if W_enc.shape != (d_in, d_latent):
        raise ValueError(
            f"W_enc shape mismatch: got {tuple(W_enc.shape)}, expected {(d_in, d_latent)} (or transpose)"
        )
    if W_dec.shape != (d_latent, d_out):
        raise ValueError(
            f"W_dec shape mismatch: got {tuple(W_dec.shape)}, expected {(d_latent, d_out)} (or transpose)"
        )
    if b_enc.shape != (d_latent,):
        raise ValueError(f"b_enc shape mismatch: got {tuple(b_enc.shape)}, expected {(d_latent,)}")
    if b_dec.shape != (d_out,):
        raise ValueError(f"b_dec shape mismatch: got {tuple(b_dec.shape)}, expected {(d_out,)}")

    threshold: Optional[torch.Tensor] = None
    if "threshold" in arrays:
        threshold = torch.from_numpy(arrays["threshold"]).to(device=device, dtype=dtype).reshape(-1)
        if threshold.shape != (d_latent,):
            raise ValueError(
                f"threshold shape mismatch: got {tuple(threshold.shape)}, expected ({d_latent},)"
            )
    return W_enc, W_dec, b_enc, b_dec, threshold


def _resolve_local_run_dir(
    root: Path,
    *,
    layer: int,
    width: str,
    run_name: Optional[str],
    l0_target: Optional[int],
) -> Path:
    base = root / f"layer_{int(layer)}" / f"width_{str(width)}"
    if not base.exists():
        raise FileNotFoundError(f"Missing CLT directory: {str(base)}")
    if run_name is not None:
        run_dir = base / str(run_name)
        if not run_dir.exists():
            available = sorted([p.name for p in base.iterdir() if p.is_dir()])
            raise FileNotFoundError(f"Missing run directory: {str(run_dir)} (available runs: {available})")
        return run_dir

    candidates = [p for p in base.iterdir() if p.is_dir()]
    if l0_target is not None:
        tag = f"average_l0_{int(l0_target)}"
        candidates = [p for p in candidates if tag in p.name]

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) < 1:
        raise FileNotFoundError(f"No run directories found under {str(base)}")
    raise ValueError(
        f"Multiple CLT runs under {str(base)}; specify run_name (one of {[p.name for p in candidates]})"
    )


def list_clt_runs(
    repo_id_or_path: str,
    *,
    layer: int,
    width: str = "16k",
    revision: Optional[str] = None,
    local_files_only: bool = False,
) -> list[str]:
    """
    List CLT run directories for a given layer/width.

    Supports local paths and HF repo IDs.
    """
    root = Path(repo_id_or_path)
    if root.exists():
        base = root / f"layer_{int(layer)}" / f"width_{str(width)}"
        if not base.exists():
            raise FileNotFoundError(f"Missing CLT directory: {str(base)}")
        return sorted([p.name for p in base.iterdir() if p.is_dir()])

    if bool(local_files_only):
        raise ValueError("Cannot list CLT runs for HF repo with local_files_only=True; provide a local path.")

    try:
        from huggingface_hub import HfApi
    except ImportError as e:  # pragma: no cover
        raise ImportError("huggingface_hub is required to list CLT runs") from e

    api = HfApi()
    prefix = f"layer_{int(layer)}/width_{str(width)}/"
    files = api.list_repo_files(repo_id_or_path, revision=revision)
    return sorted({Path(f).parent.name for f in files if f.startswith(prefix) and f.endswith("params.npz")})


def load_clt(
    repo_id_or_path: str,
    *,
    layer: int,
    width: str = "16k",
    run_name: Optional[str] = None,
    l0_target: Optional[int] = None,
    device: str = "mps",
    dtype: str | torch.dtype = "float32",
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
) -> tuple[CLTProtocol, CLTMetadata]:
    """
    Load a CLT module from local directory or HF repo.
    """

    device_t = torch.device(str(device))
    dtype_t = _as_torch_dtype(dtype)

    root = Path(repo_id_or_path)
    if root.exists():
        run_dir = _resolve_local_run_dir(root, layer=layer, width=width, run_name=run_name, l0_target=l0_target)
        params_path = run_dir / "params.npz"
        cfg_path: Optional[Path] = run_dir / "cfg.json"
    else:
        if run_name is None and l0_target is None:
            raise ValueError("For HF repo ids, specify run_name or l0_target to disambiguate run directory.")
        try:
            import huggingface_hub
            from huggingface_hub.utils import EntryNotFoundError
        except ImportError as e:  # pragma: no cover
            raise ImportError("huggingface_hub is required to download CLT artifacts") from e

        prefix = f"layer_{int(layer)}/width_{str(width)}/"
        if run_name is not None:
            run_dir_rel = str(Path(prefix) / str(run_name))
        else:
            run_names = list_clt_runs(
                repo_id_or_path,
                layer=int(layer),
                width=str(width),
                revision=revision,
                local_files_only=bool(local_files_only),
            )
            tag = f"average_l0_{int(l0_target)}"
            run_names = [n for n in run_names if tag in n]
            if len(run_names) != 1:
                run_dirs = [str(Path(prefix) / n) for n in run_names]
                raise ValueError(f"Expected exactly 1 run dir under {prefix}; found {run_dirs}")
            run_dir_rel = str(Path(prefix) / run_names[0])

        try:
            params_path = Path(
                huggingface_hub.hf_hub_download(
                    repo_id_or_path,
                    filename=str(Path(run_dir_rel) / "params.npz"),
                    revision=revision,
                    cache_dir=cache_dir,
                    local_files_only=bool(local_files_only),
                )
            )
        except EntryNotFoundError as e:
            try:
                runs = list_clt_runs(repo_id_or_path, layer=int(layer), width=str(width), revision=revision)
            except Exception:
                runs = []
            raise FileNotFoundError(
                f"Missing params.npz at {run_dir_rel!r} in repo {repo_id_or_path!r} (available runs: {runs})"
            ) from e

        run_dir = Path(run_dir_rel)
        cfg_path = None

    arrays = _load_params_npz(params_path)

    if root.exists():
        if cfg_path is not None and cfg_path.exists():
            cfg = _load_cfg(cfg_path)
        else:
            cfg = _infer_cfg_from_weights(arrays)
            cfg_path = None
    else:
        try:
            import huggingface_hub
            from huggingface_hub.utils import EntryNotFoundError

            cfg_path = Path(
                huggingface_hub.hf_hub_download(
                    repo_id_or_path,
                    filename=str(Path(run_dir) / "cfg.json"),
                    revision=revision,
                    cache_dir=cache_dir,
                    local_files_only=bool(local_files_only),
                )
            )
            cfg = _load_cfg(cfg_path)
        except EntryNotFoundError:
            cfg = _infer_cfg_from_weights(arrays)
            cfg_path = None

    # Enforce site contract for v1.
    site = CLTSiteSpec(
        encode_site=str(cfg.get("encode_site", "resid_post")),
        decode_site=str(cfg.get("decode_site", cfg.get("encode_site", "resid_post"))),
        writeback_site=str(cfg.get("writeback_site", cfg.get("decode_site", cfg.get("encode_site", "resid_post")))),
        site_mode=str(cfg.get("site_mode", "same_site_v1")),
    )
    cfg["encode_site"] = site.encode_site
    cfg["decode_site"] = site.decode_site
    cfg["writeback_site"] = site.writeback_site
    cfg["site_mode"] = site.site_mode

    W_enc, W_dec, b_enc, b_dec, threshold = _canonicalize_weights(
        cfg=cfg, arrays=arrays, dtype=dtype_t, device=device_t
    )
    clt = LinearCLT(W_enc=W_enc, W_dec=W_dec, b_enc=b_enc, b_dec=b_dec, cfg=cfg, threshold=threshold)

    meta = CLTMetadata(
        repo_id_or_path=str(repo_id_or_path),
        layer=int(layer),
        width=str(width),
        run_name=str(run_dir.name),
        d_in=int(cfg["d_in"]),
        d_latent=int(cfg["d_latent"]),
        d_out=int(cfg.get("d_out", cfg["d_in"])),
        encode_site=site.encode_site,
        decode_site=site.decode_site,
        writeback_site=site.writeback_site,
        site_mode=site.site_mode,
        params_path=str(params_path),
        cfg_path=str(cfg_path) if cfg_path is not None else None,
        inferred_from_weights=bool(cfg.get("inferred_from_weights", False)),
    )
    return clt, meta


@torch.no_grad()
def calibrate_clt_scale(
    *,
    model: nn.Module,
    clt: CLTProtocol,
    layer: int,
    scales: Tuple[float, ...] = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
    calibration_input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    shift: float | None = None,
) -> tuple[float, Dict[float, float]]:
    """
    Grid search over CLT input scale to minimize reconstruction MSE at the hooked tensor.

    This captures the exact block output tensor targeted by patching hooks.
    """
    if calibration_input_ids is None:
        raise ValueError("calibration_input_ids is required")

    blocks = get_decoder_blocks(model)
    if int(layer) < 0 or int(layer) >= len(blocks):
        raise ValueError(f"layer {int(layer)} out of range [0, {len(blocks)})")

    captured: list[torch.Tensor] = []

    def _capture(_module: nn.Module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if not isinstance(h, torch.Tensor) or h.ndim != 3:
            raise ValueError("expected hidden tensor (B,S,H) from block output")
        captured.append(h.detach())

    handle = blocks[int(layer)].register_forward_hook(_capture)
    try:
        model(
            input_ids=calibration_input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
    finally:
        handle.remove()

    if len(captured) != 1:
        raise RuntimeError(f"expected exactly one capture; got {len(captured)}")
    hidden = captured[0]

    # Align hidden tensor to CLT param device/dtype unless explicit device override is requested.
    clt_param = next(clt.parameters(), None) if isinstance(clt, nn.Module) else None
    if device is not None:
        hidden = hidden.to(device=device)
    elif clt_param is not None:
        hidden = hidden.to(device=clt_param.device, dtype=clt_param.dtype)

    best_scale = float(scales[0])
    best_mse = float("inf")
    mse_by_scale: Dict[float, float] = {}

    for s in scales:
        tr = CLTInputTransform(scale=float(s), shift=shift)
        x_in = tr.forward(hidden)
        z = clt.encode(x_in)
        recon = clt.decode(z)
        recon_model = tr.inverse(recon)
        mse = float(torch.mean((recon_model - hidden) ** 2).item())
        mse_by_scale[float(s)] = mse
        if mse < best_mse - 1e-12:
            best_mse = mse
            best_scale = float(s)
    return best_scale, mse_by_scale
