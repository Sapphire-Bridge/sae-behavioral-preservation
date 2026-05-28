from __future__ import annotations

import importlib
import os
import platform
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import torch


DeterminismMode = Literal["strict", "best_effort", "off"]


@dataclass(frozen=True)
class ReproConfig:
    seed: int
    determinism: DeterminismMode


def _safe_version(module_name: str) -> str:
    try:
        mod = importlib.import_module(module_name)
        v = getattr(mod, "__version__", None)
        return "" if v is None else str(v)
    except Exception:
        return ""


def collect_versions() -> dict[str, str]:
    return {
        "python": str(sys.version.split()[0]),
        "platform": str(platform.platform()),
        "numpy": _safe_version("numpy"),
        "torch": _safe_version("torch"),
        "transformers": _safe_version("transformers"),
        "tokenizers": _safe_version("tokenizers"),
    }


def get_git_commit_hash(*, repo_root: str | Path | None = None, required: bool = False) -> str:
    """
    Return the current git commit hash (HEAD).

    If `required=True`, raise a RuntimeError when the hash cannot be determined.
    """
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if out:
            return str(out)
    except Exception:
        pass
    if required:
        raise RuntimeError(f"Failed to determine git commit hash (expected a git checkout at {root})")
    return ""


def _device_backend(device: Any) -> str:
    if device is None:
        return ""
    try:
        import torch

        if isinstance(device, torch.device):
            return str(device.type)
        return str(torch.device(str(device)).type)
    except Exception:
        return str(device)


def seed_everything(cfg: ReproConfig, *, device: Any = None) -> dict[str, Any]:
    """
    Seed common RNGs and configure determinism knobs.

    Notes:
    - This function always sets RNG seeds.
    - determinism=off: seed only (do not request deterministic algorithms).
    - determinism=best_effort: attempt deterministic algorithms when supported (warn-only when possible).
    - determinism=strict: enforce deterministic algorithms (fail-fast on nondeterministic ops).
    - On MPS, strict determinism is treated as unsupported (enforced=False). Use CPU for strict reproducibility.
    """
    import torch

    seed = int(cfg.seed)
    mode: DeterminismMode = str(cfg.determinism)  # type: ignore[assignment]
    backend = _device_backend(device)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    enforced = False
    reason: Optional[str] = None
    torch_deterministic_algorithms: Optional[bool] = None
    torch_deterministic_warn_only: Optional[bool] = None
    cudnn_deterministic: Optional[bool] = None
    cudnn_benchmark: Optional[bool] = None
    cublas_workspace_config: Optional[str] = None

    if mode == "off":
        reason = "determinism=off"
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(False)
                torch_deterministic_algorithms = False
            except Exception:
                # Best-effort cleanup; some torch builds may not support toggling.
                pass
    elif backend == "mps":
        if mode == "strict":
            reason = "MPS backend: strict determinism is unsupported (use --device cpu)"
        else:
            reason = "MPS backend: seeded RNGs only (determinism not guaranteed)"
    else:
        if not hasattr(torch, "use_deterministic_algorithms"):
            if mode == "best_effort":
                reason = "best_effort: torch.use_deterministic_algorithms unavailable"
            else:
                reason = "torch.use_deterministic_algorithms is unavailable"
        else:
            try:
                if mode == "best_effort":
                    # Try to enable deterministic algorithms without failing the run when
                    # a nondeterministic op is encountered (warn-only when supported).
                    try:
                        torch.use_deterministic_algorithms(True, warn_only=True)
                        torch_deterministic_warn_only = True
                        torch_deterministic_algorithms = True
                        enforced = True
                    except TypeError:
                        # Older torch without warn_only; best-effort degrades to strict-ish.
                        torch.use_deterministic_algorithms(True)
                        torch_deterministic_algorithms = True
                        torch_deterministic_warn_only = False
                        enforced = True
                        reason = "best_effort: warn_only unsupported; deterministic_algorithms=True"
                elif mode == "strict":
                    torch.use_deterministic_algorithms(True)
                    torch_deterministic_algorithms = True
                    torch_deterministic_warn_only = False
                    enforced = True
                else:
                    reason = f"Unknown determinism mode: {mode!r}"
            except Exception as e:
                reason = f"torch.use_deterministic_algorithms failed: {type(e).__name__}: {e}"

        if enforced and backend == "cuda":
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                cudnn_deterministic = bool(torch.backends.cudnn.deterministic)
                cudnn_benchmark = bool(torch.backends.cudnn.benchmark)
            except Exception as e:
                enforced = False
                if reason is None:
                    reason = f"Failed to set cudnn deterministic flags: {type(e).__name__}: {e}"

    if backend == "cuda":
        cublas_workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG", None)
        if mode == "strict" and cublas_workspace_config is None:
            note = (
                "CUBLAS_WORKSPACE_CONFIG is unset; set it for fully deterministic CUDA GEMMs "
                "(e.g. ':16:8' or ':4096:8')."
            )
            reason = note if reason is None else f"{reason} | {note}"

    return {
        "seed": seed,
        "determinism_requested": str(mode),
        "device_backend": str(backend),
        "determinism_enforced": bool(enforced),
        "determinism_reason": None if reason is None else str(reason),
        "torch_deterministic_algorithms": torch_deterministic_algorithms,
        "torch_deterministic_warn_only": torch_deterministic_warn_only,
        "cudnn_deterministic": cudnn_deterministic,
        "cudnn_benchmark": cudnn_benchmark,
        "cublas_workspace_config": cublas_workspace_config,
    }
