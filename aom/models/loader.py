from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


def _safe_get_commit_hash(obj) -> str | None:
    try:
        cfg = getattr(obj, "config", None)
        if cfg is not None:
            v = getattr(cfg, "_commit_hash", None)
            if v:
                return str(v)
        v2 = getattr(obj, "_commit_hash", None)
        return str(v2) if v2 else None
    except Exception:
        return None


@dataclass(frozen=True)
class LoadedModel:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    architecture: str
    model_commit_hash: Optional[str] = None
    tokenizer_revision_effective: Optional[str] = None


def _looks_like_hf_repo_id(model_name_or_path: str) -> bool:
    path = Path(str(model_name_or_path))
    if path.exists():
        return False
    raw = str(model_name_or_path)
    return "/" in raw and not raw.startswith(("./", "../", "/"))


def _resolve_local_snapshot_path(model_name_or_path: str, revision: str | None) -> str:
    if not _looks_like_hf_repo_id(model_name_or_path):
        return model_name_or_path

    from huggingface_hub import scan_cache_dir

    target_revision = str(revision or "").strip()
    for repo in scan_cache_dir().repos:
        if repo.repo_id != model_name_or_path:
            continue
        if target_revision:
            for cached_revision in repo.revisions:
                if cached_revision.commit_hash == target_revision:
                    return str(cached_revision.snapshot_path)
            raise FileNotFoundError(
                f"Cached revision {target_revision!r} not found for model {model_name_or_path!r}"
            )
        if repo.revisions:
            chosen = sorted(repo.revisions, key=lambda item: item.commit_hash)[-1]
            return str(chosen.snapshot_path)
    raise FileNotFoundError(f"No cached snapshot found for model {model_name_or_path!r}")


def load_causal_lm(
    model_name_or_path: str,
    device: torch.device,
    *,
    torch_dtype: Optional[str] = None,
    revision: str | None = None,
    tokenizer_revision: str | None = None,
    local_files_only: bool = False,
    trust_remote_code: bool = False,
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "eager",
    device_map: Optional[str] = None,
) -> LoadedModel:
    dtype = None
    torch_dtype_s = None if torch_dtype is None else str(torch_dtype).strip().lower()
    if torch_dtype_s in {None, "", "none", "null", "auto"}:
        if str(getattr(device, "type", "")).lower() == "mps":
            dtype = torch.float16
        elif "qwen" in model_name_or_path.lower():
            dtype = torch.bfloat16
    else:
        dtype = getattr(torch, str(torch_dtype_s))

    if dtype is None and "qwen" in model_name_or_path.lower():
        dtype = torch.bfloat16

    resolved_model_name_or_path = model_name_or_path
    use_revision_kwargs = True
    if local_files_only:
        resolved_model_name_or_path = _resolve_local_snapshot_path(model_name_or_path, revision)
        if resolved_model_name_or_path != model_name_or_path:
            use_revision_kwargs = False

    tokenizer_kwargs: dict = {
        "local_files_only": local_files_only,
        "trust_remote_code": trust_remote_code,
        "use_fast": True,
    }
    tok_rev = tokenizer_revision if tokenizer_revision is not None else revision
    if tok_rev is not None and use_revision_kwargs:
        tokenizer_kwargs["revision"] = str(tok_rev)

    tokenizer = AutoTokenizer.from_pretrained(
        resolved_model_name_or_path,
        **tokenizer_kwargs,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict = {
        "local_files_only": local_files_only,
        "trust_remote_code": trust_remote_code,
        "torch_dtype": dtype,
        "attn_implementation": attn_implementation,
    }
    if revision is not None and use_revision_kwargs:
        model_kwargs["revision"] = str(revision)
    if device_map is not None:
        model_kwargs["device_map"] = device_map

    def _load_model(kwargs: dict) -> PreTrainedModel:
        try:
            return AutoModelForCausalLM.from_pretrained(resolved_model_name_or_path, **kwargs)
        except ImportError as e:
            if device_map is not None:
                raise ImportError(
                    "Loading with `device_map` requires the `accelerate` package. "
                    "Install it (e.g. `pip install accelerate`) or rerun without `device_map`."
                ) from e
            raise

    try:
        model = _load_model(model_kwargs)
    except TypeError:
        # Some architectures/configs may not accept attn_implementation; retry without it.
        model_kwargs.pop("attn_implementation", None)
        model = _load_model(model_kwargs)

    if device_map is None:
        model.to(device)
    model.eval()

    from aom.interventions.activation_patching import detect_architecture

    arch = detect_architecture(model)

    model_commit_hash = _safe_get_commit_hash(model)

    tokenizer_revision_effective = None
    try:
        init_kwargs = getattr(tokenizer, "init_kwargs", None)
        if isinstance(init_kwargs, dict):
            tokenizer_revision_effective = init_kwargs.get("revision", None)
    except Exception:
        tokenizer_revision_effective = None

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        architecture=arch,
        model_commit_hash=model_commit_hash,
        tokenizer_revision_effective=tokenizer_revision_effective,
    )
