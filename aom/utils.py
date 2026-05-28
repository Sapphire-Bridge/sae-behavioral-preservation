from __future__ import annotations

import json
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .stats import MetricValue, bootstrap_ci, bootstrap_ci_metric

_LOGPROBS_DTYPE: torch.dtype = torch.float32
_STRICT_FINITE: bool = True
_SCORE_BATCH_SIZE: int = 1
_USE_PREFIX_CACHE: bool = False


def configure_logprob_computation(*, logprobs_dtype: torch.dtype, strict_finite: bool) -> None:
    """
    Configure how log-probabilities are computed across experiment scripts.

    Default behavior is set at import time (float32 log-softmax + strict finite checks).
    The limitation runner wires this up before scoring.
    """
    global _LOGPROBS_DTYPE, _STRICT_FINITE
    _LOGPROBS_DTYPE = logprobs_dtype
    _STRICT_FINITE = bool(strict_finite)


def get_logprob_computation_config() -> Tuple[torch.dtype, bool]:
    return _LOGPROBS_DTYPE, _STRICT_FINITE


def configure_scoring_performance(*, score_batch_size: int, use_prefix_cache: bool) -> None:
    """
    Configure performance knobs for continuation scoring.

    score_batch_size controls how many continuations are scored per forward pass.
    use_prefix_cache enables prompt prefill KV-cache reuse across continuations.
    """
    global _SCORE_BATCH_SIZE, _USE_PREFIX_CACHE
    _SCORE_BATCH_SIZE = max(1, int(score_batch_size))
    _USE_PREFIX_CACHE = bool(use_prefix_cache)


def get_scoring_performance_config() -> Tuple[int, bool]:
    return int(_SCORE_BATCH_SIZE), bool(_USE_PREFIX_CACHE)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {e}") from e
    return rows


def write_jsonl(rows: Iterable[Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for row in rows:
            if is_dataclass(row):
                row = asdict(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


@torch.no_grad()
def logprob_of_continuation_ids(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    continuation_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    normalize_by_length: bool = False,
) -> torch.Tensor:
    """
    Compute log P(continuation | prompt) for a batch, using already-tokenized IDs.

    Shapes:
      prompt_ids: (B, P)
      continuation_ids: (B, C)
    Returns:
      logp: (B,) (sum or mean per token)
    """
    if prompt_ids.ndim != 2 or continuation_ids.ndim != 2:
        raise ValueError("prompt_ids and continuation_ids must be rank-2 tensors")
    if prompt_ids.size(0) != continuation_ids.size(0):
        raise ValueError("prompt_ids and continuation_ids must have same batch size")
    if continuation_ids.size(1) < 1:
        raise ValueError("continuation_ids must have length >= 1")

    full_ids = torch.cat([prompt_ids, continuation_ids], dim=1)
    if attention_mask is None:
        attention_mask = torch.ones_like(full_ids)
    else:
        if attention_mask.shape != full_ids.shape:
            raise ValueError("attention_mask must match full_ids shape")

    out = model(input_ids=full_ids, attention_mask=attention_mask, use_cache=False)
    logits = out.logits  # type: ignore[attr-defined]

    # continuation tokens live at positions [P, P+C-1]; each is predicted by logits at [P-1, P+C-2].
    P = prompt_ids.size(1)
    C = continuation_ids.size(1)
    logits_slice = logits[:, P - 1 : P + C - 1, :].to(dtype=_LOGPROBS_DTYPE)
    log_probs = F.log_softmax(logits_slice, dim=-1)
    gathered = log_probs.gather(2, continuation_ids.unsqueeze(-1)).squeeze(-1)  # (B, C)
    if not torch.isfinite(gathered).all():
        if _STRICT_FINITE:
            raise FloatingPointError(
                "Non-finite log-probability detected in continuation scoring "
                f"(device={gathered.device}, logits_dtype={logits.dtype}, logprobs_dtype={gathered.dtype})."
            )
        gathered = torch.where(torch.isfinite(gathered), gathered, torch.full_like(gathered, -1e9))
    if normalize_by_length:
        return gathered.mean(dim=1)
    return gathered.sum(dim=1)


def _as_legacy_past_key_values(past_key_values: Any) -> tuple[tuple[torch.Tensor, ...], ...]:
    if isinstance(past_key_values, tuple):
        return past_key_values
    to_legacy = getattr(past_key_values, "to_legacy_cache", None)
    if callable(to_legacy):
        legacy = to_legacy()
        if isinstance(legacy, tuple):
            return legacy
    raise TypeError(f"Unsupported cache type for legacy conversion: {type(past_key_values)!r}")


def _repeat_legacy_past_key_values(
    legacy_past_key_values: tuple[tuple[torch.Tensor, ...], ...],
    repeats: int,
) -> tuple[tuple[torch.Tensor, ...], ...]:
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    out_layers: list[tuple[torch.Tensor, ...]] = []
    for layer in legacy_past_key_values:
        if not isinstance(layer, tuple):
            raise TypeError("legacy past_key_values must be a tuple of tuples")
        out_layers.append(tuple(t.repeat_interleave(int(repeats), dim=0) for t in layer))
    return tuple(out_layers)


def _rebuild_past_like(reference_past_key_values: Any, legacy_past_key_values: tuple[tuple[torch.Tensor, ...], ...]) -> Any:
    if isinstance(reference_past_key_values, tuple):
        return legacy_past_key_values
    cls = type(reference_past_key_values)
    from_legacy = getattr(cls, "from_legacy_cache", None)
    if callable(from_legacy):
        try:
            return from_legacy(legacy_past_key_values)
        except Exception:
            return legacy_past_key_values
    return legacy_past_key_values


def _check_finite(gathered: torch.Tensor, *, mask: Optional[torch.Tensor], context: str) -> torch.Tensor:
    if mask is not None:
        valid = gathered[mask]
    else:
        valid = gathered.reshape(-1)
    if torch.isfinite(valid).all():
        return gathered
    if _STRICT_FINITE:
        raise FloatingPointError(context)
    return torch.where(torch.isfinite(gathered), gathered, torch.full_like(gathered, -1e9))


@torch.no_grad()
def logprob_of_continuation_candidates_with_prefill(
    *,
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    continuation_id_list: Sequence[torch.Tensor],
    prefill_logits_last: torch.Tensor,
    prefill_past_key_values: Any,
    normalize_by_length: bool = False,
    batch_size: int = 1,
    pad_token_id: int = 0,
) -> torch.Tensor:
    """
    Score many continuations for one prompt using a precomputed prompt prefill.

    prefill_logits_last must be the next-token logits after the prompt, shape (1, vocab).
    prefill_past_key_values must correspond to the same prompt prefill.
    """
    if prompt_ids.ndim != 2 or prompt_ids.size(0) != 1:
        raise ValueError("prompt_ids must have shape (1, P)")
    if prefill_logits_last.ndim != 2 or prefill_logits_last.size(0) != 1:
        raise ValueError("prefill_logits_last must have shape (1, vocab)")
    if not continuation_id_list:
        return torch.empty((0,), device=prompt_ids.device, dtype=_LOGPROBS_DTYPE)

    device = prompt_ids.device
    prompt_len = int(prompt_ids.size(1))
    batch_size = max(1, int(batch_size))

    normalized_conts: list[torch.Tensor] = []
    for cont in continuation_id_list:
        if cont.ndim == 2:
            if cont.size(0) != 1:
                raise ValueError("each continuation must have batch size 1 when rank-2")
            cont = cont.squeeze(0)
        if cont.ndim != 1:
            raise ValueError("each continuation must be rank-1 or rank-2 with batch size 1")
        if cont.numel() < 1:
            raise ValueError("each continuation must have length >= 1")
        normalized_conts.append(cont.to(device=device, dtype=torch.long))

    legacy_prefill = _as_legacy_past_key_values(prefill_past_key_values)
    prefix_log_probs = F.log_softmax(prefill_logits_last.to(dtype=_LOGPROBS_DTYPE), dim=-1)  # (1, vocab)

    all_scores: list[torch.Tensor] = []
    for chunk in batched(normalized_conts, batch_size):
        B = int(len(chunk))
        lengths = torch.tensor([int(c.numel()) for c in chunk], device=device, dtype=torch.long)
        first_targets = torch.tensor([int(c[0].item()) for c in chunk], device=device, dtype=torch.long).unsqueeze(-1)
        first_lp = prefix_log_probs.expand(B, -1).gather(1, first_targets).squeeze(-1)  # (B,)
        first_lp = _check_finite(
            first_lp,
            mask=None,
            context=(
                "Non-finite log-probability detected in prefix-cache first-token scoring "
                f"(device={first_lp.device}, logprobs_dtype={first_lp.dtype})."
            ),
        )

        max_rest = int(max(max(0, int(c.numel()) - 1) for c in chunk))
        rest_lp = torch.zeros((B,), device=device, dtype=first_lp.dtype)
        if max_rest > 0:
            prev_pad = torch.full((B, max_rest), int(pad_token_id), device=device, dtype=torch.long)
            tgt_pad = torch.zeros((B, max_rest), device=device, dtype=torch.long)
            rest_mask = torch.zeros((B, max_rest), device=device, dtype=torch.bool)
            for i, c in enumerate(chunk):
                r = max(0, int(c.numel()) - 1)
                if r > 0:
                    prev_pad[i, :r] = c[:-1]
                    tgt_pad[i, :r] = c[1:]
                    rest_mask[i, :r] = True

            repeated_legacy = _repeat_legacy_past_key_values(legacy_prefill, repeats=B)
            past_for_batch = _rebuild_past_like(prefill_past_key_values, repeated_legacy)
            attn_mask = torch.cat(
                [
                    torch.ones((B, prompt_len), device=device, dtype=torch.long),
                    rest_mask.to(dtype=torch.long),
                ],
                dim=1,
            )
            out = model(
                input_ids=prev_pad,
                attention_mask=attn_mask,
                past_key_values=past_for_batch,
                use_cache=False,
                return_dict=True,
            )
            logits = out.logits  # type: ignore[attr-defined]
            rest_log_probs = F.log_softmax(logits.to(dtype=_LOGPROBS_DTYPE), dim=-1)
            gathered_rest = rest_log_probs.gather(2, tgt_pad.unsqueeze(-1)).squeeze(-1)
            gathered_rest = _check_finite(
                gathered_rest,
                mask=rest_mask,
                context=(
                    "Non-finite log-probability detected in prefix-cache continuation scoring "
                    f"(device={gathered_rest.device}, logits_dtype={logits.dtype}, logprobs_dtype={gathered_rest.dtype})."
                ),
            )
            gathered_rest = torch.where(rest_mask, gathered_rest, torch.zeros_like(gathered_rest))
            rest_lp = gathered_rest.sum(dim=1)

        total_lp = first_lp + rest_lp
        if normalize_by_length:
            total_lp = total_lp / lengths.to(dtype=total_lp.dtype)
        all_scores.append(total_lp)

    return torch.cat(all_scores, dim=0)


@torch.no_grad()
def logprob_of_continuation_candidates_shared_prompt(
    *,
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    continuation_id_list: Sequence[torch.Tensor],
    normalize_by_length: bool = False,
    batch_size: int = 1,
    pad_token_id: int = 0,
    use_prefix_cache: bool = False,
) -> torch.Tensor:
    """
    Score many continuations for one prompt, optionally reusing a prompt KV cache.
    """
    if prompt_ids.ndim != 2 or prompt_ids.size(0) != 1:
        raise ValueError("prompt_ids must have shape (1, P)")
    if not continuation_id_list:
        return torch.empty((0,), device=prompt_ids.device, dtype=_LOGPROBS_DTYPE)
    batch_size = max(1, int(batch_size))
    device = prompt_ids.device
    prompt_len = int(prompt_ids.size(1))

    normalized_conts: list[torch.Tensor] = []
    for cont in continuation_id_list:
        if cont.ndim == 2:
            if cont.size(0) != 1:
                raise ValueError("each continuation must have batch size 1 when rank-2")
            cont = cont.squeeze(0)
        if cont.ndim != 1:
            raise ValueError("each continuation must be rank-1 or rank-2 with batch size 1")
        if cont.numel() < 1:
            raise ValueError("each continuation must have length >= 1")
        normalized_conts.append(cont.to(device=device, dtype=torch.long))

    def _full_forward() -> torch.Tensor:
        all_scores: list[torch.Tensor] = []
        for chunk in batched(normalized_conts, batch_size):
            B = int(len(chunk))
            lengths = torch.tensor([int(c.numel()) for c in chunk], device=device, dtype=torch.long)
            max_len = int(max(int(c.numel()) for c in chunk))
            cont_pad = torch.full((B, max_len), int(pad_token_id), device=device, dtype=torch.long)
            cont_mask = torch.zeros((B, max_len), device=device, dtype=torch.bool)
            for i, c in enumerate(chunk):
                L = int(c.numel())
                cont_pad[i, :L] = c
                cont_mask[i, :L] = True

            full_ids = torch.cat([prompt_ids.expand(B, -1), cont_pad], dim=1)
            attn_mask = torch.cat(
                [
                    torch.ones((B, prompt_len), device=device, dtype=torch.long),
                    cont_mask.to(dtype=torch.long),
                ],
                dim=1,
            )

            out = model(input_ids=full_ids, attention_mask=attn_mask, use_cache=False, return_dict=True)
            logits = out.logits  # type: ignore[attr-defined]
            logits_slice = logits[:, prompt_len - 1 : prompt_len + max_len - 1, :].to(dtype=_LOGPROBS_DTYPE)
            log_probs = F.log_softmax(logits_slice, dim=-1)
            gathered = log_probs.gather(2, cont_pad.unsqueeze(-1)).squeeze(-1)
            gathered = _check_finite(
                gathered,
                mask=cont_mask,
                context=(
                    "Non-finite log-probability detected in continuation scoring "
                    f"(device={gathered.device}, logits_dtype={logits.dtype}, logprobs_dtype={gathered.dtype})."
                ),
            )
            gathered = torch.where(cont_mask, gathered, torch.zeros_like(gathered))
            lp = gathered.sum(dim=1)
            if normalize_by_length:
                lp = lp / lengths.to(dtype=lp.dtype)
            all_scores.append(lp)
        return torch.cat(all_scores, dim=0)

    if not bool(use_prefix_cache):
        return _full_forward()

    try:
        prefill_out = model(
            input_ids=prompt_ids,
            attention_mask=torch.ones_like(prompt_ids),
            use_cache=True,
            return_dict=True,
        )
        prefill_logits_last = prefill_out.logits[:, -1, :]  # type: ignore[attr-defined]
        prefill_past = prefill_out.past_key_values  # type: ignore[attr-defined]
        return logprob_of_continuation_candidates_with_prefill(
            model=model,
            prompt_ids=prompt_ids,
            continuation_id_list=normalized_conts,
            prefill_logits_last=prefill_logits_last,
            prefill_past_key_values=prefill_past,
            normalize_by_length=normalize_by_length,
            batch_size=batch_size,
            pad_token_id=int(pad_token_id),
        )
    except Exception:
        # Fallback preserves correctness when a given architecture/cache implementation
        # does not support this prefill path.
        return _full_forward()


def batched(iterable: Sequence[Any], batch_size: int) -> Iterator[Sequence[Any]]:
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]
