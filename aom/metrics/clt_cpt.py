from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aom.data.schemas import DisambPair
from aom.interventions.activation_patching import PatchSpanSite, get_block_outputs
from aom.interventions.clt_adapter import CLTInputTransform, CLTPatchConfig, CLTProtocol
from aom.interventions.clt_patch import (
    IdentityLatentPolicy,
    LatentPolicy,
    ReplaceLatentDimsAtIndicesPolicy,
    ReplaceLatentsAtIndicesPolicy,
    forward_with_clt_latent_patching_span,
)
from aom.metrics.disamb import LabelScores, _encode_prompt, _logmeanexp, _margin, score_labels_next_continuations
from aom.stats import bootstrap_ci_pair_cluster, bootstrap_ratio_ci_pair_cluster
from aom.token_spans import token_span_for_substring
from aom.utils import bootstrap_ci, get_logprob_computation_config, get_scoring_performance_config


def _infer_clt_device_dtype(clt: CLTProtocol) -> tuple[torch.device, torch.dtype]:
    if isinstance(clt, torch.nn.Module):
        p = next(clt.parameters(), None)
        if p is not None:
            return p.device, p.dtype
    W_dec = getattr(clt, "W_dec", None)
    if isinstance(W_dec, torch.Tensor):
        return W_dec.device, W_dec.dtype
    return torch.device("cpu"), torch.float32


def _to_cpu_f64(x: torch.Tensor) -> torch.Tensor:
    # On MPS, cast after moving to CPU to avoid unsupported float64-on-MPS conversion.
    return x.to(device="cpu").to(dtype=torch.float64)


def _assert_prompt_span_alignment(
    prompt_ids: torch.Tensor,
    span: Sequence[int],
    span_token_ids: Sequence[int],
    *,
    side_name: str,
) -> None:
    if prompt_ids.ndim != 2 or int(prompt_ids.size(0)) != 1:
        raise ValueError("prompt_ids must have shape (1, seq)")
    flat_ids = [int(x) for x in prompt_ids[0].tolist()]
    try:
        extracted = [int(flat_ids[int(i)]) for i in span]
    except IndexError as e:
        raise RuntimeError(f"{side_name} span contains out-of-range token index: span={list(span)}") from e
    expected = [int(x) for x in span_token_ids]
    if extracted != expected:
        raise RuntimeError(
            f"{side_name} span token IDs mismatch prompt encoding: "
            f"extracted={extracted}, expected={expected}, span={list(span)}"
        )


@dataclass(frozen=True)
class CLTLatentDeltaStats:
    layer: int
    d_latent: int
    n_directions_total: int
    n_directions_used: int
    n_directions_skipped_misaligned: int
    mean_delta: torch.Tensor  # (d_latent,) float64 CPU
    mean_abs_delta: torch.Tensor  # (d_latent,) float64 CPU
    std_delta: torch.Tensor  # (d_latent,) float64 CPU
    t_stat: torch.Tensor  # (d_latent,) float64 CPU
    frac_positive: torch.Tensor  # (d_latent,) float64 CPU


@torch.no_grad()
def compute_clt_latent_delta_stats(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    items: Sequence[DisambPair],
    device: torch.device,
    *,
    layer: int,
    clt: CLTProtocol,
    transform: CLTInputTransform,
    require_token_id_match: bool = True,
    span_aggregation: Literal["mean", "sum"] = "mean",
) -> CLTLatentDeltaStats:
    """
    Compute per-latent statistics for context-swap deltas Δz = z_donor - z_receiver at the target span.

    Returns CPU float64 tensors suitable for ranking features by |t_stat| or mean_abs_delta.
    """

    if span_aggregation not in ("mean", "sum"):
        raise ValueError(f"Unknown span_aggregation={span_aggregation!r}")

    d_latent = int(getattr(clt, "d_latent"))
    if d_latent <= 0:
        raise ValueError("clt.d_latent must be positive")

    sum_delta = torch.zeros(d_latent, dtype=torch.float64)
    sum_sq_delta = torch.zeros(d_latent, dtype=torch.float64)
    sum_abs_delta = torch.zeros(d_latent, dtype=torch.float64)
    count_pos = torch.zeros(d_latent, dtype=torch.float64)

    n_total_directions = 0
    n_used = 0
    n_skipped_misaligned = 0

    for it in items:
        # Count both directions regardless of skip (matches CPT counters).
        n_total_directions += 2

        span_a, tok_ids_a = token_span_for_substring(tokenizer, it.a.prompt, it.target, it.target_occurrence)
        span_b, tok_ids_b = token_span_for_substring(tokenizer, it.b.prompt, it.target, it.target_occurrence)

        aligned = (len(span_a) == len(span_b)) and (not require_token_id_match or tok_ids_a == tok_ids_b)
        if not aligned:
            n_skipped_misaligned += 2
            continue

        ids_a = _encode_prompt(tokenizer, it.a.prompt, device=device)
        ids_b = _encode_prompt(tokenizer, it.b.prompt, device=device)
        _assert_prompt_span_alignment(ids_a, span_a, tok_ids_a, side_name="donor_a")
        _assert_prompt_span_alignment(ids_b, span_b, tok_ids_b, side_name="donor_b")
        out_a = get_block_outputs(model, ids_a, layers=[int(layer)])
        out_b = get_block_outputs(model, ids_b, layers=[int(layer)])

        clt_device, clt_dtype = _infer_clt_device_dtype(clt)
        slice_a = out_a[int(layer)][0, span_a, :].detach().unsqueeze(0).to(device=clt_device, dtype=clt_dtype)
        slice_b = out_b[int(layer)][0, span_b, :].detach().unsqueeze(0).to(device=clt_device, dtype=clt_dtype)

        z_a = clt.encode(transform.forward(slice_a))
        z_b = clt.encode(transform.forward(slice_b))
        if z_a.ndim != 3 or z_b.ndim != 3 or int(z_a.size(-1)) != d_latent or int(z_b.size(-1)) != d_latent:
            raise RuntimeError("CLT encode returned unexpected latent shape")

        for donor_z, recv_z in ((z_a, z_b), (z_b, z_a)):
            delta = donor_z - recv_z  # (1, span_len, d_latent)
            if span_aggregation == "sum":
                delta_vec = delta.sum(dim=1).squeeze(0)
            else:
                delta_vec = delta.mean(dim=1).squeeze(0)
            delta_cpu = _to_cpu_f64(delta_vec)
            sum_delta += delta_cpu
            sum_sq_delta += delta_cpu * delta_cpu
            sum_abs_delta += delta_cpu.abs()
            count_pos += (delta_cpu > 0).to(dtype=torch.float64)
            n_used += 1

    if n_used <= 0:
        mean_delta = torch.zeros(d_latent, dtype=torch.float64)
        mean_abs_delta = torch.zeros(d_latent, dtype=torch.float64)
        std_delta = torch.zeros(d_latent, dtype=torch.float64)
        t_stat = torch.zeros(d_latent, dtype=torch.float64)
        frac_positive = torch.zeros(d_latent, dtype=torch.float64)
        return CLTLatentDeltaStats(
            layer=int(layer),
            d_latent=d_latent,
            n_directions_total=int(n_total_directions),
            n_directions_used=int(n_used),
            n_directions_skipped_misaligned=int(n_skipped_misaligned),
            mean_delta=mean_delta,
            mean_abs_delta=mean_abs_delta,
            std_delta=std_delta,
            t_stat=t_stat,
            frac_positive=frac_positive,
        )

    mean_delta = sum_delta / float(n_used)
    mean_abs_delta = sum_abs_delta / float(n_used)
    frac_positive = count_pos / float(n_used)

    if n_used > 1:
        var = (sum_sq_delta - (sum_delta * sum_delta) / float(n_used)) / float(n_used - 1)
        var = torch.clamp(var, min=0.0)
        std_delta = torch.sqrt(var)
        denom = std_delta / (float(n_used) ** 0.5)
        t_stat = torch.where(denom > 0, mean_delta / denom, torch.zeros_like(mean_delta))
    else:
        std_delta = torch.zeros_like(mean_delta)
        t_stat = torch.zeros_like(mean_delta)

    return CLTLatentDeltaStats(
        layer=int(layer),
        d_latent=d_latent,
        n_directions_total=int(n_total_directions),
        n_directions_used=int(n_used),
        n_directions_skipped_misaligned=int(n_skipped_misaligned),
        mean_delta=mean_delta,
        mean_abs_delta=mean_abs_delta,
        std_delta=std_delta,
        t_stat=t_stat,
        frac_positive=frac_positive,
    )


def topk_clt_latents(
    stats: CLTLatentDeltaStats,
    *,
    k: int,
    score: Literal["abs_t_stat", "mean_abs_delta", "abs_mean_delta"] = "abs_t_stat",
) -> List[int]:
    if k < 0:
        raise ValueError("k must be >= 0")
    if k == 0:
        return []
    if int(stats.d_latent) <= 0:
        raise ValueError("stats.d_latent must be positive")
    k = min(int(k), int(stats.d_latent))

    if score == "abs_t_stat":
        vals = stats.t_stat.abs()
    elif score == "mean_abs_delta":
        vals = stats.mean_abs_delta
    elif score == "abs_mean_delta":
        vals = stats.mean_delta.abs()
    else:
        raise ValueError(f"Unknown score={score!r}")

    top = torch.topk(vals, k=k, largest=True, sorted=True).indices
    return [int(i) for i in top.tolist()]


def _random_latent_dims(
    *,
    d_latent: int,
    k: int,
    seed: int,
    excluded: Optional[Sequence[int]] = None,
) -> List[int]:
    if d_latent <= 0:
        raise ValueError("d_latent must be positive")
    if k < 0:
        raise ValueError("k must be >= 0")
    if k == 0:
        return []
    k = min(int(k), int(d_latent))

    excluded_set = {int(i) for i in (excluded or [])}
    pool = [i for i in range(int(d_latent)) if i not in excluded_set]
    if len(pool) < k:
        pool = list(range(int(d_latent)))

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    perm = torch.randperm(len(pool), generator=g).tolist()
    return [int(pool[i]) for i in perm[:k]]


@torch.no_grad()
def compute_clt_cpt_context_swap_patching_topk(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    items: Sequence[DisambPair],
    device: torch.device,
    *,
    layer: int,
    clt: CLTProtocol,
    transform: CLTInputTransform,
    latent_dims: Sequence[int],
    config: Optional[CLTPatchConfig] = None,
    normalize_by_length: bool = True,
    require_token_id_match: bool = True,
    include_random_k: bool = True,
    random_k_seed: int = 0,
    recovery_eps: float = 1e-6,
    ci: float = 0.95,
    bootstrap_n: int = 1000,
    bootstrap_seed: int = 42,
) -> Dict[str, Any]:
    """
    CPT-style context swap patching at a fixed layer, comparing:
      - full latent replacement (all dims)
      - selective replacement of `latent_dims` (top-k)
      - matched random-k dims (optional)
      - sham replacement (receiver latents injected back)
      - identity policy (no-op) [abs effect only]

    Uses the same behavioral target as `compute_clt_cpt_context_swap_patching`:
      effect = margin_patched(donor_expected) - margin_base(donor_expected)
    """

    d_latent = int(getattr(clt, "d_latent"))
    if d_latent <= 0:
        raise ValueError("clt.d_latent must be positive")
    latent_dims_i = [int(i) for i in latent_dims]
    for i in latent_dims_i:
        if int(i) < 0 or int(i) >= d_latent:
            raise ValueError(f"latent dim {int(i)} out of range [0, {d_latent})")

    random_dims: List[int] = []
    if include_random_k:
        random_dims = _random_latent_dims(
            d_latent=d_latent,
            k=len(latent_dims_i),
            seed=int(random_k_seed),
            excluded=latent_dims_i,
        )

    effects_full: List[float] = []
    effects_topk: List[float] = []
    effects_random: List[float] = []
    effects_sham: List[float] = []
    abs_identity_effects: List[float] = []

    flips_full: List[float] = []
    flips_topk: List[float] = []
    flips_random: List[float] = []

    n_total_directions = 0
    n_skipped_misaligned = 0

    for it in items:
        for donor, recv in ((it.a, it.b), (it.b, it.a)):
            n_total_directions += 1
            donor_expected = donor.expected_label

            donor_span, donor_token_ids = token_span_for_substring(
                tokenizer, donor.prompt, it.target, it.target_occurrence
            )
            recv_span, recv_token_ids = token_span_for_substring(
                tokenizer, recv.prompt, it.target, it.target_occurrence
            )
            if len(donor_span) != len(recv_span) or (require_token_id_match and donor_token_ids != recv_token_ids):
                n_skipped_misaligned += 1
                continue

            base_scores = score_labels_next_continuations(
                model, tokenizer, recv.prompt, it.choices, device, normalize_by_length=normalize_by_length
            )
            base_pred = base_scores.argmax_label()
            base_margin = _margin(base_scores, expected=donor_expected)

            donor_ids = _encode_prompt(tokenizer, donor.prompt, device=device)
            recv_ids = _encode_prompt(tokenizer, recv.prompt, device=device)
            _assert_prompt_span_alignment(donor_ids, donor_span, donor_token_ids, side_name="donor")
            _assert_prompt_span_alignment(recv_ids, recv_span, recv_token_ids, side_name="receiver")
            donor_out = get_block_outputs(model, donor_ids, layers=[int(layer)])
            recv_out = get_block_outputs(model, recv_ids, layers=[int(layer)])

            clt_device, clt_dtype = _infer_clt_device_dtype(clt)
            donor_slice = (
                donor_out[int(layer)][0, donor_span, :].detach().unsqueeze(0).to(device=clt_device, dtype=clt_dtype)
            )
            recv_slice = (
                recv_out[int(layer)][0, recv_span, :].detach().unsqueeze(0).to(device=clt_device, dtype=clt_dtype)
            )

            donor_latents = clt.encode(transform.forward(donor_slice))
            recv_latents = clt.encode(transform.forward(recv_slice))

            full_policy = ReplaceLatentsAtIndicesPolicy(
                token_indices=list(recv_span),
                replacement_latents=donor_latents,
            )
            topk_policy = ReplaceLatentDimsAtIndicesPolicy(
                token_indices=list(recv_span),
                replacement_latents=donor_latents,
                latent_dims=latent_dims_i,
            )
            sham_policy = ReplaceLatentsAtIndicesPolicy(
                token_indices=list(recv_span),
                replacement_latents=recv_latents,
            )
            identity_policy = IdentityLatentPolicy()

            full_scores = score_labels_next_continuations_clt_patched(
                model,
                tokenizer,
                recv.prompt,
                it.choices,
                device,
                layer=int(layer),
                token_indices=list(recv_span),
                clt=clt,
                transform=transform,
                policy=full_policy,
                config=config,
                normalize_by_length=normalize_by_length,
            )
            topk_scores = score_labels_next_continuations_clt_patched(
                model,
                tokenizer,
                recv.prompt,
                it.choices,
                device,
                layer=int(layer),
                token_indices=list(recv_span),
                clt=clt,
                transform=transform,
                policy=topk_policy,
                config=config,
                normalize_by_length=normalize_by_length,
            )
            sham_scores = score_labels_next_continuations_clt_patched(
                model,
                tokenizer,
                recv.prompt,
                it.choices,
                device,
                layer=int(layer),
                token_indices=list(recv_span),
                clt=clt,
                transform=transform,
                policy=sham_policy,
                config=config,
                normalize_by_length=normalize_by_length,
            )
            identity_scores = score_labels_next_continuations_clt_patched(
                model,
                tokenizer,
                recv.prompt,
                it.choices,
                device,
                layer=int(layer),
                token_indices=list(recv_span),
                clt=clt,
                transform=transform,
                policy=identity_policy,
                config=config,
                normalize_by_length=normalize_by_length,
            )

            full_margin = _margin(full_scores, expected=donor_expected)
            topk_margin = _margin(topk_scores, expected=donor_expected)
            sham_margin = _margin(sham_scores, expected=donor_expected)
            identity_margin = _margin(identity_scores, expected=donor_expected)

            effects_full.append(float(full_margin - base_margin))
            effects_topk.append(float(topk_margin - base_margin))
            effects_sham.append(float(sham_margin - base_margin))
            abs_identity_effects.append(float(abs(identity_margin - base_margin)))

            full_pred = full_scores.argmax_label()
            topk_pred = topk_scores.argmax_label()
            flips_full.append(float((base_pred != donor_expected) and (full_pred == donor_expected)))
            flips_topk.append(float((base_pred != donor_expected) and (topk_pred == donor_expected)))

            if include_random_k:
                random_policy = ReplaceLatentDimsAtIndicesPolicy(
                    token_indices=list(recv_span),
                    replacement_latents=donor_latents,
                    latent_dims=random_dims,
                )
                random_scores = score_labels_next_continuations_clt_patched(
                    model,
                    tokenizer,
                    recv.prompt,
                    it.choices,
                    device,
                    layer=int(layer),
                    token_indices=list(recv_span),
                    clt=clt,
                    transform=transform,
                    policy=random_policy,
                    config=config,
                    normalize_by_length=normalize_by_length,
                )
                random_margin = _margin(random_scores, expected=donor_expected)
                effects_random.append(float(random_margin - base_margin))
                random_pred = random_scores.argmax_label()
                flips_random.append(float((base_pred != donor_expected) and (random_pred == donor_expected)))

    mean_full, lo_full, hi_full = bootstrap_ci(effects_full, n_bootstrap=bootstrap_n, ci=ci, seed=bootstrap_seed)
    mean_topk, lo_topk, hi_topk = bootstrap_ci(effects_topk, n_bootstrap=bootstrap_n, ci=ci, seed=bootstrap_seed)
    mean_sham, lo_sham, hi_sham = bootstrap_ci(effects_sham, n_bootstrap=bootstrap_n, ci=ci, seed=bootstrap_seed)
    mean_id_abs, lo_id_abs, hi_id_abs = bootstrap_ci(
        abs_identity_effects, n_bootstrap=bootstrap_n, ci=ci, seed=bootstrap_seed
    )

    mean_rand = float("nan")
    lo_rand = float("nan")
    hi_rand = float("nan")
    if include_random_k:
        mean_rand, lo_rand, hi_rand = bootstrap_ci(effects_random, n_bootstrap=bootstrap_n, ci=ci, seed=bootstrap_seed)

    flip_full, flip_full_lo, flip_full_hi = bootstrap_ci(flips_full, n_bootstrap=bootstrap_n, ci=ci, seed=bootstrap_seed)
    flip_topk, flip_topk_lo, flip_topk_hi = bootstrap_ci(flips_topk, n_bootstrap=bootstrap_n, ci=ci, seed=bootstrap_seed)
    flip_rand, flip_rand_lo, flip_rand_hi = float("nan"), float("nan"), float("nan")
    if include_random_k:
        flip_rand, flip_rand_lo, flip_rand_hi = bootstrap_ci(
            flips_random, n_bootstrap=bootstrap_n, ci=ci, seed=bootstrap_seed
        )

    recovery = float("nan")
    recovery_rand = float("nan")
    if abs(float(mean_full)) >= float(recovery_eps):
        recovery = float(mean_topk / mean_full)
        if include_random_k:
            recovery_rand = float(mean_rand / mean_full)

    return {
        "layer": int(layer),
        "k": int(len(latent_dims_i)),
        "mean_effect_full": float(mean_full),
        "mean_effect_full_ci_low": float(lo_full),
        "mean_effect_full_ci_high": float(hi_full),
        "mean_effect_topk": float(mean_topk),
        "mean_effect_topk_ci_low": float(lo_topk),
        "mean_effect_topk_ci_high": float(hi_topk),
        "mean_effect_randomk": float(mean_rand),
        "mean_effect_randomk_ci_low": float(lo_rand),
        "mean_effect_randomk_ci_high": float(hi_rand),
        "mean_effect_sham": float(mean_sham),
        "mean_effect_sham_ci_low": float(lo_sham),
        "mean_effect_sham_ci_high": float(hi_sham),
        "mean_identity_max_abs_effect": float(mean_id_abs),
        "mean_identity_max_abs_effect_ci_low": float(lo_id_abs),
        "mean_identity_max_abs_effect_ci_high": float(hi_id_abs),
        "flip_rate_full": float(flip_full),
        "flip_rate_full_ci_low": float(flip_full_lo),
        "flip_rate_full_ci_high": float(flip_full_hi),
        "flip_rate_topk": float(flip_topk),
        "flip_rate_topk_ci_low": float(flip_topk_lo),
        "flip_rate_topk_ci_high": float(flip_topk_hi),
        "flip_rate_randomk": float(flip_rand),
        "flip_rate_randomk_ci_low": float(flip_rand_lo),
        "flip_rate_randomk_ci_high": float(flip_rand_hi),
        "recovery_ratio_topk": float(recovery),
        "recovery_ratio_randomk": float(recovery_rand),
        "recovery_eps": float(recovery_eps),
        "n_directions_total": int(n_total_directions),
        "n_directions_patched": int(len(effects_full)),
        "n_directions_skipped_misaligned": int(n_skipped_misaligned),
    }


@torch.no_grad()
def score_labels_next_continuations_clt_patched(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    choices: Mapping[str, List[str]],
    device: torch.device,
    *,
    layer: int,
    token_indices: List[int],
    clt: CLTProtocol,
    transform: CLTInputTransform,
    policy: LatentPolicy,
    config: Optional[CLTPatchConfig] = None,
    normalize_by_length: bool = True,
) -> LabelScores:
    prompt_ids = _encode_prompt(tokenizer, prompt, device=device)
    logprobs_dtype, strict_finite = get_logprob_computation_config()
    score_batch_size, _use_prefix_cache = get_scoring_performance_config()
    pad_token_id = (
        int(tokenizer.pad_token_id)
        if tokenizer.pad_token_id is not None
        else (int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else 0)
    )

    # Flatten all continuation candidates so we can score them in a single batched
    # patched forward pass (or a small number of chunked passes).
    flat_conts: List[torch.Tensor] = []
    spans: Dict[str, tuple[int, int]] = {}
    for label, continuations in choices.items():
        if len(continuations) < 1:
            raise ValueError(f"Empty continuation list for label={label}")
        start = len(flat_conts)
        for cont in continuations:
            cont_ids = tokenizer(str(cont), return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
            if cont_ids.ndim != 2 or int(cont_ids.size(0)) != 1:
                raise ValueError("expected continuation token IDs of shape (1, seq)")
            cont_1d = cont_ids.squeeze(0)
            if int(cont_1d.numel()) < 1:
                raise ValueError(f"Empty continuation encoding for label={label} cont={cont!r}")
            flat_conts.append(cont_1d)
        spans[str(label)] = (start, len(flat_conts))

    if not flat_conts:
        # Should be impossible given checks above, but keep behavior explicit.
        return LabelScores(by_label={str(k): float("-inf") for k in choices.keys()})

    P = int(prompt_ids.size(1))
    chunk_size = int(max(1, score_batch_size))
    flat_scores: List[torch.Tensor] = []
    for start in range(0, len(flat_conts), chunk_size):
        batch_conts = flat_conts[start : start + chunk_size]
        B = int(len(batch_conts))
        lengths = torch.tensor([int(c.numel()) for c in batch_conts], device=device, dtype=torch.long)
        max_len = int(lengths.max().item())

        cont_pad = torch.full((B, max_len), int(pad_token_id), device=device, dtype=torch.long)
        cont_mask = torch.zeros((B, max_len), device=device, dtype=torch.bool)
        for i, c in enumerate(batch_conts):
            L = int(c.numel())
            cont_pad[i, :L] = c
            cont_mask[i, :L] = True

        full_ids = torch.cat([prompt_ids.expand(B, -1), cont_pad], dim=1)
        attn_mask = torch.cat(
            [
                torch.ones((B, P), device=device, dtype=torch.long),
                cont_mask.to(dtype=torch.long),
            ],
            dim=1,
        )

        logits = forward_with_clt_latent_patching_span(
            model,
            input_ids=full_ids,
            attention_mask=attn_mask,
            site=PatchSpanSite(layer=int(layer), token_indices=tuple(int(x) for x in token_indices)),
            clt=clt,
            policy=policy,
            transform=transform,
            config=config,
        )

        logits_slice = logits[:, P - 1 : P + max_len - 1, :].to(dtype=logprobs_dtype)
        log_probs = torch.log_softmax(logits_slice, dim=-1)
        gathered = log_probs.gather(2, cont_pad.unsqueeze(-1)).squeeze(-1)  # (B, max_len)
        if not torch.isfinite(gathered[cont_mask]).all():
            if strict_finite:
                raise FloatingPointError("Non-finite log-probability detected in CLT patched scoring.")
            gathered = torch.where(torch.isfinite(gathered), gathered, torch.full_like(gathered, -1e9))
        gathered = torch.where(cont_mask, gathered, torch.zeros_like(gathered))
        lp = gathered.sum(dim=1)
        if normalize_by_length:
            lp = lp / lengths.to(dtype=lp.dtype)
        flat_scores.append(lp)

    flat_logps = torch.cat(flat_scores, dim=0)
    if int(flat_logps.numel()) != len(flat_conts):
        raise RuntimeError("CLT patched scoring produced a size mismatch against flattened continuations")

    scores: Dict[str, float] = {}
    for label, (lo, hi) in spans.items():
        vals = [float(x) for x in flat_logps[int(lo) : int(hi)].tolist()]
        scores[str(label)] = _logmeanexp(vals)
    return LabelScores(by_label=scores)


@torch.no_grad()
def compute_clt_cpt_context_swap_patching(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    items: List[DisambPair],
    device: torch.device,
    *,
    clt_by_layer: Mapping[int, CLTProtocol],
    transform_by_layer: Mapping[int, CLTInputTransform],
    layers: Optional[List[int]] = None,
    config: Optional[CLTPatchConfig] = None,
    normalize_by_length: bool = True,
    require_token_id_match: bool = True,
    ci: float = 0.95,
    bootstrap_n: int = 1000,
    bootstrap_seed: int = 42,
) -> Dict[str, Any]:
    if layers is None:
        layers = sorted(int(l) for l in clt_by_layer.keys())
    layers = [int(l) for l in layers]
    for l in layers:
        if l not in clt_by_layer:
            raise ValueError(f"Missing CLT for layer {l}")
        if l not in transform_by_layer:
            raise ValueError(f"Missing transform for layer {l}")

    per_layer_sum = {int(l): 0.0 for l in layers}
    per_layer_n = {int(l): 0 for l in layers}
    sham_per_layer_sum = {int(l): 0.0 for l in layers}
    sham_per_layer_n = {int(l): 0 for l in layers}
    identity_per_layer_sum = {int(l): 0.0 for l in layers}
    identity_abs_per_layer_sum = {int(l): 0.0 for l in layers}
    identity_per_layer_n = {int(l): 0 for l in layers}

    max_effects: List[float] = []
    max_layers: List[int] = []
    max_effects_norm: List[float] = []
    max_flips: List[float] = []
    sham_max_effects: List[float] = []
    identity_max_abs_effects: List[float] = []

    n_total_directions = 0
    n_skipped_misaligned = 0

    for it in items:
        for donor, recv in ((it.a, it.b), (it.b, it.a)):
            n_total_directions += 1
            donor_expected = donor.expected_label
            donor_span, donor_token_ids = token_span_for_substring(
                tokenizer, donor.prompt, it.target, it.target_occurrence
            )
            recv_span, recv_token_ids = token_span_for_substring(tokenizer, recv.prompt, it.target, it.target_occurrence)

            if len(donor_span) != len(recv_span) or (require_token_id_match and donor_token_ids != recv_token_ids):
                n_skipped_misaligned += 1
                continue

            base_scores = score_labels_next_continuations(
                model, tokenizer, recv.prompt, it.choices, device, normalize_by_length=normalize_by_length
            )
            base_pred = base_scores.argmax_label()
            base_margin_for_expected = _margin(base_scores, expected=donor_expected)

            donor_ids = _encode_prompt(tokenizer, donor.prompt, device=device)
            recv_ids = _encode_prompt(tokenizer, recv.prompt, device=device)
            _assert_prompt_span_alignment(donor_ids, donor_span, donor_token_ids, side_name="donor")
            _assert_prompt_span_alignment(recv_ids, recv_span, recv_token_ids, side_name="receiver")
            donor_out = get_block_outputs(model, donor_ids, layers=layers)
            recv_out = get_block_outputs(model, recv_ids, layers=layers)

            best_for_direction = None
            best_layer_for_direction = None
            best_sham_for_direction = None
            best_flip_for_direction = None
            best_norm_effect_for_direction = None
            best_identity_abs_for_direction = None

            for layer in layers:
                clt = clt_by_layer[int(layer)]
                transform = transform_by_layer[int(layer)]
                clt_device, clt_dtype = _infer_clt_device_dtype(clt)

                donor_slice = (
                    donor_out[int(layer)][0, donor_span, :]
                    .detach()
                    .unsqueeze(0)
                    .to(device=clt_device, dtype=clt_dtype)
                )
                recv_slice = (
                    recv_out[int(layer)][0, recv_span, :]
                    .detach()
                    .unsqueeze(0)
                    .to(device=clt_device, dtype=clt_dtype)
                )

                donor_latents = clt.encode(transform.forward(donor_slice))
                recv_latents = clt.encode(transform.forward(recv_slice))

                policy = ReplaceLatentsAtIndicesPolicy(token_indices=list(recv_span), replacement_latents=donor_latents)
                sham_policy = ReplaceLatentsAtIndicesPolicy(token_indices=list(recv_span), replacement_latents=recv_latents)
                identity_policy = IdentityLatentPolicy()

                patched_scores = score_labels_next_continuations_clt_patched(
                    model,
                    tokenizer,
                    recv.prompt,
                    it.choices,
                    device,
                    layer=int(layer),
                    token_indices=list(recv_span),
                    clt=clt,
                    transform=transform,
                    policy=policy,
                    config=config,
                    normalize_by_length=normalize_by_length,
                )
                sham_scores = score_labels_next_continuations_clt_patched(
                    model,
                    tokenizer,
                    recv.prompt,
                    it.choices,
                    device,
                    layer=int(layer),
                    token_indices=list(recv_span),
                    clt=clt,
                    transform=transform,
                    policy=sham_policy,
                    config=config,
                    normalize_by_length=normalize_by_length,
                )
                identity_scores = score_labels_next_continuations_clt_patched(
                    model,
                    tokenizer,
                    recv.prompt,
                    it.choices,
                    device,
                    layer=int(layer),
                    token_indices=list(recv_span),
                    clt=clt,
                    transform=transform,
                    policy=identity_policy,
                    config=config,
                    normalize_by_length=normalize_by_length,
                )

                base_margin = base_margin_for_expected
                patched_margin = _margin(patched_scores, expected=donor_expected)
                effect = float(patched_margin - base_margin)
                norm_effect = float(effect / (abs(base_margin) + 1e-8))

                patched_pred = patched_scores.argmax_label()
                flipped = float((base_pred != donor_expected) and (patched_pred == donor_expected))

                sham_margin = _margin(sham_scores, expected=donor_expected)
                sham_effect = float(sham_margin - base_margin)
                identity_margin = _margin(identity_scores, expected=donor_expected)
                identity_effect = float(identity_margin - base_margin)

                per_layer_sum[int(layer)] += effect
                per_layer_n[int(layer)] += 1
                sham_per_layer_sum[int(layer)] += sham_effect
                sham_per_layer_n[int(layer)] += 1
                identity_per_layer_sum[int(layer)] += identity_effect
                identity_abs_per_layer_sum[int(layer)] += abs(identity_effect)
                identity_per_layer_n[int(layer)] += 1

                if best_for_direction is None or effect > best_for_direction:
                    best_for_direction = effect
                    best_layer_for_direction = int(layer)
                    best_flip_for_direction = flipped
                    best_norm_effect_for_direction = norm_effect
                if best_sham_for_direction is None or sham_effect > best_sham_for_direction:
                    best_sham_for_direction = sham_effect
                abs_identity_effect = abs(identity_effect)
                if best_identity_abs_for_direction is None or abs_identity_effect > best_identity_abs_for_direction:
                    best_identity_abs_for_direction = abs_identity_effect

            if best_for_direction is not None and best_layer_for_direction is not None:
                max_effects.append(float(best_for_direction))
                max_layers.append(int(best_layer_for_direction))
                if best_flip_for_direction is not None:
                    max_flips.append(float(best_flip_for_direction))
                if best_norm_effect_for_direction is not None:
                    max_effects_norm.append(float(best_norm_effect_for_direction))
            if best_sham_for_direction is not None:
                sham_max_effects.append(float(best_sham_for_direction))
            if best_identity_abs_for_direction is not None:
                identity_max_abs_effects.append(float(best_identity_abs_for_direction))

    mean_max_effect, mean_max_effect_lo, mean_max_effect_hi = bootstrap_ci(
        max_effects, n_bootstrap=bootstrap_n, ci=ci, seed=bootstrap_seed
    )
    flip_rate, flip_lo, flip_hi = bootstrap_ci(max_flips, n_bootstrap=bootstrap_n, ci=ci, seed=bootstrap_seed)
    mean_norm, mean_norm_lo, mean_norm_hi = bootstrap_ci(
        max_effects_norm, n_bootstrap=bootstrap_n, ci=ci, seed=bootstrap_seed
    )
    mean_sham, mean_sham_lo, mean_sham_hi = bootstrap_ci(
        sham_max_effects, n_bootstrap=bootstrap_n, ci=ci, seed=bootstrap_seed
    )
    mean_identity_max_abs, mean_identity_max_abs_lo, mean_identity_max_abs_hi = bootstrap_ci(
        identity_max_abs_effects, n_bootstrap=bootstrap_n, ci=ci, seed=bootstrap_seed
    )

    return {
        "mean_max_effect": mean_max_effect,
        "mean_max_effect_ci_low": mean_max_effect_lo,
        "mean_max_effect_ci_high": mean_max_effect_hi,
        "mean_argmax_layer": float(sum(max_layers) / max(1, len(max_layers))) if max_layers else 0.0,
        "flip_rate_at_best_layer": flip_rate,
        "flip_rate_ci_low": flip_lo,
        "flip_rate_ci_high": flip_hi,
        "mean_norm_max_effect": mean_norm,
        "mean_norm_max_effect_ci_low": mean_norm_lo,
        "mean_norm_max_effect_ci_high": mean_norm_hi,
        "mean_sham_max_effect": mean_sham,
        "mean_sham_max_effect_ci_low": mean_sham_lo,
        "mean_sham_max_effect_ci_high": mean_sham_hi,
        "mean_identity_max_abs_effect": mean_identity_max_abs,
        "mean_identity_max_abs_effect_ci_low": mean_identity_max_abs_lo,
        "mean_identity_max_abs_effect_ci_high": mean_identity_max_abs_hi,
        "n_directions_total": int(n_total_directions),
        "n_directions_patched": int(len(max_effects)),
        "n_directions_skipped_misaligned": int(n_skipped_misaligned),
        **{f"effect_layer_{l}": per_layer_sum[l] / max(1, per_layer_n[l]) for l in layers},
        **{f"sham_effect_layer_{l}": sham_per_layer_sum[l] / max(1, sham_per_layer_n[l]) for l in layers},
        **{f"identity_effect_layer_{l}": identity_per_layer_sum[l] / max(1, identity_per_layer_n[l]) for l in layers},
        **{
            f"identity_abs_effect_layer_{l}": identity_abs_per_layer_sum[l] / max(1, identity_per_layer_n[l])
            for l in layers
        },
    }


@dataclass(frozen=True)
class TopKRecoverySpec:
    layers: Tuple[int, ...] = (4, 8, 12)
    ks: Tuple[int, ...] = (1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 4000, 8000, 16384)
    split_seed: int = 0
    frac_selection: float = 0.5
    random_k_seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    eps: float = 1e-6
    bootstrap_B: int = 1000
    ci: float = 0.95
    token_reduce: Literal["mean"] = "mean"
    include_logz: bool = False
    logz_ks: Tuple[int, ...] = (20, 50, 200, 16384)
    random_control_mode: Literal["complement", "matched_bin"] = "complement"
    matched_bin_n_bins: int = 10


@dataclass(frozen=True)
class _TopKDirectionCase:
    pair_id: str
    direction: str
    donor_expected: str
    receiver_prompt: str
    choices: Mapping[str, List[str]]
    receiver_span: Tuple[int, ...]
    donor_latents: torch.Tensor
    base_margin: float
    base_pred: str


def split_pairs_deterministic(
    pair_ids: Sequence[str],
    seed: int,
    frac_selection: float = 0.5,
) -> tuple[set[str], set[str]]:
    if not (0.0 <= float(frac_selection) <= 1.0):
        raise ValueError("frac_selection must be in [0, 1]")

    uniq = sorted({str(pid) for pid in pair_ids})
    keyed: List[Tuple[str, str]] = []
    for pid in uniq:
        h = hashlib.sha256(f"{int(seed)}:{pid}".encode("utf-8")).hexdigest()
        keyed.append((h, pid))
    keyed.sort(key=lambda x: (x[0], x[1]))

    n_total = int(len(keyed))
    n_sel = int(round(float(frac_selection) * float(n_total)))
    n_sel = min(max(0, n_sel), n_total)
    sel = {pid for _, pid in keyed[:n_sel]}
    eva = {pid for _, pid in keyed[n_sel:]}
    return sel, eva


def rank_latents_delta_times_decoder_norm(
    recv_latents: torch.Tensor,  # (n_ex, span, d_latent)
    donor_latents: torch.Tensor,  # (n_ex, span, d_latent)
    w_dec: Optional[torch.Tensor] = None,  # (d_latent, d_out)
    w_norm: Optional[torch.Tensor] = None,  # (d_latent,)
    token_reduce: Literal["mean"] = "mean",
) -> torch.Tensor:
    if token_reduce != "mean":
        raise ValueError("token_reduce must be 'mean'")
    if recv_latents.ndim != 3 or donor_latents.ndim != 3:
        raise ValueError("recv_latents and donor_latents must have shape (n_ex, span, d_latent)")
    if recv_latents.shape != donor_latents.shape:
        raise ValueError("recv_latents and donor_latents must have identical shapes")
    d_latent = int(recv_latents.size(-1))
    if (w_dec is None) == (w_norm is None):
        raise ValueError("Provide exactly one of w_dec or w_norm")
    if w_norm is None:
        if w_dec.ndim != 2:
            raise ValueError("w_dec must have shape (d_latent, d_out)")
        if int(w_dec.size(0)) != d_latent:
            raise ValueError("w_dec first dimension must match latent width")
        w_norm_t = torch.linalg.norm(w_dec.to(dtype=torch.float32), dim=1)
    else:
        if w_norm.ndim != 1:
            raise ValueError("w_norm must have shape (d_latent,)")
        if int(w_norm.size(0)) != d_latent:
            raise ValueError("w_norm length must match latent width")
        w_norm_t = w_norm.to(dtype=torch.float32)
    if int(recv_latents.size(0)) < 1:
        return torch.zeros(d_latent, dtype=torch.float32, device="cpu")

    delta = donor_latents.to(dtype=torch.float32) - recv_latents.to(dtype=torch.float32)
    token_mean = delta.abs().mean(dim=1)  # (n_ex, d_latent)
    w_norm_work = w_norm_t.to(device=token_mean.device, dtype=torch.float32)
    per_ex_score = token_mean * w_norm_work.unsqueeze(0)
    return per_ex_score.mean(dim=0).to(device="cpu", dtype=torch.float32)


def _resolved_ks(ks: Sequence[int], *, d_latent: int) -> List[int]:
    out: List[int] = []
    for raw in ks:
        k = int(raw)
        if k < 1:
            continue
        k = min(int(k), int(d_latent))
        if k not in out:
            out.append(k)
    if not out:
        raise ValueError("No valid k values after clamping")
    return out


def _seed_for_random_k(*, split_seed: int, layer: int, k: int, seed: int) -> int:
    h = hashlib.sha256(f"{int(split_seed)}:{int(layer)}:{int(k)}:{int(seed)}".encode("utf-8")).hexdigest()
    return int(h[:16], 16) % (2**31 - 1)


def _sample_random_dims_from_complement(
    *,
    d_latent: int,
    k: int,
    topk_dims: Sequence[int],
    seed: int,
) -> List[int]:
    if int(k) >= int(d_latent):
        return list(range(int(d_latent)))
    topk_set = {int(x) for x in topk_dims}
    pool = [i for i in range(int(d_latent)) if i not in topk_set]
    if len(pool) < int(k):
        pool = list(range(int(d_latent)))
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    idx = torch.randperm(len(pool), generator=g)[: int(k)].tolist()
    return [int(pool[i]) for i in idx]


def _rank_bins(values: torch.Tensor, *, n_bins: int) -> torch.Tensor:
    x = _to_cpu_f64(values.detach()).flatten()
    n = int(x.numel())
    if n < 1:
        return torch.zeros(0, dtype=torch.int64)
    if int(n_bins) < 1:
        raise ValueError("n_bins must be >= 1")
    order = torch.argsort(x, descending=False)
    out = torch.empty(n, dtype=torch.int64)
    for rank, idx in enumerate(order.tolist()):
        b = min(int(n_bins) - 1, int((rank * int(n_bins)) // max(1, n)))
        out[int(idx)] = int(b)
    return out


def _feature_bins_for_matched_random(
    *,
    firing_rate: torch.Tensor,
    mean_abs_activation: torch.Tensor,
    n_bins: int,
) -> Dict[int, int]:
    if firing_rate.ndim != 1 or mean_abs_activation.ndim != 1:
        raise ValueError("firing_rate and mean_abs_activation must be rank-1 tensors")
    if int(firing_rate.numel()) != int(mean_abs_activation.numel()):
        raise ValueError("firing_rate and mean_abs_activation lengths must match")
    fire_bins = _rank_bins(firing_rate, n_bins=int(n_bins))
    act_bins = _rank_bins(mean_abs_activation, n_bins=int(n_bins))
    out: Dict[int, int] = {}
    for fid in range(int(fire_bins.numel())):
        out[int(fid)] = int(fire_bins[int(fid)].item()) * int(n_bins) + int(act_bins[int(fid)].item())
    return out


def _sample_random_dims_matched_bin(
    *,
    d_latent: int,
    k: int,
    topk_dims: Sequence[int],
    feature_bin_by_id: Mapping[int, int],
    seed: int,
) -> List[int]:
    if int(k) >= int(d_latent):
        return list(range(int(d_latent)))
    topk_list = [int(x) for x in topk_dims[: int(k)]]
    topk_set = set(topk_list)

    by_bin: Dict[int, List[int]] = {}
    for fid in range(int(d_latent)):
        if fid in topk_set:
            continue
        b = int(feature_bin_by_id.get(int(fid), -1))
        by_bin.setdefault(int(b), []).append(int(fid))
    for b in list(by_bin.keys()):
        by_bin[int(b)] = sorted(by_bin[int(b)])

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    chosen: List[int] = []
    chosen_set: set[int] = set()
    global_pool = [i for i in range(int(d_latent)) if i not in topk_set]
    if not global_pool:
        global_pool = list(range(int(d_latent)))

    def _pick_one(cands: List[int]) -> Optional[int]:
        if not cands:
            return None
        idx = int(torch.randint(low=0, high=len(cands), size=(1,), generator=g).item())
        return int(cands[idx])

    for src in topk_list:
        b = int(feature_bin_by_id.get(int(src), -1))
        cands = [x for x in by_bin.get(int(b), []) if x not in chosen_set]
        pick = _pick_one(cands)
        if pick is None:
            fallback = [x for x in global_pool if x not in chosen_set]
            pick = _pick_one(fallback)
        if pick is None:
            break
        chosen.append(int(pick))
        chosen_set.add(int(pick))
        if len(chosen) >= int(k):
            break

    if len(chosen) < int(k):
        tail = [x for x in global_pool if x not in chosen_set]
        if len(tail) < int(k) - len(chosen):
            tail = [x for x in range(int(d_latent)) if x not in chosen_set]
        if tail:
            perm = torch.randperm(len(tail), generator=g).tolist()
            for i in perm:
                chosen.append(int(tail[int(i)]))
                if len(chosen) >= int(k):
                    break

    return chosen[: int(k)]


def _gini_nonnegative(values: torch.Tensor) -> float:
    x = _to_cpu_f64(values.detach()).flatten()
    if int(x.numel()) < 1:
        return float("nan")
    x = torch.clamp(x, min=0.0)
    s = float(x.sum().item())
    if s <= 0.0:
        return 0.0
    x_sorted, _ = torch.sort(x, descending=False)
    n = int(x_sorted.numel())
    idx = torch.arange(1, n + 1, dtype=torch.float64)
    g = (2.0 * float((idx * x_sorted).sum().item())) / (float(n) * s) - (float(n + 1) / float(n))
    return float(max(0.0, min(1.0, g)))


def _logits_slice_for_continuation_base(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    continuation: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    prompt_ids = _encode_prompt(tokenizer, prompt, device=device)
    cont_ids = tokenizer(str(continuation), return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    full_ids = torch.cat([prompt_ids, cont_ids], dim=1)
    logits = model(input_ids=full_ids, use_cache=False, return_dict=True).logits
    p = int(prompt_ids.size(1))
    c = int(cont_ids.size(1))
    logits_slice = logits[:, p - 1 : p + c - 1, :].to(dtype=torch.float32)
    return logits_slice, cont_ids


def _logits_slice_for_continuation_clt_patched(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    continuation: str,
    device: torch.device,
    layer: int,
    token_indices: Sequence[int],
    clt: CLTProtocol,
    transform: CLTInputTransform,
    policy: LatentPolicy,
    config: Optional[CLTPatchConfig],
) -> Tuple[torch.Tensor, torch.Tensor]:
    prompt_ids = _encode_prompt(tokenizer, prompt, device=device)
    cont_ids = tokenizer(str(continuation), return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    full_ids = torch.cat([prompt_ids, cont_ids], dim=1)
    logits = forward_with_clt_latent_patching_span(
        model,
        input_ids=full_ids,
        site=PatchSpanSite(layer=int(layer), token_indices=tuple(int(x) for x in token_indices)),
        clt=clt,
        policy=policy,
        transform=transform,
        config=config,
    )
    p = int(prompt_ids.size(1))
    c = int(cont_ids.size(1))
    logits_slice = logits[:, p - 1 : p + c - 1, :].to(dtype=torch.float32)
    return logits_slice, cont_ids


def _decomp_deltas(base_logits: torch.Tensor, patched_logits: torch.Tensor, cont_ids: torch.Tensor) -> Dict[str, float]:
    if base_logits.shape != patched_logits.shape:
        raise ValueError("base_logits and patched_logits must have same shape")
    if cont_ids.ndim != 2 or int(cont_ids.size(0)) != int(base_logits.size(0)):
        raise ValueError("continuation IDs shape mismatch")

    base_logz = torch.logsumexp(base_logits, dim=-1)
    patch_logz = torch.logsumexp(patched_logits, dim=-1)
    base_logprobs = torch.log_softmax(base_logits, dim=-1)
    patch_logprobs = torch.log_softmax(patched_logits, dim=-1)

    gather_idx = cont_ids.unsqueeze(-1)
    base_target_logit = base_logits.gather(2, gather_idx).squeeze(-1)
    patch_target_logit = patched_logits.gather(2, gather_idx).squeeze(-1)
    base_target_logprob = base_logprobs.gather(2, gather_idx).squeeze(-1)
    patch_target_logprob = patch_logprobs.gather(2, gather_idx).squeeze(-1)

    return {
        "delta_logit_target_mean": float((patch_target_logit - base_target_logit).mean().item()),
        "delta_logz_mean": float((patch_logz - base_logz).mean().item()),
        "delta_logprob_target_mean": float((patch_target_logprob - base_target_logprob).mean().item()),
    }


def _aggregate_random_effects_for_recovery(
    rows: Sequence[Dict[str, Any]],
) -> Tuple[List[str], List[Tuple[str, str]], List[float]]:
    by_key: Dict[Tuple[str, str], List[float]] = {}
    for r in rows:
        key = (str(r["pair_id"]), str(r["direction"]))
        by_key.setdefault(key, []).append(float(r["effect"]))
    pair_ids: List[str] = []
    keys: List[Tuple[str, str]] = []
    vals: List[float] = []
    for key in sorted(by_key.keys()):
        ys = [float(v) for v in by_key[key]]
        if ys:
            keys.append(key)
            pair_ids.append(str(key[0]))
            vals.append(float(sum(ys) / len(ys)))
    return pair_ids, keys, vals


def _effect_ci_from_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    ci: float,
    bootstrap_B: int,
    seed: int,
) -> Dict[str, float]:
    pair_ids = [str(r["pair_id"]) for r in rows]
    vals = [float(r["effect"]) for r in rows]
    mean_v, lo, hi = bootstrap_ci_pair_cluster(pair_ids, vals, n_bootstrap=bootstrap_B, ci=ci, seed=seed)
    return {
        "mean": float(mean_v),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_rows": int(len(vals)),
        "n_pairs": int(len(set(pair_ids))),
    }


@torch.no_grad()
def run_clt_topk_feature_recovery(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    items: Sequence[DisambPair],
    device: torch.device,
    *,
    clt_by_layer: Mapping[int, CLTProtocol],
    transform_by_layer: Mapping[int, CLTInputTransform],
    spec: TopKRecoverySpec,
    config: Optional[CLTPatchConfig] = None,
    normalize_by_length: bool = True,
    require_token_id_match: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    layers = [int(x) for x in spec.layers]
    if not layers:
        raise ValueError("spec.layers must be non-empty")
    for l in layers:
        if int(l) not in clt_by_layer:
            raise ValueError(f"Missing CLT for layer {int(l)}")
        if int(l) not in transform_by_layer:
            raise ValueError(f"Missing transform for layer {int(l)}")

    pair_ids_all = sorted({str(it.pair_id) for it in items})
    pair_ids_s, pair_ids_e = split_pairs_deterministic(
        pair_ids_all,
        seed=int(spec.split_seed),
        frac_selection=float(spec.frac_selection),
    )

    telemetry_rows: List[Dict[str, Any]] = []
    ranking_by_layer: Dict[str, Any] = {}
    concentration_by_layer: Dict[str, Any] = {}
    curves_by_layer: Dict[str, Any] = {}
    logz_by_layer: Dict[str, Any] = {}

    for layer in layers:
        clt = clt_by_layer[int(layer)]
        transform = transform_by_layer[int(layer)]
        clt_device, clt_dtype = _infer_clt_device_dtype(clt)
        d_latent = int(getattr(clt, "d_latent"))
        if d_latent <= 0:
            raise ValueError(f"Invalid clt.d_latent at layer {int(layer)}")

        ks = _resolved_ks(spec.ks, d_latent=d_latent)
        logz_ks = _resolved_ks(spec.logz_ks, d_latent=d_latent) if bool(spec.include_logz) else []

        w_dec = getattr(clt, "W_dec", None)
        if not isinstance(w_dec, torch.Tensor) or w_dec.ndim != 2:
            raise ValueError("CLT must expose W_dec tensor with shape (d_latent, d_out)")
        if int(w_dec.size(0)) != d_latent:
            raise ValueError("W_dec first dim must match clt.d_latent")

        w_norm_work32 = torch.linalg.norm(
            w_dec.detach().to(device=clt_device, dtype=torch.float32), dim=1
        )
        # On MPS, do device move then dtype cast to avoid float64-on-MPS conversion errors.
        w_norm_cpu64 = w_norm_work32.to(device="cpu").to(dtype=torch.float64)

        rank_sum = torch.zeros(d_latent, dtype=torch.float64)
        n_rank_examples = 0
        selection_token_count = 0
        selection_abs_activation_sum = torch.zeros(d_latent, dtype=torch.float64)
        selection_active_count = torch.zeros(d_latent, dtype=torch.float64)
        n_total_directions = 0
        n_skipped_misaligned = 0
        eval_cases: List[_TopKDirectionCase] = []

        for it in items:
            for direction_name, donor, recv in (("a_to_b", it.a, it.b), ("b_to_a", it.b, it.a)):
                n_total_directions += 1
                donor_span, donor_token_ids = token_span_for_substring(
                    tokenizer, donor.prompt, it.target, it.target_occurrence
                )
                recv_span, recv_token_ids = token_span_for_substring(
                    tokenizer, recv.prompt, it.target, it.target_occurrence
                )
                if len(donor_span) != len(recv_span) or (
                    require_token_id_match and donor_token_ids != recv_token_ids
                ):
                    n_skipped_misaligned += 1
                    continue

                donor_ids = _encode_prompt(tokenizer, donor.prompt, device=device)
                recv_ids = _encode_prompt(tokenizer, recv.prompt, device=device)
                _assert_prompt_span_alignment(donor_ids, donor_span, donor_token_ids, side_name="donor")
                _assert_prompt_span_alignment(recv_ids, recv_span, recv_token_ids, side_name="receiver")
                donor_out = get_block_outputs(model, donor_ids, layers=[int(layer)])
                recv_out = get_block_outputs(model, recv_ids, layers=[int(layer)])

                donor_slice = (
                    donor_out[int(layer)][0, donor_span, :].detach().unsqueeze(0).to(device=clt_device, dtype=clt_dtype)
                )
                recv_slice = (
                    recv_out[int(layer)][0, recv_span, :].detach().unsqueeze(0).to(device=clt_device, dtype=clt_dtype)
                )

                donor_latents = clt.encode(transform.forward(donor_slice))
                recv_latents = clt.encode(transform.forward(recv_slice))

                pair_id = str(it.pair_id)
                if pair_id in pair_ids_s:
                    rank_vec = rank_latents_delta_times_decoder_norm(
                        recv_latents=recv_latents,
                        donor_latents=donor_latents,
                        w_norm=w_norm_work32,
                        token_reduce=spec.token_reduce,
                    )
                    rank_sum += _to_cpu_f64(rank_vec)
                    n_rank_examples += 1
                    recv_latents_cpu = _to_cpu_f64(recv_latents.detach())
                    selection_token_count += int(recv_latents_cpu.size(0) * recv_latents_cpu.size(1))
                    selection_abs_activation_sum += recv_latents_cpu.abs().sum(dim=(0, 1))
                    selection_active_count += (recv_latents_cpu.abs() > float(spec.eps)).to(dtype=torch.float64).sum(
                        dim=(0, 1)
                    )

                if pair_id in pair_ids_e:
                    base_scores = score_labels_next_continuations(
                        model,
                        tokenizer,
                        recv.prompt,
                        it.choices,
                        device,
                        normalize_by_length=normalize_by_length,
                    )
                    donor_expected = str(donor.expected_label)
                    base_margin = _margin(base_scores, expected=donor_expected)
                    eval_cases.append(
                        _TopKDirectionCase(
                            pair_id=pair_id,
                            direction=str(direction_name),
                            donor_expected=donor_expected,
                            receiver_prompt=str(recv.prompt),
                            choices=it.choices,
                            receiver_span=tuple(int(i) for i in recv_span),
                            donor_latents=donor_latents,
                            base_margin=float(base_margin),
                            base_pred=str(base_scores.argmax_label()),
                        )
                    )

        if n_rank_examples > 0:
            rank_score = (rank_sum / float(n_rank_examples)).to(dtype=torch.float32)
        else:
            rank_score = torch.zeros(d_latent, dtype=torch.float32)
        rank_order = torch.argsort(rank_score, descending=True)

        if int(selection_token_count) > 0:
            selection_mean_abs_activation = selection_abs_activation_sum / float(selection_token_count)
            selection_firing_rate = selection_active_count / float(selection_token_count)
        else:
            selection_mean_abs_activation = torch.zeros(d_latent, dtype=torch.float64)
            selection_firing_rate = torch.zeros(d_latent, dtype=torch.float64)

        random_mode = str(getattr(spec, "random_control_mode", "complement")).strip().lower()
        if random_mode not in {"complement", "matched_bin"}:
            raise ValueError(f"Unsupported random_control_mode={random_mode!r}")
        feature_bin_by_id: Dict[int, int] = {}
        if random_mode == "matched_bin":
            feature_bin_by_id = _feature_bins_for_matched_random(
                firing_rate=selection_firing_rate,
                mean_abs_activation=selection_mean_abs_activation,
                n_bins=int(spec.matched_bin_n_bins),
            )

        topk_dims_by_k: Dict[int, List[int]] = {}
        bottomk_dims_by_k: Dict[int, List[int]] = {}
        random_dims_by_k_seed: Dict[Tuple[int, int], List[int]] = {}
        for k in ks:
            top = [int(x) for x in rank_order[: int(k)].tolist()]
            bottom = [int(x) for x in rank_order[-int(k) :].tolist()]
            topk_dims_by_k[int(k)] = top
            bottomk_dims_by_k[int(k)] = bottom
            for seed in spec.random_k_seeds:
                rnd_seed = _seed_for_random_k(split_seed=spec.split_seed, layer=layer, k=int(k), seed=int(seed))
                if random_mode == "matched_bin":
                    random_dims_by_k_seed[(int(k), int(seed))] = _sample_random_dims_matched_bin(
                        d_latent=d_latent,
                        k=int(k),
                        topk_dims=top,
                        feature_bin_by_id=feature_bin_by_id,
                        seed=rnd_seed,
                    )
                else:
                    random_dims_by_k_seed[(int(k), int(seed))] = _sample_random_dims_from_complement(
                        d_latent=d_latent,
                        k=int(k),
                        topk_dims=top,
                        seed=rnd_seed,
                    )

        total_rank_mass = float(rank_score.sum().item())
        concentration_by_layer[str(layer)] = {
            "gini": float(_gini_nonnegative(rank_score)),
            "rank_score_sum": float(total_rank_mass),
            "mass_at_20": float(rank_score[rank_order[: min(20, d_latent)]].sum().item() / total_rank_mass)
            if total_rank_mass > 0.0
            else float("nan"),
            "mass_at_50": float(rank_score[rank_order[: min(50, d_latent)]].sum().item() / total_rank_mass)
            if total_rank_mass > 0.0
            else float("nan"),
            "mass_at_100": float(rank_score[rank_order[: min(100, d_latent)]].sum().item() / total_rank_mass)
            if total_rank_mass > 0.0
            else float("nan"),
        }
        ranking_by_layer[str(layer)] = {
            "score_definition": "mean_t(abs(delta_latent[t,j])) * l2_norm(W_dec[j,:])",
            "token_reduce": str(spec.token_reduce),
            "n_selection_examples": int(n_rank_examples),
            "n_selection_tokens": int(selection_token_count),
            "d_latent": int(d_latent),
            "random_control_mode": str(random_mode),
            "matched_bin_n_bins": int(spec.matched_bin_n_bins) if random_mode == "matched_bin" else None,
            "top_features": [
                {
                    "rank": int(i + 1),
                    "feature_id": int(fid),
                    "rank_score": float(rank_score[int(fid)].item()),
                    "w_dec_norm": float(w_norm_cpu64[int(fid)].item()),
                    "selection_firing_rate": float(selection_firing_rate[int(fid)].item()),
                    "selection_mean_abs_activation": float(selection_mean_abs_activation[int(fid)].item()),
                    "matched_bin_id": int(feature_bin_by_id.get(int(fid), -1)) if random_mode == "matched_bin" else None,
                }
                for i, fid in enumerate(rank_order[: min(20, d_latent)].tolist())
            ],
        }

        layer_rows: List[Dict[str, Any]] = []
        layer_logz_rows: List[Dict[str, Any]] = []
        for case in eval_cases:
            full_policy = ReplaceLatentsAtIndicesPolicy(
                token_indices=list(case.receiver_span),
                replacement_latents=case.donor_latents,
            )
            full_scores = score_labels_next_continuations_clt_patched(
                model,
                tokenizer,
                case.receiver_prompt,
                case.choices,
                device,
                layer=int(layer),
                token_indices=list(case.receiver_span),
                clt=clt,
                transform=transform,
                policy=full_policy,
                config=config,
                normalize_by_length=normalize_by_length,
            )
            full_margin = _margin(full_scores, expected=case.donor_expected)
            full_effect = float(full_margin - float(case.base_margin))
            full_flip = int((case.base_pred != case.donor_expected) and (full_scores.argmax_label() == case.donor_expected))

            if bool(spec.include_logz) and int(d_latent) in logz_ks:
                conts = case.choices.get(case.donor_expected, [])
                if conts:
                    cont0 = str(conts[0])
                    base_logits, cont_ids = _logits_slice_for_continuation_base(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=case.receiver_prompt,
                        continuation=cont0,
                        device=device,
                    )
                    full_logits, _ = _logits_slice_for_continuation_clt_patched(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=case.receiver_prompt,
                        continuation=cont0,
                        device=device,
                        layer=int(layer),
                        token_indices=case.receiver_span,
                        clt=clt,
                        transform=transform,
                        policy=full_policy,
                        config=config,
                    )
                    dvals = _decomp_deltas(base_logits, full_logits, cont_ids)
                    layer_logz_rows.append(
                        {
                            "pair_id": str(case.pair_id),
                            "direction": str(case.direction),
                            "layer": int(layer),
                            "k": int(d_latent),
                            "arm": "full",
                            "continuation_used": str(cont0),
                            **dvals,
                        }
                    )

            for k in ks:
                row_full = {
                    "pair_id": str(case.pair_id),
                    "direction": str(case.direction),
                    "split": "E",
                    "layer": int(layer),
                    "k": int(k),
                    "arm": "full",
                    "random_seed": None,
                    "base_margin": float(case.base_margin),
                    "patched_margin": float(full_margin),
                    "effect": float(full_effect),
                    "flipped": int(full_flip),
                }
                telemetry_rows.append(dict(row_full))
                layer_rows.append(dict(row_full))

                if int(k) >= int(d_latent):
                    row_top = dict(row_full)
                    row_top["arm"] = "topk"
                    row_bottom = dict(row_full)
                    row_bottom["arm"] = "bottomk"
                    telemetry_rows.extend([row_top, row_bottom])
                    layer_rows.extend([row_top, row_bottom])
                    for seed in spec.random_k_seeds:
                        row_rand = dict(row_full)
                        row_rand["arm"] = "randomk"
                        row_rand["random_seed"] = int(seed)
                        telemetry_rows.append(row_rand)
                        layer_rows.append(row_rand)
                    continue

                topk_policy = ReplaceLatentDimsAtIndicesPolicy(
                    token_indices=list(case.receiver_span),
                    replacement_latents=case.donor_latents,
                    latent_dims=topk_dims_by_k[int(k)],
                )
                topk_scores = score_labels_next_continuations_clt_patched(
                    model,
                    tokenizer,
                    case.receiver_prompt,
                    case.choices,
                    device,
                    layer=int(layer),
                    token_indices=list(case.receiver_span),
                    clt=clt,
                    transform=transform,
                    policy=topk_policy,
                    config=config,
                    normalize_by_length=normalize_by_length,
                )
                topk_margin = _margin(topk_scores, expected=case.donor_expected)
                topk_effect = float(topk_margin - float(case.base_margin))
                topk_flip = int((case.base_pred != case.donor_expected) and (topk_scores.argmax_label() == case.donor_expected))
                row_topk = {
                    "pair_id": str(case.pair_id),
                    "direction": str(case.direction),
                    "split": "E",
                    "layer": int(layer),
                    "k": int(k),
                    "arm": "topk",
                    "random_seed": None,
                    "base_margin": float(case.base_margin),
                    "patched_margin": float(topk_margin),
                    "effect": float(topk_effect),
                    "flipped": int(topk_flip),
                }
                telemetry_rows.append(dict(row_topk))
                layer_rows.append(dict(row_topk))

                bottom_policy = ReplaceLatentDimsAtIndicesPolicy(
                    token_indices=list(case.receiver_span),
                    replacement_latents=case.donor_latents,
                    latent_dims=bottomk_dims_by_k[int(k)],
                )
                bottom_scores = score_labels_next_continuations_clt_patched(
                    model,
                    tokenizer,
                    case.receiver_prompt,
                    case.choices,
                    device,
                    layer=int(layer),
                    token_indices=list(case.receiver_span),
                    clt=clt,
                    transform=transform,
                    policy=bottom_policy,
                    config=config,
                    normalize_by_length=normalize_by_length,
                )
                bottom_margin = _margin(bottom_scores, expected=case.donor_expected)
                bottom_effect = float(bottom_margin - float(case.base_margin))
                bottom_flip = int(
                    (case.base_pred != case.donor_expected) and (bottom_scores.argmax_label() == case.donor_expected)
                )
                row_bottom = {
                    "pair_id": str(case.pair_id),
                    "direction": str(case.direction),
                    "split": "E",
                    "layer": int(layer),
                    "k": int(k),
                    "arm": "bottomk",
                    "random_seed": None,
                    "base_margin": float(case.base_margin),
                    "patched_margin": float(bottom_margin),
                    "effect": float(bottom_effect),
                    "flipped": int(bottom_flip),
                }
                telemetry_rows.append(dict(row_bottom))
                layer_rows.append(dict(row_bottom))

                if bool(spec.include_logz) and int(k) in logz_ks:
                    conts = case.choices.get(case.donor_expected, [])
                    if conts:
                        cont0 = str(conts[0])
                        base_logits, cont_ids = _logits_slice_for_continuation_base(
                            model=model,
                            tokenizer=tokenizer,
                            prompt=case.receiver_prompt,
                            continuation=cont0,
                            device=device,
                        )
                        topk_logits, _ = _logits_slice_for_continuation_clt_patched(
                            model=model,
                            tokenizer=tokenizer,
                            prompt=case.receiver_prompt,
                            continuation=cont0,
                            device=device,
                            layer=int(layer),
                            token_indices=case.receiver_span,
                            clt=clt,
                            transform=transform,
                            policy=topk_policy,
                            config=config,
                        )
                        dvals = _decomp_deltas(base_logits, topk_logits, cont_ids)
                        layer_logz_rows.append(
                            {
                                "pair_id": str(case.pair_id),
                                "direction": str(case.direction),
                                "layer": int(layer),
                                "k": int(k),
                                "arm": "topk",
                                "continuation_used": str(cont0),
                                **dvals,
                            }
                        )

                for seed in spec.random_k_seeds:
                    rnd_dims = random_dims_by_k_seed[(int(k), int(seed))]
                    random_policy = ReplaceLatentDimsAtIndicesPolicy(
                        token_indices=list(case.receiver_span),
                        replacement_latents=case.donor_latents,
                        latent_dims=rnd_dims,
                    )
                    random_scores = score_labels_next_continuations_clt_patched(
                        model,
                        tokenizer,
                        case.receiver_prompt,
                        case.choices,
                        device,
                        layer=int(layer),
                        token_indices=list(case.receiver_span),
                        clt=clt,
                        transform=transform,
                        policy=random_policy,
                        config=config,
                        normalize_by_length=normalize_by_length,
                    )
                    random_margin = _margin(random_scores, expected=case.donor_expected)
                    random_effect = float(random_margin - float(case.base_margin))
                    random_flip = int(
                        (case.base_pred != case.donor_expected)
                        and (random_scores.argmax_label() == case.donor_expected)
                    )
                    row_rand = {
                        "pair_id": str(case.pair_id),
                        "direction": str(case.direction),
                        "split": "E",
                        "layer": int(layer),
                        "k": int(k),
                        "arm": "randomk",
                        "random_seed": int(seed),
                        "base_margin": float(case.base_margin),
                        "patched_margin": float(random_margin),
                        "effect": float(random_effect),
                        "flipped": int(random_flip),
                    }
                    telemetry_rows.append(dict(row_rand))
                    layer_rows.append(dict(row_rand))

        effects: Dict[str, Dict[str, Dict[str, float]]] = {
            "topk": {},
            "bottomk": {},
            "full": {},
            "randomk_mean": {},
        }
        recovery: Dict[str, Dict[str, Dict[str, float]]] = {
            "topk": {},
            "bottomk": {},
            "randomk_mean": {},
        }
        randomk_per_seed: Dict[str, Dict[str, Dict[str, float]]] = {
            str(int(seed)): {} for seed in spec.random_k_seeds
        }

        for k in ks:
            rows_full = [r for r in layer_rows if int(r["k"]) == int(k) and str(r["arm"]) == "full"]
            rows_topk = [r for r in layer_rows if int(r["k"]) == int(k) and str(r["arm"]) == "topk"]
            rows_bottom = [r for r in layer_rows if int(r["k"]) == int(k) and str(r["arm"]) == "bottomk"]
            rows_random = [r for r in layer_rows if int(r["k"]) == int(k) and str(r["arm"]) == "randomk"]

            effects["full"][str(int(k))] = _effect_ci_from_rows(
                rows_full,
                ci=float(spec.ci),
                bootstrap_B=int(spec.bootstrap_B),
                seed=int(spec.split_seed + 101 + int(layer) + int(k)),
            )
            effects["topk"][str(int(k))] = _effect_ci_from_rows(
                rows_topk,
                ci=float(spec.ci),
                bootstrap_B=int(spec.bootstrap_B),
                seed=int(spec.split_seed + 201 + int(layer) + int(k)),
            )
            effects["bottomk"][str(int(k))] = _effect_ci_from_rows(
                rows_bottom,
                ci=float(spec.ci),
                bootstrap_B=int(spec.bootstrap_B),
                seed=int(spec.split_seed + 301 + int(layer) + int(k)),
            )

            rows_by_seed: Dict[int, List[Dict[str, Any]]] = {}
            for r in rows_random:
                rs = int(r["random_seed"])
                rows_by_seed.setdefault(rs, []).append(r)
            for seed in spec.random_k_seeds:
                randomk_per_seed[str(int(seed))][str(int(k))] = _effect_ci_from_rows(
                    rows_by_seed.get(int(seed), []),
                    ci=float(spec.ci),
                    bootstrap_B=int(spec.bootstrap_B),
                    seed=int(spec.split_seed + 401 + int(layer) + int(k) + int(seed)),
                )

            rand_pair_ids, rand_keys, rand_vals = _aggregate_random_effects_for_recovery(rows_random)
            rand_mean, rand_lo, rand_hi = bootstrap_ci_pair_cluster(
                rand_pair_ids,
                rand_vals,
                n_bootstrap=int(spec.bootstrap_B),
                ci=float(spec.ci),
                seed=int(spec.split_seed + 501 + int(layer) + int(k)),
            )
            effects["randomk_mean"][str(int(k))] = {
                "mean": float(rand_mean),
                "ci_low": float(rand_lo),
                "ci_high": float(rand_hi),
                "n_rows": int(len(rand_vals)),
                "n_pairs": int(len(set(rand_pair_ids))),
            }

            full_by_key = {(str(r["pair_id"]), str(r["direction"])): float(r["effect"]) for r in rows_full}

            def _recovery_for_rows(cur_rows: Sequence[Dict[str, Any]], seed_offset: int) -> Dict[str, float]:
                pair_ids: List[str] = []
                nums: List[float] = []
                dens: List[float] = []
                for rr in cur_rows:
                    key = (str(rr["pair_id"]), str(rr["direction"]))
                    if key not in full_by_key:
                        continue
                    pair_ids.append(str(key[0]))
                    nums.append(float(rr["effect"]))
                    dens.append(float(full_by_key[key]))
                if not pair_ids:
                    return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_rows": 0}
                m, lo, hi = bootstrap_ratio_ci_pair_cluster(
                    pair_ids,
                    nums,
                    dens,
                    eps=float(spec.eps),
                    n_bootstrap=int(spec.bootstrap_B),
                    ci=float(spec.ci),
                    seed=int(spec.split_seed + seed_offset + int(layer) + int(k)),
                )
                return {"mean": float(m), "ci_low": float(lo), "ci_high": float(hi), "n_rows": int(len(pair_ids))}

            recovery["topk"][str(int(k))] = _recovery_for_rows(rows_topk, seed_offset=601)
            recovery["bottomk"][str(int(k))] = _recovery_for_rows(rows_bottom, seed_offset=701)

            rand_rows_recovery: List[Dict[str, Any]] = []
            rand_vals_by_key = {kk: vv for kk, vv in zip(rand_keys, rand_vals)}
            for key, val in rand_vals_by_key.items():
                rand_rows_recovery.append(
                    {
                        "pair_id": str(key[0]),
                        "direction": str(key[1]),
                        "effect": float(val),
                    }
                )
            recovery["randomk_mean"][str(int(k))] = _recovery_for_rows(rand_rows_recovery, seed_offset=801)

        curves_by_layer[str(layer)] = {
            "effects": effects,
            "recovery": recovery,
            "randomk_per_seed": randomk_per_seed,
            "full_is_duplicated_across_k": True,
            "n_eval_cases": int(len(eval_cases)),
            "n_total_directions": int(n_total_directions),
            "n_skipped_misaligned": int(n_skipped_misaligned),
        }

        if bool(spec.include_logz):
            per_layer_logz: Dict[str, Dict[str, Dict[str, float]]] = {}
            for k in sorted({int(r["k"]) for r in layer_logz_rows}):
                per_layer_logz[str(int(k))] = {}
                for arm in sorted({str(r["arm"]) for r in layer_logz_rows if int(r["k"]) == int(k)}):
                    rows_ka = [r for r in layer_logz_rows if int(r["k"]) == int(k) and str(r["arm"]) == arm]
                    if not rows_ka:
                        continue
                    per_layer_logz[str(int(k))][str(arm)] = {
                        "delta_logit_target_mean": float(
                            sum(float(r["delta_logit_target_mean"]) for r in rows_ka) / len(rows_ka)
                        ),
                        "delta_logz_mean": float(sum(float(r["delta_logz_mean"]) for r in rows_ka) / len(rows_ka)),
                        "delta_logprob_target_mean": float(
                            sum(float(r["delta_logprob_target_mean"]) for r in rows_ka) / len(rows_ka)
                        ),
                        "n_rows": int(len(rows_ka)),
                    }
            logz_by_layer[str(layer)] = per_layer_logz

    summary: Dict[str, Any] = {
        "schema_version": "topk_recovery_v1",
        "spec": {
            "layers": [int(x) for x in spec.layers],
            "ks": [int(x) for x in spec.ks],
            "split_seed": int(spec.split_seed),
            "frac_selection": float(spec.frac_selection),
            "random_k_seeds": [int(x) for x in spec.random_k_seeds],
            "eps": float(spec.eps),
            "bootstrap_B": int(spec.bootstrap_B),
            "ci": float(spec.ci),
            "token_reduce": str(spec.token_reduce),
            "include_logz": bool(spec.include_logz),
            "logz_ks": [int(x) for x in spec.logz_ks],
            "random_control_mode": str(spec.random_control_mode),
            "matched_bin_n_bins": int(spec.matched_bin_n_bins),
            "normalize_by_length": bool(normalize_by_length),
            "require_token_id_match": bool(require_token_id_match),
        },
        "split": {
            "n_pairs_total": int(len(pair_ids_all)),
            "n_pairs_S": int(len(pair_ids_s)),
            "n_pairs_E": int(len(pair_ids_e)),
            "pair_ids_S": sorted(pair_ids_s),
            "pair_ids_E": sorted(pair_ids_e),
        },
        "ranking_by_layer": ranking_by_layer,
        "concentration_by_layer": concentration_by_layer,
        "curves_by_layer": curves_by_layer,
    }
    if bool(spec.include_logz):
        summary["logz_mode"] = "first_expected_continuation"
        summary["logz_decomp"] = logz_by_layer
    if "4" in ranking_by_layer:
        summary["top_features_L4"] = ranking_by_layer["4"]["top_features"]

    return telemetry_rows, summary
