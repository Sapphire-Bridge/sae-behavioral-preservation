from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..data.schemas import DisambPair, PromptSide
from ..interventions.activation_patching import (
    PatchSite,
    PatchSpanSite,
    forward_with_patched_block_output_span,
    get_hidden_states,
    prefill_with_patched_block_output_span,
)
from ..token_spans import token_span_for_substring
from ..utils import (
    bootstrap_ci,
    bootstrap_ci_metric,
    get_logprob_computation_config,
    get_scoring_performance_config,
    logprob_of_continuation_candidates_shared_prompt,
    logprob_of_continuation_candidates_with_prefill,
)


@dataclass(frozen=True)
class LabelScores:
    by_label: Dict[str, float]

    def argmax_label(self) -> str:
        return max(self.by_label.items(), key=lambda kv: kv[1])[0]


def _logmeanexp(xs: Sequence[float]) -> float:
    if len(xs) < 1:
        raise ValueError("logmeanexp requires at least one value")
    t = torch.tensor(list(xs), dtype=torch.float64)
    return float(torch.logsumexp(t, dim=0) - math.log(len(xs)))


def _encode_prompt(tokenizer: PreTrainedTokenizerBase, text: str, device: torch.device) -> torch.Tensor:
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    return enc["input_ids"].to(device)


def _encode_continuation(tokenizer: PreTrainedTokenizerBase, text: str, device: torch.device) -> torch.Tensor:
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    return enc["input_ids"].to(device)


@torch.no_grad()
def score_labels_next_continuations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    choices: Mapping[str, List[str]],
    device: torch.device,
    *,
    normalize_by_length: bool = True,
) -> LabelScores:
    """
    Score each label via log-mean-exp over per-continuation log-probabilities.

    By default, each continuation is length-normalized (mean logprob per token), and then we
    aggregate a label's continuation set with:
      logmeanexp(vals) = logsumexp(vals) - log(len(vals))
    which is invariant to the number of continuations per label (up to sampling variance).
    """
    prompt_ids = _encode_prompt(tokenizer, prompt, device=device)
    score_batch_size, use_prefix_cache = get_scoring_performance_config()
    pad_token_id = (
        int(tokenizer.pad_token_id)
        if tokenizer.pad_token_id is not None
        else (int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else 0)
    )
    flat_conts: List[torch.Tensor] = []
    spans: Dict[str, tuple[int, int]] = {}
    scores: Dict[str, float] = {}
    for label, continuations in choices.items():
        if len(continuations) < 1:
            raise ValueError(f"Empty continuation list for label={label}")
        start = len(flat_conts)
        encoded = [_encode_continuation(tokenizer, cont, device=device).squeeze(0) for cont in continuations]
        flat_conts.extend(encoded)
        spans[str(label)] = (start, len(flat_conts))

    flat_logps = logprob_of_continuation_candidates_shared_prompt(
        model=model,
        prompt_ids=prompt_ids,
        continuation_id_list=flat_conts,
        normalize_by_length=normalize_by_length,
        batch_size=int(score_batch_size),
        pad_token_id=int(pad_token_id),
        use_prefix_cache=bool(use_prefix_cache),
    )
    for label in spans.keys():
        start, end = spans[str(label)]
        vals = [float(x) for x in flat_logps[start:end].tolist()]
        scores[str(label)] = _logmeanexp(vals)
    return LabelScores(by_label=scores)


def _margin(scores: LabelScores, expected: str) -> float:
    exp = scores.by_label[expected]
    best_other = max(v for k, v in scores.by_label.items() if k != expected)
    return float(exp - best_other)


@torch.no_grad()
def score_labels_next_continuations_patched(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    choices: Mapping[str, List[str]],
    device: torch.device,
    *,
    patch_site: PatchSite | PatchSpanSite,
    replacement: torch.Tensor,
    normalize_by_length: bool = True,
) -> LabelScores:
    """
    Score choices under a patched forward pass (block output patched at `patch_site`).

    This is an intentionally transparent implementation that re-runs the patched model per continuation.
    """
    prompt_ids = _encode_prompt(tokenizer, prompt, device=device)
    score_batch_size, use_prefix_cache = get_scoring_performance_config()
    logprobs_dtype, strict_finite = get_logprob_computation_config()
    pad_token_id = (
        int(tokenizer.pad_token_id)
        if tokenizer.pad_token_id is not None
        else (int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else 0)
    )
    patch_span_site = (
        PatchSpanSite(layer=int(patch_site.layer), token_indices=(int(patch_site.token_index),))
        if isinstance(patch_site, PatchSite)
        else patch_site
    )

    flat_conts: List[torch.Tensor] = []
    spans: Dict[str, tuple[int, int]] = {}
    for label, continuations in choices.items():
        if len(continuations) < 1:
            raise ValueError(f"Empty continuation list for label={label}")
        start = len(flat_conts)
        encoded = [_encode_continuation(tokenizer, cont, device=device).squeeze(0) for cont in continuations]
        flat_conts.extend(encoded)
        spans[str(label)] = (start, len(flat_conts))

    flat_logps: Optional[torch.Tensor] = None
    if bool(use_prefix_cache):
        try:
            prefill_out = prefill_with_patched_block_output_span(
                model,
                input_ids=prompt_ids,
                site=patch_span_site,
                replacement=replacement,
                attention_mask=torch.ones_like(prompt_ids),
            )
            flat_logps = logprob_of_continuation_candidates_with_prefill(
                model=model,
                prompt_ids=prompt_ids,
                continuation_id_list=flat_conts,
                prefill_logits_last=prefill_out.logits[:, -1, :],  # type: ignore[attr-defined]
                prefill_past_key_values=prefill_out.past_key_values,  # type: ignore[attr-defined]
                normalize_by_length=normalize_by_length,
                batch_size=int(score_batch_size),
                pad_token_id=int(pad_token_id),
            )
        except Exception:
            # Preserve correctness on cache incompatibilities by falling back
            # to the full-forward batched path.
            flat_logps = None
    if flat_logps is None:
        flat_scores: List[torch.Tensor] = []
        P = int(prompt_ids.size(1))
        chunk_size = int(max(1, score_batch_size))
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
            logits = forward_with_patched_block_output_span(
                model,
                input_ids=full_ids,
                site=patch_span_site,
                replacement=replacement,
                attention_mask=attn_mask,
            )
            logits_slice = logits[:, P - 1 : P + max_len - 1, :].to(dtype=logprobs_dtype)
            log_probs = torch.log_softmax(logits_slice, dim=-1)
            gathered = log_probs.gather(2, cont_pad.unsqueeze(-1)).squeeze(-1)
            if not torch.isfinite(gathered[cont_mask]).all():
                if strict_finite:
                    raise FloatingPointError(
                        "Non-finite log-probability detected in patched continuation scoring "
                        f"(device={gathered.device}, logits_dtype={logits.dtype}, logprobs_dtype={gathered.dtype})."
                    )
                gathered = torch.where(torch.isfinite(gathered), gathered, torch.full_like(gathered, -1e9))
            gathered = torch.where(cont_mask, gathered, torch.zeros_like(gathered))
            lp = gathered.sum(dim=1)
            if normalize_by_length:
                lp = lp / lengths.to(dtype=lp.dtype)
            flat_scores.append(lp)
        flat_logps = torch.cat(flat_scores, dim=0) if flat_scores else torch.empty((0,), device=device, dtype=logprobs_dtype)

    scores: Dict[str, float] = {}
    for label in spans.keys():
        start, end = spans[str(label)]
        vals = [float(x) for x in flat_logps[start:end].tolist()]
        scores[str(label)] = _logmeanexp(vals)
    return LabelScores(by_label=scores)

@torch.no_grad()
def compute_aom_disamb(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    items: List[DisambPair],
    device: torch.device,
    *,
    normalize_by_length: bool = True,
    ci: float = 0.95,
    bootstrap_n: int = 1000,
    bootstrap_seed: int = 42,
) -> Dict[str, Any]:
    def _boot(name: str, values: List[float]) -> Dict[str, Any]:
        mv, lo, hi = bootstrap_ci_metric(values, n_bootstrap=bootstrap_n, ci=ci, seed=bootstrap_seed)
        out: Dict[str, Any] = {
            name: float(mv.value),
            f"{name}_ci_low": float(lo),
            f"{name}_ci_high": float(hi),
            f"{name}_n": int(mv.n),
            f"{name}_valid": bool(mv.valid),
        }
        if mv.reason is not None:
            out[f"{name}_reason"] = str(mv.reason)
        return out

    pair_acc_samples: List[float] = []
    pair_margin_samples: List[float] = []

    for it in items:
        # Bootstrap unit is the minimal pair (not the prompt side).
        side_acc: List[float] = []
        side_margin: List[float] = []
        for side in (it.a, it.b):
            scores = score_labels_next_continuations(
                model, tokenizer, side.prompt, it.choices, device, normalize_by_length=normalize_by_length
            )
            pred = scores.argmax_label()
            side_acc.append(float(pred == side.expected_label))
            side_margin.append(_margin(scores, expected=side.expected_label))
        pair_acc = float(sum(side_acc) / max(1, len(side_acc)))
        pair_margin = float(sum(side_margin) / max(1, len(side_margin)))
        pair_acc_samples.append(pair_acc)
        pair_margin_samples.append(pair_margin)

    out: Dict[str, Any] = {
        **_boot("accuracy", pair_acc_samples),
        **_boot("mean_margin", pair_margin_samples),
        "n_pairs_total": int(len(items)),
        "n_sides_total": int(2 * len(items)),
    }
    return out


def _stable_u32_seed(*parts: str, base_seed: int) -> int:
    """
    Deterministic seed helper (stable across Python versions/processes).

    Used to choose off-target control pairs via seeded RNG without optimizing for effect size.
    """
    h = hashlib.sha256()
    h.update(str(int(base_seed)).encode("utf-8"))
    for p in parts:
        h.update(b"|")
        h.update(str(p).encode("utf-8"))
    return int.from_bytes(h.digest()[:4], byteorder="little", signed=False)


def _excluded_indices(span: Sequence[int], *, buffer: int, seq_len: int) -> set[int]:
    if buffer < 0:
        raise ValueError("buffer must be >= 0")
    if seq_len < 0:
        raise ValueError("seq_len must be >= 0")
    if not span:
        return set()
    lo = max(0, int(min(span)) - int(buffer))
    hi = min(int(seq_len) - 1, int(max(span)) + int(buffer))
    if hi < lo:
        return set()
    return set(range(lo, hi + 1))


def _select_offtarget_pairs(
    *,
    donor_ids: Sequence[int],
    recv_ids: Sequence[int],
    donor_span: Sequence[int],
    recv_span: Sequence[int],
    k: int,
    buffer: int,
    donor_buffer: int,
    position_window: Optional[int] = None,
    rng: random.Random,
) -> Optional[List[Tuple[int, int]]]:
    """
    Select k paired positions (q_donor, p_recv) such that donor_ids[q_donor] == recv_ids[p_recv].

    Excludes receiver positions within +/- `buffer` of the receiver target span, and excludes donor
    positions within +/- `donor_buffer` of the donor target span (to avoid accidentally selecting
    near-target cue tokens).

    Pairs are chosen deterministically via seeded RNG, not optimized for effect size.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if not recv_span:
        raise ValueError("recv_span must be non-empty")

    center = int(round(sum(int(i) for i in recv_span) / float(len(recv_span))))

    recv_excluded = _excluded_indices(recv_span, buffer=int(buffer), seq_len=len(recv_ids))
    donor_excluded = _excluded_indices(donor_span, buffer=int(donor_buffer), seq_len=len(donor_ids))

    donor_positions_by_id: Dict[int, List[int]] = {}
    for q, tok_id in enumerate(donor_ids):
        if q in donor_excluded:
            continue
        donor_positions_by_id.setdefault(int(tok_id), []).append(int(q))

    eligible_recv_positions_all: List[int] = []
    for p, tok_id in enumerate(recv_ids):
        if p in recv_excluded:
            continue
        if int(tok_id) in donor_positions_by_id:
            eligible_recv_positions_all.append(int(p))

    if len(eligible_recv_positions_all) < k:
        return None

    eligible_recv_positions = eligible_recv_positions_all
    if position_window is not None and int(position_window) >= 0:
        w = int(position_window)
        max_w = int(max(0, len(recv_ids)))
        while True:
            in_window = [p for p in eligible_recv_positions_all if abs(int(p) - center) <= w]
            if len(in_window) >= k:
                eligible_recv_positions = in_window
                break
            if w >= max_w:
                eligible_recv_positions = eligible_recv_positions_all
                break
            w = min(max_w, max(w * 2, w + 1))

    # Position-match: choose the k receiver positions closest to the target span center
    # (ties broken deterministically by seeded RNG shuffling).
    recv_positions = list(eligible_recv_positions)
    rng.shuffle(recv_positions)
    recv_positions.sort(key=lambda p: abs(int(p) - center))
    recv_positions = recv_positions[:k]
    used_donor_positions: set[int] = set()
    pairs: List[Tuple[int, int]] = []
    for p in recv_positions:
        tok_id = int(recv_ids[p])
        donor_positions = donor_positions_by_id[tok_id]
        unused = [q for q in donor_positions if q not in used_donor_positions]
        candidates = unused if unused else donor_positions
        # Prefer the donor position closest to the receiver position to reduce position-mismatch artifacts.
        q = int(min(candidates, key=lambda q: (abs(int(q) - int(p)), int(q))))
        used_donor_positions.add(q)
        pairs.append((q, int(p)))
    return pairs


def _select_offtarget_pairs_same_index(
    *,
    donor_ids: Sequence[int],
    recv_ids: Sequence[int],
    donor_span: Sequence[int],
    recv_span: Sequence[int],
    k: int,
    buffer: int,
    donor_buffer: int,
    position_window: Optional[int] = None,
    rng: random.Random,
) -> Optional[List[Tuple[int, int]]]:
    """
    Fallback off-target control: choose k indices i (off-target) and patch receiver i with donor i.

    This avoids the position-mismatch confound (q_donor != p_recv) when token overlap is sparse.
    Prefer indices where token IDs match at the same index, but do not require it.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if not recv_span:
        raise ValueError("recv_span must be non-empty")

    center = int(round(sum(int(i) for i in recv_span) / float(len(recv_span))))

    recv_excluded = _excluded_indices(recv_span, buffer=int(buffer), seq_len=len(recv_ids))
    donor_excluded = _excluded_indices(donor_span, buffer=int(donor_buffer), seq_len=len(donor_ids))

    max_len = min(len(donor_ids), len(recv_ids))
    eligible = [i for i in range(max_len) if i not in recv_excluded and i not in donor_excluded]
    if len(eligible) < k:
        return None

    if position_window is not None and int(position_window) >= 0:
        w = int(position_window)
        max_w = int(max_len)
        while True:
            in_window = [i for i in eligible if abs(int(i) - center) <= w]
            if len(in_window) >= k:
                eligible = in_window
                break
            if w >= max_w:
                break
            w = min(max_w, max(w * 2, w + 1))

    eligible_match = [i for i in eligible if int(donor_ids[i]) == int(recv_ids[i])]
    candidates = eligible_match if len(eligible_match) >= k else eligible
    candidates = list(candidates)
    rng.shuffle(candidates)
    candidates.sort(key=lambda i: abs(int(i) - center))
    chosen = candidates[:k]
    return [(int(i), int(i)) for i in chosen]


def _select_offtarget_pairs_with_fallback(
    *,
    donor_ids: Sequence[int],
    recv_ids: Sequence[int],
    donor_span: Sequence[int],
    recv_span: Sequence[int],
    k: int,
    buffer: int,
    donor_buffer: int,
    position_window: Optional[int] = None,
    rng: random.Random,
) -> Tuple[Optional[List[Tuple[int, int]]], str]:
    """
    Return (pairs, strategy) where strategy is one of:
      - "matched_token": token-ID matched off-target pairs (q_donor, p_recv)
      - "same_index": same-index off-target pairs (q == p) with requested buffers
      - "same_index_relaxed": same-index off-target pairs with buffers relaxed to 0 (only if needed)
      - "missing": no valid control pairs found
    """
    pairs = _select_offtarget_pairs(
        donor_ids=donor_ids,
        recv_ids=recv_ids,
        donor_span=donor_span,
        recv_span=recv_span,
        k=k,
        buffer=buffer,
        donor_buffer=donor_buffer,
        position_window=position_window,
        rng=rng,
    )
    if pairs is not None:
        return pairs, "matched_token"

    pairs = _select_offtarget_pairs_same_index(
        donor_ids=donor_ids,
        recv_ids=recv_ids,
        donor_span=donor_span,
        recv_span=recv_span,
        k=k,
        buffer=buffer,
        donor_buffer=donor_buffer,
        position_window=position_window,
        rng=rng,
    )
    if pairs is not None:
        return pairs, "same_index"

    # If the prompts are too short for the requested buffer window, relax buffers to avoid
    # evaluating specificity on a biased subset with available off-target positions.
    pairs = _select_offtarget_pairs_same_index(
        donor_ids=donor_ids,
        recv_ids=recv_ids,
        donor_span=donor_span,
        recv_span=recv_span,
        k=k,
        buffer=0,
        donor_buffer=0,
        position_window=position_window,
        rng=rng,
    )
    if pairs is not None:
        return pairs, "same_index_relaxed"

    return None, "missing"


@torch.no_grad()
def compute_cpt_context_swap_patching(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    items: List[DisambPair],
    device: torch.device,
    *,
    layers: Optional[List[int]] = None,
    normalize_by_length: bool = True,
    require_token_id_match: bool = True,
    ci: float = 0.95,
    bootstrap_n: int = 1000,
    bootstrap_seed: int = 42,
) -> Dict[str, Any]:
    """
    CPT-style causal test: patch the token-in-context state at the ambiguous target token.

    For each DisambPair, patch A->B and B->A across specified layers, and measure
    mean shift toward the donor expected label.
    """
    if layers is None:
        from ..interventions.activation_patching import get_num_layers

        n_layers = get_num_layers(model)
        layers = list(range(int(n_layers)))

    per_layer_sum = {int(l): 0.0 for l in layers}
    per_layer_n = {int(l): 0 for l in layers}
    sham_per_layer_sum = {int(l): 0.0 for l in layers}
    sham_per_layer_n = {int(l): 0 for l in layers}

    max_effects: List[float] = []
    max_layers: List[int] = []
    max_effects_norm: List[float] = []
    max_flips: List[float] = []

    sham_max_effects: List[float] = []

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

            # Baseline receiver label scores (unpatched).
            base_scores = score_labels_next_continuations(
                model, tokenizer, recv.prompt, it.choices, device, normalize_by_length=normalize_by_length
            )

            donor_ids = _encode_prompt(tokenizer, donor.prompt, device=device)
            recv_ids = _encode_prompt(tokenizer, recv.prompt, device=device)

            donor_hs = get_hidden_states(model, donor_ids)
            recv_hs = get_hidden_states(model, recv_ids)

            best_for_direction = None
            best_layer_for_direction = None
            best_sham_for_direction = None
            best_flip_for_direction = None
            best_norm_effect_for_direction = None

            base_pred = base_scores.argmax_label()
            base_margin_for_expected = _margin(base_scores, expected=donor_expected)

            for layer in layers:
                # Hidden states indexing: hs[0]=embeddings, hs[layer+1]=post-block(layer)
                replacement = donor_hs[layer + 1][0, donor_span, :].detach()
                patched_scores = score_labels_next_continuations_patched(
                    model,
                    tokenizer,
                    recv.prompt,
                    it.choices,
                    device,
                    patch_site=PatchSpanSite(layer=int(layer), token_indices=tuple(recv_span)),
                    replacement=replacement,
                    normalize_by_length=normalize_by_length,
                )

                sham_replacement = recv_hs[layer + 1][0, recv_span, :].detach()
                sham_scores = score_labels_next_continuations_patched(
                    model,
                    tokenizer,
                    recv.prompt,
                    it.choices,
                    device,
                    patch_site=PatchSpanSite(layer=int(layer), token_indices=tuple(recv_span)),
                    replacement=sham_replacement,
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

                per_layer_sum[int(layer)] += effect
                per_layer_n[int(layer)] += 1
                sham_per_layer_sum[int(layer)] += sham_effect
                sham_per_layer_n[int(layer)] += 1

                if best_for_direction is None or effect > best_for_direction:
                    best_for_direction = effect
                    best_layer_for_direction = int(layer)
                    best_flip_for_direction = flipped
                    best_norm_effect_for_direction = norm_effect
                if best_sham_for_direction is None or sham_effect > best_sham_for_direction:
                    best_sham_for_direction = sham_effect

            if best_for_direction is not None and best_layer_for_direction is not None:
                max_effects.append(float(best_for_direction))
                max_layers.append(int(best_layer_for_direction))
                if best_flip_for_direction is not None:
                    max_flips.append(float(best_flip_for_direction))
                if best_norm_effect_for_direction is not None:
                    max_effects_norm.append(float(best_norm_effect_for_direction))
            if best_sham_for_direction is not None:
                sham_max_effects.append(float(best_sham_for_direction))

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
        "n_directions_total": int(n_total_directions),
        "n_directions_patched": int(len(max_effects)),
        "n_directions_skipped_misaligned": int(n_skipped_misaligned),
        **{f"effect_layer_{l}": per_layer_sum[l] / max(1, per_layer_n[l]) for l in layers},
        **{f"sham_effect_layer_{l}": sham_per_layer_sum[l] / max(1, sham_per_layer_n[l]) for l in layers},
    }


@torch.no_grad()
def compute_cpt_target_specificity_control(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    items: List[DisambPair],
    device: torch.device,
    *,
    layer: Optional[int] = None,
    depth_frac: float = 0.25,
    buffer: int = 2,
    donor_buffer: Optional[int] = None,
    position_window: int = 8,
    selection_seed: int = 0,
    normalize_by_length: bool = True,
    require_token_id_match: bool = True,
    ci: float = 0.95,
    bootstrap_n: int = 1000,
    bootstrap_seed: int = 42,
) -> Dict[str, Any]:
    """
    Target non-specificity control at a fixed layer ℓ*.

    For each donor→receiver direction, compare:
      - target-span patch: receiver target span ← donor target span (at ℓ*)
      - off-target control: receiver positions p_i ← donor positions q_i, with k = len(target span)
        and buffer exclusions. Prefer matched-token pairs (donor_ids[q_i] == recv_ids[p_i]); fall back
        to same-index off-target patching (q_i == p_i) when token overlap is sparse.

    The off-target pairs are chosen deterministically via seeded RNG and not optimized for effect size.
    """
    if donor_buffer is None:
        donor_buffer = int(buffer)
    if buffer < 0 or donor_buffer < 0:
        raise ValueError("buffer and donor_buffer must be >= 0")
    if position_window < 0:
        raise ValueError("position_window must be >= 0")
    if not (0.0 <= depth_frac <= 1.0):
        raise ValueError("depth_frac must be in [0, 1]")

    from ..interventions.activation_patching import get_num_layers

    n_layers = int(get_num_layers(model))
    if n_layers < 1:
        raise ValueError("Model must have at least one layer")
    if layer is None:
        layer = int(round(float(depth_frac) * float(n_layers - 1)))
    if layer < 0 or layer >= n_layers:
        raise ValueError(f"layer {layer} out of range [0, {n_layers})")

    def _safe_bootstrap(values: Sequence[float]) -> Tuple[float, float, float]:
        if len(values) < 1:
            nan = float("nan")
            return nan, nan, nan
        return bootstrap_ci(values, n_bootstrap=bootstrap_n, ci=ci, seed=bootstrap_seed)

    abs_target_effects: List[float] = []
    abs_ctrl_effects: List[float] = []
    abs_deltas: List[float] = []
    win_samples_abs: List[float] = []
    signed_target_effects: List[float] = []
    signed_ctrl_effects: List[float] = []
    signed_deltas: List[float] = []
    win_samples_signed: List[float] = []

    signed_deltas_by_strategy: Dict[str, List[float]] = {"matched_token": [], "same_index": [], "same_index_relaxed": []}
    win_signed_by_strategy: Dict[str, List[float]] = {"matched_token": [], "same_index": [], "same_index_relaxed": []}

    n_total_directions = 0
    n_skipped_misaligned = 0
    n_target_patched = 0
    n_ctrl_missing = 0
    n_ctrl_patched_matched_token = 0
    n_ctrl_patched_same_index = 0
    n_ctrl_patched_same_index_relaxed = 0

    for it in items:
        for direction_idx, (donor, recv) in enumerate(((it.a, it.b), (it.b, it.a))):
            n_total_directions += 1
            direction_tag = "a_to_b" if direction_idx == 0 else "b_to_a"
            donor_expected = donor.expected_label

            donor_span, donor_token_ids = token_span_for_substring(
                tokenizer, donor.prompt, it.target, it.target_occurrence
            )
            recv_span, recv_token_ids = token_span_for_substring(tokenizer, recv.prompt, it.target, it.target_occurrence)

            if len(donor_span) != len(recv_span) or (require_token_id_match and donor_token_ids != recv_token_ids):
                n_skipped_misaligned += 1
                continue

            n_target_patched += 1

            base_scores = score_labels_next_continuations(
                model, tokenizer, recv.prompt, it.choices, device, normalize_by_length=normalize_by_length
            )
            base_margin = _margin(base_scores, expected=donor_expected)

            donor_prompt_ids = _encode_prompt(tokenizer, donor.prompt, device=device)
            recv_prompt_ids = _encode_prompt(tokenizer, recv.prompt, device=device)

            donor_hs = get_hidden_states(model, donor_prompt_ids)

            replacement_target = donor_hs[layer + 1][0, donor_span, :].detach()
            target_scores = score_labels_next_continuations_patched(
                model,
                tokenizer,
                recv.prompt,
                it.choices,
                device,
                patch_site=PatchSpanSite(layer=int(layer), token_indices=tuple(recv_span)),
                replacement=replacement_target,
                normalize_by_length=normalize_by_length,
            )
            target_margin = _margin(target_scores, expected=donor_expected)
            e_target = float(target_margin - base_margin)

            donor_ids_list = [int(x) for x in donor_prompt_ids[0].tolist()]
            recv_ids_list = [int(x) for x in recv_prompt_ids[0].tolist()]
            rng_seed = _stable_u32_seed(it.pair_id, direction_tag, base_seed=int(selection_seed))
            rng = random.Random(int(rng_seed))

            pairs, strategy = _select_offtarget_pairs_with_fallback(
                donor_ids=donor_ids_list,
                recv_ids=recv_ids_list,
                donor_span=donor_span,
                recv_span=recv_span,
                k=len(recv_span),
                buffer=int(buffer),
                donor_buffer=int(donor_buffer),
                position_window=int(position_window),
                rng=rng,
            )
            if pairs is None:
                n_ctrl_missing += 1
                continue
            if strategy == "matched_token":
                n_ctrl_patched_matched_token += 1
            elif strategy == "same_index":
                n_ctrl_patched_same_index += 1
            elif strategy == "same_index_relaxed":
                n_ctrl_patched_same_index_relaxed += 1

            pairs_sorted = sorted(pairs, key=lambda qp: int(qp[1]))
            q_list = [q for (q, _p) in pairs_sorted]
            p_list = [p for (_q, p) in pairs_sorted]
            replacement_ctrl = donor_hs[layer + 1][0, q_list, :].detach()
            ctrl_scores = score_labels_next_continuations_patched(
                model,
                tokenizer,
                recv.prompt,
                it.choices,
                device,
                patch_site=PatchSpanSite(layer=int(layer), token_indices=tuple(p_list)),
                replacement=replacement_ctrl,
                normalize_by_length=normalize_by_length,
            )
            ctrl_margin = _margin(ctrl_scores, expected=donor_expected)
            e_ctrl = float(ctrl_margin - base_margin)

            signed_target_effects.append(float(e_target))
            signed_ctrl_effects.append(float(e_ctrl))
            signed_delta = float(e_target - e_ctrl)
            signed_deltas.append(signed_delta)
            win_signed = float(e_target > e_ctrl)
            win_samples_signed.append(win_signed)
            if strategy in signed_deltas_by_strategy:
                signed_deltas_by_strategy[strategy].append(signed_delta)
                win_signed_by_strategy[strategy].append(win_signed)
            abs_target = float(abs(e_target))
            abs_ctrl = float(abs(e_ctrl))
            abs_target_effects.append(abs_target)
            abs_ctrl_effects.append(abs_ctrl)
            abs_deltas.append(float(abs_target - abs_ctrl))
            win_samples_abs.append(float(abs_target > abs_ctrl))

    mean_abs_target, mean_abs_target_lo, mean_abs_target_hi = _safe_bootstrap(abs_target_effects)
    mean_abs_ctrl, mean_abs_ctrl_lo, mean_abs_ctrl_hi = _safe_bootstrap(abs_ctrl_effects)
    mean_abs_delta, mean_abs_delta_lo, mean_abs_delta_hi = _safe_bootstrap(abs_deltas)
    win_rate, win_lo, win_hi = _safe_bootstrap(win_samples_abs)
    mean_signed_target, mean_signed_target_lo, mean_signed_target_hi = _safe_bootstrap(signed_target_effects)
    mean_signed_ctrl, mean_signed_ctrl_lo, mean_signed_ctrl_hi = _safe_bootstrap(signed_ctrl_effects)
    mean_signed_delta, mean_signed_delta_lo, mean_signed_delta_hi = _safe_bootstrap(signed_deltas)
    win_rate_signed, win_signed_lo, win_signed_hi = _safe_bootstrap(win_samples_signed)

    mt_delta, mt_delta_lo, mt_delta_hi = _safe_bootstrap(signed_deltas_by_strategy["matched_token"])
    mt_win, mt_win_lo, mt_win_hi = _safe_bootstrap(win_signed_by_strategy["matched_token"])
    si_delta, si_delta_lo, si_delta_hi = _safe_bootstrap(signed_deltas_by_strategy["same_index"])
    si_win, si_win_lo, si_win_hi = _safe_bootstrap(win_signed_by_strategy["same_index"])
    sir_delta, sir_delta_lo, sir_delta_hi = _safe_bootstrap(signed_deltas_by_strategy["same_index_relaxed"])
    sir_win, sir_win_lo, sir_win_hi = _safe_bootstrap(win_signed_by_strategy["same_index_relaxed"])

    n_ctrl_patched = int(len(abs_deltas))
    n_target_patched_i = int(n_target_patched)
    valid_rate = float(float(n_ctrl_patched) / max(1, n_target_patched_i))
    return {
        "layer": float(int(layer)),
        "depth_frac": float(depth_frac),
        "buffer": float(int(buffer)),
        "donor_buffer": float(int(donor_buffer)),
        "position_window": float(int(position_window)),
        "selection_seed": float(int(selection_seed)),
        "mean_abs_target_effect": mean_abs_target,
        "mean_abs_target_effect_ci_low": mean_abs_target_lo,
        "mean_abs_target_effect_ci_high": mean_abs_target_hi,
        "mean_abs_ctrl_effect": mean_abs_ctrl,
        "mean_abs_ctrl_effect_ci_low": mean_abs_ctrl_lo,
        "mean_abs_ctrl_effect_ci_high": mean_abs_ctrl_hi,
        "mean_abs_delta": mean_abs_delta,
        "mean_abs_delta_ci_low": mean_abs_delta_lo,
        "mean_abs_delta_ci_high": mean_abs_delta_hi,
        "win_rate": win_rate,
        "win_rate_ci_low": win_lo,
        "win_rate_ci_high": win_hi,
        "mean_signed_delta": mean_signed_delta,
        "mean_signed_delta_ci_low": mean_signed_delta_lo,
        "mean_signed_delta_ci_high": mean_signed_delta_hi,
        "win_rate_signed": win_rate_signed,
        "win_rate_signed_ci_low": win_signed_lo,
        "win_rate_signed_ci_high": win_signed_hi,
        "mean_signed_delta_matched_token": mt_delta,
        "mean_signed_delta_matched_token_ci_low": mt_delta_lo,
        "mean_signed_delta_matched_token_ci_high": mt_delta_hi,
        "win_rate_signed_matched_token": mt_win,
        "win_rate_signed_matched_token_ci_low": mt_win_lo,
        "win_rate_signed_matched_token_ci_high": mt_win_hi,
        "mean_signed_delta_same_index": si_delta,
        "mean_signed_delta_same_index_ci_low": si_delta_lo,
        "mean_signed_delta_same_index_ci_high": si_delta_hi,
        "win_rate_signed_same_index": si_win,
        "win_rate_signed_same_index_ci_low": si_win_lo,
        "win_rate_signed_same_index_ci_high": si_win_hi,
        "mean_signed_delta_same_index_relaxed": sir_delta,
        "mean_signed_delta_same_index_relaxed_ci_low": sir_delta_lo,
        "mean_signed_delta_same_index_relaxed_ci_high": sir_delta_hi,
        "win_rate_signed_same_index_relaxed": sir_win,
        "win_rate_signed_same_index_relaxed_ci_low": sir_win_lo,
        "win_rate_signed_same_index_relaxed_ci_high": sir_win_hi,
        "mean_signed_target_effect": mean_signed_target,
        "mean_signed_target_effect_ci_low": mean_signed_target_lo,
        "mean_signed_target_effect_ci_high": mean_signed_target_hi,
        "mean_signed_ctrl_effect": mean_signed_ctrl,
        "mean_signed_ctrl_effect_ci_low": mean_signed_ctrl_lo,
        "mean_signed_ctrl_effect_ci_high": mean_signed_ctrl_hi,
        "n_directions_total": int(n_total_directions),
        "n_directions_target_patched": int(n_target_patched_i),
        "n_directions_skipped_misaligned": int(n_skipped_misaligned),
        "n_directions_ctrl_patched": int(n_ctrl_patched),
        "n_directions_ctrl_patched_matched_token": int(n_ctrl_patched_matched_token),
        "n_directions_ctrl_patched_same_index": int(n_ctrl_patched_same_index),
        "n_directions_ctrl_patched_same_index_relaxed": int(n_ctrl_patched_same_index_relaxed),
        "n_directions_ctrl_missing": int(n_ctrl_missing),
        "valid_control_rate": valid_rate,
    }
