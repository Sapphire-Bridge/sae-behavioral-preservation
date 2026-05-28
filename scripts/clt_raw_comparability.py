"""
CLT/SAE writeback comparability experiment (raw vs. CLT/SAE patching).

Compares raw activation patching against CLT writeback patching on
disambiguation pairs, with identity and projection baselines as controls.

Arm legend (column suffix in output CSV: effect_<ARM>):

  Causal arms (compare effects):
    A  = raw swap          : replace recv span with donor span activations
    B  = raw delta-swap    : replace with recv + (donor - recv)
                             — algebraically equal to A; sanity gate only
    C  = CLT delta_1decode : CLT writeback patch, decode_strategy=delta_1decode
    D  = CLT safe_2decode  : CLT writeback patch, decode_strategy=safe_2decode

  Equivalence controls (paired with C, D; should give identical effect to C/D):
    Cp = raw patch with replacement = recv + clt_delta_C
    Dp = raw patch with replacement = recv + clt_delta_D

  Identity baselines (replace span with itself; effect should be ~0):
    RI = raw  identity : replacement = raw_recv
    CI = CLT  identity : CLT patch with policy = recv_latents (no swap)

  Optional projection baselines (rank matched to active CLT latent count):
    PRJ_PCA      : project raw_delta onto leading PCA components (pair-disjoint fit)
    PRJ_RAND_s*  : project raw_delta onto random orthonormal basis (matched rank,
                   one arm per seed in --random_projection_seeds)

  Optional faithfulness / stress controls:
    STRESS_RECON : patch with recv + (recon_donor - recon_recv)
    STRESS_RESID : patch with recv + (raw_delta - recon_delta)
                   Activation deltas are additive by construction:
                   raw_delta = recon_delta + resid_delta, checked by
                   stress_delta_additivity_rel_err. d_STRESS_SUM_A reports
                   whether the two behavioral effects add back to the raw
                   behavioral effect.

Invariant gates (per row; diagnostic hygiene — NOT the analysis-inclusion
criterion). analysis_included is determined by limitation_analysis_included()
in scripts/limitation_analysis_policy.py, which requires all_arms_success and
finite effect_A/effect_C but intentionally excludes gate thresholds to avoid
device-dependent inclusion boundaries.
  gate_activation : ||raw_swap - raw_delta_replacement|| / ||...|| < tol
  gate_margin     : |effect_A - effect_B| < tol
  gate_score      : |delta_score_{exp,other}_A - delta_score_{exp,other}_B| < tol
  gate_clt_equiv  : |effect_C - effect_Cp| < tol  AND  |effect_D - effect_Dp| < tol
  gate_identity   : |effect_RI| < tol  AND  |effect_CI| < tol

Pipeline:
  1. Load model + tokenizer + CLT bundles (one per layer in --layers).
  2. Optionally fit PCA bundles (global and/or leave-one-pair-out per pair).
  3. Optionally build random orthonormal bases per (layer, seed).
  4. For each disambiguation pair x direction (a_to_b, b_to_a):
       align donor/receiver target token spans;
       compute base scores and primary log-odds metadata;
       for each layer:
         build arm replacements -> evaluate all arms -> compute gates,
         primary log-odds cancellation, and decomposition deltas (A vs. C).
  5. Aggregate per layer with pair-cluster bootstrap; write CSV + summary JSON.

Outputs:
  --out_csv  : one row per (pair, direction, layer); ~250 columns
  --out_json : run config, per-layer aggregates, fail counts, provenance

See parse_args() for the full CLI reference and main() for the pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aom.data.loaders import load_disamb_pairs
from aom.interventions.activation_patching import PatchSpanSite, get_block_outputs, get_num_layers, forward_with_patched_block_output_span
from aom.interventions.clt_adapter import CLTInputTransform, CLTPatchConfig, reconstruct_with_error_preservation
from aom.interventions.clt_loader import load_clt
from aom.interventions.clt_patch import ReplaceLatentsAtIndicesPolicy, forward_with_clt_latent_patching_span
from aom.metrics.primary_logodds import (
    PRIMARY_LOGODDS_RESIDUAL_TOL_DEFAULT,
    choices_token_ids_from_strings,
    evaluate_primary_applicability,
    safe_log_prob_mass,
)
from aom.models.loader import load_causal_lm
from aom.token_spans import token_span_for_substring
from aom.utils import get_best_device, get_logprob_computation_config, set_seed
from scripts.limitation_analysis_policy import (
    limitation_analysis_included,
    limitation_analysis_policy_metadata,
)


logger = logging.getLogger(__name__)


def _analysis_logprob_dtype_for_device(device: torch.device | str) -> torch.dtype:
    device_type = device.type if isinstance(device, torch.device) else str(device)
    # MPS lacks float64 kernels for this path; use float64 everywhere else for tighter identities.
    return torch.float32 if device_type == "mps" else torch.float64


def _normalize_dtype_name(value: str | torch.dtype | None) -> str:
    if value is None:
        return ""
    if isinstance(value, torch.dtype):
        value = str(value)
    s = str(value).strip().lower().replace("torch.", "")
    aliases = {
        "float": "float32",
        "half": "float16",
        "double": "float64",
        "bf16": "bfloat16",
    }
    return aliases.get(s, s)


def _parse_int_list(arg: str) -> List[int]:
    out: List[int] = []
    for part in str(arg).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError("Expected at least one integer in --layers")
    return out


def _logmeanexp(xs: Sequence[float]) -> float:
    if len(xs) < 1:
        raise ValueError("logmeanexp requires at least one value")
    t = torch.tensor(list(xs), dtype=torch.float64)
    return float(torch.logsumexp(t, dim=0) - math.log(len(xs)))


def _argmax_label(scores: Mapping[str, float]) -> str:
    return max(scores.items(), key=lambda kv: kv[1])[0]


def _margin(scores: Mapping[str, float], expected: str) -> float:
    exp = float(scores[expected])
    best_other = max(float(v) for k, v in scores.items() if str(k) != str(expected))
    return float(exp - best_other)


def _best_other_label(scores: Mapping[str, float], expected: str) -> str:
    return max(((k, v) for k, v in scores.items() if str(k) != str(expected)), key=lambda kv: kv[1])[0]


def _encode_prompt(tokenizer, text: str, device: torch.device) -> torch.Tensor:
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    return enc["input_ids"].to(device)


def _encode_cont(tokenizer, text: str, device: torch.device) -> torch.Tensor:
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    return enc["input_ids"].to(device)


@torch.inference_mode()
def _next_token_logits_probs(
    *,
    prompt_ids: torch.Tensor,
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = forward_fn(prompt_ids)
    if logits.ndim != 3 or int(logits.size(0)) != 1 or int(logits.size(1)) < 1:
        raise ValueError("Forward output must have shape (1, seq, vocab) with seq>=1")
    next_dtype = _analysis_logprob_dtype_for_device(logits.device)
    next_logits = logits[:, -1, :].to(dtype=next_dtype)[0]
    next_probs = torch.softmax(next_logits, dim=-1)
    return next_logits, next_probs


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


def _safe_err(e: Exception) -> str:
    s = str(e).strip()
    if not s:
        s = e.__class__.__name__
    s = s.replace("\n", " ")
    return s[:400]


def _git_commit(repo_root: Path) -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
        )
    except Exception:
        return ""


def _sign(x: float, eps: float = 1e-12) -> int:
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def _infer_clt_device_dtype(clt: torch.nn.Module) -> Tuple[torch.device, torch.dtype]:
    p = next(clt.parameters(), None)
    if p is not None:
        return p.device, p.dtype
    w_dec = getattr(clt, "W_dec", None)
    if isinstance(w_dec, torch.Tensor):
        return w_dec.device, w_dec.dtype
    return torch.device("cpu"), torch.float32


def _norm_ratio(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    return float(torch.norm(a).item() / (torch.norm(b).item() + float(eps)))


def _cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    aa = a.reshape(-1)
    bb = b.reshape(-1)
    na = float(torch.norm(aa).item())
    nb = float(torch.norm(bb).item())
    if na <= eps or nb <= eps:
        return float("nan")
    return float(torch.dot(aa, bb).item() / (na * nb + eps))


@torch.inference_mode()
def _score_choices_with_forward(
    *,
    tokenizer,
    prompt_ids: torch.Tensor,
    choices: Mapping[str, List[str]],
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
    normalize_by_length: bool,
) -> Dict[str, float]:
    _configured_dtype, strict_finite = get_logprob_computation_config()
    scores: Dict[str, float] = {}
    for label, continuations in choices.items():
        if len(continuations) < 1:
            raise ValueError(f"Empty continuation list for label={label}")
        vals: List[float] = []
        for cont in continuations:
            cont_ids = _encode_cont(tokenizer, str(cont), device=device)
            full_ids = torch.cat([prompt_ids, cont_ids], dim=1)
            logits = forward_fn(full_ids)
            p_len = int(prompt_ids.size(1))
            c_len = int(cont_ids.size(1))
            logits_slice = logits[:, p_len - 1 : p_len + c_len - 1, :].to(
                dtype=_analysis_logprob_dtype_for_device(logits.device)
            )
            log_probs = torch.log_softmax(logits_slice, dim=-1)
            gathered = log_probs.gather(2, cont_ids.unsqueeze(-1)).squeeze(-1)
            if not torch.isfinite(gathered).all():
                if strict_finite:
                    raise FloatingPointError("Non-finite log-probability in scoring.")
                gathered = torch.where(torch.isfinite(gathered), gathered, torch.full_like(gathered, -1e9))
            lp = gathered.mean(dim=1) if normalize_by_length else gathered.sum(dim=1)
            vals.append(float(lp.item()))
        scores[str(label)] = _logmeanexp(vals)
    return scores


@torch.inference_mode()
def _logits_slice_for_continuation(
    *,
    tokenizer,
    prompt_ids: torch.Tensor,
    continuation: str,
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cont_ids = _encode_cont(tokenizer, continuation, device=device)
    full_ids = torch.cat([prompt_ids, cont_ids], dim=1)
    logits = forward_fn(full_ids)
    p_len = int(prompt_ids.size(1))
    c_len = int(cont_ids.size(1))
    logits_slice = logits[:, p_len - 1 : p_len + c_len - 1, :].to(
        dtype=_analysis_logprob_dtype_for_device(logits.device)
    )
    return logits_slice, cont_ids


def _decomp_deltas(base_logits: torch.Tensor, patched_logits: torch.Tensor, cont_ids: torch.Tensor) -> Dict[str, float]:
    """Per-arm logit-change diagnostics over the continuation span.

    Compares patched logits to base logits at the continuation positions and
    returns delta_logit, delta_logz, delta_logprob, KL, and RMS. Used to
    populate decomp_{exp,other}_* columns for arms in DECOMP_ARMS.
    """
    if base_logits.shape != patched_logits.shape:
        raise ValueError("base_logits and patched_logits must have the same shape")
    if cont_ids.ndim != 2 or int(cont_ids.size(0)) != int(base_logits.size(0)):
        raise ValueError("continuation IDs shape mismatch")

    base_logz = torch.logsumexp(base_logits, dim=-1)
    patch_logz = torch.logsumexp(patched_logits, dim=-1)
    base_logprobs = torch.log_softmax(base_logits, dim=-1)
    patch_logprobs = torch.log_softmax(patched_logits, dim=-1)
    base_probs = torch.softmax(base_logits, dim=-1)

    gather_idx = cont_ids.unsqueeze(-1)
    base_target_logit = base_logits.gather(2, gather_idx).squeeze(-1)
    patch_target_logit = patched_logits.gather(2, gather_idx).squeeze(-1)
    base_target_logprob = base_logprobs.gather(2, gather_idx).squeeze(-1)
    patch_target_logprob = patch_logprobs.gather(2, gather_idx).squeeze(-1)

    delta_logit = patch_target_logit - base_target_logit
    delta_logz = patch_logz - base_logz
    delta_logprob = patch_target_logprob - base_target_logprob

    kl_base_patch = (base_probs * (base_logprobs - patch_logprobs)).sum(dim=-1)
    rms_logit_change = torch.sqrt(torch.mean((patched_logits - base_logits) ** 2, dim=-1))

    return {
        "delta_logit_target_mean": float(delta_logit.mean().item()),
        "delta_logz_mean": float(delta_logz.mean().item()),
        "delta_logprob_target_mean": float(delta_logprob.mean().item()),
        "kl_base_to_patch_mean": float(kl_base_patch.mean().item()),
        "rms_logit_change_mean": float(rms_logit_change.mean().item()),
    }


def _cluster_bootstrap_mean(
    *,
    rows: Sequence[Dict[str, Any]],
    key: str,
    pair_key: str,
    n_bootstrap: int,
    ci: float,
    seed: int,
) -> Tuple[float, float, float, int, int]:
    """Pair-cluster bootstrap CI for a scalar metric.

    Resamples *pair IDs* (not individual rows) to respect the a_to_b /
    b_to_a within-pair dependency structure.
    Returns (mean, ci_low, ci_high, n_pairs, n_vals).
    """
    pair_to_vals: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        value = r.get(key, float("nan"))
        if not isinstance(value, (int, float)):
            continue
        v = float(value)
        if not math.isfinite(v):
            continue
        pair_to_vals[str(r[pair_key])].append(v)
    if not pair_to_vals:
        return float("nan"), float("nan"), float("nan"), 0, 0

    pair_means: Dict[str, float] = {
        str(pid): float(sum(vals) / len(vals)) for pid, vals in pair_to_vals.items() if len(vals) > 0
    }
    if not pair_means:
        return float("nan"), float("nan"), float("nan"), 0, 0

    mean_val = float(sum(pair_means.values()) / len(pair_means))
    pairs = list(pair_means.keys())
    n_pairs = len(pairs)
    n_vals = int(sum(len(vals) for vals in pair_to_vals.values()))

    rng = random.Random(int(seed))
    boots: List[float] = []
    for _ in range(int(n_bootstrap)):
        sampled_means: List[float] = []
        for _ in range(n_pairs):
            pid = pairs[rng.randrange(n_pairs)]
            sampled_means.append(float(pair_means[pid]))
        if sampled_means:
            boots.append(float(sum(sampled_means) / len(sampled_means)))
    if not boots:
        return mean_val, float("nan"), float("nan"), n_pairs, n_vals
    boots.sort()
    alpha = (1.0 - float(ci)) / 2.0
    lo_idx = max(0, min(len(boots) - 1, int(alpha * len(boots))))
    hi_idx = max(0, min(len(boots) - 1, int((1.0 - alpha) * len(boots)) - 1))
    return mean_val, float(boots[lo_idx]), float(boots[hi_idx]), n_pairs, n_vals


def _cluster_bootstrap_ratio(
    *,
    rows: Sequence[Dict[str, Any]],
    num_key: str,
    den_key: str,
    pair_key: str,
    n_bootstrap: int,
    ci: float,
    seed: int,
    den_eps: float,
) -> Tuple[float, float, float, int]:
    """Pair-cluster bootstrap CI for a ratio of two scalar metrics.

    Same resampling strategy as _cluster_bootstrap_mean (pair-level).
    Returns (point_estimate, ci_low, ci_high, n_pairs).
    """
    pair_to_num_den: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for r in rows:
        n = r.get(num_key, float("nan"))
        d = r.get(den_key, float("nan"))
        if not isinstance(n, (int, float)) or not isinstance(d, (int, float)):
            continue
        nf = float(n)
        df = float(d)
        if not (math.isfinite(nf) and math.isfinite(df)):
            continue
        pair_to_num_den[str(r[pair_key])].append((nf, df))
    if not pair_to_num_den:
        return float("nan"), float("nan"), float("nan"), 0

    pair_num_mean: Dict[str, float] = {}
    pair_den_mean: Dict[str, float] = {}
    for pid, vals in pair_to_num_den.items():
        if len(vals) < 1:
            continue
        pair_num_mean[str(pid)] = float(sum(v[0] for v in vals) / len(vals))
        pair_den_mean[str(pid)] = float(sum(v[1] for v in vals) / len(vals))
    if not pair_num_mean:
        return float("nan"), float("nan"), float("nan"), 0

    pairs = list(pair_num_mean.keys())
    n_pairs = len(pairs)
    mean_num = float(sum(pair_num_mean.values()) / len(pair_num_mean))
    mean_den = float(sum(pair_den_mean.values()) / len(pair_den_mean))
    point = float("nan") if abs(mean_den) <= float(den_eps) else float(mean_num / mean_den)

    rng = random.Random(int(seed))
    boots: List[float] = []
    for _ in range(int(n_bootstrap)):
        sampled_num: List[float] = []
        sampled_den: List[float] = []
        for _ in range(n_pairs):
            pid = pairs[rng.randrange(n_pairs)]
            sampled_num.append(float(pair_num_mean[pid]))
            sampled_den.append(float(pair_den_mean[pid]))
        if not sampled_num:
            continue
        b_num = float(sum(sampled_num) / len(sampled_num))
        b_den = float(sum(sampled_den) / len(sampled_den))
        if abs(b_den) <= float(den_eps):
            continue
        boots.append(float(b_num / b_den))
    if not boots:
        return point, float("nan"), float("nan"), n_pairs
    boots.sort()
    alpha = (1.0 - float(ci)) / 2.0
    lo_idx = max(0, min(len(boots) - 1, int(alpha * len(boots))))
    hi_idx = max(0, min(len(boots) - 1, int((1.0 - alpha) * len(boots)) - 1))
    return point, float(boots[lo_idx]), float(boots[hi_idx]), n_pairs


def _compute_clt_writeback_delta(
    *,
    clt,
    transform: CLTInputTransform,
    recv_slice: torch.Tensor,
    recv_latents: torch.Tensor,
    donor_latents: torch.Tensor,
    decode_strategy: str,
) -> torch.Tensor:
    """Return the residual-stream delta induced by a CLT writeback patch.

    delta_1decode : decode(donor_latents) - decode(recv_latents), inverse-transformed.
    safe_2decode  : full reconstruct_with_error_preservation pass, preserving
                    per-token reconstruction error.
    Output shape matches raw_delta: (span_len, hidden_dim).
    """
    if str(decode_strategy) == "delta_1decode":
        y_hat_t = clt.decode(recv_latents)
        y_patched_t = clt.decode(donor_latents)
        return transform.inverse_delta(y_patched_t - y_hat_t)
    if str(decode_strategy) == "safe_2decode":
        y_prime, _y_hat, _err = reconstruct_with_error_preservation(
            clt=clt,
            receiver_x=recv_slice,
            receiver_y=recv_slice,
            z_prime=donor_latents,
            transform=transform,
        )
        return y_prime - recv_slice
    raise ValueError(f"Unknown decode_strategy={decode_strategy!r}")


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    _ensure_dir(path)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fields = sorted({k for r in rows for k in r.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _aggregate_layer(
    *,
    rows: Sequence[Dict[str, Any]],
    layer: int,
    n_bootstrap: int,
    ci: float,
    seed: int,
    ratio_den_eps: float,
) -> Dict[str, Any]:
    layer_rows = [r for r in rows if int(r["layer"]) == int(layer)]
    out: Dict[str, Any] = {
        "layer": int(layer),
        "n_rows": int(len(layer_rows)),
        "n_pairs": int(len({str(r["pair_id"]) for r in layer_rows})),
    }
    metrics = [
        "effect_A",
        "effect_B",
        "effect_C",
        "effect_D",
        "effect_Cp",
        "effect_Dp",
        "effect_RI",
        "effect_CI",
        "effect_PRJ_PCA",
        "effect_PRJ_RAND_mean",
        "effect_PRJ_RAND_std",
        "effect_STRESS_RECON",
        "effect_STRESS_RESID",
        "d_BA",
        "d_CA",
        "d_DA",
        "d_CCp",
        "d_DDp",
        "d_PRJ_PCA_A",
        "d_PRJ_RAND_mean_A",
        "d_STRESS_RECON_A",
        "d_STRESS_RESID_A",
        "d_STRESS_SUM_A",
        "d_STRESS_RESID_RECON",
        "sign_agree_BA",
        "sign_agree_CA",
        "sign_agree_DA",
        "sign_agree_CCp",
        "sign_agree_DDp",
        "gate_activation_ratio",
        "gate_margin_abs_diff",
        "gate_score_abs_diff_exp",
        "gate_score_abs_diff_other",
        "gate_clt_effect_abs_diff_C",
        "gate_clt_effect_abs_diff_D",
        "gate_identity_abs_effect_raw",
        "gate_identity_abs_effect_clt",
        "raw_delta_norm",
        "raw_delta_norm_small",
        "recon_norm_ratio",
        "clt_delta_norm_ratio_C",
        "clt_delta_norm_ratio_D",
        "clt_delta_cosine_raw_C",
        "clt_delta_cosine_raw_D",
        "proj_pca_delta_norm_ratio",
        "proj_pca_delta_cosine_raw",
        "projection_rank",
        "projection_rank_max",
        "projection_n_tokens_fit",
        "projection_n_pairs_fit",
        "projection_rank_for_controls",
        "n_PRJ_RAND_success",
        "stress_delta_recon_norm_ratio",
        "stress_delta_recon_cosine_raw",
        "stress_delta_resid_norm_ratio",
        "stress_delta_resid_cosine_raw",
        "stress_delta_additivity_rel_err",
        "fidelity_rel_mse",
        "fidelity_rel_l2",
        "fidelity_cosine",
        "active_latent_frac",
        "decomp_exp_delta_logit_target_mean_C",
        "decomp_exp_delta_logz_mean_C",
        "decomp_exp_delta_logprob_target_mean_C",
        "decomp_exp_kl_base_to_patch_mean_C",
        "decomp_exp_rms_logit_change_mean_C",
        "decomp_exp_delta_logit_target_mean_A",
        "decomp_exp_delta_logz_mean_A",
        "decomp_exp_delta_logprob_target_mean_A",
        "decomp_exp_kl_base_to_patch_mean_A",
        "decomp_exp_rms_logit_change_mean_A",
        "primary_logodds_applicable_flag",
        "primary_cancellation_pass_flag",
        "P_E_base",
        "P_O_base",
        "P_rest_base",
        "P_E_patch_A",
        "P_O_patch_A",
        "P_rest_patch_A",
        "P_E_patch_C",
        "P_O_patch_C",
        "P_rest_patch_C",
        "dlogPE_A",
        "dlogPO_A",
        "dlogPE_C",
        "dlogPO_C",
        "dPE_A",
        "dPO_A",
        "dPrest_A",
        "dPE_C",
        "dPO_C",
        "dPrest_C",
        "d_CA_logPE",
        "d_CA_logPO",
        "primary_delta_m_from_logodds_A",
        "primary_delta_m_from_logodds_C",
        "primary_delta_m_residual_A",
        "primary_delta_m_residual_C",
        "primary_base_logodds_residual",
    ]
    dynamic_metrics = sorted(
        {
            str(k)
            for r in layer_rows
            for k in r.keys()
            if str(k).startswith("effect_PRJ_RAND_s")
            or str(k).startswith("delta_score_exp_PRJ_RAND_s")
            or str(k).startswith("delta_score_other_PRJ_RAND_s")
            or str(k).startswith("proj_rand_s")
        }
    )
    metrics.extend(dynamic_metrics)
    for key in metrics:
        m, lo, hi, n_pairs, n_vals = _cluster_bootstrap_mean(
            rows=layer_rows,
            key=key,
            pair_key="pair_id",
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=seed,
        )
        out[f"{key}_mean"] = m
        out[f"{key}_ci_low"] = lo
        out[f"{key}_ci_high"] = hi
        out[f"{key}_n_pairs"] = int(n_pairs)
        out[f"{key}_n_vals"] = int(n_vals)

    for num_key, den_key, name in (("effect_C", "effect_A", "crr_C_over_A"), ("effect_D", "effect_A", "crr_D_over_A")):
        ratio, lo, hi, n_pairs = _cluster_bootstrap_ratio(
            rows=layer_rows,
            num_key=num_key,
            den_key=den_key,
            pair_key="pair_id",
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=seed,
            den_eps=ratio_den_eps,
        )
        out[f"{name}_mean"] = ratio
        out[f"{name}_ci_low"] = lo
        out[f"{name}_ci_high"] = hi
        out[f"{name}_n_pairs"] = int(n_pairs)
    return out


@dataclass(frozen=True)
class LayerCLTBundle:
    clt: torch.nn.Module
    transform: CLTInputTransform
    meta: Any


@dataclass(frozen=True)
class LayerPCABundle:
    components: torch.Tensor
    max_rank: int
    n_tokens_fit: int
    n_pairs_fit: int


@torch.inference_mode()
def _robust_principal_components(mat: torch.Tensor) -> Tuple[torch.Tensor, str]:
    """Fit PCA components via SVD, with float64 and Gram-matrix fallbacks.

    Tries float32 SVD, then float64 SVD, then float64 Gram eigendecomposition.
    Returns (components, backend_name) where components has shape (rank, hidden).
    """
    if mat.ndim != 2:
        raise ValueError(f"PCA fit expects rank-2 matrix, got shape={tuple(mat.shape)}")
    mat_cpu = mat.detach().to(device="cpu")
    if not torch.isfinite(mat_cpu).all():
        raise ValueError("PCA fit matrix contains non-finite values")

    attempts: list[tuple[str, torch.Tensor]] = [
        ("svd_f32", mat_cpu.to(dtype=torch.float32)),
        ("svd_f64", mat_cpu.to(dtype=torch.float64)),
    ]
    failures: list[tuple[str, Exception]] = []
    for name, candidate in attempts:
        try:
            _u, _s, vh = torch.linalg.svd(candidate, full_matrices=False)
            return vh.to(device="cpu", dtype=torch.float32), name
        except Exception as exc:
            failures.append((name, exc))

    try:
        mat_f64 = mat_cpu.to(dtype=torch.float64)
        gram = mat_f64.transpose(0, 1).matmul(mat_f64)
        eigvals, eigvecs = torch.linalg.eigh(gram)
        order = torch.argsort(eigvals, descending=True)
        components = eigvecs[:, order].transpose(0, 1).contiguous()
        keep = eigvals[order] > 0
        if torch.any(keep):
            components = components[keep]
        return components.to(device="cpu", dtype=torch.float32), "gram_eigh_f64"
    except Exception as exc:
        if failures:
            failure_summary = ", ".join(f"{name}: {type(err).__name__}" for name, err in failures)
            raise RuntimeError(
                "PCA fit failed under float32 SVD, float64 SVD, and float64 Gram eigendecomposition "
                f"({failure_summary}; gram_eigh_f64: {type(exc).__name__})"
            ) from exc
        raise RuntimeError("PCA fit failed under float64 Gram eigendecomposition") from exc


def _build_pca_bundle(
    chunks: Sequence[torch.Tensor], *, n_pairs_fit: int
) -> Tuple[Optional[LayerPCABundle], Optional[str]]:
    if not chunks:
        return None, None
    mat = torch.cat(list(chunks), dim=0)
    if mat.ndim != 2 or int(mat.size(0)) < 1:
        return None, None
    components, backend = _robust_principal_components(mat)
    return (
        LayerPCABundle(
            components=components,
            max_rank=int(components.size(0)),
            n_tokens_fit=int(mat.size(0)),
            n_pairs_fit=int(n_pairs_fit),
        ),
        backend,
    )


def _project_rows_onto_components(rows: torch.Tensor, components: torch.Tensor, rank: int) -> torch.Tensor:
    if rows.ndim != 2:
        raise ValueError(f"rows must be rank-2 (tokens, hidden); got shape={tuple(rows.shape)}")
    if components.ndim != 2:
        raise ValueError(
            f"components must be rank-2 (rank_max, hidden); got shape={tuple(components.shape)}"
        )
    if int(rows.size(-1)) != int(components.size(-1)):
        raise ValueError(
            f"Hidden dimension mismatch: rows={int(rows.size(-1))}, components={int(components.size(-1))}"
        )
    k = max(0, min(int(rank), int(components.size(0))))
    if k == 0:
        return torch.zeros_like(rows)
    basis = components[:k, :]
    coeff = rows @ basis.t()
    return coeff @ basis


def _resolve_pca_rank(
    *,
    rank_mode: str,
    fixed_rank: int,
    active_latent_count: int,
    max_rank: int,
) -> int:
    """Determine the PCA projection rank for a single row.

    active_per_row: rank = count(|z_recv| > eps) over the receiver target-span latent tensor,
    capped at max_rank.
    fixed         : rank = args.pca_fixed_rank, capped at max_rank.
    """
    if str(rank_mode) == "active_per_row":
        return max(0, min(int(active_latent_count), int(max_rank)))
    if str(rank_mode) == "fixed":
        return max(0, min(int(fixed_rank), int(max_rank)))
    raise ValueError(f"Unknown PCA rank mode: {rank_mode!r}")


@torch.inference_mode()
def _fit_pca_bundles(
    *,
    model,
    tokenizer,
    items: Sequence[Any],
    layers: Sequence[int],
    device: torch.device,
    require_token_id_match: bool,
) -> Tuple[Dict[int, LayerPCABundle], Dict[int, Dict[str, LayerPCABundle]], Dict[int, Dict[str, Any]]]:
    deltas_by_layer_pair: Dict[int, Dict[str, List[torch.Tensor]]] = {
        int(layer): defaultdict(list) for layer in layers
    }
    skip_counts: Dict[int, int] = defaultdict(int)

    for it in items:
        for donor, recv in ((it.a, it.b), (it.b, it.a)):
            if str(donor.expected_label) not in it.choices:
                continue
            pair_id = str(it.pair_id)
            try:
                donor_span, donor_token_ids = token_span_for_substring(
                    tokenizer, donor.prompt, it.target, it.target_occurrence
                )
                recv_span, recv_token_ids = token_span_for_substring(
                    tokenizer, recv.prompt, it.target, it.target_occurrence
                )
                aligned = len(donor_span) == len(recv_span) and (
                    (not bool(require_token_id_match)) or (donor_token_ids == recv_token_ids)
                )
                if not aligned:
                    for layer in layers:
                        skip_counts[int(layer)] += 1
                    continue
                prompt_ids_donor = _encode_prompt(tokenizer, donor.prompt, device=device)
                prompt_ids_recv = _encode_prompt(tokenizer, recv.prompt, device=device)
                _assert_prompt_span_alignment(prompt_ids_donor, donor_span, donor_token_ids, side_name="donor")
                _assert_prompt_span_alignment(prompt_ids_recv, recv_span, recv_token_ids, side_name="receiver")
                donor_out = get_block_outputs(model, prompt_ids_donor, layers=layers)
                recv_out = get_block_outputs(model, prompt_ids_recv, layers=layers)
                for layer in layers:
                    raw_donor = donor_out[int(layer)][0, donor_span, :].detach()
                    raw_recv = recv_out[int(layer)][0, recv_span, :].detach()
                    if raw_donor.shape != raw_recv.shape or raw_donor.numel() == 0:
                        skip_counts[int(layer)] += 1
                        continue
                    delta = (raw_donor - raw_recv).to(device="cpu", dtype=torch.float32)
                    deltas_by_layer_pair[int(layer)][pair_id].append(delta.reshape(-1, delta.shape[-1]))
            except Exception:
                for layer in layers:
                    skip_counts[int(layer)] += 1
                continue

    global_bundles: Dict[int, LayerPCABundle] = {}
    loo_bundles: Dict[int, Dict[str, LayerPCABundle]] = {}
    fit_meta: Dict[int, Dict[str, Any]] = {}
    for layer in layers:
        l = int(layer)
        pair_chunks_raw = deltas_by_layer_pair[l]
        pair_chunks: Dict[str, torch.Tensor] = {}
        for pair_id, chunks in pair_chunks_raw.items():
            if chunks:
                pair_chunks[str(pair_id)] = torch.cat(chunks, dim=0)
        if not pair_chunks:
            fit_meta[l] = {
                "n_tokens_fit": 0,
                "max_rank": 0,
                "n_pairs_fit": 0,
                "n_skipped_fit_rows": int(skip_counts[l]),
                "n_loo_bundles": 0,
                "global_fit_backend": "",
                "loo_fit_backend_counts": {},
            }
            continue
        global_bundle, global_backend = _build_pca_bundle(list(pair_chunks.values()), n_pairs_fit=len(pair_chunks))
        if global_bundle is None:
            fit_meta[l] = {
                "n_tokens_fit": 0,
                "max_rank": 0,
                "n_pairs_fit": 0,
                "n_skipped_fit_rows": int(skip_counts[l]),
                "n_loo_bundles": 0,
                "global_fit_backend": "",
                "loo_fit_backend_counts": {},
            }
            continue
        global_bundles[l] = global_bundle

        loo_layer: Dict[str, LayerPCABundle] = {}
        loo_token_counts: List[int] = []
        loo_backend_counts: Dict[str, int] = defaultdict(int)
        for excluded_pair in sorted(pair_chunks.keys()):
            train_chunks = [v for pid, v in pair_chunks.items() if pid != excluded_pair]
            loo_bundle, loo_backend = _build_pca_bundle(train_chunks, n_pairs_fit=max(0, len(pair_chunks) - 1))
            if loo_bundle is None:
                continue
            loo_layer[str(excluded_pair)] = loo_bundle
            loo_token_counts.append(int(loo_bundle.n_tokens_fit))
            if loo_backend:
                loo_backend_counts[str(loo_backend)] += 1
        loo_bundles[l] = loo_layer

        logger.info(
            "PCA fit layer=%s global_backend=%s loo_backend_counts=%s",
            l,
            str(global_backend or ""),
            dict(sorted(loo_backend_counts.items())),
        )
        fit_meta[l] = {
            "n_tokens_fit": int(global_bundle.n_tokens_fit),
            "max_rank": int(global_bundle.max_rank),
            "n_pairs_fit": int(global_bundle.n_pairs_fit),
            "n_skipped_fit_rows": int(skip_counts[l]),
            "n_loo_bundles": int(len(loo_layer)),
            "loo_tokens_fit_min": int(min(loo_token_counts)) if loo_token_counts else 0,
            "loo_tokens_fit_max": int(max(loo_token_counts)) if loo_token_counts else 0,
            "global_fit_backend": str(global_backend or ""),
            "loo_fit_backend_counts": {k: int(v) for k, v in sorted(loo_backend_counts.items())},
        }
    return global_bundles, loo_bundles, fit_meta


def _random_orthonormal_rows(*, hidden_dim: int, max_rank: int, seed: int) -> torch.Tensor:
    h = int(hidden_dim)
    k = max(0, min(int(max_rank), h))
    if k == 0:
        return torch.zeros((0, h), dtype=torch.float32)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    rand = torch.randn((h, k), generator=gen, dtype=torch.float32)
    q, _ = torch.linalg.qr(rand, mode="reduced")
    return q.t().contiguous()


# ---------------------------------------------------------------------------
# Arm registry
# Arms always evaluated on every (pair, direction, layer) triple.
# See module docstring for the meaning of each arm label.
# ---------------------------------------------------------------------------
REQUIRED_ARMS: Tuple[str, ...] = ("A", "B", "C", "D", "Cp", "Dp", "RI", "CI")

# Arms for which per-token logit decomposition (KL, RMS, delta_logit) is
# computed. Only A and C are decomposed because the paper's primary claim
# contrasts the raw-swap arm (A) with the CLT writeback arm (C).
# B, D, Cp, Dp, RI, CI are sanity gates or controls, not analysis endpoints.
DECOMP_ARMS: frozenset = frozenset({"A", "C"})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(
        description="Raw-vs-CLT comparability run with A≈B invariant gate, decomposition telemetry, and pair-cluster bootstrap."
    )
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--disamb_path", type=str, default=str(root / "data" / "disamb_pairs.jsonl"))
    p.add_argument("--clt_repo", type=str, required=True)
    p.add_argument("--layers", type=str, default="4,8,12")
    p.add_argument("--clt_width", type=str, default="16k")
    p.add_argument("--clt_run_name", type=str, default=None)
    p.add_argument("--clt_l0_target", type=int, default=None)
    p.add_argument("--clt_dtype", type=str, default="float32")
    p.add_argument("--clt_scale", type=float, default=1.0)
    p.add_argument("--clt_dtype_policy", type=str, default="clt", choices=["clt", "model"])
    p.add_argument("--clt_eps_active", type=float, default=1e-6)

    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--torch_dtype", type=str, default=None)
    p.add_argument("--attn_implementation", type=str, default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--tokenizer_revision", type=str, default=None)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--trust_remote_code", action="store_true")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_pairs", type=int, default=0, help="If >0, limit to the first N pairs.")
    p.add_argument("--normalize_by_length", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--require_token_id_match", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--gate_activation_ratio_tol", type=float, default=1e-6)
    p.add_argument("--gate_margin_abs_tol", type=float, default=1e-4)
    p.add_argument("--gate_score_abs_tol", type=float, default=1e-4)
    p.add_argument("--gate_clt_equiv_abs_tol", type=float, default=1e-4)
    p.add_argument("--gate_identity_abs_tol", type=float, default=1e-4)
    p.add_argument(
        "--hard_fail_invariant",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Debug fail-fast option. The governed paper reporting policy treats invariant gate failures "
            "as diagnostics and keeps flagged rows in aggregates."
        ),
    )
    p.add_argument(
        "--hard_fail_primary_logodds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if primary single-token candidate-set log-odds identity residual exceeds tolerance on applicable rows.",
    )
    p.add_argument(
        "--primary_logodds_residual_tol",
        type=float,
        default=float(PRIMARY_LOGODDS_RESIDUAL_TOL_DEFAULT),
        help="Absolute tolerance for primary single-token log-odds residual checks.",
    )
    p.add_argument("--ratio_den_eps", type=float, default=1e-6)
    p.add_argument("--raw_delta_norm_eps", type=float, default=1e-8)
    p.add_argument("--fidelity_rel_mse_warn", type=float, default=0.1)
    p.add_argument(
        "--run_pca_baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Evaluate a PCA-projected raw-delta baseline arm (PRJ_PCA).",
    )
    p.add_argument(
        "--run_faithfulness_resid_control",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Backward-compatible alias; when true runs both STRESS_RECON and STRESS_RESID.",
    )
    p.add_argument(
        "--run_faithfulness_decomposition_arms",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run both faithfulness controls STRESS_RECON and STRESS_RESID.",
    )
    p.add_argument(
        "--run_random_orth_baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Evaluate matched-rank random orthogonal projection controls (PRJ_RAND_s*).",
    )
    p.add_argument(
        "--random_projection_seeds",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated integer seeds for random orthogonal projections.",
    )
    p.add_argument(
        "--pca_fit_mode",
        type=str,
        default="leave_one_pair_out",
        choices=["global", "leave_one_pair_out"],
        help="PCA fit strategy; leave_one_pair_out enforces pair-disjoint fitting at evaluation time.",
    )
    p.add_argument(
        "--pca_rank_mode",
        type=str,
        default="active_per_row",
        choices=["active_per_row", "fixed"],
        help="PCA rank policy: per-row active latent count, or fixed rank.",
    )
    p.add_argument("--pca_fixed_rank", type=int, default=256, help="Used when --pca_rank_mode=fixed.")

    p.add_argument("--bootstrap_n", type=int, default=1000)
    p.add_argument("--ci", type=float, default=0.95)
    p.add_argument("--bootstrap_seed", type=int, default=42)

    p.add_argument("--out_csv", type=str, default=str(root / "results" / "clt_raw_comparability_l4_l8_l12.csv"))
    p.add_argument(
        "--out_json", type=str, default=str(root / "results" / "clt_raw_comparability_l4_l8_l12.summary.json")
    )
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    repo_root = Path(__file__).resolve().parents[1]

    if args.device == "auto":
        device = get_best_device()
    else:
        device = torch.device({"cpu": "cpu", "cuda": "cuda", "mps": "mps"}[str(args.device)])

    items = load_disamb_pairs(str(args.disamb_path))
    if int(args.max_pairs) > 0:
        items = items[: int(args.max_pairs)]
    if not items:
        raise ValueError("No disambiguation items loaded.")
    run_stress_arms = bool(
        bool(args.run_faithfulness_decomposition_arms) or bool(args.run_faithfulness_resid_control)
    )
    random_projection_seeds: List[int] = []
    if bool(args.run_random_orth_baseline):
        random_projection_seeds = _parse_int_list(str(args.random_projection_seeds))
        if len(random_projection_seeds) < 1:
            raise ValueError("--random_projection_seeds must contain at least one seed")
        if len(set(int(s) for s in random_projection_seeds)) != len(random_projection_seeds):
            raise ValueError("--random_projection_seeds contains duplicates; use unique seed values")
        if not bool(args.run_pca_baseline):
            raise ValueError("--run_random_orth_baseline requires --run_pca_baseline for matched-rank controls")

    loaded = load_causal_lm(
        str(args.model_name_or_path),
        device=device,
        torch_dtype=args.torch_dtype,
        revision=args.revision,
        tokenizer_revision=args.tokenizer_revision,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=bool(args.trust_remote_code),
        attn_implementation=str(args.attn_implementation),
    )
    model = loaded.model
    tokenizer = loaded.tokenizer
    model.eval()
    model_dtype = next(model.parameters()).dtype
    model_dtype_name = _normalize_dtype_name(model_dtype)
    clt_dtype_name = _normalize_dtype_name(str(args.clt_dtype))
    if str(args.torch_dtype).lower() in {"float64", "torch.float64"} and str(args.clt_dtype).lower() not in {
        "float64",
        "torch.float64",
    }:
        print(
            "Warning: model is float64 but --clt_dtype is not float64; strict CPU comparability is mixed-precision.",
            file=sys.stderr,
        )
    if str(args.clt_dtype_policy) == "model" and clt_dtype_name and clt_dtype_name != model_dtype_name:
        raise ValueError(
            "clt_dtype_policy='model' requires CLT/model dtype agreement: "
            f"model_dtype={model_dtype_name}, clt_dtype={clt_dtype_name}."
        )

    n_layers_model = int(get_num_layers(model))
    layers = _parse_int_list(str(args.layers))
    for layer in layers:
        if layer < 0 or layer >= n_layers_model:
            raise ValueError(f"Layer {layer} out of range [0, {n_layers_model})")

    bundles: Dict[int, LayerCLTBundle] = {}
    for layer in layers:
        clt, meta = load_clt(
            str(args.clt_repo),
            layer=int(layer),
            width=str(args.clt_width),
            run_name=args.clt_run_name,
            l0_target=args.clt_l0_target,
            device=str(device),
            dtype=str(args.clt_dtype),
            local_files_only=bool(args.local_files_only),
        )
        if str(meta.site_mode) != "same_site_v1":
            raise ValueError(f"Layer {layer}: unsupported CLT site_mode={meta.site_mode!r}")
        if not (
            str(meta.encode_site) == "resid_post"
            and str(meta.decode_site) == "resid_post"
            and str(meta.writeback_site) == "resid_post"
        ):
            raise ValueError(
                f"Layer {layer}: CLT site mismatch, expected resid_post; "
                f"got encode={meta.encode_site}, decode={meta.decode_site}, writeback={meta.writeback_site}"
            )
        bundles[int(layer)] = LayerCLTBundle(
            clt=clt,
            transform=CLTInputTransform(scale=float(args.clt_scale)),
            meta=meta,
        )

    pca_global_bundles: Dict[int, LayerPCABundle] = {}
    pca_loo_bundles: Dict[int, Dict[str, LayerPCABundle]] = {}
    pca_fit_meta: Dict[int, Dict[str, int]] = {}
    if bool(args.run_pca_baseline):
        pca_global_bundles, pca_loo_bundles, pca_fit_meta = _fit_pca_bundles(
            model=model,
            tokenizer=tokenizer,
            items=items,
            layers=layers,
            device=device,
            require_token_id_match=bool(args.require_token_id_match),
        )
        missing_layers = [int(layer) for layer in layers if int(layer) not in pca_global_bundles]
        if missing_layers:
            raise RuntimeError(
                "PCA baseline fit failed for one or more layers. "
                f"missing_layers={missing_layers}, fit_meta={pca_fit_meta}"
            )
        if str(args.pca_fit_mode) == "leave_one_pair_out":
            expected_pairs = len({str(it.pair_id) for it in items})
            if expected_pairs < 2:
                raise ValueError("pca_fit_mode=leave_one_pair_out requires at least 2 distinct pair IDs")
            missing_loo = [
                int(layer)
                for layer in layers
                if len(pca_loo_bundles.get(int(layer), {})) < expected_pairs
            ]
            if missing_loo:
                raise RuntimeError(
                    "PCA leave-one-pair-out fit missing held-out bundles for some layers. "
                    f"layers={missing_loo}, expected_pairs={expected_pairs}, fit_meta={pca_fit_meta}"
                )

    random_orth_bases: Dict[int, Dict[int, torch.Tensor]] = {}
    if bool(args.run_random_orth_baseline):
        for layer in layers:
            l = int(layer)
            pca_bundle = pca_global_bundles[l]
            hidden_dim = int(pca_bundle.components.size(1))
            max_rank = int(pca_bundle.max_rank)
            random_orth_bases[l] = {}
            for seed in random_projection_seeds:
                random_orth_bases[l][int(seed)] = _random_orthonormal_rows(
                    hidden_dim=hidden_dim,
                    max_rank=max_rank,
                    seed=int(seed),
                )

    cfg_c = CLTPatchConfig(
        decode_strategy="delta_1decode",
        dtype_policy=str(args.clt_dtype_policy),
        eps_active=float(args.clt_eps_active),
    )
    cfg_d = CLTPatchConfig(
        decode_strategy="safe_2decode",
        dtype_policy=str(args.clt_dtype_policy),
        eps_active=float(args.clt_eps_active),
    )

    rows: List[Dict[str, Any]] = []
    fail_counts: Dict[str, int] = defaultdict(int)
    fail_reasons: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    partial_csv_path = Path(args.out_csv).with_suffix(".partial.csv")

    def _record_arm_failure(arm: str, reason: str) -> None:
        fail_counts[str(arm)] += 1
        fail_reasons[str(arm)][str(reason)] += 1

    def _checkpoint_partial_rows() -> None:
        if not rows:
            return
        _write_csv(rows, partial_csv_path)

    for it in items:
        for direction_name, donor, recv in (("a_to_b", it.a, it.b), ("b_to_a", it.b, it.a)):
            if str(donor.expected_label) not in it.choices:
                fail_counts["direction_missing_expected_label"] += 1
                continue

            donor_span, donor_token_ids = token_span_for_substring(
                tokenizer, donor.prompt, it.target, it.target_occurrence
            )
            recv_span, recv_token_ids = token_span_for_substring(
                tokenizer, recv.prompt, it.target, it.target_occurrence
            )
            aligned = len(donor_span) == len(recv_span) and (
                (not bool(args.require_token_id_match)) or (donor_token_ids == recv_token_ids)
            )

            row_common: Dict[str, Any] = {
                "pair_id": str(it.pair_id),
                "direction": str(direction_name),
                "target": str(it.target),
                "target_occurrence": int(it.target_occurrence),
                "donor_expected_label": str(donor.expected_label),
                "receiver_expected_label": str(recv.expected_label),
                "aligned_spans": bool(aligned),
                "donor_span_len": int(len(donor_span)),
                "receiver_span_len": int(len(recv_span)),
            }

            if not aligned:
                for layer in layers:
                    r = dict(row_common)
                    r.update(
                        {
                            "layer": int(layer),
                            "all_arms_success": False,
                            "invariant_all_pass": False,
                            "analysis_included": False,
                            "skip_reason": "misaligned",
                        }
                    )
                    rows.append(r)
                _checkpoint_partial_rows()
                continue

            prompt_ids_donor = _encode_prompt(tokenizer, donor.prompt, device=device)
            prompt_ids_recv = _encode_prompt(tokenizer, recv.prompt, device=device)
            _assert_prompt_span_alignment(prompt_ids_donor, donor_span, donor_token_ids, side_name="donor")
            _assert_prompt_span_alignment(prompt_ids_recv, recv_span, recv_token_ids, side_name="receiver")

            base_forward = lambda full_ids: model(input_ids=full_ids, use_cache=False, return_dict=True).logits
            base_scores = _score_choices_with_forward(
                tokenizer=tokenizer,
                prompt_ids=prompt_ids_recv,
                choices=it.choices,
                forward_fn=base_forward,
                device=device,
                normalize_by_length=bool(args.normalize_by_length),
            )
            base_pred = _argmax_label(base_scores)
            donor_expected = str(donor.expected_label)
            other_label = _best_other_label(base_scores, expected=donor_expected)
            base_margin = _margin(base_scores, expected=donor_expected)

            # Primary endpoint-native mechanism metadata (single-token candidate-set log-odds).
            choices_token_ids = choices_token_ids_from_strings(tokenizer, it.choices)
            prompt_boundary_pos = int(prompt_ids_recv.size(1) - 1)
            scored_positions: List[int] = []
            for _seq in choices_token_ids.get(donor_expected, ()):
                scored_positions.append(prompt_boundary_pos)
            for _seq in choices_token_ids.get(other_label, ()):
                scored_positions.append(prompt_boundary_pos)
            primary_meta = evaluate_primary_applicability(
                choices_token_ids=choices_token_ids,
                expected_label=donor_expected,
                other_label=other_label,
                scored_positions=scored_positions,
                require_binary_labels=True,
                require_equal_candidate_counts=True,
            )

            primary_labels_binary = bool(primary_meta.labels_binary)
            expected_all_single = bool(primary_meta.expected_all_single)
            other_all_single = bool(primary_meta.other_all_single)
            expected_tokens = [int(x) for x in primary_meta.expected_tokens]
            other_tokens = [int(x) for x in primary_meta.other_tokens]
            expected_unique = [int(x) for x in primary_meta.expected_unique_tokens]
            other_unique = [int(x) for x in primary_meta.other_unique_tokens]
            expected_no_duplicates = bool(primary_meta.no_duplicates_expected)
            other_no_duplicates = bool(primary_meta.no_duplicates_other)
            token_sets_disjoint = bool(primary_meta.token_sets_disjoint)
            candidate_count_equal = bool(primary_meta.candidate_count_equal)
            same_scored_position = bool(primary_meta.same_scored_position)
            primary_static_applicable = bool(primary_meta.static_applicable)
            primary_static_reason = str(primary_meta.reason)

            primary_base_stats: Dict[str, float] = {
                "P_E_base": float("nan"),
                "P_O_base": float("nan"),
                "P_rest_base": float("nan"),
                "logPE_base": float("nan"),
                "logPO_base": float("nan"),
                "margin_from_logodds_base": float("nan"),
                "base_logodds_residual": float("nan"),
            }
            if primary_static_applicable:
                base_next_logits, base_next_probs = _next_token_logits_probs(
                    prompt_ids=prompt_ids_recv, forward_fn=base_forward
                )
                idx_exp = torch.tensor(expected_unique, device=base_next_probs.device, dtype=torch.long)
                idx_oth = torch.tensor(other_unique, device=base_next_probs.device, dtype=torch.long)
                p_e_base = float(base_next_probs.index_select(0, idx_exp).sum().item())
                p_o_base = float(base_next_probs.index_select(0, idx_oth).sum().item())
                p_rest_base = float(1.0 - p_e_base - p_o_base)
                log_pe_base = safe_log_prob_mass(p_e_base)
                log_po_base = safe_log_prob_mass(p_o_base)
                margin_from_logodds_base = float(log_pe_base - log_po_base)
                base_logodds_residual = float(base_margin - margin_from_logodds_base)
                primary_base_stats.update(
                    {
                        "P_E_base": p_e_base,
                        "P_O_base": p_o_base,
                        "P_rest_base": p_rest_base,
                        "logPE_base": log_pe_base,
                        "logPO_base": log_po_base,
                        "margin_from_logodds_base": margin_from_logodds_base,
                        "base_logodds_residual": base_logodds_residual,
                    }
                )
                if (
                    bool(args.hard_fail_primary_logodds)
                    and abs(float(base_logodds_residual)) > float(args.primary_logodds_residual_tol)
                ):
                    raise RuntimeError(
                        "Primary log-odds base residual exceeded tolerance: "
                        f"pair={it.pair_id}, dir={direction_name}, residual={base_logodds_residual:.3e}, "
                        f"tol={float(args.primary_logodds_residual_tol):.3e}"
                    )

            donor_out = get_block_outputs(model, prompt_ids_donor, layers=layers)
            recv_out = get_block_outputs(model, prompt_ids_recv, layers=layers)

            for layer in layers:
                # --- 1/8  raw delta and activation-level A=B invariant --------
                raw_donor = donor_out[int(layer)][0, donor_span, :].detach()
                raw_recv = recv_out[int(layer)][0, recv_span, :].detach()
                raw_delta = raw_donor - raw_recv
                raw_swap_replacement = raw_donor
                raw_delta_replacement = raw_recv + raw_delta

                # A≈B activation-level invariant.
                gate_activation_ratio = _norm_ratio(raw_swap_replacement - raw_delta_replacement, raw_swap_replacement)
                gate_activation_pass = bool(gate_activation_ratio < float(args.gate_activation_ratio_tol))

                # --- 2/8  CLT encode/decode, reconstruction fidelity ----------
                bundle = bundles[int(layer)]
                clt = bundle.clt
                transform = bundle.transform
                clt_device, clt_dtype = _infer_clt_device_dtype(clt)
                donor_slice = raw_donor.unsqueeze(0).to(device=clt_device, dtype=clt_dtype)
                recv_slice = raw_recv.unsqueeze(0).to(device=clt_device, dtype=clt_dtype)
                recv_latents = clt.encode(transform.forward(recv_slice))
                donor_latents = clt.encode(transform.forward(donor_slice))
                active_latent_mask = recv_latents.abs() > float(args.clt_eps_active)
                active_latent_count = int(active_latent_mask.sum().item())
                active_latent_frac = float(active_latent_mask.to(dtype=torch.float32).mean().item())

                # Reconstruction fidelity on receiver slice.
                recon = transform.inverse(clt.decode(recv_latents))
                recon_err = recon - recv_slice
                err_sse = float((recon_err**2).sum().item())
                recv_sse = float((recv_slice**2).sum().item())
                fidelity_rel_mse = float(err_sse / (recv_sse + 1e-12))
                fidelity_rel_l2 = float(torch.norm(recon_err).item() / (torch.norm(recv_slice).item() + 1e-12))
                fidelity_cosine = _cosine(recon, recv_slice)
                recon_norm_ratio = _norm_ratio(recon, recv_slice)

                # --- 3/8  CLT writeback deltas (arms C and D) -----------------
                clt_delta_c = _compute_clt_writeback_delta(
                    clt=clt,
                    transform=transform,
                    recv_slice=recv_slice,
                    recv_latents=recv_latents,
                    donor_latents=donor_latents,
                    decode_strategy="delta_1decode",
                ).to(device=raw_delta.device, dtype=raw_delta.dtype)[0]
                clt_delta_d = _compute_clt_writeback_delta(
                    clt=clt,
                    transform=transform,
                    recv_slice=recv_slice,
                    recv_latents=recv_latents,
                    donor_latents=donor_latents,
                    decode_strategy="safe_2decode",
                ).to(device=raw_delta.device, dtype=raw_delta.dtype)[0]

                raw_delta_norm = float(torch.norm(raw_delta).item())
                clt_delta_norm_c = float(torch.norm(clt_delta_c).item())
                clt_delta_norm_d = float(torch.norm(clt_delta_d).item())
                raw_delta_norm_small = bool(raw_delta_norm <= float(args.raw_delta_norm_eps))
                clt_delta_ratio_c = (
                    float(clt_delta_norm_c / raw_delta_norm) if not raw_delta_norm_small else float("nan")
                )
                clt_delta_ratio_d = (
                    float(clt_delta_norm_d / raw_delta_norm) if not raw_delta_norm_small else float("nan")
                )
                clt_delta_cos_c = _cosine(clt_delta_c, raw_delta)
                clt_delta_cos_d = _cosine(clt_delta_d, raw_delta)

                # --- 4/8  optional PCA and random-orth projection baselines ---
                # Pre-initialize optional projection outputs to NaN so the
                # row dict always contains these columns, even when the PCA
                # baseline is disabled. NaN signals "not applicable" in CSV.
                proj_pca_delta_norm_ratio = float("nan")
                proj_pca_delta_cosine_raw = float("nan")
                projection_rank = float("nan")
                projection_rank_max = float("nan")
                projection_n_tokens_fit = float("nan")
                projection_n_pairs_fit = float("nan")
                projection_fit_mode = ""
                raw_pca_replacement: Optional[torch.Tensor] = None
                raw_rand_replacements: Dict[str, torch.Tensor] = {}
                proj_rand_norm_ratio_by_seed: Dict[int, float] = {}
                proj_rand_cosine_by_seed: Dict[int, float] = {}
                pca_rank_for_controls = 0
                if bool(args.run_pca_baseline):
                    if str(args.pca_fit_mode) == "leave_one_pair_out":
                        projection_fit_mode = "leave_one_pair_out"
                        pca_bundle = pca_loo_bundles.get(int(layer), {}).get(str(it.pair_id), None)
                    else:
                        projection_fit_mode = "global"
                        pca_bundle = pca_global_bundles.get(int(layer), None)
                    if pca_bundle is None:
                        raise RuntimeError(
                            "Missing PCA bundle for evaluation row "
                            f"(pair={it.pair_id}, layer={layer}, fit_mode={projection_fit_mode})"
                        )
                    rank = _resolve_pca_rank(
                        rank_mode=str(args.pca_rank_mode),
                        fixed_rank=int(args.pca_fixed_rank),
                        active_latent_count=int(active_latent_count),
                        max_rank=int(pca_bundle.max_rank),
                    )
                    pca_rank_for_controls = int(rank)
                    projection_rank = float(rank)
                    projection_rank_max = float(pca_bundle.max_rank)
                    projection_n_tokens_fit = float(pca_bundle.n_tokens_fit)
                    projection_n_pairs_fit = float(pca_bundle.n_pairs_fit)
                    proj_delta = _project_rows_onto_components(
                        rows=raw_delta,
                        components=pca_bundle.components.to(device=raw_delta.device, dtype=raw_delta.dtype),
                        rank=rank,
                    )
                    proj_norm = float(torch.norm(proj_delta).item())
                    proj_pca_delta_norm_ratio = (
                        float(proj_norm / raw_delta_norm) if not raw_delta_norm_small else float("nan")
                    )
                    proj_pca_delta_cosine_raw = _cosine(proj_delta, raw_delta)
                    raw_pca_replacement = raw_recv + proj_delta

                    if bool(args.run_random_orth_baseline):
                        for seed in random_projection_seeds:
                            rand_basis = random_orth_bases[int(layer)][int(seed)].to(
                                device=raw_delta.device, dtype=raw_delta.dtype
                            )
                            rand_delta = _project_rows_onto_components(
                                rows=raw_delta,
                                components=rand_basis,
                                rank=rank,
                            )
                            rand_norm = float(torch.norm(rand_delta).item())
                            proj_rand_norm_ratio_by_seed[int(seed)] = (
                                float(rand_norm / raw_delta_norm) if not raw_delta_norm_small else float("nan")
                            )
                            proj_rand_cosine_by_seed[int(seed)] = _cosine(rand_delta, raw_delta)
                            raw_rand_replacements[f"PRJ_RAND_s{int(seed)}"] = raw_recv + rand_delta

                # --- 5/8  optional STRESS faithfulness arms -------------------
                # Pre-initialize optional STRESS arm outputs to NaN for the
                # same reason: stable CSV schema regardless of --run_stress_arms.
                stress_delta_recon_norm_ratio = float("nan")
                stress_delta_recon_cosine_raw = float("nan")
                stress_delta_resid_norm_ratio = float("nan")
                stress_delta_resid_cosine_raw = float("nan")
                stress_delta_additivity_rel_err = float("nan")
                raw_stress_recon_replacement: Optional[torch.Tensor] = None
                raw_stress_resid_replacement: Optional[torch.Tensor] = None
                if bool(run_stress_arms):
                    recon_donor = transform.inverse(clt.decode(donor_latents))
                    recon_delta = (recon_donor - recon).to(device=raw_delta.device, dtype=raw_delta.dtype)[0]
                    resid_delta = raw_delta - recon_delta
                    recon_delta_norm = float(torch.norm(recon_delta).item())
                    resid_delta_norm = float(torch.norm(resid_delta).item())
                    stress_delta_recon_norm_ratio = (
                        float(recon_delta_norm / raw_delta_norm) if not raw_delta_norm_small else float("nan")
                    )
                    stress_delta_recon_cosine_raw = _cosine(recon_delta, raw_delta)
                    stress_delta_resid_norm_ratio = (
                        float(resid_delta_norm / raw_delta_norm) if not raw_delta_norm_small else float("nan")
                    )
                    stress_delta_resid_cosine_raw = _cosine(resid_delta, raw_delta)
                    stress_delta_additivity_rel_err = float(
                        torch.norm(raw_delta - (recon_delta + resid_delta)).item() / (raw_delta_norm + 1e-12)
                    )
                    raw_stress_recon_replacement = raw_recv + recon_delta
                    raw_stress_resid_replacement = raw_recv + resid_delta

                # --- 6/8  build per-arm forward functions and evaluate --------
                site = PatchSpanSite(layer=int(layer), token_indices=tuple(int(i) for i in recv_span))
                clt_policy = ReplaceLatentsAtIndicesPolicy(
                    token_indices=[int(i) for i in recv_span], replacement_latents=donor_latents
                )
                clt_identity_policy = ReplaceLatentsAtIndicesPolicy(
                    token_indices=[int(i) for i in recv_span], replacement_latents=recv_latents
                )
                raw_cprime_replacement = raw_recv + clt_delta_c
                raw_dprime_replacement = raw_recv + clt_delta_d

                arm_results: Dict[str, Dict[str, Any]] = {}
                exp_cont = str(it.choices[donor_expected][0])
                oth_cont = str(it.choices[other_label][0])
                base_exp_logits, base_exp_cont_ids = _logits_slice_for_continuation(
                    tokenizer=tokenizer,
                    prompt_ids=prompt_ids_recv,
                    continuation=exp_cont,
                    forward_fn=base_forward,
                    device=device,
                )
                base_oth_logits, base_oth_cont_ids = _logits_slice_for_continuation(
                    tokenizer=tokenizer,
                    prompt_ids=prompt_ids_recv,
                    continuation=oth_cont,
                    forward_fn=base_forward,
                    device=device,
                )

                # Per-arm forward closures. Each accepts full_ids (prompt tokens)
                # and returns logits with the patch applied at `site`. The
                # closures capture `model`, `site`, and arm-specific tensors.
                forward_a = lambda full_ids: forward_with_patched_block_output_span(
                    model=model,
                    input_ids=full_ids,
                    site=site,
                    replacement=raw_swap_replacement,
                )
                forward_b = lambda full_ids: forward_with_patched_block_output_span(
                    model=model,
                    input_ids=full_ids,
                    site=site,
                    replacement=raw_delta_replacement,
                )
                forward_c = lambda full_ids: forward_with_clt_latent_patching_span(
                    model=model,
                    input_ids=full_ids,
                    site=site,
                    clt=clt,
                    policy=clt_policy,
                    transform=transform,
                    config=cfg_c,
                )
                forward_d = lambda full_ids: forward_with_clt_latent_patching_span(
                    model=model,
                    input_ids=full_ids,
                    site=site,
                    clt=clt,
                    policy=clt_policy,
                    transform=transform,
                    config=cfg_d,
                )
                forward_cp = lambda full_ids: forward_with_patched_block_output_span(
                    model=model,
                    input_ids=full_ids,
                    site=site,
                    replacement=raw_cprime_replacement,
                )
                forward_dp = lambda full_ids: forward_with_patched_block_output_span(
                    model=model,
                    input_ids=full_ids,
                    site=site,
                    replacement=raw_dprime_replacement,
                )
                forward_ri = lambda full_ids: forward_with_patched_block_output_span(
                    model=model,
                    input_ids=full_ids,
                    site=site,
                    replacement=raw_recv,
                )
                forward_ci = lambda full_ids: forward_with_clt_latent_patching_span(
                    model=model,
                    input_ids=full_ids,
                    site=site,
                    clt=clt,
                    policy=clt_identity_policy,
                    transform=transform,
                    config=cfg_c,
                )
                optional_arm_forward_fns: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {}
                if raw_pca_replacement is not None:
                    optional_arm_forward_fns["PRJ_PCA"] = lambda full_ids: forward_with_patched_block_output_span(
                        model=model,
                        input_ids=full_ids,
                        site=site,
                        replacement=raw_pca_replacement,
                    )
                # Use a default-argument to capture `rand_replacement` by value,
                # not by reference — otherwise all PRJ_RAND_s* arms would share
                # the last loop iteration's tensor.
                for rand_arm_name, rand_replacement in raw_rand_replacements.items():
                    optional_arm_forward_fns[rand_arm_name] = (
                        lambda full_ids, _replacement=rand_replacement: forward_with_patched_block_output_span(
                            model=model,
                            input_ids=full_ids,
                            site=site,
                            replacement=_replacement,
                        )
                    )
                if raw_stress_recon_replacement is not None:
                    optional_arm_forward_fns[
                        "STRESS_RECON"
                    ] = lambda full_ids: forward_with_patched_block_output_span(
                        model=model,
                        input_ids=full_ids,
                        site=site,
                        replacement=raw_stress_recon_replacement,
                    )
                if raw_stress_resid_replacement is not None:
                    optional_arm_forward_fns[
                        "STRESS_RESID"
                    ] = lambda full_ids: forward_with_patched_block_output_span(
                        model=model,
                        input_ids=full_ids,
                        site=site,
                        replacement=raw_stress_resid_replacement,
                    )
                arm_forward_fns: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
                    "A": forward_a,
                    "B": forward_b,
                    "C": forward_c,
                    "D": forward_d,
                    "Cp": forward_cp,
                    "Dp": forward_dp,
                    "RI": forward_ri,
                    "CI": forward_ci,
                }
                arm_forward_fns.update(optional_arm_forward_fns)
                required_arms = REQUIRED_ARMS
                optional_arms = tuple(optional_arm_forward_fns.keys())
                all_arms = tuple(list(required_arms) + list(optional_arms))
                decomp_arms = DECOMP_ARMS

                def _evaluate_arm(arm_name: str, forward_fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
                    scores = _score_choices_with_forward(
                        tokenizer=tokenizer,
                        prompt_ids=prompt_ids_recv,
                        choices=it.choices,
                        forward_fn=forward_fn,
                        device=device,
                        normalize_by_length=bool(args.normalize_by_length),
                    )
                    patched_margin = _margin(scores, expected=donor_expected)
                    effect = float(patched_margin - base_margin)
                    pred = _argmax_label(scores)
                    delta_score_exp = float(scores[donor_expected] - base_scores[donor_expected])
                    delta_score_other = float(scores[other_label] - base_scores[other_label])

                    decomp_exp: Dict[str, float] = {}
                    decomp_oth: Dict[str, float] = {}
                    if arm_name in decomp_arms:
                        patch_exp_logits, _ = _logits_slice_for_continuation(
                            tokenizer=tokenizer,
                            prompt_ids=prompt_ids_recv,
                            continuation=exp_cont,
                            forward_fn=forward_fn,
                            device=device,
                        )
                        patch_oth_logits, _ = _logits_slice_for_continuation(
                            tokenizer=tokenizer,
                            prompt_ids=prompt_ids_recv,
                            continuation=oth_cont,
                            forward_fn=forward_fn,
                            device=device,
                        )
                        decomp_exp = _decomp_deltas(base_exp_logits, patch_exp_logits, base_exp_cont_ids)
                        decomp_oth = _decomp_deltas(base_oth_logits, patch_oth_logits, base_oth_cont_ids)

                    arm_results[arm_name] = {
                        "success": True,
                        "effect": effect,
                        "pred": str(pred),
                        "delta_score_exp": delta_score_exp,
                        "delta_score_other": delta_score_other,
                        "decomp_exp": decomp_exp,
                        "decomp_other": decomp_oth,
                    }

                def _arm_fail(arm_name: str, e: Exception) -> None:
                    reason = _safe_err(e)
                    _record_arm_failure(arm_name, reason)
                    arm_results[arm_name] = {"success": False, "error": reason}

                for arm in all_arms:
                    try:
                        _evaluate_arm(arm, arm_forward_fns[arm])
                    except Exception as e:  # pragma: no cover - runtime guard
                        _arm_fail(arm, e)
                all_success = all(bool(arm_results.get(arm, {}).get("success", False)) for arm in required_arms)

                # --- 7/8  primary single-token log-odds cancellation (A vs C) -
                primary_logodds_applicable = bool(all_success and primary_static_applicable)
                primary_logodds_reason = "ok" if primary_logodds_applicable else (
                    "all_arms_not_success" if not all_success else str(primary_static_reason)
                )
                # Pre-initialize primary log-odds patch outputs to NaN.
                # Written only when primary_logodds_applicable is True; NaN
                # encodes "not applicable" for rows where it is False.
                primary_P_E_patch_A = float("nan")
                primary_P_O_patch_A = float("nan")
                primary_P_rest_patch_A = float("nan")
                primary_P_E_patch_C = float("nan")
                primary_P_O_patch_C = float("nan")
                primary_P_rest_patch_C = float("nan")
                primary_dlogPE_A = float("nan")
                primary_dlogPO_A = float("nan")
                primary_dlogPE_C = float("nan")
                primary_dlogPO_C = float("nan")
                primary_dPE_A = float("nan")
                primary_dPO_A = float("nan")
                primary_dPrest_A = float("nan")
                primary_dPE_C = float("nan")
                primary_dPO_C = float("nan")
                primary_dPrest_C = float("nan")
                primary_d_CA_logPE = float("nan")
                primary_d_CA_logPO = float("nan")
                primary_delta_m_from_logodds_A = float("nan")
                primary_delta_m_from_logodds_C = float("nan")
                primary_delta_m_residual_A = float("nan")
                primary_delta_m_residual_C = float("nan")
                primary_cancellation_pass = False

                if primary_logodds_applicable:
                    idx_exp = torch.tensor(expected_unique, device=prompt_ids_recv.device, dtype=torch.long)
                    idx_oth = torch.tensor(other_unique, device=prompt_ids_recv.device, dtype=torch.long)

                    _logits_a, probs_a = _next_token_logits_probs(prompt_ids=prompt_ids_recv, forward_fn=arm_forward_fns["A"])
                    _logits_c, probs_c = _next_token_logits_probs(prompt_ids=prompt_ids_recv, forward_fn=arm_forward_fns["C"])

                    primary_P_E_patch_A = float(probs_a.index_select(0, idx_exp).sum().item())
                    primary_P_O_patch_A = float(probs_a.index_select(0, idx_oth).sum().item())
                    primary_P_rest_patch_A = float(1.0 - primary_P_E_patch_A - primary_P_O_patch_A)
                    primary_P_E_patch_C = float(probs_c.index_select(0, idx_exp).sum().item())
                    primary_P_O_patch_C = float(probs_c.index_select(0, idx_oth).sum().item())
                    primary_P_rest_patch_C = float(1.0 - primary_P_E_patch_C - primary_P_O_patch_C)

                    primary_dlogPE_A = float(
                        safe_log_prob_mass(primary_P_E_patch_A) - float(primary_base_stats["logPE_base"])
                    )
                    primary_dlogPO_A = float(
                        safe_log_prob_mass(primary_P_O_patch_A) - float(primary_base_stats["logPO_base"])
                    )
                    primary_dlogPE_C = float(
                        safe_log_prob_mass(primary_P_E_patch_C) - float(primary_base_stats["logPE_base"])
                    )
                    primary_dlogPO_C = float(
                        safe_log_prob_mass(primary_P_O_patch_C) - float(primary_base_stats["logPO_base"])
                    )

                    primary_dPE_A = float(primary_P_E_patch_A - float(primary_base_stats["P_E_base"]))
                    primary_dPO_A = float(primary_P_O_patch_A - float(primary_base_stats["P_O_base"]))
                    primary_dPrest_A = float(primary_P_rest_patch_A - float(primary_base_stats["P_rest_base"]))
                    primary_dPE_C = float(primary_P_E_patch_C - float(primary_base_stats["P_E_base"]))
                    primary_dPO_C = float(primary_P_O_patch_C - float(primary_base_stats["P_O_base"]))
                    primary_dPrest_C = float(primary_P_rest_patch_C - float(primary_base_stats["P_rest_base"]))

                    primary_d_CA_logPE = float(primary_dlogPE_C - primary_dlogPE_A)
                    primary_d_CA_logPO = float(primary_dlogPO_C - primary_dlogPO_A)

                    primary_delta_m_from_logodds_A = float(primary_dlogPE_A - primary_dlogPO_A)
                    primary_delta_m_from_logodds_C = float(primary_dlogPE_C - primary_dlogPO_C)
                    primary_delta_m_residual_A = float(float(arm_results["A"]["effect"]) - primary_delta_m_from_logodds_A)
                    primary_delta_m_residual_C = float(float(arm_results["C"]["effect"]) - primary_delta_m_from_logodds_C)
                    primary_cancellation_pass = bool(
                        abs(primary_delta_m_residual_A) <= float(args.primary_logodds_residual_tol)
                        and abs(primary_delta_m_residual_C) <= float(args.primary_logodds_residual_tol)
                    )
                    if bool(args.hard_fail_primary_logodds) and not primary_cancellation_pass:
                        raise RuntimeError(
                            "Primary log-odds cancellation residual exceeded tolerance: "
                            f"pair={it.pair_id}, dir={direction_name}, layer={layer}, "
                            f"resA={primary_delta_m_residual_A:.3e}, resC={primary_delta_m_residual_C:.3e}, "
                            f"tol={float(args.primary_logodds_residual_tol):.3e}"
                        )

                # --- 8/8  invariant gates, analysis inclusion, row assembly ---
                # Pre-initialize gate diagnostics to NaN; populated only when
                # all required arms succeeded (all_success=True).
                gate_margin_abs_diff = float("nan")
                gate_score_abs_diff_exp = float("nan")
                gate_score_abs_diff_other = float("nan")
                gate_clt_effect_abs_diff_c = float("nan")
                gate_clt_effect_abs_diff_d = float("nan")
                gate_identity_abs_effect_raw = float("nan")
                gate_identity_abs_effect_clt = float("nan")
                gate_margin_pass = False
                gate_score_pass = False
                gate_clt_equiv_pass = False
                gate_identity_pass = False

                if all_success:
                    eff_a = float(arm_results["A"]["effect"])
                    eff_b = float(arm_results["B"]["effect"])
                    gate_margin_abs_diff = abs(eff_a - eff_b)
                    gate_margin_pass = bool(gate_margin_abs_diff < float(args.gate_margin_abs_tol))
                    gate_score_abs_diff_exp = abs(
                        float(arm_results["A"]["delta_score_exp"]) - float(arm_results["B"]["delta_score_exp"])
                    )
                    gate_score_abs_diff_other = abs(
                        float(arm_results["A"]["delta_score_other"]) - float(arm_results["B"]["delta_score_other"])
                    )
                    gate_score_pass = bool(
                        gate_score_abs_diff_exp < float(args.gate_score_abs_tol)
                        and gate_score_abs_diff_other < float(args.gate_score_abs_tol)
                    )
                    gate_clt_effect_abs_diff_c = abs(float(arm_results["C"]["effect"]) - float(arm_results["Cp"]["effect"]))
                    gate_clt_effect_abs_diff_d = abs(float(arm_results["D"]["effect"]) - float(arm_results["Dp"]["effect"]))
                    gate_clt_equiv_pass = bool(
                        gate_clt_effect_abs_diff_c < float(args.gate_clt_equiv_abs_tol)
                        and gate_clt_effect_abs_diff_d < float(args.gate_clt_equiv_abs_tol)
                    )
                    gate_identity_abs_effect_raw = abs(float(arm_results["RI"]["effect"]))
                    gate_identity_abs_effect_clt = abs(float(arm_results["CI"]["effect"]))
                    gate_identity_pass = bool(
                        gate_identity_abs_effect_raw < float(args.gate_identity_abs_tol)
                        and gate_identity_abs_effect_clt < float(args.gate_identity_abs_tol)
                    )

                invariant_all_pass = bool(
                    gate_activation_pass
                    and gate_margin_pass
                    and gate_score_pass
                    and gate_clt_equiv_pass
                    and gate_identity_pass
                )
                analysis_included = limitation_analysis_included(
                    all_required_arms_success=bool(all_success),
                    effect_A=arm_results.get("A", {}).get("effect"),
                    effect_C=arm_results.get("C", {}).get("effect"),
                )

                row: Dict[str, Any] = dict(row_common)
                row.update(
                    {
                        "layer": int(layer),
                        "base_pred": str(base_pred),
                        "base_margin": float(base_margin),
                        "other_label": str(other_label),
                        "raw_delta_norm": raw_delta_norm,
                        "raw_delta_norm_small": bool(raw_delta_norm_small),
                        "clt_delta_norm_C": clt_delta_norm_c,
                        "clt_delta_norm_D": clt_delta_norm_d,
                        "clt_delta_norm_ratio_C": clt_delta_ratio_c,
                        "clt_delta_norm_ratio_D": clt_delta_ratio_d,
                        "clt_delta_cosine_raw_C": clt_delta_cos_c,
                        "clt_delta_cosine_raw_D": clt_delta_cos_d,
                        "proj_pca_delta_norm_ratio": float(proj_pca_delta_norm_ratio),
                        "proj_pca_delta_cosine_raw": float(proj_pca_delta_cosine_raw),
                        "projection_rank": float(projection_rank),
                        "projection_rank_max": float(projection_rank_max),
                        "projection_n_tokens_fit": float(projection_n_tokens_fit),
                        "projection_n_pairs_fit": float(projection_n_pairs_fit),
                        "projection_fit_mode": str(projection_fit_mode),
                        "projection_rank_for_controls": float(pca_rank_for_controls),
                        "stress_delta_recon_norm_ratio": float(stress_delta_recon_norm_ratio),
                        "stress_delta_recon_cosine_raw": float(stress_delta_recon_cosine_raw),
                        "stress_delta_resid_norm_ratio": float(stress_delta_resid_norm_ratio),
                        "stress_delta_resid_cosine_raw": float(stress_delta_resid_cosine_raw),
                        "stress_delta_additivity_rel_err": float(stress_delta_additivity_rel_err),
                        "recon_norm_ratio": float(recon_norm_ratio),
                        "active_latent_count": int(active_latent_count),
                        "active_latent_frac": float(active_latent_frac),
                        "fidelity_rel_mse": float(fidelity_rel_mse),
                        "fidelity_rel_l2": float(fidelity_rel_l2),
                        "fidelity_cosine": float(fidelity_cosine),
                        "fidelity_rel_mse_warn": bool(float(fidelity_rel_mse) > float(args.fidelity_rel_mse_warn)),
                        "gate_activation_ratio": float(gate_activation_ratio),
                        "gate_activation_pass": bool(gate_activation_pass),
                        "gate_margin_abs_diff": float(gate_margin_abs_diff),
                        "gate_margin_pass": bool(gate_margin_pass),
                        "gate_score_abs_diff_exp": float(gate_score_abs_diff_exp),
                        "gate_score_abs_diff_other": float(gate_score_abs_diff_other),
                        "gate_score_pass": bool(gate_score_pass),
                        "gate_clt_effect_abs_diff_C": float(gate_clt_effect_abs_diff_c),
                        "gate_clt_effect_abs_diff_D": float(gate_clt_effect_abs_diff_d),
                        "gate_clt_equiv_pass": bool(gate_clt_equiv_pass),
                        "gate_identity_abs_effect_raw": float(gate_identity_abs_effect_raw),
                        "gate_identity_abs_effect_clt": float(gate_identity_abs_effect_clt),
                        "gate_identity_pass": bool(gate_identity_pass),
                        "invariant_all_pass": bool(invariant_all_pass),
                        "all_arms_success": bool(all_success),
                        "analysis_included": bool(analysis_included),
                        "primary_labels_binary": bool(primary_labels_binary),
                        "primary_candidate_count_expected": int(len(expected_tokens)),
                        "primary_candidate_count_other": int(len(other_tokens)),
                        "primary_expected_all_single": bool(expected_all_single),
                        "primary_other_all_single": bool(other_all_single),
                        "primary_no_duplicates_expected": bool(expected_no_duplicates),
                        "primary_no_duplicates_other": bool(other_no_duplicates),
                        "primary_token_sets_disjoint": bool(token_sets_disjoint),
                        "primary_candidate_count_equal": bool(candidate_count_equal),
                        "primary_same_scored_position": bool(same_scored_position),
                        "primary_static_applicable": bool(primary_static_applicable),
                        "primary_logodds_applicable": bool(primary_logodds_applicable),
                        "primary_logodds_applicable_flag": float(1.0 if primary_logodds_applicable else 0.0),
                        "primary_logodds_reason": str(primary_logodds_reason),
                        "P_E_base": float(primary_base_stats["P_E_base"]),
                        "P_O_base": float(primary_base_stats["P_O_base"]),
                        "P_rest_base": float(primary_base_stats["P_rest_base"]),
                        "P_E_patch_A": float(primary_P_E_patch_A),
                        "P_O_patch_A": float(primary_P_O_patch_A),
                        "P_rest_patch_A": float(primary_P_rest_patch_A),
                        "P_E_patch_C": float(primary_P_E_patch_C),
                        "P_O_patch_C": float(primary_P_O_patch_C),
                        "P_rest_patch_C": float(primary_P_rest_patch_C),
                        "dlogPE_A": float(primary_dlogPE_A),
                        "dlogPO_A": float(primary_dlogPO_A),
                        "dlogPE_C": float(primary_dlogPE_C),
                        "dlogPO_C": float(primary_dlogPO_C),
                        "dPE_A": float(primary_dPE_A),
                        "dPO_A": float(primary_dPO_A),
                        "dPrest_A": float(primary_dPrest_A),
                        "dPE_C": float(primary_dPE_C),
                        "dPO_C": float(primary_dPO_C),
                        "dPrest_C": float(primary_dPrest_C),
                        "d_CA_logPE": float(primary_d_CA_logPE),
                        "d_CA_logPO": float(primary_d_CA_logPO),
                        "primary_delta_m_from_logodds_A": float(primary_delta_m_from_logodds_A),
                        "primary_delta_m_from_logodds_C": float(primary_delta_m_from_logodds_C),
                        "primary_delta_m_residual_A": float(primary_delta_m_residual_A),
                        "primary_delta_m_residual_C": float(primary_delta_m_residual_C),
                        "primary_base_logodds_residual": float(primary_base_stats["base_logodds_residual"]),
                        "primary_cancellation_pass": bool(primary_cancellation_pass),
                        "primary_cancellation_pass_flag": float(1.0 if primary_cancellation_pass else 0.0),
                    }
                )
                for seed in random_projection_seeds:
                    row[f"proj_rand_s{int(seed)}_delta_norm_ratio"] = float(
                        proj_rand_norm_ratio_by_seed.get(int(seed), float("nan"))
                    )
                    row[f"proj_rand_s{int(seed)}_delta_cosine_raw"] = float(
                        proj_rand_cosine_by_seed.get(int(seed), float("nan"))
                    )

                for arm in all_arms:
                    arm_ok = bool(arm_results.get(arm, {}).get("success", False))
                    row[f"success_{arm}"] = arm_ok
                    row[f"error_{arm}"] = "" if arm_ok else str(arm_results.get(arm, {}).get("error", ""))
                    if not arm_ok:
                        row[f"effect_{arm}"] = float("nan")
                        row[f"delta_score_exp_{arm}"] = float("nan")
                        row[f"delta_score_other_{arm}"] = float("nan")
                        continue

                    row[f"effect_{arm}"] = float(arm_results[arm]["effect"])
                    row[f"delta_score_exp_{arm}"] = float(arm_results[arm]["delta_score_exp"])
                    row[f"delta_score_other_{arm}"] = float(arm_results[arm]["delta_score_other"])

                    for scope in ("exp", "other"):
                        decomp = arm_results[arm][f"decomp_{scope}"]
                        for k, v in decomp.items():
                            row[f"decomp_{scope}_{k}_{arm}"] = float(v)

                rand_effects: List[float] = []
                for seed in random_projection_seeds:
                    arm_name = f"PRJ_RAND_s{int(seed)}"
                    if bool(arm_results.get(arm_name, {}).get("success", False)):
                        rand_effects.append(float(row.get(f"effect_{arm_name}", float("nan"))))
                rand_effects = [x for x in rand_effects if math.isfinite(float(x))]
                if rand_effects:
                    mean_rand = float(sum(rand_effects) / len(rand_effects))
                    var_rand = float(sum((x - mean_rand) ** 2 for x in rand_effects) / len(rand_effects))
                    row["effect_PRJ_RAND_mean"] = float(mean_rand)
                    row["effect_PRJ_RAND_std"] = float(math.sqrt(max(var_rand, 0.0)))
                    row["n_PRJ_RAND_success"] = int(len(rand_effects))
                else:
                    row["effect_PRJ_RAND_mean"] = float("nan")
                    row["effect_PRJ_RAND_std"] = float("nan")
                    row["n_PRJ_RAND_success"] = int(0)

                if all_success:
                    row["d_BA"] = float(row["effect_B"] - row["effect_A"])
                    row["d_CA"] = float(row["effect_C"] - row["effect_A"])
                    row["d_DA"] = float(row["effect_D"] - row["effect_A"])
                    row["d_CCp"] = float(row["effect_C"] - row["effect_Cp"])
                    row["d_DDp"] = float(row["effect_D"] - row["effect_Dp"])
                    row["sign_agree_BA"] = float(_sign(float(row["effect_B"])) == _sign(float(row["effect_A"])))
                    row["sign_agree_CA"] = float(_sign(float(row["effect_C"])) == _sign(float(row["effect_A"])))
                    row["sign_agree_DA"] = float(_sign(float(row["effect_D"])) == _sign(float(row["effect_A"])))
                    row["sign_agree_CCp"] = float(_sign(float(row["effect_C"])) == _sign(float(row["effect_Cp"])))
                    row["sign_agree_DDp"] = float(_sign(float(row["effect_D"])) == _sign(float(row["effect_Dp"])))
                    row["d_PRJ_PCA_A"] = (
                        float(row["effect_PRJ_PCA"] - row["effect_A"])
                        if bool(arm_results.get("PRJ_PCA", {}).get("success", False))
                        else float("nan")
                    )
                    row["d_PRJ_RAND_mean_A"] = (
                        float(row["effect_PRJ_RAND_mean"] - row["effect_A"])
                        if math.isfinite(float(row["effect_PRJ_RAND_mean"]))
                        else float("nan")
                    )
                    row["d_STRESS_RECON_A"] = (
                        float(row["effect_STRESS_RECON"] - row["effect_A"])
                        if bool(arm_results.get("STRESS_RECON", {}).get("success", False))
                        else float("nan")
                    )
                    row["d_STRESS_RESID_A"] = (
                        float(row["effect_STRESS_RESID"] - row["effect_A"])
                        if bool(arm_results.get("STRESS_RESID", {}).get("success", False))
                        else float("nan")
                    )
                    row["d_STRESS_SUM_A"] = (
                        float(row["effect_STRESS_RECON"] + row["effect_STRESS_RESID"] - row["effect_A"])
                        if (
                            bool(arm_results.get("STRESS_RECON", {}).get("success", False))
                            and bool(arm_results.get("STRESS_RESID", {}).get("success", False))
                        )
                        else float("nan")
                    )
                    row["d_STRESS_RESID_RECON"] = (
                        float(row["effect_STRESS_RESID"] - row["effect_STRESS_RECON"])
                        if (
                            bool(arm_results.get("STRESS_RECON", {}).get("success", False))
                            and bool(arm_results.get("STRESS_RESID", {}).get("success", False))
                        )
                        else float("nan")
                    )
                else:
                    row["d_BA"] = float("nan")
                    row["d_CA"] = float("nan")
                    row["d_DA"] = float("nan")
                    row["d_CCp"] = float("nan")
                    row["d_DDp"] = float("nan")
                    row["sign_agree_BA"] = float("nan")
                    row["sign_agree_CA"] = float("nan")
                    row["sign_agree_DA"] = float("nan")
                    row["sign_agree_CCp"] = float("nan")
                    row["sign_agree_DDp"] = float("nan")
                    row["d_PRJ_PCA_A"] = float("nan")
                    row["d_PRJ_RAND_mean_A"] = float("nan")
                    row["d_STRESS_RECON_A"] = float("nan")
                    row["d_STRESS_RESID_A"] = float("nan")
                    row["d_STRESS_SUM_A"] = float("nan")
                    row["d_STRESS_RESID_RECON"] = float("nan")

                rows.append(row)
                _checkpoint_partial_rows()

    _write_csv(rows, Path(args.out_csv))

    rows_success = [r for r in rows if bool(r.get("all_arms_success", False))]
    rows_analysis = [r for r in rows if bool(r.get("analysis_included", False))]
    rows_invariant_fail = [r for r in rows_success if not bool(r.get("invariant_all_pass", False))]

    per_layer: List[Dict[str, Any]] = []
    for layer in layers:
        per_layer.append(
            _aggregate_layer(
                rows=rows_analysis,
                layer=int(layer),
                n_bootstrap=int(args.bootstrap_n),
                ci=float(args.ci),
                seed=int(args.bootstrap_seed),
                ratio_den_eps=float(args.ratio_den_eps),
            )
        )

    summary: Dict[str, Any] = {
        "model_name_or_path": str(args.model_name_or_path),
        "disamb_path": str(args.disamb_path),
        "n_items_loaded": int(len(items)),
        "n_layers_model": int(n_layers_model),
        "layers": [int(l) for l in layers],
        "run_config": {
            "device": str(device),
            "torch_dtype": _normalize_dtype_name(args.torch_dtype),
            "model_revision": None if args.revision is None else str(args.revision),
            "tokenizer_revision": None if args.tokenizer_revision is None else str(args.tokenizer_revision),
            "actual_model_dtype": str(model_dtype_name),
            "normalize_by_length": bool(args.normalize_by_length),
            "require_token_id_match": bool(args.require_token_id_match),
            "seed": int(args.seed),
            "bootstrap_n": int(args.bootstrap_n),
            "ci": float(args.ci),
            "bootstrap_seed": int(args.bootstrap_seed),
            "gate_activation_ratio_tol": float(args.gate_activation_ratio_tol),
            "gate_margin_abs_tol": float(args.gate_margin_abs_tol),
            "gate_score_abs_tol": float(args.gate_score_abs_tol),
            "gate_clt_equiv_abs_tol": float(args.gate_clt_equiv_abs_tol),
            "gate_identity_abs_tol": float(args.gate_identity_abs_tol),
            "hard_fail_invariant": bool(args.hard_fail_invariant),
            "hard_fail_primary_logodds": bool(args.hard_fail_primary_logodds),
            "primary_logodds_residual_tol": float(args.primary_logodds_residual_tol),
            "ratio_den_eps": float(args.ratio_den_eps),
            "raw_delta_norm_eps": float(args.raw_delta_norm_eps),
            "fidelity_rel_mse_warn": float(args.fidelity_rel_mse_warn),
            "run_pca_baseline": bool(args.run_pca_baseline),
            "run_random_orth_baseline": bool(args.run_random_orth_baseline),
            "random_projection_seeds": [int(s) for s in random_projection_seeds],
            "run_faithfulness_resid_control": bool(args.run_faithfulness_resid_control),
            "run_faithfulness_decomposition_arms": bool(args.run_faithfulness_decomposition_arms),
            "run_stress_arms": bool(run_stress_arms),
            "pca_fit_mode": str(args.pca_fit_mode),
            "pca_rank_mode": str(args.pca_rank_mode),
            "pca_fixed_rank": int(args.pca_fixed_rank),
            "clt_decode_strategies": ["delta_1decode", "safe_2decode"],
            "clt_dtype_policy": str(args.clt_dtype_policy),
            "clt_dtype": str(clt_dtype_name),
            "clt_width": str(args.clt_width),
            "clt_scale": float(args.clt_scale),
            "clt_eps_active": float(args.clt_eps_active),
        },
        "analysis_policy": limitation_analysis_policy_metadata(),
        "counts": {
            "n_rows_total": int(len(rows)),
            "n_rows_all_arms_success": int(len(rows_success)),
            "n_rows_analysis_included": int(len(rows_analysis)),
            "n_pairs_analysis_included": int(len({str(r["pair_id"]) for r in rows_analysis})),
            "n_invariant_fail_rows": int(len(rows_invariant_fail)),
        },
        "site_equivalence": {
            "raw_site": "resid_post (decoder block output hook)",
            "clt_site_mode": "same_site_v1",
            "clt_sites": {
                "encode": sorted({str(bundles[int(l)].meta.encode_site) for l in layers}),
                "decode": sorted({str(bundles[int(l)].meta.decode_site) for l in layers}),
                "writeback": sorted({str(bundles[int(l)].meta.writeback_site) for l in layers}),
            },
        },
        "arm_fail_counts": {k: int(v) for k, v in sorted(fail_counts.items())},
        "arm_fail_reasons": {k: dict(sorted(v.items())) for k, v in sorted(fail_reasons.items())},
        "pca_baseline_fit": {
            str(int(layer)): dict(pca_fit_meta.get(int(layer), {}))
            for layer in layers
        } if bool(args.run_pca_baseline) else {},
        "provenance": {
            "repo_root": "<repo-root>",
            "git_commit": str(_git_commit(repo_root)),
            "clt_repo": str(args.clt_repo),
            "clt_width": str(args.clt_width),
            "clt_run_name": (None if args.clt_run_name is None else str(args.clt_run_name)),
            "clt_l0_target": (None if args.clt_l0_target is None else int(args.clt_l0_target)),
            "fit_eval_split_description": (
                "pair-disjoint leave-one-pair-out PCA fit for each evaluated pair"
                if str(args.pca_fit_mode) == "leave_one_pair_out"
                else "single global PCA fit on the evaluation pool"
            ) if bool(args.run_pca_baseline) else "",
        },
        "per_layer": per_layer,
    }

    out_json_path = Path(args.out_json)
    _ensure_dir(out_json_path)
    out_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if bool(args.hard_fail_invariant) and rows_invariant_fail:
        sample = rows_invariant_fail[:3]
        sample_msg = "; ".join(
            f"pair={r.get('pair_id')} dir={r.get('direction')} layer={r.get('layer')} "
            f"act={r.get('gate_activation_ratio')} margin={r.get('gate_margin_abs_diff')} "
            f"score_exp={r.get('gate_score_abs_diff_exp')} score_other={r.get('gate_score_abs_diff_other')} "
            f"clt_c={r.get('gate_clt_effect_abs_diff_C')} clt_d={r.get('gate_clt_effect_abs_diff_D')} "
            f"id_raw={r.get('gate_identity_abs_effect_raw')} id_clt={r.get('gate_identity_abs_effect_clt')}"
            for r in sample
        )
        raise RuntimeError(
            f"A≈B invariant failed on {len(rows_invariant_fail)} row(s). "
            f"See {out_json_path} and {Path(args.out_csv)}. "
            f"Examples: {sample_msg}"
        )

    print(
        json.dumps(
            {
                "out_csv": str(Path(args.out_csv)),
                "out_json": str(out_json_path),
                "n_rows_total": int(len(rows)),
                "n_rows_analysis_included": int(len(rows_analysis)),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
