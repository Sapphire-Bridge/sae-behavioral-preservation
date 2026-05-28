from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel

from aom.data.schemas import DisambPair, PromptSide
from aom.interventions.clt_adapter import CLTInputTransform, CLTPatchConfig
from aom.metrics.clt_cpt import (
    compute_clt_cpt_context_swap_patching,
    compute_clt_cpt_context_swap_patching_topk,
    compute_clt_latent_delta_stats,
    topk_clt_latents,
)


class IdentityCLT(nn.Module):
    def __init__(self, d_in: int) -> None:
        super().__init__()
        self._d_in = int(d_in)
        self._d_latent = int(d_in)
        self._d_out = int(d_in)
        self.W_dec = nn.Parameter(torch.eye(self._d_out))

    @property
    def d_in(self) -> int:
        return self._d_in

    @property
    def d_latent(self) -> int:
        return self._d_latent

    @property
    def d_out(self) -> int:
        return self._d_out

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return latents


@dataclass
class CharTokenizer:
    vocab_size: int = 128
    pad_token_id: int | None = 0
    eos_token_id: int | None = 0

    def __call__(
        self,
        text: str,
        return_tensors: str = "pt",
        add_special_tokens: bool = False,  # noqa: ARG002
        return_offsets_mapping: bool = False,
    ) -> Dict[str, torch.Tensor]:
        s = str(text)
        if len(s) == 0:
            s = " "
        ids = [ord(ch) % int(self.vocab_size) for ch in s]
        out: Dict[str, torch.Tensor] = {"input_ids": torch.tensor([ids], dtype=torch.long)}
        if return_offsets_mapping:
            offsets = [[i, i + 1] for i in range(len(s))]
            out["offset_mapping"] = torch.tensor([offsets], dtype=torch.long)
        return out


def _tiny_model(vocab_size: int = 128) -> GPT2LMHeadModel:
    config = GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=vocab_size, n_positions=128)
    model = GPT2LMHeadModel(config)
    model.eval()
    return model


def _tiny_disamb_items() -> List[DisambPair]:
    return [
        DisambPair(
            pair_id="pair-1",
            target="bank",
            target_occurrence=0,
            a=PromptSide(
                prompt="The river bank is steep and wet.",
                expected_label="river",
            ),
            b=PromptSide(
                prompt="I visited the bank and opened an account.",
                expected_label="finance",
            ),
            choices={
                "river": [" near water"],
                "finance": [" to deposit cash"],
            },
            metadata=None,
        )
    ]


def test_compute_clt_cpt_context_swap_patching_smoke_outputs():
    model = _tiny_model(vocab_size=128)
    tokenizer = CharTokenizer(vocab_size=128)
    items = _tiny_disamb_items()
    device = torch.device("cpu")
    clt = IdentityCLT(d_in=32)
    transform = CLTInputTransform(scale=1.0)

    out = compute_clt_cpt_context_swap_patching(
        model=model,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        items=items,
        device=device,
        clt_by_layer={0: clt},
        transform_by_layer={0: transform},
        layers=[0],
        config=CLTPatchConfig(decode_strategy="safe_2decode", dtype_policy="clt"),
        normalize_by_length=True,
        require_token_id_match=True,
        ci=0.95,
        bootstrap_n=20,
        bootstrap_seed=42,
    )

    # Structural keys
    expected = {
        "mean_max_effect",
        "mean_max_effect_ci_low",
        "mean_max_effect_ci_high",
        "mean_argmax_layer",
        "flip_rate_at_best_layer",
        "mean_norm_max_effect",
        "mean_sham_max_effect",
        "mean_identity_max_abs_effect",
        "n_directions_total",
        "n_directions_patched",
        "n_directions_skipped_misaligned",
        "effect_layer_0",
        "sham_effect_layer_0",
        "identity_effect_layer_0",
        "identity_abs_effect_layer_0",
    }
    assert expected.issubset(set(out.keys()))
    assert int(out["n_directions_total"]) == 2
    assert int(out["n_directions_patched"]) == 2
    assert int(out["n_directions_skipped_misaligned"]) == 0

    # Sham should be near no-op (same latents injected back).
    assert abs(float(out["mean_sham_max_effect"])) < 1e-4
    assert abs(float(out["sham_effect_layer_0"])) < 1e-4
    # Identity should be near no-op as well.
    assert abs(float(out["mean_identity_max_abs_effect"])) < 1e-4
    assert abs(float(out["identity_effect_layer_0"])) < 1e-4


def test_compute_clt_latent_delta_stats_shapes_and_counters():
    model = _tiny_model(vocab_size=128)
    tokenizer = CharTokenizer(vocab_size=128)
    items = _tiny_disamb_items()
    device = torch.device("cpu")
    clt = IdentityCLT(d_in=32)
    transform = CLTInputTransform(scale=1.0)

    stats = compute_clt_latent_delta_stats(
        model=model,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        items=items,
        device=device,
        layer=0,
        clt=clt,
        transform=transform,
        require_token_id_match=True,
    )
    assert int(stats.layer) == 0
    assert int(stats.d_latent) == 32
    assert int(stats.n_directions_total) == 2
    assert int(stats.n_directions_used) == 2
    assert int(stats.n_directions_skipped_misaligned) == 0
    assert tuple(stats.mean_delta.shape) == (32,)
    assert stats.mean_delta.dtype == torch.float64
    assert str(stats.mean_delta.device) == "cpu"

    top5 = topk_clt_latents(stats, k=5, score="abs_t_stat")
    assert len(top5) == 5
    assert len(set(top5)) == 5
    assert all(0 <= int(i) < 32 for i in top5)


def test_compute_clt_cpt_topk_matches_full_when_k_equals_width():
    model = _tiny_model(vocab_size=128)
    tokenizer = CharTokenizer(vocab_size=128)
    items = _tiny_disamb_items()
    device = torch.device("cpu")
    clt = IdentityCLT(d_in=32)
    transform = CLTInputTransform(scale=1.0)
    cfg = CLTPatchConfig(decode_strategy="safe_2decode", dtype_policy="clt")

    all_dims = list(range(32))
    out = compute_clt_cpt_context_swap_patching_topk(
        model=model,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        items=items,
        device=device,
        layer=0,
        clt=clt,
        transform=transform,
        latent_dims=all_dims,
        config=cfg,
        normalize_by_length=True,
        require_token_id_match=True,
        include_random_k=False,
        bootstrap_n=20,
        bootstrap_seed=42,
    )

    assert int(out["n_directions_total"]) == 2
    assert int(out["n_directions_patched"]) == 2
    assert int(out["n_directions_skipped_misaligned"]) == 0
    assert int(out["k"]) == 32

    # If we patch all dims, top-k patching is identical to full replacement.
    assert abs(float(out["mean_effect_full"]) - float(out["mean_effect_topk"])) < 1e-6
    assert abs(float(out["flip_rate_full"]) - float(out["flip_rate_topk"])) < 1e-6

    # Sanity: sham + identity should be near no-op.
    assert abs(float(out["mean_effect_sham"])) < 1e-4
    assert abs(float(out["mean_identity_max_abs_effect"])) < 1e-4
