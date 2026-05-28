from __future__ import annotations

import pytest
import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel

from aom.interventions.activation_patching import PatchSpanSite
from aom.interventions.clt_adapter import CLTInputTransform, CLTPatchConfig
from aom.interventions.clt_patch import (
    CLTHookState,
    CLTInterventionHook,
    IdentityLatentPolicy,
    ReplaceLatentDimsAtIndicesPolicy,
    ReplaceLatentsAtIndicesPolicy,
    forward_with_clt_latent_patching_span,
)


class IdentityCLT(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self._d_in = int(d_in)
        self._d_latent = int(d_in)
        self._d_out = int(d_in)
        # A real parameter keeps device/dtype inference well-defined.
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


class ZeroAtSitePolicy:
    def apply(
        self,
        latents: torch.Tensor,
        *,
        site_mask: torch.Tensor,
        token_mask: torch.Tensor | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        out = latents.clone()
        mask = site_mask.expand_as(out)
        out[mask] = 0.0
        return out


def test_clt_intervention_hook_identity_is_noop():
    clt = IdentityCLT(d_in=8)
    hook = CLTInterventionHook(
        clt=clt,
        token_indices=(1, 3),
        policy=IdentityLatentPolicy(),
    )
    hidden = torch.randn(2, 5, 8)
    patched = hook(nn.Identity(), (), hidden)
    assert torch.allclose(hidden, patched, atol=1e-6)


def test_forward_with_clt_latent_patching_span_identity_is_noop():
    config = GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=100, n_positions=32)
    model = GPT2LMHeadModel(config)
    model.eval()

    clt = IdentityCLT(d_in=32)
    input_ids = torch.tensor([[10, 11, 12, 13, 14]], dtype=torch.long)

    base_logits = model(input_ids, use_cache=False).logits
    state = CLTHookState()
    patched_logits = forward_with_clt_latent_patching_span(
        model,
        input_ids=input_ids,
        site=PatchSpanSite(layer=0, token_indices=(2,)),
        clt=clt,
        policy=IdentityLatentPolicy(),
        state=state,
    )

    assert torch.allclose(base_logits, patched_logits, atol=1e-6)
    assert state.total_active >= 0.0


def test_forward_with_clt_latent_patching_span_zero_policy_changes_site():
    config = GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=100, n_positions=32)
    model = GPT2LMHeadModel(config)
    model.eval()

    clt = IdentityCLT(d_in=32)
    input_ids = torch.tensor([[10, 11, 12, 13, 14]], dtype=torch.long)

    base_logits = model(input_ids, use_cache=False).logits
    patched_logits = forward_with_clt_latent_patching_span(
        model,
        input_ids=input_ids,
        site=PatchSpanSite(layer=0, token_indices=(2,)),
        clt=clt,
        policy=ZeroAtSitePolicy(),
        transform=CLTInputTransform(scale=2.0, shift=1.0),
        config=CLTPatchConfig(decode_strategy="safe_2decode"),
    )

    # Intervention should alter model logits when we zero a patched token representation.
    assert not torch.allclose(base_logits, patched_logits, atol=1e-6)


def test_clt_decode_strategies_match_for_identity_clt():
    config = GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=100, n_positions=32)
    model = GPT2LMHeadModel(config)
    model.eval()
    clt = IdentityCLT(d_in=32)
    input_ids = torch.tensor([[10, 11, 12, 13, 14]], dtype=torch.long)

    safe_logits = forward_with_clt_latent_patching_span(
        model,
        input_ids=input_ids,
        site=PatchSpanSite(layer=0, token_indices=(2,)),
        clt=clt,
        policy=ZeroAtSitePolicy(),
        transform=CLTInputTransform(scale=2.0, shift=0.5),
        config=CLTPatchConfig(decode_strategy="safe_2decode"),
    )
    delta_logits = forward_with_clt_latent_patching_span(
        model,
        input_ids=input_ids,
        site=PatchSpanSite(layer=0, token_indices=(2,)),
        clt=clt,
        policy=ZeroAtSitePolicy(),
        transform=CLTInputTransform(scale=2.0, shift=0.5),
        config=CLTPatchConfig(decode_strategy="delta_1decode"),
    )
    assert torch.allclose(safe_logits, delta_logits, atol=1e-6)


def test_replace_latents_policy_casts_to_latent_dtype():
    latents = torch.randn(1, 4, 8, dtype=torch.float32)
    replacement = torch.randn(1, 2, 8, dtype=torch.float64)
    policy = ReplaceLatentsAtIndicesPolicy(token_indices=[1, 3], replacement_latents=replacement)
    patched = policy.apply(latents, site_mask=torch.ones(1, 4, 1, dtype=torch.bool))
    assert patched.dtype == latents.dtype
    assert torch.allclose(patched[:, 1, :], replacement[:, 0, :].to(dtype=latents.dtype), atol=1e-6)
    assert torch.allclose(patched[:, 3, :], replacement[:, 1, :].to(dtype=latents.dtype), atol=1e-6)


def test_replace_latent_dims_policy_replaces_only_selected_dims_and_casts_dtype():
    latents = torch.zeros(1, 4, 8, dtype=torch.float32)
    replacement = torch.full((1, 2, 8), 3.0, dtype=torch.float64)
    policy = ReplaceLatentDimsAtIndicesPolicy(
        token_indices=[1, 3],
        replacement_latents=replacement,
        latent_dims=[2, 5],
    )
    patched = policy.apply(latents, site_mask=torch.ones(1, 4, 1, dtype=torch.bool))
    assert patched.dtype == latents.dtype

    # Patched dims should match replacement at the specified tokens.
    assert torch.allclose(patched[:, 1, [2, 5]], torch.tensor([[3.0, 3.0]], dtype=torch.float32), atol=1e-6)
    assert torch.allclose(patched[:, 3, [2, 5]], torch.tensor([[3.0, 3.0]], dtype=torch.float32), atol=1e-6)
    # Unpatched dims should remain unchanged (zeros).
    assert torch.allclose(patched[:, 1, [0, 1, 3, 4, 6, 7]], torch.zeros(1, 6), atol=0.0)
    assert torch.allclose(patched[:, 3, [0, 1, 3, 4, 6, 7]], torch.zeros(1, 6), atol=0.0)


def test_forward_with_clt_latent_patching_span_raises_on_policy_site_index_mismatch():
    config = GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=100, n_positions=32)
    model = GPT2LMHeadModel(config)
    model.eval()
    clt = IdentityCLT(d_in=32)
    input_ids = torch.tensor([[10, 11, 12, 13, 14]], dtype=torch.long)

    replacement = torch.zeros(1, 1, 32, dtype=torch.float32)
    mismatch_policy = ReplaceLatentsAtIndicesPolicy(token_indices=[3], replacement_latents=replacement)
    with pytest.raises(ValueError, match="must match patch site token indices"):
        _ = forward_with_clt_latent_patching_span(
            model,
            input_ids=input_ids,
            site=PatchSpanSite(layer=0, token_indices=(2,)),
            clt=clt,
            policy=mismatch_policy,
        )


def test_forward_with_clt_latent_patching_span_respects_attention_mask_on_writeback():
    config = GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=100, n_positions=32)
    model = GPT2LMHeadModel(config)
    model.eval()
    clt = IdentityCLT(d_in=32)
    input_ids = torch.tensor([[10, 11, 12, 13, 14]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.long)

    base_logits = model(input_ids, attention_mask=attention_mask, use_cache=False).logits
    patched_logits = forward_with_clt_latent_patching_span(
        model,
        input_ids=input_ids,
        site=PatchSpanSite(layer=0, token_indices=(4,)),
        clt=clt,
        policy=ZeroAtSitePolicy(),
        attention_mask=attention_mask,
    )
    assert torch.allclose(base_logits, patched_logits, atol=1e-6)
