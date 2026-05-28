import torch
from transformers import GPT2Config, GPT2LMHeadModel

from aom.interventions.activation_patching import (
    PatchSite,
    PatchSpanSite,
    forward_with_patched_block_output,
    forward_with_patched_block_output_span,
    get_hidden_states,
)


def test_forward_with_patched_block_output_sham_is_noop():
    config = GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=100, n_positions=32)
    model = GPT2LMHeadModel(config)
    model.eval()

    input_ids = torch.tensor([[10, 11, 12, 13, 14]], dtype=torch.long)
    hs = get_hidden_states(model, input_ids)
    replacement = hs[1][0, 2, :].detach()  # layer 0 post-block for token_index=2

    base_logits = model(input_ids, use_cache=False).logits
    patched_logits = forward_with_patched_block_output(
        model, input_ids=input_ids, site=PatchSite(layer=0, token_index=2), replacement=replacement
    )

    assert torch.allclose(base_logits, patched_logits, atol=1e-5)


def test_forward_with_patched_block_output_span_sham_is_noop():
    config = GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=100, n_positions=32)
    model = GPT2LMHeadModel(config)
    model.eval()

    input_ids = torch.tensor([[10, 11, 12, 13, 14]], dtype=torch.long)
    hs = get_hidden_states(model, input_ids)
    # Patch two token positions with their own states (should be a no-op).
    replacement = hs[1][0, [1, 3], :].detach()  # layer 0 post-block for token indices 1 and 3

    base_logits = model(input_ids, use_cache=False).logits
    patched_logits = forward_with_patched_block_output_span(
        model,
        input_ids=input_ids,
        site=PatchSpanSite(layer=0, token_indices=(1, 3)),
        replacement=replacement,
    )

    assert torch.allclose(base_logits, patched_logits, atol=1e-5)
