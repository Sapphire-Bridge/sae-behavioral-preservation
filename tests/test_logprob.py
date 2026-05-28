import torch
from transformers import GPT2Config, GPT2LMHeadModel

from aom.utils import (
    logprob_of_continuation_candidates_shared_prompt,
    logprob_of_continuation_candidates_with_prefill,
    logprob_of_continuation_ids,
)


def test_logprob_of_continuation_ids_matches_manual():
    config = GPT2Config(n_layer=1, n_head=1, n_embd=16, vocab_size=50, n_positions=32)
    model = GPT2LMHeadModel(config)
    model.eval()

    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    cont = torch.tensor([[4, 5]], dtype=torch.long)

    lp = logprob_of_continuation_ids(model, prompt_ids=prompt, continuation_ids=cont, normalize_by_length=False)

    with torch.no_grad():
        full = torch.cat([prompt, cont], dim=1)
        logits = model(full, use_cache=False).logits
        # cont tokens at positions 3 and 4 are predicted by logits at positions 2 and 3
        logp0 = torch.log_softmax(logits[:, 2, :], dim=-1).gather(1, cont[:, 0:1]).squeeze(1)
        logp1 = torch.log_softmax(logits[:, 3, :], dim=-1).gather(1, cont[:, 1:2]).squeeze(1)
        manual = logp0 + logp1

    assert torch.allclose(lp, manual, atol=1e-5)


def test_logprob_candidates_shared_prompt_matches_scalar_path():
    config = GPT2Config(n_layer=1, n_head=1, n_embd=16, vocab_size=50, n_positions=32)
    model = GPT2LMHeadModel(config)
    model.eval()

    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    continuations = [
        torch.tensor([4, 5], dtype=torch.long),
        torch.tensor([6], dtype=torch.long),
        torch.tensor([7, 8, 9], dtype=torch.long),
    ]

    expected = []
    for cont in continuations:
        lp = logprob_of_continuation_ids(
            model,
            prompt_ids=prompt,
            continuation_ids=cont.unsqueeze(0),
            normalize_by_length=False,
        )
        expected.append(float(lp.item()))

    got_no_cache = logprob_of_continuation_candidates_shared_prompt(
        model=model,
        prompt_ids=prompt,
        continuation_id_list=continuations,
        normalize_by_length=False,
        batch_size=2,
        pad_token_id=0,
        use_prefix_cache=False,
    )
    got_cache = logprob_of_continuation_candidates_shared_prompt(
        model=model,
        prompt_ids=prompt,
        continuation_id_list=continuations,
        normalize_by_length=False,
        batch_size=2,
        pad_token_id=0,
        use_prefix_cache=True,
    )

    expected_t = torch.tensor(expected, dtype=got_no_cache.dtype)
    assert torch.allclose(got_no_cache.cpu(), expected_t.cpu(), atol=1e-5)
    assert torch.allclose(got_cache.cpu(), expected_t.cpu(), atol=1e-5)


def test_logprob_candidates_with_prefill_matches_scalar_path():
    config = GPT2Config(n_layer=1, n_head=1, n_embd=16, vocab_size=50, n_positions=32)
    model = GPT2LMHeadModel(config)
    model.eval()

    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    continuations = [
        torch.tensor([4, 5], dtype=torch.long),
        torch.tensor([6], dtype=torch.long),
        torch.tensor([7, 8, 9], dtype=torch.long),
    ]

    expected = []
    for cont in continuations:
        lp = logprob_of_continuation_ids(
            model,
            prompt_ids=prompt,
            continuation_ids=cont.unsqueeze(0),
            normalize_by_length=False,
        )
        expected.append(float(lp.item()))

    prefill = model(prompt, use_cache=True, return_dict=True)
    got = logprob_of_continuation_candidates_with_prefill(
        model=model,
        prompt_ids=prompt,
        continuation_id_list=continuations,
        prefill_logits_last=prefill.logits[:, -1, :],
        prefill_past_key_values=prefill.past_key_values,
        normalize_by_length=False,
        batch_size=2,
        pad_token_id=0,
    )
    expected_t = torch.tensor(expected, dtype=got.dtype)
    assert torch.allclose(got.cpu(), expected_t.cpu(), atol=1e-5)
