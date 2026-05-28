import torch
from transformers import GPT2Config, GPT2LMHeadModel

from aom.utils import logprob_of_continuation_ids, set_seed


def test_determinism_cpu_forward_and_logprob():
    set_seed(0)
    config = GPT2Config(
        n_layer=1,
        n_head=1,
        n_embd=16,
        vocab_size=50,
        n_positions=32,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
    )
    model = GPT2LMHeadModel(config)
    model.eval()

    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    cont = torch.tensor([[4, 5]], dtype=torch.long)

    lp1 = logprob_of_continuation_ids(model, prompt_ids=prompt, continuation_ids=cont, normalize_by_length=False)
    lp2 = logprob_of_continuation_ids(model, prompt_ids=prompt, continuation_ids=cont, normalize_by_length=False)
    assert torch.equal(lp1, lp2)
