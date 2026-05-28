from __future__ import annotations

import torch
from transformers import PreTrainedTokenizerFast

from aom.metrics.disamb import _encode_prompt
from aom.token_spans import token_span_for_substring


def _make_wordlevel_tokenizer(vocab: dict[str, int]) -> PreTrainedTokenizerFast:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )


def test_token_span_for_substring_wordlevel_offsets():
    vocab = {"[PAD]": 0, "[EOS]": 1, "[UNK]": 2, "I": 3, "sat": 4, "by": 5, "the": 6, "bank": 7}
    tok = _make_wordlevel_tokenizer(vocab)
    text = "I sat by the bank"
    span, token_ids = token_span_for_substring(tok, text, "bank", 0)
    assert span == [4]
    assert token_ids == [vocab["bank"]]


def test_token_span_ids_match_encode_prompt_ids():
    vocab = {"[PAD]": 0, "[EOS]": 1, "[UNK]": 2, "I": 3, "sat": 4, "by": 5, "the": 6, "bank": 7}
    tok = _make_wordlevel_tokenizer(vocab)
    text = "I sat by the bank"
    span, token_ids = token_span_for_substring(tok, text, "bank", 0)
    prompt_ids = _encode_prompt(tok, text, device=torch.device("cpu"))[0].tolist()
    assert [int(prompt_ids[i]) for i in span] == [int(x) for x in token_ids]
