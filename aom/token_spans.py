from __future__ import annotations

from typing import List, Tuple

from transformers import PreTrainedTokenizerBase


def _find_nth(haystack: str, needle: str, n: int) -> int:
    if n < 0:
        raise ValueError("n must be >= 0")
    idx = -1
    start = 0
    for _ in range(n + 1):
        idx = haystack.find(needle, start)
        if idx < 0:
            return -1
        start = idx + len(needle)
    return idx


def token_span_for_substring(
    tokenizer: PreTrainedTokenizerBase, text: str, substring: str, occurrence: int
) -> Tuple[List[int], List[int]]:
    """
    Map a substring occurrence in `text` to a contiguous token span via offsets mapping.

    Returns (token_indices, token_ids).
    """
    char_idx = _find_nth(text, substring, occurrence)
    if char_idx < 0:
        raise ValueError(f"Target substring not found: {substring!r} (occurrence={occurrence})")
    char_end = char_idx + len(substring)

    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc.get("offset_mapping", None)
    if offsets is None:
        raise ValueError("Tokenizer must support return_offsets_mapping for patching.")
    offsets = offsets[0].tolist()
    input_ids = enc["input_ids"][0].tolist()

    span = [i for i, (s, e) in enumerate(offsets) if (s < char_end and e > char_idx)]
    if not span:
        raise ValueError("Could not map target substring to token span (offsets mapping mismatch).")

    # Require a contiguous span for patching.
    if span != list(range(span[0], span[-1] + 1)):
        raise ValueError("Target substring maps to a non-contiguous token span; adjust prompt or target.")

    return [int(i) for i in span], [int(input_ids[i]) for i in span]

