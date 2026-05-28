from __future__ import annotations

from typing import Mapping, Sequence

from .schemas import DisambPair


def _validate_prompt_cont_boundary(*, prompt: str, continuation: str, item_id: str, field: str) -> None:
    if not prompt:
        raise ValueError(f"{item_id}: empty prompt/context for {field}")
    if not continuation:
        raise ValueError(f"{item_id}: empty continuation in {field}")
    if (not prompt[-1].isspace()) and (not continuation[0].isspace()):
        raise ValueError(
            f"{item_id}: continuation boundary likely wrong in {field} "
            f"(prompt does not end with whitespace and continuation does not start with whitespace)"
        )


def _validate_choices(
    *,
    choices: Mapping[str, Sequence[str]],
    expected_labels: Sequence[str],
    prompts: Sequence[str],
    item_id: str,
    field: str,
    require_equal_counts: bool,
) -> None:
    if not choices:
        raise ValueError(f"{item_id}: empty choices for {field}")

    for lab in expected_labels:
        if lab not in choices:
            raise ValueError(f"{item_id}: expected_label={lab!r} missing from choices for {field}")

    counts = set()
    for label, conts in choices.items():
        if not isinstance(label, str) or not label:
            raise ValueError(f"{item_id}: invalid label in choices for {field}: {label!r}")
        if not conts:
            raise ValueError(f"{item_id}: empty continuation list for label={label!r} in {field}")
        counts.add(len(conts))
        for cont in conts:
            if not isinstance(cont, str):
                raise ValueError(f"{item_id}: non-string continuation for label={label!r} in {field}")
            for prompt in prompts:
                _validate_prompt_cont_boundary(prompt=prompt, continuation=cont, item_id=item_id, field=field)

    if require_equal_counts and len(counts) > 1:
        raise ValueError(f"{item_id}: unequal continuation counts per label in {field}: {sorted(counts)}")


def validate_disamb_pairs(items: Sequence[DisambPair], *, require_equal_choice_counts: bool = True) -> None:
    for it in items:
        _validate_choices(
            choices=it.choices,
            expected_labels=(it.a.expected_label, it.b.expected_label),
            prompts=(it.a.prompt, it.b.prompt),
            item_id=str(it.pair_id),
            field="disamb.choices",
            require_equal_counts=require_equal_choice_counts,
        )


def validate_evidence_metadata(*, metadata: object, item_id: str) -> None:
    """
    Validate optional metadata-first evidence annotations.

    Supported keys:
    - query_pos: int
    - evidence_spans: list[[start, end] | {"start": int, "end": int}]
      where span is half-open [start, end) in token indices.
    """
    if metadata is None:
        return
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{item_id}: metadata must be a mapping when evidence annotations are present")

    if "query_pos" in metadata:
        q = metadata.get("query_pos", None)
        if not isinstance(q, int):
            raise ValueError(f"{item_id}: metadata.query_pos must be int")

    if "evidence_spans" in metadata:
        spans = metadata.get("evidence_spans", None)
        if not isinstance(spans, Sequence) or isinstance(spans, (str, bytes, bytearray)):
            raise ValueError(f"{item_id}: metadata.evidence_spans must be a list")
        for idx, sp in enumerate(spans):
            if isinstance(sp, Mapping):
                a = sp.get("start", None)
                b = sp.get("end", None)
            elif isinstance(sp, Sequence) and not isinstance(sp, (str, bytes, bytearray)) and len(sp) == 2:
                a, b = sp[0], sp[1]
            else:
                raise ValueError(f"{item_id}: invalid evidence span at index {idx}: {sp!r}")
            if not isinstance(a, int) or not isinstance(b, int):
                raise ValueError(f"{item_id}: evidence span indices must be ints at index {idx}")
            if a < 0 or b < 0 or b <= a:
                raise ValueError(f"{item_id}: evidence span must satisfy 0 <= start < end at index {idx}")
