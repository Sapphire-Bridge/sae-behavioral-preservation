from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

PRIMARY_LOGODDS_RESIDUAL_TOL_DEFAULT = 5e-6


def tokenize_cont_ids(tokenizer, text: str) -> Tuple[int, ...]:
    ids = tokenizer(str(text), add_special_tokens=False)["input_ids"]
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError(f"Unexpected batched tokenization output for continuation: {text!r}")
        ids = ids[0]
    return tuple(int(x) for x in ids)


def choices_token_ids_from_strings(tokenizer, choices: Mapping[str, Sequence[str]]) -> dict[str, list[tuple[int, ...]]]:
    return {
        str(label): [tokenize_cont_ids(tokenizer, str(cont)) for cont in continuations]
        for label, continuations in choices.items()
    }


def safe_log_prob_mass(x: float, *, eps: float = 1e-30) -> float:
    return float(math.log(max(float(x), float(eps))))


def signed_delta_margin_from_logodds(*, dlogpe: float, dlogpo: float, effect_sign: float = 1.0) -> float:
    return float(float(effect_sign) * (float(dlogpe) - float(dlogpo)))


@dataclass(frozen=True)
class PrimaryApplicability:
    labels_binary: bool
    expected_all_single: bool
    other_all_single: bool
    no_duplicates_expected: bool
    no_duplicates_other: bool
    token_sets_disjoint: bool
    candidate_count_equal: bool
    same_scored_position: bool
    static_applicable: bool
    reason: str
    expected_tokens: Tuple[int, ...]
    other_tokens: Tuple[int, ...]
    expected_unique_tokens: Tuple[int, ...]
    other_unique_tokens: Tuple[int, ...]


def evaluate_primary_applicability(
    *,
    choices_token_ids: Mapping[str, Sequence[Sequence[int]]],
    expected_label: str,
    other_label: str,
    scored_positions: Sequence[int],
    require_binary_labels: bool = True,
    require_equal_candidate_counts: bool = True,
) -> PrimaryApplicability:
    labels_binary = bool(len(choices_token_ids) == 2) if bool(require_binary_labels) else True

    expected_token_seqs = [tuple(int(x) for x in seq) for seq in choices_token_ids.get(str(expected_label), ())]
    other_token_seqs = [tuple(int(x) for x in seq) for seq in choices_token_ids.get(str(other_label), ())]

    expected_all_single = bool(expected_token_seqs) and all(len(seq) == 1 for seq in expected_token_seqs)
    other_all_single = bool(other_token_seqs) and all(len(seq) == 1 for seq in other_token_seqs)

    expected_tokens = tuple(int(seq[0]) for seq in expected_token_seqs if len(seq) == 1)
    other_tokens = tuple(int(seq[0]) for seq in other_token_seqs if len(seq) == 1)

    expected_unique = tuple(sorted({int(t) for t in expected_tokens}))
    other_unique = tuple(sorted({int(t) for t in other_tokens}))

    no_duplicates_expected = bool(len(expected_unique) == len(expected_tokens))
    no_duplicates_other = bool(len(other_unique) == len(other_tokens))
    token_sets_disjoint = bool(set(expected_unique).isdisjoint(set(other_unique)))
    candidate_count_equal = bool(len(expected_tokens) == len(other_tokens))

    scored = tuple(int(x) for x in scored_positions)
    same_scored_position = bool(scored) and bool(len(scored) == len(expected_token_seqs) + len(other_token_seqs)) and bool(
        len(set(scored)) == 1
    )

    static_applicable = bool(
        labels_binary
        and expected_all_single
        and other_all_single
        and no_duplicates_expected
        and no_duplicates_other
        and token_sets_disjoint
        and same_scored_position
        and (candidate_count_equal if bool(require_equal_candidate_counts) else True)
    )

    if not labels_binary:
        reason = "labels_not_binary"
    elif not expected_all_single or not other_all_single:
        reason = "multi_token_candidate"
    elif not no_duplicates_expected or not no_duplicates_other:
        reason = "duplicate_token_within_label"
    elif not token_sets_disjoint:
        reason = "non_disjoint_expected_other_sets"
    elif bool(require_equal_candidate_counts) and not candidate_count_equal:
        reason = "candidate_count_mismatch_conservative_exclusion"
    elif not same_scored_position:
        reason = "not_same_scored_position"
    else:
        reason = "ok"

    return PrimaryApplicability(
        labels_binary=bool(labels_binary),
        expected_all_single=bool(expected_all_single),
        other_all_single=bool(other_all_single),
        no_duplicates_expected=bool(no_duplicates_expected),
        no_duplicates_other=bool(no_duplicates_other),
        token_sets_disjoint=bool(token_sets_disjoint),
        candidate_count_equal=bool(candidate_count_equal),
        same_scored_position=bool(same_scored_position),
        static_applicable=bool(static_applicable),
        reason=str(reason),
        expected_tokens=tuple(expected_tokens),
        other_tokens=tuple(other_tokens),
        expected_unique_tokens=tuple(expected_unique),
        other_unique_tokens=tuple(other_unique),
    )
