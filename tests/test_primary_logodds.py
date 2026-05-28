from __future__ import annotations

import pytest

from aom.metrics.primary_logodds import evaluate_primary_applicability, signed_delta_margin_from_logodds


def test_primary_applicability_happy_path() -> None:
    app = evaluate_primary_applicability(
        choices_token_ids={
            "exp": [(10,), (11,), (12,)],
            "other": [(20,), (21,), (22,)],
        },
        expected_label="exp",
        other_label="other",
        scored_positions=[5, 5, 5, 5, 5, 5],
    )
    assert app.static_applicable is True
    assert app.reason == "ok"
    assert app.same_scored_position is True


@pytest.mark.parametrize(
    ("choices_token_ids", "scored_positions", "expected_reason"),
    [
        (
            {"exp": [(10,), (11,)], "other": [(11,), (20,)]},
            [0, 0, 0, 0],
            "non_disjoint_expected_other_sets",
        ),
        (
            {"exp": [(10,), (10,)], "other": [(20,), (21,)]},
            [0, 0, 0, 0],
            "duplicate_token_within_label",
        ),
        (
            {"exp": [(10, 11), (12,)], "other": [(20,), (21,)]},
            [0, 0, 0, 0],
            "multi_token_candidate",
        ),
        (
            {"exp": [(10,), (11,), (12,)], "other": [(20,), (21,)]},
            [0, 0, 0, 0, 0],
            "candidate_count_mismatch_conservative_exclusion",
        ),
        (
            {"exp": [(10,), (11,)], "other": [(20,), (21,)]},
            [0, 0, 1, 1],
            "not_same_scored_position",
        ),
    ],
)
def test_primary_applicability_exclusions(
    choices_token_ids: dict[str, list[tuple[int, ...]]],
    scored_positions: list[int],
    expected_reason: str,
) -> None:
    app = evaluate_primary_applicability(
        choices_token_ids=choices_token_ids,
        expected_label="exp",
        other_label="other",
        scored_positions=scored_positions,
    )
    assert app.static_applicable is False
    assert app.reason == expected_reason


def test_signed_delta_margin_from_logodds_matches_effect_sign() -> None:
    assert signed_delta_margin_from_logodds(dlogpe=0.7, dlogpo=0.2, effect_sign=1.0) == pytest.approx(0.5, abs=1e-12)
    assert signed_delta_margin_from_logodds(dlogpe=0.7, dlogpo=0.2, effect_sign=-1.0) == pytest.approx(-0.5, abs=1e-12)
