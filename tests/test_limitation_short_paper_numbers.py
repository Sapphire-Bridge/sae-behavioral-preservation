from __future__ import annotations

from pathlib import Path

from scripts.check_limitation_short_paper_numbers import DEFAULT_PAPER, check_paper_numbers


def test_limitation_short_paper_number_check_passes_current_surface() -> None:
    assert check_paper_numbers(DEFAULT_PAPER) == []


def test_limitation_short_paper_number_check_reports_missing_literal(tmp_path: Path) -> None:
    paper_text = DEFAULT_PAPER.read_text(encoding="utf-8")
    broken_paper = tmp_path / "broken_short_paper.md"
    broken_paper.write_text(paper_text.replace("0.506", "0.505"), encoding="utf-8")

    problems = check_paper_numbers(broken_paper)

    assert any(problem.literal == "0.506" for problem in problems)
    assert any(problem.label == "L4 centerpiece comparability literals" for problem in problems)
    assert any(problem.literal == "0.505" for problem in problems)
    assert any(problem.label == "unmapped numeric literal" for problem in problems)


def test_limitation_short_paper_number_check_reports_missing_method_literal(tmp_path: Path) -> None:
    paper_text = DEFAULT_PAPER.read_text(encoding="utf-8")
    broken_paper = tmp_path / "broken_short_paper.md"
    broken_paper.write_text(
        paper_text.replace("normalize_by_length = true", "length normalization enabled"),
        encoding="utf-8",
    )

    problems = check_paper_numbers(broken_paper)

    assert any(problem.literal == "normalize_by_length = true" for problem in problems)
    assert any(problem.label == "limitation method-surface literals" for problem in problems)
