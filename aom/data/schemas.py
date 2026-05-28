from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional


@dataclass(frozen=True)
class PromptSide:
    prompt: str
    expected_label: str


@dataclass(frozen=True)
class DisambPair:
    """
    Minimal pair for lexical/structural disambiguation.

    `choices` maps labels -> list of continuation strings to score as next tokens/phrases.
    """
    pair_id: str
    target: str
    target_occurrence: int
    a: PromptSide
    b: PromptSide
    choices: Mapping[str, List[str]]
    metadata: Optional[Dict[str, Any]] = None
