#!/usr/bin/env python3
"""
Convert garden-path triples into DisambPair JSONL for the raw-vs-SAE CRR runner.

Input: a CSV in the format of Hanna & Mueller (NAACL 2025), data_csv/gp_same_len.csv
(https://github.com/hannamw/GP-mechanisms), with columns
    item, condition, sentence_ambiguous, sentence_gp, sentence_post
where `sentence_gp` forces the garden-path reading and `sentence_post` forces the
non-garden-path reading. The data is not vendored here because the source repo
carries no license file; pass its path explicitly.

Each row becomes one pair:
    a = sentence_gp   (expected label "gp")
    b = sentence_post (expected label "nongp")
    target = the sentence-final noun, which both sides share; the runner patches
             the donor's activation at this span into the receiver (both directions).

Continuations follow the paper's single-token operationalisation, which the
primary log-odds endpoint requires (binary labels, one single-token candidate each):
    NPZ: gp "," vs nongp " was"
    NPS: gp "." vs nongp " was"
MV/RR is excluded, as in the paper (both continuation probabilities are near zero).

Usage:
    python scripts/prepare_garden_path_pairs.py \
        --input /path/to/GP-mechanisms/data_csv/gp_same_len.csv \
        --output data_garden_path/garden_path_pairs.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aom.data.loaders import load_disamb_pairs  # noqa: E402

REQUIRED_COLUMNS = ("item", "condition", "sentence_ambiguous", "sentence_gp", "sentence_post")

CHOICES_BY_CONDITION: Dict[str, Dict[str, List[str]]] = {
    "NPZ": {"gp": [","], "nongp": [" was"]},
    "NPS": {"gp": ["."], "nongp": [" was"]},
}
SUPPORTED_CONDITIONS = tuple(CHOICES_BY_CONDITION)


def _final_word(sentence: str) -> str:
    words = sentence.split()
    if not words:
        raise ValueError("empty sentence")
    return words[-1]


def _last_occurrence_index(text: str, needle: str) -> int:
    """Zero-based index of the last non-overlapping occurrence, matching token_spans._find_nth."""
    count = 0
    start = 0
    while True:
        idx = text.find(needle, start)
        if idx < 0:
            break
        count += 1
        start = idx + len(needle)
    if count == 0:
        raise ValueError(f"{needle!r} not found in {text!r}")
    return count - 1


def row_to_pair(row: Dict[str, str]) -> Dict[str, object]:
    condition = row["condition"].strip()
    if condition not in CHOICES_BY_CONDITION:
        raise ValueError(f"unsupported condition {condition!r}; expected one of {SUPPORTED_CONDITIONS}")

    item = row["item"].strip()
    pair_id = f"gp-{condition.lower()}-{item}"
    gp = row["sentence_gp"].strip()
    post = row["sentence_post"].strip()

    target = _final_word(gp)
    if _final_word(post) != target:
        raise ValueError(f"{pair_id}: final word differs between sides ({target!r} vs {_final_word(post)!r})")

    occurrence = _last_occurrence_index(gp, target)
    if _last_occurrence_index(post, target) != occurrence:
        raise ValueError(f"{pair_id}: target {target!r} occurs a different number of times on each side")

    return {
        "pair_id": pair_id,
        "target": target,
        "target_occurrence": occurrence,
        "a": {"prompt": gp, "expected_label": "gp"},
        "b": {"prompt": post, "expected_label": "nongp"},
        "choices": CHOICES_BY_CONDITION[condition],
        "metadata": {
            "type": "structural",
            "condition": condition,
            "item": item,
            "sentence_ambiguous": row["sentence_ambiguous"].strip(),
            "source": "hanna_mueller_2025_gp_same_len",
        },
    }


def convert_rows(rows: Iterable[Dict[str, str]], conditions: Sequence[str]) -> List[Dict[str, object]]:
    wanted = set(conditions)
    return [row_to_pair(r) for r in rows if r["condition"].strip() in wanted]


def read_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        return list(reader)


def write_jsonl(pairs: Sequence[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True, help="Garden-path CSV (gp_same_len.csv format).")
    p.add_argument("--output", type=Path, required=True, help="Destination DisambPair JSONL.")
    p.add_argument(
        "--conditions",
        type=lambda s: [c.strip() for c in s.split(",") if c.strip()],
        default=list(SUPPORTED_CONDITIONS),
        help=f"Comma-separated subset of {','.join(SUPPORTED_CONDITIONS)} (default: all).",
    )
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    unknown = [c for c in args.conditions if c not in CHOICES_BY_CONDITION]
    if unknown:
        raise SystemExit(f"unsupported conditions {unknown}; expected subset of {SUPPORTED_CONDITIONS}")

    pairs = convert_rows(read_rows(args.input), args.conditions)
    if not pairs:
        raise SystemExit("no rows matched the requested conditions")
    write_jsonl(pairs, args.output)

    # Round-trip through the runner's own loader so schema/boundary errors surface here.
    loaded = load_disamb_pairs(str(args.output))
    counts: Dict[str, int] = {}
    for p in pairs:
        cond = str(p["metadata"]["condition"])  # type: ignore[index]
        counts[cond] = counts.get(cond, 0) + 1
    print(f"wrote {len(loaded)} pairs to {args.output} ({counts})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
