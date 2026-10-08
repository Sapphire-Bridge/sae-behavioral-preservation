import csv
import importlib.util
import json
from pathlib import Path

import pytest

from aom.data.loaders import load_disamb_pairs
from aom.data.schemas import DisambPair, PromptSide
from aom.data.validate import validate_disamb_pairs

BASE = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location("prepare_gp", BASE / "scripts" / "prepare_garden_path_pairs.py")
prepare_gp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prepare_gp)

# Constructed fixture rows in the gp_same_len.csv format (not taken from the source dataset).
ROWS = [
    {"item": "1", "condition": "NPZ", "sentence_ambiguous": "While the chef cooked the soup",
     "sentence_gp": "While the chef stirred the soup", "sentence_post": "While the chef smiled the soup"},
    {"item": "1", "condition": "NPS", "sentence_ambiguous": "The teacher noticed the error",
     "sentence_gp": "The teacher fixed the error", "sentence_post": "The teacher said the error"},
    {"item": "1", "condition": "MVRR", "sentence_ambiguous": "The clerk sent the letter",
     "sentence_gp": "The clerk burned the letter", "sentence_post": "The clerk shown the letter"},
]


def _write_csv(path: Path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(prepare_gp.REQUIRED_COLUMNS))
        w.writeheader()
        w.writerows(rows)


def test_convert_maps_sides_target_and_choices():
    pairs = prepare_gp.convert_rows(ROWS, ["NPZ", "NPS"])
    assert [p["pair_id"] for p in pairs] == ["gp-npz-1", "gp-nps-1"]

    npz = pairs[0]
    assert npz["target"] == "soup"
    assert npz["target_occurrence"] == 0
    assert npz["a"] == {"prompt": "While the chef stirred the soup", "expected_label": "gp"}
    assert npz["b"] == {"prompt": "While the chef smiled the soup", "expected_label": "nongp"}
    assert npz["choices"] == {"gp": [","], "nongp": [" was"]}
    assert npz["metadata"]["sentence_ambiguous"] == "While the chef cooked the soup"
    assert pairs[1]["choices"] == {"gp": ["."], "nongp": [" was"]}


def test_mvrr_excluded_by_default_and_rejected_if_requested():
    assert all(p["metadata"]["condition"] != "MVRR" for p in prepare_gp.convert_rows(ROWS, ["NPZ", "NPS"]))
    with pytest.raises(ValueError, match="unsupported condition"):
        prepare_gp.row_to_pair(ROWS[2])


def test_target_occurrence_uses_last_match():
    row = dict(ROWS[0], sentence_gp="The dog chased the dog", sentence_post="The dog barked the dog")
    assert prepare_gp.row_to_pair(row)["target_occurrence"] == 1


def test_mismatched_final_word_fails_loudly():
    row = dict(ROWS[0], sentence_post="While the chef smiled the stew")
    with pytest.raises(ValueError, match="final word differs"):
        prepare_gp.row_to_pair(row)


def test_cli_round_trips_through_runner_loader(tmp_path):
    src = tmp_path / "gp.csv"
    out = tmp_path / "pairs.jsonl"
    _write_csv(src, ROWS)

    assert prepare_gp.main(["--input", str(src), "--output", str(out)]) == 0

    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    items = load_disamb_pairs(str(out))
    assert {it.a.expected_label for it in items} == {"gp"}
    assert {it.b.expected_label for it in items} == {"nongp"}


def test_validator_accepts_punctuation_led_continuations():
    item = DisambPair(
        pair_id="p", target="soup", target_occurrence=0,
        a=PromptSide("While the chef stirred the soup", "gp"),
        b=PromptSide("While the chef smiled the soup", "nongp"),
        choices={"gp": [","], "nongp": [" was"]},
    )
    validate_disamb_pairs([item])


def test_validator_still_rejects_unspaced_word_continuations():
    item = DisambPair(
        pair_id="p", target="soup", target_occurrence=0,
        a=PromptSide("While the chef stirred the soup", "gp"),
        b=PromptSide("While the chef smiled the soup", "nongp"),
        choices={"gp": ["and"], "nongp": [" was"]},
    )
    with pytest.raises(ValueError, match="boundary likely wrong"):
        validate_disamb_pairs([item])
