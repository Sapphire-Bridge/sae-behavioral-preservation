from pathlib import Path

from aom.data.loaders import load_disamb_pairs


BASE = Path(__file__).resolve().parents[1]


def test_load_disamb_pairs_smoke():
    items = load_disamb_pairs(str(BASE / "data" / "disamb_pairs.jsonl"))
    assert len(items) >= 1
    assert items[0].pair_id
