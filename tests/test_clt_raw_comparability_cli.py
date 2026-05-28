from __future__ import annotations

from scripts.clt_raw_comparability import parse_args


def test_clt_raw_comparability_accepts_revision_flags() -> None:
    args = parse_args(
        [
            "--model_name_or_path",
            "dummy",
            "--disamb_path",
            "dummy.jsonl",
            "--clt_repo",
            "dummy",
            "--layers",
            "4",
            "--revision",
            "test-rev",
            "--tokenizer_revision",
            "test-tok-rev",
            "--out_csv",
            "/tmp/dummy.csv",
        ]
    )

    assert args.revision == "test-rev"
    assert args.tokenizer_revision == "test-tok-rev"
