#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_limitation_short_paper_numbers import (
    DEFAULT_DISAMB_DATASET,
    render_disamb_target_table,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render the Appendix DISAMB target table from the canonical hardened "
            "data_paper_hardened_v2/disamb_pairs.jsonl surface."
        )
    )
    parser.add_argument(
        "--disamb-path",
        type=Path,
        default=DEFAULT_DISAMB_DATASET,
        help="Canonical hardened DISAMB JSONL file.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    print(render_disamb_target_table(args.disamb_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
