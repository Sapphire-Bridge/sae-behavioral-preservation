# Evaluation data

The canonical evaluation dataset for the SAE behavioral preservation result is
`data_paper_hardened_v2/disamb_pairs.jsonl` (52 paired homograph disambiguation
cases). Files under `data/` are retained only for legacy compatibility; paper
claims should cite the hardened snapshot path above. The schema below documents
the `.jsonl` format.

## Disambiguation pairs (`disamb_pairs.jsonl`)

Each line is a `DisambPair`:

- `pair_id`: string
- `target`: substring to patch/track (e.g. `"bank"`)
- `target_occurrence`: 0-based occurrence of `target` in each prompt
- `a`: `{ "prompt": str, "expected_label": str }`
- `b`: `{ "prompt": str, "expected_label": str }`
- `choices`: `{ label: [continuation_str, ...], ... }`
- `metadata` (optional): generator/debug fields; recommended keys include:
  - `type`: e.g. `"lexical"`
  - `word`: ambiguous target word
  - `labels`: the label set used in `choices`
  - `variant`: e.g. `"easy"` or `"hardened"`
  - `pair_variant`: e.g. `"clean"`, `"distractor"`, `"paraphrase_clean"`, `"paraphrase_distractor"`

Continuation strings should include leading spaces (GPT-2 tokenization),
e.g. `" river"`, `" loan"`.
