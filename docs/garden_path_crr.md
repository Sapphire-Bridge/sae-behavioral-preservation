# Garden-path CRR (exploratory extension)

Runs the paper's raw-vs-SAE comparability test (CRR) on a second task:
structural ambiguity in garden-path sentences, using the same model
(Gemma 3 4B PT), the same Gemma Scope 16k SAE bundle, the same estimand and
the same runner. Only the dataset changes.

Status: exploratory. Nothing here is part of the frozen paper surface.

## What a pair is

Source rows come from Hanna & Mueller (NAACL 2025), `data_csv/gp_same_len.csv`
in <https://github.com/hannamw/GP-mechanisms>. Each row has a sentence that
forces the garden-path reading and one that forces the non-garden-path reading,
differing only in the verb:

| Side | Example (NP/Z) | Expected label |
|------|----------------|----------------|
| `a`  | Because the suspect altered the file | `gp` |
| `b`  | Because the suspect lied the file    | `nongp` |

- **Target:** the shared sentence-final noun (`file`). The runner patches the
  donor's activation at that span into the receiver, in both directions, raw
  and after SAE writeback.
- **Readout:** the paper's single-token log-odds endpoint.
  - NP/Z: `","` vs `" was"`
  - NP/S: `"."` vs `" was"`
- **MV/RR** is excluded, as in the source paper.

The source CSV is not vendored because the GP-mechanisms repo has no license
file. Its rows derive from Arehalli et al. (2022), a subset of the SAP
benchmark (MIT).

## Run

```bash
# 1. Convert (CPU, seconds)
git clone https://github.com/hannamw/GP-mechanisms.git ../GP-mechanisms
python scripts/prepare_garden_path_pairs.py \
  --input ../GP-mechanisms/data_csv/gp_same_len.csv \
  --output data_garden_path/garden_path_pairs.jsonl

# 2. Assets, once (same as the paper)
make limitation-assets

# 3. CRR sweep: same layers and settings as the paper, new dataset
BUNDLE=$(python -c "import sys; sys.path.insert(0, 'scripts'); import limitation_requirements as r; print(r.LIMITATION_LOCAL_CLT_BUNDLE_PATH)")
python scripts/clt_raw_comparability.py \
  --model_name_or_path google/gemma-3-4b-pt \
  --revision cc012e0a6d0787b4adcc0fa2c4da74402494554d \
  --tokenizer_revision cc012e0a6d0787b4adcc0fa2c4da74402494554d \
  --disamb_path data_garden_path/garden_path_pairs.jsonl \
  --clt_repo "$BUNDLE" --clt_width 16k \
  --layers 4,5,8,11,16 \
  --device mps --torch_dtype float32 --attn_implementation eager \
  --seed 42 --bootstrap_n 1000 --bootstrap_seed 42 --ci 0.95 \
  --out_csv runs/garden_path/comparability.csv \
  --out_json runs/garden_path/comparability.summary.json
```

Use `--max_pairs 4` for a first smoke run.

## Check before reading any CRR

1. **Positive control.** The raw patching effect at the final noun must be
   clearly non-zero. In NP/Z the disambiguating information (verb transitivity)
   sits before the noun. If the raw effect is small, CRR is unstable, and the
   patch site rather than the SAE is the finding.
2. **Spans.** The runner aborts on token-span mismatches. With
   `--require_token_id_match` (default), donor and receiver noun spans must
   tokenize identically.
3. **Sample size.** 24 pairs per condition is too few for tight paired CIs.
   Treat this run as a feasibility check; a confirmatory version needs the
   larger SAP set and a locked layer-pair choice.
