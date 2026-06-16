# Limitation Paper Reproduction Report

- run root: `$RUN_ROOT` (`sae_limitation_cpu_fvu_v2`)
- run root created by script: `no`
- mode: `full`
- requested_device: `cpu`
- effective_device: `cpu`
- local_files_only: `yes`
- overall_status: `PASS`

## Command Log

### 1. Gemma-3 comparability source run

- status: `PASS`
- started_at_utc: `2026-06-15T12:07:12+00:00`
- ended_at_utc: `2026-06-16T00:01:58+00:00`
- exit_code: `0`
- command:

```bash
python scripts/clt_raw_comparability.py --model_name_or_path google/gemma-3-4b-pt --revision cc012e0a6d0787b4adcc0fa2c4da74402494554d --tokenizer_revision cc012e0a6d0787b4adcc0fa2c4da74402494554d --disamb_path data_paper_hardened_v2/disamb_pairs.jsonl --clt_repo clt_bundles/sae_writeback_limitation_release --clt_width 16k --layers 4,5,8,11,16 --device cpu --attn_implementation eager --seed 42 --bootstrap_n 1000 --bootstrap_seed 42 --ci 0.95 --primary_logodds_residual_tol 5e-06 --run_pca_baseline --run_random_orth_baseline --run_faithfulness_decomposition_arms --out_csv '$RUN_ROOT/source/comparability/gemma3_4b_comparability.csv' --out_json '$RUN_ROOT/source/comparability/gemma3_4b_comparability.summary.json' --no-hard_fail_invariant --no-hard_fail_primary_logodds --torch_dtype float32 --local_files_only
```

- stdout_log: `$RUN_ROOT/logs/gemma_3_comparability_source_run.stdout.log`
- stderr_log: `$RUN_ROOT/logs/gemma_3_comparability_source_run.stderr.log`
- generated files:
  - `$RUN_ROOT/logs/gemma_3_comparability_source_run.stderr.log`
  - `$RUN_ROOT/logs/gemma_3_comparability_source_run.stdout.log`
  - `$RUN_ROOT/source/comparability/gemma3_4b_comparability.csv`
  - `$RUN_ROOT/source/comparability/gemma3_4b_comparability.partial.csv`
  - `$RUN_ROOT/source/comparability/gemma3_4b_comparability.summary.json`

### 2. Gemma-3 top-k source run

- status: `PASS`
- started_at_utc: `2026-06-16T00:01:58+00:00`
- ended_at_utc: `2026-06-16T02:59:26+00:00`
- exit_code: `0`
- command:

```bash
python analysis/aom_clt_topk_recovery.py --model_name_or_path google/gemma-3-4b-pt --revision cc012e0a6d0787b4adcc0fa2c4da74402494554d --tokenizer_revision cc012e0a6d0787b4adcc0fa2c4da74402494554d --clt_repo clt_bundles/sae_writeback_limitation_release --clt_width 16k --layers 4,8 --ks 20,50,100 --logz_ks 20,50,100 --device cpu --attn_implementation eager --disamb_path data_paper_hardened_v2/disamb_pairs.jsonl --seed 0 --split_seed 0 --frac_selection 0.5 --bootstrap_B 1000 --ci 0.95 --out_csv '$RUN_ROOT/source/topk/gemma3_4b_topk.csv' --out_summary '$RUN_ROOT/source/topk/gemma3_4b_topk.summary.json' --overwrite --torch_dtype float32 --local_files_only
```

- stdout_log: `$RUN_ROOT/logs/gemma_3_top_k_source_run.stdout.log`
- stderr_log: `$RUN_ROOT/logs/gemma_3_top_k_source_run.stderr.log`
- generated files:
  - `$RUN_ROOT/logs/gemma_3_top_k_source_run.stderr.log`
  - `$RUN_ROOT/logs/gemma_3_top_k_source_run.stdout.log`
  - `$RUN_ROOT/source/topk/gemma3_4b_topk.csv`
  - `$RUN_ROOT/source/topk/gemma3_4b_topk.summary.json`
  - `$RUN_ROOT/source/topk/gemma3_4b_topk.summary.manifest.json`

### 3. Derive paper outputs

- status: `PASS`
- started_at_utc: `2026-06-16T02:59:26+00:00`
- ended_at_utc: `2026-06-16T02:59:26+00:00`
- exit_code: `0`
- command:

```bash
internal: derive limitation paper outputs from source summaries
```

- generated files:
  - `$RUN_ROOT/derived/limitation_paper_numbers.json`
  - `$RUN_ROOT/derived/stress_arm_summary.csv`

### 4. Build release surface

- status: `PASS`
- started_at_utc: `2026-06-16T02:59:26+00:00`
- ended_at_utc: `2026-06-16T02:59:27+00:00`
- exit_code: `0`
- command:

```bash
internal: build limitation release surface from source_run_root
```

- generated files:
  - `$RUN_ROOT/release/figures/centerpiece_summary.svg`
  - `$RUN_ROOT/release/figures/topk_summary.svg`
  - `$RUN_ROOT/release/results/comparability/l4/comparability.summary.json`
  - `$RUN_ROOT/release/results/comparability/l8/comparability.summary.json`
  - `$RUN_ROOT/release/results/source/comparability/gemma3_4b_comparability.csv`
  - `$RUN_ROOT/release/results/source/comparability/gemma3_4b_comparability.summary.json`
  - `$RUN_ROOT/release/results/source/topk/gemma3_4b_topk.summary.json`
  - `$RUN_ROOT/release/results/topk/l4/topk.summary.json`
  - `$RUN_ROOT/release/results/topk/l8/topk.summary.json`
  - `$RUN_ROOT/release/tables/centerpiece_summary.csv`
  - `$RUN_ROOT/release/tables/gate_diagnostics_rows.csv`
  - `$RUN_ROOT/release/tables/gate_diagnostics_summary.csv`
  - `$RUN_ROOT/release/tables/release_manifest.json`
  - `$RUN_ROOT/release/tables/robustness_input_case_target.csv`
  - `$RUN_ROOT/release/tables/robustness_summary.csv`
  - `$RUN_ROOT/release/tables/strict_gate_sensitivity.csv`
  - `$RUN_ROOT/release/tables/topk_summary.csv`
