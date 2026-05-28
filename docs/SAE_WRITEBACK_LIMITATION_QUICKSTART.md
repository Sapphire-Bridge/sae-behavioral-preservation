# SAE Writeback Limitation Quickstart

This surface is intentionally small. It covers the public result package for the SAE writeback limitation claim family, not the full reproduction bundle.

For the full paper rerun, use:

```bash
export HF_TOKEN="hf_..."
make limitation-assets
python scripts/run_limitation_paper.py \
  --run_root /tmp/sae_limitation_full \
  --device auto \
  --require_accelerator
```

If the required model and Gemma Scope snapshots are already cached locally and
you want to force offline-only execution, use:

```bash
make limitation-assets LIMITATION_ASSETS_ARGS="--local_files_only"
python scripts/run_limitation_paper.py \
  --run_root /tmp/sae_limitation_full \
  --device auto \
  --require_accelerator \
  --local_files_only
```

## Public Surface

- `paper/sae_writeback_limitation_short_paper.md` (canonical external technical surface)
- `paper/sae_writeback_limitation_paper.md` (repo/governance-facing claim surface)
- `results/sae_writeback_limitation_release/comparability/l4/comparability.summary.json`
- `results/sae_writeback_limitation_release/comparability/l8/comparability.summary.json`
- `results/sae_writeback_limitation_release/topk/l4/topk.summary.json`
- `results/sae_writeback_limitation_release/topk/l8/topk.summary.json`
- `tables/sae_writeback_limitation_release/centerpiece_summary.csv`
- `tables/sae_writeback_limitation_release/topk_summary.csv`

## Prerequisites

- The checked-in public limitation reference surface must exist under `results/sae_writeback_limitation_release/`.
- The local SAE/CLT bundle is materialized under `clt_bundles/sae_writeback_limitation_release/`; `make limitation-assets` creates it when needed and reuses it when present.
- Fresh connected machines need accepted Hugging Face access to `google/gemma-3-4b-pt` and a valid `HF_TOKEN` before `make limitation-assets` can download/cache the pinned snapshots. No separate `huggingface-cli login` is required.

## One Command

```bash
make limitation-one-result
```

Optional accelerated sanity check:

```bash
make limitation-one-result-gpu
```

After exporting `HF_TOKEN`, the reviewer GPU path can be run as one command:

```bash
make limitation-reviewer-check-gpu
```

## Expected Outputs

- A fresh temp run root containing `comparability/l4/comparability.summary.json`
- A fresh temp run root containing `one_result_check_report.md`
- A fresh temp run root containing `one_result_check_log.json`

## PASS / FAIL

- `PASS` means the fresh CPU L4 public summary matches the checked-in L4 reference on the public headline metrics and identity fields.
- `FAIL` means the fresh L4 run drifted on either the frozen identity fields, counts, or one of the public numeric metrics.
- This quickstart does not rerun L8 or top-k and does not validate any bundle, contract, CI, or archive surface.
